"""
Pattern Extractor for the Meta Agent Workflow pipeline.

Analyzes an :class:`AlignedTraceSet` to extract structural patterns:
deterministic sequences, parameterizable steps, variable steps (for Agent
Nodes), branching points (for ConditionContext), loop patterns, and user
input boundaries.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from science_modeling_tools.automation.meta_agent.models import (
    AlignedPosition,
    AlignedTraceSet,
    AlignmentType,
    BranchPattern,
    ExtractedPatterns,
    LoopPattern,
    ParameterizableInfo,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.target_converter import (
    TargetSpec,
    TargetSpecWithFallback,
)


# Strategy stability priority (highest → lowest).
STRATEGY_PRIORITY: List[str] = [
    "data-qa",
    "data-testid",
    "id",
    "aria",
    "xpath-text",
    "xpath-class",
    "css",
    "agent",
]

_STRATEGY_RANK: Dict[str, int] = {s: i for i, s in enumerate(STRATEGY_PRIORITY)}


class PatternExtractor:
    """
    Extracts structural patterns from an aligned trace set.

    Identifies deterministic sequences, variable steps (for Agent Nodes),
    branching points (for ConditionContext), and loop patterns.
    """

    def __init__(
        self,
        strategy_priority: Optional[List[str]] = None,
    ) -> None:
        self._strategy_priority = strategy_priority or STRATEGY_PRIORITY
        self._strategy_rank: Dict[str, int] = {
            s: i for i, s in enumerate(self._strategy_priority)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, aligned_set: AlignedTraceSet) -> ExtractedPatterns:
        """Extract all patterns from the aligned trace set."""

        positions = aligned_set.positions

        # Detect structural patterns first — they may consume positions.
        loop_patterns = self._detect_loops(positions)
        branch_patterns = self._detect_branches(positions)

        # Collect indices already claimed by loops / branches.
        loop_indices: Set[int] = set()
        for lp in loop_patterns:
            loop_indices.update(range(lp.body_start, lp.body_end + 1))

        branch_indices: Set[int] = set()
        for bp in branch_patterns:
            branch_indices.add(bp.branch_point_index)

        # Classify remaining positions.
        deterministic_steps: List[AlignedPosition] = []
        parameterizable_steps: List[Tuple[AlignedPosition, ParameterizableInfo]] = []
        variable_steps: List[AlignedPosition] = []
        optional_steps: List[AlignedPosition] = []
        user_input_boundaries: List[int] = []
        step_order: List[int] = []

        for pos in positions:
            idx = pos.index
            step_order.append(idx)

            # Check for user-input boundary first (wait(True) steps).
            if self._is_user_input_boundary(pos):
                user_input_boundaries.append(idx)
                # User-input boundaries are still categorised by their
                # alignment type so every index lands in exactly one bucket.

            # Skip positions already consumed by loops or branches.
            if idx in loop_indices or idx in branch_indices:
                continue

            atype = pos.alignment_type

            if atype == AlignmentType.DETERMINISTIC:
                # Consolidate target across runs.
                best_target = self._select_best_target(pos.steps)
                self._apply_best_target(pos, best_target)
                deterministic_steps.append(pos)

            elif atype == AlignmentType.PARAMETERIZABLE:
                param_info = self._build_parameterizable_info(pos)
                parameterizable_steps.append((pos, param_info))

            elif atype == AlignmentType.VARIABLE:
                self._record_variants(pos)
                variable_steps.append(pos)

            elif atype == AlignmentType.OPTIONAL:
                optional_steps.append(pos)

            elif atype == AlignmentType.BRANCH_POINT:
                # Already handled above via _detect_branches, but if the
                # aligner tagged it and we didn't detect a branch pattern,
                # treat as variable.
                if idx not in branch_indices:
                    self._record_variants(pos)
                    variable_steps.append(pos)

        return ExtractedPatterns(
            deterministic_steps=deterministic_steps,
            parameterizable_steps=parameterizable_steps,
            variable_steps=variable_steps,
            optional_steps=optional_steps,
            branch_patterns=branch_patterns,
            loop_patterns=loop_patterns,
            user_input_boundaries=user_input_boundaries,
            step_order=step_order,
        )

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------

    def _detect_loops(
        self,
        positions: List[AlignedPosition],
    ) -> List[LoopPattern]:
        """Detect repeated subsequences indicating loops.

        Uses a rolling-hash approach: for each candidate body length *L*
        (from 1 up to half the sequence length), slide a window of size *L*
        and check whether consecutive windows have matching action-type
        signatures.  When two or more consecutive windows match, record a
        :class:`LoopPattern`.
        """

        n = len(positions)
        if n < 2:
            return []

        # Build a signature string per position for fast comparison.
        sigs = self._position_signatures(positions)

        found: List[LoopPattern] = []
        consumed: Set[int] = set()

        for body_len in range(1, n // 2 + 1):
            i = 0
            while i + body_len <= n:
                # Skip if any position in the candidate window is consumed.
                if any(k in consumed for k in range(i, i + body_len)):
                    i += 1
                    continue

                # Count consecutive repetitions of the window starting at i.
                pattern = sigs[i : i + body_len]
                reps = 1
                j = i + body_len
                while (
                    j + body_len <= n
                    and sigs[j : j + body_len] == pattern
                    and not any(k in consumed for k in range(j, j + body_len))
                ):
                    reps += 1
                    j += body_len

                if reps >= 2:
                    body_start = positions[i].index
                    body_end = positions[i + body_len - 1].index
                    body_steps = positions[i : i + body_len]
                    found.append(LoopPattern(
                        body_start=body_start,
                        body_end=body_end,
                        min_iterations=reps,
                        max_iterations=reps,
                        body_steps=body_steps,
                    ))
                    for k in range(i, j):
                        consumed.add(k)
                    i = j
                else:
                    i += 1

        return found

    # ------------------------------------------------------------------
    # Branch detection
    # ------------------------------------------------------------------

    def _detect_branches(
        self,
        positions: List[AlignedPosition],
    ) -> List[BranchPattern]:
        """Detect divergence points indicating conditional branches.

        A branch point is an aligned position explicitly tagged as
        ``BRANCH_POINT`` by the aligner, or a ``VARIABLE`` position where
        the non-None steps split into distinct action-type groups.

        ``condition_description`` is set to the placeholder
        ``"Unknown: runs diverge at this point"`` per the design doc.
        """

        found: List[BranchPattern] = []

        for pos in positions:
            if pos.alignment_type not in (
                AlignmentType.BRANCH_POINT,
            ):
                continue

            # Group non-None steps by action_type to form branches.
            branches: Dict[str, List[AlignedPosition]] = {}
            for tid, step in pos.steps.items():
                if step is None:
                    continue
                label = step.action_type
                if label not in branches:
                    branches[label] = []
                # Each branch gets a single-position list for now.
                branches[label].append(AlignedPosition(
                    index=pos.index,
                    alignment_type=AlignmentType.VARIABLE,
                    steps={tid: step},
                    confidence=pos.confidence,
                ))

            if len(branches) >= 2:
                found.append(BranchPattern(
                    branch_point_index=pos.index,
                    branches=branches,
                    condition_description="Unknown: runs diverge at this point",
                    condition_source=None,
                ))

        return found

    # ------------------------------------------------------------------
    # Target consolidation
    # ------------------------------------------------------------------

    def _select_best_target(
        self,
        steps: Dict[str, Optional[TraceStep]],
    ) -> TargetSpecWithFallback:
        """Select the most robust target from multiple observations.

        Consolidates targets across runs into a
        :class:`TargetSpecWithFallback`.  Orders strategies by stability:
        data-qa > data-testid > id > aria > xpath-text > xpath-class >
        css > agent.
        """

        seen_strategies: Dict[Tuple[str, str], int] = {}  # (strategy, value) -> count

        for step in steps.values():
            if step is None or step.target is None:
                continue
            self._collect_strategies(step.target, seen_strategies)

        if not seen_strategies:
            # No usable target — return empty fallback.
            return TargetSpecWithFallback(strategies=[])

        # De-duplicate and sort by priority, then by frequency (desc).
        unique: Dict[Tuple[str, str], int] = dict(seen_strategies)
        sorted_specs = sorted(
            unique.items(),
            key=lambda item: (
                self._strategy_rank.get(item[0][0], len(self._strategy_priority)),
                -item[1],
            ),
        )

        strategies = [
            TargetSpec(strategy=s, value=v) for (s, v), _ in sorted_specs
        ]
        return TargetSpecWithFallback(strategies=strategies)

    # ------------------------------------------------------------------
    # Parameterizable step helpers
    # ------------------------------------------------------------------

    def _build_parameterizable_info(
        self,
        pos: AlignedPosition,
    ) -> ParameterizableInfo:
        """Split args into variable_args and constant_args.

        An arg key is *constant* if every non-None step has the same value
        for it.  Otherwise it is *variable* and gets a template variable
        name inferred from the key.
        """

        non_none = [s for s in pos.steps.values() if s is not None]
        if not non_none:
            return ParameterizableInfo(variable_args={}, constant_args={})

        # Gather all arg keys across steps.
        all_keys: Set[str] = set()
        for step in non_none:
            if step.args:
                all_keys.update(step.args.keys())

        variable_args: Dict[str, str] = {}
        constant_args: Dict[str, Any] = {}

        for key in sorted(all_keys):
            values = []
            for step in non_none:
                val = (step.args or {}).get(key)
                values.append(val)

            # Check if all values are the same.
            if len(set(repr(v) for v in values)) == 1:
                constant_args[key] = values[0]
            else:
                # Infer template variable name from the arg key.
                variable_args[key] = self._infer_template_name(key, pos)

        return ParameterizableInfo(
            variable_args=variable_args,
            constant_args=constant_args,
        )

    @staticmethod
    def _infer_template_name(key: str, pos: AlignedPosition) -> str:
        """Infer a template variable name from an argument key.

        Uses the key itself as the template name, optionally prefixed with
        the action type for disambiguation.
        """

        # Use the first non-None step's action_type for context.
        action_type = None
        for step in pos.steps.values():
            if step is not None:
                action_type = step.action_type
                break

        # Simple heuristic: use the key directly.  For common keys like
        # "text" we prefix with the action type.
        if key in ("text", "value") and action_type:
            return f"{action_type}_{key}"
        return key

    # ------------------------------------------------------------------
    # Variable step helpers
    # ------------------------------------------------------------------

    def _record_variants(self, pos: AlignedPosition) -> None:
        """Record observed variants and frequencies in position metadata.

        Stores ``variants`` (action_type → count) and ``variant_details``
        (action_type → list of step summaries) in the position's first
        non-None step's metadata.
        """

        variant_counts: Counter[str] = Counter()
        variant_details: Dict[str, List[Dict[str, Any]]] = {}

        for tid, step in pos.steps.items():
            if step is None:
                continue
            atype = step.action_type
            variant_counts[atype] += 1
            if atype not in variant_details:
                variant_details[atype] = []
            variant_details[atype].append({
                "trace_id": tid,
                "target": _target_summary(step.target),
                "args": step.args,
            })

        # Store in the first non-None step's metadata.
        for step in pos.steps.values():
            if step is not None:
                step.metadata["variants"] = dict(variant_counts)
                step.metadata["variant_details"] = variant_details
                break

    # ------------------------------------------------------------------
    # User-input boundary detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_user_input_boundary(pos: AlignedPosition) -> bool:
        """Check if a position represents a user-input boundary.

        A user-input boundary is a ``wait`` action where the args contain
        ``wait=True`` or the target is ``True``.
        """

        for step in pos.steps.values():
            if step is None:
                continue
            if step.action_type != "wait":
                return False
            # Check args for wait=True.
            if step.args and step.args.get("wait") is True:
                return True
            # Check target is True.
            if step.target is True:
                return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _position_signatures(
        positions: List[AlignedPosition],
    ) -> List[str]:
        """Build a list of action-type signatures for loop detection."""

        sigs: List[str] = []
        for pos in positions:
            types = sorted(
                s.action_type
                for s in pos.steps.values()
                if s is not None
            )
            sigs.append("|".join(types) if types else "<gap>")
        return sigs

    @staticmethod
    def _apply_best_target(
        pos: AlignedPosition,
        best_target: TargetSpecWithFallback,
    ) -> None:
        """Apply the consolidated best target to all non-None steps."""

        if not best_target.strategies:
            return
        for step in pos.steps.values():
            if step is not None:
                step.target = best_target

    @staticmethod
    def _collect_strategies(
        target: Any,
        seen: Dict[Tuple[str, str], int],
    ) -> None:
        """Collect (strategy, value) pairs from a target into *seen*."""

        if isinstance(target, TargetSpecWithFallback):
            for spec in target.strategies:
                key = (spec.strategy, spec.value)
                seen[key] = seen.get(key, 0) + 1
        elif isinstance(target, TargetSpec):
            key = (target.strategy, target.value)
            seen[key] = seen.get(key, 0) + 1
        elif isinstance(target, dict):
            # Handle dict-based targets (e.g., from serialization).
            strategies = target.get("strategies", [])
            for s in strategies:
                if isinstance(s, dict):
                    strat = s.get("strategy", "")
                    val = s.get("value", "")
                    if strat and val:
                        key = (strat, val)
                        seen[key] = seen.get(key, 0) + 1
        elif isinstance(target, str) and target:
            # Plain string target — treat as agent strategy.
            key = ("agent", target)
            seen[key] = seen.get(key, 0) + 1


def _target_summary(target: Any) -> Any:
    """Return a JSON-safe summary of a target for variant recording."""

    if target is None:
        return None
    if isinstance(target, TargetSpecWithFallback):
        return [{"strategy": s.strategy, "value": s.value} for s in target.strategies]
    if isinstance(target, TargetSpec):
        return {"strategy": target.strategy, "value": target.value}
    if isinstance(target, (str, bool, int, float)):
        return target
    return str(target)

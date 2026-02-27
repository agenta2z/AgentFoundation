"""
Trace Aligner for the Meta Agent Workflow pipeline.

Aligns multiple execution traces using LCS-based sequence alignment to
identify corresponding steps across runs. Steps are matched by action type
and target equivalence — NOT by ``__id__`` comparison, which is
session-specific and unstable across runs.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from agent_foundation.automation.meta_agent.models import (
    AlignedPosition,
    AlignedTraceSet,
    AlignmentType,
    ExecutionTrace,
    TraceStep,
)

# Optional BeautifulSoup for HTML structure comparison.
try:
    from bs4 import BeautifulSoup, Tag  # type: ignore[import-untyped]

    _HAS_BS4 = True
except ImportError:  # pragma: no cover
    _HAS_BS4 = False

# Regex helpers for lightweight HTML attribute extraction.
_ATTR_RE = re.compile(r'''([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')''')
_TAG_RE = re.compile(r"<(\w+)[\s>]")


class TraceAligner:
    """
    Aligns multiple execution traces using sequence alignment.

    Uses longest common subsequence (LCS) based alignment to match
    corresponding steps across traces. Steps are matched by action type
    and target equivalence.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, traces: List[ExecutionTrace]) -> AlignedTraceSet:
        """
        Align multiple traces and classify each position.

        Args:
            traces: List of normalized execution traces.

        Returns:
            AlignedTraceSet with classified positions.
        """
        if not traces:
            return AlignedTraceSet(
                positions=[], trace_ids=[], alignment_score=1.0,
            )

        trace_ids = [t.trace_id for t in traces]

        if len(traces) == 1:
            positions = self._single_trace_positions(traces[0])
            return AlignedTraceSet(
                positions=positions,
                trace_ids=trace_ids,
                alignment_score=1.0,
            )

        # Progressive pairwise alignment: start with the first two traces,
        # then fold in each subsequent trace.
        alignment = self._pairwise_align(traces[0], traces[1])
        multi: List[Dict[str, Optional[int]]] = []
        for idx_a, idx_b in alignment:
            multi.append({
                traces[0].trace_id: idx_a,
                traces[1].trace_id: idx_b,
            })

        for trace in traces[2:]:
            multi = self._fold_trace(multi, traces, trace)

        # Build AlignedPositions from the multi-alignment.
        positions: List[AlignedPosition] = []
        total_matched = 0
        total_positions = len(multi)

        for pos_idx, mapping in enumerate(multi):
            steps: Dict[str, Optional[TraceStep]] = {}
            for tid in trace_ids:
                step_idx = mapping.get(tid)
                if step_idx is not None:
                    # Find the trace with this id and get the step.
                    trace_obj = next(t for t in traces if t.trace_id == tid)
                    steps[tid] = trace_obj.steps[step_idx]
                else:
                    steps[tid] = None

            alignment_type = self._classify_position(steps)
            non_none = [s for s in steps.values() if s is not None]
            confidence = len(non_none) / len(trace_ids) if trace_ids else 1.0
            if alignment_type == AlignmentType.DETERMINISTIC:
                total_matched += 1

            positions.append(AlignedPosition(
                index=pos_idx,
                alignment_type=alignment_type,
                steps=steps,
                confidence=confidence,
            ))

        alignment_score = (
            total_matched / total_positions if total_positions > 0 else 1.0
        )

        return AlignedTraceSet(
            positions=positions,
            trace_ids=trace_ids,
            alignment_score=alignment_score,
        )

    def merge(
        self,
        existing: AlignedTraceSet,
        new_traces: List[ExecutionTrace],
    ) -> AlignedTraceSet:
        """
        Merge new traces into an existing alignment for iterative refinement.

        Reconstructs ExecutionTraces from the existing alignment, combines
        them with the new traces, and re-aligns everything.
        """
        if not new_traces:
            return existing

        # Reconstruct traces from existing alignment.
        existing_traces = self._reconstruct_traces(existing)
        all_traces = existing_traces + list(new_traces)
        return self.align(all_traces)

    # ------------------------------------------------------------------
    # Pairwise LCS alignment
    # ------------------------------------------------------------------

    def _pairwise_align(
        self,
        trace_a: ExecutionTrace,
        trace_b: ExecutionTrace,
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Align two traces using LCS-based sequence alignment.

        Returns list of ``(index_a, index_b)`` pairs where ``None``
        indicates a gap (insertion/deletion).
        """
        steps_a = trace_a.steps
        steps_b = trace_b.steps
        n = len(steps_a)
        m = len(steps_b)

        # Build LCS table.
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if self._steps_equivalent(steps_a[i - 1], steps_b[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Backtrack to produce the alignment.
        alignment: List[Tuple[Optional[int], Optional[int]]] = []
        i, j = n, m
        while i > 0 or j > 0:
            if (
                i > 0
                and j > 0
                and self._steps_equivalent(steps_a[i - 1], steps_b[j - 1])
            ):
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
                alignment.append((None, j - 1))
                j -= 1
            else:
                alignment.append((i - 1, None))
                i -= 1

        alignment.reverse()
        return alignment

    # ------------------------------------------------------------------
    # Step equivalence
    # ------------------------------------------------------------------

    def _steps_equivalent(self, step_a: TraceStep, step_b: TraceStep) -> bool:
        """
        Determine if two steps are equivalent for alignment purposes.

        Two steps are equivalent if they have the same action type AND
        their targets refer to the same element. Since ``__id__`` values
        are session-specific, equivalence is determined by (tried in order):

        1. Same action_type (required).
        2. Target equivalence via one of:
           a. Matching stable selectors (id, css, xpath) if available.
           b. Matching HTML element structure (tag + attributes).
           c. Matching target description / reasoning text.
           d. Contextual fallback — same ``action_group_index``.
        """
        if step_a.action_type != step_b.action_type:
            return False

        # If both targets are None (e.g. visit_url with URL as target string
        # or wait actions), consider them equivalent on action_type alone.
        if step_a.target is None and step_b.target is None:
            return True

        # Try stable selector comparison.
        if self._targets_match_by_selectors(step_a.target, step_b.target):
            return True

        # Try HTML element structure comparison.
        if self._targets_match_by_html(step_a, step_b):
            return True

        # Try reasoning / description text comparison.
        if self._targets_match_by_reasoning(step_a, step_b):
            return True

        # Contextual fallback: same action_group_index within their turns.
        # This is a weak signal — only use it when both targets are plain
        # strings that look like URLs or simple values (not selectors).
        if (
            step_a.action_group_index == step_b.action_group_index
            and isinstance(step_a.target, str)
            and isinstance(step_b.target, str)
            and step_a.target == step_b.target
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Target equivalence strategies
    # ------------------------------------------------------------------

    def _targets_match_by_selectors(
        self, target_a: Any, target_b: Any,
    ) -> bool:
        """
        Strategy (a): Compare targets by stable selectors.

        Works when targets are TargetSpec, TargetSpecWithFallback, dicts
        with ``strategy``/``value`` keys, or dicts with ``strategies`` list.
        """
        selectors_a = self._extract_selectors(target_a)
        selectors_b = self._extract_selectors(target_b)

        if not selectors_a or not selectors_b:
            return False

        # If any selector pair matches, the targets are equivalent.
        return bool(selectors_a & selectors_b)

    def _targets_match_by_html(
        self, step_a: TraceStep, step_b: TraceStep,
    ) -> bool:
        """
        Strategy (b): Compare targets by HTML element structure.

        Extracts the tag name and key attributes from each step's
        ``html_before`` (or ``html_after``) and checks for a match.
        """
        sig_a = self._element_signature(step_a)
        sig_b = self._element_signature(step_b)

        if sig_a is None or sig_b is None:
            return False

        return sig_a == sig_b

    def _targets_match_by_reasoning(
        self, step_a: TraceStep, step_b: TraceStep,
    ) -> bool:
        """
        Strategy (c): Compare targets by reasoning / description text.

        If both steps have non-empty reasoning text, check if they are
        substantially similar (exact match after normalisation).
        """
        text_a = self._get_target_text(step_a)
        text_b = self._get_target_text(step_b)

        if not text_a or not text_b:
            return False

        return _normalize_text(text_a) == _normalize_text(text_b)

    # ------------------------------------------------------------------
    # Position classification
    # ------------------------------------------------------------------

    def _classify_position(
        self,
        steps: Dict[str, Optional[TraceStep]],
    ) -> AlignmentType:
        """
        Classify an aligned position based on the steps present.

        Rules (from requirements 4.3–4.6):
        - All non-None same type + target + args → DETERMINISTIC
        - All non-None same type + target, different args → PARAMETERIZABLE
        - Different type or target among non-None → VARIABLE
        - Some None (gaps) + all non-None equivalent → OPTIONAL
        - Some None (gaps) + non-None differ → VARIABLE
        """
        non_none = [s for s in steps.values() if s is not None]
        has_gaps = len(non_none) < len(steps)

        if not non_none:
            # All gaps — shouldn't normally happen, treat as variable.
            return AlignmentType.VARIABLE

        # Check if all non-None steps share action_type and target.
        reference = non_none[0]
        all_same_type_target = all(
            self._steps_equivalent(reference, s) for s in non_none[1:]
        )

        if not all_same_type_target:
            return AlignmentType.VARIABLE

        # All non-None steps have equivalent type + target.
        # Now check args.
        all_same_args = all(
            _args_equal(reference.args, s.args) for s in non_none[1:]
        )

        if has_gaps:
            if all_same_args:
                return AlignmentType.OPTIONAL
            else:
                return AlignmentType.VARIABLE

        if all_same_args:
            return AlignmentType.DETERMINISTIC
        else:
            return AlignmentType.PARAMETERIZABLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_trace_positions(
        self, trace: ExecutionTrace,
    ) -> List[AlignedPosition]:
        """Build positions for a single trace (all DETERMINISTIC)."""
        positions: List[AlignedPosition] = []
        for i, step in enumerate(trace.steps):
            positions.append(AlignedPosition(
                index=i,
                alignment_type=AlignmentType.DETERMINISTIC,
                steps={trace.trace_id: step},
                confidence=1.0,
            ))
        return positions

    def _fold_trace(
        self,
        multi: List[Dict[str, Optional[int]]],
        traces: List[ExecutionTrace],
        new_trace: ExecutionTrace,
    ) -> List[Dict[str, Optional[int]]]:
        """
        Fold a new trace into an existing multi-alignment.

        Creates a synthetic "reference" step list from the multi-alignment
        (using the first non-None step at each position), then pairwise-
        aligns the new trace against it.
        """
        # Build a reference step list from the multi-alignment.
        ref_steps: List[TraceStep] = []
        ref_indices: List[int] = []  # index into multi
        for pos_idx, mapping in enumerate(multi):
            # Pick the first non-None step.
            step = None
            for tid, sidx in mapping.items():
                if sidx is not None:
                    trace_obj = next(t for t in traces if t.trace_id == tid)
                    step = trace_obj.steps[sidx]
                    break
            if step is not None:
                ref_steps.append(step)
                ref_indices.append(pos_idx)

        ref_trace = ExecutionTrace(
            trace_id="__ref__",
            task_description="",
            steps=ref_steps,
        )

        alignment = self._pairwise_align(ref_trace, new_trace)

        new_multi: List[Dict[str, Optional[int]]] = []
        used_ref_positions: Set[int] = set()

        for ref_idx, new_idx in alignment:
            if ref_idx is not None:
                # This aligns to an existing position in multi.
                multi_pos = ref_indices[ref_idx]
                used_ref_positions.add(multi_pos)
                row = dict(multi[multi_pos])
                row[new_trace.trace_id] = new_idx
                new_multi.append(row)
            else:
                # Gap in reference — new trace has an extra step.
                row: Dict[str, Optional[int]] = {}
                for m in multi:
                    for tid in m:
                        if tid not in row:
                            row[tid] = None
                row[new_trace.trace_id] = new_idx
                new_multi.append(row)

        # Add any multi positions that weren't used (gaps in new trace).
        for pos_idx, mapping in enumerate(multi):
            if pos_idx not in used_ref_positions:
                row = dict(mapping)
                row[new_trace.trace_id] = None
                new_multi.append(row)

        # Sort by the original multi position order where possible,
        # keeping insertions in their alignment order.
        # We rely on the alignment order being correct already.
        return new_multi

    def _reconstruct_traces(
        self, aligned: AlignedTraceSet,
    ) -> List[ExecutionTrace]:
        """
        Reconstruct ExecutionTrace objects from an AlignedTraceSet.

        Used by ``merge()`` to recover the original traces before
        re-aligning with new traces.
        """
        traces_map: Dict[str, List[TraceStep]] = {
            tid: [] for tid in aligned.trace_ids
        }

        for pos in aligned.positions:
            for tid in aligned.trace_ids:
                step = pos.steps.get(tid)
                if step is not None:
                    traces_map[tid].append(step)

        return [
            ExecutionTrace(
                trace_id=tid,
                task_description="",
                steps=steps,
            )
            for tid, steps in traces_map.items()
        ]

    # ------------------------------------------------------------------
    # Selector extraction
    # ------------------------------------------------------------------

    def _extract_selectors(self, target: Any) -> Set[Tuple[str, str]]:
        """
        Extract ``(strategy, value)`` pairs from a target.

        Handles TargetSpec, TargetSpecWithFallback, and dict
        representations.
        """
        pairs: Set[Tuple[str, str]] = set()

        if target is None:
            return pairs

        # TargetSpecWithFallback or dict with "strategies" key.
        strategies = getattr(target, "strategies", None)
        if strategies is None and isinstance(target, dict):
            strategies = target.get("strategies")

        if strategies is not None:
            for spec in strategies:
                s = getattr(spec, "strategy", None)
                v = getattr(spec, "value", None)
                if s is None and isinstance(spec, dict):
                    s = spec.get("strategy")
                    v = spec.get("value")
                if s and v and s != "agent":
                    pairs.add((s, v))
            return pairs

        # Single TargetSpec or dict with "strategy"/"value".
        s = getattr(target, "strategy", None)
        v = getattr(target, "value", None)
        if s is None and isinstance(target, dict):
            s = target.get("strategy")
            v = target.get("value")
        if s and v and s != "agent":
            pairs.add((s, v))

        return pairs

    def _element_signature(
        self, step: TraceStep,
    ) -> Optional[Tuple[str, ...]]:
        """
        Extract a structural signature from the step's target element.

        Uses ``html_before`` (or ``html_after``) to find the target
        element and extract its tag name and key attributes.
        """
        html = step.html_before or step.html_after
        if not html:
            return None

        target = step.target
        framework_id = self._extract_framework_id(target)
        if not framework_id:
            return None

        element_html = self._find_element_in_html(html, framework_id)
        if not element_html:
            return None

        tag = _parse_tag_name(element_html)
        attrs = _parse_attrs(element_html)

        # Build a signature from tag + stable attributes.
        sig_parts: List[str] = [tag or "unknown"]
        for attr_name in sorted(attrs):
            if attr_name.startswith("__"):
                continue  # Skip framework-internal attributes.
            sig_parts.append(f"{attr_name}={attrs[attr_name]}")

        return tuple(sig_parts)

    def _find_element_in_html(
        self, html: str, framework_id: str,
    ) -> Optional[str]:
        """Find an element by ``__id__`` attribute in HTML."""
        if _HAS_BS4:
            try:
                soup = BeautifulSoup(html, "html.parser")
                el = soup.find(attrs={"__id__": framework_id})
                if el and isinstance(el, Tag):
                    # Return just the opening tag for signature extraction.
                    return str(el)
            except Exception:
                pass
            return None

        # Regex fallback.
        pattern = re.compile(
            r'<[^>]*__id__\s*=\s*["\']'
            + re.escape(framework_id)
            + r'["\'][^>]*>',
            re.IGNORECASE,
        )
        match = pattern.search(html)
        return match.group(0) if match else None

    @staticmethod
    def _extract_framework_id(target: Any) -> Optional[str]:
        """Extract ``__id__`` value from a target specification."""
        if target is None:
            return None

        # TargetSpecWithFallback — look for __id__ strategy.
        strategies = getattr(target, "strategies", None)
        if strategies is None and isinstance(target, dict):
            strategies = target.get("strategies")
        if strategies:
            for spec in strategies:
                s = getattr(spec, "strategy", None)
                v = getattr(spec, "value", None)
                if s is None and isinstance(spec, dict):
                    s = spec.get("strategy")
                    v = spec.get("value")
                if s == "__id__":
                    return v

        # Single TargetSpec.
        s = getattr(target, "strategy", None)
        v = getattr(target, "value", None)
        if s is None and isinstance(target, dict):
            s = target.get("strategy")
            v = target.get("value")
        if s == "__id__":
            return v

        # Plain string that looks like an __id__.
        if isinstance(target, str) and target.startswith("__id__:"):
            return target[len("__id__:"):]

        return None

    def _get_target_text(self, step: TraceStep) -> Optional[str]:
        """
        Get descriptive text for a step's target.

        Checks reasoning, then string target, then agent-strategy value.
        """
        if step.reasoning:
            return step.reasoning

        if isinstance(step.target, str):
            return step.target

        # Check for agent strategy in TargetSpecWithFallback.
        strategies = getattr(step.target, "strategies", None)
        if strategies is None and isinstance(step.target, dict):
            strategies = step.target.get("strategies")
        if strategies:
            for spec in strategies:
                s = getattr(spec, "strategy", None)
                v = getattr(spec, "value", None)
                if s is None and isinstance(spec, dict):
                    s = spec.get("strategy")
                    v = spec.get("value")
                if s == "agent" and v:
                    return v

        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy text comparison."""
    return " ".join(text.lower().split())


def _args_equal(
    args_a: Optional[Dict[str, Any]],
    args_b: Optional[Dict[str, Any]],
) -> bool:
    """Compare two argument dicts for equality."""
    if args_a is None and args_b is None:
        return True
    if args_a is None or args_b is None:
        return False
    return args_a == args_b


def _parse_attrs(element_html: str) -> Dict[str, str]:
    """Extract attributes from an HTML element's opening tag."""
    attrs: Dict[str, str] = {}
    for match in _ATTR_RE.finditer(element_html):
        name = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        attrs[name] = value
    return attrs


def _parse_tag_name(element_html: str) -> Optional[str]:
    """Extract the tag name from an HTML element string."""
    m = _TAG_RE.match(element_html)
    return m.group(1).lower() if m else None

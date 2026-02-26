"""
Graph Synthesizer for the Meta Agent Workflow pipeline.

Converts :class:`ExtractedPatterns` into an :class:`ActionGraph` using the
existing fluent API.  Supports three synthesis strategies via a strategy
pattern hierarchy:

- **RuleBasedSynthesizer** — deterministic rules for all pattern types (default)
- **LLMSynthesizer** — uses :class:`InferencerBase` for all synthesis decisions
- **HybridSynthesizer** — rules for clear patterns, LLM for ambiguous ones

Each pattern category maps to a specific graph construct:

- Deterministic steps → ``graph.action(type, target=..., args=...)``
- Parameterizable steps → actions with ``{template_var}`` placeholders
- Variable steps → Agent Node actions (agent-as-action pattern)
- Optional steps → actions with ``no_action_if_target_not_found=True``
- User input boundaries → ``graph.action("wait", target=True)``
- Branch points → ``graph.branch(...)`` with placeholder conditions
- Loop patterns → ``graph.loop(...)`` constructs
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rich_python_utils.string_utils.formatting.template_manager import (
        TemplateManager,
    )

logger = logging.getLogger(__name__)

from science_modeling_tools.automation.meta_agent.errors import GraphSynthesisError
from science_modeling_tools.automation.meta_agent.models import (
    AlignedPosition,
    AlignmentType,
    BranchPattern,
    ExtractedPatterns,
    LoopPattern,
    ParameterizableInfo,
    SynthesisReport,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.target_converter import (
    TargetSpec,
    TargetSpecWithFallback,
)


# ---------------------------------------------------------------------------
# Template variable regex — matches ``{var_name}`` placeholders.
# ---------------------------------------------------------------------------
_TEMPLATE_VAR_RE = re.compile(r"\{(\w+)\}")


# ---------------------------------------------------------------------------
# Strategy enum and decision dataclasses
# ---------------------------------------------------------------------------


class SynthesisStrategy(Enum):
    """Strategy for graph synthesis."""

    RULE_BASED = "rule_based"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class ActionDecision:
    """Decision made by the synthesizer for a single pattern position.

    Records what action was synthesized and why, for auditability.
    """

    position_index: int
    action_type: str
    target: Optional[Any] = None
    args: Optional[Dict[str, Any]] = None
    decision_source: str = "rule"  # "rule", "llm", or "hybrid"
    confidence: float = 1.0
    reasoning: Optional[str] = None


@dataclass
class SynthesisResult:
    """Extended synthesis result including per-action decisions."""

    graph: Any  # ActionGraph
    report: SynthesisReport
    decisions: List[ActionDecision] = field(default_factory=list)
    python_script: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class GraphSynthesizer(ABC):
    """
    Abstract base for graph synthesis.  Converts extracted patterns into an
    ActionGraph using the ActionGraph fluent API.

    Subclasses implement ``_decide_action`` to determine how each pattern
    position is translated into a graph action.  The base class handles
    the common graph construction logic (loops, branches, user input
    boundaries) and delegates per-position action decisions to the
    subclass.
    """

    def __init__(
        self,
        action_executor: Any,
        action_metadata: Optional[Any] = None,
        agent_action_type: str = "meta_workflow_agent",
        prompt_formatter: Optional["TemplateManager"] = None,
    ):
        self._action_executor = action_executor
        self._action_metadata = action_metadata
        self._agent_action_type = agent_action_type
        self._prompt_formatter = prompt_formatter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        patterns: ExtractedPatterns,
        task_description: str = "",
    ) -> SynthesisResult:
        """
        Synthesize an ActionGraph from extracted patterns.

        Returns a :class:`SynthesisResult` with the ActionGraph,
        :class:`SynthesisReport`, and per-action decisions.
        """
        from science_modeling_tools.automation.schema.action_graph import ActionGraph
        from science_modeling_tools.automation.schema.action_metadata import (
            ActionMetadataRegistry,
        )

        metadata = self._action_metadata or ActionMetadataRegistry()

        graph = ActionGraph(
            action_executor=self._action_executor,
            action_metadata=metadata,
        )

        # Track counts for the report.
        deterministic_count = 0
        parameterizable_count = 0
        agent_node_count = 0
        optional_count = 0
        user_input_boundary_count = 0
        branch_count = 0
        loop_count = 0
        target_strategy_coverage: Dict[str, int] = {}
        template_variables: List[str] = []
        warnings: List[str] = []
        decisions: List[ActionDecision] = []

        # Build indices for quick lookup.
        loop_indices: Set[int] = set()
        for lp in patterns.loop_patterns:
            loop_indices.update(range(lp.body_start, lp.body_end + 1))

        branch_indices: Set[int] = set()
        for bp in patterns.branch_patterns:
            branch_indices.add(bp.branch_point_index)

        param_map: Dict[int, ParameterizableInfo] = {
            pos.index: info for pos, info in patterns.parameterizable_steps
        }

        uib_set: Set[int] = set(patterns.user_input_boundaries)

        # Synthesize loops first (they span multiple positions).
        for lp in patterns.loop_patterns:
            try:
                self._synthesize_loop(graph, lp)
                loop_count += 1
            except Exception as exc:
                warnings.append(f"Loop at {lp.body_start}-{lp.body_end}: {exc}")

        # Synthesize branches.
        for bp in patterns.branch_patterns:
            try:
                self._synthesize_branch(graph, bp)
                branch_count += 1
            except Exception as exc:
                warnings.append(
                    f"Branch at index {bp.branch_point_index}: {exc}"
                )

        # Walk step_order for the remaining positions.
        for idx in patterns.step_order:
            if idx in loop_indices or idx in branch_indices:
                continue

            # User input boundary.
            if idx in uib_set:
                self._synthesize_user_input_boundary(graph)
                user_input_boundary_count += 1
                decisions.append(ActionDecision(
                    position_index=idx,
                    action_type="wait",
                    target=True,
                    decision_source=self._decision_source,
                ))
                continue

            # Deterministic.
            det_pos = self._find_position(patterns.deterministic_steps, idx)
            if det_pos is not None:
                decision = self._decide_action(det_pos, {
                    "task_description": task_description,
                    "pattern_type": "deterministic",
                })
                decisions.append(decision)
                self._apply_decision_to_graph(graph, decision, "deterministic")
                deterministic_count += 1
                self._collect_target_strategies(det_pos, target_strategy_coverage)
                continue

            # Parameterizable.
            param_pos = self._find_position(
                [p for p, _ in patterns.parameterizable_steps], idx
            )
            if param_pos is not None:
                info = param_map[idx]
                decision = self._decide_action(param_pos, {
                    "task_description": task_description,
                    "pattern_type": "parameterizable",
                    "param_info": info,
                })
                decisions.append(decision)
                self._apply_decision_to_graph(graph, decision, "parameterizable")
                parameterizable_count += 1
                template_variables.extend(info.variable_args.values())
                self._collect_target_strategies(param_pos, target_strategy_coverage)
                continue

            # Variable.
            var_pos = self._find_position(patterns.variable_steps, idx)
            if var_pos is not None:
                decision = self._decide_action(var_pos, {
                    "task_description": task_description,
                    "pattern_type": "variable",
                })
                decisions.append(decision)
                self._apply_decision_to_graph(graph, decision, "variable")
                agent_node_count += 1
                continue

            # Optional.
            opt_pos = self._find_position(patterns.optional_steps, idx)
            if opt_pos is not None:
                decision = self._decide_action(opt_pos, {
                    "task_description": task_description,
                    "pattern_type": "optional",
                })
                decisions.append(decision)
                self._apply_decision_to_graph(graph, decision, "optional")
                optional_count += 1
                self._collect_target_strategies(opt_pos, target_strategy_coverage)
                continue

        total_steps = (
            deterministic_count
            + parameterizable_count
            + agent_node_count
            + optional_count
            + user_input_boundary_count
            + branch_count
            + loop_count
        )

        report = SynthesisReport(
            total_steps=total_steps,
            deterministic_count=deterministic_count,
            parameterizable_count=parameterizable_count,
            agent_node_count=agent_node_count,
            optional_count=optional_count,
            user_input_boundary_count=user_input_boundary_count,
            branch_count=branch_count,
            loop_count=loop_count,
            synthesis_strategy=self._synthesis_strategy_name,
            target_strategy_coverage=target_strategy_coverage,
            template_variables=sorted(set(template_variables)),
            warnings=warnings,
        )

        return SynthesisResult(graph=graph, report=report, decisions=decisions)

    # ------------------------------------------------------------------
    # Abstract method — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def _decide_action(
        self,
        position: AlignedPosition,
        context: Dict[str, Any],
    ) -> ActionDecision:
        """
        Decide what action to synthesize for a given position.

        Subclasses implement this to provide strategy-specific logic.

        Args:
            position: The aligned position to synthesize
            context: Additional context (task_description, pattern_type, etc.)

        Returns:
            ActionDecision describing the synthesized action
        """
        ...

    # ------------------------------------------------------------------
    # Strategy identity — overridden by subclasses
    # ------------------------------------------------------------------

    @property
    def _synthesis_strategy_name(self) -> str:
        """Return the strategy name for the SynthesisReport."""
        return "rule_based"

    @property
    def _decision_source(self) -> str:
        """Return the default decision_source for ActionDecision."""
        return "rule"

    # ------------------------------------------------------------------
    # Per-pattern synthesis helpers
    # ------------------------------------------------------------------

    def _synthesize_deterministic(
        self,
        graph: Any,
        position: AlignedPosition,
    ) -> None:
        """Add a deterministic action to the graph."""
        step = self._representative_step(position)
        if step is None:
            return

        target = self._to_graph_target(step.target)
        graph.action(
            action_type=step.action_type,
            target=target,
            args=step.args,
        )

    def _synthesize_parameterizable(
        self,
        graph: Any,
        position: AlignedPosition,
        param_info: ParameterizableInfo,
    ) -> None:
        """Add a parameterizable action with template variable placeholders."""
        step = self._representative_step(position)
        if step is None:
            return

        # Build args with template placeholders for variable args.
        merged_args: Dict[str, Any] = dict(param_info.constant_args)
        for arg_key, var_name in param_info.variable_args.items():
            merged_args[arg_key] = "{" + var_name + "}"

        target = self._to_graph_target(step.target)
        graph.action(
            action_type=step.action_type,
            target=target,
            args=merged_args if merged_args else None,
        )

    def _synthesize_variable(
        self,
        graph: Any,
        position: AlignedPosition,
    ) -> None:
        """Add an Agent Node for a variable step."""
        step = self._representative_step(position)

        # Build a task description from observed variants.
        description = self._build_variable_description(position, step)

        graph.action(
            action_type=self._agent_action_type,
            target=description,
        )

    def _synthesize_optional(
        self,
        graph: Any,
        position: AlignedPosition,
    ) -> None:
        """Add an optional action with ``no_action_if_target_not_found=True``."""
        step = self._representative_step(position)
        if step is None:
            return

        target = self._to_graph_target(step.target)
        graph.action(
            action_type=step.action_type,
            target=target,
            args=step.args,
            no_action_if_target_not_found=True,
        )

    def _synthesize_user_input_boundary(
        self,
        graph: Any,
    ) -> None:
        """Add a ``wait(True)`` human confirmation gate."""
        graph.action(
            action_type="wait",
            target=True,
        )

    def _synthesize_branch(
        self,
        graph: Any,
        branch: BranchPattern,
    ) -> None:
        """Add a branch to the graph using the callback-based API.

        Since automated condition inference is deferred, we use a
        placeholder condition that always selects the first branch.
        """
        branch_labels = list(branch.branches.keys())
        if not branch_labels:
            return

        first_label = branch_labels[0]

        def _placeholder_condition(result: Any, **kwargs: Any) -> bool:
            """Placeholder — always takes the first branch."""
            return True

        # Build the first branch as if_true.
        def _build_first(g: Any) -> None:
            for pos in branch.branches[first_label]:
                step = self._representative_step(pos)
                if step is not None:
                    g.action(
                        action_type=step.action_type,
                        target=self._to_graph_target(step.target),
                        args=step.args,
                    )

        # Build else branch with remaining labels combined.
        remaining_positions: List[AlignedPosition] = []
        for label in branch_labels[1:]:
            remaining_positions.extend(branch.branches[label])

        if remaining_positions:

            def _build_else(g: Any) -> None:
                for pos in remaining_positions:
                    step = self._representative_step(pos)
                    if step is not None:
                        g.action(
                            action_type=step.action_type,
                            target=self._to_graph_target(step.target),
                            args=step.args,
                        )

            graph.branch(
                condition=_placeholder_condition,
                if_true=_build_first,
                if_false=_build_else,
            )
        else:
            graph.branch(
                condition=_placeholder_condition,
                if_true=_build_first,
            )

    def _synthesize_loop(
        self,
        graph: Any,
        loop: LoopPattern,
    ) -> None:
        """Add a loop construct to the graph.

        Uses a simple iteration-count condition based on the detected
        max_iterations.
        """
        max_iter = loop.max_iterations

        iteration_state = {"count": 0}

        def _loop_condition(result: Any, _s: dict = iteration_state, _m: int = max_iter, **kw: Any) -> bool:
            return _s["count"] < _m

        def _loop_advance(result: Any, _s: dict = iteration_state, **kw: Any) -> Any:
            _s["count"] += 1
            return result

        graph.loop(
            condition=_loop_condition,
            max_loop=max_iter,
            advance=_loop_advance,
        )

        # Add the loop body actions to the current node.
        for body_pos in loop.body_steps:
            step = self._representative_step(body_pos)
            if step is not None:
                graph.action(
                    action_type=step.action_type,
                    target=self._to_graph_target(step.target),
                    args=step.args,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _representative_step(position: AlignedPosition) -> Optional[TraceStep]:
        """Return the first non-None step from an aligned position."""
        for step in position.steps.values():
            if step is not None:
                return step
        return None

    @staticmethod
    def _find_position(
        positions: List[AlignedPosition],
        idx: int,
    ) -> Optional[AlignedPosition]:
        """Find a position by index in a list."""
        for pos in positions:
            if pos.index == idx:
                return pos
        return None

    @staticmethod
    def _to_graph_target(
        target: Any,
    ) -> Any:
        """Convert a meta-agent target to an ActionGraph-compatible target.

        :class:`TargetSpecWithFallback` and :class:`TargetSpec` from the
        meta-agent module are lightweight local dataclasses.  The ActionGraph
        uses its own pydantic-based models.  We convert here so the graph
        receives the right types.
        """
        if target is None:
            return None

        if isinstance(target, TargetSpecWithFallback):
            try:
                from science_modeling_tools.automation.schema.common import (
                    TargetSpecWithFallback as GraphTargetSpecWithFallback,
                    TargetSpec as GraphTargetSpec,
                )

                strategies = [
                    GraphTargetSpec(strategy=s.strategy, value=s.value)
                    for s in target.strategies
                ]
                return GraphTargetSpecWithFallback(strategies=strategies)
            except Exception:
                if target.strategies:
                    return target.strategies[0].value
                return None

        if isinstance(target, TargetSpec):
            try:
                from science_modeling_tools.automation.schema.common import (
                    TargetSpec as GraphTargetSpec,
                )

                return GraphTargetSpec(strategy=target.strategy, value=target.value)
            except Exception:
                return target.value

        # Already a graph-compatible type or plain string/bool/int.
        return target

    @staticmethod
    def _build_variable_description(
        position: AlignedPosition,
        step: Optional[TraceStep],
    ) -> str:
        """Build a human-readable task description for an Agent Node."""
        variants = {}
        if step is not None:
            variants = step.metadata.get("variants", {})

        if variants:
            parts = [f"{atype} (×{count})" for atype, count in variants.items()]
            return f"Variable step at position {position.index}: observed {', '.join(parts)}"

        return f"Variable step at position {position.index}"

    @staticmethod
    def _collect_target_strategies(
        position: AlignedPosition,
        coverage: Dict[str, int],
    ) -> None:
        """Accumulate target strategy counts into *coverage*."""
        for step in position.steps.values():
            if step is None or step.target is None:
                continue
            target = step.target
            if isinstance(target, TargetSpecWithFallback):
                for spec in target.strategies:
                    coverage[spec.strategy] = coverage.get(spec.strategy, 0) + 1
            elif isinstance(target, TargetSpec):
                coverage[target.strategy] = coverage.get(target.strategy, 0) + 1
            break

    # ------------------------------------------------------------------
    # LLM prompt construction and response parsing
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(
        self,
        position: AlignedPosition,
        context: Dict[str, Any],
    ) -> str:
        """Build a structured prompt for LLM-assisted synthesis decisions.

        When a ``prompt_formatter`` (TemplateManager) was provided at
        construction time, renders the synthesis template with it.
        Otherwise falls back to the original inline f-string for
        bit-identical backward compatibility.
        """
        from science_modeling_tools.automation.meta_agent.prompt_templates import (
            build_synthesis_feed,
            SYNTHESIS_TEMPLATE_KEY,
        )

        feed = build_synthesis_feed(position, context)

        if self._prompt_formatter is not None:
            return self._prompt_formatter(SYNTHESIS_TEMPLATE_KEY, **feed)

        # Legacy fallback — exact original f-string output.
        return (
            f"Synthesize an ActionGraph action for a {feed['pattern_type']} pattern.\n"
            f"Task: {feed['task_description']}\n"
            f"Position index: {feed['position_index']}\n"
            f"Observed steps across traces:\n{feed['steps_text']}\n"
            f"{feed['param_section']}\n"
            "Respond with a JSON object containing:\n"
            '  "action_type": string (the action type to use),\n'
            '  "target": string or null (the target element or description),\n'
            '  "args": object or null (action arguments),\n'
            '  "confidence": float between 0.0 and 1.0,\n'
            '  "reasoning": string (explanation of your decision)\n\n'
            'Example: {"action_type": "click", "target": "btn-submit", '
            '"args": null, "confidence": 0.9, '
            '"reasoning": "All traces show a click on submit button"}'
        )

    @staticmethod
    def _parse_decision(response: Any) -> Dict[str, Any]:
        """Parse an LLM synthesis response into structured decision fields.

        Handles string responses (JSON or plain text), dict responses, and
        gracefully falls back to empty overrides on parse failure.

        Returns a dict with keys: action_type, target, args, confidence,
        reasoning.  A ``None`` value for ``action_type`` signals the caller
        to use rule-based defaults.
        """
        result: Dict[str, Any] = {
            "action_type": None,
            "target": None,
            "args": None,
            "confidence": None,
            "reasoning": None,
        }

        if isinstance(response, dict):
            result["action_type"] = response.get("action_type")
            result["target"] = response.get("target")
            result["args"] = response.get("args")
            raw_conf = response.get("confidence")
            if raw_conf is not None:
                try:
                    result["confidence"] = max(0.0, min(1.0, float(raw_conf)))
                except (ValueError, TypeError):
                    pass
            result["reasoning"] = response.get("reasoning")
            return result

        text = str(response).strip()

        # Try JSON parse first.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                result["action_type"] = parsed.get("action_type")
                result["target"] = parsed.get("target")
                result["args"] = parsed.get("args")
                raw_conf = parsed.get("confidence")
                if raw_conf is not None:
                    try:
                        result["confidence"] = max(0.0, min(1.0, float(raw_conf)))
                    except (ValueError, TypeError):
                        pass
                result["reasoning"] = parsed.get("reasoning")
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Plain text fallback: treat the entire response as reasoning.
        # action_type remains None, signaling rule-based fallback.
        logger.debug("LLM response is not structured JSON, using as reasoning: %s", text)
        result["reasoning"] = text if text else None
        return result

    # ------------------------------------------------------------------
    # Decision-driven graph construction
    # ------------------------------------------------------------------

    def _apply_decision_to_graph(
        self,
        graph: Any,
        decision: ActionDecision,
        pattern_type: str,
    ) -> None:
        """Add an action to the graph based on an ActionDecision.

        This method is the single point where ActionDecision values drive
        graph construction for deterministic, parameterizable, variable,
        and optional pattern types.  Branch and loop patterns continue
        to use their dedicated helpers.
        """
        if decision.action_type is None or decision.action_type == "unknown":
            return

        target = self._to_graph_target(decision.target)

        kwargs: Dict[str, Any] = {}
        if pattern_type == "optional":
            kwargs["no_action_if_target_not_found"] = True

        graph.action(
            action_type=decision.action_type,
            target=target,
            args=decision.args,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Concrete strategy: Rule-Based
# ---------------------------------------------------------------------------


class RuleBasedSynthesizer(GraphSynthesizer):
    """
    Deterministic rule-based synthesis (default strategy).

    Uses fixed rules to map pattern types to ActionGraph constructs.
    No LLM involvement — all decisions are deterministic based on
    the pattern classification from the extraction stage.
    """

    @property
    def _synthesis_strategy_name(self) -> str:
        return SynthesisStrategy.RULE_BASED.value

    @property
    def _decision_source(self) -> str:
        return "rule"

    def _decide_action(
        self,
        position: AlignedPosition,
        context: Dict[str, Any],
    ) -> ActionDecision:
        """Apply deterministic rules to decide the action."""
        step = self._representative_step(position)
        action_type = step.action_type if step else "unknown"
        target = step.target if step else None
        args = step.args if step else None

        pattern_type = context.get("pattern_type", "unknown")

        if pattern_type == "variable":
            action_type = self._agent_action_type
            target = self._build_variable_description(position, step)
            args = None

        if pattern_type == "parameterizable":
            info = context.get("param_info")
            if info is not None:
                merged: Dict[str, Any] = dict(info.constant_args)
                for arg_key, var_name in info.variable_args.items():
                    merged[arg_key] = "{" + var_name + "}"
                args = merged if merged else None

        return ActionDecision(
            position_index=position.index,
            action_type=action_type,
            target=target,
            args=args,
            decision_source="rule",
            confidence=1.0,
        )


# ---------------------------------------------------------------------------
# Concrete strategy: LLM
# ---------------------------------------------------------------------------


class LLMSynthesizer(GraphSynthesizer):
    """
    LLM-assisted synthesis using InferencerBase.

    Uses an InferencerBase instance to make synthesis decisions for ALL
    positions.  The LLM receives the pattern context and returns
    structured decisions.

    Requires an InferencerBase instance at construction time.
    """

    def __init__(
        self,
        action_executor: Any,
        inferencer: Any,  # InferencerBase
        action_metadata: Optional[Any] = None,
        agent_action_type: str = "meta_workflow_agent",
        prompt_formatter: Optional["TemplateManager"] = None,
    ):
        super().__init__(
            action_executor, action_metadata, agent_action_type, prompt_formatter,
        )
        if inferencer is None:
            raise ValueError("LLMSynthesizer requires an InferencerBase instance")
        self._inferencer = inferencer

    @property
    def _synthesis_strategy_name(self) -> str:
        return SynthesisStrategy.LLM.value

    @property
    def _decision_source(self) -> str:
        return "llm"

    def _decide_action(
        self,
        position: AlignedPosition,
        context: Dict[str, Any],
    ) -> ActionDecision:
        """Use InferencerBase for ALL positions.

        Constructs a rich prompt with observed steps from all traces,
        calls ``self._inferencer.infer()``, and parses the structured
        decision.  Falls back to rule-based defaults if the LLM response
        cannot be parsed into a structured decision.
        """
        # Compute rule-based defaults first (always available as fallback).
        step = self._representative_step(position)
        default_action_type = step.action_type if step else "unknown"
        default_target = step.target if step else None
        default_args = step.args if step else None
        pattern_type = context.get("pattern_type", "unknown")

        if pattern_type == "variable":
            default_action_type = self._agent_action_type
            default_target = self._build_variable_description(position, step)
            default_args = None

        if pattern_type == "parameterizable":
            info = context.get("param_info")
            if info is not None:
                merged: Dict[str, Any] = dict(info.constant_args)
                for arg_key, var_name in info.variable_args.items():
                    merged[arg_key] = "{" + var_name + "}"
                default_args = merged if merged else None

        # Build rich prompt and call inferencer.
        prompt = self._build_synthesis_prompt(position, context)

        reasoning = None
        confidence = 0.8
        action_type = default_action_type
        target = default_target
        args = default_args

        try:
            response = self._inferencer.infer(prompt)
            parsed = self._parse_decision(response)

            # Use parsed values only when action_type was successfully extracted.
            if parsed["action_type"] is not None:
                action_type = parsed["action_type"]
                target = parsed["target"]  # may be None, which is valid
                args = parsed["args"]

            if parsed["confidence"] is not None:
                confidence = parsed["confidence"]

            reasoning = parsed["reasoning"]

        except Exception:
            reasoning = "LLM call failed, using rule-based fallback"
            confidence = 0.5

        return ActionDecision(
            position_index=position.index,
            action_type=action_type,
            target=target,
            args=args,
            decision_source="llm",
            confidence=confidence,
            reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# Concrete strategy: Hybrid
# ---------------------------------------------------------------------------


class HybridSynthesizer(GraphSynthesizer):
    """
    Hybrid synthesis: rule-based for clear patterns, LLM for ambiguous ones.

    Uses deterministic rules for DETERMINISTIC, PARAMETERIZABLE, and
    OPTIONAL positions.  Falls back to InferencerBase for VARIABLE and
    BRANCH_POINT positions where the rule-based approach would produce
    a generic agent node.

    Requires an InferencerBase instance at construction time.
    """

    # Pattern types that are handled by rules (clear patterns).
    _RULE_PATTERNS = {"deterministic", "parameterizable", "optional"}

    def __init__(
        self,
        action_executor: Any,
        inferencer: Any,  # InferencerBase
        action_metadata: Optional[Any] = None,
        agent_action_type: str = "meta_workflow_agent",
        prompt_formatter: Optional["TemplateManager"] = None,
    ):
        super().__init__(
            action_executor, action_metadata, agent_action_type, prompt_formatter,
        )
        if inferencer is None:
            raise ValueError("HybridSynthesizer requires an InferencerBase instance")
        self._inferencer = inferencer

    @property
    def _synthesis_strategy_name(self) -> str:
        return SynthesisStrategy.HYBRID.value

    @property
    def _decision_source(self) -> str:
        return "hybrid"

    def _decide_action(
        self,
        position: AlignedPosition,
        context: Dict[str, Any],
    ) -> ActionDecision:
        """Use rules for clear patterns, LLM for ambiguous ones.

        DETERMINISTIC/PARAMETERIZABLE/OPTIONAL → rule-based (confidence=1.0)
        VARIABLE/BRANCH_POINT → LLM-assisted (confidence from LLM response)
        """
        step = self._representative_step(position)
        action_type = step.action_type if step else "unknown"
        target = step.target if step else None
        args = step.args if step else None
        pattern_type = context.get("pattern_type", "unknown")
        task_description = context.get("task_description", "")

        # Clear patterns → rule-based
        if pattern_type in self._RULE_PATTERNS:
            if pattern_type == "parameterizable":
                info = context.get("param_info")
                if info is not None:
                    merged: Dict[str, Any] = dict(info.constant_args)
                    for arg_key, var_name in info.variable_args.items():
                        merged[arg_key] = "{" + var_name + "}"
                    args = merged if merged else None

            return ActionDecision(
                position_index=position.index,
                action_type=action_type,
                target=target,
                args=args,
                decision_source="rule",
                confidence=1.0,
            )

        # Ambiguous patterns → LLM-assisted.
        # Compute rule-based defaults for fallback.
        if pattern_type == "variable":
            default_action_type = self._agent_action_type
            default_target = self._build_variable_description(position, step)
            default_args = None
        else:
            default_action_type = action_type
            default_target = target
            default_args = args

        prompt = self._build_synthesis_prompt(position, context)

        reasoning = None
        confidence = 0.8
        final_action_type = default_action_type
        final_target = default_target
        final_args = default_args

        try:
            response = self._inferencer.infer(prompt)
            parsed = self._parse_decision(response)

            if parsed["action_type"] is not None:
                final_action_type = parsed["action_type"]
                final_target = parsed["target"]
                final_args = parsed["args"]

            if parsed["confidence"] is not None:
                confidence = parsed["confidence"]

            reasoning = parsed["reasoning"]

        except Exception:
            reasoning = "LLM call failed, using rule-based fallback"
            confidence = 0.5

        return ActionDecision(
            position_index=position.index,
            action_type=final_action_type,
            target=final_target,
            args=final_args,
            decision_source="llm",
            confidence=confidence,
            reasoning=reasoning,
        )

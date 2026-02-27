"""
Graph Validator — validates a synthesized ActionGraph by executing it
and comparing results against expected outcomes from the original traces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent_foundation.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
    ValidationResult,
    ValidationResults,
)

logger = logging.getLogger(__name__)


class GraphValidator:
    """Validates a synthesized ActionGraph by executing it and comparing
    results against expected outcomes from the original traces.

    The validator iterates over test data inputs, executes the graph for
    each input, and compares the execution trace against the corresponding
    expected trace step-by-step.  Comparison is based on action types and
    target equivalence at each step index.

    Because ``ActionGraph.execute()`` requires a live WebDriver environment,
    the validator is designed so that ``_execute_graph`` can be overridden
    (or the graph can be pre-executed) for unit-testing scenarios.
    """

    def validate(
        self,
        graph: Any,  # ActionGraph
        task_description: str,
        test_data: Optional[List[Dict[str, Any]]] = None,
        expected_traces: Optional[List[ExecutionTrace]] = None,
    ) -> ValidationResults:
        """Validate the graph against test data and/or expected traces.

        For each entry in *test_data* (paired with the corresponding
        *expected_traces* entry), the graph is executed and the result
        is compared step-by-step against the expected trace.

        Args:
            graph: The synthesized ``ActionGraph`` to validate.
            task_description: Human-readable description of the task.
            test_data: List of input dicts, one per validation run.
                       Defaults to a single run with no variables.
            expected_traces: Expected execution traces to compare against.
                             Must be the same length as *test_data* when
                             both are provided.

        Returns:
            ``ValidationResults`` with per-input results and an overall
            ``success_rate``.
        """
        if test_data is None:
            test_data = [{}]

        if expected_traces is None:
            expected_traces = [None] * len(test_data)  # type: ignore[list-item]

        results: List[ValidationResult] = []
        for idx, (data, expected) in enumerate(zip(test_data, expected_traces)):
            result = self._validate_single(graph, data, expected, idx)
            results.append(result)

        return ValidationResults(results=results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_single(
        self,
        graph: Any,
        input_data: Dict[str, Any],
        expected_trace: Optional[ExecutionTrace],
        run_index: int,
    ) -> ValidationResult:
        """Run a single validation: execute the graph and compare."""
        execution_result = self._execute_graph(graph, input_data)

        if execution_result is None:
            return ValidationResult(
                input_data=input_data,
                passed=False,
                error="Graph execution returned None",
            )

        # If execution itself failed, report immediately.
        if not getattr(execution_result, "success", True):
            error_msg = str(getattr(execution_result, "error", "")) or "Execution failed"
            return ValidationResult(
                input_data=input_data,
                passed=False,
                error=error_msg,
            )

        # No expected trace to compare against — pass if execution succeeded.
        if expected_trace is None:
            return ValidationResult(input_data=input_data, passed=True)

        return self._compare_execution(execution_result, expected_trace)

    def _execute_graph(
        self,
        graph: Any,
        input_data: Dict[str, Any],
    ) -> Any:
        """Execute the graph with the given input data.

        Returns the ``ExecutionResult`` from ``graph.execute()``.
        Catches exceptions and returns ``None`` on failure so the
        caller can report a clean validation failure.
        """
        try:
            return graph.execute(initial_variables=input_data)
        except Exception as exc:
            logger.warning("Graph execution failed: %s", exc, exc_info=True)
            return None

    def _compare_execution(
        self,
        execution_result: Any,  # ExecutionResult
        expected_trace: ExecutionTrace,
    ) -> ValidationResult:
        """Compare a graph execution result against an expected trace.

        Walks the expected trace steps and compares each against the
        corresponding step in the execution result's outputs.  The
        comparison checks action type and target equivalence.

        Returns a ``ValidationResult`` with ``passed=True`` when all
        steps match, or ``passed=False`` with ``divergence_point`` set
        to the first step index where the execution diverged.
        """
        expected_steps = expected_trace.steps
        actual_steps = self._extract_steps_from_result(execution_result)

        input_data = expected_trace.input_data

        # Walk step-by-step and compare.
        min_len = min(len(expected_steps), len(actual_steps))
        for i in range(min_len):
            if not self._steps_match(expected_steps[i], actual_steps[i]):
                return ValidationResult(
                    input_data=input_data,
                    passed=False,
                    divergence_point=i,
                    expected_outcome=self._step_summary(expected_steps[i]),
                    actual_outcome=self._step_summary(actual_steps[i]),
                )

        # Length mismatch after all common steps matched.
        if len(expected_steps) != len(actual_steps):
            divergence = min_len
            return ValidationResult(
                input_data=input_data,
                passed=False,
                divergence_point=divergence,
                expected_outcome=f"{len(expected_steps)} steps",
                actual_outcome=f"{len(actual_steps)} steps",
                error=(
                    f"Step count mismatch: expected {len(expected_steps)}, "
                    f"got {len(actual_steps)}"
                ),
            )

        return ValidationResult(input_data=input_data, passed=True)

    # ------------------------------------------------------------------
    # Step comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _steps_match(expected: TraceStep, actual: TraceStep) -> bool:
        """Return True if two steps are equivalent for validation.

        Compares action type (case-insensitive) and target value.
        """
        if expected.action_type.lower() != actual.action_type.lower():
            return False

        # Both targets None → match.
        if expected.target is None and actual.target is None:
            return True

        # One None, other not → mismatch.
        if expected.target is None or actual.target is None:
            return False

        return str(expected.target) == str(actual.target)

    @staticmethod
    def _step_summary(step: TraceStep) -> Dict[str, Any]:
        """Produce a small summary dict for a step (used in result reporting)."""
        summary: Dict[str, Any] = {"action_type": step.action_type}
        if step.target is not None:
            summary["target"] = str(step.target)
        return summary

    # ------------------------------------------------------------------
    # Extracting steps from ExecutionResult
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_steps_from_result(execution_result: Any) -> List[TraceStep]:
        """Extract a list of ``TraceStep`` from an ``ExecutionResult``.

        The ``ExecutionResult.outputs`` dict may contain an ``"actions"``
        key with a list of executed action dicts, or a ``"steps"`` key
        with ``TraceStep``-like objects.  If neither is present, we
        return an empty list (execution succeeded but produced no
        comparable trace).
        """
        outputs = getattr(execution_result, "outputs", {}) or {}

        # Direct TraceStep list.
        if "steps" in outputs:
            steps = outputs["steps"]
            if isinstance(steps, list):
                return [
                    s if isinstance(s, TraceStep) else _dict_to_trace_step(s)
                    for s in steps
                ]

        # Action dicts list.
        if "actions" in outputs:
            actions = outputs["actions"]
            if isinstance(actions, list):
                return [_dict_to_trace_step(a) for a in actions]

        return []


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _dict_to_trace_step(d: Any) -> TraceStep:
    """Best-effort conversion of a dict (or dict-like) to a TraceStep."""
    if isinstance(d, TraceStep):
        return d
    if isinstance(d, dict):
        return TraceStep(
            action_type=d.get("action_type", d.get("type", "unknown")),
            target=d.get("target"),
            args=d.get("args"),
        )
    # Fallback: treat as opaque object with attributes.
    return TraceStep(
        action_type=getattr(d, "action_type", getattr(d, "type", "unknown")),
        target=getattr(d, "target", None),
        args=getattr(d, "args", None),
    )

"""Unit tests for GraphValidator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
    ValidationResult,
    ValidationResults,
)
from science_modeling_tools.automation.meta_agent.validator import GraphValidator


# ------------------------------------------------------------------
# Helpers — lightweight fakes for ActionGraph / ExecutionResult
# ------------------------------------------------------------------


@dataclass
class FakeExecutionResult:
    """Mimics ExecutionResult from action_graph.execute()."""

    success: bool = True
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None


class FakeGraph:
    """A fake ActionGraph whose execute() returns a pre-configured result."""

    def __init__(self, result: FakeExecutionResult):
        self._result = result

    def execute(self, initial_variables: Optional[Dict[str, Any]] = None) -> FakeExecutionResult:
        return self._result


class FailingGraph:
    """A fake ActionGraph whose execute() raises an exception."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def execute(self, initial_variables: Optional[Dict[str, Any]] = None):
        raise self._exc


# ------------------------------------------------------------------
# Helpers — build traces and steps quickly
# ------------------------------------------------------------------


def _step(action_type: str, target: Optional[str] = None) -> TraceStep:
    return TraceStep(action_type=action_type, target=target)


def _trace(
    steps: List[TraceStep],
    trace_id: str = "t1",
    input_data: Optional[Dict[str, Any]] = None,
) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        task_description="test",
        steps=steps,
        input_data=input_data,
    )


def _graph_with_steps(steps: List[TraceStep]) -> FakeGraph:
    """Build a FakeGraph whose outputs contain the given steps."""
    return FakeGraph(
        FakeExecutionResult(
            success=True,
            outputs={"steps": steps},
        )
    )


# ==================================================================
# Tests
# ==================================================================


class TestMatchingExecution:
    """Requirement 8.2 — matching execution passes."""

    def test_identical_steps_pass(self):
        steps = [_step("click", "btn"), _step("input_text", "field")]
        graph = _graph_with_steps(steps)
        expected = _trace(steps)

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[expected],
        )

        assert results.all_passed
        assert results.success_rate == 1.0
        assert len(results.results) == 1
        assert results.results[0].passed is True

    def test_no_expected_trace_passes_on_success(self):
        """When no expected trace is provided, a successful execution passes."""
        graph = FakeGraph(FakeExecutionResult(success=True))
        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[None],
        )
        assert results.all_passed

    def test_empty_steps_match(self):
        graph = _graph_with_steps([])
        expected = _trace([])
        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[expected],
        )
        assert results.all_passed


class TestMismatchingExecution:
    """Requirement 8.3 — mismatching execution fails with divergence point."""

    def test_different_action_type_diverges(self):
        actual_steps = [_step("click", "btn"), _step("scroll", "page")]
        expected_steps = [_step("click", "btn"), _step("input_text", "field")]

        graph = _graph_with_steps(actual_steps)
        expected = _trace(expected_steps)

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[expected],
        )

        assert not results.all_passed
        r = results.results[0]
        assert r.passed is False
        assert r.divergence_point == 1

    def test_different_target_diverges(self):
        actual_steps = [_step("click", "btn_a")]
        expected_steps = [_step("click", "btn_b")]

        graph = _graph_with_steps(actual_steps)
        expected = _trace(expected_steps)

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[expected],
        )

        r = results.results[0]
        assert r.passed is False
        assert r.divergence_point == 0

    def test_length_mismatch_diverges(self):
        actual_steps = [_step("click", "btn")]
        expected_steps = [_step("click", "btn"), _step("input_text", "field")]

        graph = _graph_with_steps(actual_steps)
        expected = _trace(expected_steps)

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[expected],
        )

        r = results.results[0]
        assert r.passed is False
        assert r.divergence_point == 1
        assert r.error is not None
        assert "mismatch" in r.error.lower()

    def test_first_step_diverges_at_zero(self):
        actual_steps = [_step("scroll", "page")]
        expected_steps = [_step("click", "btn")]

        graph = _graph_with_steps(actual_steps)
        expected = _trace(expected_steps)

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[expected],
        )

        r = results.results[0]
        assert r.passed is False
        assert r.divergence_point == 0


class TestMultipleInputs:
    """Requirement 8.4 — per-input pass/fail and overall success_rate."""

    def test_mixed_results(self):
        # First input: matching
        steps_ok = [_step("click", "btn")]
        graph_ok = _graph_with_steps(steps_ok)
        expected_ok = _trace(steps_ok, input_data={"q": "a"})

        # Second input: mismatching
        steps_bad_actual = [_step("scroll", "page")]
        steps_bad_expected = [_step("click", "btn")]
        expected_bad = _trace(steps_bad_expected, input_data={"q": "b"})

        # We need a graph that returns different results per input.
        # Use a custom validator subclass for this test.
        class MultiGraphValidator(GraphValidator):
            def __init__(self, results_by_index):
                self._results_by_index = results_by_index
                self._call_idx = 0

            def _execute_graph(self, graph, input_data):
                result = self._results_by_index[self._call_idx]
                self._call_idx += 1
                return result

        validator = MultiGraphValidator([
            FakeExecutionResult(success=True, outputs={"steps": steps_ok}),
            FakeExecutionResult(success=True, outputs={"steps": steps_bad_actual}),
        ])

        results = validator.validate(
            graph=FakeGraph(FakeExecutionResult()),  # unused
            task_description="test",
            test_data=[{"q": "a"}, {"q": "b"}],
            expected_traces=[expected_ok, expected_bad],
        )

        assert len(results.results) == 2
        assert results.results[0].passed is True
        assert results.results[1].passed is False
        assert results.success_rate == 0.5

    def test_all_pass_rate_is_one(self):
        steps = [_step("click", "btn")]
        graph = _graph_with_steps(steps)
        expected = _trace(steps)

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}, {}, {}],
            expected_traces=[expected, expected, expected],
        )

        assert results.success_rate == 1.0
        assert results.all_passed

    def test_all_fail_rate_is_zero(self):
        actual_steps = [_step("scroll", "page")]
        expected_steps = [_step("click", "btn")]

        graph = _graph_with_steps(actual_steps)
        expected = _trace(expected_steps)

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}, {}],
            expected_traces=[expected, expected],
        )

        assert results.success_rate == 0.0


class TestExecutionFailure:
    """Edge cases: graph execution fails or returns None."""

    def test_execution_exception_fails_validation(self):
        graph = FailingGraph(RuntimeError("boom"))
        expected = _trace([_step("click", "btn")])

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[expected],
        )

        assert not results.all_passed
        r = results.results[0]
        assert r.passed is False

    def test_execution_result_not_success(self):
        graph = FakeGraph(FakeExecutionResult(success=False, error=RuntimeError("fail")))
        expected = _trace([_step("click", "btn")])

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[expected],
        )

        r = results.results[0]
        assert r.passed is False
        assert r.error is not None


class TestDefaultTestData:
    """When test_data is None, a single run with empty dict is used."""

    def test_default_single_run(self):
        graph = FakeGraph(FakeExecutionResult(success=True))
        validator = GraphValidator()
        results = validator.validate(graph=graph, task_description="test")
        assert len(results.results) == 1
        assert results.results[0].passed is True


class TestCaseInsensitiveActionType:
    """Action type comparison should be case-insensitive."""

    def test_case_insensitive_match(self):
        actual = [_step("Click", "btn")]
        expected = [_step("click", "btn")]

        graph = _graph_with_steps(actual)
        exp_trace = _trace(expected)

        validator = GraphValidator()
        results = validator.validate(
            graph=graph,
            task_description="test",
            test_data=[{}],
            expected_traces=[exp_trace],
        )
        assert results.all_passed

"""Property test for validation result correctness (Property 20).

**Validates: Requirements 8.2, 8.3**

For any pair of (actual_steps, expected_steps):
- passed=True iff execution matches expected
- When passed=False, divergence_point is non-None
- divergence_point is the correct index where the first mismatch occurs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from agent_foundation.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)
from agent_foundation.automation.meta_agent.validator import GraphValidator


# ---------------------------------------------------------------------------
# Fakes (same pattern as unit tests)
# ---------------------------------------------------------------------------


@dataclass
class FakeExecutionResult:
    success: bool = True
    outputs: Dict[str, Any] = field(default_factory=dict)


class FakeGraph:
    def __init__(self, result: FakeExecutionResult):
        self._result = result

    def execute(self, initial_variables: Optional[Dict[str, Any]] = None) -> FakeExecutionResult:
        return self._result


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url", "wait", "no_op"]
TARGETS = [None, "btn", "field", "link", "page", "#submit", ".item"]


@st.composite
def trace_step(draw) -> TraceStep:
    """Generate a random TraceStep with action_type and optional target."""
    return TraceStep(
        action_type=draw(st.sampled_from(ACTION_TYPES)),
        target=draw(st.sampled_from(TARGETS)),
    )


@st.composite
def step_list(draw, min_size: int = 0, max_size: int = 8) -> List[TraceStep]:
    """Generate a list of random TraceSteps."""
    return draw(st.lists(trace_step(), min_size=min_size, max_size=max_size))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_first_mismatch(
    expected: List[TraceStep], actual: List[TraceStep]
) -> Optional[int]:
    """Compute the expected divergence point independently of the validator.

    Returns None when the lists match entirely, or the index of the first
    mismatch (including a length mismatch after all common steps agree).
    """
    min_len = min(len(expected), len(actual))
    for i in range(min_len):
        e, a = expected[i], actual[i]
        if e.action_type.lower() != a.action_type.lower():
            return i
        # Target comparison mirrors GraphValidator._steps_match
        if e.target is None and a.target is None:
            continue
        if e.target is None or a.target is None:
            return i
        if str(e.target) != str(a.target):
            return i
    if len(expected) != len(actual):
        return min_len
    return None


def _steps_are_identical(
    expected: List[TraceStep], actual: List[TraceStep]
) -> bool:
    """Return True when the two step lists match completely."""
    return _find_first_mismatch(expected, actual) is None


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(steps=step_list(min_size=0, max_size=8))
def test_identical_steps_always_pass(steps: List[TraceStep]):
    """Property 20: When actual == expected, passed is True.

    **Validates: Requirements 8.2**
    """
    graph = FakeGraph(FakeExecutionResult(success=True, outputs={"steps": steps}))
    expected = ExecutionTrace(
        trace_id="t1", task_description="test", steps=steps
    )

    validator = GraphValidator()
    results = validator.validate(
        graph=graph,
        task_description="test",
        test_data=[{}],
        expected_traces=[expected],
    )

    r = results.results[0]
    assert r.passed is True, (
        f"Expected passed=True for identical steps, got passed={r.passed}"
    )
    assert r.divergence_point is None


@settings(max_examples=300, deadline=None)
@given(actual=step_list(min_size=0, max_size=8), expected=step_list(min_size=0, max_size=8))
def test_pass_iff_match(actual: List[TraceStep], expected: List[TraceStep]):
    """Property 20: passed=True iff execution matches expected.

    **Validates: Requirements 8.2, 8.3**
    """
    graph = FakeGraph(FakeExecutionResult(success=True, outputs={"steps": actual}))
    expected_trace = ExecutionTrace(
        trace_id="t1", task_description="test", steps=expected
    )

    validator = GraphValidator()
    results = validator.validate(
        graph=graph,
        task_description="test",
        test_data=[{}],
        expected_traces=[expected_trace],
    )

    r = results.results[0]
    match = _steps_are_identical(expected, actual)

    assert r.passed == match, (
        f"passed={r.passed} but steps_match={match}; "
        f"actual_types={[s.action_type for s in actual]}, "
        f"expected_types={[s.action_type for s in expected]}"
    )


@settings(max_examples=300, deadline=None)
@given(actual=step_list(min_size=0, max_size=8), expected=step_list(min_size=0, max_size=8))
def test_failed_has_divergence_point(actual: List[TraceStep], expected: List[TraceStep]):
    """Property 20: When passed=False, divergence_point is non-None.

    **Validates: Requirements 8.3**
    """
    graph = FakeGraph(FakeExecutionResult(success=True, outputs={"steps": actual}))
    expected_trace = ExecutionTrace(
        trace_id="t1", task_description="test", steps=expected
    )

    validator = GraphValidator()
    results = validator.validate(
        graph=graph,
        task_description="test",
        test_data=[{}],
        expected_traces=[expected_trace],
    )

    r = results.results[0]
    if not r.passed:
        assert r.divergence_point is not None, (
            "passed=False but divergence_point is None"
        )


@settings(max_examples=300, deadline=None)
@given(actual=step_list(min_size=0, max_size=8), expected=step_list(min_size=0, max_size=8))
def test_divergence_point_is_correct_index(actual: List[TraceStep], expected: List[TraceStep]):
    """Property 20: divergence_point equals the first mismatch index.

    **Validates: Requirements 8.3**
    """
    graph = FakeGraph(FakeExecutionResult(success=True, outputs={"steps": actual}))
    expected_trace = ExecutionTrace(
        trace_id="t1", task_description="test", steps=expected
    )

    validator = GraphValidator()
    results = validator.validate(
        graph=graph,
        task_description="test",
        test_data=[{}],
        expected_traces=[expected_trace],
    )

    r = results.results[0]
    expected_mismatch = _find_first_mismatch(expected, actual)

    if expected_mismatch is None:
        assert r.passed is True
        assert r.divergence_point is None
    else:
        assert r.passed is False
        assert r.divergence_point == expected_mismatch, (
            f"divergence_point={r.divergence_point} but expected {expected_mismatch}; "
            f"actual={[(s.action_type, s.target) for s in actual]}, "
            f"expected={[(s.action_type, s.target) for s in expected]}"
        )

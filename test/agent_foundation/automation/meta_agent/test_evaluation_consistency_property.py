"""Property test for trace evaluation consistency (Property 25).

**Validates: Requirements 12.1, 12.2, 12.3**

For any list of ExecutionTraces and any EvaluationStrategy, the
TraceEvaluator SHALL return exactly one EvaluationResult per input trace,
in the same order.

- EXCEPTION_ONLY: success=True → passed=True, success=False → passed=False
- RULE_BASED: passes all error-severity rules → passed=True regardless of
  warning-severity rule results
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agent_foundation.automation.meta_agent.evaluator import (
    EvaluationRule,
    EvaluationStrategy,
    TraceEvaluator,
)
from agent_foundation.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url", "wait"]


@st.composite
def trace_step(draw) -> TraceStep:
    """Generate a random TraceStep."""
    return TraceStep(
        action_type=draw(st.sampled_from(ACTION_TYPES)),
        target=draw(st.text(min_size=0, max_size=10)),
    )


@st.composite
def execution_trace(draw) -> ExecutionTrace:
    """Generate a random ExecutionTrace with random success value."""
    trace_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))))
    steps = draw(st.lists(trace_step(), min_size=0, max_size=5))
    success = draw(st.booleans())
    return ExecutionTrace(
        trace_id=trace_id,
        task_description="test task",
        steps=steps,
        success=success,
    )


@st.composite
def trace_list(draw) -> list[ExecutionTrace]:
    """Generate a list of random ExecutionTraces."""
    return draw(st.lists(execution_trace(), min_size=0, max_size=10))


@st.composite
def error_and_warning_rules(draw):
    """Generate a non-empty list of EvaluationRules with mixed severities.

    Each rule has a random pass/fail predicate. At least one rule is
    always included (RULE_BASED requires >= 1 rule).
    """
    n_error = draw(st.integers(min_value=0, max_value=3))
    n_warning = draw(st.integers(min_value=0, max_value=3))
    # Ensure at least one rule total
    if n_error + n_warning == 0:
        n_error = 1

    rules: list[EvaluationRule] = []
    for i in range(n_error):
        passes = draw(st.booleans())
        rules.append(
            EvaluationRule(
                name=f"error_rule_{i}",
                description=f"Error rule {i}",
                predicate=lambda _t, _p=passes: _p,
                severity="error",
            )
        )
    for i in range(n_warning):
        passes = draw(st.booleans())
        rules.append(
            EvaluationRule(
                name=f"warning_rule_{i}",
                description=f"Warning rule {i}",
                predicate=lambda _t, _p=passes: _p,
                severity="warning",
            )
        )
    return rules


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(traces=trace_list())
def test_exception_only_returns_one_result_per_trace(traces: list[ExecutionTrace]):
    """Property 25: EXCEPTION_ONLY — result count equals trace count.

    The evaluator SHALL return exactly one EvaluationResult per input trace.

    **Validates: Requirements 12.1**
    """
    evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
    results = evaluator.evaluate(traces)
    assert len(results) == len(traces)


@settings(max_examples=200, deadline=None)
@given(traces=trace_list())
def test_exception_only_preserves_order(traces: list[ExecutionTrace]):
    """Property 25: EXCEPTION_ONLY — results are in the same order as input.

    The i-th EvaluationResult SHALL correspond to the i-th input trace.

    **Validates: Requirements 12.1**
    """
    evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
    results = evaluator.evaluate(traces)
    for trace, result in zip(traces, results):
        assert result.trace_id == trace.trace_id


@settings(max_examples=200, deadline=None)
@given(traces=trace_list())
def test_exception_only_pass_fail_matches_success(traces: list[ExecutionTrace]):
    """Property 25: EXCEPTION_ONLY — passed mirrors success flag.

    success=True → passed=True, success=False → passed=False.

    **Validates: Requirements 12.2**
    """
    evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
    results = evaluator.evaluate(traces)
    for trace, result in zip(traces, results):
        assert result.passed == trace.success, (
            f"trace {trace.trace_id}: success={trace.success} but passed={result.passed}"
        )


@settings(max_examples=200, deadline=None)
@given(traces=trace_list(), rules=error_and_warning_rules())
def test_rule_based_returns_one_result_per_trace(
    traces: list[ExecutionTrace],
    rules: list[EvaluationRule],
):
    """Property 25: RULE_BASED — result count equals trace count.

    **Validates: Requirements 12.1**
    """
    evaluator = TraceEvaluator(
        strategy=EvaluationStrategy.RULE_BASED, rules=rules
    )
    results = evaluator.evaluate(traces)
    assert len(results) == len(traces)


@settings(max_examples=200, deadline=None)
@given(traces=trace_list(), rules=error_and_warning_rules())
def test_rule_based_preserves_order(
    traces: list[ExecutionTrace],
    rules: list[EvaluationRule],
):
    """Property 25: RULE_BASED — results are in the same order as input.

    **Validates: Requirements 12.1**
    """
    evaluator = TraceEvaluator(
        strategy=EvaluationStrategy.RULE_BASED, rules=rules
    )
    results = evaluator.evaluate(traces)
    for trace, result in zip(traces, results):
        assert result.trace_id == trace.trace_id


@settings(max_examples=200, deadline=None)
@given(traces=trace_list(), rules=error_and_warning_rules())
def test_rule_based_error_rules_determine_pass_fail(
    traces: list[ExecutionTrace],
    rules: list[EvaluationRule],
):
    """Property 25: RULE_BASED — passes all error-severity rules → passed=True.

    A trace that passes all error-severity rules SHALL be marked as passed,
    regardless of warning-severity rule results.

    **Validates: Requirements 12.3**
    """
    evaluator = TraceEvaluator(
        strategy=EvaluationStrategy.RULE_BASED, rules=rules
    )
    results = evaluator.evaluate(traces)

    error_rules = [r for r in rules if r.severity == "error"]

    for trace, result in zip(traces, results):
        # Determine expected pass/fail by checking error-severity rules only
        all_error_rules_pass = all(r.predicate(trace) for r in error_rules)
        if all_error_rules_pass:
            assert result.passed is True, (
                f"trace {trace.trace_id}: all error rules pass but passed={result.passed}"
            )
        else:
            assert result.passed is False, (
                f"trace {trace.trace_id}: some error rules fail but passed={result.passed}"
            )

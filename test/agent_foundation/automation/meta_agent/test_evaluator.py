"""Unit tests for TraceEvaluator.

Validates: Requirements 12.1, 12.2, 12.3, 12.6, 12.7
"""

import pytest

from science_modeling_tools.automation.meta_agent.evaluator import (
    EvaluationRule,
    EvaluationStrategy,
    TraceEvaluator,
)
from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace(trace_id: str = "t1", success: bool = True, steps=None) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        task_description="test task",
        steps=steps or [],
        success=success,
    )


# ---------------------------------------------------------------------------
# EXCEPTION_ONLY strategy
# ---------------------------------------------------------------------------


class TestExceptionOnly:
    """Validates: Requirement 12.2"""

    def test_success_true_passes(self):
        evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
        results = evaluator.evaluate([_trace(success=True)])
        assert results[0].passed is True
        assert results[0].score == 1.0

    def test_success_false_fails(self):
        evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
        results = evaluator.evaluate([_trace(success=False)])
        assert results[0].passed is False
        assert results[0].score == 0.0


# ---------------------------------------------------------------------------
# RULE_BASED strategy
# ---------------------------------------------------------------------------


class TestRuleBased:
    """Validates: Requirements 12.3, 12.7"""

    def test_error_severity_rule_failure_rejects_trace(self):
        rule = EvaluationRule(
            name="always_fail",
            description="Always fails",
            predicate=lambda _: False,
            severity="error",
        )
        evaluator = TraceEvaluator(
            strategy=EvaluationStrategy.RULE_BASED, rules=[rule]
        )
        results = evaluator.evaluate([_trace()])
        assert results[0].passed is False
        assert "always_fail" in results[0].failed_rules

    def test_warning_severity_only_passes_with_warnings(self):
        rule = EvaluationRule(
            name="soft_check",
            description="Warns only",
            predicate=lambda _: False,
            severity="warning",
        )
        evaluator = TraceEvaluator(
            strategy=EvaluationStrategy.RULE_BASED, rules=[rule]
        )
        results = evaluator.evaluate([_trace()])
        assert results[0].passed is True
        assert "soft_check" in results[0].warnings

    def test_mixed_error_and_warning_rules(self):
        error_rule = EvaluationRule(
            name="err", description="error", predicate=lambda _: True, severity="error"
        )
        warn_rule = EvaluationRule(
            name="warn", description="warn", predicate=lambda _: False, severity="warning"
        )
        evaluator = TraceEvaluator(
            strategy=EvaluationStrategy.RULE_BASED, rules=[error_rule, warn_rule]
        )
        results = evaluator.evaluate([_trace()])
        # Error rule passes, warning rule fails → trace passes with warnings
        assert results[0].passed is True
        assert results[0].warnings == ["warn"]
        assert results[0].failed_rules == []

    def test_without_rules_raises_value_error(self):
        with pytest.raises(ValueError, match="RULE_BASED strategy requires"):
            TraceEvaluator(strategy=EvaluationStrategy.RULE_BASED)

    def test_without_rules_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="RULE_BASED strategy requires"):
            TraceEvaluator(strategy=EvaluationStrategy.RULE_BASED, rules=[])


# ---------------------------------------------------------------------------
# LLM_JUDGE strategy
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """Validates: Requirement 12.6"""

    def test_without_inferencer_raises_value_error(self):
        with pytest.raises(ValueError, match="LLM_JUDGE strategy requires"):
            TraceEvaluator(strategy=EvaluationStrategy.LLM_JUDGE)


# ---------------------------------------------------------------------------
# Result count and order
# ---------------------------------------------------------------------------


class TestResultCountAndOrder:
    """Validates: Requirement 12.1"""

    def test_result_count_matches_input(self):
        evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
        traces = [_trace(trace_id=f"t{i}") for i in range(5)]
        results = evaluator.evaluate(traces)
        assert len(results) == len(traces)

    def test_result_order_matches_input(self):
        evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
        traces = [
            _trace(trace_id="first", success=True),
            _trace(trace_id="second", success=False),
            _trace(trace_id="third", success=True),
        ]
        results = evaluator.evaluate(traces)
        assert [r.trace_id for r in results] == ["first", "second", "third"]
        assert [r.passed for r in results] == [True, False, True]

    def test_empty_trace_list_returns_empty(self):
        evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
        results = evaluator.evaluate([])
        assert results == []

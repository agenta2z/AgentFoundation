"""Unit tests for TraceEvaluator.

Validates: Requirements 12.1, 12.2, 12.3, 12.6, 12.7
"""

import pytest
from agent_foundation.automation.meta_agent.evaluator import (
    EvaluationRule,
    EvaluationStrategy,
    TraceEvaluator,
)
from agent_foundation.automation.meta_agent.models import ExecutionTrace


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


class _FakeInferencer:
    """Minimal stand-in for an InferencerBase: a single canned ``infer`` reply.

    The migrated judge reaches the wrapped inferencer through
    ``FunctionInferencer``, which invokes exactly ``inferencer.infer(prompt)`` —
    so only ``infer`` needs to exist here.
    """

    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = 0
        self.prompts = []

    def infer(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self._exc is not None:
            raise self._exc
        return self._response


def _llm_evaluator(response=None, exc=None, min_score: float = 0.5) -> TraceEvaluator:
    return TraceEvaluator(
        strategy=EvaluationStrategy.LLM_JUDGE,
        inferencer=_FakeInferencer(response=response, exc=exc),
        min_score=min_score,
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
        evaluator = TraceEvaluator(strategy=EvaluationStrategy.RULE_BASED, rules=[rule])
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
        evaluator = TraceEvaluator(strategy=EvaluationStrategy.RULE_BASED, rules=[rule])
        results = evaluator.evaluate([_trace()])
        assert results[0].passed is True
        assert "soft_check" in results[0].warnings

    def test_mixed_error_and_warning_rules(self):
        error_rule = EvaluationRule(
            name="err", description="error", predicate=lambda _: True, severity="error"
        )
        warn_rule = EvaluationRule(
            name="warn",
            description="warn",
            predicate=lambda _: False,
            severity="warning",
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


# ---------------------------------------------------------------------------
# LLM_JUDGE decode path (migrated @agentic_function judge)
# ---------------------------------------------------------------------------


class TestLLMJudgeDecode:
    """The migrated @agentic_function judge preserves the original decode.

    Validates: Requirement 12.6
    """

    def test_dict_response_scores_and_passes(self):
        # A dict reply must reach _parse_score AS A DICT (via response.raw), not
        # a stringified form — this is what proves the judge reads .raw, not the
        # normalized .text (which would stringify the dict and fail to parse).
        response = {"score": 0.85}
        evaluator = _llm_evaluator(response=response)
        result = evaluator.evaluate([_trace(trace_id="d1")], "task")[0]
        assert result.trace_id == "d1"
        assert result.passed is True
        assert result.score == 0.85
        assert result.metadata["llm_response"] == str(response)

    def test_json_string_response_scores(self):
        evaluator = _llm_evaluator(response='{"score": 0.9}')
        result = evaluator.evaluate([_trace(trace_id="j1")], "task")[0]
        assert result.passed is True
        assert result.score == 0.9
        assert result.metadata["llm_response"] == '{"score": 0.9}'

    def test_score_below_min_fails(self):
        evaluator = _llm_evaluator(response={"score": 0.2})
        result = evaluator.evaluate([_trace(trace_id="low")], "task")[0]
        assert result.passed is False
        assert result.score == 0.2

    def test_custom_min_score_threshold(self):
        passing = _llm_evaluator(response={"score": 0.85}, min_score=0.8)
        failing = _llm_evaluator(response={"score": 0.75}, min_score=0.8)
        assert passing.evaluate([_trace()], "task")[0].passed is True
        assert failing.evaluate([_trace()], "task")[0].passed is False

    def test_unparseable_reply_defaults_to_zero(self):
        # _parse_score returns 0.0 (it does NOT raise) → success path: the score
        # is 0.0 and llm_response is recorded, NOT llm_error.
        evaluator = _llm_evaluator(response="not a score")
        result = evaluator.evaluate([_trace(trace_id="bad")], "task")[0]
        assert result.passed is False
        assert result.score == 0.0
        assert result.metadata["llm_response"] == "not a score"
        assert "llm_error" not in result.metadata

    def test_infer_exception_marks_failed(self):
        # inferencer.infer raising propagates unwrapped through the decorator to
        # the fail-closed handler → llm_error carrying the original message.
        evaluator = _llm_evaluator(exc=RuntimeError("boom"))
        result = evaluator.evaluate([_trace(trace_id="err")], "task")[0]
        assert result.passed is False
        assert result.score == 0.0
        assert result.metadata["llm_error"] == "boom"

    def test_parse_score_exception_marks_failed(self):
        # A dict whose "score" is non-numeric makes _parse_score raise ValueError
        # inside the body; it propagates unwrapped → llm_error (mirrors the old
        # try/except that wrapped _parse_score).
        evaluator = _llm_evaluator(response={"score": "abc"})
        result = evaluator.evaluate([_trace(trace_id="nan")], "task")[0]
        assert result.passed is False
        assert result.score == 0.0
        assert "llm_error" in result.metadata
        assert "float" in result.metadata["llm_error"].lower()

    def test_inferencer_receives_built_prompt(self):
        # The judge forwards the _build_llm_prompt output verbatim ({{ prompt }})
        # to the wrapped inferencer, called exactly once.
        fake = _FakeInferencer(response={"score": 1.0})
        evaluator = TraceEvaluator(
            strategy=EvaluationStrategy.LLM_JUDGE, inferencer=fake
        )
        evaluator.evaluate([_trace(trace_id="p1")], "my task")
        assert fake.calls == 1
        assert "Trace ID: p1" in fake.prompts[0]
        assert "Task: my task" in fake.prompts[0]

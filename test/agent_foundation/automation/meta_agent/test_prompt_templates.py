"""Unit tests for prompt_templates module.

Tests feed builders, template rendering, template overrides, the
create_prompt_formatter factory, and pipeline integration of the
TemplateManager-based prompt management.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_foundation.automation.meta_agent.models import (
    AlignedPosition,
    AlignmentType,
    ExecutionTrace,
    ParameterizableInfo,
    TraceActionResult,
    TraceStep,
)
from agent_foundation.automation.meta_agent.prompt_templates import (
    DEFAULT_PROMPT_TEMPLATES,
    EVALUATION_TEMPLATE_KEY,
    SYNTHESIS_TEMPLATE_KEY,
    build_evaluation_feed,
    build_synthesis_feed,
    create_prompt_formatter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace(
    trace_id: str = "t1",
    success: bool = True,
    steps: list | None = None,
    task_description: str = "test task",
) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        task_description=task_description,
        steps=steps or [],
        success=success,
    )


def _step(action_type: str, target=None, args=None, result=None) -> TraceStep:
    return TraceStep(action_type=action_type, target=target, args=args, result=result)


def _pos(
    index: int,
    atype: AlignmentType,
    steps: dict[str, TraceStep | None],
    confidence: float = 1.0,
) -> AlignedPosition:
    return AlignedPosition(
        index=index, alignment_type=atype, steps=steps, confidence=confidence,
    )


# ---------------------------------------------------------------------------
# TestFeedBuilders
# ---------------------------------------------------------------------------


class TestFeedBuilders:
    """Tests for build_evaluation_feed and build_synthesis_feed."""

    def test_evaluation_feed_returns_correct_keys(self):
        """build_evaluation_feed returns all expected keys."""
        trace = _trace(
            trace_id="t42",
            success=True,
            steps=[_step("click", target="btn")],
        )
        feed = build_evaluation_feed(trace, "search task")

        assert feed["task_description"] == "search task"
        assert feed["trace_id"] == "t42"
        assert feed["trace_success"] is True
        assert feed["step_count"] == 1
        assert "click" in feed["steps_text"]
        assert "btn" in feed["steps_text"]

    def test_evaluation_feed_no_steps(self):
        """build_evaluation_feed handles a trace with no steps."""
        trace = _trace(steps=[])
        feed = build_evaluation_feed(trace, "empty task")

        assert feed["step_count"] == 0
        assert feed["steps_text"] == "  (no steps)"

    def test_evaluation_feed_includes_result_success(self):
        """Step result success is included in steps_text."""
        trace = _trace(
            steps=[
                _step("click", target="btn", result=TraceActionResult(success=True)),
                _step("input_text", target="field", result=TraceActionResult(success=False)),
            ],
        )
        feed = build_evaluation_feed(trace, "task")

        assert "success=True" in feed["steps_text"]
        assert "success=False" in feed["steps_text"]
        assert feed["step_count"] == 2

    def test_synthesis_feed_returns_correct_keys(self):
        """build_synthesis_feed returns all expected keys."""
        pos = _pos(3, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn-submit"),
        })
        context = {
            "pattern_type": "deterministic",
            "task_description": "login flow",
        }
        feed = build_synthesis_feed(pos, context)

        assert feed["pattern_type"] == "deterministic"
        assert feed["task_description"] == "login flow"
        assert feed["position_index"] == 3
        assert "click" in feed["steps_text"]
        assert "btn-submit" in feed["steps_text"]
        assert feed["param_section"] == ""

    def test_synthesis_feed_includes_param_section(self):
        """build_synthesis_feed includes param_section when param_info present."""
        pos = _pos(0, AlignmentType.PARAMETERIZABLE, {
            "t1": _step("input_text", target="field", args={"text": "hello"}),
        })
        info = ParameterizableInfo(
            variable_args={"text": "search_query"},
            constant_args={"delay": 100},
        )
        context = {
            "pattern_type": "parameterizable",
            "task_description": "search task",
            "param_info": info,
        }
        feed = build_synthesis_feed(pos, context)

        assert "search_query" in feed["param_section"]
        assert "delay" in feed["param_section"]

    def test_synthesis_feed_absent_step(self):
        """build_synthesis_feed handles None steps (absent in some traces)."""
        pos = _pos(0, AlignmentType.OPTIONAL, {
            "t1": _step("click", target="popup"),
            "t2": None,
        })
        context = {"pattern_type": "optional", "task_description": ""}
        feed = build_synthesis_feed(pos, context)

        assert "(absent)" in feed["steps_text"]
        assert "click" in feed["steps_text"]


# ---------------------------------------------------------------------------
# TestTemplateRendering
# ---------------------------------------------------------------------------


class TestTemplateRendering:
    """Tests that templates render correctly with sample feeds."""

    def test_evaluation_template_renders_with_feed(self):
        """Evaluation template renders with all feed variables filled in."""
        trace = _trace(
            trace_id="t1",
            success=True,
            steps=[_step("click", target="btn")],
        )
        feed = build_evaluation_feed(trace, "test task")
        formatter = create_prompt_formatter()

        rendered = formatter(EVALUATION_TEMPLATE_KEY, **feed)

        assert "test task" in rendered
        assert "t1" in rendered
        assert "True" in rendered
        assert "click" in rendered
        assert "btn" in rendered
        assert "score" in rendered
        assert "0.85" in rendered

    def test_synthesis_template_renders_with_feed(self):
        """Synthesis template renders with all feed variables filled in."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn-submit"),
        })
        context = {
            "pattern_type": "deterministic",
            "task_description": "login flow",
        }
        feed = build_synthesis_feed(pos, context)
        formatter = create_prompt_formatter()

        rendered = formatter(SYNTHESIS_TEMPLATE_KEY, **feed)

        assert "deterministic" in rendered
        assert "login flow" in rendered
        assert "click" in rendered
        assert "btn-submit" in rendered
        assert "action_type" in rendered
        assert "confidence" in rendered

    def test_evaluation_template_matches_legacy_output(self):
        """TemplateManager output matches the legacy f-string output character-for-character."""
        trace = _trace(
            trace_id="trace-abc",
            success=False,
            steps=[
                _step("click", target="login-btn", result=TraceActionResult(success=True)),
                _step("input_text", target="email-field"),
            ],
        )
        task_desc = "Login to dashboard"
        feed = build_evaluation_feed(trace, task_desc)

        # Render via TemplateManager
        formatter = create_prompt_formatter()
        rendered = formatter(EVALUATION_TEMPLATE_KEY, **feed)

        # Build legacy f-string output
        legacy = (
            "Evaluate the quality of the following execution trace.\n"
            f"Task: {task_desc}\n"
            f"Trace ID: {trace.trace_id}\n"
            f"Success: {trace.success}\n"
            f"Steps ({len(trace.steps)}):\n{feed['steps_text']}\n\n"
            "Respond with a JSON object containing a single key 'score' "
            "with a float value between 0.0 and 1.0, where 1.0 is perfect quality.\n"
            'Example: {"score": 0.85}'
        )

        assert rendered == legacy


# ---------------------------------------------------------------------------
# TestTemplateOverride
# ---------------------------------------------------------------------------


class TestTemplateOverride:
    """Tests that custom templates override defaults."""

    def test_custom_evaluation_template(self):
        """Custom evaluation template overrides the default."""
        custom = {
            EVALUATION_TEMPLATE_KEY: "Rate this trace: {{ trace_id }} ({{ step_count }} steps)",
            SYNTHESIS_TEMPLATE_KEY: DEFAULT_PROMPT_TEMPLATES[SYNTHESIS_TEMPLATE_KEY],
        }
        formatter = create_prompt_formatter(custom)

        trace = _trace(trace_id="t99", steps=[_step("click")])
        feed = build_evaluation_feed(trace, "task")
        rendered = formatter(EVALUATION_TEMPLATE_KEY, **feed)

        assert rendered == "Rate this trace: t99 (1 steps)"

    def test_custom_synthesis_template(self):
        """Custom synthesis template overrides the default."""
        custom = {
            EVALUATION_TEMPLATE_KEY: DEFAULT_PROMPT_TEMPLATES[EVALUATION_TEMPLATE_KEY],
            SYNTHESIS_TEMPLATE_KEY: "Decide action for position {{ position_index }}",
        }
        formatter = create_prompt_formatter(custom)

        pos = _pos(7, AlignmentType.DETERMINISTIC, {"t1": _step("click")})
        context = {"pattern_type": "deterministic", "task_description": ""}
        feed = build_synthesis_feed(pos, context)
        rendered = formatter(SYNTHESIS_TEMPLATE_KEY, **feed)

        assert rendered == "Decide action for position 7"


# ---------------------------------------------------------------------------
# TestCreatePromptFormatter
# ---------------------------------------------------------------------------


class TestCreatePromptFormatter:
    """Tests for the create_prompt_formatter factory function."""

    def test_default_formatter_resolves_both_keys(self):
        """Default formatter can resolve both template keys."""
        formatter = create_prompt_formatter()

        # Should not raise — both keys exist
        trace = _trace(steps=[])
        eval_feed = build_evaluation_feed(trace, "task")
        eval_result = formatter(EVALUATION_TEMPLATE_KEY, **eval_feed)
        assert isinstance(eval_result, str)
        assert len(eval_result) > 0

        pos = _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click")})
        synth_feed = build_synthesis_feed(pos, {"pattern_type": "det", "task_description": ""})
        synth_result = formatter(SYNTHESIS_TEMPLATE_KEY, **synth_feed)
        assert isinstance(synth_result, str)
        assert len(synth_result) > 0

    def test_custom_templates_used(self):
        """Custom templates dict is used instead of defaults."""
        custom = {
            "custom_key": "Hello {{ name }}!",
        }
        formatter = create_prompt_formatter(custom)
        result = formatter("custom_key", name="World")
        assert result == "Hello World!"


# ---------------------------------------------------------------------------
# TestPipelineIntegration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Tests that prompt_formatter is threaded through the pipeline."""

    def test_evaluator_uses_prompt_formatter(self):
        """TraceEvaluator._build_llm_prompt uses prompt_formatter when provided."""
        from agent_foundation.automation.meta_agent.evaluator import (
            EvaluationStrategy,
            TraceEvaluator,
        )

        formatter = create_prompt_formatter()
        evaluator = TraceEvaluator(
            strategy=EvaluationStrategy.EXCEPTION_ONLY,
            prompt_formatter=formatter,
        )

        trace = _trace(
            trace_id="t1",
            steps=[_step("click", target="btn", result=TraceActionResult(success=True))],
        )
        # Call the internal method directly to test prompt_formatter usage
        prompt = evaluator._build_llm_prompt(trace, "test task")

        assert "test task" in prompt
        assert "t1" in prompt
        assert "click" in prompt

    def test_evaluator_legacy_without_formatter(self):
        """TraceEvaluator._build_llm_prompt falls back to f-string without formatter."""
        from agent_foundation.automation.meta_agent.evaluator import (
            EvaluationStrategy,
            TraceEvaluator,
        )

        evaluator = TraceEvaluator(strategy=EvaluationStrategy.EXCEPTION_ONLY)
        trace = _trace(trace_id="t1", steps=[_step("click", target="btn")])
        prompt = evaluator._build_llm_prompt(trace, "test task")

        assert "test task" in prompt
        assert "t1" in prompt
        assert "click" in prompt

    def test_synthesizer_accepts_prompt_formatter(self):
        """GraphSynthesizer subclasses accept prompt_formatter parameter."""
        from agent_foundation.automation.meta_agent.synthesizer import (
            HybridSynthesizer,
            LLMSynthesizer,
            RuleBasedSynthesizer,
        )

        formatter = create_prompt_formatter()
        executor = MagicMock()
        inferencer = MagicMock()
        inferencer.infer.return_value = "LLM response"

        # All three should accept prompt_formatter without error
        rb = RuleBasedSynthesizer(action_executor=executor, prompt_formatter=formatter)
        assert rb._prompt_formatter is formatter

        llm = LLMSynthesizer(
            action_executor=executor, inferencer=inferencer, prompt_formatter=formatter,
        )
        assert llm._prompt_formatter is formatter

        hybrid = HybridSynthesizer(
            action_executor=executor, inferencer=inferencer, prompt_formatter=formatter,
        )
        assert hybrid._prompt_formatter is formatter

    def test_evaluator_formatter_and_legacy_produce_same_content(self):
        """TemplateManager path and legacy f-string produce equivalent prompts."""
        from agent_foundation.automation.meta_agent.evaluator import (
            EvaluationStrategy,
            TraceEvaluator,
        )

        trace = _trace(
            trace_id="eval-test",
            success=True,
            steps=[
                _step("click", target="submit-btn", result=TraceActionResult(success=True)),
                _step("wait"),
            ],
        )

        # With formatter
        evaluator_with = TraceEvaluator(
            strategy=EvaluationStrategy.EXCEPTION_ONLY,
            prompt_formatter=create_prompt_formatter(),
        )
        prompt_with = evaluator_with._build_llm_prompt(trace, "evaluate this")

        # Without formatter (legacy)
        evaluator_without = TraceEvaluator(
            strategy=EvaluationStrategy.EXCEPTION_ONLY,
        )
        prompt_without = evaluator_without._build_llm_prompt(trace, "evaluate this")

        assert prompt_with == prompt_without

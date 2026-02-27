"""Unit tests for synthesis strategy hierarchy.

Tests the RuleBasedSynthesizer, LLMSynthesizer, and HybridSynthesizer
subclasses of GraphSynthesizer, verifying strategy-specific behavior,
decision_source tracking, and SynthesisReport.synthesis_strategy field.

Validates: Requirements 6.10, 6.11, 6.12
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_foundation.automation.meta_agent.models import (
    AlignedPosition,
    AlignmentType,
    ExtractedPatterns,
    ParameterizableInfo,
    SynthesisReport,
    TraceStep,
)
from agent_foundation.automation.meta_agent.synthesizer import (
    ActionDecision,
    GraphSynthesizer,
    HybridSynthesizer,
    LLMSynthesizer,
    RuleBasedSynthesizer,
    SynthesisResult,
    SynthesisStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(action_type: str, target=None, args=None, **kwargs) -> TraceStep:
    return TraceStep(action_type=action_type, target=target, args=args, **kwargs)


def _pos(
    index: int,
    atype: AlignmentType,
    steps: dict[str, TraceStep | None],
    confidence: float = 1.0,
) -> AlignedPosition:
    return AlignedPosition(
        index=index, alignment_type=atype, steps=steps, confidence=confidence,
    )


def _empty_patterns(**overrides) -> ExtractedPatterns:
    defaults = dict(
        deterministic_steps=[],
        parameterizable_steps=[],
        variable_steps=[],
        optional_steps=[],
        branch_patterns=[],
        loop_patterns=[],
        user_input_boundaries=[],
        step_order=[],
    )
    defaults.update(overrides)
    return ExtractedPatterns(**defaults)


def _mock_executor():
    return MagicMock()


def _mock_inferencer(response: str = "LLM decision"):
    """Create a mock InferencerBase that returns a fixed response."""
    inf = MagicMock()
    inf.infer.return_value = response
    return inf


# ---------------------------------------------------------------------------
# RuleBasedSynthesizer — same results as original (Requirement 6.10)
# ---------------------------------------------------------------------------

class TestRuleBasedSynthesizer:
    """RuleBasedSynthesizer produces deterministic, rule-based results."""

    def test_deterministic_step_produces_action(self):
        """A deterministic step is synthesized identically to the original."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn-submit"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        synth = RuleBasedSynthesizer(action_executor=_mock_executor())
        result = synth.synthesize(patterns)

        assert len(result.graph._nodes[0]._actions) == 1
        action = result.graph._nodes[0]._actions[0]
        assert action.type == "click"
        assert result.report.deterministic_count == 1

    def test_variable_step_creates_agent_node(self):
        """Variable steps become agent nodes, same as original behavior."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click"), "t2": _step("scroll"),
        })
        patterns = _empty_patterns(variable_steps=[pos], step_order=[0])

        synth = RuleBasedSynthesizer(action_executor=_mock_executor())
        result = synth.synthesize(patterns)

        action = result.graph._nodes[0]._actions[0]
        assert action.type == "meta_workflow_agent"
        assert result.report.agent_node_count == 1

    def test_parameterizable_step_with_template_vars(self):
        """Parameterizable steps get template placeholders."""
        pos = _pos(0, AlignmentType.PARAMETERIZABLE, {
            "t1": _step("input_text", target="field", args={"text": "hello"}),
        })
        info = ParameterizableInfo(
            variable_args={"text": "search_query"}, constant_args={"delay": 100},
        )
        patterns = _empty_patterns(
            parameterizable_steps=[(pos, info)], step_order=[0],
        )

        synth = RuleBasedSynthesizer(action_executor=_mock_executor())
        result = synth.synthesize(patterns)

        action = result.graph._nodes[0]._actions[0]
        assert action.args["text"] == "{search_query}"
        assert action.args["delay"] == 100

    def test_synthesis_strategy_is_rule_based(self):
        """Report synthesis_strategy is 'rule_based'."""
        patterns = _empty_patterns()
        synth = RuleBasedSynthesizer(action_executor=_mock_executor())
        result = synth.synthesize(patterns)
        assert result.report.synthesis_strategy == "rule_based"

    def test_decision_source_is_rule(self):
        """All decisions have decision_source='rule'."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        synth = RuleBasedSynthesizer(action_executor=_mock_executor())
        result = synth.synthesize(patterns)

        assert len(result.decisions) == 1
        assert result.decisions[0].decision_source == "rule"
        assert result.decisions[0].confidence == 1.0


# ---------------------------------------------------------------------------
# LLMSynthesizer (Requirement 6.11)
# ---------------------------------------------------------------------------

class TestLLMSynthesizer:
    """LLMSynthesizer requires InferencerBase and uses LLM for all decisions."""

    def test_requires_inferencer(self):
        """Constructing without inferencer raises ValueError."""
        with pytest.raises(ValueError, match="requires an InferencerBase"):
            LLMSynthesizer(action_executor=_mock_executor(), inferencer=None)

    def test_synthesis_strategy_is_llm(self):
        """Report synthesis_strategy is 'llm'."""
        patterns = _empty_patterns()
        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        )
        result = synth.synthesize(patterns)
        assert result.report.synthesis_strategy == "llm"

    def test_decision_source_is_llm(self):
        """All decisions have decision_source='llm'."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        )
        result = synth.synthesize(patterns)

        assert len(result.decisions) == 1
        assert result.decisions[0].decision_source == "llm"

    def test_variable_step_creates_agent_node(self):
        """Variable steps become agent nodes even with LLM strategy."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click"), "t2": _step("scroll"),
        })
        patterns = _empty_patterns(variable_steps=[pos], step_order=[0])

        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        )
        result = synth.synthesize(patterns)

        action = result.graph._nodes[0]._actions[0]
        assert action.type == "meta_workflow_agent"

    def test_inferencer_is_called(self):
        """The inferencer.infer() is called for each position."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        inf = _mock_inferencer()
        synth = LLMSynthesizer(action_executor=_mock_executor(), inferencer=inf)
        synth.synthesize(patterns)

        inf.infer.assert_called_once()


# ---------------------------------------------------------------------------
# HybridSynthesizer (Requirement 6.12)
# ---------------------------------------------------------------------------

class TestHybridSynthesizer:
    """HybridSynthesizer uses rules for clear patterns, LLM for ambiguous."""

    def test_requires_inferencer(self):
        """Constructing without inferencer raises ValueError."""
        with pytest.raises(ValueError, match="requires an InferencerBase"):
            HybridSynthesizer(action_executor=_mock_executor(), inferencer=None)

    def test_synthesis_strategy_is_hybrid(self):
        """Report synthesis_strategy is 'hybrid'."""
        patterns = _empty_patterns()
        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        )
        result = synth.synthesize(patterns)
        assert result.report.synthesis_strategy == "hybrid"

    def test_deterministic_uses_rule_source(self):
        """DETERMINISTIC positions get decision_source='rule'."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        inf = _mock_inferencer()
        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        assert len(result.decisions) == 1
        assert result.decisions[0].decision_source == "rule"
        # Inferencer should NOT be called for deterministic
        inf.infer.assert_not_called()

    def test_parameterizable_uses_rule_source(self):
        """PARAMETERIZABLE positions get decision_source='rule'."""
        pos = _pos(0, AlignmentType.PARAMETERIZABLE, {
            "t1": _step("input_text", target="field", args={"text": "x"}),
        })
        info = ParameterizableInfo(
            variable_args={"text": "query"}, constant_args={},
        )
        patterns = _empty_patterns(
            parameterizable_steps=[(pos, info)], step_order=[0],
        )

        inf = _mock_inferencer()
        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        assert result.decisions[0].decision_source == "rule"
        inf.infer.assert_not_called()

    def test_optional_uses_rule_source(self):
        """OPTIONAL positions get decision_source='rule'."""
        pos = _pos(0, AlignmentType.OPTIONAL, {
            "t1": _step("click", target="popup"), "t2": None,
        })
        patterns = _empty_patterns(optional_steps=[pos], step_order=[0])

        inf = _mock_inferencer()
        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        assert result.decisions[0].decision_source == "rule"
        inf.infer.assert_not_called()

    def test_variable_uses_llm_source(self):
        """VARIABLE positions get decision_source='llm'."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click"), "t2": _step("scroll"),
        })
        patterns = _empty_patterns(variable_steps=[pos], step_order=[0])

        inf = _mock_inferencer()
        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        assert result.decisions[0].decision_source == "llm"
        inf.infer.assert_called_once()

    def test_mixed_patterns_correct_sources(self):
        """Mixed patterns: deterministic→rule, variable→llm."""
        det_pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        var_pos = _pos(1, AlignmentType.VARIABLE, {
            "t1": _step("click"), "t2": _step("scroll"),
        })
        patterns = _empty_patterns(
            deterministic_steps=[det_pos],
            variable_steps=[var_pos],
            step_order=[0, 1],
        )

        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        )
        result = synth.synthesize(patterns)

        decisions_by_idx = {d.position_index: d for d in result.decisions}
        assert decisions_by_idx[0].decision_source == "rule"
        assert decisions_by_idx[1].decision_source == "llm"


# ---------------------------------------------------------------------------
# SynthesisReport.synthesis_strategy field
# ---------------------------------------------------------------------------

class TestSynthesisReportStrategy:
    """SynthesisReport includes the synthesis_strategy field matching the strategy used."""

    def test_rule_based_report_strategy(self):
        patterns = _empty_patterns()
        result = RuleBasedSynthesizer(action_executor=_mock_executor()).synthesize(patterns)
        assert result.report.synthesis_strategy == SynthesisStrategy.RULE_BASED.value

    def test_llm_report_strategy(self):
        patterns = _empty_patterns()
        result = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        ).synthesize(patterns)
        assert result.report.synthesis_strategy == SynthesisStrategy.LLM.value

    def test_hybrid_report_strategy(self):
        patterns = _empty_patterns()
        result = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        ).synthesize(patterns)
        assert result.report.synthesis_strategy == SynthesisStrategy.HYBRID.value


# ---------------------------------------------------------------------------
# ActionDecision records decision_source correctly per strategy
# ---------------------------------------------------------------------------

class TestActionDecisionSource:
    """ActionDecision.decision_source is correct per strategy type."""

    def test_rule_based_all_decisions_rule(self):
        """RuleBasedSynthesizer: every decision has source='rule', confidence=1.0."""
        det = _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click", target="a")})
        var = _pos(1, AlignmentType.VARIABLE, {"t1": _step("click"), "t2": _step("scroll")})
        patterns = _empty_patterns(
            deterministic_steps=[det], variable_steps=[var], step_order=[0, 1],
        )

        result = RuleBasedSynthesizer(action_executor=_mock_executor()).synthesize(patterns)

        for d in result.decisions:
            assert d.decision_source == "rule"
            assert d.confidence == 1.0
            assert d.reasoning is None

    def test_llm_all_decisions_llm(self):
        """LLMSynthesizer: every decision has source='llm'."""
        det = _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click", target="a")})
        var = _pos(1, AlignmentType.VARIABLE, {"t1": _step("click"), "t2": _step("scroll")})
        patterns = _empty_patterns(
            deterministic_steps=[det], variable_steps=[var], step_order=[0, 1],
        )

        result = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        ).synthesize(patterns)

        for d in result.decisions:
            assert d.decision_source == "llm"

    def test_hybrid_decisions_mixed(self):
        """HybridSynthesizer: deterministic/parameterizable/optional→rule, variable→llm."""
        det = _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click", target="a")})
        param = _pos(1, AlignmentType.PARAMETERIZABLE, {
            "t1": _step("input_text", target="f", args={"text": "x"}),
        })
        info = ParameterizableInfo(variable_args={"text": "q"}, constant_args={})
        opt = _pos(2, AlignmentType.OPTIONAL, {"t1": _step("click", target="p"), "t2": None})
        var = _pos(3, AlignmentType.VARIABLE, {"t1": _step("click"), "t2": _step("scroll")})

        patterns = _empty_patterns(
            deterministic_steps=[det],
            parameterizable_steps=[(param, info)],
            optional_steps=[opt],
            variable_steps=[var],
            step_order=[0, 1, 2, 3],
        )

        result = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=_mock_inferencer(),
        ).synthesize(patterns)

        decisions_by_idx = {d.position_index: d for d in result.decisions}
        assert decisions_by_idx[0].decision_source == "rule"   # deterministic
        assert decisions_by_idx[1].decision_source == "rule"   # parameterizable
        assert decisions_by_idx[2].decision_source == "rule"   # optional
        assert decisions_by_idx[3].decision_source == "llm"    # variable


# ---------------------------------------------------------------------------
# _parse_decision unit tests
# ---------------------------------------------------------------------------

class TestParseDecision:
    """Unit tests for GraphSynthesizer._parse_decision static method."""

    def test_dict_response(self):
        result = GraphSynthesizer._parse_decision({
            "action_type": "click",
            "target": "btn",
            "args": {"x": 1},
            "confidence": 0.9,
            "reasoning": "test",
        })
        assert result["action_type"] == "click"
        assert result["target"] == "btn"
        assert result["args"] == {"x": 1}
        assert result["confidence"] == 0.9
        assert result["reasoning"] == "test"

    def test_json_string(self):
        result = GraphSynthesizer._parse_decision(
            '{"action_type": "scroll", "confidence": 0.7}'
        )
        assert result["action_type"] == "scroll"
        assert result["confidence"] == 0.7
        assert result["target"] is None
        assert result["args"] is None

    def test_plain_string_fallback(self):
        result = GraphSynthesizer._parse_decision("just some text")
        assert result["action_type"] is None
        assert result["reasoning"] == "just some text"

    def test_empty_string(self):
        result = GraphSynthesizer._parse_decision("")
        assert result["action_type"] is None
        assert result["reasoning"] is None

    def test_confidence_clamped_high(self):
        result = GraphSynthesizer._parse_decision(
            '{"action_type": "x", "confidence": 5.0}'
        )
        assert result["confidence"] == 1.0

    def test_confidence_clamped_low(self):
        result = GraphSynthesizer._parse_decision(
            '{"action_type": "x", "confidence": -0.5}'
        )
        assert result["confidence"] == 0.0

    def test_invalid_confidence_ignored(self):
        result = GraphSynthesizer._parse_decision(
            '{"action_type": "click", "confidence": "not_a_number"}'
        )
        assert result["action_type"] == "click"
        assert result["confidence"] is None


# ---------------------------------------------------------------------------
# LLM response parsing — LLMSynthesizer
# ---------------------------------------------------------------------------

class TestLLMResponseParsing:
    """Test that LLMSynthesizer parses structured LLM responses."""

    def test_json_string_response_overrides_defaults(self):
        """When LLM returns valid JSON, parsed values override defaults."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn-old"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        inf = _mock_inferencer(
            '{"action_type": "scroll", "target": "content-area", '
            '"args": {"direction": "down"}, "confidence": 0.95, '
            '"reasoning": "Scrolling is more appropriate"}'
        )
        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.action_type == "scroll"
        assert d.target == "content-area"
        assert d.args == {"direction": "down"}
        assert d.confidence == 0.95
        assert d.reasoning == "Scrolling is more appropriate"

        # Graph action should reflect the LLM decision.
        action = result.graph._nodes[0]._actions[0]
        assert action.type == "scroll"

    def test_dict_response_overrides_defaults(self):
        """When inferencer returns a dict directly, parsed values override."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click"), "t2": _step("scroll"),
        })
        patterns = _empty_patterns(variable_steps=[pos], step_order=[0])

        inf = MagicMock()
        inf.infer.return_value = {
            "action_type": "input_text",
            "target": "search-box",
            "args": {"text": "{query}"},
            "confidence": 0.85,
            "reasoning": "This is actually a search step",
        }
        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        d = result.decisions[0]
        assert d.action_type == "input_text"
        assert d.target == "search-box"
        assert d.args == {"text": "{query}"}

    def test_plain_string_falls_back_to_rule_based(self):
        """Non-JSON string response falls back to rule-based defaults."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn-submit"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        inf = _mock_inferencer("This looks like a good action")
        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        d = result.decisions[0]
        assert d.action_type == "click"  # fallback to rule-based
        assert d.reasoning == "This looks like a good action"

    def test_inferencer_exception_falls_back(self):
        """Inferencer raising exception falls back to rule-based defaults."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        inf = MagicMock()
        inf.infer.side_effect = RuntimeError("API error")
        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        d = result.decisions[0]
        assert d.action_type == "click"  # fallback
        assert d.confidence == 0.5
        assert "failed" in d.reasoning.lower()

    def test_partial_json_uses_available_fields(self):
        """JSON with only some fields uses those and falls back for rest."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        inf = _mock_inferencer('{"action_type": "hover", "confidence": 0.7}')
        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        d = result.decisions[0]
        assert d.action_type == "hover"
        assert d.target is None  # from parsed JSON (not present)
        assert d.confidence == 0.7


# ---------------------------------------------------------------------------
# LLM response parsing — HybridSynthesizer
# ---------------------------------------------------------------------------

class TestHybridLLMResponseParsing:
    """Test HybridSynthesizer parses LLM responses for ambiguous patterns."""

    def test_variable_with_json_response(self):
        """VARIABLE pattern with structured LLM response uses parsed values."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click"), "t2": _step("scroll"),
        })
        patterns = _empty_patterns(variable_steps=[pos], step_order=[0])

        inf = _mock_inferencer(
            '{"action_type": "click", "target": "nav-menu", '
            '"confidence": 0.9, "reasoning": "Navigation click"}'
        )
        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        d = result.decisions[0]
        assert d.action_type == "click"
        assert d.target == "nav-menu"
        assert d.decision_source == "llm"
        assert d.confidence == 0.9

    def test_deterministic_ignores_inferencer(self):
        """DETERMINISTIC pattern still uses rules, inferencer not called."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        patterns = _empty_patterns(deterministic_steps=[pos], step_order=[0])

        inf = _mock_inferencer(
            '{"action_type": "scroll", "confidence": 0.99}'
        )
        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        d = result.decisions[0]
        assert d.action_type == "click"  # rule-based, not LLM
        assert d.decision_source == "rule"
        inf.infer.assert_not_called()

    def test_variable_plain_string_falls_back(self):
        """VARIABLE with non-JSON string falls back to agent node."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click"), "t2": _step("scroll"),
        })
        patterns = _empty_patterns(variable_steps=[pos], step_order=[0])

        inf = _mock_inferencer("Not sure about this one")
        synth = HybridSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        result = synth.synthesize(patterns)

        d = result.decisions[0]
        assert d.action_type == "meta_workflow_agent"  # rule-based fallback
        assert d.reasoning == "Not sure about this one"


# ---------------------------------------------------------------------------
# Rich prompt content verification
# ---------------------------------------------------------------------------

class TestRichPromptContainsSteps:
    """Test that the prompt sent to the inferencer includes observed steps."""

    def test_prompt_includes_trace_steps(self):
        """The prompt passed to infer() contains observed step details."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click", target="btn-a"),
            "t2": _step("scroll", target="page"),
        })
        patterns = _empty_patterns(variable_steps=[pos], step_order=[0])

        inf = _mock_inferencer()
        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        synth.synthesize(patterns)

        call_args = inf.infer.call_args
        prompt = call_args[0][0]
        assert "click" in prompt
        assert "scroll" in prompt
        assert "btn-a" in prompt
        assert "page" in prompt
        assert "JSON" in prompt

    def test_prompt_includes_param_info(self):
        """Parameterizable info is included in the prompt."""
        pos = _pos(0, AlignmentType.PARAMETERIZABLE, {
            "t1": _step("input_text", target="field", args={"text": "hello"}),
        })
        info = ParameterizableInfo(
            variable_args={"text": "search_query"}, constant_args={"delay": 100},
        )
        patterns = _empty_patterns(
            parameterizable_steps=[(pos, info)], step_order=[0],
        )

        inf = _mock_inferencer()
        synth = LLMSynthesizer(
            action_executor=_mock_executor(), inferencer=inf,
        )
        synth.synthesize(patterns)

        call_args = inf.infer.call_args
        prompt = call_args[0][0]
        assert "search_query" in prompt
        assert "Parameterizable" in prompt

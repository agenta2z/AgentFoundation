"""Unit tests for GraphSynthesizer.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.8
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from science_modeling_tools.automation.meta_agent.models import (
    AlignedPosition,
    AlignmentType,
    BranchPattern,
    ExtractedPatterns,
    LoopPattern,
    ParameterizableInfo,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.synthesizer import GraphSynthesizer, RuleBasedSynthesizer
from science_modeling_tools.automation.meta_agent.target_converter import (
    TargetSpec,
    TargetSpecWithFallback,
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
        index=index,
        alignment_type=atype,
        steps=steps,
        confidence=confidence,
    )


def _empty_patterns(**overrides) -> ExtractedPatterns:
    """Return an ExtractedPatterns with all lists empty, then apply overrides."""
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
    """Create a mock action executor for ActionGraph construction."""
    return MagicMock()


def _synthesizer(**kwargs):
    """Create a RuleBasedSynthesizer with a mock executor."""
    return RuleBasedSynthesizer(action_executor=_mock_executor(), **kwargs)


# ---------------------------------------------------------------------------
# Deterministic step synthesis  (Requirement 6.1)
# ---------------------------------------------------------------------------

class TestDeterministicStep:
    """A single deterministic step produces a standard action in the graph."""

    def test_single_deterministic_step(self):
        """One deterministic click step → one action in the graph."""
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn-submit"),
        })
        patterns = _empty_patterns(
            deterministic_steps=[pos],
            step_order=[0],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        graph, report = result.graph, result.report

        # The graph's root node should have exactly one action.
        assert len(graph._nodes[0]._actions) == 1
        action = graph._nodes[0]._actions[0]
        assert action.type == "click"
        assert report.deterministic_count == 1

    def test_deterministic_step_with_target_spec(self):
        """Deterministic step with TargetSpecWithFallback target is converted."""
        target = TargetSpecWithFallback(strategies=[
            TargetSpec(strategy="id", value="login-btn"),
        ])
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target=target),
        })
        patterns = _empty_patterns(
            deterministic_steps=[pos],
            step_order=[0],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        graph, report = result.graph, result.report

        action = graph._nodes[0]._actions[0]
        assert action.type == "click"
        # Target should be converted to graph-compatible type.
        assert action.target is not None


# ---------------------------------------------------------------------------
# Parameterizable step synthesis  (Requirement 6.2)
# ---------------------------------------------------------------------------

class TestParameterizableStep:
    """Parameterizable steps produce actions with template variable placeholders."""

    def test_parameterizable_step_with_template_vars(self):
        """A parameterizable input_text step gets {search_query} placeholder."""
        pos = _pos(0, AlignmentType.PARAMETERIZABLE, {
            "t1": _step("input_text", target="search-box", args={"text": "hello"}),
        })
        info = ParameterizableInfo(
            variable_args={"text": "search_query"},
            constant_args={"delay": 100},
        )
        patterns = _empty_patterns(
            parameterizable_steps=[(pos, info)],
            step_order=[0],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        graph, report = result.graph, result.report

        action = graph._nodes[0]._actions[0]
        assert action.type == "input_text"
        assert action.args["text"] == "{search_query}"
        assert action.args["delay"] == 100
        assert report.parameterizable_count == 1

    def test_template_variables_in_report(self):
        """Template variable names appear in the SynthesisReport."""
        pos = _pos(0, AlignmentType.PARAMETERIZABLE, {
            "t1": _step("input_text", target="field", args={"text": "x"}),
        })
        info = ParameterizableInfo(
            variable_args={"text": "user_name"},
            constant_args={},
        )
        patterns = _empty_patterns(
            parameterizable_steps=[(pos, info)],
            step_order=[0],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)

        assert "user_name" in result.report.template_variables


# ---------------------------------------------------------------------------
# Variable step synthesis  (Requirement 6.3)
# ---------------------------------------------------------------------------

class TestVariableStep:
    """Variable steps produce Agent Node actions."""

    def test_variable_step_creates_agent_node(self):
        """A variable step is synthesized as an agent action type."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click", metadata={"variants": {"click": 2, "scroll": 1}}),
            "t2": _step("scroll"),
        })
        patterns = _empty_patterns(
            variable_steps=[pos],
            step_order=[0],
        )

        synth = _synthesizer(agent_action_type="meta_workflow_agent")
        result = synth.synthesize(patterns)
        graph, report = result.graph, result.report

        action = graph._nodes[0]._actions[0]
        assert action.type == "meta_workflow_agent"
        assert report.agent_node_count == 1

    def test_variable_step_description_includes_variants(self):
        """The agent node target describes observed variants."""
        pos = _pos(0, AlignmentType.VARIABLE, {
            "t1": _step("click", metadata={"variants": {"click": 3, "input_text": 1}}),
        })
        patterns = _empty_patterns(
            variable_steps=[pos],
            step_order=[0],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        graph = result.graph

        action = graph._nodes[0]._actions[0]
        # Target should be a description string mentioning the variants.
        target_str = str(action.target)
        assert "click" in target_str
        assert "input_text" in target_str


# ---------------------------------------------------------------------------
# Optional step synthesis  (Requirement 6.4)
# ---------------------------------------------------------------------------

class TestOptionalStep:
    """Optional steps produce actions with no_action_if_target_not_found=True."""

    def test_optional_step_sets_flag(self):
        """An optional step has no_action_if_target_not_found=True."""
        pos = _pos(0, AlignmentType.OPTIONAL, {
            "t1": _step("click", target="dismiss-popup"),
            "t2": None,
        })
        patterns = _empty_patterns(
            optional_steps=[pos],
            step_order=[0],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        graph, report = result.graph, result.report

        action = graph._nodes[0]._actions[0]
        assert action.type == "click"
        assert action.no_action_if_target_not_found is True
        assert report.optional_count == 1


# ---------------------------------------------------------------------------
# User input boundary synthesis  (Requirement 6.5)
# ---------------------------------------------------------------------------

class TestUserInputBoundary:
    """User input boundaries produce wait(True) actions."""

    def test_user_input_boundary_creates_wait_true(self):
        """A user input boundary index produces a wait action with target=True."""
        patterns = _empty_patterns(
            user_input_boundaries=[0],
            step_order=[0],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        graph, report = result.graph, result.report

        action = graph._nodes[0]._actions[0]
        assert action.type == "wait"
        assert action.target is True or action.target == 1  # True may be coerced to int
        assert report.user_input_boundary_count == 1


# ---------------------------------------------------------------------------
# SynthesisReport counts  (Requirement 6.8)
# ---------------------------------------------------------------------------

class TestSynthesisReportCounts:
    """SynthesisReport counts match the patterns that were synthesized."""

    def test_report_counts_match_mixed_patterns(self):
        """A mix of pattern types produces matching report counts."""
        det_pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target="btn"),
        })
        param_pos = _pos(1, AlignmentType.PARAMETERIZABLE, {
            "t1": _step("input_text", target="field", args={"text": "a"}),
        })
        param_info = ParameterizableInfo(
            variable_args={"text": "query"},
            constant_args={},
        )
        var_pos = _pos(2, AlignmentType.VARIABLE, {
            "t1": _step("click"),
            "t2": _step("scroll"),
        })
        opt_pos = _pos(3, AlignmentType.OPTIONAL, {
            "t1": _step("click", target="popup"),
            "t2": None,
        })

        patterns = _empty_patterns(
            deterministic_steps=[det_pos],
            parameterizable_steps=[(param_pos, param_info)],
            variable_steps=[var_pos],
            optional_steps=[opt_pos],
            user_input_boundaries=[4],
            step_order=[0, 1, 2, 3, 4],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        report = result.report

        assert report.deterministic_count == 1
        assert report.parameterizable_count == 1
        assert report.agent_node_count == 1
        assert report.optional_count == 1
        assert report.user_input_boundary_count == 1
        assert report.total_steps == 5

    def test_empty_patterns_produce_zero_counts(self):
        """Empty patterns produce a report with all zero counts."""
        patterns = _empty_patterns()

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        report = result.report

        assert report.total_steps == 0
        assert report.deterministic_count == 0
        assert report.parameterizable_count == 0
        assert report.agent_node_count == 0
        assert report.optional_count == 0
        assert report.user_input_boundary_count == 0
        assert report.branch_count == 0
        assert report.loop_count == 0

    def test_target_strategy_coverage_tracked(self):
        """Target strategy coverage is tracked in the report."""
        target = TargetSpecWithFallback(strategies=[
            TargetSpec(strategy="id", value="submit"),
            TargetSpec(strategy="css", value=".submit-btn"),
        ])
        pos = _pos(0, AlignmentType.DETERMINISTIC, {
            "t1": _step("click", target=target),
        })
        patterns = _empty_patterns(
            deterministic_steps=[pos],
            step_order=[0],
        )

        synth = _synthesizer()
        result = synth.synthesize(patterns)
        report = result.report

        assert "id" in report.target_strategy_coverage
        assert "css" in report.target_strategy_coverage

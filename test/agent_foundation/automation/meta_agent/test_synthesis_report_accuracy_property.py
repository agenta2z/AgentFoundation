"""Property test for synthesis report accuracy (Property 16).

**Validates: Requirements 6.8**

When GraphSynthesizer.synthesize() is called with ExtractedPatterns, the
resulting SynthesisReport counts SHALL match the actual pattern counts:
- deterministic_count == len(deterministic_steps)
- parameterizable_count == len(parameterizable_steps)
- agent_node_count == len(variable_steps)
- optional_count == len(optional_steps)
- user_input_boundary_count == len(user_input_boundaries)
- branch_count == len(branch_patterns)
- loop_count == len(loop_patterns)
- total_steps == sum of all individual counts
- template_variables contains all variable arg names from parameterizable steps
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from agent_foundation.automation.meta_agent.synthesizer import GraphSynthesizer, RuleBasedSynthesizer
from agent_foundation.automation.meta_agent.models import (
    AlignedPosition,
    AlignmentType,
    BranchPattern,
    ExtractedPatterns,
    LoopPattern,
    ParameterizableInfo,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url"]


@st.composite
def deterministic_step(draw, index: int) -> AlignedPosition:
    action_type = draw(st.sampled_from(ACTION_TYPES))
    step = TraceStep(action_type=action_type, target=f"det-{index}")
    return AlignedPosition(
        index=index,
        alignment_type=AlignmentType.DETERMINISTIC,
        steps={"t1": step},
    )


@st.composite
def variable_step(draw, index: int) -> AlignedPosition:
    action_type = draw(st.sampled_from(ACTION_TYPES))
    step = TraceStep(
        action_type=action_type,
        metadata={"variants": {action_type: 3}},
    )
    return AlignedPosition(
        index=index,
        alignment_type=AlignmentType.VARIABLE,
        steps={"t1": step},
    )


@st.composite
def optional_step(draw, index: int) -> AlignedPosition:
    action_type = draw(st.sampled_from(ACTION_TYPES))
    step = TraceStep(action_type=action_type, target=f"opt-{index}")
    return AlignedPosition(
        index=index,
        alignment_type=AlignmentType.OPTIONAL,
        steps={"t1": step, "t2": None},
    )


@st.composite
def parameterizable_step(draw, index: int):
    action_type = draw(st.sampled_from(ACTION_TYPES))
    var_name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))
    step = TraceStep(
        action_type=action_type,
        target=f"param-{index}",
        args={"text": "sample", "delay": 100},
    )
    pos = AlignedPosition(
        index=index,
        alignment_type=AlignmentType.PARAMETERIZABLE,
        steps={"t1": step},
    )
    info = ParameterizableInfo(
        variable_args={"text": var_name},
        constant_args={"delay": 100},
    )
    return pos, info


@st.composite
def branch_pattern(draw, index: int) -> BranchPattern:
    """Generate a branch pattern at the given index."""
    action_type_a = draw(st.sampled_from(ACTION_TYPES))
    action_type_b = draw(st.sampled_from(ACTION_TYPES))
    step_a = TraceStep(action_type=action_type_a, target=f"br-a-{index}")
    step_b = TraceStep(action_type=action_type_b, target=f"br-b-{index}")
    pos_a = AlignedPosition(
        index=index + 1000,
        alignment_type=AlignmentType.DETERMINISTIC,
        steps={"t1": step_a},
    )
    pos_b = AlignedPosition(
        index=index + 2000,
        alignment_type=AlignmentType.DETERMINISTIC,
        steps={"t1": step_b},
    )
    return BranchPattern(
        branch_point_index=index,
        branches={"branch_a": [pos_a], "branch_b": [pos_b]},
        condition_description="placeholder",
    )


@st.composite
def loop_pattern(draw, start_index: int) -> LoopPattern:
    """Generate a loop pattern starting at start_index with 2 body steps."""
    body_steps = []
    for offset in range(2):
        action_type = draw(st.sampled_from(ACTION_TYPES))
        step = TraceStep(action_type=action_type, target=f"loop-{start_index + offset}")
        body_steps.append(
            AlignedPosition(
                index=start_index + offset,
                alignment_type=AlignmentType.DETERMINISTIC,
                steps={"t1": step},
            )
        )
    return LoopPattern(
        body_start=start_index,
        body_end=start_index + 1,
        min_iterations=1,
        max_iterations=draw(st.integers(min_value=2, max_value=5)),
        body_steps=body_steps,
    )


@st.composite
def mixed_patterns_with_all_types(draw):
    """Generate ExtractedPatterns with a random mix of all step types.

    Includes deterministic, variable, optional, parameterizable steps,
    user input boundaries, branch patterns, and loop patterns.
    Index ranges are kept non-overlapping to avoid conflicts.
    """
    # Simple steps: indices 0..n-1
    n_det = draw(st.integers(min_value=0, max_value=5))
    n_var = draw(st.integers(min_value=0, max_value=5))
    n_opt = draw(st.integers(min_value=0, max_value=5))
    n_param = draw(st.integers(min_value=0, max_value=5))
    n_uib = draw(st.integers(min_value=0, max_value=3))
    n_branch = draw(st.integers(min_value=0, max_value=2))
    n_loop = draw(st.integers(min_value=0, max_value=2))

    idx = 0
    det_steps = []
    for _ in range(n_det):
        det_steps.append(draw(deterministic_step(idx)))
        idx += 1

    var_steps = []
    for _ in range(n_var):
        var_steps.append(draw(variable_step(idx)))
        idx += 1

    opt_steps = []
    for _ in range(n_opt):
        opt_steps.append(draw(optional_step(idx)))
        idx += 1

    param_steps = []
    for _ in range(n_param):
        param_steps.append(draw(parameterizable_step(idx)))
        idx += 1

    # User input boundaries use their own indices
    uib_indices = []
    for _ in range(n_uib):
        uib_indices.append(idx)
        idx += 1

    # step_order covers all simple steps + uib indices
    step_order = list(range(idx))

    # Branch patterns use high indices (500+) for branch_point_index
    branches = []
    for i in range(n_branch):
        bp = draw(branch_pattern(500 + i * 10))
        branches.append(bp)

    # Loop patterns use high indices (800+) to avoid overlap
    loops = []
    loop_start = 800
    for i in range(n_loop):
        lp = draw(loop_pattern(loop_start))
        loops.append(lp)
        loop_start += 10  # leave room between loops

    return ExtractedPatterns(
        deterministic_steps=det_steps,
        parameterizable_steps=param_steps,
        variable_steps=var_steps,
        optional_steps=opt_steps,
        branch_patterns=branches,
        loop_patterns=loops,
        user_input_boundaries=uib_indices,
        step_order=step_order,
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(patterns=mixed_patterns_with_all_types())
def test_synthesis_report_counts_match_pattern_counts(patterns: ExtractedPatterns):
    """Property 16: Synthesis report accuracy — individual counts.

    Each count in the SynthesisReport SHALL equal the number of
    corresponding patterns in the input ExtractedPatterns.

    **Validates: Requirements 6.8**
    """
    synth = RuleBasedSynthesizer(
        action_executor=MagicMock(),
        agent_action_type="meta_workflow_agent",
    )

    result = synth.synthesize(patterns)
    report = result.report

    assert report.deterministic_count == len(patterns.deterministic_steps), (
        f"deterministic_count: expected {len(patterns.deterministic_steps)}, "
        f"got {report.deterministic_count}"
    )
    assert report.parameterizable_count == len(patterns.parameterizable_steps), (
        f"parameterizable_count: expected {len(patterns.parameterizable_steps)}, "
        f"got {report.parameterizable_count}"
    )
    assert report.agent_node_count == len(patterns.variable_steps), (
        f"agent_node_count: expected {len(patterns.variable_steps)}, "
        f"got {report.agent_node_count}"
    )
    assert report.optional_count == len(patterns.optional_steps), (
        f"optional_count: expected {len(patterns.optional_steps)}, "
        f"got {report.optional_count}"
    )
    assert report.user_input_boundary_count == len(patterns.user_input_boundaries), (
        f"user_input_boundary_count: expected {len(patterns.user_input_boundaries)}, "
        f"got {report.user_input_boundary_count}"
    )
    assert report.branch_count == len(patterns.branch_patterns), (
        f"branch_count: expected {len(patterns.branch_patterns)}, "
        f"got {report.branch_count}"
    )
    assert report.loop_count == len(patterns.loop_patterns), (
        f"loop_count: expected {len(patterns.loop_patterns)}, "
        f"got {report.loop_count}"
    )


@settings(max_examples=200, deadline=None)
@given(patterns=mixed_patterns_with_all_types())
def test_synthesis_report_total_is_sum_of_counts(patterns: ExtractedPatterns):
    """Property 16: Synthesis report accuracy — total_steps is the sum.

    total_steps SHALL equal the sum of all individual category counts.

    **Validates: Requirements 6.8**
    """
    synth = RuleBasedSynthesizer(
        action_executor=MagicMock(),
        agent_action_type="meta_workflow_agent",
    )

    result = synth.synthesize(patterns)
    report = result.report

    expected_total = (
        report.deterministic_count
        + report.parameterizable_count
        + report.agent_node_count
        + report.optional_count
        + report.user_input_boundary_count
        + report.branch_count
        + report.loop_count
    )
    assert report.total_steps == expected_total, (
        f"total_steps: expected {expected_total}, got {report.total_steps}"
    )


@settings(max_examples=200, deadline=None)
@given(patterns=mixed_patterns_with_all_types())
def test_synthesis_report_template_variables_from_parameterizable(
    patterns: ExtractedPatterns,
):
    """Property 16: Synthesis report accuracy — template variables.

    template_variables SHALL contain all variable arg names from
    parameterizable steps (deduplicated and sorted).

    **Validates: Requirements 6.8**
    """
    synth = RuleBasedSynthesizer(
        action_executor=MagicMock(),
        agent_action_type="meta_workflow_agent",
    )

    result = synth.synthesize(patterns)
    report = result.report

    expected_vars = sorted(
        {
            var_name
            for _, info in patterns.parameterizable_steps
            for var_name in info.variable_args.values()
        }
    )
    assert report.template_variables == expected_vars, (
        f"template_variables: expected {expected_vars}, got {report.template_variables}"
    )

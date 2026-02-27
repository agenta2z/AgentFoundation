"""Property test for synthesis type mapping (Property 15).

**Validates: Requirements 6.1, 6.2, 6.3, 6.4**

For any extracted pattern, the synthesized ActionGraph action type SHALL be:
- A standard registered action type for Deterministic_Steps and Parameterizable_Steps
- The configured agent action type for Variable_Steps
- A standard action with ``no_action_if_target_not_found=True`` for Optional_Steps
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent_foundation.automation.meta_agent.synthesizer import GraphSynthesizer, RuleBasedSynthesizer
from agent_foundation.automation.meta_agent.models import (
    AlignedPosition,
    AlignmentType,
    ExtractedPatterns,
    ParameterizableInfo,
    TraceStep,
)
from agent_foundation.automation.meta_agent.target_converter import (
    TargetSpec,
    TargetSpecWithFallback,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url"]


@st.composite
def deterministic_step(draw, index: int) -> AlignedPosition:
    """Generate a deterministic aligned position."""
    action_type = draw(st.sampled_from(ACTION_TYPES))
    target = draw(st.just(f"target-{index}"))
    step = TraceStep(action_type=action_type, target=target)
    return AlignedPosition(
        index=index,
        alignment_type=AlignmentType.DETERMINISTIC,
        steps={"t1": step},
    )


@st.composite
def variable_step(draw, index: int) -> AlignedPosition:
    """Generate a variable aligned position with variant metadata."""
    action_type = draw(st.sampled_from(ACTION_TYPES))
    variants = {action_type: draw(st.integers(min_value=1, max_value=5))}
    step = TraceStep(
        action_type=action_type,
        metadata={"variants": variants},
    )
    return AlignedPosition(
        index=index,
        alignment_type=AlignmentType.VARIABLE,
        steps={"t1": step},
    )


@st.composite
def optional_step(draw, index: int) -> AlignedPosition:
    """Generate an optional aligned position (present in some traces, absent in others)."""
    action_type = draw(st.sampled_from(ACTION_TYPES))
    target = draw(st.just(f"opt-target-{index}"))
    step = TraceStep(action_type=action_type, target=target)
    return AlignedPosition(
        index=index,
        alignment_type=AlignmentType.OPTIONAL,
        steps={"t1": step, "t2": None},
    )


@st.composite
def parameterizable_step(draw, index: int):
    """Generate a parameterizable aligned position with ParameterizableInfo."""
    action_type = draw(st.sampled_from(ACTION_TYPES))
    target = draw(st.just(f"param-target-{index}"))
    step = TraceStep(
        action_type=action_type,
        target=target,
        args={"text": "sample", "delay": 100},
    )
    pos = AlignedPosition(
        index=index,
        alignment_type=AlignmentType.PARAMETERIZABLE,
        steps={"t1": step},
    )
    info = ParameterizableInfo(
        variable_args={"text": f"var_{index}"},
        constant_args={"delay": 100},
    )
    return pos, info


@st.composite
def mixed_patterns(draw):
    """Generate ExtractedPatterns with a random mix of step types.

    Each generated pattern has at least one step. Steps are assigned
    to deterministic, variable, optional, or parameterizable categories
    randomly. No loops or branches are generated (focused on type mapping).
    """
    n_steps = draw(st.integers(min_value=1, max_value=10))

    det_steps = []
    var_steps = []
    opt_steps = []
    param_steps = []
    step_order = []

    for i in range(n_steps):
        category = draw(st.sampled_from(["deterministic", "variable", "optional", "parameterizable"]))
        step_order.append(i)

        if category == "deterministic":
            det_steps.append(draw(deterministic_step(i)))
        elif category == "variable":
            var_steps.append(draw(variable_step(i)))
        elif category == "optional":
            opt_steps.append(draw(optional_step(i)))
        else:
            param_steps.append(draw(parameterizable_step(i)))

    return ExtractedPatterns(
        deterministic_steps=det_steps,
        parameterizable_steps=param_steps,
        variable_steps=var_steps,
        optional_steps=opt_steps,
        branch_patterns=[],
        loop_patterns=[],
        user_input_boundaries=[],
        step_order=step_order,
    )


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(patterns=mixed_patterns())
def test_synthesis_type_mapping(patterns: ExtractedPatterns):
    """Property 15: Synthesis type mapping.

    - Deterministic → registered action type (e.g., "click", "input_text")
    - Parameterizable → registered action type (same as deterministic)
    - Variable → agent action type ("meta_workflow_agent")
    - Optional → no_action_if_target_not_found=True

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """
    agent_action_type = "meta_workflow_agent"
    synth = RuleBasedSynthesizer(
        action_executor=MagicMock(),
        agent_action_type=agent_action_type,
    )

    result = synth.synthesize(patterns)
    graph = result.graph

    actions = graph._nodes[0]._actions

    # Build expected action count
    expected_count = (
        len(patterns.deterministic_steps)
        + len(patterns.parameterizable_steps)
        + len(patterns.variable_steps)
        + len(patterns.optional_steps)
    )
    assert len(actions) == expected_count

    # Walk step_order and verify each action's type mapping
    action_idx = 0

    # Build lookup sets for quick category identification
    det_indices = {p.index for p in patterns.deterministic_steps}
    var_indices = {p.index for p in patterns.variable_steps}
    opt_indices = {p.index for p in patterns.optional_steps}
    param_indices = {p.index for p, _ in patterns.parameterizable_steps}

    # Map from position index to the original TraceStep's action_type
    det_action_types = {
        p.index: next(s.action_type for s in p.steps.values() if s is not None)
        for p in patterns.deterministic_steps
    }
    param_action_types = {
        p.index: next(s.action_type for s in p.steps.values() if s is not None)
        for p, _ in patterns.parameterizable_steps
    }
    opt_action_types = {
        p.index: next(s.action_type for s in p.steps.values() if s is not None)
        for p in patterns.optional_steps
    }

    for idx in patterns.step_order:
        action = actions[action_idx]

        if idx in det_indices:
            # Deterministic → original action type
            assert action.type == det_action_types[idx], (
                f"Deterministic step at index {idx}: expected {det_action_types[idx]}, got {action.type}"
            )
            assert action.no_action_if_target_not_found is False

        elif idx in param_indices:
            # Parameterizable → original action type
            assert action.type == param_action_types[idx], (
                f"Parameterizable step at index {idx}: expected {param_action_types[idx]}, got {action.type}"
            )
            assert action.no_action_if_target_not_found is False

        elif idx in var_indices:
            # Variable → agent action type
            assert action.type == agent_action_type, (
                f"Variable step at index {idx}: expected {agent_action_type}, got {action.type}"
            )

        elif idx in opt_indices:
            # Optional → original action type with no_action_if_target_not_found=True
            assert action.type == opt_action_types[idx], (
                f"Optional step at index {idx}: expected {opt_action_types[idx]}, got {action.type}"
            )
            assert action.no_action_if_target_not_found is True, (
                f"Optional step at index {idx}: no_action_if_target_not_found should be True"
            )

        action_idx += 1

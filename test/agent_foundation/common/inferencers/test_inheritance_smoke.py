"""Smoke tests for linear-workflow-inheritance refactor.

Verifies:
- ReflectiveInferencer and PlanThenImplementInferencer inherit from LinearWorkflowInferencer
- All public attrs are preserved on both classes
- Default reset_sessions_per_iteration is False on LWI
- DualInferencer and BreakdownThenAggregateInferencer are NOT subclasses of LWI

Requirements: 9.1, 9.2, 10.1, 10.2, 3.3, 15.1, 15.2, 15.3
"""

import attr
import pytest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
    BreakdownThenAggregateInferencer,
    DualInferencer,
    LinearWorkflowInferencer,
    ReflectiveInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)


# ── Task 7.2.1: Inheritance checks ──────────────────────────────────


def test_reflective_is_subclass_of_lwi():
    assert issubclass(ReflectiveInferencer, LinearWorkflowInferencer)


def test_pti_is_subclass_of_lwi():
    assert issubclass(PlanThenImplementInferencer, LinearWorkflowInferencer)


# ── Task 7.2: Unrelated inferencers NOT subclasses of LWI ──────────


def test_dual_inferencer_not_subclass_of_lwi():
    assert not issubclass(DualInferencer, LinearWorkflowInferencer)


def test_bta_not_subclass_of_lwi():
    assert not issubclass(BreakdownThenAggregateInferencer, LinearWorkflowInferencer)


# ── Task 7.2.3: ReflectiveInferencer public attrs preserved ────────


def test_reflective_public_attrs_preserved():
    """All public attrs from the original ReflectiveInferencer must exist."""
    expected_attrs = {
        "base_inferencer",
        "reflection_inferencer",
        "num_reflections",
        "reflection_style",
        "reflection_prompt_template",
        "reflection_prompt_formatter",
        "base_response_concat",
        "unpack_single_reflection",
        "unpack_single_response",
        "response_selector",
        "reflection_prompt_placeholder_inferencer_input",
        "reflection_prompt_placeholder_inferencer_response",
    }
    actual_attrs = {a.name for a in attr.fields(ReflectiveInferencer)}
    missing = expected_attrs - actual_attrs
    assert not missing, f"Missing attrs on ReflectiveInferencer: {missing}"


# ── Task 7.2.4: PlanThenImplementInferencer public attrs preserved ──


def test_pti_public_attrs_preserved():
    """All key public attrs from the original PTI must exist."""
    expected_attrs = {
        "planner_inferencer",
        "executor_inferencer",
        "executor_prompt_builder",
        "interactive",
        "planner_phase",
        "executor_phase",
        "plan_config_key",
        "implement_config_key",
        "planner_outputs_plan_to_file",
        "enable_planning",
        "enable_implementation",
        "enable_analysis",
        "enable_multiple_iterations",
        "max_meta_iterations",
        "analyzer_inferencer",
        "analysis_config_key",
        "analyzer_outputs_to_file",
        "results_subdirs",
        "reset_sessions_per_meta_iteration",
        "approve_all_iterations",
        "resume_workspace",
        "iteration_handoff_template",
    }
    actual_attrs = {a.name for a in attr.fields(PlanThenImplementInferencer)}
    missing = expected_attrs - actual_attrs
    assert not missing, f"Missing attrs on PlanThenImplementInferencer: {missing}"


# ── Task 7.2.5: Default reset_sessions_per_iteration is False ──────


def test_default_reset_sessions_per_iteration_is_false():
    """LWI's reset_sessions_per_iteration must default to False."""
    field = next(
        a for a in attr.fields(LinearWorkflowInferencer)
        if a.name == "reset_sessions_per_iteration"
    )
    assert field.default is False


# ── Task 7.1.3: __init__.py exports all existing symbols ───────────


def test_init_exports_all_symbols():
    """The flow_inferencers __init__.py must export all expected symbols."""
    from agent_foundation.common.inferencers.agentic_inferencers import flow_inferencers

    expected = [
        "LinearWorkflowInferencer",
        "WorkflowStepConfig",
        "ReflectiveInferencer",
        "DualInferencer",
        "BreakdownThenAggregateInferencer",
    ]
    for name in expected:
        assert hasattr(flow_inferencers, name), (
            f"flow_inferencers.__init__.py missing export: {name}"
        )

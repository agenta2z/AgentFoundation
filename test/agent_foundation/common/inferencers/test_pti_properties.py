"""Property-based tests for PlanThenImplementInferencer.

Uses Hypothesis to verify correctness properties across randomized inputs.
"""

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)


def _make_mock_inferencer():
    mock = MagicMock()
    mock.set_parent_debuggable = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Feature: linear-workflow-inheritance, Property 9: PTI preserves PlanThenImplementResponse return type
# Validates: Requirements 10.6
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    plan_text=st.text(max_size=100),
    executor_text=st.text(max_size=100),
    iteration=st.integers(min_value=1, max_value=5),
    plan_approved=st.one_of(st.none(), st.booleans()),
)
def test_pti_build_response_returns_pti_response(
    plan_text, executor_text, iteration, plan_approved
):
    """_build_response_from_state always returns PlanThenImplementResponse."""
    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_mock_inferencer(),
        executor_inferencer=_make_mock_inferencer(),
    )

    state = {
        "iteration": iteration,
        "current_input": "test input",
        "original_request": "test request",
        "plan_output_text": plan_text,
        "plan_file_path": None,
        "plan_approved": plan_approved,
        "executor_output_text": executor_text,
        "should_continue": False,
        "iteration_records": [],
    }

    result = pti._build_response_from_state(state)

    assert isinstance(result, PlanThenImplementResponse), (
        f"Expected PlanThenImplementResponse, got {type(result)}"
    )
    assert result.plan_output == plan_text
    assert result.plan_approved == plan_approved

"""Property-based tests for LinearWorkflowInferencer enhancements.

Uses Hypothesis to verify correctness properties across randomized inputs.
Each test is tagged with the feature and property number it validates.
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_inferencer(return_value="mock_result"):
    """Create a mock InferencerBase-like object with ainfer."""
    mock = MagicMock(spec=[])
    mock.ainfer = AsyncMock(return_value=return_value)
    return mock


def _make_lwi_with_workspace(step_configs, response_builder, workspace_path):
    """Create an LWI with workspace but checkpointing/serialization disabled.

    Manually sets _workspace after construction (instead of using workspace_path)
    to avoid _auto_enable_checkpointing and _save_final_result side effects.
    The workspace is available for markers and iteration dirs, but no
    checkpoint/resume or final result serialization occurs.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    lwi = LinearWorkflowInferencer(
        step_configs=step_configs,
        response_builder=response_builder,
        # Do NOT set workspace_path — we set _workspace manually below
    )
    # Set workspace manually so markers and iteration dirs work,
    # but _save_final_result and _auto_enable_checkpointing are skipped
    # (_save_final_result checks self._workspace which we set, so we also
    # need to monkey-patch _save_final_result to be a no-op)
    lwi._workspace = InferencerWorkspace(root=workspace_path)
    lwi.workspace_path = workspace_path
    # Prevent _auto_enable_checkpointing from enabling resume
    lwi._result_root_override = workspace_path
    # Prevent _save_final_result from causing recursion on self-referential state
    lwi._save_final_result = lambda state: None
    return lwi


# ---------------------------------------------------------------------------
# Property 1: Iteration counter tracks loop cycles
# Feature: linear-workflow-inheritance, Property 1: Iteration counter tracks loop cycles
# Validates: Requirements 1.1, 1.4
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(n_loops=st.integers(min_value=0, max_value=5))
def test_iteration_counter_tracks_loop_cycles(n_loops):
    """For any LWI with N loop-back triggers, state['iteration'] == N+1 after completion."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    exec_count = [0]

    def loop_step_fn(input_val, state):
        exec_count[0] += 1
        if exec_count[0] <= n_loops:
            state["iteration"] += 1
        return f"iter_{state['iteration']}"

    def loop_condition(state, result):
        return exec_count[0] <= n_loops

    configs = [
        WorkflowStepConfig(
            name="work",
            step_fn=loop_step_fn,
            output_state_key="work_output",
            loop_back_to="work" if n_loops > 0 else None,
            loop_condition=loop_condition if n_loops > 0 else None,
            max_loop_iterations=max(n_loops, 1),
            enable_result_save=False,
        ),
    ]

    captured_state = {}

    def response_builder(state):
        captured_state.update(state)
        return state.get("work_output", "")

    lwi = LinearWorkflowInferencer(
        step_configs=configs,
        response_builder=response_builder,
    )

    lwi.infer("start")

    assert captured_state["iteration"] == n_loops + 1



# ---------------------------------------------------------------------------
# Property 2: Iteration workspace directory creation
# Feature: linear-workflow-inheritance, Property 2: Iteration workspace directory creation
# Validates: Requirements 1.2, 1.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(n_iters=st.integers(min_value=1, max_value=5))
def test_iteration_workspace_directory_creation(n_iters):
    """For any LWI with workspace and N iterations, iteration workspace dirs exist."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        WorkflowStepConfig,
    )

    workspace = tempfile.mkdtemp()
    n_loops = n_iters - 1
    exec_count = [0]

    def loop_step_fn(input_val, state):
        exec_count[0] += 1
        if exec_count[0] <= n_loops:
            state["iteration"] += 1
        return f"iter_{state['iteration']}"

    def loop_condition(state, result):
        return exec_count[0] <= n_loops

    need_loop = n_loops > 0
    configs = [
        WorkflowStepConfig(
            name="work",
            step_fn=loop_step_fn,
            output_state_key="work_output",
            loop_back_to="work" if need_loop else None,
            loop_condition=loop_condition if need_loop else None,
            max_loop_iterations=max(n_loops, 1),
            enable_result_save=False,
        ),
    ]

    lwi = _make_lwi_with_workspace(
        step_configs=configs,
        response_builder=lambda state: state.get("work_output", ""),
        workspace_path=workspace,
    )

    lwi.infer("start")

    # Iteration 1 uses base workspace (no subdirectory)
    # Iterations 2..n_iters use {base}/iteration_{i}/
    for i in range(2, n_iters + 1):
        iter_dir = os.path.join(workspace, f"iteration_{i}")
        assert os.path.isdir(iter_dir), f"Expected iteration dir {iter_dir} to exist"


# ---------------------------------------------------------------------------
# Property 3: Step completion markers written for all completed steps
# Feature: linear-workflow-inheritance, Property 3: Step completion markers written for all completed steps
# Validates: Requirements 4.1, 4.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(n_steps=st.integers(min_value=1, max_value=5))
def test_step_completion_markers_written(n_steps):
    """For any LWI with workspace and K enabled steps, K marker files exist after completion."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        WorkflowStepConfig,
    )

    workspace = tempfile.mkdtemp()

    configs = []
    for i in range(n_steps):
        mock_inf = _make_mock_inferencer(f"result_{i}")
        configs.append(
            WorkflowStepConfig(
                name=f"step_{i}",
                inferencer=mock_inf,
                output_state_key=f"out_{i}",
                enable_result_save=False,
            )
        )

    lwi = _make_lwi_with_workspace(
        step_configs=configs,
        response_builder=lambda state: "done",
        workspace_path=workspace,
    )

    lwi.infer("test_input")

    # Check that marker files exist for each step
    artifacts_dir = os.path.join(workspace, "artifacts")
    for i in range(n_steps):
        marker_path = os.path.join(artifacts_dir, f".step_{i}_completed")
        assert os.path.isfile(marker_path), (
            f"Expected marker file {marker_path} to exist for step_{i}"
        )


# ---------------------------------------------------------------------------
# Property 4: Response text extraction type dispatch
# Feature: linear-workflow-inheritance, Property 4: Response text extraction type dispatch
# Validates: Requirements 5.1, 5.2, 5.3, 5.4
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(text=st.text())
def test_extract_response_text_type_dispatch(text):
    """For any result type, extract_response_text returns the correct string."""
    from agent_foundation.common.inferencers.agentic_inferencers.common import (
        DualInferencerResponse,
        InferencerResponse,
        ResponseSelectors,
        extract_response_text,
    )

    # Case 1: DualInferencerResponse -> str(result.base_response)
    dual = DualInferencerResponse(base_response=text)
    assert extract_response_text(dual) == str(text)

    # Case 2: InferencerResponse -> str(result.select_response())
    ir = InferencerResponse(
        base_response=text,
        response_selector=ResponseSelectors.BaseResponse,
    )
    assert extract_response_text(ir) == str(text)

    # Case 3: Plain string -> str(result)
    assert extract_response_text(text) == str(text)

    # All results are strings
    assert isinstance(extract_response_text(dual), str)
    assert isinstance(extract_response_text(ir), str)
    assert isinstance(extract_response_text(text), str)


# ---------------------------------------------------------------------------
# Property 5: Iteration records accumulate with loop cycles
# Feature: linear-workflow-inheritance, Property 5: Iteration records accumulate with loop cycles
# Validates: Requirements 6.1, 6.2
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(n_loops=st.integers(min_value=0, max_value=5))
def test_iteration_records_accumulate(n_loops):
    """For any LWI with N completed loop cycles, iteration_records has N entries."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    exec_count = [0]

    def loop_step_fn(input_val, state):
        exec_count[0] += 1
        if exec_count[0] <= n_loops:
            state["iteration"] += 1
        return f"iter_{state['iteration']}"

    def loop_condition(state, result):
        return exec_count[0] <= n_loops

    configs = [
        WorkflowStepConfig(
            name="work",
            step_fn=loop_step_fn,
            output_state_key="work_output",
            loop_back_to="work" if n_loops > 0 else None,
            loop_condition=loop_condition if n_loops > 0 else None,
            max_loop_iterations=max(n_loops, 1),
            enable_result_save=False,
        ),
    ]

    captured_state = {}

    def response_builder(state):
        captured_state.update(state)
        return state.get("work_output", "")

    lwi = LinearWorkflowInferencer(
        step_configs=configs,
        response_builder=response_builder,
    )

    lwi.infer("start")

    # _record_iteration is called inside _setup_iteration, which fires
    # when iteration changes. Each loop-back that increments iteration
    # triggers one record. So N loop-backs = N records.
    records = captured_state.get("iteration_records", [])
    assert len(records) == n_loops, (
        f"Expected {n_loops} iteration records, got {len(records)}"
    )


# ---------------------------------------------------------------------------
# Property 6: Default iteration snapshot excludes private keys
# Feature: linear-workflow-inheritance, Property 6: Default iteration snapshot excludes private keys
# Validates: Requirements 6.4
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(data=st.dictionaries(st.text(min_size=1), st.text()))
def test_default_snapshot_excludes_private_keys(data):
    """For any state dict with mixed keys, default snapshot excludes underscore-prefixed keys."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )

    # Build a state dict with some underscore-prefixed keys
    state = {}
    for k, v in data.items():
        state[k] = v
        state[f"_{k}"] = f"private_{v}"

    # Also add the required iteration_records key
    state["iteration_records"] = []

    # Call _record_iteration with no custom builder (uses default snapshot)
    lwi = LinearWorkflowInferencer(step_configs=[])
    lwi._record_iteration(state)

    records = state["iteration_records"]
    assert len(records) == 1

    record = records[0]
    # All keys in the record should NOT start with underscore
    for key in record:
        assert not key.startswith("_"), (
            f"Record should not contain underscore-prefixed key: {key}"
        )

    # All non-underscore keys from state should be in the record
    expected_keys = {k for k in state if not k.startswith("_")}
    assert set(record.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Property 7: Child inferencer deduplication by identity
# Feature: linear-workflow-inheritance, Property 7: Child inferencer deduplication by identity
# Validates: Requirements 7.1, 7.2, 7.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(
    n_unique=st.integers(min_value=1, max_value=5),
    n_shared=st.integers(min_value=0, max_value=5),
)
@pytest.mark.asyncio
async def test_child_inferencer_deduplication(n_unique, n_shared):
    """For any set of step configs with shared/unique inferencers, aconnect/adisconnect
    call each unique inferencer exactly once."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    # Create n_unique distinct mock inferencers
    unique_infs = []
    for _ in range(n_unique):
        inf = MagicMock(spec=[])
        inf.aconnect = AsyncMock()
        inf.adisconnect = AsyncMock()
        unique_infs.append(inf)

    # Build step configs: one per unique inferencer, plus n_shared that reuse the first
    configs = []
    for i, inf in enumerate(unique_infs):
        configs.append(
            WorkflowStepConfig(
                name=f"unique_{i}",
                inferencer=inf,
                enable_result_save=False,
            )
        )

    # Add shared references to the first inferencer
    for i in range(n_shared):
        configs.append(
            WorkflowStepConfig(
                name=f"shared_{i}",
                inferencer=unique_infs[0],
                enable_result_save=False,
            )
        )

    lwi = LinearWorkflowInferencer(step_configs=configs)

    await lwi.aconnect()
    await lwi.adisconnect()

    # Each unique inferencer should be called exactly once
    for inf in unique_infs:
        inf.aconnect.assert_awaited_once()
        inf.adisconnect.assert_awaited_once()

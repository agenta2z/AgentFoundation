"""Checkpoint verification tests for LinearWorkflowInferencer.

Verifies:
1. Import paths work from all expected locations
2. Basic 2-step chain with mock inferencers executes correctly
3. Disabled step becomes no-op
4. State coherence across steps
5. Loop-back works
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ── Test 1: Import from all expected paths ──────────────────────────────────

def test_import_from_linear_workflow_inferencer_module():
    """Import directly from the module file."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )
    assert LinearWorkflowInferencer is not None
    assert WorkflowStepConfig is not None


def test_import_from_flow_inferencers_package():
    """Import from the flow_inferencers package __init__."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )
    assert LinearWorkflowInferencer is not None
    assert WorkflowStepConfig is not None


def test_import_from_agentic_inferencers_package():
    """Import from the agentic_inferencers package __init__ (lazy import)."""
    from agent_foundation.common.inferencers.agentic_inferencers import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )
    assert LinearWorkflowInferencer is not None
    assert WorkflowStepConfig is not None


def test_import_identity_across_paths():
    """All import paths resolve to the same class objects."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer as LWI_direct,
        WorkflowStepConfig as WSC_direct,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
        LinearWorkflowInferencer as LWI_flow,
        WorkflowStepConfig as WSC_flow,
    )
    from agent_foundation.common.inferencers.agentic_inferencers import (
        LinearWorkflowInferencer as LWI_agentic,
        WorkflowStepConfig as WSC_agentic,
    )
    assert LWI_direct is LWI_flow
    assert LWI_flow is LWI_agentic
    assert WSC_direct is WSC_flow
    assert WSC_flow is WSC_agentic


# ── Test 2: Basic 2-step chain with mock inferencers ────────────────────────

def _make_mock_inferencer(return_value):
    """Create a mock InferencerBase-like object with ainfer."""
    mock = MagicMock(spec=[])  # spec=[] prevents isinstance checks from matching
    mock.ainfer = AsyncMock(return_value=return_value)
    return mock


def test_two_step_chain_sync(tmp_path):
    """A 2-step chain with mock inferencers executes both steps and builds state."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    planner = _make_mock_inferencer("Here is the plan: do X then Y")
    executor = _make_mock_inferencer("Implementation complete: X and Y done")

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(
                name="plan",
                inferencer=planner,
                output_state_key="plan_output",
            ),
            WorkflowStepConfig(
                name="implement",
                inferencer=executor,
                input_builder=lambda s: s["plan_output"],
                output_state_key="impl_output",
            ),
        ],
        response_builder=lambda state: state["impl_output"],
        workspace_path=str(tmp_path),
    )

    result = lwi.infer("Design and implement a REST API")

    assert result == "Implementation complete: X and Y done"
    planner.ainfer.assert_called_once()
    executor.ainfer.assert_called_once()
    # Verify executor received the plan output as input
    executor_call_args = executor.ainfer.call_args
    assert executor_call_args[0][0] == "Here is the plan: do X then Y"


# ── Test 3: Disabled step becomes no-op ─────────────────────────────────────

def test_disabled_step_is_noop(tmp_path):
    """A disabled step should not execute its inferencer."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    step1 = _make_mock_inferencer("step1 result")
    disabled_inf = _make_mock_inferencer("should not run")
    step3 = _make_mock_inferencer("step3 result")

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(
                name="step1",
                inferencer=step1,
                output_state_key="s1",
            ),
            WorkflowStepConfig(
                name="disabled_step",
                inferencer=disabled_inf,
                output_state_key="s2",
                enabled=False,
            ),
            WorkflowStepConfig(
                name="step3",
                inferencer=step3,
                output_state_key="s3",
            ),
        ],
        response_builder=lambda state: {
            "s1": state.get("s1"),
            "s2": state.get("s2"),
            "s3": state.get("s3"),
        },
        workspace_path=str(tmp_path),
    )

    result = lwi.infer("test input")

    assert result["s1"] == "step1 result"
    assert result["s2"] is None  # disabled step didn't write to state
    assert result["s3"] == "step3 result"
    disabled_inf.ainfer.assert_not_called()


# ── Test 4: State coherence across steps ────────────────────────────────────

def test_state_coherence(tmp_path):
    """State mutations from earlier steps are visible to later steps."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    step1 = _make_mock_inferencer("alpha")
    step2 = _make_mock_inferencer("beta")

    def custom_updater(state, result):
        state["custom_key"] = f"processed_{result}"

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(
                name="step1",
                inferencer=step1,
                output_state_key="s1",
                state_updater=custom_updater,
            ),
            WorkflowStepConfig(
                name="step2",
                inferencer=step2,
                input_builder=lambda s: s.get("custom_key", "missing"),
                output_state_key="s2",
            ),
        ],
        response_builder=lambda state: state,
        workspace_path=str(tmp_path),
    )

    result = lwi.infer("input")

    assert result["s1"] == "alpha"
    assert result["custom_key"] == "processed_alpha"
    assert result["s2"] == "beta"
    # Verify step2 received the custom_key value
    step2.ainfer.assert_called_once()
    assert step2.ainfer.call_args[0][0] == "processed_alpha"


# ── Test 5: Validation — duplicate names rejected ───────────────────────────

def test_duplicate_step_names_rejected():
    """WorkflowStepConfig names must be unique."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    mock = _make_mock_inferencer("x")
    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="dup", inferencer=mock),
            WorkflowStepConfig(name="dup", inferencer=mock),
        ],
    )
    with pytest.raises(ValueError, match="unique"):
        lwi._build_steps()


# ── Test 6: Validation — invalid loop_back_to rejected ─────────────────────

def test_invalid_loop_back_to_rejected():
    """loop_back_to must reference an existing step name."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    mock = _make_mock_inferencer("x")
    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(
                name="step1",
                inferencer=mock,
                loop_back_to="nonexistent",
            ),
        ],
    )
    with pytest.raises(ValueError, match="no step with that name"):
        lwi._build_steps()


# ── Test 7: Validation — enabled step needs inferencer or step_fn ───────────

def test_enabled_step_needs_inferencer_or_step_fn():
    """An enabled step must have either inferencer or step_fn."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="empty", enabled=True),
        ],
    )
    with pytest.raises(ValueError, match="must have either"):
        lwi._build_steps()


# ── Test 8: run()/arun() blocked ───────────────────────────────────────────

def test_run_blocked():
    """run() should raise NotImplementedError."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    mock = _make_mock_inferencer("x")
    lwi = LinearWorkflowInferencer(
        step_configs=[WorkflowStepConfig(name="s", inferencer=mock)],
    )
    with pytest.raises(NotImplementedError):
        lwi.run()


@pytest.mark.asyncio
async def test_arun_blocked():
    """arun() should raise NotImplementedError."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    mock = _make_mock_inferencer("x")
    lwi = LinearWorkflowInferencer(
        step_configs=[WorkflowStepConfig(name="s", inferencer=mock)],
    )
    with pytest.raises(NotImplementedError):
        await lwi.arun()


# ── Test 9: step_fn callable works ──────────────────────────────────────────

def test_step_fn_callable(tmp_path):
    """A step using step_fn (instead of inferencer) should work."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    def my_step_fn(input_val, state):
        return f"processed: {input_val}"

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(
                name="custom",
                step_fn=my_step_fn,
                output_state_key="result",
                enable_result_save=False,
            ),
        ],
        response_builder=lambda state: state["result"],
    )

    result = lwi.infer("hello")
    assert result == "processed: hello"


# ── Test 10: Default response (no response_builder) returns full state ──────

def test_default_response_returns_state(tmp_path):
    """Without response_builder, _ainfer returns the full state dict."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    mock = _make_mock_inferencer("result_val")
    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(
                name="step1",
                inferencer=mock,
                output_state_key="out",
                enable_result_save=False,
            ),
        ],
    )

    result = lwi.infer("test")
    assert isinstance(result, dict)
    assert result["out"] == "result_val"
    assert result["original_input"] == "test"


# ── Test 11: aconnect calls aconnect on unique child inferencers ────────────

@pytest.mark.asyncio
async def test_aconnect_calls_unique_children():
    """aconnect should call aconnect on each unique child inferencer exactly once."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    inf_a = MagicMock(spec=[])
    inf_a.aconnect = AsyncMock()

    inf_b = MagicMock(spec=[])
    inf_b.aconnect = AsyncMock()

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=inf_a, enable_result_save=False),
            WorkflowStepConfig(name="s2", inferencer=inf_b, enable_result_save=False),
            # Shared instance — should NOT be called twice
            WorkflowStepConfig(name="s3", inferencer=inf_a, enable_result_save=False),
        ],
    )

    await lwi.aconnect(some_kwarg="val")

    inf_a.aconnect.assert_awaited_once_with(some_kwarg="val")
    inf_b.aconnect.assert_awaited_once_with(some_kwarg="val")


# ── Test 12: adisconnect calls adisconnect on unique child inferencers ──────

@pytest.mark.asyncio
async def test_adisconnect_calls_unique_children():
    """adisconnect should call adisconnect on each unique child inferencer exactly once."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    inf_a = MagicMock(spec=[])
    inf_a.adisconnect = AsyncMock()

    inf_b = MagicMock(spec=[])
    inf_b.adisconnect = AsyncMock()

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=inf_a, enable_result_save=False),
            WorkflowStepConfig(name="s2", inferencer=inf_b, enable_result_save=False),
            WorkflowStepConfig(name="s3", inferencer=inf_a, enable_result_save=False),
        ],
    )

    await lwi.adisconnect()

    inf_a.adisconnect.assert_awaited_once()
    inf_b.adisconnect.assert_awaited_once()


# ── Test 13: aconnect skips inferencers without aconnect method ─────────────

@pytest.mark.asyncio
async def test_aconnect_skips_inferencers_without_method():
    """aconnect should skip child inferencers that lack an aconnect method."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    inf_with = MagicMock(spec=[])
    inf_with.aconnect = AsyncMock()

    # This inferencer has no aconnect method
    inf_without = MagicMock(spec=[])

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=inf_with, enable_result_save=False),
            WorkflowStepConfig(name="s2", inferencer=inf_without, enable_result_save=False),
        ],
    )

    await lwi.aconnect()

    inf_with.aconnect.assert_awaited_once()


# ── Test 14: adisconnect skips inferencers without adisconnect method ───────

@pytest.mark.asyncio
async def test_adisconnect_skips_inferencers_without_method():
    """adisconnect should skip child inferencers that lack an adisconnect method."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    inf_with = MagicMock(spec=[])
    inf_with.adisconnect = AsyncMock()

    inf_without = MagicMock(spec=[])

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=inf_with, enable_result_save=False),
            WorkflowStepConfig(name="s2", inferencer=inf_without, enable_result_save=False),
        ],
    )

    await lwi.adisconnect()

    inf_with.adisconnect.assert_awaited_once()


# ── Test 15: async context manager calls aconnect/adisconnect ───────────────

@pytest.mark.asyncio
async def test_async_context_manager():
    """async with should call aconnect on enter and adisconnect on exit."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    inf = MagicMock(spec=[])
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=inf, enable_result_save=False),
        ],
    )

    async with lwi as ctx:
        assert ctx is lwi
        inf.aconnect.assert_awaited_once()
        inf.adisconnect.assert_not_awaited()

    inf.adisconnect.assert_awaited_once()


# ── Test 16: aconnect/adisconnect skip None inferencers (step_fn steps) ─────

@pytest.mark.asyncio
async def test_aconnect_skips_none_inferencers():
    """aconnect/adisconnect should skip steps that use step_fn (inferencer=None)."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    inf = MagicMock(spec=[])
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()

    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", step_fn=lambda x, s: x, enable_result_save=False),
            WorkflowStepConfig(name="s2", inferencer=inf, enable_result_save=False),
        ],
    )

    await lwi.aconnect()
    await lwi.adisconnect()

    inf.aconnect.assert_awaited_once()
    inf.adisconnect.assert_awaited_once()

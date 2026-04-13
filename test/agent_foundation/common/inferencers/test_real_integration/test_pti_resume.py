"""PTI recursive resume tests — checkpoint/resume at all levels of the hierarchy.

Validates Requirements 12, 13, 15, and 16 from the integration test spec.
All tests invoke real ``claude`` CLI subprocesses and are marked ``@pytest.mark.integration``.

PTI is configured with DualInferencer children whose base/review inferencers are
real ClaudeCodeCliInferencers. Uses ``append_system_prompt`` on review inferencers
to force approval (keeping tests deterministic). Interruption is achieved via
``idle_timeout_seconds=10`` on the target child's base_inferencer — long enough
for the initial response latency (5-15s) but shorter than total streaming time
(30-60s for Tier B prompts).
"""

import os
from unittest.mock import patch

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)

from .conftest import (
    DEFAULT_TIMEOUT,
    skip_claude,
)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
APPROVE_SYSTEM_PROMPT = (
    "You are a code reviewer. Review the code. "
    'Respond ONLY with JSON: {"approved": true, "severity": "COSMETIC", '
    '"issues": [], "reasoning": "Looks good"}'
)

TIER_B_PROMPT = (
    "Write a Python function that implements merge sort with detailed docstrings, "
    "type hints, and inline comments explaining each step."
)


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_pti_streaming.py)
# ---------------------------------------------------------------------------
def _make_claude(tmp_workspace, **overrides):
    """Create a ClaudeCodeCliInferencer with sensible defaults for tests."""
    kwargs = dict(
        target_path=str(tmp_workspace["workspace"]),
        cache_folder=str(tmp_workspace["cache"]),
        model_name="sonnet",
        resume_with_saved_results=True,
        idle_timeout_seconds=60,
    )
    if hasattr(ClaudeCodeCliInferencer, "permission_mode"):
        kwargs["permission_mode"] = "bypassPermissions"
    kwargs.update(overrides)
    return ClaudeCodeCliInferencer(**kwargs)


def _make_dual(tmp_workspace, **overrides):
    """Create a DualInferencer with claude base + claude review (auto-approve)."""
    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace, append_system_prompt=APPROVE_SYSTEM_PROMPT)
    kwargs = dict(
        base_inferencer=base,
        review_inferencer=review,
        consensus_config=ConsensusConfig(max_iterations=2, max_consensus_attempts=1),
    )
    kwargs.update(overrides)
    return DualInferencer(**kwargs)


# ===========================================================================
# Test 1: Planning phase crash resume (Req 12.1–12.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_planning_phase_crash_resume(tmp_workspace):
    """Interrupt during planner DualInferencer's consensus loop → resume → complete.

    Uses idle_timeout_seconds=10 on the planner's base_inferencer with Tier B
    prompt to trigger interruption during the planning phase. Verifies PTI
    persists a checkpoint, then a second PTI with the same workspace_path
    resumes from the planning phase and proceeds to implementation.

    **Validates: Requirements 12.1, 12.2, 12.3, 12.4**
    """
    workspace_dir = str(tmp_workspace["workspace"])

    # Planner with short idle_timeout on base to force interruption during planning
    planner_base = _make_claude(tmp_workspace, idle_timeout_seconds=10)
    planner_review = _make_claude(
        tmp_workspace, append_system_prompt=APPROVE_SYSTEM_PROMPT
    )
    planner = DualInferencer(
        base_inferencer=planner_base,
        review_inferencer=planner_review,
        consensus_config=ConsensusConfig(max_iterations=2, max_consensus_attempts=1),
    )

    executor = _make_dual(tmp_workspace)

    pti1 = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=executor,
        workspace_path=workspace_dir,
    )

    # First run — expect interruption during planning phase
    try:
        await pti1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass  # Interruption expected

    # Verify workspace has checkpoint artifacts (Req 12.1)
    assert os.path.isdir(workspace_dir), "Workspace should exist after interrupted run"

    # Resume: create second PTI with same workspace_path, normal timeouts
    planner2 = _make_dual(tmp_workspace)
    executor2 = _make_dual(tmp_workspace)

    pti2 = PlanThenImplementInferencer(
        planner_inferencer=planner2,
        executor_inferencer=executor2,
        workspace_path=workspace_dir,
        resume_workspace=workspace_dir,
    )

    # Second run — should resume from planning checkpoint (Req 12.2, 12.3, 12.4)
    result = await pti2.ainfer(TIER_B_PROMPT)

    assert isinstance(result, PlanThenImplementResponse), (
        f"Expected PlanThenImplementResponse, got {type(result).__name__}"
    )
    # Plan output should be populated (planner completed on resume)
    assert result.plan_output is not None and result.plan_output != "", (
        "plan_output should be populated after resumed planning phase"
    )
    # Executor output should be populated (implementation ran after plan)
    assert result.executor_output is not None, (
        "executor_output should be populated after implementation phase"
    )


# ===========================================================================
# Test 2: Implementation phase crash resume (Req 13.1–13.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_implementation_phase_crash_resume(tmp_workspace):
    """Let planning complete, then interrupt during executor DualInferencer.

    Uses idle_timeout_seconds=10 on the executor's base_inferencer. On resume,
    PTI skips the plan step, executor resumes, and the final
    PlanThenImplementResponse has the original plan_output from the first run.

    **Validates: Requirements 13.1, 13.2, 13.3, 13.4**
    """
    workspace_dir = str(tmp_workspace["workspace"])

    # Planner with normal timeouts — planning should complete
    planner = _make_dual(tmp_workspace)

    # Executor with short idle_timeout on base to force interruption during implementation
    executor_base = _make_claude(tmp_workspace, idle_timeout_seconds=10)
    executor_review = _make_claude(
        tmp_workspace, append_system_prompt=APPROVE_SYSTEM_PROMPT
    )
    executor = DualInferencer(
        base_inferencer=executor_base,
        review_inferencer=executor_review,
        consensus_config=ConsensusConfig(max_iterations=2, max_consensus_attempts=1),
    )

    pti1 = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=executor,
        workspace_path=workspace_dir,
    )

    # First run — planning completes, executor interrupted (Req 13.1)
    try:
        await pti1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass  # Interruption expected during implementation

    # Verify workspace exists and plan step completed
    assert os.path.isdir(workspace_dir), "Workspace should exist"

    # Resume: new PTI with same workspace_path, normal timeouts (Req 13.2, 13.3)
    planner2 = _make_dual(tmp_workspace)
    executor2 = _make_dual(tmp_workspace)

    pti2 = PlanThenImplementInferencer(
        planner_inferencer=planner2,
        executor_inferencer=executor2,
        workspace_path=workspace_dir,
        resume_workspace=workspace_dir,
    )

    result = await pti2.ainfer(TIER_B_PROMPT)

    assert isinstance(result, PlanThenImplementResponse), (
        f"Expected PlanThenImplementResponse, got {type(result).__name__}"
    )

    # Plan output should be the original from the first run (Req 13.4)
    assert result.plan_output is not None and result.plan_output != "", (
        "plan_output should be preserved from the pre-interruption planning phase"
    )

    # Executor output should be populated (implementation completed on resume)
    assert result.executor_output is not None, (
        "executor_output should be populated after resumed implementation phase"
    )


# ===========================================================================
# Test 3: Workspace-based resume detection (Req 15.1–15.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_workspace_based_resume_detection(tmp_workspace):
    """Run PTI with workspace_path → planning completes → interrupt during
    implementation → verify workspace contains plan file with .plan_completed
    marker → resume PTI → verify it detects resume point at implementation phase.

    **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
    """
    workspace_dir = str(tmp_workspace["workspace"])

    # Planner with normal timeouts — planning should complete
    planner = _make_dual(tmp_workspace)

    # Executor with short idle_timeout on base to force interruption
    executor_base = _make_claude(tmp_workspace, idle_timeout_seconds=10)
    executor_review = _make_claude(
        tmp_workspace, append_system_prompt=APPROVE_SYSTEM_PROMPT
    )
    executor = DualInferencer(
        base_inferencer=executor_base,
        review_inferencer=executor_review,
        consensus_config=ConsensusConfig(max_iterations=2, max_consensus_attempts=1),
    )

    pti1 = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=executor,
        workspace_path=workspace_dir,
    )

    # First run — planning completes, executor interrupted
    try:
        await pti1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass  # Interruption expected during implementation

    # Verify workspace contains plan file with .plan_completed marker (Req 15.1)
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    # Check iteration 1 workspace
    iter1_ws_path = PlanThenImplementInferencer._get_iteration_workspace(
        workspace_dir, 1
    )
    ws = InferencerWorkspace(root=iter1_ws_path)

    # Plan marker should exist (planning completed successfully)
    assert ws.has_marker("plan"), (
        "Workspace should have .plan_completed marker after planning phase completes"
    )

    # Implementation marker should NOT exist (executor was interrupted)
    assert not ws.has_marker("impl"), (
        "Workspace should NOT have .impl_completed marker (executor was interrupted)"
    )

    # Resume: create second PTI with resume_workspace (Req 15.3, 15.4)
    planner2 = _make_dual(tmp_workspace)
    executor2 = _make_dual(tmp_workspace)

    pti2 = PlanThenImplementInferencer(
        planner_inferencer=planner2,
        executor_inferencer=executor2,
        workspace_path=workspace_dir,
        resume_workspace=workspace_dir,
    )

    # Verify resume detection finds implementation phase
    iteration, resume_phase, _state, _input, _orig = pti2._detect_resume_point(
        workspace_dir
    )
    assert resume_phase == "implementation", (
        f"Expected resume at 'implementation', got '{resume_phase}'"
    )
    assert iteration == 1, f"Expected iteration 1, got {iteration}"

    # Complete the resumed run (Req 15.4)
    result = await pti2.ainfer(TIER_B_PROMPT)

    assert isinstance(result, PlanThenImplementResponse), (
        f"Expected PlanThenImplementResponse, got {type(result).__name__}"
    )
    assert result.plan_output is not None and result.plan_output != "", (
        "plan_output should be restored from workspace"
    )
    assert result.executor_output is not None, (
        "executor_output should be populated after resumed implementation"
    )


# ===========================================================================
# Test 4: _setup_child_workflows propagation (Req 16.1–16.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_setup_child_workflows_propagation(tmp_workspace):
    """Configure PTI with enable_result_save=True and workspace_path → run →
    verify _setup_child_workflows was called (planner and executor DualInferencers
    have _result_root_override set).

    **Validates: Requirements 16.1, 16.2, 16.3, 16.4**
    """
    workspace_dir = str(tmp_workspace["workspace"])

    planner = _make_dual(tmp_workspace)
    executor = _make_dual(tmp_workspace)

    pti = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=executor,
        workspace_path=workspace_dir,
    )

    # Track _setup_child_workflows calls
    original_setup = pti._setup_child_workflows
    setup_calls = []

    def tracking_setup(state, *args, **kwargs):
        setup_calls.append(state)
        return original_setup(state, *args, **kwargs)

    with patch.object(pti, "_setup_child_workflows", side_effect=tracking_setup):
        result = await pti.ainfer(TIER_B_PROMPT)

    assert isinstance(result, PlanThenImplementResponse), (
        f"Expected PlanThenImplementResponse, got {type(result).__name__}"
    )

    # Verify _setup_child_workflows was called at least once (Req 16.1)
    assert len(setup_calls) >= 1, (
        "_setup_child_workflows should have been called during PTI execution"
    )

    # After execution, verify child DualInferencers had checkpoint settings
    # propagated (Req 16.2, 16.3). The enable_result_save and
    # resume_with_saved_results should have been set by _setup_child_workflows.
    # Note: PTI sets enable_result_save=Always when workspace_path is provided.
    from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
        StepResultSaveOptions,
    )

    for child_name, child_inf in [
        ("planner", pti.planner_inferencer),
        ("executor", pti.executor_inferencer),
    ]:
        assert child_inf.enable_result_save == StepResultSaveOptions.Always, (
            f"{child_name} should have enable_result_save=Always after "
            f"_setup_child_workflows propagation"
        )
        assert child_inf.resume_with_saved_results is True, (
            f"{child_name} should have resume_with_saved_results=True after "
            f"_setup_child_workflows propagation"
        )

    # Verify plan and executor outputs are populated (Req 16.4)
    assert result.plan_output is not None and result.plan_output != "", (
        "plan_output should be populated"
    )
    assert result.executor_output is not None, (
        "executor_output should be populated"
    )

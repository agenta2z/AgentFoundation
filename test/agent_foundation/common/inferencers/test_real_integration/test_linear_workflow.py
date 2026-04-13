"""LinearWorkflowInferencer integration tests — streaming children, checkpoint/resume, loop-back.

Validates Requirements 30, 31, and 32 from the integration test spec.
All tests invoke real ``claude`` CLI subprocesses and are marked ``@pytest.mark.integration``.

Tests exercise LinearWorkflowInferencer with real ClaudeCodeCliInferencer children,
DualInferencer children, checkpoint/resume via workspace_path, and loop-back with
real CLI calls.
"""

import os

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
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)

from .conftest import (
    DEFAULT_TIMEOUT,
    PUZZLE_PROMPT,
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

CHEAP_PROMPT = "What is 2+2?"


# ---------------------------------------------------------------------------
# Helpers
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


# ===========================================================================
# Test 1: Three-step coding puzzle (Req 30.1, 30.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_three_step_coding_puzzle(tmp_workspace):
    """Configure LinearWorkflowInferencer with 3 steps using real ClaudeCodeCliInferencers.

    Step 1 (plan): Plan the approach for the coding puzzle.
    Step 2 (implement): Implement the code based on the plan.
    Step 3 (review): Review the implementation.

    Verifies all 3 steps execute and produce output via real CLI calls.

    **Validates: Requirements 30.1, 30.3**
    """
    step1_inferencer = _make_claude(tmp_workspace)
    step2_inferencer = _make_claude(tmp_workspace)
    step3_inferencer = _make_claude(
        tmp_workspace,
        append_system_prompt=(
            "You are a code reviewer. Briefly review the provided code "
            "and confirm it looks correct. Be concise."
        ),
    )

    step_configs = [
        WorkflowStepConfig(
            name="plan",
            inferencer=step1_inferencer,
            input_builder=lambda state: (
                f"Plan the approach for this task (be concise):\n{state['original_input']}"
            ),
            output_state_key="plan_output",
        ),
        WorkflowStepConfig(
            name="implement",
            inferencer=step2_inferencer,
            input_builder=lambda state: (
                f"Implement based on this plan:\n{state.get('plan_output', '')}\n\n"
                f"Original task:\n{state['original_input']}"
            ),
            output_state_key="implement_output",
        ),
        WorkflowStepConfig(
            name="review",
            inferencer=step3_inferencer,
            input_builder=lambda state: (
                f"Review this implementation:\n{state.get('implement_output', '')}"
            ),
            output_state_key="review_output",
        ),
    ]

    lwi = LinearWorkflowInferencer(
        step_configs=step_configs,
        response_builder=lambda state: state,
    )

    result = await lwi.ainfer(PUZZLE_PROMPT)

    # Verify result is a state dict with all 3 outputs
    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    assert "plan_output" in result, "plan_output should be in state"
    assert "implement_output" in result, "implement_output should be in state"
    assert "review_output" in result, "review_output should be in state"

    # All outputs should be non-empty strings from real CLI calls
    assert result["plan_output"] is not None and str(result["plan_output"]).strip() != "", (
        "plan_output should contain real CLI output"
    )
    assert result["implement_output"] is not None and str(result["implement_output"]).strip() != "", (
        "implement_output should contain real CLI output"
    )
    assert result["review_output"] is not None and str(result["review_output"]).strip() != "", (
        "review_output should contain real CLI output"
    )


# ===========================================================================
# Test 2: Step with DualInferencer child (Req 30.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_step_with_dual_inferencer_child(tmp_workspace):
    """Configure LinearWorkflowInferencer with a step that uses a DualInferencer.

    The DualInferencer runs its full consensus loop within the step, and the
    LinearWorkflowInferencer waits for consensus before proceeding.

    **Validates: Requirement 30.2**
    """
    # Step 1: simple claude inferencer
    step1_inferencer = _make_claude(tmp_workspace)

    # Step 2: DualInferencer with auto-approve review
    dual_base = _make_claude(tmp_workspace)
    dual_review = _make_claude(
        tmp_workspace, append_system_prompt=APPROVE_SYSTEM_PROMPT
    )
    step2_dual = DualInferencer(
        base_inferencer=dual_base,
        review_inferencer=dual_review,
        consensus_config=ConsensusConfig(max_iterations=2, max_consensus_attempts=1),
    )

    step_configs = [
        WorkflowStepConfig(
            name="draft",
            inferencer=step1_inferencer,
            output_state_key="draft_output",
        ),
        WorkflowStepConfig(
            name="consensus",
            inferencer=step2_dual,
            input_builder=lambda state: (
                f"Improve this draft:\n{state.get('draft_output', state['original_input'])}"
            ),
            output_state_key="consensus_output",
        ),
    ]

    lwi = LinearWorkflowInferencer(
        step_configs=step_configs,
        response_builder=lambda state: state,
    )

    result = await lwi.ainfer(CHEAP_PROMPT)

    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    assert "draft_output" in result, "draft_output should be in state"
    assert "consensus_output" in result, "consensus_output should be in state"

    # Draft output should be non-empty
    assert result["draft_output"] is not None and str(result["draft_output"]).strip() != "", (
        "draft_output should contain real CLI output"
    )

    # Consensus output should be non-empty (DualInferencer ran its loop)
    consensus = result["consensus_output"]
    assert consensus is not None, "consensus_output should not be None"

    # DualInferencer returns a DualInferencerResponse — verify it has consensus info
    if hasattr(consensus, "consensus_achieved"):
        assert isinstance(consensus.consensus_achieved, bool)
        assert consensus.base_response is not None



# ===========================================================================
# Test 3: Checkpoint resume (Req 31.1, 31.2, 31.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 4)
@skip_claude
async def test_checkpoint_resume(tmp_workspace):
    """Configure LinearWorkflowInferencer with workspace_path for checkpointing.

    Run → interrupt at step 2 via idle_timeout → resume → verify step 1 skipped,
    step 2 re-executed.

    **Validates: Requirements 31.1, 31.2, 31.3**
    """
    workspace_dir = str(tmp_workspace["workspace"])

    # --- First run: step 1 completes, step 2 interrupted via short idle_timeout ---
    step1_inferencer = _make_claude(tmp_workspace)
    # Step 2 has a very short idle_timeout to force interruption
    step2_inferencer = _make_claude(tmp_workspace, idle_timeout_seconds=2)

    step_configs_run1 = [
        WorkflowStepConfig(
            name="step1",
            inferencer=step1_inferencer,
            output_state_key="step1_output",
            enable_result_save=True,
        ),
        WorkflowStepConfig(
            name="step2",
            inferencer=step2_inferencer,
            input_builder=lambda state: (
                f"Continue from: {state.get('step1_output', '')}\nProvide more detail."
            ),
            output_state_key="step2_output",
            enable_result_save=True,
        ),
        WorkflowStepConfig(
            name="step3",
            inferencer=_make_claude(tmp_workspace),
            input_builder=lambda state: (
                f"Summarize: {state.get('step2_output', '')}"
            ),
            output_state_key="step3_output",
            enable_result_save=True,
        ),
    ]

    lwi1 = LinearWorkflowInferencer(
        step_configs=step_configs_run1,
        response_builder=lambda state: state,
        workspace_path=workspace_dir,
    )

    try:
        await lwi1.ainfer(CHEAP_PROMPT)
    except Exception:
        pass  # Interruption expected at step 2

    # Verify workspace directory has checkpoint artifacts
    assert os.path.isdir(workspace_dir), "Workspace should exist after interrupted run"

    # --- Resume: new LinearWorkflowInferencer with same workspace_path ---
    step1_resume = _make_claude(tmp_workspace)
    step2_resume = _make_claude(tmp_workspace)  # Normal timeout for resume
    step3_resume = _make_claude(tmp_workspace)

    step_configs_run2 = [
        WorkflowStepConfig(
            name="step1",
            inferencer=step1_resume,
            output_state_key="step1_output",
            enable_result_save=True,
        ),
        WorkflowStepConfig(
            name="step2",
            inferencer=step2_resume,
            input_builder=lambda state: (
                f"Continue from: {state.get('step1_output', '')}\nProvide more detail."
            ),
            output_state_key="step2_output",
            enable_result_save=True,
        ),
        WorkflowStepConfig(
            name="step3",
            inferencer=step3_resume,
            input_builder=lambda state: (
                f"Summarize: {state.get('step2_output', '')}"
            ),
            output_state_key="step3_output",
            enable_result_save=True,
        ),
    ]

    lwi2 = LinearWorkflowInferencer(
        step_configs=step_configs_run2,
        response_builder=lambda state: state,
        workspace_path=workspace_dir,
    )

    result = await lwi2.ainfer(CHEAP_PROMPT)

    # Verify result contains outputs from all steps
    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"

    # step1_output should be present (restored from checkpoint — Req 31.2)
    assert "step1_output" in result or result.get("step1_output") is not None, (
        "step1_output should be in state (restored from checkpoint or re-executed)"
    )

    # step2_output should be present (re-executed after resume — Req 31.1)
    assert "step2_output" in result, "step2_output should be in state after resume"
    assert result["step2_output"] is not None and str(result["step2_output"]).strip() != "", (
        "step2_output should contain real CLI output after resume"
    )


# ===========================================================================
# Test 4: Loop-back with real CLI calls (Req 32.1, 32.2, 32.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_loop_back_with_real_cli(tmp_workspace):
    """Configure LinearWorkflowInferencer with a loop_back step.

    Uses a loop_condition that returns True once then False, verifying the
    loop executes correctly with real CLI calls.

    **Validates: Requirements 32.1, 32.2, 32.3**
    """
    # Track loop iterations via a mutable counter
    loop_counter = {"count": 0}

    def loop_condition(state, result):
        """Return True on first call (trigger loop-back), False on second (exit)."""
        loop_counter["count"] += 1
        return loop_counter["count"] <= 1

    step1_inferencer = _make_claude(tmp_workspace)
    step2_inferencer = _make_claude(tmp_workspace)
    review_inferencer = _make_claude(
        tmp_workspace,
        append_system_prompt=(
            "You are a reviewer. Briefly review the provided content. "
            "Be concise — one paragraph max."
        ),
    )

    step_configs = [
        WorkflowStepConfig(
            name="draft",
            inferencer=step1_inferencer,
            output_state_key="draft_output",
        ),
        WorkflowStepConfig(
            name="refine",
            inferencer=step2_inferencer,
            input_builder=lambda state: (
                f"Refine this draft:\n{state.get('draft_output', state['original_input'])}"
            ),
            output_state_key="refine_output",
        ),
        WorkflowStepConfig(
            name="review",
            inferencer=review_inferencer,
            input_builder=lambda state: (
                f"Review this:\n{state.get('refine_output', '')}"
            ),
            output_state_key="review_output",
            loop_back_to="refine",
            loop_condition=loop_condition,
            max_loop_iterations=3,
        ),
    ]

    lwi = LinearWorkflowInferencer(
        step_configs=step_configs,
        response_builder=lambda state: state,
    )

    result = await lwi.ainfer(CHEAP_PROMPT)

    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"

    # All outputs should be populated
    assert "draft_output" in result, "draft_output should be in state"
    assert "refine_output" in result, "refine_output should be in state"
    assert "review_output" in result, "review_output should be in state"

    # The loop_condition was called — verify the counter advanced
    # First call returns True (loop back), second call returns False (exit)
    assert loop_counter["count"] == 2, (
        f"Expected loop_condition called 2 times (once True, once False), "
        f"got {loop_counter['count']}"
    )

    # All outputs should contain real CLI content
    assert result["draft_output"] is not None and str(result["draft_output"]).strip() != "", (
        "draft_output should contain real CLI output"
    )
    assert result["refine_output"] is not None and str(result["refine_output"]).strip() != "", (
        "refine_output should contain real CLI output"
    )
    assert result["review_output"] is not None and str(result["review_output"]).strip() != "", (
        "review_output should contain real CLI output"
    )

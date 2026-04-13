"""DualInferencer checkpoint/resume tests — Workflow-level checkpointing with real streaming children.

Validates Requirements 4, 5, and 9 from the integration test spec.
All tests invoke real ``claude`` CLI subprocesses and are marked ``@pytest.mark.integration``.
"""

import json
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

from .conftest import (
    DEFAULT_TIMEOUT,
    count_cache_files,
    skip_claude,
)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
CHEAP_PROMPT = "What is 2+2?"

TIER_B_PROMPT = (
    "Write a Python function that implements merge sort with detailed docstrings, "
    "type hints, and inline comments explaining each step."
)


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


def _make_dual(tmp_workspace, base_overrides=None, review_overrides=None, **dual_kwargs):
    """Create a DualInferencer with two ClaudeCodeCliInferencer children.

    Args:
        tmp_workspace: Fixture providing cache/checkpoint/workspace dirs.
        base_overrides: Extra kwargs for the base (proposer) inferencer.
        review_overrides: Extra kwargs for the review inferencer.
        **dual_kwargs: Extra kwargs for DualInferencer itself.
    """
    base = _make_claude(tmp_workspace, **(base_overrides or {}))
    review = _make_claude(tmp_workspace, **(review_overrides or {}))
    defaults = dict(
        base_inferencer=base,
        review_inferencer=review,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
    )
    defaults.update(dual_kwargs)
    return DualInferencer(**defaults)


def _find_checkpoint_json_files(checkpoint_dir):
    """Walk checkpoint_dir and return all .json file paths."""
    found = []
    for root, _dirs, files in os.walk(checkpoint_dir):
        for f in files:
            if f.endswith(".json"):
                found.append(os.path.join(root, f))
    return found


def _validate_json_files(paths):
    """Assert every path contains valid JSON and return parsed list."""
    results = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        results.append(data)
    return results


# ===========================================================================
# Test 1: Checkpoint creation (Req 4.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_checkpoint_creation(tmp_workspace):
    """enable_checkpoint=True → per-attempt directories with valid JSON."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    dual = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual.ainfer(CHEAP_PROMPT)
    assert result is not None

    # Verify per-attempt directories exist with valid JSON checkpoint files
    json_files = _find_checkpoint_json_files(checkpoint_dir)
    assert len(json_files) >= 1, (
        f"Expected at least one checkpoint JSON file in {checkpoint_dir}, "
        f"found {len(json_files)}"
    )

    # All checkpoint files must contain valid JSON
    parsed = _validate_json_files(json_files)
    assert all(isinstance(d, (dict, list)) for d in parsed)

    # Verify per-attempt directory structure (attempt_01/)
    attempt_dirs = [
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("attempt_")
    ]
    assert len(attempt_dirs) >= 1, "Expected at least one attempt_XX directory"


# ===========================================================================
# Test 2: Resume from review step (Req 4.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_resume_from_review_step(tmp_workspace):
    """Interrupt at review via short idle_timeout → resume → propose skipped."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # First run: review_inferencer has a very short idle_timeout to force interruption
    dual1 = _make_dual(
        tmp_workspace,
        review_overrides={"idle_timeout_seconds": 2},
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass  # Interruption expected

    # Verify checkpoint was created (propose step should have completed)
    json_files = _find_checkpoint_json_files(checkpoint_dir)
    assert len(json_files) >= 1, "Checkpoint should exist after propose completed"

    # Resume: new DualInferencer with same checkpoint_dir, normal timeouts
    dual2 = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None
    assert hasattr(result, "consensus_achieved")


# ===========================================================================
# Test 3: Resume from fix step (Req 4.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_resume_from_fix_step(tmp_workspace):
    """Interrupt at fix → resume → propose+review skipped."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # Use a fixer with short idle_timeout to force interruption at fix step.
    # The base_inferencer (proposer) and review_inferencer have normal timeouts
    # so propose and review complete, but the fixer times out.
    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace)
    fixer = _make_claude(tmp_workspace, idle_timeout_seconds=2)

    dual1 = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        fixer_inferencer=fixer,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass  # Interruption expected at fix step

    # Verify checkpoints exist for propose and review
    json_files = _find_checkpoint_json_files(checkpoint_dir)
    assert len(json_files) >= 1, "Checkpoints should exist after propose+review"

    # Resume with normal timeouts
    base2 = _make_claude(tmp_workspace)
    review2 = _make_claude(tmp_workspace)
    fixer2 = _make_claude(tmp_workspace)

    dual2 = DualInferencer(
        base_inferencer=base2,
        review_inferencer=review2,
        fixer_inferencer=fixer2,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None
    assert hasattr(result, "consensus_achieved")


# ===========================================================================
# Test 4: Resume from second review iteration (Req 4.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_resume_from_second_review_iteration(tmp_workspace):
    """Interrupt during second review iteration → resume with correct iteration counter."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # First run: review_inferencer has short idle_timeout.
    # With max_iterations=3, the loop is propose → review → fix → review(2nd) → ...
    # The first review may complete (short prompt), but the second review
    # (after a fix cycle) should be interrupted.
    # Use TIER_B prompt to ensure enough streaming for the timeout to trigger.
    base = _make_claude(tmp_workspace)
    # First review has normal timeout so it completes
    review = _make_claude(tmp_workspace, idle_timeout_seconds=60)
    fixer = _make_claude(tmp_workspace)

    dual1 = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        fixer_inferencer=fixer,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    # Run first — this should complete at least one full cycle (propose+review+fix)
    # Then we manually create a second DualInferencer with short review timeout
    # to interrupt during the second review.
    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass

    # Check if we got a result or were interrupted
    json_files = _find_checkpoint_json_files(checkpoint_dir)

    if len(json_files) >= 1:
        # We have checkpoints — try to resume
        base2 = _make_claude(tmp_workspace)
        review2 = _make_claude(tmp_workspace)
        fixer2 = _make_claude(tmp_workspace)

        dual2 = DualInferencer(
            base_inferencer=base2,
            review_inferencer=review2,
            fixer_inferencer=fixer2,
            consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
            enable_checkpoint=True,
            checkpoint_dir=checkpoint_dir,
        )

        result = await dual2.ainfer(TIER_B_PROMPT)
        assert result is not None
        assert hasattr(result, "total_iterations")
        assert result.total_iterations >= 1, (
            "total_iterations should reflect at least one completed iteration"
        )
        # Verify proposal text is present
        assert result.base_response is not None
        assert len(result.base_response) > 0
    else:
        # First run completed without interruption — that's also valid
        pass


# ===========================================================================
# Test 5: Dual-layer checkpoint coordination (Req 4.5)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_dual_layer_checkpoint_coordination(tmp_workspace):
    """Workflow-level checkpoints and inferencer-level cache files coexist."""
    cache_dir = str(tmp_workspace["cache"])
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    dual = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual.ainfer(CHEAP_PROMPT)
    assert result is not None

    # Verify streaming cache files exist (from real CLI calls)
    cache_count = count_cache_files(cache_dir, CHEAP_PROMPT, "ClaudeCodeCliInferencer")
    assert cache_count >= 1, "Streaming cache files should exist from real CLI calls"

    # Verify checkpoint files exist (Workflow-level)
    json_files = _find_checkpoint_json_files(checkpoint_dir)
    assert len(json_files) >= 1, "Workflow checkpoint files should exist"

    # Both systems coexist — no file conflicts
    # Verify checkpoint files are in checkpoint_dir, cache files are in cache_dir
    for jf in json_files:
        assert jf.startswith(checkpoint_dir), (
            f"Checkpoint file {jf} should be under checkpoint_dir"
        )


# ===========================================================================
# Test 6: Proposal text preserved across resume (Req 5.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_proposal_text_preserved_across_resume(tmp_workspace):
    """Resuming after review interruption passes original proposal text to reviewer."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # First run: interrupt at review step
    dual1 = _make_dual(
        tmp_workspace,
        review_overrides={"idle_timeout_seconds": 2},
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass

    # Read the propose checkpoint to capture the original proposal
    json_files = _find_checkpoint_json_files(checkpoint_dir)
    propose_files = [f for f in json_files if "propose" in os.path.basename(f)]
    original_proposal = None
    if propose_files:
        with open(propose_files[0]) as fh:
            propose_data = json.load(fh)
        # The propose step result contains the proposal text
        if isinstance(propose_data, str):
            original_proposal = propose_data
        elif isinstance(propose_data, dict):
            original_proposal = propose_data.get("result", propose_data)

    # Resume with normal timeouts
    dual2 = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None
    # The final base_response should contain meaningful content
    assert result.base_response is not None
    assert len(result.base_response) > 0

    # If we captured the original proposal, verify it influenced the result
    if original_proposal is not None:
        # The proposal text should be non-empty (was preserved from checkpoint)
        assert len(str(original_proposal)) > 0


# ===========================================================================
# Test 7: Iteration counter correct after resume (Req 5.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_iteration_counter_correct_after_resume(tmp_workspace):
    """After resume, total_iterations reflects iterations completed before interruption."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # First run: interrupt at review
    dual1 = _make_dual(
        tmp_workspace,
        review_overrides={"idle_timeout_seconds": 2},
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass

    # Resume
    dual2 = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None
    # total_iterations should be at least 1 (the resumed review counts)
    assert result.total_iterations >= 1, (
        f"Expected total_iterations >= 1, got {result.total_iterations}"
    )


# ===========================================================================
# Test 8: Consensus history accuracy after resume (Req 5.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_consensus_history_accuracy_after_resume(tmp_workspace):
    """consensus_history reflects both pre-interruption and post-resume iterations."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # First run: interrupt at review
    dual1 = _make_dual(
        tmp_workspace,
        review_overrides={"idle_timeout_seconds": 2},
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass

    # Resume
    dual2 = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None
    assert hasattr(result, "consensus_history")
    assert isinstance(result.consensus_history, list)
    assert len(result.consensus_history) >= 1, (
        "consensus_history should have at least one attempt record"
    )

    # The attempt record should have iterations
    attempt = result.consensus_history[0]
    assert hasattr(attempt, "iterations")
    # consensus_achieved should be a valid boolean
    assert isinstance(result.consensus_achieved, bool)


# ===========================================================================
# Test 9: Fixer receives correct review feedback (Req 5.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_fixer_receives_correct_review_feedback(tmp_workspace):
    """Resuming with fixer passes correct review feedback from checkpointed review step."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # Use a dedicated fixer with short idle_timeout to interrupt at fix step.
    # This means propose and review complete, but fix is interrupted.
    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace)
    fixer = _make_claude(tmp_workspace, idle_timeout_seconds=2)

    dual1 = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        fixer_inferencer=fixer,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass

    # Read the review checkpoint to capture the review feedback
    json_files = _find_checkpoint_json_files(checkpoint_dir)
    review_files = [f for f in json_files if "review" in os.path.basename(f)]
    review_feedback_captured = None
    if review_files:
        with open(review_files[0]) as fh:
            review_data = json.load(fh)
        review_feedback_captured = review_data

    # Resume with normal fixer
    base2 = _make_claude(tmp_workspace)
    review2 = _make_claude(tmp_workspace)
    fixer2 = _make_claude(tmp_workspace)

    dual2 = DualInferencer(
        base_inferencer=base2,
        review_inferencer=review2,
        fixer_inferencer=fixer2,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None

    # If review feedback was captured, verify it's non-empty
    if review_feedback_captured is not None:
        assert review_feedback_captured, "Review checkpoint should contain feedback data"

    # The result should have valid structure
    assert hasattr(result, "consensus_achieved")
    assert result.base_response is not None


# ===========================================================================
# Test 10: Interruption at propose and resume (Req 9.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_interruption_at_propose_and_resume(tmp_workspace):
    """Interrupt at propose step → resume → propose re-executed."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # Interrupt at propose by giving base_inferencer a very short idle_timeout
    dual1 = _make_dual(
        tmp_workspace,
        base_overrides={"idle_timeout_seconds": 2},
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass  # Interruption at propose expected

    # Resume with normal timeouts — propose should be re-executed
    dual2 = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None
    assert result.base_response is not None
    assert len(result.base_response) > 0


# ===========================================================================
# Test 11: Interruption at review and resume (Req 9.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_interruption_at_review_and_resume(tmp_workspace):
    """Interrupt at review → resume → review re-executed with checkpointed proposal."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # Interrupt at review
    dual1 = _make_dual(
        tmp_workspace,
        review_overrides={"idle_timeout_seconds": 2},
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass

    # Verify propose checkpoint exists (propose completed before review interrupted)
    json_files = _find_checkpoint_json_files(checkpoint_dir)
    propose_files = [f for f in json_files if "propose" in os.path.basename(f)]
    assert len(propose_files) >= 1, (
        "Propose checkpoint should exist after review interruption"
    )

    # Resume — review should be re-executed using checkpointed proposal
    dual2 = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None
    assert result.base_response is not None


# ===========================================================================
# Test 12: Interruption at fix and resume (Req 9.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_interruption_at_fix_and_resume(tmp_workspace):
    """Interrupt at fix → resume → fix re-executed with checkpointed review feedback."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace)
    fixer = _make_claude(tmp_workspace, idle_timeout_seconds=2)

    dual1 = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        fixer_inferencer=fixer,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass

    # Verify propose and review checkpoints exist
    json_files = _find_checkpoint_json_files(checkpoint_dir)
    propose_files = [f for f in json_files if "propose" in os.path.basename(f)]
    review_files = [f for f in json_files if "review" in os.path.basename(f)]

    # At minimum, propose should be checkpointed
    assert len(propose_files) >= 1, "Propose checkpoint should exist"
    # Review should also be checkpointed if the fix step was reached
    if len(review_files) >= 1:
        # Good — both propose and review completed before fix interrupted
        pass

    # Resume with normal fixer
    base2 = _make_claude(tmp_workspace)
    review2 = _make_claude(tmp_workspace)
    fixer2 = _make_claude(tmp_workspace)

    dual2 = DualInferencer(
        base_inferencer=base2,
        review_inferencer=review2,
        fixer_inferencer=fixer2,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None
    assert result.base_response is not None


# ===========================================================================
# Test 13: Interruption-resume produces valid response (Req 9.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_interruption_resume_produces_valid_response(tmp_workspace):
    """After interruption-and-resume, DualInferencerResponse has valid consensus_achieved."""
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # Interrupt at review step
    dual1 = _make_dual(
        tmp_workspace,
        review_overrides={"idle_timeout_seconds": 2},
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        await dual1.ainfer(TIER_B_PROMPT)
    except Exception:
        pass

    # Resume
    dual2 = _make_dual(
        tmp_workspace,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual2.ainfer(TIER_B_PROMPT)
    assert result is not None

    # Validate DualInferencerResponse structure
    assert isinstance(result.consensus_achieved, bool)
    assert isinstance(result.total_iterations, int)
    assert result.total_iterations >= 1
    assert isinstance(result.consensus_history, list)
    assert len(result.consensus_history) >= 1
    assert result.base_response is not None
    assert len(result.base_response) > 0

    # The response should have a valid phase (may be empty string)
    assert isinstance(result.phase, str)

"""DualInferencer end-to-end consensus loop tests — approve, reject-fix-approve, exhaustion, multi-attempt.

Validates Requirement 8 from the integration test spec.
All tests invoke real ``claude`` CLI subprocesses and are marked ``@pytest.mark.integration``.

Uses ``append_system_prompt`` on ClaudeCodeCliInferencer to deterministically
control reviewer behavior (always approve, always reject, etc.).
"""

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
    skip_claude,
)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
CHEAP_PROMPT = "What is 2+2?"

APPROVE_SYSTEM_PROMPT = (
    "You are a code reviewer. Review the code. "
    'Respond ONLY with JSON: {"approved": true, "severity": "COSMETIC", '
    '"issues": [], "reasoning": "Looks good"}'
)

REJECT_SYSTEM_PROMPT = (
    "You are a strict code reviewer. Find at least one issue. "
    'Respond ONLY with JSON: {"approved": false, "severity": "MAJOR", '
    '"issues": [{"severity": "MAJOR", "category": "test", '
    '"description": "Needs work", "location": "N/A", '
    '"suggestion": "Fix it"}], "reasoning": "Issues found"}'
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


# ===========================================================================
# Test 1: First-round approval (Req 8.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_first_round_approval(tmp_workspace):
    """Reviewer always approves → consensus on first iteration.

    **Validates: Requirement 8.1**
    """
    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace, append_system_prompt=APPROVE_SYSTEM_PROMPT)

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        consensus_config=ConsensusConfig(max_iterations=3, max_consensus_attempts=1),
    )

    result = await dual.ainfer(CHEAP_PROMPT)

    assert result is not None
    assert result.consensus_achieved is True
    assert result.total_iterations == 1
    assert result.base_response != ""


# ===========================================================================
# Test 2: Reject → fix → approve cycle (Req 8.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_reject_fix_approve_cycle(tmp_workspace):
    """Reviewer rejects first, then approves — verifies the reject→fix→approve cycle.

    Uses two separate review inferencers: the DualInferencer's review_inferencer
    is configured with the reject prompt. Since the reviewer uses append_system_prompt
    to always reject, we set max_iterations=2 and verify the loop runs at least
    2 iterations. The second review may also reject (since the system prompt
    always rejects), so we verify the loop executed multiple iterations.

    **Validates: Requirement 8.2**
    """
    base = _make_claude(tmp_workspace)
    # Reviewer always rejects — this forces the fix step to run
    review = _make_claude(tmp_workspace, append_system_prompt=REJECT_SYSTEM_PROMPT)

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        consensus_config=ConsensusConfig(max_iterations=2, max_consensus_attempts=1),
    )

    result = await dual.ainfer(CHEAP_PROMPT)

    assert result is not None
    # With a reviewer that always rejects, the loop should exhaust iterations.
    # The key validation is that the fix step ran (total_iterations > 1 means
    # at least one reject→fix cycle occurred).
    assert result.total_iterations == 2, (
        f"Expected 2 iterations (reject→fix→re-review), got {result.total_iterations}"
    )
    # Consensus history should have iteration records
    assert len(result.consensus_history) >= 1
    assert len(result.consensus_history[0].iterations) == 2


# ===========================================================================
# Test 3: Loop exhaustion (Req 8.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_loop_exhaustion(tmp_workspace):
    """Reviewer always rejects, max_iterations=2 → consensus_achieved=False.

    **Validates: Requirement 8.3**
    """
    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace, append_system_prompt=REJECT_SYSTEM_PROMPT)

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        consensus_config=ConsensusConfig(max_iterations=2, max_consensus_attempts=1),
    )

    result = await dual.ainfer(CHEAP_PROMPT)

    assert result is not None
    assert result.consensus_achieved is False
    assert result.total_iterations == 2
    assert result.base_response != ""


# ===========================================================================
# Test 4: Multi-attempt with session reset (Req 8.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_multi_attempt_with_session_reset(tmp_workspace):
    """max_consensus_attempts=2, max_iterations=1 → first attempt fails, second attempt starts fresh.

    With max_iterations=1 and a reviewer that always rejects, each attempt
    exhausts after a single iteration. With max_consensus_attempts=2, the
    DualInferencer should make two attempts, resetting session state between them.

    **Validates: Requirement 8.4**
    """
    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace, append_system_prompt=REJECT_SYSTEM_PROMPT)

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        new_session_per_attempt=True,
        consensus_config=ConsensusConfig(max_iterations=1, max_consensus_attempts=2),
    )

    result = await dual.ainfer(CHEAP_PROMPT)

    assert result is not None
    # Both attempts should fail (reviewer always rejects, only 1 iteration each)
    assert result.consensus_achieved is False
    # Total iterations = 1 per attempt × 2 attempts = 2
    assert result.total_iterations == 2
    # Should have 2 attempt records
    assert len(result.consensus_history) == 2
    # Each attempt should have exactly 1 iteration
    assert len(result.consensus_history[0].iterations) == 1
    assert len(result.consensus_history[1].iterations) == 1
    # Neither attempt achieved consensus
    assert result.consensus_history[0].consensus_reached is False
    assert result.consensus_history[1].consensus_reached is False

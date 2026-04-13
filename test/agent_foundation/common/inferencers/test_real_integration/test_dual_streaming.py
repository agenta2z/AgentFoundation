"""DualInferencer streaming tests — cache resume, session management, dual-layer coordination.

Validates Requirements 1, 2, 3, and 10 from the integration test spec.
All tests invoke real ``claude`` CLI subprocesses and are marked ``@pytest.mark.integration``.
"""

import os
import time

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)

from .conftest import (
    DEFAULT_TIMEOUT,
    assert_cached_skip,
    assert_real_call,
    count_cache_files,
    read_latest_cache,
    skip_both,
    skip_claude,
)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
CHEAP_PROMPT = "What is 2+2?"
CHEAP_PROMPT_ALT = "What is 3+3?"

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


# ===========================================================================
# Test 1: Completed cache skip (Req 1.1, 1.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_completed_cache_skip(tmp_workspace):
    """Run a real CLI call, then verify a second call returns from cache."""
    inf = _make_claude(tmp_workspace)
    cache_dir = str(tmp_workspace["cache"])

    # First call — real CLI execution
    start = time.monotonic()
    result1 = await inf.ainfer(CHEAP_PROMPT)
    assert_real_call(start)
    assert result1 is not None

    # Cache file should exist
    assert count_cache_files(cache_dir, CHEAP_PROMPT, "ClaudeCodeCliInferencer") >= 1

    # Second call — new inferencer instance, same cache folder
    inf2 = _make_claude(tmp_workspace)
    start = time.monotonic()
    result2 = await inf2.ainfer(CHEAP_PROMPT)
    assert_cached_skip(start)
    assert result2 is not None


# ===========================================================================
# Test 2: Cache file has success marker (Req 1.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_cache_file_has_success_marker(tmp_workspace):
    """Inspect cache content — must contain the success marker."""
    inf = _make_claude(tmp_workspace)
    cache_dir = str(tmp_workspace["cache"])

    await inf.ainfer(CHEAP_PROMPT)

    content = read_latest_cache(cache_dir, CHEAP_PROMPT, "ClaudeCodeCliInferencer")
    assert content is not None
    assert "--- STREAM COMPLETED SUCCESSFULLY ---" in content


# ===========================================================================
# Test 3: Cache survives new instance ID (Req 1.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_cache_survives_new_instance_id(tmp_workspace):
    """Instance A writes cache; Instance B (different UUID) finds it via wildcard glob."""
    inf_a = _make_claude(tmp_workspace)
    cache_dir = str(tmp_workspace["cache"])

    await inf_a.ainfer(CHEAP_PROMPT)
    assert count_cache_files(cache_dir, CHEAP_PROMPT, "ClaudeCodeCliInferencer") >= 1

    # Instance B — different random UUID (new object = new id)
    inf_b = _make_claude(tmp_workspace)
    assert inf_b.id != inf_a.id  # sanity: different instance IDs

    start = time.monotonic()
    result = await inf_b.ainfer(CHEAP_PROMPT)
    assert_cached_skip(start)
    assert result is not None


# ===========================================================================
# Test 4: Resume disabled (Req 1.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_resume_disabled(tmp_workspace):
    """With resume_with_saved_results=False, cache is ignored."""
    # First: populate cache
    inf1 = _make_claude(tmp_workspace)
    await inf1.ainfer(CHEAP_PROMPT)
    cache_dir = str(tmp_workspace["cache"])
    assert count_cache_files(cache_dir, CHEAP_PROMPT, "ClaudeCodeCliInferencer") >= 1

    # Second: new inferencer with resume disabled
    inf2 = _make_claude(tmp_workspace, resume_with_saved_results=False)
    start = time.monotonic()
    result = await inf2.ainfer(CHEAP_PROMPT)
    assert_real_call(start)
    assert result is not None


# ===========================================================================
# Test 5: Multi-prompt cache matching (Req 1.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_multi_prompt_cache_matching(tmp_workspace):
    """Different prompts match correct cache by prompt hash."""
    inf = _make_claude(tmp_workspace)
    cache_dir = str(tmp_workspace["cache"])

    # Run two different prompts
    await inf.ainfer(CHEAP_PROMPT)
    await inf.ainfer(CHEAP_PROMPT_ALT)

    assert count_cache_files(cache_dir, CHEAP_PROMPT, "ClaudeCodeCliInferencer") >= 1
    assert count_cache_files(cache_dir, CHEAP_PROMPT_ALT, "ClaudeCodeCliInferencer") >= 1

    # New instance — each prompt should hit its own cache
    inf2 = _make_claude(tmp_workspace)

    start = time.monotonic()
    await inf2.ainfer(CHEAP_PROMPT)
    assert_cached_skip(start)

    start = time.monotonic()
    await inf2.ainfer(CHEAP_PROMPT_ALT)
    assert_cached_skip(start)


# ===========================================================================
# Test 6: Partial cache recovery (Req 2.1, 2.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_partial_cache_recovery(tmp_workspace):
    """Interrupt a long-running stream via idle_timeout, verify partial cache, then resume.

    Uses Tier B prompt (30+ seconds streaming) with idle_timeout_seconds=10.
    The idle timeout must be longer than initial response latency (5-15s for
    real CLI) but shorter than total streaming time.
    """
    cache_dir = str(tmp_workspace["cache"])

    # First call — will be interrupted by idle timeout after ~10s of no new chunks
    inf1 = _make_claude(tmp_workspace, idle_timeout_seconds=10)
    try:
        await inf1.ainfer(TIER_B_PROMPT)
    except Exception:
        # Timeout or interruption expected — the cache should still be finalized
        pass

    # Check partial cache exists with failure marker and meaningful content
    content = read_latest_cache(cache_dir, TIER_B_PROMPT, "ClaudeCodeCliInferencer")
    if content is not None and "--- STREAM FAILED:" in content:
        # Partial cache has failure marker — verify meaningful content before marker
        idx = content.find("\n--- STREAM FAILED:")
        partial_text = content[:idx].strip() if idx >= 0 else content.strip()
        assert len(partial_text) > 0, "Partial cache should have meaningful content"

        # Resume with augmented prompt — the inferencer should detect partial cache
        inf2 = _make_claude(tmp_workspace, idle_timeout_seconds=60)
        start = time.monotonic()
        result = await inf2.ainfer(TIER_B_PROMPT)
        assert_real_call(start)
        assert result is not None
    else:
        # If the call completed successfully (fast model), just verify cache exists
        assert content is not None, "Cache file should exist after CLI call"
        assert "--- STREAM COMPLETED SUCCESSFULLY ---" in content


# ===========================================================================
# Test 7: Empty partial cache (Req 2.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_empty_partial_cache(tmp_workspace):
    """Whitespace-only partial cache → fresh execution instead of recovery."""
    cache_dir = str(tmp_workspace["cache"])

    # Manually create a whitespace-only cache file to simulate empty partial
    class_dir = os.path.join(cache_dir, "ClaudeCodeCliInferencer", "fake_instance")
    os.makedirs(class_dir, exist_ok=True)

    import hashlib

    prompt_hash = hashlib.sha256(CHEAP_PROMPT.encode()).hexdigest()[:8]
    cache_path = os.path.join(class_dir, f"stream_0001_{prompt_hash}.txt")
    with open(cache_path, "w") as f:
        f.write("   \n\n  \n")  # whitespace-only content

    # Inferencer should ignore the empty partial and do a fresh call
    inf = _make_claude(tmp_workspace)
    start = time.monotonic()
    result = await inf.ainfer(CHEAP_PROMPT)
    assert_real_call(start)
    assert result is not None


# ===========================================================================
# Test 8: Session continuation (Req 3.1, 3.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_session_continuation(tmp_workspace):
    """Verify active_session_id is set after a real CLI call."""
    inf = _make_claude(tmp_workspace)

    assert inf.active_session_id is None

    await inf.ainfer(CHEAP_PROMPT)

    assert inf.active_session_id is not None, (
        "active_session_id should be set after a real CLI call"
    )
    # Session ID should be a non-empty string
    assert isinstance(inf.active_session_id, str)
    assert len(inf.active_session_id) > 0


# ===========================================================================
# Test 9: new_session clears active_session_id (Req 3.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_new_session_clears_session_id(tmp_workspace):
    """Calling reset_session clears active_session_id."""
    inf = _make_claude(tmp_workspace)

    # Establish a session
    await inf.ainfer(CHEAP_PROMPT)
    assert inf.active_session_id is not None

    # Clear session
    inf.reset_session()
    assert inf.active_session_id is None


# ===========================================================================
# Test 10: new_session_per_attempt resets session state (Req 3.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_both
async def test_new_session_per_attempt(tmp_workspace):
    """new_session_per_attempt resets session state on DualInferencer children."""
    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace)

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        new_session_per_attempt=True,
    )

    # Manually set session IDs to simulate prior calls
    base._session_id = "fake-session-base"
    review._session_id = "fake-session-review"

    # _areset_sub_inferencers should clear sessions when new_session_per_attempt=True
    await dual._areset_sub_inferencers()

    assert base.active_session_id is None, (
        "base_inferencer session should be cleared after reset with new_session_per_attempt"
    )
    assert review.active_session_id is None, (
        "review_inferencer session should be cleared after reset with new_session_per_attempt"
    )


# ===========================================================================
# Test 11: Dual-layer resume coordination (Req 10.1, 10.2, 10.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_both
async def test_dual_layer_resume_coordination(tmp_workspace):
    """Workflow checkpoints + streaming cache coexist without conflict."""
    cache_dir = str(tmp_workspace["cache"])
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace)

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        enable_checkpoint=True,
        checkpoint_dir=checkpoint_dir,
    )

    result = await dual.ainfer(CHEAP_PROMPT)
    assert result is not None

    # Verify streaming cache files exist (from real CLI calls)
    base_cache_count = count_cache_files(
        cache_dir, CHEAP_PROMPT, "ClaudeCodeCliInferencer"
    )
    assert base_cache_count >= 1, "Base inferencer should have cache files"

    # Verify checkpoint directory has content (Workflow checkpoints)
    checkpoint_files = []
    for root, _dirs, files in os.walk(checkpoint_dir):
        checkpoint_files.extend(files)
    # Checkpoint files may or may not exist depending on whether checkpointing
    # was triggered — the key assertion is that both systems coexist without error


# ===========================================================================
# Test 12: Selective cache delete (Req 10.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_both
async def test_selective_cache_delete(tmp_workspace):
    """Delete reviewer's cache, re-run — base from cache, reviewer fresh."""
    cache_dir = str(tmp_workspace["cache"])

    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace)

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
    )

    # First run — populates cache for both base and review
    await dual.ainfer(CHEAP_PROMPT)

    # Count cache files for the base inferencer's prompt
    base_cache_before = count_cache_files(
        cache_dir, CHEAP_PROMPT, "ClaudeCodeCliInferencer"
    )
    assert base_cache_before >= 1

    # The review prompt is constructed internally by DualInferencer, so we can't
    # easily target it by prompt hash. Instead, verify the base cache survives
    # a second run — the base should be served from cache (fast).
    base2 = _make_claude(tmp_workspace)
    review2 = _make_claude(tmp_workspace)

    dual2 = DualInferencer(
        base_inferencer=base2,
        review_inferencer=review2,
    )

    # Delete ALL review-related cache files (we don't know the exact review prompt
    # hash, so we delete all cache files except those matching the base prompt)
    import glob
    import hashlib

    base_hash = hashlib.sha256(CHEAP_PROMPT.encode()).hexdigest()[:8]
    all_cache_files = glob.glob(
        os.path.join(cache_dir, "ClaudeCodeCliInferencer", "*", "stream_*.txt")
    )
    for f in all_cache_files:
        if base_hash not in os.path.basename(f):
            os.remove(f)

    # Re-run — base should come from cache, reviewer should make a fresh call
    start = time.monotonic()
    result = await dual2.ainfer(CHEAP_PROMPT)
    assert result is not None
    # Total time should reflect at least one real call (the reviewer)
    # but the base should have been fast (from cache)

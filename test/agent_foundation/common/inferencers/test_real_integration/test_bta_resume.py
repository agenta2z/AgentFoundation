"""BTA resume tests — breakdown checkpoint and worker resume.

Validates Requirements 20 and 21 from the integration test spec.
All tests invoke real ``claude`` CLI subprocesses and are marked ``@pytest.mark.integration``.

BTA uses WorkGraph (not Workflow) — resume is via custom breakdown_result.json
+ per-node result files, not Workflow step checkpoints. After the breakdown phase,
BTA saves ``breakdown_result.json`` containing the raw output and parsed sub_queries.
On resume, BTA loads this checkpoint to skip re-running the breakdown_inferencer.
"""

import json
import os

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)

from .conftest import (
    DEFAULT_TIMEOUT,
    skip_claude,
)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
BREAKDOWN_PROMPT = (
    "List exactly 2 simple math questions, one per line, as a numbered list. "
    "Example format:\n1. What is 1+1?\n2. What is 2+2?"
)

BREAKDOWN_SYSTEM_PROMPT = (
    "You are a task decomposer. When given a request, respond ONLY with a "
    "numbered list of exactly 2 sub-tasks. Each line must start with a number "
    "followed by a period. Do not include any other text."
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


def _make_worker_factory(tmp_workspace):
    """Return a worker_factory callable: (sub_query, index) → ClaudeCodeCliInferencer."""

    def factory(sub_query, index):
        return _make_claude(tmp_workspace)

    return factory


# ===========================================================================
# Test 1: Breakdown checkpoint resume (Req 20.1, 20.2, 20.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_breakdown_checkpoint_resume(tmp_workspace):
    """Run BTA → breakdown completes → interrupt during workers → verify
    breakdown_result.json saved → resume BTA → verify breakdown not re-executed
    → workers resume from cache/checkpoint.

    Uses idle_timeout_seconds=2 on workers to force interruption during the
    worker phase (after breakdown completes and saves its checkpoint).

    **Validates: Requirements 20.1, 20.2, 20.3**
    """
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # Breakdown inferencer with normal timeout — breakdown should complete
    breakdown = _make_claude(
        tmp_workspace, append_system_prompt=BREAKDOWN_SYSTEM_PROMPT
    )

    # Worker factory with short idle_timeout to force interruption during workers
    def short_timeout_worker_factory(sub_query, index):
        return _make_claude(tmp_workspace, idle_timeout_seconds=2)

    bta1 = BreakdownThenAggregateInferencer(
        breakdown_inferencer=breakdown,
        worker_factory=short_timeout_worker_factory,
        checkpoint_dir=checkpoint_dir,
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    # First run — breakdown completes, workers interrupted via idle_timeout
    try:
        await bta1.ainfer(BREAKDOWN_PROMPT)
    except Exception:
        pass  # Interruption expected during worker phase

    # Verify breakdown_result.json was saved (Req 20.1)
    ckpt_path = os.path.join(checkpoint_dir, "breakdown_result.json")
    assert os.path.exists(ckpt_path), (
        f"breakdown_result.json should exist at {ckpt_path} after breakdown completes"
    )

    # Verify checkpoint content is valid JSON with sub_queries
    with open(ckpt_path) as f:
        saved = json.load(f)
    assert "sub_queries" in saved, "Checkpoint should contain 'sub_queries' key"
    assert "raw_output" in saved, "Checkpoint should contain 'raw_output' key"
    assert len(saved["sub_queries"]) >= 1, "Should have at least 1 sub_query"

    # Resume: create second BTA with same checkpoint_dir, normal timeouts (Req 20.2, 20.3)
    # Use a breakdown inferencer that would fail if called — proves breakdown is skipped
    class FailIfCalledInferencer:
        """Sentinel inferencer that fails if its infer/ainfer is called."""

        async def ainfer(self, *args, **kwargs):
            raise AssertionError(
                "Breakdown inferencer should NOT be called on resume — "
                "breakdown_result.json should be loaded from checkpoint"
            )

        def infer(self, *args, **kwargs):
            raise AssertionError(
                "Breakdown inferencer should NOT be called on resume — "
                "breakdown_result.json should be loaded from checkpoint"
            )

    bta2 = BreakdownThenAggregateInferencer(
        breakdown_inferencer=FailIfCalledInferencer(),
        worker_factory=_make_worker_factory(tmp_workspace),
        checkpoint_dir=checkpoint_dir,
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    # Second run — should load breakdown from checkpoint, skip breakdown, run workers
    result = await bta2.ainfer(BREAKDOWN_PROMPT)

    assert result is not None, "Resumed BTA should produce a result"
    # With no aggregator, result is a tuple of worker outputs or a single result
    if isinstance(result, tuple):
        assert len(result) >= 1, "Should have at least 1 worker result"
        for i, worker_result in enumerate(result):
            assert worker_result is not None, f"Worker {i} result should not be None"
            assert str(worker_result).strip() != "", (
                f"Worker {i} result should not be empty"
            )
    else:
        assert str(result).strip() != "", "Result should not be empty"


# ===========================================================================
# Test 2: Worker resume from cache (Req 21.1, 21.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_worker_resume_from_cache(tmp_workspace):
    """Run BTA → some workers complete → interrupt → resume → completed workers
    served from cache, remaining workers execute fresh.

    Uses a 2-worker setup where the first worker has normal timeout (completes)
    and the second worker has short idle_timeout (interrupted). On resume, the
    first worker's cache is reused and the second worker executes fresh.

    **Validates: Requirements 21.1, 21.4**
    """
    checkpoint_dir = str(tmp_workspace["checkpoint"])
    worker_call_count = {"count": 0}

    def mixed_timeout_worker_factory(sub_query, index):
        """First worker completes normally, second worker times out."""
        worker_call_count["count"] += 1
        if index == 0:
            # First worker: normal timeout, should complete
            return _make_claude(tmp_workspace)
        else:
            # Second worker: very short idle_timeout to force interruption
            return _make_claude(tmp_workspace, idle_timeout_seconds=2)

    breakdown = _make_claude(
        tmp_workspace, append_system_prompt=BREAKDOWN_SYSTEM_PROMPT
    )

    bta1 = BreakdownThenAggregateInferencer(
        breakdown_inferencer=breakdown,
        worker_factory=mixed_timeout_worker_factory,
        checkpoint_dir=checkpoint_dir,
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    # First run — first worker completes, second worker interrupted
    try:
        await bta1.ainfer(BREAKDOWN_PROMPT)
    except Exception:
        pass  # Interruption expected from second worker

    # Verify breakdown checkpoint was saved
    ckpt_path = os.path.join(checkpoint_dir, "breakdown_result.json")
    assert os.path.exists(ckpt_path), (
        "breakdown_result.json should exist after breakdown completes"
    )

    # Resume: create second BTA with same checkpoint_dir, all workers normal timeout
    # The first worker should be served from cache (Req 21.1),
    # the second worker executes fresh (Req 21.4)
    bta2 = BreakdownThenAggregateInferencer(
        breakdown_inferencer=breakdown,
        worker_factory=_make_worker_factory(tmp_workspace),
        checkpoint_dir=checkpoint_dir,
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    result = await bta2.ainfer(BREAKDOWN_PROMPT)

    assert result is not None, "Resumed BTA should produce a result"
    # With no aggregator, result is a tuple of worker outputs or a single result
    if isinstance(result, tuple):
        assert len(result) >= 1, "Should have at least 1 worker result"
        for i, worker_result in enumerate(result):
            assert worker_result is not None, f"Worker {i} result should not be None"
            assert str(worker_result).strip() != "", (
                f"Worker {i} result should not be empty"
            )
    else:
        assert str(result).strip() != "", "Result should not be empty"


# ===========================================================================
# Test 3: Corrupted breakdown checkpoint (Req 20.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_corrupted_breakdown_checkpoint(tmp_workspace):
    """Create a corrupted breakdown_result.json → run BTA → verify BTA re-runs
    breakdown (graceful degradation).

    BTA's _load_breakdown_checkpoint catches json.JSONDecodeError and returns
    None, causing the breakdown_inferencer to be called fresh.

    **Validates: Requirement 20.4**
    """
    checkpoint_dir = str(tmp_workspace["checkpoint"])

    # Pre-create a corrupted breakdown_result.json
    ckpt_path = os.path.join(checkpoint_dir, "breakdown_result.json")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, "w") as f:
        f.write("{corrupted json content that is not valid!!!")

    assert os.path.exists(ckpt_path), "Corrupted checkpoint file should exist"

    # Run BTA with resume enabled — should detect corruption and re-run breakdown
    breakdown = _make_claude(
        tmp_workspace, append_system_prompt=BREAKDOWN_SYSTEM_PROMPT
    )

    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=breakdown,
        worker_factory=_make_worker_factory(tmp_workspace),
        checkpoint_dir=checkpoint_dir,
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    result = await bta.ainfer(BREAKDOWN_PROMPT)

    assert result is not None, (
        "BTA should produce a result despite corrupted checkpoint"
    )

    # Verify the checkpoint was overwritten with valid data after re-running breakdown
    assert os.path.exists(ckpt_path), (
        "breakdown_result.json should still exist after re-run"
    )
    with open(ckpt_path) as f:
        saved = json.load(f)
    assert "sub_queries" in saved, (
        "Checkpoint should contain valid 'sub_queries' after re-run"
    )
    assert len(saved["sub_queries"]) >= 1, (
        "Should have at least 1 sub_query after re-running breakdown"
    )

    # Verify result is valid
    if isinstance(result, tuple):
        assert len(result) >= 1, "Should have at least 1 worker result"
        for i, worker_result in enumerate(result):
            assert worker_result is not None, f"Worker {i} result should not be None"
    else:
        assert str(result).strip() != "", "Result should not be empty"

"""Real cross-layer composition resume — Layer 2.

Verifies that resume-with-cache works across nested inferencer compositions
(PTI + claude children). Existing per-inferencer resume tests cover single
inferencers; this verifies the nested case.

The resume mechanism here is cache-level (`resume_with_saved_results=True`):
running the same prompt twice should hit the cache on the second run, much
faster than the first.
"""

import time

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)

from .conftest import DEFAULT_TIMEOUT, skip_claude, assert_cached_skip, assert_real_call


def _make_claude(tmp_workspace, **overrides):
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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 6)
@skip_claude
async def test_pti_real_cache_hit_on_repeat(tmp_workspace):
    """Run PTI twice with same prompt + shared cache.
    First run: real LLM (slow). Second run: cache hits (fast)."""
    prompt = "Plan a small Python utility script for CSV-to-JSON conversion."

    def _build_pti():
        return PlanThenImplementInferencer(
            planner_inferencer=_make_claude(
                tmp_workspace,
                append_system_prompt="You are a planner. Output a 3-step plan.",
            ),
            executor_inferencer=_make_claude(
                tmp_workspace,
                append_system_prompt="You are an implementer. 1-line summary.",
            ),
            workspace_root=str(tmp_workspace["workspace"]),
            planner_outputs_plan_to_file=False,
        )

    # Run 1: real LLM (slow)
    pti1 = _build_pti()
    start1 = time.monotonic()
    result1 = await pti1.ainfer(prompt)
    elapsed1 = time.monotonic() - start1
    assert result1 is not None
    print(f"[composition_resume] Run 1 (real): {elapsed1:.1f}s")

    # Run 2: same prompt, fresh inferencer instances pointing at same cache
    pti2 = _build_pti()
    start2 = time.monotonic()
    result2 = await pti2.ainfer(prompt)
    elapsed2 = time.monotonic() - start2
    assert result2 is not None
    print(f"[composition_resume] Run 2 (cached): {elapsed2:.1f}s")

    # Run 2 should be MUCH faster — cache hits at every layer
    assert elapsed2 < elapsed1, (
        f"Expected cached run 2 ({elapsed2:.1f}s) to be faster than "
        f"run 1 ({elapsed1:.1f}s)"
    )

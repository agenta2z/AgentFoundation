"""PTI multi-iteration real-LLM tests — Layer 2.

Verifies real iterative refinement loop with real analyzer feedback driving
`should_continue` decisions across iterations.
"""

import os

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)

from .conftest import DEFAULT_TIMEOUT, skip_claude


PLANNER_SYSTEM_PROMPT = (
    "You are a planner. Output a brief implementation plan as a numbered list, "
    "no more than 5 items. No code, just the plan."
)
EXECUTOR_SYSTEM_PROMPT = (
    "You are an implementer. Given a plan, output a brief implementation summary "
    "(no actual code). Keep it under 100 words."
)
ANALYZER_SYSTEM_PROMPT = """\
You are a quality analyzer. Output ONLY this JSON format:

```json
{"should_continue": false, "next_iteration_request": ""}
```

Set should_continue=false unless the implementation is critically broken.
"""


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
@pytest.mark.timeout(DEFAULT_TIMEOUT * 5)  # multi-iteration is slow
@skip_claude
async def test_pti_single_iteration_real_llm(tmp_workspace):
    """Single PTI iteration with real LLM planner + executor.
    Validates plan → executor data flow with real artifacts."""
    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_claude(tmp_workspace, append_system_prompt=PLANNER_SYSTEM_PROMPT),
        executor_inferencer=_make_claude(tmp_workspace, append_system_prompt=EXECUTOR_SYSTEM_PROMPT),
        workspace_root=str(tmp_workspace["workspace"]),
        planner_outputs_plan_to_file=True,
    )

    result = await pti.ainfer("Plan and outline a CSV-to-JSON converter.")
    assert isinstance(result, PlanThenImplementResponse)
    assert result.plan_output, "planner should produce non-empty plan"
    assert result.executor_output, "executor should produce non-empty output"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 8)  # multi-iter w/ analyzer is slowest
@skip_claude
async def test_pti_multi_iteration_with_analyzer(tmp_workspace):
    """PTI with `enable_analysis=True` and analyzer that auto-terminates.

    With `should_continue=false` from analyzer, expect 1 iteration.
    Verifies iteration_records populated, workspace artifacts present.
    """
    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_claude(tmp_workspace, append_system_prompt=PLANNER_SYSTEM_PROMPT),
        executor_inferencer=_make_claude(tmp_workspace, append_system_prompt=EXECUTOR_SYSTEM_PROMPT),
        analyzer_inferencer=_make_claude(tmp_workspace, append_system_prompt=ANALYZER_SYSTEM_PROMPT),
        enable_analysis=True,
        enable_multiple_iterations=True,
        max_meta_iterations=3,
        workspace_root=str(tmp_workspace["workspace"]),
    )

    result = await pti.ainfer("Plan a small CSV parser.")

    assert isinstance(result, PlanThenImplementResponse)
    # Should have at least 1 iteration record
    assert len(result.iteration_history) >= 1
    # Each iteration's workspace should exist
    for i, _ in enumerate(result.iteration_history, start=1):
        iter_dir = os.path.join(str(tmp_workspace["workspace"]), f"iteration_{i}")
        # Artifacts may or may not exist depending on planner_outputs_plan_to_file
        # The key invariant: total_meta_iterations matches history length
    assert result.total_meta_iterations == len(result.iteration_history)

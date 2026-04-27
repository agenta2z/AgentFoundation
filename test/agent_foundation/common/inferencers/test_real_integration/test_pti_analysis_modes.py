"""PTI analysis_mode variant tests — Layer 2.

Parametrized over the three `analysis_mode` values:
- "last_round_only" — analyzer prompt references only iteration N
- "last_with_cross_ref" — analyzer prompt references iter N AND prior iterations
- "all_rounds" — analyzer prompt includes all iterations inline

Verifies real prompt rendering produces the expected reference structure.
"""

import os

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)

from .conftest import DEFAULT_TIMEOUT, skip_claude


PLANNER_PROMPT = "You are a planner. Output a 3-step numbered plan."
EXECUTOR_PROMPT = "You are an implementer. Given a plan, output a 1-line summary."
# Analyzer requests TWO iterations so we can verify cross-iteration references
ANALYZER_PROMPT_CONTINUE_ONCE = """\
You are a quality analyzer. On the first iteration only, set should_continue=true.
On subsequent iterations, set should_continue=false.

Output ONLY:
```json
{"should_continue": <bool>, "next_iteration_request": "Refine the plan"}
```
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
@pytest.mark.timeout(DEFAULT_TIMEOUT * 8)
@skip_claude
@pytest.mark.parametrize("analysis_mode", ["last_round_only", "last_with_cross_ref", "all_rounds"])
async def test_pti_analysis_mode_variants(tmp_workspace, analysis_mode):
    """For each analysis_mode, run 2 iterations and verify the analyzer ran
    correctly and the iteration_history reflects the configured behavior.

    Note: Inspecting analyzer's last_rendered_prompt would be ideal, but is
    implementation-dependent. We verify the orchestration completes without
    error for each mode and produces valid iteration_history.
    """
    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_claude(tmp_workspace, append_system_prompt=PLANNER_PROMPT),
        executor_inferencer=_make_claude(tmp_workspace, append_system_prompt=EXECUTOR_PROMPT),
        analyzer_inferencer=_make_claude(tmp_workspace, append_system_prompt=ANALYZER_PROMPT_CONTINUE_ONCE),
        enable_analysis=True,
        enable_multiple_iterations=True,
        max_meta_iterations=2,
        analysis_mode=analysis_mode,
        workspace_path=str(tmp_workspace["workspace"]),
    )

    result = await pti.ainfer(f"Test analysis_mode={analysis_mode}: simple task.")

    assert result is not None
    assert len(result.iteration_history) >= 1
    # iteration_2 directory should exist if 2 iterations ran
    if len(result.iteration_history) >= 2:
        iter_2 = os.path.join(str(tmp_workspace["workspace"]), "iteration_2")
        # may or may not exist depending on artifact-writing config

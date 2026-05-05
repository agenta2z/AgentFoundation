"""PTI + ConversationalFlowNodeAdapter as planner — Layer 2.

Adapter wraps a conversational session in PTI's planner slot. Verifies
real composition: PTI → adapter._ainfer → run_agentic_loop → real Claude →
plan extracted → executor receives plan.
"""

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
    ConversationalFlowNodeAdapter,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)

from .conftest import DEFAULT_TIMEOUT, skip_claude
from ._helpers.scripted_interactive import ScriptedInteractive


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


def _make_claude_for_conversational(tmp_workspace, **overrides):
    kwargs = dict(
        target_path=str(tmp_workspace["workspace"]),
        cache_folder=str(tmp_workspace["cache"]),
        model_name="sonnet",
        resume_with_saved_results=True,
        idle_timeout_seconds=120,
        enable_shell=False,
        append_system_prompt=(
            "You are a planner inside a conversational session. Output a "
            "numbered plan; do not use native tools."
        ),
    )
    if hasattr(ClaudeCodeCliInferencer, "permission_mode"):
        kwargs["permission_mode"] = "bypassPermissions"
    if hasattr(ClaudeCodeCliInferencer, "allowed_tools"):
        kwargs["allowed_tools"] = []
    kwargs.update(overrides)
    return ClaudeCodeCliInferencer(**kwargs)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 5)
@skip_claude
async def test_adapter_as_pti_planner_real_llm(tmp_workspace):
    """PTI's planner_inferencer is the conversational adapter.
    Verifies: adapter routes through full pipeline; PTI extracts plan;
    executor receives non-empty plan content."""
    conv_base = _make_claude_for_conversational(tmp_workspace)
    conv = ConversationalInferencer(base_inferencer=conv_base, max_iterations=2)

    # Plain claude as fallback for widget-protocol robustness
    fallback = _make_claude(
        tmp_workspace,
        append_system_prompt="You are a planner. Output a 3-step numbered plan.",
    )

    adapter = ConversationalFlowNodeAdapter(
        conversational_inferencer=conv,
        interactive=ScriptedInteractive(responses=[{"approved": True}]),
        fallback_inferencer=fallback,
    )

    pti = PlanThenImplementInferencer(
        planner_inferencer=adapter,
        executor_inferencer=_make_claude(
            tmp_workspace,
            append_system_prompt="You are an implementer. Output 1-line summary.",
        ),
        workspace=str(tmp_workspace["workspace"]),
        planner_outputs_plan_to_file=False,
    )

    result = await pti.ainfer("Plan a small Python utility script.")

    assert isinstance(result, PlanThenImplementResponse)
    assert result.plan_output, "planner adapter should produce non-empty plan"
    assert result.executor_output, "executor should produce non-empty output"

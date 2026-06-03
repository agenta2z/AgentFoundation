"""Integration test: SOP workflow with real LLM + mock tool responses.

Tests that the SOP auto-advances through all phases when tools return
canned responses. Uses a real LLM backend (Claude/RovoDev) but mocks
tool executors so each phase completes in seconds, not minutes.

Two scenarios:
  1. Sync: all tools forced synchronous (CLI mode pattern)
  2. Async: tools run async via inbox event loop (server mode pattern)

Usage:
    pytest test/.../test_sop_with_mock_tools.py -m integration -s -v
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

AF_ROOT = Path(__file__).resolve().parents[5]
OS_ROOT = AF_ROOT.parent / "OpenStartup"

skip_no_backend = pytest.mark.skipif(
    shutil.which("claude") is None and shutil.which("acli") is None,
    reason="No LLM backend (claude or acli) on PATH",
)

EXTRA_SOP_DIRS = OS_ROOT / "src" / "openteam" / "server" / "resources" / "sops"


def _build_base():
    """Build a real LLM backend."""
    if shutil.which("claude"):
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
            ClaudeCodeCliInferencer,
        )
        return ClaudeCodeCliInferencer(
            model_name="sonnet",
            target_path=str(Path.home() / "MyProjects"),
            idle_timeout_seconds=120,
        )

    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
        RovoDevCliInferencer,
    )
    return RovoDevCliInferencer(model_name="sonnet", yolo=True)


def _mock_tool_executor_factory(*, delay: float = 0.0):
    """Create a mock tool executor that returns canned responses."""

    async def _executor(tool_name: str, arguments: dict[str, Any]) -> Any:
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
            ToolExecutionResult,
        )

        if delay > 0:
            await asyncio.sleep(delay)

        canonical = tool_name.replace("-", "_")

        if canonical == "sop":
            from agent_foundation.resources.tools.sop.executor import execute
            return await execute(arguments, {
                "extra_sop_dirs": [EXTRA_SOP_DIRS],
            })

        if canonical == "create_role":
            return ToolExecutionResult(
                result=(
                    "Role document created successfully.\n"
                    "# Machine Learning Engineer\n"
                    "## Responsibilities\n"
                    "- Build and train ML models\n"
                    "- Design data pipelines\n"
                    "- Evaluate model performance\n"
                    "Document saved to: /tmp/mock_role_doc.md"
                ),
            )

        if canonical == "role_setup":
            return ToolExecutionResult(
                result=(
                    "Role setup complete.\n"
                    "Created 3 skills: model_training, data_pipeline, model_evaluation\n"
                    "Created 2 tools: jupyter_runner, experiment_tracker\n"
                    "Association map saved to: /tmp/mock_role_setup/"
                ),
            )

        return ToolExecutionResult(result=f"Mock tool '{tool_name}' executed OK.")

    return _executor


def _build_ci(*, yolo: bool = True, async_tools: bool = False):
    """Build a CI wired for SOP testing with mock tools."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )
    from agent_foundation.resources.tools.registry import load_all_tools

    base = _build_base()
    tool_dirs = [OS_ROOT / "src" / "openteam" / "server" / "resources" / "tools"]
    tool_registry = load_all_tools(extra_dirs=tool_dirs)

    if not async_tools:
        for td in tool_registry.values():
            if hasattr(td, "asynchronous"):
                td.asynchronous = False

    delay = 0.1 if async_tools else 0.0
    mock_executor = _mock_tool_executor_factory(delay=delay)

    ci = ConversationalInferencer(
        base_inferencer=base,
        tool_registry=tool_registry,
        tool_executor=mock_executor,
        yolo_mode=yolo,
    )

    return ci


async def _enter_sop(ci, sop_name="role_creation", yolo=True):
    """Initialize SOP state on the CI."""
    from agent_foundation.resources.tools.sop.executor import execute

    result = await execute(
        {"workflow": sop_name, "yolo": yolo},
        {"extra_sop_dirs": [EXTRA_SOP_DIRS]},
    )
    ci.update_prior_context(**result.context_updates)
    return result


# ── Sync scenario ────────────────────────────────────────────────────


@skip_no_backend
@pytest.mark.integration
def test_sop_sync_all_phases_advance(tmp_path):
    """Real LLM + mock tools (sync): SOP advances through all 5 phases."""
    from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus

    async def _run():
        ci = _build_ci(yolo=True, async_tools=False)
        await _enter_sop(ci)

        assert ci.sop_state is not None
        assert ci.sop_state.current_phase == "0"
        print(f"\n[sync] SOP entered: {ci.sop_state.sop_name}")
        print(f"[sync] Initial phase: {ci.sop_state.current_phase}")

        # Enable inbox with auto-shutdown
        ci.enable_inbox(auto_shutdown_on_sop_complete=True)
        ci.inbox_put_user("hire a machine learning engineer")

        result = await ci.run()

        print(f"[sync] Final phase: {ci.sop_state.current_phase}")
        print(f"[sync] Phase status: {ci.sop_state.phase_status}")
        print(f"[sync] Completed: {[c if isinstance(c, str) else getattr(c, 'phase', c) for c in ci.sop_state.completed_phases]}")

        assert ci.sop_state.phase_status == PhaseStatus.COMPLETED, (
            f"SOP not completed. Phase: {ci.sop_state.current_phase}, "
            f"Status: {ci.sop_state.phase_status}"
        )
        assert ci.sop_state.current_phase is None

    asyncio.run(_run())


# ── Async scenario (inbox event loop) ────────────────────────────────


@skip_no_backend
@pytest.mark.integration
def test_sop_async_inbox_drives_completion(tmp_path):
    """Real LLM + mock tools (async): inbox event loop auto-advances SOP."""
    from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus

    async def _run():
        ci = _build_ci(yolo=True, async_tools=True)
        await _enter_sop(ci)

        assert ci.sop_state is not None
        print(f"\n[async] SOP entered: {ci.sop_state.sop_name}")

        ci.enable_inbox(auto_shutdown_on_sop_complete=True)
        ci.inbox_put_user("hire a machine learning engineer")

        result = await ci.run()

        print(f"[async] Final phase: {ci.sop_state.current_phase}")
        print(f"[async] Phase status: {ci.sop_state.phase_status}")

        assert ci.sop_state.phase_status == PhaseStatus.COMPLETED, (
            f"SOP not completed via async inbox. Phase: {ci.sop_state.current_phase}, "
            f"Status: {ci.sop_state.phase_status}"
        )

    asyncio.run(_run())


# ── Minimal smoke: just phase 0 ─────────────────────────────────────


@skip_no_backend
@pytest.mark.integration
def test_sop_phase_0_advances_to_phase_1(tmp_path):
    """Real LLM: verify Phase 0 (multiple_choice + confirmation) advances to Phase 1."""
    async def _run():
        ci = _build_ci(yolo=True, async_tools=False)
        await _enter_sop(ci)

        assert ci.sop_state.current_phase == "0"

        result = await ci.run_agentic_loop("hire a machine learning engineer")

        print(f"\n[phase0] After first loop: phase={ci.sop_state.current_phase}")
        print(f"[phase0] Completed: {ci.sop_state.completed_phases}")

        # Phase 0 should have advanced (confirmation gate passed via yolo)
        completed = [
            c if isinstance(c, str) else getattr(c, "phase", c)
            for c in ci.sop_state.completed_phases
        ]
        assert "0" in completed, (
            f"Phase 0 did not complete. Current: {ci.sop_state.current_phase}"
        )

    asyncio.run(_run())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])

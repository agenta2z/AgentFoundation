"""SOP CLI — run SOPs from the terminal.

Yolo mode:       python -m agent_foundation.resources.tools.sop role_creation --yolo "hire an MLE"
Interactive mode: python -m agent_foundation.resources.tools.sop role_creation "hire an MLE"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_base_inferencer(model: str) -> Any:
    """Try available CLI backends (Claude Code → RovoDev)."""
    if shutil.which("claude"):
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
            ClaudeCodeCliInferencer,
        )
        return ClaudeCodeCliInferencer(model_name=model)

    if shutil.which("acli"):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
            RovoDevCliInferencer,
        )
        return RovoDevCliInferencer(model_name="sonnet", yolo=True)

    print("ERROR: No LLM backend found. Install Claude Code (claude) or RovoDev (acli).")
    sys.exit(1)


async def run_sop(
    sop_name: str,
    request: str,
    *,
    yolo: bool = False,
    model: str = "opus[1m]",
    extra_sop_dirs: list[str] | None = None,
    extra_tool_dirs: list[str] | None = None,
) -> int:
    """Run an SOP end-to-end. Returns exit code."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
        PausedResult,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )
    from agent_foundation.resources.tools.sop.executor import execute
    from agent_foundation.resources.tools.registry import load_all_tools
    from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus

    base = _build_base_inferencer(model)

    interactive = None
    if not yolo:
        try:
            from agent_foundation.ui.cli import RichTerminalInteractive
            interactive = RichTerminalInteractive(system_name="SOP")
        except ImportError:
            pass

    tool_dirs = [Path(d) for d in extra_tool_dirs] if extra_tool_dirs else None
    tool_registry = load_all_tools(extra_dirs=tool_dirs)

    session_context: dict[str, Any] = {
        "extra_sop_dirs": [Path(d) for d in extra_sop_dirs] if extra_sop_dirs else [],
        "extra_tool_dirs": [Path(d) for d in extra_tool_dirs] if extra_tool_dirs else [],
    }

    ci = ConversationalInferencer(
        base_inferencer=base,
        tool_registry=tool_registry,
        interactive=interactive,
    )

    result = await execute(
        {"workflow": sop_name, "yolo": yolo},
        session_context,
    )

    if "sop_state" not in (result.context_updates or {}):
        print(result.result)
        return 1

    ci.update_prior_context(**result.context_updates)
    print(f"Entered SOP: {sop_name}")

    if yolo:
        agentic_result = await ci.run_agentic_loop(request, interactive=interactive)
        if getattr(agentic_result, "exhausted_max_iterations", False):
            print("WARNING: SOP exhausted max_iterations — may be incomplete")
        if ci.sop_state and ci.sop_state.phase_status == PhaseStatus.COMPLETED:
            print("SOP completed successfully.")
        elif ci.sop_state:
            print(f"SOP ended at phase {ci.sop_state.current_phase} ({ci.sop_state.phase_status})")
    else:
        content = request
        while True:
            agentic_result = await ci.run_agentic_loop(content, interactive=interactive)
            if ci.sop_state is None:
                print("SOP exited.")
                break
            if ci.sop_state.phase_status == PhaseStatus.COMPLETED:
                print("SOP completed successfully.")
                break
            if isinstance(agentic_result, PausedResult):
                print("SOP paused.")
                break
            try:
                content = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                break

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sop",
        description="Run a Standard Operating Procedure end-to-end.",
    )
    parser.add_argument("sop_name", help="SOP name (e.g., role_creation, code_optimization)")
    parser.add_argument("request", nargs="?", default="", help="Initial request/context for the SOP")
    parser.add_argument("--yolo", action="store_true", help="Auto-resolve all confirmations")
    parser.add_argument("--model", default="opus[1m]", help="LLM model (default: opus[1m])")
    parser.add_argument("--extra-sop-dirs", nargs="*", default=[], help="Additional SOP directories")
    parser.add_argument("--extra-tool-dirs", nargs="*", default=[], help="Additional tool directories")

    args = parser.parse_args(argv)

    return asyncio.run(
        run_sop(
            sop_name=args.sop_name,
            request=args.request or f"Starting SOP: {args.sop_name}",
            yolo=args.yolo,
            model=args.model,
            extra_sop_dirs=args.extra_sop_dirs,
            extra_tool_dirs=args.extra_tool_dirs,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

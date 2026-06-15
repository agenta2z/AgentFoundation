"""SOP CLI — run SOPs from the terminal.

Yolo mode:       python -m agent_foundation.resources.tools.sop role_creation --yolo "hire an MLE"
Interactive mode: python -m agent_foundation.resources.tools.sop role_creation "hire an MLE"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_ci_from_config(
    model: str,
    backend: str | None = None,
    tool_registry: dict | None = None,
    tool_executor: Any = None,
    interactive: Any = None,
) -> Any:
    """Build a ConversationalInferencer from this tool's configs/default.yaml.

    Thin wrapper over the shared ``_ci_host.build_ci_from_config`` helper (also
    used by the task conversational router), preserving the SOP CLI behavior of
    printing + exiting on an unknown ``--backend``.
    """
    from agent_foundation.resources.tools import _ci_host

    configs_dir = Path(__file__).parent / "configs"
    try:
        return _ci_host.build_ci_from_config(
            configs_dir / "default.yaml",
            model=model,
            backend=backend,
            backend_dir=configs_dir / "base_inferencer",
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            interactive=interactive,
            target_path=Path.cwd(),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


async def run_sop(
    sop_name: str,
    request: str,
    *,
    yolo: bool = False,
    model: str = "opus[1m]",
    backend: str | None = None,
    extra_sop_dirs: list[str] | None = None,
    extra_tool_dirs: list[str] | None = None,
) -> int:
    """Run an SOP end-to-end. Returns exit code."""
    from agent_foundation.resources.tools.sop.executor import execute
    from agent_foundation.resources.tools.registry import load_all_tools
    from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus

    interactive = None
    if not yolo:
        try:
            from agent_foundation.ui.cli import RichTerminalInteractive
            interactive = RichTerminalInteractive(system_name="SOP")
        except ImportError:
            pass

    from agent_foundation.resources.tools import _ci_host

    tool_dirs = [Path(d) for d in extra_tool_dirs] if extra_tool_dirs else None
    tool_registry = load_all_tools(extra_dirs=tool_dirs)

    # Force all tools to synchronous mode for CLI.
    # In the server, async tools fire-and-forget while the chat continues.
    # In CLI, we need tools to complete inline before the loop continues.
    _ci_host.force_tools_synchronous(tool_registry)

    # Session workspace: _runtime/sop/<sop_name>__<timestamp>/
    from datetime import UTC, datetime
    from uuid import uuid4

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    session_id = f"{sop_name}__{ts}__{uuid4().hex[:8]}"
    runtime_root = Path.cwd() / "_runtime" / "sop"
    session_dir = runtime_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    turns_dir = session_dir / "turns"
    turns_dir.mkdir(exist_ok=True)

    session_context: dict[str, Any] = {
        "extra_sop_dirs": [Path(d) for d in extra_sop_dirs] if extra_sop_dirs else [],
        "extra_tool_dirs": [Path(d) for d in extra_tool_dirs] if extra_tool_dirs else [],
        "session_root": str(session_dir),
        "working_dir": str(Path.cwd()),
        "session_id": session_id,
    }

    # Turn logging callbacks — create rich turn artifacts per CI turn.
    #
    # Structure:
    #   turn_001/
    #     user_input.txt          ← one per turn (the trigger)
    #     messages.json           ← cumulative snapshot at turn end
    #     round_001/              ← per-LLM-call artifacts
    #       rendered_prompt.txt
    #       response.md
    #       template_source.txt
    #       template_feed.json
    #       template_config.json
    #     round_002/              ← second LLM call (after tool execution)
    #       ...
    #
    _turn_counter = [0]
    _round_counter = [0]
    _ci_ref = [None]  # populated after CI construction

    async def _on_new_turn(turn_number, user_input):
        _turn_counter[0] += 1
        _round_counter[0] = 0
        turn_dir = turns_dir / f"turn_{_turn_counter[0]:03d}"
        turn_dir.mkdir(exist_ok=True)
        (turn_dir / "user_input.txt").write_text(user_input or "", encoding="utf-8")
        ci = _ci_ref[0]
        if ci and hasattr(ci, "base_inferencer"):
            ci.base_inferencer.cache_folder = str(turn_dir)
        return _turn_counter[0]

    async def _on_prompt_rendered(ci_instance, raw_response):
        import json as _json
        _round_counter[0] += 1
        turn_dir = turns_dir / f"turn_{_turn_counter[0]:03d}"
        round_dir = turn_dir / f"round_{_round_counter[0]:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        rendered = getattr(ci_instance, "_last_rendered_prompt", "")
        (round_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
        tmpl_src = getattr(ci_instance, "_last_template_source", "")
        (round_dir / "template_source.txt").write_text(tmpl_src, encoding="utf-8")
        feed = getattr(ci_instance, "_last_template_feed", {})
        try:
            (round_dir / "template_feed.json").write_text(
                _json.dumps(feed, indent=2, default=str), encoding="utf-8",
            )
        except Exception:
            pass
        config = getattr(ci_instance, "_last_template_config", {})
        try:
            (round_dir / "template_config.json").write_text(
                _json.dumps(config, indent=2, default=str), encoding="utf-8",
            )
        except Exception:
            pass
        (round_dir / "response.md").write_text(
            str(raw_response) if raw_response else "", encoding="utf-8",
        )

    async def _on_turn_complete(turn_number):
        turn_dir = turns_dir / f"turn_{_turn_counter[0]:03d}"
        turn_dir.mkdir(exist_ok=True)
        ci = _ci_ref[0]
        if ci:
            import json as _json
            try:
                (turn_dir / "messages.json").write_text(
                    _json.dumps(ci._messages, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass

    logger.info("SOP session: %s", session_dir)

    # Synchronous tool executor that resolves executors from each tool.json.
    _tool_executor = _ci_host.make_tool_executor(
        session_context,
        tool_dirs=tool_dirs,
        af_tools_root=Path(__file__).resolve().parent.parent,  # AF tools root
    )

    ci = _build_ci_from_config(
        model=model,
        backend=backend,
        tool_registry=tool_registry,
        tool_executor=_tool_executor,
        interactive=interactive,
    )
    # So the /sop command + Available-SOPs list discover the same SOPs the
    # executor sees via session_context (static config; set before any turn).
    ci._extra_sop_dirs = session_context.get("extra_sop_dirs") or []
    _ci_ref[0] = ci

    result = await execute(
        {"workflow": sop_name, "yolo": yolo},
        session_context,
    )

    if "sop_state" not in (result.context_updates or {}):
        print(result.result)
        return 1

    ci.update_prior_context(**result.context_updates)
    print(f"Entered SOP: {sop_name}")
    print(f"Session: {session_dir}")

    # Enable inbox event loop with turn logging callbacks
    ci.enable_inbox(
        interactive=interactive,
        auto_shutdown_on_sop_complete=True,
        on_new_turn=_on_new_turn,
        on_prompt_rendered=_on_prompt_rendered,
        on_turn_complete=_on_turn_complete,
    )
    ci.inbox_put_user(request)

    if yolo:
        # Yolo: run() processes all phases via inbox, exits when SOP completes
        last_result = await ci.run()
        if ci.sop_state and ci.sop_state.phase_status == PhaseStatus.COMPLETED:
            print("SOP completed successfully.")
        elif ci.sop_state:
            print(f"SOP ended at phase {ci.sop_state.current_phase} ({ci.sop_state.phase_status})")
    else:
        # Interactive: run() in background, user input feeds inbox
        import asyncio as _aio

        run_task = _aio.create_task(ci.run())
        try:
            while not ci._shutdown_requested:
                try:
                    line = await _aio.to_thread(input, "\n> ")
                    ci.inbox_put_user(line)
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    ci.request_shutdown()
                    break
        finally:
            if not ci._shutdown_requested:
                ci.request_shutdown()
            await run_task

        if ci.sop_state and ci.sop_state.phase_status == PhaseStatus.COMPLETED:
            print("SOP completed successfully.")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sop",
        description="Run a Standard Operating Procedure end-to-end.",
    )
    parser.add_argument("sop_name", help="SOP name (e.g., role_creation, code_optimization)")
    parser.add_argument("request", nargs="?", default="", help="Initial request/context for the SOP (place before flags, or use --request)")
    parser.add_argument("--request", dest="request_flag", default=None, help="Initial request (alternative to positional arg, safe with nargs=* flags)")
    parser.add_argument("--yolo", action="store_true", help="Auto-resolve all confirmations")
    parser.add_argument("--model", default="opus[1m]", help="LLM model (default: opus[1m])")
    parser.add_argument("--backend", default=None, help="Backend inferencer: claude_code (default), rovodev, or custom YAML name")
    parser.add_argument("--extra-sop-dirs", nargs="*", default=[], help="Additional SOP directories")
    parser.add_argument("--extra-tool-dirs", nargs="*", default=[], help="Additional tool directories")

    args = parser.parse_args(argv)

    return asyncio.run(
        run_sop(
            sop_name=args.sop_name,
            request=args.request_flag or args.request or f"Starting SOP: {args.sop_name}",
            yolo=args.yolo,
            model=args.model,
            backend=args.backend,
            extra_sop_dirs=args.extra_sop_dirs,
            extra_tool_dirs=args.extra_tool_dirs,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

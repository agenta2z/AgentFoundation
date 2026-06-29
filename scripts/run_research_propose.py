"""Standalone driver: invoke research_propose end-to-end via the production
``derived_tool_execute`` path.

Exists because ``research_propose`` is a derived tool (no standalone ``__main__``);
production callers reach it only via the Conversational AI host. This driver wires
the same path for direct CLI testing/monitoring.

Usage::

    buck2 run //_tony_dev/CoreProjects/AgentFoundation/scripts:run_research_propose -- \
        --request "<research question>" \
        --workflow-target-path /path/to/codebase
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_research_propose",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--request", required=True, help="Research question.")
    parser.add_argument(
        "--workflow-target-path",
        required=False,
        default=None,
        help="Codebase / file to research. Defaults to cwd if omitted.",
    )
    parser.add_argument(
        "--docs-path", default=None, help="Optional documentation directory."
    )
    parser.add_argument(
        "--separate-proposal-files", action="store_true",
        help="Write each proposal to its own file.",
    )
    parser.add_argument(
        "--research-only", action="store_true",
        help="Run breakdown + research only (skip proposals + unified plan).",
    )
    parser.add_argument(
        "--breakdown-only", action="store_true",
        help="Run only the task breakdown (skip research + proposals).",
    )
    parser.add_argument(
        "--max-breakdown", default=None,
        help="Guidance for subtask count, e.g. '5 to 20'.",
    )
    parser.add_argument(
        "--max-researches", type=int, default=None,
        help="Hard cap on research workers.",
    )
    parser.add_argument(
        "--model", default=None, help="Override the LLM model."
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logger level (default: INFO).",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Absolute path of a prior workspace to resume from. "
             "Completed stages (breakdown, finished worker flows) skip; "
             "in-flight steps restart at their last saved checkpoint.",
    )
    parser.add_argument(
        "--copy-workspace", action="store_true",
        help="With --resume, copy the workspace to a sibling _resume_<ts> dir "
             "and run from the copy instead of modifying the original in place.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("run_research_propose")

    # Log inferencer-topology env-var snapshot up front for monitoring.
    for var in (
        "RESEARCH_PROPOSE__FLOW_INFERENCERS",
        "RESEARCH_PROPOSE__MAIN_INFERENCER",
        "TASK__MAIN_INFERENCER",
        "TASK__FLOW_INFERENCERS",
        "CLAUDE_CODE_INFERANCER_NO_SANDBOX",
        "CODEX_INFERANCER_NO_SANDBOX",
    ):
        log.info("env %s=%r", var, os.environ.get(var))

    from agent_foundation.resources.tools.registry import (
        derived_tool_execute,
        load_tool,
    )

    tool = load_tool("research_propose")
    log.info("Loaded tool: %s (aliases=%s, env_prefix=%s)",
             tool.name, tool.aliases,
             (tool.derived_from or {}).get("defaults", {}).get("env_prefix"))

    arguments: dict[str, object] = {"request": args.request}
    if args.workflow_target_path:
        arguments["workflow_target_path"] = str(
            Path(args.workflow_target_path).expanduser().resolve()
        )
    if args.docs_path:
        arguments["docs_path"] = str(Path(args.docs_path).expanduser().resolve())
    if args.separate_proposal_files:
        arguments["separate_proposal_files"] = True
    if args.research_only:
        arguments["research_only"] = True
    if args.breakdown_only:
        arguments["breakdown_only"] = True
    if args.max_breakdown:
        arguments["max_breakdown"] = args.max_breakdown
    if args.max_researches is not None:
        arguments["max_researches"] = args.max_researches
    if args.model:
        arguments["model"] = args.model
    if args.resume:
        # ``derived_tool_execute`` forwards ``resume`` verbatim into task_args.
        # ``task.executor.execute`` reads ``arguments["resume"]`` and calls
        # ``_apply_resume`` -> passes ``resume_workspace`` to ``_run_topology``.
        arguments["resume"] = str(Path(args.resume).expanduser().resolve())
    if args.copy_workspace:
        arguments["copy_workspace"] = True

    log.info("Invoking research_propose with arguments:\n%s",
             json.dumps(arguments, indent=2, default=str))

    async def _run():
        result = await derived_tool_execute(
            arguments,
            session_context={},
            derived_from=tool.derived_from or {},
            tool_name="research_propose",
        )
        text = getattr(result, "result", None) or str(result)
        updates = getattr(result, "context_updates", {}) or {}
        log.info("Run complete. context_updates=%s",
                 json.dumps(updates, indent=2, default=str))
        print("\n=== Result ===\n")
        print(text)
        ws = updates.get("workspace_path")
        if ws:
            print(f"\n=== Workspace ===\n{ws}")
            outputs = Path(ws) / "outputs"
            if outputs.is_dir():
                print("\nOutput artifacts:")
                for child in sorted(outputs.iterdir()):
                    print(f"  {child.name}")
        return 0 if updates.get("success") else 1

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())

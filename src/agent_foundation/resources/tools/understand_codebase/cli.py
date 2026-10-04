"""Standalone CLI for the understand_codebase tool.

Usage:
    python -m agent_foundation.resources.tools.understand_codebase /path/to/code
    python -m agent_foundation.resources.tools.understand_codebase /path/to/code --model sonnet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="understand_codebase",
        description="Deep-dive codebase investigation. Produces structured documentation.",
    )
    parser.add_argument("target", help="Target file or directory to investigate")
    parser.add_argument("--model", default=None, help="Override LLM model")
    parser.add_argument(
        "--docs-only", action="store_true", help="Skip planning, generate docs directly"
    )
    parser.add_argument(
        "--investigation-only",
        action="store_true",
        help="Investigation only, skip implementation",
    )
    parser.add_argument(
        "--override",
        action="append",
        help="YAML config override as key=value (repeatable)",
    )

    args = parser.parse_args(argv)

    arguments: dict[str, Any] = {"target": args.target}
    if args.docs_only:
        arguments["docs_only"] = True
    if args.investigation_only:
        arguments["investigation_only"] = True
    if args.model:
        arguments["model"] = args.model
    if args.override:
        arguments["override"] = args.override

    session_context: dict[str, Any] = {}

    from agent_foundation.resources.tools.registry import derived_tool_execute

    tool = json.loads((Path(__file__).parent / "tool.json").read_text(encoding="utf-8"))
    result = asyncio.run(
        derived_tool_execute(
            arguments,
            session_context,
            derived_from=tool["derived_from"],
            tool_name=tool["name"],
        )
    )

    if hasattr(result, "result"):
        print(result.result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

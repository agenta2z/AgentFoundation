"""Standalone CLI for the task executor.

Usage::

    python -m agent_foundation.resources.tools.task bta "Build auth system"
    python -m agent_foundation.resources.tools.task pti "Write docs" --model sonnet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .executor import execute


def main(argv: list[str] | None = None) -> int:
    tool_json = Path(__file__).parent / "tool.json"
    meta = json.loads(tool_json.read_text())

    parser = argparse.ArgumentParser(
        prog="task",
        description=meta.get("description", "Run an agent topology."),
    )
    for param in meta.get("parameters", []):
        name = param["name"]
        if param.get("positional"):
            parser.add_argument(name, nargs="?" if not param.get("required") else None,
                                help=param.get("description", ""))
        elif param["type"] == "flag":
            parser.add_argument(name, action="store_true",
                                help=param.get("description", ""))
        elif param.get("multi"):
            parser.add_argument(name, action="append",
                                help=param.get("description", ""))
        else:
            kwargs: dict[str, Any] = {"help": param.get("description", "")}
            if "default" in param:
                kwargs["default"] = param["default"]
            parser.add_argument(name, **kwargs)

    args = parser.parse_args(argv)
    arguments = {k.lstrip("-").replace("-", "_"): v
                 for k, v in vars(args).items() if v is not None}

    async def _run():
        result = await execute(arguments, {})
        print(result.result if hasattr(result, "result") else str(result))
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())

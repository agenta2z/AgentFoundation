
# pyre-strict

"""Load tool.json files into ToolDefinition instances.

All file I/O is deferred to function calls — the module-level _TOOLS_DIR
is a pure Path construction with no side effects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_foundation.resources.tools.models import ToolDefinition

_TOOLS_DIR: Path = Path(__file__).parent  # common/tools/


def load_tool(name: str, base_dir: Path | None = None) -> ToolDefinition:
    """Load a single tool definition from its tool.json."""
    tool_json_path = (base_dir or _TOOLS_DIR) / name / "tool.json"
    with open(tool_json_path) as f:
        data: dict[str, Any] = json.load(f)
    tool = ToolDefinition.from_dict(data)
    tool.source_path = str(tool_json_path)
    return tool


def load_all_tools(extra_dirs: list[str | Path] | None = None) -> dict[str, ToolDefinition]:
    """Load all tool definitions from framework and optional extra directories.

    Args:
        extra_dirs: Additional directories to scan for tool.json files.
            Application-specific tools (e.g., Slack, TWG) can live outside
            the framework package and be loaded via this parameter.

    Returns:
        {name: ToolDefinition} — extra_dirs tools override framework tools
        with the same name.
    """
    tools: dict[str, ToolDefinition] = {}
    dirs_to_scan = [_TOOLS_DIR] + [Path(d) for d in (extra_dirs or [])]
    for tools_dir in dirs_to_scan:
        if not tools_dir.is_dir():
            continue
        for child in sorted(tools_dir.iterdir()):
            tool_json = child / "tool.json"
            if child.is_dir() and tool_json.exists():
                tool = load_tool(child.name, base_dir=tools_dir)
                tools[tool.name] = tool
    return tools


def load_tools_by_type(tool_type: str, extra_dirs: list[str | Path] | None = None) -> dict[str, ToolDefinition]:
    """Load tools filtered by tool_type ('Action' or 'Conversation')."""
    return {n: t for n, t in load_all_tools(extra_dirs=extra_dirs).items() if t.tool_type == tool_type}


def get_bridge_tools(extra_dirs: list[str | Path] | None = None) -> list[ToolDefinition]:
    """Return only bridge-based (long-running) tools."""
    return [t for t in load_all_tools(extra_dirs=extra_dirs).values() if t.is_bridge]


def get_tool_names(extra_dirs: list[str | Path] | None = None) -> list[str]:
    """Return all tool names including aliases."""
    names: list[str] = []
    for tool in load_all_tools(extra_dirs=extra_dirs).values():
        names.append(tool.name)
        names.extend(tool.aliases)
    return names

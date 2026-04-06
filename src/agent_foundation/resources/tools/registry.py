
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


def load_tool(name: str) -> ToolDefinition:
    """Load a single tool definition from its tool.json."""
    tool_json_path = _TOOLS_DIR / name / "tool.json"
    with open(tool_json_path) as f:
        data: dict[str, Any] = json.load(f)
    return ToolDefinition.from_dict(data)


def load_all_tools() -> dict[str, ToolDefinition]:
    """Load all tool definitions. Returns {name: ToolDefinition}."""
    tools: dict[str, ToolDefinition] = {}
    for child in sorted(_TOOLS_DIR.iterdir()):
        tool_json = child / "tool.json"
        if child.is_dir() and tool_json.exists():
            tool = load_tool(child.name)
            tools[tool.name] = tool
    return tools


def load_tools_by_type(tool_type: str) -> dict[str, ToolDefinition]:
    """Load tools filtered by tool_type ('Action' or 'Conversation')."""
    return {n: t for n, t in load_all_tools().items() if t.tool_type == tool_type}


def get_bridge_tools() -> list[ToolDefinition]:
    """Return only bridge-based (long-running) tools."""
    return [t for t in load_all_tools().values() if t.is_bridge]


def get_tool_names() -> list[str]:
    """Return all tool names including aliases."""
    names: list[str] = []
    for tool in load_all_tools().values():
        names.append(tool.name)
        names.extend(tool.aliases)
    return names

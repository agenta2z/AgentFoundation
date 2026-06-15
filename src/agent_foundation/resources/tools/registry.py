
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


def load_tool(
    name: str,
    base_dir: Path | None = None,
    _all_dirs: list[Path] | None = None,
) -> ToolDefinition:
    """Load a single tool definition from its tool.json.

    If the tool declares ``derived_from.expose_params``, the named parameters
    are inherited from the parent tool (looked up in ``_all_dirs`` or the
    framework default). Child parameters with the same name take precedence.
    """
    tool_json_path = (base_dir or _TOOLS_DIR) / name / "tool.json"
    with open(tool_json_path) as f:
        data: dict[str, Any] = json.load(f)
    tool = ToolDefinition.from_dict(data)
    tool.source_path = str(tool_json_path)

    if tool.derived_from:
        expose = tool.derived_from.get("expose_params", [])
        if expose:
            parent_name = tool.derived_from["tool"]
            parent = _resolve_parent_tool(parent_name, _all_dirs or [base_dir or _TOOLS_DIR])
            if parent:
                child_names = {p.name for p in tool.parameters}
                expose_set = set(expose)
                for p in parent.parameters:
                    if p.name in expose_set and p.name not in child_names:
                        tool.parameters.append(p)

    return tool


def _resolve_parent_tool(
    name: str, search_dirs: list[Path],
) -> ToolDefinition | None:
    """Find and load a parent tool by name across search directories."""
    for d in search_dirs:
        tool_json = d / name / "tool.json"
        if tool_json.exists():
            with open(tool_json) as f:
                data: dict[str, Any] = json.load(f)
            return ToolDefinition.from_dict(data)
    return None


async def derived_tool_execute(
    arguments: dict[str, Any],
    session_context: dict[str, Any],
    *,
    derived_from: dict[str, Any],
    tool_name: str,
) -> Any:
    """Generic executor for any tool with ``derived_from`` — no per-tool code.

    Reads ``arg_mappings``, ``defaults``, ``target_path_arg`` from the
    ``derived_from`` declaration and delegates to the parent tool's executor.
    """
    from agent_foundation.resources.tools.task.executor import execute as task_execute

    arg_mappings = derived_from.get("arg_mappings", {})
    defaults = derived_from.get("defaults", {})
    target_path_arg = derived_from.get("target_path_arg")

    task_args: dict[str, Any] = {}
    for key, value in arguments.items():
        if value is None:
            continue
        normalized = key.lstrip("-").replace("-", "_")
        task_args[arg_mappings.get(normalized, normalized)] = value

    if target_path_arg:
        mapped_name = arg_mappings.get(target_path_arg, target_path_arg)
        target = task_args.pop(mapped_name, "")
        target_path = Path(target).resolve() if target else Path.cwd()
        overrides = list(task_args.pop("override", []) or [])
        overrides.append(f"_target_path={target_path}")
        task_args["request"] = task_args.get("request", "")
        task_args["override"] = overrides

    for k, v in defaults.items():
        task_args.setdefault(k, v)

    feed_mappings = derived_from.get("feed_mappings", {})
    if feed_mappings:
        config_overrides = task_args.setdefault("config_overrides", {})
        if not isinstance(config_overrides, dict):
            config_overrides = {}
            task_args["config_overrides"] = config_overrides
        for arg_name, override_path in feed_mappings.items():
            val = task_args.pop(arg_name, None)
            if val is not None:
                config_overrides[override_path] = val

    ctx = dict(session_context)
    ctx["tool_name"] = tool_name

    result = await task_execute(task_args, ctx)

    # Bridge tools all delegate to ``task_execute``, which emits the generic
    # ``workspace_path`` context-update. On a flat ``prior_context`` the last
    # bridge call wins, so a multi-bridge SOP (e.g. research_propose in Phase 3
    # then task in Phase 4) would clobber the earlier workspace. Publish an
    # additional, collision-free, tool-name-suffixed key so each workspace stays
    # addressable from the SOP body via Jinja, e.g.
    #   {{ workspace_path__research_propose }}/outputs/proposals.json
    #
    # Duck-typed (``hasattr``/``isinstance dict``) to mirror the consumer in
    # conversational_inferencer (``if hasattr(result, "context_updates")``) and
    # to avoid importing ToolExecutionResult into this low-level tool-registry
    # module. Strictly additive: the generic ``workspace_path`` is untouched.
    updates = getattr(result, "context_updates", None)
    if isinstance(updates, dict):
        workspace = updates.get("workspace_path")
        if workspace:
            # Canonicalise to a valid Jinja identifier. Callers already pass a
            # canonical (underscored) tool name today via _resolve_tool_name;
            # this guards any future caller that passes the hyphenated alias.
            safe_name = tool_name.replace("-", "_")
            updates[f"workspace_path__{safe_name}"] = workspace

    return result


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
                tool = load_tool(child.name, base_dir=tools_dir, _all_dirs=dirs_to_scan)
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

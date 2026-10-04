"""JSON Schemas and names for AgentFoundation tools exposed over MCP.

* Action tools convert from ``ToolDefinition.parameters`` (``ParameterDef``).
  Property names are the keys tool executors already read
  (``--workflow-target-path`` → ``workflow_target_path``), made valid for
  vendor schemas; ``argument_names`` maps them back, and a call's arguments
  reach the executor through ``canonical_arguments`` / ``executor_arguments``.
* Conversation tools (widgets) use hand-written schemas: their ``tool.json``
  types ``choices`` as a plain string, while the widget runtime consumes rich
  choice objects, ``output`` bindings and dashboard flags.
* SOP-control tools mirror the ``SOPController`` commands.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition

MCP_SERVER_NAME = "af"
MCP_PREFIX = f"mcp__{MCP_SERVER_NAME}__"
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_LOCAL_NAME = 64 - len(MCP_PREFIX)

_PROPERTY_INVALID = re.compile(r"[^a-zA-Z0-9_.-]")
_MAX_PROPERTY = 64

_TYPE_MAP = {"string": "string", "path": "string", "int": "integer", "flag": "boolean"}
_SUBCOMMAND = "subcommand"


def executor_key(param_name: str) -> str:
    """The key a tool executor reads a parameter under — the hosts'
    argument convention (``--workflow-target-path`` → ``workflow_target_path``)."""
    return param_name.lstrip("-").replace("-", "_")


def property_name(param_name: str) -> str:
    """A parameter's MCP argument name: its executor key, made valid for
    vendor tool schemas (property names match ``^[a-zA-Z0-9_.-]{1,64}$``)."""
    key = executor_key(param_name)
    name = _PROPERTY_INVALID.sub("_", key)
    if len(name) > _MAX_PROPERTY:
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        name = f"{name[: _MAX_PROPERTY - 9]}_{digest}"
    if not name:
        raise ValueError(f"Cannot build an argument name from {param_name!r}")
    return name


def argument_names(tool: ToolDefinition) -> dict[str, str]:
    """The reverse name map of an action tool: MCP argument name → executor
    key of the parameter (or the subcommand selector) it stands for."""
    params = [*tool.parameters, *(p for s in tool.subcommands for p in s.parameters)]
    names = {property_name(p.name): executor_key(p.name) for p in params}
    if tool.subcommands:
        names[_SUBCOMMAND] = _SUBCOMMAND
    return names


def canonical_arguments(tool: ToolDefinition, args: dict[str, Any]) -> dict[str, Any]:
    """A call's arguments under their MCP argument names, the names its schema
    validates: a parameter may also arrive under its CLI spelling
    (``--docs-path``, ``docs-path``). Arguments the tool does not declare are
    kept (executors may read keys their ``tool.json`` omits). Raises
    ``ValueError`` when two arguments name the same parameter."""
    names = argument_names(tool)
    canonical: dict[str, Any] = {}
    for name, value in args.items():
        key = name
        if name not in names:
            alias = property_name(name)
            key = alias if alias in names else name
        if key in canonical:
            raise ValueError(f"Argument {key!r} is given more than once")
        canonical[key] = value
    return canonical


def executor_arguments(tool: ToolDefinition, args: dict[str, Any]) -> dict[str, Any]:
    """Canonical call arguments (``canonical_arguments``) keyed the way the
    tool executor reads them: declared ones through the reverse name map,
    undeclared ones by the hosts' convention. Raises ``ValueError`` when two
    arguments map to the same key."""
    names = argument_names(tool)
    keyed: dict[str, Any] = {}
    for name, value in args.items():
        key = names.get(name) or executor_key(name)
        if key in keyed:
            raise ValueError(f"Argument {key!r} is given more than once")
        keyed[key] = value
    return keyed


def mcp_tool_name(local_name: str) -> str:
    """Valid MCP tool name for ``local_name`` (the vendor sees ``mcp__af__<it>``)."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", local_name)
    if len(name) > _MAX_LOCAL_NAME:
        digest = hashlib.sha256(local_name.encode()).hexdigest()[:8]
        name = f"{name[: _MAX_LOCAL_NAME - 9]}_{digest}"
    if not _NAME_RE.match(MCP_PREFIX + name):
        raise ValueError(f"Cannot build a valid MCP tool name from {local_name!r}")
    return name


def assert_unique(names: Iterable[str]) -> None:
    seen: set[str] = set()
    for n in names:
        if n in seen:
            raise ValueError(f"Duplicate AF tool name {n!r} on the MCP surface")
        seen.add(n)


def _param_schema(param: ParameterDef) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": _TYPE_MAP.get(param.type, "string")}
    desc = param.description or ""
    if param.type == "path":
        desc = (desc + " (absolute path)").strip()
    if desc:
        schema["description"] = desc
    if param.choices:
        schema["enum"] = list(param.choices)
    if param.default is not None:
        schema["default"] = param.default
    return schema


def parameter_defs_to_json_schema(params: list[ParameterDef]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    for param in params:
        key = property_name(param.name)
        if key in props:
            raise ValueError(f"Parameters collide on property {key!r}")
        props[key] = _param_schema(param)
        if param.required:
            required.append(key)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def action_tool_schema(tool: ToolDefinition) -> dict[str, Any]:
    """Schema for an action tool.

    Subcommands flatten behind a required ``subcommand`` enum; their
    parameters are all optional (JSON Schema cannot require a property per
    enum value), and each says which subcommands take or require it.

    A property is one executor argument, so every declaration of a name —
    by the tool or by any subcommand — must be the same parameter (type,
    choices and default). Such a parameter is shared: a tool-level one
    applies to every subcommand, a subcommand one lists the subcommands that
    take it. Any other collision raises ``ValueError``, as does a parameter
    named like the ``subcommand`` selector.
    """
    schema = parameter_defs_to_json_schema(tool.parameters)
    if not tool.subcommands:
        return schema
    props = schema["properties"]
    tool_params = {property_name(p.name): p for p in tool.parameters}
    uses: dict[str, list[tuple[str, ParameterDef]]] = {}
    for sub in tool.subcommands:
        keys = [property_name(p.name) for p in sub.parameters]
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        if repeated:
            raise ValueError(
                f"{tool.name} {sub.name}: parameters collide on {repeated!r}"
            )
        for key, param in zip(keys, sub.parameters):
            uses.setdefault(key, []).append((sub.name, param))
    if _SUBCOMMAND in tool_params or _SUBCOMMAND in uses:
        raise ValueError(
            f"{tool.name}: a parameter named {_SUBCOMMAND!r} would collide with "
            "the subcommand selector"
        )
    for key, declared in uses.items():
        reference = tool_params.get(key) or declared[0][1]
        for sub_name, param in declared:
            if _identity(param) != _identity(reference):
                raise ValueError(
                    f"{tool.name}: parameter {key!r} of subcommand {sub_name!r} "
                    "differs from another declaration of it (type, choices or "
                    "default); give one of them another name"
                )
        if key not in tool_params:
            props[key] = _subcommand_param_schema(declared)
    props[_SUBCOMMAND] = {
        "type": "string",
        "enum": [s.name for s in tool.subcommands],
        "description": "Which operation to run.",
    }
    schema.setdefault("required", []).insert(0, _SUBCOMMAND)
    return schema


def _identity(param: ParameterDef) -> tuple:
    return param.type, tuple(param.choices or ()), param.default


def _subcommand_param_schema(declared: list[tuple[str, ParameterDef]]) -> dict:
    schema = _param_schema(declared[0][1])
    texts = [(sub, (p.description or "").strip()) for sub, p in declared]
    distinct = {text for _, text in texts if text}
    if len(distinct) > 1:
        desc = "; ".join(f"{sub}: {text.rstrip('.')}" for sub, text in texts if text)
        desc += "."
    else:
        desc = next(iter(distinct), "")
    if declared[0][1].type == "path":
        desc = (desc + " (absolute path)").strip()
    subs = [sub for sub, _ in declared]
    required = [sub for sub, p in declared if p.required]
    note = f"subcommand{'s' if len(subs) > 1 else ''}: {', '.join(subs)}"
    if required == subs:
        note += "; required"
    elif required:
        note += f"; required by: {', '.join(required)}"
    schema["description"] = f"{desc} ({note})".strip()
    return schema


_OUTPUT = {
    "type": "array",
    "items": {"type": "string"},
    "description": (
        "Variable name(s) that receive the user's answer (as instructed by the SOP)."
    ),
}
_PARALLEL_GROUP = {
    "type": "integer",
    "description": "Questions with the same group are shown together in one form.",
}
_THEN_RUN = {
    "type": "object",
    "description": (
        "Optional AF action tool to run right after the user answers (e.g. after "
        "confirming). Arguments may reference answers as __<output var>__."
    ),
    "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
    "required": ["name"],
}
_CHOICE = {
    "anyOf": [
        {"type": "string"},
        {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "value": {"type": "string"},
                "description": {"type": "string"},
                "input": {
                    "type": "object",
                    "description": "Optional typed input shown when this choice is picked.",
                },
            },
            "required": ["label"],
        },
    ]
}


def _widget(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    base = {
        "prompt": {"type": "string", "description": "The question shown to the user."},
        "output": _OUTPUT,
        "parallel_group": _PARALLEL_GROUP,
        "then_run": _THEN_RUN,
        "default": {"type": "string", "description": "Prefilled answer."},
    }
    base.update(props)
    return {"type": "object", "properties": base, "required": ["prompt", *required]}


WIDGET_SCHEMAS: dict[str, dict[str, Any]] = {
    "clarification": _widget(
        {
            "expected_input_type": {
                "type": "string",
                "description": "free_text (default), path, or url.",
            },
            "prefix": {"type": "string"},
            "allow_multiple_input": {"type": "boolean"},
            "serialization": {"type": "string"},
        },
        [],
    ),
    "single_choice": _widget(
        {
            "choices": {"type": "array", "items": _CHOICE},
            "allow_custom": {"type": "boolean"},
        },
        ["choices"],
    ),
    "multiple_choice": _widget(
        {
            "choices": {"type": "array", "items": _CHOICE},
            "allow_custom": {"type": "boolean"},
        },
        ["choices"],
    ),
    "confirmation": _widget(
        {
            "view": {"type": "string", "description": "File to show for review."},
            "yes_label": {"type": "string"},
            "no_label": {"type": "string"},
        },
        [],
    ),
    "proposal_selection": _widget(
        {
            "proposals_path": {"type": "string", "description": "proposals.json path."},
            "preselected_ids": {"type": "array", "items": {"type": "string"}},
            "allow_zero": {"type": "boolean"},
            "experiment_hub": {
                "type": "boolean",
                "description": "Open the selection in the Experiment Hub dashboard.",
            },
            "host_dashboard": {"type": "string"},
        },
        ["proposals_path"],
    ),
}

TOOL_ARGUMENT_FORM = "tool_argument_form"
# Offered only when a host enables it: it has no tool.json. Its handler shows
# one free-text input, and the conversation-tool parser keeps no
# ``tool_name``/``fields``, so the schema carries only what reaches the widget.
TOOL_ARGUMENT_FORM_SCHEMA: tuple[str, dict[str, Any]] = (
    "Ask the user for the value a tool call needs, as one free-text answer "
    "bound to `output`. To run the tool with it, name the tool in `then_run` "
    "and reference the answer as __<output var>__ in its arguments.",
    _widget({}, []),
)

SOP_COMMAND_SCHEMAS: dict[str, tuple[str, dict[str, Any]]] = {
    "enter_sop": (
        "Enter a Standard Operating Procedure (see the SOP catalog). Do not collect "
        "the SOP's parameters first; it gathers its own inputs.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "request": {
                    "type": "string",
                    "description": "The user's request to start the SOP on.",
                },
                "yolo": {
                    "type": "boolean",
                    "description": "Run autonomously; only if the user explicitly asked.",
                },
                "fresh": {
                    "type": "boolean",
                    "description": "Start over even if this SOP is in progress.",
                },
            },
            "required": ["name"],
        },
    ),
    "resume_sop": (
        "Resume a paused or exited SOP (most recent when no name is given).",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "request": {"type": "string"}},
        },
    ),
    "pause_sop": (
        "Pause the active SOP for a short ad-hoc diversion.",
        {"type": "object", "properties": {}},
    ),
    "exit_sop": (
        "Exit the active SOP (resumable later).",
        {"type": "object", "properties": {}},
    ),
    "sop_status": (
        "Show the active SOP's status and the suspended SOPs.",
        {"type": "object", "properties": {}},
    ),
}

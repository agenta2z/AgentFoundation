
# pyre-strict

"""Dataclasses for tool definitions — ToolDefinition, ParameterDef, SubcommandDef.

Follows the to_dict()/from_dict() serialization pattern used by AppConfig and
ChatMessage throughout the rankevolve server codebase.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParameterDef:
    """Definition of a single tool parameter."""

    name: str
    type: str = "string"  # "string" | "int" | "flag" | "path"
    description: str = ""
    required: bool = False
    positional: bool = False
    default: str | int | bool | None = None
    choices: list[str] | None = None
    widget: str | None = None  # Widget type hint for interactive UIs
    enable_widget: bool = False  # Whether to use widget-based input collection
    popular: bool = False  # Show in "Common" tab of confirmation config UI

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.description:
            d["description"] = self.description
        if self.required:
            d["required"] = True
        if self.positional:
            d["positional"] = True
        if self.default is not None:
            d["default"] = self.default
        if self.choices is not None:
            d["choices"] = self.choices
        if self.widget is not None:
            d["widget"] = self.widget
        if self.enable_widget:
            d["enable_widget"] = True
        if self.popular:
            d["popular"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterDef:
        return cls(
            name=data["name"],
            type=data.get("type", "string"),
            description=data.get("description", ""),
            required=data.get("required", False),
            positional=data.get("positional", False),
            default=data.get("default"),
            choices=data.get("choices"),
            widget=data.get("widget"),
            enable_widget=data.get("enable_widget", False),
            popular=data.get("popular", False),
        )


@dataclass
class SubcommandDef:
    """Definition of a tool subcommand (e.g., /kn add, /kn search)."""

    name: str
    description: str = ""
    parameters: list[ParameterDef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.parameters:
            d["parameters"] = [p.to_dict() for p in self.parameters]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubcommandDef:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=[
                ParameterDef.from_dict(p) for p in data.get("parameters", [])
            ],
        )


@dataclass
class ToolDefinition:
    """Complete definition of a tool/command."""

    name: str
    description: str = ""
    tool_type: str = "Action"  # "Action" | "Conversation"
    category: str = "utility"  # "workflow" | "knowledge" | "session" | "utility" | "conversation"
    aliases: list[str] = field(default_factory=list)
    is_bridge: bool = False
    asynchronous: bool = False  # Fire-and-forget: tool runs in background, turn completes immediately
    parameters: list[ParameterDef] = field(default_factory=list)
    subcommands: list[SubcommandDef] = field(default_factory=list)
    returns: str = ""
    examples: list[str] = field(default_factory=list)
    usage_guidance: str = ""  # When to use this tool (rendered in prompt)
    agent_enabled: bool = True  # If False, tool is user-only (not rendered in LLM prompt)
    # Derived tool metadata — declares this tool is a specialization of
    # another tool with argument translation and a template_version override.
    # Used by tool_executor for dispatch and by prompt rendering for docs.
    derived_from: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "tool_type": self.tool_type,
            "category": self.category,
        }
        if self.description:
            d["description"] = self.description
        if self.aliases:
            d["aliases"] = self.aliases
        if self.is_bridge:
            d["is_bridge"] = True
        if self.asynchronous:
            d["asynchronous"] = True
        if self.parameters:
            d["parameters"] = [p.to_dict() for p in self.parameters]
        if self.subcommands:
            d["subcommands"] = [s.to_dict() for s in self.subcommands]
        if self.returns:
            d["returns"] = self.returns
        if self.examples:
            d["examples"] = self.examples
        if self.usage_guidance:
            d["usage_guidance"] = self.usage_guidance
        if not self.agent_enabled:
            d["agent_enabled"] = False
        if self.derived_from:
            d["derived_from"] = self.derived_from
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDefinition:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            tool_type=data.get("tool_type", "Action"),
            category=data.get("category", "utility"),
            aliases=data.get("aliases", []),
            is_bridge=data.get("is_bridge", False),
            asynchronous=data.get("asynchronous", False),
            parameters=[
                ParameterDef.from_dict(p) for p in data.get("parameters", [])
            ],
            subcommands=[
                SubcommandDef.from_dict(s) for s in data.get("subcommands", [])
            ],
            returns=data.get("returns", ""),
            examples=data.get("examples", []),
            usage_guidance=data.get("usage_guidance", ""),
            agent_enabled=data.get("agent_enabled", True),
            derived_from=data.get("derived_from"),
        )

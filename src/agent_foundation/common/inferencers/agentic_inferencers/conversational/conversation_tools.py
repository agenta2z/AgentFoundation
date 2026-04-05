# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Conversation tool data models.

Defines the structured types for conversation tools that the LLM can invoke
to interact with the user: clarification, single/multiple choice, confirmation,
and tool argument collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Conversation tool type constants
class ConversationToolType:
    CLARIFICATION = "clarification"
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    CONFIRMATION = "confirmation"
    TOOL_ARGUMENT_FORM = "tool_argument_form"


@dataclass
class ChoiceItem:
    """A single choice option for single/multiple choice tools."""

    label: str
    value: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"label": self.label, "value": self.value}
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChoiceItem:
        return cls(
            label=data.get("label", ""),
            value=data.get("value", ""),
            description=data.get("description", ""),
        )


@dataclass
class ConversationTool:
    """A conversation tool invocation parsed from the LLM response.

    Represents the LLM's request to interact with the user in a structured
    way (ask a question, present choices, collect form input, etc.).
    """

    tool_type: str  # One of ConversationToolType constants
    prompt: str = ""
    choices: list[ChoiceItem] = field(default_factory=list)
    allow_custom: bool = True
    expected_input_type: str = "free_text"  # "free_text" or "path"
    prefix: str = ""  # Path prefix for path input mode
    tool_name: str = ""  # For tool_argument_form: which tool
    fields: list[dict[str, Any]] = field(default_factory=list)  # For tool_argument_form
    output_vars: list[str] = field(default_factory=list)  # Variable names to capture
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tool_type": self.tool_type}
        if self.prompt:
            d["prompt"] = self.prompt
        if self.choices:
            d["choices"] = [c.to_dict() for c in self.choices]
        if not self.allow_custom:
            d["allow_custom"] = False
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.fields:
            d["fields"] = self.fields
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTool:
        choices = [
            ChoiceItem.from_dict(c) for c in data.get("choices", [])
        ]
        return cls(
            tool_type=data.get("tool_type", ""),
            prompt=data.get("prompt", ""),
            choices=choices,
            allow_custom=data.get("allow_custom", True),
            expected_input_type=data.get("expected_input_type", "free_text"),
            prefix=data.get("prefix", ""),
            tool_name=data.get("tool_name", ""),
            fields=data.get("fields", []),
            output_vars=data.get("output_vars", data.get("output", [])),
            metadata=data.get("metadata", {}),
        )

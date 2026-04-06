

"""Widget message protocol for interactive UI elements.

Defines the data structures for sending widget requests to the frontend
and receiving structured responses. Extends (not replaces) InputModeConfig
from input_modes.py — widgets are a richer presentation layer on top of
the existing input mode protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_foundation.ui.input_modes import InputModeConfig

# Widget type constants — plain strings to avoid Python/JS sync drift
WIDGET_TEXT_INPUT = "text_input"
WIDGET_SINGLE_CHOICE = "single_choice"
WIDGET_MULTIPLE_CHOICE = "multiple_choice"
WIDGET_DROPDOWN = "dropdown"
WIDGET_TOGGLE = "toggle"
WIDGET_TOOL_ARGUMENT_FORM = "tool_argument_form"


@dataclass
class WidgetField:
    """A single field within a compound widget (e.g., tool argument form)."""

    name: str
    label: str
    widget_type: str = WIDGET_TEXT_INPUT
    description: str = ""
    required: bool = False
    default: Any = None
    choices: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "widget_type": self.widget_type,
        }
        if self.description:
            d["description"] = self.description
        if self.required:
            d["required"] = True
        if self.default is not None:
            d["default"] = self.default
        if self.choices is not None:
            d["choices"] = self.choices
        return d


@dataclass
class WidgetMessage:
    """A widget request sent from server to frontend.

    Wraps an InputModeConfig with additional widget-specific configuration
    (title, description, fields for compound widgets, etc.).
    """

    widget_id: str
    widget_type: str
    title: str = ""
    description: str = ""
    input_mode: InputModeConfig | None = None
    fields: list[WidgetField] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type,
        }
        if self.title:
            d["title"] = self.title
        if self.description:
            d["description"] = self.description
        if self.input_mode is not None:
            d["input_mode"] = self.input_mode.to_dict()
        if self.fields:
            d["fields"] = [f.to_dict() for f in self.fields]
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class WidgetResponse:
    """A structured response from the frontend for a widget interaction."""

    widget_id: str
    values: dict[str, Any] = field(default_factory=dict)
    action: str = "submit"  # "submit" | "cancel" | "skip"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WidgetResponse:
        return cls(
            widget_id=data.get("widget_id", ""),
            values=data.get("values", {}),
            action=data.get("action", "submit"),
        )

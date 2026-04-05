# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tool input collector — handles __human_input__ sentinel detection and collection.

When the LLM invokes a tool with __human_input__ as a parameter value,
this module detects it and orchestrates input collection from the user
via either widgets (if supported) or conversation prompts (fallback).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from agent_foundation.ui.input_modes import (
    InputMode,
    InputModeConfig,
    ChoiceOption,
    single_choice,
)
from agent_foundation.ui.interactive_base import (
    InteractionFlags,
    InteractiveBase,
)
# TODO: widget_protocol module does not exist at agent_foundation.ui — needs separate migration
from agent_foundation.ui.widget_protocol import (
    WIDGET_DROPDOWN,
    WIDGET_TEXT_INPUT,
    WIDGET_TOOL_ARGUMENT_FORM,
    WidgetField,
    WidgetMessage,
    WidgetResponse,
)
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition

logger = logging.getLogger(__name__)

HUMAN_INPUT_SENTINEL = "__human_input__"


def has_human_input_sentinel(arguments: dict[str, Any]) -> bool:
    """Check if any argument value is the __human_input__ sentinel."""
    return any(
        isinstance(v, str) and v == HUMAN_INPUT_SENTINEL
        for v in arguments.values()
    )


def get_human_input_params(
    arguments: dict[str, Any],
    tool_def: Optional[ToolDefinition] = None,
) -> list[tuple[str, Optional[ParameterDef]]]:
    """Get parameter names that have __human_input__ sentinel values.

    Returns list of (param_name, param_def_or_none) tuples.
    """
    params: list[tuple[str, Optional[ParameterDef]]] = []
    param_lookup: dict[str, ParameterDef] = {}
    if tool_def:
        param_lookup = {p.name: p for p in tool_def.parameters}

    for key, value in arguments.items():
        if isinstance(value, str) and value == HUMAN_INPUT_SENTINEL:
            params.append((key, param_lookup.get(key)))

    return params


async def collect_human_inputs(
    arguments: dict[str, Any],
    tool_def: Optional[ToolDefinition],
    interactive: InteractiveBase,
) -> dict[str, Any]:
    """Collect values for all __human_input__ sentinel parameters.

    Uses widgets if the interactive transport supports them and the
    parameter has enable_widget=True. Falls back to text prompts.

    Returns a new arguments dict with sentinel values replaced by user input.
    """
    params = get_human_input_params(arguments, tool_def)
    if not params:
        return arguments

    result = dict(arguments)
    supports_widgets = getattr(interactive, "supports_widgets", False)

    # If multiple params need input and widgets are supported, try a compound form
    if supports_widgets and len(params) > 1:
        widget_fields = []
        for param_name, param_def in params:
            wf = WidgetField(
                name=param_name,
                label=param_def.description or param_name if param_def else param_name,
                widget_type=(
                    param_def.widget or _infer_widget_type(param_def)
                    if param_def
                    else WIDGET_TEXT_INPUT
                ),
                description=param_def.description if param_def else "",
                required=param_def.required if param_def else True,
                default=param_def.default if param_def else None,
                choices=param_def.choices if param_def else None,
            )
            widget_fields.append(wf)

        widget_msg = WidgetMessage(
            widget_id=f"tool-input-{uuid.uuid4().hex[:8]}",
            widget_type=WIDGET_TOOL_ARGUMENT_FORM,
            title=f"Input required for {tool_def.name}" if tool_def else "Input required",
            description=tool_def.description if tool_def else "",
            fields=widget_fields,
        )
        response = await interactive.send_widget(widget_msg)
        if response.action == "submit":
            for param_name, _ in params:
                if param_name in response.values:
                    result[param_name] = response.values[param_name]
        return result

    # Single param or no widget support — collect one by one
    for param_name, param_def in params:
        if supports_widgets and param_def and param_def.enable_widget:
            value = await _collect_via_widget(param_name, param_def, interactive)
        else:
            value = await _collect_via_conversation(
                param_name, param_def, interactive
            )
        result[param_name] = value

    return result


async def _collect_via_widget(
    param_name: str,
    param_def: ParameterDef,
    interactive: Any,
) -> str:
    """Collect a single parameter via a widget."""
    widget_type = param_def.widget or _infer_widget_type(param_def)

    widget_msg = WidgetMessage(
        widget_id=f"input-{param_name}-{uuid.uuid4().hex[:8]}",
        widget_type=widget_type,
        title=param_def.description or param_name,
    )

    if param_def.choices:
        input_mode = single_choice(
            [ChoiceOption(label=c, value=c) for c in param_def.choices],
            allow_custom=True,
            prompt=param_def.description or f"Select {param_name}:",
        )
        widget_msg.input_mode = input_mode

    response = await interactive.send_widget(widget_msg)
    if response.action == "submit" and response.values:
        return next(iter(response.values.values()), "")
    return str(param_def.default) if param_def.default is not None else ""


async def _collect_via_conversation(
    param_name: str,
    param_def: Optional[ParameterDef],
    interactive: InteractiveBase,
) -> str:
    """Collect a single parameter via text conversation prompt."""
    prompt = f"Please provide a value for '{param_name}'"
    if param_def and param_def.description:
        prompt = f"{param_def.description}"
    if param_def and param_def.choices:
        prompt += f"\nOptions: {', '.join(param_def.choices)}"
    if param_def and param_def.default is not None:
        prompt += f"\n(default: {param_def.default})"

    # Use InputModeConfig if choices are available
    if param_def and param_def.choices:
        input_mode = single_choice(
            [ChoiceOption(label=c, value=c) for c in param_def.choices],
            allow_custom=True,
            prompt=prompt,
        )
        await interactive.asend_response(
            prompt,
            flag=InteractionFlags.PendingInput,
            input_mode=input_mode,
        )
    else:
        input_mode = InputModeConfig(
            mode=InputMode.FREE_TEXT,
            prompt=prompt,
        )
        await interactive.asend_response(
            prompt,
            flag=InteractionFlags.PendingInput,
            input_mode=input_mode,
        )

    response = await interactive.aget_input()
    if response is None:
        return param_def.default if param_def and param_def.default is not None else ""

    # Extract string value from possible dict wrapper
    if isinstance(response, dict):
        return str(response.get("user_input", response.get("content", "")))
    return str(response)


def _infer_widget_type(param_def: ParameterDef) -> str:
    """Infer the best widget type from a ParameterDef."""
    if param_def.choices:
        return WIDGET_DROPDOWN
    return WIDGET_TEXT_INPUT

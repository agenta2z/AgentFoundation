# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""MultipleChoiceHandler — multi-choice widget; publishes the joined choices to
``tool.output_vars`` via ``PublishSessionVariablesEffect`` (tool_type
namespacing), like the single-choice and clarification widgets."""

from __future__ import annotations

from typing import Any, ClassVar

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    PublishSessionVariablesEffect,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    ConversationToolHandler,
    HandlerContext,
    HandlerResult,
    InferencerEffect,
)
from agent_foundation.common.ui.input_modes import (
    ChoiceOption,
    InputModeConfig,
    multiple_choices,
)


class MultipleChoiceHandler(ConversationToolHandler):
    tool_type: ClassVar[ConversationToolType] = ConversationToolType.MULTIPLE_CHOICE

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig:
        # Mirrors inline `_choice_option_from` — preserves description AND
        # serialized composite `input` spec.
        options = [
            ChoiceOption(
                label=c.label,
                value=c.value,
                description=getattr(c, "description", "") or "",
                input=c.input.to_dict()
                if getattr(c, "has_input", False) and c.input is not None
                else None,
            )
            for c in tool.choices
        ]
        return multiple_choices(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
            show_select_all=tool.show_select_all,
            select_all_text=tool.select_all_text,
        )

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult:
        if isinstance(response, dict):
            selections = response.get("selections")
            if isinstance(selections, list):
                text = ",".join(_selected_values(tool, selections))
            else:
                text = response.get("content") or response.get("custom_text") or ""
        else:
            text = str(response)

        effects: list[InferencerEffect] = []
        if text and tool.output_vars:
            effects.append(
                PublishSessionVariablesEffect(
                    {v: text for v in tool.output_vars},
                    tool_type=tool.tool_type,
                )
            )
        return HandlerResult(text=text, effects=effects)


def _selected_values(tool: ConversationTool, selections: list[Any]) -> list[str]:
    """The values of a UI answer's ``selections`` (``{"choice_index": i}`` or
    ``{"custom_text": s}`` items), as the compound decoder reads them."""
    values: list[str] = []
    for s in selections:
        if not isinstance(s, dict):
            continue
        idx = s.get("choice_index")
        if isinstance(idx, int) and 0 <= idx < len(tool.choices):
            values.append(tool.choices[idx].value)
        elif "custom_text" in s:
            values.append(str(s["custom_text"]))
    return values

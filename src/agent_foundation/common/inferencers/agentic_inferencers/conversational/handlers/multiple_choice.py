# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""MultipleChoiceHandler — multi-choice widget; binds joined choices to output_vars[0]."""

from __future__ import annotations

from typing import Any, ClassVar

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    SetPromptVariable,
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
        options = [
            ChoiceOption(label=c.label, value=c.value, description=c.description)
            for c in tool.choices
        ]
        return multiple_choices(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
        )

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult:
        if isinstance(response, dict):
            text = (
                response.get("content")
                or response.get("custom_text")
                or ""
            )
        else:
            text = str(response)

        effects: list[InferencerEffect] = []
        if tool.output_vars:
            effects.append(SetPromptVariable(tool.output_vars[0], text))
        return HandlerResult(text=text, effects=effects)

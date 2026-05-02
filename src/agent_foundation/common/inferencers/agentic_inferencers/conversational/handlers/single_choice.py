# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""SingleChoiceHandler — single-choice widget; binds choice to output_vars[0]."""

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
    single_choice,
)


class SingleChoiceHandler(ConversationToolHandler):
    tool_type: ClassVar[ConversationToolType] = ConversationToolType.SINGLE_CHOICE

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig:
        options = [
            ChoiceOption(label=c.label, value=c.value, description=c.description)
            for c in tool.choices
        ]
        return single_choice(
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
        # Extract scalar value from widget response
        if isinstance(response, dict):
            choice_value = (
                response.get("content")
                or response.get("custom_text")
                or ""
            )
            if not choice_value and "choice_index" in response:
                idx = response["choice_index"]
                if 0 <= idx < len(tool.choices):
                    choice_value = tool.choices[idx].value
        else:
            choice_value = str(response)

        effects: list[InferencerEffect] = []
        if tool.output_vars:
            effects.append(SetPromptVariable(tool.output_vars[0], str(choice_value)))
        return HandlerResult(text=str(choice_value), effects=effects)

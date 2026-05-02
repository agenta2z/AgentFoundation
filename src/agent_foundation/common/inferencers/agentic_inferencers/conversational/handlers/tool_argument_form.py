# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""ToolArgumentFormHandler — TOOL_ARGUMENT_FORM widget (additive net-new).

The enum value `ConversationToolType.TOOL_ARGUMENT_FORM` was declared but
never branched in the legacy dispatcher; it silently fell through to the
generic free-text widget. This handler ensures the registry covers all
declared types.

Today no production code path emits this type. This handler is forward-compat:
- `build_input_mode` returns the same free-text widget as the legacy fallthrough.
- `handle_response` interprets a structured response (`{"fields": {field_name: value, ...}}`)
  into `SetPromptVariable` effects per field. Falls back to a single
  `SetPromptVariable(tool.output_vars[0], scalar)` if the response is a plain
  string. Without `output_vars`, returns the text without effects.
"""

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
    InputMode,
    InputModeConfig,
)


class ToolArgumentFormHandler(ConversationToolHandler):
    tool_type: ClassVar[ConversationToolType] = ConversationToolType.TOOL_ARGUMENT_FORM

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig:
        return InputModeConfig(mode=InputMode.FREE_TEXT, prompt=tool.prompt)

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult:
        effects: list[InferencerEffect] = []
        text = ""

        if isinstance(response, dict):
            fields = response.get("fields")
            if isinstance(fields, dict):
                for k, v in fields.items():
                    effects.append(SetPromptVariable(str(k), str(v)))
                text = ", ".join(f"{k}={v}" for k, v in fields.items())
            else:
                text = response.get("content") or response.get("custom_text") or ""
                if text and tool.output_vars:
                    effects.append(SetPromptVariable(tool.output_vars[0], text))
        else:
            text = str(response)
            if text and tool.output_vars:
                effects.append(SetPromptVariable(tool.output_vars[0], text))

        return HandlerResult(text=text, effects=effects)

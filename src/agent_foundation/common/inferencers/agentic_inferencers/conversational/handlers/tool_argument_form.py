# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""ToolArgumentFormHandler — TOOL_ARGUMENT_FORM widget (additive net-new).

The enum value `ConversationToolType.TOOL_ARGUMENT_FORM` was declared but
never branched in the legacy dispatcher; it silently fell through to the
generic free-text widget. This handler ensures the registry covers all
declared types.

Today no production code path emits this type. This handler is forward-compat:
- `build_input_mode` returns the same free-text widget as the legacy fallthrough.
- `handle_response` publishes a structured response's fields
  (`{"fields": {field_name: value, ...}}`) as session variables, or a plain
  answer to `tool.output_vars`, via `PublishSessionVariablesEffect` (tool_type
  namespacing). Without fields or `output_vars`, returns the text without
  effects.
"""

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
from agent_foundation.common.ui.input_modes import InputMode, InputModeConfig


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
        variables: dict[str, str] = {}
        text = ""

        if isinstance(response, dict):
            fields = response.get("fields")
            if isinstance(fields, dict):
                variables = {str(k): str(v) for k, v in fields.items()}
                text = ", ".join(f"{k}={v}" for k, v in fields.items())
            else:
                text = response.get("content") or response.get("custom_text") or ""
        else:
            text = str(response)
        if not variables and text:
            variables = {v: text for v in tool.output_vars}

        effects: list[InferencerEffect] = []
        if variables:
            effects.append(
                PublishSessionVariablesEffect(variables, tool_type=tool.tool_type)
            )
        return HandlerResult(text=text, effects=effects)

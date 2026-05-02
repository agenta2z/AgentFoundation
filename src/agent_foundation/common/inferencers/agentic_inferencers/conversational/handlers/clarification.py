# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""ClarificationHandler — free-text widget; returns response as-is."""

from __future__ import annotations

from typing import Any, ClassVar

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    ConversationToolHandler,
    HandlerContext,
    HandlerResult,
)
from agent_foundation.common.ui.input_modes import (
    InputMode,
    InputModeConfig,
)


class ClarificationHandler(ConversationToolHandler):
    tool_type: ClassVar[ConversationToolType] = ConversationToolType.CLARIFICATION

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig:
        config = InputModeConfig(mode=InputMode.FREE_TEXT, prompt=tool.prompt)
        if tool.expected_input_type and tool.expected_input_type != "free_text":
            config.metadata = {
                "expected_input_type": tool.expected_input_type,
                "prefix": tool.prefix,
            }
        return config

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult:
        if isinstance(response, dict):
            text = response.get("content") or response.get("custom_text") or ""
        else:
            text = str(response)
        return HandlerResult(text=text)

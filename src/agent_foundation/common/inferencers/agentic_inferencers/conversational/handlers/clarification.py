# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""ClarificationHandler — free-text widget with AF typed-input finalization.

Extended beyond the RE port: when the tool declares a non-``free_text``
``expected_input_type`` (e.g. ``path``), decode via ``finalize_input_value``
(path re-join + validation + serialization) before publishing to output vars.

Publishes via ``PublishSessionVariablesEffect`` with tool_type namespacing so
the CI's ``set_session_variables`` emits both the bare and
``<tool_type>__<var>`` aliases (A1.b convention).
"""

from __future__ import annotations

from typing import Any, ClassVar

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    finalize_input_value,
)
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


class ClarificationHandler(ConversationToolHandler):
    tool_type: ClassVar[ConversationToolType] = ConversationToolType.CLARIFICATION

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig:
        # Byte-identity with inline `_build_input_mode` CLARIFICATION branch:
        # - top-level `expected_input_type` / `prefix` / `allow_multiple_input`
        #   are first-class fields on InputModeConfig (new UI reads these).
        # - the metadata block mirrors them for the legacy React path, plus
        #   `widget_type: "path_input"` for path inputs.
        # - `default` prefill from tool.metadata rides on metadata.default so
        #   the widget can pre-populate its input.
        config = InputModeConfig(
            mode=InputMode.FREE_TEXT,
            prompt=tool.prompt,
            expected_input_type=tool.expected_input_type,
            prefix=tool.prefix,
            allow_multiple_input=tool.allow_multiple_input,
        )
        if tool.expected_input_type and tool.expected_input_type != "free_text":
            metadata: dict[str, Any] = {
                "expected_input_type": tool.expected_input_type,
                "prefix": tool.prefix,
            }
            if tool.allow_multiple_input:
                metadata["allow_multiple_input"] = True
            if tool.expected_input_type == "path":
                metadata["widget_type"] = "path_input"
            config.metadata = metadata
        default_val = tool.metadata.get("default") if tool.metadata else None
        if default_val not in (None, ""):
            md = dict(config.metadata or {})
            md["default"] = default_val
            config.metadata = md
        return config

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult:
        if isinstance(response, dict):
            raw = response.get("content") or response.get("custom_text") or ""
        else:
            raw = response

        final = finalize_input_value(
            raw,
            expected_input_type=tool.expected_input_type or "free_text",
            prefix=tool.prefix or "",
            allow_multiple_input=tool.allow_multiple_input,
            serialization=tool.serialization or "auto",
            session_root=ctx.session_root or "",
        )

        effects: list[InferencerEffect] = []
        if tool.output_vars and final:
            effects.append(
                PublishSessionVariablesEffect(
                    {v: final for v in tool.output_vars},
                    tool_type=tool.tool_type,
                )
            )

        return HandlerResult(text=final, effects=effects)

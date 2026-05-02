# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Per-instance registry of conversation tool handlers.

Per-instance (not module-global) for automatic test isolation. Two lookup
methods: `get` returns Optional (used during the registry-rollout transition
when the dispatcher falls back to legacy if/elif branches), `require` raises
on missing (used post-cleanup when the registry is the sole dispatch path).
"""

from __future__ import annotations

from attr import attrib, attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    ConversationToolHandler,
)


@attrs(slots=False)
class ConversationToolHandlerRegistry:
    _handlers: dict[ConversationToolType, ConversationToolHandler] = attrib(
        factory=dict,
        init=False,
    )

    def register(self, handler: ConversationToolHandler) -> None:
        """Raises ValueError on duplicate tool_type."""
        tool_type = handler.tool_type
        if tool_type in self._handlers:
            raise ValueError(
                f"Handler already registered for {tool_type!r}: {self._handlers[tool_type]}"
            )
        self._handlers[tool_type] = handler

    def get(
        self,
        tool_type: ConversationToolType,
    ) -> ConversationToolHandler | None:
        """Optional lookup; never raises. Used during transition for fallback."""
        return self._handlers.get(tool_type)

    def require(
        self,
        tool_type: ConversationToolType,
    ) -> ConversationToolHandler:
        """Strict lookup; raises ValueError on missing handler.

        Used post-cleanup when registry is the sole dispatch path.
        """
        handler = self._handlers.get(tool_type)
        if handler is None:
            raise ValueError(
                f"No handler registered for {tool_type!r}. "
                f"Registered: {sorted(self._handlers.keys(), key=str)}"
            )
        return handler

    def list_registered(self) -> list[ConversationToolType]:
        return list(self._handlers.keys())

    def __contains__(self, tool_type: ConversationToolType) -> bool:
        return tool_type in self._handlers

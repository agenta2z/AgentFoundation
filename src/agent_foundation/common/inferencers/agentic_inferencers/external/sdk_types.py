"""Shared types for external SDK-based inferencers.

This module provides common response types and utilities used by SDK-based
inferencers (ClaudeCodeInferencer, DevmateSDKInferencer, etc.).
"""

from typing import Any, Optional

from attr import attrib, attrs


@attrs
class SDKInferencerResponse:
    """Standard response type for SDK-based inferencers.

    Provides a unified response format that captures both the inference output
    and metadata from the SDK execution.

    Attributes:
        content: The main response content (typically text).
        session_id: Optional session identifier for multi-turn conversations.
        tool_uses: Count of tool invocations during inference (Claude Code).
        tokens_received: Count of tokens streamed (Devmate SDK).
        raw_response: Optional raw SDK response for advanced use cases.
    """

    content: str = attrib(default="")
    session_id: Optional[str] = attrib(default=None)
    tool_uses: int = attrib(default=0)
    tokens_received: int = attrib(default=0)
    raw_response: Optional[Any] = attrib(default=None)

    def __str__(self) -> str:
        """Return the content as the string representation."""
        return self.content

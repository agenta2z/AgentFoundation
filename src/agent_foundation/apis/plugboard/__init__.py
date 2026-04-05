# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Plugboard API module — standard interface for Meta's internal LLM gateway."""

from agent_foundation.apis.plugboard.plugboard_llm import (
    generate_text,
    generate_text_async,
    generate_text_streaming,
)

__all__ = [
    "generate_text",
    "generate_text_async",
    "generate_text_streaming",
]

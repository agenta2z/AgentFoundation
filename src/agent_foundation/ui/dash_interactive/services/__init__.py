"""
Services module for dash_interactive.

This module provides reusable service classes that handle external integrations
such as LLM APIs, making them easily pluggable into Dash applications.
"""

from science_modeling_tools.ui.dash_interactive.services.llm_chat_service import (
    BaseLLMConfig,
    LLMChatService,
)

__all__ = [
    "LLMChatService",
    "BaseLLMConfig",
]

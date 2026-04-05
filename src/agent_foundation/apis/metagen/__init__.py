"""
MetaGen API module.

This module provides functions and classes for interacting with the MetaGen LLM API.
"""

from agent_foundation.apis.metagen.metagen_llm import (
    CompletionMode,
    generate_text,
    generate_text_async,
    generate_text_streaming,
    get_default_metagen_key,
    get_optimal_key_for_model,
    load_model_to_key_map,
    MetaGenModels,
)

__all__ = [
    "CompletionMode",
    "MetaGenModels",
    "generate_text",
    "generate_text_async",
    "generate_text_streaming",
    "load_model_to_key_map",
    "get_default_metagen_key",
    "get_optimal_key_for_model",
]

"""
MetaGen API module.

This module provides functions and classes for interacting with the MetaGen LLM API.
"""

from agent_foundation.apis.metagen.metagen_llm import (
    generate_text,
    generate_text_async,
    get_default_metagen_key,
    get_optimal_key_for_model,
    load_model_to_key_map,
    MetaGenModels,
)

__all__ = [
    "MetaGenModels",
    "generate_text",
    "generate_text_async",
    "load_model_to_key_map",
    "get_default_metagen_key",
    "get_optimal_key_for_model",
]

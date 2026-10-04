"""Backward-compatibility shim — re-exports from retrieval.utils."""

from agent_foundation.knowledge.retrieval.utils import (  # noqa: F401
    cosine_similarity,
    count_tokens,
    parse_entity_type,
    sanitize_id,
    unsanitize_id,
)

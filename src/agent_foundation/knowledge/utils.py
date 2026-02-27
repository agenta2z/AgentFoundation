"""Backward-compatibility shim — re-exports from retrieval.utils."""
from agent_foundation.knowledge.retrieval.utils import (  # noqa: F401
    sanitize_id,
    unsanitize_id,
    parse_entity_type,
    cosine_similarity,
    count_tokens,
)

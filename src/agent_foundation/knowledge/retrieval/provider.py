"""InfoType enum and default formatter for knowledge routing.

This module contains the InfoType StrEnum for well-known info type constants
and the default formatter function used by GroupedDictPostProcessor.

The KnowledgeProvider class has been removed. Use RetrievalPipeline with
GroupedDictPostProcessor instead.
"""

import logging
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple

from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece

logger = logging.getLogger(__name__)


class InfoType(StrEnum):
    """Convenience constants for well-known info types.

    The system works with any string — these are not required.
    """

    UserProfile = "user_profile"
    Instructions = "instructions"
    Context = "context"


def _default_formatter(
    metadata: Optional[EntityMetadata],
    pieces: List[Tuple[KnowledgePiece, float]],
    graph_context: List[Dict[str, Any]],
) -> str:
    """Default formatter that concatenates metadata properties and piece contents.

    Produces a simple text representation:
    - Metadata properties as "Key: Value" lines
    - Piece contents separated by newlines
    - Graph context as "RELATION → target (description)" lines

    Args:
        metadata: Optional entity metadata.
        pieces: List of (KnowledgePiece, score) tuples.
        graph_context: List of graph context dicts.

    Returns:
        Formatted string, or empty string if no data.
    """
    parts: List[str] = []

    # Format metadata properties
    if metadata and metadata.properties:
        meta_lines = []
        for key in sorted(metadata.properties.keys()):
            meta_lines.append(f"{key}: {metadata.properties[key]}")
        parts.append("\n".join(meta_lines))

    # Format pieces sorted by score descending, then piece_id
    if pieces:
        sorted_pieces = sorted(pieces, key=lambda p: (-p[1], p[0].piece_id))
        piece_lines = [piece.content for piece, _score in sorted_pieces]
        parts.append("\n".join(piece_lines))

    # Format graph context
    if graph_context:
        graph_lines = []
        sorted_ctx = sorted(
            graph_context,
            key=lambda c: (c.get("relation_type", ""), c.get("target_node_id", "")),
        )
        for ctx in sorted_ctx:
            relation = ctx.get("relation_type", "RELATED")
            target = ctx.get("target_node_id", "unknown")
            label = ctx.get("target_label", "")
            piece = ctx.get("piece")

            line = f"{relation} → {target}"
            if piece is not None and hasattr(piece, "content"):
                line += f" ({piece.content})"
            elif label:
                line += f" ({label})"
            graph_lines.append(line)
        parts.append("\n".join(graph_lines))

    return "\n\n".join(parts)

"""
KnowledgeFormatter and RetrievalResult for the Agent Knowledge Base.

KnowledgeFormatter formats a RetrievalResult into a prompt-ready string with
three sections: Metadata, Knowledge, and Relationships. Output is deterministic:
metadata keys are sorted, pieces are sorted by score then piece_id, and graph
context is sorted by relation type then target node_id.

RetrievalResult is a simple data container holding the results from all three
retrieval layers (metadata, knowledge pieces, entity graph).

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""
from typing import Any, Dict, List, Optional, Tuple

from attr import attrs, attrib

from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)


@attrs
class RetrievalResult:
    """Container for results from all three retrieval layers.

    Attributes:
        metadata: Entity-specific metadata (e.g., user profile).
        global_metadata: Global metadata shared across all entities.
        pieces: Scored knowledge pieces as (KnowledgePiece, score) tuples.
        graph_context: Graph traversal results as list of dicts with keys:
            relation_type, target_node_id, target_label, piece (optional), depth.
    """
    metadata: Optional[EntityMetadata] = attrib(default=None)
    global_metadata: Optional[EntityMetadata] = attrib(default=None)
    pieces: List[Tuple[KnowledgePiece, float]] = attrib(factory=list)
    graph_context: List[Dict[str, Any]] = attrib(factory=list)


@attrs
class KnowledgeFormatter:
    """Formats RetrievalResult into a prompt-ready string.

    The output has up to three sections:
    1. [Metadata] — key-value pairs from entity metadata
    2. [Global Metadata] — key-value pairs from global metadata (if present)
    3. [Knowledge] — relevant knowledge pieces grouped by KnowledgeType
    4. [Relationships] — graph context (related entities and their knowledge)

    Output is deterministic:
    - Metadata keys are sorted alphabetically
    - Pieces are sorted by (-score, piece_id) then grouped by KnowledgeType
    - Graph context is sorted by (relation_type, target node_id)

    Attributes:
        section_delimiter: String separating major sections.
        item_delimiter: String separating items within a section.
        include_tags: Whether to show tags below each knowledge piece.
        include_scores: Whether to show relevance scores next to pieces.
    """
    section_delimiter: str = attrib(default="\n\n")
    item_delimiter: str = attrib(default="\n---\n")
    include_tags: bool = attrib(default=True)
    include_scores: bool = attrib(default=False)

    def format(self, result: RetrievalResult) -> str:
        """Format retrieval result into prompt-ready string.

        Returns empty string for empty results (no metadata, no pieces,
        no graph_context).

        Args:
            result: The RetrievalResult to format.

        Returns:
            A formatted string ready for prompt injection, or empty string
            if the result contains no data.
        """
        sections = []

        # Section 1: Entity metadata
        if result.metadata and result.metadata.properties:
            sections.append(self._format_metadata(result.metadata, label="Metadata"))

        # Section 2: Global metadata
        if result.global_metadata and result.global_metadata.properties:
            sections.append(
                self._format_metadata(result.global_metadata, label="Global Metadata")
            )

        # Section 3: Knowledge pieces
        if result.pieces:
            sections.append(self._format_pieces(result.pieces))

        # Section 4: Relationships
        if result.graph_context:
            sections.append(self._format_graph_context(result.graph_context))

        if not sections:
            return ""

        return self.section_delimiter.join(sections)

    def _format_metadata(
        self, metadata: EntityMetadata, label: str = "Metadata"
    ) -> str:
        """Format entity metadata as sorted key-value pairs.

        Args:
            metadata: The EntityMetadata to format.
            label: Section header label (e.g., "Metadata" or "Global Metadata").

        Returns:
            Formatted string with section header and sorted key-value lines.
        """
        lines = [f"[{label}]"]
        for key in sorted(metadata.properties.keys()):
            value = metadata.properties[key]
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _format_pieces(
        self, pieces: List[Tuple[KnowledgePiece, float]]
    ) -> str:
        """Format knowledge pieces grouped by KnowledgeType.

        Pieces are first sorted by (-score, piece_id) for determinism,
        then grouped by KnowledgeType. Within each group, the sort order
        is preserved.

        Args:
            pieces: List of (KnowledgePiece, score) tuples.

        Returns:
            Formatted string with [Knowledge] header and grouped pieces.
        """
        # Sort pieces deterministically: by score descending, then piece_id ascending
        sorted_pieces = sorted(
            pieces, key=lambda p: (-p[1], p[0].piece_id)
        )

        # Group by KnowledgeType while preserving sort order
        groups: Dict[KnowledgeType, List[Tuple[KnowledgePiece, float]]] = {}
        for piece, score in sorted_pieces:
            kt = piece.knowledge_type
            if kt not in groups:
                groups[kt] = []
            groups[kt].append((piece, score))

        # Build formatted items
        formatted_items = []
        for knowledge_type in sorted(groups.keys(), key=lambda kt: kt.value):
            for piece, score in groups[knowledge_type]:
                item_lines = []
                # Main line: [type] content
                line = f"[{piece.knowledge_type.value}] {piece.content}"
                if self.include_scores:
                    line += f" (score: {score:.2f})"
                item_lines.append(line)

                # Tags line (indented)
                if self.include_tags and piece.tags:
                    item_lines.append(f"  Tags: {', '.join(sorted(piece.tags))}")

                formatted_items.append("\n".join(item_lines))

        header = "[Knowledge]"
        body = self.item_delimiter.join(formatted_items)
        return f"{header}\n{body}"

    def _format_graph_context(self, graph_context: List[Dict[str, Any]]) -> str:
        """Format graph context (relationships) as sorted entries.

        Each entry shows: RELATION_TYPE → NodeType:node_id (description)

        Graph context dicts are expected to have keys:
        - relation_type: str
        - target_node_id: str
        - target_label: str (optional, human-readable label)
        - piece: Optional[KnowledgePiece] (linked knowledge)
        - depth: int

        Sorted by (relation_type, target_node_id) for determinism.

        Args:
            graph_context: List of graph context dictionaries.

        Returns:
            Formatted string with [Relationships] header and sorted entries.
        """
        # Sort by relation_type then target_node_id for determinism
        sorted_context = sorted(
            graph_context,
            key=lambda ctx: (
                ctx.get("relation_type", ""),
                ctx.get("target_node_id", ""),
            ),
        )

        lines = ["[Relationships]"]
        for ctx in sorted_context:
            relation_type = ctx.get("relation_type", "UNKNOWN")
            target_node_id = ctx.get("target_node_id", "unknown")
            target_label = ctx.get("target_label", "")

            # Build the display name: prefer label, fall back to node_id
            # Format: RELATION_TYPE → NodeType:node_id
            # Use target_node_id which already has the type:name format
            display_target = target_node_id
            # Capitalize the node type portion for readability
            if ":" in display_target:
                parts = display_target.split(":", 1)
                display_target = f"{parts[0].capitalize()}:{parts[1]}"

            line = f"{relation_type} → {display_target}"

            # Add description from linked piece or label
            piece = ctx.get("piece")
            if piece is not None and hasattr(piece, "content"):
                line += f" ({piece.content})"
            elif target_label:
                line += f" ({target_label})"

            lines.append(line)

        return "\n".join(lines)

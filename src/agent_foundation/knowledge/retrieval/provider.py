"""KnowledgeProvider routing layer and InfoType enum.

This module contains the InfoType StrEnum for well-known info type constants
and the KnowledgeProvider class that routes KnowledgeBase retrieval results
to typed, formatted outputs for injection into agent prompt fields.
"""

import logging
import os
from enum import StrEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from agent_foundation.knowledge.retrieval.formatter import RetrievalResult
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
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


class KnowledgeProvider:
    """Routes KnowledgeBase retrieval results to typed, formatted outputs.

    Callable: (query: str) → Dict[str, str]
      1. Calls kb.retrieve(query) once
      2. Groups pieces by info_type string
      3. Routes metadata and graph edges to appropriate groups
      4. Formats each group with its configured formatter
      5. Returns Dict[str, str] keyed by info_type

    Any info_type string works — adding a new type requires zero code changes.
    Just use the string in your data and add {{ placeholder }} to the prompt template.

    Formatter resolution:
      - callable: called directly with (metadata, pieces, graph_context) -> str
      - str path ending in .j2/.jinja2/.jinja: loaded as Jinja2 template
      - str path ending in .hbs/.handlebars: loaded as Handlebars template
      - str path with other extension: loaded as file, rendered with Python .format()
      - str (not a file path): used as inline Python .format() string

    Attributes:
        kb: The underlying KnowledgeBase retrieval engine.
        formatters: Per-info-type formatter specs (keyed by string).
        default_formatter: Fallback formatter for unconfigured info types.
        metadata_info_type: Which info_type group receives metadata (default "user_profile").
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        formatters: Optional[Dict[str, Union[Callable, str]]] = None,
        default_formatter: Optional[Union[Callable, str]] = None,
        metadata_info_type: str = "user_profile",
    ):
        self.kb = kb
        self.formatters = formatters or {}
        self.metadata_info_type = metadata_info_type
        # Resolve the default formatter
        if default_formatter is not None:
            self._default_formatter = self._resolve_formatter(default_formatter)
        else:
            self._default_formatter = _default_formatter

    def __call__(self, query: str) -> Dict[str, str]:
        """Retrieve, group by info_type, format each group.

        1. Calls kb.retrieve(query) once → RetrievalResult
        2. Groups result by info_type via _group_by_info_type()
        3. Formats each group using the configured formatter (or default)
        4. Returns Dict[str, str] — keys are info_type strings, values are formatted text

        Args:
            query: The user query string.

        Returns:
            Dict mapping info_type strings to formatted knowledge strings.
            Empty groups produce empty strings.
        """
        result = self.kb.retrieve(query)
        groups = self._group_by_info_type(result)

        output: Dict[str, str] = {}
        for info_type, group_data in groups.items():
            # Resolve the formatter for this info_type
            if info_type in self.formatters:
                formatter = self._resolve_formatter(self.formatters[info_type])
            else:
                formatter = self._default_formatter

            # Call the formatter with the group data
            formatted = formatter(
                group_data["metadata"],
                group_data["pieces"],
                group_data["graph_context"],
            )
            output[info_type] = formatted

        return output

    def _group_by_info_type(self, result: RetrievalResult) -> Dict[str, dict]:
        """Group a RetrievalResult by info_type string.

        Routing rules:
        - Metadata → metadata_info_type (default "user_profile")
        - Pieces → piece.info_type (any string)
        - Graph edges with linked piece → piece.info_type
        - Graph edges from user node (no linked piece) → "user_profile"
          (uses self.kb.active_entity_id to identify user-originating edges)
        - Other graph edges (no linked piece) → "context"

        Args:
            result: The RetrievalResult from kb.retrieve().

        Returns:
            Dict mapping info_type strings to dicts with keys:
            metadata, pieces, graph_context.
        """
        groups: Dict[str, dict] = {}

        def _ensure_group(info_type: str) -> dict:
            if info_type not in groups:
                groups[info_type] = {
                    "metadata": None,
                    "pieces": [],
                    "graph_context": [],
                }
            return groups[info_type]

        # Route metadata to metadata_info_type
        if result.metadata and result.metadata.properties:
            group = _ensure_group(self.metadata_info_type)
            group["metadata"] = result.metadata

        # Route global metadata to metadata_info_type as well
        # (merge into the same group — global metadata supplements entity metadata)
        if result.global_metadata and result.global_metadata.properties:
            group = _ensure_group(self.metadata_info_type)
            if group["metadata"] is None:
                group["metadata"] = result.global_metadata

        # Route pieces by their info_type
        for piece, score in result.pieces:
            info_type = piece.info_type or "context"
            group = _ensure_group(info_type)
            group["pieces"].append((piece, score))

        # Route graph edges
        active_entity_id = self.kb.active_entity_id
        for edge in result.graph_context:
            linked_piece = edge.get("piece")

            if linked_piece is not None and hasattr(linked_piece, "info_type"):
                # Graph edge with linked piece → route to piece's info_type
                info_type = linked_piece.info_type or "context"
                group = _ensure_group(info_type)
                group["graph_context"].append(edge)
            elif edge.get("depth", 0) == 1 and active_entity_id is not None:
                # Depth-1 edge from user node (no linked piece) → "user_profile"
                # Depth-1 means direct neighbor of the active entity
                group = _ensure_group(InfoType.UserProfile)
                group["graph_context"].append(edge)
            else:
                # Other graph edges (no linked piece, deeper traversal) → "context"
                group = _ensure_group(InfoType.Context)
                group["graph_context"].append(edge)

        return groups

    def _resolve_formatter(self, spec: Union[Callable, str]) -> Callable:
        """Resolve a formatter spec to a callable.

        - callable → use directly
        - str ending in .j2/.jinja2/.jinja → Jinja2 template from file
        - str ending in .hbs/.handlebars → Handlebars template from file
        - str with other file extension → Python .format() template from file
        - str (not a file) → inline Python .format() string

        Args:
            spec: A callable or string formatter specification.

        Returns:
            A callable with signature (metadata, pieces, graph_context) -> str.

        Raises:
            ValueError: If a file-based template path does not exist.
            ImportError: If Jinja2 or pybars3 is not installed for template types.
        """
        # If it's already callable, return directly
        if callable(spec):
            return spec

        # Must be a string
        if not isinstance(spec, str):
            raise ValueError(f"Formatter spec must be callable or str, got {type(spec)}")

        # Check if it looks like a file path (has a file extension with a dot)
        _, ext = os.path.splitext(spec)

        if ext:
            # It's a file path
            if not os.path.isfile(spec):
                raise ValueError(f"Formatter template file not found: {spec}")

            if ext in (".j2", ".jinja2", ".jinja"):
                # Jinja2 template
                return self._make_jinja2_formatter(spec)
            elif ext in (".hbs", ".handlebars"):
                # Handlebars template
                return self._make_handlebars_formatter(spec)
            else:
                # Generic file-based Python .format() template
                return self._make_file_format_formatter(spec)
        else:
            # Inline Python .format() string
            return self._make_inline_format_formatter(spec)

    @staticmethod
    def _make_jinja2_formatter(template_path: str) -> Callable:
        """Create a Jinja2 template formatter from a file path.

        Args:
            template_path: Path to the Jinja2 template file.

        Returns:
            A callable with signature (metadata, pieces, graph_context) -> str.

        Raises:
            ImportError: If jinja2 is not installed.
        """
        try:
            import jinja2
        except ImportError:
            raise ImportError(
                "jinja2 is required for Jinja2 template formatters. "
                "Install it with: pip install jinja2"
            )

        with open(template_path, "r") as f:
            template_str = f.read()
        template = jinja2.Template(template_str)

        def formatter(
            metadata: Optional[EntityMetadata],
            pieces: List[Tuple[KnowledgePiece, float]],
            graph_context: List[Dict[str, Any]],
        ) -> str:
            return template.render(
                metadata=metadata,
                pieces=pieces,
                graph_context=graph_context,
            )

        return formatter

    @staticmethod
    def _make_handlebars_formatter(template_path: str) -> Callable:
        """Create a Handlebars template formatter from a file path.

        Args:
            template_path: Path to the Handlebars template file.

        Returns:
            A callable with signature (metadata, pieces, graph_context) -> str.

        Raises:
            ImportError: If pybars3 is not installed.
        """
        try:
            import pybars
        except ImportError:
            raise ImportError(
                "pybars3 is required for Handlebars template formatters. "
                "Install it with: pip install pybars3"
            )

        with open(template_path, "r") as f:
            template_str = f.read()
        compiler = pybars.Compiler()
        template = compiler.compile(template_str)

        def formatter(
            metadata: Optional[EntityMetadata],
            pieces: List[Tuple[KnowledgePiece, float]],
            graph_context: List[Dict[str, Any]],
        ) -> str:
            result = template(
                {
                    "metadata": metadata,
                    "pieces": pieces,
                    "graph_context": graph_context,
                }
            )
            return str(result)

        return formatter

    @staticmethod
    def _make_file_format_formatter(template_path: str) -> Callable:
        """Create a Python .format() formatter from a file template.

        Args:
            template_path: Path to the template file.

        Returns:
            A callable with signature (metadata, pieces, graph_context) -> str.
        """
        with open(template_path, "r") as f:
            template_str = f.read()

        def formatter(
            metadata: Optional[EntityMetadata],
            pieces: List[Tuple[KnowledgePiece, float]],
            graph_context: List[Dict[str, Any]],
        ) -> str:
            return template_str.format(
                metadata=metadata,
                pieces=pieces,
                graph_context=graph_context,
            )

        return formatter

    @staticmethod
    def _make_inline_format_formatter(format_string: str) -> Callable:
        """Create a Python .format() formatter from an inline string.

        Args:
            format_string: The format string template.

        Returns:
            A callable with signature (metadata, pieces, graph_context) -> str.
        """

        def formatter(
            metadata: Optional[EntityMetadata],
            pieces: List[Tuple[KnowledgePiece, float]],
            graph_context: List[Dict[str, Any]],
        ) -> str:
            return format_string.format(
                metadata=metadata,
                pieces=pieces,
                graph_context=graph_context,
            )

        return formatter

    def close(self):
        """Close the underlying KnowledgeBase."""
        self.kb.close()

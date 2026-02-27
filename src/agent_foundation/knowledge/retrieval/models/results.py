"""
Result types for knowledge system operations.

Provides structured result types for deduplication, merge operations,
validation, scoring, and other knowledge system processes.
"""

from typing import Any, Dict, List, Optional

from attr import attrib, attrs

from agent_foundation.knowledge.retrieval.models.enums import (
    DedupAction,
    MergeAction,
    MergeType,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece


@attrs
class DedupResult:
    """Result of three-tier deduplication."""

    action: DedupAction = attrib()
    reason: str = attrib(default="")
    existing_piece_id: Optional[str] = attrib(default=None)
    similarity_score: float = attrib(default=0.0)
    contradiction_detected: bool = attrib(default=False)


@attrs
class MergeCandidate:
    """A potential merge candidate with metadata."""

    piece_id: str = attrib()
    similarity: float = attrib()
    merge_type: MergeType = attrib()
    reason: str = attrib(default="")


@attrs
class MergeResult:
    """Result of merge strategy application."""

    action: MergeAction = attrib()
    piece_id: Optional[str] = attrib(default=None)
    merged_with: Optional[str] = attrib(default=None)
    error: Optional[str] = attrib(default=None)


@attrs
class ValidationResult:
    """Result of knowledge validation."""

    is_valid: bool = attrib(default=True)
    confidence: float = attrib(default=1.0)
    issues: List[str] = attrib(factory=list)
    suggestions: List[str] = attrib(factory=list)
    checks_passed: List[str] = attrib(factory=list)
    checks_failed: List[str] = attrib(factory=list)


@attrs
class ScoredPiece:
    """A KnowledgePiece with retrieval scores.

    The piece attribute holds the actual KnowledgePiece.
    Convenience properties provide direct access to common fields.
    """

    piece: KnowledgePiece = attrib()
    score: float = attrib(default=0.0)
    normalized_score: float = attrib(default=0.0)
    vector_score: Optional[float] = attrib(default=None)
    keyword_score: Optional[float] = attrib(default=None)

    @property
    def piece_id(self) -> str:
        """Delegate to wrapped piece's piece_id."""
        return self.piece.piece_id

    @property
    def content(self) -> str:
        """Delegate to wrapped piece's content."""
        return self.piece.content

    @property
    def info_type(self) -> str:
        """Delegate to wrapped piece's info_type."""
        return self.piece.info_type

    @property
    def updated_at(self) -> str:
        """Delegate to wrapped piece's updated_at."""
        return self.piece.updated_at


@attrs
class MergeJobResult:
    """Result of a background merge job."""

    processed: int = attrib(default=0)
    merged: int = attrib(default=0)
    suggestions_created: int = attrib(default=0)
    errors: List[str] = attrib(factory=list)
    duration_seconds: float = attrib(default=0.0)


@attrs
class OperationResult:
    """Generic result for knowledge operations (update, delete, restore).

    Attributes:
        success: Whether the operation succeeded.
        operation: The operation type ("update", "delete", "restore").
        piece_id: The ID of the affected piece.
        old_version: Version before the operation (for updates).
        new_version: Version after the operation (for updates).
        error: Error message if operation failed.
        details: Additional metadata (e.g., {"action": "replace", "summary": "..."}).
    """

    success: bool = attrib()
    operation: str = attrib()
    piece_id: Optional[str] = attrib(default=None)
    old_version: Optional[int] = attrib(default=None)
    new_version: Optional[int] = attrib(default=None)
    error: Optional[str] = attrib(default=None)
    details: Dict[str, Any] = attrib(factory=dict)

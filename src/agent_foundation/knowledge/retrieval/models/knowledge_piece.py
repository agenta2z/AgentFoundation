"""
KnowledgePiece data model and KnowledgeType enum.

Provides the core data model for unstructured knowledge chunks with metadata
for search, filtering, and lifecycle management. Each piece has content, a type
classification, tags, domain classification, and optional entity ownership.

The classification system supports two orthogonal dimensions:
1. Retrieval Classification (for FINDING knowledge):
   - domain/secondary_domains: Primary and additional domain categories
   - tags/custom_tags: Fine-grained topics for filtering

2. Injection Classification (for ORGANIZING in prompts):
   - info_type: WHERE in prompt (skills, instructions, context, etc.)
   - knowledge_type: WHAT kind (fact, instruction, procedure, etc.)

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11
"""

import hashlib
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional

from attr import attrs, attrib


class KnowledgeType(StrEnum):
    """Classification of knowledge pieces (WHAT the content is).

    Each type represents a different category of knowledge:
    - Fact: A factual statement
    - Instruction: A single rule or directive
    - Preference: A preference or opinion
    - Procedure: Multi-step process or technique
    - Note: General note or observation
    - Episodic: Past interaction summary or event record
    - Example: Worked example or case study
    """
    Fact = "fact"
    Instruction = "instruction"
    Preference = "preference"
    Procedure = "procedure"
    Note = "note"
    Episodic = "episodic"
    Example = "example"


@attrs
class KnowledgePiece:
    """An unstructured text chunk with metadata for search, filtering, and lifecycle.

    Existing Fields (order preserved for positional arg compatibility):
        content: The knowledge text (must be non-empty).
        piece_id: Unique identifier. Auto-generated UUID if None.
        knowledge_type: Classification of this knowledge piece (WHAT the content is).
        info_type: Prompt routing destination (WHERE the content goes). Plain string,
            default "context". Common values: "user_profile", "instructions", "context".
        tags: Lowercase tag strings for filtering.
        entity_id: Owning entity (None = global knowledge).
        source: Where this knowledge came from.
        embedding_text: Override text for embedding (None = use content).
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 last-update timestamp.

    New Fields (appended after existing fields, all have defaults):
        domain: Primary domain category for retrieval filtering.
        secondary_domains: Additional relevant domains.
        custom_tags: User-defined tags beyond the standard taxonomy.
        content_hash: Auto-computed SHA256 hash of whitespace-normalized content.
        embedding: Pre-computed embedding vector.
        space: Knowledge space scoping (main, personal, developmental).
        merge_strategy: Configured merge strategy for this piece.
        merge_processed: Whether merge has been processed.
        pending_merge_suggestion: Piece ID of suggested merge target.
        merge_suggestion_reason: Reason for merge suggestion.
        suggestion_status: Status of merge suggestion (pending, approved, rejected, expired).
        validation_status: Validation status (not_validated, pending, passed, failed).
        validation_issues: List of validation issues found.
        supersedes: ID of the piece this one replaces.
        is_active: Whether this piece is active (False = soft deleted).
        version: Version number for this piece.
        summary: Short summary for progressive disclosure.
    """
    # ── Existing fields (order preserved for positional arg compatibility) ──
    content: str = attrib()
    piece_id: str = attrib(default=None)
    knowledge_type: KnowledgeType = attrib(default=KnowledgeType.Fact)
    info_type: str = attrib(default="context")
    tags: List[str] = attrib(factory=list)
    entity_id: Optional[str] = attrib(default=None)
    source: Optional[str] = attrib(default=None)
    embedding_text: Optional[str] = attrib(default=None)
    created_at: str = attrib(default=None)
    updated_at: str = attrib(default=None)

    # ── New fields (all appended AFTER existing fields, all have defaults) ──

    # Retrieval Classification
    domain: str = attrib(default="general")
    secondary_domains: List[str] = attrib(factory=list)
    custom_tags: List[str] = attrib(factory=list)

    # Deduplication
    content_hash: Optional[str] = attrib(default=None)
    embedding: Optional[List[float]] = attrib(default=None)

    # Spaces
    space: str = attrib(default="main")

    # Merge Strategy
    merge_strategy: Optional[str] = attrib(default=None)
    merge_processed: bool = attrib(default=False)
    pending_merge_suggestion: Optional[str] = attrib(default=None)
    merge_suggestion_reason: Optional[str] = attrib(default=None)
    suggestion_status: Optional[str] = attrib(default=None)

    # Validation
    validation_status: str = attrib(default="not_validated")
    validation_issues: List[str] = attrib(factory=list)

    # Versioning
    supersedes: Optional[str] = attrib(default=None)
    is_active: bool = attrib(default=True)
    version: int = attrib(default=1)

    # Progressive Disclosure
    summary: Optional[str] = attrib(default=None)

    def __attrs_post_init__(self):
        if self.piece_id is None:
            self.piece_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        # Normalize tags: strip whitespace, lowercase, remove empty strings
        self.tags = [t.strip().lower() for t in self.tags if t.strip()]

        # Auto-compute content_hash if not provided
        if self.content_hash is None:
            self.content_hash = self._compute_content_hash()

    def _compute_content_hash(self) -> str:
        """Compute SHA256 hash of whitespace-normalized content.

        Returns:
            First 16 characters of the SHA256 hex digest of the
            whitespace-normalized content string.
        """
        normalized = " ".join(self.content.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of this KnowledgePiece with all fields.
        """
        return {
            "content": self.content,
            "piece_id": self.piece_id,
            "knowledge_type": self.knowledge_type.value,
            "info_type": self.info_type,
            "tags": list(self.tags),
            "entity_id": self.entity_id,
            "source": self.source,
            "embedding_text": self.embedding_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # New fields
            "domain": self.domain,
            "secondary_domains": list(self.secondary_domains),
            "custom_tags": list(self.custom_tags),
            "content_hash": self.content_hash,
            "embedding": self.embedding,
            "space": self.space,
            "merge_strategy": self.merge_strategy,
            "merge_processed": self.merge_processed,
            "pending_merge_suggestion": self.pending_merge_suggestion,
            "merge_suggestion_reason": self.merge_suggestion_reason,
            "suggestion_status": self.suggestion_status,
            "validation_status": self.validation_status,
            "validation_issues": list(self.validation_issues),
            "supersedes": self.supersedes,
            "is_active": self.is_active,
            "version": self.version,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgePiece":
        """Deserialize from dictionary. Validates required fields.

        Uses .get() with defaults for all new fields to maintain backward
        compatibility with dictionaries that omit new fields.

        Args:
            data: Dictionary containing KnowledgePiece fields.

        Returns:
            A new KnowledgePiece instance.

        Raises:
            ValueError: If content is missing, empty, or whitespace-only.
            ValueError: If required 'content' key is not present.
        """
        if "content" not in data:
            raise ValueError("Missing required field: 'content'")

        content = data["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Field 'content' must be a non-empty string")

        # Parse knowledge_type from string value if present
        knowledge_type = data.get("knowledge_type", KnowledgeType.Fact)
        if isinstance(knowledge_type, str):
            knowledge_type = KnowledgeType(knowledge_type)

        return cls(
            content=content,
            piece_id=data.get("piece_id"),
            knowledge_type=knowledge_type,
            info_type=data.get("info_type", "context"),
            tags=data.get("tags", []),
            entity_id=data.get("entity_id"),
            source=data.get("source"),
            embedding_text=data.get("embedding_text"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            # New fields (all with defaults for backward compat)
            domain=data.get("domain", "general"),
            secondary_domains=data.get("secondary_domains", []),
            custom_tags=data.get("custom_tags", []),
            content_hash=data.get("content_hash"),
            embedding=data.get("embedding"),
            space=data.get("space", "main"),
            merge_strategy=data.get("merge_strategy"),
            merge_processed=data.get("merge_processed", False),
            pending_merge_suggestion=data.get("pending_merge_suggestion"),
            merge_suggestion_reason=data.get("merge_suggestion_reason"),
            suggestion_status=data.get("suggestion_status"),
            validation_status=data.get("validation_status", "not_validated"),
            validation_issues=data.get("validation_issues", []),
            supersedes=data.get("supersedes"),
            is_active=data.get("is_active", True),
            version=data.get("version", 1),
            summary=data.get("summary"),
        )

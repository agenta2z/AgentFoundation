"""
KnowledgePiece data model and KnowledgeType enum.

Provides the core data model for unstructured knowledge chunks with metadata
for search and filtering. Each piece has content, a type classification, tags,
and optional entity ownership.

Requirements: 1.1, 1.2, 1.4, 1.5, 1.6
"""
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional

from attr import attrs, attrib


class KnowledgeType(StrEnum):
    """Classification of knowledge pieces.

    Each type represents a different category of knowledge:
    - Fact: A factual statement
    - Instruction: Behavioral directive / skill / SOP
    - Preference: A preference or opinion
    - Procedure: Step-by-step process
    - Note: General note or observation
    - Episodic: Past interaction summary
    """
    Fact = "fact"
    Instruction = "instruction"
    Preference = "preference"
    Procedure = "procedure"
    Note = "note"
    Episodic = "episodic"


@attrs
class KnowledgePiece:
    """An unstructured text chunk with metadata for search and filtering.

    Attributes:
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
    """
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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgePiece":
        """Deserialize from dictionary. Validates required fields.

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
        )

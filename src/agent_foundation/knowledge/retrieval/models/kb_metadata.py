"""KnowledgeBase-level metadata and operation log.

Provides a global view of KB operations and stats, complementing the
per-entity DataOperationRecord history. The operation log is append-only;
rollback adds a new entry rather than deleting old ones.

Persisted as ``{store_path}/_kb_metadata.json``.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import attrs
from attrs import Factory


@attrs.define
class KnowledgeBaseOperationEntry:
    """One entry in the KB-level operation log."""

    operation_id: str
    description: str
    timestamp: str
    source: Optional[str] = None
    entity_count: int = 0
    details: Dict[str, Any] = Factory(dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d: Dict[str, Any] = {
            "operation_id": self.operation_id,
            "description": self.description,
            "timestamp": self.timestamp,
        }
        if self.source is not None:
            d["source"] = self.source
        if self.entity_count:
            d["entity_count"] = self.entity_count
        if self.details:
            d["details"] = self.details
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBaseOperationEntry":
        """Reconstruct from dictionary."""
        return cls(
            operation_id=data["operation_id"],
            description=data["description"],
            timestamp=data["timestamp"],
            source=data.get("source"),
            entity_count=data.get("entity_count", 0),
            details=data.get("details", {}),
        )


@attrs.define
class KnowledgeBaseMetadata:
    """Top-level metadata for a knowledge base instance."""

    kb_id: str
    created_at: str = attrs.field(default=None)
    updated_at: str = attrs.field(default=None)
    stats: Dict[str, int] = Factory(dict)
    operations: List[KnowledgeBaseOperationEntry] = Factory(list)

    def __attrs_post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    def log_operation(
        self,
        operation_id: str,
        description: str,
        source: Optional[str] = None,
        entity_count: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an operation entry and update the timestamp."""
        self.operations.append(
            KnowledgeBaseOperationEntry(
                operation_id=operation_id,
                description=description,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=source,
                entity_count=entity_count,
                details=details or {},
            )
        )
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kb_id": self.kb_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stats": dict(self.stats),
            "operations": [op.to_dict() for op in self.operations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBaseMetadata":
        """Reconstruct from dictionary."""
        return cls(
            kb_id=data["kb_id"],
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            stats=data.get("stats", {}),
            operations=[
                KnowledgeBaseOperationEntry.from_dict(op)
                for op in data.get("operations", [])
            ],
        )

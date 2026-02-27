"""
KnowledgePieceStore abstract base class.

Defines the abstract interface for knowledge piece storage backends. All
knowledge piece store implementations (file-based, SQLite FTS5, Chroma,
LanceDB, Elasticsearch) must implement this interface.

The KnowledgePieceStore manages unstructured text chunks (knowledge pieces)
with support for CRUD operations, search, and filtering.

Requirements: 2.1
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)


class KnowledgePieceStore(ABC):
    """Abstract base class for knowledge piece storage backends.

    Provides the interface for CRUD operations and search on knowledge pieces.
    Implementations may use file-based JSON with keyword search, SQLite FTS5,
    Chroma vector search, LanceDB hybrid search, or Elasticsearch.

    All implementations must support:
    - Adding pieces (with duplicate ID detection)
    - Getting pieces by ID
    - Updating existing pieces
    - Removing pieces
    - Searching pieces by query with optional filters
    - Listing all pieces with optional filters

    The ``close()`` method is a concrete no-op by default. Subclasses that hold
    external connections (e.g., SQLite, Chroma, LanceDB, Elasticsearch) should
    override it to release resources.
    """

    @abstractmethod
    def add(self, piece: KnowledgePiece) -> str:
        """Add a knowledge piece to the store.

        Args:
            piece: The KnowledgePiece to add.

        Returns:
            The piece_id of the added piece.

        Raises:
            ValueError: If a piece with the same piece_id already exists.
        """
        ...

    @abstractmethod
    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        """Get a knowledge piece by its ID.

        Args:
            piece_id: The unique identifier of the piece.

        Returns:
            The KnowledgePiece if found, or None if not found.
        """
        ...

    @abstractmethod
    def update(self, piece: KnowledgePiece) -> bool:
        """Update an existing knowledge piece.

        Replaces the stored piece with the provided one. The piece is matched
        by its piece_id.

        Args:
            piece: The KnowledgePiece with updated fields.

        Returns:
            True if the piece was found and updated, False if not found.
        """
        ...

    @abstractmethod
    def remove(self, piece_id: str) -> bool:
        """Remove a knowledge piece from the store.

        Args:
            piece_id: The unique identifier of the piece to remove.

        Returns:
            True if the piece existed and was removed, False if not found.
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
        tags: List[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Search pieces by query with optional filters.

        Returns matching pieces ranked by relevance score in descending order.
        Scores are normalized to the range [0.0, 1.0].

        Args:
            query: The search query string.
            entity_id: If specified, search only this entity's pieces.
                       If None, search only global pieces (entity_id=None in storage).
                       Note: Cross-scope merging (entity + global) is handled by
                       KnowledgeBase, not by the store. The store does simple
                       single-scope filtering.
            knowledge_type: If specified, filter to this knowledge type only.
            tags: If specified, filter to pieces containing all of these tags.
            top_k: Maximum number of results to return.

        Returns:
            A list of (KnowledgePiece, relevance_score) tuples ordered by
            descending relevance score. Scores are in [0.0, 1.0].
        """
        ...

    @abstractmethod
    def list_all(
        self,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
    ) -> List[KnowledgePiece]:
        """List all pieces matching the given filters.

        Args:
            entity_id: If specified, list only this entity's pieces.
                       If None, list only global pieces.
            knowledge_type: If specified, filter to this knowledge type only.

        Returns:
            A list of KnowledgePiece objects matching the filter criteria.
        """
        ...

    def find_by_content_hash(
        self,
        content_hash: str,
        entity_id: Optional[str] = None,
    ) -> Optional[KnowledgePiece]:
        """Find a piece by its content hash.

        Default implementation does a linear scan over all pieces.
        Subclasses should override with indexed lookup for better performance.

        Args:
            content_hash: SHA256 hash prefix (16 chars).
            entity_id: If provided, also check entity-scoped pieces.

        Returns:
            The matching piece if found, None otherwise.
        """
        for piece in self.list_all(entity_id=None):
            if getattr(piece, "content_hash", None) == content_hash:
                return piece

        if entity_id:
            for piece in self.list_all(entity_id=entity_id):
                if getattr(piece, "content_hash", None) == content_hash:
                    return piece

        return None

    def close(self):
        """Close any underlying connections.

        Default no-op for file-based stores. Override for stores with
        external connections (e.g., SQLite, Chroma, LanceDB, Elasticsearch)
        to release resources.
        """
        pass



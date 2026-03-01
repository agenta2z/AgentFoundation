"""
RetrievalKnowledgePieceStore — KnowledgePieceStore adapter backed by RetrievalServiceBase.

Implements the KnowledgePieceStore ABC by delegating all operations to a
general-purpose RetrievalServiceBase instance. This adapter bridges the
domain-specific knowledge module with the generic retrieval service from
SciencePythonUtils.

Mapping:
    - piece_id    → doc_id
    - content     → content
    - entity_id   → namespace
    - knowledge_type, tags, entity_id, source, domain, secondary_domains, custom_tags → metadata dict
    - is_active, version, supersedes, content_hash, space → metadata dict (lifecycle)
    - validation_status, validation_issues, summary → metadata dict (lifecycle)
    - merge_strategy, merge_processed, pending_merge_suggestion → metadata dict (lifecycle)
    - merge_suggestion_reason, suggestion_status → metadata dict (lifecycle)
    - embedding_text → embedding_text
    - created_at, updated_at → created_at, updated_at

Requirements: 12.1, 12.2, 12.3, 12.4
"""
from typing import List, Optional, Tuple

from attr import attrs, attrib

from rich_python_utils.service_utils.retrieval_service.retrieval_service_base import (
    RetrievalServiceBase,
)
from rich_python_utils.service_utils.retrieval_service.document import Document
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


@attrs
class RetrievalKnowledgePieceStore(KnowledgePieceStore):
    """KnowledgePieceStore backed by any RetrievalServiceBase.

    Delegates all CRUD and search operations to a generic retrieval service,
    converting between KnowledgePiece and Document models. The entity_id
    field maps to the retrieval service namespace, and knowledge_type/tags
    are stored as metadata for filtering.

    Attributes:
        retrieval_service: The underlying retrieval service instance.
    """

    retrieval_service: RetrievalServiceBase = attrib()

    def _piece_to_doc(self, piece: KnowledgePiece) -> Document:
        """Convert KnowledgePiece to Document.

        Maps piece_id to doc_id, content to content, and packs
        knowledge_type, tags, entity_id, source, domain,
        secondary_domains, and custom_tags into metadata.

        Args:
            piece: The KnowledgePiece to convert.

        Returns:
            A Document with equivalent data.
        """
        return Document(
            doc_id=piece.piece_id,
            content=piece.content,
            metadata={
                "knowledge_type": piece.knowledge_type.value,
                "info_type": piece.info_type,
                "tags": list(piece.tags),
                "entity_id": piece.entity_id,
                "source": piece.source,
                "domain": piece.domain,
                "secondary_domains": list(piece.secondary_domains),
                "custom_tags": list(piece.custom_tags),
                # Lifecycle fields — required for soft-delete, versioning, restore
                "is_active": piece.is_active,
                "version": piece.version,
                "supersedes": piece.supersedes,
                "content_hash": piece.content_hash,
                "space": piece.space,
                "validation_status": piece.validation_status,
                "validation_issues": list(piece.validation_issues),
                "summary": piece.summary,
                "merge_strategy": piece.merge_strategy,
                "merge_processed": piece.merge_processed,
                "pending_merge_suggestion": piece.pending_merge_suggestion,
                "merge_suggestion_reason": piece.merge_suggestion_reason,
                "suggestion_status": piece.suggestion_status,
                # Multi-space membership and suggestion fields
                "spaces": list(piece.spaces),
                "pending_space_suggestions": piece.pending_space_suggestions,
                "space_suggestion_reasons": piece.space_suggestion_reasons,
                "space_suggestion_status": piece.space_suggestion_status,
            },
            embedding_text=piece.embedding_text,
            created_at=piece.created_at,
            updated_at=piece.updated_at,
        )

    def _doc_to_piece(self, doc: Document) -> KnowledgePiece:
        """Convert Document back to KnowledgePiece.

        Extracts knowledge_type, tags, entity_id, source, domain,
        secondary_domains, and custom_tags from the document's metadata
        dict and reconstructs a KnowledgePiece.

        Args:
            doc: The Document to convert.

        Returns:
            A KnowledgePiece with equivalent data.
        """
        return KnowledgePiece(
            content=doc.content,
            piece_id=doc.doc_id,
            knowledge_type=KnowledgeType(doc.metadata.get("knowledge_type", "fact")),
            info_type=doc.metadata.get("info_type", "context"),
            tags=doc.metadata.get("tags", []),
            entity_id=doc.metadata.get("entity_id"),
            source=doc.metadata.get("source"),
            embedding_text=doc.embedding_text,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            domain=doc.metadata.get("domain", "general"),
            secondary_domains=doc.metadata.get("secondary_domains", []),
            custom_tags=doc.metadata.get("custom_tags", []),
            # Lifecycle fields — defaults ensure backward compat with old JSON
            is_active=doc.metadata.get("is_active", True),
            version=doc.metadata.get("version", 1),
            supersedes=doc.metadata.get("supersedes"),
            content_hash=doc.metadata.get("content_hash"),
            space=doc.metadata.get("space", "main"),
            validation_status=doc.metadata.get("validation_status", "not_validated"),
            validation_issues=doc.metadata.get("validation_issues", []),
            summary=doc.metadata.get("summary"),
            merge_strategy=doc.metadata.get("merge_strategy"),
            merge_processed=doc.metadata.get("merge_processed", False),
            pending_merge_suggestion=doc.metadata.get("pending_merge_suggestion"),
            merge_suggestion_reason=doc.metadata.get("merge_suggestion_reason"),
            suggestion_status=doc.metadata.get("suggestion_status"),
            # Multi-space membership and suggestion fields
            # spaces=None triggers __attrs_post_init__ fallback to [space]
            spaces=doc.metadata.get("spaces"),
            pending_space_suggestions=doc.metadata.get("pending_space_suggestions"),
            space_suggestion_reasons=doc.metadata.get("space_suggestion_reasons"),
            space_suggestion_status=doc.metadata.get("space_suggestion_status"),
        )

    def _namespace(self, entity_id: Optional[str]) -> Optional[str]:
        """Map entity_id to namespace.

        None entity_id maps to None namespace (which the retrieval service
        resolves to "_default" internally).

        Args:
            entity_id: The entity ID, or None for global pieces.

        Returns:
            The namespace string, or None for the default namespace.
        """
        return entity_id

    def add(self, piece: KnowledgePiece) -> str:
        """Add a knowledge piece to the store.

        Converts the piece to a Document and delegates to the retrieval
        service's add method, using entity_id as the namespace.

        Args:
            piece: The KnowledgePiece to add.

        Returns:
            The piece_id of the added piece.

        Raises:
            ValueError: If a piece with the same piece_id already exists.
        """
        doc = self._piece_to_doc(piece)
        return self.retrieval_service.add(doc, namespace=self._namespace(piece.entity_id))

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        """Get a knowledge piece by its ID.

        Searches across all namespaces in the retrieval service to find
        the document with the given ID, then converts it back to a
        KnowledgePiece.

        Args:
            piece_id: The unique identifier of the piece.

        Returns:
            The KnowledgePiece if found, or None if not found.
        """
        # We need to search across all namespaces since we don't know
        # which entity_id (namespace) the piece belongs to.
        for ns in self.retrieval_service.namespaces():
            doc = self.retrieval_service.get_by_id(piece_id, namespace=ns)
            if doc is not None:
                return self._doc_to_piece(doc)
        # Also check the default namespace (entity_id=None)
        doc = self.retrieval_service.get_by_id(piece_id, namespace=None)
        if doc is not None:
            return self._doc_to_piece(doc)
        return None

    def update(self, piece: KnowledgePiece) -> bool:
        """Update an existing knowledge piece.

        Converts the piece to a Document and delegates to the retrieval
        service's update method, using entity_id as the namespace.

        Args:
            piece: The KnowledgePiece with updated fields.

        Returns:
            True if the piece was found and updated, False if not found.
        """
        doc = self._piece_to_doc(piece)
        return self.retrieval_service.update(doc, namespace=self._namespace(piece.entity_id))

    def remove(self, piece_id: str) -> bool:
        """Remove a knowledge piece from the store.

        Searches across all namespaces to find and remove the piece.

        Args:
            piece_id: The unique identifier of the piece to remove.

        Returns:
            True if the piece existed and was removed, False if not found.
        """
        # Try all namespaces since we don't know which one the piece is in
        for ns in self.retrieval_service.namespaces():
            if self.retrieval_service.remove(piece_id, namespace=ns):
                return True
        # Also try the default namespace
        if self.retrieval_service.remove(piece_id, namespace=None):
            return True
        return False

    def search(
        self,
        query: str,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
        tags: List[str] = None,
        top_k: int = 5,
        spaces: Optional[List[str]] = None,
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Search pieces by query with optional filters.

        Converts knowledge_type and tags to metadata filters and delegates
        to the retrieval service's search method. The entity_id maps to
        the namespace for scoping.

        The ``spaces`` parameter is accepted for ABC compliance but is NOT
        applied — this store does not support native space filtering.
        The KnowledgeBase handles over-fetch and post-filter for this store.

        Args:
            query: The search query string.
            entity_id: If specified, search only this entity's pieces.
                       If None, search only global pieces.
            knowledge_type: If specified, filter to this knowledge type only.
            tags: If specified, filter to pieces containing all of these tags.
            top_k: Maximum number of results to return.
            spaces: Accepted for ABC compliance; ignored by this store.

        Returns:
            A list of (KnowledgePiece, relevance_score) tuples ordered by
            descending relevance score.
        """
        filters = {}
        if knowledge_type:
            filters["knowledge_type"] = knowledge_type.value
        if tags:
            filters["tags"] = tags
        results = self.retrieval_service.search(
            query,
            filters=filters or None,
            namespace=self._namespace(entity_id),
            top_k=top_k,
        )
        return [(self._doc_to_piece(doc), score) for doc, score in results]

    def list_all(
        self,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
        spaces: Optional[List[str]] = None,
    ) -> List[KnowledgePiece]:
        """List all pieces matching the given filters.

        Converts knowledge_type to a metadata filter and delegates to the
        retrieval service's list_all method. The entity_id maps to the
        namespace for scoping.

        The ``spaces`` parameter is accepted for ABC compliance but is NOT
        applied — this store does not support native space filtering.

        Args:
            entity_id: If specified, list only this entity's pieces.
                       If None, list only global pieces.
            knowledge_type: If specified, filter to this knowledge type only.
            spaces: Accepted for ABC compliance; ignored by this store.

        Returns:
            A list of KnowledgePiece objects matching the filter criteria.
        """
        filters = {}
        if knowledge_type:
            filters["knowledge_type"] = knowledge_type.value
        docs = self.retrieval_service.list_all(
            filters=filters or None,
            namespace=self._namespace(entity_id),
        )
        return [self._doc_to_piece(doc) for doc in docs]

    def close(self):
        """Close the underlying retrieval service."""
        self.retrieval_service.close()

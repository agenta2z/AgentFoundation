"""
KnowledgeDeleter — handles knowledge deletion by ID or semantic matching.

Supports:
1. Soft Delete: Mark is_active=False (preserves history)
2. Hard Delete: Remove from store permanently

Design Decisions:
- Raise ConfirmationRequiredError instead of silent empty return
- Use generic OperationResult
- Check for superseding pieces before restore
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from agent_foundation.knowledge.retrieval.models.enums import DeleteMode
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import OperationResult
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore

logger = logging.getLogger(__name__)


class ConfirmationRequiredError(Exception):
    """Raised when operation requires user confirmation."""

    def __init__(
        self,
        candidates: List[Tuple[KnowledgePiece, float]],
        message: Optional[str] = None,
    ):
        self.candidates = candidates
        self.message = message or (
            f"Confirmation required. Found {len(candidates)} candidates."
        )
        super().__init__(self.message)


@dataclass
class DeleteConfig:
    """Configuration for knowledge deletion."""

    default_mode: DeleteMode = DeleteMode.SOFT
    similarity_threshold: float = 0.90
    max_deletions: int = 10
    require_confirmation: bool = True


class KnowledgeDeleter:
    """Handles knowledge deletion by ID or semantic matching.

    NOTE: This implementation assumes that piece_store.get_by_id()
    returns pieces regardless of is_active status.

    PERFORMANCE WARNING: restore_by_id() calls list_all() which
    loads all pieces for the entity into memory. For large knowledge bases,
    consider adding a find_by_supersedes() method to the store interface.
    """

    def __init__(
        self,
        piece_store: KnowledgePieceStore,
        config: Optional[DeleteConfig] = None,
    ):
        self.piece_store = piece_store
        self.config = config or DeleteConfig()

    def delete_by_id(
        self,
        piece_id: str,
        mode: Optional[DeleteMode] = None,
    ) -> OperationResult:
        """Delete a specific piece by its ID."""
        mode = mode or self.config.default_mode

        existing = self.piece_store.get_by_id(piece_id)
        if existing is None:
            return OperationResult(
                success=False,
                operation="delete",
                piece_id=piece_id,
                error=f"Piece not found: {piece_id}",
            )

        if mode == DeleteMode.SOFT:
            existing.is_active = False
            existing.updated_at = datetime.now(timezone.utc).isoformat()
            self.piece_store.update(existing)
            logger.info("Soft deleted piece: %s", piece_id)
        else:
            self.piece_store.remove(piece_id)
            logger.info("Hard deleted piece: %s", piece_id)

        return OperationResult(
            success=True,
            operation="delete",
            piece_id=piece_id,
            details={"mode": mode.value},
        )

    def find_candidates_for_deletion(
        self,
        query: str,
        entity_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Find pieces matching a query that could be deleted."""
        candidates = self.piece_store.search(
            query=query,
            entity_id=entity_id,
            top_k=self.config.max_deletions,
        )

        # Filter by threshold and active status
        matches = [
            (piece, score)
            for piece, score in candidates
            if score >= self.config.similarity_threshold
            and getattr(piece, "is_active", True)
        ]

        if domain:
            matches = [
                (p, s)
                for p, s in matches
                if p.domain == domain
                or domain in getattr(p, "secondary_domains", [])
            ]

        return matches

    def delete_by_query(
        self,
        query: str,
        entity_id: Optional[str] = None,
        domain: Optional[str] = None,
        mode: Optional[DeleteMode] = None,
        piece_ids: Optional[List[str]] = None,
    ) -> List[OperationResult]:
        """Delete pieces matching a query.

        Raises:
            ConfirmationRequiredError: If require_confirmation=True and no
                piece_ids provided.
        """
        mode = mode or self.config.default_mode

        if piece_ids:
            return [self.delete_by_id(pid, mode) for pid in piece_ids]

        candidates = self.find_candidates_for_deletion(query, entity_id, domain)

        if self.config.require_confirmation:
            raise ConfirmationRequiredError(
                candidates=candidates,
                message=(
                    f"Found {len(candidates)} pieces matching query. "
                    f"Review candidates and call delete_by_query with piece_ids."
                ),
            )

        results = []
        for piece, _ in candidates[: self.config.max_deletions]:
            result = self.delete_by_id(piece.piece_id, mode)
            results.append(result)

        return results

    def delete_by_ids(
        self,
        piece_ids: List[str],
        mode: Optional[DeleteMode] = None,
    ) -> List[OperationResult]:
        """Delete multiple pieces by their IDs."""
        return [self.delete_by_id(pid, mode) for pid in piece_ids]

    def restore_by_id(
        self,
        piece_id: str,
    ) -> OperationResult:
        """Restore a soft-deleted piece.

        PERFORMANCE WARNING: This calls list_all() which may be
        slow for large knowledge bases.
        """
        existing = self.piece_store.get_by_id(piece_id)
        if existing is None:
            return OperationResult(
                success=False,
                operation="restore",
                piece_id=piece_id,
                error=f"Piece not found: {piece_id}",
            )

        if existing.is_active:
            return OperationResult(
                success=False,
                operation="restore",
                piece_id=piece_id,
                error="Piece is already active",
            )

        # Check if any active piece supersedes this one
        all_pieces = self.piece_store.list_all(entity_id=existing.entity_id)
        superseding = [
            p
            for p in all_pieces
            if p.is_active and getattr(p, "supersedes", None) == piece_id
        ]
        if superseding:
            return OperationResult(
                success=False,
                operation="restore",
                piece_id=piece_id,
                error=(
                    f"Cannot restore: piece is superseded by "
                    f"{superseding[0].piece_id}. "
                    f"Deactivate the superseding piece first."
                ),
            )

        existing.is_active = True
        existing.updated_at = datetime.now(timezone.utc).isoformat()
        self.piece_store.update(existing)
        logger.info("Restored piece: %s", piece_id)

        return OperationResult(
            success=True,
            operation="restore",
            piece_id=piece_id,
        )

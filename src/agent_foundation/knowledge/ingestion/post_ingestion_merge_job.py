"""
Background job for post-ingestion merge strategies.

Processes pieces with deferred merge strategies:
- POST_INGESTION_AUTO: Automatically merge in background
- POST_INGESTION_SUGGESTION: Create merge suggestions in background

NOTE: This job only processes GLOBAL pieces (entity_id=None).
      Entity-scoped pieces should be processed via entity-specific jobs.
"""

import logging
import time
from typing import Callable, List, Optional, Tuple

from agent_foundation.knowledge.retrieval.models.enums import (
    MergeStrategy,
    SuggestionStatus,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import MergeJobResult
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


class PostIngestionMergeJob:
    """Background job for post-ingestion merge strategies.

    NOTE: This implementation only processes global pieces (entity_id=None).
    For entity-scoped pieces, create entity-specific job instances or
    modify _find_deferred_pieces to accept an entity_id parameter.
    """

    def __init__(
        self,
        piece_store: KnowledgePieceStore,
        detect_candidates_fn: Callable[
            [KnowledgePiece], List[Tuple[str, float, str]]
        ],
        merge_fn: Optional[
            Callable[[KnowledgePiece, KnowledgePiece], KnowledgePiece]
        ] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.piece_store = piece_store
        self.detect_candidates_fn = detect_candidates_fn
        self.merge_fn = merge_fn
        self.logger = logger or logging.getLogger(__name__)

    def run(self, space: str = "main") -> MergeJobResult:
        """Process deferred merges for global pieces."""
        start = time.time()
        deferred = self._find_deferred_pieces(space)

        errors: List[str] = []
        merged = 0
        suggestions_created = 0

        for piece in deferred:
            try:
                candidates = self.detect_candidates_fn(piece)

                if piece.merge_strategy == MergeStrategy.POST_INGESTION_AUTO.value:
                    if candidates and self.merge_fn:
                        candidate_id, _, _ = candidates[0]
                        existing_piece = self.piece_store.get_by_id(candidate_id)
                        if existing_piece:
                            merged_piece = self.merge_fn(piece, existing_piece)
                            existing_piece.is_active = False
                            self.piece_store.update(existing_piece)
                            self.piece_store.add(merged_piece)
                            self.piece_store.remove(piece.piece_id)
                            merged += 1

                elif (
                    piece.merge_strategy
                    == MergeStrategy.POST_INGESTION_SUGGESTION.value
                ):
                    if candidates:
                        candidate_id, _, reason = candidates[0]
                        piece.pending_merge_suggestion = candidate_id
                        piece.merge_suggestion_reason = reason
                        piece.suggestion_status = SuggestionStatus.PENDING.value
                        suggestions_created += 1

                piece.merge_processed = True
                self.piece_store.update(piece)

            except Exception as e:
                errors.append(f"{piece.piece_id}: {e}")
                self.logger.error(
                    "Failed to process piece %s: %s", piece.piece_id, e
                )
                continue

        return MergeJobResult(
            processed=len(deferred) - len(errors),
            merged=merged,
            suggestions_created=suggestions_created,
            errors=errors,
            duration_seconds=time.time() - start,
        )

    def _find_deferred_pieces(self, space: str) -> List[KnowledgePiece]:
        """Find global pieces with deferred merge strategies."""
        all_pieces = self.piece_store.list_all(entity_id=None)

        deferred = []
        for piece in all_pieces:
            if not getattr(piece, "merge_processed", True):
                strategy = getattr(piece, "merge_strategy", None)
                piece_space = getattr(piece, "space", "main")

                if piece_space == space and strategy in (
                    MergeStrategy.POST_INGESTION_AUTO.value,
                    MergeStrategy.POST_INGESTION_SUGGESTION.value,
                ):
                    deferred.append(piece)

        return deferred

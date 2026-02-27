"""
Merge Strategy System for knowledge pieces.

Provides configurable strategies for handling potential duplicates:
1. auto-merge-on-ingest: Automatically merge during ingestion
2. suggestion-on-ingest: Flag for human review
3. post-ingestion-auto: Defer merge to background job
4. post-ingestion-suggestion: Defer suggestion to background job
5. manual-only: No automatic merging
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from agent_foundation.knowledge.ingestion.prompts.merge_execution import (
    MERGE_EXECUTION_PROMPT,
)
from agent_foundation.knowledge.retrieval.models.enums import (
    MergeAction,
    MergeStrategy,
    SuggestionStatus,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.results import (
    MergeCandidate,
    MergeResult,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore

logger = logging.getLogger(__name__)


@dataclass
class MergeStrategyConfig:
    """Configuration for merge strategies."""

    default_by_type: Dict[KnowledgeType, MergeStrategy] = field(
        default_factory=lambda: {
            KnowledgeType.Procedure: MergeStrategy.MANUAL_ONLY,
            KnowledgeType.Instruction: MergeStrategy.SUGGESTION_ON_INGEST,
            KnowledgeType.Fact: MergeStrategy.AUTO_MERGE_ON_INGEST,
            KnowledgeType.Preference: MergeStrategy.MANUAL_ONLY,
            KnowledgeType.Episodic: MergeStrategy.POST_INGESTION_AUTO,
            KnowledgeType.Note: MergeStrategy.AUTO_MERGE_ON_INGEST,
            KnowledgeType.Example: MergeStrategy.SUGGESTION_ON_INGEST,
        }
    )
    allow_override: bool = True
    suggestion_expiry_days: int = 30


class MergeStrategyManager:
    """Manages merge strategies for knowledge pieces."""

    def __init__(
        self,
        piece_store: KnowledgePieceStore,
        llm_fn: Optional[Callable[[str], str]] = None,
        config: Optional[MergeStrategyConfig] = None,
    ):
        self.piece_store = piece_store
        self.llm_fn = llm_fn
        self.config = config or MergeStrategyConfig()

    def get_strategy(self, piece: KnowledgePiece) -> MergeStrategy:
        """Get effective merge strategy for a piece."""
        if self.config.allow_override and piece.merge_strategy:
            return MergeStrategy(piece.merge_strategy)
        return self.config.default_by_type.get(
            piece.knowledge_type, MergeStrategy.AUTO_MERGE_ON_INGEST
        )

    def apply_strategy(
        self,
        piece: KnowledgePiece,
        candidates: List[MergeCandidate],
    ) -> MergeResult:
        """Apply merge strategy to a piece with candidates."""
        strategy = self.get_strategy(piece)

        if strategy == MergeStrategy.AUTO_MERGE_ON_INGEST:
            return self._auto_merge(piece, candidates)
        if strategy == MergeStrategy.SUGGESTION_ON_INGEST:
            return self._suggest_merge(piece, candidates)
        if strategy == MergeStrategy.POST_INGESTION_AUTO:
            return self._defer(piece)
        if strategy == MergeStrategy.POST_INGESTION_SUGGESTION:
            return self._defer(piece)
        if strategy == MergeStrategy.MANUAL_ONLY:
            return MergeResult(
                action=MergeAction.NO_AUTO_MERGE, piece_id=piece.piece_id
            )

        return MergeResult(
            action=MergeAction.ERROR,
            piece_id=piece.piece_id,
            error="Unknown strategy",
        )

    def _auto_merge(
        self,
        piece: KnowledgePiece,
        candidates: List[MergeCandidate],
    ) -> MergeResult:
        """Execute automatic merge if candidates exist."""
        if not candidates:
            return MergeResult(
                action=MergeAction.NO_CANDIDATES, piece_id=piece.piece_id
            )

        if self.llm_fn is None:
            logger.warning("Auto-merge requested but no llm_fn provided")
            return self._suggest_merge(piece, candidates)

        top_candidate = candidates[0]
        existing = self.piece_store.get_by_id(top_candidate.piece_id)

        if existing is None:
            return MergeResult(
                action=MergeAction.ERROR,
                piece_id=piece.piece_id,
                error=f"Candidate {top_candidate.piece_id} not found",
            )

        try:
            merged = self._execute_merge(piece, existing)
            existing.is_active = False
            self.piece_store.update(existing)
            self.piece_store.add(merged)

            return MergeResult(
                action=MergeAction.MERGED,
                piece_id=merged.piece_id,
                merged_with=existing.piece_id,
            )
        except Exception as e:
            logger.error("Merge failed: %s", e)
            return MergeResult(
                action=MergeAction.ERROR,
                piece_id=piece.piece_id,
                error=str(e),
            )

    def _execute_merge(
        self,
        new_piece: KnowledgePiece,
        existing_piece: KnowledgePiece,
    ) -> KnowledgePiece:
        """Execute LLM-based merge of two pieces."""
        prompt = MERGE_EXECUTION_PROMPT.format(
            piece_a_content=existing_piece.content,
            piece_a_domain=existing_piece.domain,
            piece_a_tags=", ".join(existing_piece.tags),
            piece_b_content=new_piece.content,
            piece_b_domain=new_piece.domain,
            piece_b_tags=", ".join(new_piece.tags),
        )

        response = self.llm_fn(prompt)
        parsed = json.loads(response)

        return KnowledgePiece(
            content=parsed["merged_content"],
            domain=parsed.get("merged_domain", existing_piece.domain),
            tags=parsed.get(
                "merged_tags",
                list(set(existing_piece.tags + new_piece.tags)),
            ),
            knowledge_type=existing_piece.knowledge_type,
            info_type=existing_piece.info_type,
            entity_id=existing_piece.entity_id,
            supersedes=existing_piece.piece_id,
            version=existing_piece.version + 1,
            source=f"merged:{existing_piece.piece_id}+{new_piece.piece_id}",
        )

    def _suggest_merge(
        self,
        piece: KnowledgePiece,
        candidates: List[MergeCandidate],
    ) -> MergeResult:
        """Create merge suggestion for human review."""
        if not candidates:
            return MergeResult(
                action=MergeAction.NO_CANDIDATES, piece_id=piece.piece_id
            )

        top_candidate = candidates[0]
        piece.pending_merge_suggestion = top_candidate.piece_id
        piece.merge_suggestion_reason = top_candidate.reason
        piece.suggestion_status = SuggestionStatus.PENDING.value

        return MergeResult(
            action=MergeAction.PENDING_REVIEW, piece_id=piece.piece_id
        )

    def _defer(self, piece: KnowledgePiece) -> MergeResult:
        """Defer merge to background job."""
        piece.merge_processed = False
        return MergeResult(
            action=MergeAction.DEFERRED, piece_id=piece.piece_id
        )

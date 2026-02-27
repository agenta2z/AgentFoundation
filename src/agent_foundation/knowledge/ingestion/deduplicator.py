"""
Three-Tier Deduplication for knowledge ingestion.

Implements a three-tier deduplication strategy:
1. Tier 1: Content hash (exact match) - O(1) lookup with index
2. Tier 2: Embedding similarity - threshold-based
3. Tier 3: LLM Judge - semantic analysis for borderline cases
"""

import json
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from agent_foundation.knowledge.ingestion.prompts.dedup_llm_judge import (
    DEDUP_LLM_JUDGE_PROMPT,
)
from agent_foundation.knowledge.retrieval.models.enums import DedupAction
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import DedupResult
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    """Configuration for three-tier deduplication."""

    auto_dedup_threshold: float = 0.98
    llm_judge_threshold: float = 0.85
    enable_tier1: bool = True
    enable_tier2: bool = True
    enable_tier3: bool = True


class ThreeTierDeduplicator:
    """Three-tier deduplication for knowledge pieces."""

    def __init__(
        self,
        piece_store: KnowledgePieceStore,
        embedding_fn: Callable[[str], List[float]],
        llm_fn: Optional[Callable[[str], str]] = None,
        config: Optional[DedupConfig] = None,
    ):
        self.piece_store = piece_store
        self.embedding_fn = embedding_fn
        self.llm_fn = llm_fn
        self.config = config or DedupConfig()

    def deduplicate(self, piece: KnowledgePiece) -> DedupResult:
        """Run three-tier deduplication on a piece."""
        # Tier 1: Content hash
        if self.config.enable_tier1:
            result = self._tier1_hash_check(piece)
            if result.action == DedupAction.NO_OP:
                return result

        # Tier 2: Embedding similarity
        if self.config.enable_tier2:
            result, top_match = self._tier2_embedding_check(piece)
            if result.action == DedupAction.NO_OP:
                return result

            # Borderline case: top_match present means score is between thresholds
            if top_match is not None:
                # Tier 3: LLM Judge (for borderline cases)
                if self.config.enable_tier3:
                    return self._tier3_llm_judge(
                        piece, top_match, result.similarity_score
                    )
                # Tier 3 disabled, default to ADD for borderline
                return DedupResult(
                    action=DedupAction.ADD,
                    reason="Borderline similarity, Tier 3 disabled",
                    similarity_score=result.similarity_score,
                )

            # Low similarity, no match
            return result

        return DedupResult(action=DedupAction.ADD, reason="No duplicates found")

    def _tier1_hash_check(self, piece: KnowledgePiece) -> DedupResult:
        """Tier 1: Check for exact hash match."""
        if piece.content_hash is None:
            piece.content_hash = piece._compute_content_hash()

        existing = self.piece_store.find_by_content_hash(
            piece.content_hash, piece.entity_id
        )

        if existing:
            return DedupResult(
                action=DedupAction.NO_OP,
                reason="Exact content hash match",
                existing_piece_id=existing.piece_id,
            )

        return DedupResult(action=DedupAction.ADD, reason="No hash match")

    def _tier2_embedding_check(
        self, piece: KnowledgePiece
    ) -> Tuple[DedupResult, Optional[KnowledgePiece]]:
        """Tier 2: Check embedding similarity."""
        if piece.embedding is None:
            text = piece.embedding_text or piece.content
            piece.embedding = self.embedding_fn(text)

        similar = self.piece_store.search(
            query=piece.embedding_text or piece.content,
            entity_id=piece.entity_id,
            top_k=5,
        )

        if not similar:
            return (
                DedupResult(action=DedupAction.ADD, reason="No similar pieces"),
                None,
            )

        top_piece, top_score = similar[0]

        if top_score > self.config.auto_dedup_threshold:
            return DedupResult(
                action=DedupAction.NO_OP,
                reason=f"High similarity: {top_score:.3f}",
                existing_piece_id=top_piece.piece_id,
                similarity_score=top_score,
            ), top_piece

        if top_score < self.config.llm_judge_threshold:
            return DedupResult(
                action=DedupAction.ADD,
                reason=f"Low similarity: {top_score:.3f}",
                similarity_score=top_score,
            ), None

        return DedupResult(
            action=DedupAction.ADD,
            reason="Borderline similarity, needs LLM judge",
            similarity_score=top_score,
        ), top_piece

    def _tier3_llm_judge(
        self,
        new_piece: KnowledgePiece,
        existing_piece: KnowledgePiece,
        similarity: float,
    ) -> DedupResult:
        """Tier 3: LLM judge for borderline cases."""
        if self.llm_fn is None:
            logger.warning("No LLM function provided for Tier 3. Defaulting to ADD.")
            return DedupResult(
                action=DedupAction.ADD,
                reason="No LLM function available",
                similarity_score=similarity,
            )

        prompt = DEDUP_LLM_JUDGE_PROMPT.format(
            similarity=similarity,
            existing_content=existing_piece.content[:500],
            existing_domain=existing_piece.domain,
            existing_tags=", ".join(existing_piece.tags),
            existing_created_at=existing_piece.created_at or "unknown",
            new_content=new_piece.content[:500],
            new_domain=new_piece.domain,
            new_tags=", ".join(new_piece.tags),
        )

        try:
            response = self.llm_fn(prompt)
            parsed = json.loads(response)

            # Safely parse action enum
            try:
                action = DedupAction(parsed.get("action", "add").lower())
            except ValueError:
                logger.warning(
                    "Invalid action from LLM: %s. Defaulting to ADD.",
                    parsed.get("action"),
                )
                action = DedupAction.ADD

        except Exception as e:
            logger.warning("LLM Judge failed: %s. Defaulting to ADD.", e)
            return DedupResult(
                action=DedupAction.ADD,
                reason=f"LLM Judge error: {e}",
                similarity_score=similarity,
            )

        return DedupResult(
            action=action,
            reason=parsed.get("reasoning", ""),
            existing_piece_id=(
                existing_piece.piece_id if action != DedupAction.ADD else None
            ),
            similarity_score=similarity,
            contradiction_detected=parsed.get("contradiction_detected", False),
        )

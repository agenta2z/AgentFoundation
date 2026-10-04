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

from agent_foundation.common.inferencers.agentic_functions import (
    agentic_function,
    AgenticOutput,
)
from agent_foundation.common.inferencers.function_inferencer import FunctionInferencer
from agent_foundation.knowledge.prompt_templates import render_prompt
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
        self._judge: Callable[..., DedupResult] = _build_dedup_judge(
            self._invoke_llm_fn
        )

    def _invoke_llm_fn(self, prompt: str) -> str:
        # Read llm_fn dynamically (honors a post-construction swap, as the former
        # inline call did); _tier3_llm_judge guards that it is set before use.
        llm_fn = self.llm_fn
        if llm_fn is None:
            raise RuntimeError("Tier 3 judge invoked without an llm_fn")
        return llm_fn(prompt)

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

        prompt = render_prompt(
            "quality/DedupJudge",
            similarity=f"{similarity:.3f}",
            existing_content=existing_piece.content[:500],
            existing_domain=existing_piece.domain,
            existing_tags=", ".join(existing_piece.tags),
            existing_created_at=existing_piece.created_at or "unknown",
            new_content=new_piece.content[:500],
            new_domain=new_piece.domain,
            new_tags=", ".join(new_piece.tags),
        )

        try:
            return self._judge(
                prompt,
                existing_piece_id=existing_piece.piece_id,
                similarity=similarity,
            )
        except Exception as e:
            logger.warning("LLM Judge failed: %s. Defaulting to ADD.", e)
            return DedupResult(
                action=DedupAction.ADD,
                reason=f"LLM Judge error: {e}",
                similarity_score=similarity,
            )


def _build_dedup_judge(
    llm_callable: Callable[[str], str],
) -> Callable[..., DedupResult]:
    """Build the Tier-3 judge as an ``@agentic_function`` over a text LLM.

    ``FunctionInferencer`` adapts the plain ``prompt -> reply`` callable into an
    inferencer, so the already-rendered DedupJudge prompt is passed through
    verbatim (``{{ prompt }}``) and the raw string reply arrives in the body as
    ``response``. The decode mirrors the former inline judge exactly: an
    unrecognized action falls back to ADD, while a malformed reply raises out to
    ``_tier3_llm_judge``'s fail-closed handler.
    """

    @agentic_function(
        inferencer=FunctionInferencer(func=llm_callable),
        template_string="{{ prompt }}",
    )
    def judge(
        prompt: str,
        *,
        existing_piece_id: Optional[str],
        similarity: float,
        response: AgenticOutput,
    ) -> DedupResult:
        parsed = json.loads(response.text)
        try:
            action = DedupAction(parsed.get("action", "add").lower())
        except ValueError:
            logger.warning(
                "Invalid action from LLM: %s. Defaulting to ADD.",
                parsed.get("action"),
            )
            action = DedupAction.ADD
        return DedupResult(
            action=action,
            reason=parsed.get("reasoning", ""),
            existing_piece_id=(
                existing_piece_id if action != DedupAction.ADD else None
            ),
            similarity_score=similarity,
            contradiction_detected=parsed.get("contradiction_detected", False),
        )

    return judge

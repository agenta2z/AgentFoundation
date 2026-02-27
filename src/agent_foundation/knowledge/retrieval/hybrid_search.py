"""
Hybrid Search with Reciprocal Rank Fusion (RRF).

Combines vector similarity search and keyword (BM25) search using
RRF fusion for improved retrieval quality.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    rrf_k: int = 60
    candidate_multiplier: int = 3


class HybridRetriever:
    """Hybrid retrieval using RRF fusion.

    Note: domain and include_global filtering should be handled by the
    caller before passing search functions, as KnowledgePieceStore.search()
    doesn't natively support these parameters.
    """

    def __init__(
        self,
        vector_search_fn: Callable[..., List[Tuple[KnowledgePiece, float]]],
        keyword_search_fn: Callable[..., List[Tuple[KnowledgePiece, float]]],
        config: Optional[HybridSearchConfig] = None,
    ):
        self.vector_search_fn = vector_search_fn
        self.keyword_search_fn = keyword_search_fn
        self.config = config or HybridSearchConfig()

    def search(
        self,
        query: str,
        top_k: int = 10,
        entity_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ScoredPiece]:
        """Perform hybrid search with RRF fusion."""
        fetch_k = top_k * self.config.candidate_multiplier

        try:
            vector_results = self.vector_search_fn(
                query=query,
                entity_id=entity_id,
                tags=tags,
                top_k=fetch_k,
            )
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            vector_results = []

        try:
            keyword_results = self.keyword_search_fn(
                query=query,
                entity_id=entity_id,
                tags=tags,
                top_k=fetch_k,
            )
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)
            keyword_results = []

        # RRF Fusion
        scores: Dict[str, float] = {}
        piece_map: Dict[str, KnowledgePiece] = {}

        for rank, (piece, _score) in enumerate(vector_results):
            rrf_score = self.config.vector_weight / (self.config.rrf_k + rank + 1)
            scores[piece.piece_id] = scores.get(piece.piece_id, 0) + rrf_score
            piece_map[piece.piece_id] = piece

        for rank, (piece, _score) in enumerate(keyword_results):
            rrf_score = self.config.keyword_weight / (self.config.rrf_k + rank + 1)
            scores[piece.piece_id] = scores.get(piece.piece_id, 0) + rrf_score
            piece_map[piece.piece_id] = piece

        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])

        return [
            ScoredPiece(piece=piece_map[pid], score=scores[pid])
            for pid in sorted_ids[:top_k]
        ]

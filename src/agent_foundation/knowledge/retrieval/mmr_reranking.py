"""
Maximal Marginal Relevance (MMR) for diversity re-ranking.

Uses pure Python cosine_similarity from utils.py (no numpy).
"""

from dataclasses import dataclass
from typing import List

from agent_foundation.knowledge.retrieval.models.results import ScoredPiece
from agent_foundation.knowledge.retrieval.utils import cosine_similarity


@dataclass
class MMRConfig:
    """Configuration for MMR re-ranking."""

    enabled: bool = True
    lambda_param: float = 0.7


def apply_mmr_reranking(
    pieces: List[ScoredPiece],
    config: MMRConfig,
    top_k: int = 10,
) -> List[ScoredPiece]:
    """Apply MMR diversity re-ranking using pure Python cosine similarity.

    Greedy selection: at each step, pick the piece maximizing
    ``lambda * relevance - (1 - lambda) * max_similarity_to_selected``.

    Pieces without embeddings are skipped during diversity calculation
    and appended after embedding-based selection.

    When MMR is disabled or the input list is smaller than top_k,
    the input list is returned truncated to top_k.
    """
    if not config.enabled or len(pieces) <= top_k:
        return pieces[:top_k]

    pieces_with_emb = [p for p in pieces if p.piece.embedding is not None]
    pieces_without_emb = [p for p in pieces if p.piece.embedding is None]

    if not pieces_with_emb:
        return pieces[:top_k]

    # Normalize scores to [0, 1]
    min_score = min(p.score for p in pieces_with_emb)
    max_score = max(p.score for p in pieces_with_emb)
    score_range = max_score - min_score

    if score_range > 0:
        for p in pieces_with_emb:
            p.normalized_score = (p.score - min_score) / score_range
    else:
        for p in pieces_with_emb:
            p.normalized_score = 1.0

    # MMR selection
    selected: List[ScoredPiece] = []
    remaining = pieces_with_emb.copy()

    while len(selected) < top_k and remaining:
        mmr_scores = []

        for piece in remaining:
            relevance = piece.normalized_score

            if selected:
                similarities = [
                    cosine_similarity(piece.piece.embedding, s.piece.embedding)
                    for s in selected
                    if s.piece.embedding is not None
                ]
                max_sim = max(similarities) if similarities else 0
            else:
                max_sim = 0

            mmr = config.lambda_param * relevance - (1 - config.lambda_param) * max_sim
            mmr_scores.append((piece, mmr))

        best_piece, _ = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_piece)
        remaining.remove(best_piece)

    # Append pieces without embeddings after embedding-based selection
    if len(selected) < top_k:
        selected.extend(pieces_without_emb[: top_k - len(selected)])

    return selected

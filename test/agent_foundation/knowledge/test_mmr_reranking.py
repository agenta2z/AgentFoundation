"""
Unit tests for MMR diversity re-ranking.

Tests cover:
- MMRConfig defaults and custom values
- Greedy MMR selection with embeddings
- Handling of pieces without embeddings
- Passthrough when disabled or input <= top_k
- Score normalization
"""
import sys
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from agent_foundation.knowledge.retrieval.mmr_reranking import (
    MMRConfig,
    apply_mmr_reranking,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece


def _make_scored_piece(
    piece_id: str,
    score: float,
    embedding=None,
    content: str = "test",
) -> ScoredPiece:
    piece = KnowledgePiece(content=content, piece_id=piece_id, embedding=embedding)
    return ScoredPiece(piece=piece, score=score)


class TestMMRConfig:
    def test_default_values(self):
        config = MMRConfig()
        assert config.enabled is True
        assert config.lambda_param == 0.7

    def test_custom_values(self):
        config = MMRConfig(enabled=False, lambda_param=0.5)
        assert config.enabled is False
        assert config.lambda_param == 0.5


class TestApplyMMRReranking:
    def test_disabled_returns_truncated(self):
        """When MMR is disabled, return input truncated to top_k."""
        pieces = [_make_scored_piece(f"p{i}", 1.0 - i * 0.1, embedding=[1.0, 0.0]) for i in range(5)]
        config = MMRConfig(enabled=False)
        result = apply_mmr_reranking(pieces, config, top_k=3)
        assert len(result) == 3
        assert [r.piece_id for r in result] == ["p0", "p1", "p2"]

    def test_input_smaller_than_top_k_returns_all(self):
        """When input length <= top_k, return input as-is."""
        pieces = [_make_scored_piece("p0", 0.9, embedding=[1.0, 0.0])]
        config = MMRConfig()
        result = apply_mmr_reranking(pieces, config, top_k=5)
        assert len(result) == 1
        assert result[0].piece_id == "p0"

    def test_input_equal_to_top_k_returns_all(self):
        """When input length == top_k, return input as-is."""
        pieces = [_make_scored_piece(f"p{i}", 0.9 - i * 0.1, embedding=[1.0, 0.0]) for i in range(3)]
        config = MMRConfig()
        result = apply_mmr_reranking(pieces, config, top_k=3)
        assert len(result) == 3

    def test_empty_input(self):
        """Empty input returns empty output."""
        config = MMRConfig()
        result = apply_mmr_reranking([], config, top_k=5)
        assert result == []

    def test_all_pieces_without_embeddings(self):
        """When no pieces have embeddings, return input truncated to top_k."""
        pieces = [_make_scored_piece(f"p{i}", 1.0 - i * 0.1) for i in range(5)]
        config = MMRConfig()
        result = apply_mmr_reranking(pieces, config, top_k=3)
        assert len(result) == 3

    def test_mmr_selects_diverse_pieces(self):
        """MMR should prefer diverse pieces over redundant ones."""
        # p0 and p1 have identical embeddings (redundant), p2 is different
        p0 = _make_scored_piece("p0", 1.0, embedding=[1.0, 0.0])
        p1 = _make_scored_piece("p1", 0.95, embedding=[1.0, 0.0])
        p2 = _make_scored_piece("p2", 0.9, embedding=[0.0, 1.0])

        config = MMRConfig(lambda_param=0.5)
        result = apply_mmr_reranking([p0, p1, p2], config, top_k=2)

        assert len(result) == 2
        # p0 should be first (highest score), p2 should be second (diverse)
        assert result[0].piece_id == "p0"
        assert result[1].piece_id == "p2"

    def test_high_lambda_prefers_relevance(self):
        """With high lambda, relevance dominates over diversity."""
        p0 = _make_scored_piece("p0", 1.0, embedding=[1.0, 0.0])
        p1 = _make_scored_piece("p1", 0.99, embedding=[1.0, 0.0])  # redundant but high score
        p2 = _make_scored_piece("p2", 0.5, embedding=[0.0, 1.0])  # diverse but low score

        config = MMRConfig(lambda_param=0.99)
        result = apply_mmr_reranking([p0, p1, p2], config, top_k=2)

        assert result[0].piece_id == "p0"
        assert result[1].piece_id == "p1"

    def test_pieces_without_embeddings_appended_after(self):
        """Pieces without embeddings are appended after embedding-based selection."""
        p0 = _make_scored_piece("p0", 1.0, embedding=[1.0, 0.0])
        p1 = _make_scored_piece("p1", 0.9)  # no embedding
        p2 = _make_scored_piece("p2", 0.8, embedding=[0.0, 1.0])
        p3 = _make_scored_piece("p3", 0.7, embedding=[0.5, 0.5])

        config = MMRConfig()
        result = apply_mmr_reranking([p0, p1, p2, p3], config, top_k=3)

        assert len(result) == 3
        # MMR selects from embedding pieces first; p1 (no embedding) only fills remaining slots
        result_ids = [r.piece_id for r in result]
        # All 3 slots should be filled by embedding pieces since there are 3 with embeddings
        assert "p1" not in result_ids

    def test_output_length_respects_top_k(self):
        """Output never exceeds top_k."""
        pieces = [_make_scored_piece(f"p{i}", 1.0 - i * 0.05, embedding=[float(i), 1.0]) for i in range(10)]
        config = MMRConfig()
        result = apply_mmr_reranking(pieces, config, top_k=3)
        assert len(result) == 3

    def test_all_output_pieces_from_input(self):
        """Every piece in the output must be present in the input."""
        pieces = [_make_scored_piece(f"p{i}", 1.0 - i * 0.1, embedding=[float(i), 1.0]) for i in range(5)]
        config = MMRConfig()
        result = apply_mmr_reranking(pieces, config, top_k=3)

        input_ids = {p.piece_id for p in pieces}
        for r in result:
            assert r.piece_id in input_ids

    def test_no_embedding_pieces_fill_remaining_slots(self):
        """When MMR exhausts embedding pieces, no-embedding pieces fill remaining slots."""
        p0 = _make_scored_piece("p0", 1.0, embedding=[1.0, 0.0])
        p1 = _make_scored_piece("p1", 0.9)  # no embedding
        p2 = _make_scored_piece("p2", 0.8)  # no embedding
        p3 = _make_scored_piece("p3", 0.7, embedding=[0.0, 1.0])

        config = MMRConfig()
        # top_k=3 but only 2 embedding pieces, so 1 no-embedding piece fills the gap
        result = apply_mmr_reranking([p0, p1, p2, p3], config, top_k=3)

        assert len(result) == 3
        result_ids = [r.piece_id for r in result]
        # Both embedding pieces should be selected first
        assert "p0" in result_ids
        assert "p3" in result_ids
        # One no-embedding piece fills the last slot (p1 comes first in the no-emb list)
        assert "p1" in result_ids

    def test_score_normalization_uniform_scores(self):
        """When all scores are equal, normalized scores should be 1.0."""
        pieces = [_make_scored_piece(f"p{i}", 0.5, embedding=[float(i), 1.0]) for i in range(5)]
        config = MMRConfig()
        result = apply_mmr_reranking(pieces, config, top_k=3)

        assert len(result) == 3
        # All normalized scores should be 1.0 since all input scores are equal
        for r in result:
            assert r.normalized_score == 1.0

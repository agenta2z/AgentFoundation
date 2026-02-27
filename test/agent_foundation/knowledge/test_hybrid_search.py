"""
Unit tests for HybridRetriever and HybridSearchConfig.

Tests cover:
- RRF score computation and merging
- Graceful handling of search function failures
- Result ordering and top_k limiting
- Default config values
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

from agent_foundation.knowledge.retrieval.hybrid_search import (
    HybridRetriever,
    HybridSearchConfig,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece


def _make_piece(piece_id: str, content: str = "test") -> KnowledgePiece:
    return KnowledgePiece(content=content, piece_id=piece_id)


class TestHybridSearchConfig:
    def test_default_values(self):
        config = HybridSearchConfig()
        assert config.vector_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.rrf_k == 60
        assert config.candidate_multiplier == 3

    def test_custom_values(self):
        config = HybridSearchConfig(vector_weight=0.5, keyword_weight=0.5, rrf_k=30, candidate_multiplier=5)
        assert config.vector_weight == 0.5
        assert config.keyword_weight == 0.5
        assert config.rrf_k == 30
        assert config.candidate_multiplier == 5


class TestHybridRetriever:
    def test_basic_rrf_fusion(self):
        """Results from both search functions are fused with correct RRF scores."""
        p1 = _make_piece("p1")
        p2 = _make_piece("p2")

        def vector_fn(**kwargs):
            return [(p1, 0.9), (p2, 0.7)]

        def keyword_fn(**kwargs):
            return [(p2, 0.8), (p1, 0.6)]

        config = HybridSearchConfig(vector_weight=0.7, keyword_weight=0.3, rrf_k=60)
        retriever = HybridRetriever(vector_fn, keyword_fn, config)
        results = retriever.search("test query", top_k=10)

        assert len(results) == 2

        # p1: vector rank 0 + keyword rank 1
        expected_p1 = 0.7 / (60 + 0 + 1) + 0.3 / (60 + 1 + 1)
        # p2: vector rank 1 + keyword rank 0
        expected_p2 = 0.7 / (60 + 1 + 1) + 0.3 / (60 + 0 + 1)

        scores = {r.piece_id: r.score for r in results}
        assert abs(scores["p1"] - expected_p1) < 1e-10
        assert abs(scores["p2"] - expected_p2) < 1e-10

    def test_results_sorted_descending(self):
        """Results are sorted by descending fused score."""
        p1 = _make_piece("p1")
        p2 = _make_piece("p2")
        p3 = _make_piece("p3")

        def vector_fn(**kwargs):
            return [(p1, 0.9), (p2, 0.7), (p3, 0.5)]

        def keyword_fn(**kwargs):
            return [(p3, 0.9), (p2, 0.7), (p1, 0.5)]

        retriever = HybridRetriever(vector_fn, keyword_fn)
        results = retriever.search("test", top_k=10)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self):
        """Output is limited to top_k results."""
        pieces = [_make_piece(f"p{i}") for i in range(10)]

        def vector_fn(**kwargs):
            return [(p, 1.0 - i * 0.1) for i, p in enumerate(pieces)]

        def keyword_fn(**kwargs):
            return []

        retriever = HybridRetriever(vector_fn, keyword_fn)
        results = retriever.search("test", top_k=3)

        assert len(results) == 3

    def test_vector_search_failure_uses_keyword_only(self):
        """When vector search fails, keyword results are still returned."""
        p1 = _make_piece("p1")

        def vector_fn(**kwargs):
            raise RuntimeError("Vector search unavailable")

        def keyword_fn(**kwargs):
            return [(p1, 0.8)]

        retriever = HybridRetriever(vector_fn, keyword_fn)
        results = retriever.search("test", top_k=10)

        assert len(results) == 1
        assert results[0].piece_id == "p1"

    def test_keyword_search_failure_uses_vector_only(self):
        """When keyword search fails, vector results are still returned."""
        p1 = _make_piece("p1")

        def vector_fn(**kwargs):
            return [(p1, 0.9)]

        def keyword_fn(**kwargs):
            raise RuntimeError("Keyword search unavailable")

        retriever = HybridRetriever(vector_fn, keyword_fn)
        results = retriever.search("test", top_k=10)

        assert len(results) == 1
        assert results[0].piece_id == "p1"

    def test_both_searches_fail_returns_empty(self):
        """When both search functions fail, an empty list is returned."""

        def vector_fn(**kwargs):
            raise RuntimeError("fail")

        def keyword_fn(**kwargs):
            raise RuntimeError("fail")

        retriever = HybridRetriever(vector_fn, keyword_fn)
        results = retriever.search("test", top_k=10)

        assert results == []

    def test_piece_in_both_results_scores_summed(self):
        """A piece appearing in both result sets has its RRF scores summed."""
        p1 = _make_piece("p1")

        def vector_fn(**kwargs):
            return [(p1, 0.9)]

        def keyword_fn(**kwargs):
            return [(p1, 0.8)]

        config = HybridSearchConfig(vector_weight=0.7, keyword_weight=0.3, rrf_k=60)
        retriever = HybridRetriever(vector_fn, keyword_fn, config)
        results = retriever.search("test", top_k=10)

        assert len(results) == 1
        expected = 0.7 / (60 + 0 + 1) + 0.3 / (60 + 0 + 1)
        assert abs(results[0].score - expected) < 1e-10

    def test_candidate_multiplier_affects_fetch_k(self):
        """The candidate_multiplier scales the number of results fetched from each search fn."""
        captured_top_k = []

        def vector_fn(**kwargs):
            captured_top_k.append(kwargs.get("top_k"))
            return []

        def keyword_fn(**kwargs):
            captured_top_k.append(kwargs.get("top_k"))
            return []

        config = HybridSearchConfig(candidate_multiplier=5)
        retriever = HybridRetriever(vector_fn, keyword_fn, config)
        retriever.search("test", top_k=10)

        assert captured_top_k == [50, 50]

    def test_default_config_when_none(self):
        """When config is None, default HybridSearchConfig is used."""
        retriever = HybridRetriever(lambda **kw: [], lambda **kw: [], config=None)
        assert retriever.config.vector_weight == 0.7
        assert retriever.config.rrf_k == 60

    def test_empty_results_from_both(self):
        """When both search functions return empty lists, result is empty."""
        retriever = HybridRetriever(lambda **kw: [], lambda **kw: [])
        results = retriever.search("test", top_k=10)
        assert results == []

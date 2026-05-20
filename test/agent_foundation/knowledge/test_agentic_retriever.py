"""
Unit tests for AgenticRetriever, SubQuery, AgenticRetrievalResult,
create_domain_decomposer, and create_llm_decomposer.

Tests cover:
- Score aggregation strategies (max, sum, weighted_sum)
- Fallback to unfiltered search when results are insufficient
- Domain-based decomposer factory
- LLM-backed decomposer factory with error handling
- SubQuery and AgenticRetrievalResult defaults
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from agent_foundation.knowledge.retrieval.agentic_retriever import (
    AgenticRetriever,
    AgenticRetrievalResult,
    SubQuery,
    create_domain_decomposer,
    create_llm_decomposer,
)
from agent_foundation.knowledge.retrieval.formatter import RetrievalResult
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece


def _make_piece(piece_id: str, content: str = "test") -> KnowledgePiece:
    return KnowledgePiece(content=content, piece_id=piece_id)


def _make_kb_mock(pieces: List[Tuple[KnowledgePiece, float]]):
    """Create a mock KnowledgeBase that returns the given pieces."""
    kb = MagicMock()
    result = RetrievalResult()
    result.pieces = pieces
    kb.retrieve.return_value = result
    return kb


def _make_kb_mock_per_call(call_results: List[List[Tuple[KnowledgePiece, float]]]):
    """Create a mock KB that returns different results on successive calls."""
    kb = MagicMock()
    results = []
    for pieces in call_results:
        r = RetrievalResult()
        r.pieces = pieces
        results.append(r)
    kb.retrieve.side_effect = results
    return kb


class TestSubQuery:
    def test_defaults(self):
        sq = SubQuery(query="test")
        assert sq.query == "test"
        assert sq.domain is None
        assert sq.tags is None
        assert sq.weight == 1.0

    def test_custom_values(self):
        sq = SubQuery(query="q", domain="ml", tags=["gpu"], weight=0.8)
        assert sq.domain == "ml"
        assert sq.tags == ["gpu"]
        assert sq.weight == 0.8


class TestAgenticRetrievalResult:
    def test_defaults(self):
        r = AgenticRetrievalResult()
        assert r.pieces == []
        assert r.sub_queries == []
        assert r.used_fallback is False


class TestAgenticRetriever:
    def test_no_decomposer_single_query(self):
        """Without a decomposer, the original query is used as-is."""
        p1 = _make_piece("p1")
        kb = _make_kb_mock([(p1, 0.9)])

        retriever = AgenticRetriever(kb=kb, min_results=1)
        result = retriever.retrieve("test query")

        assert len(result.pieces) == 1
        assert result.pieces[0].piece.piece_id == "p1"
        assert result.used_fallback is False
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0].query == "test query"

    def test_with_decomposer(self):
        """Decomposer splits query into multiple sub-queries."""
        p1 = _make_piece("p1")
        p2 = _make_piece("p2")

        # First call returns p1, second returns p2, third (fallback) not needed
        kb = _make_kb_mock_per_call([
            [(p1, 0.9)],
            [(p2, 0.8)],
        ])

        def decomposer(query):
            return [
                SubQuery(query="sub1", domain="d1"),
                SubQuery(query="sub2", domain="d2"),
            ]

        retriever = AgenticRetriever(
            kb=kb, query_decomposer=decomposer, min_results=1
        )
        result = retriever.retrieve("complex query")

        assert len(result.pieces) == 2
        assert result.used_fallback is False
        assert len(result.sub_queries) == 2

    def test_aggregation_max(self):
        """Max aggregation takes the highest weighted score per piece."""
        p1 = _make_piece("p1")

        # Same piece returned by two sub-queries with different scores
        kb = _make_kb_mock_per_call([
            [(p1, 0.9)],
            [(p1, 0.7)],
        ])

        def decomposer(query):
            return [SubQuery(query="a"), SubQuery(query="b")]

        retriever = AgenticRetriever(
            kb=kb,
            query_decomposer=decomposer,
            aggregation_strategy="max",
            min_results=1,
        )
        result = retriever.retrieve("test")

        assert len(result.pieces) == 1
        assert abs(result.pieces[0].score - 0.9) < 1e-10

    def test_aggregation_sum(self):
        """Sum aggregation adds all weighted scores per piece."""
        p1 = _make_piece("p1")

        kb = _make_kb_mock_per_call([
            [(p1, 0.9)],
            [(p1, 0.7)],
        ])

        def decomposer(query):
            return [SubQuery(query="a"), SubQuery(query="b")]

        retriever = AgenticRetriever(
            kb=kb,
            query_decomposer=decomposer,
            aggregation_strategy="sum",
            min_results=1,
        )
        result = retriever.retrieve("test")

        assert len(result.pieces) == 1
        assert abs(result.pieces[0].score - 1.6) < 1e-10

    def test_aggregation_weighted_sum(self):
        """Weighted sum uses sub-query weights to scale scores."""
        p1 = _make_piece("p1")

        kb = _make_kb_mock_per_call([
            [(p1, 1.0)],
            [(p1, 1.0)],
        ])

        def decomposer(query):
            return [
                SubQuery(query="a", weight=2.0),
                SubQuery(query="b", weight=0.5),
            ]

        retriever = AgenticRetriever(
            kb=kb,
            query_decomposer=decomposer,
            aggregation_strategy="weighted_sum",
            min_results=1,
        )
        result = retriever.retrieve("test")

        assert len(result.pieces) == 1
        # 1.0 * 2.0 + 1.0 * 0.5 = 2.5
        assert abs(result.pieces[0].score - 2.5) < 1e-10

    def test_fallback_triggered(self):
        """Fallback is triggered when aggregated results < min_results."""
        p1 = _make_piece("p1")
        p2 = _make_piece("p2")
        p3 = _make_piece("p3")

        # Sub-query returns 1 piece, fallback returns 3
        kb = _make_kb_mock_per_call([
            [(p1, 0.9)],
            [(p1, 0.8), (p2, 0.7), (p3, 0.6)],
        ])

        retriever = AgenticRetriever(kb=kb, min_results=3)
        result = retriever.retrieve("test")

        assert result.used_fallback is True
        assert len(result.pieces) >= 1

    def test_fallback_not_triggered(self):
        """Fallback is not triggered when results >= min_results."""
        pieces = [_make_piece(f"p{i}") for i in range(5)]
        kb = _make_kb_mock([(p, 0.9 - i * 0.1) for i, p in enumerate(pieces)])

        retriever = AgenticRetriever(kb=kb, min_results=3)
        result = retriever.retrieve("test")

        assert result.used_fallback is False

    def test_top_k_limits_output(self):
        """Output is limited to top_k results."""
        pieces = [_make_piece(f"p{i}") for i in range(10)]
        kb = _make_kb_mock([(p, 1.0 - i * 0.05) for i, p in enumerate(pieces)])

        retriever = AgenticRetriever(kb=kb, top_k=3, min_results=1)
        result = retriever.retrieve("test")

        assert len(result.pieces) <= 3

    def test_results_sorted_descending(self):
        """Results are sorted by descending score."""
        p1 = _make_piece("p1")
        p2 = _make_piece("p2")
        p3 = _make_piece("p3")

        kb = _make_kb_mock([(p2, 0.5), (p1, 0.9), (p3, 0.3)])

        retriever = AgenticRetriever(kb=kb, min_results=1)
        result = retriever.retrieve("test")

        scores = [sp.score for sp in result.pieces]
        assert scores == sorted(scores, reverse=True)

    def test_merge_with_fallback_deduplicates(self):
        """Fallback merge deduplicates by piece_id, keeping higher score."""
        p1 = _make_piece("p1")

        # Sub-query returns p1 with score 0.5, fallback returns p1 with score 0.9
        kb = _make_kb_mock_per_call([
            [(p1, 0.5)],
            [(p1, 0.9)],
        ])

        retriever = AgenticRetriever(kb=kb, min_results=5)
        result = retriever.retrieve("test")

        assert result.used_fallback is True
        p1_results = [sp for sp in result.pieces if sp.piece.piece_id == "p1"]
        assert len(p1_results) == 1
        assert abs(p1_results[0].score - 0.9) < 1e-10

    def test_entity_id_passed_through(self):
        """entity_id is forwarded to the KnowledgeBase."""
        kb = _make_kb_mock([])
        retriever = AgenticRetriever(kb=kb, min_results=0)
        retriever.retrieve("test", entity_id="user123")

        call_args = kb.retrieve.call_args
        assert call_args.kwargs.get("entity_id") == "user123"


class TestCreateDomainDecomposer:
    def test_creates_one_subquery_per_domain(self):
        decomposer = create_domain_decomposer(["ml", "infra", "data"])
        sub_queries = decomposer("how to optimize training")

        assert len(sub_queries) == 3
        domains = [sq.domain for sq in sub_queries]
        assert domains == ["ml", "infra", "data"]

    def test_all_subqueries_use_original_query(self):
        decomposer = create_domain_decomposer(["ml", "infra"])
        sub_queries = decomposer("my query")

        for sq in sub_queries:
            assert sq.query == "my query"
            assert sq.weight == 1.0

    def test_empty_domains(self):
        decomposer = create_domain_decomposer([])
        sub_queries = decomposer("test")
        assert sub_queries == []


class TestCreateLlmDecomposer:
    def test_parses_llm_json_response(self):
        response = json.dumps([
            {"query": "sub1", "domain": "ml", "weight": 1.0},
            {"query": "sub2", "domain": "infra", "weight": 0.8},
        ])

        def llm_fn(prompt):
            return response

        decomposer = create_llm_decomposer(llm_fn, domains=["ml", "infra"])
        sub_queries = decomposer("complex query")

        assert len(sub_queries) == 2
        assert sub_queries[0].query == "sub1"
        assert sub_queries[0].domain == "ml"
        assert sub_queries[1].weight == 0.8

    def test_fallback_on_llm_failure(self):
        def llm_fn(prompt):
            raise RuntimeError("LLM unavailable")

        decomposer = create_llm_decomposer(llm_fn)
        sub_queries = decomposer("test query")

        assert len(sub_queries) == 1
        assert sub_queries[0].query == "test query"
        assert sub_queries[0].domain is None

    def test_fallback_on_invalid_json(self):
        def llm_fn(prompt):
            return "not valid json"

        decomposer = create_llm_decomposer(llm_fn)
        sub_queries = decomposer("test query")

        assert len(sub_queries) == 1
        assert sub_queries[0].query == "test query"

    def test_tags_parsed_from_llm_response(self):
        response = json.dumps([
            {"query": "sub1", "domain": "ml", "tags": ["gpu", "cuda"], "weight": 1.0},
        ])

        def llm_fn(prompt):
            return response

        decomposer = create_llm_decomposer(llm_fn)
        sub_queries = decomposer("test")

        assert sub_queries[0].tags == ["gpu", "cuda"]

    def test_no_domains_provided(self):
        response = json.dumps([{"query": "sub1", "weight": 1.0}])

        def llm_fn(prompt):
            return response

        decomposer = create_llm_decomposer(llm_fn)
        sub_queries = decomposer("test")

        assert len(sub_queries) == 1
        assert sub_queries[0].domain is None

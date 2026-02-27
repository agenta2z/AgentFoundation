"""
Property-based tests for AgenticRetriever score aggregation.

Feature: knowledge-module-migration
- Property 11: AgenticRetriever score aggregation

**Validates: Requirements 8.3**
"""
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

from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.retrieval.agentic_retriever import (
    AgenticRetriever,
    SubQuery,
)
from agent_foundation.knowledge.retrieval.formatter import RetrievalResult
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece


# ── Strategies ────────────────────────────────────────────────────────────────

_piece_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)


@st.composite
def sub_query_results(draw):
    """Generate sub-queries with known weights and pieces with known scores.

    Returns:
        - sub_queries: list of SubQuery objects with weights
        - per_subquery_pieces: list of lists of (KnowledgePiece, float) tuples
          (one list per sub-query, representing what the KB returns for each)
    """
    # Generate a pool of unique piece IDs
    all_ids = draw(
        st.lists(_piece_id_strategy, min_size=1, max_size=10, unique=True)
    )
    assume(len(all_ids) >= 1)

    pieces = {
        pid: KnowledgePiece(content=f"content_{pid}", piece_id=pid)
        for pid in all_ids
    }

    # Generate 1-4 sub-queries
    num_sub_queries = draw(st.integers(min_value=1, max_value=4))

    sub_queries = []
    per_subquery_pieces = []

    for _ in range(num_sub_queries):
        weight = draw(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
        )
        sq = SubQuery(query="test_query", weight=weight)
        sub_queries.append(sq)

        # Each sub-query returns a subset of pieces with scores
        num_pieces = draw(st.integers(min_value=0, max_value=len(all_ids)))
        chosen_ids = draw(
            st.lists(
                st.sampled_from(all_ids),
                min_size=num_pieces,
                max_size=num_pieces,
                unique=True,
            )
        )
        piece_scores = []
        for pid in chosen_ids:
            score = draw(
                st.floats(
                    min_value=0.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False,
                )
            )
            piece_scores.append((pieces[pid], score))
        per_subquery_pieces.append(piece_scores)

    # Ensure at least one sub-query returns at least one piece
    total_pieces = sum(len(ps) for ps in per_subquery_pieces)
    assume(total_pieces > 0)

    return sub_queries, per_subquery_pieces


def _make_kb_mock(per_subquery_pieces: List[List[Tuple[KnowledgePiece, float]]]):
    """Create a mock KnowledgeBase that returns specific pieces per sub-query call."""
    kb = MagicMock()
    results = []
    for pieces in per_subquery_pieces:
        r = RetrievalResult()
        r.pieces = pieces
        results.append(r)
    # Add an extra empty result for potential fallback call
    fallback = RetrievalResult()
    fallback.pieces = []
    results.append(fallback)
    kb.retrieve.side_effect = results
    return kb


def _compute_expected_scores(
    sub_queries: List[SubQuery],
    per_subquery_pieces: List[List[Tuple[KnowledgePiece, float]]],
    strategy: str,
) -> Dict[str, float]:
    """Manually compute expected aggregated scores.

    Mirrors the logic in AgenticRetriever._aggregate_scores:
    - For each sub-query, weighted_score = score * weight
    - "max": final = max of weighted scores across sub-queries
    - "sum" or "weighted_sum": final = sum of weighted scores
    """
    piece_weighted_scores: Dict[str, List[float]] = {}

    for sq, pieces in zip(sub_queries, per_subquery_pieces):
        for piece, score in pieces:
            pid = piece.piece_id
            if pid not in piece_weighted_scores:
                piece_weighted_scores[pid] = []
            piece_weighted_scores[pid].append(score * sq.weight)

    expected = {}
    for pid, scores in piece_weighted_scores.items():
        if strategy == "max":
            expected[pid] = max(scores)
        else:  # sum or weighted_sum
            expected[pid] = sum(scores)

    return expected


# ── Property 11: AgenticRetriever score aggregation ───────────────────────────


class TestAgenticRetrieverScoreAggregation:
    """Property 11: AgenticRetriever score aggregation.

    For any set of sub-query results with known scores and weights,
    when aggregation_strategy is "max", the final score for each piece
    should be the maximum of its weighted scores across sub-queries.
    When "sum" or "weighted_sum", the final score should be the sum
    of weighted scores.

    **Validates: Requirements 8.3**
    """

    @given(data=sub_query_results(), strategy=st.sampled_from(["max", "sum", "weighted_sum"]))
    @settings(max_examples=100)
    def test_aggregated_scores_match_expected(self, data, strategy):
        """Aggregated scores match manually computed expected values for all strategies.

        **Validates: Requirements 8.3**
        """
        sub_queries, per_subquery_pieces = data

        kb = _make_kb_mock(per_subquery_pieces)

        def decomposer(query: str) -> List[SubQuery]:
            return sub_queries

        retriever = AgenticRetriever(
            kb=kb,
            query_decomposer=decomposer,
            aggregation_strategy=strategy,
            min_results=0,  # disable fallback
            top_k=100,
        )
        result = retriever.retrieve("test")

        expected = _compute_expected_scores(sub_queries, per_subquery_pieces, strategy)

        # Every piece in the result should have the expected score
        for sp in result.pieces:
            pid = sp.piece.piece_id
            assert pid in expected, f"Unexpected piece {pid} in results"
            assert abs(sp.score - expected[pid]) < 1e-9, (
                f"Score mismatch for {pid} with strategy={strategy}: "
                f"got {sp.score}, expected {expected[pid]}"
            )

        # Every expected piece should be in the result
        result_ids = {sp.piece.piece_id for sp in result.pieces}
        for pid in expected:
            assert pid in result_ids, (
                f"Expected piece {pid} missing from results with strategy={strategy}"
            )

    @given(data=sub_query_results())
    @settings(max_examples=100)
    def test_max_strategy_picks_maximum_weighted_score(self, data):
        """With "max" strategy, each piece's score is the max of its weighted scores.

        **Validates: Requirements 8.3**
        """
        sub_queries, per_subquery_pieces = data

        kb = _make_kb_mock(per_subquery_pieces)

        def decomposer(query: str) -> List[SubQuery]:
            return sub_queries

        retriever = AgenticRetriever(
            kb=kb,
            query_decomposer=decomposer,
            aggregation_strategy="max",
            min_results=0,
            top_k=100,
        )
        result = retriever.retrieve("test")

        expected = _compute_expected_scores(sub_queries, per_subquery_pieces, "max")

        for sp in result.pieces:
            pid = sp.piece.piece_id
            assert abs(sp.score - expected[pid]) < 1e-9

    @given(data=sub_query_results())
    @settings(max_examples=100)
    def test_sum_strategy_sums_weighted_scores(self, data):
        """With "sum" strategy, each piece's score is the sum of its weighted scores.

        **Validates: Requirements 8.3**
        """
        sub_queries, per_subquery_pieces = data

        kb = _make_kb_mock(per_subquery_pieces)

        def decomposer(query: str) -> List[SubQuery]:
            return sub_queries

        retriever = AgenticRetriever(
            kb=kb,
            query_decomposer=decomposer,
            aggregation_strategy="sum",
            min_results=0,
            top_k=100,
        )
        result = retriever.retrieve("test")

        expected = _compute_expected_scores(sub_queries, per_subquery_pieces, "sum")

        for sp in result.pieces:
            pid = sp.piece.piece_id
            assert abs(sp.score - expected[pid]) < 1e-9

    @given(data=sub_query_results())
    @settings(max_examples=100)
    def test_weighted_sum_equals_sum_behavior(self, data):
        """With "weighted_sum" strategy, behavior matches "sum" (both sum weighted scores).

        **Validates: Requirements 8.3**
        """
        sub_queries, per_subquery_pieces = data

        kb = _make_kb_mock(per_subquery_pieces)

        def decomposer(query: str) -> List[SubQuery]:
            return sub_queries

        retriever = AgenticRetriever(
            kb=kb,
            query_decomposer=decomposer,
            aggregation_strategy="weighted_sum",
            min_results=0,
            top_k=100,
        )
        result = retriever.retrieve("test")

        expected = _compute_expected_scores(sub_queries, per_subquery_pieces, "weighted_sum")

        for sp in result.pieces:
            pid = sp.piece.piece_id
            assert abs(sp.score - expected[pid]) < 1e-9

    @given(data=sub_query_results(), strategy=st.sampled_from(["max", "sum", "weighted_sum"]))
    @settings(max_examples=100)
    def test_results_sorted_descending(self, data, strategy):
        """Aggregated results are sorted by descending score.

        **Validates: Requirements 8.3**
        """
        sub_queries, per_subquery_pieces = data

        kb = _make_kb_mock(per_subquery_pieces)

        def decomposer(query: str) -> List[SubQuery]:
            return sub_queries

        retriever = AgenticRetriever(
            kb=kb,
            query_decomposer=decomposer,
            aggregation_strategy=strategy,
            min_results=0,
            top_k=100,
        )
        result = retriever.retrieve("test")

        scores = [sp.score for sp in result.pieces]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending with strategy={strategy}: {scores}"
        )

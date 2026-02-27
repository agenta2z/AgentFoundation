"""
Property-based tests for HybridRetriever RRF score computation.

Feature: knowledge-module-migration
- Property 7: HybridRetriever RRF score computation

**Validates: Requirements 5.2, 5.3, 5.4**
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

from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.retrieval.hybrid_search import (
    HybridRetriever,
    HybridSearchConfig,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece


# ── Strategies ────────────────────────────────────────────────────────────────

_piece_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)


@st.composite
def unique_piece_ids(draw, min_size=1, max_size=10):
    """Generate a list of unique piece IDs."""
    ids = draw(
        st.lists(
            _piece_id_strategy,
            min_size=min_size,
            max_size=max_size,
            unique=True,
        )
    )
    return ids


@st.composite
def hybrid_search_inputs(draw):
    """Generate vector results, keyword results, config, and top_k for hybrid search.

    Produces two lists of (KnowledgePiece, float) tuples with unique piece_ids
    within each list, a HybridSearchConfig, and a top_k value.
    """
    # Generate a pool of unique piece IDs
    all_ids = draw(st.lists(_piece_id_strategy, min_size=1, max_size=15, unique=True))
    assume(len(all_ids) >= 1)

    # Split IDs into: vector-only, keyword-only, and shared
    indices = list(range(len(all_ids)))
    vector_indices = draw(
        st.lists(st.sampled_from(indices), min_size=0, max_size=len(indices), unique=True)
    )
    keyword_indices = draw(
        st.lists(st.sampled_from(indices), min_size=0, max_size=len(indices), unique=True)
    )
    # Ensure at least one result exists
    assume(len(vector_indices) > 0 or len(keyword_indices) > 0)

    # Build pieces
    pieces = {pid: KnowledgePiece(content=f"content_{pid}", piece_id=pid) for pid in all_ids}

    # Build result lists with scores
    score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    vector_results = [(pieces[all_ids[i]], draw(score_st)) for i in vector_indices]
    keyword_results = [(pieces[all_ids[i]], draw(score_st)) for i in keyword_indices]

    # Config
    vector_weight = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    keyword_weight = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    rrf_k = draw(st.integers(min_value=1, max_value=100))
    config = HybridSearchConfig(
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        rrf_k=rrf_k,
    )

    top_k = draw(st.integers(min_value=1, max_value=max(len(all_ids), 1)))

    return vector_results, keyword_results, config, top_k


# ── Property 7: HybridRetriever RRF score computation ────────────────────────


class TestHybridRetrieverRRFScoreComputation:
    """Property 7: HybridRetriever RRF score computation.

    For any set of vector results and keyword results, the fused score for a
    piece appearing in both lists should equal
    `vector_weight / (rrf_k + vector_rank + 1) + keyword_weight / (rrf_k + keyword_rank + 1)`.
    The output should be sorted by descending fused score and limited to top_k.

    **Validates: Requirements 5.2, 5.3, 5.4**
    """

    @given(data=hybrid_search_inputs())
    @settings(max_examples=100)
    def test_rrf_scores_match_formula(self, data):
        """Fused RRF scores match the expected formula for all pieces.

        **Validates: Requirements 5.2, 5.3**
        """
        vector_results, keyword_results, config, top_k = data

        def vector_fn(**kwargs):
            return vector_results

        def keyword_fn(**kwargs):
            return keyword_results

        retriever = HybridRetriever(vector_fn, keyword_fn, config)
        results = retriever.search("test", top_k=top_k)

        # Compute expected scores manually
        expected_scores = {}
        for rank, (piece, _score) in enumerate(vector_results):
            rrf = config.vector_weight / (config.rrf_k + rank + 1)
            expected_scores[piece.piece_id] = expected_scores.get(piece.piece_id, 0) + rrf

        for rank, (piece, _score) in enumerate(keyword_results):
            rrf = config.keyword_weight / (config.rrf_k + rank + 1)
            expected_scores[piece.piece_id] = expected_scores.get(piece.piece_id, 0) + rrf

        # Verify each result's score matches expected
        for scored_piece in results:
            pid = scored_piece.piece_id
            assert pid in expected_scores, f"Unexpected piece {pid} in results"
            assert abs(scored_piece.score - expected_scores[pid]) < 1e-9, (
                f"Score mismatch for {pid}: got {scored_piece.score}, expected {expected_scores[pid]}"
            )

    @given(data=hybrid_search_inputs())
    @settings(max_examples=100)
    def test_output_sorted_descending_by_score(self, data):
        """Output is sorted by descending fused score.

        **Validates: Requirements 5.4**
        """
        vector_results, keyword_results, config, top_k = data

        def vector_fn(**kwargs):
            return vector_results

        def keyword_fn(**kwargs):
            return keyword_results

        retriever = HybridRetriever(vector_fn, keyword_fn, config)
        results = retriever.search("test", top_k=top_k)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending: {scores}"
        )

    @given(data=hybrid_search_inputs())
    @settings(max_examples=100)
    def test_output_limited_to_top_k(self, data):
        """Output length is at most top_k.

        **Validates: Requirements 5.4**
        """
        vector_results, keyword_results, config, top_k = data

        def vector_fn(**kwargs):
            return vector_results

        def keyword_fn(**kwargs):
            return keyword_results

        retriever = HybridRetriever(vector_fn, keyword_fn, config)
        results = retriever.search("test", top_k=top_k)

        assert len(results) <= top_k, (
            f"Got {len(results)} results, expected at most {top_k}"
        )

    @given(data=hybrid_search_inputs())
    @settings(max_examples=100)
    def test_pieces_in_both_lists_have_summed_scores(self, data):
        """A piece appearing in both vector and keyword results has scores summed.

        **Validates: Requirements 5.2, 5.3**
        """
        vector_results, keyword_results, config, top_k = data

        # Find pieces that appear in both lists
        vector_ids = {piece.piece_id for piece, _ in vector_results}
        keyword_ids = {piece.piece_id for piece, _ in keyword_results}
        shared_ids = vector_ids & keyword_ids

        if not shared_ids:
            return  # nothing to check if no overlap

        def vector_fn(**kwargs):
            return vector_results

        def keyword_fn(**kwargs):
            return keyword_results

        retriever = HybridRetriever(vector_fn, keyword_fn, config)
        results = retriever.search("test", top_k=len(vector_ids | keyword_ids))

        result_map = {r.piece_id: r.score for r in results}

        for pid in shared_ids:
            if pid not in result_map:
                continue  # may have been cut by top_k

            # Compute expected: vector contribution + keyword contribution
            v_rank = next(i for i, (p, _) in enumerate(vector_results) if p.piece_id == pid)
            k_rank = next(i for i, (p, _) in enumerate(keyword_results) if p.piece_id == pid)
            expected = (
                config.vector_weight / (config.rrf_k + v_rank + 1)
                + config.keyword_weight / (config.rrf_k + k_rank + 1)
            )
            assert abs(result_map[pid] - expected) < 1e-9

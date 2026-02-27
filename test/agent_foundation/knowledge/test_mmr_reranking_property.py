"""
Property-based tests for MMR diversity re-ranking.

Feature: knowledge-module-migration
- Property 10: MMR output constraints

**Validates: Requirements 6.2, 6.4**
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

from agent_foundation.knowledge.retrieval.mmr_reranking import (
    apply_mmr_reranking,
    MMRConfig,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece


# ── Strategies ────────────────────────────────────────────────────────────────

_embedding_dim = st.shared(st.integers(min_value=2, max_value=8), key="emb_dim")

_bounded_float = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def embedding_vector(draw, dim=None):
    """Generate a bounded float vector of a given dimension."""
    d = dim if dim is not None else draw(_embedding_dim)
    return draw(st.lists(_bounded_float, min_size=d, max_size=d))


@st.composite
def scored_piece_with_embedding(draw, dim=None):
    """Generate a ScoredPiece whose KnowledgePiece has an embedding."""
    emb = draw(embedding_vector(dim=dim))
    piece = KnowledgePiece(
        content=draw(st.text(min_size=1, max_size=30).filter(lambda s: s.strip())),
        embedding=emb,
    )
    score = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    return ScoredPiece(piece=piece, score=score)


@st.composite
def scored_piece_without_embedding(draw):
    """Generate a ScoredPiece whose KnowledgePiece has no embedding."""
    piece = KnowledgePiece(
        content=draw(st.text(min_size=1, max_size=30).filter(lambda s: s.strip())),
        embedding=None,
    )
    score = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    return ScoredPiece(piece=piece, score=score)


@st.composite
def mmr_inputs(draw):
    """Generate a list of ScoredPieces (with embeddings), an MMRConfig, and top_k."""
    dim = draw(st.integers(min_value=2, max_value=8))
    n = draw(st.integers(min_value=1, max_value=15))
    pieces = draw(st.lists(scored_piece_with_embedding(dim=dim), min_size=n, max_size=n))
    config = MMRConfig(
        enabled=True,
        lambda_param=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    )
    top_k = draw(st.integers(min_value=1, max_value=max(len(pieces), 1)))
    return pieces, config, top_k


@st.composite
def mmr_inputs_mixed(draw):
    """Generate a mixed list of ScoredPieces (some with, some without embeddings)."""
    dim = draw(st.integers(min_value=2, max_value=8))
    with_emb = draw(st.lists(scored_piece_with_embedding(dim=dim), min_size=0, max_size=10))
    without_emb = draw(st.lists(scored_piece_without_embedding(), min_size=0, max_size=5))
    pieces = with_emb + without_emb
    assume(len(pieces) >= 1)
    config = MMRConfig(
        enabled=True,
        lambda_param=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    )
    top_k = draw(st.integers(min_value=1, max_value=max(len(pieces), 1)))
    return pieces, config, top_k


# ── Property 10: MMR output constraints ──────────────────────────────────────


class TestMMROutputConstraints:
    """Property 10: MMR output constraints.

    For any list of ScoredPieces with embeddings and an MMRConfig, the output
    of apply_mmr_reranking should have length <= top_k, and every piece in the
    output should be present in the input. When MMR is disabled or input length
    <= top_k, the output should equal the input truncated to top_k.

    **Validates: Requirements 6.2, 6.4**
    """

    @given(data=mmr_inputs())
    @settings(max_examples=100)
    def test_output_length_at_most_top_k(self, data):
        """Output length is always <= top_k.

        **Validates: Requirements 6.2**
        """
        pieces, config, top_k = data
        result = apply_mmr_reranking(pieces, config, top_k)
        assert len(result) <= top_k, (
            f"Got {len(result)} results, expected at most {top_k}"
        )

    @given(data=mmr_inputs_mixed())
    @settings(max_examples=100)
    def test_every_output_piece_in_input(self, data):
        """Every piece in the output is present in the input.

        **Validates: Requirements 6.2**
        """
        pieces, config, top_k = data
        result = apply_mmr_reranking(pieces, config, top_k)
        input_ids = {p.piece.piece_id for p in pieces}
        for sp in result:
            assert sp.piece.piece_id in input_ids, (
                f"Output piece {sp.piece.piece_id} not found in input"
            )

    @given(
        pieces=st.lists(scored_piece_with_embedding(dim=3), min_size=1, max_size=15),
        top_k=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_disabled_returns_input_truncated(self, pieces, top_k):
        """When MMR is disabled, output equals input[:top_k].

        **Validates: Requirements 6.4**
        """
        config = MMRConfig(enabled=False)
        result = apply_mmr_reranking(pieces, config, top_k)
        expected = pieces[:top_k]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.piece.piece_id == e.piece.piece_id

    @given(
        dim=st.integers(min_value=2, max_value=8),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_small_input_returns_input_truncated(self, dim, data):
        """When input length <= top_k, output equals input[:top_k].

        **Validates: Requirements 6.4**
        """
        n = data.draw(st.integers(min_value=1, max_value=10))
        pieces = data.draw(
            st.lists(scored_piece_with_embedding(dim=dim), min_size=n, max_size=n)
        )
        # top_k >= len(pieces) so the short-circuit path triggers
        top_k = data.draw(st.integers(min_value=len(pieces), max_value=len(pieces) + 10))
        config = MMRConfig(enabled=True, lambda_param=0.7)
        result = apply_mmr_reranking(pieces, config, top_k)
        expected = pieces[:top_k]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.piece.piece_id == e.piece.piece_id

"""
Property-based tests for LanceDB store multi-space support.

Feature: knowledge-space-restructuring
- Property 3: LanceDB Storage Round-Trip for Spaces
- Property 4: Space-Filtered Retrieval Correctness

Uses hypothesis with a minimum of 100 iterations per property test.
Integration tests use in-memory LanceDB (temp directory) for isolation.
"""
import sys
import tempfile
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

_rpu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))

from hypothesis import given, settings, strategies as st

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.lancedb_store import (
    LanceDBKnowledgePieceStore,
)

# Import strategies from conftest
_test_dir = Path(__file__).resolve().parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
from conftest import knowledge_piece_strategy

# Deterministic embedding function for testing (384-dim to match typical models)
_EMBED_DIM = 384
_dummy_embed = lambda text: [0.1] * _EMBED_DIM

# Strategy for generating valid space names
_space_strategy = st.sampled_from(["main", "personal", "developmental"])

# Strategy for generating non-empty deduplicated space lists
_spaces_strategy = st.lists(
    _space_strategy, min_size=1, max_size=3
).map(lambda xs: list(dict.fromkeys(xs)))


# ── Property 3: LanceDB Storage Round-Trip for Spaces ────────────────────────


class TestLanceDBStorageRoundTrip:
    """Property 3: LanceDB Storage Round-Trip for Spaces.

    For any valid KnowledgePiece with arbitrary spaces values, storing the
    piece in LanceDB via add() then retrieving it via get_by_id() SHALL
    produce a piece with an equivalent spaces list.

    **Validates: Requirements 2.1, 2.2, 2.6**
    """

    @given(
        piece=knowledge_piece_strategy(include_new_fields=True),
        spaces=_spaces_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_round_trip_preserves_spaces(self, piece: KnowledgePiece, spaces: list):
        """Store a piece with arbitrary spaces in LanceDB, retrieve by ID,
        and verify the spaces list is equivalent.

        **Validates: Requirements 2.1, 2.2, 2.6**
        """
        # Override the piece's spaces with the generated spaces
        piece.spaces = spaces
        piece.space = spaces[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LanceDBKnowledgePieceStore(
                db_path=tmp_dir,
                embedding_function=_dummy_embed,
            )
            store.add(piece)
            retrieved = store.get_by_id(piece.piece_id)

            assert retrieved is not None, f"Piece {piece.piece_id!r} not found after add()"
            assert retrieved.spaces == piece.spaces, (
                f"Spaces mismatch: stored {piece.spaces}, got {retrieved.spaces}"
            )
            assert retrieved.space == piece.space, (
                f"Primary space mismatch: stored {piece.space}, got {retrieved.space}"
            )


# ── Property 4: Space-Filtered Retrieval Correctness ─────────────────────────


# Strategy for generating a small set of pieces with random spaces
@st.composite
def _pieces_with_spaces(draw):
    """Generate 2-5 pieces, each with a random non-empty spaces list and unique piece_id."""
    n = draw(st.integers(min_value=2, max_value=5))
    pieces = []
    for i in range(n):
        spaces = draw(_spaces_strategy)
        piece = KnowledgePiece(
            content=f"piece content {i} {draw(st.text(min_size=1, max_size=20))}",
            piece_id=f"piece-{i}-{draw(st.uuids())}",
            spaces=spaces,
        )
        pieces.append(piece)
    return pieces


class TestSpaceFilteredRetrievalCorrectness:
    """Property 4: Space-Filtered Retrieval Correctness.

    For any set of knowledge pieces with various space assignments and any
    non-empty space filter list, calling list_all() with that space filter
    SHALL return only pieces whose spaces list has at least one element in
    common with the filter list. No piece outside the requested spaces SHALL
    appear in the results.

    **Validates: Requirements 2.3, 2.4, 2.5**
    """

    @given(
        pieces=_pieces_with_spaces(),
        filter_spaces=_spaces_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_list_all_with_space_filter_returns_only_matching_pieces(
        self, pieces: list, filter_spaces: list
    ):
        """Store pieces with various spaces, list_all with a space filter,
        and verify every returned piece has at least one space in common
        with the filter.

        **Validates: Requirements 2.3, 2.4, 2.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LanceDBKnowledgePieceStore(
                db_path=tmp_dir,
                embedding_function=_dummy_embed,
            )
            for piece in pieces:
                store.add(piece)

            results = store.list_all(spaces=filter_spaces)
            result_ids = {r.piece_id for r in results}
            filter_set = set(filter_spaces)

            # Every returned piece must have at least one space in common with the filter
            for r in results:
                assert set(r.spaces) & filter_set, (
                    f"Piece {r.piece_id!r} with spaces {r.spaces} does not match "
                    f"filter {filter_spaces}"
                )

            # Every piece that SHOULD match must be in the results (completeness)
            for piece in pieces:
                if set(piece.spaces) & filter_set:
                    assert piece.piece_id in result_ids, (
                        f"Piece {piece.piece_id!r} with spaces {piece.spaces} should "
                        f"match filter {filter_spaces} but was not returned"
                    )
                else:
                    assert piece.piece_id not in result_ids, (
                        f"Piece {piece.piece_id!r} with spaces {piece.spaces} should "
                        f"NOT match filter {filter_spaces} but was returned"
                    )

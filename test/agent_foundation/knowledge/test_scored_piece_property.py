"""
Property-based tests for ScoredPiece convenience property delegation.

Feature: knowledge-module-migration
Property 4: ScoredPiece convenience property delegation

Validates: Requirements 2.3
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

_test_dir = _current_file.parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))

from hypothesis import given, settings, strategies as st

from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece

from conftest import knowledge_piece_strategy


@st.composite
def scored_piece_strategy(draw):
    """Generate a ScoredPiece wrapping a random KnowledgePiece with random scores."""
    piece = draw(knowledge_piece_strategy())
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    normalized_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    vector_score = draw(st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ))
    keyword_score = draw(st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ))
    return ScoredPiece(
        piece=piece,
        score=score,
        normalized_score=normalized_score,
        vector_score=vector_score,
        keyword_score=keyword_score,
    )


class TestScoredPiecePropertyDelegation:
    """Property 4: ScoredPiece convenience property delegation.

    For any KnowledgePiece wrapped in a ScoredPiece, the convenience properties
    piece_id, content, info_type, and updated_at should return the same values
    as the wrapped piece's corresponding attributes.

    **Validates: Requirements 2.3**
    """

    @given(scored_piece=scored_piece_strategy())
    @settings(max_examples=100)
    def test_piece_id_delegates_to_wrapped_piece(self, scored_piece: ScoredPiece):
        """ScoredPiece.piece_id returns the wrapped piece's piece_id."""
        assert scored_piece.piece_id == scored_piece.piece.piece_id

    @given(scored_piece=scored_piece_strategy())
    @settings(max_examples=100)
    def test_content_delegates_to_wrapped_piece(self, scored_piece: ScoredPiece):
        """ScoredPiece.content returns the wrapped piece's content."""
        assert scored_piece.content == scored_piece.piece.content

    @given(scored_piece=scored_piece_strategy())
    @settings(max_examples=100)
    def test_info_type_delegates_to_wrapped_piece(self, scored_piece: ScoredPiece):
        """ScoredPiece.info_type returns the wrapped piece's info_type."""
        assert scored_piece.info_type == scored_piece.piece.info_type

    @given(scored_piece=scored_piece_strategy())
    @settings(max_examples=100)
    def test_updated_at_delegates_to_wrapped_piece(self, scored_piece: ScoredPiece):
        """ScoredPiece.updated_at returns the wrapped piece's updated_at."""
        assert scored_piece.updated_at == scored_piece.piece.updated_at

"""
Property-based tests for RetrievalKnowledgePieceStore round-trip serialization.

Feature: knowledge-space-restructuring
- Property 15: File-Based Store Round-Trip for Spaces

Tests that _piece_to_doc() → _doc_to_piece() preserves spaces, pending_space_suggestions,
space_suggestion_reasons, and space_suggestion_status fields.

**Validates: Requirements 2.7, 1.9**
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

# Also add RichPythonUtils src to path
_rpu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))

from hypothesis import given, settings, strategies as st

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)

# Import strategies from conftest
_test_dir = Path(__file__).resolve().parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
from conftest import knowledge_piece_strategy


# ── Strategies for space-related fields ──────────────────────────────────────

_valid_space = st.sampled_from(["main", "personal", "developmental"])

_spaces_list = st.lists(_valid_space, min_size=1, max_size=3).map(
    lambda xs: list(dict.fromkeys(xs))  # deduplicate, preserve order
)

_space_suggestion_status = st.sampled_from([None, "pending", "approved", "rejected"])

_optional_space_list = st.one_of(
    st.none(),
    st.lists(_valid_space, min_size=1, max_size=3).map(
        lambda xs: list(dict.fromkeys(xs))
    ),
)

_optional_reason_list = st.one_of(
    st.none(),
    st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=3),
)


@st.composite
def knowledge_piece_with_spaces_strategy(draw):
    """Generate a KnowledgePiece with explicit spaces and suggestion fields."""
    piece = draw(knowledge_piece_strategy(include_new_fields=True))

    spaces = draw(_spaces_list)
    pending_space_suggestions = draw(_optional_space_list)
    space_suggestion_reasons = draw(_optional_reason_list)
    space_suggestion_status = draw(_space_suggestion_status)

    return KnowledgePiece(
        content=piece.content,
        piece_id=piece.piece_id,
        knowledge_type=piece.knowledge_type,
        info_type=piece.info_type,
        tags=piece.tags,
        entity_id=piece.entity_id,
        source=piece.source,
        embedding_text=piece.embedding_text,
        created_at=piece.created_at,
        updated_at=piece.updated_at,
        domain=piece.domain,
        secondary_domains=piece.secondary_domains,
        custom_tags=piece.custom_tags,
        space=piece.space,
        spaces=spaces,
        is_active=piece.is_active,
        version=piece.version,
        supersedes=piece.supersedes,
        content_hash=piece.content_hash,
        validation_status=piece.validation_status,
        validation_issues=piece.validation_issues,
        summary=piece.summary,
        merge_strategy=piece.merge_strategy,
        merge_processed=piece.merge_processed,
        pending_merge_suggestion=piece.pending_merge_suggestion,
        merge_suggestion_reason=piece.merge_suggestion_reason,
        suggestion_status=piece.suggestion_status,
        pending_space_suggestions=pending_space_suggestions,
        space_suggestion_reasons=space_suggestion_reasons,
        space_suggestion_status=space_suggestion_status,
    )


# Feature: knowledge-space-restructuring, Property 15: File-Based Store Round-Trip for Spaces


class TestFileBasedStoreRoundTripForSpaces:
    """Property 15: File-Based Store Round-Trip for Spaces.

    For any valid KnowledgePiece with arbitrary spaces, pending_space_suggestions,
    space_suggestion_reasons, and space_suggestion_status values, converting via
    _piece_to_doc() then _doc_to_piece() SHALL preserve all four fields.

    **Validates: Requirements 2.7, 1.9**
    """

    @given(piece=knowledge_piece_with_spaces_strategy())
    @settings(max_examples=100)
    def test_piece_to_doc_to_piece_preserves_spaces(self, piece):
        """_piece_to_doc() → _doc_to_piece() preserves spaces field.

        **Validates: Requirements 2.7, 1.9**
        """
        store = RetrievalKnowledgePieceStore(
            retrieval_service=MemoryRetrievalService()
        )
        doc = store._piece_to_doc(piece)
        restored = store._doc_to_piece(doc)

        assert restored.spaces == piece.spaces
        assert restored.space == piece.space
        # Sync invariant holds
        assert restored.space == restored.spaces[0]

    @given(piece=knowledge_piece_with_spaces_strategy())
    @settings(max_examples=100)
    def test_piece_to_doc_to_piece_preserves_suggestion_fields(self, piece):
        """_piece_to_doc() → _doc_to_piece() preserves space suggestion fields.

        **Validates: Requirements 2.7, 1.9**
        """
        store = RetrievalKnowledgePieceStore(
            retrieval_service=MemoryRetrievalService()
        )
        doc = store._piece_to_doc(piece)
        restored = store._doc_to_piece(doc)

        assert restored.pending_space_suggestions == piece.pending_space_suggestions
        assert restored.space_suggestion_reasons == piece.space_suggestion_reasons
        assert restored.space_suggestion_status == piece.space_suggestion_status

    @given(piece=knowledge_piece_strategy(include_new_fields=True))
    @settings(max_examples=100)
    def test_piece_without_spaces_metadata_falls_back_to_space(self, piece):
        """When spaces is absent from Document metadata, _doc_to_piece derives
        spaces from the space field via __attrs_post_init__.

        **Validates: Requirements 2.7, 1.9**
        """
        store = RetrievalKnowledgePieceStore(
            retrieval_service=MemoryRetrievalService()
        )
        doc = store._piece_to_doc(piece)

        # Simulate old-format document without spaces in metadata
        del doc.metadata["spaces"]
        del doc.metadata["pending_space_suggestions"]
        del doc.metadata["space_suggestion_reasons"]
        del doc.metadata["space_suggestion_status"]

        restored = store._doc_to_piece(doc)

        # spaces should be derived from space via __attrs_post_init__
        assert restored.spaces == [restored.space]
        assert restored.space == piece.space
        # Suggestion fields should be None
        assert restored.pending_space_suggestions is None
        assert restored.space_suggestion_reasons is None
        assert restored.space_suggestion_status is None

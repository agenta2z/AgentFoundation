"""
Property-based tests for the knowledge deleter module.

Feature: knowledge-module-migration
- Property 27: Soft delete then restore round trip
- Property 28: Restore blocked by superseding piece
- Property 29: Delete by query with confirmation raises error

**Validates: Requirements 17.1, 17.3, 17.4, 17.5**
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pytest
from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.ingestion.knowledge_deleter import (
    ConfirmationRequiredError,
    DeleteConfig,
    KnowledgeDeleter,
)
from agent_foundation.knowledge.retrieval.models.enums import DeleteMode
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


# ── Test Helpers ──────────────────────────────────────────────────────────────


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for property testing."""

    def __init__(
        self,
        pieces: Optional[List[KnowledgePiece]] = None,
        search_score: float = 0.95,
    ):
        self._pieces: dict[str, KnowledgePiece] = {}
        for p in pieces or []:
            self._pieces[p.piece_id] = p
        self._search_score = search_score

    def add(self, piece: KnowledgePiece) -> str:
        self._pieces[piece.piece_id] = piece
        return piece.piece_id

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        return self._pieces.get(piece_id)

    def update(self, piece: KnowledgePiece) -> bool:
        if piece.piece_id in self._pieces:
            self._pieces[piece.piece_id] = piece
            return True
        return False

    def remove(self, piece_id: str) -> bool:
        return self._pieces.pop(piece_id, None) is not None

    def search(
        self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5
    ) -> List[Tuple[KnowledgePiece, float]]:
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            results.append((p, self._search_score))
        return results[:top_k]

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            results.append(p)
        return results


# ── Strategies ────────────────────────────────────────────────────────────────

# Content strings — non-empty, printable text
_content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=300,
)

_entity_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=30,
)

_domain_strategy = st.sampled_from(
    ["general", "testing", "debugging", "infrastructure", "model_optimization"]
)

_knowledge_type_strategy = st.sampled_from(list(KnowledgeType))

_tags_strategy = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    min_size=0,
    max_size=5,
)


# Feature: knowledge-module-migration, Property 27: Soft delete then restore round trip


class TestSoftDeleteThenRestoreRoundTrip:
    """Property 27: Soft delete then restore round trip.

    For any active piece, soft-deleting it should set is_active=False, and
    restoring it (when no superseding piece exists) should set is_active=True.
    The piece content and other fields should remain unchanged through the
    round trip.

    **Validates: Requirements 17.1, 17.4**
    """

    @given(
        content=_content_strategy,
        entity_id=_entity_id_strategy,
        domain=_domain_strategy,
        knowledge_type=_knowledge_type_strategy,
        tags=_tags_strategy,
    )
    @settings(max_examples=100)
    def test_round_trip_preserves_content_and_fields(
        self,
        content: str,
        entity_id: str,
        domain: str,
        knowledge_type: KnowledgeType,
        tags: List[str],
    ):
        """Soft delete sets is_active=False, restore sets is_active=True,
        and content + metadata fields are unchanged through the round trip.

        **Validates: Requirements 17.1, 17.4**
        """
        assume(len(content.strip()) > 0)
        assume(len(entity_id.strip()) > 0)

        piece = KnowledgePiece(
            content=content,
            entity_id=entity_id,
            domain=domain,
            knowledge_type=knowledge_type,
            tags=tags,
            is_active=True,
        )
        piece_id = piece.piece_id
        original_content = piece.content
        original_domain = piece.domain
        original_knowledge_type = piece.knowledge_type
        original_tags = list(piece.tags)
        original_entity_id = piece.entity_id
        original_content_hash = piece.content_hash

        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        # Soft delete
        del_result = deleter.delete_by_id(piece_id, mode=DeleteMode.SOFT)
        assert del_result.success is True

        deleted_piece = store.get_by_id(piece_id)
        assert deleted_piece.is_active is False

        # Restore
        restore_result = deleter.restore_by_id(piece_id)
        assert restore_result.success is True

        restored = store.get_by_id(piece_id)
        assert restored.is_active is True
        assert restored.content == original_content
        assert restored.domain == original_domain
        assert restored.knowledge_type == original_knowledge_type
        assert restored.tags == original_tags
        assert restored.entity_id == original_entity_id
        assert restored.content_hash == original_content_hash

    @given(
        content=_content_strategy,
        version=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_soft_delete_sets_inactive(
        self,
        content: str,
        version: int,
    ):
        """Soft delete should set is_active=False for any active piece.

        **Validates: Requirements 17.1**
        """
        assume(len(content.strip()) > 0)

        piece = KnowledgePiece(
            content=content,
            version=version,
            is_active=True,
        )

        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.delete_by_id(piece.piece_id, mode=DeleteMode.SOFT)
        assert result.success is True
        assert result.operation == "delete"
        assert result.details["mode"] == "soft"

        updated = store.get_by_id(piece.piece_id)
        assert updated.is_active is False

    @given(
        content=_content_strategy,
    )
    @settings(max_examples=100)
    def test_restore_sets_active(
        self,
        content: str,
    ):
        """Restore should set is_active=True for any soft-deleted piece
        with no superseding piece.

        **Validates: Requirements 17.4**
        """
        assume(len(content.strip()) > 0)

        piece = KnowledgePiece(
            content=content,
            is_active=False,
        )

        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.restore_by_id(piece.piece_id)
        assert result.success is True
        assert result.operation == "restore"

        restored = store.get_by_id(piece.piece_id)
        assert restored.is_active is True


# Feature: knowledge-module-migration, Property 28: Restore blocked by superseding piece


class TestRestoreBlockedBySupersedingPiece:
    """Property 28: Restore blocked by superseding piece.

    For any soft-deleted piece that has an active piece with `supersedes`
    pointing to it, restore_by_id() should return a failed OperationResult
    with an error message.

    **Validates: Requirements 17.5**
    """

    @given(
        old_content=_content_strategy,
        new_content=_content_strategy,
        entity_id=_entity_id_strategy,
    )
    @settings(max_examples=100)
    def test_restore_fails_when_superseded(
        self,
        old_content: str,
        new_content: str,
        entity_id: str,
    ):
        """Restoring a piece that is superseded by an active piece should fail.

        **Validates: Requirements 17.5**
        """
        assume(len(old_content.strip()) > 0)
        assume(len(new_content.strip()) > 0)
        assume(len(entity_id.strip()) > 0)

        old_piece = KnowledgePiece(
            content=old_content,
            entity_id=entity_id,
            is_active=False,
        )
        new_piece = KnowledgePiece(
            content=new_content,
            entity_id=entity_id,
            is_active=True,
            supersedes=old_piece.piece_id,
        )

        store = InMemoryPieceStore(pieces=[old_piece, new_piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.restore_by_id(old_piece.piece_id)
        assert result.success is False
        assert result.operation == "restore"
        assert result.error is not None
        assert "superseded" in result.error.lower() or "supersed" in result.error.lower()

    @given(
        old_content=_content_strategy,
        new_content=_content_strategy,
        entity_id=_entity_id_strategy,
    )
    @settings(max_examples=100)
    def test_restore_succeeds_when_superseding_piece_inactive(
        self,
        old_content: str,
        new_content: str,
        entity_id: str,
    ):
        """Restoring should succeed when the superseding piece is also inactive.

        **Validates: Requirements 17.5**
        """
        assume(len(old_content.strip()) > 0)
        assume(len(new_content.strip()) > 0)
        assume(len(entity_id.strip()) > 0)

        old_piece = KnowledgePiece(
            content=old_content,
            entity_id=entity_id,
            is_active=False,
        )
        new_piece = KnowledgePiece(
            content=new_content,
            entity_id=entity_id,
            is_active=False,
            supersedes=old_piece.piece_id,
        )

        store = InMemoryPieceStore(pieces=[old_piece, new_piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.restore_by_id(old_piece.piece_id)
        assert result.success is True
        assert store.get_by_id(old_piece.piece_id).is_active is True


# Feature: knowledge-module-migration, Property 29: Delete by query with confirmation raises error


class TestDeleteByQueryWithConfirmationRaisesError:
    """Property 29: Delete by query with confirmation raises error.

    For any query-based deletion with require_confirmation=True and no
    explicit piece_ids, delete_by_query() should raise
    ConfirmationRequiredError containing the list of candidate pieces.

    **Validates: Requirements 17.3**
    """

    @given(
        query=_content_strategy,
        num_pieces=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_confirmation_raises_with_candidates(
        self,
        query: str,
        num_pieces: int,
    ):
        """delete_by_query with require_confirmation=True and no piece_ids
        should raise ConfirmationRequiredError with candidate pieces.

        **Validates: Requirements 17.3**
        """
        assume(len(query.strip()) > 0)

        pieces = [
            KnowledgePiece(content=f"Piece content {i}", is_active=True)
            for i in range(num_pieces)
        ]

        store = InMemoryPieceStore(pieces=pieces, search_score=0.95)
        config = DeleteConfig(require_confirmation=True)
        deleter = KnowledgeDeleter(store, config=config)

        with pytest.raises(ConfirmationRequiredError) as exc_info:
            deleter.delete_by_query(query=query)

        err = exc_info.value
        assert isinstance(err.candidates, list)
        # All candidates should be (KnowledgePiece, float) tuples
        for candidate in err.candidates:
            assert isinstance(candidate, tuple)
            assert len(candidate) == 2
            assert isinstance(candidate[0], KnowledgePiece)
            assert isinstance(candidate[1], float)

    @given(
        query=_content_strategy,
    )
    @settings(max_examples=100)
    def test_confirmation_not_raised_when_piece_ids_provided(
        self,
        query: str,
    ):
        """delete_by_query should NOT raise ConfirmationRequiredError when
        explicit piece_ids are provided, even with require_confirmation=True.

        **Validates: Requirements 17.3**
        """
        assume(len(query.strip()) > 0)

        piece = KnowledgePiece(content="Test content", is_active=True)
        store = InMemoryPieceStore(pieces=[piece])
        config = DeleteConfig(require_confirmation=True)
        deleter = KnowledgeDeleter(store, config=config)

        # Should not raise — piece_ids bypass confirmation
        results = deleter.delete_by_query(
            query=query,
            piece_ids=[piece.piece_id],
        )
        assert len(results) == 1
        assert results[0].success is True

    @given(
        query=_content_strategy,
        threshold=st.floats(min_value=0.91, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_confirmation_error_respects_threshold(
        self,
        query: str,
        threshold: float,
    ):
        """ConfirmationRequiredError candidates should only include pieces
        above the similarity threshold.

        **Validates: Requirements 17.3**
        """
        assume(len(query.strip()) > 0)

        piece = KnowledgePiece(content="Test content", is_active=True)
        # Score below threshold — no candidates should match
        store = InMemoryPieceStore(pieces=[piece], search_score=0.50)
        config = DeleteConfig(
            require_confirmation=True,
            similarity_threshold=threshold,
        )
        deleter = KnowledgeDeleter(store, config=config)

        with pytest.raises(ConfirmationRequiredError) as exc_info:
            deleter.delete_by_query(query=query)

        # Candidates should be empty since score < threshold
        assert len(exc_info.value.candidates) == 0

"""Unit tests for KnowledgeDeleter."""

from typing import List, Optional, Tuple

import pytest

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


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for testing."""

    def __init__(
        self,
        pieces: Optional[List[KnowledgePiece]] = None,
        search_score: float = 0.95,
    ):
        self._pieces = {p.piece_id: p for p in (pieces or [])}
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
        if piece_id in self._pieces:
            del self._pieces[piece_id]
            return True
        return False

    def search(
        self,
        query: str,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
        tags: List[str] = None,
        top_k: int = 5,
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


def _make_piece(
    content: str = "Test content",
    piece_id: str = None,
    entity_id: str = "entity-1",
    is_active: bool = True,
    supersedes: str = None,
    domain: str = "general",
    secondary_domains: list = None,
) -> KnowledgePiece:
    piece = KnowledgePiece(
        content=content,
        entity_id=entity_id,
        is_active=is_active,
        supersedes=supersedes,
        domain=domain,
        secondary_domains=secondary_domains or [],
    )
    if piece_id:
        piece.piece_id = piece_id
    return piece


# ── DeleteConfig Tests ───────────────────────────────────────────────────


class TestDeleteConfig:
    def test_defaults(self):
        config = DeleteConfig()
        assert config.default_mode == DeleteMode.SOFT
        assert config.similarity_threshold == 0.90
        assert config.max_deletions == 10
        assert config.require_confirmation is True

    def test_custom_values(self):
        config = DeleteConfig(
            default_mode=DeleteMode.HARD,
            similarity_threshold=0.80,
            max_deletions=5,
            require_confirmation=False,
        )
        assert config.default_mode == DeleteMode.HARD
        assert config.similarity_threshold == 0.80
        assert config.max_deletions == 5
        assert config.require_confirmation is False


# ── ConfirmationRequiredError Tests ──────────────────────────────────────


class TestConfirmationRequiredError:
    def test_error_has_candidates(self):
        piece = _make_piece()
        candidates = [(piece, 0.95)]
        err = ConfirmationRequiredError(candidates=candidates)
        assert err.candidates == candidates
        assert "1 candidates" in str(err)

    def test_custom_message(self):
        err = ConfirmationRequiredError(candidates=[], message="Custom msg")
        assert str(err) == "Custom msg"
        assert err.message == "Custom msg"

    def test_is_exception(self):
        err = ConfirmationRequiredError(candidates=[])
        assert isinstance(err, Exception)


# ── Delete by ID Tests ───────────────────────────────────────────────────


class TestDeleteById:
    def test_piece_not_found_returns_failure(self):
        store = InMemoryPieceStore()
        deleter = KnowledgeDeleter(store)
        result = deleter.delete_by_id("nonexistent")
        assert result.success is False
        assert result.operation == "delete"
        assert "not found" in result.error

    def test_soft_delete_sets_inactive(self):
        piece = _make_piece(piece_id="p1")
        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.delete_by_id("p1", mode=DeleteMode.SOFT)
        assert result.success is True
        assert result.details["mode"] == "soft"

        updated = store.get_by_id("p1")
        assert updated.is_active is False
        assert updated.updated_at is not None

    def test_hard_delete_removes_piece(self):
        piece = _make_piece(piece_id="p1")
        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.delete_by_id("p1", mode=DeleteMode.HARD)
        assert result.success is True
        assert result.details["mode"] == "hard"
        assert store.get_by_id("p1") is None

    def test_default_mode_from_config(self):
        piece = _make_piece(piece_id="p1")
        store = InMemoryPieceStore(pieces=[piece])
        config = DeleteConfig(default_mode=DeleteMode.HARD)
        deleter = KnowledgeDeleter(store, config=config)

        result = deleter.delete_by_id("p1")
        assert result.success is True
        assert store.get_by_id("p1") is None

    def test_soft_delete_updates_timestamp(self):
        piece = _make_piece(piece_id="p1")
        piece.updated_at = "2020-01-01T00:00:00+00:00"
        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        deleter.delete_by_id("p1", mode=DeleteMode.SOFT)
        updated = store.get_by_id("p1")
        assert updated.updated_at != "2020-01-01T00:00:00+00:00"


# ── Delete by Query Tests ────────────────────────────────────────────────


class TestDeleteByQuery:
    def test_with_piece_ids_deletes_directly(self):
        p1 = _make_piece(piece_id="p1")
        p2 = _make_piece(piece_id="p2")
        store = InMemoryPieceStore(pieces=[p1, p2])
        deleter = KnowledgeDeleter(store)

        results = deleter.delete_by_query(
            query="anything",
            piece_ids=["p1", "p2"],
            mode=DeleteMode.HARD,
        )
        assert len(results) == 2
        assert all(r.success for r in results)
        assert store.get_by_id("p1") is None
        assert store.get_by_id("p2") is None

    def test_confirmation_required_raises_error(self):
        piece = _make_piece(piece_id="p1")
        store = InMemoryPieceStore(pieces=[piece])
        config = DeleteConfig(require_confirmation=True)
        deleter = KnowledgeDeleter(store, config=config)

        with pytest.raises(ConfirmationRequiredError) as exc_info:
            deleter.delete_by_query(query="test")

        assert len(exc_info.value.candidates) == 1

    def test_no_confirmation_deletes_candidates(self):
        p1 = _make_piece(piece_id="p1")
        p2 = _make_piece(piece_id="p2")
        store = InMemoryPieceStore(pieces=[p1, p2])
        config = DeleteConfig(require_confirmation=False)
        deleter = KnowledgeDeleter(store, config=config)

        results = deleter.delete_by_query(query="test", mode=DeleteMode.SOFT)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_max_deletions_respected(self):
        pieces = [_make_piece(piece_id=f"p{i}") for i in range(5)]
        store = InMemoryPieceStore(pieces=pieces)
        config = DeleteConfig(require_confirmation=False, max_deletions=2)
        deleter = KnowledgeDeleter(store, config=config)

        results = deleter.delete_by_query(query="test", mode=DeleteMode.HARD)
        assert len(results) == 2

    def test_threshold_filters_low_scores(self):
        piece = _make_piece(piece_id="p1")
        store = InMemoryPieceStore(pieces=[piece], search_score=0.50)
        config = DeleteConfig(require_confirmation=False, similarity_threshold=0.90)
        deleter = KnowledgeDeleter(store, config=config)

        results = deleter.delete_by_query(query="test")
        assert len(results) == 0

    def test_domain_filter_applied(self):
        p1 = _make_piece(piece_id="p1", domain="testing")
        p2 = _make_piece(piece_id="p2", domain="general")
        store = InMemoryPieceStore(pieces=[p1, p2])
        config = DeleteConfig(require_confirmation=False)
        deleter = KnowledgeDeleter(store, config=config)

        results = deleter.delete_by_query(query="test", domain="testing")
        assert len(results) == 1
        assert results[0].piece_id == "p1"

    def test_domain_filter_includes_secondary_domains(self):
        p1 = _make_piece(
            piece_id="p1",
            domain="general",
            secondary_domains=["testing"],
        )
        store = InMemoryPieceStore(pieces=[p1])
        config = DeleteConfig(require_confirmation=False)
        deleter = KnowledgeDeleter(store, config=config)

        results = deleter.delete_by_query(query="test", domain="testing")
        assert len(results) == 1

    def test_inactive_pieces_filtered_from_candidates(self):
        p1 = _make_piece(piece_id="p1", is_active=False)
        store = InMemoryPieceStore(pieces=[p1])
        config = DeleteConfig(require_confirmation=False)
        deleter = KnowledgeDeleter(store, config=config)

        results = deleter.delete_by_query(query="test")
        assert len(results) == 0


# ── Delete by IDs Tests ──────────────────────────────────────────────────


class TestDeleteByIds:
    def test_deletes_multiple(self):
        p1 = _make_piece(piece_id="p1")
        p2 = _make_piece(piece_id="p2")
        store = InMemoryPieceStore(pieces=[p1, p2])
        deleter = KnowledgeDeleter(store)

        results = deleter.delete_by_ids(["p1", "p2"], mode=DeleteMode.HARD)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_partial_failure(self):
        p1 = _make_piece(piece_id="p1")
        store = InMemoryPieceStore(pieces=[p1])
        deleter = KnowledgeDeleter(store)

        results = deleter.delete_by_ids(["p1", "nonexistent"])
        assert results[0].success is True
        assert results[1].success is False


# ── Restore by ID Tests ─────────────────────────────────────────────────


class TestRestoreById:
    def test_piece_not_found_returns_failure(self):
        store = InMemoryPieceStore()
        deleter = KnowledgeDeleter(store)
        result = deleter.restore_by_id("nonexistent")
        assert result.success is False
        assert "not found" in result.error

    def test_already_active_returns_failure(self):
        piece = _make_piece(piece_id="p1", is_active=True)
        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.restore_by_id("p1")
        assert result.success is False
        assert "already active" in result.error

    def test_restore_soft_deleted_piece(self):
        piece = _make_piece(piece_id="p1", is_active=False)
        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.restore_by_id("p1")
        assert result.success is True
        assert result.operation == "restore"

        restored = store.get_by_id("p1")
        assert restored.is_active is True

    def test_restore_updates_timestamp(self):
        piece = _make_piece(piece_id="p1", is_active=False)
        piece.updated_at = "2020-01-01T00:00:00+00:00"
        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        deleter.restore_by_id("p1")
        restored = store.get_by_id("p1")
        assert restored.updated_at != "2020-01-01T00:00:00+00:00"

    def test_restore_blocked_by_superseding_piece(self):
        old_piece = _make_piece(piece_id="old-1", is_active=False, entity_id="e1")
        new_piece = _make_piece(
            piece_id="new-1",
            is_active=True,
            entity_id="e1",
            supersedes="old-1",
        )
        store = InMemoryPieceStore(pieces=[old_piece, new_piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.restore_by_id("old-1")
        assert result.success is False
        assert "superseded" in result.error
        assert "new-1" in result.error

    def test_restore_allowed_when_superseding_piece_inactive(self):
        old_piece = _make_piece(piece_id="old-1", is_active=False, entity_id="e1")
        new_piece = _make_piece(
            piece_id="new-1",
            is_active=False,
            entity_id="e1",
            supersedes="old-1",
        )
        store = InMemoryPieceStore(pieces=[old_piece, new_piece])
        deleter = KnowledgeDeleter(store)

        result = deleter.restore_by_id("old-1")
        assert result.success is True

    def test_soft_delete_then_restore_round_trip(self):
        piece = _make_piece(piece_id="p1", content="Important knowledge")
        store = InMemoryPieceStore(pieces=[piece])
        deleter = KnowledgeDeleter(store)

        # Soft delete
        del_result = deleter.delete_by_id("p1", mode=DeleteMode.SOFT)
        assert del_result.success is True
        assert store.get_by_id("p1").is_active is False

        # Restore
        restore_result = deleter.restore_by_id("p1")
        assert restore_result.success is True

        restored = store.get_by_id("p1")
        assert restored.is_active is True
        assert restored.content == "Important knowledge"

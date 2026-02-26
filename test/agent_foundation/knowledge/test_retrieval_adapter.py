"""
Unit tests for RetrievalKnowledgePieceStore adapter.

Tests that the adapter correctly implements the KnowledgePieceStore ABC by
delegating to a MemoryRetrievalService backend. Covers CRUD operations,
search with filters, entity_id scoping, round-trip serialization, and
close delegation.

Requirements: 12.1, 12.2, 12.3, 12.4
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

# Also add SciencePythonUtils src to path
_spu_src = Path(__file__).resolve().parents[4] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest

from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from science_modeling_tools.knowledge.stores.pieces.base import KnowledgePieceStore
from science_modeling_tools.knowledge.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)


@pytest.fixture
def retrieval_service():
    """Create a fresh MemoryRetrievalService for each test."""
    return MemoryRetrievalService()


@pytest.fixture
def store(retrieval_service):
    """Create a RetrievalKnowledgePieceStore backed by MemoryRetrievalService."""
    return RetrievalKnowledgePieceStore(retrieval_service=retrieval_service)


class TestImplementsABC:
    """Requirement 12.1: RetrievalKnowledgePieceStore implements KnowledgePieceStore ABC."""

    def test_is_instance_of_knowledge_piece_store(self, store):
        """RetrievalKnowledgePieceStore should be an instance of KnowledgePieceStore."""
        assert isinstance(store, KnowledgePieceStore)

    def test_has_all_abstract_methods(self, store):
        """RetrievalKnowledgePieceStore should implement all KnowledgePieceStore methods."""
        assert hasattr(store, "add")
        assert hasattr(store, "get_by_id")
        assert hasattr(store, "update")
        assert hasattr(store, "remove")
        assert hasattr(store, "search")
        assert hasattr(store, "list_all")
        assert hasattr(store, "close")


class TestAddAndGetById:
    """Requirements 12.2, 12.4: add then get_by_id returns equivalent piece."""

    def test_add_then_get_by_id_round_trip(self, store):
        """Adding a piece and getting it back should preserve all fields."""
        piece = KnowledgePiece(
            content="Python is a programming language",
            piece_id="piece-001",
            knowledge_type=KnowledgeType.Fact,
            tags=["python", "programming"],
            entity_id=None,
            source="manual",
            embedding_text="python programming language",
        )
        returned_id = store.add(piece)
        assert returned_id == "piece-001"

        result = store.get_by_id("piece-001")
        assert result is not None
        assert result.piece_id == piece.piece_id
        assert result.content == piece.content
        assert result.knowledge_type == piece.knowledge_type
        assert result.tags == piece.tags
        assert result.entity_id == piece.entity_id
        assert result.source == piece.source
        assert result.embedding_text == piece.embedding_text
        assert result.created_at == piece.created_at
        assert result.updated_at == piece.updated_at

    def test_add_piece_with_entity_id(self, store):
        """Adding a piece with entity_id should scope it to that namespace."""
        piece = KnowledgePiece(
            content="Entity-specific knowledge",
            piece_id="piece-entity-001",
            entity_id="user:alice",
        )
        store.add(piece)

        result = store.get_by_id("piece-entity-001")
        assert result is not None
        assert result.entity_id == "user:alice"

    def test_get_by_id_nonexistent_returns_none(self, store):
        """Getting a non-existent piece should return None."""
        result = store.get_by_id("nonexistent-id")
        assert result is None

    def test_duplicate_add_raises_value_error(self, store):
        """Adding a piece with a duplicate piece_id should raise ValueError."""
        piece = KnowledgePiece(
            content="First piece",
            piece_id="dup-001",
        )
        store.add(piece)

        duplicate = KnowledgePiece(
            content="Duplicate piece",
            piece_id="dup-001",
        )
        with pytest.raises(ValueError):
            store.add(duplicate)

    def test_add_uses_entity_id_as_namespace(self, store, retrieval_service):
        """add should use entity_id as the retrieval service namespace."""
        piece = KnowledgePiece(
            content="Scoped knowledge",
            piece_id="ns-001",
            entity_id="project:alpha",
        )
        store.add(piece)

        # Verify the document is stored under the "project:alpha" namespace
        doc = retrieval_service.get_by_id("ns-001", namespace="project:alpha")
        assert doc is not None
        assert doc.doc_id == "ns-001"

    def test_add_global_piece_uses_default_namespace(self, store, retrieval_service):
        """add with entity_id=None should use the default namespace."""
        piece = KnowledgePiece(
            content="Global knowledge",
            piece_id="global-001",
            entity_id=None,
        )
        store.add(piece)

        # Verify the document is stored under the default namespace (None → _default)
        doc = retrieval_service.get_by_id("global-001", namespace=None)
        assert doc is not None


class TestSearch:
    """Requirement 12.3: search with knowledge_type and tags filters."""

    def _add_test_pieces(self, store):
        """Add a set of test pieces for search tests."""
        pieces = [
            KnowledgePiece(
                content="Python is great for data science",
                piece_id="p1",
                knowledge_type=KnowledgeType.Fact,
                tags=["python", "data"],
            ),
            KnowledgePiece(
                content="Use pytest for testing Python code",
                piece_id="p2",
                knowledge_type=KnowledgeType.Instruction,
                tags=["python", "testing"],
            ),
            KnowledgePiece(
                content="Machine learning requires data preprocessing",
                piece_id="p3",
                knowledge_type=KnowledgeType.Fact,
                tags=["ml", "data"],
            ),
        ]
        for p in pieces:
            store.add(p)
        return pieces

    def test_search_returns_matching_pieces(self, store):
        """Search should return pieces matching the query."""
        self._add_test_pieces(store)

        results = store.search("python")
        assert len(results) > 0
        # Results are (KnowledgePiece, score) tuples
        for piece, score in results:
            assert isinstance(piece, KnowledgePiece)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_search_with_knowledge_type_filter(self, store):
        """Search with knowledge_type filter should return only matching types."""
        self._add_test_pieces(store)

        results = store.search("python data", knowledge_type=KnowledgeType.Fact)
        assert len(results) > 0
        for piece, score in results:
            assert piece.knowledge_type == KnowledgeType.Fact

    def test_search_with_tags_filter(self, store):
        """Search with tags filter should return only pieces with all specified tags."""
        self._add_test_pieces(store)

        results = store.search("python", tags=["python", "testing"])
        assert len(results) > 0
        for piece, score in results:
            assert "python" in piece.tags
            assert "testing" in piece.tags

    def test_search_with_entity_id_scoping(self, store):
        """Search with entity_id should only return pieces from that entity."""
        # Add global piece
        store.add(KnowledgePiece(
            content="Global python knowledge",
            piece_id="global-p1",
            entity_id=None,
        ))
        # Add entity-scoped piece
        store.add(KnowledgePiece(
            content="Alice python knowledge",
            piece_id="alice-p1",
            entity_id="user:alice",
        ))

        # Search in alice's scope
        results = store.search("python", entity_id="user:alice")
        piece_ids = [p.piece_id for p, _ in results]
        assert "alice-p1" in piece_ids
        assert "global-p1" not in piece_ids

        # Search in global scope
        results = store.search("python", entity_id=None)
        piece_ids = [p.piece_id for p, _ in results]
        assert "global-p1" in piece_ids
        assert "alice-p1" not in piece_ids

    def test_search_respects_top_k(self, store):
        """Search should return at most top_k results."""
        for i in range(10):
            store.add(KnowledgePiece(
                content=f"Python knowledge item {i}",
                piece_id=f"topk-{i}",
            ))

        results = store.search("python", top_k=3)
        assert len(results) <= 3


class TestUpdate:
    """Requirement 12.4: update preserves fields through round-trip."""

    def test_update_existing_piece(self, store):
        """Updating an existing piece should modify its content."""
        piece = KnowledgePiece(
            content="Original content",
            piece_id="upd-001",
            knowledge_type=KnowledgeType.Fact,
            tags=["original"],
        )
        store.add(piece)

        updated_piece = KnowledgePiece(
            content="Updated content",
            piece_id="upd-001",
            knowledge_type=KnowledgeType.Instruction,
            tags=["updated"],
        )
        result = store.update(updated_piece)
        assert result is True

        retrieved = store.get_by_id("upd-001")
        assert retrieved is not None
        assert retrieved.content == "Updated content"
        assert retrieved.knowledge_type == KnowledgeType.Instruction
        assert retrieved.tags == ["updated"]

    def test_update_nonexistent_returns_false(self, store):
        """Updating a non-existent piece should return False."""
        piece = KnowledgePiece(
            content="Does not exist",
            piece_id="nonexistent-001",
        )
        result = store.update(piece)
        assert result is False


class TestRemove:
    """Requirement 12.1: remove operations via adapter."""

    def test_remove_existing_piece(self, store):
        """Removing an existing piece should return True."""
        piece = KnowledgePiece(
            content="To be removed",
            piece_id="rm-001",
        )
        store.add(piece)
        assert store.remove("rm-001") is True

    def test_remove_nonexistent_returns_false(self, store):
        """Removing a non-existent piece should return False."""
        assert store.remove("nonexistent-id") is False

    def test_get_after_remove_returns_none(self, store):
        """Getting a piece after removal should return None."""
        piece = KnowledgePiece(
            content="Temporary piece",
            piece_id="rm-002",
        )
        store.add(piece)
        store.remove("rm-002")
        assert store.get_by_id("rm-002") is None

    def test_remove_entity_scoped_piece(self, store):
        """Removing an entity-scoped piece should work correctly."""
        piece = KnowledgePiece(
            content="Entity piece",
            piece_id="rm-entity-001",
            entity_id="user:bob",
        )
        store.add(piece)
        assert store.remove("rm-entity-001") is True
        assert store.get_by_id("rm-entity-001") is None


class TestListAll:
    """Requirements 12.1, 12.3: list_all with filters."""

    def test_list_all_returns_all_global_pieces(self, store):
        """list_all with entity_id=None should return all global pieces."""
        store.add(KnowledgePiece(content="Piece A", piece_id="la-001"))
        store.add(KnowledgePiece(content="Piece B", piece_id="la-002"))

        results = store.list_all()
        assert len(results) == 2
        piece_ids = {p.piece_id for p in results}
        assert piece_ids == {"la-001", "la-002"}

    def test_list_all_with_entity_id_scoping(self, store):
        """list_all with entity_id should return only that entity's pieces."""
        store.add(KnowledgePiece(
            content="Global piece",
            piece_id="la-global",
            entity_id=None,
        ))
        store.add(KnowledgePiece(
            content="Alice piece",
            piece_id="la-alice",
            entity_id="user:alice",
        ))

        global_pieces = store.list_all(entity_id=None)
        assert len(global_pieces) == 1
        assert global_pieces[0].piece_id == "la-global"

        alice_pieces = store.list_all(entity_id="user:alice")
        assert len(alice_pieces) == 1
        assert alice_pieces[0].piece_id == "la-alice"

    def test_list_all_with_knowledge_type_filter(self, store):
        """list_all with knowledge_type should return only matching pieces."""
        store.add(KnowledgePiece(
            content="A fact",
            piece_id="la-fact",
            knowledge_type=KnowledgeType.Fact,
        ))
        store.add(KnowledgePiece(
            content="An instruction",
            piece_id="la-instr",
            knowledge_type=KnowledgeType.Instruction,
        ))

        facts = store.list_all(knowledge_type=KnowledgeType.Fact)
        assert len(facts) == 1
        assert facts[0].piece_id == "la-fact"
        assert facts[0].knowledge_type == KnowledgeType.Fact

    def test_list_all_empty_store(self, store):
        """list_all on an empty store should return an empty list."""
        assert store.list_all() == []


class TestClose:
    """Adapter close should delegate to the underlying retrieval service."""

    def test_close_delegates_to_retrieval_service(self, store, retrieval_service):
        """close() should delegate to the underlying retrieval service's close()."""
        store.close()
        # MemoryRetrievalService sets _closed=True on close
        assert retrieval_service._closed is True

    def test_close_is_idempotent(self, store):
        """Calling close() multiple times should not raise."""
        store.close()
        store.close()  # Should not raise

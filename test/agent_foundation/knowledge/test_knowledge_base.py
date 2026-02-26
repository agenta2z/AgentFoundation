"""
Unit tests for KnowledgeBase orchestrator.

Tests CRUD operations, context manager, sensitive content rejection,
bulk_load, merge logic, graph knowledge extraction, and retrieval.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5,
              8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.4
"""
import json
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

import pytest

from science_modeling_tools.knowledge.knowledge_base import KnowledgeBase
from science_modeling_tools.knowledge.formatter import RetrievalResult
from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)
from science_modeling_tools.knowledge.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from science_modeling_tools.knowledge.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from science_modeling_tools.knowledge.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def stores():
    """Create adapter-backed stores using in-memory services."""
    metadata_store = KeyValueMetadataStore(kv_service=MemoryKeyValueService())
    piece_store = RetrievalKnowledgePieceStore(retrieval_service=MemoryRetrievalService())
    graph_store = GraphServiceEntityGraphStore(graph_service=MemoryGraphService())
    return metadata_store, piece_store, graph_store


@pytest.fixture
def kb(stores):
    """Create a KnowledgeBase with adapter-backed stores."""
    metadata_store, piece_store, graph_store = stores
    return KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:xinli",
    )


@pytest.fixture
def populated_kb(kb, stores):
    """Create a KnowledgeBase with some pre-loaded data."""
    metadata_store, piece_store, graph_store = stores

    # Add metadata
    user_meta = EntityMetadata(
        entity_id="user:xinli",
        entity_type="user",
        properties={"name": "Xinli", "location": "Seattle"},
    )
    metadata_store.save_metadata(user_meta)

    global_meta = EntityMetadata(
        entity_id="global",
        entity_type="global",
        properties={"app_version": "1.0"},
    )
    metadata_store.save_metadata(global_meta)

    # Add knowledge pieces (entity-scoped)
    piece1 = KnowledgePiece(
        content="Prefers organic eggs",
        piece_id="piece-1",
        knowledge_type=KnowledgeType.Preference,
        tags=["grocery", "eggs"],
        entity_id="user:xinli",
    )
    piece_store.add(piece1)

    piece2 = KnowledgePiece(
        content="Costco membership is Executive tier",
        piece_id="piece-2",
        knowledge_type=KnowledgeType.Fact,
        tags=["costco", "membership"],
        entity_id="user:xinli",
    )
    piece_store.add(piece2)

    # Add global piece
    global_piece = KnowledgePiece(
        content="Egg prices vary by season",
        piece_id="piece-global",
        knowledge_type=KnowledgeType.Fact,
        tags=["eggs", "pricing"],
        entity_id=None,
    )
    piece_store.add(global_piece)

    # Add graph nodes and relations using GraphNode/GraphEdge
    user_node = GraphNode(
        node_id="user:xinli", node_type="user", label="Xinli"
    )
    store_node = GraphNode(
        node_id="store:costco", node_type="store", label="Costco"
    )
    graph_store.add_node(user_node)
    graph_store.add_node(store_node)

    relation = GraphEdge(
        source_id="user:xinli",
        target_id="store:costco",
        edge_type="SHOPS_AT",
        properties={"membership_tier": "Executive"},
    )
    graph_store.add_relation(relation)

    return kb


# ── CRUD Tests ───────────────────────────────────────────────────────────────


class TestAddPiece:
    """Tests for KnowledgeBase.add_piece."""

    def test_add_piece_returns_id(self, kb):
        piece = KnowledgePiece(content="Test knowledge", piece_id="test-1")
        result = kb.add_piece(piece)
        assert result == "test-1"

    def test_add_piece_persists(self, kb, stores):
        _, piece_store, _ = stores
        piece = KnowledgePiece(content="Persisted knowledge", piece_id="test-2")
        kb.add_piece(piece)
        retrieved = piece_store.get_by_id("test-2")
        assert retrieved is not None
        assert retrieved.content == "Persisted knowledge"

    def test_add_piece_rejects_empty_content(self, kb):
        piece = KnowledgePiece.__new__(KnowledgePiece)
        piece.content = ""
        piece.piece_id = "empty-1"
        with pytest.raises(ValueError, match="non-empty"):
            kb.add_piece(piece)

    def test_add_piece_rejects_whitespace_content(self, kb):
        piece = KnowledgePiece.__new__(KnowledgePiece)
        piece.content = "   \t\n  "
        piece.piece_id = "ws-1"
        with pytest.raises(ValueError, match="non-empty"):
            kb.add_piece(piece)


class TestUpdatePiece:
    """Tests for KnowledgeBase.update_piece."""

    def test_update_existing_piece(self, kb):
        piece = KnowledgePiece(content="Original", piece_id="upd-1")
        kb.add_piece(piece)

        piece.content = "Updated content"
        result = kb.update_piece(piece)
        assert result is True

    def test_update_refreshes_timestamp(self, kb, stores):
        _, piece_store, _ = stores
        piece = KnowledgePiece(content="Original", piece_id="upd-2")
        kb.add_piece(piece)
        original_updated = piece.updated_at

        piece.content = "Updated content"
        kb.update_piece(piece)

        retrieved = piece_store.get_by_id("upd-2")
        assert retrieved.content == "Updated content"
        assert retrieved.updated_at >= original_updated

    def test_update_nonexistent_returns_false(self, kb):
        piece = KnowledgePiece(content="Ghost", piece_id="nonexistent")
        result = kb.update_piece(piece)
        assert result is False

    def test_update_rejects_sensitive_content(self, kb):
        piece = KnowledgePiece(content="Safe content", piece_id="upd-3")
        kb.add_piece(piece)

        piece.content = "api_key= sk-12345"
        with pytest.raises(ValueError, match="sensitive"):
            kb.update_piece(piece)


class TestRemovePiece:
    """Tests for KnowledgeBase.remove_piece."""

    def test_remove_existing_piece(self, kb):
        piece = KnowledgePiece(content="To be removed", piece_id="rm-1")
        kb.add_piece(piece)
        result = kb.remove_piece("rm-1")
        assert result is True

    def test_remove_nonexistent_returns_false(self, kb):
        result = kb.remove_piece("nonexistent")
        assert result is False

    def test_remove_then_get_returns_none(self, kb, stores):
        _, piece_store, _ = stores
        piece = KnowledgePiece(content="Temporary", piece_id="rm-2")
        kb.add_piece(piece)
        kb.remove_piece("rm-2")
        assert piece_store.get_by_id("rm-2") is None

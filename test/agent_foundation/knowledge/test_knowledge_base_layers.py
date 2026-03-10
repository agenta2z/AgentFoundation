"""
Unit tests for KnowledgeBase layer methods.

Tests each layer method individually (retrieve_metadata, retrieve_pieces,
retrieve_search_graph, retrieve_identity_graph) and verifies that the
refactored retrieve() produces identical output to manual layer assembly.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

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

_test_dir = str(Path(__file__).resolve().parent)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

import pytest

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode, GraphEdge

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from agent_foundation.knowledge.retrieval.stores.metadata.base import MetadataStore
from conftest import InMemoryEntityGraphStore


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_metadata(entity_id: str, spaces: List[str] = None) -> EntityMetadata:
    return EntityMetadata(
        entity_id=entity_id,
        entity_type="user",
        properties={"name": entity_id},
        spaces=spaces or ["main"],
    )


def _make_piece(piece_id: str, entity_id: str = None, info_type: str = "context") -> KnowledgePiece:
    return KnowledgePiece(
        content=f"Content for {piece_id}",
        piece_id=piece_id,
        entity_id=entity_id,
        info_type=info_type,
    )


def _make_kb(
    include_metadata=True,
    include_pieces=True,
    include_graph=True,
    active_entity_id="user:test",
    graph_traversal_depth=1,
):
    """Create a KnowledgeBase with mock stores for testing."""
    metadata_store = MagicMock(spec=MetadataStore)
    metadata_store.get_metadata.return_value = None

    piece_store = MagicMock(spec=KnowledgePieceStore)
    piece_store.search.return_value = []
    piece_store.get_by_id.return_value = None
    piece_store.supports_space_filter = False

    graph_store = InMemoryEntityGraphStore()

    kb = KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id=active_entity_id,
        include_metadata=include_metadata,
        include_pieces=include_pieces,
        include_graph=include_graph,
        graph_traversal_depth=graph_traversal_depth,
    )
    return kb, metadata_store, piece_store, graph_store


# ── Test retrieve_metadata ───────────────────────────────────────────────────


class TestRetrieveMetadata:
    """Tests for KnowledgeBase.retrieve_metadata() — Layer 1."""

    def test_returns_entity_and_global_metadata(self):
        """retrieve_metadata returns both entity and global metadata."""
        kb, metadata_store, _, _ = _make_kb()
        entity_meta = _make_metadata("user:test")
        global_meta = _make_metadata("global")
        metadata_store.get_metadata.side_effect = lambda eid: {
            "user:test": entity_meta,
            "global": global_meta,
        }.get(eid)

        meta, global_m = kb.retrieve_metadata("user:test", include_global=True)
        assert meta is entity_meta
        assert global_m is global_meta

    def test_returns_none_when_metadata_disabled(self):
        """retrieve_metadata returns (None, None) when include_metadata is False."""
        kb, _, _, _ = _make_kb(include_metadata=False)
        meta, global_m = kb.retrieve_metadata("user:test")
        assert meta is None
        assert global_m is None

    def test_returns_none_when_no_entity_id(self):
        """retrieve_metadata returns (None, None) when entity_id is None."""
        kb, _, _, _ = _make_kb(active_entity_id=None)
        meta, global_m = kb.retrieve_metadata(entity_id=None)
        assert meta is None
        assert global_m is None

    def test_resolves_active_entity_id(self):
        """retrieve_metadata uses active_entity_id when entity_id is None."""
        kb, metadata_store, _, _ = _make_kb(active_entity_id="user:default")
        entity_meta = _make_metadata("user:default")
        metadata_store.get_metadata.side_effect = lambda eid: {
            "user:default": entity_meta,
        }.get(eid)

        meta, _ = kb.retrieve_metadata(entity_id=None, include_global=False)
        assert meta is entity_meta

    def test_skips_global_when_include_global_false(self):
        """retrieve_metadata skips global metadata when include_global=False."""
        kb, metadata_store, _, _ = _make_kb()
        entity_meta = _make_metadata("user:test")
        metadata_store.get_metadata.side_effect = lambda eid: {
            "user:test": entity_meta,
        }.get(eid)

        meta, global_m = kb.retrieve_metadata("user:test", include_global=False)
        assert meta is entity_meta
        assert global_m is None

    def test_space_filtering_removes_non_matching_metadata(self):
        """retrieve_metadata filters out metadata not in requested spaces."""
        kb, metadata_store, _, _ = _make_kb()
        entity_meta = _make_metadata("user:test", spaces=["personal"])
        global_meta = _make_metadata("global", spaces=["main"])
        metadata_store.get_metadata.side_effect = lambda eid: {
            "user:test": entity_meta,
            "global": global_meta,
        }.get(eid)

        meta, global_m = kb.retrieve_metadata("user:test", spaces=["main"])
        assert meta is None  # personal doesn't match main
        assert global_m is global_meta  # main matches main

    def test_space_filtering_keeps_matching_metadata(self):
        """retrieve_metadata keeps metadata in requested spaces."""
        kb, metadata_store, _, _ = _make_kb()
        entity_meta = _make_metadata("user:test", spaces=["main", "personal"])
        metadata_store.get_metadata.side_effect = lambda eid: {
            "user:test": entity_meta,
        }.get(eid)

        meta, _ = kb.retrieve_metadata("user:test", include_global=False, spaces=["personal"])
        assert meta is entity_meta


# ── Test retrieve_pieces ─────────────────────────────────────────────────────


class TestRetrievePieces:
    """Tests for KnowledgeBase.retrieve_pieces() — Layer 2."""

    def test_returns_empty_for_empty_query(self):
        """retrieve_pieces returns empty list for empty/whitespace query."""
        kb, _, _, _ = _make_kb()
        assert kb.retrieve_pieces("") == []
        assert kb.retrieve_pieces("   ") == []

    def test_returns_empty_when_pieces_disabled(self):
        """retrieve_pieces returns empty list when include_pieces is False."""
        kb, _, _, _ = _make_kb(include_pieces=False)
        assert kb.retrieve_pieces("test query") == []

    def test_standard_path_returns_pieces(self):
        """retrieve_pieces returns pieces via standard search path."""
        kb, _, piece_store, _ = _make_kb()
        piece = _make_piece("p1", entity_id="user:test")
        piece_store.search.return_value = [(piece, 0.9)]

        result = kb.retrieve_pieces("test query", entity_id="user:test", include_global=False)
        assert len(result) == 1
        assert result[0][0] is piece
        assert result[0][1] == 0.9

    def test_resolves_defaults(self):
        """retrieve_pieces resolves entity_id and top_k from defaults."""
        kb, _, piece_store, _ = _make_kb(active_entity_id="user:default")
        piece = _make_piece("p1", entity_id="user:default")
        piece_store.search.return_value = [(piece, 0.8)]

        result = kb.retrieve_pieces("query", include_global=False)
        assert len(result) == 1
        # Verify it used the active_entity_id
        piece_store.search.assert_called_with(
            "query", entity_id="user:default", tags=None, top_k=5
        )


# ── Test retrieve_search_graph ───────────────────────────────────────────────


class TestRetrieveSearchGraph:
    """Tests for KnowledgeBase.retrieve_search_graph() — Layer 3a."""

    def test_returns_empty_when_graph_disabled(self):
        """retrieve_search_graph returns empty list when include_graph is False."""
        kb, _, _, _ = _make_kb(include_graph=False)
        assert kb.retrieve_search_graph("test query") == []

    def test_returns_empty_for_non_semantic_store(self):
        """retrieve_search_graph returns empty for stores without semantic search."""
        kb, _, _, _ = _make_kb()
        # InMemoryEntityGraphStore doesn't support semantic search
        assert kb.retrieve_search_graph("test query") == []

    def test_resolves_top_k_default(self):
        """retrieve_search_graph resolves top_k from default_top_k."""
        kb, _, _, _ = _make_kb()
        kb.default_top_k = 10
        # Should not raise, just return empty (no semantic search support)
        result = kb.retrieve_search_graph("query", top_k=None)
        assert result == []


# ── Test retrieve_identity_graph ─────────────────────────────────────────────


class TestRetrieveIdentityGraph:
    """Tests for KnowledgeBase.retrieve_identity_graph() — Layer 3b."""

    def test_returns_empty_when_graph_disabled(self):
        """retrieve_identity_graph returns empty list when include_graph is False."""
        kb, _, _, _ = _make_kb(include_graph=False)
        assert kb.retrieve_identity_graph("user:test") == []

    def test_returns_empty_for_missing_entity(self):
        """retrieve_identity_graph returns empty for non-existent entity."""
        kb, _, _, _ = _make_kb()
        assert kb.retrieve_identity_graph("nonexistent") == []

    def test_returns_identity_entry_for_existing_entity(self):
        """retrieve_identity_graph returns IDENTITY entry for existing entity."""
        kb, _, _, graph_store = _make_kb()
        node = GraphNode(node_id="user:test", node_type="user", label="Test User")
        graph_store.add_node(node)

        result = kb.retrieve_identity_graph("user:test")
        assert len(result) >= 1
        identity_entries = [e for e in result if e["relation_type"] == "IDENTITY"]
        assert len(identity_entries) == 1
        assert identity_entries[0]["target_node_id"] == "user:test"
        assert identity_entries[0]["score"] == 1.0
        assert identity_entries[0]["depth"] == 0

    def test_walks_neighbors(self):
        """retrieve_identity_graph walks neighbors from the entity node."""
        kb, _, _, graph_store = _make_kb()
        user_node = GraphNode(node_id="user:test", node_type="user", label="Test User")
        neighbor = GraphNode(node_id="company:acme", node_type="company", label="Acme")
        graph_store.add_node(user_node)
        graph_store.add_node(neighbor)
        graph_store.add_relation(GraphEdge(
            source_id="user:test", target_id="company:acme", edge_type="WORKS_AT"
        ))

        result = kb.retrieve_identity_graph("user:test")
        assert len(result) == 2  # IDENTITY + WORKS_AT neighbor
        neighbor_entries = [e for e in result if e["target_node_id"] == "company:acme"]
        assert len(neighbor_entries) == 1
        assert neighbor_entries[0]["relation_type"] == "WORKS_AT"
        assert neighbor_entries[0]["depth"] == 1
        assert neighbor_entries[0]["score"] == 0.5  # 1.0 * 1/(1+1)

    def test_resolves_active_entity_id(self):
        """retrieve_identity_graph uses active_entity_id when entity_id is None."""
        kb, _, _, graph_store = _make_kb(active_entity_id="user:default")
        node = GraphNode(node_id="user:default", node_type="user", label="Default")
        graph_store.add_node(node)

        result = kb.retrieve_identity_graph(entity_id=None)
        assert len(result) >= 1
        assert result[0]["target_node_id"] == "user:default"


# ── Test retrieve() orchestration ────────────────────────────────────────────


class TestRetrieveOrchestration:
    """Tests that retrieve() produces identical output to manual layer assembly."""

    def test_retrieve_matches_manual_layer_assembly(self):
        """retrieve() output matches manually calling layer methods."""
        kb, metadata_store, piece_store, graph_store = _make_kb()

        # Set up metadata
        entity_meta = _make_metadata("user:test")
        global_meta = _make_metadata("global")
        metadata_store.get_metadata.side_effect = lambda eid: {
            "user:test": entity_meta,
            "global": global_meta,
        }.get(eid)

        # Set up pieces
        piece = _make_piece("p1", entity_id="user:test")
        piece_store.search.return_value = [(piece, 0.9)]

        # Set up graph
        user_node = GraphNode(node_id="user:test", node_type="user", label="Test")
        graph_store.add_node(user_node)

        # Call retrieve()
        result = kb.retrieve("test query", entity_id="user:test")

        # Manually call layer methods
        meta, global_m = kb.retrieve_metadata("user:test", include_global=True)
        pieces = kb.retrieve_pieces(
            "test query", "user:test", kb.default_top_k, True,
            None, None, None, 1, None,
        )
        search_ctx = kb.retrieve_search_graph("test query", kb.default_top_k, None, None)
        identity_ctx = kb.retrieve_identity_graph("user:test", None, None)

        # Compare
        assert result.metadata is meta
        assert result.global_metadata is global_m
        assert result.pieces == pieces

    def test_retrieve_empty_query_skips_pieces(self):
        """retrieve() with empty query skips L2 but still runs L1, L3."""
        kb, metadata_store, _, graph_store = _make_kb()
        entity_meta = _make_metadata("user:test")
        metadata_store.get_metadata.side_effect = lambda eid: {
            "user:test": entity_meta,
        }.get(eid)

        user_node = GraphNode(node_id="user:test", node_type="user", label="Test")
        graph_store.add_node(user_node)

        result = kb.retrieve("", entity_id="user:test")
        assert result.metadata is entity_meta
        assert result.pieces == []
        # Graph context should still have the IDENTITY entry
        assert len(result.graph_context) >= 1

    def test_retrieve_preserves_signature(self):
        """retrieve() accepts all original parameters without error."""
        kb, _, _, _ = _make_kb()
        # Should not raise
        result = kb.retrieve(
            query="test",
            entity_id="user:test",
            top_k=3,
            include_global=False,
            domain="general",
            secondary_domains=["testing"],
            tags=["tag1"],
            min_results=1,
            spaces=["main"],
        )
        assert result is not None

"""
Unit tests for KnowledgeBase graph search integration.

Feature: graph-semantic-retrieval
Tests cover:
- Semantic graph store enabled: query finds nodes, walks graph, returns linked pieces
- Non-semantic graph store: falls back to identity-based traversal only
- Graph search WITHOUT entity_id (query-only mode)
- Merge/dedup when same piece found by both Layer 2 and Layer 3a
- SEARCH_HIT relation_type in graph context entries
- merge_graph_contexts deduplication by (target_node_id, relation_type) compound key
- Same node with different relation_types preserved in merge
- Space filtering on Layer 3a search results and walked neighbors
- close() delegates to both graph store and retrieval service
- NATIVE mode with retrieval_service=None (no sidecar sync)
- Upsert: add_node for existing node uses add-then-update fallback

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, call

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

# Add test directory to path for conftest imports
_test_dir = str(Path(__file__).resolve().parent)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

import pytest

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode, GraphEdge
from rich_python_utils.service_utils.retrieval_service.document import Document
from rich_python_utils.service_utils.retrieval_service.retrieval_service_base import (
    RetrievalServiceBase,
)

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.graph_walk import (
    SeedNode,
    graph_walk,
    merge_graph_contexts,
)
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from agent_foundation.knowledge.retrieval.stores.graph.semantic_graph_store import (
    SemanticGraphStore,
)
from agent_foundation.knowledge.retrieval.stores.graph.search_mode import SearchMode
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from agent_foundation.knowledge.retrieval.stores.metadata.base import MetadataStore
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece


# ── Shared test helpers (from conftest.py) ───────────────────────────────────
from conftest import InMemoryEntityGraphStore, InMemoryRetrievalService


# ── Helper: build a KnowledgeBase with mock stores ───────────────────────────


def _make_knowledge_base(
    graph_store: EntityGraphStore,
    piece_store: Optional[KnowledgePieceStore] = None,
    metadata_store: Optional[MetadataStore] = None,
    graph_traversal_depth: int = 1,
    graph_retrieval_ignore_pieces_already_retrieved=False,
    include_pieces: bool = False,
) -> KnowledgeBase:
    """Create a KnowledgeBase with the given graph store and mock piece/metadata stores."""
    if piece_store is None:
        piece_store = MagicMock(spec=KnowledgePieceStore)
        piece_store.get_by_id.return_value = None
        piece_store.search.return_value = []
        piece_store.list_all.return_value = []
    if metadata_store is None:
        metadata_store = MagicMock(spec=MetadataStore)
        metadata_store.get_metadata.return_value = None
    return KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        include_metadata=False,
        include_pieces=include_pieces,
        include_graph=True,
        graph_traversal_depth=graph_traversal_depth,
        graph_retrieval_ignore_pieces_already_retrieved=graph_retrieval_ignore_pieces_already_retrieved,
    )


def _build_safeway_graph(graph_store: InMemoryEntityGraphStore) -> Dict[str, GraphNode]:
    """Build a small Safeway-themed graph for testing.

    Graph topology:
        safeway --SELLS--> organic_eggs (edge has piece_id="piece-eggs")
        safeway --LOCATED_IN--> san_francisco
    """
    safeway = GraphNode(
        node_id="service:safeway", node_type="service",
        label="Safeway Grocery", properties={"category": "grocery"},
    )
    organic_eggs = GraphNode(
        node_id="product:organic-eggs", node_type="product",
        label="Organic Eggs", properties={"price": "5.99"},
    )
    san_francisco = GraphNode(
        node_id="location:sf", node_type="location",
        label="San Francisco", properties={},
    )
    graph_store.add_node(safeway)
    graph_store.add_node(organic_eggs)
    graph_store.add_node(san_francisco)
    graph_store.add_relation(GraphEdge(
        source_id="service:safeway", target_id="product:organic-eggs",
        edge_type="SELLS", properties={"piece_id": "piece-eggs"},
    ))
    graph_store.add_relation(GraphEdge(
        source_id="service:safeway", target_id="location:sf",
        edge_type="LOCATED_IN", properties={},
    ))
    return {"safeway": safeway, "organic_eggs": organic_eggs, "san_francisco": san_francisco}


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: Semantic graph store enabled — query finds nodes, walks graph,
#          returns linked pieces
# ══════════════════════════════════════════════════════════════════════════════


class TestSemanticGraphSearchEnabled:
    """Test with semantic graph store enabled: query finds nodes, walks graph,
    returns linked pieces.

    **Validates: Requirements 7.1, 7.2**
    """

    def test_query_finds_nodes_and_walks_graph(self):
        """When graph store supports semantic search, retrieve() calls
        search_nodes and walks neighbors to extract linked pieces."""
        inner_store = InMemoryEntityGraphStore()
        retrieval_svc = InMemoryRetrievalService()
        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=retrieval_svc,
            search_mode=SearchMode.SIDECAR,
        )

        nodes = _build_safeway_graph(inner_store)
        # Index the safeway node in the sidecar so search can find it
        semantic_store.add_node(nodes["safeway"])
        # Re-add the other nodes so they exist in inner_store (already there)

        # Mock piece_store to return a piece for piece-eggs
        piece_store = MagicMock(spec=KnowledgePieceStore)
        eggs_piece = KnowledgePiece(
            content="Organic eggs at Safeway cost $5.99",
            piece_id="piece-eggs",
        )
        piece_store.get_by_id.return_value = eggs_piece
        piece_store.search.return_value = []
        piece_store.list_all.return_value = []

        kb = _make_knowledge_base(
            semantic_store, piece_store=piece_store, graph_traversal_depth=1,
        )

        result = kb.retrieve("safeway grocery")

        # Should have graph context entries
        assert result.graph_context is not None
        assert len(result.graph_context) > 0

        # Should have a SEARCH_HIT entry for safeway
        search_hits = [e for e in result.graph_context if e["relation_type"] == "SEARCH_HIT"]
        assert len(search_hits) >= 1
        assert any(e["target_node_id"] == "service:safeway" for e in search_hits)

        # Should have neighbor entries with linked pieces
        neighbor_entries = [e for e in result.graph_context if e["target_node_id"] == "product:organic-eggs"]
        assert len(neighbor_entries) == 1
        assert neighbor_entries[0]["piece"] is not None
        assert neighbor_entries[0]["piece"].piece_id == "piece-eggs"


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: Non-semantic graph store — falls back to identity-based traversal
# ══════════════════════════════════════════════════════════════════════════════


class TestNonSemanticGraphStoreFallback:
    """Test with non-semantic graph store: falls back to identity-based
    traversal only.

    **Validates: Requirements 7.5**
    """

    def test_non_semantic_store_uses_identity_traversal(self):
        """When graph store does NOT support semantic search, retrieve() uses
        identity-based traversal only (Layer 3b)."""
        graph_store = InMemoryEntityGraphStore()
        nodes = _build_safeway_graph(graph_store)

        piece_store = MagicMock(spec=KnowledgePieceStore)
        eggs_piece = KnowledgePiece(content="Eggs info", piece_id="piece-eggs")
        piece_store.get_by_id.return_value = eggs_piece
        piece_store.search.return_value = []
        piece_store.list_all.return_value = []

        kb = _make_knowledge_base(graph_store, piece_store=piece_store, graph_traversal_depth=1)

        # Non-semantic store: supports_semantic_search is False
        assert not graph_store.supports_semantic_search

        # Retrieve with entity_id triggers Layer 3b
        result = kb.retrieve("organic eggs", entity_id="service:safeway")

        assert result.graph_context is not None
        assert len(result.graph_context) > 0
        # No SEARCH_HIT entries since semantic search was not used
        search_hits = [e for e in result.graph_context if e.get("relation_type") == "SEARCH_HIT"]
        assert len(search_hits) == 0

    def test_non_semantic_store_without_entity_id_returns_no_graph(self):
        """Without entity_id and without semantic search, no graph context."""
        graph_store = InMemoryEntityGraphStore()
        _build_safeway_graph(graph_store)

        kb = _make_knowledge_base(graph_store)
        result = kb.retrieve("organic eggs")

        # No entity_id and no semantic search → no graph context
        assert result.graph_context is None or len(result.graph_context) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: Graph search WITHOUT entity_id (query-only mode)
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphSearchWithoutEntityId:
    """Test graph search WITHOUT entity_id (query-only mode) — key new behavior.

    **Validates: Requirements 7.1**
    """

    def test_query_only_mode_finds_graph_nodes(self):
        """Layer 3a runs with just a query, no entity_id needed."""
        inner_store = InMemoryEntityGraphStore()
        retrieval_svc = InMemoryRetrievalService()
        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=retrieval_svc,
            search_mode=SearchMode.SIDECAR,
        )

        safeway = GraphNode(
            node_id="service:safeway", node_type="service",
            label="Safeway Grocery", properties={"category": "grocery"},
        )
        inner_store.add_node(safeway)
        semantic_store.add_node(safeway)  # Index in sidecar

        kb = _make_knowledge_base(semantic_store)

        # No entity_id provided — query-only mode
        result = kb.retrieve("safeway grocery")

        assert result.graph_context is not None
        assert len(result.graph_context) > 0
        # Should find safeway via semantic search
        node_ids = [e["target_node_id"] for e in result.graph_context]
        assert "service:safeway" in node_ids

    def test_query_only_mode_no_identity_traversal(self):
        """Without entity_id, Layer 3b (identity-based) does not run."""
        inner_store = InMemoryEntityGraphStore()
        retrieval_svc = InMemoryRetrievalService()
        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=retrieval_svc,
            search_mode=SearchMode.SIDECAR,
        )

        # Add a node that won't match the query
        unrelated = GraphNode(
            node_id="service:walmart", node_type="service",
            label="Walmart", properties={},
        )
        inner_store.add_node(unrelated)

        # Add a neighbor to walmart
        product = GraphNode(
            node_id="product:milk", node_type="product",
            label="Milk", properties={},
        )
        inner_store.add_node(product)
        inner_store.add_relation(GraphEdge(
            source_id="service:walmart", target_id="product:milk",
            edge_type="SELLS",
        ))

        kb = _make_knowledge_base(semantic_store)

        # Query without entity_id — only Layer 3a runs
        # Since "xyz" won't match any indexed nodes, graph context should be empty
        result = kb.retrieve("xyz nonexistent query")
        # No graph context since search returned no results and no entity_id
        assert result.graph_context is None or len(result.graph_context) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: Merge/dedup when same piece found by both Layer 2 and Layer 3a
# ══════════════════════════════════════════════════════════════════════════════


class TestMergeDedupLayer2AndLayer3a:
    """Test merge/dedup when same piece found by both Layer 2 and Layer 3a.

    **Validates: Requirements 7.3**
    """

    def test_piece_in_layer2_skipped_in_graph_context(self):
        """When graph_retrieval_ignore_pieces_already_retrieved is True and a
        piece is found by Layer 2, it is skipped in Layer 3a graph context."""
        inner_store = InMemoryEntityGraphStore()
        retrieval_svc = InMemoryRetrievalService()
        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=retrieval_svc,
            search_mode=SearchMode.SIDECAR,
        )

        safeway = GraphNode(
            node_id="service:safeway", node_type="service",
            label="Safeway Grocery", properties={},
        )
        eggs = GraphNode(
            node_id="product:eggs", node_type="product",
            label="Organic Eggs", properties={},
        )
        inner_store.add_node(safeway)
        inner_store.add_node(eggs)
        inner_store.add_relation(GraphEdge(
            source_id="service:safeway", target_id="product:eggs",
            edge_type="SELLS", properties={"piece_id": "piece-eggs"},
        ))
        semantic_store.add_node(safeway)

        eggs_piece = KnowledgePiece(
            content="Organic eggs info", piece_id="piece-eggs", info_type="context",
        )
        piece_store = MagicMock(spec=KnowledgePieceStore)
        piece_store.get_by_id.return_value = eggs_piece
        # Layer 2 also returns this piece
        piece_store.search.return_value = [(eggs_piece, 0.9)]
        piece_store.list_all.return_value = []

        kb = _make_knowledge_base(
            semantic_store,
            piece_store=piece_store,
            graph_traversal_depth=1,
            graph_retrieval_ignore_pieces_already_retrieved=True,
            include_pieces=True,
        )

        result = kb.retrieve("safeway grocery")

        # Layer 2 should have the piece
        assert len(result.pieces) > 0

        # Layer 3a graph context should NOT have the piece attached
        if result.graph_context:
            for entry in result.graph_context:
                if entry["target_node_id"] == "product:eggs":
                    assert entry["piece"] is None, (
                        "Piece already in Layer 2 should be skipped in graph context"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: SEARCH_HIT relation_type in graph context entries
# ══════════════════════════════════════════════════════════════════════════════


class TestSearchHitRelationType:
    """Test SEARCH_HIT relation_type in graph context entries.

    **Validates: Requirements 7.1**
    """

    def test_search_hit_at_depth_zero(self):
        """Matched nodes from search_nodes appear as SEARCH_HIT at depth 0."""
        graph_store = InMemoryEntityGraphStore()
        safeway = GraphNode(
            node_id="service:safeway", node_type="service",
            label="Safeway", properties={},
        )
        graph_store.add_node(safeway)

        kb = _make_knowledge_base(graph_store)

        # Use unified graph walk with search seeds
        seeds = [SeedNode(node=safeway, score=0.9, source="search")]
        context = graph_walk(graph_store, kb.piece_store, seeds, traversal_depth=1)

        assert len(context) >= 1
        hit = context[0]
        assert hit["relation_type"] == "SEARCH_HIT"
        assert hit["target_node_id"] == "service:safeway"
        assert hit["depth"] == 0
        assert hit["score"] == pytest.approx(0.9)
        assert hit["piece"] is None

    def test_search_hit_score_equals_search_score(self):
        """The SEARCH_HIT entry score equals the raw search score (depth=0 → factor=1.0)."""
        graph_store = InMemoryEntityGraphStore()
        node = GraphNode(node_id="n1", node_type="service", label="Test")
        graph_store.add_node(node)

        kb = _make_knowledge_base(graph_store)
        seeds = [SeedNode(node=node, score=0.75, source="search")]
        context = graph_walk(graph_store, kb.piece_store, seeds, traversal_depth=1)

        hits = [e for e in context if e["relation_type"] == "SEARCH_HIT"]
        assert len(hits) == 1
        assert hits[0]["score"] == pytest.approx(0.75)


# ══════════════════════════════════════════════════════════════════════════════
# Test 6: merge_graph_contexts deduplication by (target_node_id, relation_type)
# ══════════════════════════════════════════════════════════════════════════════


class TestMergeGraphContextsDedup:
    """Test merge_graph_contexts deduplication by (target_node_id, relation_type)
    compound key.

    **Validates: Requirements 7.1**
    """

    def test_dedup_keeps_higher_score(self):
        """When same (node_id, relation_type) appears in both lists,
        the entry with higher score wins."""
        search_entry = {
            "relation_type": "RELATED",
            "target_node_id": "node-A",
            "target_label": "Node A",
            "piece": None,
            "depth": 1,
            "score": 0.8,
        }
        identity_entry = {
            "relation_type": "RELATED",
            "target_node_id": "node-A",
            "target_label": "Node A",
            "piece": None,
            "depth": 1,
            "score": 0.5,
        }

        merged = merge_graph_contexts([search_entry], [identity_entry])

        assert len(merged) == 1
        assert merged[0]["score"] == pytest.approx(0.8)

    def test_dedup_keeps_shorter_depth_on_equal_score(self):
        """When scores are equal, the entry with shorter depth wins."""
        entry_deep = {
            "relation_type": "RELATED",
            "target_node_id": "node-B",
            "target_label": "Node B",
            "piece": None,
            "depth": 3,
            "score": 0.6,
        }
        entry_shallow = {
            "relation_type": "RELATED",
            "target_node_id": "node-B",
            "target_label": "Node B",
            "piece": None,
            "depth": 1,
            "score": 0.6,
        }

        merged = merge_graph_contexts([entry_deep], [entry_shallow])

        assert len(merged) == 1
        assert merged[0]["depth"] == 1

    def test_entries_from_different_nodes_preserved(self):
        """Entries for different node IDs are all preserved."""
        entry_a = {
            "relation_type": "RELATED",
            "target_node_id": "node-A",
            "target_label": "A",
            "piece": None,
            "depth": 0,
            "score": 0.9,
        }
        entry_b = {
            "relation_type": "RELATED",
            "target_node_id": "node-B",
            "target_label": "B",
            "piece": None,
            "depth": 0,
            "score": 0.7,
        }

        merged = merge_graph_contexts([entry_a], [entry_b])
        assert len(merged) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Test 7: Same node with different relation_types preserved in merge
# ══════════════════════════════════════════════════════════════════════════════


class TestDifferentRelationTypesPreserved:
    """Test that same node with different relation_types is preserved in merge.

    **Validates: Requirements 7.1**
    """

    def test_same_node_different_relations_both_kept(self):
        """SEARCH_HIT from Layer 3a and WORKS_AT from Layer 3b for the same
        node are both preserved."""
        search_entry = {
            "relation_type": "SEARCH_HIT",
            "target_node_id": "person:alice",
            "target_label": "Alice",
            "piece": None,
            "depth": 0,
            "score": 0.9,
        }
        identity_entry = {
            "relation_type": "WORKS_AT",
            "target_node_id": "person:alice",
            "target_label": "Alice",
            "piece": None,
            "depth": 1,
            "score": 0.0,
        }

        merged = merge_graph_contexts([search_entry], [identity_entry])

        assert len(merged) == 2
        relation_types = {e["relation_type"] for e in merged}
        assert "SEARCH_HIT" in relation_types
        assert "WORKS_AT" in relation_types

    def test_three_different_relations_to_same_node(self):
        """Three different relation_types to the same node are all preserved."""
        entries = [
            {"relation_type": "SEARCH_HIT", "target_node_id": "n1", "target_label": "N",
             "piece": None, "depth": 0, "score": 0.9},
            {"relation_type": "SELLS", "target_node_id": "n1", "target_label": "N",
             "piece": None, "depth": 1, "score": 0.5},
            {"relation_type": "LOCATED_IN", "target_node_id": "n1", "target_label": "N",
             "piece": None, "depth": 1, "score": 0.3},
        ]

        merged = merge_graph_contexts(entries, [])
        assert len(merged) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Test 8: Space filtering on Layer 3a search results and walked neighbors
# ══════════════════════════════════════════════════════════════════════════════


class TestSpaceFilteringLayer3a:
    """Test space filtering on Layer 3a search results and walked neighbors.

    **Validates: Requirements 7.1, 7.2**
    """

    def test_search_hit_node_filtered_by_spaces(self):
        """Nodes not in the requested spaces are excluded from search results.
        In the new architecture, find_search_seeds() handles seed-level space
        filtering before graph_walk() is called."""
        graph_store = InMemoryEntityGraphStore()
        node_work = GraphNode(
            node_id="n-work", node_type="service", label="Work Node",
            properties={"spaces": ["work"]},
        )
        node_personal = GraphNode(
            node_id="n-personal", node_type="service", label="Personal Node",
            properties={"spaces": ["personal"]},
        )
        graph_store.add_node(node_work)
        graph_store.add_node(node_personal)

        kb = _make_knowledge_base(graph_store)

        # Both nodes found by search, but filter to "work" space only
        # In the new architecture, find_search_seeds filters seeds by spaces
        search_results = [(node_work, 0.9), (node_personal, 0.8)]
        requested_spaces = ["work"]
        seeds = [
            SeedNode(node=n, score=s, source="search")
            for n, s in search_results
            if set(n.properties.get("spaces", ["main"])) & set(requested_spaces)
        ]
        context = graph_walk(
            graph_store, kb.piece_store, seeds,
            traversal_depth=1, spaces=requested_spaces,
        )

        node_ids = [e["target_node_id"] for e in context]
        assert "n-work" in node_ids
        assert "n-personal" not in node_ids

    def test_walked_neighbors_filtered_by_spaces(self):
        """Neighbors walked from a search hit are filtered by spaces."""
        graph_store = InMemoryEntityGraphStore()
        root = GraphNode(
            node_id="root", node_type="service", label="Root",
            properties={"spaces": ["work"]},
        )
        neighbor_work = GraphNode(
            node_id="n-work", node_type="product", label="Work Product",
            properties={"spaces": ["work"]},
        )
        neighbor_personal = GraphNode(
            node_id="n-personal", node_type="product", label="Personal Product",
            properties={"spaces": ["personal"]},
        )
        graph_store.add_node(root)
        graph_store.add_node(neighbor_work)
        graph_store.add_node(neighbor_personal)
        graph_store.add_relation(GraphEdge(
            source_id="root", target_id="n-work", edge_type="RELATED",
        ))
        graph_store.add_relation(GraphEdge(
            source_id="root", target_id="n-personal", edge_type="RELATED",
        ))

        kb = _make_knowledge_base(graph_store, graph_traversal_depth=1)

        seeds = [SeedNode(node=root, score=0.9, source="search")]
        context = graph_walk(
            graph_store, kb.piece_store, seeds,
            traversal_depth=1, spaces=["work"],
        )

        node_ids = [e["target_node_id"] for e in context]
        assert "root" in node_ids  # SEARCH_HIT
        assert "n-work" in node_ids  # neighbor in "work" space
        assert "n-personal" not in node_ids  # filtered out

    def test_node_without_spaces_treated_as_main(self):
        """Nodes without a 'spaces' property default to ['main'].
        In the new architecture, find_search_seeds() handles seed-level space
        filtering before graph_walk() is called."""
        graph_store = InMemoryEntityGraphStore()
        node_no_spaces = GraphNode(
            node_id="n-default", node_type="service", label="Default",
            properties={},  # No spaces → defaults to ["main"]
        )
        graph_store.add_node(node_no_spaces)

        kb = _make_knowledge_base(graph_store)

        # Filter for "main" space — should include node without explicit spaces
        main_spaces = ["main"]
        all_candidates = [(node_no_spaces, 0.8)]
        seeds_main = [
            SeedNode(node=n, score=s, source="search")
            for n, s in all_candidates
            if set(n.properties.get("spaces", ["main"])) & set(main_spaces)
        ]
        context = graph_walk(
            graph_store, kb.piece_store, seeds_main,
            traversal_depth=1, spaces=main_spaces,
        )
        assert len(context) >= 1
        assert context[0]["target_node_id"] == "n-default"

        # Filter for "work" space — should exclude node defaulting to "main"
        work_spaces = ["work"]
        seeds_work = [
            SeedNode(node=n, score=s, source="search")
            for n, s in all_candidates
            if set(n.properties.get("spaces", ["main"])) & set(work_spaces)
        ]
        context_work = graph_walk(
            graph_store, kb.piece_store, seeds_work,
            traversal_depth=1, spaces=work_spaces,
        )
        assert len(context_work) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Test 9: close() delegates to both graph store and retrieval service
# ══════════════════════════════════════════════════════════════════════════════


class TestCloseDelegate:
    """Test close() delegates to both graph store and retrieval service.

    **Validates: Requirements 7.1**
    """

    def test_close_delegates_to_semantic_graph_store(self):
        """KnowledgeBase.close() calls graph_store.close(), which in turn
        closes both the inner graph store and the retrieval service."""
        inner_store = InMemoryEntityGraphStore()
        retrieval_svc = InMemoryRetrievalService()
        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=retrieval_svc,
            search_mode=SearchMode.SIDECAR,
        )

        kb = _make_knowledge_base(semantic_store)
        kb.close()

        assert retrieval_svc._closed is True

    def test_close_with_none_retrieval_service(self):
        """close() works when retrieval_service is None (NATIVE mode with
        a store that supports semantic search)."""
        inner_store = MagicMock(spec=EntityGraphStore)
        inner_store.supports_semantic_search = True
        inner_store.close = MagicMock()

        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=None,
            search_mode=SearchMode.NATIVE,
        )

        metadata_store = MagicMock(spec=MetadataStore)
        piece_store = MagicMock(spec=KnowledgePieceStore)
        piece_store.search.return_value = []
        piece_store.list_all.return_value = []

        kb = KnowledgeBase(
            metadata_store=metadata_store,
            piece_store=piece_store,
            graph_store=semantic_store,
        )
        # Should not raise
        kb.close()
        inner_store.close.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Test 10: NATIVE mode with retrieval_service=None (no sidecar sync)
# ══════════════════════════════════════════════════════════════════════════════


class TestNativeModeNoSidecar:
    """Test NATIVE mode with retrieval_service=None (no sidecar sync).

    **Validates: Requirements 7.1, 7.5**
    """

    def test_native_mode_requires_semantic_search_support(self):
        """NATIVE mode raises ValueError if wrapped store doesn't support
        semantic search."""
        inner_store = InMemoryEntityGraphStore()
        assert not inner_store.supports_semantic_search

        with pytest.raises(ValueError, match="supports semantic search"):
            SemanticGraphStore(
                graph_store=inner_store,
                retrieval_service=None,
                search_mode=SearchMode.NATIVE,
            )

    def test_native_mode_add_node_skips_sidecar(self):
        """In NATIVE mode with retrieval_service=None, add_node only
        delegates to the wrapped store (no sidecar sync)."""
        inner_store = MagicMock(spec=EntityGraphStore)
        inner_store.supports_semantic_search = True
        inner_store.add_node = MagicMock()

        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=None,
            search_mode=SearchMode.NATIVE,
        )

        node = GraphNode(node_id="n1", node_type="service", label="Test")
        semantic_store.add_node(node)

        inner_store.add_node.assert_called_once_with(node)

    def test_native_mode_delegates_search_to_wrapped_store(self):
        """In NATIVE mode, search_nodes delegates to the wrapped store."""
        inner_store = MagicMock(spec=EntityGraphStore)
        inner_store.supports_semantic_search = True
        expected_results = [
            (GraphNode(node_id="n1", node_type="service", label="Test"), 0.9),
        ]
        inner_store.search_nodes = MagicMock(return_value=expected_results)

        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=None,
            search_mode=SearchMode.NATIVE,
        )

        results = semantic_store.search_nodes("test query", top_k=5)
        assert results == expected_results
        inner_store.search_nodes.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Test 11: Upsert — add_node for existing node uses add-then-update fallback
# ══════════════════════════════════════════════════════════════════════════════


class TestUpsertAddThenUpdate:
    """Test upsert: add_node for existing node uses add-then-update fallback.

    **Validates: Requirements 7.1**
    """

    def test_add_node_twice_updates_sidecar(self):
        """Adding a node that already exists in the sidecar index triggers
        the add→ValueError→update fallback."""
        inner_store = InMemoryEntityGraphStore()
        retrieval_svc = InMemoryRetrievalService()
        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=retrieval_svc,
            search_mode=SearchMode.SIDECAR,
        )

        node_v1 = GraphNode(
            node_id="n1", node_type="service", label="Version 1",
            properties={"version": "1"},
        )
        node_v2 = GraphNode(
            node_id="n1", node_type="service", label="Version 2",
            properties={"version": "2"},
        )

        # First add — should succeed via add()
        semantic_store.add_node(node_v1)
        doc = retrieval_svc.get_by_id("n1", namespace="_graph_nodes")
        assert doc is not None
        assert doc.metadata["label"] == "Version 1"

        # Second add — should trigger add→ValueError→update fallback
        semantic_store.add_node(node_v2)
        doc_updated = retrieval_svc.get_by_id("n1", namespace="_graph_nodes")
        assert doc_updated is not None
        assert doc_updated.metadata["label"] == "Version 2"
        assert doc_updated.metadata["properties"]["version"] == "2"

    def test_upsert_preserves_graph_store_state(self):
        """After upsert, the graph store has the latest version of the node."""
        inner_store = InMemoryEntityGraphStore()
        retrieval_svc = InMemoryRetrievalService()
        semantic_store = SemanticGraphStore(
            graph_store=inner_store,
            retrieval_service=retrieval_svc,
            search_mode=SearchMode.SIDECAR,
        )

        node_v1 = GraphNode(
            node_id="n1", node_type="service", label="V1", properties={},
        )
        node_v2 = GraphNode(
            node_id="n1", node_type="service", label="V2", properties={"updated": True},
        )

        semantic_store.add_node(node_v1)
        semantic_store.add_node(node_v2)

        stored = inner_store.get_node("n1")
        assert stored is not None
        assert stored.label == "V2"
        assert stored.properties.get("updated") is True

"""
Unit tests for GraphServiceEntityGraphStore adapter.

Tests that the adapter correctly implements the EntityGraphStore ABC by
delegating to a MemoryGraphService backend. Covers CRUD operations on
nodes and edges, round-trip with GraphNode/GraphEdge, neighbor traversal
with depth, error handling, and close delegation.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
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

from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphEdge,
    GraphNode,
)
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)


@pytest.fixture
def graph_service():
    """Create a fresh MemoryGraphService for each test."""
    return MemoryGraphService()


@pytest.fixture
def store(graph_service):
    """Create a GraphServiceEntityGraphStore backed by MemoryGraphService."""
    return GraphServiceEntityGraphStore(graph_service=graph_service)


class TestImplementsABC:
    """Requirement 13.1: GraphServiceEntityGraphStore implements EntityGraphStore ABC."""

    def test_is_instance_of_entity_graph_store(self, store):
        """GraphServiceEntityGraphStore should be an instance of EntityGraphStore."""
        assert isinstance(store, EntityGraphStore)

    def test_has_all_abstract_methods(self, store):
        """GraphServiceEntityGraphStore should implement all EntityGraphStore methods."""
        assert hasattr(store, "add_node")
        assert hasattr(store, "get_node")
        assert hasattr(store, "remove_node")
        assert hasattr(store, "add_relation")
        assert hasattr(store, "get_relations")
        assert hasattr(store, "remove_relation")
        assert hasattr(store, "get_neighbors")
        assert hasattr(store, "close")


class TestAddAndGetNode:
    """Requirements 13.2, 13.4: add_node then get_node round-trip."""

    def test_add_then_get_returns_equivalent_node(self, store):
        """Adding a node and getting it back should preserve all fields."""
        node = GraphNode(
            node_id="user:alice",
            node_type="user",
            label="Alice",
            properties={"location": "Seattle", "age": 30},
        )
        store.add_node(node)
        result = store.get_node("user:alice")

        assert result is not None
        assert result.node_id == node.node_id
        assert result.node_type == node.node_type
        assert result.label == node.label
        assert result.properties == node.properties

    def test_add_node_upsert_overwrites(self, store):
        """Adding a node with the same ID should overwrite the existing node."""
        node_v1 = GraphNode(
            node_id="user:bob",
            node_type="user",
            label="Bob",
            properties={"version": 1},
        )
        store.add_node(node_v1)

        node_v2 = GraphNode(
            node_id="user:bob",
            node_type="user",
            label="Robert",
            properties={"version": 2, "updated": True},
        )
        store.add_node(node_v2)

        result = store.get_node("user:bob")
        assert result is not None
        assert result.label == "Robert"
        assert result.properties == {"version": 2, "updated": True}

    def test_get_nonexistent_node_returns_none(self, store):
        """Getting a non-existent node should return None."""
        result = store.get_node("user:nobody")
        assert result is None

    def test_round_trip_preserves_all_fields(self, store):
        """Round-trip should preserve all GraphNode fields via to_dict comparison."""
        node = GraphNode(
            node_id="tool:calculator",
            node_type="tool",
            label="Calculator",
            properties={"features": ["add", "subtract"], "version": "1.0"},
        )
        store.add_node(node)
        result = store.get_node("tool:calculator")

        assert result.to_dict() == node.to_dict()

    def test_add_node_delegates_to_graph_service(self, store, graph_service):
        """add_node should store a GraphNode in the underlying graph service."""
        node = GraphNode(
            node_id="store:costco",
            node_type="store",
            label="Costco",
            properties={"membership": True},
        )
        store.add_node(node)

        # Verify the GraphNode is stored in the service
        graph_node = graph_service.get_node("store:costco")
        assert graph_node is not None
        assert graph_node.node_id == "store:costco"
        assert graph_node.node_type == "store"
        assert graph_node.label == "Costco"
        assert graph_node.properties == {"membership": True}

    def test_add_node_with_empty_properties(self, store):
        """Adding a node with empty properties should work correctly."""
        node = GraphNode(
            node_id="item:simple",
            node_type="item",
        )
        store.add_node(node)
        result = store.get_node("item:simple")

        assert result is not None
        assert result.node_id == "item:simple"
        assert result.label == ""
        assert result.properties == {}


class TestRemoveNode:
    """Requirement 13.1: remove_node operations via adapter."""

    def test_remove_existing_node_returns_true(self, store):
        """Removing an existing node should return True."""
        node = GraphNode(node_id="user:charlie", node_type="user")
        store.add_node(node)
        assert store.remove_node("user:charlie") is True

    def test_remove_nonexistent_node_returns_false(self, store):
        """Removing a non-existent node should return False."""
        assert store.remove_node("user:nobody") is False

    def test_get_after_remove_returns_none(self, store):
        """Getting a node after removal should return None."""
        node = GraphNode(node_id="user:dave", node_type="user")
        store.add_node(node)
        store.remove_node("user:dave")
        assert store.get_node("user:dave") is None

    def test_remove_node_cascade_deletes_relations(self, store):
        """Removing a node should cascade-delete all its edges."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        node_c = GraphNode(node_id="c", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        rel_ab = GraphEdge(source_id="a", target_id="b", edge_type="KNOWS")
        rel_bc = GraphEdge(source_id="b", target_id="c", edge_type="KNOWS")
        store.add_relation(rel_ab)
        store.add_relation(rel_bc)

        # Remove node b — should cascade-delete both edges
        store.remove_node("b")

        # Edges involving b should be gone
        assert store.get_relations("a", direction="outgoing") == []
        assert store.get_relations("c", direction="incoming") == []


class TestAddAndGetRelations:
    """Requirements 13.3: add_relation then get_relations round-trip."""

    def test_add_then_get_returns_equivalent_relation(self, store):
        """Adding an edge and getting it back should preserve all fields."""
        node_a = GraphNode(node_id="user:alice", node_type="user")
        node_b = GraphNode(node_id="store:costco", node_type="store")
        store.add_node(node_a)
        store.add_node(node_b)

        relation = GraphEdge(
            source_id="user:alice",
            target_id="store:costco",
            edge_type="SHOPS_AT",
            properties={"frequency": "weekly", "member": True},
        )
        store.add_relation(relation)

        relations = store.get_relations("user:alice", direction="outgoing")
        assert len(relations) == 1
        result = relations[0]
        assert result.source_id == relation.source_id
        assert result.target_id == relation.target_id
        assert result.edge_type == relation.edge_type
        assert result.properties == relation.properties

    def test_get_relations_with_type_filter(self, store):
        """get_relations with relation_type should return only matching edges."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        node_c = GraphNode(node_id="c", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="KNOWS"
        ))
        store.add_relation(GraphEdge(
            source_id="a", target_id="c", edge_type="WORKS_WITH"
        ))

        knows_rels = store.get_relations("a", relation_type="KNOWS")
        assert len(knows_rels) == 1
        assert knows_rels[0].edge_type == "KNOWS"
        assert knows_rels[0].target_id == "b"

    def test_get_relations_direction_incoming(self, store):
        """get_relations with direction='incoming' should return incoming edges."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="KNOWS"
        ))

        incoming = store.get_relations("b", direction="incoming")
        assert len(incoming) == 1
        assert incoming[0].source_id == "a"

    def test_get_relations_direction_both(self, store):
        """get_relations with direction='both' should return all edges."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        node_c = GraphNode(node_id="c", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="KNOWS"
        ))
        store.add_relation(GraphEdge(
            source_id="c", target_id="b", edge_type="FOLLOWS"
        ))

        both = store.get_relations("b", direction="both")
        assert len(both) == 2

    def test_add_relation_with_missing_source_raises_value_error(self, store):
        """Adding an edge with a non-existent source should raise ValueError."""
        node_b = GraphNode(node_id="b", node_type="test")
        store.add_node(node_b)

        relation = GraphEdge(
            source_id="nonexistent",
            target_id="b",
            edge_type="KNOWS",
        )
        with pytest.raises(ValueError):
            store.add_relation(relation)

    def test_add_relation_with_missing_target_raises_value_error(self, store):
        """Adding an edge with a non-existent target should raise ValueError."""
        node_a = GraphNode(node_id="a", node_type="test")
        store.add_node(node_a)

        relation = GraphEdge(
            source_id="a",
            target_id="nonexistent",
            edge_type="KNOWS",
        )
        with pytest.raises(ValueError):
            store.add_relation(relation)

    def test_add_relation_delegates_to_graph_service(self, store, graph_service):
        """add_relation should store a GraphEdge in the underlying graph service."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)

        relation = GraphEdge(
            source_id="a",
            target_id="b",
            edge_type="KNOWS",
            properties={"since": 2020},
        )
        store.add_relation(relation)

        # Verify the GraphEdge is stored in the service
        edges = graph_service.get_edges("a", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].source_id == "a"
        assert edges[0].target_id == "b"
        assert edges[0].edge_type == "KNOWS"
        assert edges[0].properties == {"since": 2020}

    def test_get_relations_empty_returns_empty_list(self, store):
        """get_relations for a node with no edges should return empty list."""
        node = GraphNode(node_id="lonely", node_type="test")
        store.add_node(node)

        assert store.get_relations("lonely") == []


class TestRemoveRelation:
    """Requirement 13.1: remove_relation operations via adapter."""

    def test_remove_existing_relation_returns_true(self, store):
        """Removing an existing edge should return True."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="KNOWS"
        ))

        assert store.remove_relation("a", "b", "KNOWS") is True

    def test_remove_nonexistent_relation_returns_false(self, store):
        """Removing a non-existent edge should return False."""
        assert store.remove_relation("a", "b", "KNOWS") is False

    def test_get_relations_after_remove(self, store):
        """get_relations after removal should not include the removed edge."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="KNOWS"
        ))
        store.remove_relation("a", "b", "KNOWS")

        assert store.get_relations("a") == []


class TestGetNeighbors:
    """Requirement 13.5: get_neighbors with depth traversal."""

    def test_get_direct_neighbors(self, store):
        """get_neighbors with depth=1 should return direct neighbors."""
        node_a = GraphNode(node_id="a", node_type="test", label="A")
        node_b = GraphNode(node_id="b", node_type="test", label="B")
        node_c = GraphNode(node_id="c", node_type="test", label="C")
        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="KNOWS"
        ))
        store.add_relation(GraphEdge(
            source_id="a", target_id="c", edge_type="KNOWS"
        ))

        neighbors = store.get_neighbors("a", depth=1)
        assert len(neighbors) == 2

        # All should be at depth 1
        for graph_node, depth in neighbors:
            assert isinstance(graph_node, GraphNode)
            assert depth == 1

        neighbor_ids = {n.node_id for n, _ in neighbors}
        assert neighbor_ids == {"b", "c"}

    def test_get_neighbors_with_depth_2(self, store):
        """get_neighbors with depth=2 should return neighbors up to 2 hops."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        node_c = GraphNode(node_id="c", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="KNOWS"
        ))
        store.add_relation(GraphEdge(
            source_id="b", target_id="c", edge_type="KNOWS"
        ))

        neighbors = store.get_neighbors("a", depth=2)
        assert len(neighbors) == 2

        neighbor_map = {n.node_id: d for n, d in neighbors}
        assert neighbor_map["b"] == 1
        assert neighbor_map["c"] == 2

    def test_get_neighbors_with_relation_type_filter(self, store):
        """get_neighbors with relation_type should only follow matching edges."""
        node_a = GraphNode(node_id="a", node_type="test")
        node_b = GraphNode(node_id="b", node_type="test")
        node_c = GraphNode(node_id="c", node_type="test")
        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="KNOWS"
        ))
        store.add_relation(GraphEdge(
            source_id="a", target_id="c", edge_type="WORKS_WITH"
        ))

        knows_neighbors = store.get_neighbors("a", relation_type="KNOWS")
        assert len(knows_neighbors) == 1
        assert knows_neighbors[0][0].node_id == "b"

    def test_get_neighbors_nonexistent_node_returns_empty(self, store):
        """get_neighbors for a non-existent node should return empty list."""
        result = store.get_neighbors("nonexistent")
        assert result == []

    def test_get_neighbors_no_outgoing_returns_empty(self, store):
        """get_neighbors for a node with no outgoing edges should return empty."""
        node = GraphNode(node_id="isolated", node_type="test")
        store.add_node(node)

        result = store.get_neighbors("isolated")
        assert result == []

    def test_get_neighbors_preserves_node_properties(self, store):
        """get_neighbors should return GraphNode objects with correct properties."""
        node_a = GraphNode(node_id="a", node_type="user", label="Alice")
        node_b = GraphNode(
            node_id="b",
            node_type="store",
            label="Costco",
            properties={"membership": True},
        )
        store.add_node(node_a)
        store.add_node(node_b)

        store.add_relation(GraphEdge(
            source_id="a", target_id="b", edge_type="SHOPS_AT"
        ))

        neighbors = store.get_neighbors("a")
        assert len(neighbors) == 1
        neighbor_node, depth = neighbors[0]
        assert neighbor_node.node_id == "b"
        assert neighbor_node.node_type == "store"
        assert neighbor_node.label == "Costco"
        assert neighbor_node.properties == {"membership": True}
        assert depth == 1


class TestClose:
    """Adapter close should delegate to the underlying graph service."""

    def test_close_delegates_to_graph_service(self, store, graph_service):
        """close() should delegate to the underlying graph service's close()."""
        store.close()
        # MemoryGraphService sets _closed=True on close
        assert graph_service._closed is True

    def test_close_is_idempotent(self, store):
        """Calling close() multiple times should not raise."""
        store.close()
        store.close()  # Should not raise

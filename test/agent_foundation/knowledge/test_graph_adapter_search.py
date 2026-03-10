"""
Unit tests for GraphServiceEntityGraphStore search delegation.

Tests that the adapter correctly delegates supports_semantic_search and
search_nodes() to the underlying graph service, with proper inactive
node filtering.
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

_spu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode
from rich_python_utils.service_utils.graph_service.file_graph_service import (
    FileGraphService,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)


class TestSupportsSemanticSearch:
    """Test that adapter correctly reports supports_semantic_search."""

    def test_true_with_file_graph_service(self, tmp_path):
        """FileGraphService supports search, so adapter should too."""
        svc = FileGraphService(base_dir=str(tmp_path))
        store = GraphServiceEntityGraphStore(graph_service=svc)
        assert store.supports_semantic_search is True

    def test_false_with_memory_graph_service(self):
        """MemoryGraphService does not support search."""
        svc = MemoryGraphService()
        store = GraphServiceEntityGraphStore(graph_service=svc)
        assert store.supports_semantic_search is False


class TestSearchNodesDelegation:
    """Test that adapter correctly delegates search_nodes()."""

    def test_search_delegates_to_file_graph_service(self, tmp_path):
        svc = FileGraphService(base_dir=str(tmp_path))
        store = GraphServiceEntityGraphStore(graph_service=svc)
        store.add_node(GraphNode(
            node_id="n1", node_type="person", label="Alice",
        ))
        results = store.search_nodes("alice")
        assert len(results) == 1
        assert results[0][0].node_id == "n1"
        assert results[0][1] > 0.0

    def test_search_filters_inactive_nodes(self, tmp_path):
        """Inactive (soft-deleted) nodes should be excluded from results."""
        svc = FileGraphService(base_dir=str(tmp_path))
        store = GraphServiceEntityGraphStore(graph_service=svc)
        # Add an active node
        store.add_node(GraphNode(
            node_id="n1", node_type="person", label="Alice",
        ))
        # Add an inactive node directly via the underlying service
        inactive = GraphNode(
            node_id="n2", node_type="person", label="Alice Inactive",
            is_active=False,
        )
        svc.add_node(inactive)
        # Search should only return the active node
        results = store.search_nodes("alice")
        assert len(results) == 1
        assert results[0][0].node_id == "n1"

    def test_search_returns_empty_for_no_match(self, tmp_path):
        svc = FileGraphService(base_dir=str(tmp_path))
        store = GraphServiceEntityGraphStore(graph_service=svc)
        store.add_node(GraphNode(
            node_id="n1", node_type="person", label="Alice",
        ))
        assert store.search_nodes("nonexistent") == []

    def test_search_passes_node_type_filter(self, tmp_path):
        svc = FileGraphService(base_dir=str(tmp_path))
        store = GraphServiceEntityGraphStore(graph_service=svc)
        store.add_node(GraphNode(
            node_id="n1", node_type="person", label="Alice",
        ))
        store.add_node(GraphNode(
            node_id="n2", node_type="place", label="Alice Springs",
        ))
        results = store.search_nodes("alice", node_type="person")
        assert len(results) == 1
        assert results[0][0].node_type == "person"

    def test_search_passes_top_k(self, tmp_path):
        svc = FileGraphService(base_dir=str(tmp_path))
        store = GraphServiceEntityGraphStore(graph_service=svc)
        for i in range(10):
            store.add_node(GraphNode(
                node_id=f"n{i}", node_type="person", label=f"Person {i}",
            ))
        results = store.search_nodes("person", top_k=3)
        assert len(results) == 3

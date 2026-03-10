"""
Unit tests for SemanticGraphStore.

Feature: graph-semantic-retrieval
Tests cover:
- Construction validation (search mode + retrieval service + semantic support)
- Search mode routing (empty query, sidecar, native, both)
- Reindex validation and behavior
- close() lifecycle delegation

**Validates: Requirements 3.1, 3.2, 3.3, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 8.1, 8.2, 8.3**
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple
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
from rich_python_utils.service_utils.retrieval_service.retrieval_service_base import (
    RetrievalServiceBase,
)

from agent_foundation.knowledge.retrieval.stores.graph.semantic_graph_store import (
    SemanticGraphStore,
)
from agent_foundation.knowledge.retrieval.stores.graph.search_mode import SearchMode
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore

from conftest import InMemoryEntityGraphStore, InMemoryRetrievalService


# ── Construction validation ──────────────────────────────────────────────────


class TestSemanticGraphStoreConstruction:
    """Validate constructor raises on invalid search mode / service combos."""

    def test_sidecar_mode_requires_retrieval_service(self):
        """SIDECAR mode with retrieval_service=None raises ValueError."""
        with pytest.raises(ValueError, match="requires a retrieval_service"):
            SemanticGraphStore(
                graph_store=InMemoryEntityGraphStore(),
                retrieval_service=None,
                search_mode=SearchMode.SIDECAR,
            )

    def test_both_mode_requires_retrieval_service(self):
        """BOTH mode with retrieval_service=None raises ValueError."""
        with pytest.raises(ValueError, match="requires a retrieval_service"):
            SemanticGraphStore(
                graph_store=InMemoryEntityGraphStore(),
                retrieval_service=None,
                search_mode=SearchMode.BOTH,
            )

    def test_native_mode_requires_semantic_search_support(self):
        """NATIVE mode with non-semantic store raises ValueError."""
        graph = InMemoryEntityGraphStore()
        assert not graph.supports_semantic_search
        with pytest.raises(ValueError, match="supports semantic search"):
            SemanticGraphStore(
                graph_store=graph,
                retrieval_service=None,
                search_mode=SearchMode.NATIVE,
            )

    def test_both_mode_requires_semantic_search_support(self):
        """BOTH mode with non-semantic store raises ValueError."""
        graph = InMemoryEntityGraphStore()
        with pytest.raises(ValueError, match="supports semantic search"):
            SemanticGraphStore(
                graph_store=graph,
                retrieval_service=InMemoryRetrievalService(),
                search_mode=SearchMode.BOTH,
            )

    def test_sidecar_mode_with_retrieval_service_succeeds(self):
        """SIDECAR mode with valid retrieval_service succeeds."""
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        assert store.search_mode == SearchMode.SIDECAR

    def test_native_mode_with_none_retrieval_service_succeeds(self):
        """NATIVE mode with retrieval_service=None is valid when store supports semantic search."""
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = True
        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=None,
            search_mode=SearchMode.NATIVE,
        )
        assert store.search_mode == SearchMode.NATIVE

    def test_supports_semantic_search_always_true(self):
        """SemanticGraphStore.supports_semantic_search is always True."""
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        assert store.supports_semantic_search is True


# ── Search mode routing ──────────────────────────────────────────────────────


class TestSearchModeRouting:
    """Validate search_nodes routes to the correct backend(s)."""

    def test_empty_query_returns_empty_list(self):
        """search_nodes with empty string returns []."""
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        assert store.search_nodes("") == []

    def test_whitespace_query_returns_empty_list(self):
        """search_nodes with whitespace-only string returns []."""
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        assert store.search_nodes("   ") == []

    def test_sidecar_mode_searches_retrieval_service(self):
        """SIDECAR mode calls retrieval_service.search() and returns graph nodes."""
        graph = InMemoryEntityGraphStore()
        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=graph,
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        # Add a node so it gets indexed in the sidecar
        node = GraphNode(node_id="n1", node_type="service", label="Test Service")
        store.add_node(node)

        results = store.search_nodes("test", top_k=5)
        assert len(results) >= 1
        node_ids = [n.node_id for n, _ in results]
        assert "n1" in node_ids

    def test_native_mode_delegates_to_wrapped_store(self):
        """NATIVE mode calls graph_store.search_nodes()."""
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = True
        expected_node = GraphNode(node_id="n1", node_type="service", label="Test")
        mock_graph.search_nodes.return_value = [(expected_node, 0.9)]

        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=None,
            search_mode=SearchMode.NATIVE,
        )
        results = store.search_nodes("test query", top_k=3)

        mock_graph.search_nodes.assert_called_once_with("test query", top_k=3, node_type=None)
        assert len(results) == 1
        assert results[0][0].node_id == "n1"

    def test_both_mode_calls_both_and_merges(self):
        """BOTH mode calls both backends and merges via _rrf_merge."""
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = True
        native_node = GraphNode(node_id="native1", node_type="service", label="Native")
        mock_graph.search_nodes.return_value = [(native_node, 0.9)]
        mock_graph.get_node.return_value = None  # force _doc_to_node fallback

        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=retrieval,
            search_mode=SearchMode.BOTH,
        )

        # Manually add a doc to the retrieval service to simulate sidecar content
        from rich_python_utils.service_utils.retrieval_service.document import Document
        doc = Document(
            doc_id="sidecar1",
            content="Sidecar Node",
            metadata={"node_type": "product", "label": "Sidecar", "is_active": True, "properties": {}},
        )
        retrieval.add(doc, namespace="_graph_nodes")

        results = store.search_nodes("sidecar", top_k=10)

        # Both backends should have been called
        mock_graph.search_nodes.assert_called_once()
        # Results should contain nodes from both sources
        result_ids = {n.node_id for n, _ in results}
        assert "native1" in result_ids
        assert "sidecar1" in result_ids


# ── Reindex validation ───────────────────────────────────────────────────────


class TestReindexValidation:
    """Validate reindex() behavior and error conditions."""

    def test_reindex_raises_when_no_retrieval_service(self):
        """reindex() raises ValueError when retrieval_service is None."""
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = True
        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=None,
            search_mode=SearchMode.NATIVE,
        )
        with pytest.raises(ValueError, match="Cannot reindex"):
            store.reindex()

    def test_reindex_clears_and_rebuilds(self):
        """reindex() clears the namespace and re-indexes all active nodes."""
        graph = InMemoryEntityGraphStore()
        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=graph,
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )

        # Add nodes directly to graph (bypassing sidecar sync)
        active = GraphNode(node_id="a1", node_type="service", label="Active", is_active=True)
        inactive = GraphNode(node_id="i1", node_type="service", label="Inactive", is_active=False)
        graph.add_node(active)
        graph.add_node(inactive)

        count = store.reindex()

        # Only active nodes should be indexed
        assert count == 1
        assert retrieval.size(namespace="_graph_nodes") == 1
        assert retrieval.get_by_id("a1", namespace="_graph_nodes") is not None
        assert retrieval.get_by_id("i1", namespace="_graph_nodes") is None


# ── Close lifecycle ──────────────────────────────────────────────────────────


class TestCloseLifecycle:
    """Validate close() delegates properly."""

    def test_close_delegates_to_both(self):
        """close() calls close on both wrapped store and retrieval service."""
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = False
        mock_retrieval = MagicMock(spec=RetrievalServiceBase)

        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=mock_retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        store.close()

        mock_graph.close.assert_called_once()
        mock_retrieval.close.assert_called_once()

    def test_close_with_none_retrieval_service(self):
        """close() works when retrieval_service is None."""
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = True

        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=None,
            search_mode=SearchMode.NATIVE,
        )
        store.close()

        mock_graph.close.assert_called_once()

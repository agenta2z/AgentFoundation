"""
Property-based tests for SemanticGraphStore.

Feature: graph-semantic-retrieval
- Property 1: GraphNode ↔ Document Round-Trip
- Property 2: Default Node Text Builder Includes All Fields
- Property 3: Embedding Text Override
- Property 4: Add/Update Node Syncs Sidecar Index
- Property 5: Remove Node Syncs Sidecar Index
- Property 6: Edge Operations Do Not Modify Sidecar Index
- Property 7: Retrieval Service Failure Does Not Block Graph Store
- Property 8: Node Type Filter Returns Only Matching Types
- Property 9: RRF Fusion Produces Correct Scores and Ordering
- Property 10: Reindex Rebuilds Index With Only Active Nodes
- Property 11: Non-Search CRUD Behaves Identically to Wrapped Store
- Property 15: kwargs Passthrough Preserves Extended Parameters
- Property 16: Search Node Full Fidelity

**Validates: Requirements 3.3, 2.2, 2.3, 4.1, 4.2, 4.3, 4.4, 4.5, 1.4, 6.1, 6.2, 6.3, 8.1, 8.2, 8.3, 9.2, 5.2, 5.3**
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

# Add test directory to path for conftest imports
_test_dir = str(Path(__file__).resolve().parent)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

from hypothesis import given, settings, strategies as st, assume

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode, GraphEdge
from rich_python_utils.service_utils.retrieval_service.document import Document
from rich_python_utils.service_utils.retrieval_service.retrieval_service_base import (
    RetrievalServiceBase,
)

from agent_foundation.knowledge.retrieval.stores.graph.semantic_graph_store import (
    SemanticGraphStore,
)
from agent_foundation.knowledge.retrieval.stores.graph.search_mode import SearchMode
from agent_foundation.knowledge.retrieval.stores.graph.node_text_builder import (
    default_node_text_builder,
)
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore


# ── Hypothesis strategies ────────────────────────────────────────────────────

# JSON-serializable leaf values
_json_leaf = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(max_size=50),
)

_json_value = st.recursive(
    _json_leaf,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(min_size=1, max_size=15), children, max_size=3),
    ),
    max_leaves=8,
)

_identifier_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
)

_properties_strategy = st.dictionaries(
    st.text(min_size=1, max_size=20),
    _json_value,
    max_size=5,
)

_node_type_strategy = st.sampled_from(["service", "person", "product", "location", "concept"])


@st.composite
def graph_node_strategy(draw):
    """Generate a random GraphNode with JSON-serializable properties."""
    node_id = draw(_identifier_text)
    node_type = draw(_node_type_strategy)
    label = draw(st.text(max_size=50))
    properties = draw(_properties_strategy)
    is_active = draw(st.booleans())
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        properties=properties,
        is_active=is_active,
    )


@st.composite
def graph_node_list_strategy(draw, min_size=0, max_size=10):
    """Generate a list of GraphNodes with mixed types and active/inactive states."""
    return draw(st.lists(graph_node_strategy(), min_size=min_size, max_size=max_size))


@st.composite
def ranked_list_strategy(draw, min_size=0, max_size=5):
    """Generate a ranked list of (GraphNode, float) tuples for RRF testing."""
    nodes = draw(st.lists(graph_node_strategy(), min_size=min_size, max_size=max_size))
    scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=len(nodes),
            max_size=len(nodes),
        )
    )
    return list(zip(nodes, scores))


# ── Shared test helpers (from conftest.py) ───────────────────────────────────
from conftest import InMemoryEntityGraphStore, InMemoryRetrievalService


# ── Property 1: GraphNode ↔ Document Round-Trip ─────────────────────────────
# Feature: graph-semantic-retrieval, Property 1: GraphNode ↔ Document Round-Trip


class TestGraphNodeDocumentRoundTrip:
    """Property 1: GraphNode ↔ Document Round-Trip.

    For any valid GraphNode with JSON-serializable properties, converting to
    Document via _node_to_doc and back via _doc_to_node produces equivalent
    node_id, node_type, label, and properties. History is excluded.

    **Validates: Requirements 3.3**
    """

    @given(node=graph_node_strategy())
    @settings(max_examples=100)
    def test_round_trip_preserves_core_fields(self, node: GraphNode):
        """_node_to_doc → _doc_to_node preserves node_id, node_type, label, properties.

        **Validates: Requirements 3.3**
        """
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        doc = store._node_to_doc(node)
        restored = store._doc_to_node(doc)

        assert restored.node_id == node.node_id
        assert restored.node_type == node.node_type
        assert restored.label == node.label
        assert restored.properties == node.properties
        # History is explicitly excluded from round-trip
        assert restored.history == []


# ── Property 2: Default Node Text Builder Includes All Fields ────────────────
# Feature: graph-semantic-retrieval, Property 2: Default Node Text Builder Includes All Fields


class TestDefaultNodeTextBuilderIncludesAllFields:
    """Property 2: Default Node Text Builder Includes All Fields.

    For any GraphNode with non-empty node_type, label, and properties, the
    output of default_node_text_builder contains the node_type, label, and
    string representations of all property values (excluding embedding_text key).

    **Validates: Requirements 2.2**
    """

    @given(node=graph_node_strategy())
    @settings(max_examples=100)
    def test_builder_includes_node_type_label_and_properties(self, node: GraphNode):
        """default_node_text_builder output contains node_type, label, and property values.

        **Validates: Requirements 2.2**
        """
        assume(node.node_type.strip())
        assume(node.label.strip())
        assume(len(node.properties) > 0)

        text = default_node_text_builder(node)

        assert node.node_type in text
        assert node.label in text

        for key in sorted(node.properties.keys()):
            if key == "embedding_text":
                continue
            val = node.properties[key]
            if isinstance(val, str):
                if val:  # non-empty strings are included
                    assert val in text
            else:
                expected = f"{key}: {val}"
                assert expected in text


# ── Property 3: Embedding Text Override ──────────────────────────────────────
# Feature: graph-semantic-retrieval, Property 3: Embedding Text Override


class TestEmbeddingTextOverride:
    """Property 3: Embedding Text Override.

    For any GraphNode whose properties dict contains an embedding_text key,
    the Document produced by _node_to_doc has its embedding_text field set
    to that value.

    **Validates: Requirements 2.3**
    """

    @given(node=graph_node_strategy(), embed_text=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_embedding_text_property_overrides_doc_embedding_text(self, node: GraphNode, embed_text: str):
        """When properties has embedding_text key, Document.embedding_text uses that value.

        **Validates: Requirements 2.3**
        """
        node.properties["embedding_text"] = embed_text

        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        doc = store._node_to_doc(node)

        assert doc.embedding_text == embed_text


# ── Property 4: Add/Update Node Syncs Sidecar Index ─────────────────────────
# Feature: graph-semantic-retrieval, Property 4: Add/Update Node Syncs Sidecar Index


class TestAddUpdateNodeSyncsSidecarIndex:
    """Property 4: Add/Update Node Syncs Sidecar Index.

    For any GraphNode added or updated via SemanticGraphStore, the sidecar
    RetrievalServiceBase contains a Document with doc_id equal to the node's
    node_id, and metadata reflects the node's current state. Include upsert
    via add-then-update fallback.

    **Validates: Requirements 4.1, 4.3**
    """

    @given(node=graph_node_strategy())
    @settings(max_examples=100)
    def test_add_node_indexes_in_sidecar(self, node: GraphNode):
        """Adding a node creates a corresponding Document in the sidecar.

        **Validates: Requirements 4.1, 4.3**
        """
        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        store.add_node(node)

        doc = retrieval.get_by_id(node.node_id, namespace=store.index_namespace)
        assert doc is not None
        assert doc.doc_id == node.node_id
        assert doc.metadata["node_type"] == node.node_type
        assert doc.metadata["label"] == node.label
        assert doc.metadata["properties"] == node.properties

    @given(node=graph_node_strategy(), new_label=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_upsert_node_updates_sidecar(self, node: GraphNode, new_label: str):
        """Re-adding a node with same node_id updates the sidecar Document (upsert).

        **Validates: Requirements 4.1, 4.3**
        """
        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        store.add_node(node)

        # Update the node with a new label
        updated_node = GraphNode(
            node_id=node.node_id,
            node_type=node.node_type,
            label=new_label,
            properties=node.properties,
            is_active=node.is_active,
        )
        store.add_node(updated_node)

        doc = retrieval.get_by_id(node.node_id, namespace=store.index_namespace)
        assert doc is not None
        assert doc.metadata["label"] == new_label


# ── Property 5: Remove Node Syncs Sidecar Index ─────────────────────────────
# Feature: graph-semantic-retrieval, Property 5: Remove Node Syncs Sidecar Index


class TestRemoveNodeSyncsSidecarIndex:
    """Property 5: Remove Node Syncs Sidecar Index.

    For any GraphNode added then removed, the sidecar no longer contains a
    Document with that node_id.

    **Validates: Requirements 4.2**
    """

    @given(node=graph_node_strategy())
    @settings(max_examples=100)
    def test_remove_node_removes_from_sidecar(self, node: GraphNode):
        """Removing a node removes the corresponding Document from the sidecar.

        **Validates: Requirements 4.2**
        """
        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        store.add_node(node)
        assert retrieval.get_by_id(node.node_id, namespace=store.index_namespace) is not None

        store.remove_node(node.node_id)
        assert retrieval.get_by_id(node.node_id, namespace=store.index_namespace) is None



# ── Property 6: Edge Operations Do Not Modify Sidecar Index ─────────────────
# Feature: graph-semantic-retrieval, Property 6: Edge Operations Do Not Modify Sidecar Index


class TestEdgeOperationsDoNotModifySidecarIndex:
    """Property 6: Edge Operations Do Not Modify Sidecar Index.

    For any sequence of edge operations, the number of documents in the sidecar
    index remains unchanged.

    **Validates: Requirements 4.4**
    """

    @given(
        node_a=graph_node_strategy(),
        node_b=graph_node_strategy(),
        edge_type=_identifier_text,
    )
    @settings(max_examples=100)
    def test_edge_ops_do_not_change_sidecar_doc_count(
        self, node_a: GraphNode, node_b: GraphNode, edge_type: str
    ):
        """Adding/removing edges does not change the sidecar document count.

        **Validates: Requirements 4.4**
        """
        assume(node_a.node_id != node_b.node_id)

        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        store.add_node(node_a)
        store.add_node(node_b)

        count_before = retrieval.size(namespace=store.index_namespace)

        edge = GraphEdge(
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            edge_type=edge_type,
        )
        store.add_relation(edge)
        assert retrieval.size(namespace=store.index_namespace) == count_before

        store.remove_relation(node_a.node_id, node_b.node_id, edge_type)
        assert retrieval.size(namespace=store.index_namespace) == count_before


# ── Property 7: Retrieval Service Failure Does Not Block Graph Store ─────────
# Feature: graph-semantic-retrieval, Property 7: Retrieval Service Failure Does Not Block Graph Store


class TestRetrievalServiceFailureDoesNotBlockGraphStore:
    """Property 7: Retrieval Service Failure Does Not Block Graph Store.

    For any node mutation where the sidecar raises an exception, the underlying
    EntityGraphStore still contains the expected node state.

    **Validates: Requirements 4.5**
    """

    @given(node=graph_node_strategy())
    @settings(max_examples=100)
    def test_add_node_succeeds_despite_sidecar_failure(self, node: GraphNode):
        """Graph store add_node succeeds even when sidecar raises.

        **Validates: Requirements 4.5**
        """
        graph = InMemoryEntityGraphStore()
        failing_retrieval = MagicMock(spec=RetrievalServiceBase)
        failing_retrieval.add.side_effect = RuntimeError("sidecar failure")

        store = SemanticGraphStore(
            graph_store=graph,
            retrieval_service=failing_retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        store.add_node(node)

        assert graph.get_node(node.node_id) is not None
        assert graph.get_node(node.node_id).node_id == node.node_id

    @given(node=graph_node_strategy())
    @settings(max_examples=100)
    def test_remove_node_succeeds_despite_sidecar_failure(self, node: GraphNode):
        """Graph store remove_node succeeds even when sidecar raises.

        **Validates: Requirements 4.5**
        """
        graph = InMemoryEntityGraphStore()
        graph.add_node(node)

        failing_retrieval = MagicMock(spec=RetrievalServiceBase)
        failing_retrieval.remove.side_effect = RuntimeError("sidecar failure")

        store = SemanticGraphStore(
            graph_store=graph,
            retrieval_service=failing_retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        store.remove_node(node.node_id)

        assert graph.get_node(node.node_id) is None


# ── Property 8: Node Type Filter Returns Only Matching Types ────────────────
# Feature: graph-semantic-retrieval, Property 8: Node Type Filter Returns Only Matching Types


class TestNodeTypeFilterReturnsOnlyMatchingTypes:
    """Property 8: Node Type Filter Returns Only Matching Types.

    For any set of indexed GraphNodes with mixed node_type values and any
    node_type filter, search_nodes returns only nodes whose node_type matches
    the filter.

    **Validates: Requirements 1.4**
    """

    @given(
        nodes=st.lists(graph_node_strategy(), min_size=2, max_size=8),
        filter_type=_node_type_strategy,
    )
    @settings(max_examples=100)
    def test_search_with_node_type_filter_returns_only_matching(
        self, nodes: List[GraphNode], filter_type: str
    ):
        """search_nodes with node_type filter returns only nodes of that type.

        **Validates: Requirements 1.4**
        """
        # Deduplicate by node_id
        seen_ids = set()
        unique_nodes = []
        for n in nodes:
            if n.node_id not in seen_ids:
                seen_ids.add(n.node_id)
                unique_nodes.append(n)
        assume(len(unique_nodes) >= 2)

        retrieval = InMemoryRetrievalService()
        graph = InMemoryEntityGraphStore()
        store = SemanticGraphStore(
            graph_store=graph,
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )

        for node in unique_nodes:
            store.add_node(node)

        results = store.search_nodes("test query", top_k=100, node_type=filter_type)

        for result_node, score in results:
            assert result_node.node_type == filter_type



# ── Property 9: RRF Fusion Produces Correct Scores and Ordering ─────────────
# Feature: graph-semantic-retrieval, Property 9: RRF Fusion Produces Correct Scores and Ordering


class TestRRFFusionProducesCorrectScoresAndOrdering:
    """Property 9: RRF Fusion Produces Correct Scores and Ordering.

    For any two ranked lists (including both empty), the RRF merge produces
    correct scores (sum of 1/(rrf_k + rank + 1)), descending order, and at
    most top_k items.

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    @given(
        list_a=ranked_list_strategy(min_size=0, max_size=5),
        list_b=ranked_list_strategy(min_size=0, max_size=5),
        rrf_k=st.integers(min_value=1, max_value=100),
        top_k=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_rrf_merge_scores_ordering_and_truncation(
        self, list_a, list_b, rrf_k: int, top_k: int
    ):
        """RRF merge produces correct scores, descending order, and at most top_k items.

        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        store = SemanticGraphStore(
            graph_store=InMemoryEntityGraphStore(),
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
            rrf_k=rrf_k,
        )

        result = store._rrf_merge(list_a, list_b, top_k)

        # Both empty → empty result
        if not list_a and not list_b:
            assert result == []
            return

        # At most top_k items
        assert len(result) <= top_k

        # Compute expected scores
        expected_scores: Dict[str, float] = {}
        for rank, (node, _) in enumerate(list_a):
            rrf_score = 1.0 / (rrf_k + rank + 1)
            expected_scores[node.node_id] = expected_scores.get(node.node_id, 0) + rrf_score
        for rank, (node, _) in enumerate(list_b):
            rrf_score = 1.0 / (rrf_k + rank + 1)
            expected_scores[node.node_id] = expected_scores.get(node.node_id, 0) + rrf_score

        # Verify scores match
        for node, score in result:
            assert abs(score - expected_scores[node.node_id]) < 1e-9

        # Verify descending order
        scores = [score for _, score in result]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ── Property 10: Reindex Rebuilds Index With Only Active Nodes ───────────────
# Feature: graph-semantic-retrieval, Property 10: Reindex Rebuilds Index With Only Active Nodes


class TestReindexRebuildsIndexWithOnlyActiveNodes:
    """Property 10: Reindex Rebuilds Index With Only Active Nodes.

    For any graph store with mixed active/inactive nodes, after reindex(),
    the sidecar contains exactly one Document per active node and zero for
    inactive.

    **Validates: Requirements 8.1, 8.2, 8.3**
    """

    @given(nodes=graph_node_list_strategy(min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_reindex_indexes_only_active_nodes(self, nodes: List[GraphNode]):
        """After reindex(), sidecar has exactly one doc per active node.

        **Validates: Requirements 8.1, 8.2, 8.3**
        """
        # Deduplicate by node_id
        seen_ids = set()
        unique_nodes = []
        for n in nodes:
            if n.node_id not in seen_ids:
                seen_ids.add(n.node_id)
                unique_nodes.append(n)
        assume(len(unique_nodes) >= 1)

        graph = InMemoryEntityGraphStore()
        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=graph,
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )

        # Add all nodes directly to the graph store (bypassing sidecar sync)
        for node in unique_nodes:
            graph.add_node(node)

        # Reindex
        count = store.reindex()

        active_nodes = [n for n in unique_nodes if n.is_active]
        inactive_nodes = [n for n in unique_nodes if not n.is_active]

        assert count == len(active_nodes)
        assert retrieval.size(namespace=store.index_namespace) == len(active_nodes)

        # Each active node has a doc
        for node in active_nodes:
            doc = retrieval.get_by_id(node.node_id, namespace=store.index_namespace)
            assert doc is not None
            assert doc.doc_id == node.node_id

        # No inactive node has a doc
        for node in inactive_nodes:
            doc = retrieval.get_by_id(node.node_id, namespace=store.index_namespace)
            assert doc is None


# ── Property 11: Non-Search CRUD Behaves Identically to Wrapped Store ────────
# Feature: graph-semantic-retrieval, Property 11: Non-Search CRUD Behaves Identically to Wrapped Store


class TestNonSearchCRUDBehavesIdenticallyToWrappedStore:
    """Property 11: Non-Search CRUD Behaves Identically to Wrapped Store.

    For any sequence of non-search operations with arbitrary **kwargs, return
    values are identical to the wrapped store.

    **Validates: Requirements 9.2**
    """

    @given(
        node_a=graph_node_strategy(),
        node_b=graph_node_strategy(),
        edge_type=_identifier_text,
    )
    @settings(max_examples=100)
    def test_crud_returns_match_wrapped_store(
        self, node_a: GraphNode, node_b: GraphNode, edge_type: str
    ):
        """Non-search CRUD operations return identical results to the wrapped store.

        **Validates: Requirements 9.2**
        """
        assume(node_a.node_id != node_b.node_id)

        # Direct store
        direct = InMemoryEntityGraphStore()
        direct.add_node(node_a)
        direct.add_node(node_b)

        # Wrapped store
        wrapped_graph = InMemoryEntityGraphStore()
        store = SemanticGraphStore(
            graph_store=wrapped_graph,
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        store.add_node(node_a)
        store.add_node(node_b)

        # get_node
        direct_result = direct.get_node(node_a.node_id)
        wrapped_result = store.get_node(node_a.node_id)
        assert direct_result.node_id == wrapped_result.node_id
        assert direct_result.node_type == wrapped_result.node_type
        assert direct_result.label == wrapped_result.label
        assert direct_result.properties == wrapped_result.properties

        # add_relation + get_relations
        edge = GraphEdge(
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            edge_type=edge_type,
        )
        direct.add_relation(edge)
        store.add_relation(edge)

        direct_rels = direct.get_relations(node_a.node_id)
        wrapped_rels = store.get_relations(node_a.node_id)
        assert len(direct_rels) == len(wrapped_rels)

        # get_neighbors
        direct_neighbors = direct.get_neighbors(node_a.node_id)
        wrapped_neighbors = store.get_neighbors(node_a.node_id)
        assert len(direct_neighbors) == len(wrapped_neighbors)

        # remove_relation
        direct_removed = direct.remove_relation(node_a.node_id, node_b.node_id, edge_type)
        wrapped_removed = store.remove_relation(node_a.node_id, node_b.node_id, edge_type)
        assert direct_removed == wrapped_removed

        # remove_node
        direct_removed = direct.remove_node(node_a.node_id)
        wrapped_removed = store.remove_node(node_a.node_id)
        assert direct_removed == wrapped_removed

        # get_node after removal
        assert direct.get_node(node_a.node_id) is None
        assert store.get_node(node_a.node_id) is None



# ── Property 15: kwargs Passthrough Preserves Extended Parameters ────────────
# Feature: graph-semantic-retrieval, Property 15: kwargs Passthrough Preserves Extended Parameters


class TestKwargsPassthroughPreservesExtendedParameters:
    """Property 15: kwargs Passthrough Preserves Extended Parameters.

    For any call with kwargs like operation_id="x", the wrapped store receives
    the same kwargs.

    **Validates: Requirements 9.2**
    """

    @given(
        node=graph_node_strategy(),
        operation_id=_identifier_text,
    )
    @settings(max_examples=100)
    def test_add_node_passes_kwargs_to_wrapped_store(self, node: GraphNode, operation_id: str):
        """add_node passes **kwargs through to the wrapped store.

        **Validates: Requirements 9.2**
        """
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = False

        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        store.add_node(node, operation_id=operation_id)

        mock_graph.add_node.assert_called_once_with(node, operation_id=operation_id)

    @given(
        node_id=_identifier_text,
        include_inactive=st.booleans(),
    )
    @settings(max_examples=100)
    def test_get_node_passes_kwargs_to_wrapped_store(self, node_id: str, include_inactive: bool):
        """get_node passes **kwargs through to the wrapped store.

        **Validates: Requirements 9.2**
        """
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = False
        mock_graph.get_node.return_value = None

        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        store.get_node(node_id, include_inactive=include_inactive)

        mock_graph.get_node.assert_called_once_with(node_id, include_inactive=include_inactive)

    @given(
        node_id=_identifier_text,
        operation_id=_identifier_text,
    )
    @settings(max_examples=100)
    def test_remove_node_passes_kwargs_to_wrapped_store(self, node_id: str, operation_id: str):
        """remove_node passes **kwargs through to the wrapped store.

        **Validates: Requirements 9.2**
        """
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = False
        mock_graph.remove_node.return_value = True

        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        store.remove_node(node_id, operation_id=operation_id)

        mock_graph.remove_node.assert_called_once_with(node_id, operation_id=operation_id)

    @given(edge_type=_identifier_text, operation_id=_identifier_text)
    @settings(max_examples=100)
    def test_add_relation_passes_kwargs_to_wrapped_store(self, edge_type: str, operation_id: str):
        """add_relation passes **kwargs through to the wrapped store.

        **Validates: Requirements 9.2**
        """
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = False

        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        edge = GraphEdge(source_id="a", target_id="b", edge_type=edge_type)
        store.add_relation(edge, operation_id=operation_id)

        mock_graph.add_relation.assert_called_once_with(edge, operation_id=operation_id)

    @given(node_id=_identifier_text, operation_id=_identifier_text)
    @settings(max_examples=100)
    def test_get_neighbors_passes_kwargs_to_wrapped_store(self, node_id: str, operation_id: str):
        """get_neighbors passes **kwargs through to the wrapped store.

        **Validates: Requirements 9.2**
        """
        mock_graph = MagicMock(spec=EntityGraphStore)
        mock_graph.supports_semantic_search = False
        mock_graph.get_neighbors.return_value = []

        store = SemanticGraphStore(
            graph_store=mock_graph,
            retrieval_service=InMemoryRetrievalService(),
            search_mode=SearchMode.SIDECAR,
        )
        store.get_neighbors(node_id, operation_id=operation_id)

        mock_graph.get_neighbors.assert_called_once_with(
            node_id, relation_type=None, depth=1, operation_id=operation_id
        )


# ── Property 16: Search Node Full Fidelity ──────────────────────────────────
# Feature: graph-semantic-retrieval, Property 16: Search Node Full Fidelity


class TestSearchNodeFullFidelity:
    """Property 16: Search Node Full Fidelity.

    For any node in both sidecar and graph store, search_nodes in sidecar mode
    returns a GraphNode identical to graph_store.get_node(doc.doc_id).

    **Validates: Requirements 5.2, 5.3**
    """

    @given(node=graph_node_strategy())
    @settings(max_examples=100)
    def test_search_returns_full_fidelity_node_from_graph_store(self, node: GraphNode):
        """search_nodes returns the full node from graph store, not the lossy _doc_to_node.

        **Validates: Requirements 5.2, 5.3**
        """
        # Give the node some history to verify full fidelity
        node.is_active = True

        graph = InMemoryEntityGraphStore()
        retrieval = InMemoryRetrievalService()
        store = SemanticGraphStore(
            graph_store=graph,
            retrieval_service=retrieval,
            search_mode=SearchMode.SIDECAR,
        )
        store.add_node(node)

        # Mock the retrieval service search to return the indexed doc
        doc = retrieval.get_by_id(node.node_id, namespace=store.index_namespace)
        assert doc is not None

        # Patch search to return this doc
        original_search = retrieval.search
        def mock_search(query, filters=None, namespace=None, top_k=5):
            return [(doc, 0.9)]
        retrieval.search = mock_search

        results = store.search_nodes("test query", top_k=5)
        assert len(results) == 1

        result_node, score = results[0]
        graph_node = graph.get_node(node.node_id)

        # Full fidelity: the returned node should be the same object from graph store
        assert result_node.node_id == graph_node.node_id
        assert result_node.node_type == graph_node.node_type
        assert result_node.label == graph_node.label
        assert result_node.properties == graph_node.properties
        assert result_node.is_active == graph_node.is_active
        assert result_node.history == graph_node.history

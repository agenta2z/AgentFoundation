"""
Property-based tests for KnowledgeBase space-aware retrieval.

Feature: knowledge-space-restructuring
- Property 5: No Space Filter Returns All Spaces
- Property 7: Metadata Space Filtering
- Property 8: Graph Traversal Space Filtering
- Property 14: KnowledgeBase Callable Spaces Passthrough

**Validates: Requirements 3.2, 3.5, 4.3, 4.4, 5.3, 5.4, 11.2**

These tests use simple in-memory stub stores that implement the ABC interfaces
minimally. The piece store stub has supports_space_filter=False so the KB
handles filtering via over-fetch + post-filter.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from hypothesis import given, settings, strategies as st, assume

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.formatter import RetrievalResult
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from agent_foundation.knowledge.retrieval.stores.metadata.base import MetadataStore
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)


# ── Stub Store Implementations ───────────────────────────────────────────────


class StubPieceStore(KnowledgePieceStore):
    """In-memory piece store stub. supports_space_filter=False so KB post-filters."""

    def __init__(self):
        self._pieces: Dict[str, KnowledgePiece] = {}

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
        spaces: Optional[List[str]] = None,
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Return all matching pieces with a fixed score of 0.5.

        Does NOT apply space filtering — KB handles that via post-filter.
        """
        results = []
        for piece in self._pieces.values():
            if entity_id is not None and piece.entity_id != entity_id:
                continue
            if entity_id is None and piece.entity_id is not None:
                continue
            if knowledge_type and piece.knowledge_type != knowledge_type:
                continue
            if tags and not all(t in piece.tags for t in tags):
                continue
            results.append((piece, 0.5))
        return results[:top_k]

    def list_all(
        self,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
        spaces: Optional[List[str]] = None,
    ) -> List[KnowledgePiece]:
        results = []
        for piece in self._pieces.values():
            if entity_id is not None and piece.entity_id != entity_id:
                continue
            if entity_id is None and piece.entity_id is not None:
                continue
            if knowledge_type and piece.knowledge_type != knowledge_type:
                continue
            results.append(piece)
        return results


class StubMetadataStore(MetadataStore):
    """In-memory metadata store stub."""

    def __init__(self):
        self._metadata: Dict[str, EntityMetadata] = {}

    def get_metadata(self, entity_id: str) -> Optional[EntityMetadata]:
        return self._metadata.get(entity_id)

    def save_metadata(self, metadata: EntityMetadata) -> None:
        self._metadata[metadata.entity_id] = metadata

    def delete_metadata(self, entity_id: str) -> bool:
        if entity_id in self._metadata:
            del self._metadata[entity_id]
            return True
        return False

    def list_entities(self, entity_type: str = None) -> List[str]:
        if entity_type:
            return [
                eid for eid, m in self._metadata.items()
                if m.entity_type == entity_type
            ]
        return list(self._metadata.keys())


class StubGraphStore(EntityGraphStore):
    """In-memory graph store stub."""

    def __init__(self):
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._edges = [
                e for e in self._edges
                if e.source_id != node_id and e.target_id != node_id
            ]
            return True
        return False

    def add_relation(self, relation: GraphEdge) -> None:
        self._edges.append(relation)

    def get_relations(
        self,
        node_id: str,
        relation_type: str = None,
        direction: str = "outgoing",
    ) -> List[GraphEdge]:
        results = []
        for e in self._edges:
            if direction in ("outgoing", "both") and e.source_id == node_id:
                if relation_type is None or e.edge_type == relation_type:
                    results.append(e)
            if direction in ("incoming", "both") and e.target_id == node_id:
                if relation_type is None or e.edge_type == relation_type:
                    results.append(e)
        return results

    def remove_relation(
        self, source_id: str, target_id: str, relation_type: str
    ) -> bool:
        for i, e in enumerate(self._edges):
            if (
                e.source_id == source_id
                and e.target_id == target_id
                and e.edge_type == relation_type
            ):
                self._edges.pop(i)
                return True
        return False

    def get_neighbors(
        self,
        node_id: str,
        relation_type: str = None,
        depth: int = 1,
    ) -> List[Tuple[GraphNode, int]]:
        """Simple BFS to depth 1 only (sufficient for tests)."""
        results = []
        if depth < 1:
            return results
        for e in self._edges:
            if e.source_id == node_id:
                if relation_type is None or e.edge_type == relation_type:
                    target = self._nodes.get(e.target_id)
                    if target:
                        results.append((target, 1))
        return results


# ── Hypothesis Strategies ────────────────────────────────────────────────────

_valid_space = st.sampled_from(["main", "personal", "developmental"])

_spaces_list = st.lists(_valid_space, min_size=1, max_size=3).map(
    lambda xs: list(dict.fromkeys(xs))  # deduplicate, preserve order
)

_space_filter = st.lists(_valid_space, min_size=1, max_size=3).map(
    lambda xs: list(dict.fromkeys(xs))
)


@st.composite
def piece_with_spaces(draw, entity_id=None):
    """Generate a KnowledgePiece with a specific spaces assignment."""
    spaces = draw(_spaces_list)
    piece_id = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        )
    )
    return KnowledgePiece(
        content=f"content for {piece_id}",
        piece_id=piece_id,
        knowledge_type=KnowledgeType.Fact,
        info_type="context",
        spaces=spaces,
        entity_id=entity_id,
    )


@st.composite
def metadata_with_spaces(draw, entity_id=None):
    """Generate an EntityMetadata with a specific spaces assignment."""
    spaces = draw(_spaces_list)
    eid = entity_id or draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        )
    )
    return EntityMetadata(
        entity_id=eid,
        entity_type="user",
        properties={"key": "value"},
        spaces=spaces,
    )


@st.composite
def graph_node_with_spaces(draw, node_id=None):
    """Generate a GraphNode with spaces in its properties."""
    spaces = draw(_spaces_list)
    nid = node_id or draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        )
    )
    return GraphNode(
        node_id=nid,
        node_type="entity",
        label=nid,
        properties={"spaces": spaces},
    )


# ── Helper to build a fresh KB with stub stores ─────────────────────────────


def _make_kb():
    """Create a KnowledgeBase with fresh stub stores."""
    piece_store = StubPieceStore()
    metadata_store = StubMetadataStore()
    graph_store = StubGraphStore()
    kb = KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:test",
        default_top_k=100,  # high to avoid truncation in tests
    )
    return kb, piece_store, metadata_store, graph_store


# ── Property 5: No Space Filter Returns All Spaces ──────────────────────────


class TestNoSpaceFilterReturnsAllSpaces:
    """Property 5: No Space Filter Returns All Spaces.

    For any set of knowledge pieces across different spaces, calling
    retrieve() without a spaces parameter (or with spaces=None) SHALL
    return results from all spaces — equivalent to no filtering.

    **Validates: Requirements 3.2, 4.4, 5.4, 11.2**
    """

    @given(
        pieces_data=st.lists(
            st.tuples(_spaces_list, st.sampled_from(["p1", "p2", "p3", "p4", "p5"])),
            min_size=1,
            max_size=5,
        ).map(lambda xs: [(s, pid) for s, pid in dict([(pid, s) for s, pid in xs]).items()])
    )
    @settings(max_examples=100)
    def test_retrieve_without_spaces_returns_all_pieces(self, pieces_data):
        """retrieve() without spaces param returns pieces from all spaces.

        **Validates: Requirements 3.2, 11.2**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        # Add pieces with various spaces, all entity-scoped
        added_ids = set()
        for pid, spaces in pieces_data:
            piece = KnowledgePiece(
                content=f"content {pid}",
                piece_id=pid,
                knowledge_type=KnowledgeType.Fact,
                info_type="context",
                spaces=spaces,
                entity_id="user:test",
            )
            piece_store.add(piece)
            added_ids.add(pid)

        # Retrieve without spaces filter
        result = kb.retrieve("content", include_global=False)

        returned_ids = {p.piece_id for p, _ in result.pieces}
        assert returned_ids == added_ids, (
            f"Expected all pieces {added_ids}, got {returned_ids}"
        )

    @given(
        meta_spaces=_spaces_list,
        global_meta_spaces=_spaces_list,
    )
    @settings(max_examples=100)
    def test_retrieve_without_spaces_returns_all_metadata(
        self, meta_spaces, global_meta_spaces
    ):
        """retrieve() without spaces param returns metadata regardless of space.

        **Validates: Requirements 4.4, 11.2**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        user_meta = EntityMetadata(
            entity_id="user:test",
            entity_type="user",
            properties={"name": "Test"},
            spaces=meta_spaces,
        )
        global_meta = EntityMetadata(
            entity_id="global",
            entity_type="global",
            properties={"version": "1.0"},
            spaces=global_meta_spaces,
        )
        metadata_store.save_metadata(user_meta)
        metadata_store.save_metadata(global_meta)

        result = kb.retrieve("query")

        # Both metadata should be returned regardless of their spaces
        assert result.metadata is not None, "Entity metadata should be returned"
        assert result.global_metadata is not None, "Global metadata should be returned"

    @given(node_spaces=_spaces_list)
    @settings(max_examples=100)
    def test_retrieve_without_spaces_returns_all_graph_neighbors(self, node_spaces):
        """retrieve() without spaces param returns all graph neighbors.

        **Validates: Requirements 5.4, 11.2**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        # Add source node and a neighbor with specific spaces
        source = GraphNode(
            node_id="user:test", node_type="user", label="Test"
        )
        neighbor = GraphNode(
            node_id="neighbor:1",
            node_type="entity",
            label="Neighbor",
            properties={"spaces": node_spaces},
        )
        graph_store.add_node(source)
        graph_store.add_node(neighbor)
        graph_store.add_relation(GraphEdge(
            source_id="user:test",
            target_id="neighbor:1",
            edge_type="RELATED",
        ))

        result = kb.retrieve("query")

        # Graph neighbor should be returned regardless of its spaces
        assert len(result.graph_context) == 1, (
            f"Expected 1 graph neighbor, got {len(result.graph_context)}"
        )


# ── Property 7: Metadata Space Filtering ─────────────────────────────────────


class TestMetadataSpaceFiltering:
    """Property 7: Metadata Space Filtering.

    For any set of EntityMetadata objects with various space assignments and
    any space filter, the KnowledgeBase SHALL return only metadata whose
    spaces list intersects with the requested spaces.

    **Validates: Requirements 4.3**
    """

    @given(
        meta_spaces=_spaces_list,
        space_filter=_space_filter,
    )
    @settings(max_examples=100)
    def test_entity_metadata_filtered_by_spaces(self, meta_spaces, space_filter):
        """Entity metadata is included only when its spaces intersect the filter.

        **Validates: Requirements 4.3**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        user_meta = EntityMetadata(
            entity_id="user:test",
            entity_type="user",
            properties={"name": "Test"},
            spaces=meta_spaces,
        )
        metadata_store.save_metadata(user_meta)

        result = kb.retrieve("query", spaces=space_filter, include_global=False)

        has_intersection = bool(set(meta_spaces) & set(space_filter))
        if has_intersection:
            assert result.metadata is not None, (
                f"Metadata with spaces={meta_spaces} should be returned "
                f"for filter={space_filter}"
            )
        else:
            assert result.metadata is None, (
                f"Metadata with spaces={meta_spaces} should NOT be returned "
                f"for filter={space_filter}"
            )

    @given(
        global_meta_spaces=_spaces_list,
        space_filter=_space_filter,
    )
    @settings(max_examples=100)
    def test_global_metadata_filtered_by_spaces(self, global_meta_spaces, space_filter):
        """Global metadata is included only when its spaces intersect the filter.

        **Validates: Requirements 4.3**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        global_meta = EntityMetadata(
            entity_id="global",
            entity_type="global",
            properties={"version": "1.0"},
            spaces=global_meta_spaces,
        )
        metadata_store.save_metadata(global_meta)

        result = kb.retrieve("query", spaces=space_filter)

        has_intersection = bool(set(global_meta_spaces) & set(space_filter))
        if has_intersection:
            assert result.global_metadata is not None, (
                f"Global metadata with spaces={global_meta_spaces} should be "
                f"returned for filter={space_filter}"
            )
        else:
            assert result.global_metadata is None, (
                f"Global metadata with spaces={global_meta_spaces} should NOT "
                f"be returned for filter={space_filter}"
            )


# ── Property 8: Graph Traversal Space Filtering ─────────────────────────────


class TestGraphTraversalSpaceFiltering:
    """Property 8: Graph Traversal Space Filtering.

    For any entity graph with nodes carrying various spaces properties and
    any space filter, graph traversal SHALL return only neighbor nodes whose
    spaces property intersects with the requested spaces.

    **Validates: Requirements 5.3**
    """

    @given(
        neighbor_spaces_list=st.lists(_spaces_list, min_size=1, max_size=5),
        space_filter=_space_filter,
    )
    @settings(max_examples=100)
    def test_graph_neighbors_filtered_by_spaces(
        self, neighbor_spaces_list, space_filter
    ):
        """Only graph neighbors whose spaces intersect the filter are returned.

        **Validates: Requirements 5.3**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        # Add source node
        source = GraphNode(
            node_id="user:test", node_type="user", label="Test"
        )
        graph_store.add_node(source)

        # Add neighbors with various spaces
        expected_neighbor_ids = set()
        all_neighbor_ids = set()
        for i, spaces in enumerate(neighbor_spaces_list):
            nid = f"neighbor:{i}"
            all_neighbor_ids.add(nid)
            neighbor = GraphNode(
                node_id=nid,
                node_type="entity",
                label=f"Neighbor {i}",
                properties={"spaces": spaces},
            )
            graph_store.add_node(neighbor)
            graph_store.add_relation(GraphEdge(
                source_id="user:test",
                target_id=nid,
                edge_type="RELATED",
            ))
            if set(spaces) & set(space_filter):
                expected_neighbor_ids.add(nid)

        result = kb.retrieve("query", spaces=space_filter)

        returned_ids = {
            ctx["target_node_id"] for ctx in result.graph_context
        }
        assert returned_ids == expected_neighbor_ids, (
            f"Expected neighbors {expected_neighbor_ids}, got {returned_ids}. "
            f"Filter={space_filter}, neighbor_spaces={neighbor_spaces_list}"
        )

    @given(space_filter=_space_filter)
    @settings(max_examples=100)
    def test_nodes_without_spaces_property_treated_as_main(self, space_filter):
        """Nodes missing the spaces property are treated as ["main"].

        **Validates: Requirements 5.3**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        source = GraphNode(
            node_id="user:test", node_type="user", label="Test"
        )
        # Neighbor with no spaces property at all
        neighbor = GraphNode(
            node_id="neighbor:no_spaces",
            node_type="entity",
            label="No Spaces",
            properties={},  # no "spaces" key
        )
        graph_store.add_node(source)
        graph_store.add_node(neighbor)
        graph_store.add_relation(GraphEdge(
            source_id="user:test",
            target_id="neighbor:no_spaces",
            edge_type="RELATED",
        ))

        result = kb.retrieve("query", spaces=space_filter)

        # Node without spaces property defaults to ["main"]
        should_be_included = "main" in space_filter
        returned_ids = {
            ctx["target_node_id"] for ctx in result.graph_context
        }
        if should_be_included:
            assert "neighbor:no_spaces" in returned_ids
        else:
            assert "neighbor:no_spaces" not in returned_ids


# ── Property 14: KnowledgeBase Callable Spaces Passthrough ───────────────────


class TestKnowledgeBaseCallableSpacesPassthrough:
    """Property 14: KnowledgeBase Callable Spaces Passthrough.

    For any query string and any spaces filter, calling kb(query, spaces=filter)
    SHALL produce the same formatted result as
    kb.formatter.format(kb.retrieve(query, spaces=filter)).

    **Validates: Requirements 3.5**
    """

    @given(
        pieces_data=st.lists(
            st.tuples(
                st.sampled_from(["p1", "p2", "p3", "p4", "p5"]),
                _spaces_list,
            ),
            min_size=0,
            max_size=5,
        ).map(lambda xs: {pid: s for pid, s in xs}),
        space_filter=_space_filter,
    )
    @settings(max_examples=100)
    def test_callable_matches_explicit_retrieve_format(
        self, pieces_data, space_filter
    ):
        """kb(query, spaces=filter) == kb.formatter.format(kb.retrieve(query, spaces=filter)).

        **Validates: Requirements 3.5**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        # Add pieces
        for pid, spaces in pieces_data.items():
            piece = KnowledgePiece(
                content=f"content {pid}",
                piece_id=pid,
                knowledge_type=KnowledgeType.Fact,
                info_type="context",
                spaces=spaces,
                entity_id="user:test",
            )
            piece_store.add(piece)

        # Add metadata
        user_meta = EntityMetadata(
            entity_id="user:test",
            entity_type="user",
            properties={"name": "Test"},
            spaces=["main"],
        )
        metadata_store.save_metadata(user_meta)

        query = "content"

        # Call via __call__
        callable_result = kb(query, spaces=space_filter)

        # Call via explicit retrieve + format
        retrieve_result = kb.retrieve(query, spaces=space_filter)
        explicit_result = kb.formatter.format(retrieve_result)

        assert callable_result == explicit_result, (
            f"Callable result differs from explicit retrieve+format.\n"
            f"Callable: {callable_result!r}\n"
            f"Explicit: {explicit_result!r}"
        )

    @given(
        pieces_data=st.lists(
            st.tuples(
                st.sampled_from(["p1", "p2", "p3"]),
                _spaces_list,
            ),
            min_size=0,
            max_size=3,
        ).map(lambda xs: {pid: s for pid, s in xs}),
    )
    @settings(max_examples=100)
    def test_callable_without_spaces_matches_retrieve(self, pieces_data):
        """kb(query) == kb.formatter.format(kb.retrieve(query)) — no spaces arg.

        **Validates: Requirements 3.5**
        """
        kb, piece_store, metadata_store, graph_store = _make_kb()

        for pid, spaces in pieces_data.items():
            piece = KnowledgePiece(
                content=f"content {pid}",
                piece_id=pid,
                knowledge_type=KnowledgeType.Fact,
                info_type="context",
                spaces=spaces,
                entity_id="user:test",
            )
            piece_store.add(piece)

        query = "content"

        callable_result = kb(query)
        retrieve_result = kb.retrieve(query)
        explicit_result = kb.formatter.format(retrieve_result)

        assert callable_result == explicit_result

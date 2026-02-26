"""
Property-based tests for knowledge module adapters.

# Feature: knowledge-service-extraction
# Property 17: Metadata adapter round-trip
# Property 18: Metadata adapter list_entities filtering
# Property 19: Piece adapter round-trip
# Property 20: Piece adapter search filter delegation

Uses Hypothesis to verify universal correctness properties of the
KeyValueMetadataStore adapter backed by MemoryKeyValueService and
RetrievalKnowledgePieceStore adapter backed by MemoryRetrievalService.
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
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from science_modeling_tools.knowledge.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)


# ── Hypothesis strategies ────────────────────────────────────────────────────

# Entity type names: simple non-empty identifiers (letters/digits)
_entity_type_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)

# Entity name part: non-empty identifier text
_entity_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)

# ISO 8601 timestamp strategy
_timestamp_strategy = st.from_regex(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\+00:00",
    fullmatch=True,
)

# JSON-serializable leaf values for properties dicts
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
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
    ),
    max_leaves=10,
)

# Properties dict strategy
_properties_strategy = st.dictionaries(
    st.text(min_size=1, max_size=20),
    _json_value,
    max_size=5,
)


@st.composite
def entity_metadata_with_typed_id(draw):
    """Generate EntityMetadata with entity_id in 'entity_type:name' format.

    This ensures parse_entity_type(entity_id) returns the correct entity_type,
    which is required for the adapter's namespace mapping to work correctly.
    """
    entity_type = draw(_entity_type_strategy)
    name = draw(_entity_name_strategy)
    entity_id = f"{entity_type}:{name}"
    properties = draw(_properties_strategy)
    created_at = draw(_timestamp_strategy)
    updated_at = draw(_timestamp_strategy)

    return EntityMetadata(
        entity_id=entity_id,
        entity_type=entity_type,
        properties=properties,
        created_at=created_at,
        updated_at=updated_at,
    )


# Shared settings for property tests using function-scoped fixtures
_fixture_settings = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store():
    """Create a KeyValueMetadataStore backed by MemoryKeyValueService."""
    kv_service = MemoryKeyValueService()
    return KeyValueMetadataStore(kv_service=kv_service)


# ── Property 17: Metadata adapter round-trip ─────────────────────────────────


class TestMetadataAdapterRoundTrip:
    """Property 17: Metadata adapter round-trip.

    For any valid EntityMetadata, saving it through KeyValueMetadataStore
    and then retrieving it with get_metadata should return an EntityMetadata
    with equivalent entity_id, entity_type, properties, created_at, and
    updated_at.

    **Validates: Requirements 11.2, 11.3**
    """

    @given(metadata=entity_metadata_with_typed_id())
    @_fixture_settings
    def test_save_get_round_trip(self, store, metadata):
        """save_metadata(m) then get_metadata(m.entity_id) returns equivalent metadata.

        **Validates: Requirements 11.2, 11.3**
        """
        store.save_metadata(metadata)
        retrieved = store.get_metadata(metadata.entity_id)

        assert retrieved is not None, (
            f"get_metadata returned None for entity_id={metadata.entity_id!r}"
        )
        assert retrieved.entity_id == metadata.entity_id, (
            f"entity_id mismatch: {retrieved.entity_id!r} != {metadata.entity_id!r}"
        )
        assert retrieved.entity_type == metadata.entity_type, (
            f"entity_type mismatch: {retrieved.entity_type!r} != {metadata.entity_type!r}"
        )
        assert retrieved.properties == metadata.properties, (
            f"properties mismatch: {retrieved.properties!r} != {metadata.properties!r}"
        )
        assert retrieved.created_at == metadata.created_at, (
            f"created_at mismatch: {retrieved.created_at!r} != {metadata.created_at!r}"
        )
        assert retrieved.updated_at == metadata.updated_at, (
            f"updated_at mismatch: {retrieved.updated_at!r} != {metadata.updated_at!r}"
        )


# ── Property 18: Metadata adapter list_entities filtering ────────────────────


class TestMetadataAdapterListEntitiesFiltering:
    """Property 18: Metadata adapter list_entities filtering.

    For any set of EntityMetadata objects with various entity_types saved
    through KeyValueMetadataStore, calling list_entities with a specific
    entity_type should return exactly the entity_ids of that type, and
    calling list_entities without a filter should return all entity_ids.

    **Validates: Requirements 11.4, 11.5**
    """

    @given(
        metadata_list=st.lists(
            entity_metadata_with_typed_id(),
            min_size=1,
            max_size=15,
        )
    )
    @settings(max_examples=100)
    def test_list_entities_filtered_returns_correct_type(self, metadata_list):
        """list_entities(entity_type=T) returns exactly the entity_ids of type T.

        **Validates: Requirements 11.4, 11.5**
        """
        # Fresh store per iteration to avoid cross-iteration data leakage
        kv_service = MemoryKeyValueService()
        store = KeyValueMetadataStore(kv_service=kv_service)

        # Deduplicate by entity_id (last write wins, matching upsert semantics)
        unique_metadata = {}
        for m in metadata_list:
            unique_metadata[m.entity_id] = m

        # Save all metadata
        for m in unique_metadata.values():
            store.save_metadata(m)

        # Collect expected entity_ids grouped by entity_type
        expected_by_type = {}
        for m in unique_metadata.values():
            expected_by_type.setdefault(m.entity_type, set()).add(m.entity_id)

        # Test filtered listing for each entity_type present
        for entity_type, expected_ids in expected_by_type.items():
            result = store.list_entities(entity_type=entity_type)
            assert set(result) == expected_ids, (
                f"list_entities(entity_type={entity_type!r}) returned "
                f"{sorted(result)!r}, expected {sorted(expected_ids)!r}"
            )

        # Test unfiltered listing returns all entity_ids
        all_result = store.list_entities()
        all_expected = set(unique_metadata.keys())
        assert set(all_result) == all_expected, (
            f"list_entities() returned {sorted(all_result)!r}, "
            f"expected {sorted(all_expected)!r}"
        )


# ── Additional imports for piece adapter tests ───────────────────────────────

from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from science_modeling_tools.knowledge.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)


# ── Hypothesis strategies for KnowledgePiece ─────────────────────────────────

# Safe string strategy for identifiers (letters and digits only)
_safe_string = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)

# Content strategy: space-separated words so term overlap search works
_word = st.text(
    alphabet=st.characters(whitelist_categories=("L",)),
    min_size=2,
    max_size=10,
)

_content_strategy = st.lists(_word, min_size=1, max_size=10).map(
    lambda words: " ".join(words)
)

# Tag strategy: lowercase safe strings
_tag_strategy_piece = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=15,
).map(str.lower)

# Fixed timestamps to avoid auto-generation variability
_fixed_timestamp = st.from_regex(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\+00:00",
    fullmatch=True,
)


@st.composite
def knowledge_piece_for_adapter(draw):
    """Generate a KnowledgePiece suitable for adapter round-trip testing.

    Uses space-separated word content for search compatibility,
    explicit timestamps to avoid auto-generation variability,
    and safe strings for identifiers.
    """
    content = draw(_content_strategy)
    piece_id = draw(_safe_string)
    knowledge_type = draw(st.sampled_from(list(KnowledgeType)))
    tags = draw(st.lists(_tag_strategy_piece, min_size=0, max_size=5))
    entity_id = draw(st.one_of(st.none(), _safe_string))
    source = draw(st.one_of(st.none(), _safe_string))
    embedding_text = draw(st.one_of(st.none(), _content_strategy))
    created_at = draw(_fixed_timestamp)
    updated_at = draw(_fixed_timestamp)

    return KnowledgePiece(
        content=content,
        piece_id=piece_id,
        knowledge_type=knowledge_type,
        tags=tags,
        entity_id=entity_id,
        source=source,
        embedding_text=embedding_text,
        created_at=created_at,
        updated_at=updated_at,
    )


# ── Property 19: Piece adapter round-trip ────────────────────────────────────
# Feature: knowledge-service-extraction, Property 19: Piece adapter round-trip


class TestPieceAdapterRoundTrip:
    """Property 19: Piece adapter round-trip.

    For any valid KnowledgePiece, adding it through RetrievalKnowledgePieceStore
    and then retrieving it with get_by_id should return a KnowledgePiece with
    equivalent content, piece_id, knowledge_type, tags, entity_id, source,
    embedding_text, created_at, and updated_at.

    **Validates: Requirements 12.2, 12.4**
    """

    @given(piece=knowledge_piece_for_adapter())
    @settings(max_examples=100)
    def test_property_19_add_get_round_trip(self, piece):
        """add(piece) then get_by_id(piece.piece_id) returns equivalent piece.

        **Validates: Requirements 12.2, 12.4**
        """
        # Fresh store per iteration to avoid cross-iteration data leakage
        retrieval_service = MemoryRetrievalService()
        store = RetrievalKnowledgePieceStore(retrieval_service=retrieval_service)

        store.add(piece)
        retrieved = store.get_by_id(piece.piece_id)

        assert retrieved is not None, (
            f"get_by_id returned None for piece_id={piece.piece_id!r}"
        )
        assert retrieved.content == piece.content, (
            f"content mismatch: {retrieved.content!r} != {piece.content!r}"
        )
        assert retrieved.piece_id == piece.piece_id, (
            f"piece_id mismatch: {retrieved.piece_id!r} != {piece.piece_id!r}"
        )
        assert retrieved.knowledge_type == piece.knowledge_type, (
            f"knowledge_type mismatch: {retrieved.knowledge_type!r} != {piece.knowledge_type!r}"
        )
        assert retrieved.tags == piece.tags, (
            f"tags mismatch: {retrieved.tags!r} != {piece.tags!r}"
        )
        assert retrieved.entity_id == piece.entity_id, (
            f"entity_id mismatch: {retrieved.entity_id!r} != {piece.entity_id!r}"
        )
        assert retrieved.source == piece.source, (
            f"source mismatch: {retrieved.source!r} != {piece.source!r}"
        )
        assert retrieved.embedding_text == piece.embedding_text, (
            f"embedding_text mismatch: {retrieved.embedding_text!r} != {piece.embedding_text!r}"
        )
        assert retrieved.created_at == piece.created_at, (
            f"created_at mismatch: {retrieved.created_at!r} != {piece.created_at!r}"
        )
        assert retrieved.updated_at == piece.updated_at, (
            f"updated_at mismatch: {retrieved.updated_at!r} != {piece.updated_at!r}"
        )


# ── Property 20: Piece adapter search filter delegation ──────────────────────
# Feature: knowledge-service-extraction, Property 20: Piece adapter search filter delegation


class TestPieceAdapterSearchFilterDelegation:
    """Property 20: Piece adapter search filter delegation.

    For any set of KnowledgePieces with various knowledge_types and tags,
    searching through RetrievalKnowledgePieceStore with knowledge_type and
    tags filters should return only pieces matching those filters.

    **Validates: Requirements 12.3**
    """

    @given(
        pieces=st.lists(
            knowledge_piece_for_adapter(),
            min_size=2,
            max_size=15,
        ),
        filter_type=st.sampled_from(list(KnowledgeType)),
        filter_tags=st.lists(_tag_strategy_piece, min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_property_20_search_with_filters(self, pieces, filter_type, filter_tags):
        """search with knowledge_type and tags filters returns only matching pieces.

        **Validates: Requirements 12.3**
        """
        # Fresh store per iteration
        retrieval_service = MemoryRetrievalService()
        store = RetrievalKnowledgePieceStore(retrieval_service=retrieval_service)

        # Deduplicate by piece_id (last write wins) and use same entity_id
        # so all pieces are in the same namespace for search
        shared_entity_id = None  # Use default namespace for all pieces
        unique_pieces = {}
        for p in pieces:
            # Override entity_id so all pieces are in the same namespace
            deduped = KnowledgePiece(
                content=p.content,
                piece_id=p.piece_id,
                knowledge_type=p.knowledge_type,
                tags=p.tags,
                entity_id=shared_entity_id,
                source=p.source,
                embedding_text=p.embedding_text,
                created_at=p.created_at,
                updated_at=p.updated_at,
            )
            unique_pieces[deduped.piece_id] = deduped

        # Add all unique pieces
        for p in unique_pieces.values():
            store.add(p)

        # Collect all unique words from all pieces' content for a broad query
        all_words = set()
        for p in unique_pieces.values():
            all_words.update(p.content.lower().split())
        # Use a query that matches all pieces (all words)
        query = " ".join(all_words) if all_words else "test"

        # Search with knowledge_type filter only
        results_by_type = store.search(
            query=query,
            entity_id=shared_entity_id,
            knowledge_type=filter_type,
            top_k=len(unique_pieces) + 10,
        )
        for result_piece, score in results_by_type:
            assert result_piece.knowledge_type == filter_type, (
                f"Expected knowledge_type={filter_type!r}, "
                f"got {result_piece.knowledge_type!r} for piece_id={result_piece.piece_id!r}"
            )

        # Verify all matching pieces are returned (that have non-zero search score)
        expected_type_ids = {
            pid for pid, p in unique_pieces.items()
            if p.knowledge_type == filter_type
        }
        returned_type_ids = {p.piece_id for p, _ in results_by_type}
        # returned_type_ids should be a subset of expected (search may miss some
        # due to zero score), but should not contain non-matching pieces
        assert returned_type_ids <= expected_type_ids, (
            f"Search returned pieces not matching filter_type={filter_type!r}: "
            f"unexpected={returned_type_ids - expected_type_ids}"
        )

        # Search with tags filter only
        results_by_tags = store.search(
            query=query,
            entity_id=shared_entity_id,
            tags=filter_tags,
            top_k=len(unique_pieces) + 10,
        )
        for result_piece, score in results_by_tags:
            # All filter tags must be present in the piece's tags (AND containment)
            for tag in filter_tags:
                assert tag in result_piece.tags, (
                    f"Expected tag {tag!r} in piece tags {result_piece.tags!r} "
                    f"for piece_id={result_piece.piece_id!r}"
                )

        # Verify returned pieces are a subset of those that actually match
        expected_tag_ids = {
            pid for pid, p in unique_pieces.items()
            if all(t in p.tags for t in filter_tags)
        }
        returned_tag_ids = {p.piece_id for p, _ in results_by_tags}
        assert returned_tag_ids <= expected_tag_ids, (
            f"Search returned pieces not matching filter_tags={filter_tags!r}: "
            f"unexpected={returned_tag_ids - expected_tag_ids}"
        )

        # Search with both knowledge_type AND tags filters
        results_both = store.search(
            query=query,
            entity_id=shared_entity_id,
            knowledge_type=filter_type,
            tags=filter_tags,
            top_k=len(unique_pieces) + 10,
        )
        for result_piece, score in results_both:
            assert result_piece.knowledge_type == filter_type, (
                f"Expected knowledge_type={filter_type!r}, "
                f"got {result_piece.knowledge_type!r}"
            )
            for tag in filter_tags:
                assert tag in result_piece.tags, (
                    f"Expected tag {tag!r} in piece tags {result_piece.tags!r}"
                )

        # Verify combined filter results are subset of intersection
        expected_both_ids = expected_type_ids & expected_tag_ids
        returned_both_ids = {p.piece_id for p, _ in results_both}
        assert returned_both_ids <= expected_both_ids, (
            f"Search returned pieces not matching combined filters: "
            f"unexpected={returned_both_ids - expected_both_ids}"
        )


# ── Additional imports for graph adapter tests ───────────────────────────────

from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)
from science_modeling_tools.knowledge.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)

# Import strategies from conftest (pytest conftest is not directly importable,
# so we define local strategies using the same pattern)

# Safe identifier strategy for node/edge fields
_graph_identifier = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
)

# JSON-serializable properties for graph nodes/edges
_graph_json_leaf = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(max_size=50),
)

_graph_json_value = st.recursive(
    _graph_json_leaf,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
    ),
    max_leaves=10,
)

_graph_properties_strategy = st.dictionaries(
    st.text(min_size=1, max_size=20),
    _graph_json_value,
    max_size=5,
)


@st.composite
def graph_node_strategy(draw):
    """Generate a random GraphNode instance for adapter testing."""
    node_id = draw(_graph_identifier)
    node_type = draw(_graph_identifier)
    label = draw(st.text(max_size=50))
    properties = draw(_graph_properties_strategy)
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        properties=properties,
    )


@st.composite
def graph_edge_strategy(draw):
    """Generate a random GraphEdge instance for adapter testing."""
    source_id = draw(_graph_identifier)
    target_id = draw(_graph_identifier)
    edge_type = draw(_graph_identifier)
    properties = draw(_graph_properties_strategy)
    return GraphEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        properties=properties,
    )


# ── Property 21: Graph adapter node round-trip ───────────────────────────────
# Feature: knowledge-service-extraction, Property 21: Graph adapter node round-trip


class TestGraphAdapterNodeRoundTrip:
    """Property 21: Graph adapter node round-trip.

    For any valid GraphNode, adding it through GraphServiceEntityGraphStore
    and then retrieving it with get_node should return a GraphNode with
    equivalent node_id, node_type, label, and properties.

    **Validates: Requirements 13.2, 13.4**
    """

    @given(node=graph_node_strategy())
    @settings(max_examples=100)
    def test_property_21_add_get_node_round_trip(self, node):
        """add_node(node) then get_node(node.node_id) returns equivalent node.

        **Validates: Requirements 13.2, 13.4**
        """
        # Fresh store per iteration to avoid cross-iteration data leakage
        graph_service = MemoryGraphService()
        store = GraphServiceEntityGraphStore(graph_service=graph_service)

        store.add_node(node)
        retrieved = store.get_node(node.node_id)

        assert retrieved is not None, (
            f"get_node returned None for node_id={node.node_id!r}"
        )
        assert retrieved.node_id == node.node_id, (
            f"node_id mismatch: {retrieved.node_id!r} != {node.node_id!r}"
        )
        assert retrieved.node_type == node.node_type, (
            f"node_type mismatch: {retrieved.node_type!r} != {node.node_type!r}"
        )
        assert retrieved.label == node.label, (
            f"label mismatch: {retrieved.label!r} != {node.label!r}"
        )
        assert retrieved.properties == node.properties, (
            f"properties mismatch: {retrieved.properties!r} != {node.properties!r}"
        )


# ── Property 22: Graph adapter edge round-trip ───────────────────────────────
# Feature: knowledge-service-extraction, Property 22: Graph adapter edge round-trip


class TestGraphAdapterEdgeRoundTrip:
    """Property 22: Graph adapter edge round-trip.

    For any valid GraphEdge between two existing nodes, adding it through
    GraphServiceEntityGraphStore and then calling get_relations should
    include an edge with equivalent source_id, target_id, edge_type, and
    properties.

    **Validates: Requirements 13.3**
    """

    @given(
        source_node=graph_node_strategy(),
        target_node=graph_node_strategy(),
        edge=graph_edge_strategy(),
    )
    @settings(max_examples=100)
    def test_property_22_add_get_edge_round_trip(self, source_node, target_node, edge):
        """add_relation(edge) then get_relations(source_id) includes equivalent edge.

        **Validates: Requirements 13.3**
        """
        # Ensure source and target have distinct node_ids
        assume(source_node.node_id != target_node.node_id)

        # Override edge source/target to match our nodes
        edge = GraphEdge(
            source_id=source_node.node_id,
            target_id=target_node.node_id,
            edge_type=edge.edge_type,
            properties=edge.properties,
        )

        # Fresh store per iteration
        graph_service = MemoryGraphService()
        store = GraphServiceEntityGraphStore(graph_service=graph_service)

        # Add nodes first (required for edge creation)
        store.add_node(source_node)
        store.add_node(target_node)

        # Add edge
        store.add_relation(edge)

        # Retrieve edges from source node
        relations = store.get_relations(source_node.node_id, direction="outgoing")

        # Find the matching edge
        matching = [
            r for r in relations
            if r.source_id == edge.source_id
            and r.target_id == edge.target_id
            and r.edge_type == edge.edge_type
        ]

        assert len(matching) >= 1, (
            f"Expected at least one matching edge from {source_node.node_id!r} "
            f"to {target_node.node_id!r} with type {edge.edge_type!r}, "
            f"but found {len(matching)}. All relations: {relations!r}"
        )

        matched_edge = matching[0]
        assert matched_edge.source_id == edge.source_id, (
            f"source_id mismatch: {matched_edge.source_id!r} != {edge.source_id!r}"
        )
        assert matched_edge.target_id == edge.target_id, (
            f"target_id mismatch: {matched_edge.target_id!r} != {edge.target_id!r}"
        )
        assert matched_edge.edge_type == edge.edge_type, (
            f"edge_type mismatch: {matched_edge.edge_type!r} != {edge.edge_type!r}"
        )
        assert matched_edge.properties == edge.properties, (
            f"properties mismatch: {matched_edge.properties!r} != {edge.properties!r}"
        )


# ── Property 23: Graph adapter get_neighbors preserves depth ─────────────────
# Feature: knowledge-service-extraction, Property 23: Graph adapter get_neighbors preserves depth


class TestGraphAdapterGetNeighborsPreservesDepth:
    """Property 23: Graph adapter get_neighbors preserves depth.

    For any graph with edges stored through GraphServiceEntityGraphStore,
    calling get_neighbors should return GraphNode objects at the correct
    depth values matching the underlying graph structure.

    **Validates: Requirements 13.5**
    """

    @given(
        edge_type=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=15,
        ),
        chain_length=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=100)
    def test_property_23_get_neighbors_depth_values(self, edge_type, chain_length):
        """get_neighbors returns nodes at correct depth values for a chain graph.

        Creates a linear chain: n0 -> n1 -> n2 -> ... -> n_{chain_length-1}
        Then verifies that get_neighbors from n0 with sufficient depth returns
        each node at the correct depth value.

        **Validates: Requirements 13.5**
        """
        # Fresh store per iteration
        graph_service = MemoryGraphService()
        store = GraphServiceEntityGraphStore(graph_service=graph_service)

        # Create a chain of nodes
        nodes = []
        for i in range(chain_length):
            node = GraphNode(
                node_id=f"n{i}",
                node_type="test_node",
                label=f"Node {i}",
            )
            nodes.append(node)
            store.add_node(node)

        # Create edges forming a chain: n0 -> n1 -> n2 -> ...
        for i in range(chain_length - 1):
            edge = GraphEdge(
                source_id=f"n{i}",
                target_id=f"n{i+1}",
                edge_type=edge_type,
            )
            store.add_relation(edge)

        # Query neighbors from n0 with full depth
        max_depth = chain_length - 1
        neighbors = store.get_neighbors(
            node_id="n0",
            relation_type=edge_type,
            depth=max_depth,
        )

        # Build a dict of node_id -> depth from results
        neighbor_depths = {n.node_id: d for n, d in neighbors}

        # Verify each node in the chain (except n0) is at the correct depth
        for i in range(1, chain_length):
            expected_depth = i
            node_id = f"n{i}"
            assert node_id in neighbor_depths, (
                f"Node {node_id!r} not found in neighbors. "
                f"Got: {neighbor_depths!r}"
            )
            assert neighbor_depths[node_id] == expected_depth, (
                f"Depth mismatch for {node_id!r}: "
                f"got {neighbor_depths[node_id]}, expected {expected_depth}"
            )

        # Verify the source node (n0) is NOT in the results
        assert "n0" not in neighbor_depths, (
            f"Source node 'n0' should not be in neighbors, "
            f"but found at depth {neighbor_depths.get('n0')}"
        )

        # Verify total count matches expected
        assert len(neighbors) == chain_length - 1, (
            f"Expected {chain_length - 1} neighbors, got {len(neighbors)}"
        )

    @given(
        edge_type=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=15,
        ),
    )
    @settings(max_examples=100)
    def test_property_23_depth_limits_results(self, edge_type):
        """get_neighbors with depth=1 returns only direct neighbors.

        Creates a chain n0 -> n1 -> n2 -> n3 and verifies that
        depth=1 returns only n1, depth=2 returns n1 and n2, etc.

        **Validates: Requirements 13.5**
        """
        # Fresh store per iteration
        graph_service = MemoryGraphService()
        store = GraphServiceEntityGraphStore(graph_service=graph_service)

        # Create a 4-node chain: n0 -> n1 -> n2 -> n3
        for i in range(4):
            store.add_node(GraphNode(
                node_id=f"n{i}",
                node_type="test_node",
                label=f"Node {i}",
            ))
        for i in range(3):
            store.add_relation(GraphEdge(
                source_id=f"n{i}",
                target_id=f"n{i+1}",
                edge_type=edge_type,
            ))

        # Test depth=1: should only return n1
        neighbors_d1 = store.get_neighbors("n0", relation_type=edge_type, depth=1)
        d1_ids = {n.node_id for n, _ in neighbors_d1}
        assert d1_ids == {"n1"}, (
            f"depth=1 should return only n1, got {d1_ids}"
        )
        for node, depth in neighbors_d1:
            assert depth == 1, f"All depth=1 neighbors should have depth 1, got {depth}"

        # Test depth=2: should return n1 (depth=1) and n2 (depth=2)
        neighbors_d2 = store.get_neighbors("n0", relation_type=edge_type, depth=2)
        d2_map = {n.node_id: d for n, d in neighbors_d2}
        assert set(d2_map.keys()) == {"n1", "n2"}, (
            f"depth=2 should return n1 and n2, got {set(d2_map.keys())}"
        )
        assert d2_map["n1"] == 1, f"n1 should be at depth 1, got {d2_map['n1']}"
        assert d2_map["n2"] == 2, f"n2 should be at depth 2, got {d2_map['n2']}"

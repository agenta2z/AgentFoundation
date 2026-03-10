"""
Property-based tests for the graph_walk module (seed finders).

Feature: retrieval-pipeline-refactor
- Property 1: Search Seed Construction Preserves Score and Tags Source
- Property 2: Identity Seed Construction Uses Score 1.0
- Property 3: Seed Space Filtering (OR Semantics)
- Property 4: Seed Finder Precondition Guards

**Validates: Requirements 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4**
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, PropertyMock

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
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode

from agent_foundation.knowledge.retrieval.graph_walk import (
    SeedNode,
    find_search_seeds,
    find_identity_seeds,
)
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from conftest import InMemoryEntityGraphStore


# ── Hypothesis strategies ────────────────────────────────────────────────────

_identifier_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=30,
)

_space_strategy = st.sampled_from(["main", "personal", "developmental", "work", "testing"])

_node_type_strategy = st.sampled_from(["service", "person", "product", "location", "concept"])


@st.composite
def graph_node_with_spaces_strategy(draw, spaces=None):
    """Generate a GraphNode with explicit spaces in properties."""
    node_id = draw(_identifier_text)
    node_type = draw(_node_type_strategy)
    label = draw(st.text(max_size=30))
    if spaces is None:
        spaces = draw(st.lists(_space_strategy, min_size=0, max_size=3))
    properties = {"spaces": spaces}
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        properties=properties,
        is_active=True,
    )


def search_results_strategy(min_size=1, max_size=5):
    """Generate a list of (GraphNode, float) search results with spaces."""
    return st.lists(
        st.tuples(
            graph_node_with_spaces_strategy(),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=min_size,
        max_size=max_size,
    )


def _make_semantic_graph_store(search_results: List[Tuple[GraphNode, float]]):
    """Create a mock graph store that supports semantic search."""
    store = MagicMock(spec=EntityGraphStore)
    type(store).supports_semantic_search = PropertyMock(return_value=True)
    store.search_nodes.return_value = search_results
    return store


def _make_non_semantic_graph_store():
    """Create a mock graph store that does NOT support semantic search."""
    store = MagicMock(spec=EntityGraphStore)
    type(store).supports_semantic_search = PropertyMock(return_value=False)
    return store


# ── Property 1: Search Seed Construction Preserves Score and Tags Source ─────
# Feature: retrieval-pipeline-refactor, Property 1


class TestSearchSeedConstruction:
    """Property 1: Search Seed Construction Preserves Score and Tags Source.

    **Validates: Requirements 1.2, 1.4, 2.1**
    """

    @given(search_results=search_results_strategy(min_size=1, max_size=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_search_seed_preserves_score_and_source(
        self, search_results: List[Tuple[GraphNode, float]]
    ):
        """For any search results, find_search_seeds() produces SeedNodes where
        each score equals the original search score, source is 'search', and
        node is the original GraphNode."""
        # Feature: retrieval-pipeline-refactor, Property 1: Search Seed Construction
        store = _make_semantic_graph_store(search_results)

        seeds = find_search_seeds(store, "test query", top_k=10)

        # Every seed must preserve score, source, and node reference
        assert len(seeds) == len(search_results)
        for seed, (original_node, original_score) in zip(seeds, search_results):
            assert seed.score == original_score
            assert seed.source == "search"
            assert seed.node is original_node
            assert seed.node.node_id == original_node.node_id
            assert seed.node.label == original_node.label
            assert seed.node.properties == original_node.properties


# ── Property 2: Identity Seed Construction Uses Score 1.0 ───────────────────
# Feature: retrieval-pipeline-refactor, Property 2


class TestIdentitySeedConstruction:
    """Property 2: Identity Seed Construction Uses Score 1.0.

    **Validates: Requirements 1.3, 1.4, 3.1**
    """

    @given(node=graph_node_with_spaces_strategy())
    @settings(max_examples=100)
    def test_identity_seed_score_is_one(self, node: GraphNode):
        """For any valid entity_id that exists in the graph store,
        find_identity_seeds() produces exactly one SeedNode with score=1.0,
        source='identity', and node equal to the store's node."""
        # Feature: retrieval-pipeline-refactor, Property 2: Identity Seed Construction
        store = InMemoryEntityGraphStore()
        store.add_node(node)

        seeds = find_identity_seeds(store, node.node_id)

        assert len(seeds) == 1
        seed = seeds[0]
        assert seed.score == 1.0
        assert seed.source == "identity"
        assert seed.node.node_id == node.node_id
        assert seed.node.label == node.label


# ── Property 3: Seed Space Filtering (OR Semantics) ─────────────────────────
# Feature: retrieval-pipeline-refactor, Property 3


class TestSeedSpaceFiltering:
    """Property 3: Seed Space Filtering (OR Semantics).

    **Validates: Requirements 2.2, 3.2**
    """

    @given(
        search_results=search_results_strategy(min_size=1, max_size=5),
        filter_spaces=st.lists(_space_strategy, min_size=1, max_size=3),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_search_seeds_filtered_by_spaces(
        self,
        search_results: List[Tuple[GraphNode, float]],
        filter_spaces: List[str],
    ):
        """All SeedNodes returned by find_search_seeds() with a spaces filter
        have at least one space intersecting with the requested spaces."""
        # Feature: retrieval-pipeline-refactor, Property 3: Seed Space Filtering
        store = _make_semantic_graph_store(search_results)

        seeds = find_search_seeds(store, "test query", top_k=10, spaces=filter_spaces)

        filter_set = set(filter_spaces)
        for seed in seeds:
            node_spaces = set(seed.node.properties.get("spaces", []))
            assert node_spaces & filter_set, (
                f"Seed node {seed.node.node_id} has spaces {node_spaces} "
                f"which don't intersect with filter {filter_set}"
            )

    @given(
        node=graph_node_with_spaces_strategy(),
        filter_spaces=st.lists(_space_strategy, min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_identity_seeds_not_filtered_by_spaces(
        self,
        node: GraphNode,
        filter_spaces: List[str],
    ):
        """find_identity_seeds() always returns the identity node regardless
        of spaces filter — the identity node is the user's own entity and
        must not be filtered out.  Space filtering is applied to neighbors
        during graph_walk() instead."""
        # Feature: retrieval-pipeline-refactor, Property 3: Seed Space Filtering
        store = InMemoryEntityGraphStore()
        store.add_node(node)

        seeds = find_identity_seeds(store, node.node_id, spaces=filter_spaces)

        # Identity seed is always returned (not filtered by spaces)
        assert len(seeds) == 1, (
            f"Expected 1 identity seed, got {len(seeds)} "
            f"(node spaces={node.properties.get('spaces', [])}, "
            f"filter={filter_spaces})"
        )
        assert seeds[0].node.node_id == node.node_id
        assert seeds[0].score == 1.0
        assert seeds[0].source == "identity"


# ── Property 4: Seed Finder Precondition Guards ─────────────────────────────
# Feature: retrieval-pipeline-refactor, Property 4


class TestSeedFinderPreconditionGuards:
    """Property 4: Seed Finder Precondition Guards.

    **Validates: Requirements 2.3, 2.4, 3.3, 3.4**
    """

    @given(
        query=st.sampled_from(["", "   ", "\t", "\n", "  \n\t  "]),
    )
    @settings(max_examples=100)
    def test_search_seeds_empty_for_empty_query(self, query: str):
        """find_search_seeds() returns empty list when query is empty or
        whitespace-only."""
        # Feature: retrieval-pipeline-refactor, Property 4: Precondition Guards
        store = _make_semantic_graph_store([])

        seeds = find_search_seeds(store, query, top_k=5)

        assert seeds == []
        # search_nodes should NOT have been called
        store.search_nodes.assert_not_called()

    @given(
        query=st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_search_seeds_empty_for_unsupported_store(self, query: str):
        """find_search_seeds() returns empty list when the graph store does
        not support semantic search."""
        # Feature: retrieval-pipeline-refactor, Property 4: Precondition Guards
        store = _make_non_semantic_graph_store()

        seeds = find_search_seeds(store, query, top_k=5)

        assert seeds == []

    @given(
        entity_id=st.sampled_from(["", None]),
    )
    @settings(max_examples=100)
    def test_identity_seeds_empty_for_missing_entity_id(self, entity_id):
        """find_identity_seeds() returns empty list when entity_id is None
        or empty."""
        # Feature: retrieval-pipeline-refactor, Property 4: Precondition Guards
        store = InMemoryEntityGraphStore()

        seeds = find_identity_seeds(store, entity_id)

        assert seeds == []

    @given(
        entity_id=_identifier_text,
    )
    @settings(max_examples=100)
    def test_identity_seeds_empty_for_nonexistent_entity(self, entity_id: str):
        """find_identity_seeds() returns empty list when entity_id does not
        exist in the graph store."""
        # Feature: retrieval-pipeline-refactor, Property 4: Precondition Guards
        store = InMemoryEntityGraphStore()
        # Don't add any nodes — entity_id won't be found

        seeds = find_identity_seeds(store, entity_id)

        assert seeds == []


# ── Additional imports for graph walk property tests ─────────────────────────

from unittest.mock import patch
from agent_foundation.knowledge.retrieval.graph_walk import (
    graph_walk,
    merge_graph_contexts,
    _should_skip_piece,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from rich_python_utils.service_utils.graph_service.graph_node import GraphEdge


# ── In-memory piece store for property tests ─────────────────────────────────

class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory piece store for graph walk property tests."""

    def __init__(self):
        self._pieces: dict = {}

    def add(self, piece: KnowledgePiece) -> str:
        self._pieces[piece.piece_id] = piece
        return piece.piece_id

    def get_by_id(self, piece_id: str):
        return self._pieces.get(piece_id)

    def update(self, piece: KnowledgePiece) -> bool:
        if piece.piece_id in self._pieces:
            self._pieces[piece.piece_id] = piece
            return True
        return False

    def remove(self, piece_id: str) -> bool:
        return self._pieces.pop(piece_id, None) is not None

    def search(self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5, spaces=None):
        return [(p, 0.5) for p in list(self._pieces.values())[:top_k]]

    def list_all(self, entity_id=None, knowledge_type=None, spaces=None):
        return list(self._pieces.values())


# ── Strategies for graph walk tests ──────────────────────────────────────────

_relation_type_strategy = st.sampled_from([
    "WORKS_AT", "KNOWS", "USES", "MANAGES", "BELONGS_TO", "CREATED_BY",
])

_info_type_strategy = st.sampled_from(["context", "user_profile", "instructions", "episodic"])


@st.composite
def seed_node_strategy(draw, source=None, spaces=None):
    """Generate a SeedNode with random node, score, and source."""
    node = draw(graph_node_with_spaces_strategy(spaces=spaces))
    score = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    if source is None:
        source = draw(st.sampled_from(["search", "identity"]))
    return SeedNode(node=node, score=score, source=source)


@st.composite
def graph_context_entry_strategy(draw, min_depth=0, max_depth=3):
    """Generate a graph context entry dict."""
    return {
        "relation_type": draw(st.sampled_from(["SEARCH_HIT", "IDENTITY", "WORKS_AT", "KNOWS", "RELATED"])),
        "target_node_id": draw(_identifier_text),
        "target_label": draw(st.text(max_size=20)),
        "piece": None,
        "depth": draw(st.integers(min_value=min_depth, max_value=max_depth)),
        "score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    }


# ── Property 5: Depth-Decayed Scoring Formula ───────────────────────────────
# Feature: retrieval-pipeline-refactor, Property 5


class TestDepthDecayedScoring:
    """Property 5: Depth-Decayed Scoring Formula.

    **Validates: Requirements 4.2, 4.5**
    """

    @given(
        seed=seed_node_strategy(),
        traversal_depth=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_depth_decayed_scoring(self, seed: SeedNode, traversal_depth: int):
        """For any seed node with score s and any neighbor at depth D,
        the graph context entry's score equals s × 1/(D+1). At depth 0
        the score equals s."""
        # Feature: retrieval-pipeline-refactor, Property 5: Depth-Decayed Scoring Formula
        graph_store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        # Add seed node
        graph_store.add_node(seed.node)

        # Create a chain of neighbors at increasing depths
        prev_node_id = seed.node.node_id
        for d in range(1, traversal_depth + 1):
            neighbor = GraphNode(
                node_id=f"neighbor_{d}_{seed.node.node_id}",
                node_type="concept",
                label=f"Neighbor {d}",
                properties={"spaces": seed.node.properties.get("spaces", [])},
                is_active=True,
            )
            graph_store.add_node(neighbor)
            graph_store.add_relation(GraphEdge(
                source_id=prev_node_id,
                target_id=neighbor.node_id,
                edge_type="RELATED",
            ))
            prev_node_id = neighbor.node_id

        result = graph_walk(graph_store, piece_store, [seed], traversal_depth=traversal_depth)

        # Depth-0 entry should have score == seed.score
        depth_0_entries = [e for e in result if e["depth"] == 0]
        assert len(depth_0_entries) == 1
        assert depth_0_entries[0]["score"] == seed.score

        # Each neighbor at depth D should have score == seed.score * 1/(D+1)
        for entry in result:
            d = entry["depth"]
            expected_score = seed.score * (1.0 / (d + 1))
            assert abs(entry["score"] - expected_score) < 1e-9, (
                f"At depth {d}: expected {expected_score}, got {entry['score']}"
            )


# ── Property 6: Graph Walk Space Filtering ───────────────────────────────────
# Feature: retrieval-pipeline-refactor, Property 6


class TestGraphWalkSpaceFiltering:
    """Property 6: Graph Walk Space Filtering.

    **Validates: Requirements 4.4**
    """

    @given(
        seed_spaces=st.lists(_space_strategy, min_size=1, max_size=2),
        filter_spaces=st.lists(_space_strategy, min_size=1, max_size=2),
        neighbor_spaces_list=st.lists(
            st.lists(_space_strategy, min_size=0, max_size=3),
            min_size=1,
            max_size=4,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_walk_space_filtering(
        self,
        seed_spaces: List[str],
        filter_spaces: List[str],
        neighbor_spaces_list: List[List[str]],
    ):
        """All graph context entries in the output correspond to nodes whose
        spaces intersect with the requested spaces. No entry for a node
        outside the requested spaces shall appear."""
        # Feature: retrieval-pipeline-refactor, Property 6: Graph Walk Space Filtering
        graph_store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        # Seed node must pass the filter (seed finders handle this)
        seed_node = GraphNode(
            node_id="seed_0",
            node_type="person",
            label="Seed",
            properties={"spaces": seed_spaces},
            is_active=True,
        )
        graph_store.add_node(seed_node)
        seed = SeedNode(node=seed_node, score=0.8, source="search")

        # Add neighbors with various spaces
        for i, ns in enumerate(neighbor_spaces_list):
            neighbor = GraphNode(
                node_id=f"neighbor_{i}",
                node_type="concept",
                label=f"Neighbor {i}",
                properties={"spaces": ns},
                is_active=True,
            )
            graph_store.add_node(neighbor)
            graph_store.add_relation(GraphEdge(
                source_id=seed_node.node_id,
                target_id=neighbor.node_id,
                edge_type="KNOWS",
            ))

        result = graph_walk(
            graph_store, piece_store, [seed],
            traversal_depth=1, spaces=filter_spaces,
        )

        filter_set = set(filter_spaces)
        for entry in result:
            if entry["depth"] == 0:
                # Depth-0 is the seed itself — seed filtering is done by seed finders
                continue
            # All neighbor entries must have spaces intersecting with filter
            node = graph_store.get_node(entry["target_node_id"])
            if node:
                node_spaces = set(node.properties.get("spaces", []))
                assert node_spaces & filter_set, (
                    f"Node {entry['target_node_id']} has spaces {node_spaces} "
                    f"which don't intersect with filter {filter_set}"
                )


# ── Property 7: Graph Walk Edge Relation Lookup ─────────────────────────────
# Feature: retrieval-pipeline-refactor, Property 7


class TestGraphWalkEdgeRelationLookup:
    """Property 7: Graph Walk Edge Relation Lookup.

    **Validates: Requirements 4.6**
    """

    @given(
        edge_types=st.lists(
            _relation_type_strategy,
            min_size=1,
            max_size=4,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_walk_edge_relation_lookup(self, edge_types: List[str]):
        """For depth-1 neighbors, the entry's relation_type equals the edge_type
        of the actual edge. For depth > 1, relation_type is 'RELATED'."""
        # Feature: retrieval-pipeline-refactor, Property 7: Graph Walk Edge Relation Lookup
        graph_store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        seed_node = GraphNode(
            node_id="seed_edge_test",
            node_type="person",
            label="Seed",
            properties={"spaces": ["main"]},
            is_active=True,
        )
        graph_store.add_node(seed_node)
        seed = SeedNode(node=seed_node, score=0.9, source="search")

        # Add depth-1 neighbors with specific edge types
        expected_relations = {}
        for i, edge_type in enumerate(edge_types):
            neighbor = GraphNode(
                node_id=f"d1_neighbor_{i}",
                node_type="concept",
                label=f"D1 Neighbor {i}",
                properties={"spaces": ["main"]},
                is_active=True,
            )
            graph_store.add_node(neighbor)
            graph_store.add_relation(GraphEdge(
                source_id=seed_node.node_id,
                target_id=neighbor.node_id,
                edge_type=edge_type,
            ))
            expected_relations[neighbor.node_id] = edge_type

            # Add a depth-2 neighbor from each depth-1 neighbor
            d2_neighbor = GraphNode(
                node_id=f"d2_neighbor_{i}",
                node_type="concept",
                label=f"D2 Neighbor {i}",
                properties={"spaces": ["main"]},
                is_active=True,
            )
            graph_store.add_node(d2_neighbor)
            graph_store.add_relation(GraphEdge(
                source_id=neighbor.node_id,
                target_id=d2_neighbor.node_id,
                edge_type="CHILD_OF",
            ))

        result = graph_walk(graph_store, piece_store, [seed], traversal_depth=2)

        for entry in result:
            if entry["depth"] == 0:
                continue
            elif entry["depth"] == 1:
                # Depth-1: relation_type should match the edge_type
                assert entry["target_node_id"] in expected_relations, (
                    f"Unexpected depth-1 node: {entry['target_node_id']}"
                )
                assert entry["relation_type"] == expected_relations[entry["target_node_id"]], (
                    f"Expected relation_type={expected_relations[entry['target_node_id']]}, "
                    f"got {entry['relation_type']}"
                )
            else:
                # Depth > 1: relation_type should be "RELATED"
                assert entry["relation_type"] == "RELATED", (
                    f"Expected RELATED for depth {entry['depth']}, got {entry['relation_type']}"
                )


# ── Property 8: Graph Walk Piece Dedup ───────────────────────────────────────
# Feature: retrieval-pipeline-refactor, Property 8


class TestGraphWalkPieceDedup:
    """Property 8: Graph Walk Piece Dedup.

    **Validates: Requirements 4.7**
    """

    @given(
        info_type=_info_type_strategy,
        ignore_mode=st.sampled_from(["bool_true", "list_match", "list_no_match", "false"]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_walk_piece_dedup(self, info_type: str, ignore_mode: str):
        """When a depth-1 neighbor's edge has a piece_id in already_retrieved_piece_ids
        and ignore_already_retrieved is enabled, the entry's piece is None.
        Otherwise, the piece is looked up from the piece store."""
        # Feature: retrieval-pipeline-refactor, Property 8: Graph Walk Piece Dedup
        graph_store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        # Create a piece
        piece = KnowledgePiece(
            content="Test content",
            piece_id="piece_1",
            info_type=info_type,
        )
        piece_store.add(piece)

        # Create seed and neighbor with edge linking to the piece
        seed_node = GraphNode(
            node_id="seed_dedup",
            node_type="person",
            label="Seed",
            properties={"spaces": ["main"]},
            is_active=True,
        )
        neighbor = GraphNode(
            node_id="neighbor_dedup",
            node_type="concept",
            label="Neighbor",
            properties={"spaces": ["main"]},
            is_active=True,
        )
        graph_store.add_node(seed_node)
        graph_store.add_node(neighbor)
        graph_store.add_relation(GraphEdge(
            source_id=seed_node.node_id,
            target_id=neighbor.node_id,
            edge_type="WORKS_AT",
            properties={"piece_id": "piece_1"},
        ))

        seed = SeedNode(node=seed_node, score=0.9, source="search")
        already_retrieved = {"piece_1": info_type}

        # Configure ignore mode
        if ignore_mode == "bool_true":
            ignore = True
            should_skip = True
        elif ignore_mode == "list_match":
            ignore = [info_type]
            should_skip = True
        elif ignore_mode == "list_no_match":
            ignore = ["nonexistent_type"]
            should_skip = False
        else:  # "false"
            ignore = False
            should_skip = False

        result = graph_walk(
            graph_store, piece_store, [seed],
            traversal_depth=1,
            already_retrieved_piece_ids=already_retrieved,
            ignore_already_retrieved=ignore,
        )

        # Find the depth-1 entry
        depth_1_entries = [e for e in result if e["depth"] == 1]
        assert len(depth_1_entries) == 1

        if should_skip:
            assert depth_1_entries[0]["piece"] is None, (
                f"Expected piece=None when dedup is active (mode={ignore_mode})"
            )
        else:
            assert depth_1_entries[0]["piece"] is not None, (
                f"Expected piece to be present when dedup is inactive (mode={ignore_mode})"
            )
            assert depth_1_entries[0]["piece"].piece_id == "piece_1"


# ── Property 9: Graph Walk Preserves Per-Seed Provenance ─────────────────────
# Feature: retrieval-pipeline-refactor, Property 9


class TestGraphWalkPreservesProvenance:
    """Property 9: Graph Walk Preserves Per-Seed Provenance.

    **Validates: Requirements 4.2, 4.8**
    """

    @given(
        num_search_seeds=st.integers(min_value=0, max_value=3),
        num_identity_seeds=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_walk_preserves_provenance(
        self,
        num_search_seeds: int,
        num_identity_seeds: int,
    ):
        """For N seed nodes, the output contains exactly N depth-0 entries.
        Search seeds have relation_type='SEARCH_HIT', identity seeds have
        relation_type='IDENTITY'."""
        # Feature: retrieval-pipeline-refactor, Property 9: Per-Seed Provenance
        graph_store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        seeds = []
        for i in range(num_search_seeds):
            node = GraphNode(
                node_id=f"search_seed_{i}",
                node_type="concept",
                label=f"Search Seed {i}",
                properties={"spaces": ["main"]},
                is_active=True,
            )
            graph_store.add_node(node)
            seeds.append(SeedNode(node=node, score=0.5 + i * 0.1, source="search"))

        for i in range(num_identity_seeds):
            node = GraphNode(
                node_id=f"identity_seed_{i}",
                node_type="person",
                label=f"Identity Seed {i}",
                properties={"spaces": ["main"]},
                is_active=True,
            )
            graph_store.add_node(node)
            seeds.append(SeedNode(node=node, score=1.0, source="identity"))

        result = graph_walk(graph_store, piece_store, seeds, traversal_depth=1)

        # Count depth-0 entries
        depth_0_entries = [e for e in result if e["depth"] == 0]
        total_seeds = num_search_seeds + num_identity_seeds
        assert len(depth_0_entries) == total_seeds, (
            f"Expected {total_seeds} depth-0 entries, got {len(depth_0_entries)}"
        )

        # Verify relation_type matches source
        search_d0 = [e for e in depth_0_entries if e["relation_type"] == "SEARCH_HIT"]
        identity_d0 = [e for e in depth_0_entries if e["relation_type"] == "IDENTITY"]
        assert len(search_d0) == num_search_seeds
        assert len(identity_d0) == num_identity_seeds


# ── Property 10: Merge Dedup by (node_id, relation_type) with Score/Depth Tiebreaker ──
# Feature: retrieval-pipeline-refactor, Property 10


class TestMergeDedupAndTiebreaker:
    """Property 10: Merge Dedup by (node_id, relation_type) with Score/Depth Tiebreaker.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    """

    @given(
        search_entries=st.lists(graph_context_entry_strategy(), min_size=0, max_size=6),
        identity_entries=st.lists(graph_context_entry_strategy(), min_size=0, max_size=6),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_merge_dedup_and_tiebreaker(
        self,
        search_entries: List[Dict[str, Any]],
        identity_entries: List[Dict[str, Any]],
    ):
        """merge_graph_contexts() produces a result where:
        1. No two entries share the same (target_node_id, relation_type) pair.
        2. When duplicates existed, the entry with the higher score is kept.
        3. When duplicates had equal scores, the entry with the shorter depth is kept.
        4. The same target_node_id CAN appear with different relation_type values."""
        # Feature: retrieval-pipeline-refactor, Property 10: Merge Dedup and Tiebreaker
        merged = merge_graph_contexts(search_entries, identity_entries)

        # Property 1: No duplicate (target_node_id, relation_type) keys
        keys = [(e["target_node_id"], e.get("relation_type", "RELATED")) for e in merged]
        assert len(keys) == len(set(keys)), "Duplicate keys found in merged result"

        # Property 2 & 3: For each key, the kept entry has the best score/depth
        all_entries = search_entries + identity_entries
        for entry in merged:
            key = (entry["target_node_id"], entry.get("relation_type", "RELATED"))
            # Find all entries with this key from the input
            candidates = [
                e for e in all_entries
                if (e["target_node_id"], e.get("relation_type", "RELATED")) == key
            ]
            assert len(candidates) >= 1

            # The kept entry should be the best candidate
            best = candidates[0]
            for c in candidates[1:]:
                c_score = c.get("score", 0)
                best_score = best.get("score", 0)
                if c_score > best_score or (
                    c_score == best_score and c["depth"] < best["depth"]
                ):
                    best = c

            assert entry.get("score", 0) == best.get("score", 0), (
                f"Expected score {best.get('score', 0)}, got {entry.get('score', 0)}"
            )
            assert entry["depth"] == best["depth"], (
                f"Expected depth {best['depth']}, got {entry['depth']}"
            )

        # Property 4: Same node_id CAN appear with different relation_types
        # (This is verified implicitly — we just check no false dedup)
        node_ids_in_merged = [e["target_node_id"] for e in merged]
        for nid in set(node_ids_in_merged):
            entries_for_node = [e for e in merged if e["target_node_id"] == nid]
            relation_types = [e.get("relation_type", "RELATED") for e in entries_for_node]
            # All relation_types for the same node must be unique
            assert len(relation_types) == len(set(relation_types))

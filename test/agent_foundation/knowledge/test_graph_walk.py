"""
Unit tests for the graph_walk module.

Tests edge cases, error handling, space filtering, and merge behavior
for the unified graph walk functions.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5.1, 5.2, 5.3, 5.4
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

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

from agent_foundation.knowledge.retrieval.graph_walk import (
    SeedNode,
    find_search_seeds,
    find_identity_seeds,
    graph_walk,
    merge_graph_contexts,
    _should_skip_piece,
    _node_passes_space_filter,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from conftest import InMemoryEntityGraphStore


# ── In-memory piece store for unit tests ─────────────────────────────────────


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory piece store for graph walk unit tests."""

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


class FailingPieceStore(KnowledgePieceStore):
    """Piece store that raises on get_by_id."""

    def add(self, piece):
        return piece.piece_id

    def get_by_id(self, piece_id):
        raise RuntimeError("piece store failure")

    def update(self, piece):
        return False

    def remove(self, piece_id):
        return False

    def search(self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5, spaces=None):
        return []

    def list_all(self, entity_id=None, knowledge_type=None, spaces=None):
        return []


# ── Helper to build a simple graph ──────────────────────────────────────────


def _make_node(node_id: str, spaces: Optional[List[str]] = None, label: str = "") -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type="concept",
        label=label or node_id,
        properties={"spaces": spaces or []},
        is_active=True,
    )


def _make_seed(node: GraphNode, score: float = 0.8, source: str = "search") -> SeedNode:
    return SeedNode(node=node, score=score, source=source)


# ── 1. Empty seeds → empty result ───────────────────────────────────────────


class TestEmptySeeds:
    """graph_walk with empty seeds list returns empty list.

    Validates: Requirement 4.1
    """

    def test_empty_seeds_returns_empty(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        result = graph_walk(store, piece_store, [], traversal_depth=1)

        assert result == []

    def test_empty_seeds_with_spaces_returns_empty(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        result = graph_walk(store, piece_store, [], traversal_depth=1, spaces=["main"])

        assert result == []


# ── 2. Empty graph (seed exists, no neighbors) ──────────────────────────────


class TestEmptyGraph:
    """Seed node exists but has no neighbors → only depth-0 entry returned.

    Validates: Requirements 4.1, 4.2
    """

    def test_seed_with_no_neighbors(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        node = _make_node("lonely", spaces=["main"])
        store.add_node(node)
        seed = _make_seed(node, score=0.7, source="search")

        result = graph_walk(store, piece_store, [seed], traversal_depth=1)

        assert len(result) == 1
        assert result[0]["depth"] == 0
        assert result[0]["relation_type"] == "SEARCH_HIT"
        assert result[0]["target_node_id"] == "lonely"
        assert result[0]["score"] == 0.7
        assert result[0]["piece"] is None


# ── 3. Single-node graph ────────────────────────────────────────────────────


class TestSingleNodeGraph:
    """One seed, no neighbors → one depth-0 entry.

    Validates: Requirements 4.1, 4.2
    """

    def test_single_identity_seed_no_neighbors(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        node = _make_node("user_1", spaces=["personal"])
        store.add_node(node)
        seed = _make_seed(node, score=1.0, source="identity")

        result = graph_walk(store, piece_store, [seed], traversal_depth=1)

        assert len(result) == 1
        assert result[0]["depth"] == 0
        assert result[0]["relation_type"] == "IDENTITY"
        assert result[0]["target_node_id"] == "user_1"
        assert result[0]["score"] == 1.0


# ── 4. Depth=0 traversal ────────────────────────────────────────────────────


class TestDepthZeroTraversal:
    """depth=0 should emit depth-0 entries but no neighbors.

    Validates: Requirements 4.1, 4.2, 4.3
    """

    def test_depth_zero_emits_only_seed_entries(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        seed_node = _make_node("seed", spaces=["main"])
        neighbor = _make_node("neighbor", spaces=["main"])
        store.add_node(seed_node)
        store.add_node(neighbor)
        store.add_relation(GraphEdge(
            source_id="seed", target_id="neighbor", edge_type="KNOWS",
        ))
        seed = _make_seed(seed_node, score=0.9, source="search")

        result = graph_walk(store, piece_store, [seed], traversal_depth=0)

        assert len(result) == 1
        assert result[0]["depth"] == 0
        assert result[0]["target_node_id"] == "seed"


# ── 5. Graph store get_neighbors exception ───────────────────────────────────


class TestGetNeighborsException:
    """graph_store.get_neighbors() exception: catch, log warning, skip that seed, continue.

    Validates: Requirements 4.3, 4.8
    """

    def test_get_neighbors_exception_skips_seed_walk(self):
        """When get_neighbors raises, the depth-0 entry is still emitted
        but no neighbor entries appear. Other seeds are unaffected."""
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        good_node = _make_node("good_seed", spaces=["main"])
        bad_node = _make_node("bad_seed", spaces=["main"])
        neighbor = _make_node("neighbor_of_good", spaces=["main"])
        store.add_node(good_node)
        store.add_node(bad_node)
        store.add_node(neighbor)
        store.add_relation(GraphEdge(
            source_id="good_seed", target_id="neighbor_of_good", edge_type="KNOWS",
        ))

        good_seed = _make_seed(good_node, score=0.8, source="search")
        bad_seed = _make_seed(bad_node, score=0.6, source="search")

        # Patch get_neighbors to fail only for bad_seed
        original_get_neighbors = store.get_neighbors

        def patched_get_neighbors(node_id, **kwargs):
            if node_id == "bad_seed":
                raise RuntimeError("graph store failure")
            return original_get_neighbors(node_id, **kwargs)

        store.get_neighbors = patched_get_neighbors

        result = graph_walk(store, piece_store, [bad_seed, good_seed], traversal_depth=1)

        # bad_seed: depth-0 entry only (walk skipped)
        # good_seed: depth-0 + depth-1 neighbor
        depth_0 = [e for e in result if e["depth"] == 0]
        assert len(depth_0) == 2  # both seeds emit depth-0

        depth_1 = [e for e in result if e["depth"] == 1]
        assert len(depth_1) == 1
        assert depth_1[0]["target_node_id"] == "neighbor_of_good"


# ── 6. Graph store get_relations exception ───────────────────────────────────


class TestGetRelationsException:
    """graph_store.get_relations() exception: catch, set relation_type='RELATED' and piece=None.

    Validates: Requirements 4.6
    """

    def test_get_relations_exception_uses_related_fallback(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        seed_node = _make_node("seed_rel", spaces=["main"])
        neighbor = _make_node("neighbor_rel", spaces=["main"])
        store.add_node(seed_node)
        store.add_node(neighbor)
        store.add_relation(GraphEdge(
            source_id="seed_rel", target_id="neighbor_rel", edge_type="WORKS_AT",
            properties={"piece_id": "some_piece"},
        ))

        seed = _make_seed(seed_node, score=0.9, source="search")

        # Patch get_relations to raise
        original_get_relations = store.get_relations
        store.get_relations = MagicMock(side_effect=RuntimeError("relations failure"))

        result = graph_walk(store, piece_store, [seed], traversal_depth=1)

        depth_1 = [e for e in result if e["depth"] == 1]
        assert len(depth_1) == 1
        assert depth_1[0]["relation_type"] == "RELATED"
        assert depth_1[0]["piece"] is None


# ── 7. Piece store get_by_id returns None ────────────────────────────────────


class TestPieceStoreReturnsNone:
    """piece_store.get_by_id returns None → entry's piece stays None.

    Validates: Requirements 4.7
    """

    def test_piece_store_returns_none(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()  # empty — get_by_id returns None

        seed_node = _make_node("seed_pn", spaces=["main"])
        neighbor = _make_node("neighbor_pn", spaces=["main"])
        store.add_node(seed_node)
        store.add_node(neighbor)
        store.add_relation(GraphEdge(
            source_id="seed_pn", target_id="neighbor_pn", edge_type="USES",
            properties={"piece_id": "nonexistent_piece"},
        ))

        seed = _make_seed(seed_node, score=0.8, source="search")

        result = graph_walk(store, piece_store, [seed], traversal_depth=1)

        depth_1 = [e for e in result if e["depth"] == 1]
        assert len(depth_1) == 1
        assert depth_1[0]["piece"] is None


# ── 8. Piece store get_by_id raises exception ───────────────────────────────


class TestPieceStoreException:
    """piece_store.get_by_id raises → catch, set piece=None.

    Validates: Requirements 4.7
    """

    def test_piece_store_exception_sets_piece_none(self):
        store = InMemoryEntityGraphStore()
        piece_store = FailingPieceStore()

        seed_node = _make_node("seed_pe", spaces=["main"])
        neighbor = _make_node("neighbor_pe", spaces=["main"])
        store.add_node(seed_node)
        store.add_node(neighbor)
        store.add_relation(GraphEdge(
            source_id="seed_pe", target_id="neighbor_pe", edge_type="MANAGES",
            properties={"piece_id": "will_fail"},
        ))

        seed = _make_seed(seed_node, score=0.7, source="search")

        result = graph_walk(store, piece_store, [seed], traversal_depth=1)

        depth_1 = [e for e in result if e["depth"] == 1]
        assert len(depth_1) == 1
        assert depth_1[0]["piece"] is None
        assert depth_1[0]["relation_type"] == "MANAGES"


# ── 9. Space filtering: matching spaces kept, non-matching filtered ──────────


class TestSpaceFiltering:
    """Neighbors with matching spaces kept, non-matching filtered out.

    Validates: Requirements 4.4
    """

    def test_matching_spaces_kept(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        seed_node = _make_node("seed_sf", spaces=["main", "work"])
        match_neighbor = _make_node("match", spaces=["main"])
        no_match_neighbor = _make_node("no_match", spaces=["personal"])
        overlap_neighbor = _make_node("overlap", spaces=["work", "personal"])

        store.add_node(seed_node)
        store.add_node(match_neighbor)
        store.add_node(no_match_neighbor)
        store.add_node(overlap_neighbor)

        store.add_relation(GraphEdge(source_id="seed_sf", target_id="match", edge_type="KNOWS"))
        store.add_relation(GraphEdge(source_id="seed_sf", target_id="no_match", edge_type="KNOWS"))
        store.add_relation(GraphEdge(source_id="seed_sf", target_id="overlap", edge_type="KNOWS"))

        seed = _make_seed(seed_node, score=0.9, source="search")

        result = graph_walk(
            store, piece_store, [seed],
            traversal_depth=1, spaces=["main", "work"],
        )

        depth_1_ids = {e["target_node_id"] for e in result if e["depth"] == 1}
        assert "match" in depth_1_ids
        assert "overlap" in depth_1_ids
        assert "no_match" not in depth_1_ids

    def test_no_spaces_filter_keeps_all(self):
        """When spaces=None, all neighbors are kept."""
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        seed_node = _make_node("seed_ns", spaces=[])
        neighbor = _make_node("neighbor_ns", spaces=["anything"])
        store.add_node(seed_node)
        store.add_node(neighbor)
        store.add_relation(GraphEdge(source_id="seed_ns", target_id="neighbor_ns", edge_type="KNOWS"))

        seed = _make_seed(seed_node, score=0.5, source="search")

        result = graph_walk(store, piece_store, [seed], traversal_depth=1, spaces=None)

        depth_1 = [e for e in result if e["depth"] == 1]
        assert len(depth_1) == 1


# ── 10. Space filtering with empty spaces on node ───────────────────────────


class TestSpaceFilteringEmptyNodeSpaces:
    """Node with no spaces filtered out when spaces filter is provided.

    Validates: Requirements 4.4
    """

    def test_empty_spaces_node_filtered_out(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        seed_node = _make_node("seed_es", spaces=["main"])
        empty_spaces_neighbor = _make_node("empty_spaces", spaces=[])
        has_spaces_neighbor = _make_node("has_spaces", spaces=["main"])

        store.add_node(seed_node)
        store.add_node(empty_spaces_neighbor)
        store.add_node(has_spaces_neighbor)

        store.add_relation(GraphEdge(source_id="seed_es", target_id="empty_spaces", edge_type="KNOWS"))
        store.add_relation(GraphEdge(source_id="seed_es", target_id="has_spaces", edge_type="KNOWS"))

        seed = _make_seed(seed_node, score=0.8, source="search")

        result = graph_walk(
            store, piece_store, [seed],
            traversal_depth=1, spaces=["main"],
        )

        depth_1_ids = {e["target_node_id"] for e in result if e["depth"] == 1}
        assert "has_spaces" in depth_1_ids
        assert "empty_spaces" not in depth_1_ids


# ── 11. Merge with empty lists ──────────────────────────────────────────────


class TestMergeEmptyLists:
    """merge_graph_contexts([], []) returns [].

    Validates: Requirements 5.1
    """

    def test_merge_both_empty(self):
        result = merge_graph_contexts([], [])
        assert result == []


# ── 12. Merge with one empty list ───────────────────────────────────────────


class TestMergeOneEmpty:
    """merge_graph_contexts(entries, []) returns entries.

    Validates: Requirements 5.1
    """

    def test_merge_identity_empty(self):
        entries = [
            {
                "relation_type": "SEARCH_HIT",
                "target_node_id": "n1",
                "target_label": "Node 1",
                "piece": None,
                "depth": 0,
                "score": 0.9,
            },
        ]
        result = merge_graph_contexts(entries, [])
        assert len(result) == 1
        assert result[0]["target_node_id"] == "n1"

    def test_merge_search_empty(self):
        entries = [
            {
                "relation_type": "IDENTITY",
                "target_node_id": "n2",
                "target_label": "Node 2",
                "piece": None,
                "depth": 0,
                "score": 1.0,
            },
        ]
        result = merge_graph_contexts([], entries)
        assert len(result) == 1
        assert result[0]["target_node_id"] == "n2"


# ── 13. Merge dedup: duplicate (node_id, relation_type) keeps higher score ──


class TestMergeDedup:
    """Duplicate (node_id, relation_type) keeps higher score.

    Validates: Requirements 5.1, 5.2
    """

    def test_merge_keeps_higher_score(self):
        search = [
            {
                "relation_type": "KNOWS",
                "target_node_id": "shared",
                "target_label": "Shared",
                "piece": None,
                "depth": 1,
                "score": 0.3,
            },
        ]
        identity = [
            {
                "relation_type": "KNOWS",
                "target_node_id": "shared",
                "target_label": "Shared",
                "piece": None,
                "depth": 1,
                "score": 0.5,
            },
        ]
        result = merge_graph_contexts(search, identity)

        assert len(result) == 1
        assert result[0]["score"] == 0.5


# ── 14. Merge tiebreaker: equal scores keeps shorter depth ──────────────────


class TestMergeTiebreaker:
    """Equal scores keeps shorter depth.

    Validates: Requirements 5.3
    """

    def test_merge_tiebreaker_shorter_depth(self):
        search = [
            {
                "relation_type": "WORKS_AT",
                "target_node_id": "company",
                "target_label": "Company",
                "piece": None,
                "depth": 2,
                "score": 0.5,
            },
        ]
        identity = [
            {
                "relation_type": "WORKS_AT",
                "target_node_id": "company",
                "target_label": "Company",
                "piece": None,
                "depth": 1,
                "score": 0.5,
            },
        ]
        result = merge_graph_contexts(search, identity)

        assert len(result) == 1
        assert result[0]["depth"] == 1


# ── 15. Merge allows same node with different relation_types ────────────────


class TestMergeDifferentRelationTypes:
    """Same node with different relation_types both kept.

    Validates: Requirements 5.4
    """

    def test_merge_same_node_different_relations(self):
        search = [
            {
                "relation_type": "SEARCH_HIT",
                "target_node_id": "multi",
                "target_label": "Multi",
                "piece": None,
                "depth": 0,
                "score": 0.8,
            },
        ]
        identity = [
            {
                "relation_type": "WORKS_AT",
                "target_node_id": "multi",
                "target_label": "Multi",
                "piece": None,
                "depth": 1,
                "score": 0.5,
            },
        ]
        result = merge_graph_contexts(search, identity)

        assert len(result) == 2
        relation_types = {e["relation_type"] for e in result}
        assert "SEARCH_HIT" in relation_types
        assert "WORKS_AT" in relation_types


# ── Helper function unit tests ──────────────────────────────────────────────


class TestNodePassesSpaceFilter:
    """Unit tests for _node_passes_space_filter helper."""

    def test_no_filter_passes(self):
        node = _make_node("n", spaces=["main"])
        assert _node_passes_space_filter(node, None) is True

    def test_empty_filter_passes(self):
        node = _make_node("n", spaces=["main"])
        assert _node_passes_space_filter(node, []) is True

    def test_matching_space_passes(self):
        node = _make_node("n", spaces=["main", "work"])
        assert _node_passes_space_filter(node, ["main"]) is True

    def test_no_matching_space_fails(self):
        node = _make_node("n", spaces=["personal"])
        assert _node_passes_space_filter(node, ["main"]) is False

    def test_empty_node_spaces_fails(self):
        node = _make_node("n", spaces=[])
        assert _node_passes_space_filter(node, ["main"]) is False


class TestShouldSkipPiece:
    """Unit tests for _should_skip_piece helper."""

    def test_no_already_retrieved(self):
        assert _should_skip_piece("p1", None, True) is False

    def test_piece_not_in_already_retrieved(self):
        assert _should_skip_piece("p1", {"p2": "context"}, True) is False

    def test_bool_true_skips(self):
        assert _should_skip_piece("p1", {"p1": "context"}, True) is True

    def test_bool_false_does_not_skip(self):
        assert _should_skip_piece("p1", {"p1": "context"}, False) is False

    def test_list_match_skips(self):
        assert _should_skip_piece("p1", {"p1": "context"}, ["context"]) is True

    def test_list_no_match_does_not_skip(self):
        assert _should_skip_piece("p1", {"p1": "context"}, ["user_profile"]) is False

    def test_tuple_match_skips(self):
        assert _should_skip_piece("p1", {"p1": "episodic"}, ("episodic",)) is True


# ── 16. Multi-edge to same neighbor ──────────────────────────────────────────


class TestMultiEdgeToSameNeighbor:
    """When a seed has multiple edges to the same neighbor (e.g., MEMBER_OF
    and SHOPS_AT), graph_walk should emit a separate context entry for each.

    This tests the fix for the break-at-first-edge bug.
    """

    def test_multi_edge_emits_all_relation_types(self):
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        # Add a piece for one of the edges
        piece = KnowledgePiece(
            piece_id="membership_piece",
            content="Safeway membership info",
            knowledge_type="factual",
            info_type="user_profile",
        )
        piece_store.add(piece)

        user_node = _make_node("user", spaces=["main"])
        store_node = _make_node("store", spaces=["main"])
        store.add_node(user_node)
        store.add_node(store_node)

        # Two edges to the same target
        store.add_relation(GraphEdge(
            source_id="user", target_id="store",
            edge_type="MEMBER_OF",
            properties={"piece_id": "membership_piece"},
        ))
        store.add_relation(GraphEdge(
            source_id="user", target_id="store",
            edge_type="SHOPS_AT",
            properties={},
        ))

        seed = _make_seed(user_node, score=1.0, source="identity")

        result = graph_walk(store, piece_store, [seed], traversal_depth=1)

        # Depth-0: IDENTITY entry for user
        depth_0 = [e for e in result if e["depth"] == 0]
        assert len(depth_0) == 1
        assert depth_0[0]["relation_type"] == "IDENTITY"

        # Depth-1: should have both MEMBER_OF and SHOPS_AT
        depth_1 = [e for e in result if e["depth"] == 1]
        store_rels = [
            e["relation_type"] for e in depth_1
            if e["target_node_id"] == "store"
        ]
        assert "MEMBER_OF" in store_rels
        assert "SHOPS_AT" in store_rels
        assert len(store_rels) == 2

        # The MEMBER_OF entry should have the piece
        member_entry = next(
            e for e in depth_1 if e["relation_type"] == "MEMBER_OF"
        )
        assert member_entry["piece"] is not None
        assert member_entry["piece"].piece_id == "membership_piece"

        # The SHOPS_AT entry should have no piece
        shops_entry = next(
            e for e in depth_1 if e["relation_type"] == "SHOPS_AT"
        )
        assert shops_entry["piece"] is None

    def test_single_edge_still_works(self):
        """Regression: single-edge case should still produce exactly one entry."""
        store = InMemoryEntityGraphStore()
        piece_store = InMemoryPieceStore()

        seed_node = _make_node("s", spaces=["main"])
        neighbor = _make_node("n", spaces=["main"])
        store.add_node(seed_node)
        store.add_node(neighbor)
        store.add_relation(GraphEdge(
            source_id="s", target_id="n", edge_type="KNOWS",
        ))

        seed = _make_seed(seed_node, score=0.8, source="search")

        result = graph_walk(store, piece_store, [seed], traversal_depth=1)

        depth_1 = [e for e in result if e["depth"] == 1]
        assert len(depth_1) == 1
        assert depth_1[0]["relation_type"] == "KNOWS"

"""
Property-based tests for KnowledgeBase graph search integration.

Feature: graph-semantic-retrieval
- Property 12: Depth-Based Scoring
- Property 13: Graph-Search Piece Dedup via Skip
- Property 14: Graph Context Merge Deduplicates by (target_node_id, relation_type)

**Validates: Requirements 7.4, 7.3, 7.1**
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

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
from hypothesis import given, settings, strategies as st, assume

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode, GraphEdge

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


# ── Hypothesis strategies ────────────────────────────────────────────────────

_identifier_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=30,
)

_node_type_strategy = st.sampled_from(["service", "person", "product", "location", "concept"])

_relation_type_strategy = st.sampled_from([
    "SEARCH_HIT", "RELATED", "WORKS_AT", "SELLS", "LOCATED_IN", "KNOWS",
])


@st.composite
def graph_node_strategy(draw):
    """Generate a random GraphNode with JSON-serializable properties."""
    node_id = draw(_identifier_text)
    node_type = draw(_node_type_strategy)
    label = draw(st.text(max_size=30))
    properties = draw(st.dictionaries(
        st.text(min_size=1, max_size=15),
        st.text(max_size=20),
        max_size=3,
    ))
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        properties=properties,
        is_active=True,
    )


@st.composite
def graph_context_entry_strategy(draw):
    """Generate a graph context entry dict."""
    return {
        "relation_type": draw(_relation_type_strategy),
        "target_node_id": draw(_identifier_text),
        "target_label": draw(st.text(max_size=20)),
        "piece": None,
        "depth": draw(st.integers(min_value=0, max_value=5)),
        "score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    }


# ── Shared test helpers (from conftest.py) ───────────────────────────────────
from conftest import InMemoryEntityGraphStore


# ── Helper: build a KnowledgeBase with mock stores ───────────────────────────


def _make_knowledge_base(
    graph_store: EntityGraphStore,
    piece_store: Optional[KnowledgePieceStore] = None,
    metadata_store: Optional[MetadataStore] = None,
    graph_traversal_depth: int = 2,
    graph_retrieval_ignore_pieces_already_retrieved=False,
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
        include_pieces=False,
        include_graph=True,
        graph_traversal_depth=graph_traversal_depth,
        graph_retrieval_ignore_pieces_already_retrieved=graph_retrieval_ignore_pieces_already_retrieved,
    )


# ── Property 12: Depth-Based Scoring ────────────────────────────────────────
# Feature: graph-semantic-retrieval, Property 12: Depth-Based Scoring
# For any graph search result node at traversal depth d with semantic search
# score s, the combined score equals s × 1/(d+1): yielding s × 1.0 at depth 0,
# s × 0.5 at depth 1, s × 0.33 at depth 2, etc.


class TestDepthBasedScoring:
    """Property 12: Depth-Based Scoring.

    **Validates: Requirements 7.4**
    """

    @given(
        search_score=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        num_hops=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_depth_based_scoring_formula(self, search_score: float, num_hops: int):
        """For a search-hit node with neighbors at various depths, the combined
        score for each entry equals search_score × 1/(depth+1)."""
        # Feature: graph-semantic-retrieval, Property 12: Depth-Based Scoring

        # Build a linear chain: root -> n1 -> n2 -> n3
        graph_store = InMemoryEntityGraphStore()
        root = GraphNode(node_id="root", node_type="service", label="Root")
        graph_store.add_node(root)

        prev_id = "root"
        for i in range(1, num_hops + 1):
            neighbor = GraphNode(
                node_id=f"n{i}", node_type="service", label=f"Node {i}"
            )
            graph_store.add_node(neighbor)
            graph_store.add_relation(GraphEdge(
                source_id=prev_id, target_id=f"n{i}", edge_type="RELATED"
            ))
            prev_id = f"n{i}"

        kb = _make_knowledge_base(graph_store, graph_traversal_depth=num_hops)

        # Use unified graph walk with search seeds
        search_results = [(root, search_score)]
        seeds = [SeedNode(node=node, score=score, source="search") for node, score in search_results]
        context = graph_walk(graph_store, kb.piece_store, seeds, traversal_depth=num_hops)

        # Verify depth-0 entry (the search hit itself)
        depth_0_entries = [e for e in context if e["depth"] == 0]
        assert len(depth_0_entries) == 1
        assert depth_0_entries[0]["score"] == pytest.approx(search_score, abs=1e-9)
        assert depth_0_entries[0]["relation_type"] == "SEARCH_HIT"

        # Verify entries at each depth
        for d in range(1, num_hops + 1):
            depth_d_entries = [e for e in context if e["depth"] == d]
            assert len(depth_d_entries) >= 1, f"Expected entry at depth {d}"
            expected_score = search_score * (1.0 / (d + 1))
            for entry in depth_d_entries:
                assert entry["score"] == pytest.approx(expected_score, abs=1e-9), (
                    f"At depth {d}: expected {expected_score}, got {entry['score']}"
                )

    @given(
        search_score=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_depth_scoring_is_strictly_monotonic_decreasing(self, search_score: float):
        """Each additional hop reduces the score — strictly monotonic decay."""
        # Feature: graph-semantic-retrieval, Property 12: Depth-Based Scoring

        # Build a chain of 3 hops
        graph_store = InMemoryEntityGraphStore()
        root = GraphNode(node_id="root", node_type="service", label="Root")
        graph_store.add_node(root)
        for i in range(1, 4):
            n = GraphNode(node_id=f"n{i}", node_type="service", label=f"N{i}")
            graph_store.add_node(n)
            prev = "root" if i == 1 else f"n{i-1}"
            graph_store.add_relation(GraphEdge(
                source_id=prev, target_id=f"n{i}", edge_type="RELATED"
            ))

        kb = _make_knowledge_base(graph_store, graph_traversal_depth=3)
        seeds = [SeedNode(node=root, score=search_score, source="search")]
        context = graph_walk(graph_store, kb.piece_store, seeds, traversal_depth=3)

        # Collect scores by depth
        scores_by_depth = {}
        for entry in context:
            d = entry["depth"]
            scores_by_depth[d] = entry["score"]

        # Verify strictly decreasing
        for d in range(len(scores_by_depth) - 1):
            assert scores_by_depth[d] > scores_by_depth[d + 1], (
                f"Score at depth {d} ({scores_by_depth[d]}) should be > "
                f"score at depth {d+1} ({scores_by_depth[d+1]})"
            )


# ── Property 13: Graph-Search Piece Dedup via Skip ──────────────────────────
# Feature: graph-semantic-retrieval, Property 13: Graph-Search Piece Dedup via Skip
# For any piece linked to a graph-search-found edge, if that piece's piece_id
# already appears in the Layer 2 results (tracked via already_retrieved_piece_ids)
# and graph_retrieval_ignore_pieces_already_retrieved is set, the piece shall NOT
# appear in the graph_context entries.


class TestGraphSearchPieceDedupViaSkip:
    """Property 13: Graph-Search Piece Dedup via Skip.

    **Validates: Requirements 7.3**
    """

    @given(
        piece_id=_identifier_text,
        info_type=st.sampled_from(["context", "user_profile", "instructions"]),
        search_score=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_piece_already_in_layer2_is_skipped_when_flag_set(
        self, piece_id: str, info_type: str, search_score: float
    ):
        """When graph_retrieval_ignore_pieces_already_retrieved is True and a
        piece_id is in already_retrieved_piece_ids, the piece is NOT attached
        to graph_context entries."""
        # Feature: graph-semantic-retrieval, Property 13: Graph-Search Piece Dedup via Skip

        # Build graph: root -> neighbor, edge has piece_id in properties
        graph_store = InMemoryEntityGraphStore()
        root = GraphNode(node_id="root", node_type="service", label="Root")
        neighbor = GraphNode(node_id="neighbor", node_type="product", label="Neighbor")
        graph_store.add_node(root)
        graph_store.add_node(neighbor)
        graph_store.add_relation(GraphEdge(
            source_id="root",
            target_id="neighbor",
            edge_type="SELLS",
            properties={"piece_id": piece_id},
        ))

        # Mock piece_store that would return a piece for this piece_id
        mock_piece = KnowledgePiece(content="test content", piece_id=piece_id, info_type=info_type)
        piece_store = MagicMock(spec=KnowledgePieceStore)
        piece_store.get_by_id.return_value = mock_piece
        piece_store.search.return_value = []
        piece_store.list_all.return_value = []

        kb = _make_knowledge_base(
            graph_store,
            piece_store=piece_store,
            graph_traversal_depth=1,
            graph_retrieval_ignore_pieces_already_retrieved=True,
        )

        # Simulate: piece_id already found by Layer 2
        already_retrieved = {piece_id: info_type}

        seeds = [SeedNode(node=root, score=search_score, source="search")]
        context = graph_walk(
            graph_store, piece_store, seeds,
            traversal_depth=1,
            already_retrieved_piece_ids=already_retrieved,
            ignore_already_retrieved=True,
        )

        # The piece should NOT appear in any graph_context entry
        for entry in context:
            assert entry.get("piece") is None, (
                f"Piece {piece_id} should have been skipped but was found in "
                f"graph_context entry for {entry['target_node_id']}"
            )

    @given(
        piece_id=_identifier_text,
        search_score=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_piece_not_in_layer2_is_included(
        self, piece_id: str, search_score: float
    ):
        """When a piece_id is NOT in already_retrieved_piece_ids, the piece
        IS attached to the graph_context entry (even when dedup flag is set)."""
        # Feature: graph-semantic-retrieval, Property 13: Graph-Search Piece Dedup via Skip

        # Build graph: root -> neighbor, edge has piece_id
        graph_store = InMemoryEntityGraphStore()
        root = GraphNode(node_id="root", node_type="service", label="Root")
        neighbor = GraphNode(node_id="neighbor", node_type="product", label="Neighbor")
        graph_store.add_node(root)
        graph_store.add_node(neighbor)
        graph_store.add_relation(GraphEdge(
            source_id="root",
            target_id="neighbor",
            edge_type="SELLS",
            properties={"piece_id": piece_id},
        ))

        # Mock piece_store returns a piece
        mock_piece = KnowledgePiece(content="test content", piece_id=piece_id)
        piece_store = MagicMock(spec=KnowledgePieceStore)
        piece_store.get_by_id.return_value = mock_piece
        piece_store.search.return_value = []
        piece_store.list_all.return_value = []

        kb = _make_knowledge_base(
            graph_store,
            piece_store=piece_store,
            graph_traversal_depth=1,
            graph_retrieval_ignore_pieces_already_retrieved=True,
        )

        # No pieces already retrieved — empty dict
        already_retrieved = {}

        seeds = [SeedNode(node=root, score=search_score, source="search")]
        context = graph_walk(
            graph_store, piece_store, seeds,
            traversal_depth=1,
            already_retrieved_piece_ids=already_retrieved,
            ignore_already_retrieved=True,
        )

        # The neighbor entry at depth 1 should have the piece attached
        neighbor_entries = [e for e in context if e["target_node_id"] == "neighbor"]
        assert len(neighbor_entries) == 1
        assert neighbor_entries[0]["piece"] is not None
        assert neighbor_entries[0]["piece"].piece_id == piece_id


# ── Property 14: Graph Context Merge Deduplicates by (target_node_id, relation_type) ──
# Feature: graph-semantic-retrieval, Property 14: Graph Context Merge Deduplicates
# by (target_node_id, relation_type)
# For any two lists of graph context entries, the merged result contains each
# (target_node_id, relation_type) pair at most once, keeping the entry with the
# higher score (or shorter depth if scores equal). Different relation_types to
# the same node are preserved.


class TestGraphContextMergeDeduplicates:
    """Property 14: Graph Context Merge Deduplicates by (target_node_id, relation_type).

    **Validates: Requirements 7.1**
    """

    @given(
        search_entries=st.lists(graph_context_entry_strategy(), min_size=0, max_size=8),
        identity_entries=st.lists(graph_context_entry_strategy(), min_size=0, max_size=8),
    )
    @settings(max_examples=100)
    def test_merged_has_unique_node_relation_pairs(
        self,
        search_entries: List[Dict[str, Any]],
        identity_entries: List[Dict[str, Any]],
    ):
        """The merged result contains each (target_node_id, relation_type) pair
        at most once."""
        # Feature: graph-semantic-retrieval, Property 14: Graph Context Merge
        # Deduplicates by (target_node_id, relation_type)

        merged = merge_graph_contexts(search_entries, identity_entries)

        # Check uniqueness of (target_node_id, relation_type) pairs
        seen_keys = set()
        for entry in merged:
            key = (entry["target_node_id"], entry.get("relation_type", "RELATED"))
            assert key not in seen_keys, (
                f"Duplicate key {key} found in merged graph context"
            )
            seen_keys.add(key)

    @given(
        search_entries=st.lists(graph_context_entry_strategy(), min_size=0, max_size=8),
        identity_entries=st.lists(graph_context_entry_strategy(), min_size=0, max_size=8),
    )
    @settings(max_examples=100)
    def test_merged_keeps_higher_score_entry(
        self,
        search_entries: List[Dict[str, Any]],
        identity_entries: List[Dict[str, Any]],
    ):
        """When the same (target_node_id, relation_type) appears in both lists,
        the entry with the higher score is kept."""
        # Feature: graph-semantic-retrieval, Property 14: Graph Context Merge
        # Deduplicates by (target_node_id, relation_type)

        merged = merge_graph_contexts(search_entries, identity_entries)

        # Build a map of all entries by key from both input lists
        all_entries_by_key: Dict[Tuple, List[Dict]] = {}
        for entry in search_entries + identity_entries:
            key = (entry["target_node_id"], entry.get("relation_type", "RELATED"))
            all_entries_by_key.setdefault(key, []).append(entry)

        # For each merged entry, verify it's the best one
        for entry in merged:
            key = (entry["target_node_id"], entry.get("relation_type", "RELATED"))
            candidates = all_entries_by_key.get(key, [])
            if len(candidates) > 1:
                merged_score = entry.get("score", 0)
                merged_depth = entry["depth"]
                for candidate in candidates:
                    cand_score = candidate.get("score", 0)
                    cand_depth = candidate["depth"]
                    # The merged entry should be at least as good as any candidate
                    assert (
                        merged_score > cand_score
                        or (merged_score == cand_score and merged_depth <= cand_depth)
                        or (merged_score == cand_score and merged_depth == cand_depth)
                    ), (
                        f"Merged entry for {key} has score={merged_score}, depth={merged_depth} "
                        f"but candidate has score={cand_score}, depth={cand_depth}"
                    )

    @given(
        node_id=_identifier_text,
        rel_type_a=_relation_type_strategy,
        rel_type_b=_relation_type_strategy,
        score_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        score_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_different_relation_types_to_same_node_preserved(
        self,
        node_id: str,
        rel_type_a: str,
        rel_type_b: str,
        score_a: float,
        score_b: float,
    ):
        """Different relation_types to the same node are preserved as separate entries."""
        # Feature: graph-semantic-retrieval, Property 14: Graph Context Merge
        # Deduplicates by (target_node_id, relation_type)
        assume(rel_type_a != rel_type_b)

        entry_a = {
            "relation_type": rel_type_a,
            "target_node_id": node_id,
            "target_label": "Node",
            "piece": None,
            "depth": 0,
            "score": score_a,
        }
        entry_b = {
            "relation_type": rel_type_b,
            "target_node_id": node_id,
            "target_label": "Node",
            "piece": None,
            "depth": 1,
            "score": score_b,
        }

        merged = merge_graph_contexts([entry_a], [entry_b])

        # Both entries should be preserved since they have different relation_types
        assert len(merged) == 2
        merged_keys = {(e["target_node_id"], e["relation_type"]) for e in merged}
        assert (node_id, rel_type_a) in merged_keys
        assert (node_id, rel_type_b) in merged_keys

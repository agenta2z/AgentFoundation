"""
Property-based tests for KnowledgeBase layer method decomposition.

Feature: retrieval-pipeline-refactor, Property 18: Layer Method Decomposition Equivalence

For any valid combination of (query, entity_id, top_k, include_global, domain,
secondary_domains, tags, min_results, spaces), calling retrieve() shall produce
a RetrievalResult identical to manually calling retrieve_metadata(),
retrieve_pieces(), retrieve_search_graph(), and retrieve_identity_graph()
with the same parameters and assembling the result.

**Validates: Requirements 13.5, 13.6**
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

_test_dir = str(Path(__file__).resolve().parent)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

import pytest
from hypothesis import given, settings, strategies as st, assume

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode, GraphEdge

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.graph_walk import merge_graph_contexts
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from agent_foundation.knowledge.retrieval.stores.metadata.base import MetadataStore
from conftest import InMemoryEntityGraphStore


# ── Strategies ───────────────────────────────────────────────────────────────

_space_strategy = st.sampled_from(["main", "personal", "developmental"])
_spaces_strategy = st.one_of(
    st.none(),
    st.lists(_space_strategy, min_size=1, max_size=3).map(
        lambda xs: list(dict.fromkeys(xs))
    ),
)

_query_strategy = st.one_of(
    st.just(""),
    st.just("   "),
    st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
)

_entity_id_strategy = st.one_of(
    st.none(),
    st.just("user:test"),
)

_top_k_strategy = st.one_of(
    st.none(),
    st.integers(min_value=1, max_value=10),
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_metadata(entity_id: str, spaces: List[str] = None) -> EntityMetadata:
    return EntityMetadata(
        entity_id=entity_id,
        entity_type="user",
        properties={"name": entity_id},
        spaces=spaces or ["main"],
    )


def _make_piece(piece_id: str, entity_id: str = None) -> KnowledgePiece:
    return KnowledgePiece(
        content=f"Content for {piece_id}",
        piece_id=piece_id,
        entity_id=entity_id,
        info_type="context",
    )


def _build_kb():
    """Build a KnowledgeBase with deterministic mock stores.

    Returns a KB with:
    - Metadata store that returns entity and global metadata
    - Piece store that returns a fixed piece for any non-empty search
    - Graph store with a user node and one neighbor
    """
    entity_meta = _make_metadata("user:test", spaces=["main"])
    global_meta = _make_metadata("global", spaces=["main"])

    metadata_store = MagicMock(spec=MetadataStore)
    metadata_store.get_metadata.side_effect = lambda eid: {
        "user:test": entity_meta,
        "global": global_meta,
    }.get(eid)

    piece = _make_piece("p1", entity_id="user:test")
    piece_store = MagicMock(spec=KnowledgePieceStore)
    piece_store.search.return_value = [(piece, 0.85)]
    piece_store.get_by_id.return_value = None
    piece_store.supports_space_filter = False

    graph_store = InMemoryEntityGraphStore()
    user_node = GraphNode(
        node_id="user:test", node_type="user", label="Test User",
        properties={"spaces": ["main"]},
    )
    neighbor = GraphNode(
        node_id="company:acme", node_type="company", label="Acme Corp",
        properties={"spaces": ["main"]},
    )
    graph_store.add_node(user_node)
    graph_store.add_node(neighbor)
    graph_store.add_relation(GraphEdge(
        source_id="user:test", target_id="company:acme", edge_type="WORKS_AT",
    ))

    kb = KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:test",
        include_metadata=True,
        include_pieces=True,
        include_graph=True,
        graph_traversal_depth=1,
    )
    return kb


# ── Property 18: Layer Method Decomposition Equivalence ──────────────────────
# Feature: retrieval-pipeline-refactor, Property 18: Layer Method Decomposition Equivalence


class TestLayerMethodDecompositionEquivalence:
    """Property 18: Layer Method Decomposition Equivalence.

    For any valid combination of parameters, calling retrieve() shall produce
    a RetrievalResult identical to manually calling the four layer methods
    with the same parameters and assembling the result.

    **Validates: Requirements 13.5, 13.6**
    """

    @given(
        query=_query_strategy,
        entity_id=_entity_id_strategy,
        top_k=_top_k_strategy,
        include_global=st.booleans(),
        spaces=_spaces_strategy,
    )
    @settings(max_examples=100)
    def test_layer_method_decomposition_equivalence(
        self,
        query: str,
        entity_id: Optional[str],
        top_k: Optional[int],
        include_global: bool,
        spaces: Optional[List[str]],
    ):
        """retrieve() == manual layer assembly for any valid parameter combination.

        **Validates: Requirements 13.5, 13.6**
        """
        kb = _build_kb()

        # Resolve defaults the same way retrieve() does
        resolved_entity_id = entity_id or kb.active_entity_id
        resolved_top_k = top_k if top_k is not None else kb.default_top_k

        # Call retrieve()
        result = kb.retrieve(
            query=query,
            entity_id=entity_id,
            top_k=top_k,
            include_global=include_global,
            spaces=spaces,
        )

        # Manually call layer methods with the same resolved parameters
        meta, global_m = kb.retrieve_metadata(
            resolved_entity_id, include_global, spaces
        )

        pieces = []
        if query and query.strip() and kb.include_pieces:
            pieces = kb.retrieve_pieces(
                query, resolved_entity_id, resolved_top_k, include_global,
                None, None, None, 1, spaces,
            )

        # Build dedup set
        already_retrieved_piece_ids = None
        if kb.graph_retrieval_ignore_pieces_already_retrieved and pieces:
            already_retrieved_piece_ids = {
                p.piece_id: p.info_type for p, _ in pieces
            }

        search_ctx = kb.retrieve_search_graph(
            query, resolved_top_k, spaces, already_retrieved_piece_ids
        )
        identity_ctx = kb.retrieve_identity_graph(
            resolved_entity_id, spaces, already_retrieved_piece_ids
        )

        # Assemble expected graph context
        expected_graph_context = []
        if search_ctx or identity_ctx:
            expected_graph_context = merge_graph_contexts(search_ctx, identity_ctx)

        # Compare results
        assert result.metadata is meta, (
            f"Metadata mismatch: retrieve()={result.metadata}, manual={meta}"
        )
        assert result.global_metadata is global_m, (
            f"Global metadata mismatch: retrieve()={result.global_metadata}, manual={global_m}"
        )
        assert result.pieces == pieces, (
            f"Pieces mismatch: retrieve() has {len(result.pieces)} pieces, "
            f"manual has {len(pieces)} pieces"
        )

        # Compare graph context by (target_node_id, relation_type) keys
        result_keys = {
            (e["target_node_id"], e["relation_type"])
            for e in (result.graph_context or [])
        }
        expected_keys = {
            (e["target_node_id"], e["relation_type"])
            for e in expected_graph_context
        }
        assert result_keys == expected_keys, (
            f"Graph context keys mismatch:\n"
            f"  retrieve(): {result_keys}\n"
            f"  manual: {expected_keys}"
        )

        # Compare scores for matching entries
        result_scores = {
            (e["target_node_id"], e["relation_type"]): e.get("score", 0)
            for e in (result.graph_context or [])
        }
        expected_scores = {
            (e["target_node_id"], e["relation_type"]): e.get("score", 0)
            for e in expected_graph_context
        }
        for key in result_keys:
            assert abs(result_scores[key] - expected_scores[key]) < 1e-9, (
                f"Score mismatch for {key}: "
                f"retrieve()={result_scores[key]}, manual={expected_scores[key]}"
            )

"""
Property-based tests for KnowledgeBase orchestrator.

Tests global/entity piece merge, sensitive content rejection,
callable interface, and bulk loading.

# Feature: agent-knowledge-base
# Properties 11, 15, 18, 19
"""
import json
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

import pytest
from hypothesis import given, settings, assume, strategies as st

from science_modeling_tools.knowledge.knowledge_base import KnowledgeBase
from science_modeling_tools.knowledge.formatter import RetrievalResult
from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)
from science_modeling_tools.knowledge.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from science_modeling_tools.knowledge.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from science_modeling_tools.knowledge.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)


# ── Shared strategies ────────────────────────────────────────────────────────

# Non-empty content that does NOT match sensitive patterns
_safe_content = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=80,
).filter(lambda s: s.strip())

# Simple alphanumeric identifiers safe for filenames and piece_ids
_safe_id = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)

# Entity ID strategy: type:name format
_entity_id_strategy = st.tuples(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
).map(lambda t: f"{t[0]}:{t[1]}")

# Knowledge type strategy
_knowledge_type_strategy = st.sampled_from(list(KnowledgeType))

# Tag strategy: pre-normalized lowercase alphanumeric
_normalized_tag = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=15,
)

# Non-empty query strings (at least one non-whitespace char)
_query_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=40,
).filter(lambda s: s.strip())


@st.composite
def _safe_piece_strategy(draw, entity_id=None, piece_id=None):
    """Generate a KnowledgePiece with safe content (no sensitive patterns).

    Optionally fix entity_id and/or piece_id.
    """
    content = draw(_safe_content)
    pid = piece_id if piece_id is not None else draw(_safe_id)
    knowledge_type = draw(_knowledge_type_strategy)
    tags = draw(st.lists(_normalized_tag, max_size=4))
    eid = entity_id if entity_id is not None else draw(
        st.one_of(st.none(), _entity_id_strategy)
    )

    return KnowledgePiece(
        content=content,
        piece_id=pid,
        knowledge_type=knowledge_type,
        tags=tags,
        entity_id=eid,
    )


def _make_kb(active_entity_id=None):
    """Create a KnowledgeBase with adapter-backed in-memory stores."""
    return KnowledgeBase(
        metadata_store=KeyValueMetadataStore(kv_service=MemoryKeyValueService()),
        piece_store=RetrievalKnowledgePieceStore(retrieval_service=MemoryRetrievalService()),
        graph_store=GraphServiceEntityGraphStore(graph_service=MemoryGraphService()),
        active_entity_id=active_entity_id,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Property 11: Global and entity piece merge
# ══════════════════════════════════════════════════════════════════════════════


class TestGlobalAndEntityPieceMerge:
    """Property 11: Global and entity piece merge.

    For any set of KnowledgePiece instances where some are global
    (entity_id=None) and some are entity-scoped, the KnowledgeBase.retrieve
    method with include_global=True SHALL return pieces from both the active
    entity and the global pool. Specifically, the merged result set (before
    top_k truncation) SHALL contain every piece that would appear in either
    the entity-only search or the global-only search.

    **Validates: Requirements 5.3**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_global_and_entity_merge(self, data):
        """Retrieve with include_global=True merges entity and global pieces.

        **Validates: Requirements 5.3**
        """
        entity_id = data.draw(_entity_id_strategy)

        # Generate 1-3 entity-scoped pieces with unique IDs
        n_entity = data.draw(st.integers(min_value=1, max_value=3))
        entity_pieces = []
        for i in range(n_entity):
            piece = data.draw(
                _safe_piece_strategy(entity_id=entity_id, piece_id=f"ent{i}")
            )
            entity_pieces.append(piece)

        # Generate 1-3 global pieces with unique IDs
        n_global = data.draw(st.integers(min_value=1, max_value=3))
        global_pieces = []
        for i in range(n_global):
            piece = data.draw(
                _safe_piece_strategy(entity_id=None, piece_id=f"glb{i}")
            )
            global_pieces.append(piece)

        # Build a query from content words of all pieces to ensure matches
        all_words = set()
        for p in entity_pieces + global_pieces:
            all_words.update(p.content.split())
        # Use a subset of words as query (at least 1)
        query_words = list(all_words)[:5]
        assume(len(query_words) > 0)
        query = " ".join(query_words)
        assume(query.strip())

        # Set up KB with large top_k to avoid truncation
        kb = _make_kb(active_entity_id=entity_id)
        kb.default_top_k = n_entity + n_global + 10

        # Add all pieces
        for p in entity_pieces + global_pieces:
            kb.add_piece(p)

        # Retrieve with include_global=True
        result = kb.retrieve(query, include_global=True)

        # Get entity-only results
        entity_only = kb.piece_store.search(
            query, entity_id=entity_id, top_k=kb.default_top_k
        )
        entity_only_ids = {p.piece_id for p, _ in entity_only}

        # Get global-only results
        global_only = kb.piece_store.search(
            query, entity_id=None, top_k=kb.default_top_k
        )
        global_only_ids = {p.piece_id for p, _ in global_only}

        # The merged result should contain every piece from either search
        merged_ids = {p.piece_id for p, _ in result.pieces}
        expected_ids = entity_only_ids | global_only_ids

        assert expected_ids.issubset(merged_ids), (
            f"Missing pieces in merge: {expected_ids - merged_ids}. "
            f"Entity-only: {entity_only_ids}, Global-only: {global_only_ids}, "
            f"Merged: {merged_ids}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Property 15: Sensitive content rejection
# ══════════════════════════════════════════════════════════════════════════════


# Strategy for generating strings that match sensitive patterns
_sensitive_prefix = st.sampled_from([
    "api_key=",
    "api-key=",
    "apikey=",
    "secret=",
    "secret:",
    "password=",
    "password:",
    "token=",
    "token:",
    "credential=",
    "credential:",
    "API_KEY =",
    "Secret =",
    "Password:",
])

_sensitive_bearer = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~+/",
    min_size=5,
    max_size=30,
).map(lambda s: f"Bearer {s}")


@st.composite
def _sensitive_content_strategy(draw):
    """Generate content that contains a sensitive pattern."""
    use_bearer = draw(st.booleans())
    if use_bearer:
        return draw(_sensitive_bearer)
    else:
        prefix = draw(_sensitive_prefix)
        value = draw(st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            min_size=3,
            max_size=20,
        ))
        # Optionally add surrounding text
        before = draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
            max_size=20,
        ))
        after = draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
            max_size=20,
        ))
        return f"{before} {prefix}{value} {after}".strip()


class TestSensitiveContentRejection:
    """Property 15: Sensitive content rejection.

    For any string matching common secret patterns (containing substrings
    like api_key=, password:, Bearer tokens), the KnowledgeBase SHALL reject
    the piece during add and raise a validation error.

    **Validates: Requirements 9.2**
    """

    @given(sensitive_content=_sensitive_content_strategy())
    @settings(max_examples=100)
    def test_sensitive_content_rejected(self, sensitive_content):
        """Adding a piece with sensitive content raises ValueError.

        **Validates: Requirements 9.2**
        """
        kb = _make_kb()

        piece = KnowledgePiece(
            content=sensitive_content,
            piece_id="sensitive-test",
        )

        with pytest.raises(ValueError, match="sensitive"):
            kb.add_piece(piece)


# ══════════════════════════════════════════════════════════════════════════════
# Property 18: Callable returns formatted string
# ══════════════════════════════════════════════════════════════════════════════


class TestCallableReturnsFormattedString:
    """Property 18: Callable returns formatted string.

    For any KnowledgeBase with pieces and any non-empty query string, calling
    the KnowledgeBase as a callable SHALL return a str value. If pieces match,
    the string SHALL be non-empty and contain content from at least one
    matching piece.

    **Validates: Requirements 5.6**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_callable_returns_formatted_string(self, data):
        """Calling KnowledgeBase as callable returns a str with matching content.

        **Validates: Requirements 5.6**
        """
        entity_id = data.draw(_entity_id_strategy)
        kb = _make_kb(active_entity_id=entity_id)

        # Generate 1-3 pieces with unique IDs
        n_pieces = data.draw(st.integers(min_value=1, max_value=3))
        pieces = []
        for i in range(n_pieces):
            piece = data.draw(
                _safe_piece_strategy(entity_id=entity_id, piece_id=f"call{i}")
            )
            pieces.append(piece)
            kb.add_piece(piece)

        # Build query from piece content words to ensure matches
        all_words = set()
        for p in pieces:
            all_words.update(p.content.split())
        query_words = list(all_words)[:5]
        assume(len(query_words) > 0)
        query = " ".join(query_words)
        assume(query.strip())

        # Call as callable
        result = kb(query)

        # Must return a string
        assert isinstance(result, str), (
            f"Expected str, got {type(result).__name__}"
        )

        # If pieces match, the result should be non-empty and contain
        # content from at least one piece
        retrieve_result = kb.retrieve(query)
        if retrieve_result.pieces:
            assert result != "", "Expected non-empty string when pieces match"
            # At least one piece's content should appear in the formatted output
            found_any = any(
                piece.content in result
                for piece, _ in retrieve_result.pieces
            )
            assert found_any, (
                f"Formatted output does not contain any matching piece content. "
                f"Output: {result!r}, "
                f"Piece contents: {[p.content for p, _ in retrieve_result.pieces]}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Property 19: Bulk load adds all valid items
# ══════════════════════════════════════════════════════════════════════════════


@st.composite
def _bulk_load_items_strategy(draw):
    """Generate a list of valid KnowledgePiece dicts for bulk loading.

    Each item has unique piece_id and safe content.
    """
    n = draw(st.integers(min_value=1, max_value=5))
    items = []
    for i in range(n):
        content = draw(_safe_content)
        knowledge_type = draw(_knowledge_type_strategy)
        tags = draw(st.lists(_normalized_tag, max_size=3))
        entity_id = draw(st.one_of(st.none(), _entity_id_strategy))

        items.append({
            "content": content,
            "piece_id": f"bulk{i}",
            "knowledge_type": knowledge_type.value,
            "tags": tags,
            "entity_id": entity_id,
        })
    return items


class TestBulkLoadAddsAllValidItems:
    """Property 19: Bulk load adds all valid items.

    For any JSON file containing N valid KnowledgePiece dictionaries,
    bulk_load SHALL add exactly N pieces to the store, and each SHALL be
    retrievable by its identifier.

    **Validates: Requirements 8.4**
    """

    @given(items=_bulk_load_items_strategy())
    @settings(max_examples=100)
    def test_bulk_load_adds_all_valid(self, items, tmp_path_factory):
        """Bulk loading N valid items adds exactly N pieces to the store.

        **Validates: Requirements 8.4**
        """
        tmp_path = tmp_path_factory.mktemp("bulk")
        kb = _make_kb()

        # Write items to a JSON file
        json_path = str(tmp_path / "bulk_items.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(items, f)

        # Bulk load
        count = kb.bulk_load(json_path)

        # Should have loaded all items
        assert count == len(items), (
            f"Expected {len(items)} items loaded, got {count}"
        )

        # Each item should be retrievable by its piece_id
        for item_dict in items:
            piece = kb.piece_store.get_by_id(item_dict["piece_id"])
            assert piece is not None, (
                f"Piece {item_dict['piece_id']!r} not found after bulk load"
            )
            assert piece.content == item_dict["content"], (
                f"Content mismatch for {item_dict['piece_id']!r}: "
                f"expected {item_dict['content']!r}, got {piece.content!r}"
            )

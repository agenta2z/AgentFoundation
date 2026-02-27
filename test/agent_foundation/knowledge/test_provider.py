"""
Unit tests for KnowledgeProvider.

Tests __call__ return type and keys, _group_by_info_type routing for metadata,
pieces, and graph edges, custom info_type grouping, and _resolve_formatter
for callable and inline string specs.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
_spu_src = Path(__file__).resolve().parents[3] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest

from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)
from agent_foundation.knowledge.retrieval.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.provider import KnowledgeProvider, InfoType
from agent_foundation.knowledge.retrieval.formatter import RetrievalResult
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_kb(active_entity_id: str = "user:test") -> KnowledgeBase:
    """Create a KnowledgeBase with in-memory stores."""
    metadata_store = KeyValueMetadataStore(MemoryKeyValueService())
    piece_store = RetrievalKnowledgePieceStore(MemoryRetrievalService())
    graph_store = GraphServiceEntityGraphStore(MemoryGraphService())
    return KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id=active_entity_id,
    )


def _create_populated_provider() -> KnowledgeProvider:
    """Create a KnowledgeProvider with populated data for testing.

    Populates:
    - User metadata (FirstName, Location)
    - Two pieces: one with info_type="user_profile", one with info_type="instructions"
    - Graph nodes and edges linking user to a store
    """
    kb = _create_kb(active_entity_id="user:alice")

    # Add user metadata
    user_meta = EntityMetadata(
        entity_id="user:alice",
        entity_type="user",
        properties={"FirstName": "Alice", "Location": "Seattle"},
    )
    kb.metadata_store.save_metadata(user_meta)

    # Add knowledge pieces with different info_types
    profile_piece = KnowledgePiece(
        content="Alice is a Safeway member with free delivery.",
        piece_id="safeway-membership",
        knowledge_type=KnowledgeType.Fact,
        info_type="user_profile",
        tags=["grocery", "safeway"],
        entity_id="user:alice",
        embedding_text="Safeway grocery store membership delivery",
    )
    kb.add_piece(profile_piece)

    instruction_piece = KnowledgePiece(
        content="Grocery Procedure: 1) Log in. 2) Add items. 3) Checkout.",
        piece_id="grocery-procedure",
        knowledge_type=KnowledgeType.Procedure,
        info_type="instructions",
        tags=["grocery", "procedure"],
        entity_id=None,
        embedding_text="grocery shopping steps procedure checkout",
    )
    kb.add_piece(instruction_piece)

    # Add graph nodes
    user_node = GraphNode(node_id="user:alice", node_type="user", label="Alice")
    store_node = GraphNode(node_id="store:safeway", node_type="store", label="Safeway")
    kb.graph_store.add_node(user_node)
    kb.graph_store.add_node(store_node)

    # Add graph edge with linked piece
    edge = GraphEdge(
        source_id="user:alice",
        target_id="store:safeway",
        edge_type="SHOPS_AT",
        properties={"piece_id": "safeway-membership"},
    )
    kb.graph_store.add_relation(edge)

    return KnowledgeProvider(kb)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def provider():
    """Create a populated KnowledgeProvider for testing."""
    return _create_populated_provider()


@pytest.fixture
def empty_provider():
    """Create a KnowledgeProvider with an empty KnowledgeBase."""
    kb = _create_kb()
    return KnowledgeProvider(kb)


# ── Test: __call__ returns Dict[str, str] ────────────────────────────────────


class TestCallReturnType:
    """Test KnowledgeProvider.__call__ return type.

    Validates: Requirements 2.1, 2.5
    """

    def test_call_returns_dict_str_str(self, provider):
        """Calling provider(query) returns a Dict[str, str] with all str values."""
        result = provider("grocery shopping")

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str), f"Key {key!r} is not a str"
            assert isinstance(value, str), f"Value for key {key!r} is not a str"

    def test_call_on_empty_provider_returns_dict(self, empty_provider):
        """Calling an empty provider returns an empty dict (or dict with empty values)."""
        result = empty_provider("anything")

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestCallReturnsExpectedKeys:
    """Test KnowledgeProvider.__call__ returns expected info_type keys.

    Validates: Requirements 2.2, 2.5
    """

    def test_call_returns_expected_keys(self, provider):
        """Provider populated with user_profile and instructions pieces returns those keys."""
        result = provider("grocery safeway shopping")

        assert "user_profile" in result
        assert "instructions" in result

    def test_user_profile_contains_metadata(self, provider):
        """The user_profile value contains metadata properties."""
        result = provider("grocery safeway")

        assert "Alice" in result.get("user_profile", "")

    def test_instructions_contains_procedure(self, provider):
        """The instructions value contains the procedure content."""
        result = provider("grocery shopping procedure checkout")

        assert "instructions" in result
        assert "Procedure" in result["instructions"] or "procedure" in result["instructions"].lower()


# ── Test: _group_by_info_type routes metadata ────────────────────────────────


class TestGroupByInfoTypeMetadata:
    """Test _group_by_info_type routes metadata to metadata_info_type.

    Validates: Requirement 2.3
    """

    def test_metadata_routes_to_user_profile(self, provider):
        """Metadata is routed to the metadata_info_type group (default 'user_profile')."""
        metadata = EntityMetadata(
            entity_id="user:alice",
            entity_type="user",
            properties={"FirstName": "Alice", "Location": "Seattle"},
        )
        result = RetrievalResult(metadata=metadata)

        groups = provider._group_by_info_type(result)

        assert "user_profile" in groups
        assert groups["user_profile"]["metadata"] is not None
        assert groups["user_profile"]["metadata"].properties["FirstName"] == "Alice"

    def test_metadata_routes_to_custom_info_type(self):
        """Metadata routes to a custom metadata_info_type when configured."""
        kb = _create_kb()
        custom_provider = KnowledgeProvider(kb, metadata_info_type="profile_data")

        metadata = EntityMetadata(
            entity_id="user:test",
            entity_type="user",
            properties={"Name": "Test"},
        )
        result = RetrievalResult(metadata=metadata)

        groups = custom_provider._group_by_info_type(result)

        assert "profile_data" in groups
        assert groups["profile_data"]["metadata"] is not None

    def test_empty_metadata_not_routed(self, provider):
        """Metadata with no properties is not routed to any group."""
        metadata = EntityMetadata(
            entity_id="user:alice",
            entity_type="user",
            properties={},
        )
        result = RetrievalResult(metadata=metadata)

        groups = provider._group_by_info_type(result)

        # Empty metadata should not create a group
        assert "user_profile" not in groups or groups.get("user_profile", {}).get("metadata") is None


# ── Test: _group_by_info_type routes pieces ──────────────────────────────────


class TestGroupByInfoTypePieces:
    """Test _group_by_info_type routes pieces by their info_type.

    Validates: Requirement 2.2
    """

    def test_pieces_route_by_info_type(self, provider):
        """Pieces are grouped by their info_type string."""
        piece_profile = KnowledgePiece(
            content="User is a member.",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            info_type="user_profile",
        )
        piece_instructions = KnowledgePiece(
            content="Step 1: Log in.",
            piece_id="p2",
            knowledge_type=KnowledgeType.Instruction,
            info_type="instructions",
        )
        result = RetrievalResult(pieces=[(piece_profile, 0.9), (piece_instructions, 0.8)])

        groups = provider._group_by_info_type(result)

        assert "user_profile" in groups
        assert "instructions" in groups
        assert len(groups["user_profile"]["pieces"]) == 1
        assert len(groups["instructions"]["pieces"]) == 1
        assert groups["user_profile"]["pieces"][0][0].piece_id == "p1"
        assert groups["instructions"]["pieces"][0][0].piece_id == "p2"

    def test_pieces_with_same_info_type_grouped_together(self, provider):
        """Multiple pieces with the same info_type are grouped together."""
        piece1 = KnowledgePiece(
            content="Fact one.", piece_id="f1", info_type="user_profile"
        )
        piece2 = KnowledgePiece(
            content="Fact two.", piece_id="f2", info_type="user_profile"
        )
        result = RetrievalResult(pieces=[(piece1, 0.9), (piece2, 0.8)])

        groups = provider._group_by_info_type(result)

        assert len(groups["user_profile"]["pieces"]) == 2

    def test_piece_with_no_info_type_defaults_to_context(self, provider):
        """A piece with info_type=None defaults to 'context' group."""
        piece = KnowledgePiece(
            content="Some context info.", piece_id="ctx1"
        )
        piece.info_type = None
        result = RetrievalResult(pieces=[(piece, 0.5)])

        groups = provider._group_by_info_type(result)

        assert "context" in groups
        assert len(groups["context"]["pieces"]) == 1


# ── Test: _group_by_info_type routes graph edges ─────────────────────────────


class TestGroupByInfoTypeGraphEdges:
    """Test _group_by_info_type routes graph edges correctly.

    Validates: Requirement 2.4
    """

    def test_graph_edge_with_linked_piece_routes_by_piece_info_type(self, provider):
        """Graph edge with a linked piece routes to the piece's info_type."""
        linked_piece = KnowledgePiece(
            content="Safeway membership info.",
            piece_id="linked-1",
            info_type="user_profile",
        )
        edge = {
            "relation_type": "SHOPS_AT",
            "target_node_id": "store:safeway",
            "target_label": "Safeway",
            "piece": linked_piece,
            "depth": 1,
        }
        result = RetrievalResult(graph_context=[edge])

        groups = provider._group_by_info_type(result)

        assert "user_profile" in groups
        assert len(groups["user_profile"]["graph_context"]) == 1
        assert groups["user_profile"]["graph_context"][0]["relation_type"] == "SHOPS_AT"

    def test_graph_edge_with_linked_instructions_piece(self, provider):
        """Graph edge with a linked instructions piece routes to 'instructions'."""
        linked_piece = KnowledgePiece(
            content="Shopping procedure.",
            piece_id="linked-2",
            info_type="instructions",
        )
        edge = {
            "relation_type": "HAS_SKILL",
            "target_node_id": "procedure:shopping",
            "target_label": "Shopping",
            "piece": linked_piece,
            "depth": 1,
        }
        result = RetrievalResult(graph_context=[edge])

        groups = provider._group_by_info_type(result)

        assert "instructions" in groups
        assert len(groups["instructions"]["graph_context"]) == 1

    def test_graph_edge_depth1_no_piece_routes_to_user_profile(self, provider):
        """Depth-1 graph edge without linked piece routes to 'user_profile'."""
        edge = {
            "relation_type": "KNOWS",
            "target_node_id": "person:bob",
            "target_label": "Bob",
            "piece": None,
            "depth": 1,
        }
        result = RetrievalResult(graph_context=[edge])

        groups = provider._group_by_info_type(result)

        assert "user_profile" in groups
        assert len(groups["user_profile"]["graph_context"]) == 1

    def test_graph_edge_deeper_no_piece_routes_to_context(self, provider):
        """Deeper graph edge (depth > 1) without linked piece routes to 'context'."""
        edge = {
            "relation_type": "RELATED",
            "target_node_id": "entity:far",
            "target_label": "Far Entity",
            "piece": None,
            "depth": 2,
        }
        result = RetrievalResult(graph_context=[edge])

        groups = provider._group_by_info_type(result)

        assert "context" in groups
        assert len(groups["context"]["graph_context"]) == 1


# ── Test: Custom info_type groups under that key ─────────────────────────────


class TestCustomInfoType:
    """Test that custom info_type strings group under their own key.

    Validates: Requirements 2.1, 2.2
    """

    def test_custom_info_type_groups_under_key(self):
        """A piece with info_type='safety' appears under the 'safety' key."""
        kb = _create_kb(active_entity_id="user:alice")

        safety_piece = KnowledgePiece(
            content="Always wear safety goggles in the lab.",
            piece_id="safety-1",
            knowledge_type=KnowledgeType.Instruction,
            info_type="safety",
            tags=["safety", "lab"],
            entity_id=None,
            embedding_text="safety goggles lab protection",
        )
        kb.add_piece(safety_piece)

        provider = KnowledgeProvider(kb)
        result = provider("safety goggles lab")

        assert "safety" in result
        assert "safety goggles" in result["safety"].lower() or "goggles" in result["safety"]

    def test_custom_info_type_via_group_by(self):
        """_group_by_info_type routes a piece with custom info_type correctly."""
        kb = _create_kb()
        provider = KnowledgeProvider(kb)

        piece = KnowledgePiece(
            content="Custom domain info.",
            piece_id="custom-1",
            info_type="my_custom_type",
        )
        result = RetrievalResult(pieces=[(piece, 0.7)])

        groups = provider._group_by_info_type(result)

        assert "my_custom_type" in groups
        assert len(groups["my_custom_type"]["pieces"]) == 1


# ── Test: _resolve_formatter ─────────────────────────────────────────────────


class TestResolveFormatter:
    """Test _resolve_formatter handles callable and inline string.

    Validates: Requirement 2.5
    """

    def test_resolve_formatter_callable(self):
        """A callable formatter is returned as-is."""
        kb = _create_kb()
        provider = KnowledgeProvider(kb)

        def my_formatter(metadata, pieces, graph_context):
            return "custom output"

        resolved = provider._resolve_formatter(my_formatter)

        assert resolved is my_formatter
        assert resolved(None, [], []) == "custom output"

    def test_resolve_formatter_inline_string(self):
        """An inline format string is resolved to a callable that formats correctly."""
        kb = _create_kb()
        provider = KnowledgeProvider(kb)

        format_str = "Meta: {metadata}, Pieces: {pieces}, Graph: {graph_context}"
        resolved = provider._resolve_formatter(format_str)

        assert callable(resolved)
        output = resolved("test_meta", ["p1", "p2"], ["g1"])
        assert "test_meta" in output
        assert "p1" in output

    def test_resolve_formatter_callable_used_in_call(self):
        """A callable formatter configured per info_type is used during __call__."""
        kb = _create_kb(active_entity_id="user:alice")

        piece = KnowledgePiece(
            content="Important fact.",
            piece_id="fmt-test",
            knowledge_type=KnowledgeType.Fact,
            info_type="user_profile",
            entity_id="user:alice",
            embedding_text="important fact test",
        )
        kb.add_piece(piece)

        def custom_fmt(metadata, pieces, graph_context):
            piece_texts = [p.content for p, _s in pieces]
            return "CUSTOM: " + "; ".join(piece_texts)

        provider = KnowledgeProvider(
            kb,
            formatters={"user_profile": custom_fmt},
        )
        result = provider("important fact")

        assert "user_profile" in result
        assert result["user_profile"].startswith("CUSTOM: ")

    def test_resolve_formatter_inline_string_used_in_call(self):
        """An inline format string configured as default_formatter is used during __call__."""
        kb = _create_kb(active_entity_id="user:alice")

        meta = EntityMetadata(
            entity_id="user:alice",
            entity_type="user",
            properties={"Name": "Alice"},
        )
        kb.metadata_store.save_metadata(meta)

        provider = KnowledgeProvider(
            kb,
            default_formatter="Profile: {metadata} | Items: {pieces} | Relations: {graph_context}",
        )
        result = provider("anything")

        # The user_profile key should exist (from metadata routing)
        assert "user_profile" in result
        assert "Profile:" in result["user_profile"]

    def test_resolve_formatter_rejects_non_str_non_callable(self):
        """_resolve_formatter raises ValueError for non-str, non-callable input."""
        kb = _create_kb()
        provider = KnowledgeProvider(kb)

        with pytest.raises(ValueError, match="callable or str"):
            provider._resolve_formatter(12345)

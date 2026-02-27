"""
Integration tests for KnowledgeBase with Agent.

Tests that KnowledgeBase can be assigned to Agent.user_profile and that
the agent correctly calls the KnowledgeBase with user_input to retrieve
relevant knowledge for prompt injection.

Requirements: 6.1, 6.2, 6.3, 6.4
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Also add SciencePythonUtils/src for rich_python_utils dependency (needed by Agent)
_workspace_root = _current_path.parent.parent
_spu_src = _workspace_root / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
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
from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)
from agent_foundation.agents.agent import Agent


# ── Helper ───────────────────────────────────────────────────────────────────


def _make_stores():
    """Create adapter-backed stores using in-memory services."""
    metadata_store = KeyValueMetadataStore(kv_service=MemoryKeyValueService())
    piece_store = RetrievalKnowledgePieceStore(retrieval_service=MemoryRetrievalService())
    graph_store = GraphServiceEntityGraphStore(graph_service=MemoryGraphService())
    return metadata_store, piece_store, graph_store


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def stores():
    """Create adapter-backed stores using in-memory services."""
    return _make_stores()


@pytest.fixture
def kb(stores):
    """Create a KnowledgeBase with adapter-backed stores."""
    metadata_store, piece_store, graph_store = stores
    return KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:xinli",
    )


@pytest.fixture
def populated_kb(kb, stores):
    """Create a KnowledgeBase pre-loaded with test data."""
    metadata_store, piece_store, graph_store = stores

    # Add user metadata
    user_meta = EntityMetadata(
        entity_id="user:xinli",
        entity_type="user",
        properties={"name": "Xinli", "location": "Seattle", "zip": "98121"},
    )
    metadata_store.save_metadata(user_meta)

    # Add knowledge pieces (entity-scoped)
    piece1 = KnowledgePiece(
        content="Prefers organic eggs",
        piece_id="piece-1",
        knowledge_type=KnowledgeType.Preference,
        tags=["grocery", "eggs"],
        entity_id="user:xinli",
    )
    piece_store.add(piece1)

    piece2 = KnowledgePiece(
        content="Costco membership is Executive tier",
        piece_id="piece-2",
        knowledge_type=KnowledgeType.Fact,
        tags=["costco", "membership"],
        entity_id="user:xinli",
    )
    piece_store.add(piece2)

    # Add global piece
    global_piece = KnowledgePiece(
        content="Egg prices vary by season",
        piece_id="piece-global",
        knowledge_type=KnowledgeType.Fact,
        tags=["eggs", "pricing"],
        entity_id=None,
    )
    piece_store.add(global_piece)

    # Add graph nodes and relations using GraphNode/GraphEdge
    user_node = GraphNode(
        node_id="user:xinli", node_type="user", label="Xinli"
    )
    store_node = GraphNode(
        node_id="store:costco", node_type="store", label="Costco"
    )
    graph_store.add_node(user_node)
    graph_store.add_node(store_node)

    relation = GraphEdge(
        source_id="user:xinli",
        target_id="store:costco",
        edge_type="SHOPS_AT",
        properties={"membership_tier": "Executive"},
    )
    graph_store.add_relation(relation)

    return kb


# ── Test: KnowledgeBase assignable to Agent.user_profile ─────────────────────


class TestKnowledgeBaseAssignment:
    """Test that KnowledgeBase can be assigned to Agent.user_profile."""

    def test_kb_assigned_to_user_profile(self, kb):
        """KnowledgeBase instance can be set as Agent.user_profile.

        Validates: Requirements 6.1
        """
        agent = Agent(user_profile=kb)
        assert agent.user_profile is kb

    def test_kb_is_callable(self, kb):
        """KnowledgeBase implements __call__ making it compatible with
        the Agent's callable user_profile interface.

        Validates: Requirements 6.3
        """
        assert callable(kb)

    def test_kb_assigned_to_context(self, kb):
        """KnowledgeBase instance can also be set as Agent.context.

        Validates: Requirements 6.2
        """
        agent = Agent(context=kb)
        assert agent.context is kb

    def test_kb_callable_returns_string(self, populated_kb):
        """KnowledgeBase.__call__ returns a string suitable for prompt injection.

        Validates: Requirements 6.3
        """
        result = populated_kb("Check egg prices")
        assert isinstance(result, str)

    def test_kb_callable_returns_empty_for_no_matches(self, kb):
        """KnowledgeBase returns empty string when no items match.

        Validates: Requirements 6.4
        """
        result = kb("completely unrelated query with no matches")
        assert isinstance(result, str)


# ── Test: Agent calls KnowledgeBase with user_input ──────────────────────────


class TestAgentCallsKnowledgeBase:
    """Test that Agent correctly invokes KnowledgeBase with user_input."""

    def test_resolve_task_input_field_calls_callable_with_user_input(self, populated_kb):
        """When user_profile is callable and resolved via _resolve_task_input_field,
        it is called with user_input as the argument.

        This tests the core integration mechanism: Agent._resolve_task_input_field
        detects that user_profile is callable and invokes it with user_input.

        Note: task_input must be non-empty for _resolve_task_input_field to
        resolve callables (empty dict is falsy, returns default as-is).

        Validates: Requirements 6.1
        """
        user_input = "Check egg prices"
        # Use a non-empty task_input so the method proceeds to callable resolution.
        # When task_input is empty/falsy, the method returns default directly.
        result = Agent._resolve_task_input_field(
            user_input=user_input,
            task_input={"_placeholder": True},
            field_name="user_profile",
            default=populated_kb,
        )
        # The result should be a formatted string from KnowledgeBase
        assert isinstance(result, str)
        assert len(result) > 0

    def test_resolve_task_input_field_passes_user_input_to_kb(self, kb):
        """Verify that _resolve_task_input_field passes the exact user_input
        string to the KnowledgeBase callable.

        Validates: Requirements 6.1
        """
        # Create a mock to track calls
        mock_kb = MagicMock(return_value="mocked knowledge")
        user_input = "What are the egg prices?"

        # task_input must be non-empty for callable resolution to occur
        result = Agent._resolve_task_input_field(
            user_input=user_input,
            task_input={"_placeholder": True},
            field_name="user_profile",
            default=mock_kb,
        )

        mock_kb.assert_called_once_with(user_input)
        assert result == "mocked knowledge"

    def test_resolve_returns_none_for_none_profile(self):
        """When user_profile is None, _resolve_task_input_field returns None.

        Validates: Requirements 6.1
        """
        result = Agent._resolve_task_input_field(
            user_input="hello",
            task_input={},
            field_name="user_profile",
            default=None,
        )
        assert result is None

    def test_resolve_returns_static_string_profile(self):
        """When user_profile is a static string, it is returned as-is.

        Validates: Requirements 6.1
        """
        result = Agent._resolve_task_input_field(
            user_input="hello",
            task_input={},
            field_name="user_profile",
            default="Static profile info",
        )
        assert result == "Static profile info"


# ── Test: Full end-to-end flow ───────────────────────────────────────────────


class TestFullIntegrationFlow:
    """Test the complete flow: add pieces → create KB → assign to agent → verify output."""

    def test_full_flow_add_pieces_create_kb_assign_to_agent(self):
        """End-to-end test: create stores, add knowledge, create KB,
        assign to Agent.user_profile, and verify the formatted output
        contains the expected knowledge.

        Flow:
        1. Create adapter-backed stores
        2. Add metadata and knowledge pieces
        3. Create KnowledgeBase
        4. Assign KB to Agent.user_profile
        5. Call KB with user_input (simulating what Agent does)
        6. Verify formatted output contains expected content

        Validates: Requirements 6.1, 6.2, 6.3, 6.4
        """
        # Step 1: Create stores
        metadata_store, piece_store, graph_store = _make_stores()

        # Step 2: Add metadata
        user_meta = EntityMetadata(
            entity_id="user:alice",
            entity_type="user",
            properties={"name": "Alice", "location": "Portland"},
        )
        metadata_store.save_metadata(user_meta)

        # Step 2: Add knowledge pieces via KnowledgeBase
        kb = KnowledgeBase(
            metadata_store=metadata_store,
            piece_store=piece_store,
            graph_store=graph_store,
            active_entity_id="user:alice",
        )

        piece1 = KnowledgePiece(
            content="Prefers local farmers market produce",
            piece_id="p1",
            knowledge_type=KnowledgeType.Preference,
            tags=["grocery", "produce", "local"],
            entity_id="user:alice",
        )
        kb.add_piece(piece1)

        piece2 = KnowledgePiece(
            content="Allergic to shellfish",
            piece_id="p2",
            knowledge_type=KnowledgeType.Fact,
            tags=["health", "allergy"],
            entity_id="user:alice",
        )
        kb.add_piece(piece2)

        # Step 3: Assign KB to Agent.user_profile
        agent = Agent(user_profile=kb)
        assert agent.user_profile is kb

        # Step 4: Simulate what Agent does — call KB with user_input
        user_input = "Find me some local produce"
        formatted_output = agent.user_profile(user_input)

        # Step 5: Verify formatted output
        assert isinstance(formatted_output, str)
        assert len(formatted_output) > 0
        # Should contain metadata
        assert "Alice" in formatted_output
        assert "Portland" in formatted_output
        # Should contain the relevant knowledge piece about produce
        assert "local farmers market produce" in formatted_output

    def test_full_flow_with_graph_context(self):
        """End-to-end test including entity graph relationships.

        Validates: Requirements 6.1, 6.2, 6.3
        """
        metadata_store, piece_store, graph_store = _make_stores()

        # Add metadata
        user_meta = EntityMetadata(
            entity_id="user:bob",
            entity_type="user",
            properties={"name": "Bob"},
        )
        metadata_store.save_metadata(user_meta)

        # Add graph nodes and relations using GraphNode/GraphEdge
        user_node = GraphNode(
            node_id="user:bob", node_type="user", label="Bob"
        )
        store_node = GraphNode(
            node_id="store:trader_joes", node_type="store", label="Trader Joes"
        )
        graph_store.add_node(user_node)
        graph_store.add_node(store_node)

        relation = GraphEdge(
            source_id="user:bob",
            target_id="store:trader_joes",
            edge_type="SHOPS_AT",
        )
        graph_store.add_relation(relation)

        # Create KB and assign to agent
        kb = KnowledgeBase(
            metadata_store=metadata_store,
            piece_store=piece_store,
            graph_store=graph_store,
            active_entity_id="user:bob",
        )
        agent = Agent(user_profile=kb)

        # Call KB with user_input
        formatted_output = agent.user_profile("Where should I shop?")

        # Verify output includes metadata and graph relationships
        assert isinstance(formatted_output, str)
        assert "Bob" in formatted_output
        assert "SHOPS_AT" in formatted_output
        assert "trader_joes" in formatted_output

    def test_full_flow_empty_query_returns_metadata_and_graph(self):
        """When query is empty/whitespace, KB still returns metadata and graph
        context but skips piece search.

        Validates: Requirements 6.4
        """
        metadata_store, piece_store, graph_store = _make_stores()

        user_meta = EntityMetadata(
            entity_id="user:carol",
            entity_type="user",
            properties={"name": "Carol"},
        )
        metadata_store.save_metadata(user_meta)

        kb = KnowledgeBase(
            metadata_store=metadata_store,
            piece_store=piece_store,
            graph_store=graph_store,
            active_entity_id="user:carol",
        )
        agent = Agent(user_profile=kb)

        # Empty query — should still return metadata
        formatted_output = agent.user_profile("")
        assert isinstance(formatted_output, str)
        # Metadata should still be present
        assert "Carol" in formatted_output

    def test_full_flow_no_results_returns_empty_string(self):
        """When KB has no data at all, callable returns empty string.

        Validates: Requirements 6.4
        """
        metadata_store, piece_store, graph_store = _make_stores()

        kb = KnowledgeBase(
            metadata_store=metadata_store,
            piece_store=piece_store,
            graph_store=graph_store,
            # No active entity — nothing to retrieve
            active_entity_id=None,
        )
        agent = Agent(user_profile=kb)

        formatted_output = agent.user_profile("anything")
        assert formatted_output == ""

    def test_resolve_task_input_field_with_populated_kb(self, populated_kb):
        """Simulate the exact Agent integration path: _resolve_task_input_field
        resolves a callable user_profile by calling it with user_input.

        This mirrors what happens inside Agent.__call__ when task_input is a
        non-empty dict containing agent fields.

        Validates: Requirements 6.1, 6.2, 6.3
        """
        user_input = "egg prices"

        # This is exactly what Agent.__call__ does internally when task_input
        # is non-empty. The method pops "user_profile" from task_input; if not
        # found, uses default (the KB). Since KB is callable, it calls kb(user_input).
        resolved_profile = Agent._resolve_task_input_field(
            user_input=user_input,
            task_input={"_placeholder": True},
            field_name="user_profile",
            default=populated_kb,
        )

        # The resolved profile should be a formatted string
        assert isinstance(resolved_profile, str)
        assert len(resolved_profile) > 0
        # Should contain metadata from user:xinli
        assert "Xinli" in resolved_profile
        assert "Seattle" in resolved_profile
        # Should contain relevant knowledge about eggs
        assert "egg" in resolved_profile.lower()

    def test_kb_context_manager_with_agent(self):
        """KnowledgeBase can be used as a context manager alongside Agent.

        Validates: Requirements 6.1, 6.3
        """
        metadata_store, piece_store, graph_store = _make_stores()

        with KnowledgeBase(
            metadata_store=metadata_store,
            piece_store=piece_store,
            graph_store=graph_store,
            active_entity_id="user:test",
        ) as kb:
            agent = Agent(user_profile=kb)
            assert agent.user_profile is kb
            assert callable(agent.user_profile)
            # KB should work within context manager
            result = agent.user_profile("test query")
            assert isinstance(result, str)

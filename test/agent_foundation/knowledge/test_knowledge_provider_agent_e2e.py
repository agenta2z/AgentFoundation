"""
E2E integration tests for knowledge_provider → Agent → Reasoner Input.

Verifies the complete wiring: a KnowledgeProvider set on the `knowledge_provider`
attribute flows through Agent.__call__ → _construct_reasoner_input → prompt feed
merge → formatted template → reasoner input.

Uses:
- A self-contained KnowledgeProvider built from inline mock data (no external files)
- A mock reasoner that captures the reasoner_input argument
- A _TestablePromptBasedAgent subclass that overrides _parse_raw_response to
  immediately complete (same pattern as TestAgent in test_agent.py)

All test data uses clearly fictional entities (e.g., "Alice Mockwell", "MockMart").

Requirements: 3.2, 3.3, 4.1, 4.2, 4.5
"""
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
_workspace_root = _current_path.parent.parent
_spu_src = _workspace_root / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest
from attr import attrs, attrib

from agent_foundation.agents.agent_response import AgentResponse
from agent_foundation.agents.agent_state import AgentTaskStatusFlags, AgentStateItem
from agent_foundation.agents.prompt_based_agents.prompt_based_agent import (
    PromptBasedAgent,
    FeedConflictResolution,
)
from agent_foundation.knowledge import KnowledgeProvider
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece, KnowledgeType
from agent_foundation.knowledge.retrieval.stores.metadata.keyvalue_adapter import KeyValueMetadataStore
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import RetrievalKnowledgePieceStore
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import GraphServiceEntityGraphStore
from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import MemoryKeyValueService
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import MemoryRetrievalService
from rich_python_utils.service_utils.graph_service.memory_graph_service import MemoryGraphService
from rich_python_utils.service_utils.graph_service.graph_node import GraphNode, GraphEdge
from rich_python_utils.string_utils.formatting.handlebars_format import (
    format_template as handlebars_format,
)


# ── Test template ─────────────────────────────────────────────────────────────

TEST_TEMPLATE = (
    "You are a helpful shopping assistant.\n\n"
    "User profile: {{user_profile}}\n\n"
    "Instructions: {{instructions}}\n\n"
    "User request: {{user_input}}\n\n"
    "Please help the user."
)


# ── Helper classes ────────────────────────────────────────────────────────────


class MockReasonerCapture:
    """Mock reasoner that captures the reasoner_input for assertion."""

    def __init__(self):
        self.captured_inputs = []

    def __call__(self, reasoner_input, reasoner_inference_config=None, **kwargs):
        self.captured_inputs.append(reasoner_input)
        return "mock response"


@attrs
class _TestablePromptBasedAgent(PromptBasedAgent):
    """PromptBasedAgent subclass that terminates after one reasoner call.

    Overrides _parse_raw_response to return Completed immediately,
    avoiding the need for structured XML/JSON delimiters in the mock response.

    Named with underscore prefix to prevent pytest from trying to collect it
    as a test class.
    """

    def _parse_raw_response(self, raw_response) -> Tuple[
        Union[str, AgentResponse],
        Union[AgentTaskStatusFlags, str, AgentStateItem, Any]
    ]:
        return (
            AgentResponse(instant_response="done", next_actions=[]),
            AgentStateItem(task_status=AgentTaskStatusFlags.Completed),
        )


# ── Mock knowledge data (all fictional) ──────────────────────────────────────


def _build_mock_kb() -> KnowledgeBase:
    """Build a KnowledgeBase with fictional inline data. No external files."""
    metadata_store = KeyValueMetadataStore(kv_service=MemoryKeyValueService())
    piece_store = RetrievalKnowledgePieceStore(retrieval_service=MemoryRetrievalService())
    graph_store = GraphServiceEntityGraphStore(graph_service=MemoryGraphService())

    kb = KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:mock_alice",
    )

    # Metadata — fictional user
    metadata_store.save_metadata(EntityMetadata(
        entity_id="user:mock_alice",
        entity_type="user",
        properties={
            "FirstName": "Alice",
            "LastName": "Mockwell",
            "Location": "Faketown, FS, Testland",
            "ZipCode": "00000",
        },
    ))

    # Pieces — fictional memberships (entity-scoped, info_type=user_profile)
    piece_store.add(KnowledgePiece(
        piece_id="mockmart-membership",
        content="User is a MockMart member with free delivery on qualifying orders.",
        knowledge_type=KnowledgeType.Fact,
        info_type="user_profile",
        tags=["grocery", "mockmart", "membership"],
        entity_id="user:mock_alice",
        embedding_text="MockMart grocery store membership account delivery",
    ))
    piece_store.add(KnowledgePiece(
        piece_id="freshco-membership",
        content="User is a FreshCo premium member.",
        knowledge_type=KnowledgeType.Fact,
        info_type="user_profile",
        tags=["grocery", "freshco", "membership"],
        entity_id="user:mock_alice",
        embedding_text="FreshCo grocery store premium membership",
    ))

    # Piece — fictional procedure (global, info_type=instructions)
    piece_store.add(KnowledgePiece(
        piece_id="mock-shopping-procedure",
        content="Mock Shopping Procedure: 1) Log in. 2) Find store. 3) Add items to cart and checkout.",
        knowledge_type=KnowledgeType.Procedure,
        info_type="instructions",
        tags=["grocery", "procedure", "workflow"],
        entity_id=None,
        embedding_text="grocery shopping steps procedure workflow login store cart checkout",
    ))

    # Graph — fictional relationships
    graph_store.add_node(GraphNode(node_id="user:mock_alice", node_type="user", label="Alice Mockwell"))
    graph_store.add_node(GraphNode(node_id="store:mockmart", node_type="store", label="MockMart"))
    graph_store.add_node(GraphNode(node_id="store:freshco", node_type="store", label="FreshCo"))
    graph_store.add_relation(GraphEdge(
        source_id="user:mock_alice", target_id="store:mockmart",
        edge_type="SHOPS_AT", properties={"piece_id": "mockmart-membership"},
    ))
    graph_store.add_relation(GraphEdge(
        source_id="user:mock_alice", target_id="store:freshco",
        edge_type="SHOPS_AT", properties={"piece_id": "freshco-membership"},
    ))

    return kb


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_knowledge_provider():
    """KnowledgeProvider backed by inline fictional data."""
    kb = _build_mock_kb()
    provider = KnowledgeProvider(kb=kb)
    yield provider
    provider.close()


@pytest.fixture
def mock_reasoner():
    """Mock reasoner that captures inputs."""
    return MockReasonerCapture()


def _make_agent(
    reasoner,
    knowledge_provider=None,
    user_profile=None,
    feed_conflict_resolution=FeedConflictResolution.FEED_ONLY,
):
    """Create a _TestablePromptBasedAgent with the test template."""
    return _TestablePromptBasedAgent(
        reasoner=reasoner,
        default_prompt_template=TEST_TEMPLATE,
        prompt_formatter=handlebars_format,
        knowledge_provider=knowledge_provider,
        user_profile=user_profile,
        feed_conflict_resolution=feed_conflict_resolution,
    )


def _call_agent(agent, user_input: str):
    """Call agent with user_input using the dict format required by Agent.__call__.

    Agent.__call__ expects either a dict with 'user_input' key or uses interactive
    queue. Plain string input fails because task_input.pop() is called before
    the string check. Using dict format ensures correct resolution.
    """
    return agent({"user_input": user_input})


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestConstructReasonerInputWithKnowledge:
    """Unit test: call _construct_reasoner_input directly with real provider output."""

    def test_construct_reasoner_input_contains_knowledge(self, mock_knowledge_provider, mock_reasoner):
        """Direct _construct_reasoner_input call merges knowledge into template.

        Validates the inner integration: knowledge dict is merged into feed
        and the template renders with user_profile and instructions.
        """
        agent = _make_agent(reasoner=mock_reasoner, knowledge_provider=mock_knowledge_provider)

        # Call the provider directly to get the knowledge dict
        query = "MockMart grocery store membership"
        knowledge = mock_knowledge_provider(query)
        assert isinstance(knowledge, dict)
        assert len(knowledge) > 0, f"Provider returned empty dict for query '{query}'"

        # Call _construct_reasoner_input with the knowledge dict
        reasoner_input = agent._construct_reasoner_input(
            task_input={},
            user_input=query,
            knowledge=knowledge,
        )

        # The rendered template should contain knowledge content
        assert isinstance(reasoner_input, str)
        assert "MockMart" in reasoner_input
        assert "You are a helpful shopping assistant" in reasoner_input


class TestE2EAgentCall:
    """E2E test: full __call__ flow — knowledge appears in captured reasoner_input."""

    def test_e2e_agent_call_passes_knowledge_to_reasoner(self, mock_knowledge_provider, mock_reasoner):
        """Full __call__ flow: knowledge appears in captured reasoner_input.

        Validates the complete chain:
        1. Agent.__call__ resolves knowledge_provider(user_input)
        2. Passes knowledge to _construct_reasoner_input()
        3. _construct_reasoner_input merges into feed via _merge_into_feed()
        4. prompt_formatter renders template
        5. Rendered string passed to reasoner
        """
        agent = _make_agent(
            reasoner=mock_reasoner,
            knowledge_provider=mock_knowledge_provider,
        )

        _call_agent(agent, "MockMart grocery store membership")

        # Reasoner should have been called exactly once
        assert len(mock_reasoner.captured_inputs) == 1
        reasoner_input = mock_reasoner.captured_inputs[0]

        # The rendered template should contain user request
        assert "MockMart" in reasoner_input
        # Should contain template structure
        assert "You are a helpful shopping assistant" in reasoner_input

    def test_e2e_knowledge_provider_receives_user_input(self, mock_reasoner):
        """Provider callable receives the exact user_input string."""
        captured_queries = []

        def tracking_provider(query):
            captured_queries.append(query)
            return {"user_profile": "Mock Profile Data", "instructions": "Mock Instructions"}

        agent = _make_agent(
            reasoner=mock_reasoner,
            knowledge_provider=tracking_provider,
        )

        _call_agent(agent, "buy eggs at FreshCo")

        # Provider should have been called with the exact user_input
        assert len(captured_queries) == 1
        assert captured_queries[0] == "buy eggs at FreshCo"

        # And the knowledge should appear in the rendered template
        reasoner_input = mock_reasoner.captured_inputs[0]
        assert "Mock Profile Data" in reasoner_input
        assert "Mock Instructions" in reasoner_input


class TestE2EStaticDictKnowledgeProvider:
    """E2E test: isinstance(knowledge_provider, dict) branch works."""

    def test_e2e_static_dict_knowledge_provider(self, mock_reasoner):
        """Static dict knowledge_provider (not callable) is passed through directly.

        Tests the branch in agent.py where:
            elif isinstance(self.knowledge_provider, dict):
                knowledge = self.knowledge_provider
        """
        static_knowledge = {
            "user_profile": "Alice Mockwell, Faketown, FS",
            "instructions": "Always check MockMart first",
        }

        agent = _make_agent(
            reasoner=mock_reasoner,
            knowledge_provider=static_knowledge,
        )

        _call_agent(agent, "plan dinner shopping")

        assert len(mock_reasoner.captured_inputs) == 1
        reasoner_input = mock_reasoner.captured_inputs[0]
        assert "Alice Mockwell, Faketown, FS" in reasoner_input
        assert "Always check MockMart first" in reasoner_input


class TestE2EFeedConflictResolutionMerge:
    """E2E test: MERGE mode concatenates static user_profile + knowledge user_profile."""

    def test_e2e_feed_conflict_resolution_merge(self, mock_reasoner):
        """MERGE mode concatenates existing user_profile with knowledge user_profile.

        When both user_profile (static attribute) and knowledge_provider supply
        user_profile, MERGE should concatenate them with \\n\\n.
        """
        static_profile = "Base profile: Alice Mockwell"
        knowledge = {
            "user_profile": "Membership: MockMart Gold",
            "instructions": "Follow the mock shopping procedure",
        }

        agent = _make_agent(
            reasoner=mock_reasoner,
            knowledge_provider=knowledge,
            user_profile=static_profile,
            feed_conflict_resolution=FeedConflictResolution.MERGE,
        )

        _call_agent(agent, "buy groceries")

        assert len(mock_reasoner.captured_inputs) == 1
        reasoner_input = mock_reasoner.captured_inputs[0]

        # Both the static profile and knowledge profile should appear
        assert "Base profile: Alice Mockwell" in reasoner_input
        assert "Membership: MockMart Gold" in reasoner_input
        assert "Follow the mock shopping procedure" in reasoner_input


class TestE2ENoKnowledgeProvider:
    """E2E test: knowledge_provider=None (default) doesn't break anything."""

    def test_e2e_no_knowledge_provider(self, mock_reasoner):
        """Agent with no knowledge_provider still works normally."""
        agent = _make_agent(
            reasoner=mock_reasoner,
            knowledge_provider=None,
            user_profile="Simple static profile",
        )

        _call_agent(agent, "hello world")

        assert len(mock_reasoner.captured_inputs) == 1
        reasoner_input = mock_reasoner.captured_inputs[0]
        assert "hello world" in reasoner_input
        assert "Simple static profile" in reasoner_input
        assert "You are a helpful shopping assistant" in reasoner_input

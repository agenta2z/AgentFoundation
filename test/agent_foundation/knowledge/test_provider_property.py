"""
Property-based tests for KnowledgeProvider callable.

Property 2: KnowledgeProvider returns Dict[str, str] for any query.
For any populated KnowledgeProvider instance and any non-empty query string,
calling provider(query) should return a Dict[str, str] where every key is str
and every value is str.

# Feature: knowledge-agent-integration
# Property 2: KnowledgeProvider returns Dict[str, str] for any query

**Validates: Requirements 2.2, 2.5, 3.3**
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
_spu_src = Path(__file__).resolve().parents[3] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

from hypothesis import given, settings, strategies as st

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
from agent_foundation.knowledge.retrieval.provider import KnowledgeProvider
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)


# ── Module-level populated provider (created ONCE) ───────────────────────────


def _create_populated_provider() -> KnowledgeProvider:
    """Create a KnowledgeProvider populated with diverse data.

    Populates:
    - User metadata (FirstName, Location, ZipCode)
    - Three pieces with different info_types: user_profile, instructions, context
    - Graph nodes for user, store, and procedure
    - Graph edges: SHOPS_AT (with linked piece), HAS_SKILL (with linked piece),
      KNOWS (no linked piece, depth-1 → user_profile)
    """
    metadata_store = KeyValueMetadataStore(MemoryKeyValueService())
    piece_store = RetrievalKnowledgePieceStore(MemoryRetrievalService())
    graph_store = GraphServiceEntityGraphStore(MemoryGraphService())
    kb = KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:alice",
    )

    # Add user metadata
    user_meta = EntityMetadata(
        entity_id="user:alice",
        entity_type="user",
        properties={"FirstName": "Alice", "Location": "Seattle", "ZipCode": "98101"},
    )
    kb.metadata_store.save_metadata(user_meta)

    # Add knowledge pieces with different info_types
    profile_piece = KnowledgePiece(
        content="Alice is a Safeway member with free delivery benefits.",
        piece_id="safeway-membership",
        knowledge_type=KnowledgeType.Fact,
        info_type="user_profile",
        tags=["grocery", "safeway", "membership"],
        entity_id="user:alice",
        embedding_text="Safeway grocery store membership delivery benefits",
    )
    kb.add_piece(profile_piece)

    instruction_piece = KnowledgePiece(
        content="Grocery Procedure: 1) Log in. 2) Find store. 3) Add items. 4) Checkout.",
        piece_id="grocery-procedure",
        knowledge_type=KnowledgeType.Procedure,
        info_type="instructions",
        tags=["grocery", "procedure", "workflow"],
        entity_id=None,
        embedding_text="grocery shopping steps procedure login checkout workflow",
    )
    kb.add_piece(instruction_piece)

    context_piece = KnowledgePiece(
        content="Seattle has many grocery delivery options available year-round.",
        piece_id="seattle-grocery-context",
        knowledge_type=KnowledgeType.Note,
        info_type="context",
        tags=["seattle", "grocery", "delivery"],
        entity_id=None,
        embedding_text="Seattle grocery delivery options availability",
    )
    kb.add_piece(context_piece)

    # Add graph nodes
    user_node = GraphNode(node_id="user:alice", node_type="user", label="Alice")
    store_node = GraphNode(node_id="store:safeway", node_type="store", label="Safeway")
    procedure_node = GraphNode(
        node_id="procedure:grocery_shopping",
        node_type="procedure",
        label="Grocery Shopping Procedure",
    )
    friend_node = GraphNode(node_id="person:bob", node_type="person", label="Bob")
    kb.graph_store.add_node(user_node)
    kb.graph_store.add_node(store_node)
    kb.graph_store.add_node(procedure_node)
    kb.graph_store.add_node(friend_node)

    # Add graph edges
    # Edge with linked piece (routes by piece info_type)
    shops_edge = GraphEdge(
        source_id="user:alice",
        target_id="store:safeway",
        edge_type="SHOPS_AT",
        properties={"piece_id": "safeway-membership"},
    )
    kb.graph_store.add_relation(shops_edge)

    skill_edge = GraphEdge(
        source_id="user:alice",
        target_id="procedure:grocery_shopping",
        edge_type="HAS_SKILL",
        properties={"piece_id": "grocery-procedure"},
    )
    kb.graph_store.add_relation(skill_edge)

    # Edge without linked piece (depth-1 from user → user_profile)
    knows_edge = GraphEdge(
        source_id="user:alice",
        target_id="person:bob",
        edge_type="KNOWS",
        properties={},
    )
    kb.graph_store.add_relation(knows_edge)

    return KnowledgeProvider(kb)


# Create the provider ONCE at module level
_PROVIDER = _create_populated_provider()


# ── Hypothesis strategy for non-empty query strings ──────────────────────────

_query_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=60,
).filter(lambda s: s.strip())


# ══════════════════════════════════════════════════════════════════════════════
# Property 2: KnowledgeProvider returns Dict[str, str] for any query
# ══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeProviderReturnType:
    """Property 2: KnowledgeProvider returns Dict[str, str] for any query.

    For any populated KnowledgeProvider instance and any non-empty query string,
    calling provider(query) should return a Dict[str, str] where every key is
    str and every value is str.

    **Validates: Requirements 2.2, 2.5, 3.3**
    """

    @given(query=_query_strategy)
    @settings(max_examples=100)
    def test_provider_returns_dict_str_str(self, query: str):
        """Calling provider(query) returns Dict[str, str] for any non-empty query.

        **Validates: Requirements 2.2, 2.5, 3.3**
        """
        result = _PROVIDER(query)

        # Return type must be dict
        assert isinstance(result, dict), (
            f"Expected dict, got {type(result).__name__} for query {query!r}"
        )

        # Every key must be str
        for key in result.keys():
            assert isinstance(key, str), (
                f"Key {key!r} is not str (type={type(key).__name__}) "
                f"for query {query!r}"
            )

        # Every value must be str
        for key, value in result.items():
            assert isinstance(value, str), (
                f"Value for key {key!r} is not str "
                f"(type={type(value).__name__}, value={value!r}) "
                f"for query {query!r}"
            )

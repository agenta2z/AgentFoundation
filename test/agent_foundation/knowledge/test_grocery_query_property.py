"""
Property-based tests for grocery query relevance.

Property 3: Grocery query retrieves relevant knowledge.
For any KnowledgeProvider populated with the grocery store knowledge data and
any query string containing a grocery store name (e.g., "safeway", "qfc",
"wholefoods"), provider(query)["user_profile"] should contain that store's
membership information, and provider(query)["instructions"] should contain
the grocery shopping procedure.

# Feature: knowledge-agent-integration
# Property 3: Grocery query retrieves relevant knowledge

**Validates: Requirements 6.3**
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
from science_modeling_tools.knowledge.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from science_modeling_tools.knowledge.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from science_modeling_tools.knowledge.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from science_modeling_tools.knowledge.knowledge_base import KnowledgeBase
from science_modeling_tools.knowledge.data_loader import KnowledgeDataLoader
from science_modeling_tools.knowledge.provider import KnowledgeProvider


# ── Path to grocery data file ────────────────────────────────────────────────

_workspace_root = Path(__file__).resolve().parents[4]
GROCERY_DATA_FILE = str(
    _workspace_root
    / "WebAgent"
    / "test"
    / "webaxon"
    / "webaxon"
    / "grocery_store_testcase"
    / "knowledge_data.json"
)


# ── Module-level populated provider (created ONCE) ───────────────────────────


def _create_grocery_provider() -> KnowledgeProvider:
    """Create a KnowledgeProvider populated from the grocery store knowledge data file."""
    metadata_store = KeyValueMetadataStore(MemoryKeyValueService())
    piece_store = RetrievalKnowledgePieceStore(MemoryRetrievalService())
    graph_store = GraphServiceEntityGraphStore(MemoryGraphService())
    kb = KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:[name]",
    )
    KnowledgeDataLoader.load(kb, GROCERY_DATA_FILE)
    return KnowledgeProvider(kb)


# Create the provider ONCE at module level
_PROVIDER = _create_grocery_provider()


# ── Mapping from store name to expected membership content ───────────────────

_STORE_MEMBERSHIP_MARKERS = {
    "safeway": "Safeway member",
    "qfc": "QFC member",
    "wholefoods": "Whole Foods prime member",
}


# ── Hypothesis strategy for grocery queries ──────────────────────────────────

@st.composite
def grocery_query_strategy(draw):
    """Generate a query containing a grocery store name and a context word.

    The MemoryRetrievalService uses simple keyword matching (term overlap).
    Including a context word like "grocery" or "shopping" ensures the procedure
    piece's embedding_text is also matched.
    """
    store = draw(st.sampled_from(["safeway", "qfc", "wholefoods"]))
    context_word = draw(st.sampled_from(["grocery", "shopping", "store", "delivery"]))
    return f"{context_word} {store}"


# ══════════════════════════════════════════════════════════════════════════════
# Property 3: Grocery query retrieves relevant knowledge
# ══════════════════════════════════════════════════════════════════════════════


class TestGroceryQueryRelevance:
    """Property 3: Grocery query retrieves relevant knowledge.

    For any KnowledgeProvider populated with the grocery store knowledge data
    and any query string containing a grocery store name, the provider should
    return user_profile containing that store's membership information and
    instructions containing the grocery shopping procedure.

    **Validates: Requirements 6.3**
    """

    @given(query=grocery_query_strategy())
    @settings(max_examples=100)
    def test_grocery_query_returns_membership_and_procedure(self, query: str):
        """Querying with a store name returns membership info and procedure.

        **Validates: Requirements 6.3**
        """
        result = _PROVIDER(query)

        # Determine which store was queried
        store_name = None
        for name in _STORE_MEMBERSHIP_MARKERS:
            if name in query.lower():
                store_name = name
                break
        assert store_name is not None, (
            f"Could not determine store name from query: {query!r}"
        )

        # user_profile must be present and contain the store's membership info
        assert "user_profile" in result, (
            f"Result missing 'user_profile' key for query {query!r}. "
            f"Keys: {list(result.keys())}"
        )
        membership_marker = _STORE_MEMBERSHIP_MARKERS[store_name]
        assert membership_marker in result["user_profile"], (
            f"user_profile does not contain '{membership_marker}' "
            f"for query {query!r}. "
            f"user_profile content: {result['user_profile']!r}"
        )

        # instructions must be present and contain the grocery procedure
        assert "instructions" in result, (
            f"Result missing 'instructions' key for query {query!r}. "
            f"Keys: {list(result.keys())}"
        )
        assert "Grocery Shopping Procedure" in result["instructions"], (
            f"instructions does not contain 'Grocery Shopping Procedure' "
            f"for query {query!r}. "
            f"instructions content: {result['instructions']!r}"
        )

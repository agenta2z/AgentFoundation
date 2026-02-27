"""
Unit tests validating the grocery store knowledge data file structure.

Loads the grocery knowledge data JSON file and verifies:
- Top-level sections (metadata, pieces, graph)
- Metadata entries and their properties
- Knowledge piece counts, types, and fields
- Graph node and edge counts and types
- End-to-end loading via KnowledgeDataLoader into a KnowledgeBase

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
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

_spu_src = Path(__file__).resolve().parents[3] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest

from agent_foundation.knowledge.retrieval.data_loader import KnowledgeDataLoader
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
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

# ── Path to grocery data file ────────────────────────────────────────────────

_workspace_root = Path(__file__).resolve().parents[4]  # Go up from test/agent_foundation/knowledge/ to workspace root
GROCERY_DATA_FILE = str(
    _workspace_root
    / "WebAgent"
    / "test"
    / "webaxon"
    / "webaxon"
    / "grocery_store_testcase"
    / "knowledge_data.json"
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_grocery_data() -> dict:
    """Load and return the raw grocery data JSON as a dict."""
    with open(GROCERY_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _create_test_kb() -> KnowledgeBase:
    """Create a KnowledgeBase with in-memory stores for testing."""
    metadata_store = KeyValueMetadataStore(MemoryKeyValueService())
    piece_store = RetrievalKnowledgePieceStore(MemoryRetrievalService())
    graph_store = GraphServiceEntityGraphStore(MemoryGraphService())
    return KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:test",
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def grocery_data():
    """Load the grocery data JSON file."""
    return _load_grocery_data()


@pytest.fixture
def kb():
    """Create a fresh KnowledgeBase with in-memory stores."""
    return _create_test_kb()


# ── Test 1: Top-level structure ──────────────────────────────────────────────


class TestGroceryDataStructure:
    """Verify the grocery data file has the three required top-level sections.

    Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """

    def test_file_exists(self):
        """The grocery data file exists at the expected path."""
        assert Path(GROCERY_DATA_FILE).exists(), (
            f"Grocery data file not found at {GROCERY_DATA_FILE}"
        )

    def test_has_required_sections(self, grocery_data):
        """The file has metadata, pieces, and graph sections."""
        assert "metadata" in grocery_data
        assert "pieces" in grocery_data
        assert "graph" in grocery_data


# ── Test 2: Metadata entries ─────────────────────────────────────────────────


class TestGroceryMetadata:
    """Verify metadata has 4 entries with expected properties.

    Validates: Requirements 5.1, 5.6
    """

    def test_metadata_has_four_entries(self, grocery_data):
        """Metadata section has exactly 4 entries: user + 3 stores."""
        metadata = grocery_data["metadata"]
        assert len(metadata) == 4

    def test_metadata_has_expected_keys(self, grocery_data):
        """Metadata contains user:[name], store:safeway, store:qfc, store:wholefoods."""
        metadata = grocery_data["metadata"]
        keys = set(metadata.keys())
        assert "user:[name]" in keys
        assert "store:safeway" in keys
        assert "store:qfc" in keys
        assert "store:wholefoods" in keys

    def test_user_metadata_has_required_properties(self, grocery_data):
        """User metadata has FirstName, LastName, Location, ZipCode, PhoneNumber, Family."""
        user_props = grocery_data["metadata"]["user:[name]"]["properties"]
        required_fields = {"FirstName", "LastName", "Location", "ZipCode", "PhoneNumber", "Family"}
        assert required_fields.issubset(set(user_props.keys()))

    def test_store_metadata_has_required_properties(self, grocery_data):
        """Each store metadata has Name, URL, MemberEmail."""
        for store_key in ["store:safeway", "store:qfc", "store:wholefoods"]:
            store_props = grocery_data["metadata"][store_key]["properties"]
            assert "Name" in store_props, f"{store_key} missing Name"
            assert "URL" in store_props, f"{store_key} missing URL"
            assert "MemberEmail" in store_props, f"{store_key} missing MemberEmail"


# ── Test 3: Knowledge pieces ────────────────────────────────────────────────


class TestGroceryPieces:
    """Verify pieces section has 4 entries with correct types.

    Validates: Requirements 5.2, 5.3
    """

    def test_pieces_has_four_entries(self, grocery_data):
        """Pieces section has exactly 4 entries: 3 memberships + 1 procedure."""
        pieces = grocery_data["pieces"]
        assert len(pieces) == 4

    def test_membership_pieces_have_correct_types(self, grocery_data):
        """The 3 membership pieces have knowledge_type=fact, info_type=user_profile."""
        pieces = grocery_data["pieces"]
        membership_pieces = [
            p for p in pieces if p["piece_id"].endswith("-membership")
        ]
        assert len(membership_pieces) == 3

        for piece in membership_pieces:
            assert piece["knowledge_type"] == "fact", (
                f"Piece {piece['piece_id']} should have knowledge_type=fact"
            )
            assert piece["info_type"] == "user_profile", (
                f"Piece {piece['piece_id']} should have info_type=user_profile"
            )

    def test_procedure_piece_has_correct_types(self, grocery_data):
        """The procedure piece has knowledge_type=procedure, info_type=instructions."""
        pieces = grocery_data["pieces"]
        procedure_pieces = [
            p for p in pieces if p["knowledge_type"] == "procedure"
        ]
        assert len(procedure_pieces) == 1

        procedure = procedure_pieces[0]
        assert procedure["knowledge_type"] == "procedure"
        assert procedure["info_type"] == "instructions"

    def test_all_pieces_have_embedding_text(self, grocery_data):
        """Every piece has a non-empty embedding_text field."""
        pieces = grocery_data["pieces"]
        for piece in pieces:
            assert "embedding_text" in piece, (
                f"Piece {piece['piece_id']} missing embedding_text"
            )
            assert piece["embedding_text"], (
                f"Piece {piece['piece_id']} has empty embedding_text"
            )


# ── Test 4: Graph nodes and edges ───────────────────────────────────────────


class TestGroceryGraph:
    """Verify graph has 5 nodes and 7 edges with correct types.

    Validates: Requirements 5.4, 5.5
    """

    def test_graph_has_five_nodes(self, grocery_data):
        """Graph has exactly 5 nodes."""
        nodes = grocery_data["graph"]["nodes"]
        assert len(nodes) == 5

    def test_graph_has_seven_edges(self, grocery_data):
        """Graph has exactly 7 edges."""
        edges = grocery_data["graph"]["edges"]
        assert len(edges) == 7

    def test_edge_types_correct_counts(self, grocery_data):
        """Graph has 3 SHOPS_AT, 1 HAS_SKILL, 3 USES_PROCEDURE edges."""
        edges = grocery_data["graph"]["edges"]
        edge_type_counts = {}
        for edge in edges:
            t = edge["edge_type"]
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1

        assert edge_type_counts.get("SHOPS_AT", 0) == 3
        assert edge_type_counts.get("HAS_SKILL", 0) == 1
        assert edge_type_counts.get("USES_PROCEDURE", 0) == 3


# ── Test 5: Load via KnowledgeDataLoader into KnowledgeBase ─────────────────


class TestGroceryDataLoaderIntegration:
    """Load the grocery data file via KnowledgeDataLoader and verify stores.

    Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """

    def test_loader_returns_correct_counts(self, kb):
        """KnowledgeDataLoader.load returns correct counts for all sections."""
        result = KnowledgeDataLoader.load(kb, GROCERY_DATA_FILE)

        assert result["metadata"] == 4
        assert result["pieces"] == 4
        assert result["graph_nodes"] == 5
        assert result["graph_edges"] == 7

    def test_loader_populates_metadata_store(self, kb):
        """All metadata entries are retrievable from the metadata store."""
        KnowledgeDataLoader.load(kb, GROCERY_DATA_FILE)

        # Verify user metadata
        user_meta = kb.metadata_store.get_metadata("user:[name]")
        assert user_meta is not None
        assert "FirstName" in user_meta.properties
        assert "LastName" in user_meta.properties

        # Verify all store metadata entries
        for store_id in ["store:safeway", "store:qfc", "store:wholefoods"]:
            store_meta = kb.metadata_store.get_metadata(store_id)
            assert store_meta is not None, f"Missing metadata for {store_id}"
            assert "Name" in store_meta.properties
            assert "URL" in store_meta.properties

    def test_loader_populates_piece_store(self, kb):
        """All knowledge pieces are retrievable from the piece store."""
        KnowledgeDataLoader.load(kb, GROCERY_DATA_FILE)

        # Verify membership pieces
        for piece_id in ["safeway-membership", "qfc-membership", "wholefoods-membership"]:
            piece = kb.piece_store.get_by_id(piece_id)
            assert piece is not None, f"Missing piece: {piece_id}"
            assert piece.knowledge_type.value == "fact"
            assert piece.info_type == "user_profile"

        # Verify procedure piece
        procedure = kb.piece_store.get_by_id("grocery-shopping-procedure")
        assert procedure is not None
        assert procedure.knowledge_type.value == "procedure"
        assert procedure.info_type == "instructions"

    def test_loader_populates_graph_store(self, kb):
        """All graph nodes and edges are present in the graph store."""
        KnowledgeDataLoader.load(kb, GROCERY_DATA_FILE)

        # Verify all nodes exist
        for node_id in [
            "user:[name]",
            "store:safeway",
            "store:qfc",
            "store:wholefoods",
            "procedure:grocery_shopping",
        ]:
            node = kb.graph_store.get_node(node_id)
            assert node is not None, f"Missing graph node: {node_id}"

        # Verify user edges (SHOPS_AT + HAS_SKILL)
        user_edges = kb.graph_store.get_relations("user:[name]", direction="outgoing")
        assert len(user_edges) == 4  # 3 SHOPS_AT + 1 HAS_SKILL
        edge_types = [e.edge_type for e in user_edges]
        assert edge_types.count("SHOPS_AT") == 3
        assert edge_types.count("HAS_SKILL") == 1

        # Verify store USES_PROCEDURE edges
        for store_id in ["store:safeway", "store:qfc", "store:wholefoods"]:
            store_edges = kb.graph_store.get_relations(store_id, direction="outgoing")
            uses_proc = [e for e in store_edges if e.edge_type == "USES_PROCEDURE"]
            assert len(uses_proc) == 1, f"{store_id} should have 1 USES_PROCEDURE edge"

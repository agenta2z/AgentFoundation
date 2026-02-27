"""
Unit tests for KnowledgeDataLoader.

Tests loading knowledge data from JSON files into a KnowledgeBase,
including valid files, missing files, malformed JSON, invalid knowledge_type,
and sensitive content handling.

Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 1.7
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


# ── Helpers ──────────────────────────────────────────────────────────────────


def create_test_kb():
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


def make_valid_data():
    """Return a valid knowledge data dict with metadata, pieces, and graph."""
    return {
        "metadata": {
            "user:alice": {
                "entity_type": "user",
                "properties": {
                    "FirstName": "Alice",
                    "LastName": "Smith",
                    "Location": "Seattle, WA",
                },
            },
            "store:acme": {
                "entity_type": "store",
                "properties": {
                    "Name": "Acme Store",
                    "URL": "https://www.acme.com",
                },
            },
        },
        "pieces": [
            {
                "piece_id": "acme-membership",
                "content": "Alice is an Acme Store member with free delivery.",
                "knowledge_type": "fact",
                "info_type": "user_profile",
                "tags": ["grocery", "acme"],
                "entity_id": "user:alice",
                "embedding_text": "Acme Store membership delivery",
            },
            {
                "piece_id": "shopping-procedure",
                "content": "Shopping Procedure: 1) Log in. 2) Add items. 3) Checkout.",
                "knowledge_type": "procedure",
                "info_type": "instructions",
                "tags": ["shopping", "procedure"],
                "entity_id": None,
                "embedding_text": "shopping steps procedure checkout",
            },
        ],
        "graph": {
            "nodes": [
                {
                    "node_id": "user:alice",
                    "node_type": "user",
                    "label": "Alice Smith",
                },
                {
                    "node_id": "store:acme",
                    "node_type": "store",
                    "label": "Acme Store",
                },
                {
                    "node_id": "procedure:shopping",
                    "node_type": "procedure",
                    "label": "Shopping Procedure",
                },
            ],
            "edges": [
                {
                    "source_id": "user:alice",
                    "target_id": "store:acme",
                    "edge_type": "SHOPS_AT",
                    "properties": {"piece_id": "acme-membership"},
                },
                {
                    "source_id": "user:alice",
                    "target_id": "procedure:shopping",
                    "edge_type": "HAS_SKILL",
                    "properties": {"piece_id": "shopping-procedure"},
                },
            ],
        },
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def kb():
    """Create a fresh KnowledgeBase with in-memory stores."""
    return create_test_kb()


@pytest.fixture
def valid_data_file(tmp_path):
    """Create a valid knowledge data JSON file in a temp directory."""
    data = make_valid_data()
    file_path = tmp_path / "knowledge_data.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return str(file_path)


# ── Test: Valid file populates all three stores ──────────────────────────────


class TestLoadValidFile:
    """Test KnowledgeDataLoader.load() with a valid knowledge data file.

    Validates: Requirements 1.1, 1.2, 1.3
    """

    def test_load_returns_correct_counts(self, kb, valid_data_file):
        """Load returns a dict with correct counts for each section."""
        result = KnowledgeDataLoader.load(kb, valid_data_file)

        assert result["metadata"] == 2
        assert result["pieces"] == 2
        assert result["graph_nodes"] == 3
        assert result["graph_edges"] == 2

    def test_load_populates_metadata_store(self, kb, valid_data_file):
        """Metadata entries are retrievable from the metadata store after load."""
        KnowledgeDataLoader.load(kb, valid_data_file)

        # Verify user metadata
        user_meta = kb.metadata_store.get_metadata("user:alice")
        assert user_meta is not None
        assert user_meta.properties["FirstName"] == "Alice"
        assert user_meta.properties["LastName"] == "Smith"
        assert user_meta.properties["Location"] == "Seattle, WA"

        # Verify store metadata
        store_meta = kb.metadata_store.get_metadata("store:acme")
        assert store_meta is not None
        assert store_meta.properties["Name"] == "Acme Store"

    def test_load_populates_piece_store(self, kb, valid_data_file):
        """Knowledge pieces are retrievable from the piece store after load."""
        KnowledgeDataLoader.load(kb, valid_data_file)

        # Verify pieces are searchable
        piece = kb.piece_store.get_by_id("acme-membership")
        assert piece is not None
        assert "Acme Store member" in piece.content
        assert piece.info_type == "user_profile"

        procedure = kb.piece_store.get_by_id("shopping-procedure")
        assert procedure is not None
        assert "Shopping Procedure" in procedure.content
        assert procedure.info_type == "instructions"

    def test_load_populates_graph_store(self, kb, valid_data_file):
        """Graph nodes and edges are present in the graph store after load."""
        KnowledgeDataLoader.load(kb, valid_data_file)

        # Verify nodes
        user_node = kb.graph_store.get_node("user:alice")
        assert user_node is not None
        assert user_node.label == "Alice Smith"

        store_node = kb.graph_store.get_node("store:acme")
        assert store_node is not None
        assert store_node.label == "Acme Store"

        proc_node = kb.graph_store.get_node("procedure:shopping")
        assert proc_node is not None

        # Verify edges
        edges = kb.graph_store.get_relations("user:alice", direction="outgoing")
        assert len(edges) == 2
        edge_types = {e.edge_type for e in edges}
        assert "SHOPS_AT" in edge_types
        assert "HAS_SKILL" in edge_types


# ── Test: Missing file raises FileNotFoundError ─────────────────────────────


class TestLoadMissingFile:
    """Test KnowledgeDataLoader.load() with a non-existent file.

    Validates: Requirement 1.6
    """

    def test_missing_file_raises_file_not_found_error(self, kb, tmp_path):
        """Loading a non-existent file raises FileNotFoundError."""
        missing_path = str(tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError, match="nonexistent.json"):
            KnowledgeDataLoader.load(kb, missing_path)


# ── Test: Malformed JSON raises ValueError ───────────────────────────────────


class TestLoadMalformedJSON:
    """Test KnowledgeDataLoader.load() with malformed JSON files.

    Validates: Requirement 1.6
    """

    def test_invalid_json_raises_value_error(self, kb, tmp_path):
        """Loading a file with invalid JSON raises ValueError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json!!!", encoding="utf-8")

        with pytest.raises(ValueError, match="not valid JSON"):
            KnowledgeDataLoader.load(kb, str(bad_file))

    def test_missing_required_sections_raises_value_error(self, kb, tmp_path):
        """Loading a file missing required sections raises ValueError."""
        incomplete_file = tmp_path / "incomplete.json"
        incomplete_file.write_text(
            json.dumps({"metadata": {}}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="missing required sections"):
            KnowledgeDataLoader.load(kb, str(incomplete_file))

    def test_non_object_json_raises_value_error(self, kb, tmp_path):
        """Loading a file with a JSON array (not object) raises ValueError."""
        array_file = tmp_path / "array.json"
        array_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        with pytest.raises(ValueError, match="JSON object"):
            KnowledgeDataLoader.load(kb, str(array_file))


# ── Test: Invalid knowledge_type skips the bad piece ─────────────────────────


class TestLoadInvalidKnowledgeType:
    """Test KnowledgeDataLoader.load() with invalid knowledge_type values.

    Validates: Requirement 1.5
    """

    def test_invalid_knowledge_type_skips_piece(self, kb, tmp_path):
        """A piece with an invalid knowledge_type is skipped, others load."""
        data = {
            "metadata": {},
            "pieces": [
                {
                    "piece_id": "good-piece",
                    "content": "This is a valid fact.",
                    "knowledge_type": "fact",
                    "info_type": "context",
                    "tags": [],
                },
                {
                    "piece_id": "bad-piece",
                    "content": "This has an invalid type.",
                    "knowledge_type": "invalid_type_xyz",
                    "info_type": "context",
                    "tags": [],
                },
            ],
            "graph": {"nodes": [], "edges": []},
        }
        file_path = tmp_path / "invalid_type.json"
        file_path.write_text(json.dumps(data), encoding="utf-8")

        result = KnowledgeDataLoader.load(kb, str(file_path))

        # Only the valid piece should be loaded
        assert result["pieces"] == 1

        # Verify the good piece is present
        good = kb.piece_store.get_by_id("good-piece")
        assert good is not None
        assert good.content == "This is a valid fact."

        # Verify the bad piece was skipped
        bad = kb.piece_store.get_by_id("bad-piece")
        assert bad is None


# ── Test: Sensitive content skips the bad piece ──────────────────────────────


class TestLoadSensitiveContent:
    """Test KnowledgeDataLoader.load() with sensitive content in pieces.

    Validates: Requirement 1.7
    """

    def test_sensitive_password_content_skips_piece(self, kb, tmp_path):
        """A piece containing password patterns is skipped."""
        data = {
            "metadata": {},
            "pieces": [
                {
                    "piece_id": "safe-piece",
                    "content": "This is safe content about groceries.",
                    "knowledge_type": "fact",
                    "info_type": "context",
                    "tags": [],
                },
                {
                    "piece_id": "sensitive-piece",
                    "content": "my password= secret123",
                    "knowledge_type": "fact",
                    "info_type": "context",
                    "tags": [],
                },
            ],
            "graph": {"nodes": [], "edges": []},
        }
        file_path = tmp_path / "sensitive.json"
        file_path.write_text(json.dumps(data), encoding="utf-8")

        result = KnowledgeDataLoader.load(kb, str(file_path))

        # Only the safe piece should be loaded
        assert result["pieces"] == 1

        # Verify the safe piece is present
        safe = kb.piece_store.get_by_id("safe-piece")
        assert safe is not None

        # Verify the sensitive piece was skipped
        sensitive = kb.piece_store.get_by_id("sensitive-piece")
        assert sensitive is None

    def test_sensitive_api_key_content_skips_piece(self, kb, tmp_path):
        """A piece containing API key patterns is skipped."""
        data = {
            "metadata": {},
            "pieces": [
                {
                    "piece_id": "api-key-piece",
                    "content": "api_key= abc123xyz",
                    "knowledge_type": "fact",
                    "info_type": "context",
                    "tags": [],
                },
            ],
            "graph": {"nodes": [], "edges": []},
        }
        file_path = tmp_path / "api_key.json"
        file_path.write_text(json.dumps(data), encoding="utf-8")

        result = KnowledgeDataLoader.load(kb, str(file_path))

        assert result["pieces"] == 0
        assert kb.piece_store.get_by_id("api-key-piece") is None

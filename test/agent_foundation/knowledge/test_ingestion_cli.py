"""
Unit tests for KnowledgeIngestionCLI.

Tests the CLI tool for ingesting free-form text into a KnowledgeBase via
LLM structuring, including valid/invalid JSON handling, schema validation,
STRUCTURING_PROMPT content, check-in saving, and store path behavior.

Requirements: 8.1, 8.2, 8.3, 8.7, 8.9
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

from science_modeling_tools.knowledge import KnowledgeIngestionCLI, KnowledgeBase
from science_modeling_tools.knowledge.ingestion_cli import STRUCTURING_PROMPT
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


# ── Helpers ──────────────────────────────────────────────────────────────────

VALID_JSON_RESPONSE = json.dumps({
    "metadata": {
        "user:test": {
            "entity_type": "user",
            "properties": {"FirstName": "Test", "LastName": "User"},
        }
    },
    "pieces": [
        {
            "piece_id": "test-fact",
            "content": "Test user is a member of TestService",
            "knowledge_type": "fact",
            "info_type": "user_profile",
            "tags": ["test"],
            "entity_id": "user:test",
            "embedding_text": "test membership service",
        }
    ],
    "graph": {
        "nodes": [
            {
                "node_id": "user:test",
                "node_type": "user",
                "label": "Test User",
                "properties": {},
            }
        ],
        "edges": [],
    },
})


def _make_kb():
    """Create a KnowledgeBase with in-memory stores for testing."""
    return KnowledgeBase(
        metadata_store=KeyValueMetadataStore(kv_service=MemoryKeyValueService()),
        piece_store=RetrievalKnowledgePieceStore(
            retrieval_service=MemoryRetrievalService()
        ),
        graph_store=GraphServiceEntityGraphStore(
            graph_service=MemoryGraphService()
        ),
        active_entity_id="user:test",
    )


# ── Test: Ingest with valid JSON populates KB ────────────────────────────────


class TestIngestValidJson:
    """Test ingest() with a mock inferencer returning valid JSON."""

    def test_ingest_valid_json_populates_kb(self):
        """Mock inferencer returns VALID_JSON_RESPONSE; verify KB is populated."""
        mock_inferencer = lambda prompt: VALID_JSON_RESPONSE
        cli = KnowledgeIngestionCLI(inferencer=mock_inferencer)
        kb = _make_kb()

        counts = cli.ingest("My name is Test User", kb)

        assert counts["metadata"] >= 1
        assert counts["pieces"] >= 1
        assert counts["graph_nodes"] >= 1


# ── Test: Ingest with invalid JSON retries and raises ────────────────────────


class TestIngestInvalidJson:
    """Test ingest() with a mock inferencer that always returns invalid JSON."""

    def test_ingest_invalid_json_retries_and_raises(self):
        """Mock inferencer always returns 'not json'; verify ValueError after max_retries."""
        mock_inferencer = lambda prompt: "not json"
        cli = KnowledgeIngestionCLI(inferencer=mock_inferencer, max_retries=2)
        kb = _make_kb()

        with pytest.raises(ValueError, match="Failed to get valid structured data"):
            cli.ingest("Some text", kb)


# ── Test: _parse_and_validate ────────────────────────────────────────────────


class TestParseAndValidate:
    """Test _parse_and_validate accepts valid schema and rejects invalid ones."""

    def test_parse_and_validate_accepts_valid(self):
        """Valid JSON string with all required sections is accepted."""
        cli = KnowledgeIngestionCLI(inferencer=lambda p: "")
        result = cli._parse_and_validate(VALID_JSON_RESPONSE)

        assert "metadata" in result
        assert "pieces" in result
        assert "graph" in result

    def test_parse_and_validate_rejects_missing_metadata(self):
        """Missing 'metadata' section raises ValueError."""
        cli = KnowledgeIngestionCLI(inferencer=lambda p: "")
        data = {"pieces": [], "graph": {"nodes": [], "edges": []}}

        with pytest.raises(ValueError, match="Missing required sections.*metadata"):
            cli._parse_and_validate(json.dumps(data))

    def test_parse_and_validate_rejects_missing_pieces(self):
        """Missing 'pieces' section raises ValueError."""
        cli = KnowledgeIngestionCLI(inferencer=lambda p: "")
        data = {"metadata": {}, "graph": {"nodes": [], "edges": []}}

        with pytest.raises(ValueError, match="Missing required sections.*pieces"):
            cli._parse_and_validate(json.dumps(data))

    def test_parse_and_validate_rejects_missing_graph(self):
        """Missing 'graph' section raises ValueError."""
        cli = KnowledgeIngestionCLI(inferencer=lambda p: "")
        data = {"metadata": {}, "pieces": []}

        with pytest.raises(ValueError, match="Missing required sections.*graph"):
            cli._parse_and_validate(json.dumps(data))

    def test_parse_and_validate_rejects_piece_missing_fields(self):
        """Piece missing piece_id raises ValueError."""
        cli = KnowledgeIngestionCLI(inferencer=lambda p: "")
        data = {
            "metadata": {},
            "pieces": [
                {
                    "content": "some content",
                    "knowledge_type": "fact",
                    "info_type": "user_profile",
                    # missing piece_id
                }
            ],
            "graph": {"nodes": [], "edges": []},
        }

        with pytest.raises(ValueError, match="missing required fields.*piece_id"):
            cli._parse_and_validate(json.dumps(data))


# ── Test: STRUCTURING_PROMPT contains required schema ────────────────────────


class TestStructuringPrompt:
    """Test that STRUCTURING_PROMPT contains required schema structure."""

    def test_structuring_prompt_contains_schema(self):
        """STRUCTURING_PROMPT contains 'metadata', 'pieces', 'graph', 'knowledge_type', 'info_type'."""
        assert "metadata" in STRUCTURING_PROMPT
        assert "pieces" in STRUCTURING_PROMPT
        assert "graph" in STRUCTURING_PROMPT
        assert "knowledge_type" in STRUCTURING_PROMPT
        assert "info_type" in STRUCTURING_PROMPT


# ── Test: _save_check_in creates files ───────────────────────────────────────


class TestSaveCheckIn:
    """Test _save_check_in creates folder with raw_input.txt and structured.json."""

    def test_save_check_in_creates_files(self, tmp_path):
        """Use tmp_path; verify raw_input.txt and structured.json are created."""
        cli = KnowledgeIngestionCLI(
            inferencer=lambda p: "",
            raw_files_store_path=str(tmp_path),
        )
        structured = json.loads(VALID_JSON_RESPONSE)

        result = cli._save_check_in("Hello world", structured)

        assert result is not None
        checkin_dir = Path(result)
        assert checkin_dir.exists()
        assert (checkin_dir / "raw_input.txt").exists()
        assert (checkin_dir / "structured.json").exists()

        raw_text = (checkin_dir / "raw_input.txt").read_text(encoding="utf-8")
        assert raw_text == "Hello world"

        saved_json = json.loads(
            (checkin_dir / "structured.json").read_text(encoding="utf-8")
        )
        assert saved_json == structured


# ── Test: Ingest with store path saves check-in ─────────────────────────────


class TestIngestWithStorePath:
    """Test ingest with raw_files_store_path saves check-in before loading."""

    def test_ingest_with_store_path_saves_checkin(self, tmp_path):
        """Mock inferencer + tmp_path; verify check-in folder is created."""
        mock_inferencer = lambda prompt: VALID_JSON_RESPONSE
        cli = KnowledgeIngestionCLI(
            inferencer=mock_inferencer,
            raw_files_store_path=str(tmp_path),
        )
        kb = _make_kb()

        cli.ingest("My name is Test User", kb)

        # At least one check-in folder should exist
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(subdirs) == 1
        checkin_dir = subdirs[0]
        assert (checkin_dir / "raw_input.txt").exists()
        assert (checkin_dir / "structured.json").exists()


# ── Test: Ingest without store path skips saving ─────────────────────────────


class TestIngestWithoutStorePath:
    """Test ingest without raw_files_store_path skips saving (no error)."""

    def test_ingest_without_store_path_skips_saving(self, tmp_path):
        """No raw_files_store_path; verify no error and no folders created."""
        mock_inferencer = lambda prompt: VALID_JSON_RESPONSE
        cli = KnowledgeIngestionCLI(inferencer=mock_inferencer)
        kb = _make_kb()

        # Should not raise
        counts = cli.ingest("My name is Test User", kb)

        assert counts["metadata"] >= 1

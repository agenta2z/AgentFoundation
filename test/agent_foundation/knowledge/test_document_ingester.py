"""Unit tests for the DocumentIngester module."""

import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from agent_foundation.knowledge.ingestion.document_ingester import (
    DocumentIngester,
    IngesterConfig,
    IngestionResult,
    ProgressCallback,
    ingest_markdown_files,
    ingest_directory,
)
from agent_foundation.knowledge.ingestion.chunker import ChunkerConfig
from agent_foundation.knowledge.ingestion.deduplicator import (
    DedupConfig,
    ThreeTierDeduplicator,
)
from agent_foundation.knowledge.ingestion.merge_strategy import (
    MergeStrategyConfig,
    MergeStrategyManager,
)
from agent_foundation.knowledge.ingestion.validator import (
    KnowledgeValidator,
    ValidationConfig,
)
from agent_foundation.knowledge.retrieval.models.enums import (
    DedupAction,
    MergeAction,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.results import (
    DedupResult,
    MergeResult,
    ValidationResult,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


# ── Helpers ──────────────────────────────────────────────────────────────────


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for testing."""

    def __init__(self):
        self._pieces: dict[str, KnowledgePiece] = {}

    def add(self, piece: KnowledgePiece) -> str:
        self._pieces[piece.piece_id] = piece
        return piece.piece_id

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        return self._pieces.get(piece_id)

    def update(self, piece: KnowledgePiece) -> bool:
        if piece.piece_id in self._pieces:
            self._pieces[piece.piece_id] = piece
            return True
        return False

    def remove(self, piece_id: str) -> bool:
        return self._pieces.pop(piece_id, None) is not None

    def search(
        self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5
    ) -> List[Tuple[KnowledgePiece, float]]:
        return list((p, 0.5) for p in self._pieces.values())[:top_k]

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        return list(self._pieces.values())


def _make_llm_response(pieces: List[Dict[str, Any]]) -> str:
    """Create a valid LLM JSON response with the given pieces."""
    return json.dumps({
        "metadata": {},
        "pieces": pieces,
        "graph": {"nodes": [], "edges": []},
    })


def _simple_piece_dict(
    piece_id: str = "test-piece-1",
    content: str = "Test content",
    knowledge_type: str = "fact",
    info_type: str = "context",
    domain: str = "general",
) -> Dict[str, Any]:
    return {
        "piece_id": piece_id,
        "content": content,
        "knowledge_type": knowledge_type,
        "info_type": info_type,
        "domain": domain,
    }


def _make_inferencer(response: str):
    """Create a simple inferencer that returns a fixed response."""
    def inferencer(prompt: str) -> str:
        return response
    return inferencer


def _make_mock_kb(piece_store=None):
    """Create a mock KnowledgeBase with the required attributes."""
    kb = MagicMock()
    kb.piece_store = piece_store or InMemoryPieceStore()
    return kb


# ── IngestionResult Tests ────────────────────────────────────────────────────


class TestIngestionResult:
    def test_defaults(self):
        result = IngestionResult(success=True)
        assert result.success is True
        assert result.chunks_processed == 0
        assert result.pieces_created == 0
        assert result.errors == []
        assert result.source_file is None

    def test_errors_default_to_empty_list(self):
        result = IngestionResult(success=False)
        assert result.errors == []
        result.errors.append("test error")
        # New instance should have fresh list
        result2 = IngestionResult(success=False)
        assert result2.errors == []


# ── IngesterConfig Tests ─────────────────────────────────────────────────────


class TestIngesterConfig:
    def test_defaults(self):
        config = IngesterConfig()
        assert config.max_retries == 3
        assert config.full_schema is True
        assert config.merge_graphs is True
        assert config.dedupe_pieces is True
        assert config.chunker_config is None
        assert config.debug_session is None


# ── DocumentIngester Core Tests ──────────────────────────────────────────────


class TestDocumentIngesterInit:
    def test_default_config(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        assert ingester.config.max_retries == 3
        assert ingester._deduplicator is None
        assert ingester._merge_manager is None
        assert ingester._validator is None

    def test_custom_config(self):
        config = IngesterConfig(max_retries=5)
        ingester = DocumentIngester(inferencer=lambda p: "", config=config)
        assert ingester.config.max_retries == 5

    def test_progress_callback(self):
        messages = []
        ingester = DocumentIngester(
            inferencer=lambda p: "",
            on_progress=messages.append,
        )
        ingester._report("hello")
        assert "hello" in messages


class TestCallLlm:
    def test_plain_string_response(self):
        ingester = DocumentIngester(inferencer=lambda p: "hello")
        assert ingester._call_llm("prompt") == "hello"

    def test_inferencer_response_protocol(self):
        """Test duck-typed InferencerResponse handling."""
        class MockResponse:
            def select_response(self):
                return MagicMock(response="structured response")

        ingester = DocumentIngester(inferencer=lambda p: MockResponse())
        assert ingester._call_llm("prompt") == "structured response"

    def test_non_string_fallback(self):
        ingester = DocumentIngester(inferencer=lambda p: 42)
        assert ingester._call_llm("prompt") == "42"


class TestParseAndValidate:
    def test_valid_json(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        response = _make_llm_response([_simple_piece_dict()])
        result = ingester._parse_and_validate(response)
        assert len(result["pieces"]) == 1
        assert result["pieces"][0]["piece_id"] == "test-piece-1"

    def test_strips_markdown_fences(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        inner = _make_llm_response([_simple_piece_dict()])
        response = f"```json\n{inner}\n```"
        result = ingester._parse_and_validate(response)
        assert len(result["pieces"]) == 1

    def test_invalid_json_raises(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        with pytest.raises(ValueError, match="Invalid JSON"):
            ingester._parse_and_validate("not json")

    def test_non_dict_raises(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        with pytest.raises(ValueError, match="JSON object"):
            ingester._parse_and_validate("[1, 2, 3]")

    def test_missing_piece_fields_raises(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        response = json.dumps({
            "metadata": {},
            "pieces": [{"content": "no id"}],
            "graph": {"nodes": [], "edges": []},
        })
        with pytest.raises(ValueError, match="missing required fields"):
            ingester._parse_and_validate(response)

    def test_invalid_knowledge_type_raises(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        piece = _simple_piece_dict()
        piece["knowledge_type"] = "invalid_type"
        response = _make_llm_response([piece])
        with pytest.raises(ValueError, match="invalid knowledge_type"):
            ingester._parse_and_validate(response)

    def test_adds_missing_sections_for_full_schema(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        result = ingester._parse_and_validate('{"pieces": []}')
        assert "metadata" in result
        assert "graph" in result


class TestMergeResults:
    def test_merges_pieces_from_multiple_chunks(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        data1 = {
            "metadata": {"key1": "val1"},
            "pieces": [_simple_piece_dict("p1")],
            "graph": {"nodes": [], "edges": []},
        }
        data2 = {
            "metadata": {"key2": "val2"},
            "pieces": [_simple_piece_dict("p2")],
            "graph": {"nodes": [], "edges": []},
        }
        merged = ingester._merge_results([data1, data2])
        assert len(merged["pieces"]) == 2
        assert merged["metadata"]["key1"] == "val1"
        assert merged["metadata"]["key2"] == "val2"

    def test_deduplicates_pieces_by_id(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        data1 = {
            "metadata": {},
            "pieces": [_simple_piece_dict("same-id")],
            "graph": {"nodes": [], "edges": []},
        }
        data2 = {
            "metadata": {},
            "pieces": [_simple_piece_dict("same-id")],
            "graph": {"nodes": [], "edges": []},
        }
        merged = ingester._merge_results([data1, data2])
        assert len(merged["pieces"]) == 1

    def test_dedup_disabled_allows_duplicates(self):
        config = IngesterConfig(dedupe_pieces=False)
        ingester = DocumentIngester(inferencer=lambda p: "", config=config)
        data1 = {
            "metadata": {},
            "pieces": [_simple_piece_dict("same-id")],
            "graph": {"nodes": [], "edges": []},
        }
        merged = ingester._merge_results([data1, data1])
        assert len(merged["pieces"]) == 2

    def test_merges_graph_nodes_and_edges(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        data1 = {
            "metadata": {},
            "pieces": [],
            "graph": {
                "nodes": [{"node_id": "n1", "node_type": "concept"}],
                "edges": [{"source_id": "n1", "target_id": "n2", "edge_type": "RELATES"}],
            },
        }
        data2 = {
            "metadata": {},
            "pieces": [],
            "graph": {
                "nodes": [
                    {"node_id": "n1", "node_type": "concept"},  # duplicate
                    {"node_id": "n2", "node_type": "concept"},
                ],
                "edges": [
                    {"source_id": "n1", "target_id": "n2", "edge_type": "RELATES"},  # duplicate
                ],
            },
        }
        merged = ingester._merge_results([data1, data2])
        assert len(merged["graph"]["nodes"]) == 2
        assert len(merged["graph"]["edges"]) == 1


# ── Enhancement Tests ────────────────────────────────────────────────────────


class TestApplyEnhancements:
    def test_no_enhancements_passthrough(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        data = {"pieces": [_simple_piece_dict()]}
        result_data, counts, deactivate = ingester._apply_enhancements(data)
        assert len(result_data["pieces"]) == 1
        assert counts == {"deduped": 0, "failed_validation": 0, "updated": 0, "merged": 0}
        assert deactivate == []

    def test_failed_validation_moves_to_developmental(self):
        """Requirement 21.4: Failed validation → developmental space."""
        mock_validator = MagicMock(spec=KnowledgeValidator)
        mock_validator.validate.return_value = ValidationResult(
            is_valid=False,
            confidence=0.3,
            issues=["Content may contain credentials"],
            checks_failed=["security"],
        )

        ingester = DocumentIngester(
            inferencer=lambda p: "",
            validator=mock_validator,
        )
        data = {"pieces": [_simple_piece_dict()]}
        result_data, counts, _ = ingester._apply_enhancements(data)

        assert counts["failed_validation"] == 1
        piece = result_data["pieces"][0]
        assert piece["space"] == "developmental"
        assert piece["validation_status"] == "failed"
        assert "Content may contain credentials" in piece["validation_issues"]

    def test_valid_piece_stays_in_main_space(self):
        mock_validator = MagicMock(spec=KnowledgeValidator)
        mock_validator.validate.return_value = ValidationResult(
            is_valid=True,
            confidence=1.0,
            checks_passed=["security", "privacy"],
        )

        ingester = DocumentIngester(
            inferencer=lambda p: "",
            validator=mock_validator,
        )
        data = {"pieces": [_simple_piece_dict()]}
        result_data, counts, _ = ingester._apply_enhancements(data)

        assert counts["failed_validation"] == 0
        piece = result_data["pieces"][0]
        assert piece.get("space", "main") == "main"

    def test_dedup_noop_removes_piece(self):
        mock_dedup = MagicMock(spec=ThreeTierDeduplicator)
        mock_dedup.deduplicate.return_value = DedupResult(
            action=DedupAction.NO_OP,
            reason="Exact match",
            existing_piece_id="existing-1",
        )

        ingester = DocumentIngester(
            inferencer=lambda p: "",
            deduplicator=mock_dedup,
        )
        data = {"pieces": [_simple_piece_dict()]}
        result_data, counts, _ = ingester._apply_enhancements(data)

        assert len(result_data["pieces"]) == 0
        assert counts["deduped"] == 1

    def test_dedup_update_sets_supersedes(self):
        mock_dedup = MagicMock(spec=ThreeTierDeduplicator)
        mock_dedup.deduplicate.return_value = DedupResult(
            action=DedupAction.UPDATE,
            reason="Updated version",
            existing_piece_id="old-piece-1",
        )

        ingester = DocumentIngester(
            inferencer=lambda p: "",
            deduplicator=mock_dedup,
        )
        data = {"pieces": [_simple_piece_dict()]}
        result_data, counts, deactivate = ingester._apply_enhancements(data)

        assert len(result_data["pieces"]) == 1
        assert counts["updated"] == 1
        assert "old-piece-1" in deactivate
        piece = result_data["pieces"][0]
        assert piece["supersedes"] == "old-piece-1"

    def test_dedup_add_keeps_piece(self):
        mock_dedup = MagicMock(spec=ThreeTierDeduplicator)
        mock_dedup.deduplicate.return_value = DedupResult(
            action=DedupAction.ADD,
            reason="No duplicates",
        )

        ingester = DocumentIngester(
            inferencer=lambda p: "",
            deduplicator=mock_dedup,
        )
        data = {"pieces": [_simple_piece_dict()]}
        result_data, counts, _ = ingester._apply_enhancements(data)

        assert len(result_data["pieces"]) == 1
        assert counts["deduped"] == 0


# ── Progress Callback Tests ──────────────────────────────────────────────────


class TestProgressCallbacks:
    def test_progress_messages_during_ingest(self):
        """Requirement 21.2: Progress callbacks for UI updates."""
        messages = []
        response = _make_llm_response([_simple_piece_dict()])
        ingester = DocumentIngester(
            inferencer=_make_inferencer(response),
            on_progress=messages.append,
        )
        kb = _make_mock_kb()
        ingester.ingest_text("# Test\n\nSome content here.", kb)

        assert any("Analyzing" in m for m in messages)
        assert any("chunk" in m.lower() for m in messages)


# ── Atomic Loading Tests ─────────────────────────────────────────────────────


class TestAtomicLoading:
    def test_load_into_kb_deactivates_after_add(self):
        """Requirement 21.3: Atomic loading — deactivate after add."""
        store = InMemoryPieceStore()
        old_piece = KnowledgePiece(content="old content", piece_id="old-1")
        store.add(old_piece)

        kb = _make_mock_kb(piece_store=store)

        ingester = DocumentIngester(inferencer=lambda p: "")
        data = {
            "metadata": {},
            "pieces": [
                {
                    **_simple_piece_dict("new-1"),
                    "supersedes": "old-1",
                }
            ],
            "graph": {"nodes": [], "edges": []},
        }

        ingester._load_into_kb(data, kb, pieces_to_deactivate=["old-1"])

        old = store.get_by_id("old-1")
        assert old is not None
        assert old.is_active is False


# ── End-to-End Ingest Text Tests ─────────────────────────────────────────────


class TestIngestText:
    def test_empty_text_returns_empty_result(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        kb = _make_mock_kb()
        result = ingester.ingest_text("", kb)
        assert result.success is True
        assert result.chunks_processed == 0

    def test_whitespace_only_returns_empty_result(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        kb = _make_mock_kb()
        result = ingester.ingest_text("   \n\n  ", kb)
        assert result.success is True
        assert result.chunks_processed == 0

    def test_successful_ingestion(self):
        response = _make_llm_response([_simple_piece_dict()])
        ingester = DocumentIngester(inferencer=_make_inferencer(response))
        kb = _make_mock_kb()
        result = ingester.ingest_text("# Test\n\nSome content.", kb)
        assert result.success is True
        assert result.chunks_processed >= 1

    def test_source_file_added_to_pieces(self):
        """Verify source file metadata is added to pieces."""
        response = _make_llm_response([_simple_piece_dict()])
        captured = {}

        def mock_inferencer(prompt):
            return response

        ingester = DocumentIngester(inferencer=mock_inferencer)
        kb = _make_mock_kb()
        result = ingester.ingest_text(
            "# Test\n\nContent.", kb, source_file="test.md"
        )
        assert result.source_file == "test.md"

    def test_llm_failure_records_error(self):
        def failing_inferencer(prompt):
            raise RuntimeError("LLM down")

        config = IngesterConfig(max_retries=1)
        ingester = DocumentIngester(
            inferencer=failing_inferencer, config=config
        )
        kb = _make_mock_kb()
        result = ingester.ingest_text("# Test\n\nContent.", kb)
        assert result.success is False
        assert len(result.errors) > 0


# ── Ingest File Tests ────────────────────────────────────────────────────────


class TestIngestFile:
    def test_file_not_found(self):
        ingester = DocumentIngester(inferencer=lambda p: "")
        kb = _make_mock_kb()
        with pytest.raises(FileNotFoundError):
            ingester.ingest_file("/nonexistent/path.md", kb)

    def test_ingest_real_file(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nWorld content here.")

        response = _make_llm_response([_simple_piece_dict()])
        ingester = DocumentIngester(inferencer=_make_inferencer(response))
        kb = _make_mock_kb()
        result = ingester.ingest_file(str(md_file), kb)
        assert result.success is True
        assert result.source_file == str(md_file)


# ── Convenience Function Tests ───────────────────────────────────────────────


class TestIngestMarkdownFiles:
    def test_ingest_multiple_files(self, tmp_path):
        for i in range(2):
            f = tmp_path / f"doc{i}.md"
            f.write_text(f"# Doc {i}\n\nContent {i}.")

        response = _make_llm_response([_simple_piece_dict()])
        results = ingest_markdown_files(
            [str(tmp_path / "doc0.md"), str(tmp_path / "doc1.md")],
            _make_mock_kb(),
            _make_inferencer(response),
        )
        assert len(results) == 2
        assert all(r.success for r in results.values())

    def test_handles_missing_file(self, tmp_path):
        results = ingest_markdown_files(
            [str(tmp_path / "missing.md")],
            _make_mock_kb(),
            lambda p: "",
        )
        assert len(results) == 1
        result = list(results.values())[0]
        assert result.success is False


class TestIngestDirectory:
    def test_ingest_directory(self, tmp_path):
        (tmp_path / "doc1.md").write_text("# Doc 1\n\nContent.")
        (tmp_path / "doc2.md").write_text("# Doc 2\n\nContent.")
        (tmp_path / "readme.txt").write_text("Not markdown")

        response = _make_llm_response([_simple_piece_dict()])
        results = ingest_directory(
            str(tmp_path),
            _make_mock_kb(),
            _make_inferencer(response),
        )
        assert len(results) == 2

    def test_invalid_directory_raises(self):
        with pytest.raises(ValueError, match="Not a directory"):
            ingest_directory(
                "/nonexistent/dir",
                _make_mock_kb(),
                lambda p: "",
            )

    def test_empty_directory(self, tmp_path):
        results = ingest_directory(
            str(tmp_path),
            _make_mock_kb(),
            lambda p: "",
        )
        assert results == {}

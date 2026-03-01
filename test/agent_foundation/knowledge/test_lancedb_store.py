"""Tests for LanceDB store serialization and find_by_content_hash.

Tests the _piece_to_record and _record_to_piece functions for correct
serialization/deserialization of all KnowledgePiece fields, including
new fields stored as JSON strings. Also tests the find_by_content_hash
override using SQL WHERE clause.

Requirements: 23.3, 23.4, 23.5
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.lancedb_store import (
    _piece_to_record,
    _record_to_piece,
    _GLOBAL_ENTITY_SENTINEL,
    _escape_sql_like,
    _build_where_clause,
)


class TestPieceToRecord:
    """Tests for _piece_to_record serialization."""

    def test_includes_new_scalar_fields(self):
        piece = KnowledgePiece(
            content="test content",
            domain="model_optimization",
            content_hash="abc123",
            space="personal",
            is_active=False,
            version=3,
            summary="A summary",
            validation_status="passed",
            merge_strategy="auto-merge-on-ingest",
            supersedes="old-piece-id",
        )
        record = _piece_to_record(piece, [0.1, 0.2])

        assert record["domain"] == "model_optimization"
        assert record["content_hash"] == "abc123"
        assert record["space"] == "personal"
        assert record["is_active"] is False
        assert record["version"] == 3
        assert record["summary"] == "A summary"
        assert record["validation_status"] == "passed"
        assert record["merge_strategy"] == "auto-merge-on-ingest"
        assert record["supersedes"] == "old-piece-id"

    def test_list_fields_stored_as_json_strings(self):
        piece = KnowledgePiece(
            content="test content",
            secondary_domains=["data_engineering", "testing"],
            custom_tags=["gpu", "optimization"],
            validation_issues=["stale content", "missing source"],
        )
        record = _piece_to_record(piece, [0.1])

        assert isinstance(record["secondary_domains"], str)
        assert json.loads(record["secondary_domains"]) == ["data_engineering", "testing"]
        assert isinstance(record["custom_tags"], str)
        assert json.loads(record["custom_tags"]) == ["gpu", "optimization"]
        assert isinstance(record["validation_issues"], str)
        assert json.loads(record["validation_issues"]) == ["stale content", "missing source"]

    def test_defaults_for_none_optional_fields(self):
        piece = KnowledgePiece(content="test")
        record = _piece_to_record(piece, [0.0])

        assert record["domain"] == "general"
        assert record["space"] == "main"
        assert record["summary"] == ""
        assert record["merge_strategy"] == ""
        assert record["supersedes"] == ""
        assert record["validation_status"] == "not_validated"

    def test_empty_lists_stored_as_json(self):
        piece = KnowledgePiece(content="test")
        record = _piece_to_record(piece, [0.0])

        assert json.loads(record["secondary_domains"]) == []
        assert json.loads(record["custom_tags"]) == []
        assert json.loads(record["validation_issues"]) == []


class TestRecordToPiece:
    """Tests for _record_to_piece deserialization."""

    def test_deserializes_new_scalar_fields(self):
        record = {
            "content": "test content",
            "piece_id": "p1",
            "knowledge_type": "fact",
            "tags": "[]",
            "entity_id": _GLOBAL_ENTITY_SENTINEL,
            "source": "",
            "embedding_text": "",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "domain": "testing",
            "content_hash": "abc123def456",
            "space": "developmental",
            "is_active": False,
            "version": 2,
            "summary": "A summary",
            "validation_status": "failed",
            "merge_strategy": "manual-only",
            "supersedes": "old-id",
            "secondary_domains": "[]",
            "custom_tags": "[]",
            "validation_issues": "[]",
        }
        piece = _record_to_piece(record)

        assert piece.domain == "testing"
        assert piece.content_hash == "abc123def456"
        assert piece.space == "developmental"
        assert piece.is_active is False
        assert piece.version == 2
        assert piece.summary == "A summary"
        assert piece.validation_status == "failed"
        assert piece.merge_strategy == "manual-only"
        assert piece.supersedes == "old-id"

    def test_deserializes_json_list_fields(self):
        record = {
            "content": "test",
            "piece_id": "p1",
            "knowledge_type": "fact",
            "tags": "[]",
            "entity_id": _GLOBAL_ENTITY_SENTINEL,
            "source": "",
            "embedding_text": "",
            "created_at": "",
            "updated_at": "",
            "secondary_domains": json.dumps(["infra", "testing"]),
            "custom_tags": json.dumps(["gpu"]),
            "validation_issues": json.dumps(["issue1"]),
        }
        piece = _record_to_piece(record)

        assert piece.secondary_domains == ["infra", "testing"]
        assert piece.custom_tags == ["gpu"]
        assert piece.validation_issues == ["issue1"]

    def test_defaults_for_missing_new_fields(self):
        """Records without new fields should get defaults."""
        record = {
            "content": "test",
            "piece_id": "p1",
            "knowledge_type": "fact",
            "tags": "[]",
            "entity_id": _GLOBAL_ENTITY_SENTINEL,
            "source": "",
            "embedding_text": "",
            "created_at": "",
            "updated_at": "",
        }
        piece = _record_to_piece(record)

        assert piece.domain == "general"
        assert piece.space == "main"
        assert piece.is_active is True
        assert piece.version == 1
        assert piece.validation_status == "not_validated"
        assert piece.secondary_domains == []
        assert piece.custom_tags == []
        assert piece.validation_issues == []
        assert piece.summary is None
        assert piece.merge_strategy is None
        assert piece.supersedes is None

    def test_empty_string_optional_fields_become_none(self):
        record = {
            "content": "test",
            "piece_id": "p1",
            "knowledge_type": "fact",
            "tags": "[]",
            "entity_id": _GLOBAL_ENTITY_SENTINEL,
            "source": "",
            "embedding_text": "",
            "created_at": "",
            "updated_at": "",
            "summary": "",
            "merge_strategy": "",
            "supersedes": "",
            "content_hash": "",
        }
        piece = _record_to_piece(record)

        assert piece.summary is None
        assert piece.merge_strategy is None
        assert piece.supersedes is None
        # content_hash will be recomputed by __attrs_post_init__ when None
        assert piece.content_hash is not None


class TestRoundTrip:
    """Tests for _piece_to_record → _record_to_piece round trip."""

    def test_round_trip_preserves_all_fields(self):
        piece = KnowledgePiece(
            content="round trip test",
            knowledge_type=KnowledgeType.Procedure,
            info_type="skills",
            tags=["ml", "gpu"],
            entity_id="user-1",
            source="doc.md",
            domain="model_optimization",
            secondary_domains=["training_efficiency"],
            custom_tags=["cuda"],
            space="personal",
            is_active=False,
            version=5,
            summary="A procedure",
            validation_status="passed",
            validation_issues=["minor issue"],
            merge_strategy="auto-merge-on-ingest",
            supersedes="prev-id",
        )
        vector = [0.1, 0.2, 0.3]
        record = _piece_to_record(piece, vector)
        restored = _record_to_piece(record)

        assert restored.content == piece.content
        assert restored.piece_id == piece.piece_id
        assert restored.knowledge_type == piece.knowledge_type
        assert restored.tags == piece.tags
        assert restored.entity_id == piece.entity_id
        assert restored.source == piece.source
        assert restored.domain == piece.domain
        assert restored.secondary_domains == piece.secondary_domains
        assert restored.custom_tags == piece.custom_tags
        assert restored.content_hash == piece.content_hash
        assert restored.space == piece.space
        assert restored.is_active == piece.is_active
        assert restored.version == piece.version
        assert restored.summary == piece.summary
        assert restored.validation_status == piece.validation_status
        assert restored.validation_issues == piece.validation_issues
        assert restored.merge_strategy == piece.merge_strategy
        assert restored.supersedes == piece.supersedes


class TestFindByContentHash:
    """Tests for LanceDBKnowledgePieceStore.find_by_content_hash override."""

    def test_returns_none_when_table_is_none(self):
        """find_by_content_hash returns None when no table exists."""
        with patch("agent_foundation.knowledge.retrieval.stores.pieces.lancedb_store.lancedb", create=True):
            store = MagicMock()
            store._table = None
            # Call the unbound method logic directly
            from agent_foundation.knowledge.retrieval.stores.pieces.lancedb_store import LanceDBKnowledgePieceStore
            result = LanceDBKnowledgePieceStore.find_by_content_hash(store, "abc123")
            assert result is None

    def test_returns_none_for_empty_hash(self):
        """find_by_content_hash returns None for empty content_hash."""
        store = MagicMock()
        store._table = MagicMock()
        from agent_foundation.knowledge.retrieval.stores.pieces.lancedb_store import LanceDBKnowledgePieceStore
        result = LanceDBKnowledgePieceStore.find_by_content_hash(store, "")
        assert result is None

    def test_finds_global_piece_by_hash(self):
        """find_by_content_hash finds a global piece via SQL WHERE."""
        mock_record = {
            "content": "found it",
            "piece_id": "p1",
            "knowledge_type": "fact",
            "tags": "[]",
            "entity_id": _GLOBAL_ENTITY_SENTINEL,
            "source": "",
            "embedding_text": "",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "content_hash": "abc123",
            "domain": "general",
            "space": "main",
            "is_active": True,
            "version": 1,
            "summary": "",
            "validation_status": "not_validated",
            "merge_strategy": "",
            "supersedes": "",
            "secondary_domains": "[]",
            "custom_tags": "[]",
            "validation_issues": "[]",
        }

        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = [mock_record]

        store = MagicMock()
        store._table = mock_table

        from agent_foundation.knowledge.retrieval.stores.pieces.lancedb_store import LanceDBKnowledgePieceStore
        result = LanceDBKnowledgePieceStore.find_by_content_hash(store, "abc123")

        assert result is not None
        assert result.content == "found it"
        assert result.piece_id == "p1"

    def test_falls_back_to_entity_search(self):
        """find_by_content_hash checks entity-scoped pieces when global returns nothing."""
        mock_record = {
            "content": "entity piece",
            "piece_id": "p2",
            "knowledge_type": "fact",
            "tags": "[]",
            "entity_id": "user-1",
            "source": "",
            "embedding_text": "",
            "created_at": "",
            "updated_at": "",
            "content_hash": "def456",
            "domain": "general",
            "space": "main",
            "is_active": True,
            "version": 1,
            "summary": "",
            "validation_status": "not_validated",
            "merge_strategy": "",
            "supersedes": "",
            "secondary_domains": "[]",
            "custom_tags": "[]",
            "validation_issues": "[]",
        }

        mock_table = MagicMock()
        # First call (global) returns empty, second call (entity) returns result
        mock_table.search.return_value.where.return_value.limit.return_value.to_list.side_effect = [
            [],  # global search
            [mock_record],  # entity search
        ]

        store = MagicMock()
        store._table = mock_table

        from agent_foundation.knowledge.retrieval.stores.pieces.lancedb_store import LanceDBKnowledgePieceStore
        result = LanceDBKnowledgePieceStore.find_by_content_hash(store, "def456", entity_id="user-1")

        assert result is not None
        assert result.content == "entity piece"


class TestPieceToRecordSpaces:
    """Tests for _piece_to_record multi-space serialization (Task 3.1)."""

    def test_stores_spaces_as_json_string(self):
        piece = KnowledgePiece(content="test", spaces=["personal", "main"])
        record = _piece_to_record(piece, [0.1])

        assert isinstance(record["spaces"], str)
        assert json.loads(record["spaces"]) == ["personal", "main"]

    def test_stores_primary_space_as_scalar(self):
        piece = KnowledgePiece(content="test", spaces=["personal", "main"])
        record = _piece_to_record(piece, [0.1])

        assert record["primary_space"] == "personal"

    def test_default_spaces_produces_main(self):
        piece = KnowledgePiece(content="test")
        record = _piece_to_record(piece, [0.1])

        assert json.loads(record["spaces"]) == ["main"]
        assert record["primary_space"] == "main"

    def test_suggestion_fields_stored_as_json_when_present(self):
        piece = KnowledgePiece(
            content="test",
            pending_space_suggestions=["developmental"],
            space_suggestion_reasons=["low quality"],
            space_suggestion_status="pending",
        )
        record = _piece_to_record(piece, [0.1])

        assert json.loads(record["pending_space_suggestions"]) == ["developmental"]
        assert json.loads(record["space_suggestion_reasons"]) == ["low quality"]
        assert record["space_suggestion_status"] == "pending"

    def test_suggestion_fields_empty_string_when_none(self):
        piece = KnowledgePiece(content="test")
        record = _piece_to_record(piece, [0.1])

        assert record["pending_space_suggestions"] == ""
        assert record["space_suggestion_reasons"] == ""
        assert record["space_suggestion_status"] == ""


class TestRecordToPieceSpaces:
    """Tests for _record_to_piece multi-space deserialization (Task 3.2)."""

    def _base_record(self, **overrides):
        record = {
            "content": "test",
            "piece_id": "p1",
            "knowledge_type": "fact",
            "tags": "[]",
            "entity_id": _GLOBAL_ENTITY_SENTINEL,
            "source": "",
            "embedding_text": "",
            "created_at": "",
            "updated_at": "",
        }
        record.update(overrides)
        return record

    def test_deserializes_spaces_from_json(self):
        record = self._base_record(
            spaces=json.dumps(["personal", "main"]),
            primary_space="personal",
        )
        piece = _record_to_piece(record)

        assert piece.spaces == ["personal", "main"]

    def test_falls_back_to_space_when_spaces_absent(self):
        record = self._base_record(space="developmental")
        piece = _record_to_piece(record)

        assert piece.spaces == ["developmental"]

    def test_falls_back_to_main_when_both_absent(self):
        record = self._base_record()
        piece = _record_to_piece(record)

        assert piece.spaces == ["main"]

    def test_ignores_primary_space_column(self):
        """primary_space is derived from spaces[0], not stored on the piece."""
        record = self._base_record(
            spaces=json.dumps(["personal", "main"]),
            primary_space="main",  # intentionally wrong — should be ignored
        )
        piece = _record_to_piece(record)

        assert piece.spaces == ["personal", "main"]
        assert piece.space == "personal"  # derived from spaces[0]

    def test_deserializes_suggestion_fields(self):
        record = self._base_record(
            spaces=json.dumps(["main"]),
            primary_space="main",
            pending_space_suggestions=json.dumps(["personal"]),
            space_suggestion_reasons=json.dumps(["user entity"]),
            space_suggestion_status="pending",
        )
        piece = _record_to_piece(record)

        assert piece.pending_space_suggestions == ["personal"]
        assert piece.space_suggestion_reasons == ["user entity"]
        assert piece.space_suggestion_status == "pending"

    def test_empty_string_suggestion_fields_become_none(self):
        record = self._base_record(
            spaces=json.dumps(["main"]),
            primary_space="main",
            pending_space_suggestions="",
            space_suggestion_reasons="",
            space_suggestion_status="",
        )
        piece = _record_to_piece(record)

        assert piece.pending_space_suggestions is None
        assert piece.space_suggestion_reasons is None
        assert piece.space_suggestion_status is None


class TestEscapeSqlLike:
    """Tests for _escape_sql_like helper (Task 3.3)."""

    def test_escapes_single_quotes(self):
        assert _escape_sql_like("it's") == "it''s"

    def test_escapes_percent(self):
        assert _escape_sql_like("100%") == "100\\%"

    def test_escapes_underscore(self):
        assert _escape_sql_like("my_space") == "my\\_space"

    def test_escapes_all_wildcards(self):
        assert _escape_sql_like("it's_100%") == "it''s\\_100\\%"

    def test_plain_string_unchanged(self):
        assert _escape_sql_like("personal") == "personal"


class TestBuildWhereClauseSpaces:
    """Tests for _build_where_clause with spaces parameter (Task 3.3)."""

    def test_no_spaces_produces_no_space_condition(self):
        clause = _build_where_clause("user:1", None)
        assert "primary_space" not in clause
        assert "spaces" not in clause

    def test_single_space_produces_dual_strategy(self):
        clause = _build_where_clause("user:1", None, spaces=["personal"])
        assert "primary_space IN ('personal')" in clause
        assert "spaces LIKE" in clause
        assert '"personal"' in clause

    def test_multiple_spaces_produces_in_clause(self):
        clause = _build_where_clause("user:1", None, spaces=["personal", "main"])
        assert "primary_space IN ('personal', 'main')" in clause
        assert "spaces LIKE '%\"personal\"%'" in clause
        assert "spaces LIKE '%\"main\"%'" in clause

    def test_spaces_combined_with_entity_id(self):
        clause = _build_where_clause("user:1", None, spaces=["personal"])
        assert clause.startswith("entity_id = 'user:1'")
        assert "AND" in clause
        assert "primary_space" in clause

    def test_spaces_with_special_chars_escaped(self):
        clause = _build_where_clause(None, None, spaces=["my_space"])
        assert "my\\_space" in clause

    def test_empty_spaces_list_ignored(self):
        clause = _build_where_clause("user:1", None, spaces=[])
        assert "primary_space" not in clause

    def test_spaces_none_ignored(self):
        clause = _build_where_clause("user:1", None, spaces=None)
        assert "primary_space" not in clause


class TestRoundTripSpaces:
    """Tests for _piece_to_record → _record_to_piece round trip with spaces."""

    def test_round_trip_preserves_spaces(self):
        piece = KnowledgePiece(
            content="multi-space test",
            spaces=["personal", "main"],
        )
        record = _piece_to_record(piece, [0.1])
        restored = _record_to_piece(record)

        assert restored.spaces == piece.spaces
        assert restored.space == piece.space

    def test_round_trip_preserves_suggestion_fields(self):
        piece = KnowledgePiece(
            content="suggestion test",
            pending_space_suggestions=["developmental"],
            space_suggestion_reasons=["failed validation"],
            space_suggestion_status="pending",
        )
        record = _piece_to_record(piece, [0.1])
        restored = _record_to_piece(record)

        assert restored.pending_space_suggestions == piece.pending_space_suggestions
        assert restored.space_suggestion_reasons == piece.space_suggestion_reasons
        assert restored.space_suggestion_status == piece.space_suggestion_status

    def test_round_trip_none_suggestions_preserved(self):
        piece = KnowledgePiece(content="no suggestions")
        record = _piece_to_record(piece, [0.1])
        restored = _record_to_piece(record)

        assert restored.pending_space_suggestions is None
        assert restored.space_suggestion_reasons is None
        assert restored.space_suggestion_status is None

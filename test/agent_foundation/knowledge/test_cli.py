"""Unit tests for the knowledge CLI module."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_foundation.knowledge.cli import (
    _build_parser,
    _create_kb,
    _get_default_data_dir,
    main,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.formatter import RetrievalResult


class TestGetDefaultDataDir:
    """Tests for _get_default_data_dir."""

    def test_returns_agent_foundation_cache_path(self):
        result = _get_default_data_dir()
        assert result == Path.home() / ".cache" / "agent_foundation" / "knowledge"

    def test_returns_path_object(self):
        result = _get_default_data_dir()
        assert isinstance(result, Path)


class TestBuildParser:
    """Tests for _build_parser."""

    def test_parses_add_command(self):
        parser = _build_parser()
        args = parser.parse_args(["add", "hello world"])
        assert args.command == "add"
        assert args.text == "hello world"

    def test_parses_search_with_filters(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["search", "query", "--entity-id", "user1", "--domain", "debugging"]
        )
        assert args.command == "search"
        assert args.text == "query"
        assert args.entity_id == "user1"
        assert args.domain == "debugging"

    def test_parses_list_command(self):
        parser = _build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"
        assert args.text == ""

    def test_parses_clear_command(self):
        parser = _build_parser()
        args = parser.parse_args(["clear"])
        assert args.command == "clear"

    def test_entity_id_defaults_to_none(self):
        parser = _build_parser()
        args = parser.parse_args(["list"])
        assert args.entity_id is None

    def test_domain_defaults_to_none(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "q"])
        assert args.domain is None


class TestMainAdd:
    """Tests for the 'add' command."""

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_add_outputs_json_with_piece_id(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        mock_kb.add_piece.return_value = "piece-123"
        mock_create_kb.return_value = mock_kb

        main(["add", "some knowledge text"])

        output = json.loads(capsys.readouterr().out)
        assert output == {"ok": True, "piece_id": "piece-123"}

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_add_passes_entity_id(self, mock_create_kb):
        mock_kb = MagicMock()
        mock_kb.add_piece.return_value = "piece-456"
        mock_create_kb.return_value = mock_kb

        main(["add", "text", "--entity-id", "user1"])

        piece_arg = mock_kb.add_piece.call_args[0][0]
        assert piece_arg.entity_id == "user1"

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_add_no_text_exits_with_error(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        mock_create_kb.return_value = mock_kb

        with pytest.raises(SystemExit) as exc_info:
            main(["add"])

        assert exc_info.value.code == 1
        output = json.loads(capsys.readouterr().out)
        assert output == {"error": "No text provided"}

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_add_closes_kb(self, mock_create_kb):
        mock_kb = MagicMock()
        mock_kb.add_piece.return_value = "id"
        mock_create_kb.return_value = mock_kb

        main(["add", "text"])

        mock_kb.close.assert_called_once()


class TestMainSearch:
    """Tests for the 'search' command."""

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_search_outputs_json_items(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        piece = KnowledgePiece(content="found content", piece_id="p1")
        result = RetrievalResult()
        result.pieces = [(piece, 0.95)]
        mock_kb.retrieve.return_value = result
        mock_create_kb.return_value = mock_kb

        main(["search", "my query"])

        output = json.loads(capsys.readouterr().out)
        assert output["ok"] is True
        assert len(output["items"]) == 1
        assert output["items"][0]["piece_id"] == "p1"
        assert output["items"][0]["score"] == 0.95

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_search_passes_entity_id_and_domain(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        result = RetrievalResult()
        result.pieces = []
        mock_kb.retrieve.return_value = result
        mock_create_kb.return_value = mock_kb

        main(["search", "q", "--entity-id", "e1", "--domain", "testing"])

        mock_kb.retrieve.assert_called_once_with(
            "q", entity_id="e1", domain="testing"
        )

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_search_no_query_exits_with_error(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        mock_create_kb.return_value = mock_kb

        with pytest.raises(SystemExit) as exc_info:
            main(["search"])

        assert exc_info.value.code == 1
        output = json.loads(capsys.readouterr().out)
        assert output == {"error": "No query provided"}

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_search_truncates_content_to_200(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        piece = KnowledgePiece(content="x" * 500, piece_id="p1")
        result = RetrievalResult()
        result.pieces = [(piece, 0.5)]
        mock_kb.retrieve.return_value = result
        mock_create_kb.return_value = mock_kb

        main(["search", "q"])

        output = json.loads(capsys.readouterr().out)
        assert len(output["items"][0]["content"]) == 200


class TestMainList:
    """Tests for the 'list' command."""

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_list_outputs_all_pieces(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        pieces = [
            KnowledgePiece(content="first", piece_id="p1"),
            KnowledgePiece(content="second", piece_id="p2"),
        ]
        mock_kb.piece_store.list_all.return_value = pieces
        mock_create_kb.return_value = mock_kb

        main(["list"])

        output = json.loads(capsys.readouterr().out)
        assert output["ok"] is True
        assert len(output["items"]) == 2

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_list_passes_entity_id(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        mock_kb.piece_store.list_all.return_value = []
        mock_create_kb.return_value = mock_kb

        main(["list", "--entity-id", "user1"])

        mock_kb.piece_store.list_all.assert_called_once_with(entity_id="user1")

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_list_empty_store(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        mock_kb.piece_store.list_all.return_value = []
        mock_create_kb.return_value = mock_kb

        main(["list"])

        output = json.loads(capsys.readouterr().out)
        assert output == {"ok": True, "items": []}


class TestMainClear:
    """Tests for the 'clear' command."""

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_clear_removes_all_pieces(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        pieces = [
            KnowledgePiece(content="a", piece_id="p1"),
            KnowledgePiece(content="b", piece_id="p2"),
        ]
        mock_kb.piece_store.list_all.return_value = pieces
        mock_create_kb.return_value = mock_kb

        main(["clear"])

        assert mock_kb.remove_piece.call_count == 2
        output = json.loads(capsys.readouterr().out)
        assert output == {"ok": True, "cleared": 2}

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_clear_with_entity_id(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        mock_kb.piece_store.list_all.return_value = []
        mock_create_kb.return_value = mock_kb

        main(["clear", "--entity-id", "user1"])

        mock_kb.piece_store.list_all.assert_called_once_with(entity_id="user1")
        output = json.loads(capsys.readouterr().out)
        assert output == {"ok": True, "cleared": 0}

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_clear_closes_kb(self, mock_create_kb):
        mock_kb = MagicMock()
        mock_kb.piece_store.list_all.return_value = []
        mock_create_kb.return_value = mock_kb

        main(["clear"])

        mock_kb.close.assert_called_once()


class TestMainJsonOutput:
    """Tests verifying JSON subprocess communication."""

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_all_outputs_are_valid_json(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        mock_kb.piece_store.list_all.return_value = []
        mock_create_kb.return_value = mock_kb

        main(["list"])

        raw = capsys.readouterr().out.strip()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    @patch("agent_foundation.knowledge.cli._create_kb")
    def test_error_output_is_valid_json(self, mock_create_kb, capsys):
        mock_kb = MagicMock()
        mock_create_kb.return_value = mock_kb

        with pytest.raises(SystemExit):
            main(["add"])

        raw = capsys.readouterr().out.strip()
        parsed = json.loads(raw)
        assert "error" in parsed

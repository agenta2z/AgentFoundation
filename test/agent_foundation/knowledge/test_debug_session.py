"""Unit tests for IngestionDebugSession."""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from agent_foundation.knowledge.ingestion.debug_session import (
    IngestionDebugSession,
    get_ingestion_runtime_dir,
    get_knowledge_base_dir,
    list_all_ingestion_sessions,
)


class TestGetKnowledgeBaseDir:
    """Tests for get_knowledge_base_dir."""

    def test_default_uses_home_cache(self, monkeypatch):
        monkeypatch.delenv("AGENT_FOUNDATION_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = get_knowledge_base_dir()
        assert result == Path.home() / ".cache" / "knowledge_ingestion"

    def test_xdg_cache_home(self, monkeypatch):
        monkeypatch.delenv("AGENT_FOUNDATION_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg_cache")
        result = get_knowledge_base_dir()
        assert result == Path("/tmp/xdg_cache") / "knowledge_ingestion"

    def test_agent_foundation_cache_dir_takes_priority(self, monkeypatch):
        monkeypatch.setenv("AGENT_FOUNDATION_CACHE_DIR", "/tmp/af_cache")
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg_cache")
        result = get_knowledge_base_dir()
        assert result == Path("/tmp/af_cache") / "knowledge_ingestion"


class TestGetIngestionRuntimeDir:
    """Tests for get_ingestion_runtime_dir."""

    def test_creates_timestamped_dir(self):
        ts = datetime(2024, 1, 15, 10, 30, 45)
        result = get_ingestion_runtime_dir(timestamp=ts)
        assert result.name == "ingest_20240115_103045"

    def test_custom_root(self, tmp_path):
        ts = datetime(2024, 6, 1, 12, 0, 0)
        result = get_ingestion_runtime_dir(timestamp=ts, custom_root=tmp_path)
        assert result == tmp_path / "ingest_20240601_120000"

    def test_default_timestamp_uses_now(self):
        result = get_ingestion_runtime_dir()
        assert result.name.startswith("ingest_")


class TestListAllIngestionSessions:
    """Tests for list_all_ingestion_sessions."""

    def test_empty_when_no_dir(self, tmp_path):
        result = list_all_ingestion_sessions(custom_root=tmp_path / "nonexistent")
        assert result == []

    def test_lists_sessions_sorted_descending(self, tmp_path):
        (tmp_path / "ingest_20240101_000000").mkdir()
        (tmp_path / "ingest_20240601_120000").mkdir()
        (tmp_path / "ingest_20240301_060000").mkdir()
        (tmp_path / "other_dir").mkdir()  # should be excluded

        result = list_all_ingestion_sessions(custom_root=tmp_path)
        assert len(result) == 3
        assert result[0].name == "ingest_20240601_120000"
        assert result[2].name == "ingest_20240101_000000"


class TestIngestionDebugSession:
    """Tests for IngestionDebugSession."""

    def test_creates_subdirectories(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path / "session", enable_file_logging=False)
        assert (tmp_path / "session" / "chunks").is_dir()
        assert (tmp_path / "session" / "prompts").is_dir()
        assert (tmp_path / "session" / "responses").is_dir()
        assert (tmp_path / "session" / "structured").is_dir()
        assert (tmp_path / "session" / "merged").is_dir()
        assert (tmp_path / "session" / "logs").is_dir()

    def test_save_chunk(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        path = session.save_chunk("doc1", 0, "Hello world")
        assert path.exists()
        assert path.read_text() == "Hello world"
        assert path in session.get_files_saved()

    def test_save_chunk_with_metadata(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        meta = {"source": "test.md", "line": 10}
        path = session.save_chunk("doc1", 1, "Content", metadata=meta)
        text = path.read_text()
        assert text.startswith("# Metadata:")
        assert "Content" in text

    def test_save_prompt(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        path = session.save_prompt("doc1", 0, "Classify this text")
        assert path.exists()
        assert path.read_text() == "Classify this text"
        assert "prompt" in path.name

    def test_save_response(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        path = session.save_response("doc1", 0, '{"domain": "general"}')
        assert path.exists()
        assert "response" in path.name

    def test_save_structured(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        data = {"domain": "testing", "tags": ["unit"]}
        path = session.save_structured("doc1", 0, data)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_save_merged(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        data = {"pieces": [{"content": "merged content"}]}
        path = session.save_merged("doc1", data)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_save_log(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        path = session.save_log("doc1", "Processing started")
        assert path.exists()
        text = path.read_text()
        assert "Processing started" in text

    def test_save_log_appends(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        session.save_log("doc1", "First message")
        session.save_log("doc1", "Second message")
        path = session.logs_dir / "doc1_log.txt"
        text = path.read_text()
        assert "First message" in text
        assert "Second message" in text

    def test_save_summary(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        summary = {"total_chunks": 5, "errors": 0}
        path = session.save_summary(summary)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["total_chunks"] == 5
        assert "runtime_dir" in loaded
        assert "timestamp" in loaded
        assert "files_saved" in loaded

    def test_get_files_saved_returns_copy(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        session.save_chunk("doc1", 0, "test")
        files = session.get_files_saved()
        files.clear()
        assert len(session.get_files_saved()) > 0

    def test_sanitize_filename(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False)
        path = session.save_chunk("path/to/doc with spaces", 0, "content")
        assert path.exists()
        assert "/" not in path.name
        assert " " not in path.name

    def test_context_manager(self, tmp_path):
        with IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=False) as session:
            session.save_chunk("doc1", 0, "test")
            assert len(session.get_files_saved()) == 1

    def test_file_logging_creates_log_file(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=True)
        log_file = tmp_path / "logs" / "ingestion.log"
        assert log_file.exists()
        session.close()

    def test_custom_root_creates_timestamped_dir(self, tmp_path):
        session = IngestionDebugSession(custom_root=tmp_path, enable_file_logging=False)
        assert session.runtime_dir.parent == tmp_path
        assert session.runtime_dir.name.startswith("ingest_")

    def test_close_cleans_up_handlers(self, tmp_path):
        session = IngestionDebugSession(runtime_dir=tmp_path, enable_file_logging=True)
        assert len(session._file_handlers) == 1
        session.close()
        # Handler should be closed (no error on double close)
        session.close()

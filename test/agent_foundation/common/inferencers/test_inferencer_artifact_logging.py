"""Tests for per-call is_artifact=True in inferencer session logging.

Verifies that:
- InferenceInput / InferenceResponse produce .parts/ files (artifact-marked)
- InferenceArgs does NOT produce .parts/ files (metadata, not artifact)
- Small artifacts are still extracted (is_artifact bypasses parts_min_size)
- Async path produces the same artifact structure as sync
- PostProcessedResponse / MergedResponse produce .parts/ when triggered
- No crash when inferencer has no workspace
- Session JSONL content correctly distinguishes inline vs __parts_file__
"""

import asyncio
import glob as _glob
import json
import os
import tempfile

import pytest
from attr import attrs, attrib

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


@attrs
class _StubInferencer(InferencerBase):
    """Minimal inferencer that returns a scripted response."""

    scripted_response: str = attrib(default="stub response")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.scripted_response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self.scripted_response


def _find_session_parts(ws_root):
    """Find the session.jsonl.parts directory under workspace logs."""
    logs_dir = os.path.join(ws_root, "logs")
    if not os.path.isdir(logs_dir):
        return None
    for dirpath, dirnames, _ in os.walk(logs_dir):
        for d in dirnames:
            if d.endswith(".jsonl.parts"):
                return os.path.join(dirpath, d)
    return None


def _list_subdirs(parts_dir):
    """List subdirectory names inside a .parts/ directory."""
    if parts_dir is None or not os.path.isdir(parts_dir):
        return []
    return [
        name for name in os.listdir(parts_dir)
        if os.path.isdir(os.path.join(parts_dir, name))
    ]


def _count_files_in(parts_dir, subdir):
    """Count files in a specific subdirectory of .parts/."""
    target = os.path.join(parts_dir, subdir)
    if not os.path.isdir(target):
        return 0
    return sum(len(files) for _, _, files in os.walk(target))


class TestInferencerArtifactLogging:
    """Verify selective artifact extraction in inferencer session logging."""

    def test_artifact_log_calls_produce_parts(self, tmp_path):
        """InferenceInput and InferenceResponse should produce .parts/ subdirs."""
        ws = InferencerWorkspace(root=str(tmp_path / "workspace"))
        ws.ensure_dirs()
        inf = _StubInferencer(
            workspace=ws,
            debug_mode=True,
            always_add_logging_based_logger=False,
        )

        inf.infer("test prompt input")

        parts_dir = _find_session_parts(str(tmp_path / "workspace"))
        assert parts_dir is not None, "No session.jsonl.parts directory found"

        subdirs = _list_subdirs(parts_dir)
        assert "InferenceInput" in subdirs, (
            f"InferenceInput subdir missing. Found: {subdirs}"
        )
        assert "InferenceResponse" in subdirs, (
            f"InferenceResponse subdir missing. Found: {subdirs}"
        )

        assert _count_files_in(parts_dir, "InferenceInput") >= 1
        assert _count_files_in(parts_dir, "InferenceResponse") >= 1

    def test_metadata_log_calls_no_parts(self, tmp_path):
        """InferenceArgs should NOT produce a .parts/ subdir."""
        ws = InferencerWorkspace(root=str(tmp_path / "workspace"))
        ws.ensure_dirs()
        inf = _StubInferencer(
            workspace=ws,
            debug_mode=True,
            always_add_logging_based_logger=False,
        )

        inf.infer("test prompt")

        parts_dir = _find_session_parts(str(tmp_path / "workspace"))
        subdirs = _list_subdirs(parts_dir) if parts_dir else []

        assert "InferenceArgs" not in subdirs, (
            f"InferenceArgs should NOT have .parts/ subdir, but found in {subdirs}"
        )

    def test_small_artifacts_still_extracted(self, tmp_path):
        """Even a tiny input/response should produce .parts/ files."""
        ws = InferencerWorkspace(root=str(tmp_path / "workspace"))
        ws.ensure_dirs()
        inf = _StubInferencer(
            scripted_response="ok",
            workspace=ws,
            debug_mode=True,
            always_add_logging_based_logger=False,
        )

        inf.infer("hi")

        parts_dir = _find_session_parts(str(tmp_path / "workspace"))
        assert parts_dir is not None, "No .parts/ dir for small artifacts"

        assert _count_files_in(parts_dir, "InferenceInput") >= 1, (
            "Small InferenceInput ('hi') should still be extracted as artifact"
        )
        assert _count_files_in(parts_dir, "InferenceResponse") >= 1, (
            "Small InferenceResponse ('ok') should still be extracted as artifact"
        )

    @pytest.mark.asyncio
    async def test_async_path_produces_parts(self, tmp_path):
        """ainfer() should produce the same artifact .parts/ as infer()."""
        ws = InferencerWorkspace(root=str(tmp_path / "workspace"))
        ws.ensure_dirs()
        inf = _StubInferencer(
            workspace=ws,
            debug_mode=True,
            always_add_logging_based_logger=False,
        )

        await inf.ainfer("async test prompt")

        parts_dir = _find_session_parts(str(tmp_path / "workspace"))
        assert parts_dir is not None, "No session.jsonl.parts directory found for async"

        subdirs = _list_subdirs(parts_dir)
        assert "InferenceInput" in subdirs, (
            f"InferenceInput missing in async path. Found: {subdirs}"
        )
        assert "InferenceResponse" in subdirs, (
            f"InferenceResponse missing in async path. Found: {subdirs}"
        )

    def test_post_processed_response_produces_parts(self, tmp_path):
        """PostProcessedResponse should produce .parts/ when post-processor is set."""
        ws = InferencerWorkspace(root=str(tmp_path / "workspace"))
        ws.ensure_dirs()
        inf = _StubInferencer(
            scripted_response="raw output",
            response_post_processor=lambda x: f"processed: {x}",
            workspace=ws,
            debug_mode=True,
            always_add_logging_based_logger=False,
        )

        result = inf.infer("test")
        assert "processed:" in str(result)

        parts_dir = _find_session_parts(str(tmp_path / "workspace"))
        subdirs = _list_subdirs(parts_dir) if parts_dir else []
        assert "PostProcessedResponse" in subdirs, (
            f"PostProcessedResponse subdir missing. Found: {subdirs}"
        )

    def test_merged_response_produces_parts(self, tmp_path):
        """MergedResponse should produce .parts/ when post_response_merger is set."""
        ws = InferencerWorkspace(root=str(tmp_path / "workspace"))
        ws.ensure_dirs()
        inf = _StubInferencer(
            scripted_response="chunk",
            post_response_merger=lambda results: " + ".join(str(r) for r in results),
            workspace=ws,
            debug_mode=True,
            always_add_logging_based_logger=False,
        )

        result = inf.infer(iter(["prompt1", "prompt2"]))

        parts_dir = _find_session_parts(str(tmp_path / "workspace"))
        subdirs = _list_subdirs(parts_dir) if parts_dir else []
        assert "MergedResponse" in subdirs, (
            f"MergedResponse subdir missing. Found: {subdirs}"
        )

    def test_no_workspace_no_crash(self):
        """is_artifact=True kwargs should be harmless when no workspace logger exists."""
        inf = _StubInferencer(
            debug_mode=True,
            always_add_logging_based_logger=False,
        )

        result = inf.infer("test without workspace")
        assert result == "stub response"

    def test_session_jsonl_inline_vs_parts(self, tmp_path):
        """Non-artifact entries should be fully inline; artifact entries should use __parts_file__."""
        ws = InferencerWorkspace(root=str(tmp_path / "workspace"))
        ws.ensure_dirs()
        inf = _StubInferencer(
            workspace=ws,
            debug_mode=True,
            always_add_logging_based_logger=False,
        )

        inf.infer("test prompt for jsonl check")

        logs_dir = os.path.join(str(tmp_path / "workspace"), "logs")
        jsonl_files = [
            f for f in _glob.glob(os.path.join(logs_dir, "**", "*.jsonl"), recursive=True)
            if '.parts' not in f
        ]
        assert len(jsonl_files) >= 1, "No session.jsonl file found"

        with open(jsonl_files[0]) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        has_parts_ref = lambda entry: '__parts_file__' in json.dumps(entry)

        args_entries = [e for e in lines if e.get('type') == 'InferenceArgs']
        artifact_entries = [e for e in lines if has_parts_ref(e)]
        non_artifact_entries = [e for e in lines if not has_parts_ref(e)]

        assert len(args_entries) >= 1, "No InferenceArgs entry in session.jsonl"
        assert len(artifact_entries) >= 1, "No artifact entries with __parts_file__ references"
        assert len(non_artifact_entries) >= 1, "Expected at least one non-artifact inline entry"

        for entry in args_entries:
            assert not has_parts_ref(entry), (
                "InferenceArgs should be fully inline (no __parts_file__)"
            )

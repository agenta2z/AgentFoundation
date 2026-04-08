"""Tests for InferencerWorkspace."""

import json
import os
import tempfile

import pytest

from agent_foundation.common.inferencers.inferencer_workspace import (
    InferencerWorkspace,
)


@pytest.fixture
def tmp_root(tmp_path):
    """Provide a temporary workspace root path as a string."""
    return str(tmp_path / "workspace")


@pytest.fixture
def ws(tmp_root):
    """Create an InferencerWorkspace with ensure_dirs() called."""
    workspace = InferencerWorkspace(root=tmp_root)
    workspace.ensure_dirs()
    return workspace


# -- Directory properties --


class TestDirectoryProperties:
    def test_outputs_dir(self, ws):
        assert ws.outputs_dir == os.path.join(ws.root, "outputs")

    def test_artifacts_dir(self, ws):
        assert ws.artifacts_dir == os.path.join(ws.root, "artifacts")

    def test_checkpoints_dir(self, ws):
        assert ws.checkpoints_dir == os.path.join(ws.root, "checkpoints")

    def test_logs_dir(self, ws):
        assert ws.logs_dir == os.path.join(ws.root, "logs")

    def test_children_dir(self, ws):
        assert ws.children_dir == os.path.join(ws.root, "children")


# -- ensure_dirs --


class TestEnsureDirs:
    def test_creates_4_core_dirs(self, tmp_root):
        ws = InferencerWorkspace(root=tmp_root)
        ws.ensure_dirs()
        assert os.path.isdir(ws.outputs_dir)
        assert os.path.isdir(ws.artifacts_dir)
        assert os.path.isdir(ws.checkpoints_dir)
        assert os.path.isdir(ws.logs_dir)

    def test_does_not_create_children(self, ws):
        assert not os.path.exists(ws.children_dir)

    def test_does_not_create_optional_dirs(self, ws):
        for name in ("analysis", "results", "_runtime"):
            assert not os.path.exists(os.path.join(ws.root, name))

    def test_extra_subdirs(self, tmp_root):
        ws = InferencerWorkspace(root=tmp_root)
        ws.ensure_dirs("analysis", "results", "_runtime")
        for name in ("analysis", "results", "_runtime"):
            assert os.path.isdir(os.path.join(ws.root, name))

    def test_idempotent(self, ws):
        # Write a file into outputs/
        test_file = os.path.join(ws.outputs_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello")
        # Call ensure_dirs again — file must survive
        ws.ensure_dirs()
        assert os.path.isfile(test_file)
        with open(test_file) as f:
            assert f.read() == "hello"


# -- Path resolution --


class TestPathResolution:
    def test_output_path(self, ws):
        assert ws.output_path("plan.md") == os.path.join(
            ws.root, "outputs", "plan.md"
        )

    def test_artifact_path(self, ws):
        assert ws.artifact_path("round01_plan.md") == os.path.join(
            ws.root, "artifacts", "round01_plan.md"
        )

    def test_checkpoint_path(self, ws):
        assert ws.checkpoint_path("step_1.json") == os.path.join(
            ws.root, "checkpoints", "step_1.json"
        )

    def test_checkpoint_path_nested(self, ws):
        result = ws.checkpoint_path(os.path.join("attempt_01", "step_1.json"))
        assert result == os.path.join(
            ws.root, "checkpoints", "attempt_01", "step_1.json"
        )

    def test_log_path(self, ws):
        assert ws.log_path("Round01") == os.path.join(
            ws.root, "logs", "Round01"
        )

    def test_analysis_path(self, ws):
        assert ws.analysis_path("iter_1.md") == os.path.join(
            ws.root, "analysis", "iter_1.md"
        )

    def test_results_path(self, ws):
        assert ws.results_path("summary.json") == os.path.join(
            ws.root, "results", "summary.json"
        )

    def test_subdir(self, ws):
        assert ws.subdir("custom") == os.path.join(ws.root, "custom")
        # subdir does NOT create the directory
        assert not os.path.exists(ws.subdir("custom"))


# -- Child workspace --


class TestChildWorkspace:
    def test_child_returns_workspace(self, ws):
        child = ws.child("planner")
        assert isinstance(child, InferencerWorkspace)
        assert child.root == os.path.join(ws.root, "children", "planner")

    def test_child_does_not_create_dirs(self, ws):
        child = ws.child("planner")
        assert not os.path.exists(child.root)

    def test_child_ensure_dirs_creates_structure(self, ws):
        child = ws.child("planner")
        child.ensure_dirs()
        assert os.path.isdir(child.outputs_dir)
        assert os.path.isdir(child.artifacts_dir)
        assert os.path.isdir(child.checkpoints_dir)
        assert os.path.isdir(child.logs_dir)

    def test_child_is_independent(self, ws):
        child = ws.child("planner")
        child.ensure_dirs()
        # Writing to child doesn't affect parent
        with open(child.output_path("plan.md"), "w") as f:
            f.write("test plan")
        assert not os.path.exists(ws.output_path("plan.md"))

    def test_child_output(self, ws):
        expected = os.path.join(
            ws.root, "children", "planner", "outputs", "plan.md"
        )
        assert ws.child_output("planner", "plan.md") == expected

    def test_nested_children(self, ws):
        child = ws.child("worker_0")
        grandchild = child.child("planner")
        assert grandchild.root == os.path.join(
            ws.root, "children", "worker_0", "children", "planner"
        )


# -- Path traversal validation --


class TestPathTraversal:
    def test_rejects_dotdot(self, ws):
        with pytest.raises(ValueError, match="Invalid child workspace name"):
            ws.child("..")

    def test_rejects_dotdot_in_path(self, ws):
        with pytest.raises(ValueError):
            ws.child("foo/../bar")

    def test_rejects_forward_slash(self, ws):
        with pytest.raises(ValueError):
            ws.child("foo/bar")

    def test_rejects_backslash(self, ws):
        with pytest.raises(ValueError):
            ws.child("foo\\bar")

    def test_rejects_dot(self, ws):
        with pytest.raises(ValueError):
            ws.child(".")

    def test_rejects_empty(self, ws):
        with pytest.raises(ValueError):
            ws.child("")

    def test_child_output_validates_name(self, ws):
        with pytest.raises(ValueError):
            ws.child_output("../../etc", "passwd")

    def test_allows_valid_names(self, ws):
        # These should NOT raise
        ws.child("planner")
        ws.child("worker_0")
        ws.child("analyzer")
        ws.child("my-component")


# -- Glob --


class TestGlob:
    def test_glob_outputs_empty(self, ws):
        assert ws.glob_outputs("*.md") == []

    def test_glob_outputs(self, ws):
        for name in ("round01_plan.md", "round02_plan.md"):
            with open(ws.output_path(name), "w") as f:
                f.write("test")
        result = ws.glob_outputs("round*_plan.md")
        assert len(result) == 2
        assert "round01_plan.md" in os.path.basename(result[0])
        assert "round02_plan.md" in os.path.basename(result[1])

    def test_glob_artifacts(self, ws):
        for name in ("round01_plan.md", "round02_plan.md"):
            with open(ws.artifact_path(name), "w") as f:
                f.write("test")
        result = ws.glob_artifacts("round*_plan.md")
        assert len(result) == 2

    def test_glob_returns_sorted(self, ws):
        for name in ("c.txt", "a.txt", "b.txt"):
            with open(ws.output_path(name), "w") as f:
                f.write("test")
        result = ws.glob_outputs("*.txt")
        basenames = [os.path.basename(p) for p in result]
        assert basenames == ["a.txt", "b.txt", "c.txt"]


# -- Markers --


class TestMarkers:
    def test_write_and_has_marker(self, ws):
        assert not ws.has_marker("plan")
        ws.write_marker("plan")
        assert ws.has_marker("plan")

    def test_marker_content(self, ws):
        ws.write_marker("plan")
        marker_path = ws.artifact_path(".plan_completed")
        with open(marker_path) as f:
            data = json.load(f)
        assert "completed_at" in data
        assert data["step"] == "plan"

    def test_marker_custom_metadata(self, ws):
        ws.write_marker("plan", metadata={"custom": "value"})
        marker_path = ws.artifact_path(".plan_completed")
        with open(marker_path) as f:
            data = json.load(f)
        assert data == {"custom": "value"}

    def test_has_marker_legacy_fallback(self, ws):
        """has_marker checks artifacts/ first, then outputs/ for backward compat."""
        # Write marker to legacy outputs/ location
        legacy_path = ws.output_path(".plan_completed")
        with open(legacy_path, "w") as f:
            json.dump({"step": "plan"}, f)
        # Should find it via fallback
        assert ws.has_marker("plan")

    def test_has_marker_prefers_artifacts(self, ws):
        """When marker exists in both locations, artifacts/ is checked first."""
        # Write to both locations with different content
        ws.write_marker("plan", metadata={"location": "artifacts"})
        with open(ws.output_path(".plan_completed"), "w") as f:
            json.dump({"location": "outputs"}, f)
        # has_marker returns True (found in artifacts/)
        assert ws.has_marker("plan")

    def test_clear_marker(self, ws):
        ws.write_marker("plan")
        assert ws.has_marker("plan")
        ws.clear_marker("plan")
        assert not ws.has_marker("plan")

    def test_clear_marker_both_locations(self, ws):
        """clear_marker removes from both artifacts/ and outputs/."""
        ws.write_marker("impl")
        # Also write to legacy location
        with open(ws.output_path(".impl_completed"), "w") as f:
            json.dump({"step": "impl"}, f)
        ws.clear_marker("impl")
        assert not os.path.isfile(ws.artifact_path(".impl_completed"))
        assert not os.path.isfile(ws.output_path(".impl_completed"))

    def test_clear_marker_nonexistent(self, ws):
        """clear_marker on non-existent marker doesn't raise."""
        ws.clear_marker("nonexistent")  # should not raise


# -- attrs behavior --


class TestAttrs:
    def test_eq(self, tmp_root):
        ws1 = InferencerWorkspace(root=tmp_root)
        ws2 = InferencerWorkspace(root=tmp_root)
        assert ws1 == ws2

    def test_repr(self, tmp_root):
        ws = InferencerWorkspace(root=tmp_root)
        r = repr(ws)
        assert "InferencerWorkspace" in r
        assert "workspace" in r

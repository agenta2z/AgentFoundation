"""Phase 0 unit tests for the deliverable boundary semantics plan.

Covers:
- AC0 (workspace child propagation + ensure_dirs creates deliverables_dir)
- T7 (grandchild flag propagation)
- surface_outputs_from primitive
"""

import os

import pytest

from agent_foundation.common.inferencers.inferencer_workspace import (
    InferencerWorkspace,
)


# ---------------------------------------------------------------------------
# Fix 1: child() propagates use_final_deliverables_folder
# ---------------------------------------------------------------------------


def test_child_propagates_use_final_deliverables_folder_true(tmp_path):
    """child() must inherit use_final_deliverables_folder=True from parent."""
    parent = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=True
    )
    child = parent.child("worker_0")
    assert child.use_final_deliverables_folder is True
    assert child.deliverables_dir is not None
    assert child.deliverables_dir.endswith("worker_0/outputs/final_deliverables")


def test_child_propagates_use_final_deliverables_folder_false(tmp_path):
    """child() must inherit use_final_deliverables_folder=False from parent."""
    parent = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=False
    )
    child = parent.child("worker_0")
    assert child.use_final_deliverables_folder is False
    assert child.deliverables_dir is None


def test_child_propagates_use_final_deliverables_folder_str(tmp_path):
    """child() must inherit string-typed use_final_deliverables_folder."""
    parent = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder="custom_dir"
    )
    child = parent.child("worker_0")
    assert child.use_final_deliverables_folder == "custom_dir"
    assert child.deliverables_dir.endswith("worker_0/outputs/custom_dir")


def test_grandchild_propagation(tmp_path):
    """T7: ws.child("a").child("b") propagates the flag through 2 hops."""
    parent = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=True
    )
    grandchild = parent.child("a").child("b")
    assert grandchild.use_final_deliverables_folder is True
    assert grandchild.deliverables_dir is not None
    assert "children/a/children/b/outputs/final_deliverables" in (
        grandchild.deliverables_dir
    )


# ---------------------------------------------------------------------------
# Fix 2: ensure_dirs() creates deliverables_dir
# ---------------------------------------------------------------------------


def test_ensure_dirs_creates_deliverables_when_enabled(tmp_path):
    ws = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=True
    )
    ws.ensure_dirs()
    assert os.path.isdir(ws.deliverables_dir)


def test_ensure_dirs_skips_deliverables_when_disabled(tmp_path):
    ws = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=False
    )
    ws.ensure_dirs()
    assert ws.deliverables_dir is None
    # outputs/final_deliverables should NOT exist
    assert not os.path.exists(
        os.path.join(str(tmp_path), "outputs", "final_deliverables")
    )


# ---------------------------------------------------------------------------
# has_deliverables
# ---------------------------------------------------------------------------


def test_has_deliverables_false_when_dir_does_not_exist(tmp_path):
    ws = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=True
    )
    # No ensure_dirs() called
    assert ws.has_deliverables is False


def test_has_deliverables_false_when_dir_empty(tmp_path):
    ws = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=True
    )
    ws.ensure_dirs()
    assert ws.has_deliverables is False  # Empty


def test_has_deliverables_true_when_dir_has_file(tmp_path):
    ws = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=True
    )
    ws.ensure_dirs()
    with open(os.path.join(ws.deliverables_dir, "out.md"), "w") as f:
        f.write("hello")
    assert ws.has_deliverables is True


def test_has_deliverables_false_when_disabled(tmp_path):
    ws = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=False
    )
    assert ws.has_deliverables is False


# ---------------------------------------------------------------------------
# deliverable_paths
# ---------------------------------------------------------------------------


def test_deliverable_paths_returns_relative_paths(tmp_path):
    ws = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=True
    )
    ws.ensure_dirs()
    with open(os.path.join(ws.deliverables_dir, "a.md"), "w") as f:
        f.write("a")
    sub = os.path.join(ws.deliverables_dir, "sub")
    os.makedirs(sub)
    with open(os.path.join(sub, "b.md"), "w") as f:
        f.write("b")
    paths = ws.deliverable_paths()
    assert paths == ["a.md", os.path.join("sub", "b.md")]


def test_deliverable_paths_empty_when_no_dir(tmp_path):
    ws = InferencerWorkspace(
        root=str(tmp_path), use_final_deliverables_folder=False
    )
    assert ws.deliverable_paths() == []


# ---------------------------------------------------------------------------
# surface_outputs_from primitive
# ---------------------------------------------------------------------------


def test_surface_outputs_from_simple_copy(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(os.path.join(child.deliverables_dir, "result.md"), "w") as f:
        f.write("worker output")

    copied = parent.surface_outputs_from(child)
    assert "result.md" in copied
    assert os.path.isfile(
        os.path.join(parent.deliverables_dir, "result.md")
    )


def test_surface_outputs_from_with_namespace(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(os.path.join(child.deliverables_dir, "result.md"), "w") as f:
        f.write("worker output")

    copied = parent.surface_outputs_from(
        child, namespace="workers/worker_0"
    )
    expected_rel = os.path.join("workers", "worker_0", "result.md")
    assert expected_rel in copied
    assert os.path.isfile(
        os.path.join(parent.deliverables_dir, "workers", "worker_0", "result.md")
    )


def test_surface_outputs_from_skip_existing(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(os.path.join(child.deliverables_dir, "result.md"), "w") as f:
        f.write("new")
    # Pre-populate dest
    with open(os.path.join(parent.deliverables_dir, "result.md"), "w") as f:
        f.write("existing")

    copied = parent.surface_outputs_from(child, skip_existing=True)
    assert copied == []  # Skipped
    with open(os.path.join(parent.deliverables_dir, "result.md")) as f:
        assert f.read() == "existing"


def test_surface_outputs_from_overwrite(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(os.path.join(child.deliverables_dir, "result.md"), "w") as f:
        f.write("new")
    with open(os.path.join(parent.deliverables_dir, "result.md"), "w") as f:
        f.write("existing")

    copied = parent.surface_outputs_from(child, skip_existing=False)
    assert "result.md" in copied
    with open(os.path.join(parent.deliverables_dir, "result.md")) as f:
        assert f.read() == "new"


def test_surface_outputs_from_noop_when_source_empty(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()  # empty deliverables
    assert parent.surface_outputs_from(child) == []


def test_surface_outputs_from_noop_when_parent_disabled(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=False
    )
    parent.ensure_dirs()
    child = parent.child("worker_0")
    # Child also has flag=False because of propagation
    assert parent.surface_outputs_from(child) == []


def test_surface_outputs_from_validates_namespace(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(os.path.join(child.deliverables_dir, "result.md"), "w") as f:
        f.write("x")
    with pytest.raises(ValueError):
        parent.surface_outputs_from(child, namespace="../escape")

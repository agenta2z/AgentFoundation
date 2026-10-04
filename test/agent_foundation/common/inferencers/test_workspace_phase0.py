"""Phase 0 unit tests for the deliverable boundary semantics plan.

Part 2 (two-axis model) retired the ``final_deliverables/`` subfolder and the
``use_final_deliverables_folder`` workspace flag: deliverables now live directly
in ``outputs/``. The former "flag propagation" tests
(``test_child_propagates_use_final_deliverables_folder_*``,
``test_grandchild_propagation``, ``test_ensure_dirs_*_deliverables_*``) asserted
that deleted mechanism and were removed. The surviving tests below exercise the
retargeted helpers (``has_deliverables`` / ``deliverable_paths`` /
``surface_outputs_from``) against ``outputs/``.

Covers:
- ``has_deliverables`` (outputs/ non-empty)
- ``deliverable_paths`` (recursive listing under outputs/)
- ``surface_outputs_from`` primitive (child outputs/ → parent outputs/)
"""

import os

import pytest
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


# ---------------------------------------------------------------------------
# has_deliverables — outputs/ non-empty (final_deliverables/ retired)
# ---------------------------------------------------------------------------


def test_has_deliverables_false_when_dir_does_not_exist(tmp_path):
    ws = InferencerWorkspace(root=str(tmp_path))
    # No ensure_dirs() called — outputs/ does not exist yet.
    assert ws.has_deliverables is False


def test_has_deliverables_false_when_dir_empty(tmp_path):
    ws = InferencerWorkspace(root=str(tmp_path))
    ws.ensure_dirs()
    assert ws.has_deliverables is False  # outputs/ exists but empty


def test_has_deliverables_true_when_dir_has_file(tmp_path):
    ws = InferencerWorkspace(root=str(tmp_path))
    ws.ensure_dirs()
    with open(ws.output_path("out.md"), "w") as f:
        f.write("hello")
    assert ws.has_deliverables is True


# ---------------------------------------------------------------------------
# deliverable_paths — recursive listing under outputs/
# ---------------------------------------------------------------------------


def test_deliverable_paths_returns_relative_paths(tmp_path):
    ws = InferencerWorkspace(root=str(tmp_path))
    ws.ensure_dirs()
    with open(ws.output_path("a.md"), "w") as f:
        f.write("a")
    sub = ws.output_path("sub")
    os.makedirs(sub)
    with open(os.path.join(sub, "b.md"), "w") as f:
        f.write("b")
    paths = ws.deliverable_paths()
    # Paths are relative to outputs/ (no final_deliverables/ prefix).
    assert paths == ["a.md", os.path.join("sub", "b.md")]


def test_deliverable_paths_empty_when_no_dir(tmp_path):
    ws = InferencerWorkspace(root=str(tmp_path))
    assert ws.deliverable_paths() == []


# ---------------------------------------------------------------------------
# surface_outputs_from primitive — child outputs/ → parent outputs/
# ---------------------------------------------------------------------------


def test_surface_outputs_from_simple_copy(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(child.output_path("result.md"), "w") as f:
        f.write("worker output")

    copied = parent.surface_outputs_from(child)
    assert "result.md" in copied
    assert os.path.isfile(parent.output_path("result.md"))


def test_surface_outputs_from_with_namespace(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(child.output_path("result.md"), "w") as f:
        f.write("worker output")

    copied = parent.surface_outputs_from(child, namespace="workers/worker_0")
    expected_rel = os.path.join("workers", "worker_0", "result.md")
    assert expected_rel in copied
    assert os.path.isfile(
        parent.output_path(os.path.join("workers", "worker_0", "result.md"))
    )


def test_surface_outputs_from_skip_existing(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(child.output_path("result.md"), "w") as f:
        f.write("new")
    # Pre-populate dest
    with open(parent.output_path("result.md"), "w") as f:
        f.write("existing")

    copied = parent.surface_outputs_from(child, skip_existing=True)
    assert copied == []  # Skipped
    with open(parent.output_path("result.md")) as f:
        assert f.read() == "existing"


def test_surface_outputs_from_overwrite(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(child.output_path("result.md"), "w") as f:
        f.write("new")
    with open(parent.output_path("result.md"), "w") as f:
        f.write("existing")

    copied = parent.surface_outputs_from(child, skip_existing=False)
    assert "result.md" in copied
    with open(parent.output_path("result.md")) as f:
        assert f.read() == "new"


def test_surface_outputs_from_noop_when_source_empty(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()  # empty outputs/
    assert parent.surface_outputs_from(child) == []


def test_surface_outputs_from_validates_namespace(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(child.output_path("result.md"), "w") as f:
        f.write("x")
    with pytest.raises(ValueError):
        parent.surface_outputs_from(child, namespace="../escape")

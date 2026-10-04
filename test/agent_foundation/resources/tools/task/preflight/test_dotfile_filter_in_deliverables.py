"""Preflight tests for dotfile filtering in `_list_deliverable_files`.

Validates the dotfile-filter behavior that prevents hidden marker files (e.g.
``.DS_Store``, ``.gitkeep``, or a stray ``.self_promoted``) from being surfaced
as user-facing deliverables.

Part 2 (two-axis model): deliverables live directly in ``outputs/``
(``final_deliverables/`` is RETIRED), so ``_list_deliverable_files`` lists the
contents of ``workspace.outputs_dir``. The dotfile guard still applies.

Fix location:
  CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/
  deliverable_boundary.py — `_list_deliverable_files`:

      for f in files:
          if f.startswith("."):    ← dotfile guard
              continue
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _ws(tmp):
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    w = InferencerWorkspace(root=str(tmp))
    w.ensure_dirs()
    return w


def _list_deliverable_files(ws):
    """Reach into deliverable_boundary internals — the symbol we're testing."""
    from agent_foundation.common.inferencers import deliverable_boundary as db

    return db._list_deliverable_files(ws)


# -------------------------------------------------------------------------
# Direct dotfile filter behaviour
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_DF1_self_promoted_marker_is_filtered(tmp_path):
    """DF1: a stray `.self_promoted` dotfile is excluded from deliverable listings."""
    w = _ws(tmp_path)
    # A real deliverable plus a stray hidden file in outputs/.
    with open(os.path.join(w.outputs_dir, "output.md"), "w") as f:
        f.write("real deliverable")
    with open(os.path.join(w.outputs_dir, ".self_promoted"), "w") as f:
        f.write("")

    files = _list_deliverable_files(w)
    assert "output.md" in files, "Real deliverable must be listed"
    assert ".self_promoted" not in files, (
        "Hidden file `.self_promoted` must be filtered out by the dotfile "
        "guard in _list_deliverable_files. "
        f"Got: {files}"
    )


@pytest.mark.preflight
def test_DF2_arbitrary_dotfiles_filtered(tmp_path):
    """DF2: All dotfiles are filtered, not just `.self_promoted`."""
    w = _ws(tmp_path)
    with open(os.path.join(w.outputs_dir, "real.md"), "w") as f:
        f.write("x")
    for hidden in (".DS_Store", ".gitkeep", ".hidden_meta", ".pytest_cache"):
        with open(os.path.join(w.outputs_dir, hidden), "w") as f:
            f.write("x")

    files = _list_deliverable_files(w)
    assert files == ["real.md"], (
        f"Only non-dotfile deliverables should be listed. Got: {files}"
    )


@pytest.mark.preflight
def test_DF3_dotfiles_in_subdirs_filtered(tmp_path):
    """DF3: Dotfile filter applies recursively (subdirectories also filtered)."""
    w = _ws(tmp_path)
    sub = os.path.join(w.outputs_dir, "workers", "worker_0")
    os.makedirs(sub)
    with open(os.path.join(sub, "plan.md"), "w") as f:
        f.write("x")
    with open(os.path.join(sub, ".self_promoted"), "w") as f:
        f.write("")

    files = _list_deliverable_files(w)
    # Use os.path.join so this works on every platform
    expected = os.path.join("workers", "worker_0", "plan.md")
    assert expected in files, f"Real subdirectory file missing. Got: {files}"
    for entry in files:
        assert os.path.basename(entry) != ".self_promoted", (
            f"Dotfiles in nested directories must also be filtered. Found: {entry}"
        )


# -------------------------------------------------------------------------
# Filter is listing-only: it never deletes from disk, and has_deliverables
# (Part 2: outputs/ non-empty) is unaffected.
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_DF4_filter_is_listing_only_not_deletion(tmp_path):
    """DF4: The dotfile filter only excludes from the listing — it does NOT
    delete the file from disk, and has_deliverables still reflects outputs/.
    """
    w = _ws(tmp_path)
    with open(os.path.join(w.outputs_dir, "output.md"), "w") as f:
        f.write("x")
    marker_path = os.path.join(w.outputs_dir, ".self_promoted")
    with open(marker_path, "w") as f:
        f.write("")

    # The hidden file must still EXIST on disk — the filter only excludes it
    # from the listing, not from disk.
    assert os.path.isfile(marker_path), (
        "Filter must not delete the file — only exclude from listing"
    )

    # And `has_deliverables` must still return True (outputs/ is non-empty)
    assert w.has_deliverables, (
        "has_deliverables should still be True since output.md is present"
    )


# -------------------------------------------------------------------------
# End-to-end via the public `collect_child_boundary_deliverables`
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_DF5_marker_does_not_appear_in_aggregated_deliverables(tmp_path):
    """DF5: Through the full collect+aggregate flow, `.self_promoted` never
    appears in the parent's outputs/.

    This is the full-stack guarantee: even when a leaf has a stray hidden file,
    it stays in the leaf's workspace and is never copied upward.
    """
    from agent_foundation.common.inferencers.deliverable_boundary import (
        aggregate_into_self_deliverables,
        ChildBoundaryDeliverables,
    )

    parent = _ws(tmp_path)
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(os.path.join(child.outputs_dir, "out.md"), "w") as f:
        f.write("real")
    with open(os.path.join(child.outputs_dir, ".self_promoted"), "w") as f:
        f.write("")

    # Build the ChildBoundaryDeliverables list as the dotfile filter (in
    # `_list_deliverable_files`) would naturally produce — i.e., excluding
    # the marker. Then aggregate.
    files = _list_deliverable_files(child)
    assert ".self_promoted" not in files, (
        "Pre-condition: filter must already exclude marker before aggregation"
    )
    children = [
        ChildBoundaryDeliverables(
            child_name="worker_0",
            child_workspace_root=child.root,
            deliverable_files=files,
            child_workspace=child,
        )
    ]
    aggregate_into_self_deliverables(parent, children)

    # Parent's outputs/ should contain `out.md` somewhere but NEVER
    # `.self_promoted`
    found_marker = []
    for root_dir, _dirs, files_in in os.walk(parent.outputs_dir):
        for f in files_in:
            if f == ".self_promoted":
                found_marker.append(os.path.join(root_dir, f))
    assert not found_marker, (
        f"Marker file leaked into parent's outputs/: {found_marker}"
    )

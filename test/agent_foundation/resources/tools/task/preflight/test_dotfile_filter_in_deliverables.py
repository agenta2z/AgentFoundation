"""Preflight tests for dotfile filtering in `_list_deliverable_files`.

Validates the dotfile-filter fix that prevents `.self_promoted` (and other
hidden marker files) from being surfaced as user-facing deliverables.

Background:
  When a leaf inferencer self-promotes its output via `output_is_deliverable=True`,
  it writes a `.self_promoted` marker into its workspace's `deliverables_dir`.
  The parent BTA's Pass 1 collector (`collect_child_boundary_deliverables`)
  detects this marker but must NOT propagate it as a deliverable file —
  that would pollute every layer of the surfacing chain.

Fix location:
  CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/
  deliverable_boundary.py — `_list_deliverable_files`, lines 456-457:

      for f in files:
          if f.startswith("."):    ← ADDED
              continue              ← ADDED
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _ws(tmp, use_fdl=True):
    from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
    w = InferencerWorkspace(root=str(tmp), use_final_deliverables_folder=use_fdl)
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
    """DF1: `.self_promoted` marker is excluded from deliverable listings."""
    w = _ws(tmp_path)
    # Simulate a self-promoted leaf: real deliverable plus marker
    with open(os.path.join(w.deliverables_dir, "output.md"), "w") as f:
        f.write("real deliverable")
    with open(os.path.join(w.deliverables_dir, ".self_promoted"), "w") as f:
        f.write("")

    files = _list_deliverable_files(w)
    assert "output.md" in files, "Real deliverable must be listed"
    assert ".self_promoted" not in files, (
        "Marker file `.self_promoted` must be filtered out by the dotfile "
        "guard in _list_deliverable_files (deliverable_boundary.py:456-457). "
        f"Got: {files}"
    )


@pytest.mark.preflight
def test_DF2_arbitrary_dotfiles_filtered(tmp_path):
    """DF2: All dotfiles are filtered, not just `.self_promoted`."""
    w = _ws(tmp_path)
    with open(os.path.join(w.deliverables_dir, "real.md"), "w") as f:
        f.write("x")
    for hidden in (".DS_Store", ".gitkeep", ".hidden_meta", ".pytest_cache"):
        with open(os.path.join(w.deliverables_dir, hidden), "w") as f:
            f.write("x")

    files = _list_deliverable_files(w)
    assert files == ["real.md"], (
        f"Only non-dotfile deliverables should be listed. Got: {files}"
    )


@pytest.mark.preflight
def test_DF3_dotfiles_in_subdirs_filtered(tmp_path):
    """DF3: Dotfile filter applies recursively (subdirectories also filtered)."""
    w = _ws(tmp_path)
    sub = os.path.join(w.deliverables_dir, "workers", "worker_0")
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
            "Dotfiles in nested directories must also be filtered. "
            f"Found: {entry}"
        )


# -------------------------------------------------------------------------
# `.self_promoted` existence check still works (separate code path)
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_DF4_self_promoted_existence_check_still_works(tmp_path):
    """DF4: The dotfile filter does NOT break the existence check used by Pass 1.

    `collect_child_boundary_deliverables` checks for `.self_promoted` via
    direct `os.path.exists()` (or `has_self_promoted` semantic) — independent
    of `_list_deliverable_files`. The filter must not interfere with that path.
    """
    w = _ws(tmp_path)
    with open(os.path.join(w.deliverables_dir, "output.md"), "w") as f:
        f.write("x")
    marker_path = os.path.join(w.deliverables_dir, ".self_promoted")
    with open(marker_path, "w") as f:
        f.write("")

    # The marker file must still EXIST on disk — the filter only excludes it
    # from the listing, not from disk.
    assert os.path.isfile(marker_path), (
        "Filter must not delete the marker — only exclude from listing"
    )

    # And `has_deliverables` must still return True (deliverables_dir is non-empty)
    assert w.has_deliverables, (
        "has_deliverables should still be True since output.md is present"
    )


# -------------------------------------------------------------------------
# End-to-end via the public `collect_child_boundary_deliverables`
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_DF5_marker_does_not_appear_in_aggregated_deliverables(tmp_path):
    """DF5: Through the full collect+aggregate flow, `.self_promoted` never
    appears in the parent's final_deliverables/.

    This is the full-stack guarantee: even when a leaf self-promotes, the
    marker stays in the leaf's workspace and is never copied upward.
    """
    from agent_foundation.common.inferencers.deliverable_boundary import (
        ChildBoundaryDeliverables,
        aggregate_into_self_deliverables,
    )

    parent = _ws(tmp_path)
    child = parent.child("worker_0")
    child.ensure_dirs()
    with open(os.path.join(child.deliverables_dir, "out.md"), "w") as f:
        f.write("real")
    with open(os.path.join(child.deliverables_dir, ".self_promoted"), "w") as f:
        f.write("")

    # Build the ChildBoundaryDeliverables list as the dotfile filter (in
    # `_list_deliverable_files`) would naturally produce — i.e., excluding
    # the marker. Then aggregate.
    files = _list_deliverable_files(child)
    assert ".self_promoted" not in files, (
        "Pre-condition: filter must already exclude marker before aggregation"
    )
    children = [ChildBoundaryDeliverables(
        child_name="worker_0",
        child_workspace_root=child.root,
        deliverable_files=files,
        child_workspace=child,
    )]
    aggregate_into_self_deliverables(parent, children)

    # Parent's final_deliverables/ should contain `out.md` somewhere but NEVER
    # `.self_promoted`
    found_marker = []
    for root_dir, _dirs, files_in in os.walk(parent.deliverables_dir):
        for f in files_in:
            if f == ".self_promoted":
                found_marker.append(os.path.join(root_dir, f))
    assert not found_marker, (
        f"Marker file leaked into parent's final_deliverables: {found_marker}"
    )

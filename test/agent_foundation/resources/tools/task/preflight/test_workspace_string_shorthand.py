"""Preflight test for the ``workspace="<path>"`` string-shorthand convenience.

Background:
    Pre-2026-05-05, ``LinearWorkflowInferencer`` and friends declared a
    separate ``workspace_root: Optional[str]`` attrib as a YAML-friendly
    convenience that constructed ``InferencerWorkspace(root=<str>)`` with
    default flags. That attrib was removed because it duplicated the
    canonical ``workspace: Optional[InferencerWorkspace]`` attrib on
    ``InferencerBase`` and led to a subtle clobber bug.

    To preserve the YAML/Python terseness benefit without the duplicate
    field, ``InferencerBase.__attrs_post_init__`` now accepts a string
    value for ``workspace`` and auto-converts it to
    ``InferencerWorkspace(root=<str>)``.

What this test verifies:
    1. ``Inferencer(workspace="/tmp/foo")`` creates an ``InferencerWorkspace``
       with ``root="/tmp/foo"`` (no manual conversion needed).
    2. The string shorthand is byte-equivalent to passing the explicit
       ``InferencerWorkspace(root=...)`` object — same root, same default
       flags.
    3. Both ``self.workspace`` and ``self._workspace`` end up as the same
       ``InferencerWorkspace`` instance after post-init.
    4. Default flags on the constructed workspace are preserved
       (``use_final_deliverables_folder=False`` by default).

These run without LLM calls; expected runtime < 5s each.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_ws(tmp_path):
    """A tmp directory string usable as a workspace root."""
    return str(tmp_path / "shorthand_ws")


def _make_minimal_lwi(workspace_arg):
    """Construct a LinearWorkflowInferencer with no real steps,
    passing ``workspace_arg`` (str OR InferencerWorkspace OR None)."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )
    return LinearWorkflowInferencer(
        step_configs=[],
        workspace=workspace_arg,
    )


def test_string_shorthand_creates_inferencer_workspace(tmp_ws):
    """Preflight 1: passing a string yields a real InferencerWorkspace."""
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    inf = _make_minimal_lwi(tmp_ws)

    assert isinstance(inf.workspace, InferencerWorkspace), (
        f"Expected workspace to be InferencerWorkspace after string shorthand, "
        f"got {type(inf.workspace).__name__}"
    )
    assert inf.workspace.root == tmp_ws, (
        f"Expected workspace.root={tmp_ws!r}, got {inf.workspace.root!r}"
    )


def test_string_shorthand_equivalent_to_explicit_form(tmp_ws):
    """Preflight 2: workspace="<path>" == workspace=InferencerWorkspace(root="<path>")."""
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    inf_string = _make_minimal_lwi(tmp_ws)
    inf_explicit = _make_minimal_lwi(InferencerWorkspace(root=tmp_ws))

    # Same root
    assert inf_string.workspace.root == inf_explicit.workspace.root
    # Same default flag value (use_final_deliverables_folder)
    assert (
        inf_string.workspace.use_final_deliverables_folder
        == inf_explicit.workspace.use_final_deliverables_folder
    )


def test_string_shorthand_syncs_to_underscore_workspace(tmp_ws):
    """Preflight 3: self._workspace IS self.workspace after post-init."""
    inf = _make_minimal_lwi(tmp_ws)

    assert inf._workspace is not None, (
        "_workspace should be set after post-init synced from workspace"
    )
    assert inf._workspace is inf.workspace, (
        "_workspace and workspace should be the SAME object after post-init"
    )
    assert inf._workspace.root == tmp_ws


def test_string_shorthand_preserves_default_flags(tmp_ws):
    """Preflight 4: Default workspace flags are unset (use_final_deliverables_folder=False)."""
    inf = _make_minimal_lwi(tmp_ws)

    assert inf.workspace.use_final_deliverables_folder is False, (
        "String shorthand should produce a workspace with default flags "
        "(use_final_deliverables_folder=False); use the explicit "
        "InferencerWorkspace(root=..., use_final_deliverables_folder=True) "
        "form when you need flags."
    )


def test_explicit_workspace_object_pass_through(tmp_ws):
    """Preflight 5: an explicit InferencerWorkspace passes through unchanged
    (string conversion only triggers on str input — not other types)."""
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    explicit_ws = InferencerWorkspace(root=tmp_ws, use_final_deliverables_folder=True)
    inf = _make_minimal_lwi(explicit_ws)

    assert inf.workspace is explicit_ws, (
        "Explicit InferencerWorkspace should pass through unchanged "
        "(no re-construction)."
    )
    assert inf.workspace.use_final_deliverables_folder is True, (
        "Flags on the explicit workspace must be preserved."
    )


def test_none_workspace_remains_none():
    """Preflight 6: workspace=None remains None (no spurious conversion)."""
    inf = _make_minimal_lwi(None)
    assert inf.workspace is None
    assert inf._workspace is None

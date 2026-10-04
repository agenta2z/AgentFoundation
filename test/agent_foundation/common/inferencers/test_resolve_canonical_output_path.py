"""Tests for the ``resolve_canonical_output_path`` helper.

Part 2 (two-axis model) retired the ``final_deliverables/`` subfolder: deliverables
live directly in ``outputs/``, so ``resolve_canonical_output_path`` resolves straight
to ``outputs/<filename>`` (the former Tier-1 ``final_deliverables/`` probe and its
``deliverables_fallback`` policies are gone; the parameter is kept for call-site
compatibility but is no longer consulted). The Tier-1-only tests
(``test_tier1_deliverable_exists``, ``test_tier1_first_match_fallback``,
``test_tier1_alphabetical_scan_filters_dotfiles``,
``test_canonical_path_prefers_deliverables_over_outputs``) asserted that deleted
mechanism and were removed.

Surviving coverage:
  * None workspace → None
  * nothing on disk → None
  * ``outputs/<filename>`` present → absolute path (the canonical case)
  * custom filename resolution
  * a non-matching file in ``outputs/`` does not shadow the requested filename
  * absolute-path guarantee
"""

import os
import tempfile

from agent_foundation.common.inferencers.inferencer_workspace import (
    InferencerWorkspace,
    resolve_canonical_output_path,
)


def _make_workspace(tmpdir: str) -> InferencerWorkspace:
    """Create an InferencerWorkspace rooted at tmpdir with standard layout.

    Part 2: deliverables live directly in ``outputs/`` (no
    ``use_final_deliverables_folder`` flag, no separate deliverable subfolder).
    """
    ws = InferencerWorkspace(root=tmpdir)
    ws.ensure_dirs()
    return ws


def _write_file(path: str, content: str = "test") -> None:
    """Write content to path, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# --------------------------------------------------------------------------
# None workspace → None
# --------------------------------------------------------------------------


def test_none_workspace_returns_none():
    """Helper must return None for a None workspace, not crash."""
    assert resolve_canonical_output_path(None) is None
    assert resolve_canonical_output_path(None, filename="anything.md") is None


# --------------------------------------------------------------------------
# Nothing on disk → None
# --------------------------------------------------------------------------


def test_no_outputs_returns_none():
    """No ``outputs/<filename>`` on disk → None."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        # Workspace exists but outputs/output.md was never written.
        assert resolve_canonical_output_path(ws) is None


# --------------------------------------------------------------------------
# outputs/<filename> present → absolute path (the canonical case)
# --------------------------------------------------------------------------


def test_resolves_outputs_file():
    """Part 2: deliverables live in ``outputs/`` — ``outputs/output.md`` resolves
    to its absolute path.

    This is the common production case: leaf CLI inferencers (RovoDevCli,
    ClaudeCodeCli) AND orchestrators alike write their canonical output to
    ``outputs/output.md`` (there is no separate ``final_deliverables/`` folder).
    """
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        outputs_path = ws.output_path("output.md")
        _write_file(outputs_path, "canonical output")

        result = resolve_canonical_output_path(ws)
        assert result is not None
        assert os.path.isabs(result), f"Expected absolute path, got {result}"
        assert result == os.path.abspath(outputs_path)


def test_resolves_custom_filename():
    """Helper resolves a non-default filename from ``outputs/``."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        custom_path = ws.output_path("implementation.md")
        _write_file(custom_path, "custom content")

        # Default filename "output.md" is absent → None; the custom filename
        # resolves directly.
        assert resolve_canonical_output_path(ws) is None
        result = resolve_canonical_output_path(ws, filename="implementation.md")
        assert result == os.path.abspath(custom_path)


# --------------------------------------------------------------------------
# A non-matching file in outputs/ does not shadow the requested filename
# --------------------------------------------------------------------------


def test_non_matching_output_file_is_ignored():
    """Only the REQUESTED filename resolves — an unrelated file in ``outputs/``
    does not cause a false positive.

    Part 2: with the ``final_deliverables/`` fallback policies removed, resolution
    is a single exact ``outputs/<filename>`` probe. A workspace containing only
    ``wrong_name.md`` must resolve ``output.md`` to None, and resolve
    ``output.md`` once it is actually present.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        _write_file(ws.output_path("wrong_name.md"), "unrelated")
        # Requested filename absent → None (no fallback to the unrelated file).
        assert resolve_canonical_output_path(ws, filename="output.md") is None

        # Once the requested filename exists, it resolves.
        outputs_path = ws.output_path("output.md")
        _write_file(outputs_path, "the real output")
        result = resolve_canonical_output_path(ws, filename="output.md")
        assert result == os.path.abspath(outputs_path)


# --------------------------------------------------------------------------
# Absolute path guarantee
# --------------------------------------------------------------------------


def test_absolute_path_guarantee():
    """All returns must be absolute paths (CWD-independent)."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        outputs_path = ws.output_path("output.md")
        _write_file(outputs_path, "content")

        result = resolve_canonical_output_path(ws)
        assert result is not None
        assert os.path.isabs(result), f"Returned path must be absolute: {result}"
        assert result == os.path.abspath(outputs_path)

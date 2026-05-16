"""Phase 0a tests for ``resolve_canonical_output_path`` helper.

10 tests covering 3-tier resolution semantics:
  Tier 1: deliverables (with first_match / alphabetical_scan / none policies)
  Tier 2: outputs/<filename> (CRITICAL for leaf CLI inferencers)
  Tier 3: None (no usable file)

See ``orchestrator_path_aware_INTEGRATED_plan.md`` §4 for the full helper spec.
"""

import os
import tempfile

import pytest

from agent_foundation.common.inferencers.inferencer_workspace import (
    InferencerWorkspace,
    resolve_canonical_output_path,
)


def _make_workspace(tmpdir: str, with_deliverables: bool = True) -> InferencerWorkspace:
    """Create an InferencerWorkspace rooted at tmpdir with standard layout.

    Note: ``use_final_deliverables_folder`` defaults to True so deliverables_dir is set.
    """
    ws = InferencerWorkspace(
        root=tmpdir,
        use_final_deliverables_folder=with_deliverables,
    )
    ws.ensure_dirs()
    return ws


def _write_file(path: str, content: str = "test") -> None:
    """Write content to path, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# --------------------------------------------------------------------------
# Test 1: None workspace → None
# --------------------------------------------------------------------------

def test_none_workspace_returns_none():
    """Helper must return None for a None workspace, not crash."""
    assert resolve_canonical_output_path(None) is None
    assert resolve_canonical_output_path(None, filename="anything.md") is None


# --------------------------------------------------------------------------
# Test 2: No deliverables AND no outputs/output.md → None
# --------------------------------------------------------------------------

def test_no_deliverables_no_outputs_returns_none():
    """Tier 1 + Tier 2 both empty → None (Tier 3)."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        # Workspace exists but has no deliverable, no outputs/output.md
        assert resolve_canonical_output_path(ws) is None


# --------------------------------------------------------------------------
# Test 3: Tier 1 deliverable exists → returns absolute path
# --------------------------------------------------------------------------

def test_tier1_deliverable_exists():
    """Tier 1 hit: final_deliverables/output.md exists → return its abspath."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        # Create a deliverable
        deliv_dir = ws.deliverables_dir
        assert deliv_dir is not None
        os.makedirs(deliv_dir, exist_ok=True)
        deliv_path = os.path.join(deliv_dir, "output.md")
        _write_file(deliv_path, "deliverable content")

        result = resolve_canonical_output_path(ws)
        assert result is not None
        assert os.path.isabs(result), f"Expected absolute path, got {result}"
        assert result == os.path.abspath(deliv_path)


# --------------------------------------------------------------------------
# Test 4: Tier 1 with custom filename
# --------------------------------------------------------------------------

def test_tier1_custom_filename():
    """Helper resolves a non-default filename in Tier 1."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        deliv_dir = ws.deliverables_dir
        assert deliv_dir is not None
        os.makedirs(deliv_dir, exist_ok=True)
        custom_path = os.path.join(deliv_dir, "implementation.md")
        _write_file(custom_path, "custom content")

        # Default filename "output.md" → not found, falls back per policy.
        # With custom filename, found in Tier 1 directly.
        result = resolve_canonical_output_path(ws, filename="implementation.md")
        assert result == os.path.abspath(custom_path)


# --------------------------------------------------------------------------
# Test 5: Tier 1 "first_match" fallback
# --------------------------------------------------------------------------

def test_tier1_first_match_fallback():
    """deliverables_fallback='first_match' returns deliverable_paths()[0]."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        deliv_dir = ws.deliverables_dir
        assert deliv_dir is not None
        os.makedirs(deliv_dir, exist_ok=True)
        # Create a deliverable but NOT named "output.md"
        alt_path = os.path.join(deliv_dir, "zzz_alt.md")
        _write_file(alt_path, "alternative deliverable")

        result = resolve_canonical_output_path(
            ws, filename="output.md", deliverables_fallback="first_match"
        )
        # first_match returns deliverable_paths()[0] - the first listed
        assert result is not None
        assert os.path.isfile(result)


# --------------------------------------------------------------------------
# Test 6: Tier 1 "alphabetical_scan" fallback (with dotfile filter)
# --------------------------------------------------------------------------

def test_tier1_alphabetical_scan_filters_dotfiles():
    """alphabetical_scan returns first NON-DOTFILE in sorted order."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        deliv_dir = ws.deliverables_dir
        assert deliv_dir is not None
        os.makedirs(deliv_dir, exist_ok=True)
        # Create dotfile (should be filtered) AND regular file
        _write_file(os.path.join(deliv_dir, ".self_promoted"), "marker")
        _write_file(os.path.join(deliv_dir, "real_doc.md"), "real content")

        result = resolve_canonical_output_path(
            ws,
            filename="missing_preferred.md",
            deliverables_fallback="alphabetical_scan",
        )
        # Should return real_doc.md (not the dotfile)
        assert result is not None
        assert "real_doc.md" in result
        assert ".self_promoted" not in result


# --------------------------------------------------------------------------
# Test 7: Tier 1 "none" fallback skips to Tier 2
# --------------------------------------------------------------------------

def test_tier1_none_fallback_skips_to_tier2():
    """deliverables_fallback='none' skips Tier 1 fallback, goes to Tier 2."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        # Set up: has_deliverables=True (deliv_dir exists with files)
        # but NO matching filename, AND outputs/output.md DOES exist
        deliv_dir = ws.deliverables_dir
        assert deliv_dir is not None
        os.makedirs(deliv_dir, exist_ok=True)
        _write_file(os.path.join(deliv_dir, "wrong_name.md"), "wrong")
        # Tier 2: outputs/output.md exists
        outputs_path = ws.output_path("output.md")
        _write_file(outputs_path, "tier 2 content")

        # With deliverables_fallback="none", Tier 1 fallback is skipped,
        # so we proceed to Tier 2 and find outputs/output.md
        result = resolve_canonical_output_path(
            ws, filename="output.md", deliverables_fallback="none"
        )
        assert result == os.path.abspath(outputs_path)


# --------------------------------------------------------------------------
# Test 8: Tier 2 ONLY (CRITICAL leaf-CLI scenario)
# --------------------------------------------------------------------------

def test_tier2_only_leaf_cli_scenario():
    """CRITICAL: workspace has NO deliverables but outputs/output.md exists.

    This is the most common production case: leaf CLI inferencers
    (RovoDevCli, ClaudeCodeCli) write to outputs/output.md but DON'T
    promote to final_deliverables/. Without Tier 2 this would return None.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        # NO deliverables created — has_deliverables would be False
        # Only outputs/output.md
        outputs_path = ws.output_path("output.md")
        _write_file(outputs_path, "leaf CLI output")

        # Pre-condition check: has_deliverables MUST be False
        assert not ws.has_deliverables, (
            "Test setup error: workspace should have no deliverables"
        )

        result = resolve_canonical_output_path(ws)
        assert result == os.path.abspath(outputs_path), (
            f"Tier 2 fallback failed: got {result}, expected {outputs_path}"
        )


# --------------------------------------------------------------------------
# Test 9: Tier 1 → Tier 2 cascade
# --------------------------------------------------------------------------

def test_tier1_fail_tier2_success_cascade():
    """has_deliverables=True but Tier 1 finds nothing → falls through to Tier 2."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        deliv_dir = ws.deliverables_dir
        assert deliv_dir is not None
        os.makedirs(deliv_dir, exist_ok=True)
        # Create deliverables that do NOT match filename "output.md"
        # AND set fallback to "none" so Tier 1 fallback is skipped
        _write_file(os.path.join(deliv_dir, "other.md"), "wrong")
        # Tier 2 has the right file
        outputs_path = ws.output_path("output.md")
        _write_file(outputs_path, "tier 2 cascade hit")

        result = resolve_canonical_output_path(
            ws, filename="output.md", deliverables_fallback="none"
        )
        assert result == os.path.abspath(outputs_path)


# --------------------------------------------------------------------------
# Test 10: Absolute path guarantee
# --------------------------------------------------------------------------

def test_absolute_path_guarantee():
    """All returns must be absolute paths (CWD-independent)."""
    with tempfile.TemporaryDirectory() as tmp:
        # Use a relative path for tmp by going through abspath
        ws = _make_workspace(tmp)
        deliv_dir = ws.deliverables_dir
        assert deliv_dir is not None
        os.makedirs(deliv_dir, exist_ok=True)
        deliv_path = os.path.join(deliv_dir, "output.md")
        _write_file(deliv_path, "content")

        result = resolve_canonical_output_path(ws)
        assert result is not None
        assert os.path.isabs(result), f"Returned path must be absolute: {result}"

        # Also test Tier 2 returns absolute
        with tempfile.TemporaryDirectory() as tmp2:
            ws2 = _make_workspace(tmp2)
            outputs_path = ws2.output_path("output.md")
            _write_file(outputs_path, "tier 2")
            result2 = resolve_canonical_output_path(ws2)
            assert result2 is not None
            assert os.path.isabs(result2)


# --------------------------------------------------------------------------
# Fix #2: Canonical path resolution — Tier 1 vs Tier 2 precedence
# --------------------------------------------------------------------------

def test_canonical_path_prefers_deliverables_over_outputs():
    """Fix #2: When both deliverables/ AND outputs/ contain output.md,
    Tier 1 (deliverables) wins.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        # Create Tier 1: final_deliverables/output.md
        deliv_dir = ws.deliverables_dir
        assert deliv_dir is not None
        os.makedirs(deliv_dir, exist_ok=True)
        deliv_path = os.path.join(deliv_dir, "output.md")
        _write_file(deliv_path, "deliverables content")
        # Create Tier 2: outputs/output.md
        outputs_path = ws.output_path("output.md")
        _write_file(outputs_path, "outputs content")

        result = resolve_canonical_output_path(ws, filename="output.md")
        assert result is not None
        # Must resolve to Tier 1 (deliverables), NOT Tier 2 (outputs)
        assert result == os.path.abspath(deliv_path), (
            f"Expected Tier 1 (deliverables) path {deliv_path}, got {result}"
        )


def test_canonical_path_falls_back_to_outputs():
    """Fix #2: When only outputs/ contains output.md (no deliverables),
    Tier 2 (outputs/) is returned.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ws = _make_workspace(tmp)
        # No deliverables created — only outputs/output.md
        outputs_path = ws.output_path("output.md")
        _write_file(outputs_path, "outputs-only content")

        # Verify no deliverables exist
        assert not ws.has_deliverables, "Test setup: workspace should have no deliverables"

        result = resolve_canonical_output_path(ws, filename="output.md")
        assert result is not None
        assert result == os.path.abspath(outputs_path), (
            f"Expected Tier 2 (outputs) path {outputs_path}, got {result}"
        )

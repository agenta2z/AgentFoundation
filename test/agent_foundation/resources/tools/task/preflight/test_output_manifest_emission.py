"""Preflight tests for output_manifest emission (post-finalize hook).

Validates the manifest-emission contract added in the cached-hennessy fix plan
(Issue G — Output Manifest Index). Specifically:

  • InferencerBase has `output_manifest_index: bool = attrib(default=False)`
  • InferencerBase has `output_is_deliverable: bool = attrib(default=False)`
  • `_post_finalize_deliverable_and_manifest()` is wired into the finalize path
    in all four call sites (sync/async × fresh/resume).
  • Setting `output_is_deliverable=True` auto-enables manifest emission.
  • The manifest file is written next to the output as `<basename>_manifest.json`
    with the documented schema (schema_version, output, contributors, stats).
  • The manifest is NOT emitted when both flags are False.
  • `output_is_deliverable=True` PROMOTES agent-written outputs/ content into
    final_deliverables/ (MOVE, not copy). A framework-written <Response> summary
    (when the agent didn't write output_path) is a reference that STAYS in
    outputs/ — only agent-written files are deliverables. Self-promotion is
    detected upward via a non-empty final_deliverables/
    (workspace.has_deliverables), NOT a `.self_promoted` marker (retired).

These tests use a non-local-access stub inferencer (returns a `<Response>`
summary and writes no file); M7/M8 simulate the agent writing its deliverable
to outputs/ to exercise the promotion path — see InferencerBase._finalize_output
(and TestDeliverablePromotion for the full agent-wrote / non-writer matrix).

YAML config under test (where the flags live):
  src/agent_foundation/resources/tools/task/configs/
    default.yaml
  (line numbers shift over time — search for `output_is_deliverable`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _make_manifest_stub(output_path: str = "output.md",
                        output_is_deliverable: bool = False,
                        output_manifest_index: bool = False):
    """Stub InferencerBase with template-style file output (NOT has_local_access).

    Returns a `<Response>...</Response>`-delimited string so that
    `_finalize_output` writes it to `workspace.outputs_dir/<output_path>`.
    """
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    from attr import attrib, attrs

    @attrs(auto_attribs=False)
    class ManifestStub(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **_inference_args):
            # Wrap in <Response> tags so extract_delimited can parse it
            return "<Response>stub-content</Response>"

    return ManifestStub(
        output_path=output_path,
        output_is_deliverable=output_is_deliverable,
        output_manifest_index=output_manifest_index,
    )


def _ws(tmp, use_fdl=True):
    from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
    w = InferencerWorkspace(root=str(tmp), use_final_deliverables_folder=use_fdl)
    w.ensure_dirs()
    return w


# -------------------------------------------------------------------------
# Attribute presence (regression: someone deleting the attrs)
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_M1_output_is_deliverable_attr_exists():
    """M1: InferencerBase exposes `output_is_deliverable` defaulting to False."""
    stub = _make_manifest_stub()
    assert hasattr(stub, "output_is_deliverable"), (
        "InferencerBase must expose `output_is_deliverable` attrib for "
        "leaf-as-deliverable promotion (cached-hennessy plan, Issue D)."
    )
    assert stub.output_is_deliverable is False, "Default must be False"


@pytest.mark.preflight
def test_M2_output_manifest_index_attr_exists():
    """M2: InferencerBase exposes `output_manifest_index` defaulting to False."""
    stub = _make_manifest_stub()
    assert hasattr(stub, "output_manifest_index"), (
        "InferencerBase must expose `output_manifest_index` attrib for "
        "provenance tracking (cached-hennessy plan, Issue G)."
    )
    assert stub.output_manifest_index is False, "Default must be False"


# -------------------------------------------------------------------------
# Negative case: no flags → no manifest, no deliverable copy
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_M3_no_flags_no_manifest_emitted(tmp_path):
    """M3: When both flags are False, no manifest file is written."""
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_is_deliverable=False, output_manifest_index=False)
    stub._workspace = w
    stub.infer("input")

    output_file = os.path.join(w.outputs_dir, "output.md")
    assert os.path.isfile(output_file), "Output file should still be written"

    manifest_file = os.path.join(w.outputs_dir, "output_manifest.json")
    assert not os.path.exists(manifest_file), (
        "Manifest must NOT be emitted when both flags are False — found "
        f"unexpected manifest at {manifest_file}"
    )


# -------------------------------------------------------------------------
# Positive case: explicit manifest flag
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_M4_manifest_emitted_when_flag_set(tmp_path):
    """M4: Setting `output_manifest_index=True` emits the manifest file."""
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_manifest_index=True)
    stub._workspace = w
    stub.infer("input")

    manifest_file = os.path.join(w.outputs_dir, "output_manifest.json")
    assert os.path.isfile(manifest_file), (
        f"Manifest expected at {manifest_file} but not found. "
        "Verify _post_finalize_deliverable_and_manifest is wired in "
        "_infer_single (inferencer_base.py:1066)."
    )


# -------------------------------------------------------------------------
# Auto-enable: deliverable flag implies manifest
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_M5_deliverable_auto_enables_manifest(tmp_path):
    """M5: `output_is_deliverable=True` auto-enables manifest (per inferencer_base.py:764)."""
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_is_deliverable=True, output_manifest_index=False)
    stub._workspace = w
    stub.infer("input")

    manifest_file = os.path.join(w.outputs_dir, "output_manifest.json")
    assert os.path.isfile(manifest_file), (
        "Manifest should auto-enable when output_is_deliverable=True "
        "(see inferencer_base.py condition `OR self.output_is_deliverable`)."
    )


# -------------------------------------------------------------------------
# Schema: manifest content matches the documented v1.0 contract
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_M6_manifest_schema_v1(tmp_path):
    """M6: Manifest JSON has schema_version, output{path,size_bytes,produced_by,workspace_root},
    contributors[], stats{total}.
    """
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_manifest_index=True)
    stub._workspace = w
    stub.infer("input")

    manifest_file = os.path.join(w.outputs_dir, "output_manifest.json")
    assert os.path.isfile(manifest_file)
    with open(manifest_file) as f:
        manifest = json.load(f)

    assert manifest.get("schema_version") == "1.0", (
        f"schema_version must be '1.0', got {manifest.get('schema_version')!r}"
    )

    out = manifest.get("output")
    assert isinstance(out, dict), "output block must be a dict"
    assert "path" in out and out["path"].endswith("output.md")
    assert isinstance(out.get("size_bytes"), int) and out["size_bytes"] >= 0
    assert out.get("produced_by", "").endswith("ManifestStub") or \
           "Stub" in out.get("produced_by", ""), (
        f"produced_by should reflect the inferencer class, got {out.get('produced_by')!r}"
    )
    assert "workspace_root" in out

    assert isinstance(manifest.get("contributors"), list), "contributors must be a list"
    stats = manifest.get("stats")
    assert isinstance(stats, dict) and "total" in stats
    assert stats["total"] == len(manifest["contributors"])


# -------------------------------------------------------------------------
# Deliverable copy + .self_promoted marker
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_M7_agent_written_deliverable_moves_to_deliverables_dir(tmp_path):
    """M7: with output_is_deliverable=True, agent-written outputs/ content is
    PROMOTED (moved) into final_deliverables/.

    Design (see InferencerBase._finalize_output + TestDeliverablePromotion):
    only files the agent physically wrote to outputs/ are deliverables, and they
    are MOVED (not copied) into final_deliverables/. (A framework-written
    <Response> summary, when the agent didn't write output_path, is a reference
    that stays in outputs/ — covered by TestDeliverablePromotion.)
    """
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_is_deliverable=True)
    stub._workspace = w
    # Simulate the agent writing its deliverable to outputs/ (what a local-access
    # leaf does before finalize); the stub itself only returns a <Response>.
    with open(os.path.join(w.outputs_dir, "output.md"), "w") as f:
        f.write("# Agent deliverable\nfull content")
    stub.infer("input")

    src = os.path.join(w.outputs_dir, "output.md")
    dst = os.path.join(w.deliverables_dir, "output.md")
    assert os.path.isfile(dst), (
        f"agent-written output.md should be promoted to {dst}."
    )
    assert "Agent deliverable" in open(dst).read()
    assert not os.path.isfile(src), (
        "MOVE (not copy) semantics: output.md must NOT remain in outputs_dir "
        "after promotion to final_deliverables/."
    )


@pytest.mark.preflight
def test_M8_promoted_deliverable_self_promotes_via_has_deliverables(tmp_path):
    """M8: a promoted deliverable makes the workspace report has_deliverables —
    the marker-free self-promotion signal.

    The legacy `.self_promoted` marker FILE was retired (no longer written by
    _finalize_output). Upward surfacing via `collect_child_boundary_deliverables`
    Pass 1 now keys on a NON-EMPTY `final_deliverables/`
    (`workspace.has_deliverables`), not a marker file.
    """
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_is_deliverable=True)
    stub._workspace = w
    with open(os.path.join(w.outputs_dir, "output.md"), "w") as f:
        f.write("# Agent deliverable")
    stub.infer("input")

    assert w.has_deliverables, (
        "After promotion, deliverables_dir must be non-empty so parent BTAs "
        "detect this leaf as a self-promoted deliverable (Pass 1 keys on "
        "workspace.has_deliverables, not a marker file)."
    )
    assert os.path.isfile(os.path.join(w.deliverables_dir, "output.md"))


# -------------------------------------------------------------------------
# No-workspace safety: don't crash when workspace is None
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_M9_no_workspace_no_crash(tmp_path):
    """M9: With no workspace assigned, manifest hook is a no-op (no crash)."""
    stub = _make_manifest_stub(output_is_deliverable=True, output_manifest_index=True)
    # Do NOT assign _workspace
    # Should not crash — just returns the response unchanged
    result = stub.infer("input")
    assert result is not None

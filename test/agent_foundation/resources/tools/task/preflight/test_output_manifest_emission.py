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
  • The deliverable copy logic also writes a `.self_promoted` marker into
    deliverables_dir (required for upward surfacing via Pass 1 detection).

These tests use a non-local-access stub inferencer (so `_finalize_output`
actually writes a file) — see InferencerBase._finalize_output gate at
inferencer_base.py:723.

YAML config under test (where the flags live):
  src/openteam/server/resources/tools/task/topologies/
    breakdown_multiflow_plan_then_implement.yaml
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
        "(see inferencer_base.py:764 condition `OR self.output_is_deliverable`)."
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
def test_M7_deliverable_copies_to_deliverables_dir(tmp_path):
    """M7: `output_is_deliverable=True` copies output.md into deliverables_dir."""
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_is_deliverable=True)
    stub._workspace = w
    stub.infer("input")

    src = os.path.join(w.outputs_dir, "output.md")
    dst = os.path.join(w.deliverables_dir, "output.md")
    assert os.path.isfile(src), "Source output should exist in outputs_dir"
    assert os.path.isfile(dst), (
        f"Output should be copied to {dst}. Verify the deliverable-copy "
        "block in _post_finalize_deliverable_and_manifest "
        "(inferencer_base.py:753-762)."
    )


@pytest.mark.preflight
def test_M8_self_promoted_marker_written(tmp_path):
    """M8: `output_is_deliverable=True` writes `.self_promoted` marker.

    This marker is required for Pass 1 detection in
    `collect_child_boundary_deliverables` to surface non-boundary children's
    deliverables (cached-hennessy plan, Step 5a).
    """
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_is_deliverable=True)
    stub._workspace = w
    stub.infer("input")

    marker = os.path.join(w.deliverables_dir, ".self_promoted")
    assert os.path.isfile(marker), (
        "`.self_promoted` marker must be written into deliverables_dir so "
        "that parent BTAs detect this leaf as a self-promoted deliverable. "
        "See inferencer_base.py:760-762."
    )


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

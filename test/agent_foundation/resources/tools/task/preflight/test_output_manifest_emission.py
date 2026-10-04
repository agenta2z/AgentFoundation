"""Preflight tests for output_manifest emission (post-finalize hook).

Validates the manifest-emission contract under the Part 2 two-axis model:

  • InferencerBase has `output_manifest_index: bool = attrib(default=False)`.
  • `_finalize_output` emits the manifest ONLY when `output_manifest_index=True`
    (the old `output_is_deliverable` auto-enable was RETIRED along with the flag).
  • The manifest is written to ``artifacts/`` (Axis A = framework bookkeeping),
    named ``<basename>_manifest.json``, with the documented schema
    (schema_version, output, contributors, stats).
  • The manifest is NOT emitted when `output_manifest_index` is False.
  • ``outputs/`` IS the deliverable set — there is no move to
    ``final_deliverables/`` (retired) and no ``.self_promoted`` marker.

These tests use a non-local-access stub inferencer (returns a `<Response>`
summary and writes no file) so that `_finalize_output` materializes the
summary at ``output_path`` under ``outputs/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _make_manifest_stub(
    output_path: str = "output.md", output_manifest_index: bool = False
):
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
        output_manifest_index=output_manifest_index,
    )


def _ws(tmp):
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    w = InferencerWorkspace(root=str(tmp))
    w.ensure_dirs()
    return w


# -------------------------------------------------------------------------
# Attribute presence (regression: someone deleting the attr)
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_M2_output_manifest_index_attr_exists():
    """M2: InferencerBase exposes `output_manifest_index` defaulting to False."""
    stub = _make_manifest_stub()
    assert hasattr(stub, "output_manifest_index"), (
        "InferencerBase must expose `output_manifest_index` attrib for "
        "provenance tracking (Axis A: manifest → artifacts/)."
    )
    assert stub.output_manifest_index is False, "Default must be False"


@pytest.mark.preflight
def test_M2b_retired_deliverable_flag_absent():
    """M2b: The retired `output_is_deliverable` attrib must NOT exist (Part 2)."""
    stub = _make_manifest_stub()
    assert not hasattr(stub, "output_is_deliverable"), (
        "Retired attrib `output_is_deliverable` reappeared on InferencerBase; "
        "Part 2 removed it — outputs/ IS the deliverable set and promotion is "
        "role-based via promote_child."
    )


# -------------------------------------------------------------------------
# Negative case: flag off → no manifest
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_M3_no_flag_no_manifest_emitted(tmp_path):
    """M3: When `output_manifest_index` is False, no manifest file is written."""
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_manifest_index=False)
    stub._workspace = w
    stub.infer("input")

    output_file = os.path.join(w.outputs_dir, "output.md")
    assert os.path.isfile(output_file), "Output file should still be written"

    # Manifest must not appear in EITHER outputs/ or artifacts/.
    for cand in (
        os.path.join(w.outputs_dir, "output_manifest.json"),
        os.path.join(w.artifacts_dir, "output_manifest.json"),
    ):
        assert not os.path.exists(cand), (
            "Manifest must NOT be emitted when output_manifest_index is False — "
            f"found unexpected manifest at {cand}"
        )


# -------------------------------------------------------------------------
# Positive case: explicit manifest flag → emit to artifacts/
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_M4_manifest_emitted_to_artifacts_when_flag_set(tmp_path):
    """M4: Setting `output_manifest_index=True` emits the manifest to artifacts/."""
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_manifest_index=True)
    stub._workspace = w
    stub.infer("input")

    manifest_file = os.path.join(w.artifacts_dir, "output_manifest.json")
    assert os.path.isfile(manifest_file), (
        f"Manifest expected at {manifest_file} (artifacts/, Axis A) but not "
        "found. Verify _finalize_output emits when output_manifest_index=True."
    )
    # It must NOT be written into outputs/ (that is the deliverable set).
    assert not os.path.exists(os.path.join(w.outputs_dir, "output_manifest.json")), (
        "Manifest must live in artifacts/, not outputs/ (the deliverable set)."
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

    manifest_file = os.path.join(w.artifacts_dir, "output_manifest.json")
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
    assert out.get("produced_by", "").endswith("ManifestStub") or "Stub" in out.get(
        "produced_by", ""
    ), (
        f"produced_by should reflect the inferencer class, got {out.get('produced_by')!r}"
    )
    assert "workspace_root" in out

    assert isinstance(manifest.get("contributors"), list), "contributors must be a list"
    stats = manifest.get("stats")
    assert isinstance(stats, dict) and "total" in stats
    assert stats["total"] == len(manifest["contributors"])


# -------------------------------------------------------------------------
# outputs/ IS the deliverable set (no move to final_deliverables/, no marker)
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_M7_agent_written_output_stays_in_outputs(tmp_path):
    """M7: an agent-written file in outputs/ STAYS in outputs/ — outputs/ IS the
    deliverable set (Part 2). There is no move to a `final_deliverables/` folder.
    """
    w = _ws(tmp_path)
    stub = _make_manifest_stub(output_manifest_index=True)
    stub._workspace = w
    # Simulate the agent writing its deliverable to outputs/ (what a local-access
    # leaf does before finalize); the stub itself only returns a <Response>.
    with open(os.path.join(w.outputs_dir, "output.md"), "w") as f:
        f.write("# Agent deliverable\nfull content")
    stub.infer("input")

    src = os.path.join(w.outputs_dir, "output.md")
    assert os.path.isfile(src), (
        "agent-written output.md must remain in outputs/ (the deliverable set); "
        "Part 2 retired the move to final_deliverables/."
    )
    assert "Agent deliverable" in open(src).read()


@pytest.mark.preflight
def test_M8_nonempty_outputs_reports_has_deliverables(tmp_path):
    """M8: a non-empty outputs/ makes the workspace report has_deliverables —
    the marker-free deliverable signal (Part 2).

    The legacy `.self_promoted` marker FILE was retired. Upward surfacing keys
    on a NON-EMPTY ``outputs/`` (``workspace.has_deliverables``), not a marker.
    """
    w = _ws(tmp_path)
    stub = _make_manifest_stub()
    stub._workspace = w
    with open(os.path.join(w.outputs_dir, "output.md"), "w") as f:
        f.write("# Agent deliverable")
    stub.infer("input")

    assert w.has_deliverables, (
        "outputs/ is non-empty so has_deliverables must be True — this is the "
        "marker-free self-promotion signal parents key on."
    )
    # No retired marker file should be written by finalize.
    assert not os.path.exists(os.path.join(w.outputs_dir, ".self_promoted")), (
        "The `.self_promoted` marker was retired and must not be written."
    )


# -------------------------------------------------------------------------
# No-workspace safety: don't crash when workspace is None
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_M9_no_workspace_no_crash(tmp_path):
    """M9: With no workspace assigned, manifest hook is a no-op (no crash)."""
    stub = _make_manifest_stub(output_manifest_index=True)
    # Do NOT assign _workspace
    # Should not crash — just returns the response unchanged
    result = stub.infer("input")
    assert result is not None

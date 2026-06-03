"""Preflight tests for the per-round audit log + nav symlinks.

Validates the round-audit framework added in the cached-hennessy plan
(Issues E + F). Specifically:

  • DualInferencer exposes `enable_round_audit: bool = attrib(default=True)`
  • `_record_round_audit(round_idx, phase, inferencer, extra=None)` is callable
  • Calling it appends a structured JSONL entry to `outputs/round_log.jsonl`
  • The entry contains: round, phase, inferencer_class, inferencer_workspace,
    timestamp, and any `extra` keys passed through.
  • A navigation symlink is created at `children/round_NN/<phase>` pointing
    to the inferencer's workspace root.
  • If `enable_round_audit=False`, nothing is written.
  • If the audit raises (e.g., disk full mid-write), it is caught and logged
    but never propagates — the inference run must not be crashed by audit.

Fix locations:
  • dual_inferencer.py:448-494 — fail-safe `_record_round_audit`
  • multi_flow_dual_inferencer.py:625-629 — always-log dispatch (Fix #4)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _ws(tmp, name="dual_root", use_fdl=True):
    """Create a workspace under tmp_path/<name>/ with full layout."""
    from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
    root = os.path.join(str(tmp), name)
    os.makedirs(root, exist_ok=True)
    w = InferencerWorkspace(root=root, use_final_deliverables_folder=use_fdl)
    w.ensure_dirs()
    # children_dir is created on demand — touch it now
    os.makedirs(w.children_dir, exist_ok=True)
    return w


def _make_dual_with_audit(parent_ws, child_ws, enable_round_audit=True):
    """Construct a minimal DualInferencer-like object with the audit method.

    Rather than constructing a real Dual (which has many required slots),
    we instantiate the actual DualInferencer class but only exercise the
    `_record_round_audit` method directly, which is the unit under test.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    from attr import attrib, attrs

    # A minimal child inferencer with a workspace assigned
    @attrs(auto_attribs=False)
    class _ChildStub(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **_inference_args):
            return "child"

    child = _ChildStub()
    child._workspace = child_ws

    # Use object.__new__ to bypass attrs init — we only need the audit method
    # and the two attributes it reads. This avoids constructing all required
    # Dual slots for a unit-level test of one method.
    dual = object.__new__(DualInferencer)
    dual._workspace = parent_ws
    dual.enable_round_audit = enable_round_audit
    return dual, child


# -------------------------------------------------------------------------
# Attribute presence
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_RA1_enable_round_audit_attr_exists():
    """RA1: DualInferencer declares `enable_round_audit` defaulting to True."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )
    import attr

    fields = {f.name: f for f in attr.fields(DualInferencer)}
    assert "enable_round_audit" in fields, (
        "DualInferencer must declare `enable_round_audit` attrib for round "
        "audit toggling (cached-hennessy plan, Step 7)."
    )
    assert fields["enable_round_audit"].default is True, (
        "Default must be True so audit is on by default for new runs."
    )


@pytest.mark.preflight
def test_RA2_record_round_audit_method_exists():
    """RA2: DualInferencer exposes `_record_round_audit` method."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )

    assert hasattr(DualInferencer, "_record_round_audit"), (
        "DualInferencer must expose `_record_round_audit` "
        "(cached-hennessy plan, Step 7)."
    )
    assert callable(DualInferencer._record_round_audit)


# -------------------------------------------------------------------------
# Functional: log file written + content shape
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_RA3_audit_writes_jsonl_entry(tmp_path):
    """RA3: A single _record_round_audit call appends one valid JSONL line."""
    parent = _ws(tmp_path, "parent")
    child = _ws(tmp_path, "child")
    dual, child_inf = _make_dual_with_audit(parent, child)

    dual._record_round_audit(
        round_idx=1, phase="propose", inferencer=child_inf,
    )

    log_path = os.path.join(parent.outputs_dir, "round_log.jsonl")
    assert os.path.isfile(log_path), (
        f"round_log.jsonl expected at {log_path} but missing. "
        "Verify dual_inferencer.py:474-477 writes correctly."
    )

    with open(log_path) as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == 1, f"Expected exactly 1 JSONL line, got {len(lines)}"

    entry = json.loads(lines[0])
    for key in ("round", "phase", "inferencer_class",
                "inferencer_workspace", "timestamp"):
        assert key in entry, f"Missing required key {key!r} in entry: {entry}"
    assert entry["round"] == 1
    assert entry["phase"] == "propose"
    assert entry["inferencer_class"].endswith("ChildStub") or \
           "Stub" in entry["inferencer_class"]
    assert entry["inferencer_workspace"].startswith(child.root)


@pytest.mark.preflight
def test_RA4_extra_kwargs_merged_into_entry(tmp_path):
    """RA4: `extra=` kwargs (e.g., MFDual dispatch metadata) are merged in."""
    parent = _ws(tmp_path, "parent")
    child = _ws(tmp_path, "child")
    dual, child_inf = _make_dual_with_audit(parent, child)

    dual._record_round_audit(
        round_idx=2, phase="review_dispatch", inferencer=child_inf,
        extra={"mfdual_dispatch": {"winner_idx": 1, "ranking": [1, 0, 2]}},
    )

    log_path = os.path.join(parent.outputs_dir, "round_log.jsonl")
    with open(log_path) as f:
        entry = json.loads(f.readline())

    assert "mfdual_dispatch" in entry, (
        "Extra kwargs must be merged into the JSONL entry (used by MFDual "
        "to record review_dispatch metadata)."
    )
    assert entry["mfdual_dispatch"]["winner_idx"] == 1
    assert entry["mfdual_dispatch"]["ranking"] == [1, 0, 2]


@pytest.mark.preflight
def test_RA5_multiple_calls_append(tmp_path):
    """RA5: Multiple calls produce a JSONL file with one line per call."""
    parent = _ws(tmp_path, "parent")
    child = _ws(tmp_path, "child")
    dual, child_inf = _make_dual_with_audit(parent, child)

    for round_idx, phase in [(1, "propose"), (1, "review"), (1, "fix"),
                             (2, "review"), (2, "fix")]:
        dual._record_round_audit(round_idx, phase, child_inf)

    log_path = os.path.join(parent.outputs_dir, "round_log.jsonl")
    with open(log_path) as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == 5, f"Expected 5 entries, got {len(lines)}"

    rounds = [json.loads(ln)["round"] for ln in lines]
    phases = [json.loads(ln)["phase"] for ln in lines]
    assert rounds == [1, 1, 1, 2, 2]
    assert phases == ["propose", "review", "fix", "review", "fix"]


# -------------------------------------------------------------------------
# Navigation symlinks (or pointer file fallback)
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_RA6_navigation_symlink_or_pointer_created(tmp_path):
    """RA6: A navigation entry exists at children/round_NN/<phase>.

    Accepts either a real symlink (POSIX) or a `<phase>.pointer.txt`
    fallback (Windows / restricted environments).
    """
    parent = _ws(tmp_path, "parent")
    child = _ws(tmp_path, "child")
    dual, child_inf = _make_dual_with_audit(parent, child)

    dual._record_round_audit(round_idx=1, phase="propose", inferencer=child_inf)

    nav_dir = os.path.join(parent.children_dir, "round_01")
    assert os.path.isdir(nav_dir), f"Nav dir not created: {nav_dir}"

    link_path = os.path.join(nav_dir, "propose")
    pointer_path = os.path.join(nav_dir, "propose.pointer.txt")
    assert os.path.islink(link_path) or os.path.isfile(pointer_path), (
        f"Expected either a symlink at {link_path} or pointer file at "
        f"{pointer_path}; neither found."
    )


# -------------------------------------------------------------------------
# Toggle: disabled → no log, no nav
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_RA7_disabled_audit_writes_nothing(tmp_path):
    """RA7: With `enable_round_audit=False`, nothing is written."""
    parent = _ws(tmp_path, "parent")
    child = _ws(tmp_path, "child")
    dual, child_inf = _make_dual_with_audit(parent, child, enable_round_audit=False)

    dual._record_round_audit(round_idx=1, phase="propose", inferencer=child_inf)

    log_path = os.path.join(parent.outputs_dir, "round_log.jsonl")
    nav_dir = os.path.join(parent.children_dir, "round_01")
    assert not os.path.exists(log_path), (
        "round_log.jsonl must NOT be written when enable_round_audit=False"
    )
    assert not os.path.exists(nav_dir), (
        "Nav directory must NOT be created when enable_round_audit=False"
    )


# -------------------------------------------------------------------------
# Fail-safe: audit failures do NOT crash the run
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_RA8_failsafe_audit_swallows_exceptions(tmp_path, monkeypatch):
    """RA8: When the audit internals raise (e.g., disk full), the call returns
    cleanly. This is the fail-safe try/except wrapper added in Fix #7.
    """
    parent = _ws(tmp_path, "parent")
    child = _ws(tmp_path, "child")
    dual, child_inf = _make_dual_with_audit(parent, child)

    # Force a write failure by replacing `open` at the target module level
    # with one that raises OSError for the round_log.jsonl path.
    import builtins
    real_open = builtins.open

    def _hostile_open(path, *a, **kw):
        if str(path).endswith("round_log.jsonl"):
            raise OSError("simulated disk full")
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", _hostile_open)

    # Must NOT raise — the wrapper should catch and log a warning.
    dual._record_round_audit(round_idx=1, phase="propose", inferencer=child_inf)


# -------------------------------------------------------------------------
# No-workspace / no-child-workspace safety
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_RA9_no_parent_workspace_is_noop(tmp_path):
    """RA9: If parent has no workspace, audit is a no-op (no crash)."""
    child = _ws(tmp_path, "child")
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    from attr import attrs

    @attrs(auto_attribs=False)
    class _ChildStub(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **_inference_args):
            return "x"

    child_inf = _ChildStub()
    child_inf._workspace = child

    dual = object.__new__(DualInferencer)
    dual._workspace = None
    dual.enable_round_audit = True

    # Should not raise
    dual._record_round_audit(round_idx=1, phase="propose", inferencer=child_inf)


@pytest.mark.preflight
def test_RA10_no_child_workspace_is_noop(tmp_path):
    """RA10: If the inferencer being audited has no workspace, audit is a no-op."""
    parent = _ws(tmp_path, "parent")
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    from attr import attrs

    @attrs(auto_attribs=False)
    class _ChildStub(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **_inference_args):
            return "x"

    child_inf = _ChildStub()  # No _workspace assigned

    dual = object.__new__(DualInferencer)
    dual._workspace = parent
    dual.enable_round_audit = True

    # Should not raise — early return at dual_inferencer.py:457
    dual._record_round_audit(round_idx=1, phase="propose", inferencer=child_inf)

    # And no log file should be created
    log_path = os.path.join(parent.outputs_dir, "round_log.jsonl")
    assert not os.path.exists(log_path), (
        "Audit must early-return when child has no workspace — no log file"
    )

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Phase K10 — SOPController serialization round-trip parity.

Load-bearing assertion for the SOP extraction: pre- and post-migration
``_conversation_blob`` produce identical dicts for the SOP portion; and
``restore(state)`` → ``serialize()`` is stable.

Uses a synthesized blob (structurally equivalent to what
`_conversation_blob` emitted pre-Phase K) rather than a captured artifact,
because the migration has already shipped in this repo. The invariant
being tested is: the SOPController.serialize/restore contract preserves
byte-identity for the two owned keys (`sop_state`, `suspended_sops`).
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
    SOPController,
)
from agent_foundation.common.workflow.sop_state import SOPState


def _fake_state(name: str, current: str = "1") -> SOPState:
    """Fresh SOPState suitable for serialize/restore parity checks.

    Pre-populates `sop_name`, sets `.sop` to a stub (so `reload_sop_definition`
    early-returns and does NOT try to hit the real SOP registry).
    """
    s = SOPState(sop_name=name, current_phase=current)
    s.phase_required_tools = {current: {"toolA", "toolB"}}
    s.phase_executed_tools = {current: {"toolA"}}
    s.phase_outputs = {}
    s.goto_counts = {}
    s.completed_phases = []
    s.user_input_gate_passed = False
    # Stub SOP so reload_sop_definition sees `state.sop is not None` and skips the registry.
    s.sop = SimpleNamespace(phases=[], phase_required_tools={}, tool_to_phase_map={})
    return s


def test_empty_serialize_stable() -> None:
    """Fresh controller round-trips to {sop_state:None, suspended_sops:[]}."""
    ctrl = SOPController()
    blob = ctrl.serialize()
    assert blob == {"sop_state": None, "suspended_sops": []}

    # Restore + re-serialize must be idempotent.
    ctrl2 = SOPController()
    ctrl2.restore(blob)
    assert ctrl2.serialize() == blob


def test_active_sop_round_trip_progress_stable() -> None:
    """serialize → restore → serialize is stable for progress fields.

    Not byte-identical: ``reload_sop_definition`` intentionally REFRESHES
    ``phase_required_tools`` + ``tool_phase_map`` from the freshly-loaded
    SOP graph (Fix D — so a session resumed after an SOP edit doesn't run
    the new graph with stale persisted maps). Progress fields
    (``completed_phases``, ``phase_executed_tools``, ``phase_outputs``,
    ``current_phase``, ``user_input_gate_passed``) MUST be stable.
    """
    ctrl = SOPController()
    ctrl.sop_state = _fake_state("code_optimization", "0")
    blob_1 = ctrl.serialize()

    ctrl2 = SOPController()
    ctrl2.restore(blob_1)
    blob_2 = ctrl2.serialize()

    # Progress fields — MUST be byte-identical.
    for stable_key in (
        "sop_name",
        "current_phase",
        "phase_status",
        "completed_phases",
        "phase_outputs",
        "phase_executed_tools",
        "goto_counts",
        "user_input_gate_passed",
        "yolo_mode",
    ):
        assert blob_1["sop_state"][stable_key] == blob_2["sop_state"][stable_key], (
            f"{stable_key} must round-trip byte-identically"
        )

    # Second round-trip must now be fully stable (definition-refresh only
    # happens once — from stale-map to fresh-map).
    ctrl3 = SOPController()
    ctrl3.restore(blob_2)
    blob_3 = ctrl3.serialize()
    assert blob_2 == blob_3, "Second round-trip must be stable"


def test_reattach_sop_false_host_pre_set_state_survives() -> None:
    """OpenStartup round-resume host contract: reattach_sop=False MUST leave
    the pre-set sop_state / _suspended_sops UNCHANGED."""
    ctrl = SOPController()
    host_prepared_state = _fake_state("code_optimization", "2")
    ctrl.sop_state = host_prepared_state
    ctrl._suspended_sops = [_fake_state("model_optimization")]

    # An incoming blob that WOULD replace state — but reattach_sop=False must skip.
    incoming = {
        "sop_state": {"sop_name": "would_replace", "current_phase": "5"},
        "suspended_sops": [],
    }
    ctrl.restore(incoming, reattach_sop=False)

    assert ctrl.sop_state is host_prepared_state, (
        "Pre-set sop_state must survive reattach_sop=False (OpenStartup contract)"
    )
    assert len(ctrl._suspended_sops) == 1


def test_suspended_stack_round_trip_preserves_order() -> None:
    """Multi-SOP suspension stack round-trips with most-recent-first ordering."""
    ctrl = SOPController()
    ctrl._suspended_sops = [
        _fake_state("model_optimization"),
        _fake_state("code_optimization"),
        _fake_state("sop_creation"),
    ]
    blob = ctrl.serialize()
    assert [s["sop_name"] for s in blob["suspended_sops"]] == [
        "model_optimization",
        "code_optimization",
        "sop_creation",
    ]

    ctrl2 = SOPController()
    ctrl2.restore(blob)
    assert [s.sop_name for s in ctrl2._suspended_sops] == [
        "model_optimization",
        "code_optimization",
        "sop_creation",
    ]


def test_restore_does_not_reset_paused_flag() -> None:
    """Per Phase K6: `_paused` reset is CI's responsibility (unconditional loop-frame),
    NOT SOPController.restore(). The restore path must leave `_paused` untouched
    so the CI can control it independently."""
    ctrl = SOPController()
    ctrl._paused = True

    ctrl.restore({"sop_state": None, "suspended_sops": []})

    # `_paused` was NOT touched by restore.
    assert ctrl._paused is True

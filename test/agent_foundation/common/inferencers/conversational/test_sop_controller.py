# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""SOPController unit tests — Phase K extraction.

Focused on the load-bearing invariants:
- ``is_paused`` public property (getter+setter) drives the field.
- ``serialize()`` emits the ``sop_state`` + ``suspended_sops`` keys
  byte-identical to the pre-extraction ``_conversation_blob`` payload.
- ``restore(state, reattach_sop=True)`` round-trips serialize state.
- ``restore(state, reattach_sop=False)`` DOES NOT touch sop_state /
  suspended_sops — preserves the OpenStartup round-resume host contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
    SOPController,
)


def test_default_state_matches_pre_migration_post_init() -> None:
    ctrl = SOPController(prompt_renderer_ref=None)
    assert ctrl.sop_state is None
    assert ctrl._suspended_sops == []
    assert ctrl._paused is False
    assert ctrl._pending_followup is None
    assert ctrl._auto_shutdown_on_sop_complete is False
    assert ctrl.is_paused is False
    assert ctrl.is_active is False


def test_is_paused_property_getter_and_setter() -> None:
    ctrl = SOPController(prompt_renderer_ref=None)
    assert ctrl.is_paused is False
    ctrl.is_paused = True
    assert ctrl.is_paused is True
    assert ctrl._paused is True
    ctrl.is_paused = False
    assert ctrl._paused is False


def test_serialize_empty_state() -> None:
    ctrl = SOPController(prompt_renderer_ref=None)
    blob = ctrl.serialize()
    assert blob == {"sop_state": None, "suspended_sops": []}


def test_serialize_with_active_sop_state() -> None:
    ctrl = SOPController(prompt_renderer_ref=None)
    fake_state = MagicMock()
    fake_state.to_dict.return_value = {"sop_name": "foo", "current_phase": "1"}
    ctrl.sop_state = fake_state
    blob = ctrl.serialize()
    assert blob["sop_state"] == {"sop_name": "foo", "current_phase": "1"}
    assert blob["suspended_sops"] == []


def test_serialize_with_suspended_stack() -> None:
    ctrl = SOPController(prompt_renderer_ref=None)
    s1 = MagicMock()
    s1.to_dict.return_value = {"sop_name": "a"}
    s2 = MagicMock()
    s2.to_dict.return_value = {"sop_name": "b"}
    ctrl._suspended_sops = [s1, s2]
    blob = ctrl.serialize()
    assert blob["suspended_sops"] == [{"sop_name": "a"}, {"sop_name": "b"}]


def test_restore_reattach_false_is_noop_on_sop_state() -> None:
    """OpenStartup round-resume host contract: reattach_sop=False MUST leave
    the pre-set sop_state / _suspended_sops UNCHANGED (host already restored
    them via extra-dirs-aware factory)."""
    ctrl = SOPController(prompt_renderer_ref=None)
    pre_state = MagicMock()
    pre_state.to_dict.return_value = {"sop_name": "host_prepared"}
    ctrl.sop_state = pre_state
    ctrl._suspended_sops = [MagicMock()]

    # A blob that WOULD replace state — but reattach_sop=False must skip.
    incoming = {"sop_state": {"sop_name": "would_replace"}, "suspended_sops": []}
    ctrl.restore(incoming, reattach_sop=False)

    # sop_state and _suspended_sops are STILL the host-set values.
    assert ctrl.sop_state is pre_state
    assert len(ctrl._suspended_sops) == 1


def test_restore_reattach_true_clears_when_no_sop_state() -> None:
    ctrl = SOPController(prompt_renderer_ref=None)
    ctrl.sop_state = MagicMock()
    ctrl._suspended_sops = [MagicMock()]

    ctrl.restore({"sop_state": None, "suspended_sops": []}, reattach_sop=True)

    assert ctrl.sop_state is None
    assert ctrl._suspended_sops == []

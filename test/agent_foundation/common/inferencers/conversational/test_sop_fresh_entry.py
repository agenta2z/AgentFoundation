"""``SOPController.enter`` keeps at most one instance per SOP name (active or
suspended): ``fresh`` discards the in-progress instance and starts over, and
an SOP that already has an instance is entered only with ``fresh``. The
classic ``/sop`` command and the native ``enter_sop`` tool both enter here."""

from __future__ import annotations

import unittest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
    SOPController,
)

_SOP = "code_optimization"


class FreshEntryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = SOPController()

    def test_fresh_discards_the_suspended_instance(self) -> None:
        self.ctrl.enter(_SOP)
        discarded = self.ctrl.sop_state
        self.ctrl.cmd_exit_sop()
        text = self.ctrl.enter(_SOP, fresh=True)
        self.assertEqual(text, f"Entered SOP '{_SOP}'.")
        self.assertNotEqual(self.ctrl.sop_state.instance_id, discarded.instance_id)
        self.assertEqual(self.ctrl._suspended_sops, [])
        self.assertEqual(self.ctrl.format_suspended_sops(), ("", ""))
        self.assertEqual(self.ctrl.resume(_SOP), "No suspended SOPs to resume.")
        self.assertEqual(self.ctrl.serialize()["suspended_sops"], [])

    def test_suspending_the_fresh_instance_leaves_one_entry_for_the_name(self) -> None:
        self.ctrl.enter(_SOP)
        self.ctrl.cmd_pause_sop()
        self.ctrl.enter(_SOP, fresh=True)
        fresh = self.ctrl.sop_state
        self.ctrl.cmd_exit_sop()
        self.assertEqual(self.ctrl._suspended_sops, [fresh])
        self.assertEqual(
            self.ctrl.format_suspended_sops(),
            ("", f"- **{_SOP}** ({fresh.sop_status})"),
        )
        self.assertIn(f"Resumed SOP '{_SOP}'", self.ctrl.resume(_SOP))
        self.assertIs(self.ctrl.sop_state, fresh)
        self.assertEqual(self.ctrl._suspended_sops, [])

    def test_fresh_pauses_a_different_active_sop(self) -> None:
        self.ctrl.enter(_SOP)
        self.ctrl.cmd_exit_sop()
        self.ctrl.enter("model_optimization")
        other = self.ctrl.sop_state
        self.ctrl.enter(_SOP, fresh=True, request="from scratch")
        self.assertEqual(self.ctrl.sop_state.sop_name, _SOP)
        self.assertEqual(self.ctrl._suspended_sops, [other])
        self.assertEqual(other.suspension_reason, "paused")
        self.assertEqual(self.ctrl.consume_pending_followup(), "from scratch")

    def test_fresh_replaces_the_active_instance_of_the_same_sop(self) -> None:
        self.ctrl.enter(_SOP)
        discarded = self.ctrl.sop_state
        text = self.ctrl.enter(_SOP, fresh=True)
        self.assertEqual(text, f"Entered SOP '{_SOP}'.")
        self.assertIsNot(self.ctrl.sop_state, discarded)
        self.assertEqual(self.ctrl._suspended_sops, [])

    def test_entering_the_active_sop_without_fresh_is_refused(self) -> None:
        self.ctrl.enter(_SOP)
        active = self.ctrl.sop_state
        text = self.ctrl.enter(_SOP, request="again")
        self.assertEqual(
            text,
            f"SOP '{_SOP}' is already active ({active.sop_status}). "
            f"Continue it, or use /sop {_SOP} --fresh to start over.",
        )
        self.assertIs(self.ctrl.sop_state, active)
        self.assertEqual(self.ctrl._suspended_sops, [])
        self.assertIsNone(self.ctrl.consume_pending_followup())

    def test_a_failed_fresh_entry_discards_nothing(self) -> None:
        self.ctrl.enter(_SOP)
        self.ctrl.cmd_exit_sop()
        self.ctrl.enter("model_optimization")
        active, suspended = self.ctrl.sop_state, list(self.ctrl._suspended_sops)

        def failing_build(name: str, *, yolo: bool = False):
            return None, f"SOP '{name}' could not be loaded"

        text = self.ctrl.enter(_SOP, fresh=True, build_state=failing_build)
        self.assertEqual(text, f"SOP '{_SOP}' could not be loaded")
        self.assertIs(self.ctrl.sop_state, active)
        self.assertEqual(self.ctrl._suspended_sops, suspended)

    def test_the_fresh_flag_of_the_sop_command(self) -> None:
        self.ctrl.cmd_sop(_SOP)
        self.ctrl.cmd_exit_sop()
        self.assertIn("--fresh to start over", self.ctrl.cmd_sop(_SOP))
        self.assertEqual(self.ctrl.cmd_sop(f"{_SOP} --fresh"), f"Entered SOP '{_SOP}'.")
        self.assertEqual(self.ctrl._suspended_sops, [])

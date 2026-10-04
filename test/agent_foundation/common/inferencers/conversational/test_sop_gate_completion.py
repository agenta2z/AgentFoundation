"""A satisfied user-input gate completing a "requires user input" phase that has
no tools (``prepare_sop_for_turn``, run before every classic render and every
native vendor turn) moves the SOP on exactly as ``check_phase_completion`` does
for every other phase completion."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
    SOPController,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_feed import (
    build_sop_feed,
    prepare_sop_for_turn,
)
from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus

_SOPS = {
    "gated": """# Gated

Fixture SOP whose first phase only asks the user.

## Phase 0 -- Ask
[__initial__]

[__requires user input__] Ask the user what to do.

## Phase 1 -- Act
[__depends on__ Phase 0]

Act on the answer.
""",
    "ask_only": """# Ask Only

Fixture SOP whose only phase asks the user.

## Phase 0 -- Ask
[__initial__]

[__requires user input__] Ask the user what to do.
""",
    "parallel": """# Parallel

Fixture SOP whose input phase runs alongside a tool phase.

## Phase 0 -- Build
[__initial__]

Build it.

**Tools**[__required__]:
- /build-it

## Phase 1 -- Confirm

[__requires user input__] Ask the user to confirm the plan.

## Phase 2 -- Finish
[__depends on__ Phase 0, Phase 1]

Finish.
""",
}

_STATE_FIELDS = (
    "current_phase",
    "phase_status",
    "completed_phases",
    "user_input_gate_passed",
)


class _FakeBase:
    effective_cwd = "/workspace/project"
    system_prompt = ""


class GateCompletionTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="af_sop_gate_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.extra_dirs = [tmp / "sops"]
        for name, body in _SOPS.items():
            folder = tmp / "sops" / name
            folder.mkdir(parents=True)
            (folder / "SOP.md").write_text(body)

    def _entered_with_open_gate(
        self, name: str, ctrl: SOPController | None = None
    ) -> SOPController:
        ctrl = ctrl or SOPController(extra_sop_dirs=self.extra_dirs)
        ctrl.enter(name)
        ctrl.sop_state.user_input_gate_passed = True
        return ctrl

    def test_completing_the_current_phase_makes_the_next_one_current(self) -> None:
        ctrl = self._entered_with_open_gate("gated")
        prepare_sop_for_turn(ctrl)
        state = ctrl.sop_state
        self.assertEqual(state.completed_phase_ids(), ["0"])
        self.assertEqual(state.current_phase, "1")
        self.assertEqual(state.phase_status, PhaseStatus.RUNNING)
        self.assertFalse(state.user_input_gate_passed)
        self.assertEqual(
            [line for line in state.sop_outline.splitlines() if "(current)" in line],
            ["1 Act ▶ (current)"],
        )
        self.assertEqual(state.sop_status, "Current Phase (1 of 2): Act")
        self.assertIn(
            "Act on the answer.", build_sop_feed(ctrl, {}).sop_nextstep_guidance
        )

    def test_the_gate_leaves_the_state_check_phase_completion_leaves(self) -> None:
        by_gate = self._entered_with_open_gate("gated")
        self.assertTrue(by_gate.consume_gate_for_no_tools_requires_input_phase())
        by_check = self._entered_with_open_gate("gated")
        by_check.check_phase_completion()
        self.assertEqual(
            {f: getattr(by_gate.sop_state, f) for f in _STATE_FIELDS},
            {f: getattr(by_check.sop_state, f) for f in _STATE_FIELDS},
        )

    def test_completing_the_last_phase_completes_the_sop(self) -> None:
        shutdowns: list[bool] = []
        ctrl = SOPController(
            extra_sop_dirs=self.extra_dirs,
            request_shutdown=lambda: shutdowns.append(True),
        )
        ctrl._auto_shutdown_on_sop_complete = True
        self._entered_with_open_gate("ask_only", ctrl)
        prepare_sop_for_turn(ctrl)
        state = ctrl.sop_state
        self.assertEqual(state.completed_phase_ids(), ["0"])
        self.assertIsNone(state.current_phase)
        self.assertEqual(state.phase_status, PhaseStatus.COMPLETED)
        self.assertEqual(shutdowns, [True])

    def test_completing_another_available_phase_keeps_the_current_one(self) -> None:
        ctrl = self._entered_with_open_gate("parallel")
        prepare_sop_for_turn(ctrl)
        state = ctrl.sop_state
        self.assertEqual(state.completed_phase_ids(), ["1"])
        self.assertEqual(state.current_phase, "0")
        self.assertEqual(state.phase_status, PhaseStatus.IDLE)
        self.assertFalse(state.user_input_gate_passed)

    def test_classic_prompt_shows_the_guided_phase_as_current(self) -> None:
        ci = ConversationalInferencer(
            base_inferencer=_FakeBase(), extra_sop_dirs=self.extra_dirs
        )
        ci.sop_controller.enter("gated")
        ci.sop_state.user_input_gate_passed = True
        rendered = ci._render_prompt("go on")
        self.assertIn("1 Act ▶ (current)\n</SOPPhases>", rendered)
        self.assertIn("<SOPStatus>\nCurrent Phase (1 of 2): Act\n", rendered)
        self.assertIn("#### Act\n\nAct on the answer.", rendered)

    def test_classic_prompt_marks_the_completed_phase_done(self) -> None:
        ci = ConversationalInferencer(
            base_inferencer=_FakeBase(), extra_sop_dirs=self.extra_dirs
        )
        ci.sop_controller.enter("gated")
        ci.sop_state.user_input_gate_passed = True
        rendered = ci._render_prompt("go on")
        self.assertIn(
            "<SOPPhases>\n0 Ask ✓ (done)\n1 Act ▶ (current)\n</SOPPhases>", rendered
        )

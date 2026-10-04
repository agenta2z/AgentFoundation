"""``SOPController.enter`` / ``resume`` (typed SOP entry shared by the slash
commands and the native SOP tools), the pause/exit guards, extra SOP
directories (``load_sop(name, extra_dirs=...)``, definition reload, the CI's
``extra_sop_dirs`` forwarding to the controller)."""

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
from agent_foundation.common.workflow.sop_state import SOPState
from agent_foundation.resources.sops.registry import _SOPS_DIR, load_sop, SOPNotFound

_EXTRA_SOP = "extra_only"
_EXTRA_SOP_MD = """# Extra Only

Fixture SOP that exists only in an extra SOP directory.

## Phase 0 -- Start
[__initial__]

Greet the user.

## Phase 1 -- Finish
[__depends on__ Phase 0]

Say goodbye.
"""


def _write_sop(root: Path, name: str, body: str = _EXTRA_SOP_MD) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SOP.md").write_text(body)
    return folder


class _FakeBase:
    effective_cwd = "/workspace/project"
    system_prompt = ""


def _make_ci(**kwargs) -> ConversationalInferencer:
    return ConversationalInferencer(base_inferencer=_FakeBase(), **kwargs)


class _TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="af_sop_enter_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.extra_dir = self.tmp / "extra"
        _write_sop(self.extra_dir, _EXTRA_SOP)


class EnterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = SOPController()

    def test_enter_from_idle(self) -> None:
        text = self.ctrl.enter("code_optimization")
        self.assertEqual(text, "Entered SOP 'code_optimization'.")
        self.assertEqual(self.ctrl.sop_state.sop_name, "code_optimization")
        self.assertIsNone(self.ctrl.consume_pending_followup())

    def test_request_is_taken_as_given(self) -> None:
        request = '--fresh  please, "quoted"  --yolo'
        text = self.ctrl.enter("code_optimization", request=f"  {request} ")
        self.assertEqual(
            text, f"Entered SOP 'code_optimization'. Starting on: {request}"
        )
        self.assertFalse(self.ctrl.sop_state.yolo_mode)
        self.assertEqual(self.ctrl.consume_pending_followup(), request)

    def test_blank_request_sets_no_followup(self) -> None:
        self.ctrl.enter("code_optimization", request="   ")
        self.assertIsNone(self.ctrl.consume_pending_followup())

    def test_entering_pauses_the_active_sop(self) -> None:
        self.ctrl.enter("code_optimization")
        first = self.ctrl.sop_state
        self.ctrl.enter("model_optimization")
        self.assertEqual(self.ctrl.sop_state.sop_name, "model_optimization")
        self.assertEqual(self.ctrl._suspended_sops, [first])
        self.assertEqual(first.suspension_reason, "paused")
        self.assertTrue(first.suspended_at)

    def test_in_progress_sop_is_refused_unless_fresh(self) -> None:
        self.ctrl.enter("code_optimization")
        self.ctrl.cmd_exit_sop()
        text = self.ctrl.enter("code_optimization", request="again")
        self.assertIn("You have an in-progress 'code_optimization'", text)
        self.assertIsNone(self.ctrl.sop_state)
        self.assertIsNone(self.ctrl.consume_pending_followup())

        text = self.ctrl.enter("code_optimization", fresh=True)
        self.assertEqual(text, "Entered SOP 'code_optimization'.")
        self.assertEqual(self.ctrl.sop_state.sop_name, "code_optimization")

    def test_unknown_sop_reports_the_error_and_changes_nothing(self) -> None:
        self.ctrl.enter("code_optimization")
        active = self.ctrl.sop_state
        text = self.ctrl.enter("no_such_sop", request="x")
        self.assertIn("SOP 'no_such_sop' not found", text)
        self.assertIs(self.ctrl.sop_state, active)
        self.assertEqual(self.ctrl._suspended_sops, [])
        self.assertIsNone(self.ctrl.consume_pending_followup())

    def test_yolo_reaches_the_state_and_the_host_setter(self) -> None:
        calls: list[bool] = []
        self.ctrl.enter("code_optimization", yolo=True, yolo_mode_setter=calls.append)
        self.assertTrue(self.ctrl.sop_state.yolo_mode)
        self.assertEqual(calls, [True])

    def test_injected_state_builder_replaces_the_registry(self) -> None:
        built: list[tuple[str, bool]] = []

        def build_state(name: str, *, yolo: bool = False):
            built.append((name, yolo))
            return SOPState(sop=None, sop_name=name, yolo_mode=yolo), None

        self.ctrl.enter("anything", yolo=True, build_state=build_state)
        self.assertEqual(built, [("anything", True)])
        self.assertEqual(self.ctrl.sop_state.sop_name, "anything")

    def test_cmd_sop_parses_the_argument_line_into_enter(self) -> None:
        text = self.ctrl.cmd_sop("code_optimization --yolo tidy   the  repo")
        self.assertEqual(
            text, "Entered SOP 'code_optimization'. Starting on: tidy the repo"
        )
        self.assertTrue(self.ctrl.sop_state.yolo_mode)
        self.assertEqual(
            self.ctrl.cmd_sop(""), "Usage: /sop <name> [--yolo] [--fresh] [request...]"
        )


class ResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = SOPController()
        self.ctrl.enter("code_optimization")
        self.ctrl.enter("model_optimization")
        self.ctrl.cmd_exit_sop()
        # Suspended, most recent first: model_optimization, code_optimization.

    def test_nothing_suspended(self) -> None:
        self.assertEqual(SOPController().resume(), "No suspended SOPs to resume.")

    def test_most_recent_with_a_request_that_starts_with_a_sop_name(self) -> None:
        text = self.ctrl.resume(request="code_optimization  first, please")
        self.assertEqual(self.ctrl.sop_state.sop_name, "model_optimization")
        self.assertTrue(
            text.endswith("Continuing on: code_optimization  first, please")
        )
        self.assertEqual(self.ctrl.sop_state.suspension_reason, "")
        self.assertEqual(self.ctrl.sop_state.suspended_at, "")

    def test_by_name_pauses_the_active_sop(self) -> None:
        self.ctrl.resume("model_optimization")
        text = self.ctrl.resume("code_optimization")
        self.assertTrue(text.startswith("Resumed SOP 'code_optimization'"))
        self.assertEqual(
            [(s.sop_name, s.suspension_reason) for s in self.ctrl._suspended_sops],
            [("model_optimization", "paused")],
        )

    def test_unknown_name(self) -> None:
        text = self.ctrl.resume("sop_creation")
        self.assertEqual(
            text,
            "No suspended SOP named 'sop_creation'. "
            "In-progress: model_optimization, code_optimization",
        )
        self.assertIsNone(self.ctrl.sop_state)

    def test_injected_reloader_replaces_the_registry(self) -> None:
        reloaded: list[str] = []
        self.ctrl.resume(reload=lambda state: reloaded.append(state.sop_name))
        self.assertEqual(reloaded, ["model_optimization"])

    def test_cmd_resume_sop_reads_a_name_only_when_it_is_suspended(self) -> None:
        text = self.ctrl.cmd_resume_sop("code_optimization go on")
        self.assertEqual(self.ctrl.sop_state.sop_name, "code_optimization")
        self.assertTrue(text.endswith("Continuing on: go on"))
        text = self.ctrl.cmd_resume_sop("go on please")
        self.assertIn("No suspended SOP named 'go on please'", text)


class PauseExitGuardTest(unittest.TestCase):
    def test_pause_without_an_active_sop(self) -> None:
        ctrl = SOPController()
        self.assertEqual(ctrl.cmd_pause_sop(), "No active SOP to pause.")
        self.assertIsNone(ctrl.sop_state)
        self.assertEqual(ctrl._suspended_sops, [])

    def test_exit_without_an_active_sop(self) -> None:
        ctrl = SOPController()
        self.assertEqual(ctrl.cmd_exit_sop(), "No active SOP to exit.")
        self.assertIsNone(ctrl.sop_state)
        self.assertEqual(ctrl._suspended_sops, [])

    def test_pause_and_exit_suspend_with_their_reason(self) -> None:
        ctrl = SOPController()
        ctrl.enter("code_optimization")
        self.assertIn("paused", ctrl.cmd_pause_sop())
        ctrl.enter("model_optimization")
        self.assertIn("Exited SOP 'model_optimization'", ctrl.cmd_exit_sop())
        self.assertEqual(
            [(s.sop_name, s.suspension_reason) for s in ctrl._suspended_sops],
            [("model_optimization", "exited"), ("code_optimization", "paused")],
        )


class ExtraSopDirsTest(_TempDirTestCase):
    def test_reload_attaches_a_definition_from_an_extra_dir(self) -> None:
        ctrl = SOPController(extra_sop_dirs=[self.extra_dir])
        state = SOPState(sop=None, sop_name=_EXTRA_SOP)
        ctrl.reload_sop_definition(state)
        self.assertEqual([p.id for p in state.sop.phases], ["0", "1"])
        self.assertEqual(state.tool_phase_map, state.sop.tool_to_phase_map)

    def test_restore_reattaches_extra_dir_sops(self) -> None:
        ctrl = SOPController(extra_sop_dirs=[self.extra_dir])
        ctrl.enter(_EXTRA_SOP)
        ctrl.cmd_pause_sop()
        ctrl.enter("code_optimization")
        restored = SOPController(extra_sop_dirs=[self.extra_dir])
        restored.restore(ctrl.serialize())
        self.assertEqual(restored.sop_state.sop_name, "code_optimization")
        self.assertEqual(restored._suspended_sops[0].sop_name, _EXTRA_SOP)
        self.assertIsNotNone(restored._suspended_sops[0].sop)

    def test_ci_constructor_seeds_the_controller(self) -> None:
        ci = _make_ci(extra_sop_dirs=[self.extra_dir])
        self.assertEqual(ci.sop_controller.extra_sop_dirs, [self.extra_dir])
        self.assertEqual(ci.extra_sop_dirs, [self.extra_dir])
        self.assertIs(ci._extra_sop_dirs, ci.sop_controller.extra_sop_dirs)

    def test_assigned_dirs_reach_entry_reload_and_catalog(self) -> None:
        ci = _make_ci()
        self.assertIn("not found", ci.sop_controller.enter(_EXTRA_SOP))
        ci.extra_sop_dirs = [self.extra_dir]
        self.assertEqual(ci.sop_controller.extra_sop_dirs, [self.extra_dir])
        self.assertIn(_EXTRA_SOP, ci._filtered_sops())
        ci.sop_controller.enter(_EXTRA_SOP)
        ci.sop_controller.cmd_pause_sop()
        ci._suspended_sops[0].sop = None
        self.assertIn("Resumed SOP 'extra_only'", ci.sop_controller.resume())
        self.assertIsNotNone(ci.sop_state.sop)

    def test_legacy_private_name_forwards_to_the_controller(self) -> None:
        ci = _make_ci()
        ci._extra_sop_dirs = [self.extra_dir]
        self.assertEqual(ci.sop_controller.extra_sop_dirs, [self.extra_dir])
        ci._extra_sop_dirs.append(self.tmp)
        self.assertEqual(ci.extra_sop_dirs, [self.extra_dir, self.tmp])

    def test_public_getter_returns_a_copy(self) -> None:
        ci = _make_ci(extra_sop_dirs=[self.extra_dir])
        ci.extra_sop_dirs.append(self.tmp)
        self.assertEqual(ci.sop_controller.extra_sop_dirs, [self.extra_dir])


class LoadSopTest(_TempDirTestCase):
    def test_later_directories_override_earlier_ones(self) -> None:
        first, second = self.tmp / "first", self.tmp / "second"
        _write_sop(first, "code_optimization")
        _write_sop(second, "code_optimization")
        self.assertEqual(
            load_sop("code_optimization").folder, _SOPS_DIR / "code_optimization"
        )
        self.assertEqual(
            load_sop("code_optimization", extra_dirs=[first]).folder,
            first / "code_optimization",
        )
        self.assertEqual(
            load_sop("code_optimization", extra_dirs=[first, second]).folder,
            second / "code_optimization",
        )
        self.assertEqual(
            load_sop("model_optimization", extra_dirs=[first, second]).folder,
            _SOPS_DIR / "model_optimization",
        )

    def test_base_dir_loads_exactly_that_directory(self) -> None:
        info = load_sop(_EXTRA_SOP, base_dir=self.extra_dir)
        self.assertEqual(info.folder, self.extra_dir / _EXTRA_SOP)
        with self.assertRaises(SOPNotFound):
            load_sop("code_optimization", base_dir=self.extra_dir)

    def test_base_dir_and_extra_dirs_together_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_sop(_EXTRA_SOP, base_dir=self.extra_dir, extra_dirs=[self.tmp])

    def test_missing_sop_raises(self) -> None:
        with self.assertRaises(SOPNotFound):
            load_sop(_EXTRA_SOP)
        with self.assertRaises(SOPNotFound):
            load_sop("no_such_sop", extra_dirs=[self.extra_dir])

"""``sop_feed``: per-turn SOP preparation and the SOP-derived prompt values
shared by the classic and native orchestrators."""

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
    filtered_sops,
    prepare_sop_for_turn,
    resolve_feed,
    SopFeed,
)
from jinja2 import Template

_GATED_SOP_MD = """# Gated

Fixture SOP whose first phase only asks the user.

## Phase 0 -- Ask
[__initial__]

[__requires user input__] Ask the user what to do.

## Phase 1 -- Act
[__depends on__ Phase 0]

Act on {{ session_root_path }}.
"""


def _render(template: str, ctx: dict) -> str:
    return Template(template).render(**ctx)


class _Renderer:
    def __init__(self, sop_file: Path | None) -> None:
        self._sop_file = sop_file

    def find_sop_file(self) -> Path | None:
        return self._sop_file


class _FakeBase:
    effective_cwd = "/workspace/project"
    system_prompt = ""


class _SopDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="af_sop_feed_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        folder = self.tmp / "sops" / "gated"
        folder.mkdir(parents=True)
        self.sop_file = folder / "SOP.md"
        self.sop_file.write_text(_GATED_SOP_MD)
        self.extra_dirs = [self.tmp / "sops"]


class PrepareSopForTurnTest(_SopDirTestCase):
    def test_idle_adopts_the_sop_file_next_to_the_template(self) -> None:
        ctrl = SOPController()
        prepare_sop_for_turn(ctrl, prompt_renderer=_Renderer(self.sop_file))
        self.assertEqual(ctrl.sop_state.sop_name, "Gated")
        self.assertEqual([p.id for p in ctrl.sop_state.sop.phases], ["0", "1"])

    def test_active_sop_is_kept(self) -> None:
        ctrl = SOPController()
        ctrl.enter("code_optimization")
        active = ctrl.sop_state
        prepare_sop_for_turn(ctrl, prompt_renderer=_Renderer(self.sop_file))
        self.assertIs(ctrl.sop_state, active)

    def test_no_renderer_or_no_file_leaves_idle(self) -> None:
        ctrl = SOPController()
        prepare_sop_for_turn(ctrl)
        prepare_sop_for_turn(ctrl, prompt_renderer=_Renderer(None))
        self.assertIsNone(ctrl.sop_state)

    def test_satisfied_gate_completes_a_no_tools_input_phase(self) -> None:
        ctrl = SOPController(extra_sop_dirs=self.extra_dirs)
        ctrl.enter("gated")
        ctrl.sop_state.user_input_gate_passed = True
        prepare_sop_for_turn(ctrl)
        self.assertEqual(ctrl.sop_state.completed_phase_ids(), ["0"])
        self.assertFalse(ctrl.sop_state.user_input_gate_passed)


class BuildSopFeedTest(_SopDirTestCase):
    def test_idle_lists_the_filtered_catalog(self) -> None:
        ctrl = SOPController()
        feed = build_sop_feed(
            ctrl,
            {},
            extra_sop_dirs=self.extra_dirs,
            allowed=["gated", "code_optimization", "model_optimization"],
            disallowed=["model_optimization"],
        )
        self.assertFalse(feed.sop_active)
        self.assertEqual(feed.sop_nextstep_guidance, "")
        self.assertIn("(`gated`)", feed.available_sops)
        self.assertIn("(`code_optimization`)", feed.available_sops)
        self.assertNotIn("model_optimization", feed.available_sops)
        self.assertNotIn("sop_creation", feed.available_sops)

    def test_active_sop_guidance_and_catalog_modes(self) -> None:
        ctrl = SOPController(extra_sop_dirs=self.extra_dirs)
        ctrl.enter("code_optimization")
        ctrl.cmd_pause_sop()
        ctrl.enter("gated")
        when_idle = build_sop_feed(ctrl, {}, extra_sop_dirs=self.extra_dirs)
        self.assertTrue(when_idle.sop_active)
        self.assertIn("Ask the user what to do.", when_idle.sop_nextstep_guidance)
        self.assertEqual(when_idle.available_sops, "")
        self.assertTrue(when_idle.paused_sop.startswith("code_optimization ("))
        self.assertEqual(when_idle.inprogress_sops, "")
        always = build_sop_feed(
            ctrl, {}, catalog_mode="always", extra_sop_dirs=self.extra_dirs
        )
        self.assertIn("(`gated`)", always.available_sops)

    def test_unknown_catalog_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_sop_feed(SOPController(), {}, catalog_mode="never")

    def test_template_values(self) -> None:
        feed = SopFeed(sop=object(), available_sops="catalog")
        self.assertEqual(
            feed.template_values(),
            {
                "sop_nextstep_guidance": "",
                "available_sops": "catalog",
                "paused_sop": "",
                "inprogress_sops": "",
                "sop_active": True,
            },
        )

    def test_filtered_sops_without_filters_lists_everything(self) -> None:
        names = set(filtered_sops(extra_sop_dirs=self.extra_dirs))
        self.assertTrue(
            {"gated", "code_optimization", "model_optimization"} <= names, names
        )

    def test_ci_delegators_match_the_functional_api(self) -> None:
        ci = ConversationalInferencer(
            base_inferencer=_FakeBase(),
            extra_sop_dirs=self.extra_dirs,
            allowed_sops=["gated", "code_optimization"],
        )
        ci.sop_controller.enter("code_optimization")
        feed = build_sop_feed(
            ci.sop_controller,
            ci.prior_context,
            catalog_mode="always",
            extra_sop_dirs=self.extra_dirs,
            allowed=["gated", "code_optimization"],
        )
        self.assertEqual(
            ci._build_sop_feed(catalog_mode="always"),
            {"sop": feed.sop, **feed.template_values()},
        )
        self.assertEqual(set(ci._filtered_sops()), {"gated", "code_optimization"})


class ResolveFeedTest(unittest.TestCase):
    def test_values_referencing_other_values_are_rendered(self) -> None:
        feed = {
            "guidance": "Act on {{ session_root_path }}.",
            "session_root_path": "/r",
        }
        self.assertEqual(resolve_feed(feed, _render)["guidance"], "Act on /r.")

    def test_a_cycle_leaves_the_feed_unresolved(self) -> None:
        feed = {"a": "{{ b }}", "b": "{{ a }}"}
        with self.assertLogs(
            "agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_feed",
            level="WARNING",
        ):
            self.assertIs(resolve_feed(feed, _render), feed)

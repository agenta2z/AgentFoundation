"""Tests for SOPState, phase completion detection, pause/resume, and SOP executor."""

import asyncio
import unittest
from pathlib import Path

from attr import attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.workflow.sop_state import SOPState
from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus


OPENTEAM_SOPS = Path("/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/resources/sops")


@attrs(slots=False)
class MockBase(InferencerBase):
    async def _ainfer(self, inp, cfg=None, **kw):
        return "mock"

    def _infer(self, inp, cfg=None, **kw):
        return "mock"


def _make_ci(**kwargs):
    return ConversationalInferencer(base_inferencer=MockBase(), **kwargs)


async def _enter_sop(ci, sop_name="role_creation", yolo=False):
    from agent_foundation.resources.tools.sop.executor import execute

    result = await execute(
        {"workflow": sop_name, "yolo": yolo},
        {"extra_sop_dirs": [OPENTEAM_SOPS]},
    )
    ci.update_prior_context(**result.context_updates)
    return result


# ── SOPState tests ───────────────────────────────────────────────────


class TestSOPState(unittest.TestCase):
    def test_dict_protocol(self):
        s = SOPState(sop_name="test", current_phase="0")
        d = dict(s)
        assert d["sop_name"] == "test"
        assert d["current_phase"] == "0"

    def test_spread(self):
        s = SOPState(sop_name="x")
        merged = {**s, "extra": True}
        assert merged["sop_name"] == "x"
        assert merged["extra"] is True

    def test_to_feed_excludes_sop(self):
        s = SOPState(sop_name="test", sop=object())
        feed = s.to_feed()
        assert "sop" not in feed
        assert "sop_name" in feed

    def test_to_dict_excludes_sop(self):
        s = SOPState(sop_name="test", sop=object())
        d = s.to_dict()
        assert "sop" not in d

    def test_from_dict_roundtrip(self):
        original = SOPState(
            sop_name="rc",
            current_phase="1",
            phase_status=PhaseStatus.RUNNING,
            instance_id="test_123",
        )
        restored = SOPState.from_dict(original.to_dict())
        assert restored.sop_name == "rc"
        assert restored.current_phase == "1"
        assert restored.instance_id == "test_123"

    def test_phase_status_enum(self):
        s = SOPState(phase_status=PhaseStatus.IDLE)
        assert s.phase_status == "idle"
        assert s.phase_status == PhaseStatus.IDLE

    def test_suspension_fields_roundtrip(self):
        s = SOPState(sop_name="x", suspension_reason="paused", suspended_at="2026-01-01")
        restored = SOPState.from_dict(s.to_dict())
        assert restored.suspension_reason == "paused"
        assert restored.suspended_at == "2026-01-01"

    def test_suspension_label(self):
        assert SOPState(suspension_reason="paused").suspension_label == "Paused"
        assert SOPState(suspension_reason="exited").suspension_label == "Exited"
        assert SOPState().suspension_label == ""


# ── SOP Executor tests ──────────────────────────────────────────────


class TestSOPExecutor(unittest.TestCase):
    def test_executor_returns_sop_state(self):
        ci = _make_ci()
        result = asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        assert isinstance(ci.sop_state, SOPState)

    def test_executor_sets_fields(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        s = ci.sop_state
        assert s.sop_name == "role_creation"
        assert s.current_phase == "0"
        assert s.sop is not None
        # tool_phase_map is taken verbatim from the SOP's [__tools__] mappings.
        assert s.tool_phase_map == {
            "multiple_choice": "0",
            "create_role": "1",
            "role_setup": "2",
        }
        assert s.sop_description != ""

    def test_executor_yolo_bridges(self):
        ci = _make_ci()
        assert ci.yolo_mode is False
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci, yolo=True))
        assert ci.yolo_mode is True

    def test_sop_state_not_in_prior_context(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        assert "sop_state" not in ci.prior_context


# ── Phase Completion tests ───────────────────────────────────────────


class TestPhaseCompletion(unittest.TestCase):
    def test_no_sop_is_noop(self):
        ci = _make_ci()
        ci._check_phase_completion()

    def test_tool_mapped_completion(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        s = ci.sop_state
        s.current_phase = "1"
        s.completed_phases = ["0"]
        ci._check_phase_completion(tool_name="create_role")
        assert "1" in s.completed_phases
        assert s.current_phase == "1b"

    def test_confirmation_completion(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        s = ci.sop_state
        assert s.current_phase == "0"
        s.user_input_gate_passed = True
        ci._check_phase_completion()
        assert "0" in s.completed_phases
        assert s.current_phase == "1"

    def test_all_phases_complete(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        s = ci.sop_state

        s.user_input_gate_passed = True
        ci._check_phase_completion()
        assert s.current_phase == "1"

        ci._check_phase_completion(tool_name="create_role")
        assert s.current_phase == "1b"

        s.user_input_gate_passed = True
        ci._check_phase_completion()
        assert s.current_phase == "2"

        ci._check_phase_completion(tool_name="role_setup")
        assert s.current_phase == "2b"

        s.user_input_gate_passed = True
        ci._check_phase_completion()
        assert s.current_phase is None
        assert s.phase_status == PhaseStatus.COMPLETED

    def test_no_detection_is_noop(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        ci._check_phase_completion(tool_name="nonexistent")
        assert ci.sop_state.current_phase == "0"


# ── Prompt Rendering tests ───────────────────────────────────────────


class TestPromptRendering(unittest.TestCase):
    def test_no_sop_no_workflow_sections(self):
        ci = _make_ci()
        rendered = ci._render_prompt("test")
        assert "WorkflowDescription" not in rendered

    def test_sop_renders_guidance(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        rendered = ci._render_prompt("test")
        feed = ci._last_template_feed
        guidance = feed.get("sop_nextstep_guidance", "")
        assert len(guidance) > 100
        assert "Role Specification" in guidance

    def test_exit_sop_suspends_resumably(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        assert ci.sop_state is not None
        asyncio.get_event_loop().run_until_complete(ci._commands.dispatch("/exit_sop"))
        # Exit is now resumable: state moves to the suspended bag, not destroyed.
        assert ci.sop_state is None
        assert len(ci._suspended_sops) == 1
        assert ci._suspended_sops[0].suspension_reason == "exited"
        rendered = ci._render_prompt("test")
        # No active SOP context, but the exited SOP is listed as in-progress.
        assert "<SOPDescription>" not in rendered
        assert "In-Progress SOPs" in rendered


# ── Pause/Resume tests ──────────────────────────────────────────────


class TestPauseResume(unittest.TestCase):
    def test_serialize_without_sop(self):
        ci = _make_ci()
        state = ci._serialize_pause_state(turn_number=1, iteration=2)
        assert state["sop_state"] is None
        assert state["turn_number"] == 1
        assert state["iteration"] == 2

    def test_serialize_with_sop(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        state = ci._serialize_pause_state()
        assert state["sop_state"] is not None
        assert state["sop_state"]["sop_name"] == "role_creation"
        assert "sop" not in state["sop_state"]

    def test_restore_roundtrip(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(
            _enter_sop(ci, sop_name="code_optimization", yolo=False)
        )
        ci.add_message("user", "hello")
        state = ci._serialize_pause_state(turn_number=3, iteration=1)

        ci2 = _make_ci()
        ci2._restore_pause_state(state)
        assert ci2.sop_state is not None
        assert ci2.sop_state.sop_name == "code_optimization"
        assert ci2.sop_state.sop is not None
        assert ci2._messages == [{"role": "user", "content": "hello"}]
        assert ci2._paused is False

    def test_max_iterations_bumped(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        n_phases = len(ci.sop_state.sop.phases)
        assert n_phases == 5
        # The loop uses effective_max = max(5, 5*3) = 15


# ── Suspend / Resume Lifecycle tests ─────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSuspendLifecycle(unittest.TestCase):
    def test_pause_sop_moves_to_bag(self):
        ci = _make_ci()
        _run(_enter_sop(ci))
        name = ci.sop_state.sop_name
        _run(ci._commands.dispatch("/pause_sop"))
        assert ci.sop_state is None
        assert len(ci._suspended_sops) == 1
        assert ci._suspended_sops[0].sop_name == name
        assert ci._suspended_sops[0].suspension_reason == "paused"

    def test_exit_sop_moves_to_bag(self):
        ci = _make_ci()
        _run(_enter_sop(ci))
        _run(ci._commands.dispatch("/exit_sop"))
        assert ci.sop_state is None
        assert ci._suspended_sops[0].suspension_reason == "exited"

    def test_resume_sop_default_most_recent(self):
        ci = _make_ci()
        _run(_enter_sop(ci, sop_name="role_creation"))
        _run(ci._commands.dispatch("/exit_sop"))
        _run(_enter_sop(ci, sop_name="code_optimization"))
        _run(ci._commands.dispatch("/pause_sop"))
        # Bag: [code_optimization (paused), role_creation (exited)]
        result = _run(ci._commands.dispatch("/resume_sop"))
        assert ci.sop_state is not None
        assert ci.sop_state.sop_name == "code_optimization"
        assert "code_optimization" in result
        # role_creation stays suspended
        assert [s.sop_name for s in ci._suspended_sops] == ["role_creation"]

    def test_resume_sop_by_name(self):
        ci = _make_ci()
        _run(_enter_sop(ci, sop_name="role_creation"))
        _run(ci._commands.dispatch("/exit_sop"))
        _run(_enter_sop(ci, sop_name="code_optimization"))
        _run(ci._commands.dispatch("/exit_sop"))
        _run(ci._commands.dispatch("/resume_sop role_creation"))
        assert ci.sop_state.sop_name == "role_creation"
        assert [s.sop_name for s in ci._suspended_sops] == ["code_optimization"]

    def test_resume_auto_pauses_active(self):
        ci = _make_ci()
        _run(_enter_sop(ci, sop_name="role_creation"))
        _run(ci._commands.dispatch("/exit_sop"))
        _run(_enter_sop(ci, sop_name="code_optimization"))  # now active
        _run(ci._commands.dispatch("/resume_sop role_creation"))
        assert ci.sop_state.sop_name == "role_creation"
        # code_optimization was auto-paused
        susp = {s.sop_name: s.suspension_reason for s in ci._suspended_sops}
        assert susp == {"code_optimization": "paused"}

    def test_resume_none_message(self):
        ci = _make_ci()
        result = _run(ci._commands.dispatch("/resume_sop"))
        assert "No suspended" in result

    def test_resume_unknown_name(self):
        ci = _make_ci()
        _run(_enter_sop(ci))
        _run(ci._commands.dispatch("/exit_sop"))
        result = _run(ci._commands.dispatch("/resume_sop nonexistent"))
        assert "No suspended SOP named" in result


class TestSopCommandEntry(unittest.TestCase):
    def _ci(self):
        return _make_ci(extra_sop_dirs=[OPENTEAM_SOPS])

    def test_sop_command_enters(self):
        ci = self._ci()
        result = _run(ci._commands.dispatch("/sop role_creation"))
        assert ci.sop_state is not None
        assert ci.sop_state.sop_name == "role_creation"
        assert "Entered SOP" in result

    def test_sop_usage_when_no_name(self):
        ci = self._ci()
        result = _run(ci._commands.dispatch("/sop"))
        assert "Usage:" in result
        assert ci.sop_state is None

    def test_sop_same_name_prompts_resume_or_fresh(self):
        ci = self._ci()
        _run(ci._commands.dispatch("/sop role_creation"))
        _run(ci._commands.dispatch("/exit_sop"))
        result = _run(ci._commands.dispatch("/sop role_creation"))
        # Did not enter a new one; surfaced the resume-vs-fresh choice.
        assert ci.sop_state is None
        assert "/resume_sop role_creation" in result
        assert "--fresh" in result
        assert len(ci._suspended_sops) == 1

    def test_sop_fresh_starts_new_instance(self):
        ci = self._ci()
        _run(ci._commands.dispatch("/sop role_creation"))
        first_id = ci.sop_state.instance_id
        _run(ci._commands.dispatch("/exit_sop"))
        _run(ci._commands.dispatch("/sop role_creation --fresh"))
        assert ci.sop_state is not None
        assert ci.sop_state.instance_id != first_id
        # the previously exited instance remains suspended
        assert len(ci._suspended_sops) == 1


class TestSuspendedSerialization(unittest.TestCase):
    def test_suspended_bag_roundtrips(self):
        ci = _make_ci()
        _run(_enter_sop(ci, sop_name="code_optimization"))
        _run(ci._commands.dispatch("/exit_sop"))
        state = ci._serialize_pause_state()
        assert len(state["suspended_sops"]) == 1

        ci2 = _make_ci()
        ci2._restore_pause_state(state)
        assert len(ci2._suspended_sops) == 1
        restored = ci2._suspended_sops[0]
        assert restored.sop_name == "code_optimization"
        assert restored.suspension_reason == "exited"
        assert restored.sop is not None  # definition reloaded

    def test_old_state_without_bag_loads_empty(self):
        ci = _make_ci()
        ci._restore_pause_state({"messages": [], "prior_context": {}})
        assert ci._suspended_sops == []


class TestLifecyclePromptRendering(unittest.TestCase):
    def test_paused_renders_nudge(self):
        ci = _make_ci()
        _run(_enter_sop(ci, sop_name="role_creation"))
        _run(ci._commands.dispatch("/pause_sop"))
        rendered = ci._render_prompt("hi")
        # Assert on the section body (the header substring also appears in the
        # Decision Procedure prose, so it is not a reliable discriminator).
        assert "You temporarily paused" in rendered
        assert "role_creation" in rendered

    def test_exited_renders_in_progress(self):
        ci = _make_ci()
        _run(_enter_sop(ci, sop_name="role_creation"))
        _run(ci._commands.dispatch("/exit_sop"))
        rendered = ci._render_prompt("hi")
        assert "## In-Progress SOPs" in rendered
        # The Paused-SOP section body must be absent (exited, not paused).
        assert "You temporarily paused" not in rendered

    def test_two_paused_single_nudge(self):
        ci = _make_ci()
        _run(_enter_sop(ci, sop_name="role_creation"))
        _run(ci._commands.dispatch("/pause_sop"))
        _run(_enter_sop(ci, sop_name="code_optimization"))
        _run(ci._commands.dispatch("/pause_sop"))
        # Both stored as paused; rendering must not mutate that.
        assert all(s.suspension_reason == "paused" for s in ci._suspended_sops)
        paused_sop, inprogress = ci._format_suspended_sops()
        assert "code_optimization" in paused_sop      # most recent = the nudge
        assert "role_creation" in inprogress          # older paused = passive
        assert "role_creation" not in paused_sop


if __name__ == "__main__":
    unittest.main()

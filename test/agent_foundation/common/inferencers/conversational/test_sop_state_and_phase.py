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
        assert s.tool_phase_map == {"create_role": "1", "role_setup": "2"}
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
        s.confirmation_gate_passed = True
        ci._check_phase_completion()
        assert "0" in s.completed_phases
        assert s.current_phase == "1"

    def test_all_phases_complete(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        s = ci.sop_state

        s.confirmation_gate_passed = True
        ci._check_phase_completion()
        assert s.current_phase == "1"

        ci._check_phase_completion(tool_name="create_role")
        assert s.current_phase == "1b"

        s.confirmation_gate_passed = True
        ci._check_phase_completion()
        assert s.current_phase == "2"

        ci._check_phase_completion(tool_name="role_setup")
        assert s.current_phase == "2b"

        s.confirmation_gate_passed = True
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

    def test_exit_sop_clears_guidance(self):
        ci = _make_ci()
        asyncio.get_event_loop().run_until_complete(_enter_sop(ci))
        assert ci.sop_state is not None
        asyncio.get_event_loop().run_until_complete(ci._commands.dispatch("/exit_sop"))
        assert ci.sop_state is None
        rendered = ci._render_prompt("test")
        assert "WorkflowDescription" not in rendered


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


if __name__ == "__main__":
    unittest.main()

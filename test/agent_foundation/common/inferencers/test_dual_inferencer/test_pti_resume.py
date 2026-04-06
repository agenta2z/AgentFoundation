

"""Automated tests for PlanThenImplementInferencer resume + Workflow checkpoint.

Tests file-based resume detection (backward compat) and native Workflow
checkpoint/resume for the PTI → DualInferencer recursive hierarchy.
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    DualInferencerResponse,
    ReflectionStyles,
    ResponseSelectors,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
    _PHASE_TO_STEP_INDEX,
)


def _make_mock_inferencer(response_text="mock response", id_="mock"):
    """Create a minimal mock InferencerBase with ainfer, aconnect, adisconnect."""
    mock = MagicMock()
    mock.id = id_
    mock.ainfer = AsyncMock(
        return_value=DualInferencerResponse(
            base_response=response_text,
            reflection_response=None,
            reflection_style=ReflectionStyles.NoReflection,
            response_selector=ResponseSelectors.BaseResponse,
            consensus_achieved=True,
            consensus_history=[],
            total_iterations=1,
        )
    )
    mock.aconnect = AsyncMock()
    mock.adisconnect = AsyncMock()
    mock.set_parent_debuggable = MagicMock()
    return mock


def _normpath(p: str) -> str:
    """Normalize path separators for cross-platform comparison."""
    return p.replace("\\", "/")


class TestPTIWorkspaceHelpers(unittest.TestCase):
    """Tests for PTI workspace helpers (iteration workspace paths, etc.)."""

    def test_get_iteration_workspace_iter1(self):
        ws = PlanThenImplementInferencer._get_iteration_workspace("/base", 1)
        self.assertEqual(_normpath(ws), "/base")

    def test_get_iteration_workspace_iter2(self):
        ws = PlanThenImplementInferencer._get_iteration_workspace("/base", 2)
        self.assertEqual(_normpath(ws), "/base/followup_iterations/iteration_2")

    def test_get_iteration_workspace_iter5(self):
        ws = PlanThenImplementInferencer._get_iteration_workspace("/base", 5)
        self.assertEqual(_normpath(ws), "/base/followup_iterations/iteration_5")


class TestPTIResumeDetection(unittest.TestCase):
    """Tests for PTI file-based resume detection (backward compat)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.planner = _make_mock_inferencer("plan text", "planner")
        self.executor = _make_mock_inferencer("impl text", "executor")
        self.analyzer = _make_mock_inferencer(
            '```json\n{"should_continue": false}\n```', "analyzer"
        )

    def _make_pti(self, **kwargs):
        defaults = dict(
            planner_inferencer=self.planner,
            executor_inferencer=self.executor,
            analyzer_inferencer=self.analyzer,
            workspace_path=self.tmpdir,
            resume_workspace=self.tmpdir,
            enable_analysis=True,
            analysis_mode="last_round_only",
        )
        defaults.update(kwargs)
        return PlanThenImplementInferencer(**defaults)

    def test_detect_fresh_workspace(self):
        """Empty workspace → planning phase."""
        os.makedirs(os.path.join(self.tmpdir, "outputs"), exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")

        pti = self._make_pti()
        iteration, phase, state, current_input, original = pti._detect_resume_point(
            self.tmpdir
        )
        self.assertEqual(iteration, 1)
        self.assertEqual(phase, "planning")

    def test_detect_plan_done_no_impl(self):
        """Plan file exists, no implementation → implementation phase."""
        outputs = os.path.join(self.tmpdir, "outputs")
        os.makedirs(outputs, exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")
        with open(os.path.join(outputs, "round0_plan.md"), "w") as f:
            f.write("# Plan\nDo stuff")
        # Write plan completion marker (Tier 2)
        with open(os.path.join(outputs, ".plan_completed"), "w") as f:
            f.write("{}")

        pti = self._make_pti()
        iteration, phase, state, current_input, original = pti._detect_resume_point(
            self.tmpdir
        )
        self.assertEqual(iteration, 1)
        self.assertEqual(phase, "implementation")
        self.assertTrue(state.plan_done)
        self.assertFalse(state.impl_done)

    def test_detect_impl_done_no_analysis(self):
        """Plan + impl done, no analysis → analysis phase."""
        outputs = os.path.join(self.tmpdir, "outputs")
        os.makedirs(outputs, exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, "results"), exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")
        with open(os.path.join(outputs, "round0_plan.md"), "w") as f:
            f.write("plan")
        with open(os.path.join(outputs, "round0_implementation.md"), "w") as f:
            f.write("impl")
        # Write completion markers (Tier 2)
        with open(os.path.join(outputs, ".plan_completed"), "w") as f:
            f.write("{}")
        with open(os.path.join(outputs, ".impl_completed"), "w") as f:
            f.write("{}")

        pti = self._make_pti()
        iteration, phase, state, current_input, original = pti._detect_resume_point(
            self.tmpdir
        )
        self.assertEqual(iteration, 1)
        self.assertEqual(phase, "analysis")

    def test_detect_complete(self):
        """Plan + impl + analysis done, should_continue=False → complete."""
        outputs = os.path.join(self.tmpdir, "outputs")
        results = os.path.join(self.tmpdir, "results")
        os.makedirs(outputs, exist_ok=True)
        os.makedirs(results, exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")
        with open(os.path.join(outputs, "round0_plan.md"), "w") as f:
            f.write("plan")
        with open(os.path.join(outputs, "round0_implementation.md"), "w") as f:
            f.write("impl")
        # Write completion markers (Tier 2)
        with open(os.path.join(outputs, ".plan_completed"), "w") as f:
            f.write("{}")
        with open(os.path.join(outputs, ".impl_completed"), "w") as f:
            f.write("{}")
        with open(os.path.join(results, "analysis_summary.json"), "w") as f:
            json.dump({"should_continue": False, "summary": "done"}, f)

        pti = self._make_pti()
        iteration, phase, state, current_input, original = pti._detect_resume_point(
            self.tmpdir
        )
        self.assertEqual(phase, "complete")

    def test_detect_new_iteration(self):
        """should_continue=True → new_iteration."""
        outputs = os.path.join(self.tmpdir, "outputs")
        results = os.path.join(self.tmpdir, "results")
        os.makedirs(outputs, exist_ok=True)
        os.makedirs(results, exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")
        with open(os.path.join(outputs, "round0_plan.md"), "w") as f:
            f.write("plan")
        with open(os.path.join(outputs, "round0_implementation.md"), "w") as f:
            f.write("impl")
        # Write completion markers (Tier 2)
        with open(os.path.join(outputs, ".plan_completed"), "w") as f:
            f.write("{}")
        with open(os.path.join(outputs, ".impl_completed"), "w") as f:
            f.write("{}")
        with open(os.path.join(results, "analysis_summary.json"), "w") as f:
            json.dump(
                {
                    "should_continue": True,
                    "summary": "needs work",
                    "next_iteration_request": "fix bugs",
                },
                f,
            )

        pti = self._make_pti(enable_multiple_iterations=True)
        iteration, phase, state, current_input, original = pti._detect_resume_point(
            self.tmpdir
        )
        self.assertEqual(phase, "new_iteration")
        self.assertEqual(iteration, 2)
        self.assertIn("fix bugs", current_input)


class TestPTISynthesizeCheckpoint(unittest.TestCase):
    """Tests for backward-compat checkpoint synthesis from workspace state."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.planner = _make_mock_inferencer("plan text", "planner")
        self.executor = _make_mock_inferencer("impl text", "executor")
        self.analyzer = _make_mock_inferencer("analysis", "analyzer")

    def _make_pti(self, **kwargs):
        defaults = dict(
            planner_inferencer=self.planner,
            executor_inferencer=self.executor,
            analyzer_inferencer=self.analyzer,
            workspace_path=self.tmpdir,
            resume_workspace=self.tmpdir,
            enable_analysis=True,
            analysis_mode="last_round_only",
        )
        defaults.update(kwargs)
        return PlanThenImplementInferencer(**defaults)

    def test_synthesize_planning_phase(self):
        """Fresh workspace → synthesized checkpoint at step 0 (plan)."""
        os.makedirs(os.path.join(self.tmpdir, "outputs"), exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")

        pti = self._make_pti()
        pti._current_base_workspace = self.tmpdir
        pti._current_iteration_workspace = self.tmpdir
        ckpt = pti._synthesize_checkpoint_from_workspace()

        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt["next_step_index"], 0)  # plan step
        self.assertEqual(ckpt["state"]["iteration"], 1)

    def test_synthesize_implementation_phase(self):
        """Plan done → synthesized checkpoint at step 2 (implement)."""
        outputs = os.path.join(self.tmpdir, "outputs")
        os.makedirs(outputs, exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")
        with open(os.path.join(outputs, "round0_plan.md"), "w") as f:
            f.write("plan content")
        # Write plan completion marker (Tier 2)
        with open(os.path.join(outputs, ".plan_completed"), "w") as f:
            f.write("{}")

        pti = self._make_pti()
        pti._current_base_workspace = self.tmpdir
        pti._current_iteration_workspace = self.tmpdir
        ckpt = pti._synthesize_checkpoint_from_workspace()

        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt["next_step_index"], 2)  # implement step
        self.assertEqual(ckpt["state"]["plan_output_text"], "plan content")
        self.assertTrue(ckpt["state"]["plan_approved"])

    def test_synthesize_creates_sentinel_file(self):
        """Synthesized checkpoint creates a sentinel result file for _arun validation."""
        os.makedirs(os.path.join(self.tmpdir, "outputs"), exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")

        pti = self._make_pti()
        pti._current_base_workspace = self.tmpdir
        pti._current_iteration_workspace = self.tmpdir
        ckpt = pti._synthesize_checkpoint_from_workspace()

        self.assertIsNotNone(ckpt)
        # result_id must NOT be None — it must point to a real sentinel file
        sentinel_id = ckpt["result_id"]
        self.assertIsNotNone(sentinel_id)
        self.assertEqual(sentinel_id, "__synth_sentinel__")

        # The sentinel file must actually exist on disk
        sentinel_path = pti._resolve_result_path(sentinel_id)
        self.assertTrue(
            os.path.exists(sentinel_path),
            f"Sentinel file not found at {sentinel_path}",
        )

    def test_synthesize_complete_returns_none(self):
        """Complete workspace → None (no checkpoint needed)."""
        outputs = os.path.join(self.tmpdir, "outputs")
        results = os.path.join(self.tmpdir, "results")
        os.makedirs(outputs, exist_ok=True)
        os.makedirs(results, exist_ok=True)
        with open(os.path.join(self.tmpdir, "request.txt"), "w") as f:
            f.write("test request")
        with open(os.path.join(outputs, "round0_plan.md"), "w") as f:
            f.write("plan")
        with open(os.path.join(outputs, "round0_implementation.md"), "w") as f:
            f.write("impl")
        # Write completion markers (Tier 2)
        with open(os.path.join(outputs, ".plan_completed"), "w") as f:
            f.write("{}")
        with open(os.path.join(outputs, ".impl_completed"), "w") as f:
            f.write("{}")
        with open(os.path.join(results, "analysis_summary.json"), "w") as f:
            json.dump({"should_continue": False, "summary": "done"}, f)

        pti = self._make_pti()
        pti._current_base_workspace = self.tmpdir
        pti._current_iteration_workspace = self.tmpdir
        ckpt = pti._synthesize_checkpoint_from_workspace()

        self.assertIsNone(ckpt)


class TestPhaseToStepIndex(unittest.TestCase):
    """Test the phase-to-step-index mapping."""

    def test_planning_maps_to_0(self):
        self.assertEqual(_PHASE_TO_STEP_INDEX["planning"], 0)

    def test_implementation_maps_to_2(self):
        self.assertEqual(_PHASE_TO_STEP_INDEX["implementation"], 2)

    def test_analysis_maps_to_3(self):
        self.assertEqual(_PHASE_TO_STEP_INDEX["analysis"], 3)

    def test_new_iteration_maps_to_0(self):
        self.assertEqual(_PHASE_TO_STEP_INDEX["new_iteration"], 0)


class TestPTIBuildResponse(unittest.TestCase):
    """Test _build_response_from_state."""

    def setUp(self):
        self.planner = _make_mock_inferencer("plan", "planner")
        self.executor = _make_mock_inferencer("impl", "executor")

    def test_basic_response(self):
        pti = PlanThenImplementInferencer(
            planner_inferencer=self.planner,
            executor_inferencer=self.executor,
        )
        state = {
            "iteration": 1,
            "current_input": "req",
            "original_request": "req",
            "plan_output_text": "the plan",
            "plan_file_path": "/tmp/plan.md",
            "plan_approved": True,
            "executor_output_text": "the implementation",
            "should_continue": False,
            "iteration_records": [],
        }
        resp = pti._build_response_from_state(state)
        self.assertIsInstance(resp, PlanThenImplementResponse)
        self.assertEqual(resp.plan_output, "the plan")
        self.assertEqual(str(resp.base_response), "the implementation")
        self.assertTrue(resp.plan_approved)

    def test_none_state(self):
        pti = PlanThenImplementInferencer(
            planner_inferencer=self.planner,
            executor_inferencer=self.executor,
        )
        resp = pti._build_response_from_state(None)
        self.assertIsInstance(resp, PlanThenImplementResponse)
        self.assertEqual(resp.plan_output, "")


class TestPTIExtractResponseText(unittest.TestCase):
    """Test _extract_response_text static method."""

    def test_dual_inferencer_response(self):
        resp = DualInferencerResponse(
            base_response="plan text",
            reflection_response=None,
            reflection_style=ReflectionStyles.NoReflection,
            response_selector=ResponseSelectors.BaseResponse,
            consensus_achieved=True,
            consensus_history=[],
            total_iterations=1,
        )
        self.assertEqual(
            PlanThenImplementInferencer._extract_response_text(resp), "plan text"
        )

    def test_string(self):
        self.assertEqual(
            PlanThenImplementInferencer._extract_response_text("hello"), "hello"
        )

    def test_none(self):
        self.assertEqual(
            PlanThenImplementInferencer._extract_response_text(None), "None"
        )


class TestPTIBackwardScanDisabled(unittest.TestCase):
    """Verify _backward_scan_resume is disabled for PTI."""

    def test_returns_negative_one(self):
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        result = pti._backward_scan_resume(True)
        self.assertEqual(result, (-1, None))


class TestPTIBlockDirectRunArun(unittest.TestCase):
    """Verify run() and arun() are blocked."""

    def test_run_raises(self):
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        with self.assertRaises(NotImplementedError):
            pti.run()

    def test_arun_raises(self):
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        with self.assertRaises(NotImplementedError):
            asyncio.get_event_loop().run_until_complete(pti.arun())


class TestPTIGetResultPath(unittest.TestCase):
    """Test _get_result_path returns correct checkpoint paths using STABLE base workspace."""

    def test_with_base_workspace(self):
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        pti._current_base_workspace = "/tmp/workspace"
        path = pti._get_result_path("plan")
        self.assertEqual(
            _normpath(path), "/tmp/workspace/checkpoints/pti/step_plan.json"
        )

    def test_without_base_workspace(self):
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        pti._current_base_workspace = None
        path = pti._get_result_path("plan")
        self.assertEqual(path, "")

    def test_stable_across_iteration_changes(self):
        """_get_result_path uses _current_base_workspace, NOT _current_iteration_workspace.

        This is the core regression test for Issue #1: checkpoint path must
        remain stable even when closures change _current_iteration_workspace.
        """
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        pti._current_base_workspace = "/tmp/base"

        # Simulate iteration 1
        pti._current_iteration_workspace = "/tmp/base"
        path1 = pti._get_result_path("__wf_checkpoint__")

        # Simulate iteration 2 (closure changes _current_iteration_workspace)
        pti._current_iteration_workspace = "/tmp/base/followup_iterations/iteration_2"
        path2 = pti._get_result_path("__wf_checkpoint__")

        # Both should resolve to the SAME path under _current_base_workspace
        self.assertEqual(path1, path2)
        self.assertEqual(
            _normpath(path1),
            "/tmp/base/checkpoints/pti/step___wf_checkpoint__.json",
        )


class TestPTIParseAnalysis(unittest.TestCase):
    """Test _parse_analysis_response."""

    def test_parse_json_block(self):
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        text = '```json\n{"should_continue": true, "next_iteration_request": "fix bugs"}\n```'
        should_continue, next_req = pti._parse_analysis_response(text)
        self.assertTrue(should_continue)
        self.assertEqual(next_req, "fix bugs")

    def test_parse_plain_json(self):
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        text = '{"should_continue": false, "next_iteration_request": ""}'
        should_continue, next_req = pti._parse_analysis_response(text)
        self.assertFalse(should_continue)

    def test_parse_invalid(self):
        planner = _make_mock_inferencer("plan", "planner")
        executor = _make_mock_inferencer("impl", "executor")
        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )
        text = "This is not JSON at all."
        should_continue, next_req = pti._parse_analysis_response(text)
        self.assertFalse(should_continue)


if __name__ == "__main__":
    unittest.main()

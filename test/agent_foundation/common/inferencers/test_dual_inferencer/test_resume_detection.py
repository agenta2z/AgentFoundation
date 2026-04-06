

"""Mock unit tests for PlanThenImplementInferencer resume detection,
analysis-only mode, and backward compatibility.

Uses tempfile workspaces to test _detect_resume_point, _detect_workspace_state,
and full _ainfer flow with mock inferencers.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)


@attrs
class MockInferencer(InferencerBase):
    """Minimal mock inferencer for unit testing."""

    _response: str = attrib(default="mock response", alias="_response")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._response


def _create_file(path: str, content: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _create_completion_marker(ws: str, phase: str) -> None:
    """Write a Tier 2 step-completion marker file.

    Creates ``<ws>/outputs/.<phase>_completed`` to signal that the step
    finished successfully.
    """
    marker_path = os.path.join(ws, "outputs", f".{phase}_completed")
    _create_file(
        marker_path,
        json.dumps({"completed_at": "2026-01-01T00:00:00+00:00", "step": phase}),
    )


def _make_pti_for_resume(ws: str, **kwargs) -> PlanThenImplementInferencer:
    """Create a minimal PTI suitable for testing resume detection."""
    defaults = dict(
        planner_inferencer=MockInferencer(),
        executor_inferencer=MockInferencer(),
        analyzer_inferencer=MockInferencer(),
        enable_analysis=True,
        resume_workspace=ws,
    )
    defaults.update(kwargs)
    return PlanThenImplementInferencer(**defaults)


# =============================================================================
# Phase 2: Resume Detection Tests
# =============================================================================


class ResumeDetectionTest(unittest.TestCase):
    """Tests for _detect_resume_point and _detect_workspace_state."""

    def test_resume_analysis_ready(self):
        """2a: Workspace like task_20260310_074446 — plan+impl done, no analysis."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round1_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round1_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            iteration, phase, state, current_input, original_request = (
                pti._detect_resume_point(ws)
            )

            self.assertEqual(iteration, 1)
            self.assertEqual(phase, "analysis")
            self.assertTrue(state.plan_done)
            self.assertTrue(state.impl_done)
            self.assertFalse(state.analysis_done)
            self.assertEqual(current_input, "original request")
            self.assertEqual(original_request, "original request")

    def test_resume_plan_done_impl_pending(self):
        """2b: Workspace like task_20260310_053048 — plan done, no impl."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_completion_marker(ws, "plan")
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            iteration, phase, state, current_input, original_request = (
                pti._detect_resume_point(ws)
            )

            self.assertEqual(iteration, 1)
            self.assertEqual(phase, "implementation")
            self.assertTrue(state.plan_done)
            self.assertFalse(state.impl_done)
            self.assertEqual(current_input, "original request")

    def test_resume_empty_workspace(self):
        """2c: Empty workspace — nothing in outputs."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            os.makedirs(os.path.join(ws, "outputs"), exist_ok=True)
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            iteration, phase, state, current_input, original_request = (
                pti._detect_resume_point(ws)
            )

            self.assertEqual(iteration, 1)
            self.assertEqual(phase, "planning")
            self.assertFalse(state.plan_done)
            self.assertFalse(state.impl_done)

    def test_resume_completed_should_continue(self):
        """2d: Completed iteration with should_continue=True."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            _create_file(
                os.path.join(ws, "results", "analysis_summary.json"),
                json.dumps(
                    {
                        "should_continue": True,
                        "next_iteration_request": "Fix tests",
                        "summary": "Tests failing",
                        "analysis_doc_path": None,
                    }
                ),
            )

            pti = _make_pti_for_resume(ws)
            iteration, phase, state, current_input, original_request = (
                pti._detect_resume_point(ws)
            )

            self.assertEqual(iteration, 2)
            self.assertEqual(phase, "new_iteration")
            self.assertIn("iteration 2", current_input.lower())
            self.assertIn("Fix tests", current_input)
            self.assertEqual(original_request, "original request")

    def test_resume_completed_should_not_continue(self):
        """2e: Completed iteration with should_continue=False."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            _create_file(
                os.path.join(ws, "results", "analysis_summary.json"),
                json.dumps(
                    {
                        "should_continue": False,
                        "next_iteration_request": "",
                        "summary": "All good",
                        "analysis_doc_path": None,
                    }
                ),
            )

            pti = _make_pti_for_resume(ws)
            iteration, phase, state, current_input, original_request = (
                pti._detect_resume_point(ws)
            )

            self.assertEqual(iteration, 1)
            self.assertEqual(phase, "complete")
            self.assertEqual(current_input, "")

    def test_resume_followup_iteration_incomplete(self):
        """2f: Followup iteration_2 has plan but no impl."""
        with tempfile.TemporaryDirectory() as ws:
            # Iteration 1: complete
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            _create_file(
                os.path.join(ws, "results", "analysis_summary.json"),
                json.dumps(
                    {
                        "should_continue": True,
                        "next_iteration_request": "Fix tests",
                        "summary": "Tests failing",
                        "analysis_doc_path": None,
                    }
                ),
            )
            # Iteration 2: plan done, impl pending
            iter2_ws = os.path.join(ws, "followup_iterations", "iteration_2")
            _create_file(
                os.path.join(iter2_ws, "request.txt"), "iteration 2 handoff"
            )
            _create_file(
                os.path.join(iter2_ws, "outputs", "round0_plan.md"), "plan v2"
            )
            _create_completion_marker(iter2_ws, "plan")
            os.makedirs(os.path.join(iter2_ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            iteration, phase, state, current_input, original_request = (
                pti._detect_resume_point(ws)
            )

            self.assertEqual(iteration, 2)
            self.assertEqual(phase, "implementation")
            self.assertEqual(current_input, "iteration 2 handoff")
            self.assertEqual(original_request, "original request")


# =============================================================================
# Phase 3: Analysis-Only Mode Tests
# =============================================================================


class AnalysisOnlyModeTest(unittest.IsolatedAsyncioTestCase):
    """Tests for analysis-only mode (plan+impl disabled, analysis enabled)."""

    async def test_analysis_only_loads_from_disk(self):
        """3a: Only analyzer runs; planner/executor never called."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan text")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl text"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            _create_file(
                os.path.join(
                    ws, "outputs", "benchmarks", "round0", "results.json"
                ),
                '{"metric": 42}',
            )
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            mock_analyzer = MockInferencer(
                _response=(
                    '<Response>```json\n'
                    '{"should_continue": false, "summary": "done", '
                    '"next_iteration_request": ""}\n'
                    "```</Response>"
                )
            )
            planner = MockInferencer()
            executor = MockInferencer()

            planner._ainfer = AsyncMock(return_value="should not be called")
            executor._ainfer = AsyncMock(return_value="should not be called")

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                analyzer_inferencer=mock_analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("ignored input")

            planner._ainfer.assert_not_called()
            executor._ainfer.assert_not_called()

            summary_path = os.path.join(ws, "results", "analysis_summary.json")
            self.assertTrue(os.path.isfile(summary_path))

    async def test_analysis_only_no_results_skips(self):
        """3b: No benchmark/test files → analyzer skipped."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            mock_analyzer = MockInferencer()
            mock_analyzer._ainfer = AsyncMock(return_value="should not run")

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=mock_analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("ignored")

            mock_analyzer._ainfer.assert_not_called()
            self.assertIsInstance(result, PlanThenImplementResponse)


# =============================================================================
# Phase 4: Backward Compatibility Tests
# =============================================================================


class BackwardCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    """Tests verifying default config produces identical behavior."""

    async def test_default_config_identical_result(self):
        """4a: Default PTI with mock inferencers produces correct structure."""
        planner = MockInferencer(_response="plan text")
        executor = MockInferencer(_response="impl text")

        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )

        result = await pti._ainfer("test request")

        self.assertIsInstance(result, PlanThenImplementResponse)
        self.assertEqual(result.base_response, "impl text")
        self.assertEqual(result.plan_output, "plan text")
        self.assertEqual(len(result.iteration_history), 1)
        self.assertEqual(result.total_meta_iterations, 1)
        self.assertFalse(result.meta_iterations_exhausted)
        self.assertTrue(result.plan_approved)  # auto-approved (no interactive)

    async def test_exception_propagation_default_mode(self):
        """4b: Exceptions propagate without try/except in default mode."""
        planner = MockInferencer(_response="plan text")
        executor = MockInferencer()
        executor._ainfer = AsyncMock(side_effect=RuntimeError("test error"))

        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )

        with self.assertRaises(RuntimeError) as ctx:
            await pti._ainfer("test request")

        self.assertEqual(str(ctx.exception), "test error")

    async def test_plan_rejection(self):
        """4c: Plan rejection returns plan_str as base_response."""
        planner = MockInferencer(_response="plan text")
        executor = MockInferencer(_response="should not run")
        executor._ainfer = AsyncMock(return_value="should not run")

        mock_interactive = MagicMock()
        mock_interactive.send_response = MagicMock()
        mock_interactive.get_input = MagicMock(return_value="n")

        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
            interactive=mock_interactive,
        )

        result = await pti._ainfer("test request")

        self.assertEqual(result.base_response, "plan text")
        self.assertFalse(result.plan_approved)
        executor._ainfer.assert_not_called()

    async def test_config_splitting(self):
        """4d: Plan config goes to planner, impl config goes to executor."""
        planner = MockInferencer()
        executor = MockInferencer()

        captured_plan_config = {}
        captured_impl_config = {}

        async def mock_plan_ainfer(inp, inference_config=None, **kw):
            captured_plan_config.update(inference_config or {})
            return "plan"

        async def mock_impl_ainfer(inp, inference_config=None, **kw):
            captured_impl_config.update(inference_config or {})
            return "impl"

        planner._ainfer = mock_plan_ainfer
        executor._ainfer = mock_impl_ainfer

        pti = PlanThenImplementInferencer(
            planner_inferencer=planner,
            executor_inferencer=executor,
        )

        await pti._ainfer(
            "test",
            inference_config={
                "plan_config": {"key": "plan_val"},
                "implement_config": {"key": "impl_val"},
            },
        )

        self.assertEqual(captured_plan_config.get("key"), "plan_val")
        self.assertEqual(captured_impl_config.get("key"), "impl_val")


# =============================================================================
# Phase 5: Regression — Analysis-Only with Stale Artifacts
# =============================================================================


class StaleAnalysisArtifactsRegressionTest(unittest.IsolatedAsyncioTestCase):
    """Regression tests for analysis-only resume with pre-existing analysis artifacts.

    Reproduces the bug where --analysis-only on a workspace that already has
    results/analysis_summary.json (from a previous analysis run) causes PTI to
    short-circuit via the "complete" early-return path instead of running a
    fresh analysis.

    Root cause chain:
        1. copytree copies stale analysis_summary.json into the new workspace
        2. _detect_resume_point sees analysis_done=True → returns "complete"
        3. _ainfer early-returns with base_response=executor_output (impl text)
        4. User sees old implementation report instead of new analysis

    See: task_20260310_161045_analysis_20260313_193927 incident.
    """

    def _make_workspace_with_stale_analysis(
        self,
        ws: str,
        should_continue: bool = False,
    ) -> None:
        """Create a workspace simulating a copy from a previously-analyzed workspace.

        The workspace has:
        - outputs/round0_plan.md (plan done)
        - outputs/round0_implementation.md (impl done)
        - outputs/benchmarks/round0/results.json (benchmark results for analysis)
        - results/analysis_summary.json (STALE — from a prior analysis run)
        - analysis/iteration_1_analysis.md (STALE — prior analysis output)
        """
        _create_file(os.path.join(ws, "request.txt"), "original request")
        _create_file(
            os.path.join(ws, "outputs", "round0_plan.md"),
            "# Plan\nImplement the feature.",
        )
        _create_file(
            os.path.join(ws, "outputs", "round0_implementation.md"),
            "# Implementation Report\nThe feature was implemented.",
        )
        _create_completion_marker(ws, "plan")
        _create_completion_marker(ws, "impl")
        _create_file(
            os.path.join(ws, "outputs", "benchmarks", "round0", "results.json"),
            '{"metric": 42, "passed": true}',
        )
        _create_file(
            os.path.join(ws, "results", "analysis_summary.json"),
            json.dumps(
                {
                    "should_continue": should_continue,
                    "summary": "Stale analysis from previous run.",
                    "next_iteration_request": "Fix tests"
                    if should_continue
                    else "",
                    "analysis_doc_path": os.path.join(
                        ws, "analysis", "iteration_1_analysis.md"
                    ),
                }
            ),
        )
        _create_file(
            os.path.join(ws, "analysis", "iteration_1_analysis.md"),
            "# Old Analysis\nThis is stale analysis output from a previous run.",
        )

    async def test_analysis_only_with_stale_summary_still_runs_analyzer(self):
        """PRIMARY REGRESSION: Analyzer must be called even when analysis_summary.json exists.

        Bug: _detect_resume_point returns "complete" because analysis_done=True,
        causing _ainfer to early-return with old impl text. The analyzer is never called.
        """
        analyzer_called = False

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            nonlocal analyzer_called
            analyzer_called = True
            return (
                '<Response>```json\n'
                '{"should_continue": false, "summary": "Fresh analysis.", '
                '"next_iteration_request": ""}\n'
                "```</Response>"
            )

        with tempfile.TemporaryDirectory() as ws:
            self._make_workspace_with_stale_analysis(ws, should_continue=False)

            planner = MockInferencer()
            executor = MockInferencer()
            analyzer = MockInferencer()

            planner._ainfer = AsyncMock(return_value="should not run")
            executor._ainfer = AsyncMock(return_value="should not run")
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("Run analysis on existing workspace")

            # Core assertion: analyzer MUST be called
            self.assertTrue(
                analyzer_called,
                "Analyzer was not called — PTI short-circuited due to stale "
                "analysis_summary.json. This is the primary regression bug.",
            )
            # Planner and executor must NOT be called
            planner._ainfer.assert_not_called()
            executor._ainfer.assert_not_called()

    async def test_analysis_only_with_stale_continue_true_still_runs_analyzer(self):
        """Stale summary with should_continue=True must not trigger a new iteration.

        When analysis-only, the user wants a fresh analysis, not to continue
        iterating. Even if the stale summary says "continue", we should just
        run analysis on the current workspace.
        """
        analyzer_called = False

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            nonlocal analyzer_called
            analyzer_called = True
            return (
                '<Response>```json\n'
                '{"should_continue": false, "summary": "Fresh re-analysis.", '
                '"next_iteration_request": ""}\n'
                "```</Response>"
            )

        with tempfile.TemporaryDirectory() as ws:
            self._make_workspace_with_stale_analysis(ws, should_continue=True)

            analyzer = MockInferencer()
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("Run analysis on existing workspace")

            self.assertTrue(
                analyzer_called,
                "Analyzer was not called — stale should_continue=True caused "
                "PTI to skip analysis or attempt a new iteration.",
            )

    async def test_analysis_only_overwrites_stale_summary(self):
        """After fresh analysis, analysis_summary.json must contain new results.

        Verifies the stale summary is replaced, not preserved.
        """

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            return (
                '<Response>```json\n'
                '{"should_continue": false, '
                '"summary": "FRESH_ANALYSIS_MARKER", '
                '"next_iteration_request": ""}\n'
                "```</Response>"
            )

        with tempfile.TemporaryDirectory() as ws:
            self._make_workspace_with_stale_analysis(ws, should_continue=False)

            analyzer = MockInferencer()
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            await pti._ainfer("Run analysis on existing workspace")

            summary_path = os.path.join(ws, "results", "analysis_summary.json")
            self.assertTrue(os.path.isfile(summary_path))
            with open(summary_path) as f:
                data = json.load(f)
            self.assertIn(
                "FRESH_ANALYSIS_MARKER",
                data.get("summary", ""),
                "analysis_summary.json still contains stale data — "
                "the fresh analysis did not overwrite it.",
            )

    async def test_analysis_only_response_is_not_impl_text(self):
        """base_response must NOT be the old implementation text.

        Bug: Even when analysis runs, base_response is set to
        executor_output (impl text) rather than analysis output.
        """
        IMPL_TEXT = "# Implementation Report\nThe feature was implemented."
        ANALYSIS_TEXT = (
            '<Response>```json\n'
            '{"should_continue": false, "summary": "ANALYSIS_OUTPUT_HERE", '
            '"next_iteration_request": ""}\n'
            "```</Response>"
        )

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            return ANALYSIS_TEXT

        with tempfile.TemporaryDirectory() as ws:
            self._make_workspace_with_stale_analysis(ws, should_continue=False)

            analyzer = MockInferencer()
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("Run analysis on existing workspace")

            self.assertNotEqual(
                str(result.base_response),
                IMPL_TEXT,
                "base_response is the old implementation text — "
                "PTI returned cached impl output instead of analysis output.",
            )

    async def test_analysis_only_no_stale_summary_still_works(self):
        """Baseline: analysis-only without stale artifacts works correctly.

        Ensures the stale-artifact tests are meaningful by confirming the
        non-stale case works.
        """
        analyzer_called = False

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            nonlocal analyzer_called
            analyzer_called = True
            return (
                '<Response>```json\n'
                '{"should_continue": false, "summary": "done", '
                '"next_iteration_request": ""}\n'
                "```</Response>"
            )

        with tempfile.TemporaryDirectory() as ws:
            # Workspace WITHOUT stale analysis_summary.json
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            _create_file(
                os.path.join(
                    ws, "outputs", "benchmarks", "round0", "results.json"
                ),
                '{"metric": 42}',
            )
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            analyzer = MockInferencer()
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("Run analysis on existing workspace")

            self.assertTrue(
                analyzer_called,
                "Analyzer should be called when no stale summary exists.",
            )


# =============================================================================
# Phase 6: Integration — Analysis-Only Resume End-to-End
# =============================================================================


class AnalysisOnlyResumeIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """End-to-end integration tests for analysis-only resume flow.

    Unlike the Phase 5 regression tests (which verify the analyzer is called),
    these tests exercise the FULL pipeline: workspace setup → _ainfer →
    analyzer config vars → disk artifacts → response structure.
    """

    @staticmethod
    def _make_multi_round_workspace(
        ws: str,
        *,
        stale_summary: bool = True,
        stale_continue: bool = False,
        num_plan_rounds: int = 2,
        num_impl_rounds: int = 2,
        include_tests: bool = False,
    ) -> None:
        """Build a realistic multi-round workspace with optional stale analysis.

        Creates:
          - request.txt
          - outputs/round{0..N}_plan.md
          - outputs/round{0..N}_implementation.md
          - outputs/benchmarks/round{0..N}/results.json
          - outputs/tests/round0/output.txt  (if include_tests)
          - results/analysis_summary.json    (if stale_summary)
          - analysis/iteration_1_analysis.md (if stale_summary)
        """
        _create_file(os.path.join(ws, "request.txt"), "Design a REST API")

        for i in range(num_plan_rounds):
            _create_file(
                os.path.join(ws, "outputs", f"round{i}_plan.md"),
                f"# Plan v{i}\nDesign iteration {i}.",
            )

        for i in range(num_impl_rounds):
            _create_file(
                os.path.join(ws, "outputs", f"round{i}_implementation.md"),
                f"# Implementation Report (Round {i})\n"
                f"Changes applied in round {i}.",
            )

        _create_completion_marker(ws, "plan")
        _create_completion_marker(ws, "impl")

        for i in range(num_impl_rounds):
            _create_file(
                os.path.join(
                    ws, "outputs", "benchmarks", f"round{i}", "results.json"
                ),
                json.dumps({"round": i, "latency_ms": 100 - i * 10}),
            )

        if include_tests:
            _create_file(
                os.path.join(ws, "outputs", "tests", "round0", "output.txt"),
                "PASS: 12 tests passed, 0 failed",
            )

        os.makedirs(os.path.join(ws, "results"), exist_ok=True)

        if stale_summary:
            _create_file(
                os.path.join(ws, "results", "analysis_summary.json"),
                json.dumps(
                    {
                        "should_continue": stale_continue,
                        "summary": "STALE_SUMMARY_MARKER",
                        "next_iteration_request": "fix perf"
                        if stale_continue
                        else "",
                        "analysis_doc_path": os.path.join(
                            ws, "analysis", "iteration_1_analysis.md"
                        ),
                    }
                ),
            )
            _create_file(
                os.path.join(ws, "analysis", "iteration_1_analysis.md"),
                "# Stale Analysis\nThis is old.",
            )

    async def test_e2e_stale_workspace_config_vars_and_disk_artifacts(self):
        """Full E2E: multi-round workspace with stale summary.

        Verifies:
        1. Analyzer receives correct inference_config template vars
        2. Correct implementation file (latest round) is referenced
        3. analysis_summary.json is overwritten with fresh content
        4. Planner / executor never invoked
        5. Both benchmarks and tests result types detected
        """
        captured_config = {}
        captured_input = None

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            nonlocal captured_config, captured_input
            captured_config.update(inference_config or {})
            captured_input = inp
            return (
                '<Response>```json\n'
                '{"should_continue": false, '
                '"summary": "FRESH_E2E_ANALYSIS", '
                '"next_iteration_request": ""}\n'
                "```</Response>"
            )

        with tempfile.TemporaryDirectory() as ws:
            self._make_multi_round_workspace(
                ws,
                stale_summary=True,
                stale_continue=False,
                num_plan_rounds=2,
                num_impl_rounds=2,
                include_tests=True,
            )

            planner = MockInferencer()
            executor = MockInferencer()
            analyzer = MockInferencer()
            planner._ainfer = AsyncMock(return_value="fail")
            executor._ainfer = AsyncMock(return_value="fail")
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("Run analysis")

            # ---- Phase isolation ----
            planner._ainfer.assert_not_called()
            executor._ainfer.assert_not_called()

            # ---- Analyzer received correct input ----
            self.assertEqual(captured_input, "Design a REST API")

            # ---- Config vars ----
            self.assertEqual(captured_config.get("result_type"), "benchmarks")
            self.assertIn("benchmarks", captured_config.get("result_path_base", ""))
            self.assertIn("round1", captured_config.get("result_path_latest", ""))
            self.assertEqual(
                captured_config.get("result_path"),
                captured_config.get("result_path_latest"),
            )
            self.assertEqual(captured_config.get("meta_iteration"), 1)
            self.assertIn(
                "round1_implementation.md",
                captured_config.get("implementation_output_path", ""),
            )
            self.assertIsNotNone(captured_config.get("analysis_request"))
            self.assertIn(
                "first iteration",
                captured_config.get("previous_iteration_paths", ""),
            )

            # ---- Multiple result types detected ----
            analysis_req = captured_config.get("analysis_request", "")
            self.assertIn("tests", analysis_req)

            # ---- Disk: stale summary overwritten ----
            summary_path = os.path.join(ws, "results", "analysis_summary.json")
            with open(summary_path) as f:
                summary = json.load(f)
            self.assertIn("FRESH_E2E_ANALYSIS", summary["summary"])
            self.assertNotIn("STALE_SUMMARY_MARKER", summary["summary"])

    async def test_e2e_stale_continue_true_no_multi_iteration(self):
        """Stale should_continue=True must NOT trigger a second iteration.

        In analysis-only mode the user wants a fresh analysis, not a new
        plan+implement cycle.  Verifies:
        1. Exactly one iteration runs
        2. Analyzer called once
        3. No followup_iterations/ directory created
        """
        call_count = 0

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            nonlocal call_count
            call_count += 1
            return (
                '<Response>```json\n'
                '{"should_continue": false, '
                '"summary": "Fresh analysis done.", '
                '"next_iteration_request": ""}\n'
                "```</Response>"
            )

        with tempfile.TemporaryDirectory() as ws:
            self._make_multi_round_workspace(
                ws,
                stale_summary=True,
                stale_continue=True,
            )

            analyzer = MockInferencer()
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("Run analysis")

            self.assertEqual(call_count, 1)
            self.assertEqual(result.total_meta_iterations, 1)
            self.assertFalse(
                os.path.isdir(os.path.join(ws, "followup_iterations")),
                "followup_iterations/ should not be created in analysis-only mode",
            )

    async def test_e2e_response_structure(self):
        """Verify PlanThenImplementResponse fields for analysis-only resume."""

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            return (
                '<Response>```json\n'
                '{"should_continue": false, '
                '"summary": "All benchmarks passed.", '
                '"next_iteration_request": ""}\n'
                "```</Response>"
            )

        with tempfile.TemporaryDirectory() as ws:
            self._make_multi_round_workspace(
                ws,
                stale_summary=True,
                num_plan_rounds=1,
                num_impl_rounds=1,
            )

            analyzer = MockInferencer()
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("Run analysis")

            # ---- Response type ----
            self.assertIsInstance(result, PlanThenImplementResponse)

            # ---- base_response is the analysis text, not impl text ----
            self.assertNotIn(
                "Implementation Report",
                str(result.base_response),
            )

            # ---- Iteration history ----
            self.assertEqual(len(result.iteration_history), 1)
            rec = result.iteration_history[0]
            self.assertEqual(rec.iteration, 1)
            self.assertTrue(rec.test_results_found)
            self.assertIsNotNone(rec.analysis_output)
            self.assertIn("should_continue", rec.analysis_output)
            self.assertFalse(rec.should_continue)

            # ---- Plan/impl loaded from disk ----
            self.assertIn("Plan v0", rec.plan_output)
            self.assertIn("Implementation Report", rec.executor_output)

            # ---- Meta-iteration bookkeeping ----
            self.assertEqual(result.total_meta_iterations, 1)
            self.assertFalse(result.meta_iterations_exhausted)

    async def test_e2e_fresh_workspace_no_stale_artifacts(self):
        """Baseline E2E: analysis-only on workspace without stale summary.

        Ensures the integration tests are meaningful by confirming the
        non-stale case works identically.
        """
        captured_config = {}

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            captured_config.update(inference_config or {})
            return (
                '<Response>```json\n'
                '{"should_continue": false, '
                '"summary": "Baseline analysis.", '
                '"next_iteration_request": ""}\n'
                "```</Response>"
            )

        with tempfile.TemporaryDirectory() as ws:
            self._make_multi_round_workspace(
                ws,
                stale_summary=False,
                num_plan_rounds=1,
                num_impl_rounds=1,
            )

            analyzer = MockInferencer()
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
            )

            result = await pti._ainfer("Run analysis")

            self.assertEqual(captured_config.get("result_type"), "benchmarks")
            self.assertIn("round0", captured_config.get("result_path_latest", ""))
            summary_path = os.path.join(ws, "results", "analysis_summary.json")
            self.assertTrue(os.path.isfile(summary_path))


# =============================================================================
# Phase 7: Resume Detection — Stale Artifacts
# =============================================================================


class StaleArtifactsResumeDetectionTest(unittest.TestCase):
    """Unit tests for _detect_resume_point with stale analysis artifacts.

    Tests the pure detection logic to verify the resume point calculation
    matches expected behavior for analysis-only scenarios.
    """

    def test_complete_workspace_detected_as_complete(self):
        """Document current behavior: workspace with all phases done → "complete".

        This is the detection-level root cause. _detect_resume_point does not
        know about enable_planning/enable_implementation, so it correctly
        returns "complete". The fix must be in _ainfer, which overrides the
        resume_phase for analysis-only mode.
        """
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            _create_file(
                os.path.join(ws, "results", "analysis_summary.json"),
                json.dumps(
                    {
                        "should_continue": False,
                        "summary": "done",
                        "next_iteration_request": "",
                        "analysis_doc_path": None,
                    }
                ),
            )

            pti = _make_pti_for_resume(ws)
            iteration, phase, state, current_input, original_request = (
                pti._detect_resume_point(ws)
            )

            # _detect_resume_point correctly identifies this as "complete"
            # because it doesn't know about analysis-only mode.
            # The fix for the stale-artifacts bug is in _ainfer, which must
            # override this to "analysis" when analysis-only mode is active.
            self.assertEqual(phase, "complete")
            self.assertEqual(iteration, 1)
            self.assertTrue(state.analysis_done)

    def test_workspace_state_detects_analysis_done(self):
        """_detect_workspace_state correctly reads analysis_summary.json."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(
                os.path.join(ws, "results", "analysis_summary.json"),
                json.dumps(
                    {
                        "should_continue": False,
                        "summary": "done",
                        "next_iteration_request": "",
                        "analysis_doc_path": "/some/path.md",
                    }
                ),
            )
            os.makedirs(os.path.join(ws, "outputs"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            state = pti._detect_workspace_state(ws, 1)

            self.assertTrue(state.analysis_done)
            self.assertEqual(state.analysis_doc_path, "/some/path.md")

    def test_workspace_state_no_summary_means_not_done(self):
        """Without analysis_summary.json, analysis_done is False."""
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)
            os.makedirs(os.path.join(ws, "outputs"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            state = pti._detect_workspace_state(ws, 1)

            self.assertFalse(state.analysis_done)
            self.assertIsNone(state.analysis_doc_path)


# =============================================================================
# Phase 8: Initial Plan File — Skip Proposal, Start at Review
# =============================================================================


class InitialPlanFileTest(unittest.IsolatedAsyncioTestCase):
    """Tests for --initial-plan: inject a plan file, skip proposer, start at review."""

    async def test_initial_plan_file_injects_override_into_planner_config(self):
        """PTI with initial_plan_file passes initial_response_override to planner.

        The PTI's responsibility is to:
        1. Load the file
        2. Inject 'initial_response_override' into the plan inference_config
        The DualInferencer then uses that to skip propose and start at review.
        """
        captured_config = {}

        async def mock_planner_ainfer(inp, inference_config=None, **kw):
            captured_config.update(inference_config or {})
            return "plan consensus result"

        with tempfile.TemporaryDirectory() as ws:
            plan_file = os.path.join(ws, "my_plan.md")
            with open(plan_file, "w") as f:
                f.write("# My Custom Plan\nStep 1: Do the thing.\nStep 2: Done.")

            planner = MockInferencer()
            planner._ainfer = mock_planner_ainfer

            executor = MockInferencer()
            executor._ainfer = AsyncMock(return_value="impl output")

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                enable_planning=True,
                enable_implementation=True,
                initial_plan_file=plan_file,
            )

            result = await pti._ainfer("Build a REST API")

            self.assertIn(
                "initial_response_override",
                captured_config,
                "PTI must pass initial_response_override in planner config",
            )
            self.assertEqual(
                captured_config["initial_response_override"],
                "# My Custom Plan\nStep 1: Do the thing.\nStep 2: Done.",
            )

    async def test_initial_plan_file_plan_only_mode(self):
        """--plan-only --initial-plan: injects override, no implementation."""
        captured_config = {}

        async def mock_planner_ainfer(inp, inference_config=None, **kw):
            captured_config.update(inference_config or {})
            return "plan reviewed"

        with tempfile.TemporaryDirectory() as ws:
            plan_file = os.path.join(ws, "plan.md")
            with open(plan_file, "w") as f:
                f.write("# Plan\nDo X, then Y.")

            planner = MockInferencer()
            planner._ainfer = mock_planner_ainfer

            executor_mock = AsyncMock(return_value="should not run")
            executor = MockInferencer()
            executor._ainfer = executor_mock

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                enable_planning=True,
                enable_implementation=False,
                initial_plan_file=plan_file,
            )

            result = await pti._ainfer("Review this plan")

            self.assertIn("initial_response_override", captured_config)
            self.assertEqual(
                captured_config["initial_response_override"],
                "# Plan\nDo X, then Y.",
            )
            executor_mock.assert_not_called()

    async def test_initial_plan_only_injected_on_first_iteration(self):
        """initial_plan_file override is only injected on the first iteration."""
        call_configs = []

        async def mock_planner_ainfer(inp, inference_config=None, **kw):
            call_configs.append(dict(inference_config or {}))
            return "plan result"

        with tempfile.TemporaryDirectory() as ws:
            plan_file = os.path.join(ws, "plan.md")
            with open(plan_file, "w") as f:
                f.write("# Plan\nDo X.")

            planner = MockInferencer()
            planner._ainfer = mock_planner_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=MockInferencer(),
                enable_planning=True,
                enable_implementation=False,
                initial_plan_file=plan_file,
            )

            await pti._ainfer("Review this plan")

            self.assertTrue(len(call_configs) >= 1)
            self.assertIn(
                "initial_response_override", call_configs[0],
                "First iteration must have the override",
            )

    async def test_initial_plan_file_not_found_raises_error(self):
        """Non-existent plan file raises FileNotFoundError before any LLM call."""
        pti = PlanThenImplementInferencer(
            planner_inferencer=MockInferencer(),
            executor_inferencer=MockInferencer(),
            enable_planning=True,
            enable_implementation=False,
            initial_plan_file="/nonexistent/path/plan.md",
        )

        with self.assertRaises(FileNotFoundError):
            await pti._ainfer("Review this plan")


# =============================================================================
# Phase 9: Step-In-Progress Marker Lifecycle & Resume Detection
# =============================================================================


class StepInProgressMarkerTest(unittest.IsolatedAsyncioTestCase):
    """Tests for the pre-execution step marker lifecycle and resume detection.

    Verifies that __wf_step_in_progress__.json is correctly created before
    step execution, cleared after success, persists on failure, and drives
    resume-context injection in the executor prompt.
    """

    @staticmethod
    def _get_marker_path(ws: str) -> str:
        """Return the expected marker file path for PTI checkpoints."""
        return os.path.join(
            ws, "checkpoints", "pti", "step___wf_step_in_progress__.json"
        )

    async def test_marker_written_before_step_execution(self):
        """Verify __wf_step_in_progress__.json exists while the step runs."""
        marker_seen_during_execution = False

        with tempfile.TemporaryDirectory() as ws:
            marker_path = self._get_marker_path(ws)

            async def mock_executor_ainfer(inp, inference_config=None, **kw):
                nonlocal marker_seen_during_execution
                marker_seen_during_execution = os.path.isfile(marker_path)
                return "impl output"

            planner = MockInferencer(_response="plan")
            executor = MockInferencer()
            executor._ainfer = mock_executor_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                workspace_path=ws,
            )
            await pti._ainfer("test request")

            self.assertTrue(
                marker_seen_during_execution,
                "Marker file should exist during step execution",
            )

    async def test_marker_cleared_after_step_success(self):
        """Verify marker is removed when step completes successfully."""
        with tempfile.TemporaryDirectory() as ws:
            marker_path = self._get_marker_path(ws)

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(_response="plan"),
                executor_inferencer=MockInferencer(_response="impl"),
                workspace_path=ws,
            )
            await pti._ainfer("test request")

            self.assertFalse(
                os.path.isfile(marker_path),
                "Marker file should be removed after successful step completion",
            )

    async def test_marker_persists_on_step_failure(self):
        """Verify marker remains when step raises an exception."""
        with tempfile.TemporaryDirectory() as ws:
            marker_path = self._get_marker_path(ws)

            executor = MockInferencer()
            executor._ainfer = AsyncMock(side_effect=RuntimeError("step failed"))

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(_response="plan"),
                executor_inferencer=executor,
                workspace_path=ws,
            )

            with self.assertRaises(RuntimeError):
                await pti._ainfer("test request")

            self.assertTrue(
                os.path.isfile(marker_path),
                "Marker file should persist when step fails",
            )

    async def test_resume_detects_step_attempted(self):
        """Load checkpoint + marker, verify _step_was_previously_attempted."""
        with tempfile.TemporaryDirectory() as ws:
            ckpt_dir = os.path.join(ws, "checkpoints", "pti")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Create workspace files so checkpoint validation passes
            _create_file(os.path.join(ws, "request.txt"), "test request")
            _create_file(
                os.path.join(ws, "outputs", "round0_plan.md"), "plan"
            )
            _create_completion_marker(ws, "plan")
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            # Write checkpoint: next_step_index=2 (implement step)
            ckpt = {
                "version": 1,
                "exec_seq": 1,
                "step_index": 1,
                "result_id": "__synth_sentinel__",
                "next_step_index": 2,
                "loop_counts": {},
                "state": {
                    "iteration": 1,
                    "current_input": "test request",
                    "original_request": "test request",
                    "plan_output_text": "plan",
                    "plan_file_path": None,
                    "plan_approved": True,
                    "executor_output_text": "",
                    "should_continue": False,
                    "next_iteration_request": "",
                    "iteration_records": [],
                },
            }
            with open(
                os.path.join(ckpt_dir, "step___wf_checkpoint__.json"), "w"
            ) as f:
                json.dump(ckpt, f)

            # Write sentinel result
            with open(
                os.path.join(ckpt_dir, "step___synth_sentinel__.json"), "w"
            ) as f:
                json.dump({"_synthetic": True}, f)

            # Write in-progress marker for step 2
            marker = {
                "step_index": 2,
                "step_name": "implement",
                "started_at": "2026-03-15T00:00:00",
                "attempt": 1,
            }
            with open(
                os.path.join(
                    ckpt_dir, "step___wf_step_in_progress__.json"
                ),
                "w",
            ) as f:
                json.dump(marker, f)

            was_attempted = False

            async def mock_executor_ainfer(inp, inference_config=None, **kw):
                nonlocal was_attempted
                was_attempted = getattr(
                    pti, "_step_was_previously_attempted", False
                )
                return "impl output"

            executor = MockInferencer()
            executor._ainfer = mock_executor_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(_response="plan"),
                executor_inferencer=executor,
                resume_workspace=ws,
            )

            await pti._ainfer("test request")

            self.assertTrue(
                was_attempted,
                "_step_was_previously_attempted should be True when marker exists",
            )

    def test_resume_context_in_executor_prompt(self):
        """Verify _build_executor_input() includes resume warning."""
        pti = PlanThenImplementInferencer(
            planner_inferencer=MockInferencer(),
            executor_inferencer=MockInferencer(),
        )
        pti._step_was_previously_attempted = True

        result = pti._build_executor_input("original task", "the plan")

        self.assertIn("Resume Context", result)
        self.assertIn("sl status", result)
        self.assertIn("partial changes", result)

    def test_no_resume_context_when_not_attempted(self):
        """Verify _build_executor_input() omits resume warning normally."""
        pti = PlanThenImplementInferencer(
            planner_inferencer=MockInferencer(),
            executor_inferencer=MockInferencer(),
        )

        result = pti._build_executor_input("task", "plan")

        self.assertNotIn("Resume Context", result)

    async def test_attempt_counter_increments(self):
        """Verify marker records attempt=1 on first failure."""
        with tempfile.TemporaryDirectory() as ws:
            marker_path = self._get_marker_path(ws)

            executor = MockInferencer()
            executor._ainfer = AsyncMock(
                side_effect=RuntimeError("first attempt fails")
            )

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(_response="plan"),
                executor_inferencer=executor,
                workspace_path=ws,
            )

            with self.assertRaises(RuntimeError):
                await pti._ainfer("test request")

            self.assertTrue(os.path.isfile(marker_path))
            with open(marker_path) as f:
                marker_data = json.load(f)
            self.assertEqual(marker_data["step_index"], 2)
            self.assertEqual(marker_data["attempt"], 1)
            self.assertEqual(marker_data["step_name"], "implement")
            self.assertIn("started_at", marker_data)


    async def test_flag_reset_after_successful_resume(self):
        """After a successful resumed step, _step_was_previously_attempted resets.

        Prevents false resume warnings on subsequent loop iterations.
        """
        with tempfile.TemporaryDirectory() as ws:
            ckpt_dir = os.path.join(ws, "checkpoints", "pti")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Write checkpoint: next_step_index=2 (implement step)
            ckpt = {
                "version": 1,
                "exec_seq": 1,
                "step_index": 1,
                "result_id": "__synth_sentinel__",
                "next_step_index": 2,
                "loop_counts": {},
                "state": {
                    "iteration": 1,
                    "current_input": "test request",
                    "original_request": "test request",
                    "plan_output_text": "plan",
                    "plan_file_path": None,
                    "plan_approved": True,
                    "executor_output_text": "",
                    "should_continue": False,
                    "next_iteration_request": "",
                    "iteration_records": [],
                },
            }
            with open(
                os.path.join(ckpt_dir, "step___wf_checkpoint__.json"), "w"
            ) as f:
                json.dump(ckpt, f)

            # Write sentinel result
            with open(
                os.path.join(ckpt_dir, "step___synth_sentinel__.json"), "w"
            ) as f:
                json.dump({"_synthetic": True}, f)

            # Write in-progress marker for step 2
            marker = {
                "step_index": 2,
                "step_name": "implement",
                "started_at": "2026-03-15T00:00:00+00:00",
                "attempt": 1,
            }
            with open(
                os.path.join(
                    ckpt_dir, "step___wf_step_in_progress__.json"
                ),
                "w",
            ) as f:
                json.dump(marker, f)

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(_response="plan"),
                executor_inferencer=MockInferencer(_response="impl"),
                resume_workspace=ws,
            )

            await pti._ainfer("test request")

            self.assertFalse(
                pti._step_was_previously_attempted,
                "_step_was_previously_attempted must reset after successful "
                "step completion to avoid false resume warnings on loop "
                "iterations",
            )
            self.assertIsNone(pti._previous_attempt_info)


class Tier2CompletionMarkerTest(unittest.IsolatedAsyncioTestCase):
    """Tests for Tier 2 (file-based) completion markers.

    Tier 2 uses ``.plan_completed`` / ``.impl_completed`` marker files to
    distinguish between "output file exists AND step finished" vs "output
    file exists but step may have been interrupted."
    """

    # ------------------------------------------------------------------
    # Test 1: impl_done=True when both output file AND .impl_completed exist
    # ------------------------------------------------------------------
    def test_impl_fully_completed(self):
        """impl_done=True when output AND marker both exist."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "request")
            _create_file(
                os.path.join(ws, "outputs", "round0_plan.md"), "plan"
            )
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            state = pti._detect_workspace_state(ws, iteration=1)

            self.assertTrue(state.impl_done)
            self.assertFalse(state.impl_partial)

    # ------------------------------------------------------------------
    # Test 2: impl_done=False, impl_partial=True when output exists but
    #         .impl_completed is missing (the false-positive fix)
    # ------------------------------------------------------------------
    def test_impl_partial_when_marker_missing(self):
        """impl_done=False, impl_partial=True when marker is absent."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "request")
            _create_file(
                os.path.join(ws, "outputs", "round0_plan.md"), "plan"
            )
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            # Intentionally NO .impl_completed marker
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            state = pti._detect_workspace_state(ws, iteration=1)

            self.assertFalse(state.impl_done)
            self.assertTrue(state.impl_partial)

    # ------------------------------------------------------------------
    # Test 3: impl_done=False, impl_partial=False when no output exists
    # ------------------------------------------------------------------
    def test_impl_never_attempted(self):
        """impl_done=False, impl_partial=False when no impl file exists."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "request")
            _create_file(
                os.path.join(ws, "outputs", "round0_plan.md"), "plan"
            )
            _create_completion_marker(ws, "plan")
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            state = pti._detect_workspace_state(ws, iteration=1)

            self.assertFalse(state.impl_done)
            self.assertFalse(state.impl_partial)

    # ------------------------------------------------------------------
    # Test 4: Three-state logic for plan phase
    # ------------------------------------------------------------------
    def test_plan_fully_completed(self):
        """plan_done=True, plan_partial=False when plan marker exists."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "request")
            _create_file(
                os.path.join(ws, "outputs", "round0_plan.md"), "plan"
            )
            _create_completion_marker(ws, "plan")
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            state = pti._detect_workspace_state(ws, iteration=1)

            self.assertTrue(state.plan_done)
            self.assertFalse(state.plan_partial)

    def test_plan_partial_when_marker_missing(self):
        """plan_done=True, plan_partial=True when marker is absent.

        Plan content IS usable (DualInferencer wrote consensus output) but
        completion was never confirmed.  plan_done stays True for backward
        compat.
        """
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "request")
            _create_file(
                os.path.join(ws, "outputs", "round0_plan.md"), "plan"
            )
            # Intentionally NO .plan_completed marker
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            state = pti._detect_workspace_state(ws, iteration=1)

            self.assertTrue(state.plan_done)
            self.assertTrue(state.plan_partial)

    def test_plan_never_attempted(self):
        """plan_done=False, plan_partial=False when no plan file exists."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "request")
            os.makedirs(os.path.join(ws, "outputs"), exist_ok=True)
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = _make_pti_for_resume(ws)
            state = pti._detect_workspace_state(ws, iteration=1)

            self.assertFalse(state.plan_done)
            self.assertFalse(state.plan_partial)

    # ------------------------------------------------------------------
    # Test 5: _synthesize_checkpoint sets _impl_was_partially_attempted
    #         in the state dict when impl_partial=True
    # ------------------------------------------------------------------
    async def test_synthesize_sets_previously_attempted_on_impl_partial(self):
        """_synthesize_checkpoint_from_workspace sets state flag when impl_partial."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(
                os.path.join(ws, "outputs", "round0_plan.md"), "plan"
            )
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            # NO .impl_completed → impl_partial=True
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(_response="plan"),
                executor_inferencer=MockInferencer(_response="impl"),
                resume_workspace=ws,
            )

            checkpoint = pti._synthesize_checkpoint_from_workspace()

            self.assertIsNotNone(checkpoint)
            # The flag is stored in the state dict (not on self) because
            # _arun() resets self._step_was_previously_attempted.
            self.assertTrue(
                checkpoint["state"].get("_impl_was_partially_attempted"),
                "impl_partial should set _impl_was_partially_attempted in state dict",
            )
            self.assertEqual(pti._step_attempt_counts.get(2), 1)

    # ------------------------------------------------------------------
    # Test 6: Backward compatibility — workspace WITH markers works
    #         identically to current behavior for fully-completed runs
    # ------------------------------------------------------------------
    async def test_backward_compat_with_markers(self):
        """Workspace with markers produces same resume behavior as before."""
        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(
                os.path.join(ws, "outputs", "round0_plan.md"), "plan"
            )
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_completion_marker(ws, "plan")
            _create_completion_marker(ws, "impl")
            _create_file(
                os.path.join(ws, "results", "analysis_summary.json"),
                json.dumps(
                    {
                        "should_continue": False,
                        "summary": "All good",
                        "analysis_doc_path": None,
                    }
                ),
            )

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(_response="plan"),
                executor_inferencer=MockInferencer(_response="impl"),
                resume_workspace=ws,
            )

            iteration, phase, state, _, _ = pti._detect_resume_point(ws)

            self.assertEqual(iteration, 1)
            self.assertEqual(phase, "complete")
            self.assertTrue(state.plan_done)
            self.assertFalse(state.plan_partial)
            self.assertTrue(state.impl_done)
            self.assertFalse(state.impl_partial)
            self.assertTrue(state.analysis_done)

            # _step_was_previously_attempted should NOT be set for a
            # fully-completed workspace
            self.assertFalse(
                getattr(pti, "_step_was_previously_attempted", False),
            )


if __name__ == "__main__":
    unittest.main()

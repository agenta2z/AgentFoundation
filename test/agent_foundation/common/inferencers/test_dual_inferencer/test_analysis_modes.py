

"""Unit tests for analysis mode support in PlanThenImplementInferencer.

Tests the three analysis modes (last_round_only, last_with_cross_ref, all_rounds)
and their supporting methods: _find_latest_round_dir, _load_analysis_request_template,
_build_analysis_config_vars, _has_results, as well as CLI flag parsing.
"""

import importlib
import os
import tempfile
import unittest

from attr import attrib, attrs


def _has_server_factories() -> bool:
    """Check if agent_foundation.server.factories is available."""
    try:
        importlib.import_module("agent_foundation.server.factories")
        return True
    except ImportError:
        return False
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    ANALYSIS_MODE_CLI_MAP,
    ANALYSIS_MODE_FILE_MAP,
    PlanThenImplementInferencer,
    VALID_ANALYSIS_MODES,
)
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)


@attrs
class MockInferencer(InferencerBase):
    """Minimal mock inferencer for unit testing."""

    _response: str = attrib(default="mock response")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._response


def _create_file(path: str, content: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _make_pti(templates_dir: str | None = None, **kwargs) -> PlanThenImplementInferencer:
    """Create a minimal PTI with analysis enabled for testing."""
    with tempfile.TemporaryDirectory() as ws:
        defaults = dict(
            planner_inferencer=MockInferencer(),
            executor_inferencer=MockInferencer(),
            analyzer_inferencer=MockInferencer(),
            enable_analysis=True,
            resume_workspace=kwargs.pop("resume_workspace", ws),
            analysis_templates_dir=templates_dir,
        )
        defaults.update(kwargs)
        return PlanThenImplementInferencer(**defaults)


# =============================================================================
# Mode Constants and Validation
# =============================================================================


class AnalysisModeConstantsTest(unittest.TestCase):
    """Tests for mode constants and validation logic."""

    def test_valid_modes_set(self):
        """All three analysis modes are recognized."""
        self.assertEqual(
            VALID_ANALYSIS_MODES,
            {"last_round_only", "last_with_cross_ref", "all_rounds"},
        )

    def test_cli_map_covers_all_modes(self):
        """Every CLI short name maps to a valid internal mode."""
        for cli_name, internal_name in ANALYSIS_MODE_CLI_MAP.items():
            self.assertIn(internal_name, VALID_ANALYSIS_MODES, f"CLI name '{cli_name}' maps to unknown mode '{internal_name}'")

    def test_file_map_covers_all_modes(self):
        """Every valid mode has a corresponding file map entry."""
        for mode in VALID_ANALYSIS_MODES:
            self.assertIn(mode, ANALYSIS_MODE_FILE_MAP, f"Mode '{mode}' has no file map entry")

    def test_default_mode_is_valid(self):
        """Default analysis_mode is a valid mode."""
        pti = _make_pti()
        self.assertIn(pti.analysis_mode, VALID_ANALYSIS_MODES)

    def test_invalid_mode_raises(self):
        """Invalid analysis_mode raises ValueError at construction."""
        with self.assertRaises(ValueError) as ctx:
            _make_pti(analysis_mode="banana")
        self.assertIn("banana", str(ctx.exception))
        self.assertIn("Invalid analysis_mode", str(ctx.exception))

    def test_each_valid_mode_accepted(self):
        """Each valid mode is accepted without error."""
        for mode in VALID_ANALYSIS_MODES:
            pti = _make_pti(analysis_mode=mode)
            self.assertEqual(pti.analysis_mode, mode)


# =============================================================================
# _find_latest_round_dir
# =============================================================================


class FindLatestRoundDirTest(unittest.TestCase):
    """Tests for _find_latest_round_dir()."""

    def setUp(self):
        self.pti = _make_pti()

    def test_multiple_round_dirs(self):
        """Returns highest-numbered round directory."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "round0"))
            os.makedirs(os.path.join(d, "round1"))
            os.makedirs(os.path.join(d, "round2"))
            result = self.pti._find_latest_round_dir(d)
            self.assertEqual(result, "round2")

    def test_single_round_dir(self):
        """Returns the only round directory."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "round0"))
            result = self.pti._find_latest_round_dir(d)
            self.assertEqual(result, "round0")

    def test_no_round_dirs(self):
        """Returns None when no roundN/ directories exist (flat files)."""
        with tempfile.TemporaryDirectory() as d:
            _create_file(os.path.join(d, "results.json"), "{}")
            result = self.pti._find_latest_round_dir(d)
            self.assertIsNone(result)

    def test_empty_directory(self):
        """Returns None for empty directory."""
        with tempfile.TemporaryDirectory() as d:
            result = self.pti._find_latest_round_dir(d)
            self.assertIsNone(result)

    def test_nonexistent_directory(self):
        """Returns None for non-existent path."""
        result = self.pti._find_latest_round_dir("/nonexistent/path/xyz")
        self.assertIsNone(result)

    def test_non_directory_roundN_entries(self):
        """Files named 'roundN' are ignored (only directories count)."""
        with tempfile.TemporaryDirectory() as d:
            _create_file(os.path.join(d, "round0"), "not a dir")
            os.makedirs(os.path.join(d, "round1"))
            result = self.pti._find_latest_round_dir(d)
            self.assertEqual(result, "round1")

    def test_gaps_in_round_numbers(self):
        """Handles gaps (round0, round5) correctly."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "round0"))
            os.makedirs(os.path.join(d, "round5"))
            result = self.pti._find_latest_round_dir(d)
            self.assertEqual(result, "round5")

    def test_ignores_non_round_dirs(self):
        """Ignores directories not matching 'roundN' pattern."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "round0"))
            os.makedirs(os.path.join(d, "backup"))
            os.makedirs(os.path.join(d, "round_extra"))
            result = self.pti._find_latest_round_dir(d)
            self.assertEqual(result, "round0")


# =============================================================================
# _has_results
# =============================================================================


class HasResultsTest(unittest.TestCase):
    """Tests for _has_results() lightweight check."""

    def setUp(self):
        self.pti = _make_pti()

    def test_benchmarks_with_files(self):
        """Returns True when benchmarks/ has files."""
        with tempfile.TemporaryDirectory() as d:
            _create_file(os.path.join(d, "benchmarks", "results.json"), "{}")
            self.assertTrue(self.pti._has_results(d))

    def test_tests_with_files(self):
        """Returns True when tests/ has files."""
        with tempfile.TemporaryDirectory() as d:
            _create_file(os.path.join(d, "tests", "output.txt"), "pass")
            self.assertTrue(self.pti._has_results(d))

    def test_empty_subdirs(self):
        """Returns False when result subdirs exist but are empty."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "benchmarks"))
            os.makedirs(os.path.join(d, "tests"))
            self.assertFalse(self.pti._has_results(d))

    def test_no_subdirs(self):
        """Returns False when result subdirs don't exist."""
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self.pti._has_results(d))

    def test_nested_round_dirs(self):
        """Returns True when results are in roundN/ subdirectories."""
        with tempfile.TemporaryDirectory() as d:
            _create_file(
                os.path.join(d, "benchmarks", "round0", "results.json"), "{}"
            )
            self.assertTrue(self.pti._has_results(d))


# =============================================================================
# _load_analysis_request_template
# =============================================================================


class LoadAnalysisRequestTemplateTest(unittest.TestCase):
    """Tests for _load_analysis_request_template()."""

    def test_loads_cross_ref_template(self):
        """Loads the cross_ref.jinja2 template file for last_with_cross_ref mode."""
        with tempfile.TemporaryDirectory() as templates_dir:
            var_dir = os.path.join(
                templates_dir, "analysis", "main", "_variables", "analysis_request"
            )
            _create_file(
                os.path.join(var_dir, "cross_ref.jinja2"),
                "Cross-ref template content with {{ result_path_latest }}",
            )
            pti = _make_pti(
                templates_dir=templates_dir,
                analysis_mode="last_with_cross_ref",
            )
            result = pti._load_analysis_request_template("last_with_cross_ref")
            self.assertIn("Cross-ref template content", result)
            self.assertIn("{{ result_path_latest }}", result)

    def test_loads_last_template(self):
        """Loads the last.jinja2 template file for last_round_only mode."""
        with tempfile.TemporaryDirectory() as templates_dir:
            var_dir = os.path.join(
                templates_dir, "analysis", "main", "_variables", "analysis_request"
            )
            _create_file(os.path.join(var_dir, "last.jinja2"), "Last-only content")
            pti = _make_pti(
                templates_dir=templates_dir,
                analysis_mode="last_round_only",
            )
            result = pti._load_analysis_request_template("last_round_only")
            self.assertIn("Last-only content", result)

    def test_loads_all_rounds_template(self):
        """Loads the all_rounds.jinja2 template file for all_rounds mode."""
        with tempfile.TemporaryDirectory() as templates_dir:
            var_dir = os.path.join(
                templates_dir, "analysis", "main", "_variables", "analysis_request"
            )
            _create_file(os.path.join(var_dir, "all_rounds.jinja2"), "All rounds content")
            pti = _make_pti(
                templates_dir=templates_dir,
                analysis_mode="all_rounds",
            )
            result = pti._load_analysis_request_template("all_rounds")
            self.assertIn("All rounds content", result)

    def test_fallback_when_templates_dir_is_none(self):
        """Returns generic fallback when analysis_templates_dir is None."""
        pti = _make_pti(templates_dir=None)
        result = pti._load_analysis_request_template("last_with_cross_ref")
        self.assertIn("Read and analyze", result)

    def test_fallback_when_file_missing(self):
        """Returns generic fallback when template file doesn't exist."""
        with tempfile.TemporaryDirectory() as templates_dir:
            pti = _make_pti(templates_dir=templates_dir)
            result = pti._load_analysis_request_template("last_with_cross_ref")
            self.assertIn("Read and analyze", result)


# =============================================================================
# _build_analysis_config_vars
# =============================================================================


class BuildAnalysisConfigVarsTest(unittest.TestCase):
    """Tests for _build_analysis_config_vars()."""

    def test_single_result_type_with_rounds(self):
        """Builds correct vars for benchmarks/ with roundN/ dirs."""
        with tempfile.TemporaryDirectory() as outputs_dir:
            _create_file(
                os.path.join(outputs_dir, "benchmarks", "round0", "results.json"),
                "{}",
            )
            _create_file(
                os.path.join(outputs_dir, "benchmarks", "round1", "results.json"),
                "{}",
            )
            _create_file(
                os.path.join(outputs_dir, "round1_implementation.md"),
                "impl content",
            )
            pti = _make_pti(analysis_mode="last_with_cross_ref")
            vars_ = pti._build_analysis_config_vars(outputs_dir, iteration=1)

            self.assertEqual(vars_["result_type"], "benchmarks")
            self.assertIn("benchmarks", vars_["result_path_base"])
            self.assertIn("round1", vars_["result_path_latest"])
            self.assertEqual(vars_["result_path"], vars_["result_path_latest"])
            self.assertEqual(vars_["meta_iteration"], 1)
            self.assertIn("round1_implementation.md", vars_["implementation_output_path"])
            self.assertIsInstance(vars_["analysis_request"], str)
            self.assertTrue(len(vars_["analysis_request"]) > 0)
            # First iteration has no prior history
            self.assertIn("first iteration", vars_["previous_iteration_paths"])

    def test_flat_results_no_rounds(self):
        """Handles flat file structure (no roundN/ dirs)."""
        with tempfile.TemporaryDirectory() as outputs_dir:
            _create_file(
                os.path.join(outputs_dir, "benchmarks", "results.json"), "{}"
            )
            pti = _make_pti(analysis_mode="last_round_only")
            vars_ = pti._build_analysis_config_vars(outputs_dir, iteration=1)

            self.assertEqual(vars_["result_type"], "benchmarks")
            # When no round dirs, result_path_base == result_path_latest
            self.assertEqual(vars_["result_path_base"], vars_["result_path_latest"])

    def test_multiple_result_types(self):
        """Appends extra result types to analysis_request when both benchmarks and tests have data."""
        with tempfile.TemporaryDirectory() as outputs_dir:
            _create_file(
                os.path.join(outputs_dir, "benchmarks", "round0", "bench.json"),
                "{}",
            )
            _create_file(
                os.path.join(outputs_dir, "tests", "round0", "test_output.txt"),
                "PASS",
            )
            pti = _make_pti(analysis_mode="all_rounds")
            vars_ = pti._build_analysis_config_vars(outputs_dir, iteration=1)

            self.assertEqual(vars_["result_type"], "benchmarks")
            self.assertIn("Additional result types", vars_["analysis_request"])
            self.assertIn("tests", vars_["analysis_request"])

    def test_no_results_fallback(self):
        """Falls back to 'results'/outputs_dir when no result subdirs have content."""
        with tempfile.TemporaryDirectory() as outputs_dir:
            pti = _make_pti(analysis_mode="last_with_cross_ref")
            vars_ = pti._build_analysis_config_vars(outputs_dir, iteration=1)

            self.assertEqual(vars_["result_type"], "results")
            self.assertEqual(vars_["result_path_base"], outputs_dir)
            self.assertEqual(vars_["result_path_latest"], outputs_dir)

    def test_no_implementation_files_fallback(self):
        """Uses descriptive fallback when no implementation output files exist."""
        with tempfile.TemporaryDirectory() as outputs_dir:
            _create_file(
                os.path.join(outputs_dir, "benchmarks", "results.json"), "{}"
            )
            pti = _make_pti(analysis_mode="last_round_only")
            vars_ = pti._build_analysis_config_vars(outputs_dir, iteration=1)

            self.assertIn("no implementation report", vars_["implementation_output_path"])

    def test_multiple_implementation_files_picks_latest(self):
        """Picks the highest-round implementation file."""
        with tempfile.TemporaryDirectory() as outputs_dir:
            _create_file(
                os.path.join(outputs_dir, "benchmarks", "data.json"), "{}"
            )
            _create_file(
                os.path.join(outputs_dir, "round0_implementation.md"), "v0"
            )
            _create_file(
                os.path.join(outputs_dir, "round1_implementation.md"), "v1"
            )
            _create_file(
                os.path.join(outputs_dir, "round2_implementation.md"), "v2"
            )
            pti = _make_pti()
            vars_ = pti._build_analysis_config_vars(outputs_dir, iteration=1)

            self.assertIn("round2_implementation.md", vars_["implementation_output_path"])

    def test_analysis_request_uses_correct_mode(self):
        """analysis_request content differs by mode (when templates exist)."""
        with tempfile.TemporaryDirectory() as templates_dir:
            var_dir = os.path.join(
                templates_dir, "analysis", "main", "_variables", "analysis_request"
            )
            _create_file(os.path.join(var_dir, "last.jinja2"), "LAST_MODE_MARKER")
            _create_file(os.path.join(var_dir, "cross_ref.jinja2"), "CROSS_REF_MARKER")
            _create_file(os.path.join(var_dir, "all_rounds.jinja2"), "ALL_ROUNDS_MARKER")

            with tempfile.TemporaryDirectory() as outputs_dir:
                _create_file(
                    os.path.join(outputs_dir, "benchmarks", "data.json"), "{}"
                )

                for mode, marker in [
                    ("last_round_only", "LAST_MODE_MARKER"),
                    ("last_with_cross_ref", "CROSS_REF_MARKER"),
                    ("all_rounds", "ALL_ROUNDS_MARKER"),
                ]:
                    pti = _make_pti(
                        templates_dir=templates_dir, analysis_mode=mode
                    )
                    vars_ = pti._build_analysis_config_vars(outputs_dir, iteration=1)
                    self.assertIn(
                        marker,
                        vars_["analysis_request"],
                        f"Mode '{mode}' should use template with '{marker}'",
                      )


# =============================================================================
# _build_previous_iteration_paths
# =============================================================================


class BuildPreviousIterationPathsTest(unittest.TestCase):
    """Tests for _build_previous_iteration_paths()."""

    def test_first_iteration_no_history(self):
        """Iteration 1 has no prior history."""
        with tempfile.TemporaryDirectory() as outputs_dir:
            pti = _make_pti()
            result = pti._build_previous_iteration_paths(outputs_dir, iteration=1)
            self.assertIn("first iteration", result)

    def test_iteration_2_references_root_workspace(self):
        """Iteration 2 should reference the root workspace (iteration 1)."""
        with tempfile.TemporaryDirectory() as root_ws:
            # Set up root workspace (iteration 1)
            _create_file(os.path.join(root_ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(root_ws, "outputs", "round0_implementation.md"), "impl"
            )
            # Set up iteration 2 workspace
            iter2_ws = os.path.join(root_ws, "followup_iterations", "iteration_2")
            iter2_outputs = os.path.join(iter2_ws, "outputs")
            os.makedirs(iter2_outputs, exist_ok=True)

            pti = _make_pti(resume_workspace=root_ws)
            result = pti._build_previous_iteration_paths(iter2_outputs, iteration=2)

            self.assertIn("Iteration 1", result)
            self.assertIn(root_ws, result)
            self.assertNotIn("Iteration 2", result)

    def test_iteration_3_references_root_and_iter2(self):
        """Iteration 3 should reference both root (iter 1) and followup iter 2."""
        with tempfile.TemporaryDirectory() as root_ws:
            # Set up root workspace (iteration 1)
            _create_file(os.path.join(root_ws, "outputs", "round0_plan.md"), "plan")
            # Set up iteration 2 workspace
            iter2_ws = os.path.join(root_ws, "followup_iterations", "iteration_2")
            _create_file(os.path.join(iter2_ws, "outputs", "round0_plan.md"), "plan2")
            # Set up iteration 3 workspace
            iter3_ws = os.path.join(root_ws, "followup_iterations", "iteration_3")
            iter3_outputs = os.path.join(iter3_ws, "outputs")
            os.makedirs(iter3_outputs, exist_ok=True)

            pti = _make_pti(resume_workspace=root_ws)
            result = pti._build_previous_iteration_paths(iter3_outputs, iteration=3)

            self.assertIn("Iteration 1", result)
            self.assertIn(root_ws, result)
            self.assertIn("Iteration 2", result)
            self.assertIn(iter2_ws, result)
            self.assertNotIn("Iteration 3", result)

    def test_missing_followup_dir_graceful(self):
        """Iteration 2+ without followup_iterations/ dir still references root."""
        with tempfile.TemporaryDirectory() as root_ws:
            _create_file(os.path.join(root_ws, "outputs", "round0_plan.md"), "plan")
            # No followup_iterations/ directory exists
            outputs_dir = os.path.join(root_ws, "some_path", "outputs")
            os.makedirs(outputs_dir, exist_ok=True)

            pti = _make_pti(resume_workspace=root_ws)
            result = pti._build_previous_iteration_paths(outputs_dir, iteration=2)

            self.assertIn("Iteration 1", result)

    def test_config_vars_includes_previous_iteration_paths(self):
        """_build_analysis_config_vars includes previous_iteration_paths key."""
        with tempfile.TemporaryDirectory() as outputs_dir:
            _create_file(
                os.path.join(outputs_dir, "benchmarks", "data.json"), "{}"
            )
            pti = _make_pti(analysis_mode="last_with_cross_ref")
            vars_ = pti._build_analysis_config_vars(outputs_dir, iteration=1)

            self.assertIn("previous_iteration_paths", vars_)
            self.assertIn("first iteration", vars_["previous_iteration_paths"])


# =============================================================================
# CLI Flag Parsing
# =============================================================================


@unittest.skipUnless(
    _has_server_factories(),
    "agent_foundation.server.factories not migrated yet",
)
class ParseTaskOptionsAnalysisModeTest(unittest.TestCase):
    """Tests for --analysis-mode parsing in parse_task_options()."""

    def test_valid_mode_last(self):
        """--analysis-mode last maps to last_round_only."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--analysis-mode last Do analysis"
        )
        self.assertEqual(pti_flags["analysis_mode"], "last_round_only")
        self.assertEqual(request, "Do analysis")

    def test_valid_mode_cross_ref(self):
        """--analysis-mode cross-ref maps to last_with_cross_ref."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--analysis-mode cross-ref Do analysis"
        )
        self.assertEqual(pti_flags["analysis_mode"], "last_with_cross_ref")
        self.assertEqual(request, "Do analysis")

    def test_valid_mode_all_rounds(self):
        """--analysis-mode all-rounds maps to all_rounds."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--analysis-mode all-rounds Do analysis"
        )
        self.assertEqual(pti_flags["analysis_mode"], "all_rounds")
        self.assertEqual(request, "Do analysis")

    def test_invalid_mode_stores_error(self):
        """Invalid --analysis-mode stores error in pti_flags."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--analysis-mode banana Do analysis"
        )
        self.assertNotIn("analysis_mode", pti_flags)
        self.assertIn("_analysis_mode_error", pti_flags)
        self.assertIn("banana", pti_flags["_analysis_mode_error"])
        self.assertEqual(request, "Do analysis")

    def test_combined_flags(self):
        """--analysis-mode works with other flags."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--analysis --analysis-mode last --multi-iter Do work"
        )
        self.assertEqual(pti_flags["analysis_mode"], "last_round_only")
        self.assertTrue(pti_flags["enable_analysis"])
        self.assertTrue(pti_flags["enable_multiple_iterations"])
        self.assertEqual(request, "Do work")

    def test_analysis_only_with_mode(self):
        """--analysis-only with --analysis-mode works together."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--analysis-mode cross-ref --analysis-only /path/to/ws"
        )
        self.assertEqual(pti_flags["analysis_mode"], "last_with_cross_ref")
        self.assertEqual(pti_flags["resume_workspace"], "/path/to/ws")
        self.assertTrue(pti_flags["enable_analysis"])
        self.assertFalse(pti_flags["enable_planning"])
        self.assertFalse(pti_flags["enable_implementation"])

    def test_analysis_only_disables_planning_and_implementation(self):
        """--analysis-only explicitly sets enable_planning=False and enable_implementation=False."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--analysis-only /path/to/ws"
        )
        self.assertFalse(pti_flags["enable_planning"])
        self.assertFalse(pti_flags["enable_implementation"])
        self.assertTrue(pti_flags["enable_analysis"])
        self.assertEqual(pti_flags["resume_workspace"], "/path/to/ws")


@unittest.skipUnless(
    _has_server_factories(),
    "agent_foundation.server.factories not migrated yet",
)
class ParseTaskOptionsInitialPlanTest(unittest.TestCase):
    """Tests for --initial-plan parsing in parse_task_options()."""

    def test_initial_plan_with_request(self):
        """--initial-plan <path> <request> parses both correctly."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--initial-plan /tmp/my_plan.md Review this plan"
        )
        self.assertEqual(pti_flags["initial_plan_file"], "/tmp/my_plan.md")
        self.assertEqual(request, "Review this plan")

    def test_initial_plan_without_request(self):
        """--initial-plan <path> with no request works."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--initial-plan /tmp/plan.md"
        )
        self.assertEqual(pti_flags["initial_plan_file"], "/tmp/plan.md")
        self.assertEqual(request, "")

    def test_initial_plan_with_plan_only(self):
        """--plan --initial-plan <path> <request> combines correctly."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, task_mode, pti_flags = parse_task_options(
            "--plan --initial-plan /tmp/plan.md Review and refine"
        )
        self.assertEqual(pti_flags["initial_plan_file"], "/tmp/plan.md")
        self.assertEqual(request, "Review and refine")
        from agent_foundation.server.task_types import TaskMode
        self.assertEqual(task_mode, TaskMode.PLAN_ONLY)

    def test_initial_plan_combined_with_other_flags(self):
        """--initial-plan works alongside --analysis, --multi-iter, etc."""
        from agent_foundation.server.factories import parse_task_options

        request, _, _, _, pti_flags = parse_task_options(
            "--analysis --initial-plan /tmp/plan.md --multi-iter Do work"
        )
        self.assertEqual(pti_flags["initial_plan_file"], "/tmp/plan.md")
        self.assertTrue(pti_flags["enable_analysis"])
        self.assertTrue(pti_flags["enable_multiple_iterations"])
        self.assertEqual(request, "Do work")


# =============================================================================
# Template File Integration
# =============================================================================


class TemplateFileIntegrationTest(unittest.TestCase):
    """Tests that the actual _variables/ template files exist and have correct content."""

    def _get_real_templates_dir(self) -> str | None:
        """Locate the shared prompt_templates directory."""
        candidate = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "..",
            "src", "prompt_templates",
        )
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate):
            return candidate
        # Try alternative path (deeper nesting)
        alt = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "..", "..",
            "src", "prompt_templates",
        )
        alt = os.path.normpath(alt)
        return alt if os.path.isdir(alt) else None

    def test_all_template_files_exist(self):
        """All three analysis request template files exist on disk."""
        templates_dir = self._get_real_templates_dir()
        if templates_dir is None:
            self.skipTest("Cannot locate src/prompt_templates directory")
        for filename in ("last.jinja2", "cross_ref.jinja2", "all_rounds.jinja2"):
            path = os.path.join(
                templates_dir, "analysis", "main",
                "_variables", "analysis_request", filename,
            )
            self.assertTrue(
                os.path.isfile(path),
                f"Template file missing: {path}",
            )

    def test_template_files_contain_expected_placeholders(self):
        """Template files contain {{ result_type }} and {{ result_path_latest }}."""
        templates_dir = self._get_real_templates_dir()
        if templates_dir is None:
            self.skipTest("Cannot locate src/prompt_templates directory")
        var_dir = os.path.join(
            templates_dir, "analysis", "main", "_variables", "analysis_request"
        )
        for filename in ("last.jinja2", "cross_ref.jinja2", "all_rounds.jinja2"):
            with open(os.path.join(var_dir, filename)) as f:
                content = f.read()
            self.assertIn(
                "{{ result_type }}", content,
                f"{filename} should reference {{{{ result_type }}}}",
            )
            self.assertIn(
                "{{ implementation_output_path }}", content,
                f"{filename} should reference {{{{ implementation_output_path }}}}",
            )

    def test_cross_ref_references_both_paths(self):
        """cross_ref.md references both result_path_latest and result_path_base."""
        templates_dir = self._get_real_templates_dir()
        if templates_dir is None:
            self.skipTest("Cannot locate src/prompt_templates directory")
        path = os.path.join(
            templates_dir, "analysis", "main",
            "_variables", "analysis_request", "cross_ref.jinja2",
        )
        with open(path) as f:
            content = f.read()
        self.assertIn("{{ result_path_latest }}", content)
        self.assertIn("{{ result_path_base }}", content)

    def test_all_rounds_references_base_path(self):
        """all_rounds.md references result_path_base for full traversal."""
        templates_dir = self._get_real_templates_dir()
        if templates_dir is None:
            self.skipTest("Cannot locate src/prompt_templates directory")
        path = os.path.join(
            templates_dir, "analysis", "main",
            "_variables", "analysis_request", "all_rounds.jinja2",
        )
        with open(path) as f:
            content = f.read()
        self.assertIn("{{ result_path_base }}", content)


# =============================================================================
# End-to-End: Analysis Mode with Mock Workspace
# =============================================================================


class AnalysisModeEndToEndTest(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests verifying analysis mode with mock workspaces."""

    async def test_analysis_only_cross_ref_mode(self):
        """Analysis-only with cross-ref mode passes correct config to analyzer."""
        captured_config = {}

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            captured_config.update(inference_config or {})
            return (
                '<Response>```json\n'
                '{"should_continue": false, "summary": "done", '
                '"next_iteration_request": ""}\n'
                '```</Response>'
            )

        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_file(
                os.path.join(ws, "outputs", "benchmarks", "round0", "results.json"),
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
                analysis_mode="last_with_cross_ref",
            )

            await pti._ainfer("ignored")

            # Verify key template variables were passed
            self.assertEqual(captured_config.get("result_type"), "benchmarks")
            self.assertIn("benchmarks", captured_config.get("result_path_base", ""))
            self.assertIn("round0", captured_config.get("result_path_latest", ""))
            self.assertEqual(
                captured_config.get("result_path"),
                captured_config.get("result_path_latest"),
            )
            self.assertIn(
                "round0_implementation.md",
                captured_config.get("implementation_output_path", ""),
            )
            self.assertIsNotNone(captured_config.get("analysis_request"))
            self.assertIsNotNone(captured_config.get("output_path"))

    async def test_analysis_only_no_results_skips_analyzer(self):
        """When no results exist, analyzer is not called."""
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as ws:
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)
            # No benchmarks/ or tests/ dirs → no results

            analyzer = MockInferencer()
            analyzer._ainfer = AsyncMock(return_value="should not run")

            pti = PlanThenImplementInferencer(
                planner_inferencer=MockInferencer(),
                executor_inferencer=MockInferencer(),
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
                analysis_mode="all_rounds",
            )

            await pti._ainfer("ignored")
            analyzer._ainfer.assert_not_called()

    async def test_analysis_mode_last_only_templates_dir_provided(self):
        """last_round_only mode loads template from templates_dir."""
        captured_config = {}

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            captured_config.update(inference_config or {})
            return (
                '<Response>```json\n'
                '{"should_continue": false, "summary": "done", '
                '"next_iteration_request": ""}\n'
                '```</Response>'
            )

        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as templates_dir:
            # Set up workspace
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan")
            _create_file(
                os.path.join(ws, "outputs", "round0_implementation.md"), "impl"
            )
            _create_file(
                os.path.join(ws, "outputs", "benchmarks", "round0", "data.json"),
                "{}",
            )
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            # Set up template
            var_dir = os.path.join(
                templates_dir, "analysis", "main", "_variables", "analysis_request"
            )
            _create_file(
                os.path.join(var_dir, "last.jinja2"),
                "LAST_ONLY: Read {{ result_path_latest }}",
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
                analysis_mode="last_round_only",
                analysis_templates_dir=templates_dir,
            )

            await pti._ainfer("ignored")

            # The analysis_request should come from our custom template
            analysis_req = captured_config.get("analysis_request", "")
            self.assertIn("LAST_ONLY", analysis_req)

    async def test_analysis_only_incomplete_workspace_skips_plan_and_impl(self):
        """On incomplete workspace (plan done, no impl), planner/executor must NOT run."""
        from unittest.mock import AsyncMock

        captured_config = {}

        async def mock_analyzer_ainfer(inp, inference_config=None, **kw):
            captured_config.update(inference_config or {})
            return (
                '<Response>```json\n'
                '{"should_continue": false, "summary": "done", '
                '"next_iteration_request": ""}\n'
                '```</Response>'
            )

        with tempfile.TemporaryDirectory() as ws:
            # Incomplete workspace: plan done, but NO implementation file
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan text")
            # Deliberately NO round*_implementation.md
            _create_file(
                os.path.join(ws, "outputs", "benchmarks", "round0", "results.json"),
                '{"metric": 42}',
            )
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            planner = MockInferencer()
            executor = MockInferencer()
            analyzer = MockInferencer()

            planner._ainfer = AsyncMock(return_value="should not be called")
            executor._ainfer = AsyncMock(return_value="should not be called")
            analyzer._ainfer = mock_analyzer_ainfer

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
                analysis_mode="last_with_cross_ref",
            )

            result = await pti._ainfer("ignored input")

            # Core assertions: planner and executor NEVER called
            planner._ainfer.assert_not_called()
            executor._ainfer.assert_not_called()

            # Analyzer WAS called (benchmark results exist)
            self.assertIn("result_type", captured_config)
            self.assertEqual(captured_config["result_type"], "benchmarks")

            # Analysis summary was saved
            summary_path = os.path.join(ws, "results", "analysis_summary.json")
            self.assertTrue(os.path.isfile(summary_path))

    async def test_analysis_only_incomplete_workspace_no_results_skips_all(self):
        """On incomplete workspace with no results, all phases are skipped gracefully."""
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as ws:
            # Incomplete workspace: plan done, no impl, no benchmark results
            _create_file(os.path.join(ws, "request.txt"), "original request")
            _create_file(os.path.join(ws, "outputs", "round0_plan.md"), "plan text")
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            planner = MockInferencer()
            executor = MockInferencer()
            analyzer = MockInferencer()

            planner._ainfer = AsyncMock(return_value="should not be called")
            executor._ainfer = AsyncMock(return_value="should not be called")
            analyzer._ainfer = AsyncMock(return_value="should not be called")

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
                analysis_mode="all_rounds",
            )

            result = await pti._ainfer("ignored input")

            planner._ainfer.assert_not_called()
            executor._ainfer.assert_not_called()
            analyzer._ainfer.assert_not_called()

    async def test_analysis_only_empty_workspace_skips_all(self):
        """On empty workspace (nothing done), all phases are skipped gracefully."""
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as ws:
            # Empty workspace: no plan, no impl, no results
            _create_file(os.path.join(ws, "request.txt"), "original request")
            os.makedirs(os.path.join(ws, "outputs"), exist_ok=True)
            os.makedirs(os.path.join(ws, "results"), exist_ok=True)

            planner = MockInferencer()
            executor = MockInferencer()
            analyzer = MockInferencer()

            planner._ainfer = AsyncMock(return_value="should not be called")
            executor._ainfer = AsyncMock(return_value="should not be called")
            analyzer._ainfer = AsyncMock(return_value="should not be called")

            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                analyzer_inferencer=analyzer,
                enable_planning=False,
                enable_implementation=False,
                enable_analysis=True,
                resume_workspace=ws,
                analysis_mode="last_round_only",
            )

            result = await pti._ainfer("ignored input")

            planner._ainfer.assert_not_called()
            executor._ainfer.assert_not_called()
            analyzer._ainfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()

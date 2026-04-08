

"""PlanThenImplementInferencer — two-phase flow: plan first, then implement.

Chains a planner and executor inferencer sequentially. The planner's output
is fed as input to the executor. Optionally pauses for human approval between
phases using an InteractiveBase instance.

Supports multi-iteration refinement loops with analysis, follow-up workspaces,
and resume from interruption. All new features are opt-in via flags that
default to preserving exact current behavior (two-way door).

Both planner and executor can be any InferencerBase subclass (e.g., DualInferencer
for consensus-based planning/execution, or a simple API inferencer).

Inherits from Workflow for native checkpoint/resume with loop support
and automatic recursive resume of child DualInferencers.
"""

import copy
import glob
import json
import os
import re
import shutil
from datetime import datetime, timezone
from enum import Flag, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.common import (
    DualInferencerResponse,
    InferencerResponse,
    ReflectionStyles,
    ResponseSelectors,
)
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)
from agent_foundation.ui.interactive_base import (
    InteractionFlags,
    InteractiveBase,
)
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.serializable import SerializationMode
from rich_python_utils.common_objects.workflow.common.exceptions import (
    WorkflowAborted,
)
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import (
    ResultPassDownMode,
)
from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
    StepResultSaveOptions,
)
from rich_python_utils.common_objects.workflow.common.step_wrapper import StepWrapper
from rich_python_utils.common_objects.workflow.workflow import Workflow
from rich_python_utils.io_utils.artifact import artifact_field, artifact_type


# region Data Structures


@artifact_field("plan_output", type="md", alias="plan", group="outputs")
@artifact_field("executor_output", type="md", alias="implementation", group="outputs")
@artifact_field("analysis_output", type="md", alias="analysis", group="analysis")
@attrs
class MetaIterationRecord:
    """Record of a single meta-iteration (plan + implement + analysis cycle).

    Attributes:
        iteration: 1-based iteration number.
        workspace_path: Path to this iteration's workspace directory.
        plan_response: Full planner response object.
        plan_output: Extracted plan text.
        executor_response: Full executor response object.
        executor_output: Extracted executor output text.
        plan_file_path: Resolved path to the plan file on disk.
        plan_approved: None if not required, True/False for approval result.
        test_results_found: Whether test/benchmark results were found.
        analysis_response: Full analyzer response object.
        analysis_output: Extracted analysis text.
        analysis_doc_path: Path to the analysis document on disk.
        should_continue: Whether the analysis recommended continuing.
        resumed_from_phase: Phase name if this iteration was resumed.
        error: Error message if the iteration failed.
    """

    iteration: int = attrib()
    workspace_path: Optional[str] = attrib(default=None)
    plan_response: Any = attrib(default=None)
    plan_output: str = attrib(default="")
    executor_response: Any = attrib(default=None)
    executor_output: str = attrib(default="")
    plan_file_path: Optional[str] = attrib(default=None)
    plan_approved: Optional[bool] = attrib(default=None)
    test_results_found: bool = attrib(default=False)
    analysis_response: Optional[Any] = attrib(default=None)
    analysis_output: Optional[str] = attrib(default=None)
    analysis_doc_path: Optional[str] = attrib(default=None)
    should_continue: bool = attrib(default=False)
    resumed_from_phase: Optional[str] = attrib(default=None)
    error: Optional[str] = attrib(default=None)


@attrs
class WorkspaceState:
    """Detected state of a workspace directory for resume support.

    Attributes:
        workspace_path: Path to the workspace directory.
        iteration: Iteration number this workspace corresponds to.
        plan_done: Whether plan output files exist.
        plan_partial: Whether plan files exist but .plan_completed marker is missing.
        plan_file_path: Path to the highest-round plan file.
        impl_done: Whether implementation output files exist AND confirmed complete.
        impl_partial: Whether impl files exist but .impl_completed marker is missing.
        analysis_done: Whether analysis_summary.json exists.
        analysis_doc_path: Path to the analysis document.
        has_test_results: Whether test result files exist.
        has_benchmark_results: Whether benchmark result files exist.
    """

    workspace_path: str = attrib()
    iteration: int = attrib()
    plan_done: bool = attrib(default=False)
    plan_partial: bool = attrib(default=False)
    plan_file_path: Optional[str] = attrib(default=None)
    impl_done: bool = attrib(default=False)
    impl_partial: bool = attrib(default=False)
    analysis_done: bool = attrib(default=False)
    analysis_doc_path: Optional[str] = attrib(default=None)
    has_test_results: bool = attrib(default=False)
    has_benchmark_results: bool = attrib(default=False)


# endregion


# region Default Templates

_DEFAULT_ITERATION_HANDOFF_TEMPLATE = """\
This is iteration {iteration} (of up to {max_iterations}) of a multi-iteration \
refinement process.

## Original Request

{original_request}

## Previous Iteration Analysis

We completed iteration {prev_iteration} which included planning and implementation.
{analysis_reference}

## Focus for This Iteration

Based on the analysis of iteration {prev_iteration}, the following areas need attention:

{next_iteration_request}

Please read the analysis document for full context before creating your plan. \
Build upon what worked in the previous iteration and address the identified issues.\
"""

# endregion


# Canonical mode constants and mappings — single source of truth
VALID_ANALYSIS_MODES = {"last_round_only", "last_with_cross_ref", "all_rounds"}
ANALYSIS_MODE_CLI_MAP = {
    "last": "last_round_only",
    "cross-ref": "last_with_cross_ref",
    "all-rounds": "all_rounds",
}
ANALYSIS_MODE_FILE_MAP = {
    "last_round_only": "last",
    "last_with_cross_ref": "cross_ref",
    "all_rounds": "all_rounds",
}


@attrs
class PlanThenImplementResponse(InferencerResponse):
    """Response from PlanThenImplementInferencer.

    Extends InferencerResponse with plan/executor metadata.
    base_response holds the executor's final output text (or plan text if rejected).

    Attributes:
        plan_response: The full planner response object.
        plan_output: The extracted plan text that was sent to the executor.
        executor_output: The full executor response object.
        plan_approved: None if approval not required, True if approved, False if rejected.
        planner_phase: Label for the planning phase.
        executor_phase: Label for the execution phase.
        iteration_history: List of MetaIterationRecord for multi-iteration runs.
        total_meta_iterations: Number of meta-iterations completed.
        meta_iterations_exhausted: Whether the max iteration count was reached.
    """

    plan_response: Any = attrib(default=None)
    plan_output: str = attrib(default="")
    executor_output: Any = attrib(default=None)
    plan_approved: Optional[bool] = attrib(default=None)
    plan_file_path: Optional[str] = attrib(default=None)
    planner_phase: str = attrib(default="plan")
    executor_phase: str = attrib(default="implementation")
    iteration_history: List[MetaIterationRecord] = attrib(factory=list)
    total_meta_iterations: int = attrib(default=1)
    meta_iterations_exhausted: bool = attrib(default=False)


# Phase-to-step-index mapping for backward-compat checkpoint synthesis
_PHASE_TO_STEP_INDEX = {
    "planning": 0,
    "implementation": 2,  # skip approval — plan already approved
    "analysis": 3,
    "new_iteration": 0,  # loop back to plan with incremented iteration
}


class PTIOutputMode(Flag):
    """Which child outputs PTI surfaces as its own final deliverables."""

    PLAN = auto()
    IMPLEMENTATION = auto()
    ANALYSIS = auto()

    PLAN_AND_IMPLEMENTATION = PLAN | IMPLEMENTATION
    ALL = PLAN | IMPLEMENTATION | ANALYSIS


_OUTPUT_MODE_MAP = {
    PTIOutputMode.PLAN: ("planner", "plan.md"),
    PTIOutputMode.IMPLEMENTATION: ("executor", "implementation.md"),
    PTIOutputMode.ANALYSIS: ("analyzer", "analysis.md"),
}

# Attr name → (short workspace name, default output_path)
_CHILD_DEFAULTS = {
    "planner_inferencer": ("planner", "plan.md"),
    "executor_inferencer": ("executor", "implementation.md"),
    "analyzer_inferencer": ("analyzer", "analysis.md"),
}

# Attr name → short workspace name (for _setup_child_workflows)
_CHILD_NAME_MAP = {k: v[0] for k, v in _CHILD_DEFAULTS.items()}


@artifact_type(Workflow, type="json", group="workflows")
@attrs(slots=False)
class PlanThenImplementInferencer(InferencerBase, Workflow):
    """Two-phase inferencer: plan first, then implement.

    Chains a planner and executor inferencer sequentially. The planner's output
    (base_response) is combined with the original input and fed to the executor.
    Optionally pauses for human approval between phases.

    Inherits from Workflow for native checkpoint/resume with loop support
    and automatic recursive resume of child DualInferencers via
    ``@artifact_type(Workflow, ...)`` + ``_setup_child_workflows``.

    Supports multi-iteration refinement loops with an analyzer phase that
    evaluates results and decides whether to continue. All new features are
    opt-in via flags that default to preserving exact current behavior.

    Usage:
        # Simple chaining (no approval):
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
        )
        result = pti("Design and implement a REST API")

        # With human approval (terminal):
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
            interactive=TerminalInteractive(),
        )

        # Multi-iteration with analysis:
        pti = PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
            analyzer_inferencer=analyzer_dual,
            enable_analysis=True,
            enable_multiple_iterations=True,
            workspace_path="/path/to/workspace",
        )

        # Async (recommended):
        async with PlanThenImplementInferencer(...) as pti:
            result = await pti.ainfer("Design and implement a REST API")

    Attributes:
        planner_inferencer: Inferencer for the planning phase.
        executor_inferencer: Inferencer for the execution/implementation phase.
        executor_prompt_builder: Optional callable (plan_str, input_str) -> str
            for custom executor input construction.
        interactive: InteractiveBase instance for human approval gate.
        planner_phase: Label for the planning phase (for logging).
        executor_phase: Label for the execution phase (for logging).
        plan_config_key: Key in inference_config for planner-specific config.
        implement_config_key: Key in inference_config for executor-specific config.
        planner_outputs_plan_to_file: When True, scans the output directory for
            the plan file with the largest round index.
        enable_planning: Whether to run the planning phase.
        enable_implementation: Whether to run the implementation phase.
        enable_analysis: Whether to run the analysis phase.
        enable_multiple_iterations: Whether to run multiple plan-implement-analyze cycles.
        max_meta_iterations: Maximum number of meta-iterations.
        analyzer_inferencer: Inferencer for the analysis phase.
        analysis_config_key: Key in inference_config for analyzer-specific config.
        analyzer_outputs_to_file: Whether the analyzer writes output to a file.
        results_subdirs: Subdirectories under outputs/ to scan for results.
        reset_sessions_per_meta_iteration: Whether to reset sub-inferencer sessions.
        approve_all_iterations: Whether to require approval for every iteration.
        workspace_path: Base workspace directory for multi-iteration runs.
        resume_workspace: Workspace to resume from (enables resume detection).
        iteration_handoff_template: Custom template for iteration handoff text.
    """

    planner_inferencer: InferencerBase = attrib(default=None)
    executor_inferencer: InferencerBase = attrib(default=None)

    # Optional custom executor prompt builder: (plan_str, input_str) -> str
    executor_prompt_builder: Optional[Callable] = attrib(default=None)

    # Human approval gate
    interactive: Optional[InteractiveBase] = attrib(default=None)

    # Phase labels
    planner_phase: str = attrib(default="plan")
    executor_phase: str = attrib(default="implementation")

    # inference_config keys for phase-specific config splitting
    plan_config_key: str = attrib(default="plan_config")
    implement_config_key: str = attrib(default="implement_config")

    # When True and planner writes output to a file (via output_path in
    # inference_config), the executor prompt references the plan file path
    # so the implementer can read the full plan. When False, the plan's
    # base_response text is embedded inline (original behavior).
    planner_outputs_plan_to_file: bool = attrib(default=True)

    # Phase enable flags (defaults preserve current behavior)
    enable_planning: bool = attrib(default=True)
    enable_implementation: bool = attrib(default=True)
    enable_analysis: bool = attrib(default=False)
    enable_multiple_iterations: bool = attrib(default=False)

    # Multi-iteration config
    max_meta_iterations: int = attrib(default=3)
    analyzer_inferencer: Optional[InferencerBase] = attrib(default=None)
    analysis_config_key: str = attrib(default="analysis_config")
    analyzer_outputs_to_file: bool = attrib(default=True)
    results_subdirs: Tuple[str, ...] = attrib(default=("benchmarks", "tests"))
    reset_sessions_per_meta_iteration: bool = attrib(default=True)
    approve_all_iterations: bool = attrib(default=False)

    # Workspace config
    workspace_path: Optional[str] = attrib(default=None)
    resume_workspace: Optional[str] = attrib(default=None)
    iteration_handoff_template: Optional[str] = attrib(default=None)

    # Declarative output selection — controls which child outputs
    # _finalize_outputs() copies to workspace root outputs/.
    output_mode: PTIOutputMode = attrib(default=PTIOutputMode.IMPLEMENTATION)

    # Analysis mode config
    analysis_mode: str = attrib(default="last_with_cross_ref")
    analysis_templates_dir: Optional[str] = attrib(default=None)

    # Initial plan override: skip plan proposal, start with plan review
    initial_plan_file: Optional[str] = attrib(default=None)

    # Enhanced checkpoint flags (all opt-in, backward compatible)
    enable_checkpoint_plan_review: bool = attrib(default=False)
    enable_checkpoint_pre_implementation: bool = attrib(default=False)
    enable_checkpoint_post_implementation: bool = attrib(default=False)
    enable_checkpoint_analysis_review: bool = attrib(default=False)
    enable_checkpoint_iteration_handoff: bool = attrib(default=False)

    # --- Workflow params suppressed (not user-facing) ---
    result_pass_down_mode = attrib(default=ResultPassDownMode.NoPassDown, init=False)
    unpack_single_result = attrib(default=False, init=False)
    ignore_stop_flag_from_saved_results = attrib(default=True, init=False)
    auto_mode = attrib(default=SerializationMode.PREFER_CLEAR_TEXT, init=False)
    checkpoint_mode = attrib(default="jsonfy", init=False)

    # Internal state (not user-facing)
    _next_iteration_input: Optional[str] = attrib(default=None, init=False)
    _partial_iteration_history: Optional[List[MetaIterationRecord]] = attrib(
        default=None, init=False
    )
    _current_base_workspace: Optional[str] = attrib(default=None, init=False)
    _current_iteration_workspace: Optional[str] = attrib(default=None, init=False)
    _current_inference_config: Optional[dict] = attrib(default=None, init=False)
    _current_inference_args: Optional[dict] = attrib(default=None, init=False)
    _pending_pti_state: Optional[dict] = attrib(default=None, init=False)

    def __attrs_post_init__(self):
        super(PlanThenImplementInferencer, self).__attrs_post_init__()

        if (not self.response_types) or self.response_types == (str,):
            self.response_types = (str, InferencerResponse, PlanThenImplementResponse)

        # Validation: enable_multiple_iterations implies enable_analysis
        if self.enable_multiple_iterations:
            self.enable_analysis = True

        # Validation: analysis_mode must be one of the known modes
        if self.analysis_mode not in VALID_ANALYSIS_MODES:
            raise ValueError(
                f"Invalid analysis_mode '{self.analysis_mode}'. "
                f"Must be one of: {VALID_ANALYSIS_MODES}"
            )

        # Validation: analysis requires analyzer_inferencer
        if self.enable_analysis and self.analyzer_inferencer is None:
            raise ValueError(
                "enable_analysis=True requires analyzer_inferencer to be set"
            )

        # Validation: analysis or multi-iteration requires workspace
        if (self.enable_analysis or self.enable_multiple_iterations) and (
            self.workspace_path is None and self.resume_workspace is None
        ):
            raise ValueError(
                "enable_analysis=True or enable_multiple_iterations=True "
                "requires workspace_path or resume_workspace to be set"
            )

        # Set parent debuggable for nested inferencers (deduplicate by identity)
        seen_ids = set()
        for inf in (
            self.planner_inferencer,
            self.executor_inferencer,
            self.analyzer_inferencer,
        ):
            if (
                inf is not None
                and isinstance(inf, Debuggable)
                and id(inf) not in seen_ids
            ):
                seen_ids.add(id(inf))
                inf.set_parent_debuggable(self)

    # --- Block direct Workflow.run()/arun() --- use ainfer() instead ---

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "Use ainfer() or _infer() instead of run(). "
            "PTI manages Workflow._arun() internally."
        )

    async def arun(self, *args, **kwargs):
        raise NotImplementedError(
            "Use ainfer() instead of arun(). PTI manages Workflow._arun() internally."
        )

    # region Static/Instance Helpers

    def _build_executor_input(
        self,
        inference_input: str,
        plan_str: str,
        plan_file_path: Optional[str] = None,
    ) -> str:
        """Build the executor's input from the original request and plan.

        When a custom executor_prompt_builder callable is provided, delegates
        to it. Otherwise, builds a simple string that either references the
        plan file path (so the executor can read the full plan) or embeds the
        plan text inline.

        If ``_step_was_previously_attempted`` is set (by the step-in-progress
        marker system), appends a resume context warning so the executor
        checks for partial changes from the interrupted attempt.

        Args:
            inference_input: The original user request.
            plan_str: The extracted plan text (concise base_response).
            plan_file_path: Optional resolved path to the final plan file.

        Returns:
            Executor input string.
        """
        if self.executor_prompt_builder is not None:
            return self.executor_prompt_builder(plan_str, inference_input)

        if plan_file_path:
            base = (
                f"{inference_input}\n\n"
                f"## Approved Plan\n\n"
                f"The full approved plan is at: `{plan_file_path}`\n"
                f"Read that file to understand the complete plan details."
            )
        else:
            base = f"{inference_input}\n\n## Approved Plan\n\n{plan_str}"

        if getattr(self, "_step_was_previously_attempted", False):
            base += (
                "\n\n## \u26a0\ufe0f Resume Context\n\n"
                "A previous implementation attempt was interrupted before "
                "completion. The target repository may contain partial "
                "changes from that attempt.\n\n"
                "**IMPORTANT**: Before implementing, run `sl status` "
                "(or `git status`) in the target repository to check for "
                "existing modifications. If you find partial work that "
                "aligns with the plan, build upon it rather than starting "
                "from scratch. If you find conflicting or broken changes, "
                "revert them first.\n"
            )

        return base

    @staticmethod
    def _resolve_plan_file_path(plan_inference_config: dict) -> Optional[str]:
        """Resolve the finalized plan file path by scanning the output directory.

        Derives a glob pattern from plan_inference_config["output_path"] by
        replacing {{ round_index }} with *, then finds the file with the
        largest round index.

        Args:
            plan_inference_config: The planner's inference_config dict
                containing "output_path" (a Jinja2 template string).

        Returns:
            Resolved file path string, or None if output_path not in config
            or no matching files found.
        """
        output_path_template = plan_inference_config.get("output_path", "")
        if not output_path_template:
            return None

        glob_pattern = output_path_template.replace("{{ round_index }}", "*")
        glob_pattern = glob_pattern.replace("{{round_index}}", "*")

        if glob_pattern == output_path_template:
            return (
                output_path_template if os.path.isfile(output_path_template) else None
            )

        matching_files = glob.glob(glob_pattern)
        if not matching_files:
            return None

        def extract_round_index(path: str) -> int:
            match = re.search(r"round(\d+)", os.path.basename(path))
            return int(match.group(1)) if match else -1

        return max(matching_files, key=extract_round_index)

    @staticmethod
    def _extract_response_text(result: Any) -> str:
        """Extract text from an inferencer result.

        Handles DualInferencerResponse (uses base_response),
        InferencerResponse (uses select_response()), and
        other types (uses str()).

        Args:
            result: An inferencer response of any type.

        Returns:
            Extracted text string.
        """
        if isinstance(result, DualInferencerResponse):
            return str(result.base_response)
        elif isinstance(result, InferencerResponse):
            return str(result.select_response())
        else:
            return str(result)

    # endregion

    # region Workspace Helpers (3a-3c)

    @staticmethod
    def _get_iteration_workspace(base_workspace: str, iteration: int) -> str:
        """Get the workspace directory for a given iteration.

        Iteration 1 uses the base workspace. Subsequent iterations use
        followup_iterations/iteration_N/ subdirectories.
        """
        if iteration == 1:
            return base_workspace
        return os.path.join(
            base_workspace, "followup_iterations", f"iteration_{iteration}"
        )

    @staticmethod
    def _setup_iteration_workspace(
        workspace_path: str, iteration: int, request_text: str
    ) -> None:
        """Create workspace directory structure and write request.txt."""
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        ws = InferencerWorkspace(root=workspace_path)
        ws.ensure_dirs("results", "analysis", "_runtime")
        with open(ws.artifact_path("request.txt"), "w") as f:
            f.write(request_text)

    def _build_iteration_config(
        self, inference_config: dict, iter_workspace: str, iteration: int
    ) -> dict:
        """Build a per-iteration inference_config with workspace-local output paths.

        Deep-copies the original config and rewrites output_path values to
        point to the iteration's workspace directory.  Skips rewriting for
        children that already have a workspace assigned (full composition mode).
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        iter_config = copy.deepcopy(inference_config)
        ws = InferencerWorkspace(root=iter_workspace)

        # Only rewrite output_path for children without workspace
        planner_has_ws = getattr(self.planner_inferencer, "_workspace", None) is not None
        executor_has_ws = getattr(self.executor_inferencer, "_workspace", None) is not None

        for config_key, child_has_ws in (
            (self.plan_config_key, planner_has_ws),
            (self.implement_config_key, executor_has_ws),
        ):
            if child_has_ws:
                continue
            sub_config = iter_config.get(config_key)
            if isinstance(sub_config, dict) and "output_path" in sub_config:
                basename = os.path.basename(sub_config["output_path"])
                sub_config["output_path"] = ws.output_path(basename)

        # Analyzer: skip if it has workspace; otherwise write to analysis/
        analyzer_has_ws = (
            self.analyzer_inferencer is not None
            and getattr(self.analyzer_inferencer, "_workspace", None) is not None
        )
        if not analyzer_has_ws:
            analysis_sub = iter_config.get(self.analysis_config_key)
            if not isinstance(analysis_sub, dict):
                analysis_sub = {}
                iter_config[self.analysis_config_key] = analysis_sub
            analysis_sub["output_path"] = ws.analysis_path(
                f"iteration_{iteration}_analysis.md"
            )
            os.makedirs(os.path.dirname(analysis_sub["output_path"]), exist_ok=True)

        return iter_config

    # endregion

    # region Results Collection and Analysis (3d-3i)

    def _collect_results(self, outputs_dir: str) -> Dict[str, Any]:
        """Scan outputs directory for test/benchmark result files.

        Returns a dict with one key per results_subdir, mapping filenames
        to their content, plus a 'has_results' boolean.
        """
        results: Dict[str, Any] = {"has_results": False}
        for subdir in self.results_subdirs:
            subdir_path = os.path.join(outputs_dir, subdir)
            subdir_results: Dict[str, str] = {}
            if os.path.isdir(subdir_path):
                for root, _, files in os.walk(subdir_path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath) as f:
                                content = f.read()
                            rel_path = os.path.relpath(fpath, subdir_path)
                            subdir_results[rel_path] = content
                        except Exception:
                            pass
            results[subdir] = subdir_results
            if subdir_results:
                results["has_results"] = True
        return results

    def _has_results(self, outputs_dir: str) -> bool:
        """Lightweight check for whether any result files exist.

        Unlike ``_collect_results()``, this does NOT read file contents —
        it only checks for existence of at least one file in any of the
        ``results_subdirs``.
        """
        for subdir in self.results_subdirs:
            subdir_path = os.path.join(outputs_dir, subdir)
            if os.path.isdir(subdir_path):
                try:
                    if any(True for _ in os.scandir(subdir_path)):
                        return True
                except OSError:
                    pass
        return False

    def _find_latest_round_dir(self, result_path_base: str) -> Optional[str]:
        """Find the highest roundN/ subdirectory under result_path_base.

        Returns the directory name (e.g. "round2") or None if no round
        subdirectories exist (flat file structure).
        """
        if not os.path.isdir(result_path_base):
            return None
        round_dirs = []
        for entry in os.listdir(result_path_base):
            match = re.match(r"^round(\d+)$", entry)
            if match and os.path.isdir(os.path.join(result_path_base, entry)):
                round_dirs.append((int(match.group(1)), entry))
        if not round_dirs:
            return None
        round_dirs.sort()
        return round_dirs[-1][1]

    def _load_analysis_request_template(self, mode: str) -> str:
        """Read the analysis request template for the given mode.

        Reads the appropriate ``_variables/analysis_request/{mode}.md`` file
        from ``self.analysis_templates_dir``.  Falls back to a generic string
        when the template directory or file is unavailable.
        """
        if self.analysis_templates_dir is None:
            self.log_info(
                "analysis_templates_dir is None — using generic analysis request",
                "AnalysisTemplateFallback",
            )
            return "Read and analyze the results. Evaluate correctness and performance."
        filename = ANALYSIS_MODE_FILE_MAP.get(mode, "cross_ref")
        templates_dir: str = self.analysis_templates_dir  # type: ignore[assignment]
        var_path = os.path.join(
            templates_dir,
            "analysis",
            "main",
            "_variables",
            "analysis_request",
            f"{filename}.jinja2",
        )
        if os.path.isfile(var_path):
            with open(var_path) as f:
                return f.read()
        self.log_info(
            f"Analysis request template not found at {var_path} — using generic fallback",
            "AnalysisTemplateFallback",
        )
        return "Read and analyze the results. Evaluate correctness and performance."

    def _build_analysis_config_vars(
        self, outputs_dir: str, iteration: int
    ) -> Dict[str, Any]:
        """Build template variables for the analysis prompt.

        Returns a dict of variables to merge into ``inference_config``.
        Handles multiple result types (both benchmarks and tests if both
        have data), discovers the latest implementation output file, and
        builds ``previous_iteration_paths`` for cross-iteration context.
        """
        result_entries: list = []
        for subdir in self.results_subdirs:
            subdir_path = os.path.join(outputs_dir, subdir)
            if os.path.isdir(subdir_path) and os.listdir(subdir_path):
                latest = self._find_latest_round_dir(subdir_path)
                latest_path = (
                    os.path.join(subdir_path, latest) if latest else subdir_path
                )
                result_entries.append((subdir, subdir_path, latest_path))

        if result_entries:
            result_type, result_path_base, result_path_latest = result_entries[0]
        else:
            result_type, result_path_base, result_path_latest = (
                "results",
                outputs_dir,
                outputs_dir,
            )

        # Discover the latest plan file (round*_plan.md)
        plan_pattern = os.path.join(outputs_dir, "round*_plan.md")
        plan_files = glob.glob(plan_pattern)

        def _extract_round_num(path: str) -> int:
            match = re.search(r"round(\d+)", os.path.basename(path))
            return int(match.group(1)) if match else -1

        if plan_files:
            implementation_plan = max(plan_files, key=_extract_round_num)
        else:
            implementation_plan = "(no implementation plan found)"
            self.log_info(
                f"No plan files matching {plan_pattern}",
                "AnalysisWarning",
            )

        # Discover the latest implementation output file (round*_implementation.md)
        impl_pattern = os.path.join(outputs_dir, "round*_implementation.md")
        impl_files = glob.glob(impl_pattern)
        if impl_files:
            implementation_output_path = max(impl_files, key=_extract_round_num)
        else:
            implementation_output_path = "(no implementation report found)"
            self.log_info(
                f"No implementation output files matching {impl_pattern}",
                "AnalysisWarning",
            )

        # Build previous iteration paths for cross-iteration analysis.
        previous_iteration_paths = self._build_previous_iteration_paths(
            outputs_dir, iteration
        )

        analysis_request = self._load_analysis_request_template(self.analysis_mode)

        config_vars: Dict[str, Any] = {
            "result_type": result_type,
            "result_path": result_path_latest,
            "result_path_base": result_path_base,
            "result_path_latest": result_path_latest,
            "implementation_plan": implementation_plan,
            "implementation_output_path": implementation_output_path,
            "meta_iteration": iteration,
            "analysis_request": analysis_request,
            "previous_iteration_paths": previous_iteration_paths,
        }

        if len(result_entries) > 1:
            extra = "\n\nAdditional result types available:\n"
            for rtype, rbase, rlatest in result_entries[1:]:
                extra += f"- {rtype}: latest at {rlatest}, all rounds at {rbase}\n"
            config_vars["analysis_request"] = analysis_request + extra

        return config_vars

    def _build_previous_iteration_paths(self, outputs_dir: str, iteration: int) -> str:
        """Build a human-readable list of previous iteration workspace paths."""
        if iteration <= 1:
            return "(first iteration — no prior history)"

        iter_workspace = os.path.dirname(outputs_dir)
        root_ws = self.resume_workspace or self.workspace_path
        if root_ws is None:
            root_ws = os.path.dirname(os.path.dirname(iter_workspace))

        prev_paths: list[str] = []

        iter1_outputs = os.path.join(root_ws, "outputs")
        if os.path.isdir(iter1_outputs):
            prev_paths.append(f"- Iteration 1: {root_ws}")

        followup_dir = os.path.join(root_ws, "followup_iterations")
        if os.path.isdir(followup_dir):
            for i in range(2, iteration):
                iter_ws = os.path.join(followup_dir, f"iteration_{i}")
                if os.path.isdir(iter_ws):
                    prev_paths.append(f"- Iteration {i}: {iter_ws}")

        return "\n".join(prev_paths) if prev_paths else "(no prior iterations found)"

    @staticmethod
    def _build_analysis_input(
        original_request: str,
        executor_output: str,
        collected_results: dict,
        iteration: int,
    ) -> str:
        """Build the analysis prompt from request, output, and results.

        .. deprecated::
            Superseded by the template-driven approach using
            ``_build_analysis_config_vars()`` + ``_load_analysis_request_template()``.
            Kept for backward compatibility.
        """
        parts = [
            f"# Analysis of Iteration {iteration}\n",
            f"## Original Request\n\n{original_request}\n",
            f"## Implementation Output\n\n{executor_output}\n",
        ]
        for subdir, files in collected_results.items():
            if subdir == "has_results" or not isinstance(files, dict) or not files:
                continue
            parts.append(f"\n## {subdir.title()} Results\n")
            for fname, content in files.items():
                parts.append(f"\n### {fname}\n\n```\n{content}\n```\n")

        parts.append(
            "\n## Task\n\n"
            "Analyze the implementation output and test/benchmark results above. "
            "Determine if the implementation meets the requirements or if further "
            "iterations are needed.\n\n"
            "Respond with a JSON object inside a ```json``` code block:\n"
            "```json\n"
            "{\n"
            '  "should_continue": true/false,\n'
            '  "summary": "Brief analysis summary",\n'
            '  "next_iteration_request": "What to focus on next (if continuing)"\n'
            "}\n"
            "```\n"
        )
        return "\n".join(parts)

    def _parse_analysis_response(self, analysis_text: str) -> Tuple[bool, str]:
        """Parse analysis response JSON to extract continuation decision."""
        from agent_foundation.common.response_parsers import (
            extract_delimited,
        )

        try:
            cleaned = extract_delimited(analysis_text)
        except Exception:
            cleaned = analysis_text

        json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return (
                    parsed.get("should_continue", False),
                    parsed.get("next_iteration_request", ""),
                )
            except (json.JSONDecodeError, AttributeError):
                pass

        try:
            parsed = json.loads(cleaned)
            return (
                parsed.get("should_continue", False),
                parsed.get("next_iteration_request", ""),
            )
        except (json.JSONDecodeError, AttributeError):
            pass

        self.log_info(
            "Could not parse analysis response as JSON. Stopping iteration.",
            "AnalysisParseError",
        )
        return (False, "")

    def _build_iteration_handoff(
        self,
        original_request: str,
        iteration: int,
        max_iterations: int,
        next_iteration_request: str,
        analysis_doc_path: Optional[str],
        analysis_summary: str = "",
    ) -> str:
        """Build handoff text for the next iteration."""
        template = (
            self.iteration_handoff_template or _DEFAULT_ITERATION_HANDOFF_TEMPLATE
        )

        if analysis_doc_path:
            analysis_reference = (
                f"The full analysis document is at: `{analysis_doc_path}`\n"
                f"Read that file for the complete analysis."
            )
        else:
            analysis_reference = (
                f"Analysis summary:\n{analysis_summary}" if analysis_summary else ""
            )

        return template.format(
            iteration=iteration,
            max_iterations=max_iterations,
            original_request=original_request,
            prev_iteration=iteration - 1,
            analysis_reference=analysis_reference,
            next_iteration_request=next_iteration_request,
        )

    @staticmethod
    def _resolve_analysis_file_path(analysis_config: dict) -> Optional[str]:
        """Resolve the analysis output file path."""
        output_path = analysis_config.get("output_path", "")
        if not output_path:
            return None
        return output_path if os.path.isfile(output_path) else None

    @staticmethod
    def _save_analysis_summary(
        workspace_or_results_dir: str,
        should_continue: bool,
        summary: str,
        next_iteration_request: str,
        analysis_doc_path: Optional[str],
    ) -> None:
        """Write analysis_summary.json to artifacts/ (new) or results/ (legacy).

        Accepts either a workspace root path (new callers) or a results/
        subdirectory path (legacy callers).  Detects which by checking if
        an ``artifacts/`` subdirectory exists or can be created at the root.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        data = {
            "should_continue": should_continue,
            "summary": summary,
            "next_iteration_request": next_iteration_request,
            "analysis_doc_path": analysis_doc_path,
        }
        # Try workspace mode: if the path has artifacts/ subdir or we can
        # identify it as a workspace root (not ending with "results")
        artifacts_dir = os.path.join(workspace_or_results_dir, "artifacts")
        if os.path.isdir(artifacts_dir) or not workspace_or_results_dir.rstrip(
            os.sep
        ).endswith("results"):
            ws = InferencerWorkspace(root=workspace_or_results_dir)
            path = ws.artifact_path("analysis_summary.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            # Legacy: caller passed the results/ subdirectory directly
            path = os.path.join(workspace_or_results_dir, "analysis_summary.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # endregion

    # region Load Existing Outputs (3j-3k)

    def _load_existing_plan(self, iter_workspace: str) -> Tuple[str, Optional[str]]:
        """Load the highest-round plan file from a workspace.

        Checks child workspace (``children/planner/``) first, then falls
        back to legacy (``outputs/``).

        Returns:
            Tuple of (plan_text, plan_file_path). Empty string and None if
            no plan files found.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        ws = InferencerWorkspace(root=iter_workspace)

        # New: child workspace final output
        child_final = ws.child_output("planner", "plan.md")
        if os.path.isfile(child_final):
            try:
                with open(child_final) as f:
                    return f.read(), child_final
            except Exception:
                pass

        # New: child workspace artifacts (round files)
        child_art_dir = os.path.join(ws.children_dir, "planner", "artifacts")
        if os.path.isdir(child_art_dir):
            child_files = glob.glob(
                os.path.join(child_art_dir, "round*_plan.md")
            )
            if child_files:
                best = max(child_files, key=self._extract_round)
                try:
                    with open(best) as f:
                        return f.read(), best
                except Exception:
                    pass

        # Legacy: outputs/ directly
        pattern = os.path.join(iter_workspace, "outputs", "round*_plan.md")
        matching = glob.glob(pattern)
        if not matching:
            self.log_info(
                f"No plan files found in {iter_workspace}",
                "LoadExistingPlanNotFound",
            )
            return "", None

        best = max(matching, key=self._extract_round)
        try:
            with open(best) as f:
                content = f.read()
        except Exception:
            content = ""
        return content, best

    def _load_existing_implementation(self, iter_workspace: str) -> str:
        """Load the highest-round implementation file from a workspace.

        Checks child workspace (``children/executor/``) first, then falls
        back to legacy (``outputs/``).

        Returns:
            Implementation text. Empty string if no files found.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        ws = InferencerWorkspace(root=iter_workspace)

        # New: child workspace final output
        child_final = ws.child_output("executor", "implementation.md")
        if os.path.isfile(child_final):
            try:
                with open(child_final) as f:
                    return f.read()
            except Exception:
                pass

        # New: child workspace artifacts (round files)
        child_art_dir = os.path.join(ws.children_dir, "executor", "artifacts")
        if os.path.isdir(child_art_dir):
            child_files = glob.glob(
                os.path.join(child_art_dir, "round*_implementation.md")
            )
            if child_files:
                best = max(child_files, key=self._extract_round)
                try:
                    with open(best) as f:
                        return f.read()
                except Exception:
                    pass

        # Legacy: outputs/ directly
        pattern = os.path.join(iter_workspace, "outputs", "round*_implementation.md")
        matching = glob.glob(pattern)
        if not matching:
            self.log_info(
                f"No implementation files found in {iter_workspace}",
                "LoadExistingImplNotFound",
            )
            return ""

        best = max(matching, key=self._extract_round)
        try:
            with open(best) as f:
                return f.read()
        except Exception:
            return ""

    # endregion

    # region Session Reset (3l)

    async def _reset_sub_inferencers_for_meta_iteration(self) -> None:
        """Reset all sub-inferencer sessions for a new meta-iteration.

        Disconnects, optionally calls reset_session(), then reconnects.
        """
        seen_ids: set = set()
        for inf in (
            self.planner_inferencer,
            self.executor_inferencer,
            self.analyzer_inferencer,
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                await inf.adisconnect()
                reset_fn = getattr(inf, "reset_session", None)
                if reset_fn is not None and callable(reset_fn):
                    reset_fn()
                await inf.aconnect()

    # endregion

    # region Resume Detection (Phase 4) — file-based, kept for backward compat

    @staticmethod
    def _extract_round(path: str) -> int:
        """Extract round number from a filename like ``round03_plan.md``."""
        m = re.search(r"round(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    @staticmethod
    def _find_analysis_summary(workspace_path: str) -> Optional[str]:
        """Find analysis_summary.json in artifacts/ (new) or results/ (legacy)."""
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        ws = InferencerWorkspace(root=workspace_path)
        path = ws.artifact_path("analysis_summary.json")
        if os.path.isfile(path):
            return path
        legacy = os.path.join(workspace_path, "results", "analysis_summary.json")
        return legacy if os.path.isfile(legacy) else None

    def _detect_workspace_state(
        self, workspace_path: str, iteration: int
    ) -> WorkspaceState:
        """Detect the completion state of a workspace directory.

        Checks both new composition locations (``children/<name>/``) and
        legacy locations (``outputs/``) for backward compatibility.
        Markers are checked via ``InferencerWorkspace.has_marker()`` which
        looks in ``artifacts/`` then ``outputs/``.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        state = WorkspaceState(workspace_path=workspace_path, iteration=iteration)
        ws = InferencerWorkspace(root=workspace_path)

        # -- Plan detection: child workspace then legacy --
        plan_file = None

        child_plan_final = ws.child_output("planner", "plan.md")
        if os.path.isfile(child_plan_final):
            plan_file = child_plan_final
        else:
            child_art_dir = os.path.join(ws.children_dir, "planner", "artifacts")
            if os.path.isdir(child_art_dir):
                child_files = glob.glob(
                    os.path.join(child_art_dir, "round*_plan.md")
                )
                if child_files:
                    plan_file = max(child_files, key=self._extract_round)

        if plan_file is None:
            legacy_files = glob.glob(
                os.path.join(workspace_path, "outputs", "round*_plan.md")
            )
            if legacy_files:
                plan_file = max(legacy_files, key=self._extract_round)

        if plan_file is not None:
            state.plan_file_path = plan_file
            if ws.has_marker("plan"):
                state.plan_done = True
                state.plan_partial = False
            else:
                state.plan_done = True
                state.plan_partial = True

        # -- Implementation detection: child workspace then legacy --
        impl_file = None

        child_impl_final = ws.child_output("executor", "implementation.md")
        if os.path.isfile(child_impl_final):
            impl_file = child_impl_final
        else:
            child_art_dir = os.path.join(ws.children_dir, "executor", "artifacts")
            if os.path.isdir(child_art_dir):
                child_files = glob.glob(
                    os.path.join(child_art_dir, "round*_implementation.md")
                )
                if child_files:
                    impl_file = max(child_files, key=self._extract_round)

        if impl_file is None:
            legacy_files = glob.glob(
                os.path.join(workspace_path, "outputs", "round*_implementation.md")
            )
            if legacy_files:
                impl_file = max(legacy_files, key=self._extract_round)

        if impl_file is not None:
            if ws.has_marker("impl"):
                state.impl_done = True
                state.impl_partial = False
            else:
                state.impl_done = False
                state.impl_partial = True

        # -- Analysis detection (artifacts/ then results/ fallback) --
        summary_path = self._find_analysis_summary(workspace_path)
        if summary_path is not None:
            state.analysis_done = True
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                state.analysis_doc_path = data.get("analysis_doc_path")
            except Exception:
                pass

        for subdir in self.results_subdirs:
            subdir_path = os.path.join(workspace_path, "outputs", subdir)
            if os.path.isdir(subdir_path) and os.listdir(subdir_path):
                if subdir == "benchmarks":
                    state.has_benchmark_results = True
                elif subdir == "tests":
                    state.has_test_results = True

        return state

    def _detect_resume_point(
        self, base_workspace: str
    ) -> Tuple[int, str, WorkspaceState, str, str]:
        """Detect where to resume from in a workspace.

        Returns:
            Tuple of (iteration, resume_phase, state, current_input, original_request).
            resume_phase is one of: "planning", "implementation", "analysis",
            "new_iteration", "complete".
        """
        request_path = os.path.join(base_workspace, "request.txt")
        original_request = ""
        if os.path.isfile(request_path):
            with open(request_path) as f:
                original_request = f.read()

        state = self._detect_workspace_state(base_workspace, 1)

        if not state.plan_done:
            return (1, "planning", state, original_request, original_request)
        if state.plan_partial:
            # Plan content exists but was never confirmed complete (no
            # .plan_completed marker).  Resume at planning so the existing
            # plan can be reviewed rather than skipping straight to
            # implementation with an unreviewed plan.
            return (1, "planning", state, original_request, original_request)
        if not state.impl_done:
            return (1, "implementation", state, original_request, original_request)
        if self.enable_analysis and not state.analysis_done:
            return (1, "analysis", state, original_request, original_request)

        followup_dir = os.path.join(base_workspace, "followup_iterations")
        if not os.path.isdir(followup_dir):
            return self._check_should_continue_or_complete(
                base_workspace, 1, state, original_request
            )

        iter_dirs = sorted(
            [
                d
                for d in os.listdir(followup_dir)
                if d.startswith("iteration_")
                and os.path.isdir(os.path.join(followup_dir, d))
            ]
        )
        if not iter_dirs:
            return self._check_should_continue_or_complete(
                base_workspace, 1, state, original_request
            )

        def _extract_iter_num(d: str) -> int:
            m = re.match(r"iteration_(\d+)", d)
            return int(m.group(1)) if m else 0

        highest_dir = max(iter_dirs, key=_extract_iter_num)
        n = _extract_iter_num(highest_dir)
        iter_ws = os.path.join(followup_dir, highest_dir)
        iter_state = self._detect_workspace_state(iter_ws, n)

        if not iter_state.plan_done:
            current_input = self._read_request_txt(iter_ws)
            return (n, "planning", iter_state, current_input, original_request)
        if iter_state.plan_partial:
            current_input = self._read_request_txt(iter_ws)
            return (n, "planning", iter_state, current_input, original_request)
        if not iter_state.impl_done:
            current_input = self._read_request_txt(iter_ws)
            return (n, "implementation", iter_state, current_input, original_request)
        if self.enable_analysis and not iter_state.analysis_done:
            current_input = self._read_request_txt(iter_ws)
            return (n, "analysis", iter_state, current_input, original_request)

        return self._check_should_continue_or_complete(
            base_workspace, n, iter_state, original_request
        )

    def _check_should_continue_or_complete(
        self,
        base_workspace: str,
        iteration: int,
        state: WorkspaceState,
        original_request: str,
    ) -> Tuple[int, str, WorkspaceState, str, str]:
        """Check if the last completed iteration's analysis says to continue."""
        iter_ws = self._get_iteration_workspace(base_workspace, iteration)
        summary_path = os.path.join(iter_ws, "results", "analysis_summary.json")
        if os.path.isfile(summary_path):
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                if data.get("should_continue", False):
                    next_request = data.get("next_iteration_request", "")
                    analysis_doc_path = data.get("analysis_doc_path")
                    handoff = self._build_iteration_handoff(
                        original_request,
                        iteration + 1,
                        self.max_meta_iterations,
                        next_request,
                        analysis_doc_path,
                        data.get("summary", ""),
                    )
                    return (
                        iteration + 1,
                        "new_iteration",
                        state,
                        handoff,
                        original_request,
                    )
            except Exception:
                pass
        return (iteration, "complete", state, "", original_request)

    @staticmethod
    def _read_request_txt(workspace_path: str) -> str:
        """Read request.txt from a workspace directory."""
        req_path = os.path.join(workspace_path, "request.txt")
        if os.path.isfile(req_path):
            with open(req_path) as f:
                return f.read()
        return ""

    def _load_completed_iteration(
        self, workspace: str, iteration: int
    ) -> MetaIterationRecord:
        """Load a completed iteration's results from disk."""
        record = MetaIterationRecord(iteration=iteration, workspace_path=workspace)

        plan_str, plan_file = self._load_existing_plan(workspace)
        record.plan_output = plan_str
        record.plan_file_path = plan_file

        impl_str = self._load_existing_implementation(workspace)
        record.executor_output = impl_str

        summary_path = os.path.join(workspace, "results", "analysis_summary.json")
        if os.path.isfile(summary_path):
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                record.analysis_output = data.get("summary", "")
                record.analysis_doc_path = data.get("analysis_doc_path")
                record.should_continue = data.get("should_continue", False)
            except Exception:
                pass

        return record

    # endregion

    def _setup_child_workflows(self, state, *args, **kwargs):
        """Override to provide per-iteration child workflow directories.

        Workspace-mode children (``_workspace`` set by
        ``_setup_iteration_children``) manage their own checkpoint paths,
        so ``_result_root_override`` is set to ``None`` (the child's
        ``_get_result_path`` returns full absolute paths).

        Legacy children get manual path isolation:
        ``<base>/checkpoints/pti/iter_<N>/<attr_name>/``
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        if state is None:
            return

        has_metadata = getattr(type(self), "__artifact_types__", None) or getattr(
            type(state), "__artifact_types__", None
        )
        if not has_metadata:
            return

        base = getattr(self, "_current_base_workspace", None)
        if not base:
            return

        iteration = state.get("iteration", 1) if isinstance(state, dict) else 1

        all_children = {}
        all_children.update(self._find_child_workflows_in(self))
        all_children.update(self._find_child_workflows_in(state))

        for attr_name, (child, entry) in all_children.items():
            child_has_workspace = getattr(child, "_workspace", None) is not None

            if child_has_workspace:
                # Full composition: child manages own checkpoint paths.
                # _resolve_result_path with None returns as-is.
                child._result_root_override = None
            else:
                # Legacy: manual path for checkpoint isolation
                ws = InferencerWorkspace(root=base)
                child_dir = os.path.join(
                    ws.checkpoint_path("pti"),
                    f"iter_{iteration}",
                    attr_name,
                )
                os.makedirs(child_dir, exist_ok=True)
                child._result_root_override = child_dir

            child.enable_result_save = self.enable_result_save
            child.resume_with_saved_results = self.resume_with_saved_results
            child.checkpoint_mode = self.checkpoint_mode

    # region Workflow Method Overrides (Phase D.3)

    def _get_result_path(self, result_id, *args, **kwargs):
        """Return path for checkpoint files under the STABLE base workspace.

        Uses _current_base_workspace (not _current_iteration_workspace) so
        that the __wf_checkpoint__ file and step result files always resolve
        to the same directory regardless of which loop iteration is active.
        Step results are distinguished by ___seqN suffixes (monotonically
        increasing), so they never collide even in a single directory.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        base = getattr(self, "_current_base_workspace", None)
        if not base:
            return ""
        ws = InferencerWorkspace(root=base)
        return ws.checkpoint_path(os.path.join("pti", f"step_{result_id}.json"))

    def _save_result(self, result, output_path: str):
        """Save step result as JSON with explicit dict__ pre-conversion."""
        from rich_python_utils.common_utils.map_helper import dict__
        from rich_python_utils.io_utils.json_io import write_json

        write_json(dict__(result, recursive=True), output_path, indent=2)

    def _load_result(self, result_id, result_path_or_preloaded_result):
        """Load step result from JSON file."""
        from rich_python_utils.io_utils.json_io import read_json

        if isinstance(result_path_or_preloaded_result, str):
            return read_json(result_path_or_preloaded_result)
        return result_path_or_preloaded_result

    def _save_loop_checkpoint(
        self, step_index, next_step_index, last_saved_result_id, state, *args, **kwargs
    ):
        """Save loop checkpoint — JSON mode, stringify keys."""
        # Re-setup child workflows before saving (matches base behaviour)
        self._setup_child_workflows(state, *args, **kwargs)

        # Validate JSON serializability on first checkpoint only
        if not getattr(self, "_state_picklability_verified", False):
            try:
                import json as _json

                from rich_python_utils.common_utils.map_helper import dict__

                _json.dumps(dict__(state, recursive=True))
            except Exception as e:
                raise TypeError(
                    f"Workflow state is not JSON-serializable and cannot be "
                    f"checkpointed. Make state serializable or set "
                    f"enable_result_save=False. Original error: {e}"
                ) from e
            self._state_picklability_verified = True

        self._save_checkpoint(
            {
                "version": 1,
                "exec_seq": self._exec_seq,
                "step_index": step_index,
                "result_id": last_saved_result_id,
                "next_step_index": next_step_index,
                "loop_counts": {str(k): v for k, v in self._loop_counts.items()},
                "state": state,
            },
            *args,
            **kwargs,
        )

    def _try_load_checkpoint(self, *args, **kwargs) -> Optional[dict]:
        """Try loading checkpoint: real checkpoint first, then backward-compat synthesis."""
        # 1. Try real checkpoint (base Workflow logic)
        try:
            ckpt_id = "__wf_checkpoint__"
            ckpt_path = self._resolve_result_path(ckpt_id, *args, **kwargs)
            exists = self._exists_result(result_id=ckpt_id, result_path=ckpt_path)
            if exists:
                ckpt = self._load_result(
                    result_id=ckpt_id,
                    result_path_or_preloaded_result=(
                        ckpt_path if isinstance(exists, bool) else exists
                    ),
                )
                if isinstance(ckpt, dict) and "next_step_index" in ckpt:
                    # Fix int keys: JSON serializes dict keys as strings
                    lc = ckpt.get("loop_counts", {})
                    if isinstance(lc, dict):
                        ckpt["loop_counts"] = {int(k): v for k, v in lc.items()}

                    # Validate checkpoint against file markers: if the
                    # checkpoint claims the plan is approved but
                    # .plan_completed was deleted, the user wants to
                    # re-review the plan.  Discard the stale checkpoint
                    # and fall through to Tier 2 file-based detection.
                    if self.resume_workspace:
                        ckpt_state = ckpt.get("state", {})
                        if (
                            ckpt_state.get("plan_approved")
                            and ckpt.get("next_step_index", 0) > 0
                        ):
                            ws = self._get_iteration_workspace(
                                self.resume_workspace,
                                ckpt_state.get("iteration", 1),
                            )
                            plan_marker = os.path.join(
                                ws or self.resume_workspace,
                                "outputs",
                                ".plan_completed",
                            )
                            if not os.path.isfile(plan_marker):
                                self.log_info(
                                    "Checkpoint says plan_approved=True but "
                                    ".plan_completed marker is missing — "
                                    "discarding stale checkpoint for re-review"
                                )
                                # Fall through to Tier 2
                                pass
                            elif ckpt.get("next_step_index", 0) > 2:
                                # Checkpoint says we're past implementation.
                                # Validate that implementation actually ran
                                # (not just skipped via --no-implementation).
                                impl_marker = os.path.join(
                                    ws or self.resume_workspace,
                                    "outputs",
                                    ".impl_completed",
                                )
                                impl_files = glob.glob(
                                    os.path.join(
                                        ws or self.resume_workspace,
                                        "outputs",
                                        "round*_implementation.md",
                                    )
                                )
                                if not os.path.isfile(impl_marker) and not impl_files:
                                    self.log_info(
                                        "Checkpoint says past implementation but "
                                        "no .impl_completed marker or implementation "
                                        "files found — implementation was skipped, "
                                        "discarding stale checkpoint"
                                    )
                                    # Fall through to Tier 2
                                    pass
                                else:
                                    ckpt_state_final = ckpt.get("state")
                                    if ckpt_state_final is not None:
                                        self._setup_child_workflows(
                                            ckpt_state_final, *args, **kwargs
                                        )
                                    return ckpt
                            else:
                                ckpt_state_final = ckpt.get("state")
                                if ckpt_state_final is not None:
                                    self._setup_child_workflows(
                                        ckpt_state_final, *args, **kwargs
                                    )
                                return ckpt
                        else:
                            ckpt_state_final = ckpt.get("state")
                            if ckpt_state_final is not None:
                                self._setup_child_workflows(
                                    ckpt_state_final, *args, **kwargs
                                )
                            return ckpt
                    else:
                        ckpt_state = ckpt.get("state")
                        if ckpt_state is not None:
                            self._setup_child_workflows(ckpt_state, *args, **kwargs)
                        return ckpt
        except Exception:
            pass

        # 2. Backward-compat: synthesize from file-based workspace detection
        if self.resume_workspace:
            return self._synthesize_checkpoint_from_workspace()
        return None

    def _synthesize_checkpoint_from_workspace(self) -> Optional[dict]:
        """Synthesize a Workflow checkpoint from file-based workspace state detection.

        Used for backward compatibility: existing workspaces created before
        the Workflow migration have no __wf_checkpoint__ file, but have output
        files that _detect_resume_point() can analyze.
        """
        base_workspace = self.resume_workspace
        if not base_workspace:
            return None

        try:
            (
                iteration,
                resume_phase,
                resume_state,
                current_input,
                original_request,
            ) = self._detect_resume_point(base_workspace)
        except Exception:
            return None

        # Analysis-only override: mirror the _ainfer logic so the
        # synthesized checkpoint lands on the analysis step instead of
        # returning None for "complete" workspaces.
        _is_analysis_only = (
            self.enable_analysis
            and not self.enable_planning
            and not self.enable_implementation
        )
        if _is_analysis_only and resume_phase in (
            "complete",
            "new_iteration",
            "implementation",
        ):
            if resume_phase == "new_iteration":
                iteration = max(1, iteration - 1)
            resume_phase = "analysis"

        if resume_phase == "complete":
            return None

        next_step_index = _PHASE_TO_STEP_INDEX.get(resume_phase)
        if next_step_index is None:
            return None

        # Build the synthetic state dict
        state = {
            "iteration": iteration,
            "current_input": current_input,
            "original_request": original_request,
            "plan_output_text": "",
            "plan_file_path": None,
            "plan_approved": None,
            "executor_output_text": "",
            "should_continue": False,
            "next_iteration_request": "",
            "iteration_records": [],
        }

        # Load existing plan for review when resuming at planning with
        # plan_partial=True (plan content exists but wasn't confirmed
        # complete).  Feed it as initial_response_override so the
        # DualInferencer skips propose and goes directly to review.
        if resume_phase == "planning" and resume_state.plan_partial:
            iter_ws = self._get_iteration_workspace(base_workspace, iteration)
            plan_str, plan_file = self._load_existing_plan(iter_ws)
            if plan_str:
                state["plan_output_text"] = plan_str
                state["plan_file_path"] = plan_file
                state["_resume_plan_for_review"] = True

        # Load completed data for partially-done iterations
        if resume_phase in ("implementation", "analysis", "new_iteration"):
            plan_str, plan_file = self._load_existing_plan(
                self._get_iteration_workspace(base_workspace, iteration)
            )
            state["plan_output_text"] = plan_str
            state["plan_file_path"] = plan_file
            state["plan_approved"] = True

        if resume_phase in ("analysis", "new_iteration"):
            executor_str = self._load_existing_implementation(
                self._get_iteration_workspace(base_workspace, iteration)
            )
            state["executor_output_text"] = executor_str

        # Update workspace to match iteration
        ws = self._get_iteration_workspace(base_workspace, iteration)
        self._current_iteration_workspace = ws
        self._current_base_workspace = base_workspace

        # Tier 2 partial detection: if impl files exist but the
        # completion marker is missing, record this in the *state dict*
        # (not on self) because _arun() unconditionally resets
        # self._step_was_previously_attempted = False before the Tier 1
        # marker check.  The state dict survives _arun init, so we
        # restore it in _step_implement_impl before _build_executor_input.
        if resume_phase == "implementation" and resume_state.impl_partial:
            state["_impl_was_partially_attempted"] = True
            if not hasattr(self, "_step_attempt_counts"):
                self._step_attempt_counts = {}
            self._step_attempt_counts[2] = 1  # step index 2 = implement

        # Create sentinel result file so _arun's result_id validation passes.
        # _arun loads checkpoint["result_id"] via _resolve_result_path — if the
        # file doesn't exist, the checkpoint is silently discarded (line 689-693).
        sentinel_id = "__synth_sentinel__"
        sentinel_path = self._resolve_result_path(sentinel_id)
        if sentinel_path:
            os.makedirs(os.path.dirname(sentinel_path), exist_ok=True)
            self._save_result(
                {"_synthetic": True, "phase": resume_phase}, sentinel_path
            )

        return {
            "version": 1,
            "exec_seq": 0,
            "step_index": max(0, next_step_index - 1),
            "result_id": sentinel_id,
            "next_step_index": next_step_index,
            "loop_counts": {},
            "state": state,
        }

    # ------------------------------------------------------------------
    # Step-in-progress marker overrides (use _get_result_path for
    # consistent pathing under checkpoints/pti/)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_step_completion_marker(workspace_path: str, phase_name: str) -> None:
        """Write a JSON marker confirming that a step completed successfully.

        Creates ``<workspace>/artifacts/.<phase>_completed`` containing a
        timestamp and step name.  Used by Tier 2 (file-based) resume
        detection in ``_detect_workspace_state`` to distinguish between
        "output file exists AND step finished" vs "output file exists but
        step may have been interrupted."

        Legacy note: markers previously lived in ``outputs/``. The
        ``InferencerWorkspace.has_marker()`` method checks both locations
        for backward compatibility.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        try:
            ws = InferencerWorkspace(root=workspace_path)
            ws.write_marker(phase_name)
        except Exception:
            pass

    def _save_step_in_progress_marker(
        self, step_index, step_name, state, *args, **kwargs
    ):
        """Override: write marker under checkpoints/pti/ via _get_result_path."""
        marker_path = self._get_result_path("__wf_step_in_progress__")
        if not marker_path:
            return
        try:
            os.makedirs(os.path.dirname(marker_path), exist_ok=True)
            marker = {
                "step_index": step_index,
                "step_name": step_name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "attempt": self._step_attempt_counts.get(step_index, 0),
            }
            with open(marker_path, "w") as f:
                json.dump(marker, f, indent=2)
        except Exception:
            pass

    def _clear_step_in_progress_marker(self, *args, **kwargs):
        """Override: remove marker under checkpoints/pti/."""
        marker_path = self._get_result_path("__wf_step_in_progress__")
        if not marker_path:
            return
        try:
            os.remove(marker_path)
        except OSError:
            pass

    def _load_step_in_progress_marker(self, *args, **kwargs) -> Optional[dict]:
        """Override: load marker from checkpoints/pti/."""
        marker_path = self._get_result_path("__wf_step_in_progress__")
        if not marker_path:
            return None
        try:
            if os.path.isfile(marker_path):
                with open(marker_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _backward_scan_resume(self, _has_loops, *args, **kwargs):
        """DISABLED for PTI — all resume goes through _try_load_checkpoint."""
        return -1, None

    def _init_state(self) -> dict:
        """Return the pending PTI state prepared in _ainfer."""
        return self._pending_pti_state or {}

    def _handle_abort(self, abort_exc, step_result, state):
        """Handle WorkflowAborted (plan rejection) — return state."""
        return state

    def _exists_result(self, result_id, result_path):
        """Check if a JSON result file exists."""
        if not result_path:
            return False
        json_path = result_path
        if not json_path.endswith(".json"):
            json_path = result_path + ".json"
        return os.path.exists(json_path) or os.path.exists(result_path)

    # endregion

    # region Step Closures (Phase C.2 + D.4)

    def _build_iteration_steps(self):
        """Build Workflow steps with native loop support."""

        def _update_state_fn(state, result):
            return state

        async def _step_plan_impl(*args, **kwargs):
            state = self._state
            iteration = state["iteration"]
            base_workspace = self._current_base_workspace
            ws = self._get_iteration_workspace(base_workspace, iteration)
            self._current_iteration_workspace = ws
            # RE-CALL to update children's _result_root_override for this iteration
            self._setup_child_workflows(state)

            if not self.enable_planning:
                if ws:
                    plan_str, plan_path = self._load_existing_plan(ws)
                    state["plan_output_text"] = plan_str
                    state["plan_file_path"] = plan_path
                self._state = state
                return state.get("plan_output_text", "")

            inference_config = self._current_inference_config or {}
            _inference_args = self._current_inference_args or {}

            # Build iteration config
            if base_workspace:
                iter_config = self._build_iteration_config(
                    inference_config, ws, iteration
                )
                self._setup_iteration_workspace(ws, iteration, state["current_input"])
            else:
                iter_config = inference_config

            plan_ic = iter_config.get(self.plan_config_key, iter_config)

            # Session reset on iteration > 1
            if iteration > 1 and self.reset_sessions_per_meta_iteration:
                await self._reset_sub_inferencers_for_meta_iteration()

            # Inject initial plan file override (first iteration only)
            if self.initial_plan_file and iteration == 1:
                if not os.path.isfile(self.initial_plan_file):
                    raise FileNotFoundError(
                        f"Initial plan file not found: {self.initial_plan_file}"
                    )
                with open(self.initial_plan_file) as f:
                    plan_text = f.read()
                # Copy initial plan to workspace as round0_plan.md
                if ws:
                    initial_plan_output = os.path.join(ws, "outputs", "round0_plan.md")
                    os.makedirs(os.path.dirname(initial_plan_output), exist_ok=True)
                    with open(initial_plan_output, "w") as f:
                        f.write(plan_text)
                    self.log_info(
                        f"[{self.planner_phase}] Copied initial plan to "
                        f"{initial_plan_output}"
                    )
                plan_ic = {**plan_ic, "initial_response_override": plan_text}

            # Resume with existing plan for review: plan_partial=True means
            # the plan content exists but was never confirmed complete (e.g.,
            # ran with max_iterations=0 then deleted .plan_completed).  Feed
            # the existing plan as initial_response_override so the
            # DualInferencer skips propose and goes directly to review.
            if state.get("_resume_plan_for_review") and not self.initial_plan_file:
                existing_plan = state.get("plan_output_text", "")
                if existing_plan:
                    plan_ic = {**plan_ic, "initial_response_override": existing_plan}
                    self.log_info(
                        f"[{self.planner_phase}] Resuming with existing plan "
                        f"for review ({len(existing_plan)} chars)"
                    )

            self.log_info(f"[{self.planner_phase}] Starting planning phase")

            result = await self.planner_inferencer.ainfer(
                state["current_input"],
                inference_config=plan_ic,
                **_inference_args,
            )
            plan_str = self._extract_response_text(result)

            plan_file_path = None
            if self.planner_outputs_plan_to_file:
                plan_file_path = self._resolve_plan_file_path(plan_ic)

            state["plan_output_text"] = plan_str
            state["plan_file_path"] = plan_file_path
            # Store iter_config in state for implement/analysis steps
            state["_iter_config"] = iter_config
            state["_plan_response"] = result
            self._state = state

            # Tier 2: write file-based completion marker
            ws = self._current_iteration_workspace
            if ws:
                self._write_step_completion_marker(ws, "plan")

            return plan_str

        async def _step_approval_impl(*args, **kwargs):
            state = self._state
            plan_str = state.get("plan_output_text", "")

            if self.interactive is None:
                state["plan_approved"] = True
                self._state = state
                return plan_str

            interactive = self.interactive

            # Use enhanced checkpoint if enabled
            if self.enable_checkpoint_plan_review:
                # TODO: interactive_checkpoint module does not exist at agent_foundation.ui — needs separate migration
                from agent_foundation.ui.interactive_checkpoint import (
                    checkpoint_plan_review,
                )

                plan_summary = (
                    f"{plan_str[:1000]}{'...' if len(plan_str) > 1000 else ''}"
                )
                result = await checkpoint_plan_review(
                    interactive, plan_summary, default_action="approve"
                )

                if result.action == "approve":
                    self.log_info(f"[{self.planner_phase}] Plan approved by human")
                    state["plan_approved"] = True
                    self._state = state
                    return plan_str
                elif result.action == "modify":
                    self.log_info(
                        f"[{self.planner_phase}] Plan modification requested: {result.user_input[:200]}",
                        "PlanModify",
                    )
                    state["plan_approved"] = False
                    state["plan_modification_request"] = result.user_input
                    self._state = state
                    raise WorkflowAborted(
                        f"Plan modification requested: {result.user_input}"
                    )
                else:  # reject
                    self.log_info("Plan rejected by human", "PlanRejected")
                    state["plan_approved"] = False
                    self._state = state
                    raise WorkflowAborted("Plan rejected by human")

            # Fallback: original y/N text prompt
            approval_message = (
                f"=== Plan Phase Complete ===\n\n"
                f"Plan summary (first 500 chars):\n"
                f"{plan_str[:500]}{'...' if len(plan_str) > 500 else ''}\n\n"
                f"Approve this plan and proceed to implementation? (y/N)"
            )
            interactive.send_response(
                approval_message,
                flag=InteractionFlags.PendingInput,
            )
            user_input = interactive.get_input()
            approved = (
                user_input.strip().lower() in ("y", "yes", "")
                if user_input is not None
                else False
            )

            if not approved:
                self.log_info("Plan rejected by human", "PlanRejected")
                state["plan_approved"] = False
                self._state = state
                raise WorkflowAborted("Plan rejected by human")

            self.log_info(f"[{self.planner_phase}] Plan approved by human")
            state["plan_approved"] = True
            self._state = state
            return plan_str

        async def _step_implement_impl(*args, **kwargs):
            state = self._state

            base_workspace = self._current_base_workspace
            # Derive workspace (safe even on resume past plan step)
            ws = self._get_iteration_workspace(base_workspace, state["iteration"])
            self._current_iteration_workspace = ws

            if not self.enable_implementation:
                if ws:
                    impl_str = self._load_existing_implementation(ws)
                    state["executor_output_text"] = impl_str
                self._state = state
                return state.get("executor_output_text", "")

            inference_config = self._current_inference_config or {}
            _inference_args = self._current_inference_args or {}

            iter_config = state.get("_iter_config")
            if iter_config is None and base_workspace:
                iter_config = self._build_iteration_config(
                    inference_config, ws, state["iteration"]
                )
            elif iter_config is None:
                iter_config = inference_config

            impl_ic = iter_config.get(self.implement_config_key, iter_config)

            # Restore Tier 2 partial-attempt flag from state dict.
            # _synthesize_checkpoint_from_workspace() stores this in state
            # because _arun() resets self._step_was_previously_attempted.
            # Pop the flag so it doesn't persist across loop iterations.
            if state.pop("_impl_was_partially_attempted", False):
                self._step_was_previously_attempted = True

            plan_str = state.get("plan_output_text", "")
            plan_file_path = state.get("plan_file_path")
            executor_input = self._build_executor_input(
                state["current_input"], plan_str, plan_file_path=plan_file_path
            )

            self.log_info(f"[{self.executor_phase}] Starting execution phase")

            result = await self.executor_inferencer.ainfer(
                executor_input,
                inference_config=impl_ic,
                **_inference_args,
            )
            executor_str = self._extract_response_text(result)

            state["executor_output_text"] = executor_str
            state["_executor_response"] = result
            self._state = state

            # Tier 2: write file-based completion marker
            ws = self._current_iteration_workspace
            if ws:
                self._write_step_completion_marker(ws, "impl")

            return executor_str

        async def _step_analysis_impl(*args, **kwargs):
            state = self._state
            base_workspace = self._current_base_workspace
            ws = self._get_iteration_workspace(base_workspace, state["iteration"])
            self._current_iteration_workspace = ws

            inference_config = self._current_inference_config or {}
            _inference_args = self._current_inference_args or {}

            iter_config = state.get("_iter_config")
            if iter_config is None and base_workspace:
                iter_config = self._build_iteration_config(
                    inference_config, ws, state["iteration"]
                )
            elif iter_config is None:
                iter_config = inference_config

            should_continue = False
            if self.enable_analysis and ws:
                outputs_dir = os.path.join(ws, "outputs")
                has_results = self._has_results(outputs_dir)

                if has_results and self.analyzer_inferencer is not None:
                    analysis_ic = iter_config.get(self.analysis_config_key, iter_config)
                    analysis_vars = self._build_analysis_config_vars(
                        outputs_dir, state["iteration"]
                    )
                    analysis_ic = {**analysis_ic, **analysis_vars}

                    analysis_result = await self.analyzer_inferencer.ainfer(
                        state.get("original_request", state["current_input"]),
                        inference_config=analysis_ic,
                        **_inference_args,
                    )
                    analysis_str = self._extract_response_text(analysis_result)
                    state["_analysis_result_text"] = analysis_str
                    should_continue, next_request = self._parse_analysis_response(
                        analysis_str
                    )
                    analysis_doc_path = (
                        self._resolve_analysis_file_path(analysis_ic)
                        if self.analyzer_outputs_to_file
                        else None
                    )

                    results_dir = os.path.join(ws, "results")
                    os.makedirs(results_dir, exist_ok=True)
                    self._save_analysis_summary(
                        results_dir,
                        should_continue,
                        analysis_str[:500],
                        next_request,
                        analysis_doc_path,
                    )

                    # Record in iteration_records
                    from rich_python_utils.common_utils.map_helper import dict__

                    record = MetaIterationRecord(
                        iteration=state["iteration"],
                        workspace_path=ws,
                        plan_output=state.get("plan_output_text", ""),
                        executor_output=state.get("executor_output_text", ""),
                        plan_file_path=state.get("plan_file_path"),
                        plan_approved=state.get("plan_approved"),
                        analysis_output=analysis_str,
                        analysis_doc_path=analysis_doc_path,
                        should_continue=should_continue,
                        test_results_found=has_results,
                    )
                    state.setdefault("iteration_records", []).append(
                        dict__(record, recursive=True)
                    )
                else:
                    record = MetaIterationRecord(
                        iteration=state["iteration"],
                        workspace_path=ws,
                        plan_output=state.get("plan_output_text", ""),
                        executor_output=state.get("executor_output_text", ""),
                        plan_file_path=state.get("plan_file_path"),
                        plan_approved=state.get("plan_approved"),
                    )
                    from rich_python_utils.common_utils.map_helper import dict__

                    state.setdefault("iteration_records", []).append(
                        dict__(record, recursive=True)
                    )
            else:
                record = MetaIterationRecord(
                    iteration=state["iteration"],
                    workspace_path=ws,
                    plan_output=state.get("plan_output_text", ""),
                    executor_output=state.get("executor_output_text", ""),
                    plan_file_path=state.get("plan_file_path"),
                    plan_approved=state.get("plan_approved"),
                    analysis_output=analysis_str if "analysis_str" in dir() else None,
                    test_results_found=has_results if "has_results" in dir() else False,
                )
                from rich_python_utils.common_utils.map_helper import dict__

                state.setdefault("iteration_records", []).append(
                    dict__(record, recursive=True)
                )

            state["should_continue"] = should_continue
            if should_continue:
                next_request_text = next_request if "next_request" in dir() else ""
                analysis_doc = (
                    analysis_doc_path if "analysis_doc_path" in dir() else None
                )
                analysis_summary = analysis_str[:500] if "analysis_str" in dir() else ""
                state["iteration"] += 1
                state["current_input"] = self._build_iteration_handoff(
                    state.get("original_request", ""),
                    state["iteration"],
                    self.max_meta_iterations,
                    next_request_text,
                    analysis_doc,
                    analysis_summary,
                )
                state["plan_output_text"] = ""
                state["executor_output_text"] = ""
                state["plan_file_path"] = None
                state["plan_approved"] = None
                # Set up next iteration workspace
                if base_workspace:
                    next_ws = self._get_iteration_workspace(
                        base_workspace, state["iteration"]
                    )
                    self._setup_iteration_workspace(
                        next_ws, state["iteration"], state["current_input"]
                    )

            self._state = state
            return state.get(
                "_analysis_result_text", state.get("executor_output_text", "")
            )

        return [
            StepWrapper(
                _step_plan_impl,
                name="plan",
                update_state=_update_state_fn,
                # Don't checkpoint disabled steps — a saved empty result
                # blocks resume when the step is later enabled.
                enable_result_save=(
                    self.enable_result_save if self.enable_planning else False
                ),
            ),
            StepWrapper(
                _step_approval_impl,
                name="approval",
                update_state=_update_state_fn,
            ),
            StepWrapper(
                _step_implement_impl,
                name="implement",
                update_state=_update_state_fn,
                enable_result_save=(
                    self.enable_result_save if self.enable_implementation else False
                ),
            ),
            StepWrapper(
                _step_analysis_impl,
                name="analysis",
                update_state=_update_state_fn,
                loop_back_to="plan",
                loop_condition=lambda state, result: state.get(
                    "should_continue", False
                ),
                max_loop_iterations=self.max_meta_iterations - 1,
            ),
        ]

    # endregion

    # region Build Response from State

    def _build_response_from_state(self, state: dict) -> PlanThenImplementResponse:
        """Build PlanThenImplementResponse from the final workflow state."""
        if state is None:
            state = {}

        iteration_records = state.get("iteration_records", [])

        # Try to find the last record for response metadata
        last_plan_output = state.get("plan_output_text", "")
        last_executor_output = state.get("executor_output_text", "")
        plan_file_path = state.get("plan_file_path")
        plan_approved = state.get("plan_approved")
        plan_response = state.get("_plan_response")
        executor_response = state.get("_executor_response")

        final_base = (
            state.get("_analysis_result_text")
            or last_executor_output
            or last_plan_output
        )

        # Convert raw dicts back to MetaIterationRecord for the response
        history = []
        for rec in iteration_records:
            if isinstance(rec, dict):
                history.append(
                    MetaIterationRecord(
                        iteration=rec.get("iteration", 0),
                        workspace_path=rec.get("workspace_path"),
                        plan_output=rec.get("plan_output", ""),
                        executor_output=rec.get("executor_output", ""),
                        plan_file_path=rec.get("plan_file_path"),
                        plan_approved=rec.get("plan_approved"),
                        analysis_output=rec.get("analysis_output"),
                        analysis_doc_path=rec.get("analysis_doc_path"),
                        should_continue=rec.get("should_continue", False),
                        test_results_found=rec.get("test_results_found", False),
                    )
                )
            elif isinstance(rec, MetaIterationRecord):
                history.append(rec)

        return PlanThenImplementResponse(
            base_response=final_base,
            reflection_response=None,
            reflection_style=ReflectionStyles.NoReflection,
            response_selector=ResponseSelectors.BaseResponse,
            plan_response=plan_response,
            plan_output=last_plan_output,
            executor_output=executor_response,
            plan_approved=plan_approved,
            plan_file_path=plan_file_path,
            planner_phase=self.planner_phase,
            executor_phase=self.executor_phase,
            iteration_history=history,
            total_meta_iterations=len(history) or 1,
            meta_iterations_exhausted=(
                self.enable_multiple_iterations
                and len(history) >= self.max_meta_iterations
            ),
        )

    # endregion

    # region Sync/Async Bridge

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        """Sync bridge — delegates to _ainfer() via _run_async().

        For multi-call usage, prefer the async interface:
            async with PlanThenImplementInferencer(...) as pti:
                result = await pti.ainfer("task")
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(
            self._ainfer(inference_input, inference_config, **_inference_args)
        )

    # endregion

    # region Workspace Composition Helpers

    def _setup_iteration_children(self, state):
        """Set up child workspaces for the current iteration.

        Called once per iteration before any step runs.  Uses
        ``_CHILD_DEFAULTS`` to map attr names to short workspace names
        and default output_path values.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        if self._workspace is None:
            return
        iter_ws_path = self._get_iteration_workspace(
            self._current_base_workspace, state["iteration"]
        )
        iter_ws = InferencerWorkspace(root=iter_ws_path)

        for attr_name, (short_name, default_output) in _CHILD_DEFAULTS.items():
            child = getattr(self, attr_name, None)
            if child is None or not isinstance(child, InferencerBase):
                continue
            child_ws = iter_ws.child(short_name)
            child_ws.ensure_dirs()
            child._workspace = child_ws
            if not child.output_path:
                child.output_path = default_output

    def _finalize_outputs(self):
        """Copy selected child outputs from LAST iteration to workspace root.

        Driven by ``output_mode`` and ``_OUTPUT_MODE_MAP``.  Only runs in
        workspace mode.  Idempotent — safe on re-run after crash.
        """
        if self._workspace is None:
            return
        state = self._state or {}
        last_iter = state.get("iteration", 1)

        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        iter_ws_path = self._get_iteration_workspace(
            self._workspace.root, last_iter
        )
        iter_ws = InferencerWorkspace(root=iter_ws_path)

        os.makedirs(self._workspace.outputs_dir, exist_ok=True)
        for flag, (child_name, filename) in _OUTPUT_MODE_MAP.items():
            if flag in self.output_mode:
                src = iter_ws.child_output(child_name, filename)
                dst = self._workspace.output_path(filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

    def resolve_output_path(self, runtime_override=None):
        """PTI override: resolve based on output_mode."""
        if runtime_override is not None:
            return super().resolve_output_path(runtime_override)
        if self._workspace is None:
            return None
        active = [f for f in _OUTPUT_MODE_MAP if f in self.output_mode]
        if len(active) == 1:
            _, filename = _OUTPUT_MODE_MAP[active[0]]
            return self._workspace.output_path(filename)
        return self._workspace.outputs_dir

    # endregion

    # region Core Two-Phase Flow (Phase D.5)

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        """Async inference — core plan-then-implement flow.

        Now delegates to Workflow._arun() for native checkpoint/resume
        with loop support and automatic recursive child workflow resume.
        """
        if inference_config is None:
            inference_config = {}
        elif not isinstance(inference_config, dict):
            raise ValueError("'inference_config' must be a dict")

        self._partial_iteration_history = None

        base_workspace = self.resume_workspace or self.workspace_path

        # Handle "complete" early return for resume
        if self.resume_workspace:
            try:
                (
                    start_iter,
                    resume_phase,
                    _resume_state,
                    _current_input,
                    _original_request,
                ) = self._detect_resume_point(self.resume_workspace)

                # Analysis-only override
                _is_analysis_only = (
                    self.enable_analysis
                    and not self.enable_planning
                    and not self.enable_implementation
                )
                if _is_analysis_only and resume_phase in (
                    "complete",
                    "new_iteration",
                    "implementation",
                ):
                    if resume_phase == "new_iteration":
                        start_iter = max(1, start_iter - 1)
                    resume_phase = "analysis"

                if resume_phase == "complete":
                    final_ws = self._get_iteration_workspace(base_workspace, start_iter)
                    final_record = self._load_completed_iteration(final_ws, start_iter)
                    return PlanThenImplementResponse(
                        base_response=(
                            final_record.analysis_output or final_record.executor_output
                        ),
                        reflection_response=None,
                        reflection_style=ReflectionStyles.NoReflection,
                        response_selector=ResponseSelectors.BaseResponse,
                        plan_response=final_record.plan_response,
                        plan_output=final_record.plan_output,
                        executor_output=final_record.executor_response,
                        plan_file_path=final_record.plan_file_path,
                        planner_phase=self.planner_phase,
                        executor_phase=self.executor_phase,
                        iteration_history=[final_record],
                        total_meta_iterations=start_iter,
                        meta_iterations_exhausted=(
                            start_iter >= self.max_meta_iterations
                        ),
                    )
            except Exception:
                pass

        # Store context for closures
        self._current_base_workspace = base_workspace
        self._current_inference_config = inference_config
        self._current_inference_args = _inference_args

        # Workspace reconstruction
        if base_workspace:
            from agent_foundation.common.inferencers.inferencer_workspace import (
                InferencerWorkspace,
            )

            self._workspace = InferencerWorkspace(root=base_workspace)
        else:
            self._workspace = None

        # CRITICAL: set before _arun so _setup_child_workflows gets valid paths
        self._current_iteration_workspace = (
            self._get_iteration_workspace(base_workspace, 1) if base_workspace else None
        )

        # Build initial state
        self._pending_pti_state = {
            "iteration": 1,
            "current_input": inference_input,
            "original_request": inference_input,
            "plan_output_text": "",
            "plan_file_path": None,
            "plan_approved": None,
            "executor_output_text": "",
            "should_continue": False,
            "next_iteration_request": "",
            "iteration_records": [],
        }

        self._steps = self._build_iteration_steps()

        # Enable checkpointing
        if self._result_root_override is not None:
            pass  # Parent already configured
        elif base_workspace:
            self.enable_result_save = StepResultSaveOptions.Always
            self.resume_with_saved_results = True

        # Run the workflow
        await Workflow._arun(self, inference_input, **_inference_args)

        # Copy selected child outputs to workspace root outputs/
        self._finalize_outputs()

        return self._build_response_from_state(self._state)

    # endregion

    # region Lifecycle

    async def aconnect(self, **kwargs):
        """Establish connections for all sub-inferencers."""
        seen_ids = set()
        for inf in (
            self.planner_inferencer,
            self.executor_inferencer,
            self.analyzer_inferencer,
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                await inf.aconnect(**kwargs)

    async def adisconnect(self):
        """Disconnect all sub-inferencers."""
        seen_ids = set()
        for inf in (
            self.planner_inferencer,
            self.executor_inferencer,
            self.analyzer_inferencer,
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                await inf.adisconnect()

    # endregion

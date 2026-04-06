

"""Manual E2E test script for PlanThenImplementInferencer.

Chains two DualInferencer instances (plan + implementation) via
PlanThenImplementInferencer. Each DualInferencer runs its own consensus
loop with configurable external inferencers.

Usage:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_plan_then_implement -- \
        --request "Design and implement a REST API for user management" \
        --plan-base-inferencer claude_code \
        --plan-review-inferencer claude_code \
        --impl-base-inferencer claude_code \
        --impl-review-inferencer claude_code \
        --root-folder /path/to/repo

    # With human approval between plan and implementation:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_plan_then_implement -- \
        --request "Design and implement a REST API" \
        --plan-base-inferencer claude_code \
        --plan-review-inferencer devmate_cli \
        --impl-base-inferencer claude_code \
        --impl-review-inferencer devmate_cli \
        --require-approval \
        --root-folder /path/to/repo

    # Multi-iteration with analysis:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_plan_then_implement -- \
        --request "Design and implement a REST API" \
        --enable-analysis \
        --enable-multiple-iterations \
        --analyzer-inferencer claude_code \
        --workspace /path/to/workspace \
        --root-folder /path/to/repo

    # Resume from a previous workspace:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_plan_then_implement -- \
        --request "ignored when resuming" \
        --resume-workspace /path/to/previous/workspace \
        --root-folder /path/to/repo
"""

import asyncio
import glob
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import click
from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)
from agent_foundation.common.response_parsers import extract_delimited
from agent_foundation.common.ui.terminal_interactive import (
    TerminalInteractive,
)
from rich_python_utils.common_objects.debuggable import LoggerConfig
from rich_python_utils.io_utils.json_io import JsonLogger, SpaceExtMode
from rich_python_utils.string_utils.formatting.template_manager import (
    TemplateManager,
)

from .analysis_templates import (
    ANALYSIS_FOLLOWUP_TEMPLATE,
    ANALYSIS_INITIAL_TEMPLATE,
    ANALYSIS_REVIEW_TEMPLATE,
)
from .implementation_templates import (
    IMPLEMENTATION_FOLLOWUP_TEMPLATE,
    IMPLEMENTATION_INITIAL_TEMPLATE,
    IMPLEMENTATION_REVIEW_TEMPLATE,
)
from .plan_templates import (
    PLAN_FOLLOWUP_TEMPLATE,
    PLAN_INITIAL_TEMPLATE,
    PLAN_REVIEW_TEMPLATE,
)

logger = logging.getLogger(__name__)


# Map of template version → {role → template string}
TEMPLATES = {
    "plan": {
        "initial": PLAN_INITIAL_TEMPLATE,
        "review": PLAN_REVIEW_TEMPLATE,
        "followup": PLAN_FOLLOWUP_TEMPLATE,
    },
    "implementation": {
        "initial": IMPLEMENTATION_INITIAL_TEMPLATE,
        "review": IMPLEMENTATION_REVIEW_TEMPLATE,
        "followup": IMPLEMENTATION_FOLLOWUP_TEMPLATE,
    },
    "analysis": {
        "initial": ANALYSIS_INITIAL_TEMPLATE,
        "review": ANALYSIS_REVIEW_TEMPLATE,
        "followup": ANALYSIS_FOLLOWUP_TEMPLATE,
    },
}

INFERENCER_CHOICES = ["claude_code", "claude_code_cli", "devmate_sdk", "devmate_cli"]


# ---------------------------------------------------------------------------
# Inferencer factory
# ---------------------------------------------------------------------------


def create_inferencer(
    inferencer_type: str,
    root_folder: str,
    model: str | None,
    system_prompt: str,
    timeout: int,
    inferencer_logger=None,
    inferencer_id: str | None = None,
    cache_folder: str | None = None,
    large_arg_temp_dir: str | None = None,
    tool_use_timeout: int = 0,
) -> InferencerBase:
    """Build an inferencer instance based on type string.

    Note: This function mirrors __main__.create_inferencer(). They are kept
    separate because BUCK python_binary targets have isolated source trees,
    making cross-binary imports impractical without additional library targets.
    """
    if inferencer_type == "claude_code":
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_inferencer import (  # noqa: E501
            ClaudeCodeInferencer,
        )

        return ClaudeCodeInferencer(
            target_path=root_folder,
            model_id=model or "",
            system_prompt=system_prompt,
            idle_timeout_seconds=timeout,
            tool_use_idle_timeout_seconds=tool_use_timeout,
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
            logger=inferencer_logger,
            id=inferencer_id,
            cache_folder=cache_folder,
        )
    elif inferencer_type == "claude_code_cli":
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (  # noqa: E501
            ClaudeCodeCliInferencer,
        )

        kwargs: dict = dict(
            target_path=root_folder,
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
            idle_timeout_seconds=timeout,
            tool_use_idle_timeout_seconds=tool_use_timeout,
        )
        if model is not None:
            kwargs["model_name"] = model
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if inferencer_logger is not None:
            kwargs["logger"] = inferencer_logger
        if inferencer_id is not None:
            kwargs["id"] = inferencer_id
        if cache_folder is not None:
            kwargs["cache_folder"] = cache_folder
        if large_arg_temp_dir is not None:
            kwargs["large_arg_temp_dir"] = large_arg_temp_dir
        return ClaudeCodeCliInferencer(**kwargs)
    elif inferencer_type == "devmate_sdk":
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (  # noqa: E501
            DevmateSDKInferencer,
        )

        kwargs: dict = dict(
            target_path=root_folder,
            total_timeout_seconds=timeout,
            idle_timeout_seconds=timeout,
            tool_use_idle_timeout_seconds=tool_use_timeout,
        )
        if model is not None:
            kwargs["model_name"] = model
        if inferencer_logger is not None:
            kwargs["logger"] = inferencer_logger
        if inferencer_id is not None:
            kwargs["id"] = inferencer_id
        if cache_folder is not None:
            kwargs["cache_folder"] = cache_folder
        return DevmateSDKInferencer(**kwargs)
    elif inferencer_type == "devmate_cli":
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (  # noqa: E501
            DevmateCliInferencer,
        )

        kwargs = dict(
            target_path=root_folder,
            idle_timeout_seconds=timeout,
            tool_use_idle_timeout_seconds=tool_use_timeout,
        )
        if model is not None:
            kwargs["model_name"] = model
        if inferencer_logger is not None:
            kwargs["logger"] = inferencer_logger
        if inferencer_id is not None:
            kwargs["id"] = inferencer_id
        if cache_folder is not None:
            kwargs["cache_folder"] = cache_folder
        if large_arg_temp_dir is not None:
            kwargs["large_arg_temp_dir"] = large_arg_temp_dir
        return DevmateCliInferencer(**kwargs)
    else:
        raise ValueError(f"Unknown inferencer type: {inferencer_type}")


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def setup_workspace(
    workspace: Path, request_text: str, enable_analysis: bool = False
) -> None:
    """Create workspace directory structure and write template files."""
    (workspace / "prompt_templates" / "plan" / "main").mkdir(
        parents=True, exist_ok=True
    )
    (workspace / "prompt_templates" / "implementation" / "main").mkdir(
        parents=True, exist_ok=True
    )
    (workspace / "results").mkdir(parents=True, exist_ok=True)
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)

    if enable_analysis:
        (workspace / "prompt_templates" / "analysis" / "main").mkdir(
            parents=True, exist_ok=True
        )
        (workspace / "analysis").mkdir(parents=True, exist_ok=True)

    # Save original request
    (workspace / "request.txt").write_text(request_text)

    # Write adapted template files
    for version, templates in TEMPLATES.items():
        for role, content in templates.items():
            template_path = (
                workspace / "prompt_templates" / version / "main" / f"{role}.jinja2"
            )
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text(content)
            logger.info("Wrote template: %s", template_path)


def save_phase_results(
    results_dir: Path,
    output_dir: Path,
    phase: str,
    result,
) -> None:
    """Save DualInferencerResponse results for a single phase.

    Scans the outputs directory for files matching the phase pattern and
    saves per-iteration data.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.common import (
        DualInferencerResponse,
    )

    if not isinstance(result, DualInferencerResponse):
        # For non-DualInferencer results, just save the raw output
        (results_dir / f"{phase}_output.txt").write_text(str(result))
        return

    # Find the final output file by scanning for largest round index
    pattern = str(output_dir / f"round*_{phase}.md")
    output_files = glob.glob(pattern)

    if output_files:

        def _extract_round(path: str) -> int:
            match = re.search(r"round(\d+)_", Path(path).name)
            return int(match.group(1)) if match else -1

        latest_file = max(output_files, key=_extract_round)
        content = Path(latest_file).read_text()
        (results_dir / f"{phase}_final_output.txt").write_text(content)
        logger.info("[%s] Final output from: %s", phase, latest_file)
    else:
        (results_dir / f"{phase}_final_output.txt").write_text(
            str(result.base_response)
        )

    # Save consensus summary
    summary = {
        "phase": phase,
        "consensus_achieved": result.consensus_achieved,
        "total_iterations": result.total_iterations,
    }
    (results_dir / f"{phase}_consensus_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Save per-iteration files and full history
    history = []
    for attempt in result.consensus_history:
        attempt_data = {
            "attempt": attempt.attempt,
            "consensus_reached": attempt.consensus_reached,
            "final_output_length": len(attempt.final_output),
            "iterations": [],
        }
        for iter_rec in attempt.iterations:
            iter_data = {
                "iteration": iter_rec.iteration,
                "consensus_reached": iter_rec.consensus_reached,
                "review_feedback": iter_rec.review_feedback,
                "counter_feedback": iter_rec.counter_feedback,
            }
            attempt_data["iterations"].append(iter_data)

            prefix = f"{phase}_attempt_{attempt.attempt}_iter_{iter_rec.iteration}"
            (results_dir / f"{prefix}_proposal.txt").write_text(iter_rec.base_output)
            if iter_rec.review_feedback is not None:
                (results_dir / f"{prefix}_review.json").write_text(
                    json.dumps(iter_rec.review_feedback, indent=2)
                )
            if iter_rec.counter_feedback is not None:
                (results_dir / f"{prefix}_counter_feedback.json").write_text(
                    json.dumps(iter_rec.counter_feedback, indent=2)
                )

        history.append(attempt_data)

    (results_dir / f"{phase}_consensus_history.json").write_text(
        json.dumps(history, indent=2, default=str)
    )


def print_summary(workspace: Path, result: PlanThenImplementResponse) -> None:
    """Print a concise summary to stdout."""
    plan_status = "N/A"
    impl_status = "N/A"

    from agent_foundation.common.inferencers.agentic_inferencers.common import (
        DualInferencerResponse,
    )

    if isinstance(result.plan_response, DualInferencerResponse):
        plan_status = (
            "ACHIEVED" if result.plan_response.consensus_achieved else "NOT ACHIEVED"
        )

    if result.plan_approved is False:
        impl_status = "SKIPPED (plan rejected)"
    elif isinstance(result.executor_output, DualInferencerResponse):
        impl_status = (
            "ACHIEVED" if result.executor_output.consensus_achieved else "NOT ACHIEVED"
        )

    approval_str = {
        None: "Not required",
        True: "Approved",
        False: "Rejected",
    }.get(result.plan_approved, "Unknown")

    click.echo("")
    click.echo("=" * 60)
    click.echo("PlanThenImplementInferencer Test Complete")
    click.echo("=" * 60)
    click.echo(f"Workspace:         {workspace}")
    click.echo(f"Plan consensus:    {plan_status}")
    click.echo(f"Human approval:    {approval_str}")
    click.echo(f"Impl consensus:    {impl_status}")
    click.echo(f"Meta-iterations:   {result.total_meta_iterations}")
    click.echo(f"Exhausted:         {result.meta_iterations_exhausted}")
    click.echo(f"Outputs:           {workspace / 'outputs'}")
    click.echo(f"Results:           {workspace / 'results'}")
    click.echo("=" * 60)
    click.echo("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--request",
    "-r",
    required=True,
    help="Request string, or path to a file containing the request.",
)
@click.option(
    "--workspace",
    "-w",
    default=None,
    help="Workspace directory. Default: ./_workspace_test_plan_then_implement/{datetime}/",
)
@click.option(
    "--plan-base-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="claude_code",
    help="Inferencer for the plan phase base (proposer) role.",
)
@click.option(
    "--plan-review-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="claude_code",
    help="Inferencer for the plan phase reviewer role.",
)
@click.option(
    "--impl-base-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="claude_code",
    help="Inferencer for the implementation phase base (proposer) role.",
)
@click.option(
    "--impl-review-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="claude_code",
    help="Inferencer for the implementation phase reviewer role.",
)
@click.option(
    "--require-approval",
    is_flag=True,
    default=False,
    help="Require human approval between plan and implementation phases.",
)
@click.option(
    "--root-folder",
    default=None,
    help="Working directory for agentic inferencers. Default: cwd.",
)
@click.option(
    "--model",
    default=None,
    help="Model ID/name for the inferencers.",
)
@click.option(
    "--max-iterations", default=5, help="Max consensus iterations per DualInferencer."
)
@click.option("--max-attempts", default=1, help="Max fresh-start consensus attempts.")
@click.option(
    "--system-prompt",
    default="",
    help="System prompt for ClaudeCodeInferencer (ignored by devmate).",
)
@click.option(
    "--timeout", default=1800, help="Per-message idle timeout seconds for inferencers."
)
@click.option(
    "--tool-use-timeout",
    default=7200,
    help="Idle timeout seconds during tool-use activity (0 = use --timeout).",
)
@click.option(
    "--consensus-threshold",
    type=click.Choice([s.name for s in Severity]),
    default="COSMETIC",
    help="Max acceptable severity for consensus.",
)
@click.option(
    "--no-counter-feedback",
    is_flag=True,
    default=False,
    help="Disable counter-feedback from fixer.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Python logging level.",
)
# New multi-iteration / analysis options
@click.option(
    "--enable-planning/--no-planning",
    default=True,
    help="Enable/disable the planning phase.",
)
@click.option(
    "--enable-implementation/--no-implementation",
    default=True,
    help="Enable/disable the implementation phase.",
)
@click.option(
    "--enable-analysis",
    is_flag=True,
    default=False,
    help="Enable the analysis phase after implementation.",
)
@click.option(
    "--enable-multiple-iterations",
    is_flag=True,
    default=False,
    help="Enable multi-iteration refinement loop (implies --enable-analysis).",
)
@click.option(
    "--max-meta-iterations",
    default=3,
    help="Maximum number of meta-iterations (plan+impl+analysis cycles).",
)
@click.option(
    "--analyzer-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default=None,
    help="Inferencer type for the analysis phase.",
)
@click.option(
    "--resume-workspace",
    default=None,
    help="Path to an existing workspace to resume from.",
)
def main(
    request: str,
    workspace: str | None,
    plan_base_inferencer: str,
    plan_review_inferencer: str,
    impl_base_inferencer: str,
    impl_review_inferencer: str,
    require_approval: bool,
    root_folder: str | None,
    model: str | None,
    max_iterations: int,
    max_attempts: int,
    system_prompt: str,
    timeout: int,
    tool_use_timeout: int,
    consensus_threshold: str,
    no_counter_feedback: bool,
    log_level: str,
    enable_planning: bool,
    enable_implementation: bool,
    enable_analysis: bool,
    enable_multiple_iterations: bool,
    max_meta_iterations: int,
    analyzer_inferencer: str | None,
    resume_workspace: str | None,
) -> None:
    """Run PlanThenImplementInferencer with two chained DualInferencers."""
    # 1. Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # 2. Resolve request
    if os.path.isfile(request):
        logger.info("Reading request from file: %s", request)
        request_text = Path(request).read_text()
    else:
        request_text = request

    # 3. Set up workspace (always create a timestamp subfolder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resume_workspace is not None:
        workspace_path = Path(resume_workspace)
    elif workspace is None:
        workspace_path = Path(f"./_workspace_test_plan_then_implement/{timestamp}")
    else:
        workspace_path = Path(workspace) / timestamp

    logger.info("Workspace: %s", workspace_path)

    if resume_workspace is None:
        setup_workspace(workspace_path, request_text, enable_analysis=enable_analysis)

    # 4. Resolve root folder
    if root_folder is None:
        root_folder = os.getcwd()

    # 5. Set up structured logging
    logs_dir = workspace_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = workspace_path / "_runtime" / "inferencer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_folder = str(cache_dir)

    tmp_offload_dir = workspace_path / "_runtime" / "tmp_output_files"
    tmp_offload_dir.mkdir(parents=True, exist_ok=True)
    large_arg_temp_dir = str(tmp_offload_dir)

    json_logger = JsonLogger(
        file_path=str(logs_dir / "session.jsonl"),
        append=True,
        is_artifact=True,
        parts_min_size=0,
        space_ext_mode=SpaceExtMode.MOVE,
        parts_file_namer=lambda obj: obj.get("type", "")
        if isinstance(obj, dict)
        else "",
    )
    dual_logger = [
        (json_logger, LoggerConfig(pass_item_key_as="parts_key_path_root")),
        print,
    ]

    enable_counter_feedback = not no_counter_feedback
    consensus_config = ConsensusConfig(
        max_iterations=max_iterations,
        max_consensus_attempts=max_attempts,
        consensus_threshold=Severity[consensus_threshold],
        enable_counter_feedback=enable_counter_feedback,
    )

    # 6. Build sub-inferencers for both phases
    logger.info(
        "Building plan inferencers: base=%s, review=%s",
        plan_base_inferencer,
        plan_review_inferencer,
    )
    plan_base_inf = create_inferencer(
        plan_base_inferencer,
        root_folder,
        model,
        system_prompt,
        timeout,
        inferencer_logger=dual_logger,
        inferencer_id=f"plan_{plan_base_inferencer}_base",
        cache_folder=cache_folder,
        large_arg_temp_dir=large_arg_temp_dir,
        tool_use_timeout=tool_use_timeout,
    )
    plan_review_inf = create_inferencer(
        plan_review_inferencer,
        root_folder,
        model,
        system_prompt,
        timeout,
        inferencer_logger=dual_logger,
        inferencer_id=f"plan_{plan_review_inferencer}_review",
        cache_folder=cache_folder,
        large_arg_temp_dir=large_arg_temp_dir,
        tool_use_timeout=tool_use_timeout,
    )

    logger.info(
        "Building impl inferencers: base=%s, review=%s",
        impl_base_inferencer,
        impl_review_inferencer,
    )
    impl_base_inf = create_inferencer(
        impl_base_inferencer,
        root_folder,
        model,
        system_prompt,
        timeout,
        inferencer_logger=dual_logger,
        inferencer_id=f"impl_{impl_base_inferencer}_base",
        cache_folder=cache_folder,
        large_arg_temp_dir=large_arg_temp_dir,
        tool_use_timeout=tool_use_timeout,
    )
    impl_review_inf = create_inferencer(
        impl_review_inferencer,
        root_folder,
        model,
        system_prompt,
        timeout,
        inferencer_logger=dual_logger,
        inferencer_id=f"impl_{impl_review_inferencer}_review",
        cache_folder=cache_folder,
        large_arg_temp_dir=large_arg_temp_dir,
        tool_use_timeout=tool_use_timeout,
    )

    # Build analyzer inferencer if requested
    analyzer_inf = None
    if analyzer_inferencer is not None:
        logger.info("Building analyzer inferencer: %s", analyzer_inferencer)
        analyzer_inf = create_inferencer(
            analyzer_inferencer,
            root_folder,
            model,
            system_prompt,
            timeout,
            inferencer_logger=dual_logger,
            inferencer_id=f"analyzer_{analyzer_inferencer}",
            cache_folder=cache_folder,
            large_arg_temp_dir=large_arg_temp_dir,
            tool_use_timeout=tool_use_timeout,
        )

    # 7. Create TemplateManagers for each phase
    plan_tm = TemplateManager(
        templates=str(workspace_path / "prompt_templates"),
        active_template_root_space="plan",
        enable_templated_feed=True,
    )
    impl_tm = TemplateManager(
        templates=str(workspace_path / "prompt_templates"),
        active_template_root_space="implementation",
        enable_templated_feed=True,
    )

    # 8. Build output path templates
    output_dir = workspace_path.resolve() / "outputs"
    plan_output_path = str(output_dir) + "/round{{ round_index }}_plan.md"
    impl_output_path = str(output_dir) + "/round{{ round_index }}_implementation.md"

    # 9. Construct DualInferencers
    plan_dual = DualInferencer(
        base_inferencer=plan_base_inf,
        review_inferencer=plan_review_inf,
        consensus_config=consensus_config,
        prompt_formatter=plan_tm,
        initial_prompt="initial",
        review_prompt="review",
        followup_prompt="followup",
        placeholder_proposal="main_response",
        phase="plan",
        response_parser=extract_delimited,
        logger=dual_logger,
        debug_mode=True,
        id="PlanDualInferencer",
    )

    impl_dual = DualInferencer(
        base_inferencer=impl_base_inf,
        review_inferencer=impl_review_inf,
        consensus_config=consensus_config,
        prompt_formatter=impl_tm,
        initial_prompt="initial",
        review_prompt="review",
        followup_prompt="followup",
        placeholder_proposal="main_response",
        phase="implementation",
        response_parser=extract_delimited,
        logger=dual_logger,
        debug_mode=True,
        id="ImplDualInferencer",
    )

    # 10. Set up human approval (if requested)
    interactive = None
    if require_approval:
        interactive = TerminalInteractive(
            system_name="PlanThenImplement",
            user_name="Human",
        )

    # 11. Construct PlanThenImplementInferencer
    pti_kwargs = dict(
        planner_inferencer=plan_dual,
        executor_inferencer=impl_dual,
        interactive=interactive,
        planner_phase="plan",
        executor_phase="implementation",
        enable_planning=enable_planning,
        enable_implementation=enable_implementation,
        enable_analysis=enable_analysis,
        enable_multiple_iterations=enable_multiple_iterations,
        max_meta_iterations=max_meta_iterations,
        logger=dual_logger,
        debug_mode=True,
        id="PlanThenImplementInferencer",
    )

    if analyzer_inf is not None:
        pti_kwargs["analyzer_inferencer"] = analyzer_inf

    # Always pass workspace_path so Tier 1 (Workflow JSON) checkpoints are
    # enabled for every run, not just analysis/multi-iteration modes.  This
    # makes --resume-workspace recovery precise (exact step + state) instead
    # of relying on the less-reliable Tier 2 file-existence heuristic.
    pti_kwargs["workspace_path"] = str(workspace_path)

    if resume_workspace is not None:
        pti_kwargs["resume_workspace"] = resume_workspace

    pti = PlanThenImplementInferencer(**pti_kwargs)

    # 12. Run with proper lifecycle management
    async def run():
        async with pti:
            return await pti.ainfer(
                request_text,
                inference_config={
                    "plan_config": {"output_path": plan_output_path},
                    "implement_config": {"output_path": impl_output_path},
                },
            )

    logger.info("Starting PlanThenImplementInferencer...")
    result = asyncio.run(run())

    # 13. Save results
    results_dir = workspace_path / "results"
    output_dir_path = workspace_path / "outputs"

    # Save plan phase results
    if result.plan_response is not None:
        save_phase_results(results_dir, output_dir_path, "plan", result.plan_response)

    # Save implementation phase results
    if result.executor_output is not None:
        save_phase_results(
            results_dir, output_dir_path, "implementation", result.executor_output
        )

    # Save overall summary
    overall_summary = {
        "plan_approved": result.plan_approved,
        "plan_output_length": len(result.plan_output),
        "final_output_length": len(str(result.base_response)),
        "plan_base_inferencer": plan_base_inferencer,
        "plan_review_inferencer": plan_review_inferencer,
        "impl_base_inferencer": impl_base_inferencer,
        "impl_review_inferencer": impl_review_inferencer,
        "max_iterations": max_iterations,
        "max_attempts": max_attempts,
        "consensus_threshold": consensus_threshold,
        "enable_counter_feedback": enable_counter_feedback,
        "require_approval": require_approval,
        "enable_planning": enable_planning,
        "enable_implementation": enable_implementation,
        "enable_analysis": enable_analysis,
        "enable_multiple_iterations": enable_multiple_iterations,
        "max_meta_iterations": max_meta_iterations,
        "total_meta_iterations": result.total_meta_iterations,
        "meta_iterations_exhausted": result.meta_iterations_exhausted,
    }
    (results_dir / "overall_summary.json").write_text(
        json.dumps(overall_summary, indent=2)
    )

    # Save final output
    (results_dir / "final_output.txt").write_text(str(result.base_response))

    # 14. Print summary
    print_summary(workspace_path, result)


if __name__ == "__main__":
    main()

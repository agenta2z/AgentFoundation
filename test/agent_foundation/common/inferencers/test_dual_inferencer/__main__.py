"""Manual test script for DualInferencer.

Creates a workspace, writes adapted prompt templates, runs the DualInferencer
consensus loop with configurable external inferencers, and saves results.

Usage:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_dual_inferencer -- \
        --request "Design a REST API for user management" \
        --template-version plan \
        --base-inferencer claude_code \
        --review-inferencer claude_code \
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
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)
from agent_foundation.common.response_parsers import extract_delimited
from rich_python_utils.common_objects.debuggable import LoggerConfig
from rich_python_utils.io_utils.json_io import JsonLogger, SpaceExtMode
from rich_python_utils.string_utils.formatting.template_manager import (
    TemplateManager,
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
}

INFERENCER_CHOICES = ["claude_code", "devmate_sdk", "devmate_cli"]


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
) -> InferencerBase:
    """Build an inferencer instance based on type string.

    Uses absolute imports to avoid import-path issues.
    """
    if inferencer_type == "claude_code":
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_inferencer import (  # noqa: E501
            ClaudeCodeInferencer,
        )

        return ClaudeCodeInferencer(
            root_folder=root_folder,
            model_id=model or "",
            system_prompt=system_prompt,
            idle_timeout_seconds=timeout,
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
            logger=inferencer_logger,
            id=inferencer_id,
            cache_folder=cache_folder,
        )
    elif inferencer_type == "devmate_sdk":
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (  # noqa: E501
            DevmateSDKInferencer,
        )

        kwargs: dict = dict(
            root_folder=root_folder,
            total_timeout_seconds=timeout,
            idle_timeout_seconds=timeout,
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

        kwargs = dict(repo_path=root_folder)
        if model is not None:
            kwargs["model_name"] = model
        if inferencer_logger is not None:
            kwargs["logger"] = inferencer_logger
        if inferencer_id is not None:
            kwargs["id"] = inferencer_id
        if cache_folder is not None:
            kwargs["cache_folder"] = cache_folder
        return DevmateCliInferencer(**kwargs)
    else:
        raise ValueError(f"Unknown inferencer type: {inferencer_type}")


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def setup_workspace(workspace: Path, request_text: str) -> None:
    """Create workspace directory structure and write template files."""
    # Create directories (with "main" subdirectory to match TemplateManager's
    # active_template_type="main" default, producing flat keys like "plan/main")
    (workspace / "prompt_templates" / "plan" / "main").mkdir(
        parents=True, exist_ok=True
    )
    (workspace / "prompt_templates" / "implementation" / "main").mkdir(
        parents=True, exist_ok=True
    )
    (workspace / "results").mkdir(parents=True, exist_ok=True)
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)

    # Save original request
    (workspace / "request.txt").write_text(request_text)

    # Write adapted template files
    for version, templates in TEMPLATES.items():
        for role, content in templates.items():
            template_path = (
                workspace / "prompt_templates" / version / "main" / f"{role}.jinja2"
            )
            template_path.write_text(content)
            logger.info("Wrote template: %s", template_path)


def save_results(workspace: Path, result, cli_config: dict) -> None:
    """Save DualInferencerResponse results to workspace/results/.

    Scans the outputs directory for files matching the template version
    pattern and copies the one with the largest round index as the final output.
    """
    results_dir = workspace / "results"
    output_dir = workspace / "outputs"
    template_version = cli_config.get("template_version", "")

    # Find the final output file by scanning for largest round index
    pattern = str(output_dir / f"round*_{template_version}.md")
    output_files = glob.glob(pattern)

    if output_files:

        def _extract_round(path: str) -> int:
            match = re.search(r"round(\d+)_", Path(path).name)
            return int(match.group(1)) if match else -1

        latest_file = max(output_files, key=_extract_round)
        content = Path(latest_file).read_text()
        (results_dir / "final_output.txt").write_text(content)
        logger.info("Final output from: %s", latest_file)
    else:
        # Fallback: save the raw response
        (results_dir / "final_output.txt").write_text(str(result.base_response))

    # Save summary
    summary = {
        "consensus_achieved": result.consensus_achieved,
        "total_iterations": result.total_iterations,
        "phase": result.phase,
        **cli_config,
    }
    (results_dir / "consensus_summary.json").write_text(json.dumps(summary, indent=2))

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

            # Save individual iteration files
            prefix = f"attempt_{attempt.attempt}_iter_{iter_rec.iteration}"
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

    (results_dir / "consensus_history.json").write_text(
        json.dumps(history, indent=2, default=str)
    )


def print_summary(workspace: Path, result) -> None:
    """Print a concise summary to stdout."""
    status = "ACHIEVED" if result.consensus_achieved else "NOT ACHIEVED"
    attempts = len(result.consensus_history)
    click.echo("")
    click.echo("=" * 55)
    click.echo("DualInferencer Test Complete")
    click.echo("=" * 55)
    click.echo(f"Workspace:    {workspace}")
    click.echo(f"Phase:        {result.phase}")
    click.echo(
        f"Consensus:    {status} "
        f"({result.total_iterations} iterations, {attempts} attempt(s))"
    )
    click.echo(f"Outputs:      {workspace / 'outputs'}")
    click.echo(f"Final output: {workspace / 'results' / 'final_output.txt'}")
    click.echo("=" * 55)
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
    help="Workspace directory. Default: ./_workspace/test_dual_agent_{datetime}/",
)
@click.option(
    "--template-version",
    "-t",
    type=click.Choice(["plan", "implementation"]),
    default="plan",
    help="Which adapted template set to use.",
)
@click.option(
    "--base-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="claude_code",
    help="Inferencer for the base (proposer) role.",
)
@click.option(
    "--review-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="claude_code",
    help="Inferencer for the reviewer role.",
)
@click.option(
    "--fixer-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default=None,
    help="Inferencer for the fixer role. Omit to reuse base inferencer.",
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
    "--max-iterations", default=5, help="Max consensus iterations per attempt."
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
def main(
    request: str,
    workspace: str | None,
    template_version: str,
    base_inferencer: str,
    review_inferencer: str,
    fixer_inferencer: str | None,
    root_folder: str | None,
    model: str | None,
    max_iterations: int,
    max_attempts: int,
    system_prompt: str,
    timeout: int,
    consensus_threshold: str,
    no_counter_feedback: bool,
    log_level: str,
) -> None:
    """Run DualInferencer consensus loop with configurable inferencers."""
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

    # 3. Set up workspace
    if workspace is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_path = Path(f"./_workspace/test_dual_agent_{timestamp}")
    else:
        workspace_path = Path(workspace)

    logger.info("Workspace: %s", workspace_path)
    setup_workspace(workspace_path, request_text)

    # 4. Resolve root folder
    if root_folder is None:
        root_folder = os.getcwd()

    # 5. Set up JsonLogger for structured prompt/response logging
    logs_dir = workspace_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 5b. Set up inferencer cache directory
    cache_dir = workspace_path / "_runtime" / "inferencer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_folder = str(cache_dir)

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

    # 6. Build inferencers
    logger.info(
        "Building inferencers: base=%s, review=%s, fixer=%s",
        base_inferencer,
        review_inferencer,
        fixer_inferencer,
    )

    base_inf = create_inferencer(
        base_inferencer,
        root_folder,
        model,
        system_prompt,
        timeout,
        inferencer_logger=dual_logger,
        inferencer_id=f"{base_inferencer}_base",
        cache_folder=cache_folder,
    )
    review_inf = create_inferencer(
        review_inferencer,
        root_folder,
        model,
        system_prompt,
        timeout,
        inferencer_logger=dual_logger,
        inferencer_id=f"{review_inferencer}_review",
        cache_folder=cache_folder,
    )
    fixer_inf = None
    if fixer_inferencer is not None:
        fixer_inf = create_inferencer(
            fixer_inferencer,
            root_folder,
            model,
            system_prompt,
            timeout,
            inferencer_logger=dual_logger,
            inferencer_id=f"{fixer_inferencer}_fixer",
            cache_folder=cache_folder,
        )

    # 7. Create TemplateManager from prompt_templates directory
    prompt_tm = TemplateManager(
        templates=str(workspace_path / "prompt_templates"),
        active_template_root_space=template_version,
        enable_templated_feed=True,
    )

    # 8. Output path template (uses Jinja2 syntax for round_index)
    # Must be absolute so Claude Code (with arbitrary root_folder) can find it
    output_dir = workspace_path.resolve() / "outputs"
    output_path = (
        str(output_dir) + "/round{{ round_index }}_" + template_version + ".md"
    )

    # 9. Construct DualInferencer
    enable_counter_feedback = not no_counter_feedback

    dual = DualInferencer(
        base_inferencer=base_inf,
        review_inferencer=review_inf,
        fixer_inferencer=fixer_inf,
        consensus_config=ConsensusConfig(
            max_iterations=max_iterations,
            max_consensus_attempts=max_attempts,
            consensus_threshold=Severity[consensus_threshold],
            enable_counter_feedback=enable_counter_feedback,
        ),
        prompt_formatter=prompt_tm,
        initial_prompt="initial",
        review_prompt="review",
        followup_prompt="followup",
        placeholder_proposal="main_response",
        phase=template_version,
        response_parser=extract_delimited,
        logger=dual_logger,
        debug_mode=True,
        id="DualInferencer",
    )

    # 10. Run with proper lifecycle management
    async def run():
        async with dual:
            return await dual.ainfer(
                request_text,
                inference_config={"output_path": output_path},
            )

    logger.info("Starting DualInferencer consensus loop...")
    result = asyncio.run(run())

    # 11. Save results
    cli_config = {
        "template_version": template_version,
        "base_inferencer": base_inferencer,
        "review_inferencer": review_inferencer,
        "fixer_inferencer": fixer_inferencer,
        "model": model,
        "max_iterations": max_iterations,
        "max_attempts": max_attempts,
        "consensus_threshold": consensus_threshold,
        "enable_counter_feedback": enable_counter_feedback,
    }
    save_results(workspace_path, result, cli_config)

    # 12. Print summary
    print_summary(workspace_path, result)


if __name__ == "__main__":
    main()

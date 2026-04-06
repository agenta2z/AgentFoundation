

"""Manual E2E integration test for /understand-codebase workflow.

Exercises the full DualInferencerBridge flow with template_version="understand_codebase",
using devmate_cli (or other inferencer) for both base and reviewer roles.

Usage:
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_understand_codebase -- \
        --workflow-target-path /path/to/codebase

    # Planning only (skip implementation/documentation generation):
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_understand_codebase -- \
        --workflow-target-path /path/to/codebase --no-implementation

    # Implementation only (skip planning, use existing plan):
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_understand_codebase -- \
        --workflow-target-path /path/to/codebase --no-planning \
        --workspace /path/to/previous/workspace

    # Initial plan only (no review loop, for iterating on prompt quality):
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_understand_codebase -- \
        --workflow-target-path /path/to/codebase --initial-plan-only

    # Custom inferencers and root folder:
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_understand_codebase -- \
        --workflow-target-path /path/to/codebase \
        --base-inferencer claude_code \
        --review-inferencer claude_code \
        --root-folder /path/to/repo
"""

import asyncio
import logging
import os
from pathlib import Path

import click

logger = logging.getLogger(__name__)

INFERENCER_CHOICES = ["claude_code", "claude_code_cli", "devmate_sdk", "devmate_cli"]


@click.command()
@click.option(
    "--workflow-target-path",
    required=True,
    help="Path to the codebase or file to understand.",
)
@click.option(
    "--root-folder",
    default=None,
    help="Working directory for agentic inferencers. Defaults to parent of workflow-target-path.",
)
@click.option(
    "--base-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="devmate_cli",
    help="Inferencer for the base (proposer) role.",
)
@click.option(
    "--review-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="devmate_cli",
    help="Inferencer for the reviewer role.",
)
@click.option(
    "--workspace",
    "-w",
    default=None,
    help="Workspace directory. Default: auto-generated timestamped folder.",
)
@click.option(
    "--model",
    default=None,
    help="Model ID/name for the inferencers.",
)
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
    "--timeout",
    default=1800,
    help="Per-message idle timeout seconds for inferencers.",
)
@click.option(
    "--max-iterations",
    default=5,
    help="Max consensus iterations per DualInferencer.",
)
@click.option(
    "--initial-plan-only",
    is_flag=True,
    default=False,
    help="Run only the initial plan proposal (no review/consensus loop). Useful for iterating on prompt quality.",
)
@click.option(
    "--initial-plan-file",
    default=None,
    help="Path to an existing plan file. Skips proposal and jumps directly to review of this plan.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Python logging level.",
)
def main(
    workflow_target_path: str,
    root_folder: str | None,
    base_inferencer: str,
    review_inferencer: str,
    workspace: str | None,
    model: str | None,
    enable_planning: bool,
    enable_implementation: bool,
    timeout: int,
    max_iterations: int,
    initial_plan_only: bool,
    initial_plan_file: str | None,
    log_level: str,
) -> None:
    """Run /understand-codebase E2E via DualInferencerBridge."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve root folder
    if root_folder is None:
        target_path = Path(workflow_target_path)
        if target_path.is_file():
            root_folder = str(target_path.parent)
        elif target_path.is_dir():
            root_folder = str(target_path)
        else:
            root_folder = os.getcwd()

    # Build session_context (mirrors how the server passes workflow_target_path)
    session_context = {
        "target_path": root_folder,
        "workflow_target_path": workflow_target_path,
    }

    # Initial-plan-only mode: skip review/consensus, just run the first proposal
    if initial_plan_only:
        max_iterations = 0
        enable_implementation = False

    # Build request text (mirrors command_router.py)
    request_text = workflow_target_path

    click.echo("=" * 60)
    click.echo("Understand Codebase Integration Test")
    click.echo("=" * 60)
    click.echo(f"Target:            {workflow_target_path}")
    click.echo(f"Root folder:       {root_folder}")
    click.echo(f"Base inferencer:   {base_inferencer}")
    click.echo(f"Review inferencer: {review_inferencer}")
    click.echo(f"Planning:          {enable_planning}")
    click.echo(f"Implementation:    {enable_implementation}")
    click.echo(f"Initial plan only: {initial_plan_only}")
    click.echo(f"Max iterations:    {max_iterations}")
    click.echo(f"Resume workspace:  {workspace or '(none — fresh run)'}")
    click.echo(f"Template version:  understand_codebase")
    click.echo("=" * 60)

    # Import here to avoid import overhead if --help
    from agent_foundation.server.dual_inferencer_bridge import DualInferencerBridge
    from agent_foundation.server.task_types import TaskMode

    bridge_kwargs = dict(
        root_folder=Path(root_folder),
        claude_model=model,
        session_context=session_context,
        base_inferencer_type=base_inferencer,
        review_inferencer_type=review_inferencer,
        template_version="understand_codebase",
        enable_planning=enable_planning,
        enable_implementation=enable_implementation,
        max_iterations=max_iterations,
        timeout=timeout,
    )
    if workspace:
        bridge_kwargs["resume_workspace"] = workspace
        bridge_kwargs["copy_workspace"] = False
    if initial_plan_file:
        bridge_kwargs["initial_plan_file"] = initial_plan_file
    bridge = DualInferencerBridge(**bridge_kwargs)

    click.echo(f"Workspace:         {bridge.workspace}")
    click.echo("")

    async def run():
        return await bridge.run(request_text, task_mode=TaskMode.FULL_WORKFLOW)

    logger.info("Starting understand-codebase workflow...")
    result = asyncio.run(run())

    # Print summary
    click.echo("")
    click.echo("=" * 60)
    click.echo("Understand Codebase Test Complete")
    click.echo("=" * 60)
    click.echo(f"Workspace:       {bridge.workspace}")
    click.echo(f"Result length:   {len(result)} chars")
    click.echo(f"Outputs:         {bridge.workspace / 'outputs'}")
    click.echo(f"Results:         {bridge.workspace / 'results'}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()

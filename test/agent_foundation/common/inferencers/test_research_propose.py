# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Manual E2E integration test for /research-propose workflow.

Exercises the full ResearchProposeBridge flow (BreakdownThenAggregate pipeline):
breakdown -> parallel worker research -> unified proposal aggregation.

Usage:
    # Breakdown only (task decomposition, no workers):
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_research_propose -- \
        --workflow-target-path /path/to/code \
        --docs-path /path/to/docs \
        --breakdown-only

    # Custom research query:
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_research_propose -- \
        --workflow-target-path /path/to/codebase \
        --query "How can we improve inference latency for the MHTA module?"

    # Full pipeline with custom max-queries range:
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_research_propose -- \
        --workflow-target-path /path/to/codebase \
        --docs-path /path/to/docs \
        --max-queries "5 to 20"

    # Resume from previous workspace:
    buck2 run --prefer-remote fbcode//rankevolve/test/agentic_foundation:test_research_propose -- \
        --workflow-target-path /path/to/codebase \
        --workspace /path/to/previous/workspace
"""

import asyncio
import logging
import os
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _build_default_query(workflow_target_path: str, docs_path: str | None) -> str:
    """Build the default research query, optionally including docs path."""
    if docs_path:
        lines = [
            f"Analyze the documentation at {docs_path} alongside its",
            f"implementation ({workflow_target_path}) and direct dependencies.",
        ]
    else:
        lines = [
            f"Analyze the code at {workflow_target_path} and all its direct",
            "dependencies.",
        ]
    lines.append("")
    lines.append(
        "Your goal is to identify concrete improvement opportunities worth"
        " deeper investigation."
    )
    lines.append("")
    lines.append("For each opportunity, explain:")
    lines.append("1. What the current behavior or design is")
    lines.append("2. What the gap, inefficiency, or limitation is")
    lines.append(
        "3. Why it matters (impact on performance, reliability,"
        " maintainability, or user experience)"
    )
    lines.append("4. What alternatives exist, and what are the tradeoffs of each?")
    lines.append("5. What a deeper investigation would need to answer")
    lines.append("")
    if docs_path:
        lines.append(
            f"You may leverage existing opportunity studies available under"
            f" {docs_path} if any, but only as a starting point."
        )
    lines.append(
        "DO NOT constrain your horizon to what is already covered. Go as far"
        " as possible with your innovation and imagination and surface"
        " inspiring new opportunities."
    )
    return "\n".join(lines)


INFERENCER_CHOICES = [
    "devmate_cli",
    "devmate_sdk",
    "claude_code",
    "claude_code_cli",
    "metamate_sdk",
    "metamate_cli",
]


@click.command()
@click.option(
    "--workflow-target-path",
    required=True,
    help="Path to the codebase or file to research.",
)
@click.option(
    "--docs-path",
    default=None,
    help="Path to documentation directory (included in default query and session context).",
)
@click.option(
    "--root-folder",
    default=None,
    help="Working directory for agentic inferencers. Defaults to workflow-target-path.",
)
@click.option(
    "--base-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default="devmate_cli",
    help="Inferencer type for breakdown, workers, and aggregation.",
)
@click.option(
    "--query",
    "-q",
    default=None,
    help="Custom research query. Defaults to deep-dive analysis of target path.",
)
@click.option(
    "--max-breakdown",
    default="5 to 20 (or you really want to suggest more)",
    help="Guidance for how many subtasks to decompose into (passed to the LLM).",
)
@click.option(
    "--max-researches",
    type=int,
    default=None,
    help="Hard limit on how many research workers to run (default: all subtasks).",
)
@click.option(
    "--model",
    default=None,
    help="Model ID/name override for the inferencers.",
)
@click.option(
    "--research-only",
    is_flag=True,
    default=False,
    help="Run breakdown + deep research only, skip unified proposal.",
)
@click.option(
    "--breakdown-only",
    is_flag=True,
    default=False,
    help="Run only the task breakdown (decomposition), skip workers and aggregation.",
)
@click.option(
    "--research-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default=None,
    help="Inferencer type for research phase (e.g., metamate_cli for knowledge search). Defaults to --base-inferencer.",
)
@click.option(
    "--proposal-inferencer",
    type=click.Choice(INFERENCER_CHOICES),
    default=None,
    help="Inferencer type for proposal phase (e.g., devmate_cli for code reading). Defaults to --base-inferencer.",
)
@click.option(
    "--workspace",
    "-w",
    default=None,
    help="Resume from a previous workspace directory.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Python logging level.",
)
def main(
    workflow_target_path: str,
    docs_path: str | None,
    root_folder: str | None,
    base_inferencer: str,
    query: str | None,
    max_breakdown: str,
    max_researches: int | None,
    model: str | None,
    research_only: bool,
    breakdown_only: bool,
    research_inferencer: str | None,
    proposal_inferencer: str | None,
    workspace: str | None,
    log_level: str,
) -> None:
    """Run /research-propose E2E via ResearchProposeBridge."""
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

    # Use default query if none provided
    request_text = query or _build_default_query(workflow_target_path, docs_path)

    # Build session context (mirrors how the server passes context)
    session_context: dict[str, str] = {
        "target_path": root_folder,
        "workflow_target_path": workflow_target_path,
    }
    if docs_path:
        session_context["docs_path"] = docs_path

    click.echo("=" * 60)
    click.echo("Research-Propose Integration Test")
    click.echo("=" * 60)
    click.echo(f"Target:           {workflow_target_path}")
    if docs_path:
        click.echo(f"Docs path:        {docs_path}")
    click.echo(f"Root folder:      {root_folder}")
    click.echo(f"Base inferencer:  {base_inferencer}")
    if research_inferencer:
        click.echo(f"Research inferencer: {research_inferencer}")
    if proposal_inferencer:
        click.echo(f"Proposal inferencer: {proposal_inferencer}")
    click.echo(f"Max breakdown:    {max_breakdown}")
    if max_researches is not None:
        click.echo(f"Max researches:   {max_researches}")
    click.echo(f"Research only:    {research_only}")
    click.echo(f"Breakdown only:   {breakdown_only}")
    click.echo(f"Resume workspace: {workspace or '(none - fresh run)'}")
    click.echo(f"Query:            {request_text[:80]}...")
    click.echo("=" * 60)

    # Import here to avoid import overhead if --help
    from agent_foundation.server.research_propose_bridge import ResearchProposeBridge

    bridge = ResearchProposeBridge(
        root_folder=Path(root_folder),
        model=model,
        research_only=research_only,
        breakdown_only=breakdown_only,
        max_breakdown=max_breakdown,
        max_researches=max_researches,
        session_context=session_context,
        resume_workspace=workspace,
        base_inferencer_type=base_inferencer,
        research_inferencer_type=research_inferencer,
        proposal_inferencer_type=proposal_inferencer,
    )

    click.echo(f"Workspace:        {bridge.workspace}")
    click.echo("")

    async def run() -> str:
        return await bridge.run(request_text)

    logger.info("Starting research-propose workflow...")
    result = asyncio.run(run())

    # Print summary
    click.echo("")
    click.echo("=" * 60)
    click.echo("Research-Propose Test Complete")
    click.echo("=" * 60)
    click.echo(f"Workspace:      {bridge.workspace}")
    click.echo(f"Result length:  {len(result)} chars")
    click.echo(f"Outputs:        {bridge.workspace / 'outputs'}")

    # Validate outputs
    for name in ("breakdown_result.md", "final_result.md"):
        output_file = bridge.workspace / "outputs" / name
        if output_file.exists():
            click.echo(f"Output file:    {output_file} (exists)")

    click.echo("=" * 60)

    # Print first 500 chars of result as preview
    if result:
        click.echo("")
        click.echo("Result preview:")
        click.echo("-" * 40)
        click.echo(result[:500])
        if len(result) > 500:
            click.echo(f"... ({len(result) - 500} more chars)")
        click.echo("-" * 40)


if __name__ == "__main__":
    main()

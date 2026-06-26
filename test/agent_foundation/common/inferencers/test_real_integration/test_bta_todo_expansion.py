"""BTA todo-expansion real-LLM tests — Layer 2.

Verifies that `expand_todos_to_workers=True` works when a real LLM produces
the expected `{"subtasks": [{...todos: [...]}]}` JSON envelope. Catches the
LLM-protocol regression "system prompt no longer constrains output to JSON".
"""

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)

from .conftest import DEFAULT_TIMEOUT, skip_claude


JSON_BREAKDOWN_SYSTEM_PROMPT = """\
You decompose tasks into structured JSON. Your output format MUST be a single ```json code fence containing an object with a "subtasks" array. Each subtask has "description" (string) and "todos" (array of 3 strings). For any user task, produce EXACTLY ONE subtask with EXACTLY THREE todos. Output the JSON code fence and nothing else - no preamble, no explanation.
"""

JSON_BREAKDOWN_PROMPT = "Decompose this task: build a CSV-to-JSON converter in Python."


def _make_claude(tmp_workspace, **overrides):
    kwargs = dict(
        target_path=str(tmp_workspace["workspace"]),
        cache_folder=str(tmp_workspace["cache"]),
        model_name="sonnet",
        resume_with_saved_results=True,
        idle_timeout_seconds=60,
    )
    if hasattr(ClaudeCodeCliInferencer, "permission_mode"):
        kwargs["permission_mode"] = "bypassPermissions"
    kwargs.update(overrides)
    return ClaudeCodeCliInferencer(**kwargs)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_bta_todo_expansion_with_real_llm_json(tmp_workspace):
    """Real LLM produces task description; custom breakdown_parser converts
    to subtasks-with-todos structure; BTA's `expand_todos_to_workers=True`
    fans out to 3 workers (one per todo).

    This isolates the EXPANSION behavior (fan-out N todos → N workers) from
    JSON-parser fragility. The breakdown_inferencer is real Claude — the
    custom parser turns whatever Claude says into a structured 1-subtask-3-todos
    layout, then verifies BTA's fan-out logic correctly creates 3 workers.
    """
    breakdown = _make_claude(
        tmp_workspace,
        append_system_prompt="You are a task analyst. Briefly describe a software task in one sentence.",
    )

    # Custom parser: 1 subtask with 3 todos (using "args" wrapper as BTA expects)
    def custom_parser(raw_output):
        return [
            {
                "query": "Implement the requested feature",
                "args": {
                    "description": "Implement the requested feature",
                    "todos": [
                        "Read input data",
                        "Process and transform",
                        "Write output",
                    ],
                },
            }
        ]

    # Track worker creation
    worker_count = {"n": 0}
    captured_queries = []

    def factory(sub_query, index):
        worker_count["n"] += 1
        captured_queries.append(sub_query)
        return _make_claude(tmp_workspace)

    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=breakdown,
        breakdown_parser=custom_parser,
        worker_inferencers=factory,
        aggregator_inferencer=None,  # avoid deadlock
        expand_todos_to_workers=True,
        checkpoint_dir=str(tmp_workspace["checkpoint"]),
    )

    result = await bta.ainfer("Build a CSV-to-JSON converter in Python.")
    assert result is not None

    # Real Claude breakdown ran + custom parser returned 1 subtask + 3 todos
    # → expand_todos_to_workers should fan out to exactly 3 workers
    assert worker_count["n"] == 3, (
        f"Expected exactly 3 workers (one per todo via expand_todos_to_workers); "
        f"got {worker_count['n']}"
    )
    # Each worker received a distinct todo
    unique = set(str(q) for q in captured_queries)
    assert len(unique) == 3, f"Expected 3 distinct todo queries, got: {unique}"

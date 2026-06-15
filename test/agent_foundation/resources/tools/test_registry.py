"""Commit 6a: bridge dispatcher publishes a collision-free workspace key.

``derived_tool_execute`` augments the generic ``workspace_path`` context-update
emitted by ``task_execute`` with a tool-name-suffixed copy, so multi-bridge SOPs
can address each workspace individually from the SOP body via Jinja.
"""
import asyncio

from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.resources.tools import registry


def _patch_task_execute(monkeypatch, return_value):
    async def fake_execute(task_args, ctx):
        if callable(return_value):
            return return_value(task_args, ctx)
        return return_value

    monkeypatch.setattr(
        "agent_foundation.resources.tools.task.executor.execute", fake_execute
    )


def _run(tool_name, monkeypatch, return_value):
    _patch_task_execute(monkeypatch, return_value)
    return asyncio.run(
        registry.derived_tool_execute(
            {"request": "x"}, {}, derived_from={"tool": "task"}, tool_name=tool_name
        )
    )


def test_emits_suffixed_key_alongside_generic(monkeypatch):
    result = _run(
        "research_propose",
        monkeypatch,
        ToolExecutionResult(
            result="done", context_updates={"workspace_path": "/tmp/foo", "success": True}
        ),
    )
    assert result.context_updates["workspace_path"] == "/tmp/foo"
    assert result.context_updates["workspace_path__research_propose"] == "/tmp/foo"
    # Generic key preserved untouched (back-compat).
    assert result.context_updates["success"] is True


def test_non_toolexecutionresult_passthrough(monkeypatch):
    """Legacy/raw return value must pass through unchanged (no AttributeError)."""
    result = _run("research_propose", monkeypatch, "raw string result")
    assert result == "raw string result"


def test_no_workspace_path_no_suffix_added(monkeypatch):
    result = _run(
        "research_propose",
        monkeypatch,
        ToolExecutionResult(result="ok", context_updates={"success": True}),
    )
    assert "workspace_path__research_propose" not in result.context_updates
    assert result.context_updates == {"success": True}


def test_hyphenated_tool_name_is_canonicalised(monkeypatch):
    """Defensive guard: a hyphen alias still yields a valid Jinja identifier key."""
    result = _run(
        "research-propose",
        monkeypatch,
        ToolExecutionResult(result="ok", context_updates={"workspace_path": "/w"}),
    )
    assert result.context_updates["workspace_path__research_propose"] == "/w"
    assert "workspace_path__research-propose" not in result.context_updates


def test_proposal_selection_tool_json_loads():
    """Commit 4: the new conversation tool ships a loadable tool.json."""
    from agent_foundation.resources.tools.registry import load_all_tools

    tools = load_all_tools()
    assert "proposal_selection" in tools
    t = tools["proposal_selection"]
    assert t.tool_type == "Conversation"
    assert getattr(t, "yolo_default", None) == {"mode": "select_all"}


def test_distinct_keys_for_distinct_bridges(monkeypatch):
    rp = _run(
        "research_propose",
        monkeypatch,
        ToolExecutionResult(result="ok", context_updates={"workspace_path": "/rp"}),
    )
    tk = _run(
        "task",
        monkeypatch,
        ToolExecutionResult(result="ok", context_updates={"workspace_path": "/tk"}),
    )
    assert rp.context_updates["workspace_path__research_propose"] == "/rp"
    assert tk.context_updates["workspace_path__task"] == "/tk"

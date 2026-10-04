"""Conversational tool bodies run at a ``tool/<name>`` child of the turn's ctx."""

import asyncio
import types

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    RunContext,
)
from agent_foundation.common.inferencers.run_context.bridge import enter_run, exit_run
from agent_foundation.resources.tools.models import ToolDefinition
from attr import attrs

SEEN = []


@pytest.fixture(autouse=True)
def _fresh_seen():
    SEEN.clear()
    yield
    SEEN.clear()


@attrs(slots=False)
class _NoopBase(InferencerBase):
    def _infer(self, inp, cfg=None, **kw):
        return ""

    async def _ainfer(self, inp, cfg=None, **kw):
        return ""


async def _recording_executor(name, arguments):
    ctx = active_run_context()
    SEEN.append((ctx.path, ctx.workspace) if ctx else None)
    return types.SimpleNamespace(result=f"{name} ok", context_updates={})


def _make_ci(*async_tools):
    return ConversationalInferencer(
        base_inferencer=_NoopBase(),
        tool_registry={
            name: ToolDefinition(name=name, asynchronous=True) for name in async_tools
        },
        tool_executor=_recording_executor,
    )


async def _call_tool(ci, name):
    result = await ci._execute_tool_call(types.SimpleNamespace(name=name, arguments={}))
    task = getattr(ci, "_active_async_task", None)
    if task is not None:
        await task
        ci._active_async_task = None
    return result


def _under(caller, coro):
    token = enter_run(caller)
    try:
        return asyncio.run(coro)
    finally:
        exit_run(token)


def test_sync_tool_runs_at_tool_child_keeping_the_caller_workspace():
    caller = RunContext.root(workspace=None).child("x")
    _under(caller, _call_tool(_make_ci(), "search"))

    assert SEEN == [("/x/tool/search", None)]


def test_async_tool_dispatches_run_at_distinct_numbered_children():
    ci = _make_ci("research")

    async def _twice():
        await _call_tool(ci, "research")
        await _call_tool(ci, "research")

    _under(RunContext.root(workspace=None).child("x"), _twice())

    assert [path for path, _ in SEEN] == [
        "/x/tool/research/async_0",
        "/x/tool/research/async_1",
    ]


def test_tool_keeps_the_caller_workspace_object(tmp_path):
    ws = InferencerWorkspace(root=str(tmp_path / "ws"))
    caller = RunContext.root(workspace=ws).child("x")
    ci = _make_ci("research")

    async def _both():
        await _call_tool(ci, "search")
        await _call_tool(ci, "research")

    _under(caller, _both())

    assert [w for _, w in SEEN] == [caller.workspace, caller.workspace]


def test_caller_ctx_is_restored_after_sync_and_async_tools():
    caller = RunContext.root(workspace=None).child("x")
    ci = _make_ci("research")

    async def _both():
        await _call_tool(ci, "search")
        assert active_run_context() is caller
        await _call_tool(ci, "research")
        assert active_run_context() is caller

    _under(caller, _both())


def test_tool_without_a_caller_ctx_runs_ctx_less():
    ci = _make_ci("research")

    async def _both():
        await _call_tool(ci, "search")
        await _call_tool(ci, "research")

    asyncio.run(_both())

    assert SEEN == [None, None]
    assert active_run_context() is None


def test_unsafe_tool_names_are_sanitized_under_a_workspace(tmp_path):
    caller = RunContext.root(
        workspace=InferencerWorkspace(root=str(tmp_path / "ws"))
    ).child("x")
    ci = _make_ci()

    async def _both():
        await _call_tool(ci, "a/b")
        await _call_tool(ci, "../up")

    _under(caller, _both())

    assert [path for path, _ in SEEN] == ["/x/tool/a_b", "/x/tool/__up"]


def test_command_invoked_as_tool_stays_at_the_turn_ctx():
    caller = RunContext.root(workspace=None).child("x")
    ci = _make_ci()
    ci._commands.is_command_name = lambda name: name == "set_model"

    async def _dispatch(name, arguments):
        SEEN.append(active_run_context().path)
        return "ok"

    ci._commands.dispatch_as_tool = _dispatch
    _under(caller, _call_tool(ci, "set_model"))

    assert SEEN == ["/x"]


def test_rc_child_chains_subslots_with_workspace_on_the_deepest_only(tmp_path):
    base_ws = InferencerWorkspace(root=str(tmp_path / "ws"))
    explicit = InferencerWorkspace(root=str(tmp_path / "explicit"))
    caller = RunContext.root(workspace=base_ws).child("x")
    token = enter_run(caller)
    try:
        child = _NoopBase()._rc_child("a/b", "c..d", ".", workspace=explicit)
        derived = _NoopBase()._rc_child("a/b", "c..d")
        assert active_run_context() is caller
    finally:
        exit_run(token)

    assert child.path == "/x/a_b/c_d/child"
    assert child.workspace is explicit
    assert derived.path == "/x/a_b/c_d"
    assert derived.workspace.root == caller.workspace.child("a_b").child("c_d").root


def test_rc_child_without_a_ctx_is_none():
    assert _NoopBase()._rc_child("tool", "search") is None

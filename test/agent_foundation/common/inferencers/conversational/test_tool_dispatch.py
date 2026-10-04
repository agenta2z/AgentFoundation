"""tool_dispatch: ``execute`` / ``apply_outcome`` over a ``ToolDispatchHost``
double, and ``AsyncToolTasks`` keeping every background run referenced."""

from __future__ import annotations

import asyncio
import gc
import types
import weakref
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    command,
    CommandRegistry,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_dispatch import (
    apply_outcome,
    AsyncToolTasks,
    execute,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    RunContext,
)
from agent_foundation.resources.tools.models import ToolDefinition
from later.unittest import TestCase


class _SOP:
    """Records the SOPController calls dispatch makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.pending_followup: Optional[str] = None

    def check_phase_completion(self, tool_name: str = "") -> None:
        self.calls.append(("check", tool_name))

    def mark_async_tool_phase_running(self, canonical: str) -> None:
        self.calls.append(("running", canonical))

    def consume_pending_followup(self) -> Optional[str]:
        followup, self.pending_followup = self.pending_followup, None
        return followup


class _Host:
    def __init__(self, executor: Any = None, *, tools: tuple = ()) -> None:
        self.tool_registry = {t.name: t for t in tools}
        self.tool_executor = executor
        self.sop_controller = _SOP()
        self.async_tool_tasks = AsyncToolTasks()
        self.prior_context: dict[str, Any] = {}
        self.commands = CommandRegistry(self)
        self.command_ctx_paths: list[str] = []

    def update_prior_context(self, **updates: Any) -> None:
        self.prior_context.update(updates)

    @command("greet", requires_args=True)
    async def _cmd_greet(self, who: str = "") -> str:
        self.command_ctx_paths.append(active_run_context().path)
        self.sop_controller.pending_followup = f"start with {who}"
        return f"hello {who}"


class _Executor:
    def __init__(self, updates: Optional[dict] = None, error: str = "") -> None:
        self.updates = updates or {}
        self.error = error
        self.calls: list[tuple[str, Any, Optional[str]]] = []

    async def __call__(self, name: str, arguments: Any) -> Any:
        ctx = active_run_context()
        self.calls.append((name, arguments, ctx.path if ctx else None))
        if self.error:
            raise RuntimeError(self.error)
        return types.SimpleNamespace(
            result=f"{name} ok", context_updates=dict(self.updates)
        )


def _turn() -> RunContext:
    return RunContext.root(workspace=None).child("turn")


class ExecuteSyncTest(TestCase):
    async def test_outcome_carries_the_result_and_nothing_is_applied_yet(
        self,
    ) -> None:
        executor = _Executor(updates={"brief_path": "/tmp/b.md"})
        host = _Host(
            executor,
            tools=(ToolDefinition(name="write_brief", aliases=["brief"]),),
        )

        outcome = await execute(host, "brief", {"topic": "x"}, run_ctx=_turn())

        self.assertEqual(outcome.tool_name, "write_brief")
        self.assertEqual(outcome.text, "write_brief ok")
        self.assertEqual(dict(outcome.context_updates), {"brief_path": "/tmp/b.md"})
        self.assertFalse(outcome.is_async or outcome.is_command)
        self.assertIsNone(outcome.error)
        self.assertEqual(
            executor.calls, [("write_brief", {"topic": "x"}, "/turn/tool/write_brief")]
        )
        self.assertEqual(host.prior_context, {})
        self.assertEqual(host.sop_controller.calls, [])

        self.assertIs(apply_outcome(host, outcome), outcome)
        self.assertEqual(host.prior_context, {"brief_path": "/tmp/b.md"})
        self.assertEqual(host.sop_controller.calls, [("check", "write_brief")])

    async def test_without_a_run_context_the_tool_runs_context_free(self) -> None:
        executor = _Executor()
        host = _Host(executor, tools=(ToolDefinition(name="search"),))
        await execute(host, "/search", {})
        self.assertEqual(executor.calls, [("search", {}, None)])


class ExecuteCommandTest(TestCase):
    async def test_command_runs_under_the_turn_context_and_returns_its_followup(
        self,
    ) -> None:
        host = _Host(_Executor())

        outcome = await execute(host, "greet", {"who": "ada"}, run_ctx=_turn())

        self.assertTrue(outcome.is_command)
        self.assertEqual(outcome.text, "hello ada")
        self.assertEqual(outcome.followup_text, "start with ada")
        self.assertIsNone(host.sop_controller.pending_followup)
        self.assertEqual(host.command_ctx_paths, ["/turn"])
        self.assertEqual(host.sop_controller.calls, [])

        apply_outcome(host, outcome)
        self.assertEqual(host.sop_controller.calls, [("check", "greet")])


class ExecuteErrorTest(TestCase):
    async def test_failing_tool_reports_an_error_and_applies_nothing(self) -> None:
        host = _Host(_Executor(error="boom"), tools=(ToolDefinition(name="t"),))

        outcome = apply_outcome(host, await execute(host, "t", {}))

        self.assertEqual(outcome.error, "boom")
        self.assertEqual(outcome.text, "Error executing t: boom")
        self.assertEqual(host.sop_controller.calls, [])

    async def test_missing_executor_is_an_error(self) -> None:
        outcome = await execute(_Host(None), "t", {})
        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.text, "No tool executor configured for: t")

    async def test_a_result_that_cannot_be_applied_fails_the_call(self) -> None:
        host = _Host(_Executor(updates={"k": "v"}), tools=(ToolDefinition(name="t"),))

        def broken_update(**_updates: Any) -> None:
            raise ValueError("bad update")

        host.update_prior_context = broken_update

        outcome = apply_outcome(host, await execute(host, "t", {}))

        self.assertEqual(outcome.error, "bad update")
        self.assertEqual(outcome.text, "Error executing t: bad update")


class ExecuteAsyncTest(TestCase):
    async def test_async_tool_starts_in_the_background_and_applies_when_done(
        self,
    ) -> None:
        release = asyncio.Event()
        done: list[tuple[str, str, dict, list]] = []

        class _Slow(_Executor):
            async def __call__(self, name: str, arguments: Any) -> Any:
                await release.wait()
                return await super().__call__(name, arguments)

        host = _Host(
            _Slow(updates={"report": "r.md"}),
            tools=(ToolDefinition(name="research", asynchronous=True),),
        )

        def on_done(name: str, result: Any) -> None:
            done.append(
                (
                    name,
                    result.result,
                    dict(host.prior_context),
                    host.sop_controller.calls[:],
                )
            )

        outcome = await execute(
            host, "research", {}, run_ctx=_turn(), on_async_done=on_done
        )

        self.assertTrue(outcome.is_async)
        self.assertIn("launched asynchronously", outcome.text)
        (task,) = host.async_tool_tasks.pending
        self.assertIs(host.async_tool_tasks.latest, task)
        self.assertEqual(host.sop_controller.calls, [("running", "research")])
        # Starting a run applies nothing more.
        self.assertIs(apply_outcome(host, outcome), outcome)
        self.assertEqual(host.prior_context, {})

        release.set()
        await task

        # Effects first, then the host's completion hook.
        self.assertEqual(
            done,
            [
                (
                    "research",
                    "research ok",
                    {"report": "r.md"},
                    [("running", "research"), ("check", "research")],
                )
            ],
        )
        self.assertEqual(host.async_tool_tasks.pending, frozenset())
        self.assertEqual(
            host.tool_executor.calls,
            [("research", {}, "/turn/tool/research/async_0")],
        )

    async def test_concurrent_runs_stay_referenced_until_each_finishes(self) -> None:
        """Each run waits on a future nothing but the run itself references, so
        a run the host does not hold is unreachable and collectable."""
        waiting: list[weakref.ref] = []

        class _Parked(_Executor):
            async def __call__(self, name: str, arguments: Any) -> Any:
                fut = asyncio.get_running_loop().create_future()
                waiting.append(weakref.ref(fut))
                await fut
                return await super().__call__(name, arguments)

        host = _Host(
            _Parked(),
            tools=(
                ToolDefinition(name="a", asynchronous=True),
                ToolDefinition(name="b", asynchronous=True),
            ),
        )
        finished: list[str] = []

        def on_done(name: str, _result: Any) -> None:
            finished.append(name)

        await execute(host, "a", {}, on_async_done=on_done)
        await execute(host, "b", {}, on_async_done=on_done)
        while len(waiting) < 2:
            await asyncio.sleep(0)
        gc.collect()

        self.assertEqual(len(host.async_tool_tasks.pending), 2)
        futures = [ref() for ref in waiting]
        self.assertNotIn(None, futures)
        for fut in futures:
            fut.set_result(None)
        await asyncio.gather(*host.async_tool_tasks.pending)

        self.assertEqual(sorted(finished), ["a", "b"])
        self.assertEqual(host.async_tool_tasks.pending, frozenset())

    async def test_failed_background_run_is_released(self) -> None:
        host = _Host(
            _Executor(error="boom"),
            tools=(ToolDefinition(name="research", asynchronous=True),),
        )
        finished: list[str] = []
        await execute(
            host, "research", {}, on_async_done=lambda n, _r: finished.append(n)
        )
        (task,) = host.async_tool_tasks.pending
        await task
        self.assertEqual(finished, [])
        self.assertEqual(host.async_tool_tasks.pending, frozenset())

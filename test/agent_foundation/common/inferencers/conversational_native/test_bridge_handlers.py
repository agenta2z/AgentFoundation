"""The AF tool bridge's handler pipeline (plan §5.3-§5.4, §6.3): dispatch,
result sizing, state updates, refusals, turn-ending causes and what a call
returning after its turn applies.

Plan §12.1 row coverage (assertion: tests; Class.method or a whole Class):
- action -> tool_dispatch.execute, context_updates, CompletedAction:
    ActionDispatchTest
    NativeTurnTest.test_action_tool_runs_with_normalized_arguments
- truncation / spill before L3:
    ResultSizingTest
- enter_sop -> follow-up in the result + the first phase's L3:
    NativeTurnTest.test_enter_sop_tool_returns_state_update
    NativeTurnTest.test_enter_sop_request_reaches_the_sop_verbatim_and_flags_typed
    ResultSizingTest.test_a_huge_result_keeps_its_whole_state_update_within_the_budget
- OpenStartup-style async -> "started" + phase running + no notice:
    AsyncToolTest.test_a_host_managed_async_tool_runs_its_phase_and_queues_no_notice
- generic host -> outbox once:
    AsyncToolTest.test_on_a_generic_host_the_result_reaches_the_next_turn_once
    OutboxDeliveryTest
- refusals (current rule: a call refused because its turn is ending gets a
  non-error AF_END_TURN result; only a call without a turn is an error):
    no turn: RefusalTest
    subagent: NativeTurnTest.test_subagent_calls_are_refused
    non-widget after widget: NativeTurnTest.test_non_widget_call_after_a_widget_is_refused,
      BridgeGateTest, CompoundWidgetRuleTest, CompoundWidgetBridgeTest
- arguments validated against the session's schema (plan §5.3 step 4):
    ArgumentValidationTest
- effects skipped when the gate closed mid-call (current rule: a result
  applies only while its turn is live and its vendor timeout has not passed;
  a stop cause raised mid-call, e.g. a pause, still applies it:
  CooperativePauseTest):
    ResultAfterTurnEndTest
    VendorToolTimeoutTest
- run context bound from a foreign task:
    RunContextBindingTest
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    PausedResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge import (
    tool_bridge,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    AFToolBridge,
    END_TURN_MARKER,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    TextDelta,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.turn import (
    QueuedWidget,
    StopCause,
    TurnOrigin,
    TurnScope,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    RunContext,
)
from agent_foundation.resources.tools.models import ToolDefinition
from fakes import (
    FAKE_CAPABILITIES,
    FakeBackendFactory,
    subagent_tool,
    text,
    tools,
    turn_end,
)
from helpers import make_native, RecordingExecutor, shared_dir, wait_for
from later.unittest import TestCase
from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus


class NativeTurnTest(TestCase):
    async def test_enter_sop_tool_returns_state_update(self) -> None:
        native, factory, _, _ = make_native(
            [
                [
                    tools(
                        (
                            "enter_sop",
                            {"name": "mini_research", "request": "quantum sensors"},
                        )
                    ),
                    text("Started."),
                ]
            ]
        )
        async with native:
            await native.run_agentic_loop("research quantum sensors")
            self.assertEqual(native.sop_state.sop_name, "mini_research")
            name, result, is_error = factory.last.tool_results[0]
            self.assertFalse(is_error)
            self.assertIn("Entered SOP 'mini_research'", result)
            self.assertIn("<af_state_update", result)

    async def test_enter_sop_request_reaches_the_sop_verbatim_and_flags_typed(
        self,
    ) -> None:
        request = '--fresh  please, "quoted"'
        native, factory, _, _ = make_native(
            [
                [
                    tools(
                        (
                            "enter_sop",
                            {"name": "mini_research", "request": request, "yolo": True},
                        )
                    ),
                    text("Started."),
                ]
            ]
        )
        async with native:
            await native.run_agentic_loop("run the mini research sop")
            _, result, is_error = factory.last.tool_results[0]
            self.assertFalse(is_error)
            self.assertIn(
                f"Entered SOP 'mini_research'. Starting on: {request}", result
            )
            self.assertTrue(native.sop_state.yolo_mode)
            self.assertIsNone(native._consume_pending_followup())

    async def test_request_text_never_acts_as_the_fresh_flag(self) -> None:
        native, factory, _, _ = make_native(
            [
                [
                    tools(
                        (
                            "enter_sop",
                            {"name": "mini_research", "request": "--fresh start over"},
                        )
                    ),
                    text("It is in progress."),
                ]
            ]
        )
        async with native:
            native.sop_controller.cmd_sop("mini_research")
            paused = native.sop_state
            native.sop_controller.cmd_pause_sop()
            await native.run_agentic_loop("start mini research over")
            _, result, _ = factory.last.tool_results[0]
            self.assertIn("You have an in-progress 'mini_research'", result)
            self.assertIsNone(native.sop_state)
            self.assertEqual(native.suspended_sops, [paused])

    async def test_resume_sop_request_is_never_read_as_a_name(self) -> None:
        native, factory, _, _ = make_native(
            [
                [
                    tools(
                        ("resume_sop", {"request": "mini_research next step please"})
                    ),
                    text("Resumed."),
                ]
            ]
        )
        async with native:
            native.sop_controller.cmd_sop("mini_research")
            native.sop_controller.cmd_exit_sop()
            await native.run_agentic_loop("continue where we left off")
            _, result, _ = factory.last.tool_results[0]
            self.assertIn("Continuing on: mini_research next step please", result)
            self.assertEqual(native.sop_state.sop_name, "mini_research")

    async def test_sop_results_name_the_sop_tools_not_slash_commands(self) -> None:
        native, factory, _, _ = make_native(
            [
                [tools(("exit_sop", {})), text("Exited.")],
                [tools(("enter_sop", {"name": "mini_research"})), text("Refused.")],
            ]
        )
        async with native:
            native.sop_controller.cmd_sop("mini_research")
            await native.run_agentic_loop("stop the sop")
            await native.run_agentic_loop("start mini research")
            exited, refused = (r for _, r, _ in factory.last.tool_results)
            self.assertIn(
                "Resume anytime with `mcp__af__resume_sop` (name `mini_research`).",
                exited,
            )
            self.assertIn(
                "Use `mcp__af__resume_sop` (name `mini_research`) to resume, or "
                "`mcp__af__enter_sop` with `fresh: true` to start over.",
                refused,
            )
            for result in (exited, refused):
                self.assertNotIn("/resume_sop", result)
                self.assertNotIn("/sop ", result)
        # What the user's own slash commands answer keeps naming slash commands.
        native.sop_controller.cmd_sop("mini_research --fresh")
        self.assertIn(
            "Resume anytime with /resume_sop mini_research.",
            native.sop_controller.cmd_exit_sop(),
        )

    async def test_sop_results_name_a_user_command_for_a_tool_not_offered(
        self,
    ) -> None:
        native, factory, _, _ = make_native(
            [[tools(("exit_sop", {})), text("Exited.")]],
            sop_control_tools=["enter_sop", "exit_sop"],
        )
        async with native:
            native.sop_controller.cmd_sop("mini_research")
            await native.run_agentic_loop("stop the sop")
            _, exited, _ = factory.last.tool_results[0]
            self.assertIn(
                "Resume anytime with a `/resume_sop mini_research` message from "
                "the user.",
                exited,
            )
            self.assertNotIn("mcp__af__resume_sop", exited)

    async def test_non_widget_call_after_a_widget_is_refused(self) -> None:
        scripts = [
            [
                tools(
                    ("clarification", {"prompt": "Topic?", "output": ["topic"]}),
                    ("write_brief", {"topic": "x"}),
                )
            ],
            [text("ok")],
        ]
        native, factory, _, executor = make_native(scripts, answers=["lidar"])
        async with native:
            await native.run_agentic_loop("go")
            _, refusal, is_error = factory.last.tool_results[1]
            self.assertFalse(is_error)
            self.assertTrue(refusal.startswith(f"{END_TURN_MARKER} — not run"))
            self.assertIn("a question for the user is pending", refusal)
            self.assertEqual(executor.calls, [])

    async def test_a_call_after_an_async_dispatch_is_refused_with_its_reason(
        self,
    ) -> None:
        scripts = [
            [tools(("write_brief", {"topic": "a"}), ("write_brief", {"topic": "b"}))]
        ]
        native, factory, _, executor = make_native(scripts, async_brief=True)
        async with native:
            await native.run_agentic_loop("write two")
            _, refusal, is_error = factory.last.tool_results[1]
            self.assertFalse(is_error)
            self.assertTrue(refusal.startswith(f"{END_TURN_MARKER} — not run"))
            self.assertIn("a background task started", refusal)
            self.assertNotIn({"topic": "b"}, [args for _, args in executor.calls])

    async def test_action_tool_runs_with_normalized_arguments(self) -> None:
        scripts = [
            [tools(("write_brief", {"topic": "lidar", "depth": "deep"})), text("Done.")]
        ]
        native, factory, _, executor = make_native(scripts)
        async with native:
            result = await native.run_agentic_loop("write it")
            self.assertEqual(
                executor.calls, [("write_brief", {"topic": "lidar", "depth": "deep"})]
            )
            self.assertEqual(factory.last.tool_results[0][1], "write_brief done")
            self.assertEqual(
                [a.tool for a in result.completed_actions], ["write_brief"]
            )

    async def test_async_tool_ends_the_vendor_turn(self) -> None:
        scripts = [[tools(("write_brief", {"topic": "lidar"})), text("never streamed")]]
        native, factory, _, _ = make_native(scripts, async_brief=True)
        async with native:
            result = await native.run_agentic_loop("write it")
            self.assertIn("AF_END_TURN", factory.last.tool_results[0][1])
            self.assertNotIn("never streamed", result.text)

    async def test_subagent_calls_are_refused(self) -> None:
        native, factory, _, executor = make_native(
            [[subagent_tool("write_brief", {"topic": "x"}), text("ok")]]
        )
        async with native:
            await native.run_agentic_loop("go")
            name, reason, is_error = factory.last.tool_results[0]
            self.assertTrue(is_error)
            self.assertIn("main agent", reason)
            self.assertEqual(executor.calls, [])


def _units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _spilled(test: unittest.TestCase, result: str, what: str) -> str:
    """The private file a cut ``result`` names for ``what``, read."""
    match = re.search(rf"\n\.\.\. \(truncated; full {what}: (\S+)\)", result)
    test.assertIsNotNone(match, f"no {what} file named in the result")
    path = match.group(1)
    test.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class _SizingHost:
    """What result sizing reads from the bridge host; ``state_update`` is the
    L3 of a call that changed the SOP state ("" for one that did not)."""

    def __init__(self, *, owns_spill: bool = False, state_update: str = "") -> None:
        self.native_tool_result_max_chars = 4000
        self._owns_spill = owns_spill
        self._state_update = state_update
        self._paused = False
        self._spill_dir = Path(tempfile.mkdtemp(prefix="af_native_sizing_"))

    def vendor_owns_result_spill(self) -> bool:
        return self._owns_spill

    def sop_fingerprint(self) -> tuple:
        return ("changed",) if self._state_update else ("unchanged",)

    def render_state_update(self) -> str:
        return self._state_update

    def spill_dir(self) -> Path:
        return self._spill_dir


def _finish(host: _SizingHost, output: str, stop: StopCause | None = None) -> str:
    turn = TurnScope(
        run_ctx=None, interactive=None, turn_number=1, origin=TurnOrigin.USER
    )
    if stop is not None:
        turn.request_stop(stop)
    return AFToolBridge(host, asyncio.Lock())._finish(turn, output, ("unchanged",))


class ResultSizingTest(TestCase):
    # Whether the vendor spills oversized results itself (Claude Code's
    # maxResultSizeChars) or the bridge truncates them.
    _SIZING_PATHS = {"bridge truncates": False, "vendor spills": True}

    async def test_oversized_result_is_spilled_to_a_private_file(self) -> None:
        executor = RecordingExecutor(results={"write_brief": "Y" * 5000})
        native, factory, _, _ = make_native(
            [[tools(("write_brief", {"topic": "x"})), text("done")]],
            executor=executor,
            native_tool_result_max_chars=2000,
        )
        async with native:
            await native.run_agentic_loop("go")
            result = factory.last.tool_results[0][1]
            self.assertTrue(result.startswith("Y" * 1000))
            self.assertLessEqual(_units(result), 2000)
            self.assertRegex(result, r"\(truncated; full output: \S+\)$")
            self.assertEqual(_spilled(self, result, "output"), "Y" * 5000)

    async def test_vendor_owned_spill_leaves_the_result_whole(self) -> None:
        factory = FakeBackendFactory(
            capabilities=dataclasses.replace(FAKE_CAPABILITIES, owns_result_spill=True)
        )
        executor = RecordingExecutor(results={"write_brief": "Z" * 500})
        native, factory, _, _ = make_native(
            [[tools(("write_brief", {"topic": "x"})), text("done")]],
            factory=factory,
            executor=executor,
            native_tool_result_max_chars=50,
        )
        async with native:
            await native.run_agentic_loop("go")
            self.assertEqual(factory.last.tool_results[0][1], "Z" * 500)

    async def test_a_huge_result_keeps_its_whole_state_update_within_the_budget(
        self,
    ) -> None:
        # A vendor spill would put the state update, at the end, behind a
        # preview of the head: the bridge cuts the output on both paths.
        request = "R" * 60_000
        enter = ("enter_sop", {"name": "mini_research", "request": request})
        for name, owns_spill in self._SIZING_PATHS.items():
            with self.subTest(name):
                capabilities = dataclasses.replace(
                    FAKE_CAPABILITIES, owns_result_spill=owns_spill
                )
                native, factory, _, _ = make_native(
                    [[tools(enter), text("ok")]],
                    factory=FakeBackendFactory(capabilities=capabilities),
                    native_tool_result_max_chars=4000,
                )
                async with native:
                    await native.run_agentic_loop("go")
                    result = factory.last.tool_results[0][1]
                self.assertLessEqual(_units(result), 4000)
                self.assertTrue(result.startswith("Entered SOP 'mini_research'."))
                state = result.index("\n\n<af_state_update")
                self.assertLess(result.index("(truncated; full output:"), state)
                self.assertIn("Current Phase (0 of 3): Topic", result[state:])
                self.assertTrue(result.rstrip().endswith("</af_state_update>"))
                self.assertEqual(
                    _spilled(self, result, "output"),
                    f"Entered SOP 'mini_research'. Starting on: {request}",
                )

    async def test_a_huge_end_turn_result_leads_with_the_directive(self) -> None:
        update = "<af_state_update>Phase 1 is next.</af_state_update>"
        for name, owns_spill in self._SIZING_PATHS.items():
            with self.subTest(name):
                host = _SizingHost(owns_spill=owns_spill, state_update=update)
                result = _finish(host, "O" * 60_000, StopCause.ASYNC_DISPATCH)
                self.assertLessEqual(_units(result), 4000)
                self.assertTrue(result.startswith(END_TURN_MARKER))
                self.assertTrue(result.endswith(f"\n\n{update}"))
                self.assertEqual(_spilled(self, result, "output"), "O" * 60_000)

    async def test_a_vendor_spilled_result_without_a_state_update_stays_whole(
        self,
    ) -> None:
        # The vendor's preview of a spilled result is its head, the directive.
        result = _finish(
            _SizingHost(owns_spill=True), "O" * 60_000, StopCause.WIDGET_QUEUED
        )
        self.assertTrue(result.startswith(END_TURN_MARKER))
        self.assertTrue(result.endswith("O" * 60_000))

    async def test_a_state_update_larger_than_the_budget_is_cut_and_spilled(
        self,
    ) -> None:
        update = f"<af_state_update>{'S' * 10_000}</af_state_update>"
        result = _finish(_SizingHost(state_update=update), "Brief written.")
        self.assertLessEqual(_units(result), 4000)
        self.assertTrue(result.startswith("Brief written.\n\n<af_state_update>SSS"))
        self.assertEqual(_spilled(self, result, "state update"), update)
        self.assertNotIn("full output:", result)

    async def test_the_budget_counts_utf16_units(self) -> None:
        result = _finish(_SizingHost(), "\U0001f600" * 3000)
        self.assertLessEqual(_units(result), 4000)
        mark = result.index("\n... (truncated; full output:")
        self.assertEqual(
            result[:mark], "\U0001f600" * ((4000 - _units(result[mark:])) // 2)
        )


class RunContextBindingTest(TestCase):
    async def test_tool_runs_under_the_turn_context_from_a_foreign_task(self) -> None:
        seen = []

        class _CtxExecutor(RecordingExecutor):
            async def __call__(self, name, arguments):
                seen.append((active_run_context(), asyncio.current_task()))
                return await super().__call__(name, arguments)

        native, _, _, _ = make_native(
            [[tools(("write_brief", {"topic": "x"})), text("done")]],
            executor=_CtxExecutor(),
        )
        async with native:
            root = RunContext.root()
            await native.run_agentic_loop("go", run_context=root)
            ctx, task = seen[0]
            # The executor runs in the vendor's task, under a per-tool child of
            # the turn's context (same run-state store as the host's root).
            self.assertIsNot(task, asyncio.current_task())
            self.assertIs(ctx.store, root.store)
            self.assertEqual(ctx.path, "/tool/write_brief")


class _NoticeExecutor(RecordingExecutor):
    """Queues a host notice while the tool runs (i.e. mid vendor turn)."""

    native = None

    async def __call__(self, name, arguments):
        result = await super().__call__(name, arguments)
        self.native._queue_notice("tool_completion", body="BG-RESULT-91", tool="bg_job")
        return result


class _PausingExecutor(RecordingExecutor):
    """Requests a cooperative pause from inside a running AF tool."""

    native = None

    async def __call__(self, name, arguments):
        self.native.request_pause()
        return await super().__call__(name, arguments)


class BridgeGateTest(TestCase):
    async def test_call_queued_behind_a_widget_is_refused_under_the_lock(self) -> None:
        results = []

        async def concurrent(backend, request):
            lock = native.bridge._lock
            await lock.acquire()
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            widget = asyncio.ensure_future(
                handlers["clarification"]({"prompt": "Topic?", "output": ["topic"]})
            )
            action = asyncio.ensure_future(handlers["write_brief"]({"topic": "x"}))
            await asyncio.sleep(
                0.01
            )  # both calls pass the entry checks and wait on the lock
            lock.release()
            results.extend(await asyncio.gather(widget, action))
            yield turn_end(backend)

        native, _, _, executor = make_native(
            [concurrent, [text("ok")]], answers=["lidar"]
        )
        async with native:
            await native.run_agentic_loop("go")
            widget_result, action_result = results
            self.assertTrue(widget_result.text.startswith(END_TURN_MARKER))
            self.assertFalse(action_result.is_error)
            self.assertTrue(
                action_result.text.startswith(f"{END_TURN_MARKER} — not run")
            )
            self.assertIn("already ended this turn", action_result.text)
            self.assertEqual(executor.calls, [])

    async def test_a_refusal_ending_its_message_lets_claude_stop_the_turn(
        self,
    ) -> None:
        """Claude Code stops a turn only from ``PostToolUse``, which it runs
        for a result that is not an error; for an error it runs
        ``PostToolUseFailure`` and ignores a stop (claude 2.1.288)."""
        results, stops = [], []

        async def claude(backend, request):
            hooks = backend.open_request.hooks
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            yield MessageEnd(message_id="m1", text="", af_tool_use_ids=("a1", "a2"))
            for tool_use, (name, args) in (
                ("a1", ("clarification", _TOPIC)),
                ("a2", ("write_brief", {"topic": "x"})),
            ):
                await hooks.before_af_tool(name, tool_use, None)
                result = await handlers[name](dict(args))
                results.append(result)
                stop = await hooks.after_af_tool(name, tool_use)
                stops.append(stop and not result.is_error)
            yield turn_end(backend)

        native, _, _, executor = make_native([claude, [text("ok")]], answers=["lidar"])
        async with native:
            await native.run_agentic_loop("go")
        self.assertEqual(stops, [False, True])
        self.assertTrue(results[1].text.startswith(f"{END_TURN_MARKER} — not run"))
        self.assertEqual(executor.calls, [])


_TOPIC = {"prompt": "Topic?", "output": ["topic"]}
_DEPTH = {"prompt": "Depth?", "choices": ["quick", "deep"], "output": ["depth"]}


def _scope() -> TurnScope:
    return TurnScope(
        run_ctx=None, interactive=None, turn_number=1, origin=TurnOrigin.USER
    )


def _widget() -> QueuedWidget:
    return QueuedWidget(tool=SimpleNamespace(prompt="?"))


class CompoundWidgetRuleTest(unittest.TestCase):
    """Plan §5.4: once a widget is queued, only widget calls from the same
    assistant message join it."""

    def test_announced_calls_join_only_from_the_first_widgets_message(self) -> None:
        turn = _scope()
        turn.note_af_message(("a1", "a2"), message_id="m1")
        turn.note_af_call("a1")
        turn.queue_widget(_widget(), "a1")
        turn.note_af_call("a2")
        self.assertTrue(turn.accepts_widget("a2"))
        turn.note_af_message(("b1",), message_id="m2")
        self.assertFalse(turn.accepts_widget("b1"))

    def test_without_message_ids_announced_calls_join_until_the_stop(self) -> None:
        turn = _scope()
        turn.note_af_message(("a1",))
        turn.note_af_message(("a2",))
        turn.note_af_call("a1")
        turn.queue_widget(_widget(), "a1")
        self.assertFalse(turn.finish_af_call("a1"))  # a2 of the message is open
        turn.note_af_call("a2")
        self.assertTrue(turn.accepts_widget("a2"))
        turn.queue_widget(_widget(), "a2")
        self.assertTrue(turn.finish_af_call("a2"))
        self.assertTrue(turn.stop_signalled)
        turn.note_af_message(("b1",))
        self.assertFalse(turn.accepts_widget("b1"))

    def test_unattributed_calls_join_until_the_vendor_reports_a_message(
        self,
    ) -> None:
        turn = _scope()
        turn.queue_widget(_widget(), None)
        self.assertTrue(turn.accepts_widget(None))
        turn.note_af_message(("w1",), message_id="step-1")
        self.assertFalse(turn.accepts_widget(None))

    def test_other_stop_causes_refuse_widgets(self) -> None:
        turn = _scope()
        turn.request_stop(StopCause.ASYNC_DISPATCH)
        self.assertFalse(turn.accepts_widget(None))
        self.assertFalse(turn.accepts_widget("a1"))


class CompoundWidgetBridgeTest(TestCase):
    async def _run(self, vendor, answer):
        native, factory, interactive, _ = make_native(
            [vendor, [text("Thanks.")]], answers=[answer]
        )
        async with native:
            await native.run_agentic_loop("ask me")
        return factory, interactive

    @staticmethod
    async def _noted(backend, tool_use_id: str) -> None:
        turn = backend.open_request.hooks.host.current_turn
        await wait_for(lambda: tool_use_id in turn.known_af_tool_uses)

    def _assert_queued(self, results, expected: list[bool]) -> None:
        """Every result is an end-turn result, not an error; ``expected``
        says which calls joined the compound widget."""
        self.assertEqual([r.is_error for r in results], [False] * len(results))
        self.assertTrue(all(r.text.startswith(END_TURN_MARKER) for r in results))
        self.assertEqual(["question queued" in r.text for r in results], expected)
        for result, queued in zip(results, expected):
            if not queued:
                self.assertIn("earlier message is already pending", result.text)

    async def test_unattributed_calls_of_one_message_form_one_compound_widget(
        self,
    ) -> None:
        results = []

        async def vendor(backend, request):  # dm: a step is reported after its calls
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            results.append(await handlers["clarification"](dict(_TOPIC)))
            results.append(await handlers["single_choice"](dict(_DEPTH)))
            yield MessageEnd(message_id="step-1", text="", af_tool_use_ids=("i1", "i2"))
            yield turn_end(backend)

        factory, interactive = await self._run(
            vendor, {"values": {"topic": "lidar", "depth": "deep"}}
        )
        self.assertEqual([r.is_error for r in results], [False, False])
        self.assertEqual(len(interactive.widgets), 1)
        self.assertTrue(interactive.widgets[0]["input_mode"].metadata.get("compound"))
        self.assertIn("depth: deep", factory.last.turn_requests[1].text)

    async def test_unattributed_widget_after_a_reported_message_is_refused(
        self,
    ) -> None:
        results = []

        async def vendor(backend, request):
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            results.append(await handlers["clarification"](dict(_TOPIC)))
            yield MessageEnd(message_id="step-1", text="", af_tool_use_ids=("i1",))
            await self._noted(backend, "i1")
            results.append(await handlers["single_choice"](dict(_DEPTH)))
            yield turn_end(backend)

        factory, interactive = await self._run(vendor, "lidar")
        self._assert_queued(results, [True, False])
        self.assertEqual(len(interactive.widgets), 1)
        answer = factory.last.turn_requests[1].text
        self.assertIn("topic: lidar", answer)
        self.assertNotIn("depth", answer)

    async def test_announced_widget_from_another_message_is_refused(self) -> None:
        results = []

        async def vendor(backend, request):  # message ids reported per tool use
            hooks = backend.open_request.hooks
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            turn = hooks.host.current_turn
            turn.note_af_message(("a1", "a2"), message_id="m1")
            for tool_use, (name, args) in (
                ("a1", ("clarification", _TOPIC)),
                ("a2", ("single_choice", {**_DEPTH, "output": ["depth_a"]})),
            ):
                await hooks.before_af_tool(name, tool_use, None)
                results.append(await handlers[name](dict(args)))
            turn.note_af_message(("b1",), message_id="m2")
            await hooks.before_af_tool("single_choice", "b1", None)
            results.append(await handlers["single_choice"](dict(_DEPTH)))
            yield turn_end(backend)

        factory, interactive = await self._run(
            vendor, {"values": {"topic": "lidar", "depth_a": "deep"}}
        )
        self._assert_queued(results, [True, True, False])
        self.assertEqual(len(interactive.widgets), 1)
        answer = factory.last.turn_requests[1].text
        self.assertIn("depth_a: deep", answer)
        self.assertNotIn("depth:", answer)

    async def test_a_widget_from_a_later_claude_message_is_refused(self) -> None:
        """Claude reports each content block as its own message event sharing
        the API message id; a widget call of the next message must not join
        the compound widget, even before the stop was signalled."""
        results = []

        async def vendor(backend, request):
            hooks = backend.open_request.hooks
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            for block in ("a1", "a2"):
                yield MessageEnd(message_id="msg_1", text="", af_tool_use_ids=(block,))
            await self._noted(backend, "a2")
            for tool_use, (name, args) in (
                ("a1", ("clarification", _TOPIC)),
                ("a2", ("single_choice", {**_DEPTH, "output": ["depth_a"]})),
            ):
                await hooks.before_af_tool(name, tool_use, None)
                results.append(await handlers[name](dict(args)))
            yield MessageEnd(message_id="msg_2", text="", af_tool_use_ids=("b1",))
            await self._noted(backend, "b1")
            await hooks.before_af_tool("single_choice", "b1", None)
            results.append(await handlers["single_choice"](dict(_DEPTH)))
            yield turn_end(backend)

        factory, interactive = await self._run(
            vendor, {"values": {"topic": "lidar", "depth_a": "deep"}}
        )
        self._assert_queued(results, [True, True, False])
        self.assertEqual(len(interactive.widgets), 1)
        self.assertNotIn("depth:", factory.last.turn_requests[1].text)

    async def test_announced_widget_after_the_stop_was_signalled_is_refused(
        self,
    ) -> None:
        results, stops = [], []

        async def vendor(backend, request):  # ignores the turn stop
            hooks = backend.open_request.hooks
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            yield MessageEnd(message_id="m1", text="", af_tool_use_ids=("a1",))
            await self._noted(backend, "a1")
            await hooks.before_af_tool("clarification", "a1", None)
            results.append(await handlers["clarification"](dict(_TOPIC)))
            stops.append(await hooks.after_af_tool("clarification", "a1"))
            yield MessageEnd(message_id="m2", text="", af_tool_use_ids=("b1",))
            await self._noted(backend, "b1")
            await hooks.before_af_tool("single_choice", "b1", None)
            results.append(await handlers["single_choice"](dict(_DEPTH)))
            yield turn_end(backend)

        factory, interactive = await self._run(vendor, "lidar")
        self.assertEqual(stops, [True])
        self._assert_queued(results, [True, False])
        self.assertEqual(len(interactive.widgets), 1)
        self.assertNotIn("depth", factory.last.turn_requests[1].text)


class _FailingExecutor(RecordingExecutor):
    async def __call__(self, name, arguments):
        await super().__call__(name, arguments)
        raise RuntimeError("disk full")


class _ParkedExecutor(RecordingExecutor):
    """Holds the tool body until released; its result carries a context update."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, name, arguments):
        await super().__call__(name, arguments)
        self.started.set()
        await self.release.wait()
        return ToolExecutionResult(
            result="brief written", context_updates={"brief_path": "/tmp/brief.md"}
        )


class ToolFailureTest(TestCase):
    async def test_failing_tool_is_an_error_result_for_the_vendor(self) -> None:
        native, factory, _, _ = make_native(
            [[tools(("write_brief", {"topic": "x"})), text("ok")]],
            executor=_FailingExecutor(),
        )
        async with native:
            await native.run_agentic_loop("go")
            name, result, is_error = factory.last.tool_results[0]
            self.assertEqual(name, "write_brief")
            self.assertTrue(is_error)
            self.assertIn("Error executing write_brief: disk full", result)


class ResultAfterTurnEndTest(TestCase):
    """A vendor need not cancel a running AF tool when the host cancels the
    turn; whatever the tool returns afterwards must not change host state."""

    async def _cancel_while_running(self, call: tuple[str, dict], *, yolo=False):
        executor = _ParkedExecutor()
        handler_task = {}

        async def vendor(backend, request):
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            name, args = call
            handler_task["task"] = asyncio.ensure_future(handlers[name](dict(args)))
            yield TextDelta(message_id="m0", text="Working on it.")
            await asyncio.Event().wait()

        native, _, _, _ = make_native(
            [vendor], executor=executor, vendor_drain_timeout_s=0.5
        )
        async with native:
            native.sop_controller.cmd_sop("mini_research")
            # In the brief phase, which a write_brief result completes.
            native.sop_state.completed_phases = ["0"]
            native.sop_state.current_phase = "1"
            native.sop_state.yolo_mode = yolo
            turn = asyncio.ensure_future(native.run_agentic_loop("write the brief"))
            await asyncio.wait_for(executor.started.wait(), 5)
            self.assertEqual(native.sop_state.current_phase, "1")
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            executor.release.set()
            result = await asyncio.wait_for(handler_task["task"], 5)
            return native, executor, result

    def _assert_nothing_applied(self, native, result) -> None:
        self.assertTrue(result.is_error)
        self.assertIn("did not apply its result", result.text)
        self.assertNotIn("<af_state_update", result.text)
        self.assertNotIn("brief_path", native.prior_context)
        self.assertEqual(native.sop_state.completed_phase_ids(), ["0"])
        self.assertEqual(native.sop_state.current_phase, "1")
        self.assertEqual(native.dynamic_context.completed_actions, [])

    async def test_action_tool_result_is_not_applied(self) -> None:
        native, executor, result = await self._cancel_while_running(
            ("write_brief", {"topic": "lidar"})
        )
        self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])
        self._assert_nothing_applied(native, result)

    async def test_yolo_then_run_result_is_not_applied(self) -> None:
        widget = {
            "prompt": "Topic?",
            "output": ["topic"],
            "then_run": {"name": "write_brief", "arguments": {"topic": "lidar"}},
        }
        native, executor, result = await self._cancel_while_running(
            ("clarification", widget), yolo=True
        )
        self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])
        self._assert_nothing_applied(native, result)


class _SlowExecutor(RecordingExecutor):
    async def __call__(self, name, arguments):
        await super().__call__(name, arguments)
        await asyncio.sleep(0.3)
        return ToolExecutionResult(
            result="BRIEF-WRITTEN-17", context_updates={"brief_path": "/tmp/b.md"}
        )


class VendorToolTimeoutTest(TestCase):
    """The vendor stops waiting for an AF call after its tool-call timeout
    and tells the model it failed (spike S5), while the handler runs on."""

    def _native(self, scripts, **kwargs):
        shared = shared_dir()
        native, factory, _, executor = make_native(
            scripts,
            executor=_SlowExecutor(),
            backend={"kind": "claude_sdk", "cwd": shared, "mcp_tool_timeout_ms": 100},
            session_dir=shared,
            **kwargs,
        )
        native.sop_controller.cmd_sop("mini_research")
        # In the brief phase, which a write_brief result completes.
        native.sop_state.completed_phases = ["0"]
        native.sop_state.current_phase = "1"
        return native, factory, executor

    def _assert_nothing_applied(self, native, result) -> None:
        self.assertNotIn("brief_path", native.prior_context)
        self.assertEqual(native.sop_state.completed_phase_ids(), ["0"])
        self.assertEqual(result.completed_actions, [])

    async def test_a_late_result_is_not_applied_and_reported_once(self) -> None:
        native, factory, executor = self._native(
            [
                [tools(("write_brief", {"topic": "lidar"})), text("It timed out.")],
                [text("I see.")],
                [text("ok")],
            ]
        )
        async with native:
            result = await native.run_agentic_loop("write the brief")
            _, tool_text, is_error = factory.last.tool_results[0]
            self.assertTrue(is_error)
            self.assertIn("outlasted the agent's tool-call timeout", tool_text)
            self._assert_nothing_applied(native, result)
            record = native._load_record()
            self.assertEqual(
                [(n["type"], n["tool"], n["ran"]) for n in record.pending_notices()],
                [("late_tool_result", "mcp__af__write_brief", True)],
            )
            self.assertNotIn("BRIEF-WRITTEN-17", json.dumps(record.to_dict()))
            await native.run_agentic_loop("what happened?")
            await native.run_agentic_loop("and now?")
            first, second = factory.last.l2_seen[1:3]
            self.assertIn('type="late_tool_result"', first)
            self.assertIn(
                "`mcp__af__write_brief` finished after your 0.1 s tool-call timeout",
                first,
            )
            self.assertIn("were not applied", first)
            self.assertIn("BRIEF-WRITTEN-17", first)
            self.assertNotIn("late_tool_result", second)
            self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])

    async def test_a_late_result_retained_nowhere_is_still_reported(self) -> None:
        native, factory, _ = self._native(
            [[tools(("write_brief", {"topic": "lidar"}))], [text("I see.")]]
        )
        async with native:
            await native.run_agentic_loop("write the brief")
            native._notice_bodies.clear()  # as after a host restart
            await native.run_agentic_loop("what happened?")
            l2 = factory.last.l2_seen[1]
            self.assertIn("were not applied", l2)
            self.assertIn("Its result was not retained.", l2)

    async def test_a_call_whose_timeout_passed_while_waiting_is_not_run(
        self,
    ) -> None:
        results = []

        async def vendor(backend, request):
            lock = backend.open_request.hooks.host.bridge._lock
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            await lock.acquire()
            call = asyncio.ensure_future(handlers["write_brief"]({"topic": "x"}))
            await asyncio.sleep(0.2)
            lock.release()
            results.append(await call)
            yield turn_end(backend)

        native, factory, executor = self._native([vendor, [text("I see.")]])
        async with native:
            result = await native.run_agentic_loop("write the brief")
            self.assertTrue(results[0].is_error)
            self.assertEqual(executor.calls, [])
            self._assert_nothing_applied(native, result)
            await native.run_agentic_loop("what happened?")
            self.assertIn("`mcp__af__write_brief` was not run", factory.last.l2_seen[1])

    async def test_a_yolo_then_run_after_the_timeout_is_not_applied(self) -> None:
        widget = {
            "prompt": "Topic?",
            "output": ["topic"],
            "then_run": {"name": "write_brief", "arguments": {"topic": "__topic__"}},
        }
        native, factory, executor = self._native(
            [[tools(("clarification", widget)), text("Done.")], [text("I see.")]]
        )
        async with native:
            native.sop_state.yolo_mode = True
            result = await native.run_agentic_loop("go")
            self.assertTrue(factory.last.tool_results[0][2])
            self.assertEqual(len(executor.calls), 1)
            self._assert_nothing_applied(native, result)
            await native.run_agentic_loop("what happened?")
            l2 = factory.last.l2_seen[1]
            self.assertIn("`mcp__af__clarification` finished after", l2)
            self.assertIn("Answered autonomously", l2)
            self.assertNotIn("BRIEF-WRITTEN-17", l2)


class AsyncThenRunTest(TestCase):
    async def test_async_then_run_between_turns_does_not_end_a_later_call(
        self,
    ) -> None:
        """An async ``then_run`` started after the answer (between vendor
        turns; it ends that host call, D6) must not make a later synchronous
        call end its turn."""
        widget = {
            "prompt": "Topic?",
            "output": ["topic"],
            "then_run": {"name": "write_brief", "arguments": {"topic": "__topic__"}},
        }
        native, factory, _, executor = make_native(
            [
                [tools(("clarification", widget))],
                [tools(("lookup", {})), text("Looked it up.")],
            ],
            answers=["lidar"],
            async_brief=True,
        )
        native.tool_registry["lookup"] = ToolDefinition(
            name="lookup", description="Look something up.", tool_type="Action"
        )
        async with native:
            await native.run_agentic_loop("go")
            await asyncio.gather(*native.async_tool_tasks.pending)
            result = await native.run_agentic_loop("look it up")
            self.assertEqual(
                executor.calls, [("write_brief", {"topic": "lidar"}), ("lookup", {})]
            )
            self.assertEqual(
                factory.last.tool_results[-1], ("lookup", "lookup done", False)
            )
            self.assertEqual(result.text, "Looked it up.")


class OutboxDeliveryTest(TestCase):
    async def test_notice_queued_mid_turn_is_delivered_next_turn(self) -> None:
        executor = _NoticeExecutor()
        native, factory, _, _ = make_native(
            [[tools(("write_brief", {"topic": "x"})), text("done")], [text("next")]],
            executor=executor,
        )
        executor.native = native
        async with native:
            native._queue_notice("interrupted")
            await native.run_agentic_loop("first")
            backend = factory.last
            self.assertIn('type="interrupted"', backend.l2_seen[0])
            self.assertNotIn("BG-RESULT-91", backend.l2_seen[0])
            self.assertEqual(
                [n["type"] for n in native._load_record().pending_notices()],
                ["tool_completion"],
            )
            await native.run_agentic_loop("second")
            self.assertIn("BG-RESULT-91", backend.l2_seen[1])
            self.assertNotIn('type="interrupted"', backend.l2_seen[1])
            self.assertEqual(native._load_record().pending_notices(), [])


class CooperativePauseTest(TestCase):
    async def test_pause_requested_during_a_tool_ends_the_turn_with_paused_result(
        self,
    ) -> None:
        executor = _PausingExecutor()
        native, factory, _, _ = make_native(
            [[tools(("write_brief", {"topic": "x"})), text("never streamed")]],
            executor=executor,
        )
        executor.native = native
        async with native:
            result = await native.run_agentic_loop("go")
            self.assertIsInstance(result, PausedResult)
            self.assertTrue(result.paused)
            self.assertEqual(result.pause_state["schema"], "native/v1")
            self.assertTrue(factory.last.tool_results[0][1].startswith(END_TURN_MARKER))
            self.assertNotIn("never streamed", result.text)
            self.assertTrue(native._paused)
            native.restore_state(result.pause_state)
            self.assertFalse(native._paused)


class _UpdatingExecutor(RecordingExecutor):
    async def __call__(self, name, arguments):
        await super().__call__(name, arguments)
        return ToolExecutionResult(
            result="brief written", context_updates={"brief_path": "/tmp/brief.md"}
        )


class ActionDispatchTest(TestCase):
    """Plan §5.3: an action call runs through ``tool_dispatch.execute``, merges
    the tool's ``context_updates``, completes the SOP phase under the tool's
    unprefixed name and records a ``CompletedAction``."""

    async def test_an_action_applies_its_context_updates_and_is_recorded(
        self,
    ) -> None:
        native, factory, _, executor = make_native(
            [[tools(("write_brief", {"topic": "lidar"})), text("Done.")]],
            executor=_UpdatingExecutor(),
        )
        execute = mock.patch.object(tool_bridge, "execute", wraps=tool_bridge.execute)
        async with native:
            native.sop_controller.cmd_sop("mini_research")
            native.sop_state.completed_phases = ["0"]
            native.sop_state.current_phase = "1"
            with execute as executed:
                result = await native.run_agentic_loop("write the brief")
        executed.assert_called_once()
        self.assertIs(executed.call_args.args[0], native)
        self.assertEqual(
            executed.call_args.args[1:], ("write_brief", {"topic": "lidar"})
        )
        self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])
        self.assertEqual(native.prior_context["brief_path"], "/tmp/brief.md")
        self.assertEqual(native.sop_state.completed_phase_ids(), ["0", "1"])
        name, output, is_error = factory.last.tool_results[0]
        self.assertFalse(is_error)
        self.assertTrue(output.startswith("brief written\n\n<af_state_update"))
        self.assertEqual(
            [(a.tool, a.summary) for a in result.completed_actions],
            [("write_brief", "brief written")],
        )
        self.assertEqual(
            [a.tool for a in native.dynamic_context.completed_actions],
            ["write_brief"],
        )


class AsyncToolTest(TestCase):
    """Plan §6.3: a background tool ends the vendor turn with "started". A
    host that reports completions itself (OpenStartup) gets the phase marked
    running at dispatch and no outbox notice; any other host gets the result
    as one notice in the next turn's L2."""

    async def test_a_host_managed_async_tool_runs_its_phase_and_queues_no_notice(
        self,
    ) -> None:
        executor = _ParkedExecutor()
        native, factory, _, _ = make_native(
            [
                [tools(("write_brief", {"topic": "lidar"})), text("never streamed")],
                [text("The brief is ready.")],
            ],
            executor=executor,
            async_brief=True,
            host_manages_async_results=True,
        )
        async with native:
            native.sop_controller.cmd_sop("mini_research")
            result = await native.run_agentic_loop("write the brief")
            await asyncio.wait_for(executor.started.wait(), 5)
            self.assertEqual(native.sop_state.current_phase, "1")
            self.assertEqual(native.sop_state.phase_status, PhaseStatus.RUNNING)
            started = factory.last.tool_results[0][1]
            self.assertTrue(
                started.startswith(
                    f"{END_TURN_MARKER} — the tool started in the background."
                )
            )
            self.assertNotIn("never streamed", result.text)
            executor.release.set()
            await asyncio.gather(*native.async_tool_tasks.pending)
            self.assertEqual(native.prior_context["brief_path"], "/tmp/brief.md")
            self.assertEqual(native._load_record().pending_notices(), [])
            await native.run_agentic_loop(
                "[System notification: write_brief finished]", origin="host_event"
            )
        l2 = factory.last.l2_seen[1]
        self.assertIn('origin="host_event"', l2)
        self.assertNotIn("tool_completion", l2)
        self.assertNotIn("brief written", l2)

    async def test_on_a_generic_host_the_result_reaches_the_next_turn_once(
        self,
    ) -> None:
        native, factory, _, _ = make_native(
            [
                [tools(("write_brief", {"topic": "lidar"}))],
                [text("It is done.")],
                [text("ok")],
            ],
            async_brief=True,
        )
        async with native:
            await native.run_agentic_loop("write the brief")
            await asyncio.gather(*native.async_tool_tasks.pending)
            self.assertEqual(
                [
                    (n["type"], n["tool"])
                    for n in native._load_record().pending_notices()
                ],
                [("tool_completion", "write_brief")],
            )
            await native.run_agentic_loop("is it done?")
            await native.run_agentic_loop("thanks")
            self.assertEqual(native._load_record().pending_notices(), [])
        _, delivered, after = factory.last.l2_seen
        self.assertIn(
            '<notice type="tool_completion">write_brief finished:\nwrite_brief done',
            delivered,
        )
        self.assertNotIn("tool_completion", after)


class RefusalTest(TestCase):
    async def test_a_call_outside_any_turn_is_refused_and_runs_nothing(
        self,
    ) -> None:
        native, _, interactive, executor = make_native([])
        async with native:
            for name, args in (
                ("write_brief", {"topic": "x"}),
                ("clarification", {"prompt": "Topic?"}),
                ("enter_sop", {"name": "mini_research"}),
            ):
                with self.subTest(name):
                    result = await native.bridge.call(name, args)
                    self.assertTrue(result.is_error)
                    self.assertEqual(
                        result.text, "No active AgentFoundation turn; tool refused."
                    )
            self.assertEqual(
                await native.before_af_tool("write_brief", "tu1", None),
                "No active AgentFoundation turn.",
            )
            self.assertFalse(await native.after_af_tool("write_brief", "tu1"))
        self.assertEqual(executor.calls, [])
        self.assertEqual(interactive.widgets, [])
        self.assertIsNone(native.sop_state)


class ArgumentValidationTest(TestCase):
    """Plan §5.3 step 4: arguments are validated against the session's tool
    schema before anything runs; an invalid call is an error result and the
    turn goes on."""

    async def test_an_invalid_call_is_an_error_result_and_runs_nothing(self) -> None:
        cases = [
            (("write_brief", {}), "Invalid arguments: 'topic' is a required property"),
            (("write_brief", {"topic": ""}), "Missing required argument(s): topic"),
            (
                ("write_brief", {"topic": "x", "depth": "medium"}),
                "Invalid arguments at depth: 'medium' is not one of ['quick', 'deep']",
            ),
            (
                ("clarification", {"output": ["t"]}),
                "Invalid arguments: 'prompt' is a required property",
            ),
            (
                ("single_choice", {"prompt": "Depth?"}),
                "Invalid arguments: 'choices' is a required property",
            ),
            (
                ("enter_sop", {"name": 7}),
                "Invalid arguments at name: 7 is not of type 'string'",
            ),
        ]
        native, factory, interactive, executor = make_native(
            [[tools(*(call for call, _ in cases)), text("Let me fix those.")]]
        )
        async with native:
            result = await native.run_agentic_loop("go")
        self.assertEqual(
            factory.last.tool_results,
            [(call[0], error, True) for call, error in cases],
        )
        self.assertEqual(executor.calls, [])
        self.assertEqual(interactive.widgets, [])
        self.assertIsNone(native.sop_state)
        self.assertEqual(result.text, "Let me fix those.")

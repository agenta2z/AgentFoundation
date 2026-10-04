"""Submission safety and cancellation of native vendor turns (plan §6.4):
submission states, no replay, interrupts, drains and stalls.

Plan §12.1 row coverage (assertion: tests; Class.method or a whole Class):
- prepared -> retry allowed:
    RetryWhilePreparedTest
    StoppedBeforeTheModelTest.test_a_prompt_blocked_after_the_session_started_stays_prepared
    SubmissionStateTest.test_unsubmitted_vendor_error_is_not_treated_as_acceptance
- submitted failure -> uncertain, never replayed:
    SubmissionStateTest, NoReplayTest, UnfinishedTurnTest
    FailureHandlingTest.test_failed_turn_after_a_tool_ran_is_never_replayed
- cancel -> drain + interrupt concurrently:
    CancelDrainTest.test_interrupt_runs_while_the_turn_drains_and_its_tail_is_dropped
- drain timeout -> kill (the session is closed; the next turn resumes it):
    CancelDrainTest.test_a_drain_timeout_kills_the_turn_stream_and_leaves_it_uncertain
    SdkDrainTimeoutTest
- interrupted notice next turn:
    CancelDrainTest (both tests)
    NativeSessionLifecycleTest.test_cancelled_turn_is_marked_and_announced_not_replayed
- no stale events:
    CancelDrainTest, CancelDuringStreamTest, CancelledFirstTurnTest,
    SdkDrainTimeoutTest
Also here: the stall watchdog (StallWatchdogTest), and the prompt manifest
("View Prompt") of a turn that did not complete (ManifestOutcomeTest).
"""

from __future__ import annotations

import asyncio
import copy
import tempfile
import time
from unittest import mock

import claude_agent_sdk
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    VendorTurnFailed,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    SessionStarted,
    TextDelta,
    TurnEnd,
    VendorError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    L2Channel,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.claude_sdk import (
    ClaudeSdkBackend,
)
from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock
from fakes import FAKE_CAPABILITIES, FakeBackendFactory, text, turn_end
from helpers import make_native, wait_for
from later.unittest import TestCase


def _stopped_after_session_start(*, submitted: bool):
    """A fresh Claude CLI turn as the CLI reports it: ``init`` confirms the
    session, then the turn ends before any model request."""

    async def turn(backend, request):
        yield SessionStarted(session_id=backend.session_id)
        yield VendorError(message="the turn was not run", submitted=submitted)

    return turn


class StoppedBeforeTheModelTest(TestCase):
    async def test_a_prompt_blocked_after_the_session_started_stays_prepared(
        self,
    ) -> None:
        native, factory, _, _ = make_native(
            [_stopped_after_session_start(submitted=False), [text("two")]]
        )
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("one")
            record = native._load_record()
            self.assertEqual(record.submission, "prepared")
            self.assertNotIn(
                "turn_failed", [n["type"] for n in record.pending_notices()]
            )
            # The vendor created the session, so the next turn resumes it.
            self.assertTrue(record.started)
            await native.runtime_manager.evict_conversation(native.conversation_key)
            await native.run_agentic_loop("two")
            self.assertTrue(factory.last.open_request.resume)
            self.assertNotIn('type="turn_failed"', factory.last.l2_seen[0])

    async def test_a_turn_stopped_after_its_prompt_was_recorded_is_uncertain(
        self,
    ) -> None:
        native, _, _, _ = make_native([_stopped_after_session_start(submitted=True)])
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("one")
            record = native._load_record()
            self.assertEqual(record.submission, "uncertain")
            self.assertEqual(record.pending_notices()[-1]["type"], "turn_failed")


class UnfinishedTurnTest(TestCase):
    """A host that died during a vendor turn leaves the record ``submitted``;
    the next host call settles it before anything else."""

    async def _crash_during_a_turn(self, *, queue_widget: bool) -> tuple:
        shared, store = (
            tempfile.mkdtemp(prefix="af_native_test_"),
            InMemoryRecordStore(),
        )
        crash = {}

        async def vendor(backend, request):
            yield MessageEnd(message_id="m0", text="", af_tool_use_ids=("tu1",))
            turn = backend.open_request.hooks.host.current_turn
            while not turn.accepted:
                await asyncio.sleep(0.005)
            if queue_widget:
                handlers = {t.name: t.handler for t in backend.open_request.tools}
                await handlers["clarification"]({"prompt": "Topic?", "output": ["t"]})
            # The host dies here; the store's copy is all that survives.
            crash["record"] = copy.deepcopy(store._records["conv-test"])
            yield TurnEnd(session_id=backend.session_id, num_turns=1)

        native, factory, _, _ = make_native(
            [vendor, [text("ok")]],
            answers=["lidar"],
            record_store=store,
            session_dir=shared,
        )
        async with native:
            await native.run_agentic_loop("ask me")
        restarted = InMemoryRecordStore()
        restarted._records["conv-test"] = crash["record"]
        recovered, factory2, _, _ = make_native(
            [[text("Where were we?")]], record_store=restarted, session_dir=shared
        )
        async with recovered:
            await recovered.run_agentic_loop("still there?")
        return crash["record"], factory, factory2, restarted.load("conv-test")

    async def test_a_widget_queued_before_the_crash_is_announced_as_unshown(
        self,
    ) -> None:
        crashed, first, after, record = await self._crash_during_a_turn(
            queue_widget=True
        )
        self.assertEqual(crashed["submission"], "submitted")
        self.assertTrue(crashed["pending_widget"])
        backend = after.last
        self.assertTrue(backend.open_request.resume)
        self.assertEqual(
            backend.open_request.session_id, first.last.open_request.session_id
        )
        self.assertEqual([r.text for r in backend.turn_requests], ["still there?"])
        l2 = backend.l2_seen[0]
        self.assertIn('type="interrupted"', l2)
        self.assertIn('type="widget_unshown"', l2)
        self.assertEqual(record.submission, "committed")
        self.assertFalse(record.pending_widget)
        self.assertEqual(record.pending_notices(), [])

    async def test_a_crash_without_a_queued_widget_is_announced_as_interrupted(
        self,
    ) -> None:
        crashed, _, after, _ = await self._crash_during_a_turn(queue_widget=False)
        self.assertEqual(crashed["submission"], "submitted")
        l2 = after.last.l2_seen[0]
        self.assertIn('type="interrupted"', l2)
        self.assertNotIn('type="widget_unshown"', l2)

    async def test_a_cancel_after_a_widget_was_queued_announces_it_unshown(
        self,
    ) -> None:
        queued = asyncio.Event()

        async def vendor(backend, request):
            yield MessageEnd(message_id="m0", text="", af_tool_use_ids=("tu1",))
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            await handlers["clarification"]({"prompt": "Topic?", "output": ["t"]})
            queued.set()
            await asyncio.Event().wait()
            yield TurnEnd(session_id=backend.session_id)  # pragma: no cover

        native, factory, interactive, _ = make_native(
            [vendor, [text("ok")]], vendor_drain_timeout_s=0.2
        )
        async with native:
            turn = asyncio.ensure_future(native.run_agentic_loop("ask me"))
            await asyncio.wait_for(queued.wait(), 5)
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            record = native._load_record()
            self.assertFalse(record.pending_widget)
            self.assertEqual(
                [n["type"] for n in record.pending_notices()],
                ["interrupted", "widget_unshown"],
            )
            self.assertEqual(interactive.widgets, [])
            await native.run_agentic_loop("next")
            self.assertIn('type="widget_unshown"', factory.last.l2_seen[-1])


class NoReplayTest(TestCase):
    async def test_retry_budget_is_zero_and_recovery_reraises(self) -> None:
        native, _, _, _ = make_native([])
        async with native:
            self.assertEqual(native.max_retry, 0)
            with self.assertRaises(ValueError):
                await native._ainfer_recovery("x", last_exception=ValueError("boom"))


class SubmissionStateTest(TestCase):
    async def test_prepared_then_submitted_then_committed(self) -> None:
        seen = {}

        async def probe(backend, request):
            seen["before"] = native._load_record().submission
            yield TextDelta(message_id="m0", text="hello")
            await wait_for(lambda: native._load_record().submission == "submitted")
            seen["after"] = native._load_record().submission
            yield MessageEnd(message_id="m0", text="hello", message_uuid="u0")
            yield turn_end(backend)

        native, _, _, _ = make_native([probe])
        async with native:
            await native.run_agentic_loop("one")
            self.assertEqual(seen, {"before": "prepared", "after": "submitted"})
            self.assertEqual(native._load_record().submission, "committed")

    async def test_silent_stream_end_is_uncertain(self) -> None:
        """The prompt was handed to the vendor (no backend error said
        otherwise), so the turn is not definitely unsubmitted."""

        async def silent(backend, request):
            return
            yield  # pragma: no cover - makes this an async generator

        native, _, _, _ = make_native([silent])
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("one")
            record = native._load_record()
            self.assertEqual(record.submission, "uncertain")
            self.assertIn("turn_failed", [n["type"] for n in record.pending_notices()])

    async def test_failure_after_events_is_uncertain_and_announced(self) -> None:
        async def partial(backend, request):
            yield TextDelta(message_id="m0", text="half")
            yield VendorError(message="connection reset")

        native, _, _, _ = make_native([partial])
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("one")
            record = native._load_record()
            self.assertEqual(record.submission, "uncertain")
            self.assertEqual(record.pending_notices()[-1]["type"], "turn_failed")

    async def test_unsubmitted_vendor_error_is_not_treated_as_acceptance(self) -> None:
        """PRODUCTION BUG (turn_loop._stream_turn / _fail_turn): a backend that
        reports ``VendorError(submitted=False)`` (e.g. "spawn failed") as its
        only event is marked accepted -> ``uncertain`` + ``turn_failed`` notice,
        although nothing reached the vendor."""

        async def spawn_failed(backend, request):
            yield VendorError(message="spawn failed", submitted=False)

        native, _, _, _ = make_native([spawn_failed])
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("one")
            record = native._load_record()
            self.assertEqual(record.submission, "prepared")
            self.assertNotIn(
                "turn_failed", [n["type"] for n in record.pending_notices()]
            )


class FailureHandlingTest(TestCase):
    async def test_failed_turn_after_a_tool_ran_is_never_replayed(self) -> None:
        async def fail_after_tool(backend, request):
            yield MessageEnd(
                message_id="m0",
                text="",
                tool_use_ids=("tu1",),
                af_tool_use_ids=("tu1",),
                message_uuid="u0",
            )
            await backend.call_tool("write_brief", {"topic": "x"}, tool_use_id="tu1")
            yield VendorError(message="stream broke")

        native, factory, _, executor = make_native(
            [fail_after_tool, [text("recovered")]]
        )
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("go")
            self.assertEqual(native._load_record().submission, "uncertain")
            await native.run_agentic_loop("continue")
            self.assertEqual(executor.calls, [("write_brief", {"topic": "x"})])
            backend = factory.last
            self.assertEqual(
                [r.text for r in backend.turn_requests], ["go", "continue"]
            )
            self.assertIn('type="turn_failed"', backend.l2_seen[1])


class NativeSessionLifecycleTest(TestCase):
    async def test_cancelled_turn_is_marked_and_announced_not_replayed(self) -> None:
        started = asyncio.Event()

        class _SlowFactory:
            def __init__(self, inner):
                self.inner = inner
                self.capabilities = inner.capabilities

            def __call__(self, spec, **kw):
                backend = self.inner(spec, **kw)
                original = backend.run_turn

                async def slow(request):
                    started.set()
                    await asyncio.sleep(30)
                    async for event in original(request):
                        yield event

                backend.run_turn = slow
                return backend

        native, factory, _, _ = make_native(
            [[text("late")]], vendor_drain_timeout_s=0.2
        )
        native.backend_factory = _SlowFactory(factory)
        async with native:
            task = asyncio.ensure_future(native.run_agentic_loop("slow"))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            record = native._load_record()
            self.assertIn(record.submission, ("interrupted", "uncertain"))
            self.assertEqual(record.pending_notices()[-1]["type"], "interrupted")


class CancelDuringStreamTest(TestCase):
    async def test_cancel_mid_round_stops_its_token_stream(self) -> None:
        """A host cancel while a round streams must not leave the round's
        token-stream task pending (it would be destroyed with its generator
        still running)."""
        streaming = asyncio.Event()

        async def endless(backend, request):
            yield TextDelta(message_id="m0", text="once upon a time")
            streaming.set()
            await asyncio.Event().wait()
            yield turn_end(backend)  # pragma: no cover - never reached

        native, _, _, _ = make_native([endless], vendor_drain_timeout_s=0.5)
        async with native:
            turn = asyncio.ensure_future(native.run_agentic_loop("tell a story"))
            await asyncio.wait_for(streaming.wait(), 5)
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            pending = [
                t
                for t in asyncio.all_tasks()
                if t is not asyncio.current_task()
                and "stream_token_batches" in repr(t.get_coro())
            ]
            self.assertEqual(pending, [])
            self.assertEqual(native._load_record().submission, "uncertain")


class CancelledFirstTurnTest(TestCase):
    async def test_a_session_the_vendor_started_is_resumed_after_a_cancel(self) -> None:
        """A first turn cancelled after the vendor began it must not leave the
        record "new": the next turn would re-create the pinned session id
        (Claude CLI: "Session ID ... is already in use")."""
        streaming = asyncio.Event()

        async def endless(backend, request):
            yield TextDelta(message_id="m0", text="working on it")
            streaming.set()
            await asyncio.Event().wait()
            yield turn_end(backend)  # pragma: no cover - never reached

        native, factory, _, _ = make_native(
            [endless, [text("second")]], vendor_drain_timeout_s=0.2
        )
        async with native:
            turn = asyncio.ensure_future(native.run_agentic_loop("one", turn_number=1))
            await asyncio.wait_for(streaming.wait(), 5)
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            first = factory.instances[0]
            self.assertTrue(native._load_record().started)
            await native.runtime_manager.evict_conversation(native.conversation_key)
            await native.run_agentic_loop("two", turn_number=2)
            reopened = factory.last
            self.assertIsNot(reopened, first)
            self.assertTrue(reopened.open_request.resume)
            self.assertEqual(
                reopened.open_request.session_id, first.open_request.session_id
            )


class StallWatchdogTest(TestCase):
    async def test_silent_vendor_is_interrupted_and_the_turn_fails(self) -> None:
        async def hang(backend, request):
            yield TextDelta(message_id="m0", text="thinking")
            await backend.interrupted.wait()

        native, factory, _, _ = make_native(
            [hang], vendor_stall_timeout_s=0.3, vendor_drain_timeout_s=1.0
        )
        async with native:
            start = time.monotonic()
            with self.assertRaises(VendorTurnFailed) as ctx:
                await native.run_agentic_loop("go")
            self.assertLess(time.monotonic() - start, 5.0)
            self.assertIn("no events", str(ctx.exception))
            self.assertGreaterEqual(factory.last.interrupts, 1)
            self.assertEqual(native._load_record().submission, "uncertain")


class ManifestOutcomeTest(TestCase):
    """ "View Prompt" of a vendor turn that did not complete: the turn's
    prompt manifest (recorded before submission) stays the last prompt data
    and states how the turn ended."""

    def _assert_manifest(
        self, native, user: str, outcome: str, detail: str
    ) -> tuple[str, dict]:
        data = native.last_prompt_data()
        rendered, feed = data["rendered_prompt"], data["template_feed"]
        routes = FAKE_CAPABILITIES.prompt_routes(L2Channel.HOOK)
        record = native._load_record()
        l1 = (native.session_dir() / f"l1_{record.generation}.md").read_text(
            encoding="utf-8"
        )
        self.assertTrue(rendered.startswith("## Prompt manifest (approximate)\n"))
        self.assertIn(f"\n## Turn outcome — {outcome}\n", rendered)
        self.assertIn(f"## Session instructions — {routes.l1}\n{l1}\n", rendered)
        self.assertIn(f"## User message — {routes.user}\n{user}\n", rendered)
        self.assertLess(
            rendered.index("## Turn outcome"), rendered.index("## Session instructions")
        )
        self.assertEqual(feed["turn_outcome"], outcome)
        self.assertIn(detail, feed["turn_outcome_detail"])
        self.assertIn(detail, rendered)
        self.assertEqual(feed["l1_core_hash"], record.l1_core_hash)
        self.assertEqual(feed["l2_route"], routes.l2)
        return rendered, feed

    async def test_a_turn_that_fails_after_submission_is_marked_uncertain(
        self,
    ) -> None:
        async def partial(backend, request):
            yield TextDelta(message_id="m0", text="half")
            yield VendorError(message="connection reset")

        native, _, _, _ = make_native([[text("one")], partial])
        async with native:
            await native.run_agentic_loop("first question")
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("second question")
            rendered, _ = self._assert_manifest(
                native, "second question", "uncertain", "connection reset"
            )
        self.assertIn("it is not re-sent", rendered)
        self.assertNotIn("first question", rendered)

    async def test_the_first_turn_failing_keeps_its_turn_context(self) -> None:
        async def broken(backend, request):
            yield TextDelta(message_id="m0", text="half")
            yield VendorError(message="stream broke")

        native, _, _, _ = make_native([broken])
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("hello")
            rendered, _ = self._assert_manifest(
                native, "hello", "uncertain", "stream broke"
            )
        routes = FAKE_CAPABILITIES.prompt_routes(L2Channel.HOOK)
        self.assertIn(f"## Turn context — {routes.l2}\n<af_context", rendered)

    async def test_a_turn_that_never_reached_the_agent_is_marked_failed(self) -> None:
        async def spawn_failed(backend, request):
            yield VendorError(message="spawn failed", submitted=False)

        native, _, _, _ = make_native([spawn_failed])
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("hello")
            rendered, _ = self._assert_manifest(
                native, "hello", "failed", "spawn failed"
            )
        self.assertIn("failed before it reached the agent", rendered)

    async def test_a_stalled_turn_is_marked_uncertain(self) -> None:
        async def hang(backend, request):
            yield TextDelta(message_id="m0", text="thinking")
            await backend.interrupted.wait()

        native, _, _, _ = make_native(
            [hang], vendor_stall_timeout_s=0.3, vendor_drain_timeout_s=1.0
        )
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("go")
            self._assert_manifest(native, "go", "uncertain", "produced no events")

    async def _cancelled(self, script, user: str, **kwargs):
        streaming = asyncio.Event()

        async def vendor(backend, request):
            async for event in script(backend, streaming):
                yield event

        native, _, _, _ = make_native([vendor], **kwargs)
        async with native:
            turn = asyncio.ensure_future(native.run_agentic_loop(user))
            await asyncio.wait_for(streaming.wait(), 5)
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
        return native

    async def test_a_cancel_the_agent_confirms_is_marked_interrupted(self) -> None:
        async def stops_on_interrupt(backend, streaming):
            yield TextDelta(message_id="m0", text="working")
            streaming.set()
            await backend.interrupted.wait()

        native = await self._cancelled(stops_on_interrupt, "long task")
        self.assertEqual(native._load_record().submission, "interrupted")
        self._assert_manifest(
            native, "long task", "interrupted", "The host cancelled the turn."
        )

    async def test_a_cancel_the_agent_does_not_confirm_is_marked_uncertain(
        self,
    ) -> None:
        async def ignores_interrupt(backend, streaming):
            yield TextDelta(message_id="m0", text="working")
            streaming.set()
            await asyncio.Event().wait()
            yield turn_end(backend)  # pragma: no cover - never reached

        native = await self._cancelled(
            ignores_interrupt, "long task", vendor_drain_timeout_s=0.2
        )
        self.assertEqual(native._load_record().submission, "uncertain")
        self._assert_manifest(
            native, "long task", "uncertain", "did not confirm the stop"
        )

    async def test_a_completed_turn_states_no_outcome(self) -> None:
        async def broken(backend, request):
            yield VendorError(message="stream broke")

        native, _, _, _ = make_native([broken, [text("fine")]])
        async with native:
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("one")
            result = await native.run_agentic_loop("two")
            data = native.last_prompt_data()
        self.assertNotIn("## Turn outcome", data["rendered_prompt"])
        self.assertNotIn("turn_outcome", data["template_feed"])
        self.assertIn("\ntwo\n", data["rendered_prompt"])
        self.assertEqual(result.last_rendered_prompt, data["rendered_prompt"])


class _SpawnFailsOnceFactory(FakeBackendFactory):
    """The first backend cannot open its session (a vendor CLI that does not
    spawn); later ones work."""

    def __call__(self, spec, **runtime):
        backend = super().__call__(spec, **runtime)
        if len(self.instances) == 1:

            async def spawn_failed(request) -> None:
                raise RuntimeError("spawn failed")

            backend.open = spawn_failed
        return backend


class RetryWhilePreparedTest(TestCase):
    async def test_a_turn_that_never_reached_the_vendor_is_safe_to_send_again(
        self,
    ) -> None:
        """Plan §6.4: a failure while ``prepared`` (spawn/connect) leaves the
        turn definitely unsubmitted — no notice, nothing to settle — so the
        host may send it again and the vendor sees it once."""
        native, factory, _, _ = make_native(
            [[text("hello")]], factory=_SpawnFailsOnceFactory()
        )
        async with native:
            with self.assertRaisesRegex(RuntimeError, "spawn failed"):
                await native.run_agentic_loop("hi")
            record = native._load_record()
            self.assertEqual(record.submission, "prepared")
            self.assertEqual(record.pending_notices(), [])
            self.assertFalse(record.started)
            result = await native.run_agentic_loop("hi")
        failed, working = factory.instances
        self.assertEqual(failed.turn_requests, [])
        self.assertEqual([r.text for r in working.turn_requests], ["hi"])
        self.assertFalse(working.open_request.resume)
        for notice in ("interrupted", "turn_failed"):
            self.assertNotIn(notice, working.l2_seen[0])
        self.assertEqual(result.text, "hello")
        self.assertEqual([m["content"] for m in native.get_messages()], ["hi", "hello"])


class CancelDrainTest(TestCase):
    """Plan §6.4: a host cancel drains the vendor turn while ``interrupt()``
    runs (both bounded). Acknowledged → ``interrupted``; a drain that times
    out kills the turn's stream → ``uncertain``. Either way the next L2 says
    so once and nothing of the cancelled turn reaches the host."""

    @staticmethod
    async def _accepted(native) -> None:
        await wait_for(
            lambda: native.current_turn is not None and native.current_turn.accepted,
            timeout=5,
        )

    async def test_interrupt_runs_while_the_turn_drains_and_its_tail_is_dropped(
        self,
    ) -> None:
        drained = []

        async def vendor(backend, request):
            yield TextDelta(message_id="m0", text="working")
            await backend.interrupted.wait()
            yield TextDelta(message_id="m0", text=" STALE-TAIL")
            yield MessageEnd(
                message_id="m0", text="working STALE-TAIL", message_uuid="u0"
            )
            drained.append(True)
            yield turn_end(backend)

        native, factory, interactive, _ = make_native(
            [vendor, [text("next")], [text("after")]], vendor_drain_timeout_s=5.0
        )
        displays = []

        async def on_round_complete(_inf, iteration, turn, raw, clean, display, conv):
            displays.append(display)

        async with native:
            turn = asyncio.ensure_future(
                native.run_agentic_loop("go", on_round_complete=on_round_complete)
            )
            await self._accepted(native)
            backend = factory.last
            interrupt = backend.interrupt

            async def acknowledged_once_drained() -> None:
                # Returns only after the vendor drained: an interrupt awaited
                # before draining would time out instead.
                await interrupt()
                await wait_for(lambda: drained, timeout=5)

            backend.interrupt = acknowledged_once_drained
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            record = native._load_record()
            self.assertEqual(record.submission, "interrupted")
            self.assertEqual(
                [n["type"] for n in record.pending_notices()], ["interrupted"]
            )
            self.assertEqual(backend.interrupts, 1)
            await native.run_agentic_loop("next", on_round_complete=on_round_complete)
            await native.run_agentic_loop("after", on_round_complete=on_round_complete)
        self.assertIn('type="interrupted"', backend.l2_seen[1])
        self.assertEqual(backend.l2_seen[2], "")
        self.assertEqual(displays, ["next", "after"])
        self.assertEqual(interactive.streamed, ["next", "after"])
        self.assertEqual(
            [m["content"] for m in native.get_messages()],
            ["go", "next", "next", "after", "after"],
        )
        self.assertEqual(len(factory.instances), 1)

    async def test_a_drain_timeout_kills_the_turn_stream_and_leaves_it_uncertain(
        self,
    ) -> None:
        killed = asyncio.Event()

        async def deaf(backend, request):
            try:
                yield TextDelta(message_id="m0", text="working")
                await asyncio.Event().wait()
                yield turn_end(backend)  # pragma: no cover - never reached
            finally:
                killed.set()

        native, factory, interactive, _ = make_native(
            [deaf, [text("next")]], vendor_drain_timeout_s=0.2
        )
        async with native:
            turn = asyncio.ensure_future(native.run_agentic_loop("go"))
            await self._accepted(native)
            loop = asyncio.get_running_loop()
            start = loop.time()
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            self.assertLess(loop.time() - start, 2.0)
            await asyncio.wait_for(killed.wait(), 2)
            first = factory.last
            self.assertEqual(first.interrupts, 1)
            # The session is closed before the cancel returns (plan §6.4 "on
            # timeout disconnect/kill"): nothing it still produces can reach
            # a later turn.
            self.assertTrue(first.closed)
            record = native._load_record()
            self.assertEqual(record.submission, "uncertain")
            self.assertEqual(
                [n["type"] for n in record.pending_notices()], ["interrupted"]
            )
            result = await native.run_agentic_loop("next")
        reopened = factory.last
        self.assertIsNot(reopened, first)
        self.assertTrue(reopened.open_request.resume)
        self.assertEqual(
            reopened.open_request.session_id, first.open_request.session_id
        )
        self.assertEqual([r.text for r in first.turn_requests], ["go"])
        self.assertEqual([r.text for r in reopened.turn_requests], ["next"])
        self.assertIn('type="interrupted"', reopened.l2_seen[0])
        self.assertEqual(result.text, "next")
        self.assertEqual(interactive.streamed, ["next"])


def _assistant(session_id: str, message_id: str, text_: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text_)],
        model="opus",
        message_id=message_id,
        uuid=f"uuid-{message_id}",
        session_id=session_id,
    )


def _result(session_id: str, text_: str) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        result=text_,
    )


class _PersistentSdkClient:
    """Stands in for ``ClaudeSDKClient`` where turns meet: one message stream
    per connected client, which ``receive_response()`` reads up to the next
    result, as the SDK does. Its ``"slow"`` turn ignores ``interrupt()`` and
    emits its last message and result once ``release`` is set — after the
    drain timeout, into whatever still reads this client."""

    instances: list = []
    # Whether the "slow" turn produces model output before it hangs.
    slow_output = True

    def __init__(self, options) -> None:
        self.options = options
        self.connected = False
        self.queries: list[str] = []
        self.interrupts = 0
        self.release = asyncio.Event()
        self.late_emitted = False
        self._messages: asyncio.Queue = asyncio.Queue()
        type(self).instances.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def get_mcp_status(self) -> dict:
        af = {"name": "af", "status": "connected", "tools": [{"name": "sop_status"}]}
        return {"mcpServers": [af]}

    async def query(self, prompt: str, session_id: str = "default") -> None:
        hook = self.options.hooks["UserPromptSubmit"][0].hooks[0]
        await hook(
            {"hook_event_name": "UserPromptSubmit", "prompt": prompt}, None, None
        )
        self.queries.append(prompt)
        sid = self.options.resume or self.options.session_id
        init = {
            "session_id": sid,
            "mcp_servers": [{"name": "af", "status": "connected"}],
            "tools": ["mcp__af__sop_status"],
        }
        self._messages.put_nowait(SystemMessage(subtype="init", data=init))
        if prompt == "slow":
            if self.slow_output:
                self._messages.put_nowait(_assistant(sid, "msg_1", "working"))
            asyncio.ensure_future(self._emit_late(sid))
        else:
            self._messages.put_nowait(_assistant(sid, "msg_2", f"reply to {prompt}"))
            self._messages.put_nowait(_result(sid, f"reply to {prompt}"))

    async def _emit_late(self, sid: str) -> None:
        await self.release.wait()
        self._messages.put_nowait(_assistant(sid, "msg_1b", "STALE-TAIL"))
        self._messages.put_nowait(_result(sid, "STALE-TAIL"))
        self.late_emitted = True

    async def receive_response(self):
        while True:
            message = await self._messages.get()
            yield message
            if isinstance(message, ResultMessage):
                return

    async def interrupt(self) -> None:
        self.interrupts += 1


class _SdkBackends(FakeBackendFactory):
    """The real ``ClaudeSdkBackend`` (its client patched in the test)."""

    capabilities = ClaudeSdkBackend.capabilities

    def __call__(self, spec, **runtime):
        backend = ClaudeSdkBackend(spec, **runtime)
        self.instances.append(backend)
        return backend


class SdkDrainTimeoutTest(TestCase):
    """The Claude SDK client outlives a turn: a turn that does not drain could
    leave its late messages, result included, to the next turn on the same
    client. Plan §6.4: on a drain timeout the session is disconnected and the
    next turn resumes the vendor session by id on a new client."""

    async def test_late_events_of_a_timed_out_turn_never_reach_the_next_turn(
        self,
    ) -> None:
        _PersistentSdkClient.instances = []
        native, factory, interactive, _ = make_native(
            [], factory=_SdkBackends(), vendor_drain_timeout_s=0.2
        )
        with mock.patch.object(
            claude_agent_sdk, "ClaudeSDKClient", _PersistentSdkClient
        ):
            async with native:
                turn = asyncio.ensure_future(native.run_agentic_loop("slow"))
                await CancelDrainTest._accepted(native)
                turn.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await turn
                (first,) = _PersistentSdkClient.instances
                self.assertEqual(first.interrupts, 1)
                self.assertFalse(first.connected)
                self.assertEqual(native._load_record().submission, "uncertain")
                first.release.set()  # the vendor goes on after the timeout
                await wait_for(lambda: first.late_emitted)
                result = await native.run_agentic_loop("next")
        self.assertEqual(result.text, "reply to next")
        _, second = _PersistentSdkClient.instances
        self.assertEqual((first.queries, second.queries), (["slow"], ["next"]))
        self.assertIsNone(first.options.resume)
        self.assertTrue(first.options.session_id)
        self.assertEqual(second.options.resume, first.options.session_id)
        self.assertNotIn("STALE", " ".join(interactive.streamed))
        self.assertNotIn("STALE", " ".join(m["content"] for m in native.get_messages()))
        self.assertIn('type="interrupted"', result.last_rendered_prompt)

    async def test_a_first_turn_abandoned_after_init_is_resumed(self) -> None:
        """Claude Code holds the session from init on (its transcript has the
        prompt), so a first turn abandoned before any model output leaves a
        session the next turn resumes: its pinned id cannot be opened as new
        again ("already in use")."""
        _PersistentSdkClient.instances = []
        native, _, _, _ = make_native(
            [], factory=_SdkBackends(), vendor_drain_timeout_s=0.2
        )
        with (
            mock.patch.object(
                claude_agent_sdk, "ClaudeSDKClient", _PersistentSdkClient
            ),
            mock.patch.object(_PersistentSdkClient, "slow_output", False),
        ):
            async with native:
                turn = asyncio.ensure_future(native.run_agentic_loop("slow"))
                instances = _PersistentSdkClient.instances
                await wait_for(
                    lambda: instances
                    and instances[0].queries
                    and instances[0]._messages.empty(),
                    timeout=5,
                )
                await asyncio.sleep(0.05)  # init reaches the turn driver
                self.assertTrue(native._load_record().started)
                turn.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await turn
                (first,) = instances
                self.assertFalse(first.connected)
                first.release.set()
                await wait_for(lambda: first.late_emitted)
                result = await native.run_agentic_loop("next")
        _, second = _PersistentSdkClient.instances
        self.assertEqual(second.options.resume, first.options.session_id)
        self.assertEqual(result.text, "reply to next")

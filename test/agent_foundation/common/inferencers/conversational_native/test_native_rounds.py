"""Rounds of the native orchestrator (plan §6.2): one round per main-thread
assistant message, opened lazily, grouped by message id.

Plan §12.1 row coverage (assertion: tests; Class.method or a whole Class):
- lazy rounds:
    RoundsTest.test_rounds_open_lazily_on_the_first_text
    RoundsTest.test_a_turn_without_assistant_output_opens_no_round
- message_id grouping:
    RoundGroupingTest.test_blocks_of_one_message_form_a_single_round
- subagent exclusion:
    RoundsTest.test_subagent_tool_calls_produce_no_round
- balanced round callbacks, incl. a round cancelled mid-stream:
    BalancedRoundsTest
- frame order tokens -> message_end -> pending_input:
    FrameOrderTest
- monotonic iteration across the vendor turns of one call:
    IterationTest
Also here: a turn boundary carries the widget round's cache folder
(RoundGroupingTest).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    SessionStarted,
    TextDelta,
)
from fakes import RecordingInteractive, subagent_tool, text, tools, turn_end
from helpers import make_native, wait_for
from later.unittest import TestCase


def _round_hooks():
    starts, completes = [], []

    async def on_round_start(iteration, turn):
        starts.append(iteration)

    async def on_round_complete(_inf, iteration, turn, raw, clean, display, conv):
        completes.append((iteration, display))

    return starts, completes, on_round_start, on_round_complete


class RoundsTest(TestCase):
    async def test_rounds_open_lazily_on_the_first_text(self) -> None:
        starts, _completes, on_start, on_complete = _round_hooks()
        probe = {}

        async def quiet_then_text(backend, request):
            yield SessionStarted(session_id=backend.session_id)
            await asyncio.sleep(0.05)
            probe["rounds_before_text"] = len(starts)
            yield TextDelta(message_id="m0", text="now")
            yield MessageEnd(message_id="m0", text="now", message_uuid="u0")
            yield turn_end(backend)

        native, _, _, _ = make_native([quiet_then_text])
        async with native:
            await native.run_agentic_loop(
                "go", on_round_start=on_start, on_round_complete=on_complete
            )
            self.assertEqual(probe["rounds_before_text"], 0)
            self.assertEqual(starts, [0])

    async def test_a_turn_without_assistant_output_opens_no_round(self) -> None:
        starts, completes, on_start, on_complete = _round_hooks()

        async def empty(backend, request):
            yield turn_end(backend)

        native, _, _, _ = make_native([empty])
        async with native:
            result = await native.run_agentic_loop(
                "go", on_round_start=on_start, on_round_complete=on_complete
            )
            self.assertEqual((starts, completes), ([], []))
            self.assertEqual(result.iterations_used, 0)

    async def test_subagent_tool_calls_produce_no_round(self) -> None:
        starts, completes, on_start, on_complete = _round_hooks()
        native, factory, _, executor = make_native(
            [[subagent_tool("write_brief", {"topic": "x"}), text("ok")]]
        )
        async with native:
            await native.run_agentic_loop(
                "go", on_round_start=on_start, on_round_complete=on_complete
            )
            self.assertTrue(factory.last.tool_results[0][2])
            self.assertEqual(executor.calls, [])
            self.assertEqual(completes, [(0, "ok")])


class RoundGroupingTest(TestCase):
    def _hooks(self):
        starts, completes = [], []

        async def on_round_start(iteration, turn):
            starts.append(iteration)
            return {"round": iteration}

        async def on_round_complete(_inf, iteration, turn, raw, clean, display, conv):
            completes.append((iteration, display))

        return starts, completes, on_round_start, on_round_complete

    async def test_blocks_of_one_message_form_a_single_round(self) -> None:
        starts, completes, on_start, on_complete = self._hooks()

        async def two_blocks(backend, request):
            yield MessageEnd(message_id="m0", text="Let me check.", message_uuid="u0a")
            yield MessageEnd(
                message_id="m0",
                text="",
                tool_use_ids=("tu_b",),
                af_tool_use_ids=("tu_b",),
                message_uuid="u0b",
            )
            await backend.call_tool("write_brief", {"topic": "x"}, tool_use_id="tu_b")
            yield turn_end(backend)

        native, _, _, executor = make_native([two_blocks])
        async with native:
            result = await native.run_agentic_loop(
                "go", on_round_start=on_start, on_round_complete=on_complete
            )
            self.assertEqual(starts, [0])
            self.assertEqual(completes, [(0, "Let me check.")])
            self.assertEqual(result.iterations_used, 1)
            self.assertEqual(executor.calls, [("write_brief", {"topic": "x"})])

    async def test_widget_only_message_gets_its_own_round(self) -> None:
        starts, completes, on_start, on_complete = self._hooks()

        async def widget_only(backend, request):
            yield TextDelta(message_id="m0", text="Hi.")
            yield MessageEnd(message_id="m0", text="Hi.", message_uuid="u0")
            yield MessageEnd(
                message_id="m1",
                text="",
                tool_use_ids=("tu_w",),
                af_tool_use_ids=("tu_w",),
                message_uuid="u1",
            )
            await wait_for(lambda: len(starts) == 2)  # the message event was processed
            await backend.call_tool(
                "clarification",
                {"prompt": "Topic?", "output": ["topic"]},
                tool_use_id="tu_w",
            )
            yield turn_end(backend)

        native, _, interactive, _ = make_native([widget_only], answers=[None])
        async with native:
            result = await native.run_agentic_loop(
                "go", on_round_start=on_start, on_round_complete=on_complete
            )
            self.assertTrue(result.has_conversation_tool)
            self.assertEqual(starts, [0, 1])
            self.assertEqual(completes, [(0, "Hi."), (1, "")])
            self.assertEqual(interactive.round_contexts[-1], {"round": 1})

    async def test_a_turn_boundary_carries_the_widget_rounds_cache_folder(
        self,
    ) -> None:
        """As in the text protocol, where a widget is its round's: while the
        widget waits, the host's cache folder is the widget round's, not that
        of a later message of the vendor turn, and the turn boundary after the
        answer carries it; the next round moves it on."""
        starts, folders = [], []

        async def on_round_start(iteration, turn):
            starts.append(iteration)
            return {"round": iteration, "cache_folder": f"/rounds/{iteration}"}

        async def on_round_complete(inf, iteration, turn, raw, clean, display, conv):
            folders.append(inf.cache_folder)

        async def next_turn(turn, user_input):
            return turn + 1

        async def ask_then_talk(backend, request):
            yield TextDelta(message_id="m0", text="Let me ask.")
            yield MessageEnd(
                message_id="m0",
                text="Let me ask.",
                tool_use_ids=("tu_w",),
                af_tool_use_ids=("tu_w",),
                message_uuid="u0",
            )
            await wait_for(lambda: len(starts) == 1)  # the widget's round is open
            await backend.call_tool(
                "clarification",
                {"prompt": "Topic?", "output": ["topic"]},
                tool_use_id="tu_w",
            )
            yield TextDelta(message_id="m1", text="Over to you.")
            yield MessageEnd(message_id="m1", text="Over to you.", message_uuid="u1")
            yield turn_end(backend)

        native, _, interactive, _ = make_native(
            [ask_then_talk, [text("Lidar it is.")]], answers=["lidar"]
        )
        async with native:
            result = await native.run_agentic_loop(
                "go",
                turn_number=1,
                on_new_turn=next_turn,
                on_round_start=on_round_start,
                on_round_complete=on_round_complete,
            )
        self.assertEqual(result.text, "Lidar it is.")
        self.assertEqual(folders, ["/rounds/0", "/rounds/1", "/rounds/2"])
        self.assertEqual([c["round"] for c in interactive.round_contexts], [0, 1, 0, 2])
        self.assertEqual(interactive.turn_boundaries, [3])
        self.assertEqual(interactive.boundary_cache_folders, ["/rounds/0"])
        self.assertEqual(native.cache_folder, "/rounds/2")


def _ask(var: str) -> list:
    return [
        text(f"Question {var}."),
        tools(("clarification", {"prompt": "?", "output": [var]})),
    ]


class BalancedRoundsTest(TestCase):
    async def test_each_round_completes_once_and_a_cancelled_round_never(
        self,
    ) -> None:
        """Every ``on_round_start`` gets one ``on_round_complete`` with its
        iteration. A round the host cancels mid-stream is aborted and never
        completed — no late completion either (the text protocol completes
        no round it cancels); the next call's rounds balance again."""
        starts, completes = [], []

        async def on_start(iteration, turn):
            starts.append(iteration)
            return {"round": iteration}

        async def on_complete(_inf, iteration, turn, raw, clean, display, conv):
            completes.append((iteration, display))

        async def second_message_hangs(backend, request):
            yield TextDelta(message_id="m0", text="one")
            yield MessageEnd(message_id="m0", text="one", message_uuid="u0")
            yield TextDelta(message_id="m1", text="two")
            await asyncio.Event().wait()
            yield turn_end(backend)  # pragma: no cover - never reached

        native, _, _, _ = make_native(
            [second_message_hangs, _ask("a"), [text("Done.")]],
            answers=["lidar"],
            vendor_drain_timeout_s=0.2,
        )
        callbacks = {"on_round_start": on_start, "on_round_complete": on_complete}
        async with native:
            turn = asyncio.ensure_future(native.run_agentic_loop("go", **callbacks))
            await wait_for(lambda: len(starts) == 2, timeout=5)  # round 1 streams
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            await asyncio.sleep(0.3)
            self.assertEqual((starts, completes), ([0, 1], [(0, "one")]))
            starts.clear()
            completes.clear()
            await native.run_agentic_loop("again", **callbacks)
        self.assertEqual(starts, [0, 1])
        self.assertEqual(completes, [(0, "Question a."), (1, "Done.")])


class FrameOrderTest(TestCase):
    async def test_tokens_then_message_end_then_pending_input(self) -> None:
        """Plan §6.2: widgets are shown only after the vendor turn ended, so
        a round's tokens and its ``message_end`` (``on_round_complete``)
        precede the widget's ``pending_input``."""
        frames = []

        class _Frames(RecordingInteractive):
            async def stream_token_batches(
                self, tokens, session_id, send_stream_end=False, turn_number=0
            ):
                async for chunk, _meta in tokens:
                    frames.append(("token", chunk))
                return ""

            async def asend_response(self, text, **kwargs):
                frames.append(("pending_input", kwargs["input_mode"].prompt))
                await super().asend_response(text, **kwargs)

        async def on_round_start(iteration, turn_number):
            return {"round": iteration}

        async def on_round_complete(_inf, iteration, turn, raw, clean, display, conv):
            frames.append(("message_end", display))

        native, _, _, _ = make_native(
            [
                [
                    text("Let me "),
                    text("ask."),
                    tools(("clarification", {"prompt": "Topic?", "output": ["t"]})),
                ],
                [text("Thanks.")],
            ]
        )
        async with native:
            await native.run_agentic_loop(
                "go",
                interactive=_Frames(["lidar"]),
                on_round_start=on_round_start,
                on_round_complete=on_round_complete,
            )
        self.assertEqual(
            frames,
            [
                ("token", "Let me "),
                ("token", "ask."),
                ("message_end", "Let me ask."),
                ("pending_input", "Topic?"),
                ("token", "Thanks."),
                ("message_end", "Thanks."),
            ],
        )


class IterationTest(TestCase):
    """Plan §6.2: ``iteration`` grows monotonically across the vendor turns of
    one call (OpenStartup derives round directories from it)."""

    async def _iterations(
        self, *, with_context: bool, scripts: Optional[list] = None
    ) -> tuple:
        starts, completes = [], []

        async def on_round_start(iteration, turn_number):
            starts.append(iteration)
            return {"round": iteration} if with_context else None

        async def on_round_complete(_inf, iteration, turn, raw, clean, display, conv):
            completes.append(iteration)

        if scripts is None:
            scripts = [_ask("a"), _ask("b"), [text("Done.")]]
        questions = len(scripts) - 1
        native, _, interactive, _ = make_native(scripts, answers=["1", "2"])
        async with native:
            result = await native.run_agentic_loop(
                "go", on_round_start=on_round_start, on_round_complete=on_round_complete
            )
        self.assertEqual(result.text, "Done.")
        self.assertEqual(len(interactive.widgets), questions)
        return starts, completes, result.iterations_used

    async def test_iteration_grows_across_the_vendor_turns_of_one_call(self) -> None:
        starts, completes, used = await self._iterations(with_context=True)
        self.assertEqual(starts, [0, 1, 2])
        self.assertEqual(completes, starts)
        self.assertEqual(used, 3)

    async def test_iteration_grows_for_a_host_whose_rounds_carry_no_context(
        self,
    ) -> None:
        """A widget is presented in its message's round also when the host's
        ``on_round_start`` returns no context: no extra round, no iteration
        seen twice."""
        starts, completes, used = await self._iterations(with_context=False)
        self.assertEqual(starts, [0, 1, 2])
        self.assertEqual(completes, starts)
        self.assertEqual(used, 3)

    async def test_a_widget_whose_message_was_never_reported_gets_one_round(
        self,
    ) -> None:
        """A widget call no message event reported still gets a round of its
        own, whose iteration the next round does not repeat."""

        async def unreported_question(backend, request):
            await backend.call_tool(
                "clarification", {"prompt": "Topic?", "output": ["topic"]}
            )
            yield turn_end(backend)

        for with_context in (True, False):
            with self.subTest(with_context=with_context):
                starts, completes, used = await self._iterations(
                    with_context=with_context,
                    scripts=[unreported_question, [text("Done.")]],
                )
                self.assertEqual(starts, [0, 1])
                self.assertEqual(completes, starts)
                self.assertEqual(used, 2)

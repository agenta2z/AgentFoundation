"""Tests for ``InboxDriver`` and the ConversationalInferencer inbox delegation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import later.unittest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox import (
    SyntheticContinue,
    ToolCompletion,
    UserMessage,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox_driver import (
    InboxDriver,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrs

_CONTINUE_AFTER_TOOLS = "Continue based on the tool execution results above."
_SYNTHETIC_CONTINUE = "Continue per SOP guidance — advance to the next phase."
_FINAL_ANSWER = "Here is my final answer."


class _RecordingTurn:
    """Fake ``run_turn``: records every call; can stop the driver or raise."""

    def __init__(self, stop_after: int, fail_on: frozenset[int] = frozenset()) -> None:
        self.driver: InboxDriver | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._stop_after = stop_after
        self._fail_on = fail_on

    async def __call__(self, content: str, **kwargs: Any) -> str:
        self.calls.append((content, kwargs))
        n = len(self.calls)
        if n >= self._stop_after and self.driver is not None:
            self.driver.request_shutdown()
        if n in self._fail_on:
            raise ValueError(f"turn {n} failed")
        return f"result-{n}"


class _FakeRunContext:
    def __init__(self) -> None:
        self.children: list[str] = []

    def child(self, name: str) -> str:
        self.children.append(name)
        return f"ctx:{name}"


def _driver(turn: _RecordingTurn, **kwargs: Any) -> InboxDriver:
    driver = InboxDriver(turn, **kwargs)
    turn.driver = driver
    return driver


class InboxDriverTest(later.unittest.TestCase):
    async def test_user_message_runs_one_turn_with_user_origin(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn)
        interactive = object()
        on_new_turn = object()
        on_prompt_rendered = object()
        on_turn_complete = object()
        driver.enable(
            interactive=interactive,
            on_new_turn=on_new_turn,
            on_prompt_rendered=on_prompt_rendered,
            on_turn_complete=on_turn_complete,
        )
        driver.put_user("hello", source="cli")

        result = await driver.run()

        self.assertEqual(result, "result-1")
        self.assertEqual(
            turn.calls,
            [
                (
                    "hello",
                    {
                        "origin": "user",
                        "interactive": interactive,
                        "turn_number": 1,
                        "on_new_turn": on_new_turn,
                        "on_prompt_rendered": on_prompt_rendered,
                        "on_turn_complete": on_turn_complete,
                        "run_context": None,
                    },
                )
            ],
        )

    async def test_tool_completion_starts_auto_continue_turn(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn)
        driver.enable()
        driver.put(ToolCompletion(tool_name="research"))

        await driver.run()

        self.assertEqual(len(turn.calls), 1)
        content, kwargs = turn.calls[0]
        self.assertEqual(content, _CONTINUE_AFTER_TOOLS)
        self.assertEqual(kwargs["origin"], "tool_completion")
        self.assertEqual(kwargs["turn_number"], 1)

    async def test_synthetic_continue_runs_host_event_turn(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn)
        driver.enable()
        driver.put(SyntheticContinue(reason="phase advance"))

        await driver.run()

        self.assertEqual(len(turn.calls), 1)
        content, kwargs = turn.calls[0]
        self.assertEqual(content, _SYNTHETIC_CONTINUE)
        self.assertEqual(kwargs["origin"], "host_event")

    async def test_items_run_in_fifo_order_with_increasing_turn_numbers(self) -> None:
        turn = _RecordingTurn(stop_after=3)
        driver = _driver(turn)
        driver.enable()
        driver.put_user("first")
        driver.put(ToolCompletion(tool_name="t"))
        driver.put(UserMessage(content="second"))

        result = await driver.run()

        self.assertEqual(result, "result-3")
        self.assertEqual(
            [(c, kw["origin"], kw["turn_number"]) for c, kw in turn.calls],
            [
                ("first", "user", 1),
                (_CONTINUE_AFTER_TOOLS, "tool_completion", 2),
                ("second", "user", 3),
            ],
        )

    async def test_unknown_item_is_skipped_without_consuming_a_turn(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn)
        driver.enable()
        driver.put(object())
        driver.put_user("real")

        await driver.run()

        self.assertEqual(
            [(c, kw["turn_number"]) for c, kw in turn.calls], [("real", 1)]
        )
        self.assertEqual(driver.turn_counter, 1)
        queue = driver.queue
        assert queue is not None
        # Every dequeued item, skipped or not, is marked done.
        await asyncio.wait_for(queue.join(), timeout=1)

    async def test_shutdown_is_checked_between_items(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn)
        driver.enable()
        driver.put_user("one")
        driver.put_user("two")

        result = await driver.run()

        self.assertEqual(result, "result-1")
        self.assertEqual([c for c, _ in turn.calls], ["one"])
        self.assertTrue(driver.shutdown_requested)
        self.assertFalse(driver.running)
        queue = driver.queue
        assert queue is not None
        self.assertEqual(queue.qsize(), 1)

    async def test_run_returns_immediately_once_shutdown_was_requested(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn)
        driver.enable()
        driver.put_user("never")
        driver.request_shutdown()

        result = await driver.run()

        self.assertIsNone(result)
        self.assertEqual(turn.calls, [])

    async def test_failed_turn_is_logged_and_loop_continues(self) -> None:
        turn = _RecordingTurn(stop_after=3, fail_on=frozenset({1, 3}))
        driver = _driver(turn)
        driver.enable()
        driver.put_user("a")
        driver.put_user("b")
        driver.put_user("c")

        with self.assertLogs(
            "agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox_driver",
            level=logging.ERROR,
        ) as logs:
            result = await driver.run()

        self.assertEqual(result, "result-2")
        self.assertEqual([c for c, _ in turn.calls], ["a", "b", "c"])
        self.assertEqual(len(logs.records), 2)
        self.assertIn("Inbox item", logs.records[0].getMessage())
        self.assertIn("turn 1 failed", logs.records[0].getMessage())
        self.assertFalse(driver.running)

    async def test_failure_is_logged_to_the_injected_logger(self) -> None:
        turn = _RecordingTurn(stop_after=1, fail_on=frozenset({1}))
        driver = _driver(turn, log=logging.getLogger("inbox_driver_test.host"))
        driver.enable()
        driver.put_user("boom")

        with self.assertLogs("inbox_driver_test.host", level=logging.ERROR) as logs:
            result = await driver.run()

        self.assertIsNone(result)
        self.assertEqual(len(logs.records), 1)

    async def test_each_turn_runs_under_its_own_child_run_context(self) -> None:
        turn = _RecordingTurn(stop_after=2)
        driver = _driver(turn)
        driver.enable()
        driver.put_user("a")
        driver.put(object())
        driver.put_user("b")
        root = _FakeRunContext()

        await driver.run(run_context=root)

        self.assertEqual(root.children, ["turn_1", "turn_2"])
        self.assertEqual(
            [kw["run_context"] for _, kw in turn.calls], ["ctx:turn_1", "ctx:turn_2"]
        )

    async def test_custom_content_for_item(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn, content_for_item=lambda item: f"custom:{item.kind}")
        driver.enable()
        driver.put(ToolCompletion(tool_name="t"))

        await driver.run()

        self.assertEqual(turn.calls[0][0], "custom:tool_completion")
        self.assertEqual(turn.calls[0][1]["origin"], "tool_completion")

    async def test_enable_and_run_guards(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn)

        with self.assertRaisesRegex(RuntimeError, "Inbox not enabled"):
            driver.put_user("early")
        with self.assertRaisesRegex(RuntimeError, "Inbox not enabled"):
            await driver.run()

        driver.enable()
        with self.assertRaisesRegex(RuntimeError, "Inbox already enabled"):
            driver.enable()

    async def test_concurrent_run_is_rejected(self) -> None:
        turn = _RecordingTurn(stop_after=1)
        driver = _driver(turn)
        driver.enable()
        task = asyncio.create_task(driver.run())
        await asyncio.sleep(0)

        self.assertTrue(driver.running)
        with self.assertRaisesRegex(RuntimeError, "already executing"):
            await driver.run()

        driver.put_user("unblock")
        result = await task
        self.assertEqual(result, "result-1")
        self.assertFalse(driver.running)


@attrs(slots=False)
class _FinalAnswerBase(InferencerBase):
    def _infer(self, inp, cfg=None, **kw):
        return _FINAL_ANSWER

    async def _ainfer(self, inp, cfg=None, **kw):
        return _FINAL_ANSWER


class ConversationalInferencerInboxTest(later.unittest.TestCase):
    async def test_run_drives_run_agentic_loop_with_origin(self) -> None:
        ci = ConversationalInferencer(base_inferencer=_FinalAnswerBase())
        real_loop = ci.run_agentic_loop
        calls: list[dict[str, Any]] = []
        completed_turns: list[int] = []

        async def recording_loop(**kwargs: Any) -> Any:
            calls.append(kwargs)
            result = await real_loop(**kwargs)
            if len(calls) == 2:
                ci.request_shutdown()
            return result

        async def on_turn_complete(turn_number: int) -> None:
            completed_turns.append(turn_number)

        ci.run_agentic_loop = recording_loop
        ci.enable_inbox(None, on_turn_complete=on_turn_complete)
        ci.inbox_put_user("hello")
        ci._inbox.put_nowait(ToolCompletion(tool_name="t"))

        result = await ci.run()

        self.assertEqual(result.text, _FINAL_ANSWER)
        self.assertEqual(
            calls,
            [
                {
                    "content": "hello",
                    "origin": "user",
                    "interactive": None,
                    "turn_number": 1,
                    "on_new_turn": None,
                    "on_prompt_rendered": None,
                    "on_turn_complete": on_turn_complete,
                    "run_context": None,
                },
                {
                    "content": _CONTINUE_AFTER_TOOLS,
                    "origin": "tool_completion",
                    "interactive": None,
                    "turn_number": 2,
                    "on_new_turn": None,
                    "on_prompt_rendered": None,
                    "on_turn_complete": on_turn_complete,
                    "run_context": None,
                },
            ],
        )
        self.assertEqual(len(completed_turns), 2)

    async def test_enable_inbox_forwards_interactive_and_callbacks(self) -> None:
        ci = ConversationalInferencer(base_inferencer=_FinalAnswerBase())
        calls: list[dict[str, Any]] = []

        async def fake_loop(**kwargs: Any) -> str:
            calls.append(kwargs)
            ci.request_shutdown()
            return "done"

        ci.run_agentic_loop = fake_loop
        interactive = object()
        on_new_turn = object()
        on_prompt_rendered = object()
        ci.enable_inbox(
            interactive,
            on_new_turn=on_new_turn,
            on_prompt_rendered=on_prompt_rendered,
        )
        ci.inbox_put_user("hi")
        root = _FakeRunContext()

        result = await ci.run(run_context=root)

        self.assertEqual(result, "done")
        self.assertEqual(calls[0]["interactive"], interactive)
        self.assertEqual(calls[0]["on_new_turn"], on_new_turn)
        self.assertEqual(calls[0]["on_prompt_rendered"], on_prompt_rendered)
        self.assertEqual(calls[0]["run_context"], "ctx:turn_1")

    async def test_failed_turn_logs_to_the_inferencer_module_logger(self) -> None:
        ci = ConversationalInferencer(base_inferencer=_FinalAnswerBase())

        async def failing_loop(**kwargs: Any) -> str:
            ci.request_shutdown()
            raise ValueError("loop exploded")

        ci.run_agentic_loop = failing_loop
        ci.enable_inbox()
        ci.inbox_put_user("hi")

        with self.assertLogs(
            "agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer",
            level=logging.ERROR,
        ) as logs:
            result = await ci.run()

        self.assertIsNone(result)
        self.assertIn("loop exploded", logs.records[0].getMessage())

    async def test_inbox_state_is_exposed_through_the_legacy_attributes(self) -> None:
        ci = ConversationalInferencer(base_inferencer=_FinalAnswerBase())

        self.assertIsNone(ci._inbox)
        self.assertFalse(ci._shutdown_requested)
        self.assertFalse(ci.shutdown_requested)

        ci.enable_inbox(auto_shutdown_on_sop_complete=True, maxsize=3)
        self.assertIsInstance(ci._inbox, asyncio.Queue)
        self.assertEqual(ci._inbox.maxsize, 3)
        self.assertTrue(ci.sop_controller._auto_shutdown_on_sop_complete)
        with self.assertRaisesRegex(RuntimeError, "Inbox already enabled"):
            ci.enable_inbox()

        ci.sop_controller.request_shutdown()
        self.assertTrue(ci._shutdown_requested)
        self.assertTrue(ci.shutdown_requested)

        ci._shutdown_requested = False
        self.assertFalse(ci.shutdown_requested)

    async def test_inbox_put_requires_enable(self) -> None:
        ci = ConversationalInferencer(base_inferencer=_FinalAnswerBase())

        with self.assertRaisesRegex(RuntimeError, "Inbox not enabled"):
            ci.inbox_put_user("early")
        with self.assertRaisesRegex(RuntimeError, "Inbox not enabled"):
            await ci.run()

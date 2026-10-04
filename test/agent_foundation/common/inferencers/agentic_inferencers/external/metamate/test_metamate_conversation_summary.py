# pyre-strict

"""Every MetaMate stream call ends with one ``MetamateConversationSummary`` record.

A parent that tolerates a failed request (a BTA quorum) keeps only the error's
type, so the record is where the server's side of a failure survives: the phase,
the polls before it, the agent's tool calls so far, and the HTTP status, debug
headers and body. A completed call records the same tool summary.
"""

from __future__ import annotations

import asyncio
import datetime
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
    metamate_sdk_inferencer,
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    summarize_tool_activity,
)

RECORDS: list[tuple[Any, Any]] = []


class _RecordingMetamate(MetamateSDKInferencer):
    def log_info(
        self, item: Any, log_type: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        RECORDS.append((log_type, item))


class _HTTPError(Exception):
    def __init__(self, response: Any) -> None:
        super().__init__("500 Server Error: Internal Server Error")
        self.response = response


def _tool(name: str, status: str, summary: str, warning: Any = None) -> Any:
    entry = SimpleNamespace(
        tool_call_display_name=name, status=status, summary=summary, warning=warning
    )
    return SimpleNamespace(
        block=SimpleNamespace(
            uuid=f"t-{name}-{summary}",
            content=SimpleNamespace(thinking_panel_entry=entry),
        )
    )


def _outputs(status: str, text: str, tools: list[Any]) -> list[Any]:
    answer = SimpleNamespace(
        block=SimpleNamespace(
            uuid="answer", content=SimpleNamespace(markdown=SimpleNamespace(value=text))
        )
    )
    message = SimpleNamespace(
        message=SimpleNamespace(role="ASSISTANT", block_uuids=["answer"], status=status)
    )
    return [message, *tools, answer]


class _FakeClient:
    polls: list[Any] = []

    def __init__(self, cat: Any = None) -> None:
        self._polls = iter(_FakeClient.polls)

    def engine_start_v2(self, **kwargs: Any) -> Any:
        return SimpleNamespace(conversation=SimpleNamespace(uuid="conv-1", fbid="fb-1"))

    def get_conversation_for_stream(self, conversation_uuid: str) -> Any:
        step = next(self._polls)
        if isinstance(step, Exception):
            raise step
        return step


SEARCH = _tool("Searching code", "COMPLETED", "repo:fbsource HSTU")
READ = _tool("Reading file", "COMPLETED", "stu.py:1-80")
BIG = _tool("Searching code", "ERROR", "repo:fbsource attention", "memory budget")


def _run(polls: list[Any]) -> Any:
    RECORDS.clear()
    _FakeClient.polls = polls
    inf = _RecordingMetamate(
        api_key=None,
        poll_interval_seconds=0,
        auto_continue=False,
        code_scope_judge=None,
    )
    with mock.patch.object(
        metamate_sdk_inferencer,
        "resolve_metamate_client_cls",
        return_value=_FakeClient,
    ):
        return asyncio.run(inf.ainfer("task"))


def _summary() -> dict[str, Any]:
    """The first stream call's record, after checking every stream call (a retry
    replays the same script) logged exactly one."""
    summaries = [
        item for kind, item in RECORDS if kind == "MetamateConversationSummary"
    ]
    starts = [item for kind, item in RECORDS if kind == "EngineStartPayload"]
    assert summaries and len(summaries) == max(len(starts), 1), (summaries, starts)
    return summaries[0]


class ConversationSummaryTest(unittest.TestCase):
    def test_a_completed_call_records_the_agents_tool_calls(self) -> None:
        result = _run(
            [
                _outputs("IN_PROGRESS", "", [SEARCH]),
                _outputs("COMPLETED", "the answer", [SEARCH, READ, BIG]),
            ]
        )
        self.assertEqual(result, "the answer")
        summary = _summary()
        self.assertEqual(summary["outcome"], "terminal")
        self.assertEqual(summary["polls"], 2)
        self.assertEqual(summary["conversation_uuid"], "conv-1")
        self.assertEqual(summary["text_chars"], len("the answer"))
        self.assertEqual(
            summary["tool_calls"], {"Searching code": 2, "Reading file": 1}
        )
        self.assertEqual(
            summary["recent_tool_calls"][-1],
            {
                "tool": "Searching code",
                "status": "ERROR",
                "summary": "repo:fbsource attention",
                "warning": "memory budget",
            },
        )
        self.assertNotIn("error", summary)

    def test_a_failed_poll_records_what_the_server_said(self) -> None:
        response = SimpleNamespace(
            status_code=500,
            reason="Internal Server Error",
            url="https://interngraph.intern.facebook.com/graphql?access_token=SECRET",
            headers={
                "X-FB-Debug": "dbg-token",
                "Content-Type": "text/html",
                "Set-Cookie": "session=SECRET",
            },
            text="",
            elapsed=datetime.timedelta(seconds=1.25),
        )
        with self.assertRaises(_HTTPError):
            _run([_outputs("IN_PROGRESS", "partial", [SEARCH]), _HTTPError(response)])
        summary = _summary()
        self.assertEqual(summary["outcome"], "failed")
        self.assertEqual(summary["phase"], "poll")
        self.assertEqual(summary["polls"], 1)
        self.assertEqual(summary["tool_calls"], {"Searching code": 1})
        error = summary["error"]
        self.assertEqual(error["type"], "_HTTPError")
        self.assertEqual(
            error["http"],
            {
                "status_code": 500,
                "reason": "Internal Server Error",
                "url": "https://interngraph.intern.facebook.com/graphql",
                "headers": {"X-FB-Debug": "dbg-token", "Content-Type": "text/html"},
                "body": "",
                "elapsed_s": 1.25,
            },
        )

    def test_an_error_without_a_response_records_no_http_details(self) -> None:
        with self.assertRaises(RuntimeError):
            _run([RuntimeError("connection reset")])
        summary = _summary()
        self.assertEqual(summary["outcome"], "failed")
        self.assertEqual(summary["polls"], 0)
        self.assertEqual(summary["blocks"], 0)
        self.assertEqual(
            summary["error"],
            {"type": "RuntimeError", "message": "connection reset", "http": None},
        )


class GeneratedTypesTest(unittest.TestCase):
    """The summary reads the client's real thrift types, not only the fakes."""

    def test_thinking_panel_entries_of_real_bridge_outputs(self) -> None:
        try:
            import metamate.sdk.sdk.thrift_types as sdk_types
        except ImportError:
            self.skipTest("the MetaMate SDK thrift types are linked only under buck")
        entry = sdk_types.BlockContentThinkingPanelEntry(
            summary="repo:fbsource HSTU",
            tool_call_display_name="Searching code",
            status=sdk_types.ThinkingPanelEntryStatus.ERROR,
            uuid="e1",
            warning="memory budget",
        )
        outputs = [
            sdk_types.BridgeOutput(
                message=sdk_types.Message(uuid="m1", conversation_uuid="c1")
            ),
            sdk_types.BridgeOutput(
                block=sdk_types.Block(
                    uuid="b1",
                    message_uuid="m1",
                    content=sdk_types.BlockContent(thinking_panel_entry=entry),
                )
            ),
            sdk_types.BridgeOutput(
                block=sdk_types.Block(
                    uuid="b2",
                    message_uuid="m1",
                    content=sdk_types.BlockContent(
                        markdown=sdk_types.BlockContentMarkdown(value="answer")
                    ),
                )
            ),
        ]
        self.assertEqual(
            summarize_tool_activity(outputs),
            {
                "blocks": 2,
                "tool_calls": {"Searching code": 1},
                "recent_tool_calls": [
                    {
                        "tool": "Searching code",
                        "status": "ERROR",
                        "summary": "repo:fbsource HSTU",
                        "warning": "memory budget",
                    }
                ],
            },
        )

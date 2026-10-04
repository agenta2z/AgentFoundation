"""ClaudeCodeCliInferencer.ainfer: the reply is the streamed text, verbatim, and
the call's outcome comes from the stream's ``result`` event — never from JSON
found in the reply. A reply containing JSON used to be mistaken for the CLI's
own JSON output (its text was replaced by the object's ``result`` field,
usually ``""``, and a ``session_id`` key in it was adopted as the session).
When a message's stream breaks off, Claude Code re-sends the message without
text deltas; the reply is then the ``result`` event's final message.

Drives the real streaming transport against a stand-in ``claude`` (a shell
script printing stream-json), so no Claude Code process is started.
"""

import asyncio
import json
import os
import shlex
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)


def _stand_in_claude(
    reply: str, *, is_error: bool = False, streamed: str | None = None
) -> str:
    """``streamed`` is the text the deltas carry (default: ``reply``); ``reply``
    is the final message, which the ``result`` event reports."""
    events = [
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": reply if streamed is None else streamed,
                },
            },
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": reply}]},
        },
        {
            "type": "result",
            "subtype": "error_during_execution" if is_error else "success",
            "is_error": is_error,
            "result": reply,
            "session_id": "sess-123",
            "num_turns": 1,
        },
    ]
    directory = tempfile.mkdtemp(prefix="claude_stand_in_")
    out = os.path.join(directory, "stdout")
    with open(out, "w") as f:
        f.write("".join(json.dumps(e) + "\n" for e in events))
    path = os.path.join(directory, "claude")
    with open(path, "w") as f:
        f.write(f"#!/bin/sh\ncat > /dev/null\ncat {shlex.quote(out)}\n")
    os.chmod(path, 0o700)
    return path


def _ainfer(reply: str, **kw):
    inferencer = ClaudeCodeCliInferencer(
        target_path=tempfile.mkdtemp(), claude_command=_stand_in_claude(reply, **kw)
    )
    return inferencer, asyncio.run(inferencer.ainfer("hello"))


class AsyncResultTest(unittest.TestCase):
    def test_a_json_reply_is_returned_verbatim(self) -> None:
        reply = '{"subtasks": ["a", "b"], "session_id": "not-a-session"}'
        inferencer, response = _ainfer(reply)
        self.assertEqual(response.output, reply)
        self.assertEqual(response.raw_output, reply)
        self.assertEqual(response.session_id, "sess-123")
        self.assertEqual(inferencer.active_session_id, "sess-123")

    def test_a_reply_with_embedded_json_is_returned_verbatim(self) -> None:
        reply = 'Here is the plan: {"step": 1} and the rest.'
        _inferencer, response = _ainfer(reply)
        self.assertEqual(response.output, reply)

    def test_raw_output_keeps_the_reply_envelope(self) -> None:
        reply = '<Response>\nDone.\n```json\n{"name": "x"}\n```\n</Response>'
        _inferencer, response = _ainfer(reply)
        self.assertEqual(response.raw_output, reply)
        self.assertTrue(response.success)

    def test_an_error_result_fails_the_call(self) -> None:
        _inferencer, response = _ainfer("API Error: overloaded", is_error=True)
        self.assertFalse(response.success)
        self.assertEqual(response.output, "API Error: overloaded")

    def test_interim_text_before_the_final_message_is_kept(self) -> None:
        final = "<Response>\nDone.\n</Response>"
        streamed = "Checking the two files first.\n" + final
        _inferencer, response = _ainfer(final, streamed=streamed)
        self.assertEqual(response.output, streamed)

    def test_a_resent_final_message_replaces_the_abandoned_stream(self) -> None:
        """The deltas stop mid-message (11.6 of 24.7 K chars in a fan-out gate's
        breakdown, cutting its subtask JSON); the message arrives again whole,
        as a non-streamed event, and the result event reports it."""
        final = '<Response>\nThree subtasks.\n```json\n{"subtasks": [1, 2, 3]}\n```\n</Response>'
        streamed = (
            'Fact-checking is done.<Response>\nI split it.\n```json\n{"subtasks": [1,'
        )
        _inferencer, response = _ainfer(final, streamed=streamed)
        self.assertEqual(response.output, final)
        self.assertEqual(response.raw_output, final)
        self.assertTrue(response.success)

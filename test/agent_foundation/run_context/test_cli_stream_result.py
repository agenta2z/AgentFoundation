"""The CLI stream's final metadata lives in the invocation (plan v8 §13, P10 c2; B28).

claude CLI's async stream records its final ``result`` event, and codex CLI's its
``thread.started`` id and ``turn.completed`` usage, for the async post-hook of the same
invocation, which adopts the streamed session id. Neither is written onto the
instance, so overlapping calls on one leaf each adopt their own stream's session.
"""

from __future__ import annotations

import asyncio
import sys

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext

# Emits its final ``result`` event at once; for a prompt containing "slow" the
# process then lingers, so the other overlapping call finishes its stream first.
FAKE_CLAUDE = r"""
import json, sys, time

prompt = sys.stdin.read()
sid = "sid-" + prompt.strip().split()[-1]
for event in (
    {"type": "stream_event", "event": {"type": "content_block_delta",
     "delta": {"type": "text_delta", "text": "reply"}}},
    {"type": "result", "subtype": "success", "result": "reply", "session_id": sid},
):
    print(json.dumps(event), flush=True)
if "slow" in prompt:
    time.sleep(0.6)
"""


def _claude(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_MAX_CONCURRENCY", raising=False)
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    return ClaudeCodeCliInferencer(
        claude_command=f"{sys.executable} {script}", target_path=str(work)
    )


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def test_overlapping_claude_calls_each_adopt_their_own_stream_session(
    tmp_path, monkeypatch
):
    claude = _claude(tmp_path, monkeypatch)

    async def main():
        slow = asyncio.ensure_future(
            claude.ainfer("slow a", run_context=_host(tmp_path / "a"))
        )
        await asyncio.sleep(0.2)
        fast = await claude.ainfer("fast b", run_context=_host(tmp_path / "b"))
        return await slow, fast

    slow, fast = asyncio.run(main())
    assert (slow.session_id, fast.session_id) == ("sid-a", "sid-b")
    assert "_last_stream_result" not in vars(claude)


def test_a_bare_call_adopts_its_stream_session(tmp_path, monkeypatch):
    claude = _claude(tmp_path, monkeypatch)
    result = asyncio.run(claude.ainfer("bare c"))
    assert result.session_id == "sid-c"
    assert claude.active_session_id == "sid-c"
    assert "_last_stream_result" not in vars(claude)

"""ClaudeCodeCliInferencer's concurrency pools.

Every ``claude`` subprocess holds a slot of its inferencer's pool for its whole
lifetime. The default pool (four slots per event loop) is shared by every
inferencer that names no other, so long runs that fill it make any further
call in it wait; an inferencer in its own pool (an interactive chat leaf) is
never held up by them.

Drives the real streaming transport against a stand-in ``claude``: a prompt
containing ``hold`` streams until the call is cancelled, any other prompt is
answered at once. Each process logs its prompt when it starts.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from later.unittest import TestCase

_STAND_IN = r"""
import json, sys, time

prompt = sys.stdin.read()
with open(__LOG__, "a") as f:
    f.write(prompt.replace("\n", " ") + "\n")
if "hold" in prompt:
    while True:
        print(json.dumps({"type": "stream_event", "event": {"type": "ping"}}), flush=True)
        time.sleep(0.05)
reply = "reply to " + prompt.strip()
delta = {"type": "text_delta", "text": reply}
print(json.dumps({"type": "stream_event",
                  "event": {"type": "content_block_delta", "delta": delta}}))
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": reply, "session_id": "sess-1"}))
"""


class ConcurrencyPoolTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.log = self.root / "started.log"
        script = self.root / "claude.py"
        script.write_text(_STAND_IN.replace("__LOG__", repr(str(self.log))))
        self.command = f"{sys.executable} {script}"
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_")}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runs: list[asyncio.Task] = []

    async def _stop_all(self) -> None:
        for run in self.runs:
            run.cancel()
        await asyncio.gather(*self.runs, return_exceptions=True)

    def _inferencer(self, **kwargs) -> ClaudeCodeCliInferencer:
        return ClaudeCodeCliInferencer(
            claude_command=self.command,
            target_path=str(self.root),
            max_retry=0,
            **kwargs,
        )

    def _started(self) -> list[str]:
        if not self.log.exists():
            return []
        return [line.strip() for line in self.log.read_text().splitlines()]

    def _start(self, prompt: str, **kwargs) -> asyncio.Task:
        run = asyncio.create_task(self._inferencer(**kwargs).ainfer(prompt))
        self.runs.append(run)
        return run

    async def _until_started(self, count: int) -> None:
        deadline = time.monotonic() + 30
        while len(self._started()) < count:
            for run in self.runs:
                self.assertFalse(run.done(), f"a call ended early: {run}")
            self.assertLess(time.monotonic(), deadline, self._started())
            await asyncio.sleep(0.05)

    async def _fill_default_pool(self) -> None:
        for i in range(4):
            self._start(f"hold {i}")
        await self._until_started(4)

    async def test_the_default_pool_runs_four_claude_processes_at_a_time(
        self,
    ) -> None:
        try:
            await self._fill_default_pool()

            waiting = self._start("hold queued")
            await asyncio.sleep(1)
            self.assertNotIn("hold queued", self._started())

            first = self.runs.pop(0)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await self._until_started(5)
            self.assertEqual(self._started()[-1], "hold queued")
            self.assertFalse(waiting.done())
        finally:
            await self._stop_all()

    async def test_a_call_in_its_own_pool_runs_while_the_default_pool_is_full(
        self,
    ) -> None:
        try:
            await self._fill_default_pool()
            queued = self._start("hold queued")

            chat = self._inferencer(
                concurrency_pool="interactive", concurrency_pool_cap=2
            )
            response = await asyncio.wait_for(chat.ainfer("chat"), timeout=30)

            self.assertEqual(response.output, "reply to chat")
            self.assertNotIn("hold queued", self._started())
            self.assertFalse(queued.done())
        finally:
            await self._stop_all()

    async def test_a_pool_cap_counts_only_the_calls_in_that_pool(self) -> None:
        chat = {"concurrency_pool": "chat", "concurrency_pool_cap": 2}
        try:
            for i in range(2):
                self._start(f"hold chat {i}", **chat)
            await self._until_started(2)
            self._start("hold chat queued", **chat)

            work = self._inferencer().ainfer("work")
            response = await asyncio.wait_for(work, timeout=30)

            self.assertEqual(response.output, "reply to work")
            self.assertNotIn("hold chat queued", self._started())
        finally:
            await self._stop_all()

    async def test_a_pool_keeps_the_cap_of_its_first_call(self) -> None:
        first = self._inferencer(concurrency_pool="chat", concurrency_pool_cap=2)
        second = self._inferencer(concurrency_pool="chat", concurrency_pool_cap=3)

        semaphore = first._concurrency_semaphore()
        with self.assertLogs(level="WARNING") as logs:
            self.assertIs(second._concurrency_semaphore(), semaphore)

        self.assertIn("'chat' already has 2 slots; ignoring cap 3", logs.output[0])
        self.assertIsNot(self._inferencer()._concurrency_semaphore(), semaphore)
        uncapped = self._inferencer(concurrency_pool="free", concurrency_pool_cap=0)
        self.assertIsNone(uncapped._concurrency_semaphore())

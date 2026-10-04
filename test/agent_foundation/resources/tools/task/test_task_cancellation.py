"""Cancelling a running ``task`` (how OpenStartup stops a background run on
session delete and server shutdown) ends it at once: every process its CLI
leaves spawned is gone and nothing writes into the workspace afterwards.

The production topology (``default.yaml``) runs with a fake CLI that never
finishes: it streams activity forever and keeps a grandchild writing into the
workspace, as a real run's tool processes do.
"""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
import time
from pathlib import Path
from unittest import mock

from agent_foundation.resources.tools.task.executor import execute
from later.unittest import TestCase

# ``$1`` tells the CLIs apart: ``claude -p ...`` vs ``codex exec ...``.
_FAKE_CLI = """#!/bin/bash
[ "$1" = "--version" ] && { echo "fake 0.0"; exit 0; }
echo "$1" >> "$FAKE_CLI_STATE/first_args.txt"
cat > /dev/null &
( while true; do echo grandchild >> "$PWD/writes.log"; sleep 0.05; done ) &
echo "$$ $!" >> "$FAKE_CLI_STATE/pids.txt"
while true; do
  echo '{"type":"stream_event","event":{"type":"ping"}}'
  echo child >> "$PWD/writes.log"
  sleep 0.05
done
"""


def _running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


def _files(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in root.rglob("*")
        if p.is_file()
    }


class TaskCancellationTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.workspace = self.root / "tasks" / "task_cancel"
        self.workspace.mkdir(parents=True)
        self.state = Path(tempfile.mkdtemp(dir=tmp.name, prefix="cli_state_"))
        fake = self.state / "cli"
        fake.write_text(_FAKE_CLI)
        fake.chmod(0o755)
        env = mock.patch.dict(
            os.environ,
            {
                "CLAUDE_CODE_COMMAND": str(fake),
                "CODEX_COMMAND": str(fake),
                "FAKE_CLI_STATE": str(self.state),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self._kill_leftovers)

    def _pids(self) -> list[int]:
        pids = self.state / "pids.txt"
        return [int(p) for p in pids.read_text().split()] if pids.exists() else []

    def _kill_leftovers(self) -> None:
        for pid in self._pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def _start_run(self, **arguments: str) -> asyncio.Task:
        run = asyncio.create_task(
            execute(
                {"request": "Document the module.", **arguments},
                {"working_dir": str(self.workspace), "session_root": str(self.root)},
            )
        )
        deadline = time.monotonic() + 60
        while not (self.workspace / "writes.log").exists():
            self.assertFalse(run.done(), "the run ended before its CLI started")
            self.assertLess(time.monotonic(), deadline, "the CLI never started")
            await asyncio.sleep(0.05)
        return run

    async def _cancel_and_check(self, run: asyncio.Task, cli_first_arg: str) -> None:
        await asyncio.sleep(0.3)

        run.cancel()
        done, _ = await asyncio.wait({run}, timeout=10)

        self.assertEqual(done, {run}, "the cancelled run did not end")
        self.assertTrue(run.cancelled())
        first_args = (self.state / "first_args.txt").read_text().split()
        self.assertEqual(first_args, [cli_first_arg])
        pids = self._pids()
        self.assertEqual(len(pids), 2)
        self.assertEqual([p for p in pids if _running(p)], [])
        before = _files(self.root)
        await asyncio.sleep(0.5)
        self.assertEqual(_files(self.root), before)

    async def test_cancel_kills_the_claude_process_tree_and_stops_all_writes(
        self,
    ) -> None:
        await self._cancel_and_check(await self._start_run(), "-p")

    async def test_cancel_kills_the_codex_process_tree_and_stops_all_writes(
        self,
    ) -> None:
        run = await self._start_run(config='{"_target_": "CodexCLI"}')
        await self._cancel_and_check(run, "exec")

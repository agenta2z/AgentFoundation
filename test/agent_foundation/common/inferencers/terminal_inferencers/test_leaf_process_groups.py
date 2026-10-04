"""Every terminal CLI leaf runs its CLI in a process group of its own, ended as a
whole.

The spawn helpers of ``TerminalInferencerBase`` / ``TerminalSessionInferencerBase``
start each CLI in its own session and register the group in ``process_groups``
until the leaf ended it. So a cancelled or abandoned call ends the CLI and
everything it spawned at once; so does a sync call's timeout (where
``subprocess.run`` killed the direct child only, then waited for the pipes its
survivors held); and an interpreter that exits without that cleanup reaps the
group.

Each leaf family (Devmate, RovoDev, Kiro, Metamate, Claude Code, Codex) runs a
fake CLI that starts a long-running grandchild holding the CLI's stdout,
appends "<cli pid> <grandchild pid>" to a file and streams activity until
killed. The interpreter-exit cases run a real interpreter (a subprocess) that
leaves one async and one sync call running and exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from typing import Any, List
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer import (
    CodexCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer import (
    KiroCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_cli_inferencer import (
    MetamateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
    common as rovodev_common,
    rovodev_cli_inferencer as rovodev_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
)
from agent_foundation.common.inferencers.terminal_inferencers import process_groups
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    TerminalInferencerBase,
)
from attr import attrs
from later.unittest import TestCase

# The executable each leaf runs: Kiro and Metamate find theirs on PATH.
_BINARY = {"kiro": "kiro-cli", "metamate": "buck"}

# Says it is working as Claude (a text delta), then as Codex (an agent message);
# the other leaves stream the lines as they are.
_FAKE_CLI = """#!/bin/bash
[ "$1" = "--version" ] && {{ echo "fake 0.0"; exit 0; }}
( while true; do sleep 0.05; done ) &
echo "$$ $!" >> {pids}
echo '{{"type":"stream_event","event":{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"working"}}}}}}'
echo '{{"type":"item.completed","item":{{"type":"agent_message","text":"working"}}}}'
while true; do
  echo '{{"type":"stream_event","event":{{"type":"ping"}}}}'
  sleep 0.05
done
"""


def _write_fake_cli(tmp: Path, name: str) -> Path:
    """The fake CLI as ``tmp / "bin" / name``; it appends to ``tmp / "cli.pids"``."""
    cli = tmp / "bin" / name
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text(_FAKE_CLI.format(pids=json.dumps(str(tmp / "cli.pids"))))
    cli.chmod(0o755)
    return cli


def build_leaf(kind: str, tmp: Path, stack: contextlib.ExitStack) -> Any:
    """A real ``kind`` leaf whose CLI is the fake, set up in ``tmp``; ``stack``
    holds the patches it needs."""
    cli = _write_fake_cli(tmp, _BINARY.get(kind, "fake-cli"))
    path = f"{cli.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    stack.enter_context(mock.patch.dict(os.environ, {"PATH": path}))
    for name in ("CLAUDE_CODE_COMMAND", "CLAUDE_CODE_MAX_CONCURRENCY", "CODEX_COMMAND"):
        os.environ.pop(name, None)
    work = tmp / "work"
    work.mkdir(exist_ok=True)
    if kind == "devmate":
        (work / ".sl").mkdir(exist_ok=True)
        return DevmateCliInferencer(target_path=str(work), cli_binary=str(cli))
    if kind == "rovodev":
        sessions = tmp / "sessions"
        sessions.mkdir(exist_ok=True)
        for name in ("find_latest_session_id", "ensure_session_metadata"):
            real = getattr(rovodev_common, name)
            fake = functools.partial(real, sessions_dir=str(sessions))
            stack.enter_context(mock.patch.object(rovodev_module, name, fake))
        return RovoDevCliInferencer(acli_path=str(cli), target_path=str(work))
    if kind == "kiro":
        return KiroCliInferencer(target_path=str(work))
    if kind == "metamate":
        return MetamateCliInferencer(target_path=str(work))
    if kind == "claude":
        return ClaudeCodeCliInferencer(claude_command=str(cli), target_path=str(work))
    if kind == "codex":
        return CodexCliInferencer(codex_command=str(cli), target_path=str(work))
    raise ValueError(kind)


def _cli_pids(tmp: Path) -> List[int]:
    """The CLI pid of each started fake, in start order."""
    pids_file = tmp / "cli.pids"
    if not pids_file.exists():
        return []
    return [int(line.split()[0]) for line in pids_file.read_text().splitlines()]


def _group_members(pgid: int) -> List[int]:
    """The live (non-zombie) processes in process group ``pgid``."""
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            members.append(int(entry))
    return members


def _wait_until_empty(pgid: int, timeout: float = 5.0) -> List[int]:
    deadline = time.monotonic() + timeout
    while (members := _group_members(pgid)) and time.monotonic() < deadline:
        time.sleep(0.05)
    return members


def _kill_fakes(tmp: Path) -> None:
    """Test cleanup: whatever the fakes left, should a test fail."""
    pids_file = tmp / "cli.pids"
    if not pids_file.exists():
        return
    for line in pids_file.read_text().splitlines():
        for pid in line.split():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid), signal.SIGKILL)


class _LeafTestMixin:
    def setUp(self) -> None:
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.stack = contextlib.ExitStack()
        self.addCleanup(_kill_fakes, self.tmp)
        self.addCleanup(self.stack.close)

    def _leaf(self, kind: str) -> Any:
        return build_leaf(kind, self.tmp, self.stack)

    def _started_group(self, count: int = 1) -> int:
        """The process group of the ``count``-th fake, once it started."""
        deadline = time.monotonic() + 30
        while len(pids := _cli_pids(self.tmp)) < count:
            self.assertLess(time.monotonic(), deadline, "the CLI never started")
            time.sleep(0.02)
        pgid = os.getpgid(pids[count - 1])
        self.assertNotEqual(pgid, os.getpgrp(), "the CLI runs in the host's group")
        self.assertIn(pgid, process_groups.registered())
        return pgid

    def _assert_ended(self, pgid: int) -> None:
        self.assertEqual(_wait_until_empty(pgid, 2.0), [], "the CLI tree survived")
        self.assertNotIn(pgid, process_groups.registered())

    def _assert_tree_of_first_fake_ended(self) -> None:
        line = (self.tmp / "cli.pids").read_text().splitlines()[0]
        for pid in map(int, line.split()):
            self.assertFalse(_alive(pid), f"{pid} survived the timeout")
        self.assertEqual(process_groups.registered(), frozenset())


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class CancelledCallTest(_LeafTestMixin, TestCase):
    """Cancelling ``ainfer`` ends the CLI's whole tree at once (not after the
    5 s exit wait meant for a CLI that already exited)."""

    async def _cancelled(self, kind: str) -> None:
        leaf = self._leaf(kind)
        call = asyncio.create_task(leaf.ainfer("hello"))
        deadline = time.monotonic() + 30
        while not _cli_pids(self.tmp):
            self.assertLess(time.monotonic(), deadline, "the CLI never started")
            await asyncio.sleep(0.02)
        pgid = self._started_group()

        started = time.monotonic()
        call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call

        self.assertLess(time.monotonic() - started, 4.0, "the cancel waited")
        self._assert_ended(pgid)

    async def test_devmate(self) -> None:
        await self._cancelled("devmate")

    async def test_rovodev(self) -> None:
        await self._cancelled("rovodev")

    async def test_kiro(self) -> None:
        await self._cancelled("kiro")

    async def test_metamate(self) -> None:
        await self._cancelled("metamate")

    async def test_claude(self) -> None:
        await self._cancelled("claude")

    async def test_codex(self) -> None:
        await self._cancelled("codex")


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class SyncStreamClosedEarlyTest(_LeafTestMixin, unittest.TestCase):
    """Closing ``infer_streaming`` mid-stream ends the CLI's whole tree (the
    native sync transport of Devmate and Claude, the threaded async one of the
    rest)."""

    def _closed_early(self, kind: str) -> None:
        stream = self._leaf(kind).infer_streaming("hello")
        for chunk in stream:
            if chunk.strip():
                break
        pgid = self._started_group()

        stream.close()

        self._assert_ended(pgid)

    def test_devmate(self) -> None:
        self._closed_early("devmate")

    def test_rovodev(self) -> None:
        self._closed_early("rovodev")

    def test_kiro(self) -> None:
        self._closed_early("kiro")

    def test_metamate(self) -> None:
        self._closed_early("metamate")

    def test_claude(self) -> None:
        self._closed_early("claude")

    def test_codex(self) -> None:
        self._closed_early("codex")


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class SyncTimeoutTest(_LeafTestMixin, unittest.TestCase):
    """A sync call's timeout ends the CLI's whole tree, and returns although
    the grandchild held the CLI's pipes."""

    def _assert_base_timeout(self, kind: str) -> None:
        leaf = self._leaf(kind)
        leaf.timeout = 1
        started = time.monotonic()

        result = leaf._infer("hello")

        self.assertLess(time.monotonic() - started, 10.0)
        self.assertEqual(result.return_code, -1)
        self._assert_tree_of_first_fake_ended()

    def _assert_raises_timeout(self, kind: str) -> None:
        leaf = self._leaf(kind)
        started = time.monotonic()

        with self.assertRaises(subprocess.TimeoutExpired):
            leaf._infer("hello", subprocess_timeout_seconds=1)

        self.assertLess(time.monotonic() - started, 10.0)
        self._assert_tree_of_first_fake_ended()

    def test_devmate(self) -> None:
        self._assert_base_timeout("devmate")

    def test_kiro(self) -> None:
        self._assert_base_timeout("kiro")

    def test_metamate(self) -> None:
        self._assert_base_timeout("metamate")

    def test_claude(self) -> None:
        self._assert_raises_timeout("claude")

    def test_codex(self) -> None:
        self._assert_raises_timeout("codex")


def _alive(pid: int, timeout: float = 2.0) -> bool:
    """Whether ``pid`` is still a live (non-zombie) process ``timeout`` seconds
    from now (False as soon as it is not)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with open(f"/proc/{pid}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except OSError:
            return False
        if state == "Z" or time.monotonic() >= deadline:
            return state != "Z"
        time.sleep(0.05)


# Leaves an ``ainfer`` task and a sync ``infer`` (in a daemon thread) running,
# prints their process groups and what is registered, and exits without the
# cleanup that would end them: the loop is never resumed nor closed.
_HOST = textwrap.dedent(
    """\
    import asyncio, contextlib, importlib.util, json, os, sys, threading, time
    from pathlib import Path

    sys.path[:0] = json.loads(sys.argv[1])
    spec = importlib.util.spec_from_file_location("leaves", sys.argv[2])
    leaves = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaves)
    from agent_foundation.common.inferencers.terminal_inferencers import (
        process_groups,
    )

    kind, tmp = sys.argv[3], Path(sys.argv[4])
    stack = contextlib.ExitStack()
    leaf = leaves.build_leaf(kind, tmp, stack)

    def started(count):
        while len(leaves._cli_pids(tmp)) < count:
            time.sleep(0.02)

    async def until_started():
        while len(leaves._cli_pids(tmp)) < 1:
            await asyncio.sleep(0.02)

    loop = asyncio.new_event_loop()
    call = loop.create_task(leaf.ainfer("hello"))
    loop.run_until_complete(until_started())
    threading.Thread(target=leaf.infer, args=("hello",), daemon=True).start()
    started(2)
    groups = [os.getpgid(pid) for pid in leaves._cli_pids(tmp)]
    print(json.dumps({"groups": groups,
                      "registered": sorted(process_groups.registered())}),
          flush=True)
    """
)


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class InterpreterExitTest(_LeafTestMixin, unittest.TestCase):
    """An interpreter that exits with a leaf's calls still running, without
    the cleanup that would end them, reaps their trees."""

    def _host_exits(self, kind: str) -> None:
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                _HOST,
                json.dumps(sys.path),
                __file__,
                kind,
                str(self.tmp),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(done.returncode, 0, done.stderr)
        report = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual(report["registered"], sorted(report["groups"]))
        for pgid in report["groups"]:
            self.assertEqual(_wait_until_empty(pgid), [], f"{kind}: a tree survived")

    def test_devmate(self) -> None:
        self._host_exits("devmate")

    def test_rovodev(self) -> None:
        self._host_exits("rovodev")

    def test_kiro(self) -> None:
        self._host_exits("kiro")

    def test_metamate(self) -> None:
        self._host_exits("metamate")

    def test_claude(self) -> None:
        self._host_exits("claude")

    def test_codex(self) -> None:
        self._host_exits("codex")


@attrs
class _ShellInferencer(TerminalInferencerBase):
    """A plain terminal inferencer running a shell command."""

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        return str(inference_input)

    def parse_output(self, stdout: str, stderr: str, return_code: int) -> Any:
        return {"output": stdout, "return_code": return_code}


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class TerminalInferencerBaseSpawnTest(_LeafTestMixin, unittest.TestCase):
    """The base's own sync paths, which no leaf above takes."""

    def _fake(self) -> str:
        return str(_write_fake_cli(self.tmp, "fake-cli"))

    def test_a_stream_closed_early_ends_the_tree(self) -> None:
        stream = _ShellInferencer()._infer_streaming(self._fake())
        next(stream)
        pgid = self._started_group()

        stream.close()

        self._assert_ended(pgid)

    def test_an_interrupted_run_ends_the_tree(self) -> None:
        tmp = self.tmp

        def interrupted(process, *args, **kwargs):
            while not _cli_pids(tmp):
                time.sleep(0.02)
            raise KeyboardInterrupt

        inferencer = _ShellInferencer()
        with mock.patch.object(subprocess.Popen, "communicate", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                inferencer._run_subprocess(self._fake())

        self._assert_tree_of_first_fake_ended()

    def test_a_run_to_the_end_ends_what_the_command_left(self) -> None:
        left = self.tmp / "left.pid"
        command = f"(sleep 300 >/dev/null 2>&1 & echo $! > {left}); echo done"

        result = _ShellInferencer()._run_subprocess(command)

        self.assertEqual((result.returncode, result.stdout), (0, "done\n"))
        self.assertFalse(_alive(int(left.read_text())), "the leftover survived")
        self.assertEqual(process_groups.registered(), frozenset())

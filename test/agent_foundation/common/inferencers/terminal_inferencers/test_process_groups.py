"""The CLI process groups a process started end with it.

``process_groups`` records every CLI group the terminal CLI leaves and the
native per-turn ``CliProcess`` start (each CLI runs in its own session, out of
reach of the host's terminal and process group) until the leaf ended it. When
the interpreter exits with groups still registered — without the cleanup that
would have ended them — ``reap_all`` ends them: at ``atexit``, or in the
handler of a terminating signal once ``install_exit_signal_reaper`` ran.

The interpreter-exit cases run a real interpreter (a subprocess) that starts a
fake CLI tree (a shell, a background subshell, their ``sleep``s) in its own
session, registers it and exits without ending it.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session import (
    cli_runner,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer import (
    CodexCliInferencer,
)
from agent_foundation.common.inferencers.terminal_inferencers import process_groups
from later.unittest import TestCase

# ``$1``: where to write "<shell pid> <subshell pid>" once both run.
_TREE = """
( while true; do sleep 0.05; done ) &
echo "$$ $!" > "$1"
while true; do sleep 0.05; done
"""

_HOST = textwrap.dedent(
    """\
    import json, os, signal, subprocess, sys, time
    sys.path[:0] = json.loads(sys.argv[1])
    from agent_foundation.common.inferencers.terminal_inferencers import (
        process_groups,
    )

    tree_script, pids_file, how = sys.argv[2:5]
    tree = subprocess.Popen(
        ["bash", "-c", tree_script, "tree", pids_file],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    while not (os.path.exists(pids_file) and open(pids_file).read().strip()):
        time.sleep(0.02)
    process_groups.register(tree.pid)

    def die_of(sig):
        os.kill(os.getpid(), sig)
        time.sleep(30)
        sys.exit("still alive")

    if how == "raise":
        raise RuntimeError("the host failed")
    if how == "exit_code":
        sys.exit(3)
    if how == "unregistered":
        process_groups.unregister(tree.pid)
    if how == "forked_child_exits":
        child = os.fork()
        if child == 0:
            sys.exit(0)  # runs the inherited atexit handlers
        os.waitpid(child, 0)
        print(json.dumps(sorted(process_groups.registered())), flush=True)
    if how in ("sigterm", "sighup", "server"):
        process_groups.install_exit_signal_reaper()
    if how == "sigterm_without_reaper":
        die_of(signal.SIGTERM)
    if how == "sigterm":
        die_of(signal.SIGTERM)
    if how == "sighup":
        die_of(signal.SIGHUP)
    if how == "server":
        # A server's own SIGTERM handler for its run (as uvicorn installs),
        # put back when it stops, then the SIGTERM it caught re-raised.
        caught = []
        found = signal.signal(signal.SIGTERM, lambda sig, frame: caught.append(sig))
        os.kill(os.getpid(), signal.SIGTERM)
        while not caught:
            time.sleep(0.01)
        signal.signal(signal.SIGTERM, found)
        signal.raise_signal(caught[0])
        sys.exit("still alive")
    """
)


def _group_members(pgid: int) -> list[int]:
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


def _wait_until_empty(pgid: int, timeout: float = 5.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while (members := _group_members(pgid)) and time.monotonic() < deadline:
        time.sleep(0.05)
    return members


def _kill_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _kill_tree_in(pids_file: Path) -> None:
    pids = pids_file.read_text().split() if pids_file.exists() else []
    if pids:
        _kill_group(int(pids[0]))


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class InterpreterExitTest(unittest.TestCase):
    def _host_exits(self, how: str) -> tuple[subprocess.CompletedProcess, int]:
        """Run a host interpreter that registers a fake CLI tree and exits as
        ``how`` says; returns how it ended and the tree's pgid."""
        pids_file = Path(tempfile.mkdtemp()) / "tree.pids"
        self.addCleanup(_kill_tree_in, pids_file)
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                _HOST,
                json.dumps(sys.path),
                _TREE,
                str(pids_file),
                how,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return done, int(pids_file.read_text().split()[0])

    def _assert_tree_ended(self, how: str, returncode: int) -> None:
        done, pgid = self._host_exits(how)
        self.assertEqual(done.returncode, returncode, done.stderr)
        self.assertEqual(_wait_until_empty(pgid), [], f"{how}: the tree survived")

    def test_a_normal_exit_ends_the_registered_tree(self) -> None:
        self._assert_tree_ended("return", 0)

    def test_an_unhandled_exception_ends_the_registered_tree(self) -> None:
        self._assert_tree_ended("raise", 1)

    def test_sys_exit_ends_the_registered_tree(self) -> None:
        self._assert_tree_ended("exit_code", 3)

    def test_sigterm_ends_the_tree_with_the_signal_reaper(self) -> None:
        self._assert_tree_ended("sigterm", -signal.SIGTERM)

    def test_sighup_ends_the_tree_with_the_signal_reaper(self) -> None:
        self._assert_tree_ended("sighup", -signal.SIGHUP)

    def test_the_signal_a_server_re_raises_under_the_restored_handler(self) -> None:
        self._assert_tree_ended("server", -signal.SIGTERM)

    def test_without_the_signal_reaper_a_sigterm_leaves_the_tree_running(
        self,
    ) -> None:
        done, pgid = self._host_exits("sigterm_without_reaper")
        self.assertEqual(done.returncode, -signal.SIGTERM, done.stderr)
        time.sleep(0.3)
        self.assertNotEqual(_group_members(pgid), [], "atexit ran on SIGTERM")

    def test_an_unregistered_group_is_left_alone(self) -> None:
        done, pgid = self._host_exits("unregistered")
        self.assertEqual(done.returncode, 0, done.stderr)
        time.sleep(0.3)
        self.assertNotEqual(_group_members(pgid), [])

    def test_a_forked_child_does_not_end_its_parents_groups(self) -> None:
        done, pgid = self._host_exits("forked_child_exits")
        self.assertEqual(done.returncode, 0, done.stderr)
        # The parent still had its group after the child's atexit ran ...
        self.assertEqual(json.loads(done.stdout), [pgid])
        # ... and ended it at its own exit.
        self.assertEqual(_wait_until_empty(pgid), [])


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class ReapAllTest(unittest.TestCase):
    def _tree(self, script: str = _TREE) -> int:
        tmp = tempfile.mkdtemp()
        pids_file = os.path.join(tmp, "tree.pids")
        tree = subprocess.Popen(
            ["bash", "-c", script, "tree", pids_file], start_new_session=True
        )
        self.addCleanup(_kill_group, tree.pid)
        self.addCleanup(process_groups.unregister, tree.pid)
        deadline = time.monotonic() + 10
        while not (os.path.exists(pids_file) and Path(pids_file).read_text()):
            self.assertLess(time.monotonic(), deadline, "the tree never started")
            time.sleep(0.02)
        return tree.pid

    def test_what_ignores_sigterm_is_killed_after_the_grace(self) -> None:
        pgid = self._tree("trap '' TERM\n" + _TREE)
        process_groups.register(pgid)

        started = time.monotonic()
        process_groups.reap_all(grace_s=0.5)

        self.assertGreaterEqual(time.monotonic() - started, 0.5)
        self.assertEqual(_wait_until_empty(pgid), [])
        self.assertNotIn(pgid, process_groups.registered())

    def test_a_tree_that_exits_on_sigterm_is_not_waited_for(self) -> None:
        pgid = self._tree()
        process_groups.register(pgid)

        started = time.monotonic()
        process_groups.reap_all(grace_s=30)

        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(_wait_until_empty(pgid), [])

    def test_the_signal_reaper_leaves_a_handled_signal_alone(self) -> None:
        handler = mock.Mock()
        previous = signal.signal(signal.SIGHUP, handler)
        self.addCleanup(signal.signal, signal.SIGHUP, previous)

        process_groups.install_exit_signal_reaper([signal.SIGHUP])

        self.assertIs(signal.getsignal(signal.SIGHUP), handler)


# Records "<pid> <subshell pid>", says it is working (as Claude, then as Codex)
# and streams activity until killed; the answering variant answers and exits.
_FOREVER_CLI = """#!/bin/bash
[ "$1" = "--version" ] && { echo "fake 0.0"; exit 0; }
cat > /dev/null &
( while true; do sleep 0.05; done ) &
echo "$$ $!" > "$FAKE_CLI_PIDS"
echo '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"working"}}}'
echo '{"type":"item.completed","item":{"type":"agent_message","text":"working"}}'
while true; do
  echo '{"type":"stream_event","event":{"type":"ping"}}'
  sleep 0.05
done
"""
_ANSWERING_CLI = """#!/bin/bash
[ "$1" = "--version" ] && { echo "fake 0.0"; exit 0; }
cat > /dev/null
echo "$$" > "$FAKE_CLI_PIDS"
echo '{"type":"result","subtype":"success","result":"done","session_id":"s1"}'
echo '{"type":"turn.completed","usage":{}}'
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
class LeavesRegisterTheirGroupsTest(TestCase):
    """Registered while the CLI runs; unregistered once the leaf ended it."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.pids_file = self.tmp / "cli.pids"
        self.addCleanup(self._kill_leftovers)

    def _kill_leftovers(self) -> None:
        if self.pids_file.exists():
            for pid in self.pids_file.read_text().split():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(pid), signal.SIGKILL)

    def _fake(self, script: str) -> str:
        path = self.tmp / "cli"
        path.write_text(script)
        path.chmod(0o755)
        return str(path)

    def _leaf(self, kind: str, script: str):
        fake = self._fake(script)
        env = {
            "CLAUDE_CODE_COMMAND": fake,
            "CODEX_COMMAND": fake,
            "FAKE_CLI_PIDS": str(self.pids_file),
        }
        patch = mock.patch.dict(os.environ, env)
        patch.start()
        self.addCleanup(patch.stop)
        work = str(self.tmp / "work")
        os.makedirs(work, exist_ok=True)
        if kind == "claude":
            return ClaudeCodeCliInferencer(target_path=work)
        return CodexCliInferencer(target_path=work)

    async def _cli_pid(self) -> int:
        deadline = time.monotonic() + 30
        while not (self.pids_file.exists() and self.pids_file.read_text().strip()):
            self.assertLess(time.monotonic(), deadline, "the CLI never started")
            await asyncio.sleep(0.02)
        return int(self.pids_file.read_text().split()[0])

    async def _closed_mid_stream(self, kind: str) -> None:
        leaf = self._leaf(kind, _FOREVER_CLI)
        async with contextlib.aclosing(leaf.ainfer_streaming("hello")) as stream:
            async for chunk in stream:
                if chunk:
                    break
            pgid = os.getpgid(await self._cli_pid())
            self.assertIn(pgid, process_groups.registered())
        self.assertNotIn(pgid, process_groups.registered())
        self.assertEqual(_wait_until_empty(pgid), [])
        # Let the loop finalize the inner stream generators the leaf left.
        await asyncio.sleep(0.1)

    async def _ran_to_the_end(self, kind: str) -> None:
        leaf = self._leaf(kind, _ANSWERING_CLI)
        before = process_groups.registered()
        async with contextlib.aclosing(leaf.ainfer_streaming("hello")) as stream:
            async for _chunk in stream:
                pass
        await self._cli_pid()
        self.assertEqual(process_groups.registered(), before)

    async def test_claude_a_stream_left_early(self) -> None:
        await self._closed_mid_stream("claude")

    async def test_claude_a_stream_run_to_the_end(self) -> None:
        await self._ran_to_the_end("claude")

    async def test_codex_a_stream_left_early(self) -> None:
        await self._closed_mid_stream("codex")

    async def test_codex_a_stream_run_to_the_end(self) -> None:
        await self._ran_to_the_end("codex")

    async def test_native_cli_process_until_killed(self) -> None:
        fake = self._fake(_FOREVER_CLI)
        with mock.patch.dict(os.environ, {"FAKE_CLI_PIDS": str(self.pids_file)}):
            proc = cli_runner.CliProcess([fake], cwd=str(self.tmp), stdin_devnull=True)
            await proc.start()
        pid = await self._cli_pid()
        self.assertIn(pid, process_groups.registered())

        await proc.kill()

        self.assertNotIn(pid, process_groups.registered())
        self.assertEqual(_wait_until_empty(pid), [])

    async def test_native_cli_process_that_exits(self) -> None:
        fake = self._fake(_ANSWERING_CLI)
        with mock.patch.dict(os.environ, {"FAKE_CLI_PIDS": str(self.pids_file)}):
            proc = cli_runner.CliProcess([fake], cwd=str(self.tmp), stdin_devnull=True)
            await proc.start()
        pid = await self._cli_pid()
        async for _event in proc.json_lines():
            pass

        self.assertEqual(await proc.wait(), 0)

        self.assertNotIn(pid, process_groups.registered())

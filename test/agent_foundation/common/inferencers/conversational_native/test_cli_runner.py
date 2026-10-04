"""CliProcess: newline-delimited JSON from a per-turn vendor CLI, and ending
the turn's whole process tree."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import tempfile
import time
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session import (
    cli_runner,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.actor import (
    SessionActor,
    TurnOutcome,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    SessionOpenRequest,
    TurnRequest,
)
from fakes import scripted_cli
from later.unittest import TestCase


async def _lines(stdout: str) -> list:
    proc = cli_runner.CliProcess(
        [scripted_cli(stdout, "", 0)], cwd=tempfile.mkdtemp(), stdin_devnull=True
    )
    await proc.start()
    out = [obj async for obj in proc.json_lines()]
    await proc.wait()
    return out


class CliProcessTest(TestCase):
    async def test_event_lines_above_64_kib_are_read(self) -> None:
        # A dm session_start or a Claude tool result routinely exceeds
        # asyncio's default 64 KiB line limit.
        big = {"type": "user", "content": "x" * (1024 * 1024)}
        out = await _lines(
            json.dumps(big) + "\n" + json.dumps({"type": "result"}) + "\n"
        )
        self.assertEqual([o["type"] for o in out], ["user", "result"])
        self.assertEqual(len(out[0]["content"]), 1024 * 1024)

    async def test_a_line_above_the_limit_is_skipped_not_fatal(self) -> None:
        stdout = (
            json.dumps({"type": "first"})
            + "\n"
            + json.dumps({"blob": "y" * 5000})
            + "\n"
            + json.dumps({"type": "result"})
            + "\n"
        )
        with mock.patch.object(cli_runner, "LINE_LIMIT", 1000):
            out = await _lines(stdout)
        self.assertEqual([o.get("type") for o in out], ["first", "result"])

    async def test_non_json_lines_are_ignored(self) -> None:
        out = await _lines('banner\n\n{"type": "result"}\n')
        self.assertEqual(out, [{"type": "result"}])


# The installed claude/codex are launchers whose agent child inherits stdout.
_LAUNCHER = """#!/bin/sh
{agent} &
wait
"""
_AGENT = """#!/bin/sh
echo $$ > {pid_file}
{body}
"""


def agent_printing(first: dict) -> str:
    """An agent body that prints ``first`` as its first event, then runs on."""
    return f"echo {shlex.quote(json.dumps(first))}\nexec sleep 60"


def background_child(command: str, ready_file: str) -> str:
    """A shell line running ``command`` as a background child that writes its
    pid to ``ready_file`` from its own exec'd shell. Once the file exists, the
    traps set before this line are in place and the child's signal handling
    is final: a group signal sent earlier can miss a child forked after it."""
    inner = f"echo $$ > {shlex.quote(ready_file)}; exec {command}"
    return f"sh -c {shlex.quote(inner)} &"


def launcher_with_agent(body: str | None = None) -> tuple[str, str]:
    """A stand-in vendor CLI that runs ``body`` (default: print one event
    and sleep) as a backgrounded agent child, which inherits stdout; returns
    ``(launcher_path, agent_pid_file)``."""
    body = body if body is not None else agent_printing({"type": "started"})
    directory = tempfile.mkdtemp(prefix="fake_launcher_")
    pid_file = os.path.join(directory, "agent.pid")
    agent = os.path.join(directory, "agent")
    launcher = os.path.join(directory, "cli")
    for path, script in (
        (agent, _AGENT.format(pid_file=shlex.quote(pid_file), body=body)),
        (launcher, _LAUNCHER.format(agent=shlex.quote(agent))),
    ):
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o700)
    return launcher, pid_file


def pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _pid_in(pid_file: str) -> int:
    try:
        with open(pid_file) as f:
            return int(f.read().strip() or 0)
    except FileNotFoundError:
        return 0


async def read_pid(pid_file: str, timeout: float = 10) -> int:
    deadline = time.monotonic() + timeout
    while not (pid := _pid_in(pid_file)):
        if time.monotonic() > deadline:
            raise AssertionError(f"{pid_file} was never written")
        await asyncio.sleep(0.02)
    return pid


async def cancel_through_actor(
    backend: object, pid_file: str
) -> tuple[TurnOutcome, object, int, float]:
    """Run one turn of ``backend`` (a per-turn CLI backend whose CLI is a
    ``launcher_with_agent``) through a SessionActor and cancel it after its
    first event, as the turn loop does when the host cancels. Returns the
    turn's outcome, that first event, the agent's pid and how long the
    cancel took; the agent is SIGKILLed afterwards if it survived."""
    directory = tempfile.mkdtemp()
    actor = SessionActor(
        backend,
        SessionOpenRequest(
            session_id="s1",
            resume=False,
            l1_text="L1",
            l1_path=os.path.join(directory, "l1.md"),
            tools=[],
            hooks=None,
            cwd=directory,
            model="",
        ),
        drain_timeout_s=10,
    )
    await actor.start()
    agent = 0
    try:
        stream = actor.run_turn(TurnRequest(text="hi"))
        first = await asyncio.wait_for(anext(stream), 10)
        agent = await read_pid(pid_file)
        started = time.monotonic()
        await stream.aclose()
        return actor.last_outcome, first, agent, time.monotonic() - started
    finally:
        await actor.close()
        if agent and pid_alive(agent):
            os.kill(agent, signal.SIGKILL)


class EndTheProcessTreeTest(TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._pids: list[int] = []

    async def asyncTearDown(self) -> None:
        for pid in self._pids:
            if pid and pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
        await super().asyncTearDown()

    async def _start(
        self, body: str | None = None
    ) -> tuple[cli_runner.CliProcess, int]:
        launcher, pid_file = launcher_with_agent(body)
        proc = cli_runner.CliProcess(
            [launcher], cwd=tempfile.mkdtemp(), stdin_devnull=True
        )
        await proc.start()
        agent = await read_pid(pid_file)
        self._pids.append(agent)
        self.assertTrue(pid_alive(agent))
        return proc, agent

    async def test_kill_ends_the_agent_the_launcher_started(self) -> None:
        proc, agent = await self._start()
        lines = proc.json_lines()
        self.assertEqual(await anext(lines), {"type": "started"})

        await asyncio.wait_for(proc.kill(), 3)

        self.assertFalse(pid_alive(agent))
        self.assertEqual([obj async for obj in lines], [])  # stdout ended
        self.assertEqual(await asyncio.wait_for(proc.wait(), 3), -signal.SIGTERM)

    async def test_sigterm_comes_first_and_the_tree_may_exit_cleanly(self) -> None:
        directory = tempfile.mkdtemp()
        marker = os.path.join(directory, "terminated")
        ready = os.path.join(directory, "ready")
        body = (
            f"trap 'echo term > {shlex.quote(marker)}; exit 0' TERM\n"
            f"{background_child('sleep 60', ready)}\nwait $!"
        )
        proc, agent = await self._start(body)
        child = await read_pid(ready)
        self._pids.append(child)

        await asyncio.wait_for(proc.kill(), 3)

        self.assertFalse(pid_alive(agent))
        self.assertFalse(pid_alive(child))
        with open(marker) as f:
            self.assertEqual(f.read().strip(), "term")

    async def test_what_ignores_sigterm_is_killed_after_the_grace(self) -> None:
        ready = os.path.join(tempfile.mkdtemp(), "ready")
        proc, agent = await self._start(
            f"trap '' TERM\n{background_child('sleep 60', ready)}\nwait"
        )
        self._pids.append(await read_pid(ready))
        started = time.monotonic()
        with mock.patch.object(cli_runner, "TERM_GRACE_S", 0.5):
            await asyncio.wait_for(proc.kill(), 3)
        self.assertGreaterEqual(time.monotonic() - started, 0.5)
        self.assertFalse(pid_alive(agent))

    async def test_a_normal_exit_ends_what_the_cli_left_running(self) -> None:
        directory = tempfile.mkdtemp()
        left = os.path.join(directory, "left.pid")
        cli = os.path.join(directory, "cli")
        with open(cli, "w") as f:
            f.write(
                "#!/bin/sh\n"
                "sleep 60 </dev/null >/dev/null 2>&1 &\n"
                f"echo $! > {shlex.quote(left)}\n"
                'echo \'{"type": "result"}\'\n'
            )
        os.chmod(cli, 0o700)
        proc = cli_runner.CliProcess([cli], cwd=directory, stdin_devnull=True)
        await proc.start()
        self.assertEqual([obj async for obj in proc.json_lines()], [{"type": "result"}])
        leftover = await read_pid(left)
        self._pids.append(leftover)

        self.assertEqual(await asyncio.wait_for(proc.wait(), 3), 0)

        self.assertFalse(pid_alive(leftover))

    async def test_once_the_tree_ended_its_group_is_never_signalled_again(
        self,
    ) -> None:
        # The group id is then free to name another (e.g. the next turn's).
        proc = cli_runner.CliProcess(
            [scripted_cli('{"type": "result"}\n', "", 0)],
            cwd=tempfile.mkdtemp(),
            stdin_devnull=True,
        )
        await proc.start()
        [_ async for _ in proc.json_lines()]
        await proc.wait()
        with mock.patch.object(cli_runner.os, "killpg") as killpg:
            await proc.kill()
            await proc.kill()
        killpg.assert_not_called()

    async def test_an_interrupt_while_spawning_ends_the_process_once_spawned(
        self,
    ) -> None:
        launcher, pid_file = launcher_with_agent()
        proc = cli_runner.CliProcess(
            [launcher], cwd=tempfile.mkdtemp(), stdin_devnull=True
        )
        spawning = asyncio.ensure_future(proc.start())
        await proc.kill()  # before the process exists
        await asyncio.wait_for(spawning, 5)
        self.assertIsNotNone(proc._proc.returncode)
        agent = _pid_in(pid_file)  # 0: the agent never started
        self._pids.append(agent)
        self.assertFalse(agent and pid_alive(agent))

    async def test_a_process_that_left_the_group_does_not_hang_kill(self) -> None:
        # setsid: a new session, outside the group, still holding stdout.
        directory = tempfile.mkdtemp()
        escaped = os.path.join(directory, "escaped.pid")
        body = (
            f"setsid sh -c 'echo $$ > {shlex.quote(escaped)}; exec sleep 60' &\n"
            "exec sleep 60"
        )
        proc, agent = await self._start(body)
        self._pids.append(await read_pid(escaped))
        with mock.patch.object(cli_runner, "REAP_TIMEOUT_S", 0.5):
            await asyncio.wait_for(proc.kill(), 5)
        self.assertFalse(pid_alive(agent))
        self.assertIsNotNone(proc._proc.returncode)

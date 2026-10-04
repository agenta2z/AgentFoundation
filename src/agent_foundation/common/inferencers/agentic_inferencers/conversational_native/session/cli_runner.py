"""Shared machinery for per-turn CLI backends (claude -p, dm -p, codex exec).

Each turn spawns one process whose stdout is newline-delimited JSON. The runner
spawns, yields parsed JSON objects, and ends the turn's whole process tree on
interrupt. Argv construction and the JSON->VendorEvent mapping live in each
backend.

The installed CLIs are launchers that run the agent as a child process which
inherits the turn's stdout (claude 2.1.288: /usr/local/bin/claude_code/claude
starts ~/.cache/claude_code_native_versions/<v>/claude; the codex launcher
starts ~/.codex/packages/.../bin/codex). So each turn runs in its own session
and process group, and stopping it signals that group: killing the launcher
alone leaves the agent running the cancelled turn, and holding the pipe that
``asyncio.subprocess.Process.wait()`` waits for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from typing import Any, AsyncIterator, Optional

from agent_foundation.common.inferencers.terminal_inferencers import process_groups

logger: logging.Logger = logging.getLogger(__name__)

# One stdout line is one vendor event, and events embed whole tool results and
# messages (asyncio's default 64 KiB line limit is exceeded by a routine dm
# session_start or a large Claude tool result).
LINE_LIMIT = 64 * 1024 * 1024
# How long a turn's processes get to exit on SIGTERM before SIGKILL.
TERM_GRACE_S = 5.0
# How long the group may take to empty, and the CLI to be reaped, after SIGKILL.
REAP_TIMEOUT_S = 5.0
_POLL_S = 0.05


class CliProcess:
    """One per-turn subprocess emitting newline-delimited JSON on stdout."""

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: Optional[dict[str, str]] = None,
        stdin_devnull: bool = False,
    ) -> None:
        self._argv = argv
        self._cwd = cwd
        self._env = env
        self._stdin_devnull = (
            stdin_devnull  # dm shuts down on stdin close -> must be closed
        )
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr: list[bytes] = []
        self._kill_requested = False
        # Set once the group is empty: its pgid may then name another group.
        self._group_ended = False

    async def start(self) -> None:
        run_env = dict(os.environ)
        if self._env:
            run_env.update(self._env)
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            cwd=self._cwd,
            env=run_env,
            stdin=asyncio.subprocess.DEVNULL if self._stdin_devnull else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=LINE_LIMIT,
            start_new_session=True,  # pgid == pid: the group ``kill()`` ends
        )
        process_groups.register(self._proc.pid)
        if self._kill_requested:  # interrupted while spawning
            await self.kill()

    async def json_lines(self) -> AsyncIterator[dict[str, Any]]:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while True:
            try:
                raw = await stdout.readline()
            except ValueError:
                # The reader dropped a line above LINE_LIMIT; keep reading.
                logger.warning("CLI output line above %d bytes skipped", LINE_LIMIT)
                continue
            if not raw:
                return
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.debug("non-JSON CLI line ignored: %.120s", line)

    async def wait(self) -> int:
        """The CLI's exit code once it exited; what it left running in its
        process group (background shells, MCP servers) is ended too."""
        assert self._proc is not None
        if self._proc.stderr is not None:
            self._stderr.append(await self._proc.stderr.read())
        rc = await self._proc.wait()
        await self._end_group()
        return rc

    @property
    def stderr_text(self) -> str:
        return b"".join(self._stderr).decode(errors="replace")

    async def kill(self) -> None:
        """End the turn's whole process tree and reap the CLI: SIGTERM to its
        process group, SIGKILL to what is left of it after ``TERM_GRACE_S``."""
        self._kill_requested = True
        proc = self._proc
        if proc is None:
            return
        await self._end_group()
        try:
            await asyncio.wait_for(proc.wait(), REAP_TIMEOUT_S)
        except asyncio.TimeoutError:
            # A process that left the group (setsid) still holds a pipe.
            logger.warning(
                "CLI pid %d pipes still open after its group ended", proc.pid
            )
            _close_pipes(proc)
            await proc.wait()

    async def _end_group(self) -> None:
        assert self._proc is not None
        if self._group_ended:
            return
        pgid = self._proc.pid
        if not await _signal_and_wait(pgid, signal.SIGTERM, TERM_GRACE_S):
            logger.warning(
                "CLI process group %d still running %gs after SIGTERM; SIGKILL",
                pgid,
                TERM_GRACE_S,
            )
            if not await _signal_and_wait(pgid, signal.SIGKILL, REAP_TIMEOUT_S):
                logger.warning(
                    "CLI process group %d not empty %gs after SIGKILL",
                    pgid,
                    REAP_TIMEOUT_S,
                )
        self._group_ended = True
        process_groups.unregister(pgid)


async def _signal_and_wait(pgid: int, sig: int, timeout: float) -> bool:
    """Signal a process group; True once no process is left in it (within
    ``timeout``)."""
    if not _signal_group(pgid, sig):
        return True
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _signal_group(pgid, 0):
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_POLL_S)
    return True


def _signal_group(pgid: int, sig: int) -> bool:
    """Signal a process group; False once no process is left in it."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:  # a member we may not signal (setuid)
        pass
    return True


def _close_pipes(proc: asyncio.subprocess.Process) -> None:
    for stream in (proc.stdout, proc.stderr):
        transport = getattr(stream, "_transport", None)
        if transport is not None and not transport.is_closing():
            transport.close()

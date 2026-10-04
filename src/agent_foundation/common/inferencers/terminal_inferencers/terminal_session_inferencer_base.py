"""Terminal Session Inferencer Base.

Extends StreamingInferencerBase with subprocess-based command execution.
Subclasses implement ``construct_command()``, ``parse_output()``, and
``_build_session_args()`` for their specific CLI tools.
"""

import asyncio
import enum
import logging
import os
import subprocess
import sys
from abc import abstractmethod
from contextlib import aclosing
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from agent_foundation.common.inferencers.run_context import publish_result, read_result
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.terminal_inferencers import process_groups
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    TerminalInferencerBase,
    TerminalStreamResult,
)
from attr import attrib, attrs

_MAX_STDOUT_LINE_BYTES: int = 16 * 1024 * 1024  # 16 MB
_STDOUT_DRAIN_CHUNK_BYTES: int = 64 * 1024  # per-read() size during the post-exit drain
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_response import (
    TerminalInferencerResponse,
)


class LargeInputMode(enum.Enum):
    """How to pass the prompt to the CLI subprocess.

    INLINE: prompt embedded in the command line (risks E2BIG for large prompts).
    STDIN:  prompt piped via stdin (safe for any size).
    FILE:   prompt offloaded to a temp file when it exceeds a threshold.
    """

    INLINE = "inline"
    STDIN = "stdin"
    FILE = "file"


logger: logging.Logger = logging.getLogger(__name__)


def _convert_large_input_mode(value: Any) -> LargeInputMode:
    """Converter for ``large_input_mode`` attrib — accepts str or enum."""
    if isinstance(value, LargeInputMode):
        return value
    if isinstance(value, str):
        return LargeInputMode(value.lower())
    raise TypeError(
        f"large_input_mode must be LargeInputMode or str, got {type(value).__name__}"
    )


@attrs
class TerminalSessionInferencerBase(TerminalInferencerBase, StreamingInferencerBase):
    """Async-streaming terminal inferencer with session management.

    Multiple inheritance: TerminalInferencerBase provides subprocess
    execution (effective_cwd, target_path, pre/post_exec_scripts, env_vars,
    timeout, _execute_command, _resolve_subprocess_cwd). StreamingInferencerBase
    provides streaming/cache scaffolding and recovery.

    MRO: TSIB → TIB → SIB → IB → Debuggable → Resumable → ABC.

    Subclasses implement ``construct_command()``, ``parse_output()``,
    and ``_build_session_args()``.
    """

    # Session-specific attributes (effective_cwd, pre_exec_scripts inherited from TIB)
    session_arg_name: str = attrib(default="--session-id")
    resume_arg_name: str = attrib(default="--resume")

    # Timeout (seconds) for draining remaining stdout after process exit.
    # CLI tools that spawn child processes (e.g., MCP servers) may hold
    # stdout/stderr pipes open after the main process exits, causing
    # ``async for line in process.stdout`` to block forever.  This timeout
    # controls how long to wait for remaining buffered output after the
    # main process is detected as exited.
    subprocess_exit_drain_timeout: float = attrib(default=5.0)

    # Polling interval (seconds) for checking if the subprocess has exited.
    _subprocess_exit_poll_interval: float = attrib(default=0.5, repr=False)

    # === Abstract Methods ===

    @abstractmethod
    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        """Build the shell command string.

        Args:
            inference_input: The input data (prompt string or dict).
            **kwargs: Additional arguments (session_id, resume, etc.).

        Returns:
            Shell command string.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse command output into result dict.

        Args:
            stdout: Standard output from command.
            stderr: Standard error from command.
            return_code: Process return code.

        Returns:
            Parsed result dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """Build CLI session arguments.

        Args:
            session_id: The session ID.
            is_resume: Whether this is a resume operation.

        Returns:
            CLI argument string.
        """
        raise NotImplementedError

    # _resolve_subprocess_cwd is inherited from TerminalInferencerBase

    # === Helpers: subprocess pipe-hang prevention ===

    async def _poll_process_exit(self, pid: int) -> Optional[int]:
        """Poll for subprocess exit without relying on pipe closure.

        ``asyncio.subprocess.Process.wait()`` waits for pipe transports to
        close, which never happens when child processes (e.g., MCP servers)
        inherit the pipes.

        On POSIX we use ``os.waitid(WNOHANG | WNOWAIT)`` to detect the actual
        exit without reaping the child: asyncio's child watcher reaps it and
        records the real return code. A child reaped here would read back as 255
        (E13). Where ``waitid`` is missing we fall back to ``os.waitpid(WNOHANG)``.
        On Windows ``os.waitpid`` blocks (no ``WNOHANG``) — we fall back to
        ``OpenProcess`` + ``GetExitCodeProcess`` via ctypes, which gives us
        the same non-blocking semantics.

        Args:
            pid: The process ID to monitor.

        Returns:
            Exit code, or ``None`` if the process was already reaped.
        """
        if hasattr(os, "waitid") and hasattr(os, "WNOWAIT"):
            return await self._poll_exit_without_reaping(pid)

        if hasattr(os, "WNOHANG"):
            # POSIX without waitid
            while True:
                try:
                    wpid, status = os.waitpid(pid, os.WNOHANG)
                    if wpid != 0:
                        return os.waitstatus_to_exitcode(status)
                except ChildProcessError:
                    return None  # already reaped by asyncio
                await asyncio.sleep(self._subprocess_exit_poll_interval)

        # Windows path — poll via Win32 API
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        handle = kernel32.OpenProcess(
            SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return None  # process not found / already reaped

        try:
            exit_code = wintypes.DWORD()
            while True:
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return None
                if exit_code.value != STILL_ACTIVE:
                    return int(exit_code.value)
                await asyncio.sleep(self._subprocess_exit_poll_interval)
        finally:
            kernel32.CloseHandle(handle)

    async def _poll_exit_without_reaping(self, pid: int) -> Optional[int]:
        """The POSIX ``waitid`` poll of :meth:`_poll_process_exit`: ``WNOWAIT``
        leaves the exited child for asyncio's watcher to reap."""
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        while True:
            try:
                info = os.waitid(os.P_PID, pid, flags)
            except ChildProcessError:
                return None  # already reaped by asyncio
            if info is not None:
                if info.si_code == os.CLD_EXITED:
                    return info.si_status
                return -info.si_status  # killed by a signal
            await asyncio.sleep(self._subprocess_exit_poll_interval)

    @staticmethod
    def _force_close_pipes(
        process: asyncio.subprocess.Process,
    ) -> None:
        """Force-close subprocess pipe transports.

        After the main process exits, child processes may still hold the
        pipes open.  Closing the transports unblocks
        ``asyncio.subprocess.Process.wait()`` and prevents indefinite hangs
        during ``asyncio.run()`` shutdown.
        """
        for stream in (process.stdout, process.stderr, process.stdin):
            if stream is not None:
                transport = getattr(stream, "_transport", None)
                if transport is not None and not transport.is_closing():
                    transport.close()

    async def _create_subprocess_shell(
        self, command: str, **kwargs: Any
    ) -> asyncio.subprocess.Process:
        """``asyncio.create_subprocess_shell(command, **kwargs)`` as the
        leader of its own registered process group (see ``_popen``); end it
        with ``_end_process_group``."""
        process = await asyncio.create_subprocess_shell(
            command, start_new_session=True, **kwargs
        )
        process_groups.register(process.pid)
        return process

    async def _safe_process_cleanup(
        self, process: asyncio.subprocess.Process, timeout: float = 5.0
    ) -> None:
        """Clean up an exited subprocess: force-close pipes, wait, and end its
        process group (what the CLI left running).

        Args:
            process: The subprocess to clean up.
            timeout: Max seconds to wait for ``process.wait()``.
        """
        self._force_close_pipes(process)
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] process.wait() timed out after %.1fs — killing process tree",
                    self.__class__.__name__,
                    timeout,
                )
                self._kill_process_group(process.pid)
                try:
                    await asyncio.wait_for(process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._end_process_group(process.pid)

    async def _abort_process(
        self, process: asyncio.subprocess.Process, timeout: float = 3.0
    ) -> None:
        """End a subprocess left before its end (cancellation, an idle
        timeout, the consumer closing the stream): its whole tree at once —
        the CLI would otherwise keep working — then reap it.

        Args:
            process: The subprocess to end.
            timeout: Max seconds to wait for ``process.wait()``.
        """
        self._kill_process_tree(process)
        self._force_close_pipes(process)
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] killed process %s not reaped after %.1fs",
                self.__class__.__name__,
                process.pid,
                timeout,
            )
        finally:
            self._end_process_group(process.pid)

    async def _read_stdout_with_exit_detection(
        self,
        process: asyncio.subprocess.Process,
    ) -> AsyncIterator[str]:
        """Read subprocess stdout lines, racing against process exit.

        Prevents the common hang where CLI tools (e.g., ``acli rovodev``)
        spawn child processes (MCP servers) that inherit stdout/stderr
        pipes.  When the main process exits, the children keep the pipes
        open, causing ``async for line in process.stdout`` to block forever.

        This method polls for *actual* process exit independently of pipe
        state (``_poll_process_exit``), and breaks out of the read loop when
        the process has exited.

        Args:
            process: The subprocess to read from.

        Yields:
            Decoded stdout lines.
        """
        exit_task = asyncio.create_task(self._poll_process_exit(process.pid))
        read_task: Optional[asyncio.Task] = None
        _fell_back_to_chunked = False

        try:
            while True:
                if _fell_back_to_chunked:
                    read_task = asyncio.create_task(
                        process.stdout.read(_MAX_STDOUT_LINE_BYTES)
                    )
                else:
                    read_task = asyncio.create_task(process.stdout.readuntil(b"\n"))

                done, _ = await asyncio.wait(
                    {read_task, exit_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if read_task in done:
                    try:
                        line = read_task.result()
                    except asyncio.IncompleteReadError as e:
                        logger.info(
                            "[%s] _exit_detect: IncompleteReadError (partial=%d bytes)",
                            self.__class__.__name__,
                            len(e.partial) if e.partial else 0,
                        )
                        if e.partial:
                            yield e.partial.decode("utf-8", errors="replace")
                        break
                    except asyncio.LimitOverrunError:
                        logger.warning(
                            "[%s] subprocess stdout line exceeded %d-byte "
                            "buffer; falling back to chunked read",
                            self.__class__.__name__,
                            _MAX_STDOUT_LINE_BYTES,
                        )
                        _fell_back_to_chunked = True
                        continue
                    if not line:  # EOF
                        logger.info(
                            "[%s] _exit_detect: EOF on stdout", self.__class__.__name__
                        )
                        break
                    yield line.decode("utf-8", errors="replace")

                    # If exit also fired simultaneously, drain and break
                    if exit_task in done:
                        logger.info(
                            "[%s] _exit_detect: read+exit simultaneous — breaking",
                            self.__class__.__name__,
                        )
                        break
                elif exit_task in done:
                    logger.info(
                        "[%s] _exit_detect: exit fired while read pending — draining (timeout=%.1fs)",
                        self.__class__.__name__,
                        self.subprocess_exit_drain_timeout,
                    )
                    # Process exited but readline is stuck on pipe.
                    # Cancel the stuck read and drain buffered output.
                    read_task.cancel()
                    try:
                        await read_task
                    except asyncio.CancelledError:
                        pass

                    # Drain remaining output in bounded CHUNKS until real EOF or an
                    # idle-quiet window. The parent exited, but a child that inherited
                    # the pipe (e.g. dm-core, or an MCP server) may still hold it open
                    # and keep producing. A single ``read()``-to-EOF here is
                    # all-or-nothing: it blocks on the child-held pipe and, on timeout,
                    # drops the ENTIRE backlog (verified repro: 78/80 post-exit lines
                    # lost). Looping ``read(N)`` captures each chunk as it arrives and
                    # stops only on EOF or ``subprocess_exit_drain_timeout`` of NO new
                    # data (an idle, pipe-holding MCP child), so real trailing output
                    # survives without reintroducing the indefinite hang this method
                    # prevents.
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                process.stdout.read(_STDOUT_DRAIN_CHUNK_BYTES),
                                timeout=self.subprocess_exit_drain_timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.debug(
                                "[%s] stdout drain: no new data for %.1fs after exit "
                                "— stopping",
                                self.__class__.__name__,
                                self.subprocess_exit_drain_timeout,
                            )
                            break
                        except Exception:
                            break
                        if not chunk:  # real EOF — all writers closed the pipe
                            break
                        yield chunk.decode("utf-8", errors="replace")
                    break
        finally:
            # A read still pending when the reader is closed early (the caller
            # cancelled) would fail once the pipes close, unretrieved.
            for task in (read_task, exit_task):
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    # === Concrete: _build_full_command ===

    def _build_full_command(self, command: str) -> str:
        """Prepend ``pre_exec_scripts`` to the main command.

        Args:
            command: The main command string.

        Returns:
            Full command string with pre-exec scripts chained via ``&&``.
        """
        parts: list[str] = []
        if self.pre_exec_scripts:
            parts.extend(self.pre_exec_scripts)
        parts.append(command)
        return " && ".join(parts)

    # === Concrete: _ainfer and _infer (non-streaming execution) ===

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Execute command via streaming pipeline and return parsed output dict.

        Delegates to ``super()._ainfer()`` which accumulates from
        ``ainfer_streaming()`` → ``_ainfer_streaming()``. This ensures:
        - Cache file writing (via ``StreamingInferencerBase.ainfer_streaming()``)
        - Per-chunk idle timeout (via ``idle_timeout_seconds``)
        - the transport's ``TerminalStreamResult`` (stdout, stderr, return code)

        Args:
            inference_input: Input data for inference.
            inference_config: Optional configuration (unused).
            **kwargs: Additional arguments passed to ``construct_command()``.

        Returns:
            Parsed result dictionary from ``parse_output()``.
        """
        accumulated = await super()._ainfer(inference_input, inference_config, **kwargs)

        streamed = read_result(self, self._TERMINAL_RESULT) or TerminalStreamResult()
        return self._wrap_parse_output(
            self._parse_streamed_output(str(accumulated), streamed)
        )

    def _parse_streamed_output(self, text: str, streamed: TerminalStreamResult) -> Any:
        """Parse a streamed call: the transport's stdout when it captured one,
        else the streamed text. Subclasses whose stream carries the call's
        outcome separately override this."""
        return self.parse_output(
            streamed.stdout or text, streamed.stderr, streamed.return_code
        )

    # _infer is inherited from TerminalInferencerBase (richer pipeline with
    # timeout, env_vars, pre/post-exec scripts, output file saving).
    # Return-type wrapping handled by _wrap_parse_output below.

    def _wrap_parse_output(self, parsed):
        """Wrap parsed output in TerminalInferencerResponse.

        Handles both dict (standard parse_output return) and
        TerminalInferencerResponse (if parse_output already wrapped).
        """
        if isinstance(parsed, TerminalInferencerResponse):
            return parsed
        return TerminalInferencerResponse.from_dict(parsed)

    def _run_pre_exec_scripts_in_subprocess_shell(self) -> bool:
        """Session subclasses chain pre-scripts via '&&' in the main shell
        (env-vars propagate to the main command). This disables TIB's
        separate _execute_scripts call to avoid double-execution.

        The streaming paths (_ainfer_streaming, _infer_streaming) call
        _build_full_command() which handles the chaining.
        """
        return True

    # === Concrete: _infer_streaming (sync subprocess line streaming) ===

    def _infer_streaming(
        self,
        inference_input: Any,
        stream_callback: Any = None,
        output_stream: Any = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Sync subprocess streaming — yields stdout lines.

        Args:
            inference_input: Input data for inference.
            stream_callback: Optional callback for each line (unused by base).
            output_stream: Optional output stream (unused by base).
            **kwargs: Additional arguments passed to ``construct_command()``.

        Yields:
            Lines from subprocess stdout.
        """
        command = self.construct_command(inference_input, **kwargs)
        full_command = self._build_full_command(command)

        process = self._popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._resolve_subprocess_cwd(),
        )

        collected: list[str] = []
        stdout_ended = False
        try:
            for line in process.stdout:
                collected.append(line)
                yield line
            stdout_ended = True
        finally:
            try:
                if not stdout_ended:
                    # Left before the CLI's end (the consumer closing the
                    # stream, an error): the CLI would keep working.
                    self._kill_process_tree(process)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "[%s] sync process.wait() timed out — killing process tree",
                        self.__class__.__name__,
                    )
                    self._kill_process_group(process.pid)
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                self._end_process_group(process.pid)
            publish_result(
                self,
                self._TERMINAL_RESULT,
                TerminalStreamResult(
                    stdout="".join(collected), return_code=process.returncode
                ),
            )

    # === Concrete: _ainfer_streaming (async subprocess line streaming) ===

    async def _ainfer_streaming(
        self, inference_input: Any, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Async subprocess streaming — yields stdout lines.

        Uses ``_read_stdout_with_exit_detection()`` to prevent hangs when
        child processes inherit stdout/stderr pipes.

        Args:
            inference_input: Input data for inference.
            **kwargs: Additional arguments passed to ``construct_command()``.

        Yields:
            Lines from subprocess stdout.
        """
        command = self.construct_command(inference_input, **kwargs)
        full_command = self._build_full_command(command)

        process = await self._create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._resolve_subprocess_cwd(),
        )

        collected: list[str] = []
        stdout_ended = False
        try:
            async with aclosing(
                self._read_stdout_with_exit_detection(process)
            ) as lines:
                async for line in lines:
                    collected.append(line)
                    yield line
            stdout_ended = True
        finally:
            try:
                if stdout_ended:
                    await self._safe_process_cleanup(process)
                else:
                    await self._abort_process(process)
            finally:
                # Also when the cleanup above is itself cancelled.
                self._end_process_group(process.pid)
                publish_result(
                    self,
                    self._TERMINAL_RESULT,
                    TerminalStreamResult(
                        stdout="".join(collected), return_code=process.returncode
                    ),
                )


# === Convenience MI class: terminal + streaming + templates ===

from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)


@attrs
class TerminalSessionTemplatedInferencerBase(
    TerminalSessionInferencerBase,
    TemplatedInferencerBase,
):
    """Terminal + streaming + templates.

    MRO: TSTIB → TSIB → TIB → SIB → TemplatedIB → IB → Debuggable → Resumable → ABC.

    Use this for CLI inferencers that need all three axes (ClaudeCode, Kiro,
    Devmate, RovoDev). Use TerminalSessionInferencerBase directly if you
    don't want templates (Metamate).
    """

    pass

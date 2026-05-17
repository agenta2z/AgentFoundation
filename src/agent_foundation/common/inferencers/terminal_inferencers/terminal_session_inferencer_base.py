"""Terminal Session Inferencer Base.

Extends StreamingInferencerBase with subprocess-based command execution.
Subclasses implement ``construct_command()``, ``parse_output()``, and
``_build_session_args()`` for their specific CLI tools.
"""

import asyncio
import enum
import logging
import os
import signal
import subprocess
import sys
from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    TerminalInferencerBase,
)

_MAX_STDOUT_LINE_BYTES: int = 16 * 1024 * 1024  # 16 MB
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

    # Internal state (_last_streaming_output and _last_streaming_return_code
    # are inherited from TerminalInferencerBase; only stderr is TSIB-specific)
    _last_streaming_stderr: str = attrib(default="", init=False, repr=False)

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

        On POSIX we use ``os.waitpid(WNOHANG)`` to detect the actual exit.
        On Windows ``os.waitpid`` blocks (no ``WNOHANG``) — we fall back to
        ``OpenProcess`` + ``GetExitCodeProcess`` via ctypes, which gives us
        the same non-blocking semantics.

        Args:
            pid: The process ID to monitor.

        Returns:
            Exit code, or ``None`` if the process was already reaped.
        """
        if hasattr(os, "WNOHANG"):
            # POSIX path
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

    @staticmethod
    def _kill_process_group(pgid: int) -> None:
        """Send SIGKILL to all processes in the given process group.

        With ``start_new_session=True``, the subprocess's PID equals its
        PGID, so pass ``process.pid`` directly.  Safe to call after the
        main process has exited — children retain the PGID even after
        reparenting to init.

        PGID collision with an unrelated process is impossible while any
        group member exists: POSIX ``setsid()`` fails with ``EPERM``
        when a process group with the target PGID is still occupied.

        No-op on Windows (``os.killpg`` is POSIX-only).
        """
        if not hasattr(os, "killpg"):
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    async def _safe_process_cleanup(
        self, process: asyncio.subprocess.Process, timeout: float = 5.0
    ) -> None:
        """Clean up subprocess: force-close pipes, wait, and kill process group.

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
            self._kill_process_group(process.pid)

    async def _read_stdout_with_exit_detection(
        self,
        process: asyncio.subprocess.Process,
    ) -> AsyncIterator[str]:
        """Read subprocess stdout lines, racing against process exit.

        Prevents the common hang where CLI tools (e.g., ``acli rovodev``)
        spawn child processes (MCP servers) that inherit stdout/stderr
        pipes.  When the main process exits, the children keep the pipes
        open, causing ``async for line in process.stdout`` to block forever.

        This method uses ``os.waitpid(WNOHANG)`` to poll for *actual*
        process exit independently of pipe state, and breaks out of the
        read loop when the process has exited.

        Args:
            process: The subprocess to read from.

        Yields:
            Decoded stdout lines.
        """
        exit_task = asyncio.create_task(self._poll_process_exit(process.pid))
        _fell_back_to_chunked = False

        try:
            while True:
                if _fell_back_to_chunked:
                    read_task = asyncio.create_task(
                        process.stdout.read(_MAX_STDOUT_LINE_BYTES)
                    )
                else:
                    read_task = asyncio.create_task(
                        process.stdout.readuntil(b"\n")
                    )

                done, _ = await asyncio.wait(
                    {read_task, exit_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if read_task in done:
                    try:
                        line = read_task.result()
                    except asyncio.IncompleteReadError as e:
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
                        break
                    yield line.decode("utf-8", errors="replace")

                    # If exit also fired simultaneously, drain and break
                    if exit_task in done:
                        break
                elif exit_task in done:
                    # Process exited but readline is stuck on pipe.
                    # Cancel the stuck read and drain buffered output.
                    read_task.cancel()
                    try:
                        await read_task
                    except asyncio.CancelledError:
                        pass

                    # Drain any remaining buffered output with timeout
                    try:
                        remaining = await asyncio.wait_for(
                            process.stdout.read(),
                            timeout=self.subprocess_exit_drain_timeout,
                        )
                        if remaining:
                            yield remaining.decode("utf-8", errors="replace")
                    except (asyncio.TimeoutError, Exception):
                        logger.debug(
                            "[%s] stdout drain timed out after process exit",
                            self.__class__.__name__,
                        )
                    break
        finally:
            if not exit_task.done():
                exit_task.cancel()
                try:
                    await exit_task
                except asyncio.CancelledError:
                    pass

    # === Concrete: _ainfer_streaming (subprocess line streaming) ===

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield lines from subprocess stdout.

        Satisfies the ``@abstractmethod`` contract from StreamingInferencerBase.

        Uses ``_read_stdout_with_exit_detection()`` to prevent hangs when
        child processes inherit stdout/stderr pipes.

        Args:
            prompt: The prompt string.
            **kwargs: Additional arguments passed to ``construct_command()``.

        Yields:
            Lines from subprocess stdout.
        """
        command = self.construct_command({"prompt": prompt}, **kwargs)
        full_command = self._build_full_command(command)

        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._resolve_subprocess_cwd(),
            limit=_MAX_STDOUT_LINE_BYTES,
            start_new_session=True,
        )

        collected_stdout: list[str] = []
        try:
            async for line in self._read_stdout_with_exit_detection(process):
                collected_stdout.append(line)
                yield line
        finally:
            # Read stderr with timeout (may also be held by children)
            try:
                stderr_bytes = await asyncio.wait_for(
                    process.stderr.read(), timeout=self.subprocess_exit_drain_timeout
                ) if process.stderr else b""
            except (asyncio.TimeoutError, Exception):
                stderr_bytes = b""
                logger.debug(
                    "[%s] stderr read timed out", self.__class__.__name__
                )
            self._last_streaming_stderr = stderr_bytes.decode("utf-8", errors="replace")
            await self._safe_process_cleanup(process)
            self._last_streaming_output = "".join(collected_stdout)
            self._last_streaming_return_code = process.returncode

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
        - stderr capture (via ``_last_streaming_stderr``)

        Args:
            inference_input: Input data for inference.
            inference_config: Optional configuration (unused).
            **kwargs: Additional arguments passed to ``construct_command()``.

        Returns:
            Parsed result dictionary from ``parse_output()``.
        """
        accumulated = await super()._ainfer(
            inference_input, inference_config, **kwargs
        )

        stdout = self._last_streaming_output or str(accumulated)
        stderr = self._last_streaming_stderr
        return_code = self._last_streaming_return_code

        result_dict = self.parse_output(stdout, stderr, return_code)
        return self._wrap_parse_output(result_dict)

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

        process = subprocess.Popen(
            full_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._resolve_subprocess_cwd(),
            start_new_session=True,
        )

        collected: list[str] = []
        try:
            for line in process.stdout:
                collected.append(line)
                yield line
        finally:
            try:
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
                self._kill_process_group(process.pid)
            self._last_streaming_output = "".join(collected)
            self._last_streaming_return_code = process.returncode

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

        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._resolve_subprocess_cwd(),
            start_new_session=True,
        )

        collected: list[str] = []
        try:
            async for line in self._read_stdout_with_exit_detection(process):
                collected.append(line)
                yield line
        finally:
            await self._safe_process_cleanup(process)
            self._last_streaming_output = "".join(collected)
            self._last_streaming_return_code = process.returncode


# === Convenience MI class: terminal + streaming + templates ===

from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)


@attrs
class TerminalSessionTemplatedInferencerBase(
    TerminalSessionInferencerBase, TemplatedInferencerBase,
):
    """Terminal + streaming + templates.

    MRO: TSTIB → TSIB → TIB → SIB → TemplatedIB → IB → Debuggable → Resumable → ABC.

    Use this for CLI inferencers that need all three axes (ClaudeCode, Kiro,
    Devmate, RovoDev). Use TerminalSessionInferencerBase directly if you
    don't want templates (Metamate).
    """

    pass

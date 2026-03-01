"""Terminal Session Inferencer Base.

Extends StreamingInferencerBase with subprocess-based command execution.
Subclasses implement ``construct_command()``, ``parse_output()``, and
``_build_session_args()`` for their specific CLI tools.
"""

import asyncio
import subprocess
from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_response import (
    TerminalInferencerResponse,
)


@attrs
class TerminalSessionInferencerBase(StreamingInferencerBase):
    """Base for CLI/terminal-based streaming inferencers.

    Executes commands via subprocess and streams stdout line-by-line.
    Subclasses implement ``construct_command()``, ``parse_output()``,
    and ``_build_session_args()``.

    Attributes:
        working_dir: Working directory for subprocess execution.
        pre_exec_scripts: Shell commands to run before the main command.
        session_arg_name: CLI argument name for session ID.
        resume_arg_name: CLI argument name for resume flag.
    """

    # Terminal-specific attributes
    working_dir: Optional[str] = attrib(default=None)
    pre_exec_scripts: Optional[List[str]] = attrib(default=None)
    session_arg_name: str = attrib(default="--session-id")
    resume_arg_name: str = attrib(default="--resume")

    # Internal state for streaming result
    _last_streaming_output: str = attrib(default="", init=False, repr=False)
    _last_streaming_stderr: str = attrib(default="", init=False, repr=False)
    _last_streaming_return_code: int = attrib(default=0, init=False, repr=False)

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

    # === Concrete: _produce_chunks (subprocess line streaming) ===

    async def _produce_chunks(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield lines from subprocess stdout.

        Satisfies the ``@abstractmethod`` contract from StreamingInferencerBase.

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
            cwd=self.working_dir,
        )

        collected_stdout: list[str] = []
        try:
            async for line_bytes in process.stdout:
                line = line_bytes.decode("utf-8", errors="replace")
                collected_stdout.append(line)
                yield line
        finally:
            stderr_bytes = await process.stderr.read() if process.stderr else b""
            self._last_streaming_stderr = stderr_bytes.decode("utf-8", errors="replace")
            await process.wait()
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
        ``ainfer_streaming()`` → ``_produce_chunks()``. This ensures:
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
        return TerminalInferencerResponse.from_dict(result_dict)

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync execution via subprocess.run().

        Args:
            inference_input: Input data for inference.
            inference_config: Optional configuration (unused).
            **kwargs: Additional arguments passed to ``construct_command()``.

        Returns:
            TerminalInferencerResponse wrapping ``parse_output()`` result.
        """
        command = self.construct_command(inference_input, **kwargs)
        full_command = self._build_full_command(command)

        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.working_dir,
        )
        result_dict = self.parse_output(result.stdout, result.stderr, result.returncode)
        return TerminalInferencerResponse.from_dict(result_dict)

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
            cwd=self.working_dir,
        )

        collected: list[str] = []
        try:
            for line in process.stdout:
                collected.append(line)
                yield line
        finally:
            process.wait()
            self._last_streaming_output = "".join(collected)
            self._last_streaming_return_code = process.returncode

    # === Concrete: _ainfer_streaming (async subprocess line streaming) ===

    async def _ainfer_streaming(
        self, inference_input: Any, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Async subprocess streaming — yields stdout lines.

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
            cwd=self.working_dir,
        )

        collected: list[str] = []
        try:
            async for line_bytes in process.stdout:
                line = line_bytes.decode("utf-8", errors="replace")
                collected.append(line)
                yield line
        finally:
            await process.wait()
            self._last_streaming_output = "".join(collected)
            self._last_streaming_return_code = process.returncode

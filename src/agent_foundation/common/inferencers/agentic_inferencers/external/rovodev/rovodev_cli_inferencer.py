"""Rovo Dev CLI run-mode inferencer.

Wraps ``acli rovodev legacy`` for single-shot and multi-turn programmatic use.
Follows the same pattern as ``ClaudeCodeCliInferencer``.

Usage::

    # Single-turn:
    inf = RovoDevCliInferencer(working_dir="/path/to/repo")
    result = inf("What does this repo do?")

    # Multi-turn (auto-resume):
    inf = RovoDevCliInferencer(working_dir="/repo")
    r1 = inf.new_session("My number is 42")
    r2 = inf("What is my number?")  # auto-resumes last session

    # Streaming:
    for chunk in inf.infer_streaming("Explain this codebase"):
        print(chunk, end="")

    # Async:
    result = await inf.ainfer("Fix this bug")

    # Structured output:
    inf = RovoDevCliInferencer(
        working_dir="/repo",
        output_schema='{"type":"object","properties":{"answer":{"type":"string"}}}',
    )
    result = inf("What is 2+2?")
"""

import logging
import os
import contextvars
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List, Optional

from attr import attrib, attrs

from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.common import (
    clean_env_for_subprocess,
    ensure_session_metadata,
    find_latest_session_id,
    ACLI_BINARY,
    ACLI_SUBCOMMAND,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_TOOL_USE_IDLE_TIMEOUT,
    RovoDevNotFoundError,
    find_acli_binary,
    strip_ansi_codes,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalInferencerResponse,
    TerminalSessionInferencerBase,
)

logger: logging.Logger = logging.getLogger(__name__)

# Per-call output file path for async parallel safety
_current_output_file: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_output_file", default=None
)


@attrs
class RovoDevCliInferencer(TerminalSessionInferencerBase):
    """Rovo Dev CLI legacy-mode inferencer (``acli rovodev legacy``).

    Inherits from ``TerminalSessionInferencerBase`` which provides:
    - ``_ainfer_streaming()`` async subprocess line streaming
    - ``_infer_streaming()`` sync subprocess line streaming
    - ``_ainfer()`` / ``_infer()`` via subprocess with ``parse_output()``
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - ``active_session_id`` property

    This class implements ``construct_command()``, ``parse_output()``, and
    ``_build_session_args()`` (the abstract methods), and overrides ``ainfer()``
    and ``infer()`` for Rovo Dev session management.

    Session Limitation:
        Rovo Dev's ``--restore`` is a boolean flag that resumes the most recent
        session for the current ``working_dir``. Unlike Claude Code, it does NOT
        support arbitrary session ID targeting. The ``active_session_id`` is set
        to a sentinel value ``"active"`` after successful inference to enable
        auto-resume on subsequent calls.

    Attributes:
        acli_path: Path to acli binary. Auto-detected via ``shutil.which`` if None.
        config_file: Path to rovodev config file.
        cloud_id: Atlassian cloud ID.
        yolo: Skip tool confirmation prompts (default True for programmatic use).
        enable_deep_plan: Enable deep planning mode.
        xid: Experience ID for analytics tagging.
        output_schema: JSON schema for structured output.
        output_file: Path for output file capture.
        agent_mode: Agent mode override (e.g., "ask", "plan").
        jira: Jira ticket URL to start work on.
        extra_cli_args: Additional CLI arguments to pass through.
        idle_timeout_seconds: Idle timeout between output chunks (default 30 min).
        tool_use_idle_timeout_seconds: Idle timeout during tool use (default 2 hr).
    """

    # --- Configuration ---
    has_local_access: bool = attrib(default=True)
    acli_path: Optional[str] = attrib(default=None)
    config_file: Optional[str] = attrib(default=None)
    cloud_id: Optional[str] = attrib(default=None)
    yolo: bool = attrib(default=True)
    enable_deep_plan: bool = attrib(default=False)
    xid: Optional[str] = attrib(default=None)
    output_schema: Optional[str] = attrib(default=None)
    output_file: Optional[str] = attrib(default=None)
    raw_output_to_file: bool = attrib(default=True)  # Always capture clean LLM output via --output-file
    agent_mode: Optional[str] = attrib(default=None)
    jira: Optional[str] = attrib(default=None)
    extra_cli_args: Optional[List[str]] = attrib(default=None)

    # --- Timeouts ---
    idle_timeout_seconds: int = attrib(default=DEFAULT_IDLE_TIMEOUT)
    tool_use_idle_timeout_seconds: int = attrib(default=DEFAULT_TOOL_USE_IDLE_TIMEOUT)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __attrs_post_init__(self) -> None:
        """Validate acli is installed and set defaults."""
        if self.acli_path is None:
            import shutil

            self.acli_path = shutil.which(ACLI_BINARY)
        if self.working_dir is None:
            self.working_dir = os.getcwd()
        super().__attrs_post_init__()

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        """Build the ``acli rovodev legacy`` command string.

        Session management (``session_id``/``resume``) is injected into kwargs
        by the ``ainfer()``/``infer()`` overrides.

        Args:
            inference_input: The input data (prompt string or dict).
            **kwargs: Additional arguments (session_id, resume, output_file, etc.).

        Returns:
            Shell command string.

        Raises:
            RovoDevNotFoundError: If acli binary is not found.
        """
        acli = self.acli_path
        if not acli:
            raise RovoDevNotFoundError()

        # Extract prompt (handle both dict and string)
        if isinstance(inference_input, dict):
            prompt = inference_input.get("prompt", str(inference_input))
        else:
            prompt = str(inference_input)

        parts = [acli, ACLI_SUBCOMMAND, "legacy", shlex.quote(prompt)]

        if self.yolo:
            parts.append("--yolo")

        # Output file for reliable output capture.
        # When raw_output_to_file=True (default), always use a temp file so
        # parse_output() gets clean LLM text (no terminal formatting, preserves
        # XML tags like <Response>). Each call gets its own temp file to support
        # parallel inference.
        out_path = kwargs.get("output_file") or self.output_file
        if not out_path and self.raw_output_to_file:
            out_path = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
            kwargs["_auto_output_file"] = out_path  # for parse_output cleanup
        if out_path:
            parts.extend(["--output-file", out_path])

        if self.config_file:
            parts.extend(["--config-file", self.config_file])
        if self.xid:
            parts.extend(["--xid", self.xid])
        if self.jira:
            parts.extend(["--jira", self.jira])
        if self.enable_deep_plan:
            parts.append("--enable-deep-plan")
        if self.agent_mode:
            parts.extend(["--agent-mode", self.agent_mode])
        if self.output_schema:
            parts.extend(["--output-schema", shlex.quote(self.output_schema)])

        # Session resumption — delegated to _build_session_args()
        session_id = kwargs.get("session_id")
        is_resume = kwargs.get("resume", False)
        if is_resume:
            session_args = self._build_session_args(
                session_id or "", is_resume
            )
            if session_args:
                parts.append(session_args)

        if self.extra_cli_args:
            parts.extend(self.extra_cli_args)

        return " ".join(parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int,
        output_file_path: Optional[str] = None,
    ) -> dict:
        """Parse ``acli rovodev legacy`` output.

        Strategy:
        1. Read from output file if available (clean, no Rich formatting)
        2. Fall back to ANSI-stripped stdout

        Args:
            stdout: Standard output from the process.
            stderr: Standard error from the process.
            return_code: Process exit code.
            output_file_path: Per-call output file path (overrides self.output_file).

        Returns:
            Dict suitable for ``TerminalInferencerResponse.from_dict()``.
        """
        output = stdout
        effective_output_file = output_file_path or _current_output_file.get(None) or self.output_file

        # Read from output file (clean, no Rich TUI formatting)
        if effective_output_file and Path(effective_output_file).exists():
            try:
                output = Path(effective_output_file).read_text(encoding="utf-8").strip()
            except OSError:
                logger.warning("Failed to read output file: %s", effective_output_file)
                output = strip_ansi_codes(stdout).strip()
        else:
            # Fall back to stdout with ANSI stripping
            output = strip_ansi_codes(stdout).strip()

        # Check for auth errors in stderr
        if return_code != 0 and stderr:
            stderr_lower = stderr.lower()
            if "unauthorized" in stderr_lower or "auth" in stderr_lower:
                logger.error("Rovo Dev authentication error: %s", stderr[:200])

        return {
            "output": output,
            "raw_output": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "success": return_code == 0,
        }

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Override to handle temp output file lifecycle and clean env."""
        # Generate per-call temp output file if raw_output_to_file is enabled
        auto_output_file = None
        if not self.output_file and self.raw_output_to_file:
            auto_output_file = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
            kwargs["output_file"] = auto_output_file

        command = self.construct_command(inference_input, **kwargs)
        full_command = self._build_full_command(command)
        env = clean_env_for_subprocess()

        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.working_dir,
            env=env,
        )
        result_dict = self.parse_output(
            result.stdout, result.stderr, result.returncode,
            output_file_path=auto_output_file,
        )

        # Clean up temp output file
        if auto_output_file:
            try:
                Path(auto_output_file).unlink(missing_ok=True)
            except OSError:
                pass

        return TerminalInferencerResponse.from_dict(result_dict)

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """Build session restore arguments.

        Rovo Dev CLI (v0.13.68+) supports ``--restore <session_id>`` to resume
        a specific session by UUID, or ``--restore`` without a value to resume
        the most recent session for the current workspace.

        Args:
            session_id: The session UUID to restore, or empty for most recent.
            is_resume: Whether this is a session resume request.

        Returns:
            ``"--restore <session_id>"`` or ``"--restore"`` if resuming.
        """
        if not is_resume:
            return ""
        if session_id:
            return f"--restore {shlex.quote(session_id)}"
        return "--restore"

    # =========================================================================
    # Override: _yield_filter() — ANSI stripping for streaming
    # =========================================================================

    async def _yield_filter(
        self, chunks: AsyncIterator[str], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Filter streaming output — strip ANSI codes from Rich TUI stdout.

        This is an async generator that transforms an entire async iterator
        of chunks, matching the ``StreamingInferencerBase._yield_filter``
        signature.

        Note:
            stdout streaming from ``acli rovodev legacy`` may be noisy due to
            Rich TUI formatting. For clean streaming, use
            ``RovoDevServeInferencer`` (serve mode).

        Args:
            chunks: Async iterator of raw stdout chunks.
            **kwargs: Additional arguments passed through to parent.

        Yields:
            Cleaned text chunks with ANSI codes removed.
        """
        async for line in super()._yield_filter(chunks, **kwargs):
            clean = strip_ansi_codes(line)
            if clean.strip():
                yield clean

    # =========================================================================
    # Override: ainfer() / infer() — Session-aware inference
    # =========================================================================

    async def ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Async inference with session management.

        Injects ``session_id`` and ``resume`` into kwargs before calling the
        base class, and updates ``active_session_id`` from the result.
        Follows the same pattern as ``ClaudeCodeCliInferencer.ainfer()``.

        Args:
            inference_input: Input for inference (prompt string or dict).
            inference_config: Optional configuration.
            **kwargs: Additional arguments (new_session, session_id, resume).

        Returns:
            ``TerminalInferencerResponse`` with the inference result.
        """
        # Generate per-call temp output file if raw_output_to_file is enabled
        auto_output_file = None
        if not self.output_file and self.raw_output_to_file:
            auto_output_file = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
            kwargs["output_file"] = auto_output_file
            _current_output_file.set(auto_output_file)

        # Handle new_session flag
        new_session = kwargs.pop("new_session", False)
        if new_session:
            self.active_session_id = None

        # Determine session context
        session_id = kwargs.get("session_id", self.active_session_id)
        is_resume = kwargs.get("resume", True)

        if session_id is None:
            if self.auto_resume and self.active_session_id:
                session_id = self.active_session_id
            else:
                is_resume = False

        kwargs["session_id"] = session_id
        kwargs["resume"] = is_resume and session_id is not None

        # Route through _ainfer_single for retry/preprocessing/timeout
        result = await self._ainfer_single(
            inference_input, inference_config, **kwargs
        )

        # Extract the real session ID from the sessions directory.
        # After a successful run, the most recently modified session in
        # ~/.rovodev/sessions/ is the one we just created/used.
        if getattr(result, "success", False):
            session_id_found = find_latest_session_id(
                workspace_path=self.working_dir
            )
            if session_id_found:
                self.active_session_id = session_id_found
                self.log_debug(f"Captured session ID: {session_id_found}", "Async")
                ensure_session_metadata(
                    session_id_found, workspace_path=self.working_dir
                )

        # Clean up temp output file
        if auto_output_file:
            try:
                Path(auto_output_file).unlink(missing_ok=True)
            except OSError:
                pass
            _current_output_file.set(None)

        return result

    def infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync inference with session management.

        Mirrors ``ainfer()`` for synchronous usage.

        Args:
            inference_input: Input for inference (prompt string or dict).
            inference_config: Optional configuration.
            **kwargs: Additional arguments (new_session, session_id, resume).

        Returns:
            ``TerminalInferencerResponse`` with the inference result.
        """
        # Handle new_session flag
        new_session = kwargs.pop("new_session", False)
        if new_session:
            self.active_session_id = None

        # Determine session context
        session_id = kwargs.get("session_id", self.active_session_id)
        is_resume = kwargs.get("resume", True)

        if session_id is None:
            if self.auto_resume and self.active_session_id:
                session_id = self.active_session_id
            else:
                is_resume = False

        kwargs["session_id"] = session_id
        kwargs["resume"] = is_resume and session_id is not None

        # Route through _infer_single for retry/preprocessing/timeout
        result = self._infer_single(
            inference_input, inference_config, **kwargs
        )

        # Extract the real session ID from the sessions directory.
        if getattr(result, "success", False):
            session_id_found = find_latest_session_id(
                workspace_path=self.working_dir
            )
            if session_id_found:
                self.active_session_id = session_id_found
                self.log_debug(f"Captured session ID: {session_id_found}", "Sync")
                ensure_session_metadata(
                    session_id_found, workspace_path=self.working_dir
                )
        return result

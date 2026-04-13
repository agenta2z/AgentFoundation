# pyre-strict

"""Kiro CLI inferencer for executing Kiro CLI commands.

Wraps ``kiro-cli chat --no-interactive`` as a subprocess-based
TerminalSessionInferencerBase implementation.
"""

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    LargeInputMode,
    TerminalInferencerResponse,
    TerminalSessionInferencerBase,
)

logger: logging.Logger = logging.getLogger(__name__)


@attrs
class KiroCliInferencer(TerminalSessionInferencerBase):
    """Kiro CLI as a terminal-based streaming inferencer with session continuation.

    Inherits from TerminalSessionInferencerBase which provides:
    - ``_ainfer_streaming()`` async subprocess line streaming
    - ``_infer_streaming()`` sync subprocess line streaming
    - ``_ainfer()`` / ``_infer()`` via subprocess with ``parse_output()``
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - ``active_session_id`` property

    This class implements ``construct_command()``, ``parse_output()``, and
    ``_build_session_args()`` (the abstract methods), and overrides ``ainfer()``
    and ``infer()`` for session management.

    Usage Patterns:
        # Simple single-call:
        inferencer = KiroCliInferencer(target_path="/path/to/repo")
        result = inferencer("Write a hello world program")

        # Multi-turn with auto-resume:
        inferencer = KiroCliInferencer(target_path="/repo", auto_resume=True)
        r1 = inferencer.new_session("My number is 42")
        r2 = inferencer.infer("What is my number?")  # Auto-resumes!

        # With specific model:
        inferencer = KiroCliInferencer(model_name="claude-haiku-4.5")
        result = inferencer("Quick question")

    Attributes:
        target_path: Absolute path to the target repository/workspace where the
            Kiro CLI agent operates. Used as the ``working_dir`` for the CLI
            subprocess. Defaults to ``os.getcwd()`` if not specified.
        agent_name: Optional custom agent via ``--agent``.
        model_name: Model to use via ``--model`` (default: ``"auto"``).
            Available models include: auto, claude-opus-4.6, claude-sonnet-4.6,
            claude-sonnet-4, claude-haiku-4.5, deepseek-3.2, and others.
            When set to ``"auto"``, the flag is omitted (kiro-cli default).
            Accepts any format (dash-separated, dot-separated, short aliases)
            — normalized via ``resolve_model_tag()`` in ``__attrs_post_init__``.
        trust_mode: Tool permission mode — ``"all"`` for ``--trust-all-tools``,
            ``"specific"`` for ``--trust-tools <list>``.
        trusted_tools: Tools for ``--trust-tools`` when ``trust_mode="specific"``.
        require_mcp_startup: Include ``--require-mcp-startup`` flag.
        extra_cli_args: Additional CLI arguments.
        idle_timeout_seconds: Per-chunk idle timeout (default: 1800).
        large_input_mode: How to pass prompt to subprocess (default: STDIN).
    """

    # Kiro CLI-specific attributes
    target_path: Optional[str] = attrib(default=None)
    agent_name: Optional[str] = attrib(default=None)
    model_name: str = attrib(default="auto")
    trust_mode: str = attrib(default="all")
    trusted_tools: Optional[List[str]] = attrib(default=None)
    require_mcp_startup: bool = attrib(default=False)
    extra_cli_args: Optional[List[str]] = attrib(default=None)
    idle_timeout_seconds: int = attrib(default=1800)
    large_input_mode: LargeInputMode = attrib(default=LargeInputMode.STDIN)

    def __attrs_post_init__(self) -> None:
        """Initialize defaults after attrs init."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.common import (
            resolve_model_tag,
        )

        if self.target_path is None:
            self.target_path = os.getcwd()
        if self.working_dir is None:
            self.working_dir = self.target_path
        if self.model_name and self.model_name != "auto":
            self.model_name = resolve_model_tag(self.model_name)
        super().__attrs_post_init__()

    # === Abstract Method Implementations ===

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """Build CLI arguments for Kiro CLI session management.

        Kiro CLI uses ``--resume`` to resume the most recent session from
        the working directory. Unlike Claude Code CLI which uses
        ``--resume '<session_id>'``, no session ID argument is needed.

        Args:
            session_id: The session ID (accepted for interface compatibility,
                not passed to CLI).
            is_resume: Whether this is a resume operation.

        Returns:
            CLI argument string.
        """
        if is_resume:
            return "--resume"
        return ""

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        """Build the shell command string for Kiro CLI.

        Args:
            inference_input: The input data (prompt string or dict).
            **kwargs: Additional arguments (session_id, resume, use_stdin).

        Returns:
            Shell command string.
        """
        # Extract prompt (handle both dict and string)
        if isinstance(inference_input, dict):
            prompt = inference_input.get("prompt", str(inference_input))
        else:
            prompt = str(inference_input)

        session_id = kwargs.get("session_id")
        is_resume = kwargs.get("resume", False)
        use_stdin = kwargs.get("use_stdin", False)

        command_parts = ["kiro-cli", "chat", "--no-interactive"]

        if self.model_name and self.model_name != "auto":
            command_parts.append(f"--model {self.model_name}")

        if self.agent_name:
            command_parts.append(f"--agent {self.agent_name}")

        # Trust mode (mutually exclusive)
        if self.trust_mode == "all":
            command_parts.append("--trust-all-tools")
        elif self.trust_mode == "specific" and self.trusted_tools:
            tools_str = ",".join(self.trusted_tools)
            command_parts.append(f"--trust-tools {tools_str}")

        if self.require_mcp_startup:
            command_parts.append("--require-mcp-startup")

        if is_resume and session_id:
            session_args = self._build_session_args(session_id, is_resume)
            if session_args:
                command_parts.append(session_args)

        # Always include --wrap never to prevent line-wrapping artifacts
        command_parts.append("--wrap never")

        if self.extra_cli_args:
            command_parts.extend(self.extra_cli_args)

        if not use_stdin:
            # Inline prompt as positional argument (last)
            escaped_prompt = self._escape_for_shell(prompt)
            command_parts.append(f'"{escaped_prompt}"')

        return " ".join(command_parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> dict:
        """Parse command output into a response dict.

        Simple text-based parsing — no JSON extraction needed.
        Returns a plain dict; the base class wraps it in
        TerminalInferencerResponse via ``from_dict()``.

        Args:
            stdout: Standard output from command.
            stderr: Standard error from command.
            return_code: Process return code.

        Returns:
            Dict with parsed fields.
        """
        result: Dict[str, Any] = {
            "output": stdout.strip() if stdout else "",
            "raw_output": stdout if stdout else "",
            "stderr": stderr if stderr else "",
            "return_code": return_code,
            "success": return_code == 0,
        }

        if not result["success"]:
            result["error"] = (
                stderr.strip()
                if stderr and stderr.strip()
                else f"Command failed with code {return_code}"
            )

        return result

    # === Helper Methods ===

    def _escape_for_shell(self, text: str) -> str:
        """Shell-escape text for use in double-quoted strings.

        Escapes: backslash, double-quote, $, backtick (in that order).

        Args:
            text: Text to escape.

        Returns:
            Shell-escaped text.
        """
        return (
            text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("$", "\\$")
            .replace("`", "\\`")
        )

    def get_response_text(self, result: Any) -> str:
        """Extract response text from result.

        Args:
            result: TerminalInferencerResponse or dict from inference.

        Returns:
            Response text or error message.
        """
        if isinstance(result, dict):
            if result.get("success"):
                return result.get("output", "")
            return result.get("error", "Unknown error occurred")
        if hasattr(result, "success"):
            if result.success:
                return result.output or ""
            return result.error or "Unknown error occurred"
        return str(result)

    def check_auth(self, timeout: float = 10.0) -> bool:
        """Check if kiro-cli is authenticated by running 'kiro-cli whoami'.

        Returns True if authenticated, False otherwise.
        Does not attempt to authenticate — callers should direct users
        to run 'kiro-cli login' manually.

        Args:
            timeout: Maximum seconds to wait for the command.

        Returns:
            True if authenticated, False otherwise.
        """
        try:
            result = subprocess.run(
                "kiro-cli whoami",
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    # === Override: ainfer() — Session-Aware ===

    async def ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Async inference with session management.

        Routes through _ainfer_single() to preserve:
        - Retry logic (max_retry, execute_with_retry)
        - Input preprocessing (input_preprocessor)
        - Response postprocessing (response_post_processor)
        - Total timeout (total_timeout_seconds)

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Additional arguments.

        Returns:
            Inference result.
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

        # Route through _ainfer_single for retry/preprocessing/timeout
        result = await self._ainfer_single(inference_input, inference_config, **kwargs)

        # Update active session from result
        result_session_id = None
        if isinstance(result, dict):
            result_session_id = result.get("session_id")
        elif hasattr(result, "session_id"):
            result_session_id = result.session_id
        if result_session_id and result_session_id != self.active_session_id:
            self.active_session_id = result_session_id
            self.log_debug(
                f"Updated active session to: {result_session_id[:8]}...", "Async"
            )

        return result

    # === Override: infer() — Sync Session-Aware ===

    def infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync inference with session management.

        Mirrors ainfer() for the sync path. Required because the inherited
        resume_session() and new_session() call self.infer() without
        resume=True, so we must inject session context here.

        Routes through _infer_single() to preserve retry/preprocessing.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Additional arguments.

        Returns:
            Inference result.
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

        # Route through _infer_single (preserves retry/preprocessing)
        result = self._infer_single(inference_input, inference_config, **kwargs)

        # Update active session from result
        result_session_id = None
        if isinstance(result, dict):
            result_session_id = result.get("session_id")
        elif hasattr(result, "session_id"):
            result_session_id = result.session_id
        if result_session_id and result_session_id != self.active_session_id:
            self.active_session_id = result_session_id
            self.log_debug(
                f"Updated active session to: {result_session_id[:8]}...", "Sync"
            )

        return result

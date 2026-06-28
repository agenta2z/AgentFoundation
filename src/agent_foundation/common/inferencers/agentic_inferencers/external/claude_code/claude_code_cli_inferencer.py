
# pyre-strict

"""Claude Code CLI inferencer for executing Claude Code CLI commands."""

import asyncio
import json
import logging
import os
import subprocess
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, TextIO

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.common import (
    EffortLevel,
    PermissionModeLiteral,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    EmptyLineMode,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    LargeInputMode,
    TerminalInferencerResponse,
    TerminalSessionTemplatedInferencerBase,
)

logger: logging.Logger = logging.getLogger(__name__)


from agent_foundation.common.inferencers.run_context import bridge_entrypoint


@attrs
class ClaudeCodeCliInferencer(TerminalSessionTemplatedInferencerBase):
    """Claude Code CLI as a terminal-based streaming inferencer with session continuation.

    Inherits from TerminalSessionInferencerBase which provides:
    - ``_ainfer_streaming()`` async subprocess line streaming
    - ``_infer_streaming()`` sync subprocess line streaming
    - ``_ainfer()`` / ``_infer()`` via subprocess with ``parse_output()``
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - ``active_session_id`` property

    This class implements ``construct_command()``, ``parse_output()``, and
    ``_build_session_args()`` (the abstract methods), and overrides ``ainfer()``
    and ``infer()`` for Claude-specific session management.

    Usage Patterns:
        # Simple single-call:
        inferencer = ClaudeCodeCliInferencer(target_path="/path/to/repo")
        result = inferencer("Write a hello world program")

        # Multi-turn with auto-resume:
        inferencer = ClaudeCodeCliInferencer(target_path="/repo", auto_resume=True)
        r1 = inferencer.new_session("My number is 42")
        r2 = inferencer.infer("What is my number?")  # Auto-resumes!

        # Async:
        r1 = await inferencer.anew_session("My number is 42")
        r2 = await inferencer.ainfer("What is my number?")  # Auto-resumes!

        # Streaming (text mode, no session metadata):
        for chunk in inferencer.infer_streaming("Explain this"):
            print(chunk, end="", flush=True)

    Attributes:
        target_path: Absolute path to the target repository/workspace where the
            Claude Code CLI agent operates (e.g., ``"/data/users/me/fbsource"``).
            Used as the ``effective_cwd`` for the CLI subprocess, so Claude's
            file operations are rooted here.
            Defaults to ``~/fbsource`` if not specified.
        model_name: Model alias or full name (default: "sonnet").
        system_prompt: Full system prompt override.
        append_system_prompt: Appended to default system prompt (requires Claude Code v2.1.58+).
        allowed_tools: List of tools Claude can use.
        permission_mode: Permission control mode (default: ``"bypassPermissions"``).
            One of ``"default"``, ``"acceptEdits"``, ``"plan"``, ``"auto"``,
            ``"dontAsk"``, or ``"bypassPermissions"`` — passed through as
            ``--permission-mode``. ``"bypassPermissions"`` is emitted as
            ``--dangerously-skip-permissions`` for backwards compatibility.
        effort: Reasoning-effort level (default: ``"max"``). One of ``"low"``,
            ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"``, or ``None`` to
            omit the flag entirely (use the model's default). Higher levels
            allocate more thinking budget at the cost of latency and tokens.
            Emitted as ``--effort <level>`` (Claude Code v2.1.119+).
        max_budget_usd: Maximum spend per call.
        extra_cli_args: Additional CLI arguments.
        disable_osx_sandbox: Pass the Meta launcher's
            ``--dangerously-disable-osx-sandbox`` flag. ``None`` (default)
            resolves from the ``CLAUDE_CODE_INFERANCER_NO_SANDBOX`` env var;
            an explicit ``True``/``False`` overrides it. Enable only when this
            inferencer's ``claude`` subprocess must run nested inside an
            already-sandboxed process — macOS forbids nesting seatbelt
            sandboxes, so otherwise ``claude`` exits 71 with no output.
            SECURITY: removes OS-level confinement of the agent's file access.
    """

    has_local_access: bool = attrib(default=True)

    # Claude Code CLI-specific attributes
    idle_timeout_seconds: int = attrib(default=1800)
    tool_use_idle_timeout_seconds: int = attrib(default=7200)
    empty_line_mode: EmptyLineMode = attrib(default=EmptyLineMode.SUPPRESS_LEADING)
    model_name: str = attrib(default="opus[1m]")
    claude_command: str = attrib(default="claude")
    large_input_mode: LargeInputMode = attrib(default=LargeInputMode.STDIN)
    system_prompt: Optional[str] = attrib(default=None)
    append_system_prompt: Optional[str] = attrib(default=None)
    allowed_tools: Optional[List[str]] = attrib(default=None)
    enable_shell: bool = attrib(default=True)
    # When set with ``enable_shell=True``, this is an informational hint about
    # which executables the Bash tool should allow. The CLI doesn't enforce
    # the list itself (Claude Code has no equivalent flag), but its presence
    # is logged for downstream tooling (e.g. devmate config generation).
    allowed_shell_commands: Optional[List[str]] = attrib(default=None)
    permission_mode: PermissionModeLiteral = attrib(default="bypassPermissions")
    effort: Optional[EffortLevel] = attrib(default="max")
    max_budget_usd: Optional[float] = attrib(default=None)
    extra_cli_args: Optional[List[str]] = attrib(default=None)
    # Disable the Meta launcher's macOS seatbelt sandbox via
    # ``--dangerously-disable-osx-sandbox``. ``None`` (default) resolves from
    # the ``CLAUDE_CODE_INFERANCER_NO_SANDBOX`` env var in __attrs_post_init__;
    # an explicit ``True``/``False`` overrides the env. Needed when this
    # inferencer's ``claude`` subprocess runs nested inside an already-sandboxed
    # process (macOS forbids nesting seatbelt sandboxes → exit 71). See
    # ``common.resolve_disable_osx_sandbox`` for the security note.
    disable_osx_sandbox: Optional[bool] = attrib(default=None)

    # Known Node.js Claude Code CLI paths to try as fallback
    _NODE_CLAUDE_PATHS: List[str] = [
        "node /opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js",
        "npx @anthropic-ai/claude-code",
    ]

    _CLAUDE_TIER_MAP = {
        "max": "opus[1m]",
        "default": "sonnet",
        "lite": "haiku",
    }

    @classmethod
    def _resolve_model_for_tier(cls, tier: str) -> str:
        return cls._CLAUDE_TIER_MAP.get(str(tier).lower(), "sonnet")

    def __attrs_post_init__(self) -> None:
        """Initialize defaults after attrs init."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.common import (
            resolve_disable_osx_sandbox,
            resolve_model_tag,
        )

        if self.target_path is None:
            self.target_path = os.path.expanduser("~/fbsource")
        if self.model_tier is not None:
            self.model_name = self._resolve_model_for_tier(self.model_tier)
        elif not self.model_name:
            self.model_name = "opus[1m]"
        self.model_name = resolve_model_tag(self.model_name)

        # Resolve the macOS-sandbox toggle to a concrete bool: an explicit
        # ctor value wins; otherwise fall back to the env var.
        self.disable_osx_sandbox = resolve_disable_osx_sandbox(
            self.disable_osx_sandbox
        )

        self._resolve_claude_command()

        # Shell-allowlist logging (no CLI flag equivalent, but downstream tools
        # such as devmate config generation may consume the list).
        if self.allowed_shell_commands and not self.enable_shell:
            logger.warning(
                "enable_shell=False takes precedence over allowed_shell_commands=%s "
                "(shell tool will be disabled).",
                self.allowed_shell_commands,
            )
        elif self.allowed_shell_commands:
            logger.info(
                "ClaudeCodeCliInferencer: allowed_shell_commands set to %s "
                "(informational — no equivalent CLI flag).",
                self.allowed_shell_commands,
            )

        super().__attrs_post_init__()

    def _resolve_claude_command(self) -> None:
        """Verify the Claude CLI command works; fall back to Node.js if not.

        The symlinked ``claude`` binary can sometimes be killed by macOS
        (SIGKILL / return code -9). In that case, fall back to invoking
        the Node.js CLI directly via ``node .../cli.js``.

        Also checks the ``CLAUDE_CODE_COMMAND`` environment variable.
        """
        import subprocess as _sp

        # Check env var override first
        env_cmd = os.environ.get("CLAUDE_CODE_COMMAND")
        if env_cmd:
            self.claude_command = env_cmd
            return

        # If user explicitly set a non-default command, trust it
        if self.claude_command != "claude":
            return

        # Test if the default 'claude' command works
        try:
            result = _sp.run(
                f"{self.claude_command} --version",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return  # Default command works
        except (_sp.TimeoutExpired, OSError):
            pass

        # Try Node.js fallbacks
        for node_cmd in self._NODE_CLAUDE_PATHS:
            try:
                result = _sp.run(
                    f"{node_cmd} --version",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    logger.info(
                        "[%s] Default 'claude' command failed; using fallback: %s",
                        self.__class__.__name__,
                        node_cmd,
                    )
                    self.claude_command = node_cmd
                    return
            except (_sp.TimeoutExpired, OSError):
                continue

        logger.warning(
            "[%s] Could not find a working Claude Code CLI. "
            "Set CLAUDE_CODE_COMMAND env var or pass claude_command parameter.",
            self.__class__.__name__,
        )

    # === Abstract Method Implementations ===

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """Build CLI arguments for Claude Code session management.

        Claude CLI uses --resume <session_id> (single combined flag).
        Unlike DevMate which has separate --resume and --session-id flags,
        there is no way to pass a session_id without resuming.
        The non-resume session_id case is intentionally unsupported.

        Args:
            session_id: The session ID to resume.
            is_resume: Whether this is a resume operation.

        Returns:
            CLI argument string.
        """
        if is_resume and session_id:
            # Use double quotes for Windows cmd.exe compatibility
            # (single quotes are not recognized by cmd.exe).
            # Session IDs are UUIDs so quoting is defensive, not strictly needed.
            return f'--resume "{session_id}"'
        return ""

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        """Build the shell command string for Claude Code CLI.

        Args:
            inference_input: The input data (prompt string or dict).
            **kwargs: Additional arguments (session_id, resume, output_format,
                use_stdin, etc.).

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
        output_format = kwargs.get("output_format")  # Injected by _ainfer()/_infer()
        verbose = kwargs.get("verbose", False)
        use_stdin = kwargs.get("use_stdin", False)

        command_parts = [self.claude_command]

        # Meta launcher option — must be understood by the launcher wrapper, so
        # emit it before the ``-p`` subcommand. Disables the macOS seatbelt
        # sandbox so ``claude`` can run nested inside an already-sandboxed
        # process (otherwise ``sandbox_apply`` fails → exit 71, empty output).
        if self.disable_osx_sandbox:
            from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.common import (
                DANGEROUSLY_DISABLE_OSX_SANDBOX,
            )

            command_parts.append(f"--{DANGEROUSLY_DISABLE_OSX_SANDBOX}")

        command_parts.append("-p")

        if output_format:
            command_parts.append(f"--output-format {output_format}")
        if verbose:
            command_parts.append("--verbose")
        if kwargs.get("include_partial_messages"):
            command_parts.append("--include-partial-messages")

        command_parts.append(f"--model {self.model_name}")

        if self.effort is not None:
            command_parts.append(f"--effort {self.effort}")

        if self.system_prompt:
            escaped_sys = self._escape_for_shell(self.system_prompt)
            command_parts.append(f'--system-prompt "{escaped_sys}"')

        if self.append_system_prompt:
            escaped_append = self._escape_for_shell(self.append_system_prompt)
            command_parts.append(f'--append-system-prompt "{escaped_append}"')

        if not self.enable_shell:
            if self.allowed_tools:
                # Filter "Bash" from the explicit allowed_tools list
                filtered = [t for t in self.allowed_tools if t != "Bash"]
                if filtered:
                    tools_str = ",".join(filtered)
                    command_parts.append(f'--allowedTools "{tools_str}"')
                else:
                    # All tools were "Bash" - use disallowedTools instead
                    command_parts.append('--disallowedTools "Bash"')
            else:
                # No explicit allowed_tools - use disallowedTools to disable Bash
                command_parts.append('--disallowedTools "Bash"')
        elif self.allowed_tools:
            # Comma-separated for safety (CLI accepts "comma or space-separated")
            tools_str = ",".join(self.allowed_tools)
            command_parts.append(f'--allowedTools "{tools_str}"')

        # Permission mode (consolidated attribute)
        if self.permission_mode == "bypassPermissions":
            command_parts.append("--dangerously-skip-permissions")
        elif self.permission_mode and self.permission_mode != "default":
            command_parts.append(f"--permission-mode {self.permission_mode}")

        if self.max_budget_usd is not None:
            command_parts.append(f"--max-budget-usd {self.max_budget_usd}")

        if is_resume and session_id:
            session_args = self._build_session_args(session_id, is_resume)
            command_parts.append(session_args)

        if self.extra_cli_args:
            command_parts.extend(self.extra_cli_args)

        if not use_stdin:
            # Inline prompt as positional argument (last)
            escaped_prompt = self._escape_for_shell(prompt)
            command_parts.append(f'"{escaped_prompt}"')

        return " ".join(command_parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse command output into a result dict.

        The base class ``_ainfer()`` and ``_infer()`` wrap this dict in
        ``TerminalInferencerResponse.from_dict()``, so return a plain dict
        to avoid double-wrapping.

        Args:
            stdout: Standard output from command.
            stderr: Standard error from command.
            return_code: Process return code.

        Returns:
            Dict with parsed fields (output, session_id, success, etc.).
        """
        result: Dict[str, Any] = {
            "raw_output": stdout.strip() if stdout else "",
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
        }

        json_data = self._extract_json_from_output(stdout)

        if json_data is not None:
            result["output"] = json_data.get("result", "")
            result["session_id"] = json_data.get("session_id")
            result["success"] = (
                not json_data.get("is_error", False) and return_code == 0
            )
            result["total_cost_usd"] = json_data.get("total_cost_usd")
            result["usage"] = json_data.get("usage")
            result["model_usage"] = json_data.get("modelUsage")
            result["num_turns"] = json_data.get("num_turns")
            result["duration_ms"] = json_data.get("duration_ms")
            result["result_type"] = json_data.get("subtype")
        else:
            # Fallback: raw text when JSON parsing fails
            result["output"] = stdout.strip() if stdout else ""
            result["success"] = return_code == 0
            self.log_debug(
                "Failed to parse JSON from stdout; falling back to raw text",
                "ParseFallback",
            )

        if not result.get("success") and "error" not in result:
            result["error"] = (
                stderr.strip()
                if stderr and stderr.strip()
                else result.get("output", f"Command failed with code {return_code}")
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

    def _extract_json_from_output(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from stdout.

        Primary: stdout is clean JSON (banners go to stderr).
        Fallback: find first '{' to last '}' (handles mixed output and multi-line JSON).

        Args:
            stdout: Standard output to parse.

        Returns:
            Parsed JSON dict, or None if parsing fails.
        """
        if not stdout or not stdout.strip():
            return None

        # Primary: entire stdout is JSON
        try:
            parsed = json.loads(stdout.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON substring (handles prefix/suffix noise
        # and multi-line JSON)
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(stdout[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    # === _ainfer() — Inherited from TerminalSessionInferencerBase ===
    #
    # NOT overridden. The base class routes _ainfer() through the streaming
    # pipeline: _ainfer() → super()._ainfer() → ainfer_streaming() →
    # _ainfer_streaming() (subprocess line-by-line), which provides:
    #   - Real-time cache writes (each line flushed to disk immediately)
    #   - Per-line idle timeout (via ainfer_streaming())
    #   - Structured result via parse_output() on accumulated text
    #
    # This is the same pattern used by DevmateCliInferencer.
    #
    # NOTE: The Claude CLI also supports --output-format json for structured
    # output (session_id, cost, usage metadata in a single JSON blob). That
    # mode requires process.communicate() which buffers all output until
    # completion — incompatible with real-time cache. If structured JSON
    # metadata is needed without streaming, callers can pass
    # output_format="json" and use_stdin=True to construct_command() directly
    # with their own subprocess management.

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield real-time text chunks from Claude Code CLI.

        Overrides the parent ``TerminalSessionInferencerBase._ainfer_streaming()``
        to use ``--output-format stream-json --verbose`` for true real-time
        streaming instead of buffered text output.

        Each ``assistant`` event with ``content[].text`` is yielded as a chunk.
        The final ``result`` event is captured into ``_last_stream_result`` so
        that ``ainfer()`` can extract session_id, cost, and usage metadata.

        Args:
            prompt: The prompt string.
            **kwargs: Additional arguments.

        Yields:
            Text chunks as they arrive from Claude.
        """
        import json as _json

        kwargs["output_format"] = "stream-json"
        kwargs["verbose"] = True
        kwargs["include_partial_messages"] = True

        # Honor large_input_mode=STDIN to avoid Windows ARG_MAX (8191 chars
        # on CMD.EXE) and Linux E2BIG. construct_command() will skip inlining
        # the prompt when use_stdin=True; we then pipe it through stdin.
        use_stdin = self.large_input_mode == LargeInputMode.STDIN
        if use_stdin:
            kwargs["use_stdin"] = True

        self._last_stream_result = None  # reset before each call

        command = self.construct_command({"prompt": prompt}, **kwargs)
        full_command = self._build_full_command(command)

        # Bump StreamReader line limit from the asyncio default (64KB) to
        # 16MB. Claude's stream-json events with tool_use input can carry
        # the entire generated document on a single line, easily exceeding
        # 64KB and triggering "Separator is not found, and chunk exceed
        # the limit" on readline.
        process = await asyncio.create_subprocess_shell(
            full_command,
            stdin=asyncio.subprocess.PIPE if use_stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._resolve_subprocess_cwd(),
            limit=16 * 1024 * 1024,
        )

        # Drain stdin/stderr concurrently with stdout so OS pipe buffers
        # (~64KB on Windows) never fill — that would block claude.exe and
        # deadlock the whole chain (observed with 73KB prompts: all 18
        # claude.exe threads stuck in Wait, 0% CPU, no API connection).
        # Mirrors what subprocess.communicate() does internally.
        async def _send_stdin() -> None:
            if not use_stdin or process.stdin is None:
                return
            try:
                process.stdin.write(prompt.encode("utf-8"))
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass  # process may have exited; stderr will surface error
            finally:
                try:
                    process.stdin.close()
                except Exception:
                    pass

        async def _drain_stderr() -> bytes:
            if process.stderr is None:
                return b""
            return await process.stderr.read()

        stdin_task = asyncio.create_task(_send_stdin())
        stderr_task = asyncio.create_task(_drain_stderr())

        try:
            async for line_bytes in process.stdout:
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = _json.loads(line)
                except _json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                # Real-time streaming: text deltas are inside
                # stream_event → event → content_block_delta → delta.text
                if event_type == "stream_event":
                    inner = event.get("event", {})
                    if inner.get("type") == "content_block_delta":
                        delta = inner.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield delta["text"]
                            continue
                    # Any other stream activity (thinking_delta,
                    # input_json_delta, content_block_start, message_start,
                    # ping, etc.) — emit an empty sentinel so the parent
                    # class's dual-timer extends to tool_use_idle_timeout
                    # (default 7200s for ClaudeCodeCli) instead of cutting
                    # off the request mid-thinking on idle_timeout (300s).
                    yield ""

                # Capture result event for session_id / cost / usage metadata
                elif event_type == "result":
                    self._last_stream_result = event
                    yield ""  # also a sign of activity
                else:
                    # system, rate_limit_event, assistant (non-stream), etc.
                    yield ""

        finally:
            try:
                await stdin_task
            except Exception:
                pass
            try:
                stderr_bytes = await stderr_task
            except Exception:
                stderr_bytes = b""
            self._last_streaming_stderr = stderr_bytes.decode("utf-8", errors="replace")
            # If cleanup ran due to timeout/cancellation, the subprocess may
            # still be alive — terminate it so process.wait() doesn't block
            # forever waiting on a hung claude.exe.
            if process.returncode is None:
                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass
            await process.wait()
            if process.returncode != 0:
                logger.warning(
                    "[%s] streaming subprocess exited with code %s. stderr: %s",
                    self.__class__.__name__,
                    process.returncode,
                    self._last_streaming_stderr[:500] if self._last_streaming_stderr else "(empty)",
                )

    def _resolve_subprocess_timeout(
        self, override: Optional[float] = None
    ) -> float:
        """Resolve the subprocess timeout in seconds.

        Args:
            override: Caller-specified timeout override. If provided, used as-is.

        Returns:
            Timeout in seconds: the override if given, otherwise
            ``max(idle_timeout_seconds, 1800)``.
        """
        if override is not None:
            return float(override)
        return float(max(self.idle_timeout_seconds, DEFAULT_SUBPROCESS_TIMEOUT_SECONDS))

    # _ainfer_streaming() — inherited from TerminalSessionInferencerBase.
    # Base class now handles stdin + stderr via large_input_mode=STDIN.

    # === Override: _infer() — Sync with JSON Output ===

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync execution with JSON output format.

        Passes the prompt via stdin (same ARG_MAX mitigation as _ainfer).
        Guarded by ``subprocess.run(timeout=...)`` to prevent indefinite
        hangs. Timeout defaults to ``max(idle_timeout_seconds, 1800)``;
        callers can override via ``subprocess_timeout_seconds`` kwarg.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration (unused).
            **kwargs: Additional arguments.

        Returns:
            Parsed result dictionary from ``parse_output()``.
        """
        kwargs["output_format"] = "json"
        kwargs["use_stdin"] = True
        timeout = self._resolve_subprocess_timeout(
            kwargs.pop("subprocess_timeout_seconds", None)
        )

        if isinstance(inference_input, dict):
            prompt = inference_input.get("prompt", str(inference_input))
        else:
            prompt = str(inference_input)

        command = self.construct_command(inference_input, **kwargs)
        full_command = self._build_full_command(command)

        try:
            result = subprocess.run(
                full_command,
                shell=True,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=self._resolve_subprocess_cwd(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "[%s] Sync subprocess timed out after %ss.",
                self.__class__.__name__,
                timeout,
            )
            raise
        result_dict = self.parse_output(result.stdout, result.stderr, result.returncode)
        return TerminalInferencerResponse.from_dict(result_dict)

    # === Override: ainfer() — Session-Aware ===

    @bridge_entrypoint
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

        # Note: This branch is reachable when session_id=None is explicitly passed
        # as a kwarg while self.active_session_id is set. In normal use (no explicit
        # session_id kwarg), it's dead code since kwargs.get() already falls back
        # to self.active_session_id. Kept for consistency with DevMate pattern.
        if session_id is None:
            if self.auto_resume and self.active_session_id:
                session_id = self.active_session_id
            else:
                is_resume = False

        kwargs["session_id"] = session_id
        kwargs["resume"] = is_resume and session_id is not None

        # Route through _ainfer_single for retry/preprocessing/timeout
        result = await self._ainfer_single(inference_input, inference_config, **kwargs)

        # Update active session from result (try TerminalInferencerResponse,
        # then dict, then fall back to _last_stream_result from streaming)
        result_session_id = getattr(result, "session_id", None)
        if result_session_id is None and isinstance(result, dict):
            result_session_id = result.get("session_id")
        if result_session_id is None and hasattr(self, "_last_stream_result"):
            stream_result = self._last_stream_result
            if isinstance(stream_result, dict):
                result_session_id = stream_result.get("session_id")
                # Also enrich the result object with stream metadata
                if isinstance(result, TerminalInferencerResponse):
                    if result_session_id and not result.session_id:
                        result.session_id = result_session_id
        if result_session_id and result_session_id != self.active_session_id:
            self.active_session_id = result_session_id
            self.log_debug(
                f"Updated active session to: {result_session_id[:8]}...", "Async"
            )

        return result

    # === Override: infer() — Sync Session-Aware ===

    @bridge_entrypoint
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
        result_session_id = getattr(result, "session_id", None)
        if result_session_id is None and isinstance(result, dict):
            result_session_id = result.get("session_id")
        if result_session_id and result_session_id != self.active_session_id:
            self.active_session_id = result_session_id
            self.log_debug(
                f"Updated active session to: {result_session_id[:8]}...", "Sync"
            )

        return result

    # === Override: _yield_filter() — Empty-line suppression + callbacks ===

    async def _yield_filter(
        self, chunks: AsyncIterator[str], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Support stream_callback/output_stream and filter_empty backward compat.

        LIMITATIONS (streaming mode):
        - Uses text mode (no --output-format flag)
        - Session ID, cost, usage metadata NOT available after streaming
        - active_session_id NOT updated — multi-turn via streaming unsupported

        For multi-turn, use ainfer() or the SDK-based ClaudeCodeSdkInferencer.
        """
        stream_callback: Optional[Callable[[str], None]] = kwargs.get("stream_callback")
        output_stream: Optional[TextIO] = kwargs.get("output_stream")

        # Backward compat: translate filter_empty to empty_line_mode
        filter_empty = kwargs.get("filter_empty")
        if filter_empty is not None:
            kwargs["empty_line_mode"] = (
                EmptyLineMode.SUPPRESS_LEADING
                if filter_empty
                else EmptyLineMode.PASS_THROUGH
            )

        async for line in super()._yield_filter(chunks, **kwargs):
            if stream_callback:
                stream_callback(line)
            if output_stream:
                output_stream.write(line)
                output_stream.flush()
            yield line

    # === Override: infer_streaming() — Sync Streaming ===

    def infer_streaming(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Iterator[str]:
        """Sync streaming with same limitations as ainfer_streaming().

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Additional arguments (stream_callback, output_stream, etc.).

        Yields:
            Text lines from Claude's response.
        """
        prompt = self._extract_prompt(inference_input)
        stream_callback: Optional[Callable[[str], None]] = kwargs.pop(
            "stream_callback", None
        )
        output_stream: Optional[TextIO] = kwargs.pop("output_stream", None)

        new_session = kwargs.pop("new_session", False)
        if new_session:
            self.active_session_id = None
        session_id = kwargs.get("session_id", self.active_session_id)
        is_resume = kwargs.get("resume", True)
        if session_id is None:
            if self.auto_resume and self.active_session_id:
                session_id = self.active_session_id
            else:
                is_resume = False
        kwargs["session_id"] = session_id
        kwargs["resume"] = is_resume and session_id is not None

        cache_file = self._open_cache_file(prompt) if self.cache_folder else None
        cache_success = False
        cache_error = None

        try:
            for line in self._infer_streaming({"prompt": prompt}, **kwargs):
                self._append_to_cache(cache_file, line)
                if stream_callback:
                    stream_callback(line)
                if output_stream:
                    output_stream.write(line)
                    output_stream.flush()
                yield line
            cache_success = True
        except Exception as e:
            cache_error = e
            raise
        finally:
            self._finalize_cache(cache_file, cache_success, cache_error)

    # === Result Extraction Methods ===

    def get_streaming_result(self) -> TerminalInferencerResponse:
        """Get parsed result after streaming. Session metadata NOT available.

        Returns:
            Response dict with output, return_code, success, stderr.
        """
        stdout = getattr(self, "_last_streaming_output", "")
        return_code = getattr(self, "_last_streaming_return_code", 0)
        result: Dict[str, Any] = {
            "output": stdout.strip(),
            "raw_output": stdout,
            "return_code": return_code,
            "success": return_code == 0,
            "stderr": getattr(self, "_last_streaming_stderr", ""),
        }
        if return_code != 0:
            result["error"] = (
                result.get("stderr") or f"Command failed with code {return_code}"
            )
        return TerminalInferencerResponse.from_dict(result)

    def get_response_text(self, result: Any) -> str:
        """Extract response text from result dict or TerminalInferencerResponse.

        Args:
            result: Result object or dictionary from inference.

        Returns:
            Response text or error message.
        """
        if isinstance(result, TerminalInferencerResponse):
            if result.success:
                return result.output or ""
            return result.error or "Unknown error occurred"
        if isinstance(result, dict):
            if result.get("success"):
                return result.get("output", "")
            return result.get("error", "Unknown error occurred")
        return str(result)

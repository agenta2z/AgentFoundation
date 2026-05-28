"""DevMate CLI inferencer for executing DevMate CLI commands."""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, TextIO, Union

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.common import (
    InferencerExecutionError,
    MaxIterationsExhaustedError,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.common import (
    _detect_fbsource_root_for,
    generate_config_with_allowed_commands,
    resolve_model_tag,
    SessionMode,
    sync_config_to_target,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalSessionTemplatedInferencerBase,
)

logger: logging.Logger = logging.getLogger(__name__)

_FATAL_DEVMATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"ERROR.*StreamingMessageHandler executor failure"),
    re.compile(r"RECV_TIMEOUT to .*srproxy\.titan\.prod"),
]

_DEVMATE_LOG_TAIL_BYTES: int = 200_000
_INTERNAL_FAILURE_WATCHER_POLL_SECONDS: float = 60.0
_INTERNAL_FAILURE_CACHE_SILENCE_THRESHOLD_SECONDS: float = 120.0

_PORT_FILE_RACE_BACKOFF_SECONDS: float = 2.5


def _looks_like_port_file_race(error_text: str) -> bool:
    return (
        "ENOENT" in error_text
        and "port" in error_text.lower()
    )


def _looks_like_max_iterations(error_text: str) -> bool:
    return (
        "MAX_ITERATIONS" in error_text
        and "Max iterations of" in error_text
        and "reached" in error_text
    )


def _looks_like_tool_use_corruption(error_text: str) -> bool:
    return (
        "tool_use" in error_text
        and "tool_result" in error_text
        and ("must be followed" in error_text or "orphan" in error_text.lower())
    )


@attrs
class DevmateCliInferencer(TerminalSessionTemplatedInferencerBase):
    """
    DevMate CLI inferencer for executing DevMate CLI commands.

    This inferencer wraps the `devmate` CLI tool to execute freeform prompts
    and other DevMate commands programmatically. Supports both sync and async
    operations with session continuation.

    Consistent Session API:
        This inferencer implements the unified session management API shared by
        all three external inferencers (ClaudeCodeSdkInferencer, DevmateSDKInferencer,
        DevmateCliInferencer). The API includes:

        Attributes:
            auto_resume (bool): If True, automatically resume previous session.
                Default True.

        Properties:
            active_session_id: Get the current session ID for resumption.

        Session Methods (sync):
            new_session(prompt, **kwargs): Start a new session, clearing previous.
            resume_session(prompt, session_id=None, **kwargs): Resume a session.
            infer(prompt, **kwargs): Infer with auto-resume if enabled.
            infer_streaming(prompt, **kwargs): Sync streaming inference.

        Session Methods (async):
            anew_session(prompt, **kwargs): Async start new session.
            aresume_session(prompt, session_id=None, **kwargs): Async resume session.
            ainfer(prompt, **kwargs): Async infer with auto-resume.
            ainfer_streaming(prompt, **kwargs): Async streaming inference.

        Keyword Arguments for infer/ainfer:
            session_id (str): Explicit session ID to resume.
            new_session (bool): If True, forces a new session (ignores auto_resume).
            resume (bool): Whether to resume a previous session.

    The inferencer automatically sets up a pre-execution script to change
    to the repo directory before executing devmate commands, which is required
    for proper devmate operation.

    Session Support:
        DevMate supports session continuation via --resume and --session-id flags.
        This inferencer inherits from TerminalSessionInferencerBase to provide:
        - Automatic session tracking (with auto_resume=True by default)
        - Multi-turn conversation support
        - Session history management

    Usage Patterns:
        # Single-turn usage:
        inferencer = DevmateCliInferencer(target_path="/path/to/repo")
        result = inferencer.infer("Help me understand this code")
        print(result["output"])

        # Multi-turn with auto-resume (recommended):
        inferencer = DevmateCliInferencer(target_path="/repo", auto_resume=True)
        r1 = inferencer.new_session("My number is 42")
        r2 = inferencer.infer("What is my number?")  # Auto-resumes!
        r3 = inferencer.infer("New topic", new_session=True)  # Force new

        # Async multi-turn:
        r1 = await inferencer.anew_session("My number is 42")
        r2 = await inferencer.ainfer("What is my number?")  # Auto-resumes!

        # Sync streaming:
        for line in inferencer.infer_streaming("Explain this"):
            print(line, end="")

        # Async streaming:
        async for line in inferencer.ainfer_streaming("Explain this"):
            print(line, end="")

    Attributes:
        target_path (str): Path to the repository the DevMate agent operates
            on. Inherited from ``InferencerBase``. Defaults to ~/fbsource if
            not explicitly specified (Devmate-specific default preserved
            from the historical ``repo_path`` attrib).
        model_name (str): Model to use for inference (e.g., 'claude-sonnet-4.5').
            Defaults to 'claude-sonnet-4.5'.
        max_tokens (int): Maximum tokens for response. Defaults to 32768.
        no_create_commit (bool): If True, prevents DevMate from creating commits.
            Defaults to True.
        context_files (List[str]): Optional list of files to include as context.
        headless (bool): If True, runs devmate in headless mode. Defaults to False.
        dump_output (bool): If True, dumps final structs to a temp file for
            reliable output parsing. Defaults to False.
        timeout_percent (int): Optional Sandcastle timeout percentage.
        config_name (str): The config to use. Defaults to 'freeform'.
            Note: Changing this mid-session or between resume calls may cause
            devmate CLI to reject the resume. Keep consistent within a session.
        privacy_type (str): Optional privacy type (PUBLIC or PRIVATE).
        extra_cli_args (List[str]): Additional CLI args to append.
            Caller is responsible for shell-safe values.
        auto_resume (bool): If True, automatically resume previous session.
            Defaults to True.

    Inherited Session Attributes:
        session_arg_name (str): CLI arg for session ID. Defaults to '--session-id'.
        resume_arg_name (str): CLI arg for resume flag. Defaults to '--resume'.
        active_session_id (str): Currently active session ID.
    """

    # Devmate runs as a CLI subprocess with file-edit / write tools, so it
    # HAS local file access. Override ``InferencerBase``'s False default
    # (inferencer_base.py:117) so that:
    #   - ``TemplatedInferencerBase._build_template_feed`` exposes
    #     ``output_path``, ``workspace_root``, and ``workspace_outputs`` to
    #     the rendered prompt — without these, the agent has no explicit
    #     absolute target and picks its own filename (observed: Devmate
    #     wrote ``/data/users/zgchen/fbsource/role_responsibility_document_mle.md``
    #     instead of the workspace's ``role_document.md``).
    #   - ``{% if has_local_access %}`` blocks in shared templates render —
    #     correct, Devmate IS local.
    #   - BTA's ``_format_worker_results_text`` emits ``(See file: <path>)``
    #     references for worker outputs instead of inlining multi-KB
    #     bodies (ARG_MAX safety).
    # Mirrors ``RovoDevCliInferencer.has_local_access=True`` (rovodev_cli_inferencer.py:111)
    # and ``ClaudeCodeCliInferencer.has_local_access=True`` (claude_code_cli_inferencer.py:87).
    has_local_access: bool = attrib(default=True)

    # Existing attributes
    # ``model_name`` is the Devmate-native model identifier (dot-separated,
    # e.g. ``claude-opus-4.7-1m``) that gets passed verbatim to the
    # ``devmate run`` CLI as a template variable. Prefer setting the
    # inherited ``model_id`` (from ``InferencerBase``) — it's the canonical
    # cross-inferencer attribute and the YAML-cascade target. When
    # ``model_id`` is non-empty, ``__attrs_post_init__`` translates it via
    # ``resolve_model_tag`` and overwrites ``model_name`` (model_id wins).
    # When ``model_id`` is empty, this default is used as-is.
    model_name: str = attrib(default="claude-opus-4.7-1m")
    max_tokens: int = attrib(default=65536)
    no_create_commit: bool = attrib(default=True)
    context_files: Optional[List[str]] = attrib(default=None)

    # Shell-tool gating (mirrors ClaudeCode + rankevolve devmate inferencers).
    # ``enable_shell`` controls whether the shell/execute_command tool is exposed
    # at all. ``allowed_shell_commands`` (when non-empty) restricts that tool to
    # a whitelist of executables via a generated config that extends the base.
    enable_shell: bool = attrib(default=True)
    allowed_shell_commands: Optional[List[str]] = attrib(default=None)

    # Enhanced CLI options
    headless: bool = attrib(default=False)
    dump_output: bool = attrib(default=False)
    timeout_percent: Optional[int] = attrib(default=None)
    config_name: str = attrib(default="freeform")
    privacy_type: Optional[str] = attrib(default=None)
    extra_cli_args: Optional[List[str]] = attrib(default=None)
    # Binary to invoke. Default ``"devmate"`` resolves via PATH to
    # ``/usr/local/bin/devmate`` (the system shell wrapper that execs
    # ``srconveyor devmate``). Point this at the standalone Rust binary
    # (``devmate_standalone/devai/devmate_terminal:devmate_terminal``,
    # also published as ``dmt``) and set ``cli_mode="print"`` to use its
    # headless non-interactive mode.
    cli_binary: str = attrib(default="devmate")
    # CLI invocation style. Controls how ``construct_command`` shapes
    # the argv:
    #   ``"run"``  — canonical devmate: ``devmate run <config>
    #                "prompt=$PROMPT" "key=value"...``. The prompt is
    #                passed as a template variable. This is what the
    #                system ``devmate`` wrapper expects.
    #   ``"print"`` — standalone dmt headless: ``dmt print "<PROMPT>"
    #                -c <config> -- "key=value"...``. The prompt is a
    #                positional arg; remaining template vars come after
    #                ``--``. Use this when ``cli_binary`` points at the
    #                ``devmate_terminal`` binary — its ``run`` subcommand
    #                drops into the TUI which panics in non-interactive
    #                contexts.
    cli_mode: str = attrib(
        default="run",
        validator=lambda _self, _attr, v: v in ("run", "print")
        or (_ for _ in ()).throw(
            ValueError(f"cli_mode must be 'run' or 'print', got {v!r}")
        ),
    )
    # Whether the configured ``cli_binary`` honors ``--no-create-commit``.
    # System ``devmate`` does; the standalone binary does not (it disables
    # commit creation internally via ``SessionOptions.with_create_commit_override(false)``).
    supports_no_create_commit_flag: bool = attrib(default=True)

    # Fault-tolerance attributes
    use_file_for_large_arg_exceeding_size: Union[bool, int] = attrib(default=100_000)
    session_mode: SessionMode = attrib(
        default=SessionMode.SAME_SESSION_ACROSS_ROUNDS,
        converter=lambda v: SessionMode(v) if isinstance(v, str) else v,
    )
    consecutive_error_threshold: int = attrib(default=2)
    _consecutive_error_count: int = attrib(default=0, init=False)
    total_timeout_on_internal_error_detection_seconds: float = attrib(default=2400)
    _internal_failure_detected: bool = attrib(default=False, init=False)
    _internal_failure_reason: str = attrib(default="", init=False)

    # Session management - inherited from StreamingInferencerBase via TerminalSessionInferencerBase:
    #   auto_resume, active_session_id, new_session, anew_session, resume_session, aresume_session

    # Internal state for dump file path (same concurrency constraints as base class)
    _output_file: Optional[str] = attrib(default=None, init=False, repr=False)

    def __attrs_post_init__(self):
        """Translate model_id, then super() runs source_path auto-detect, then apply Devmate ~/fbsource default + sync config."""
        # Translate inherited ``model_id`` (from InferencerBase) into the
        # Devmate-native ``model_name`` (dot-form, ``-1m`` suffix).
        # Mirrors rovodev's ``_translate_model_id_to_config_override`` pattern.
        # Precedence: when ``model_id`` is non-empty, it wins and overwrites
        # ``model_name`` so the YAML cascade contract (``_model_id`` →
        # leaf ``model_id``) selects the actual model for every inferencer.
        # When ``model_id`` is empty, the existing ``model_name`` default
        # (or caller-set value) is used unchanged.
        self._translate_model_id_to_model_name()

        # super() runs IB's __attrs_post_init__, which auto-detects
        # ``source_path`` via ``_detect_source_root()`` (we override below
        # to return the fbsource root — Devmate's custom configs at
        # ``tools/devmate/configs/...`` live above the project root).
        super().__attrs_post_init__()

        # Preserve Devmate's historical default of operating on ~/fbsource
        # when no target is explicitly passed. Was on repo_path pre-removal
        # (mirrored to target_path in the post-init mirror block); now
        # lives directly on target_path (the canonical name inherited from
        # InferencerBase).
        if self.target_path is None:
            self.target_path = os.path.expanduser("~/fbsource")

        # If both shell controls are set, generate an extended config that
        # restricts execute_command to the allowed commands. If only
        # ``allowed_shell_commands`` is set but ``enable_shell=False``,
        # log a precedence warning (the disable wins).
        if self.allowed_shell_commands and not self.enable_shell:
            logger.warning(
                "enable_shell=False takes precedence over allowed_shell_commands=%s "
                "(shell tool will be disabled).",
                self.allowed_shell_commands,
            )
        elif self.allowed_shell_commands and self.enable_shell:
            # Sync base config first (before we mutate self.config_name).
            sync_config_to_target(self.config_name, self.source_path, self.target_path)
            self.config_name = generate_config_with_allowed_commands(
                self.config_name, self.allowed_shell_commands, self.source_path
            )

        # Sync the (possibly updated) config to the target repo if they differ.
        sync_config_to_target(self.config_name, self.source_path, self.target_path)

        # NOTE: no shell ``cd $target_path`` needed in pre_exec_scripts —
        # the framework launches the subprocess with cwd=effective_cwd
        # (via ``_resolve_subprocess_cwd``), and TerminalSessionInferencer-
        # Base's ``_run_pre_exec_scripts_in_subprocess_shell()`` returns
        # True so pre_exec_scripts are joined with ``&&`` into the SAME
        # shell that already starts in effective_cwd. A pre-exec ``cd``
        # would be a no-op.

    @classmethod
    def _detect_source_root(cls) -> "Optional[str]":
        """Devmate-specific source-root detection: the fbsource monorepo root.

        See ``DevmateSDKInferencer._detect_source_root`` for rationale —
        Devmate's custom configs live at ``fbsource/fbcode/tools/devmate/
        configs/...`` (above the AgentFoundation project root), so the
        ``.hg/`` walk gives the correct ``source_path`` for
        ``sync_config_to_target`` to reach them.
        """
        return _detect_fbsource_root_for(cls) or super()._detect_source_root()

    def _translate_model_id_to_model_name(self) -> None:
        """Translate ``self.model_id`` → ``self.model_name`` via ``resolve_model_tag``.

        The inherited ``model_id`` attribute (declared by ``InferencerBase``)
        is the canonical cross-inferencer model selector and the YAML cascade
        target. Devmate's CLI takes the model as a template variable named
        ``model_name`` whose accepted values are defined by Devmate's
        server-side ``ModelName`` enum (e.g. ``claude-opus-4.7-1m``). This
        helper bridges between the two.

        Resolution rules (in priority order):

        1. If ``model_id`` is empty → no-op (legacy default
           ``model_name`` wins; preserves backward compat for callers that
           never set ``model_id``).
        2. If ``model_id`` is set → resolve it via :func:`resolve_model_tag`
           and overwrite ``self.model_name`` with the result. ``model_id``
           always wins when both are set.

        Accepted ``model_id`` formats (all 10 ``ClaudeModels`` enum values
        verified — see ``test_model_id_translation`` for the full matrix):

        - Anthropic API IDs: ``claude-opus-4-7``,
          ``claude-sonnet-4-5-20250929`` (date suffix stripped)
        - 1M-context bracket form: ``claude-opus-4-7[1m]`` →
          ``claude-opus-4.7-1m``
        - Short "latest" aliases: ``opus`` → ``claude-opus-4.7``,
          ``opus[1m]`` → ``claude-opus-4.7-1m``

        Unmapped values pass through unchanged — if Devmate's server rejects
        them, the failure surfaces as a server-side error at infer time
        (preserves the rovodev "errors loudly downstream" contract).
        """
        mid = getattr(self, "model_id", "") or ""
        if not mid:
            return  # no-op: keep current model_name (legacy default or caller-set)
        self.model_name = resolve_model_tag(mid)

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """
        Build CLI arguments for DevMate session management.

        DevMate uses --resume and --session-id flags for session continuation:
        - --resume: Flag to indicate resuming a previous session
        - --session-id 'uuid': The session ID to resume

        Args:
            session_id: The session ID to use.
            is_resume: Whether this is resuming an existing session.

        Returns:
            String containing the CLI arguments (e.g., "--resume --session-id 'abc123'").
        """
        if is_resume and session_id:
            return f"{self.resume_arg_name} {self.session_arg_name} '{session_id}'"
        elif session_id:
            # Just pass session ID without resume (for tracking purposes)
            return f"{self.session_arg_name} '{session_id}'"
        return ""

    def construct_command(self, inference_input: Any, **kwargs) -> str:
        """
        Construct the DevMate CLI command as a shell command string.

        This returns a properly quoted shell command string that matches
        the working bash script format:
        devmate run freeform "prompt=$PROMPT" "model_name=$MODEL" "max_tokens=32768" --no-create-commit

        For session continuation (resume), the format is:
        devmate run --resume --session-id 'uuid' freeform "prompt=$PROMPT" ...

        Args:
            inference_input: The prompt string or dict with 'prompt' key.
            **kwargs: Additional arguments:
                - model_name: Override model name
                - max_tokens: Override max tokens
                - context_files: Override context files
                - session_id: Session ID for continuation
                - resume: Whether to resume a previous session

        Returns:
            Shell command string with properly quoted arguments.
        """
        # Extract prompt
        if isinstance(inference_input, dict):
            prompt = inference_input.get("prompt", str(inference_input))
        else:
            prompt = str(inference_input)

        # Get model and token settings (allow overrides via kwargs)
        model = kwargs.get("model_name", self.model_name)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        context_files = kwargs.get("context_files", self.context_files)

        # Get session settings from kwargs (set by _infer in parent class)
        session_id = kwargs.get("session_id")
        is_resume = kwargs.get("resume", False)

        # Escape prompt for shell (order matters - backslash first)
        escaped_prompt = (
            prompt.replace("\\", "\\\\")  # Escape backslashes first
            .replace('"', '\\"')  # Escape double quotes
            .replace("$", "\\$")  # Escape variable expansion
            .replace("`", "\\`")  # Escape command substitution (backticks)
        )

        # Build the shell command string. ``self.cli_binary`` lets callers
        # swap in a different binary (e.g. the standalone devmate_terminal
        # Rust binary) without rewriting construct_command. ``self.cli_mode``
        # selects between the canonical ``devmate run`` shape and the
        # standalone ``dmt print`` shape (the standalone's ``run`` subcommand
        # drops into the TUI and panics in non-interactive contexts).
        if self.cli_mode == "print":
            # Standalone DMT print mode:
            #   dmt print "<PROMPT>" -c <config> [--resume-session-id <ID>]
            #            [--stream] -- key=value...
            # Prompt is positional; remaining template vars come after ``--``.
            command_parts = [self.cli_binary, "print", f'"{escaped_prompt}"']
            command_parts.append(f"-c {self.config_name}")
            # Note: --stream would be appended here for streaming use; the
            # streaming path uses a separate construct path so we leave this
            # for callers that need explicit streaming.
            # Resume the named session if requested. The standalone CLI
            # accepts ``--resume-session-id <UUID>`` (added to plumb through
            # SessionOptions.with_resume(true).with_resume_existing_session_id).
            if is_resume and session_id:
                command_parts.append(f"--resume-session-id {session_id}")
            command_parts.append("--")
            # Template vars (no "prompt=..." here — prompt is positional).
            command_parts.extend(
                [
                    f'"model_name={model}"',
                    f'"max_tokens={max_tokens}"',
                    f'"enable_shell={str(self.enable_shell).lower()}"',
                ]
            )
            if context_files:
                for file_path in context_files:
                    command_parts.append(f'"context_file={file_path}"')
            # Standalone print mode does not support --no-create-commit
            # (that's a server-side override set via SessionOptions). Skip
            # the no_create_commit and other server-side flags here.
            # dump_output / headless / sandcastle / privacy are also not
            # part of the standalone print interface — they'd cause
            # "unknown argument" errors.
            if self.extra_cli_args:
                command_parts.extend(self.extra_cli_args)
            return " ".join(command_parts)

        # Default cli_mode == "run" — canonical devmate invocation.
        command_parts = [self.cli_binary, "run"]

        # Add session args if resuming (before config name)
        if is_resume and session_id:
            session_args = self._build_session_args(session_id, is_resume)
            command_parts.append(session_args)

        # Add config name
        command_parts.append(self.config_name)

        # Add variables
        command_parts.extend(
            [
                f'"prompt={escaped_prompt}"',
                f'"model_name={model}"',
                f'"max_tokens={max_tokens}"',
                f'"enable_shell={str(self.enable_shell).lower()}"',
            ]
        )

        # Add context files if specified (also quoted)
        if context_files:
            for file_path in context_files:
                command_parts.append(f'"context_file={file_path}"')

        # Add headless flag
        if self.headless:
            command_parts.append("--headless")

        # Add dump output file (use mkstemp for security)
        if self.dump_output:
            fd, self._output_file = tempfile.mkstemp(
                suffix=".json", prefix="devmate_output_"
            )
            os.close(fd)  # Close fd; devmate writes by path
            command_parts.append(f"--dump-final-structs-to-file {self._output_file}")

        # Add timeout percentage (use `is not None` since 0 is a valid value)
        if self.timeout_percent is not None:
            command_parts.append(f"--sandcastle-timeout-percent {self.timeout_percent}")

        # Add privacy type
        if self.privacy_type:
            command_parts.append(f"--privacy-type {self.privacy_type}")

        # Add no_create_commit flag at the END (like the working bash script).
        # Suppressed for binaries that don't accept this flag (e.g. the
        # standalone devmate_terminal sets create-commit-override=false via
        # SessionOptions internally and rejects unknown CLI flags).
        if self.no_create_commit and self.supports_no_create_commit_flag:
            command_parts.append("--no-create-commit")

        # Add extra CLI args (caller responsible for shell-safe values)
        if self.extra_cli_args:
            command_parts.extend(self.extra_cli_args)

        # Return as a single shell command string
        return " ".join(command_parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """
        Parse the DevMate command output.

        If dump_output is enabled, reads from the dump file for more reliable
        output parsing. Falls back to stdout parsing if file is not available.

        This method cleans up the raw DevMate output by:
        1. Removing session header/footer blocks (Session ID, Trajectory, etc.)
        2. Extracting the actual response content
        3. Handling duplicate responses (takes the first one)

        Args:
            stdout: Standard output from DevMate command.
            stderr: Standard error from DevMate command.
            return_code: Process return code.

        Returns:
            Dictionary with:
                - output (str): The cleaned main response content
                - raw_output (str): The original unprocessed output
                - stderr (str): Any error output
                - return_code (int): Process return code
                - success (bool): True if command succeeded
                - session_id (str, optional): Extracted session ID
                - trajectory_url (str, optional): Extracted trajectory URL
                - dump_data (dict, optional): Full dump data if dump_output enabled
                - error (str, optional): Error message if failed
        """
        # ``success`` normally derives from return_code == 0. But the
        # standalone ``devmate_terminal`` (``cli_mode="print"``) is known to
        # exit with SIGABRT (return_code = -6 / 134) during teardown when
        # stdin isn't a TTY, even though the response was emitted to stdout
        # cleanly. Treat that specific case as success so subprocess-driven
        # invocations of the standalone CLI behave normally.
        is_standalone_print_abort = (
            self.cli_mode == "print"
            and return_code in (-6, 134)
            and bool(stdout and stdout.strip())
        )
        result = {
            "raw_output": stdout.strip() if stdout else "",
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
            "success": return_code == 0 or is_standalone_print_abort,
        }

        # Try dump file first (more reliable when enabled)
        if self.dump_output and self._output_file and os.path.exists(self._output_file):
            try:
                with open(self._output_file, "r") as f:
                    dump_data = json.load(f)

                result["output"] = self._extract_output_from_dump(dump_data)
                result["session_id"] = dump_data.get("session_id")
                result["dump_data"] = dump_data

                # Build trajectory URL if we have session_id
                if result.get("session_id"):
                    result["trajectory_url"] = (
                        f"https://www.internalfb.com/intern/devai/devmate/inspector/{result['session_id']}"
                    )

            except Exception as e:
                # Catch all exceptions (JSONDecodeError, IOError, and any unexpected errors)
                # to ensure graceful fallback to stdout parsing
                self.log_debug(f"Failed to process dump file: {e}", "ParseError")
            finally:
                # Always clean up temp file
                self._cleanup_output_file()

        # Fall back to stdout parsing if no dump data
        if "output" not in result:
            session_id = self._extract_session_id(result["raw_output"])
            trajectory_url = self._extract_trajectory_url(result["raw_output"])
            cleaned_output = self._clean_devmate_output(result["raw_output"])

            result["output"] = cleaned_output
            if session_id:
                result["session_id"] = session_id
            if trajectory_url:
                result["trajectory_url"] = trajectory_url

        # In standalone print mode, the session_id is emitted to stderr as
        # ``DMT_SESSION_ID=<uuid>``. Prefer that when present — it's set
        # unconditionally by the standalone binary regardless of TTY, while
        # the stdout regex relies on hyperlink escapes that only show in
        # canonical devmate. Falls through to the existing parsed value
        # (if any) when no marker is found.
        if self.cli_mode == "print":
            standalone_sid = self._extract_session_id_from_stderr(
                result.get("stderr", "")
            )
            if standalone_sid:
                result["session_id"] = standalone_sid

        # Extract finish_reason from diagnostic output
        finish_reason_match = re.search(
            r"FinishReason\.([A-Z_]+)", result.get("raw_output", "")
        )
        if finish_reason_match:
            result["finish_reason"] = finish_reason_match.group(1)

        # Add error if failed
        if return_code != 0 and "error" not in result:
            result["error"] = (
                stderr.strip() if stderr else f"Command failed with code {return_code}"
            )

        return result

    def _extract_output_from_dump(self, dump_data: Dict[str, Any]) -> str:
        """Extract main output from dump file structure."""
        try:
            if "final_response" in dump_data:
                return dump_data["final_response"]
            # Fallback: stringify the dump
            return json.dumps(dump_data, indent=2)
        except Exception as e:
            self.log_debug(f"Failed to extract output from dump: {e}", "ParseError")
            return ""

    def _cleanup_output_file(self) -> None:
        """Clean up temporary output file and clear state.

        Always clears ``self._output_file`` to None, even if the file path
        was already missing — leaving stale state would mislead subsequent
        calls into thinking a dump file is still active.
        """
        if self._output_file and os.path.exists(self._output_file):
            try:
                os.remove(self._output_file)
                self.log_debug(
                    f"Cleaned up output file: {self._output_file}", "Cleanup"
                )
            except OSError as e:
                self.log_debug(f"Failed to clean up output file: {e}", "Cleanup")
        self._output_file = None

    def _extract_session_id_from_stderr(self, stderr: str) -> Optional[str]:
        """Pull the ``DMT_SESSION_ID=<uuid>`` marker from standalone stderr.

        The standalone ``devmate_terminal`` always emits
        ``DMT_SESSION_ID=<uuid>`` on stderr at the start of every print-mode
        session (regardless of TTY) so subprocess callers can capture the
        session id without having to fake a PTY. Returns None when not found.
        """
        if not stderr:
            return None
        import re

        m = re.search(r"DMT_SESSION_ID=([0-9a-fA-F-]{36})", stderr)
        return m.group(1) if m else None

    def _extract_session_id(self, output: str) -> Optional[str]:
        """Extract session ID from DevMate output.

        Devmate wraps the session UUID in OSC8 hyperlink escape codes when
        writing to a terminal-like stream — the raw text looks like::

            Session ID: \\x1b]8;;https://...\\x1b\\<UUID>\\x1b]8;;\\x1b\\

        The literal ``Session ID:`` label can therefore be separated from the
        UUID by ESC bytes that ``[a-f0-9-]`` won't match. We first try to
        locate the canonical UUID v4 pattern anywhere AFTER the ``Session ID:``
        marker, then fall back to the first bare UUID in the whole output, and
        finally to a permissive ``Session ID: <hex>`` match for tests that use
        non-UUID fixtures.
        """
        import re

        # 1. UUID v4 (or any 8-4-4-4-12 hex grouping) anywhere after a
        #    ``Session ID:`` marker — survives OSC8 escape codes.
        uuid_pattern = (
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        )
        m = re.search(
            r"Session ID:.*?(" + uuid_pattern + r")", output, flags=re.DOTALL
        )
        if m:
            return m.group(1)

        # 2. Bare UUID anywhere in the output (e.g. ``Started session <UUID>``
        #    line printed by some devmate versions).
        m = re.search(uuid_pattern, output)
        if m:
            return m.group(0)

        # 3. Permissive fallback for hex-only fixtures used in unit tests
        #    (covers ``Session ID: abc123-def456-789``-style stubs).
        m = re.search(r"Session ID:\s*([a-f0-9-]+)", output)
        if m:
            return m.group(1)

        return None

    def _extract_trajectory_url(self, output: str) -> Optional[str]:
        """Extract trajectory URL from DevMate output."""
        import re

        match = re.search(r"Trajectory:\s*(https?://\S+)", output)
        if match:
            return match.group(1)
        return None

    def _clean_devmate_output(self, output: str) -> str:
        """
        Clean DevMate output by removing session headers/footers.

        DevMate output typically has this structure:
        - Header: "Starting Devmate server...", session info block
        - Content: The actual response
        - Footer: "Finished session...", session info block repeated

        This method extracts just the actual response content.
        """
        if not output:
            return ""

        import re

        # Split into lines for processing
        lines = output.split("\n")
        cleaned_lines = []

        # Patterns to identify session-related lines (case insensitive for robustness)
        session_patterns = [
            r"^Starting Devmate server",
            r"^Finished starting Devmate server",
            r"^Started session",
            r"^Finished session",
            r"^Session ID:",
            r"^Trajectory:",
            r"^Server logs available at:",
            r"^Client logs available at:",
            r"^=+$",  # Separator lines (================)
        ]

        # Compile patterns (case insensitive)
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in session_patterns]

        for line in lines:
            stripped_line = line.strip()

            # Skip empty lines at the beginning
            if not cleaned_lines and not stripped_line:
                continue

            # Check if this line matches any session pattern
            is_session_line = any(
                pattern.match(stripped_line) for pattern in compiled_patterns
            )

            if is_session_line:
                # Skip session-related lines
                continue
            else:
                # This is content
                cleaned_lines.append(line)

        # Join cleaned lines and strip trailing whitespace/empty lines
        cleaned_output = "\n".join(cleaned_lines).strip()

        # Remove any trailing session info that might be on the same line
        # e.g., "response text Finished session abc123"
        cleaned_output = re.sub(
            r"\s*Finished session\s+[a-f0-9-]+\s*$",
            "",
            cleaned_output,
            flags=re.IGNORECASE,
        )

        # Handle duplicate responses (DevMate sometimes outputs response twice)
        # Look for `---` separator that might indicate duplication
        if "---\n" in cleaned_output:
            parts = cleaned_output.split("---\n")
            if len(parts) >= 2:
                # Check if first and second parts are similar (duplicate)
                first_part = parts[0].strip()
                second_part = parts[1].strip() if len(parts) > 1 else ""

                # If they look similar (both start/end similarly), take just the first
                if first_part and second_part:
                    # Simple heuristic: if second part starts like first part
                    first_lines = first_part.split("\n")[:3]
                    second_lines = second_part.split("\n")[:3]

                    if first_lines == second_lines:
                        # It's a duplicate, return just the first part
                        cleaned_output = first_part

        return cleaned_output

    def get_response_text(self, result: Dict[str, Any]) -> str:
        """
        Extract just the response text from a result dictionary.

        This is a convenience method for getting the main output.

        Args:
            result: The result dictionary from infer().

        Returns:
            The main response text, or error message if failed.
        """
        if result.get("success"):
            return result.get("output", "")
        else:
            return result.get("error", "Unknown error occurred")

    def _apply_session_policy(self, kwargs: Dict[str, Any]) -> None:
        """Resolve ``new_session`` / ``session_id`` / ``resume`` into kwargs.

        Mirrors what ``ainfer`` used to do inline so the sync ``infer`` path
        gets the same session resolution (without this, ``inferencer(prompt)``
        in sync mode silently ignores ``active_session_id`` and never sends
        ``--resume --session-id`` to devmate).
        """
        new_session = kwargs.pop("new_session", False)
        if self.session_mode == SessionMode.NEW_SESSION_PER_CALL:
            new_session = True
        elif (
            self.session_mode == SessionMode.NEW_SESSION_ON_CONSECUTIVE_ERRORS
            and self._consecutive_error_count >= self.consecutive_error_threshold
        ):
            new_session = True
            self._consecutive_error_count = 0

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

    def infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs
    ) -> Any:
        """Sync inference with session-mode policy + active-session propagation.

        Without this override, the sync ``__call__`` path (i.e.,
        ``inferencer(prompt)``) would skip the session-id resolution and
        devmate would never see ``--resume --session-id``, silently
        creating a new session on every call.
        """
        self._apply_session_policy(kwargs)
        result = super().infer(inference_input, inference_config, **kwargs)

        # Propagate the response's session_id back into ``active_session_id``
        # so chained calls auto-resume. (Async path already does this; the
        # sync path didn't.)
        def _field(obj: Any, name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        result_session_id = _field(result, "session_id")
        if result_session_id and result_session_id != self.active_session_id:
            self.active_session_id = result_session_id

        return result

    async def ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs
    ) -> Dict[str, Any]:
        """Async inference with session-mode policies and fault-tolerance.

        Routes through ``_ainfer_single`` (matching ``RovoDevCliInferencer``,
        ``ClaudeCodeCliInferencer``, ``KiroCliInferencer``) so we inherit
        the full framework chain:

        - ``_propagate_to_children`` (no-op for leaves)
        - ``input_preprocessor``
        - ``_render_prompt`` (Jinja2 templating — load-bearing for BTA-
          aggregator usage where the prompt is a multi-KB synthesis
          directive plus ``upstream_artifacts`` from peer workers)
        - ``_atry_resume_from_cache`` (partial-stream recovery)
        - ``default_inference_args`` / ``state_graphs`` merging
        - ``timeout`` / ``fallback_mode`` / ``retry_prompt_mode``
        - ``pre_retry`` hooks for session cleanup
        - ``async_execute_with_retry`` around ``self._ainfer``
        - ``_finalize_output`` (output_path → workspace deliverable promotion)
        - ``response_post_processor`` (e.g. ``extract_delimited``)

        Pre-2026-05-26 this method called ``self._ainfer(...)`` directly,
        bypassing the entire chain — most visibly dropping the templated
        prompt (e.g. the create_role aggregator's 28 KB synthesis directive
        was silently degraded to the bare 34-byte BTA user request), which
        in turn caused Devmate's freeform agent to misread the request as
        a coding task and write ``.llms/agents/<name>.md`` files to
        fbsource with an auto-commit instead of producing the workspace's
        ``role_document.md``. The bypass had no design rationale — the
        original author wrote ``ainfer`` to add session-policy and
        fault-tolerance, and built atop the existing one-layer-shallower
        ``_ainfer`` call without noticing the framework features being lost.
        Verified across siblings: RovoDevCli (rovodev_cli_inferencer.py:757),
        ClaudeCodeCli (claude_code_cli_inferencer.py:656),
        KiroCli (kiro_cli_inferencer.py:270) all route through
        ``_ainfer_single``; only DevmateCli was the outlier.

        Fault-tolerance runs AFTER ``_ainfer_single`` returns. This is
        compatible with the framework's retry chain: ``async_execute_with_retry``
        retries on EXCEPTIONS, but Devmate's ``_ainfer`` does not raise on
        ``success=False`` — the promotion-to-exception happens here, after
        retries have exhausted. ``session_mode``'s
        ``_consecutive_error_count`` and the framework's ``max_retry`` (default 1)
        are independent counters and do not conflict.
        """
        self._apply_session_policy(kwargs)

        result = await self._ainfer_single(
            inference_input, inference_config, **kwargs
        )

        # ``_ainfer`` may return either a plain dict (legacy path) or a
        # ``TerminalInferencerResponse`` (current path). Accept both so the
        # session-id propagation and error-promotion logic work in either case.
        def _field(obj: Any, name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        result_session_id = _field(result, "session_id")
        if result_session_id and result_session_id != self.active_session_id:
            self.active_session_id = result_session_id

        # --- Fault-tolerance: promote failures to exceptions ---
        result_success = _field(result, "success", True)
        if not result_success:
            self._consecutive_error_count += 1
            error_text = _field(result, "error", "") or ""

            if self.session_mode == SessionMode.NEW_SESSION_ON_ERROR:
                self.active_session_id = None

            if _looks_like_port_file_race(error_text):
                await asyncio.sleep(_PORT_FILE_RACE_BACKOFF_SECONDS)

            if _looks_like_tool_use_corruption(error_text):
                self.active_session_id = None

            if _looks_like_max_iterations(error_text):
                self.active_session_id = None
                _m = re.search(r"Max iterations of (\d+) reached", error_text)
                raise MaxIterationsExhaustedError(
                    tool="devmate",
                    return_code=_field(result, "return_code"),
                    stderr=_field(result, "stderr", "") or "",
                    error=error_text,
                    max_iterations=int(_m.group(1)) if _m else None,
                    session_id=_field(result, "session_id"),
                )

            raise InferencerExecutionError(
                tool="devmate",
                return_code=_field(result, "return_code"),
                stderr=_field(result, "stderr", "") or "",
                error=error_text,
            )

        self._consecutive_error_count = 0
        return result

    async def ainfer_streaming(
        self,
        prompt: str,
        filter_session_info: bool = True,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Async streaming inference: yields output lines as they arrive.

        Args:
            prompt: The prompt to send to DevMate.
            filter_session_info: If True (default), filters out session header/footer.
            **kwargs: Additional arguments:
                - session_id: Session ID for continuation
                - resume: Whether to resume a previous session
                - new_session: If True, forces a new session

        Yields:
            Lines of DevMate output as they become available.

        Example:
            async for line in inferencer.ainfer_streaming("Explain this code"):
                print(line, end="")
        """
        # Temporarily disable dump_output for streaming (incompatible)
        original_dump_output = self.dump_output
        if self.dump_output:
            self.log_debug(
                "dump_output=True is incompatible with streaming; disabled for this call.",
                "AsyncStream",
            )
            self.dump_output = False

        try:
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

            # Track state for filtering
            content_started = False
            pending_empty_lines = []

            # Open cache file if configured
            cache_file = self._open_cache_file(prompt) if self.cache_folder else None
            success = False
            error = None

            try:
                async for line in self._ainfer_streaming({"prompt": prompt}, **kwargs):
                    self._append_to_cache(cache_file, line)
                    if filter_session_info:
                        if self._is_session_info_line(line):
                            continue

                        stripped = line.strip()
                        if not stripped:
                            if content_started:
                                pending_empty_lines.append(line)
                            continue

                        content_started = True

                        for empty_line in pending_empty_lines:
                            yield empty_line
                        pending_empty_lines = []

                        yield line
                    else:
                        yield line

                success = True
            except Exception as e:
                error = e
                raise
            finally:
                self._finalize_cache(cache_file, success, error)

        finally:
            self.dump_output = original_dump_output

    # === Streaming Methods ===

    def _is_session_info_line(self, line: str) -> bool:
        """
        Check if a line is session-related info that should be filtered.

        Args:
            line: The line to check.

        Returns:
            True if the line should be filtered out (session header/footer).
        """
        import re

        stripped = line.strip()

        # Empty lines pass through (will be filtered later if needed)
        if not stripped:
            return False

        # Patterns for session info lines to filter
        session_patterns = [
            r"^Starting Devmate server",
            r"^Finished starting Devmate server",
            r"^Started session",
            r"^Finished session",
            r"^Session ID:",
            r"^Trajectory:",
            r"^Server logs available at:",
            r"^Client logs available at:",
            r"^=+$",  # Separator lines (================)
        ]

        for pattern in session_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True

        return False

    def infer_streaming(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        output_stream: Optional[TextIO] = None,
        filter_session_info: bool = True,
        **kwargs,
    ) -> Iterator[str]:
        """
        Execute DevMate inference with streaming output.

        This method yields output lines as they become available from the
        DevMate CLI, allowing real-time display of responses.

        Note: dump_output mode is incompatible with streaming. If dump_output=True,
        the dump file will NOT be created during streaming (to prevent temp file leaks).
        Use standard infer() method for dump_output functionality.

        Note: The try/finally pattern in generators has a timing subtlety - the finally
        block only runs when the generator is fully consumed, explicitly closed, or
        garbage collected. In CPython this typically happens promptly due to reference
        counting. This is an inherent Python limitation.

        Args:
            prompt: The prompt to send to DevMate.
            stream_callback: Optional callback invoked for each output line.
            output_stream: Optional TextIO stream to write output to.
            filter_session_info: If True (default), filters out session ID,
                trajectory URL, and other session header/footer info from
                the streaming output. The session info is still available
                via get_streaming_result() after streaming completes.
            **kwargs: Additional arguments:
                - model_name: Override model name
                - max_tokens: Override max tokens
                - context_files: Override context files
                - session_id: Session ID for continuation
                - resume: Whether to resume a previous session
                - new_session: If True, forces a new session

        Yields:
            Lines of DevMate output as they become available.

        Note:
            After iteration completes, call get_streaming_result() to get
            the parsed output with session ID and other metadata.

        Example:
            >>> inferencer = DevmateCliInferencer()
            >>> for line in inferencer.infer_streaming("Explain this code"):
            ...     print(line, end="")  # Print as it streams
            >>> result = inferencer.get_streaming_result()
            >>> print(f"Session ID: {result.get('session_id')}")
        """
        # Temporarily disable dump_output for streaming (incompatible)
        original_dump_output = self.dump_output
        if self.dump_output:
            self.log_debug(
                "dump_output=True is incompatible with streaming; disabled for this call. "
                "Use infer() for dump_output functionality.",
                "Stream",
            )
            self.dump_output = False

        try:
            # Handle new_session flag
            new_session = kwargs.pop("new_session", False)
            if new_session:
                self.active_session_id = None

            # Determine session context (same logic as parent's _infer)
            session_id = kwargs.get("session_id", self.active_session_id)
            is_resume = kwargs.get("resume", True)

            # If no session to resume, this will be a new session
            if session_id is None:
                is_resume = False

            # Update kwargs with session info for construct_command
            kwargs["session_id"] = session_id
            kwargs["resume"] = is_resume

            if is_resume and session_id:
                self.log_debug(
                    f"Streaming with session resume: {session_id[:8]}...", "Stream"
                )
            else:
                self.log_debug("Streaming new session", "Stream")

            # Track state for filtering
            content_started = False
            pending_empty_lines = []

            # Open cache file if configured
            cache_file = self._open_cache_file(prompt) if self.cache_folder else None
            cache_success = False
            cache_error = None

            try:
                # Use the parent's _infer_streaming method with filtering
                for line in self._infer_streaming(
                    {"prompt": prompt},
                    stream_callback=None,  # We handle callback ourselves after filtering
                    output_stream=None,  # We handle output stream ourselves after filtering
                    **kwargs,
                ):
                    # Cache raw line before filtering
                    self._append_to_cache(cache_file, line)

                    if filter_session_info:
                        # Check if this is a session info line
                        if self._is_session_info_line(line):
                            # Skip session info lines
                            continue

                        # Handle empty lines
                        stripped = line.strip()
                        if not stripped:
                            # Buffer empty lines - only output them if content follows
                            if content_started:
                                pending_empty_lines.append(line)
                            continue

                        # This is actual content
                        content_started = True

                        # Output any pending empty lines first
                        for empty_line in pending_empty_lines:
                            if stream_callback:
                                stream_callback(empty_line)
                            if output_stream:
                                output_stream.write(empty_line)
                                output_stream.flush()
                            yield empty_line
                        pending_empty_lines = []

                        # Output the content line
                        if stream_callback:
                            stream_callback(line)
                        if output_stream:
                            output_stream.write(line)
                            output_stream.flush()
                        yield line
                    else:
                        # No filtering - pass through everything
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

        finally:
            # Restore original setting
            self.dump_output = original_dump_output

    def get_streaming_result(self) -> Dict[str, Any]:
        """
        Get the final parsed result after streaming completes.

        Call this method after exhausting the iterator from infer_streaming()
        to get the parsed output with session ID, trajectory URL, and other
        metadata extracted.

        Returns:
            Dictionary with:
                - output (str): The cleaned main response content
                - raw_output (str): The original unprocessed output
                - stderr (str): Any error output (empty for streaming)
                - return_code (int): Process return code
                - success (bool): True if command succeeded
                - session_id (str, optional): Extracted session ID
                - trajectory_url (str, optional): Extracted trajectory URL

        Example:
            >>> inferencer = DevmateCliInferencer()
            >>> # Consume the streaming iterator
            >>> output = "".join(inferencer.infer_streaming("Hello"))
            >>> # Now get the parsed result
            >>> result = inferencer.get_streaming_result()
            >>> session_id = result.get("session_id")
        """
        stdout = getattr(self, "_last_streaming_output", "")
        return_code = getattr(self, "_last_streaming_return_code", 0)

        # Use the existing parse_output method
        result = self.parse_output(stdout, "", return_code)

        # Update active session if we got a new session ID
        session_id = result.get("session_id")
        if session_id and session_id != self.active_session_id:
            self.active_session_id = session_id
            self.log_debug(f"Updated active session to: {session_id[:8]}...", "Stream")

        return result

"""Codex CLI Inferencer.

Wraps the OpenAI ``codex`` CLI (``codex exec``) as a CLI-based
``TerminalSessionTemplatedInferencerBase`` implementation — the Codex
counterpart of ``ClaudeCodeCliInferencer``.

``codex exec`` runs Codex non-interactively. This inferencer drives it with
``--json`` (a JSONL event stream on stdout) so it can stream the agent's reply
and capture the session/thread id for multi-turn resume.

Key differences from Claude Code (handled below):
  * **Resume is a subcommand, not a flag.** Claude uses ``--resume <id>``;
    Codex uses ``codex exec resume <id> [PROMPT]``. The resume subcommand also
    does NOT accept ``-s/--sandbox`` or ``-C/--cd`` (those are inherited from
    the original session), so ``construct_command`` omits them when resuming.
  * **Output + session id come from the ``--json`` JSONL events** rather than a
    single JSON blob: ``{"type":"thread.started","thread_id": "<uuid>"}`` carries
    the session id; ``{"type":"item.completed","item":{"type":"agent_message",
    "text": "..."}}`` carries the assistant text.

Auth: Codex authenticates via ``codex login`` (ChatGPT or API key); this class
does not manage auth. Verify with ``codex login status``.
"""

import asyncio
import json
import logging
import os
import subprocess
from typing import Any, AsyncIterator, Dict, List, Optional

from attr import attrib, attrs

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

logger = logging.getLogger(__name__)

# Valid values for ``codex exec -s/--sandbox``.
_CODEX_SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")

# Meta launcher flag (must precede the ``exec`` subcommand) that skips the macOS
# seatbelt sandbox. macOS forbids NESTING seatbelt sandboxes: when ``codex`` runs
# inside an already-sandboxed process its ``sandbox_apply`` fails and it exits 71
# ("Operation not permitted") with no output. Passing this skips the failing
# nested apply. Mirrors ClaudeCode's ``disable_osx_sandbox`` handling.
_DANGEROUSLY_DISABLE_OSX_SANDBOX = "--dangerously-disable-osx-sandbox"
# Env var that toggles the above for codex inferencers that don't set
# ``disable_osx_sandbox`` explicitly. (Spelling mirrors the Claude variable.)
_ENV_DISABLE_OSX_SANDBOX = "CODEX_INFERANCER_NO_SANDBOX"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag_enabled(name: str) -> bool:
    """Return ``True`` iff env var ``name`` is set to a truthy value
    (``1`` / ``true`` / ``yes`` / ``on``, case-insensitive, whitespace-stripped)."""
    val = os.environ.get(name)
    return val is not None and val.strip().lower() in _TRUTHY_ENV_VALUES


from agent_foundation.common.inferencers.run_context import bridge_entrypoint


@attrs
class CodexCliInferencer(TerminalSessionTemplatedInferencerBase):
    """OpenAI Codex CLI (``codex exec``) as a streaming, session-aware inferencer.

    Mirrors ``ClaudeCodeCliInferencer`` but targets the ``codex`` binary:

    - ``construct_command`` emits ``codex exec [--json] [-s ...] [-m ...] ...``,
      or ``codex exec resume <id> ...`` when resuming a session.
    - ``parse_output`` / ``_ainfer_streaming`` parse Codex's ``--json`` JSONL
      events (``thread.started`` -> session id; ``item.completed`` /
      ``agent_message`` -> text).

    All RunContext/Tier-3 session state, caching, retry, recovery, workspace and
    cwd handling are inherited unchanged from the base chain.
    """

    has_local_access: bool = attrib(default=True)

    # Streaming / timeout knobs (override base defaults; mirror Claude Code CLI).
    idle_timeout_seconds: int = attrib(default=1800)
    tool_use_idle_timeout_seconds: int = attrib(default=7200)
    empty_line_mode: EmptyLineMode = attrib(default=EmptyLineMode.SUPPRESS_LEADING)

    # Codex CLI configuration.
    codex_command: str = attrib(default="codex")
    # ``None`` -> use Codex's configured default model (from ``codex login`` /
    # ``~/.codex/config.toml``); otherwise passed via ``-m``.
    model_name: Optional[str] = attrib(default="gpt-5.5")
    large_input_mode: LargeInputMode = attrib(default=LargeInputMode.STDIN)
    # ``-s`` sandbox policy for fresh ``exec`` calls (resume inherits the
    # session's sandbox). One of ``_CODEX_SANDBOX_MODES``.
    sandbox_mode: Optional[str] = attrib(default="workspace-write")
    # ``--dangerously-bypass-approvals-and-sandbox`` (full autonomy). Note: an
    # enterprise-managed Codex config may reject this and fall back to a safer
    # policy (non-fatal). When True, ``-s`` is omitted.
    dangerously_bypass: bool = attrib(default=False)
    # ``--dangerously-disable-osx-sandbox`` (Meta launcher flag, prepended before
    # ``exec``). ``None`` (default) resolves from the ``CODEX_INFERANCER_NO_SANDBOX``
    # env var in __attrs_post_init__. Required when codex runs inside another
    # seatbelt sandbox (which would otherwise make it exit 71 with no output).
    # SECURITY: removes the OS-level confinement of the agent; only enable when an
    # outer sandbox/isolation is already in force.
    disable_osx_sandbox: Optional[bool] = attrib(default=None)
    # ``--skip-git-repo-check`` so Codex can run outside a git repo.
    skip_git_repo_check: bool = attrib(default=True)
    # ``-c model_reasoning_effort=<level>`` (Codex has no dedicated flag).
    reasoning_effort: Optional[str] = attrib(default=None)
    # Generic ``-c key=value`` TOML config overrides. Values are emitted
    # verbatim, so callers must provide TOML-valid values (e.g. quote strings).
    config_overrides: Optional[Dict[str, Any]] = attrib(default=None)
    # ``--output-schema <FILE>`` (JSON Schema for the model's final response).
    output_schema_path: Optional[str] = attrib(default=None)
    # Escape hatch: extra raw args appended verbatim to the command.
    extra_cli_args: Optional[List[str]] = attrib(default=None)

    # Anthropic/Claude model tags that may cascade via ``_model_name`` from configs
    # targeting heterogeneous CLI topologies. Codex can't use these; reset to the
    # OpenAI default so Codex uses a valid model.
    _NON_CODEX_MODEL_PREFIXES = ("opus", "sonnet", "haiku", "claude", "fable")

    def __attrs_post_init__(self) -> None:
        """Initialize defaults after attrs init."""
        if self.target_path is None:
            self.target_path = os.getcwd()
        # ``--dangerously-disable-osx-sandbox`` is a macOS-only launcher flag;
        # Linux/Windows ``codex`` builds reject it. Coerce to False off macOS
        # so a multi-OS shell rc with ``CODEX_INFERANCER_NO_SANDBOX=1`` doesn't
        # break every subprocess. (Mirrors the ClaudeCode treatment.)
        import sys as _sys
        if _sys.platform != "darwin":
            if self.disable_osx_sandbox is True or _env_flag_enabled(_ENV_DISABLE_OSX_SANDBOX):
                logger.debug(
                    "%s is a macOS-only flag; ignoring on platform=%s.",
                    _DANGEROUSLY_DISABLE_OSX_SANDBOX, _sys.platform,
                )
            self.disable_osx_sandbox = False
        elif self.disable_osx_sandbox is None:
            self.disable_osx_sandbox = _env_flag_enabled(_ENV_DISABLE_OSX_SANDBOX)
        if self.model_tier is not None:
            self.model_name = self._resolve_model_for_tier(self.model_tier)
        elif self.model_name and any(
            self.model_name.lower().startswith(p) for p in self._NON_CODEX_MODEL_PREFIXES
        ):
            logger.info(
                "[%s] Ignoring non-Codex model_name=%r (likely cascaded from "
                "_model_name); resetting to default.",
                self.__class__.__name__,
                self.model_name,
            )
            self.model_name = "gpt-5.5"
        if (
            self.sandbox_mode is not None
            and self.sandbox_mode not in _CODEX_SANDBOX_MODES
        ):
            logger.warning(
                "[%s] Unknown sandbox_mode=%r; expected one of %s. Passing through.",
                self.__class__.__name__,
                self.sandbox_mode,
                _CODEX_SANDBOX_MODES,
            )
        self._resolve_codex_command()
        super().__attrs_post_init__()

    _CODEX_TIER_MAP = {
        "max": "gpt-5.5",
        "default": "gpt-5.4",
        "lite": "gpt-5.4-mini",
    }

    @classmethod
    def _resolve_model_for_tier(cls, tier: str) -> str:
        return cls._CODEX_TIER_MAP.get(str(tier).lower(), "gpt-5.5")

    def _resolve_codex_command(self) -> None:
        """Verify the ``codex`` command works; honor the ``CODEX_COMMAND`` override.

        Precedence: ``CODEX_COMMAND`` env var > an explicitly-set non-default
        ``codex_command`` > probing ``codex --version``.
        """
        import subprocess as _sp

        env_cmd = os.environ.get("CODEX_COMMAND")
        if env_cmd:
            self.codex_command = env_cmd
            return

        # If the user explicitly set a non-default command, trust it.
        if self.codex_command != "codex":
            return

        probe = self.codex_command
        if self.disable_osx_sandbox:
            probe = f"{probe} {_DANGEROUSLY_DISABLE_OSX_SANDBOX}"
        try:
            result = _sp.run(
                f"{probe} --version",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return
        except (_sp.TimeoutExpired, OSError):
            pass

        logger.warning(
            "[%s] Could not verify the 'codex' CLI ('codex --version' failed). "
            "Set the CODEX_COMMAND env var or pass codex_command=. Continuing "
            "optimistically.",
            self.__class__.__name__,
        )

    # === Abstract method implementations ===

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """Codex resumes via the ``codex exec resume <id>`` SUBCOMMAND, not a flag.

        Session selection is therefore performed directly in
        ``construct_command`` (the verb changes from ``exec`` to
        ``exec resume <id>``). This hook exists only to satisfy the abstract
        base contract and returns no flag fragment.
        """
        return ""

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        """Build the shell command string for ``codex exec``.

        Honors kwargs ``session_id``, ``resume`` (bool), ``use_json`` (inject
        ``--json``), and ``use_stdin`` (read the prompt from stdin instead of
        inlining it).
        """
        if isinstance(inference_input, dict):
            prompt = inference_input.get("prompt", str(inference_input))
        else:
            prompt = str(inference_input)

        session_id = kwargs.get("session_id")
        is_resume = bool(kwargs.get("resume", False)) and bool(session_id)
        use_json = kwargs.get("use_json", False)
        use_stdin = kwargs.get("use_stdin", False)

        parts: List[str] = [self.codex_command]
        # ``--dangerously-disable-osx-sandbox`` is a launcher flag: it MUST precede
        # the ``exec`` subcommand.
        if self.disable_osx_sandbox:
            parts.append(_DANGEROUSLY_DISABLE_OSX_SANDBOX)
        parts.append("exec")

        # Resume is a subcommand: ``codex exec resume <session_id> [PROMPT]``.
        if is_resume:
            parts.append("resume")
            parts.append(f'"{session_id}"')

        if use_json:
            parts.append("--json")

        # Sandbox / approval policy. ``-s`` and the bypass flag are only valid on
        # a fresh ``exec`` (the resume subcommand rejects ``-s`` — it inherits
        # the original session's sandbox).
        if self.dangerously_bypass:
            parts.append("--dangerously-bypass-approvals-and-sandbox")
        elif not is_resume and self.sandbox_mode:
            parts.append(f"-s {self.sandbox_mode}")

        if self.skip_git_repo_check:
            parts.append("--skip-git-repo-check")

        if self.model_name:
            parts.append(f"-m {self._escape_for_shell(self.model_name)}")

        if self.reasoning_effort:
            parts.append(f'-c model_reasoning_effort="{self.reasoning_effort}"')

        for key, value in (self.config_overrides or {}).items():
            parts.append(f"-c {key}={value}")

        if self.output_schema_path:
            parts.append(f'--output-schema "{self.output_schema_path}"')

        if self.extra_cli_args:
            parts.extend(self.extra_cli_args)

        # Prompt placement (last positional). With stdin we pass ``-`` so Codex
        # reads the prompt from stdin (avoids ARG_MAX limits on large prompts).
        if use_stdin:
            parts.append("-")
        else:
            parts.append(f'"{self._escape_for_shell(prompt)}"')

        return " ".join(parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse ``codex exec --json`` JSONL output into a result dict.

        The base ``_ainfer``/``_infer`` wrap this dict in
        ``TerminalInferencerResponse.from_dict()``, so return a plain dict.
        """
        result: Dict[str, Any] = {
            "raw_output": stdout.strip() if stdout else "",
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
        }

        parsed = self._parse_codex_events(stdout)

        if parsed is not None and parsed.get("text") is not None:
            result["output"] = parsed["text"]
            result["session_id"] = parsed.get("session_id")
            result["usage"] = parsed.get("usage")
            result["success"] = return_code == 0
        else:
            # Fallback: the streaming path feeds the already-clean accumulated
            # text here (not JSONL), and any non-JSONL stdout lands here too.
            result["output"] = stdout.strip() if stdout else ""
            if parsed is not None and parsed.get("session_id"):
                result["session_id"] = parsed["session_id"]
            result["success"] = return_code == 0

        if not result.get("success") and "error" not in result:
            result["error"] = (
                stderr.strip()
                if stderr and stderr.strip()
                else result.get("output", f"Command failed with code {return_code}")
            )

        return result

    # === Helper methods ===

    def _parse_codex_events(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Parse a ``codex exec --json`` JSONL stream.

        Returns a dict with ``session_id`` (from ``thread.started``), ``text``
        (concatenated ``agent_message`` items, or ``None`` if none), ``usage``
        (from ``turn.completed``) and ``errors``; or ``None`` if no JSONL events
        were found (e.g. the input is already-clean accumulated text).
        """
        if not stdout or not stdout.strip():
            return None

        session_id: Optional[str] = None
        texts: List[str] = []
        usage: Optional[Dict[str, Any]] = None
        errors: List[str] = []
        saw_event = False

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            saw_event = True
            etype = event.get("type")
            if etype == "thread.started":
                session_id = event.get("thread_id") or session_id
            elif etype == "item.completed":
                item = event.get("item") or {}
                itype = item.get("type")
                if itype == "agent_message":
                    text = item.get("text")
                    if text:
                        texts.append(text)
                elif itype == "error":
                    msg = item.get("message")
                    if msg:
                        errors.append(msg)
            elif etype == "turn.completed":
                usage = event.get("usage") or usage

        if not saw_event:
            return None

        return {
            "session_id": session_id,
            "text": "\n".join(texts) if texts else None,
            "usage": usage,
            "errors": errors,
        }

    def _escape_for_shell(self, text: str) -> str:
        """Shell-escape text for use inside a double-quoted string.

        Escapes (in order): backslash, double-quote, ``$``, backtick.
        """
        return (
            text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("$", "\\$")
            .replace("`", "\\`")
        )

    def _resolve_subprocess_timeout(
        self, override: Optional[float] = None
    ) -> float:
        """Resolve the sync subprocess wall-clock timeout in seconds."""
        if override is not None:
            return float(override)
        return float(max(self.idle_timeout_seconds, DEFAULT_SUBPROCESS_TIMEOUT_SECONDS))

    # === Streaming primitive ===

    async def _ainfer_streaming(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Yield text chunks from ``codex exec --json``.

        Overrides the base stdout-line streaming to parse Codex's JSONL events:
        ``agent_message`` items are yielded as text; the ``thread.started`` id
        and ``turn.completed`` usage are captured into ``_last_stream_result``
        so ``ainfer()`` can recover the session id and metadata. Every other
        event yields an empty activity sentinel so the dual idle timer extends
        to ``tool_use_idle_timeout`` while Codex is thinking / running tools.
        """
        kwargs["use_json"] = True

        use_stdin = self.large_input_mode == LargeInputMode.STDIN
        if use_stdin:
            kwargs["use_stdin"] = True

        self._last_stream_result = None  # reset before each call

        command = self.construct_command({"prompt": prompt}, **kwargs)
        full_command = self._build_full_command(command)

        # 16 MB line limit: a single ``--json`` event (e.g. a large file_change
        # patch) can far exceed asyncio's default 64 KB readline cap.
        process = await asyncio.create_subprocess_shell(
            full_command,
            stdin=asyncio.subprocess.PIPE if use_stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._resolve_subprocess_cwd(),
            limit=16 * 1024 * 1024,
        )

        # Drain stdin/stderr concurrently with stdout so OS pipe buffers never
        # fill and deadlock the child (mirrors subprocess.communicate()).
        async def _send_stdin() -> None:
            if not use_stdin or process.stdin is None:
                return
            try:
                process.stdin.write(prompt.encode("utf-8"))
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass  # process may have exited; stderr surfaces the error
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
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue

                etype = event.get("type")

                if etype == "thread.started":
                    thread_id = event.get("thread_id")
                    if thread_id:
                        if not isinstance(self._last_stream_result, dict):
                            self._last_stream_result = {}
                        self._last_stream_result["session_id"] = thread_id
                        # Record the id on the instance backing (NOT via the
                        # active_session_id property) so it is readable after a
                        # streaming-only call AND cleared by reset_session(). A
                        # context-scoped property write would land in the per-branch
                        # connection store, which reset_session() does not touch —
                        # leaking a stale session into the next "new" conversation.
                        self._session_id = thread_id
                    yield ""
                elif etype == "item.completed":
                    item = event.get("item") or {}
                    if item.get("type") == "agent_message":
                        text = item.get("text")
                        yield text if text else ""
                    else:
                        # error / reasoning / command_execution / file_change ...
                        yield ""
                elif etype == "turn.completed":
                    usage = event.get("usage")
                    if usage is not None:
                        if not isinstance(self._last_stream_result, dict):
                            self._last_stream_result = {}
                        self._last_stream_result["usage"] = usage
                    yield ""
                else:
                    # turn.started, item.started/updated, etc.
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
            self._last_streaming_stderr = stderr_bytes.decode(
                "utf-8", errors="replace"
            )
            if process.returncode is None:
                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass
            await process.wait()
            if process.returncode != 0:
                logger.warning(
                    "[%s] codex streaming subprocess exited with code %s. stderr: %s",
                    self.__class__.__name__,
                    process.returncode,
                    self._last_streaming_stderr[:500]
                    if self._last_streaming_stderr
                    else "(empty)",
                )

    # === Sync one-shot ===

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync execution: ``codex exec --json`` with the prompt piped via stdin."""
        kwargs["use_json"] = True
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
                "[%s] Sync codex subprocess timed out after %ss.",
                self.__class__.__name__,
                timeout,
            )
            raise

        result_dict = self.parse_output(
            result.stdout, result.stderr, result.returncode
        )
        return TerminalInferencerResponse.from_dict(result_dict)

    # === Session-aware public overrides ===

    @bridge_entrypoint
    async def ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Async inference with session management (read-before / write-after)."""
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

        # Route through _ainfer_single for retry/preprocessing/timeout.
        result = await self._ainfer_single(
            inference_input, inference_config, **kwargs
        )

        # Recover the session id from the result, then the streamed metadata.
        result_session_id = getattr(result, "session_id", None)
        if result_session_id is None and isinstance(result, dict):
            result_session_id = result.get("session_id")
        if result_session_id is None and isinstance(
            getattr(self, "_last_stream_result", None), dict
        ):
            result_session_id = self._last_stream_result.get("session_id")
            if (
                result_session_id
                and isinstance(result, TerminalInferencerResponse)
                and not result.session_id
            ):
                result.session_id = result_session_id
        if result_session_id and result_session_id != self.active_session_id:
            self.active_session_id = result_session_id
            self.log_debug(
                f"Updated active session to: {result_session_id[:8]}...", "Async"
            )

        return result

    @bridge_entrypoint
    def infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync inference with session management (mirrors ``ainfer``)."""
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

        result = self._infer_single(inference_input, inference_config, **kwargs)

        result_session_id = getattr(result, "session_id", None)
        if result_session_id is None and isinstance(result, dict):
            result_session_id = result.get("session_id")
        if result_session_id and result_session_id != self.active_session_id:
            self.active_session_id = result_session_id
            self.log_debug(
                f"Updated active session to: {result_session_id[:8]}...", "Sync"
            )

        return result

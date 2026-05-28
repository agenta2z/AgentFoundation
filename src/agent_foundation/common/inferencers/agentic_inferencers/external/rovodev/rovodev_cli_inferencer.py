"""Rovo Dev CLI inferencer.

Wraps ``acli rovodev legacy`` (default) or ``acli rovodev`` (non-legacy) for
single-shot and multi-turn programmatic use.
Follows the same pattern as ``ClaudeCodeCliInferencer``.

Usage::

    # Single-turn:
    inf = RovoDevCliInferencer(target_path="/path/to/repo")
    result = inf("What does this repo do?")

    # Multi-turn (auto-resume):
    inf = RovoDevCliInferencer(target_path="/repo")
    r1 = inf.new_session("My number is 42")
    r2 = inf("What is my number?")  # auto-resumes last session

    # Streaming:
    for chunk in inf.infer_streaming("Explain this codebase"):
        print(chunk, end="")

    # Async:
    result = await inf.ainfer("Fix this bug")

    # Structured output:
    inf = RovoDevCliInferencer(
        target_path="/repo",
        output_schema='{"type":"object","properties":{"answer":{"type":"string"}}}',
    )
    result = inf("What is 2+2?")
"""

import json
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
    extract_json_from_output,
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
    TerminalSessionTemplatedInferencerBase,
)

logger: logging.Logger = logging.getLogger(__name__)

# Per-call output file path for async parallel safety
_current_output_file: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_output_file", default=None
)

# Schema auto-injected in non-legacy mode to capture clean LLM output as JSON.
# This preserves XML tags that would otherwise be eaten by Rich TUI rendering.
_NON_LEGACY_OUTPUT_SCHEMA = '{"type":"object","properties":{"response":{"type":"string"}}}'


@attrs
class RovoDevCliInferencer(TerminalSessionTemplatedInferencerBase):
    """Rovo Dev CLI inferencer (``acli rovodev legacy`` or ``acli rovodev``).

    Supports two CLI modes controlled by ``enable_legacy``:

    - **Legacy mode** (default, ``enable_legacy=True``): Uses ``acli rovodev legacy``.
      Captures clean output via ``--output-file``. Supports ``--jira``,
      ``--enable-deep-plan``, ``--agent-mode``.
    - **Non-legacy mode** (``enable_legacy=False``): Uses ``acli rovodev`` (TUI mode,
      non-interactive when a message is provided). Captures clean output by
      auto-injecting ``--output-schema`` and parsing JSON from stdout.
      Supports ``--config-override``.

    Attributes:
        acli_path: Path to acli binary. Auto-detected via ``shutil.which`` if None.
        enable_legacy: Use legacy CLI mode (default True). False uses non-legacy/TUI mode.
        config_file: Path to rovodev config file.
        cloud_id: Atlassian cloud ID.
        yolo: Skip tool confirmation prompts (default True for programmatic use).
        enable_deep_plan: Enable deep planning mode (legacy only).
        xid: Experience ID for analytics tagging.
        output_schema: JSON schema for structured output.
        output_file: Path for output file capture (legacy only).
        raw_output_to_file: Auto-capture clean output (legacy: temp file, non-legacy: output-schema).
        agent_mode: Agent mode override (legacy only, e.g., "ask", "plan").
        jira: Jira ticket URL to start work on (legacy only).
        config_override: JSON config override string (both legacy and non-legacy modes).
            Defaults to claude-opus-4-6 — opus correctly writes <Response> JSON to
            --output-file; sonnet writes to stdout only (not captured by output file).
        extra_cli_args: Additional CLI arguments to pass through.
        idle_timeout_seconds: Idle timeout between output chunks (default 30 min).
        tool_use_idle_timeout_seconds: Idle timeout during tool use (default 2 hr).
    """

    # --- Configuration ---
    has_local_access: bool = attrib(default=True)
    enable_legacy: bool = attrib(default=True)
    acli_path: Optional[str] = attrib(default=None)
    config_file: Optional[str] = attrib(default=None)
    config_override: Optional[str] = attrib(
        default='{"agent": {"modelId": "anthropic:claude-opus-4-7"}}'
    )
    cloud_id: Optional[str] = attrib(default=None)
    yolo: bool = attrib(default=True)
    enable_deep_plan: bool = attrib(default=False)
    efficiency_level: Optional[str] = attrib(default=None)
    xid: Optional[str] = attrib(default=None)
    output_schema: Optional[str] = attrib(default=None)
    output_file: Optional[str] = attrib(default=None)
    raw_output_to_file: bool = attrib(default=True)  # Always capture clean LLM output via --output-file
    agent_mode: Optional[str] = attrib(default=None)
    jira: Optional[str] = attrib(default=None)
    extra_cli_args: Optional[List[str]] = attrib(default=None)

    # CLI stdout is noisy TUI output; --output-file has clean LLM text.
    streams_differ_from_final_output: bool = True

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
        # Translate inherited ``model_id`` (from InferencerBase) into the
        # RovoDev-specific ``config_override.agent.modelId`` field.
        #
        # YAML cascade contract: ``_model_id: <id>`` at any ancestor cascades
        # to every leaf inferencer's ``model_id``. RovoDev's actual model
        # selection mechanism is ``config_override.agent.modelId``, so we
        # translate here. When ``model_id`` is empty (the inherited default),
        # ``config_override`` is left at its own framework default
        # (``claude-opus-4-7``), preserving back-compat for callers that
        # don't set ``model_id`` and rely on the legacy ``config_override``.
        self._translate_model_id_to_config_override()
        super().__attrs_post_init__()

    def _translate_model_id_to_config_override(self) -> None:
        """Translate ``self.model_id`` → ``config_override.agent.modelId``.

        RovoDev's CLI uses ``--config-override <json>`` to drive model
        selection. The inherited ``model_id`` attribute (declared by
        ``InferencerBase``) is the canonical cascade target; this helper
        bridges it to RovoDev's native format.

        Resolution rules (in priority order):

        1. If ``model_id`` is empty → no-op (legacy default
           ``config_override`` wins; preserves backward compat for callers
           that never set ``model_id``).
        2. If ``model_id`` is set → upsert ``agent.modelId`` in the existing
           ``config_override`` JSON, preserving any other keys (e.g.,
           ``toolPermissions``).

        Model ID format:

        - ``model_id`` should be a bare Anthropic ID like
          ``claude-opus-4-7`` or ``claude-sonnet-4-6``. The ``anthropic:``
          prefix is added automatically.
        - Alias forms like ``opus[1m]`` from ``ClaudeModels`` are
          PASSED THROUGH UNCHANGED — if RovoDev's router accepts them,
          they work; if not, RovoDev errors loudly. Translation does NOT
          attempt alias resolution because RovoDev owns its own router.

        Examples::

            model_id="claude-opus-4-7"
              → config_override="{\\"agent\\": {\\"modelId\\":
                \\"anthropic:claude-opus-4-7\\"}}"

            model_id="anthropic:claude-opus-4-7"  # already prefixed
              → config_override="{\\"agent\\": {\\"modelId\\":
                \\"anthropic:claude-opus-4-7\\"}}"  # unchanged
        """
        mid = getattr(self, "model_id", "") or ""
        if not mid:
            return  # no-op: keep current config_override default
        try:
            base = (
                json.loads(self.config_override)
                if self.config_override
                else {}
            )
            if not isinstance(base, dict):
                base = {}
        except (json.JSONDecodeError, TypeError):
            base = {}
        # Auto-prefix with anthropic: if missing (matches existing default)
        normalized = mid if ":" in mid else f"anthropic:{mid}"
        agent = base.setdefault("agent", {})
        agent["modelId"] = normalized
        self.config_override = json.dumps(base)

    # =========================================================================
    # Config override composition (merges base allowed_paths into acli config)
    # =========================================================================

    def _compose_config_override_for_cli(self) -> Optional[str]:
        """Return the ``--config-override`` JSON string to pass to acli.

        Starts from ``self.config_override`` (the user's / default override),
        then deep-merges the resolved ``effective_allowed_paths`` from the
        base class into ``toolPermissions.allowedExternalPaths``.

        acli's allowedExternalPaths is a flat path list with no R/W
        distinction, so the ``access`` field on each ``AllowedPath`` is
        intentionally ignored here — every path is granted whatever acli's
        binary "external" semantics imply. If/when acli gains per-path R/W
        scoping, this is the place to translate ``access`` accordingly.

        The union with any user-supplied ``toolPermissions.allowedExternalPaths``
        in ``config_override`` is computed at the leaf-list level so we don't
        clobber what the user explicitly asked for.

        Returns:
            Composed JSON string, or None if both the override and the path
            list are empty (caller should then skip the flag entirely).
        """
        try:
            base = json.loads(self.config_override) if self.config_override else {}
        except (TypeError, ValueError) as e:
            logger.warning(
                "config_override is not valid JSON, ignoring base and using only "
                "effective_allowed_paths: %s",
                e,
            )
            base = {}

        # Collect every distinct path the inferencer wants the agent to access.
        # `effective_allowed_paths` already auto-includes workspace.root and
        # deduplicates by resolved absolute path.
        path_strs = [ap.path for ap in self.effective_allowed_paths if ap and ap.path]

        if not path_strs and not self.efficiency_level:
            return self.config_override if base else None

        if not isinstance(base, dict):
            logger.warning(
                "config_override is not a JSON object, ignoring base and using only "
                "effective_allowed_paths"
            )
            base = {}

        tool_perms = base.setdefault("toolPermissions", {})
        if not isinstance(tool_perms, dict):
            # Defensive: caller wrote a non-dict at toolPermissions. Replace.
            tool_perms = {}
            base["toolPermissions"] = tool_perms

        existing = tool_perms.get("allowedExternalPaths") or []
        if not isinstance(existing, list):
            existing = []

        # Union, preserving order: existing-first, then any new from path_strs.
        seen = set()
        merged: List[str] = []
        for p in list(existing) + path_strs:
            if not isinstance(p, str) or not p:
                continue
            if p in seen:
                continue
            seen.add(p)
            merged.append(p)

        tool_perms["allowedExternalPaths"] = merged

        # Merge efficiency_level into agent config if set
        if self.efficiency_level:
            base.setdefault("agent", {})["efficiencyLevel"] = self.efficiency_level

        return json.dumps(base)

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def construct_command(self, inference_input: Any, **kwargs: Any) -> str:
        """Build the CLI command string.

        Builds ``acli rovodev legacy <prompt>`` when ``enable_legacy=True``,
        or ``acli rovodev <prompt>`` when ``enable_legacy=False``.

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

        # --- Command stem ---
        if self.enable_legacy:
            parts = [acli, ACLI_SUBCOMMAND, "legacy", shlex.quote(prompt)]
        else:
            parts = [acli, ACLI_SUBCOMMAND, shlex.quote(prompt)]

        # --- Common flags (both modes) ---
        if self.yolo:
            parts.append("--yolo")
        if self.config_file:
            parts.extend(["--config-file", self.config_file])
        if self.xid:
            parts.extend(["--xid", self.xid])
        if self.output_schema:
            parts.extend(["--output-schema", shlex.quote(self.output_schema)])

        # Session resumption — works in both modes via --restore
        session_id = kwargs.get("session_id")
        is_resume = kwargs.get("resume", False)
        if is_resume:
            session_args = self._build_session_args(
                session_id or "", is_resume
            )
            if session_args:
                parts.append(session_args)

        # --- Mode-specific flags ---
        if self.enable_legacy:
            # Output file for reliable output capture (legacy only).
            # Preserves XML tags that Rich TUI would strip from stdout.
            out_path = kwargs.get("output_file") or self.output_file
            if not out_path and self.raw_output_to_file:
                out_path = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
                kwargs["_auto_output_file"] = out_path
            if out_path:
                parts.extend(["--output-file", out_path])

            # Legacy-only flags
            if self.jira:
                parts.extend(["--jira", self.jira])
            if self.enable_deep_plan:
                parts.append("--enable-deep-plan")
            if self.agent_mode:
                parts.extend(["--agent-mode", self.agent_mode])
            # config_override works in both legacy and non-legacy modes; we
            # compose it dynamically so effective_allowed_paths (from the base)
            # gets merged into toolPermissions.allowedExternalPaths.
            composed_override = self._compose_config_override_for_cli()
            if composed_override:
                parts.extend(["--config-override", shlex.quote(composed_override)])
        else:
            # Non-legacy: auto-inject --output-schema for clean output capture
            # when user hasn't set their own schema and raw_output_to_file is on.
            if not self.output_schema and self.raw_output_to_file:
                parts.extend(["--output-schema", shlex.quote(_NON_LEGACY_OUTPUT_SCHEMA)])
                kwargs["_auto_output_schema"] = True

            # config_override works in both legacy and non-legacy modes; see above.
            composed_override = self._compose_config_override_for_cli()
            if composed_override:
                parts.extend(["--config-override", shlex.quote(composed_override)])

            # Warn about legacy-only flags that were set
            for flag_name, flag_val in [
                ("jira", self.jira),
                ("enable_deep_plan", self.enable_deep_plan),
                ("agent_mode", self.agent_mode),
            ]:
                if flag_val:
                    logger.warning(
                        "'%s' is legacy-only and ignored in non-legacy mode",
                        flag_name,
                    )
            if self.output_file:
                logger.warning(
                    "'output_file' is legacy-only and ignored in non-legacy mode"
                )

        if self.extra_cli_args:
            parts.extend(self.extra_cli_args)

        return " ".join(parts)

    def parse_output(
        self, stdout: str, stderr: str, return_code: int,
        output_file_path: Optional[str] = None,
    ) -> dict:
        """Parse CLI output.

        Strategy (in order):
        1. Non-legacy with auto-injected schema: extract ``response`` from
           trailing JSON in stdout (preserves XML tags).
        2. Read from output file if available (legacy, clean capture).
        3. Fall back to ANSI-stripped stdout.

        Args:
            stdout: Standard output from the process.
            stderr: Standard error from the process.
            return_code: Process exit code.
            output_file_path: Per-call output file path (overrides self.output_file).

        Returns:
            Dict suitable for ``TerminalInferencerResponse.from_dict()``.
        """
        output = stdout

        # Non-legacy with auto-injected schema: extract from trailing JSON
        if not self.enable_legacy and not self.output_schema:
            parsed = extract_json_from_output(stdout)
            if parsed and "response" in parsed:
                output = parsed["response"]
            else:
                logger.debug("JSON extraction failed, falling back to ANSI strip")
                output = strip_ansi_codes(stdout).strip()
        else:
            # Legacy mode: try output file first, then ANSI-stripped stdout
            effective_output_file = output_file_path or _current_output_file.get(None) or self.output_file
            if effective_output_file and Path(effective_output_file).exists():
                try:
                    output = Path(effective_output_file).read_text(encoding="utf-8").strip()
                except OSError:
                    logger.warning("Failed to read output file: %s", effective_output_file)
                    output = strip_ansi_codes(stdout).strip()
            else:
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
        # Generate per-call temp output file (legacy mode only)
        auto_output_file = None
        if self.enable_legacy and not self.output_file and self.raw_output_to_file:
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
            cwd=self._resolve_subprocess_cwd(),
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

    def _get_clean_output_for_cache(self) -> Optional[str]:
        """Read clean output from --output-file while it still exists.

        Called from the base class ``ainfer_streaming()`` finally block —
        at this point the subclass finally hasn't run yet, so the
        ``--output-file`` still exists on disk.

        Single read, dual use: the returned content overwrites the stream
        cache (base class), and is also stored in ``_last_clean_output``
        for the response object (used by ``_ainfer()`` and
        ``get_final_output()``).  The subclass finally only needs to
        handle file cleanup — no second read required.
        """
        if not self.enable_legacy:
            return None
        output_path = _current_output_file.get(None) or self.output_file
        if output_path:
            p = Path(output_path)
            if p.exists():
                try:
                    content = p.read_text(encoding="utf-8").strip()
                    if content:
                        self._last_clean_output = content
                        logger.debug(
                            "[%s] _get_clean_output_for_cache: read %d chars from %s",
                            self.__class__.__name__, len(content), p,
                        )
                        return content
                except OSError as e:
                    logger.warning(
                        "[%s] _get_clean_output_for_cache: failed to read %s: %s",
                        self.__class__.__name__, p, e,
                    )
        return None

    def get_final_output(self) -> Optional[str]:
        """Return clean LLM output from --output-file (legacy) or trailing JSON (non-legacy).

        For legacy mode: reads from self._last_clean_output which is populated by
        ainfer_streaming() BEFORE the temp output file is deleted. The contextvar
        _current_output_file is cleared before get_final_output() is called, so
        we use the pre-read instance variable instead.

        For non-legacy mode: extracts from trailing JSON in accumulated stdout.

        Returns:
            Clean final output string, or None if not available.
        """
        if self.enable_legacy:
            # Use pre-read content stored before file deletion in ainfer_streaming()
            content = getattr(self, "_last_clean_output", None)
            if content:
                logger.debug(
                    "[%s] get_final_output: returning %d chars from _last_clean_output",
                    self.__class__.__name__, len(content),
                )
                return content
        else:
            # Non-legacy: extract clean output from trailing JSON in accumulated stdout
            raw = getattr(self, "_last_raw_stdout", None)
            if raw:
                try:
                    parsed = extract_json_from_output(raw)
                    if parsed and "response" in parsed:
                        content = parsed["response"].strip()
                        if content:
                            return content
                except Exception as e:
                    logger.warning(
                        "[%s] get_final_output: failed to extract non-legacy output: %s",
                        self.__class__.__name__, e,
                    )
        return None

    async def _yield_filter(
        self, chunks: AsyncIterator[str], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Filter streaming output — strip ANSI codes from Rich TUI stdout.

        This is an async generator that transforms an entire async iterator
        of chunks, matching the ``StreamingInferencerBase._yield_filter``
        signature.

        Also accumulates raw (post-filter) stdout into ``_last_raw_stdout``
        for non-legacy ``get_final_output()`` (trailing JSON extraction).

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
        self._last_raw_stdout = ""  # reset per-call
        async for line in super()._yield_filter(chunks, **kwargs):
            clean = strip_ansi_codes(line)
            if clean.strip():
                if not self.enable_legacy:
                    # Accumulate for extract_json_from_output() in get_final_output()
                    self._last_raw_stdout += clean
                yield clean

    # =========================================================================
    # Override: ainfer() / infer() — Session-aware inference
    # =========================================================================

    async def ainfer_streaming(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Streaming inference with session management and clean output capture.

        Sets up the legacy --output-file temp path, delegates to the base class
        streaming pipeline, then reads the clean output from the file BEFORE
        deleting it. This populates self._last_clean_output for get_final_output()
        which is called by ConversationalInferencer after stream_token_batches().

        Args:
            inference_input: Input for inference (prompt string or dict).
            inference_config: Optional configuration.
            **kwargs: Additional arguments (session_id, resume, etc.).

        Yields:
            Text chunks as they arrive from the backend (noisy TUI stdout).
        """
        # Set up per-call temp output file (legacy mode only)
        auto_output_file = None
        if self.enable_legacy and not self.output_file and self.raw_output_to_file:
            auto_output_file = tempfile.mktemp(suffix=".md", prefix="rovodev_output_")
            kwargs["output_file"] = auto_output_file
            _current_output_file.set(auto_output_file)
            logger.info("[%s] ainfer_streaming --output-file: %s", self.__class__.__name__, auto_output_file)

        # Handle session context
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

        try:
            # Delegate to base class streaming pipeline
            async for chunk in super().ainfer_streaming(inference_input, inference_config, **kwargs):
                yield chunk
        finally:
            # _last_clean_output is already set by _get_clean_output_for_cache()
            # (called from the base class finally, which runs before this one).
            # We only need to handle cleanup and a defensive fallback for when
            # the cache path was disabled (no cache_file → base finally skipped
            # the _get_clean_output_for_cache call).
            if auto_output_file:
                if not getattr(self, "_last_clean_output", None):
                    try:
                        p = Path(auto_output_file)
                        if p.exists():
                            content = p.read_text(encoding="utf-8").strip()
                            self._last_clean_output = content if content else None
                        else:
                            self._last_clean_output = None
                    except OSError:
                        self._last_clean_output = None
                try:
                    Path(auto_output_file).unlink(missing_ok=True)
                except OSError:
                    pass
                _current_output_file.set(None)
            else:
                self._last_clean_output = None

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Override to return TerminalInferencerResponse with clean output.

        The base ``_ainfer()`` returns noisy TUI stdout as a plain string.
        When ``_last_clean_output`` is available (populated by
        ``ainfer_streaming``'s finally block from the ``--output-file``),
        we wrap the result in a ``TerminalInferencerResponse`` so that
        the logged ``InferenceResponse`` at the base-class level has the
        correct clean ``output`` field — matching the stream cache content.
        """
        raw = await super()._ainfer(inference_input, inference_config, **kwargs)
        clean = getattr(self, "_last_clean_output", None)
        if clean:
            return TerminalInferencerResponse(
                output=clean,
                raw_output=str(raw),
                success=True,
            )
        return raw

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
        # Do NOT set auto_output_file here — ainfer_streaming() already handles
        # creating and reading the temp --output-file, populating _last_clean_output.
        # If we create a second auto_output_file here and inject it via kwargs,
        # construct_command() uses THAT file, but ainfer_streaming() reads its OWN
        # local auto_output_file (a different temp path) → file not found →
        # _last_clean_output = None → get_final_output() returns empty string.

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

        # _ainfer_single → _ainfer → ainfer_streaming. Our _ainfer override
        # returns a TerminalInferencerResponse (clean output + raw noisy)
        # when _last_clean_output is available, so the base-class logging at
        # InferencerBase.__ainfer_single_impl logs the correct clean output.
        result = await self._ainfer_single(
            inference_input, inference_config, **kwargs
        )
        if not isinstance(result, TerminalInferencerResponse):
            clean_output = self.get_final_output() or ""
            result = TerminalInferencerResponse(
                output=clean_output,
                raw_output=str(result),
                success=True,
            )

        # Extract the real session ID from the sessions directory.
        # After a successful run, the most recently modified session in
        # ~/.rovodev/sessions/ is the one we just created/used.
        session_id_found = find_latest_session_id(workspace_path=self.effective_cwd)
        if session_id_found:
            self.active_session_id = session_id_found
            self.log_debug(f"Captured session ID: {session_id_found}", "Async")
            ensure_session_metadata(
                session_id_found, workspace_path=self.effective_cwd
            )

        # Note: temp output file cleanup is handled by ainfer_streaming() itself
        # (it creates, reads, and deletes its own auto_output_file in its finally block).

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
                workspace_path=self.effective_cwd
            )
            if session_id_found:
                self.active_session_id = session_id_found
                self.log_debug(f"Captured session ID: {session_id_found}", "Sync")
                ensure_session_metadata(
                    session_id_found, workspace_path=self.effective_cwd
                )
        return result

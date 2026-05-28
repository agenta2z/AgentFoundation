# pyre-strict
"""``ToolAsInferencer`` — wrap an arbitrary CLI tool as a streaming inferencer.

Inherits from :class:`StreamingInferencerBase` so the existing infrastructure
(per-prompt cache file, ``STREAM_DONE_MARKER`` / ``STREAM_FAIL_MARKER`` writing,
idle-timeout dual-timer, recovery, the agent service bridge's
``WorkspaceStreamTailer`` discovery) all work without new transport. The
subclass body is ~150 LOC of subprocess plumbing:

  - allowlist check on the binary
  - placeholder substitution + shell-metachar scrub via
    :mod:`rich_python_utils.cli_utils.cmd_helpers`
  - ``asyncio.create_subprocess_exec`` with a 16 MB per-line buffer
  - async-merge stdout + stderr line-by-line into the chunk stream
  - per-line ``marker_parsers`` regex callbacks (live event extraction)
  - SIGTERM → 5s grace → SIGKILL on cancel / abandon
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
from collections.abc import Callable, Mapping
from typing import Any, AsyncIterator, Literal, Optional

from attr import attrib, attrs

from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from rich_python_utils.cli_utils.cmd_helpers import render_argv


logger: logging.Logger = logging.getLogger(__name__)


# Default safety allowlist for the leading binary.
# Caller-overridable via the ``allowed_binaries`` field.
_DEFAULT_ALLOWED_BINARIES: frozenset[str] = frozenset(
    {
        "python3",
    }
)


# Per-line buffer for asyncio's subprocess StreamReader (default 64 KB).
# Pushed to 16 MB so a single ultra-long line from a tool (e.g. a JSON
# checkpoint dump) doesn't raise ``LimitOverrunError`` mid-stream.
_MAX_STREAM_LINE_BYTES: int = 16 * 1024 * 1024


@attrs(slots=False, auto_attribs=False, kw_only=True)
class ToolInferencerResponse:
    """The structured value :meth:`ToolAsInferencer._ainfer` returns.

    Behaves like a dict for ad-hoc field access (via ``__getitem__``) AND
    has ``__str__`` returning ``stdout`` so callers that just want the
    text response keep working.

    Attributes:
        stdout: Concatenated stdout content.
        stderr: Concatenated stderr content.
        return_code: Subprocess exit code (``-1`` if the process was
            cancelled or never spawned).
        success: ``success_check(return_code, stdout)`` result.
        parsed: ``output_parser(stdout, stderr, return_code)`` result;
            ``None`` if no parser was configured.
        cache_path: Path to the inferencer cache file (where the base
            class wrote the chunked stream + completion marker).
        tee_log_path: Path to the optional dual-write log file (only set
            when ``tee_log_path`` was passed to the constructor).
    """

    stdout: str = attrib(default="")
    stderr: str = attrib(default="")
    return_code: int = attrib(default=-1)
    success: bool = attrib(default=False)
    parsed: Any = attrib(default=None)
    cache_path: Optional[str] = attrib(default=None)
    tee_log_path: Optional[str] = attrib(default=None)

    def __str__(self) -> str:
        return self.stdout

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@attrs(slots=False, kw_only=True)
class ToolAsInferencer(StreamingInferencerBase):
    """Wrap a CLI invocation as a :class:`StreamingInferencerBase`.

    The argv template is composed of ``command`` + ``args_template``.
    Placeholders (``${KEY}``) in ``args_template`` are substituted from
    ``inference_input`` (when ``dict``) at call time, scrubbed for shell
    metachars, then handed to :func:`asyncio.create_subprocess_exec`.

    Each output line (stdout + stderr async-merged) is yielded as a
    streaming chunk with its trailing ``\\n`` preserved, so the base class's
    cache writer produces a faithful per-line transcript. Optional
    ``marker_parsers`` fire callbacks for regex-matched lines (live event
    extraction without consuming the stream).

    Cancel: drop the ``ainfer`` task. The generator's ``finally`` block
    SIGTERMs the subprocess, waits 5 s, then SIGKILLs.
    """

    # --- Identity / labels ------------------------------------------------
    tool_name: str = attrib()

    # --- Command construction --------------------------------------------
    command: list[str] = attrib(factory=list)
    """Argv prefix; first element is the binary."""

    args_template: list[str] = attrib(factory=list)
    """Appended after ``command``; supports ``${KEY}`` substitution from
    ``inference_input`` (when it is a ``dict``)."""

    env: Optional[dict[str, str]] = attrib(default=None)
    """Process environment overrides. ``None`` inherits the parent's env;
    a dict is merged ON TOP of ``os.environ`` (so callers don't have to
    re-supply the parent process's PATH)."""

    # --- Safety ----------------------------------------------------------
    allowed_binaries: frozenset[str] = attrib(default=_DEFAULT_ALLOWED_BINARIES)
    """Allowlist for the leading binary. Each new binary must be explicitly
    added to prevent arbitrary command execution."""

    # --- Output handling -------------------------------------------------
    output_parser: Optional[Callable[[str, str, int], Any]] = attrib(default=None)
    """Optional ``(stdout, stderr, return_code) -> Any`` callable. Result
    is exposed as :attr:`ToolInferencerResponse.parsed`."""

    success_check: Callable[[int, str], bool] = attrib(
        default=lambda rc, _stdout: rc == 0
    )
    """``(return_code, stdout) -> bool``. Default: ``return_code == 0``."""

    tee_log_path: Optional[str] = attrib(default=None)
    """Optional extra dual-write log path. ``None`` (default) writes only
    to the inferencer cache file (which the base class manages)."""

    marker_parsers: list[tuple[str, Callable[[re.Match[str]], None]]] = attrib(
        factory=list
    )
    """Per-line regex callbacks. Tuples of (pattern, callback). The
    callback runs after the line is yielded, never raises (errors are
    logged at WARNING and the stream continues)."""

    # --- Internal state (not init params) --------------------------------
    _proc: Optional[asyncio.subprocess.Process] = attrib(
        default=None, init=False, repr=False
    )
    _last_response: Optional[ToolInferencerResponse] = attrib(
        default=None, init=False, repr=False
    )
    term_grace_seconds: float = attrib(default=5.0)

    # ---------------------------------------------------------------------
    # Subprocess spawning
    # ---------------------------------------------------------------------

    def _build_argv(self, substitutions: Mapping[str, str]) -> list[str]:
        """Return the fully-resolved argv list for this invocation.

        Substitutes ``${KEY}`` placeholders in ``args_template`` using
        ``substitutions``, scrubs shell metachars, then prepends
        ``command`` (which is left literal — substitutions only happen
        in the user-facing template, not in the trusted prefix).

        Raises:
            ValueError: when ``command`` is empty or its first element is
                not in ``allowed_binaries``.
            CmdHelperError: from :func:`render_argv` on substitution or
                scrub failure.
        """
        if not self.command:
            raise ValueError(
                f"ToolAsInferencer({self.tool_name!r}): command is empty"
            )
        binary = self.command[0]
        # Allow either a bare binary name ("python3") OR an absolute path
        # whose basename matches an allowed binary.
        canonical = os.path.basename(binary) if os.sep in binary else binary
        if canonical not in self.allowed_binaries:
            raise ValueError(
                f"ToolAsInferencer({self.tool_name!r}): binary "
                f"{canonical!r} is not in allowed_binaries "
                f"{sorted(self.allowed_binaries)!r}; explicitly add it "
                "to the allowlist or pick an existing one"
            )
        rendered = render_argv(self.args_template, substitutions)
        return list(self.command) + rendered

    def _resolve_cwd(self) -> Optional[str]:
        """Pick a working directory.

        Delegates to ``InferencerBase.effective_cwd`` (priority:
        ``target_path`` > ``workspace.root`` > ``os.getcwd()``). Returns
        ``None`` only when no target_path is set AND no workspace is
        available — letting the subprocess inherit the parent's cwd in
        that case (preserves the historic "no cwd → inherit" contract for
        ToolAs callers that never set either).
        """
        if self.target_path is not None:
            return self.target_path
        ws = self._workspace
        if ws is not None and getattr(ws, "root", None) is not None:
            return str(ws.root)
        return None

    def _resolve_env(self) -> Optional[dict[str, str]]:
        """Merge ``env`` on top of ``os.environ``; return ``None`` to
        inherit unchanged when no overrides are configured."""
        if self.env is None:
            return None
        merged = dict(os.environ)
        merged.update(self.env)
        return merged

    # ---------------------------------------------------------------------
    # Streaming primitive
    # ---------------------------------------------------------------------

    async def _ainfer_streaming(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Spawn the subprocess, async-merge stdout+stderr, yield lines.

        The base class :meth:`StreamingInferencerBase.ainfer_streaming`
        wraps this generator with idle timeout, cache writing, and
        completion-marker emission. Any exception that propagates out of
        this generator triggers ``STREAM_FAIL_MARKER`` on the cache; clean
        completion triggers ``STREAM_DONE_MARKER``.

        Substitutions for ``args_template`` placeholders flow in via the
        private ``_tool_substitutions`` kwarg (set by :meth:`_ainfer`).
        Anything else passed by the caller flows through unchanged.

        Yields:
            Each subprocess line, including its trailing ``\\n``. Last
            line may lack ``\\n`` if the process emitted partial output
            before exit — yielded as-is.
        """
        substitutions = kwargs.get("_tool_substitutions") or {}
        argv = self._build_argv(substitutions)
        cwd = self._resolve_cwd()
        env = self._resolve_env()

        logger.info(
            "[ToolAsInferencer:%s] spawning %r (cwd=%s)",
            self.tool_name,
            argv,
            cwd,
        )

        # Optional dual-write to a colocated log file. Opened once, kept
        # open for the duration of the spawn so we don't pay an open()
        # per line.
        tee_handle: Optional[Any] = None
        if self.tee_log_path:
            os.makedirs(os.path.dirname(self.tee_log_path), exist_ok=True)
            tee_handle = open(self.tee_log_path, "w", encoding="utf-8")

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            limit=_MAX_STREAM_LINE_BYTES,
        )
        self._proc = proc

        # Async-merge stdout + stderr line-by-line via a queue. Two
        # pumper tasks read each pipe in parallel; the consumer (this
        # generator) drains the queue. A sentinel `(None, None)` marks
        # each pumper's completion; we yield until both have signalled.
        line_queue: asyncio.Queue[tuple[Optional[str], Optional[str]]] = (
            asyncio.Queue()
        )

        async def _pump(
            stream: Optional[asyncio.StreamReader], channel: str
        ) -> None:
            if stream is None:
                await line_queue.put((None, channel))
                return
            try:
                while True:
                    raw = await stream.readline()
                    if not raw:
                        break
                    try:
                        line = raw.decode("utf-8", errors="replace")
                    except Exception as e:  # pragma: no cover — replace mode shouldn't raise
                        logger.warning(
                            "[ToolAsInferencer:%s] decode failed on %s: %s",
                            self.tool_name,
                            channel,
                            e,
                        )
                        continue
                    await line_queue.put((line, channel))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "[ToolAsInferencer:%s] %s pump error: %s",
                    self.tool_name,
                    channel,
                    e,
                )
            finally:
                await line_queue.put((None, channel))

        pumpers = [
            asyncio.create_task(_pump(proc.stdout, "stdout")),
            asyncio.create_task(_pump(proc.stderr, "stderr")),
        ]
        pending_pumpers = len(pumpers)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        return_code = -1
        cancelled = False

        try:
            while pending_pumpers > 0:
                line, channel = await line_queue.get()
                if line is None:
                    pending_pumpers -= 1
                    continue
                if channel == "stdout":
                    stdout_parts.append(line)
                else:
                    stderr_parts.append(line)
                if tee_handle is not None:
                    tee_handle.write(line)
                    tee_handle.flush()
                self._fire_marker_callbacks(line)
                yield line
            return_code = await proc.wait()
        except (GeneratorExit, asyncio.CancelledError):
            cancelled = True
            await self._terminate_proc(proc)
            raise
        finally:
            for task in pumpers:
                if not task.done():
                    task.cancel()
            if proc.returncode is None:
                # Normal exit path didn't run (exception or generator drop)
                # — make sure the process isn't left running.
                await self._terminate_proc(proc)
            if tee_handle is not None:
                try:
                    tee_handle.close()
                except OSError:  # pragma: no cover
                    pass

            stdout_text = "".join(stdout_parts)
            stderr_text = "".join(stderr_parts)
            success = (
                self.success_check(proc.returncode or return_code, stdout_text)
                if not cancelled
                else False
            )
            parsed = (
                self.output_parser(
                    stdout_text, stderr_text, proc.returncode or return_code
                )
                if self.output_parser is not None
                else None
            )
            self._last_response = ToolInferencerResponse(
                stdout=stdout_text,
                stderr=stderr_text,
                return_code=proc.returncode if proc.returncode is not None else return_code,
                success=success,
                parsed=parsed,
                cache_path=None,  # base class owns the cache path
                tee_log_path=self.tee_log_path,
            )
            self._proc = None
            logger.info(
                "[ToolAsInferencer:%s] exit rc=%s success=%s",
                self.tool_name,
                self._last_response.return_code,
                success,
            )

    # ---------------------------------------------------------------------
    # Override _ainfer to return the structured response
    # ---------------------------------------------------------------------

    async def _ainfer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **kwargs: Any,
    ) -> ToolInferencerResponse:
        """Run :meth:`ainfer_streaming`, accumulate, return the structured
        :class:`ToolInferencerResponse` built during streaming.

        Overrides the base which returns the concatenated stream string.
        We need the structured response so workflow steps in
        :func:`make_tool_chain` can extract ``.parsed`` / ``.stdout`` /
        ``.success`` cleanly.
        """
        # Stash the raw inference_input under a private kwarg name so
        # _ainfer_streaming can read it for ${KEY} substitution. Using a
        # leading-underscore key avoids collisions with public kwargs the
        # base class threads through (notably "inference_input" which the
        # retry/recovery wrappers re-pass on every attempt — a plain
        # setdefault here would race that and raise "got multiple values
        # for argument 'inference_input'").
        if isinstance(inference_input, dict):
            kwargs["_tool_substitutions"] = {
                k: str(v) for k, v in inference_input.items()
            }
        else:
            kwargs.setdefault("_tool_substitutions", {})
        async for _chunk in self.ainfer_streaming(
            inference_input, inference_config, **kwargs
        ):
            pass
        if self._last_response is None:  # pragma: no cover — defensive
            return ToolInferencerResponse(
                stdout="", stderr="", return_code=-1, success=False
            )
        return self._last_response

    def _infer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **kwargs: Any,
    ) -> ToolInferencerResponse:
        """Sync bridge over :meth:`_ainfer`.

        ``InferencerBase._infer`` is abstract, so subclasses must provide
        a synchronous entry point even when streaming is the natural mode.
        We delegate to the async path via the standard ``_run_async``
        helper used by every other ``StreamingInferencerBase`` subclass.
        Tools are usually invoked through ``ainfer``; this exists for the
        rare sync caller and to satisfy the ABC.
        """
        from rich_python_utils.common_utils.async_function_helper import (
            _run_async,
        )

        return _run_async(self._ainfer(inference_input, inference_config, **kwargs))

    # ---------------------------------------------------------------------
    # Marker parsing
    # ---------------------------------------------------------------------

    def _fire_marker_callbacks(self, line: str) -> None:
        """Run each ``marker_parsers`` regex against ``line`` and invoke
        the callback for any match. Callback errors are logged at WARNING
        and never propagate."""
        if not self.marker_parsers:
            return
        for pattern, callback in self.marker_parsers:
            try:
                m = re.search(pattern, line)
            except re.error as e:  # pragma: no cover — bad regex from caller
                logger.warning(
                    "[ToolAsInferencer:%s] bad marker regex %r: %s",
                    self.tool_name,
                    pattern,
                    e,
                )
                continue
            if m is None:
                continue
            try:
                callback(m)
            except Exception as e:
                logger.warning(
                    "[ToolAsInferencer:%s] marker callback %r failed: %s",
                    self.tool_name,
                    callback,
                    e,
                )

    # ---------------------------------------------------------------------
    # Cancellation
    # ---------------------------------------------------------------------

    async def _terminate_proc(self, proc: asyncio.subprocess.Process) -> None:
        """SIGTERM → ``_term_grace_seconds`` wait → SIGKILL. Idempotent.

        Safe to call multiple times (already-exited processes are
        no-ops). Never raises.
        """
        if proc.returncode is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
        except (ProcessLookupError, OSError) as e:
            logger.debug(
                "[ToolAsInferencer:%s] SIGTERM raced with exit: %s",
                self.tool_name,
                e,
            )
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.term_grace_seconds)
            return
        except asyncio.TimeoutError:
            logger.warning(
                "[ToolAsInferencer:%s] SIGTERM grace expired after %.1fs; "
                "escalating to SIGKILL",
                self.tool_name,
                self.term_grace_seconds,
            )
        try:
            proc.kill()
        except (ProcessLookupError, OSError):  # pragma: no cover
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:  # pragma: no cover — kernel issue
            logger.error(
                "[ToolAsInferencer:%s] subprocess survived SIGKILL+2s wait",
                self.tool_name,
            )

    async def cancel(self) -> None:
        """Public cancel: terminate the subprocess if one is running.

        Most call sites won't need this — abandoning the ``ainfer`` task
        runs the generator's ``finally`` cleanup automatically. Provided
        for parity so callers that track per-tool cancellation explicitly
        have a uniform API.
        """
        proc = self._proc
        if proc is None:
            return
        await self._terminate_proc(proc)


# =============================================================================
# Tool chain factory (LWI specialization)
# =============================================================================


def make_tool_chain(
    name: str,
    tools: list[Any],  # InferencerBase, but kept loose for heterogeneous chains
    workspace: Optional[Any] = None,
    state_threading: Literal["independent", "stdout_to_next"] = "stdout_to_next",
) -> Any:  # returns LinearWorkflowInferencer; loose for lazy-import
    """Compose ``tools`` into a fixed-length :class:`LinearWorkflowInferencer`.

    LWI already provides per-step :class:`WorkflowStepConfig` execution,
    per-iteration workspace, ``__wf_step_in_progress__`` markers, and
    Workflow checkpoint resume — all we need is a constructor sugar that
    builds the step configs from a flat list of tools.

    Args:
        name: Chain identifier; used as the prefix for each step name
            (``f"{name}_step_0_{tool.tool_name}"`` …) so cache + log
            paths are stable across re-runs.
        tools: Heterogeneous list of inferencers. ``ToolAsInferencer``
            and any ``InferencerBase`` (including ``DualInferencer``) are
            both fine — LWI doesn't care.
        workspace: Passed through to LWI's ``workspace`` (InferencerBase).
        state_threading: ``"stdout_to_next"`` (default) feeds each step's
            ``.parsed`` (or ``.stdout`` if no parser) into the next step's
            ``inference_input``. ``"independent"`` gives every step the
            same original input.

    Returns:
        A :class:`LinearWorkflowInferencer` ready to be ``await``-run.

    Raises:
        ValueError: when ``tools`` is empty.
    """
    if not tools:
        raise ValueError("make_tool_chain: tools must be a non-empty list")

    # Lazy-imported to keep this module import-cheap (and avoid an upfront
    # dependency on the flow_inferencers package for callers that only
    # use ToolAsInferencer directly).
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
        WorkflowStepConfig,
    )

    def _step_name(idx: int, tool: Any) -> str:
        suffix = getattr(tool, "tool_name", None) or tool.__class__.__name__
        return f"{name}_step_{idx}_{suffix}"

    def _make_input_builder(idx: int) -> Callable[[Any], Any]:
        if state_threading == "independent" or idx == 0:
            return lambda state: state.get("input", state.get("original_input"))
        return lambda state: state.get("prev_output", state.get("input"))

    def _output_extractor(result: Any) -> Any:
        # ToolInferencerResponse wins; otherwise the plain string.
        parsed = getattr(result, "parsed", None)
        if parsed is not None:
            return parsed
        stdout = getattr(result, "stdout", None)
        if stdout is not None:
            return stdout
        return result

    step_configs: list[Any] = []
    for idx, tool in enumerate(tools):
        cfg = WorkflowStepConfig(
            name=_step_name(idx, tool),
            inferencer=tool,
            input_builder=_make_input_builder(idx),
            output_extractor=_output_extractor,
            output_state_key="prev_output",
        )
        step_configs.append(cfg)

    return LinearWorkflowInferencer(
        step_configs=step_configs,
        workspace=workspace,
        initial_state_factory=lambda inp: {"input": inp, "prev_output": None},
    )

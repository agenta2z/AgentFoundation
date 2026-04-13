
# pyre-strict

"""Streaming Inferencer Base.

Extracts common streaming, session management, timeout handling, and
sync-to-async bridging logic shared by ClaudeCodeInferencer,
DevmateSDKInferencer, and DevmateCliInferencer.

Subclasses implement ``_ainfer_streaming()`` — the single abstract primitive
that yields raw text chunks from the backend. All other streaming/inference
methods derive from this.

Dual-Timer Architecture:
    ``_ainfer_streaming()`` may yield two kinds of chunks:

    - **Non-empty strings** (text output): resets the standard
      ``idle_timeout_seconds`` timer.
    - **Empty strings** (``""`` — activity sentinels): signal that the
      backend is busy (e.g., executing a tool) but producing no text.
      These reset the ``tool_use_idle_timeout_seconds`` timer (if > 0).

    Empty-string sentinels are **never** yielded downstream, cached, or
    accumulated. They only serve to keep the idle timer alive during
    tool-heavy sessions.
"""

import asyncio
import enum
import hashlib
import logging
import os
import queue
import threading
import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional

from attr import attrib, attrs
import contextvars

from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
    _current_fallback_state,
)
from agent_foundation.common.inferencers.prompt_templates import (
    DEFAULT_RECOVERY_DIR,
    render_recovery_prompt,
)

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_RECOVERY_DIR = DEFAULT_RECOVERY_DIR


class EmptyLineMode(enum.Enum):
    """How to handle empty lines in streaming output.

    PASS_THROUGH: No special treatment — all lines yielded as-is.
    SUPPRESS_LEADING: Drop empty lines before first non-empty content.
        After content starts, all lines (including empty) pass through.
    BUFFER: Drop leading empties + buffer subsequent empties. Only emit
        buffered empties when non-empty content follows. Strips trailing
        empties at end of stream.
    """

    PASS_THROUGH = "pass_through"
    SUPPRESS_LEADING = "suppress_leading"
    BUFFER = "buffer"


class FallbackInferMode(enum.StrEnum):
    """How recovery uses cached partial output from a failed streaming attempt.

    This is distinct from FallbackMode (in rich_python_utils.common_utils.function_helper),
    which controls WHEN the fallback fires. The two enums are orthogonal:
    FallbackMode picks the trigger; FallbackInferMode picks the recovery strategy.

    See also: FallbackMode in rich_python_utils.common_utils.function_helper
    """

    CONTINUE = "continue"    # Feed partial back, ask model to continue. Return partial + continuation.
    REFERENCE = "reference"  # Show partial as context, request fresh response. Return only new response.
    RESTART = "restart"      # Ignore partial entirely. Equivalent to plain retry.



def _read_partial_from_cache(cache_path: str) -> Optional[str]:
    """Read a partial response from a cache file.

    Returns the raw content if non-empty, or None if the file is
    unreadable or contains only whitespace.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = f.read()
        return raw if raw.strip() else None
    except OSError:
        return None


@attrs
class StreamingInferencerBase(InferencerBase):
    """Base class for streaming inferencers with idle timeout, caching, and session management.

    Provides:
    - ``ainfer_streaming()`` — async streaming with per-chunk idle timeout + cache
    - ``infer_streaming()`` — sync bridge via thread + queue
    - ``_ainfer()`` — accumulates from ``ainfer_streaming()``
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - Cache persistence: optional ``cache_folder`` for writing intermediate output

    Subclasses must implement ``_ainfer_streaming(prompt, **kwargs)`` which yields
    raw text chunks from the backend.

    Streaming pipeline:
        ``ainfer_streaming()`` orchestrates a 3-stage pipeline:

        1. ``_ainfer_streaming()`` — yields raw chunks from the backend
        2. Cache — writes each chunk to disk (if ``cache_folder`` is set)
        3. ``_yield_filter()`` — filters what reaches the consumer

    Timeout architecture (two layers):
        ``ainfer() → _ainfer_single() [total_timeout_seconds — caps entire operation]``
          ``→ _ainfer() [accumulates from ainfer_streaming]``
            ``→ ainfer_streaming() [idle_timeout_seconds — gaps between chunks]``
              ``→ _ainfer_streaming() [abstract, subclass implements]``

    Attributes:
        cache_folder: Directory for persisting intermediate streamed content.
            None (default) disables caching.
        idle_timeout_seconds: Maximum seconds to wait for the next chunk before
            considering the stream stalled. 0 disables idle timeout. Default: 600.
        tool_use_idle_timeout_seconds: Maximum seconds to wait for the next chunk
            when the last received chunk was an empty-string activity sentinel
            (indicating tool use or other non-text backend activity). 0 (default)
            means "use idle_timeout_seconds for everything" (backward compatible).
            When > 0, the timer switches to this longer value after receiving an
            empty sentinel, and switches back to idle_timeout_seconds after
            receiving a non-empty text chunk.
        empty_line_mode: How to handle empty lines in streaming output.
            PASS_THROUGH (default): no special treatment.
            SUPPRESS_LEADING: drop empty lines before first non-empty content.
            BUFFER: drop leading empties + buffer subsequent empties, only emit
            when non-empty content follows.
        auto_resume: If True, automatically resume previous session on subsequent
            infer calls. Default: True.
    """

    # Streaming configuration
    cache_folder: Optional[str] = attrib(default=None)
    idle_timeout_seconds: int = attrib(default=600)
    tool_use_idle_timeout_seconds: int = attrib(default=0)
    empty_line_mode: EmptyLineMode = attrib(default=EmptyLineMode.PASS_THROUGH)

    # Session management
    auto_resume: bool = attrib(default=True)

    # Fallback recovery configuration
    fallback_infer_mode: FallbackInferMode = attrib(default=FallbackInferMode.REFERENCE)
    use_default_prompt_templates: bool = attrib(default=True)

    # Recovery template key root (overridable by subclasses to use custom templates)
    fallback_recovery_template_key: str = "recovery"

    # Internal state (not init params)
    _session_id: Optional[str] = attrib(default=None, init=False, repr=False)
    _generator_cleanup_timeout: Optional[float] = attrib(
        default=None, init=False, repr=False
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.use_default_prompt_templates and self.template_manager is not None:
            from rich_python_utils.string_utils.formatting.template_manager import (
                TemplateRootPriority,
            )

            self.template_manager.add_template_root(
                _DEFAULT_RECOVERY_DIR, priority=TemplateRootPriority.LOWEST
            )

    def _render_recovery_prompt(
        self,
        mode: "FallbackInferMode",
        prompt: str,
        partial_output: str,
    ) -> Optional[str]:
        """Render a recovery prompt for the given fallback mode.

        Returns ``None`` if no template system is available (caller should
        fall through to RESTART).
        """
        key = f"{self.fallback_recovery_template_key}/{mode.value}"
        if self.template_manager is not None:
            # Pass active_template_type="" to bypass the type suffix.
            # Without this, a TM with active_template_type="main" would look
            # for "recovery/main/continue" instead of "recovery/continue".
            return self.template_manager(
                key,
                active_template_type="",
                prompt=prompt,
                partial_output=partial_output,
            )
        elif self.use_default_prompt_templates:
            return render_recovery_prompt(
                key, prompt=prompt, partial_output=partial_output
            )
        else:
            return None

    # === Resumable protocol overrides ===

    def _get_result_path(self, result_id, *args, **kwargs) -> str:
        """Checkpoint path: tries ``output_path`` first, falls back to ``cache_folder``."""
        try:
            return super()._get_result_path(result_id, *args, **kwargs)
        except NotImplementedError:
            if self.cache_folder:
                return os.path.join(self.cache_folder, f"{result_id}.pkl")
            raise

    def _find_latest_cache(self, prompt: str) -> Optional[str]:
        """Find the most recent cache file for a given prompt.

        Globs ``cache_folder/{ClassName}/*/stream_*_{prompt_hash}.txt``
        and returns the path with the latest modification time.
        Returns ``None`` if no cache file exists.

        Note: ``self.id`` contains a random UUID and changes across process
        restarts, so the ``{id}_{timestamp}`` directory is wildcarded with
        ``*``.  Only ``ClassName`` and ``prompt_hash`` are deterministic anchors.
        """
        if not self.cache_folder:
            return None
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        pattern = os.path.join(
            self.cache_folder,
            self.__class__.__name__,
            "*",  # {id}_{timestamp} directories — wildcarded (id has random UUID)
            f"stream_*_{prompt_hash}.txt",
        )
        import glob as _glob
        matches = _glob.glob(pattern)
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)

    def _load_cached_or_resume(self, inference_input, inference_config=None, **kwargs):
        """Check for a previous session's cache file and return resume action.

        Returns:
            ``('completed', full_text)`` — previous run completed, skip execution.
            ``('partial', partial_text)`` — previous run failed mid-stream, resume
            via recovery.
            ``None`` — no cache found, execute from scratch.

        This is the streaming inferencer's own resume logic. It is NOT
        ``_load_result`` (which must return raw results for the Workflow
        engine contract).
        """
        if not self.cache_folder or not self.resume_with_saved_results:
            return None
        prompt = self._extract_prompt(inference_input)
        cache_path = self._find_latest_cache(prompt)
        if cache_path is None:
            return None
        content = _read_partial_from_cache(cache_path)
        if content is None:
            return None
        if "--- STREAM COMPLETED SUCCESSFULLY ---" in content:
            idx = content.find("\n--- STREAM COMPLETED SUCCESSFULLY ---")
            return ('completed', content[:idx].rstrip() if idx >= 0 else content)
        if "--- STREAM FAILED:" in content:
            partial = self._sanitize_partial(content, self.fallback_infer_mode)
            return ('partial', partial) if partial else None
        # No marker — crash before finalize (treat as partial)
        stripped = content.rstrip()
        return ('partial', stripped) if stripped else None

    def _try_resume_from_cache(self, inference_input, inference_config=None, **kwargs):
        """Sync resume hook — used by ``_infer_single``.

        For completed caches, returns the cached result directly.
        For partial caches, calls ``self._infer()`` (sync) with augmented prompt.
        """
        cached = self._load_cached_or_resume(inference_input, inference_config, **kwargs)
        if cached is None:
            return None
        status, content = cached
        if status == 'completed':
            logger.info("Resume: previous run completed, returning cached result")
            return content
        elif status == 'partial' and content:
            logger.info("Resume: previous run failed, triggering sync recovery from partial cache")
            augmented = self._render_recovery_prompt(
                self.fallback_infer_mode, self._extract_prompt(inference_input), content
            )
            if augmented is not None:
                return self._infer(augmented, inference_config, **kwargs)
        return None

    async def _atry_resume_from_cache(self, inference_input, inference_config=None, **kwargs):
        """Async resume hook — used by ``_ainfer_single``.

        For completed caches, returns the cached result directly.
        For partial caches, ``await``s ``self._ainfer()`` with augmented prompt.

        No recursion risk: ``_ainfer()`` goes to ``ainfer_streaming()`` →
        ``_ainfer_streaming()``, never re-entering ``_ainfer_single``.
        """
        cached = self._load_cached_or_resume(inference_input, inference_config, **kwargs)
        if cached is None:
            return None
        status, content = cached
        if status == 'completed':
            logger.info("Resume: previous run completed, returning cached result")
            return content
        elif status == 'partial' and content:
            logger.info("Resume: previous run failed, triggering async recovery from partial cache")
            augmented = self._render_recovery_prompt(
                self.fallback_infer_mode, self._extract_prompt(inference_input), content
            )
            if augmented is not None:
                return await self._ainfer(augmented, inference_config, **kwargs)
        return None

    # === Properties ===

    @property
    def active_session_id(self) -> Optional[str]:
        """Get the current active session ID for resumption."""
        return self._session_id

    @active_session_id.setter
    def active_session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    # === Abstract Method ===

    @abstractmethod
    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield raw text chunks from the backend.

        This is the single abstract primitive that each subclass must implement.
        All other streaming/inference methods derive from this.

        Subclasses handle their own:
        - Connection management (lazy connect, per-call client, subprocess)
        - Backend-specific message parsing (SDK message types, event handlers, stdout lines)
        - Session ID extraction (update self._session_id when received)

        Args:
            prompt: The extracted prompt string.
            **kwargs: Backend-specific arguments (session_id, new_session, etc.)

        Yields:
            Text chunks as they arrive from the backend.
        """
        raise NotImplementedError
        # Make this an async generator so type checkers are happy
        yield  # pragma: no cover

    # === Concrete Methods ===

    def _extract_prompt(self, inference_input: Any) -> str:
        """Extract prompt string from various input formats.

        Args:
            inference_input: String, or dict with "prompt" key.

        Returns:
            The prompt string.
        """
        if isinstance(inference_input, dict):
            return inference_input.get("prompt", str(inference_input))
        return str(inference_input)

    def _resolve_timeouts(
        self,
        idle_timeout: float | None,
        tool_use_timeout: float | None,
    ) -> tuple[float | None, float | None]:
        """Resolve effective idle and tool-use timeouts.

        Base implementation returns both values unchanged, enabling the
        dual-timer architecture (switches between idle and tool-use timeouts
        based on empty-string sentinels from ``_ainfer_streaming()``).

        Subclasses that cannot produce empty sentinels (e.g., CLI subprocess
        inferencers) should override to pre-merge, typically returning
        ``(max(idle, tool_use), None)``.
        """
        return idle_timeout, tool_use_timeout

    async def _yield_filter(
        self, chunks: AsyncIterator[str], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Filter cached chunks before yielding to consumers.

        Base implementation applies the empty-line handling policy configured
        by ``empty_line_mode``. Override in subclasses for backend-specific
        filtering (session headers, etc.), calling ``super()._yield_filter()``
        to preserve empty-line handling.

        Per-call override: pass ``empty_line_mode`` in kwargs.

        Args:
            chunks: Async iterator of already-cached chunks.
            **kwargs: Same kwargs passed to ``ainfer_streaming()``.

        Yields:
            Chunks to deliver to the consumer.
        """
        # Resolve mode: per-call override > instance attribute
        mode = kwargs.get("empty_line_mode", self.empty_line_mode)
        if isinstance(mode, str):
            mode = EmptyLineMode(mode)

        if mode == EmptyLineMode.PASS_THROUGH:
            async for chunk in chunks:
                yield chunk
            return

        content_started = False
        pending_empty_lines: list[str] = []

        async for line in chunks:
            stripped = line.strip()
            if not stripped:
                if not content_started:
                    continue  # suppress leading empties (both modes)
                if mode == EmptyLineMode.BUFFER:
                    pending_empty_lines.append(line)
                    continue  # buffer for later
                # SUPPRESS_LEADING + content started → pass through
                yield line
                continue

            # Non-empty content line
            content_started = True
            for empty_line in pending_empty_lines:
                yield empty_line
            pending_empty_lines = []
            yield line
        # End of stream: buffered empties are dropped (BUFFER mode)

    async def ainfer_streaming(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Async streaming inference with idle timeout, caching, and filtering.

        Pipeline: ``_ainfer_streaming() → cache → _yield_filter() → yield``

        Wraps ``_ainfer_streaming()`` to add:
        1. Idle timeout — if no chunk arrives within ``idle_timeout_seconds``, stops.
        2. Cache writing — if ``cache_folder`` is set, chunks are appended to a file.
        3. Yield filtering — ``_yield_filter()`` controls what reaches the consumer.

        Args:
            inference_input: Input for inference (string or dict with "prompt" key).
            inference_config: Optional configuration (unused by base).
            **kwargs: Passed through to ``_ainfer_streaming()`` and ``_yield_filter()``.

        Yields:
            Text chunks as they arrive from the backend.
        """
        prompt = self._extract_prompt(inference_input)

        # Open cache file if configured
        cache_file = None
        if self.cache_folder:
            cache_file = self._open_cache_file(prompt)
            # Publish cache path to _fallback_state (if set by _ainfer_single)
            fs = _current_fallback_state.get(None)
            if fs is not None:
                fs["cache_path"] = cache_file.name

        # Allow per-call idle timeout override via kwargs
        idle_timeout_override = kwargs.pop("idle_timeout_seconds", None)
        if idle_timeout_override is not None:
            idle_timeout = idle_timeout_override if idle_timeout_override > 0 else None
        else:
            idle_timeout = (
                self.idle_timeout_seconds if self.idle_timeout_seconds > 0 else None
            )

        # Resolve tool-use idle timeout
        tool_use_timeout_override = kwargs.pop("tool_use_idle_timeout_seconds", None)
        if tool_use_timeout_override is not None:
            tool_use_timeout = (
                tool_use_timeout_override if tool_use_timeout_override > 0 else None
            )
        else:
            tool_use_timeout = (
                self.tool_use_idle_timeout_seconds
                if self.tool_use_idle_timeout_seconds > 0
                else None
            )

        # Let subclasses adjust (e.g., CLI pre-merge to max)
        idle_timeout, tool_use_timeout = self._resolve_timeouts(
            idle_timeout, tool_use_timeout
        )

        def _fmt_timeout(val: int | float | None) -> str:
            return f"{val}s" if val is not None else "disabled"

        self.log_info(
            f"idle_timeout={_fmt_timeout(idle_timeout)}, "
            f"tool_use_timeout={_fmt_timeout(tool_use_timeout)} "
            f"(overrides: idle={idle_timeout_override}, "
            f"tool_use={tool_use_timeout_override}, "
            f"instance: idle={self.idle_timeout_seconds}, "
            f"tool_use={self.tool_use_idle_timeout_seconds})",
            "StreamingConfig",
        )

        # Dual-timer state: tracks which timeout to use for the next await.
        # Starts with the text idle timeout; switches to tool_use_timeout when
        # an empty sentinel is received, and back to idle_timeout on text.
        current_timeout = idle_timeout
        in_tool_use_mode = False

        success = False
        error = None

        try:
            # Phase 1: Produce chunks + cache raw output
            async def _cached_stream() -> AsyncIterator[str]:
                nonlocal current_timeout, in_tool_use_mode, success
                aiter = self._ainfer_streaming(prompt, **kwargs).__aiter__()
                try:
                    while True:
                        try:
                            if current_timeout is not None:
                                chunk = await asyncio.wait_for(
                                    aiter.__anext__(), timeout=current_timeout
                                )
                            else:
                                chunk = await aiter.__anext__()
                        except StopAsyncIteration:
                            break

                        if chunk == "":
                            # Activity sentinel: backend is busy (tool use,
                            # etc.) but no text output.  Switch to the longer
                            # tool-use timeout.
                            if tool_use_timeout is not None and not in_tool_use_mode:
                                current_timeout = tool_use_timeout
                                in_tool_use_mode = True
                                self.log_info(
                                    f"Switching to tool_use_idle_timeout="
                                    f"{tool_use_timeout}s",
                                    "DualTimer",
                                )
                            # Do NOT yield, cache, or accumulate sentinels.
                            continue

                        # Non-empty text chunk: switch back to standard idle.
                        if in_tool_use_mode:
                            current_timeout = idle_timeout
                            in_tool_use_mode = False
                            self.log_info(
                                f"Switching back to idle_timeout={idle_timeout}s",
                                "DualTimer",
                            )

                        self._append_to_cache(cache_file, chunk)
                        yield chunk

                    success = True
                finally:
                    # Generator cleanup with optional timeout guard.
                    # Subprocess-based inferencers set
                    # _generator_cleanup_timeout to prevent secondary hangs
                    # when aclose() triggers process.wait() on a running
                    # subprocess after idle timeout.
                    if self._generator_cleanup_timeout is not None:
                        try:
                            await asyncio.wait_for(
                                aiter.aclose(),
                                timeout=self._generator_cleanup_timeout,
                            )
                        except (asyncio.TimeoutError, Exception):
                            logger.warning(
                                "[%s] Generator cleanup timed out",
                                self.__class__.__name__,
                            )

            # Phase 2: Filter + empty-line handling
            async for filtered_chunk in self._yield_filter(_cached_stream(), **kwargs):
                yield filtered_chunk

        except asyncio.TimeoutError as e:
            error = e
            timeout_type = "tool_use_idle" if in_tool_use_mode else "text_idle"
            self.log_info(
                f"No new chunk for {current_timeout}s ({timeout_type} timeout)",
                "IdleTimeout",
            )
            raise
        except Exception as e:
            error = e
            raise
        finally:
            self._finalize_cache(cache_file, success, error)

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Async inference by accumulating all streaming chunks.

        Total timeout is handled by ``InferencerBase._ainfer_single()``.
        Subclasses that need ``SDKInferencerResponse`` should override this,
        call ``super()._ainfer()``, and wrap the result.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Passed through to ``ainfer_streaming()``.

        Returns:
            Concatenated response text.
        """
        content_parts: list[str] = []
        async for chunk in self.ainfer_streaming(
            inference_input, inference_config, **kwargs
        ):
            content_parts.append(chunk)
        return "".join(content_parts)

    def infer_streaming(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Iterator[str]:
        """Sync streaming inference via thread + queue bridge.

        Runs ``ainfer_streaming()`` in a background thread and yields chunks
        as they arrive.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Passed through to ``ainfer_streaming()``.

        Yields:
            Text chunks as they arrive from the backend.
        """
        chunk_queue: queue.Queue[str | None] = queue.Queue()
        error_container: list[Exception] = []

        async def _run_async_streaming() -> None:
            try:
                async for chunk in self.ainfer_streaming(
                    inference_input, inference_config, **kwargs
                ):
                    chunk_queue.put(chunk)
            except Exception as e:
                error_container.append(e)
            finally:
                chunk_queue.put(None)

        # Copy the current context so ContextVars (e.g. _current_fallback_state)
        # propagate from the calling thread into the daemon thread.
        ctx = contextvars.copy_context()

        def _run_in_thread() -> None:
            try:
                ctx.run(asyncio.run, _run_async_streaming())
            except Exception as e:
                error_container.append(e)
                chunk_queue.put(None)

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            yield chunk

        thread.join(timeout=5.0)

        if error_container:
            raise error_container[0]

    # === Session Management Methods ===

    def reset_session(self) -> None:
        """Clear session state so the next call starts a fresh session.

        This clears the stored ``_session_id`` without disconnecting the
        underlying transport (if any).  The next ``ainfer()`` / ``infer()``
        call will start a new session instead of resuming the previous one
        (regardless of ``auto_resume``).

        Use this when you want a clean conversational slate but don't need
        to tear down the connection itself (which ``adisconnect()`` handles).
        """
        self._session_id = None

    def new_session(self, prompt: str, **kwargs: Any) -> Any:
        """Start a new session, clearing any previous session.

        Args:
            prompt: The prompt to send.
            **kwargs: Additional arguments passed to ``infer()``.

        Returns:
            Inference result.
        """
        self._session_id = None
        return self.infer(prompt, new_session=True, **kwargs)

    async def anew_session(self, prompt: str, **kwargs: Any) -> Any:
        """Async: start a new session, clearing any previous session.

        Args:
            prompt: The prompt to send.
            **kwargs: Additional arguments passed to ``ainfer()``.

        Returns:
            Inference result.
        """
        await self.adisconnect()
        self._session_id = None
        return await self.ainfer(prompt, new_session=True, **kwargs)

    def resume_session(
        self, prompt: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Resume a previous session.

        Args:
            prompt: The follow-up prompt.
            session_id: Session ID to resume. If None, uses ``active_session_id``.
            **kwargs: Additional arguments passed to ``infer()``.

        Returns:
            Inference result.

        Raises:
            ValueError: If no session_id provided and no active session.
        """
        target = session_id or self._session_id
        if not target:
            raise ValueError(
                "No session_id provided and no active session. "
                "Call infer() first to start a session, or provide session_id."
            )
        return self.infer(prompt, session_id=target, **kwargs)

    async def aresume_session(
        self, prompt: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Async: resume a previous session.

        If a different ``session_id`` is provided than the current active session,
        this will disconnect and reconnect to the specified session.

        Args:
            prompt: The follow-up prompt.
            session_id: Session ID to resume. If None, uses ``active_session_id``.
            **kwargs: Additional arguments passed to ``ainfer()``.

        Returns:
            Inference result.

        Raises:
            ValueError: If no session_id provided and no active session.
        """
        target = session_id or self._session_id
        if not target:
            raise ValueError(
                "No session_id provided and no active session. "
                "Call ainfer() first to start a session, or provide session_id."
            )
        if self._session_id != target:
            await self.adisconnect()
        return await self.ainfer(prompt, session_id=target, **kwargs)

    # === Recovery Helpers ===

    def _sanitize_partial(
        self, partial: Optional[str], mode: FallbackInferMode
    ) -> Optional[str]:
        """Strip the failure marker and (for CONTINUE) truncate to the last newline.

        Always removes the trailing ``--- STREAM FAILED: ... ---`` marker
        written by ``_finalize_cache``.  In CONTINUE mode, additionally
        truncates to the last newline boundary so the model does not
        receive a half-word.

        Returns ``None`` if the result is empty after processing.
        """
        if not partial:
            return None
        text = str(partial)
        # Always strip the trailing --- STREAM FAILED ... --- marker
        marker_idx = text.find("\n--- STREAM FAILED:")
        if marker_idx >= 0:
            text = text[:marker_idx]
        # CONTINUE mode: truncate to last newline so we don't hand the model a half-word
        if mode == FallbackInferMode.CONTINUE:
            last_nl = text.rfind("\n")
            if last_nl > 0:
                text = text[:last_nl]
        return text.strip() or None

    async def _ainfer_recovery(
        self,
        inference_input: Any,
        last_exception: Optional[Exception],
        last_partial_output: Optional[str],
        inference_config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Cache-aware and session-aware async recovery for streaming inferencers.

        Recovery strategy precedence:
        1. Session-based resumption (if ``_session_id`` is set)
        2. Cache-based replay (CONTINUE or REFERENCE mode)
        3. RESTART (delegate to ``_ainfer`` with original prompt)

        Supports ``fallback_infer_mode`` as a runtime override via kwargs.
        """
        # Determine mode (support runtime override)
        mode = kwargs.pop("fallback_infer_mode", self.fallback_infer_mode)
        if isinstance(mode, str):
            mode = FallbackInferMode(mode)

        # 1. Session-based resumption (highest priority)
        if self._session_id:
            logger.info("Recovery via session resume (session_id=%s)", self._session_id)
            try:
                await self.adisconnect()
                return await self._ainfer(
                    inference_input, inference_config,
                    session_id=self._session_id, **kwargs
                )
            except Exception as e:
                logger.warning("Session resume failed: %s. Falling through.", e)
                self._session_id = None

        # 2. Cache-based recovery
        cache_path = None
        fs = _current_fallback_state.get(None)
        if fs is not None:
            cache_path = fs.get("cache_path")

        if cache_path and mode != FallbackInferMode.RESTART:
            partial = _read_partial_from_cache(cache_path)
            partial = self._sanitize_partial(partial, mode)

            if partial:
                augmented = self._render_recovery_prompt(
                    mode, self._extract_prompt(inference_input), partial
                )
                if augmented is not None:
                    if mode == FallbackInferMode.CONTINUE:
                        logger.info("Recovery via cache replay (mode=CONTINUE)")
                        continuation = await self._ainfer(augmented, inference_config, **kwargs)
                        return partial + str(continuation)
                    elif mode == FallbackInferMode.REFERENCE:
                        logger.info("Recovery via cache replay (mode=REFERENCE)")
                        return await self._ainfer(augmented, inference_config, **kwargs)

        # 3. RESTART (fallback)
        logger.info("Recovery via restart")
        return await self._ainfer(inference_input, inference_config, **kwargs)

    def _infer_recovery(
        self,
        inference_input: Any,
        last_exception: Optional[Exception],
        last_partial_output: Optional[str],
        inference_config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Cache-aware sync recovery for streaming inferencers.

        Mirrors ``_ainfer_recovery`` logic for the sync path. The sync
        ``infer_streaming`` thread bridge populates the cache, so cache-based
        recovery works identically.

        Session-based resumption is NOT available on the sync path (session
        management is async-only). Falls through directly to cache-based or
        RESTART recovery.
        """
        mode = kwargs.pop("fallback_infer_mode", self.fallback_infer_mode)
        if isinstance(mode, str):
            mode = FallbackInferMode(mode)

        # Cache-based recovery
        cache_path = None
        fs = _current_fallback_state.get(None)
        if fs is not None:
            cache_path = fs.get("cache_path")

        if cache_path and mode != FallbackInferMode.RESTART:
            partial = _read_partial_from_cache(cache_path)
            partial = self._sanitize_partial(partial, mode)

            if partial:
                augmented = self._render_recovery_prompt(
                    mode, self._extract_prompt(inference_input), partial
                )
                if augmented is not None:
                    if mode == FallbackInferMode.CONTINUE:
                        logger.info("Sync recovery via cache replay (mode=CONTINUE)")
                        continuation = self._infer(augmented, inference_config, **kwargs)
                        return partial + str(continuation)
                    elif mode == FallbackInferMode.REFERENCE:
                        logger.info("Sync recovery via cache replay (mode=REFERENCE)")
                        return self._infer(augmented, inference_config, **kwargs)

        # RESTART (fallback)
        logger.info("Sync recovery via restart")
        return self._infer(inference_input, inference_config, **kwargs)

    # === Cache Helper Methods ===

    def _open_cache_file(self, prompt: str) -> Any:
        """Open a cache file for writing streaming output.

        Creates the directory structure:
          ``cache_folder/{ClassName}/{id}_{timestamp}/stream_{timestamp}_{hash}.txt``

        Args:
            prompt: The prompt string (hashed for filename).

        Returns:
            An open file handle (caller must close via ``_finalize_cache``).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        session_dir = os.path.join(
            self.cache_folder,  # pyre-ignore[6]
            self.__class__.__name__,
            f"{self.id}_{timestamp}",
        )
        os.makedirs(session_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        cache_path = os.path.join(session_dir, f"stream_{unique_id}_{prompt_hash}.txt")
        self.log_debug(f"Cache file: {cache_path}", "CacheOpen")
        return open(cache_path, "w", encoding="utf-8")

    def _append_to_cache(self, cache_file: Any, chunk: str) -> None:
        """Append a chunk to the cache file, flush immediately."""
        if cache_file:
            cache_file.write(chunk)
            cache_file.flush()

    def _finalize_cache(
        self, cache_file: Any, success: bool, error: Exception | None = None
    ) -> None:
        """Write final status marker and close the cache file."""
        if cache_file:
            if success:
                cache_file.write("\n--- STREAM COMPLETED SUCCESSFULLY ---\n")
            else:
                msg = str(error) if error else "unknown"
                cache_file.write(f"\n--- STREAM FAILED: {msg} ---\n")
            cache_file.close()

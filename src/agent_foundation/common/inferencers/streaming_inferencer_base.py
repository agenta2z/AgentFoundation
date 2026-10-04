# pyre-strict

"""Streaming Inferencer Base.

Extracts common streaming, session management, timeout handling, and
sync-to-async bridging logic shared by ClaudeCodeSdkInferencer,
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
import time
import uuid
from abc import abstractmethod
from contextlib import aclosing
from datetime import datetime
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from agent_foundation.common.inferencers.constants.paths import DEFAULT_RECOVERY_DIR
from agent_foundation.common.inferencers.inferencer_base import (
    _current_fallback_state,
    InferencerBase,
)
from agent_foundation.common.inferencers.recovery import render_recovery_prompt
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
    frame_for,
    framed_agen,
    framed_gen,
    invocation_of,
    RuntimeKey,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.async_utils import iterate_async_in_thread

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
    """How recovery re-runs a failed streaming attempt (Option C taxonomy).

    This is distinct from FallbackMode (in rich_python_utils.common_utils.function_helper),
    which controls WHEN the fallback fires. The two enums are orthogonal:
    FallbackMode picks the trigger; FallbackInferMode picks the recovery strategy.

    Two canonical strategies:
      RETRY  — a plain re-run of the original input (for garbage: empty / off-topic /
               narration-only output). It has NO template (recovery/retry.jinja2 was
               retired); re-feeding garbage as a negative example only risks anchoring.
      UPDATE — edit/complete the prior output in place, preserving the good work
               (real-but-incomplete / truncated / format issues; subsumes old CONTINUE).

    UPDATE has a ``recovery/update.jinja2`` handler template; RETRY renders nothing
    (``_render_recovery_prompt`` returns ``None`` for it → plain re-run). The legacy
    names below are kept as backward-compat aliases (same values) so the shared
    streaming-resume path and external callers keep resolving to a valid strategy.

    See also: FallbackMode in rich_python_utils.common_utils.function_helper
    """

    RETRY = "retry"  # Plain re-run of the original input; no template (retired).
    UPDATE = "update"  # Edit/complete the prior output in place, preserving prior work.
    # --- Backward-compat aliases (old names → new values) ---
    RESTART = "retry"  # was a plain re-run → now RETRY.
    RETRY_WITH_REFERENCE = "retry"  # → RETRY.
    REFERENCE = "retry"  # legacy default alias → RETRY.
    CONTINUE = "update"  # old continue-from-cutoff → now UPDATE (edit in place).


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


class _SessionSlot(enum.Enum):
    """``RESET`` marks a branch whose session was explicitly cleared under a host
    ctx, so reads must not fall back to the shared instance backing."""

    RESET = "reset"


class LiveHandleField:
    """A property whose value is session-scoped live state (plan §5.5).

    Reads and writes follow the session policy of
    :meth:`StreamingInferencerBase._session_scoped_get` /
    :meth:`~StreamingInferencerBase._session_scoped_set`: this branch's slot under a
    host ctx, the instance's ``backing`` entry in ``__dict__`` otherwise (it may be
    the field's own name). For a session id and the conversation ids that resume
    with it (``active_session_id``, metamate's and rovochat's conversation ids);
    other private handle state uses ``_tier3_get`` / ``_tier3_set``.
    """

    def __init__(self, name: str, backing: str) -> None:
        self.name = name
        self.backing = backing

    def __get__(self, inst: Any, owner: Any = None) -> Any:
        if inst is None:
            return self
        return inst._session_scoped_get(self.name, backing=self.backing)

    def __set__(self, inst: Any, value: Any) -> None:
        inst._session_scoped_set(self.name, value, backing=self.backing)


@attrs(slots=True)
class StreamStats:
    """What one call's stream counted, for its ``_ainfer`` (B28)."""

    tokens: int = attrib(default=0)
    tool_uses: int = attrib(default=0)
    usage: Any = attrib(default=None)


@attrs
class StreamingInferencerBase(InferencerBase):
    """Streaming + cache-based recovery base. Inherits from InferencerBase.

    One of three orthogonal axes (streaming, terminal-exec, templating).
    Recovery-prompt rendering uses ``template_manager`` if present
    (duck-typed via ``getattr``) and falls back to a module-level
    Jinja-only renderer otherwise.

    Provides:
    - ``ainfer_streaming()`` — async streaming with per-chunk idle timeout + cache
    - ``infer_streaming()`` — sync bridge via an owned loop in a worker thread
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

    _HOST_PURE_CERTIFIED = True

    # Left out of a resume identity: the stream cache, the live observer and the
    # idle timeouts.
    _RESUME_IDENTITY_EXCLUDE = frozenset(
        {
            "cache_folder",
            "stream_observer",
            "idle_timeout_seconds",
            "tool_use_idle_timeout_seconds",
        }
    )

    # Streaming configuration
    cache_folder: Optional[str] = attrib(default=None)
    idle_timeout_seconds: int = attrib(default=600)
    tool_use_idle_timeout_seconds: int = attrib(default=0)
    empty_line_mode: EmptyLineMode = attrib(default=EmptyLineMode.PASS_THROUGH)

    # Session management
    auto_resume: bool = attrib(default=True)
    _FANOUT_DROPPED_ARGS = ("new_session",)
    _FANOUT_SINGLE_CALL_ARGS = ("session_id", "resume")

    # Fallback recovery configuration
    fallback_infer_mode: FallbackInferMode = attrib(default=FallbackInferMode.RETRY)
    use_default_prompt_templates: bool = attrib(default=True)

    # Recovery template key root (overridable by subclasses to use custom templates)
    fallback_recovery_template_key: str = "recovery"

    # Whether streamed chunks differ from the authoritative final output.
    # CLI-based inferencers (e.g., RovoDevCliInferencer) set True — their stdout
    # is noisy TUI output while --output-file has clean LLM text.
    # API-based inferencers (Claude, GPT) leave this False — stream IS the output.
    streams_differ_from_final_output: bool = False

    # Optional streaming observer callback. When set, called with each chunk
    # during _ainfer() accumulation. Used by graph visualization (BTA) to emit
    # per-node streaming content to the UI without bypassing the ainfer() pipeline.
    # Can be sync or async — async coroutines are awaited automatically.
    # Wrapped in try/except — visualization-only, never aborts inference.
    stream_observer: Optional[Callable] = attrib(default=None, repr=False, kw_only=True)

    # A parent hands this call's observer as the ``stream_observer=`` call keyword
    # (``_effective("stream_observer")``); the configured field is the fallback.
    _INVOCATION_KEYWORDS = MappingProxyType(
        {"stream_observer": RuntimeKey("StreamingInferencerBase.stream_observer")}
    )
    # The current attempt's stream counters (tokens, tool uses, usage).
    _STREAM_STATS = RuntimeKey(
        "StreamingInferencerBase.stream_stats", factory=StreamStats
    )

    def _reset_stream_stats(self) -> StreamStats:
        """Start this attempt's counters at zero and return them."""
        stats = StreamStats()
        invocation_of(self).put(self._STREAM_STATS, stats)
        return stats

    def _stream_stats(self) -> StreamStats:
        """This invocation's counters; outside one, a throwaway nothing reads."""
        frame = frame_for(self)
        if frame is None:
            return StreamStats()
        return frame.get_or_create(self._STREAM_STATS)

    def get_final_output(self) -> Optional[str]:
        """Return clean final output if it differs from concatenated stream.

        Subclasses where streamed tokens (stdout) differ from the actual LLM
        output (e.g., CLI inferencers with --output-file) override this to
        return the clean version after streaming completes.

        Must only be called AFTER ainfer_streaming() has completed (i.e., after
        the generator is exhausted and _ainfer_single() is returning).
        Returns None if stream == final output (default for API inferencers).

        Returns:
            Clean final output string, or None if stream == final output.
        """
        return None

    def _get_clean_output_for_cache(self) -> Optional[str]:
        """Read clean output for cache overwrite while source is still accessible.

        Called synchronously from the base class _ainfer_streaming() finally block,
        BEFORE the subclass's own finally block (where per-call cleanup and file
        deletion happen). This timing guarantee means:
        - the call's output-file component is still set (in RovoDevCliInferencer)
        - --output-file still exists on disk (not yet deleted)

        Subclasses that have a clean output source (e.g., --output-file) override
        this to read and return it. The base class returns None (no overwrite).

        Unlike get_final_output() — which is called AFTER subclass finally runs
        (file deleted, contextvar cleared) — this method is specifically designed
        for the cache overwrite use case where timing matters.

        Returns:
            Clean final output string for cache replacement, or None to skip overwrite.
        """
        return None

    # Internal state (not init params)
    _session_id: Optional[str] = attrib(default=None, init=False, repr=False)
    _generator_cleanup_timeout: Optional[float] = attrib(
        default=None, init=False, repr=False
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        tm = getattr(self, "template_manager", None)
        if self.use_default_prompt_templates and tm is not None:
            from rich_python_utils.string_utils.formatting.template_manager import (
                TemplateRootPriority,
            )

            tm.add_template_root(
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
        fall through to a plain re-run).
        """
        # RETRY is a plain re-run of the original input — it has NO template
        # (recovery/retry.jinja2 was retired). Return None so the dispatch and the
        # resume hooks (both null-check the result) fall through to the plain re-run.
        # NOTE: this governs the BASE seam; a subclass that overrides this 3-arg
        # method bypasses the guard (its own responsibility; degrades gracefully).
        if mode == FallbackInferMode.RETRY:
            return None
        key = f"{self.fallback_recovery_template_key}/{mode.value}"
        # Thread has_local_access so update.jinja2 can pick the edit-in-place (local)
        # vs re-emit-inline (no-local) branch, and the judge's <reason> (from the
        # per-call fallback state) so UPDATE is a guided fix. The recovery re-run
        # sends this rendered string straight to the backend, bypassing the feed
        # builder, so both must be injected here (not via the feed).
        hla = bool(getattr(self, "has_local_access", False))
        _fs = _current_fallback_state.get(None)
        reason = _fs.get("guardrail_reason") if _fs else None
        tm = getattr(self, "template_manager", None)
        if tm is not None:
            # Recovery templates use agent_prompt/agent_response (see
            # resources/prompt_templates/recovery/*.jinja2).
            return tm(
                key,
                active_template_type="",
                agent_prompt=prompt,
                agent_response=partial_output,
                has_local_access=hla,
                reason=reason,
            )
        elif self.use_default_prompt_templates:
            return render_recovery_prompt(
                key,
                prompt=prompt,
                partial_output=partial_output,
                has_local_access=hla,
                reason=reason,
            )
        else:
            return None

    def _output_file_ready(self) -> bool:
        """True iff ``resolve_output_path()`` points at an existing, non-empty file.

        Same predicate as ``InferencerBase._finalize_output`` /
        ``_guardrail_output_text``. Pure (no I/O side effects); safe at recovery time.
        """
        try:
            resolved = self.resolve_output_path()
        except Exception:
            return False
        return bool(
            resolved
            and os.path.isabs(resolved)
            and os.path.isfile(resolved)
            and os.path.getsize(resolved) > 0
        )

    def _update_mode_viable(self, raw_partial: Optional[str]) -> bool:
        """Whether an UPDATE recovery has something to build on.

        UPDATE (preserve + complete prior work) needs either a non-empty prior
        partial to show inline, or — for a local agent — a non-empty on-disk
        deliverable to edit. With neither, the caller degrades UPDATE to RETRY.
        """
        if raw_partial and raw_partial.strip():
            return True
        return bool(self.has_local_access and self._output_file_ready())

    def _effective_cache_folder(self) -> Optional[str]:
        """Resolve ``cache_folder`` at call time — explicit value first,
        otherwise derive from the ctx-aware workspace.

        v5 Fix #3 — the modern M7 ctx-dispatch path delivers the active
        workspace via ``ctx.handles["workspace_override"]``, which
        deliberately bypasses the ``_workspace`` setter — so
        ``_configure_for_workspace`` (which is the only thing that
        mutates ``self.cache_folder`` today) never fires on ctx-
        dispatched leaves. Result without this resolver: ``cache_folder``
        stays ``None`` and the streaming-cache gate at
        ``_ainfer_streaming_pipeline`` is always False, so no
        ``stream_*.txt`` files are written for any flow leaf (verified
        empirically: ``find <task_root> -name 'stream_*.txt'`` returned
        zero hits across all rounds).

        Write-pure (no instance mutation) because the SAME streaming
        leaf instance is reused across concurrent gathered branches on
        the modern path (verified: round01 and round02 share
        ``CodexCliInferencer-6efaf963``). A set-once mutation would
        route round02's cache under round01's per-branch workspace.
        ``self._workspace`` is the ctx-aware getter (returns the
        per-call workspace_override when published) so this resolver
        inherits its per-branch safety.

        Returns ``None`` (today's behavior) for leaves with no workspace
        and no explicit ``cache_folder``.
        """
        if self.cache_folder:
            return self.cache_folder
        ws = self._workspace  # ctx-aware getter (per-branch under M7)
        if ws is None:
            return None
        return os.path.join(str(ws.root), "_runtime", "inferencer_cache")

    # === Resumable protocol overrides ===

    def _get_result_path(self, result_id, *args, **kwargs) -> str:
        """Checkpoint path: tries ``output_path`` first, falls back to ``cache_folder``."""
        try:
            return super()._get_result_path(result_id, *args, **kwargs)
        except NotImplementedError:
            cf = self._effective_cache_folder()
            if cf:
                return os.path.join(cf, f"{result_id}.pkl")
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
        cf = self._effective_cache_folder()
        if not cf:
            return None
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        pattern = os.path.join(
            cf,
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
        if not self._effective_cache_folder() or not self.resume_with_saved_results:
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
            return ("completed", content[:idx].rstrip() if idx >= 0 else content)
        if "--- STREAM FAILED:" in content:
            partial = self._sanitize_partial(content, self.fallback_infer_mode)
            return ("partial", partial) if partial else None
        # No marker — crash before finalize (treat as partial)
        stripped = content.rstrip()
        return ("partial", stripped) if stripped else None

    def _try_resume_from_cache(self, inference_input, inference_config=None, **kwargs):
        """Sync resume hook — used by ``_infer_single``.

        For completed caches, returns the cached result directly.
        For partial caches, calls ``self._infer()`` (sync) with augmented prompt.
        """
        cached = self._load_cached_or_resume(
            inference_input, inference_config, **kwargs
        )
        if cached is None:
            return None
        status, content = cached
        if status == "completed":
            logger.info("Resume: previous run completed, returning cached result")
            return content
        elif status == "partial" and content:
            logger.info(
                "Resume: previous run failed, triggering sync recovery from partial cache"
            )
            augmented = self._render_recovery_prompt(
                self.fallback_infer_mode, self._extract_prompt(inference_input), content
            )
            if augmented is not None:
                return self._infer(augmented, inference_config, **kwargs)
        return None

    async def _atry_resume_from_cache(
        self, inference_input, inference_config=None, **kwargs
    ):
        """Async resume hook — used by ``_ainfer_single``.

        For completed caches, returns the cached result directly.
        For partial caches, ``await``s ``self._ainfer()`` with augmented prompt.

        No recursion risk: ``_ainfer()`` goes to ``ainfer_streaming()`` →
        ``_ainfer_streaming()``, never re-entering ``_ainfer_single``.
        """
        cached = self._load_cached_or_resume(
            inference_input, inference_config, **kwargs
        )
        if cached is None:
            return None
        status, content = cached
        if status == "completed":
            logger.info("Resume: previous run completed, returning cached result")
            return content
        elif status == "partial" and content:
            logger.info(
                "Resume: previous run failed, triggering async recovery from partial cache"
            )
            augmented = self._render_recovery_prompt(
                self.fallback_infer_mode, self._extract_prompt(inference_input), content
            )
            if augmented is not None:
                return await self._ainfer(augmented, inference_config, **kwargs)
        return None

    # === Properties ===

    def _session_scoped_get(
        self, name: str, default: Any = None, *, backing: Optional[str] = None
    ) -> Any:
        """Read session-scoped live state ``name`` under the session policy.

        Under a host ctx: this branch's slot in the connection-scoped store, keyed by
        ``(handle scope, path)`` (V8 isolation, V7 continuity across turns of one
        root, B32 isolation of independent roots), so a sibling branch's cold read is
        its own slot, never another branch's value; a slot reset under a host ctx
        reads as ``default``, never as the backing; an unset slot falls back to the
        instance's ``backing`` entry in ``__dict__`` (default ``_<name>``: a value set at setup).

        With no ctx or under a legacy-mint root (a read between calls, by an
        external caller, or inside a bare call): the backing when set; otherwise
        the one live value the branches hold, so a standalone leaf's state written
        under a call's ctx stays visible after the call (only when unambiguous:
        concurrent branches are always read under their own ctx); else
        ``default``.
        """
        attr = backing or f"_{name}"
        ctx = active_run_context()
        if ctx is not None and not ctx.legacy_mint:
            branch = self._get_live_handle_store().peek(ctx.live_branch_key)
            live = None if branch is None else branch.get(name)
            if live is _SessionSlot.RESET:
                return default
            if live is not None:
                return live
            fallback = self.__dict__.get(attr)
            return default if fallback is None else fallback
        value = self.__dict__.get(attr)
        if value is not None:
            return value
        store = self.__dict__.get("_live_handle_store")
        if store is not None:
            live = {
                v
                for v in (h.get(name) for h in list(store._by_path.values()))
                if v is not None and v is not _SessionSlot.RESET
            }
            if len(live) == 1:
                return next(iter(live))
        return default

    def _session_scoped_set(
        self, name: str, value: Any, *, backing: Optional[str] = None
    ) -> None:
        """Write session-scoped live state ``name`` under the session policy.

        Under a host ctx: only this branch's slot (``None`` stores a reset marker),
        never the backing other branches' cold reads fall back to. With no ctx or
        under a legacy-mint root (whose handle store is discarded on bridge exit):
        the instance's ``backing`` entry (default ``_<name>``), so the post-call
        getter still sees it. A reset there (``None``) also clears ``name`` in every
        branch the no-ctx read would surface, so it really resets (B6(b)).
        """
        attr = backing or f"_{name}"
        ctx = active_run_context()
        if ctx is not None and not ctx.legacy_mint:
            self._get_live_handle_store().get_or_create(ctx.live_branch_key).set(
                name, _SessionSlot.RESET if value is None else value
            )
            return
        self.__dict__[attr] = value
        store = self.__dict__.get("_live_handle_store")
        if value is None and store is not None:
            for branch in list(store._by_path.values()):
                if branch.get(name) is not None:
                    branch.set(name, None)

    # The current active session ID for resumption: a per-connection-branch live
    # handle under a host ctx, the instance ``_session_id`` otherwise.
    active_session_id = LiveHandleField("live_session_id", "_session_id")

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

    def ainfer_streaming(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context=None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Public async streaming entrypoint (M2/E2): returns a stream that
        installs the RunContext bridge, then yields the chunks of
        ``_ainfer_streaming_pipeline``. ``run_context=None`` legacy-mints ->
        byte-identical.

        When ``bta_inferencer`` fans this call out, the stream yields the
        fan-out's response for ``inference_input`` (sent verbatim, like the
        backend) as one chunk.

        Subclasses customize streaming by overriding the pipeline; an override
        of this entry only validates or adapts arguments, then calls ``super()``.
        """
        return self._ainfer_streaming_entry(
            inference_input, inference_config, run_context, kwargs
        )

    def _ainfer_streaming_entry(
        self,
        inference_input: Any,
        inference_config: Any,
        run_context: Any,
        kwargs: dict[str, Any],
    ) -> AsyncIterator[str]:
        """The stream's invocation (``framed_agen``): the ctx and frame are bound
        only while the pipeline runs, never across a yield to the consumer.
        ``_init_call_state`` and the fan-out decision run inside the frame, at the
        first resumption."""

        def start() -> AsyncIterator[str]:
            self._pop_invocation_keywords(frame_for(self), kwargs)
            self._init_call_state(inference_input)
            return self._astreaming_source(inference_input, inference_config, kwargs)

        return framed_agen(
            self,
            "ainfer_streaming",
            run_context,
            start,
            default_workspace=getattr(self, "_workspace", None),
        )

    def _astreaming_source(
        self, inference_input: Any, inference_config: Any, kwargs: dict[str, Any]
    ) -> AsyncIterator[str]:
        """The fan-out's one-chunk stream, or the streaming pipeline; decided
        under the entry's context."""
        if self._delegates_execution:
            return self._afanout_stream(inference_input, inference_config, kwargs)
        return self._ainfer_streaming_pipeline(
            inference_input, inference_config, **kwargs
        )

    async def _afanout_stream(
        self, inference_input: Any, inference_config: Any, kwargs: dict[str, Any]
    ) -> AsyncIterator[str]:
        self._ensure_ctx_workspace_logger()
        text, _ = await self._arun_fanout(inference_input, inference_config, kwargs)
        yield text

    async def _ainfer_streaming_pipeline(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Async streaming pipeline: idle timeout, caching, and filtering.

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

        # Open cache file if configured. v5 Fix #3 — resolve at call time
        # so ctx-dispatched leaves (whose workspace arrives via
        # ctx.handles["workspace_override"]) get a streaming cache without
        # mutating self.cache_folder. Pass the resolved value into
        # _open_cache_file so the gate and the open agree.
        cache_file = None
        _cache_folder = self._effective_cache_folder()
        if _cache_folder:
            cache_file = self._open_cache_file(prompt, _cache_folder)
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
                                # v5 Phase 1.5 — structured ToolUsePhase
                                # marker. Makes the long input→response gap
                                # in the JSONL self-explanatory (post-scan
                                # can sum elapsed tool-use phase time vs
                                # idle time per invocation). Env-gated.
                                from agent_foundation.common.inferencers.inferencer_base import (
                                    _is_verbose_correlation,
                                )

                                if _is_verbose_correlation():
                                    _tool_use_start_ts = time.monotonic()
                                    self.log_info(
                                        {
                                            "phase": "start",
                                            "timeout_s": tool_use_timeout,
                                            "ts": _tool_use_start_ts,
                                        },
                                        "ToolUsePhase",
                                    )
                                else:
                                    _tool_use_start_ts = None
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
                            # v5 Phase 1.5 — structured ToolUsePhase end
                            # marker with elapsed duration.
                            from agent_foundation.common.inferencers.inferencer_base import (
                                _is_verbose_correlation,
                            )

                            if _is_verbose_correlation():
                                _elapsed = (
                                    time.monotonic() - _tool_use_start_ts
                                    if _tool_use_start_ts is not None
                                    else None
                                )
                                self.log_info(
                                    {"phase": "end", "elapsed_s": _elapsed},
                                    "ToolUsePhase",
                                )

                        self._append_to_cache(cache_file, chunk)
                        yield chunk

                    success = True
                finally:
                    # Close the transport by ownership, so an early close or a
                    # failure runs its cleanup now, under this call's binding,
                    # rather than whenever its generator is garbage-collected.
                    # A configured _generator_cleanup_timeout bounds a transport
                    # whose aclose() could block (e.g. process.wait() on a
                    # still-running subprocess after an idle timeout).
                    aclose = getattr(aiter, "aclose", None)
                    if aclose is not None and self._generator_cleanup_timeout is None:
                        await aclose()
                    elif aclose is not None:
                        try:
                            await asyncio.wait_for(
                                aclose(),
                                timeout=self._generator_cleanup_timeout,
                            )
                        except (asyncio.TimeoutError, Exception):
                            logger.warning(
                                "[%s] Generator cleanup timed out",
                                self.__class__.__name__,
                            )

            # Phase 2: Filter + empty-line handling
            async with (
                aclosing(_cached_stream()) as cached,
                aclosing(self._yield_filter(cached, **kwargs)) as filtered,
            ):
                async for filtered_chunk in filtered:
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
            # If clean final output differs from the noisy stream, overwrite the
            # cache file with clean content so recovery inferences use correct context.
            # We call _get_clean_output_for_cache() — a synchronous hook that subclasses
            # override to read their clean output source (e.g. --output-file) while it
            # is still accessible (contextvar set, file not yet deleted). This runs
            # inside the base class finally, which is called while the subclass's own
            # finally (where contextvar cleanup happens) hasn't run yet — so the
            # output file is still present and contextvar is still valid.
            if self.streams_differ_from_final_output and cache_file:
                final = self._get_clean_output_for_cache()
                if final:
                    try:
                        cache_path = getattr(cache_file, "name", None)
                        if cache_path:
                            # Close the original handle FIRST to avoid stale-handle
                            # writes after truncation by the new 'w' open below.
                            # _finalize_cache() safely no-ops when cache_file is None.
                            cache_file.close()
                            cache_file = None
                            with open(cache_path, "w", encoding="utf-8") as _cf:
                                _cf.write(final)
                                _cf.write("\n--- STREAM COMPLETED SUCCESSFULLY ---\n")
                            success = True  # marker written; _finalize_cache skips it
                    except OSError as _e:
                        logger.warning(
                            "[%s] Failed to replace cache with clean output: %s",
                            self.__class__.__name__,
                            _e,
                        )
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
            **kwargs: Passed through to ``_ainfer_streaming_pipeline()``.

        Returns:
            Concatenated response text.
        """
        content_parts: list[str] = []
        # Hoist asyncio.iscoroutine out of the per-chunk loop for performance.
        _observer = self._effective("stream_observer")
        _iscoroutine = asyncio.iscoroutine
        async for chunk in self._ainfer_streaming_pipeline(
            inference_input, inference_config, **kwargs
        ):
            content_parts.append(chunk)
            # stream_observer: pipe chunk for live graph visualization.
            # Visualization-only — wrapped in try/except, never aborts inference.
            # We log at DEBUG level so failures can still be diagnosed if needed
            # (silent pass would hide misconfiguration of the observer).
            if _observer is not None:
                try:
                    _result = _observer(chunk)
                    if _iscoroutine(_result):
                        await _result
                except Exception as _exc:
                    logger.debug(
                        "[StreamingInferencerBase] stream_observer failed (visualization only): %s",
                        _exc,
                    )
        return "".join(content_parts)

    def infer_streaming(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context=None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Public sync streaming entrypoint, the twin of ``ainfer_streaming``:
        returns a stream that installs the RunContext bridge, then yields the
        chunks of ``_infer_streaming_pipeline``, or the fan-out's response as
        one chunk. ``run_context=None`` legacy-mints.

        Subclasses customize sync streaming by overriding that pipeline; an
        override of this entry only validates or adapts arguments, then calls
        ``super()``.
        """
        return self._infer_streaming_entry(
            inference_input, inference_config, run_context, kwargs
        )

    def _infer_streaming_entry(
        self,
        inference_input: Any,
        inference_config: Any,
        run_context: Any,
        kwargs: dict[str, Any],
    ) -> Iterator[str]:
        """Sync twin of ``_ainfer_streaming_entry`` (``framed_gen``)."""

        def start() -> Iterator[str]:
            self._pop_invocation_keywords(frame_for(self), kwargs)
            self._init_call_state(inference_input)
            return self._streaming_source(inference_input, inference_config, kwargs)

        return framed_gen(
            self,
            "infer_streaming",
            run_context,
            start,
            default_workspace=getattr(self, "_workspace", None),
        )

    def _streaming_source(
        self, inference_input: Any, inference_config: Any, kwargs: dict[str, Any]
    ) -> Iterator[str]:
        """Sync twin of ``_astreaming_source``."""
        if self._delegates_execution:
            return self._fanout_stream(inference_input, inference_config, kwargs)
        return self._infer_streaming_pipeline(
            inference_input, inference_config, **kwargs
        )

    def _fanout_stream(
        self, inference_input: Any, inference_config: Any, kwargs: dict[str, Any]
    ) -> Iterator[str]:
        self._ensure_ctx_workspace_logger()
        text, _ = self._run_fanout(inference_input, inference_config, kwargs)
        yield text

    def _infer_streaming_pipeline(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Iterator[str]:
        """Sync streaming pipeline: ``_ainfer_streaming_pipeline()`` driven by an
        owned event loop in a worker thread (``iterate_async_in_thread``).

        The thread starts at the first ``next()`` under a copy of the caller's
        context, so the pipeline sees the stream's ctx and frame for the whole
        call. Closing the stream early cancels the pipeline's task and joins the
        thread. Leaves with a native sync transport override this.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Passed through to ``_ainfer_streaming_pipeline()``.

        Returns:
            An iterator of text chunks as they arrive from the backend.
        """
        return iterate_async_in_thread(
            lambda: self._ainfer_streaming_pipeline(
                inference_input, inference_config, **kwargs
            ),
            owner=type(self).__name__,
        )

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
        # M6: route through the ``active_session_id`` compat-property so that under
        # an active context the connection-scoped ``ctx.handles.live_session_id`` is
        # cleared (not just the instance) — byte-identical without a context.
        self.active_session_id = None

    async def areset_conversation(self, *, run_context=None) -> None:
        """Start a fresh vendor conversation on one run-context branch (the
        contract of :meth:`InferencerBase.areset_conversation`): runs
        :meth:`_areset_branch_conversation` under that branch's context."""
        token = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            await self._areset_branch_conversation()
        finally:
            exit_run(token)

    async def _areset_branch_conversation(self) -> None:
        """End the active branch's vendor conversation. Here: forget its session
        (:meth:`reset_session`), which a session-keeping leaf would resume on its
        next call. Leaves whose branch also holds a live conversation (a connected
        client, a server-side chat, a persistent default session) extend this.

        Under a host context only the branch's session is cleared; under a legacy
        root, the instance's ctx-less session, which every bare call shares."""
        self.reset_session()

    async def _pre_retry(self, attempt: int, exception: BaseException) -> None:
        """When this inferencer (or its parent's recursion) is retried,
        reset in-memory session state so the next attempt does not resume
        a session that may have been corrupted by the failed attempt.

        Note on lifetimes: the reset takes effect on **subsequent**
        ``ainfer()`` calls. Within ONE ``ainfer()`` call, kwargs
        (``session_id``, ``resume``) are typically locked by the leaf's
        ``_prepare_call`` (see ``ClaudeCodeCliInferencer`` and similar
        leaves) before the retry loop starts, so inner retry
        attempts use those locked kwargs regardless of what
        ``active_session_id`` is set to here. This hook is load-bearing
        for **cross-layer recursive propagation** — when a parent
        inferencer's retry triggers child ``pre_retry``s (via
        ``InferencerBase.pre_retry``'s recursion through
        ``_iter_child_inferencers``), this clears the child's session_id
        so the parent's NEXT attempt — which will issue a fresh
        ``child.ainfer()`` call — starts from a clean slate.
        """
        self.reset_session()

    def new_session(self, prompt: str, *, run_context=None, **kwargs: Any) -> Any:
        """Start a new session, clearing any previous session.

        Args:
            prompt: The prompt to send.
            run_context: optional RunContext carrier (M3/N-Major2) forwarded to infer().
            **kwargs: Additional arguments passed to ``infer()``.

        Returns:
            Inference result.
        """
        from agent_foundation.common.inferencers.run_context import enter_run, exit_run

        # N-Major2: install the bridge FIRST so the session reset lands on the
        # intended branch's ctx.handles (not stale instance/parent state); the inner
        # infer() reuses this active ctx. Byte-identical without a context.
        _tok = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            self.active_session_id = None
            return self.infer(prompt, new_session=True, **kwargs)
        finally:
            exit_run(_tok)

    async def anew_session(
        self, prompt: str, *, run_context=None, **kwargs: Any
    ) -> Any:
        """Async: start a new session, clearing any previous session.

        Args:
            prompt: The prompt to send.
            run_context: optional RunContext carrier (M3/N-Major2) forwarded to ainfer().
            **kwargs: Additional arguments passed to ``ainfer()``.

        Returns:
            Inference result.
        """
        from agent_foundation.common.inferencers.run_context import enter_run, exit_run

        # N-Major2: bridge FIRST so adisconnect + session reset target the branch's
        # ctx.handles; the inner ainfer() reuses this active ctx.
        _tok = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            await self.adisconnect()
            self.active_session_id = None
            return await self.ainfer(prompt, new_session=True, **kwargs)
        finally:
            exit_run(_tok)

    def resume_session(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        *,
        run_context=None,
        **kwargs: Any,
    ) -> Any:
        """Resume a previous session.

        Args:
            prompt: The follow-up prompt.
            session_id: Session ID to resume. If None, uses ``active_session_id``.
            run_context: optional RunContext carrier (M3/N-Major2) forwarded to infer().
            **kwargs: Additional arguments passed to ``infer()``.

        Returns:
            Inference result.

        Raises:
            ValueError: If no session_id provided and no active session.
        """
        from agent_foundation.common.inferencers.run_context import enter_run, exit_run

        # N-Major2: bridge FIRST so active_session_id resolves the branch's live
        # session (ctx.handles), not stale instance state; inner infer() reuses it.
        _tok = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            target = session_id or self.active_session_id
            if not target:
                raise ValueError(
                    "No session_id provided and no active session. "
                    "Call infer() first to start a session, or provide session_id."
                )
            return self.infer(prompt, session_id=target, **kwargs)
        finally:
            exit_run(_tok)

    async def aresume_session(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        *,
        run_context=None,
        **kwargs: Any,
    ) -> Any:
        """Async: resume a previous session.

        If a different ``session_id`` is provided than the current active session,
        this will disconnect and reconnect to the specified session.

        Args:
            prompt: The follow-up prompt.
            session_id: Session ID to resume. If None, uses ``active_session_id``.
            run_context: optional RunContext carrier (M3/N-Major2) forwarded to ainfer().
            **kwargs: Additional arguments passed to ``ainfer()``.

        Returns:
            Inference result.

        Raises:
            ValueError: If no session_id provided and no active session.
        """
        from agent_foundation.common.inferencers.run_context import enter_run, exit_run

        # N-Major2: bridge FIRST so active_session_id + adisconnect target the
        # branch's live session (ctx.handles); inner ainfer() reuses this ctx.
        _tok = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            target = session_id or self.active_session_id
            if not target:
                raise ValueError(
                    "No session_id provided and no active session. "
                    "Call ainfer() first to start a session, or provide session_id."
                )
            if self.active_session_id != target:
                await self.adisconnect()
            return await self.ainfer(prompt, session_id=target, **kwargs)
        finally:
            exit_run(_tok)

    # === Recovery Helpers ===

    def _sanitize_partial(
        self, partial: Optional[str], mode: FallbackInferMode
    ) -> Optional[str]:
        """Strip the failure marker and (for UPDATE) truncate to the last newline.

        Always removes the trailing ``--- STREAM FAILED: ... ---`` marker
        written by ``_finalize_cache``.  In UPDATE mode, additionally
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
        # UPDATE mode: truncate to last newline so we don't hand the model a half-word
        if mode == FallbackInferMode.UPDATE:
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
        2. Recovery re-run via the mode's handler template — RETRY (fresh, prior
           attempt shown only as a negative-example reference) or UPDATE
           (edit/complete the prior output in place, preserving prior work)
        3. Plain re-run with the original prompt (no template system available)

        Supports ``fallback_infer_mode`` as a runtime override via kwargs.
        """
        # Determine mode: (1) explicit kwarg override, (2) guardrail-recommended
        # handler from last_exception (output_validator returns a handler string
        # → carried in ValueError.args[1]), (3) instance default.
        mode = kwargs.pop("fallback_infer_mode", None)
        if mode is None and last_exception is not None:
            _exc_args = getattr(last_exception, "args", ())
            if len(_exc_args) >= 2 and isinstance(_exc_args[1], str):
                try:
                    mode = FallbackInferMode(_exc_args[1])
                except ValueError:
                    pass
        if mode is None:
            mode = self.fallback_infer_mode
        if isinstance(mode, str):
            mode = FallbackInferMode(mode)

        # 1. Session-based resumption (highest priority)
        # §2.10: read via active_session_id so recovery resumes the BRANCH's live
        # session (ctx.handles), not stale instance state, under a context.
        _recover_sid = self.active_session_id
        if _recover_sid:
            logger.info("Recovery via session resume (session_id=%s)", _recover_sid)
            try:
                await self.adisconnect()
                return await self._ainfer(
                    inference_input, inference_config, session_id=_recover_sid, **kwargs
                )
            except Exception as e:
                logger.warning("Session resume failed: %s. Falling through.", e)
                self.active_session_id = None

        # 2. Recovery re-run. UPDATE renders recovery/update.jinja2 and re-runs ONCE:
        #    the agent edits/completes its prior output in place — a local agent
        #    revises the preserved on-disk deliverable, a no-local agent re-emits it
        #    inline; the cached partial (if any) is the base to build on, and the
        #    judge's <reason> guides the fix. RETRY has NO template (recovery/retry
        #    was retired) — it is a plain re-run of the original input (step 3).
        # Capability gate: UPDATE (edit/complete prior work) is only viable with
        # something to build on — a non-empty partial, or (local agents) a non-empty
        # on-disk deliverable. With neither, degrade to RETRY (which, with no partial,
        # is a plain re-run — see the gate below).
        fs = _current_fallback_state.get(None)
        cache_path = fs.get("cache_path") if fs is not None else None
        raw_partial = _read_partial_from_cache(cache_path) if cache_path else None
        if mode == FallbackInferMode.UPDATE and not self._update_mode_viable(
            raw_partial
        ):
            mode = FallbackInferMode.RETRY
        partial = self._sanitize_partial(raw_partial, mode) or ""

        # Only UPDATE renders a template (edit/complete the prior work in place, or
        # re-emit inline for a no-local agent). RETRY has no template — it is a plain
        # re-run of the original input (step 3 below).
        if mode == FallbackInferMode.UPDATE:
            augmented = self._render_recovery_prompt(
                mode, self._extract_prompt(inference_input), partial
            )
            if augmented is not None:
                logger.info("Recovery via %s", mode.value)
                return await self._ainfer(augmented, inference_config, **kwargs)

        # 3. Fallback: plain re-run with the original prompt.
        logger.info("Recovery via restart (plain re-run)")
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
        management is async-only). Falls through directly to the recovery
        re-run (RETRY/UPDATE) or a plain re-run.
        """
        mode = kwargs.pop("fallback_infer_mode", None)
        if mode is None and last_exception is not None:
            _exc_args = getattr(last_exception, "args", ())
            if len(_exc_args) >= 2 and isinstance(_exc_args[1], str):
                try:
                    mode = FallbackInferMode(_exc_args[1])
                except ValueError:
                    pass
        if mode is None:
            mode = self.fallback_infer_mode
        if isinstance(mode, str):
            mode = FallbackInferMode(mode)

        # Recovery re-run (RETRY or UPDATE) — mirrors _ainfer_recovery (sync path).
        # Capability gate: UPDATE is only viable with prior work to build on (a
        # non-empty partial, or a local agent's non-empty on-disk deliverable); else
        # degrade to RETRY (a plain re-run when there is no partial).
        fs = _current_fallback_state.get(None)
        cache_path = fs.get("cache_path") if fs is not None else None
        raw_partial = _read_partial_from_cache(cache_path) if cache_path else None
        if mode == FallbackInferMode.UPDATE and not self._update_mode_viable(
            raw_partial
        ):
            mode = FallbackInferMode.RETRY
        partial = self._sanitize_partial(raw_partial, mode) or ""

        # Only UPDATE renders a template; RETRY has no template — it is a plain
        # re-run of the original input (below).
        if mode == FallbackInferMode.UPDATE:
            augmented = self._render_recovery_prompt(
                mode, self._extract_prompt(inference_input), partial
            )
            if augmented is not None:
                logger.info("Sync recovery via %s", mode.value)
                return self._infer(augmented, inference_config, **kwargs)

        # Fallback: plain re-run with the original prompt.
        logger.info("Sync recovery via restart (plain re-run)")
        return self._infer(inference_input, inference_config, **kwargs)

    # === Cache Helper Methods ===

    def _open_cache_file(self, prompt: str, cache_folder: Optional[str] = None) -> Any:
        """Open a cache file for writing streaming output.

        Creates the directory structure:
          ``cache_folder/{ClassName}/{id}_{timestamp}/stream_{timestamp}_{hash}.txt``

        Args:
            prompt: The prompt string (hashed for filename).
            cache_folder: Pre-resolved cache folder (Fix #3). When omitted
                falls back to ``self._effective_cache_folder()`` so direct
                callers (e.g. tests) keep working without threading the
                resolved value through. The pipeline caller passes the
                value computed at the gate site to keep both ends in
                lock-step under M7 ctx-dispatch.
        """
        if cache_folder is None:
            cache_folder = self._effective_cache_folder()
        assert cache_folder, "_open_cache_file requires a non-empty cache_folder"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        session_dir = os.path.join(
            cache_folder,
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

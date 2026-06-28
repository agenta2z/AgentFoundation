"""Codex SDK Inferencer.

Wraps the official OpenAI **`openai-codex` Python SDK** (`AsyncCodex` /
`AsyncThread`) as an async-native ``StreamingInferencerBase`` — the SDK
counterpart of ``CodexCliInferencer`` and the Codex analog of
``ClaudeCodeSdkInferencer``.

The ``openai-codex`` SDK drives the local Codex app-server over JSON-RPC and
ships a pinned Codex CLI runtime, so this class needs no subprocess/CLI parsing:

  * Connection: ``client = AsyncCodex(); await client.__aenter__()`` (held open
    across calls, torn down in ``adisconnect``). The live ``AsyncCodex`` client
    and ``AsyncThread`` are stored as Tier-3 per-branch connection handles
    (V8 isolation), exactly like ``ClaudeCodeSdkInferencer`` holds its
    ``ClaudeSDKClient``.
  * Session: a Codex **thread** is the conversation. ``thread.id`` is the
    session id. Resume is a method — ``client.thread_resume(thread_id)`` — not
    a flag. Multi-turn on one connection is consecutive ``thread.turn(prompt)``.
  * Streaming: ``handle = await thread.turn(prompt)`` then
    ``async for note in handle.stream()`` yields JSON-RPC ``Notification``
    objects; ``note.method == "item/agentMessage/delta"`` carries
    ``note.payload.delta`` (token-level text).

Auth: the SDK reuses the local Codex login (``codex login``); this class does
not manage auth. The SDK is a soft dependency — the module imports cleanly
without it and raises a clear error only when a call needs it.
"""

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from attr import attrib, attrs

from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
    SDKInferencerResponse,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    EmptyLineMode,
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)

logger = logging.getLogger(__name__)

# Accepted ``sandbox_mode`` strings -> openai_codex.Sandbox members.
_SANDBOX_ALIASES = {
    "read-only": "read_only",
    "read_only": "read_only",
    "workspace-write": "workspace_write",
    "workspace_write": "workspace_write",
    "full-access": "full_access",
    "full_access": "full_access",
    "danger-full-access": "full_access",
}

# Accepted ``approval_mode`` strings -> openai_codex.ApprovalMode members.
_APPROVAL_ALIASES = {
    "auto_review": "auto_review",
    "auto-review": "auto_review",
    "deny_all": "deny_all",
    "deny-all": "deny_all",
    "never": "deny_all",
}


@attrs(slots=False)
class CodexSdkInferencer(StreamingInferencerBase, TemplatedInferencerBase):
    """OpenAI Codex SDK (``openai-codex``) as an async-native streaming inferencer.

    Mirrors ``ClaudeCodeSdkInferencer``: a persistent SDK connection held in
    Tier-3 handles, ``aconnect``/``adisconnect`` lifecycle, an ``_ainfer_streaming``
    primitive that streams text deltas, and an ``_ainfer`` override that can
    return a structured ``SDKInferencerResponse``.

    All RunContext/Tier-3 session state, caching, retry, recovery, templating,
    and the sync bridge are inherited from the base chain.
    """

    has_local_access: bool = attrib(default=True)

    # Streaming / timeout knobs (override base defaults).
    idle_timeout_seconds: int = attrib(default=1800)
    tool_use_idle_timeout_seconds: int = attrib(default=7200)
    empty_line_mode: EmptyLineMode = attrib(default=EmptyLineMode.SUPPRESS_LEADING)

    # Codex SDK configuration.
    # ``None`` -> the model configured for your Codex login; else passed to thread_start.
    model_name: Optional[str] = attrib(default=None)
    # ``-s`` sandbox policy: read-only | workspace-write | full-access.
    sandbox_mode: Optional[str] = attrib(default="workspace-write")
    # Codex ApprovalMode: auto_review (default) | deny_all. ``None`` -> SDK default.
    approval_mode: Optional[str] = attrib(default=None)
    # Codex thread instructions (system-prompt analogs).
    developer_instructions: Optional[str] = attrib(default=None)
    base_instructions: Optional[str] = attrib(default=None)
    # Reasoning effort for each turn (e.g. "low"/"medium"/"high"); None -> default.
    reasoning_effort: Optional[str] = attrib(default=None)
    # Extra env merged into the SDK runtime environment.
    sdk_env: Dict[str, str] = attrib(factory=dict)
    # kwargs forwarded verbatim to ``openai_codex.CodexConfig(**...)`` when
    # constructing the SDK client (e.g. ``codex_bin``, ``launch_args_override``,
    # ``config_overrides``, ``env``). ``None`` -> default ``AsyncCodex()`` (the
    # bundled codex binary). Use this where the bundled binary can't run/auth --
    # e.g. point the SDK at a working codex app-server:
    # ``launch_args_override=("/path/to/codex", "--dangerously-disable-osx-sandbox",
    # "app-server", "--listen", "stdio://")``.
    codex_config_kwargs: Optional[Dict[str, Any]] = attrib(default=None)

    # Per-run internal state (not per-branch; plain attribs).
    _connect_lock: Any = attrib(default=None, init=False, repr=False)
    _last_tool_use_count: int = attrib(default=0, init=False, repr=False)
    _last_usage: Any = attrib(default=None, init=False, repr=False)

    # === Tier-3 connection handles (per-branch, V8-isolated; NOT attribs) ===

    @property
    def _client(self):
        return self._tier3_get("client", None)

    @_client.setter
    def _client(self, value):
        self._tier3_set("client", value)

    @property
    def _thread(self):
        return self._tier3_get("thread", None)

    @_thread.setter
    def _thread(self, value):
        self._tier3_set("thread", value)

    @property
    def _connected_loop(self):
        return self._tier3_get("connected_loop", None)

    @_connected_loop.setter
    def _connected_loop(self, value):
        self._tier3_set("connected_loop", value)

    def __attrs_post_init__(self) -> None:
        if self.target_path is None:
            self.target_path = os.getcwd()
        if self.sandbox_mode is not None and self.sandbox_mode not in _SANDBOX_ALIASES:
            logger.warning(
                "[%s] Unknown sandbox_mode=%r; expected one of %s. Falling back to "
                "workspace-write.",
                self.__class__.__name__,
                self.sandbox_mode,
                sorted(set(_SANDBOX_ALIASES)),
            )
        super().__attrs_post_init__()

    # === SDK option resolution ===

    def _resolve_sandbox(self):
        if self.sandbox_mode is None:
            return None
        from openai_codex import Sandbox

        member = _SANDBOX_ALIASES.get(self.sandbox_mode, "workspace_write")
        return getattr(Sandbox, member)

    def _resolve_approval(self):
        if self.approval_mode is None:
            return None
        from openai_codex import ApprovalMode

        member = _APPROVAL_ALIASES.get(self.approval_mode)
        return getattr(ApprovalMode, member) if member else None

    def _thread_kwargs(self) -> Dict[str, Any]:
        """Common kwargs accepted by both ``thread_start`` and ``thread_resume``."""
        kwargs: Dict[str, Any] = {"cwd": str(self.effective_cwd)}
        sandbox = self._resolve_sandbox()
        if sandbox is not None:
            kwargs["sandbox"] = sandbox
        approval = self._resolve_approval()
        if approval is not None:
            kwargs["approval_mode"] = approval
        model = self.model_name or self.model_id
        if model:
            kwargs["model"] = model
        if self.developer_instructions:
            kwargs["developer_instructions"] = self.developer_instructions
        if self.base_instructions:
            kwargs["base_instructions"] = self.base_instructions
        return kwargs

    # === Connection lifecycle ===

    async def aconnect(self, session_id: Optional[str] = None, **kwargs: Any) -> None:
        """Open the Codex SDK client and start (or resume) the conversation thread.

        Idempotent: returns early if already connected on this branch.
        """
        if self._client is not None:
            return

        try:
            from openai_codex import AsyncCodex
        except ImportError as e:  # pragma: no cover - soft dependency
            raise RuntimeError(
                f"openai-codex SDK not available: {e}. Install it with "
                "`pip install openai-codex`."
            ) from e

        # Apply extra env for the SDK runtime (auth is reused from `codex login`).
        for k, v in (self.sdk_env or {}).items():
            os.environ.setdefault(k, v)

        if self.codex_config_kwargs:
            from openai_codex import CodexConfig

            client = AsyncCodex(CodexConfig(**self.codex_config_kwargs))
        else:
            client = AsyncCodex()
        await client.__aenter__()

        thread_kwargs = self._thread_kwargs()
        try:
            if session_id:
                thread = await client.thread_resume(session_id, **thread_kwargs)
            else:
                thread = await client.thread_start(**thread_kwargs)
        except Exception:
            # Don't leak the client if thread setup fails.
            try:
                await client.close()
            except Exception:
                pass
            raise

        self._client = client
        self._thread = thread
        self._connected_loop = asyncio.get_running_loop()
        if getattr(thread, "id", None):
            # Property -> per-branch store (correctness under fan-out); backing ->
            # reset authority (so reset_session()/new_session start fresh).
            self.active_session_id = thread.id
            self._session_id = thread.id

    async def adisconnect(self) -> None:
        """Close EVERY per-branch Codex client (drain all stored branches + backing).

        Runs at the ``__aexit__``/host-cleanup boundary where no RunContext is
        active, so it must drain by stored path — not just the active branch —
        to reclaim every connection a context opened during ``_ainfer`` (V7/V8).
        """
        for h in self._iter_live_handle_sets():
            client = h.get("client")
            if client is not None:
                try:
                    await client.close()
                except Exception as e:  # pragma: no cover - best-effort teardown
                    logger.debug("[%s] codex client.close() failed: %s",
                                 self.__class__.__name__, e)
            h.set("client", None)
            h.set("thread", None)
            h.set("connected_loop", None)

    # === Streaming primitive ===

    async def _ainfer_streaming(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream text from a Codex thread turn.

        Yields ``item/agentMessage/delta`` text; every other notification yields
        an empty activity sentinel so the dual idle timer extends to
        ``tool_use_idle_timeout`` while Codex reasons / runs tools. Captures the
        thread id (session) and token usage along the way.
        """
        # Thread-safe lazy connect (resume the saved session when appropriate).
        if self._client is None:
            if self._connect_lock is None:
                self._connect_lock = asyncio.Lock()
            async with self._connect_lock:
                if self._client is None:
                    sid = kwargs.get("session_id")
                    if sid is None and self.auto_resume and self._session_id:
                        sid = self._session_id
                    await self.aconnect(session_id=sid)

        thread = self._thread
        handle = await thread.turn(prompt)

        async for note in handle.stream():
            method = getattr(note, "method", None)
            payload = getattr(note, "payload", None)

            if method == "item/agentMessage/delta":
                delta = getattr(payload, "delta", None)
                yield delta if delta else ""
            elif method == "turn/started":
                tid = getattr(payload, "thread_id", None)
                if tid:
                    self._session_id = tid
                yield ""
            elif method == "item/completed":
                item = getattr(payload, "item", None)
                root = getattr(item, "root", item)
                itype = getattr(root, "type", None)
                # Count tool-ish items (command runs, file changes, web searches),
                # not the assistant message / user echo / reasoning.
                if itype not in (None, "agentMessage", "userMessage", "reasoning"):
                    self._last_tool_use_count += 1
                yield ""
            elif method in ("turn/completed", "thread/tokenUsage/updated"):
                usage = getattr(payload, "token_usage", None)
                if usage is None:
                    turn = getattr(payload, "turn", None)
                    usage = getattr(turn, "usage", None)
                if usage is not None:
                    self._last_usage = usage
                yield ""
            else:
                # item/started, reasoning deltas, plan updates, etc.
                yield ""

    # === Structured-response + sync overrides ===

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Override to support ``SDKInferencerResponse`` + tool-use counting."""
        self._last_tool_use_count = 0
        self._last_usage = None
        response_text = await super()._ainfer(
            inference_input, inference_config, **kwargs
        )
        if kwargs.get("return_sdk_response", False):
            return SDKInferencerResponse(
                content=response_text,
                session_id=self.active_session_id,
                tool_uses=self._last_tool_use_count,
                tokens_received=self._extract_output_tokens(self._last_usage),
                raw_response=self._last_usage,
            )
        return response_text

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync bridge with stale-loop / cross-loop guards.

        The ``AsyncCodex`` client is loop-bound, so a sync call that runs on a
        fresh event loop must not reuse a client bound to a closed/other loop.
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        # Stale-loop: a client bound to a now-closed loop is unusable -> drop it
        # so the next call reconnects on the live loop.
        loop = self._connected_loop
        if loop is not None and getattr(loop, "is_closed", lambda: False)():
            self._client = None
            self._thread = None
            self._connected_loop = None

        # Cross-loop: refuse to reuse a client from a different running loop.
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if (
            current is not None
            and self._client is not None
            and self._connected_loop is not None
            and current is not self._connected_loop
        ):
            raise RuntimeError(
                "CodexSdkInferencer client is bound to a different event loop. "
                "Use `await inferencer.ainfer(...)` from async code instead of the "
                "sync `infer(...)`."
            )

        async def _run_and_close():
            try:
                return await self._ainfer(
                    inference_input, inference_config, **kwargs
                )
            finally:
                # The sync bridge runs on a throwaway event loop, so the per-call
                # connection can't be reused — close it to avoid leaking app-server
                # connections. Multi-turn continuity is preserved across sync calls
                # via auto_resume + the persisted session id (adisconnect clears the
                # live handles but not the session backing).
                await self.adisconnect()

        return _run_async(_run_and_close())

    # === Helpers ===

    @staticmethod
    def _extract_output_tokens(usage: Any) -> int:
        """Best-effort output-token count from a Codex usage object."""
        if usage is None:
            return 0
        for attr in ("output_tokens", "total_tokens"):
            val = getattr(usage, attr, None)
            if isinstance(val, int):
                return val
        last = getattr(usage, "last", None)
        if last is not None:
            val = getattr(last, "output_tokens", None)
            if isinstance(val, int):
                return val
        return 0

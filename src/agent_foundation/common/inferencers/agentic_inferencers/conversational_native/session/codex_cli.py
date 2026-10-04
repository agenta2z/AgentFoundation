"""Codex backend over the ``codex exec`` CLI (one process per turn).

Same contract as the other per-turn CLI backends. Codex speaks streamable-HTTP
MCP natively (``-c mcp_servers.af.url`` + ``-c mcp_servers.af.bearer_token_env_var``),
so it reuses the shared ``LocalMcpHttpServer`` (the transport claude_cli proved).
L1 is appended inline as ``-c developer_instructions=<json.dumps(text)>`` (a JSON
string is a valid TOML basic string, so escaping is correct); the bearer token
rides an env var, never argv. L2 rides the user turn as a labelled ``<af_context>``
envelope; the turn ends on the ``AF_END_TURN`` directive the bridge puts at the
head of a widget/async tool result.

The session id is recorded from the ``thread.started`` event (codex has no
pinned ``--session-id``); later turns resume via ``codex exec resume <id>``.
Reads ``--json`` (JSONL events) and maps them to VendorEvents. All of
caller_tools, L1 and resume are verified by scripts/native_spikes/s15 and a
real two-turn memory resume.

AF calls are attributed from their MCP requests. Codex reaches MCP tools only
from a code-mode ``exec`` script, and every ``tools/call`` carries
``_meta.callId`` (one per call) and ``_meta.itemId`` (the model output item —
the ``exec`` call — that made it). Each AF call is announced, before it runs,
as an AF message of that item, merged into the turn's events: the questions
one script asks form one compound widget, while a question from a later model
response (another item) is refused. ``--json`` cannot tell the two apart: it
reports neither the ``exec`` item nor response boundaries, only each MCP call
once it finished (codex-cli 0.159.3, scripts/native_spikes/s18).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import math
import os
import uuid
from typing import Any, AsyncIterator, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    BridgeResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    SessionStarted,
    TextDelta,
    TurnEnd,
    VendorError,
    VendorEvent,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BackendCapabilities,
    BridgeToolSpec,
    CallerTools,
    Evidence,
    L2Channel,
    NativeBackendSpec,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.cli_runner import (
    CliProcess,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.common import (
    CODEX_BINARY,
    find_codex_binary,
    resolve_model_tag,
)

logger: logging.Logger = logging.getLogger(__name__)

_TOKEN_ENV = "AF_MCP_TOKEN"
_MISSING_SESSION = "no rollout found for thread id"
_AF_FAILED = "required MCP servers failed to initialize"
_EXITED = object()  # end of the process's events


def _toml_value(value: Any) -> str:
    # A JSON scalar/array literal is valid TOML for strings, numbers, booleans
    # and arrays of them (the only shapes MCP server configs use).
    return json.dumps(value)


def _call_ids() -> tuple[str, Optional[str]]:
    """``(call id, output item id)`` of the MCP ``tools/call`` being served,
    from its ``_meta``; a call without a ``callId`` gets a fresh id, one
    without an ``itemId`` has no output item."""
    from mcp.server.lowlevel.server import request_ctx

    try:
        meta = request_ctx.get().meta
    except LookupError:
        meta = None
    extra = (getattr(meta, "model_extra", None) or {}) if meta is not None else {}
    call_id, item_id = extra.get("callId"), extra.get("itemId")
    return (
        str(call_id) if call_id else f"af-{uuid.uuid4().hex}",
        str(item_id) if item_id else None,
    )


class CodexCliBackend:
    capabilities = BackendCapabilities(
        kind="codex_cli",
        caller_tools=CallerTools.HTTP,
        l2_channels=(L2Channel.ENVELOPE,),
        pinned_session_id=False,  # id recorded from thread.started, not pinned
        exact_fork=False,  # `exec fork` clones a whole session, not up-to-message
        turn_stop_hook=False,  # directive-only (AF_END_TURN)
        subagent_attribution=False,
        compaction_signal=False,
        persistent_process=False,
        slash_passthrough=(),
        environments=("inherit", "hermetic"),
        evidence={  # all verified by scripts/native_spikes/s15 + a real 2-turn resume
            "caller_tools": Evidence.VERIFIED,
            "l1": Evidence.VERIFIED,
            "resume": Evidence.VERIFIED,
            # --json reports no MCP status; `required = true` makes codex
            # refuse to start the thread when `af` fails to initialize.
            "af_health": Evidence.VERIFIED,
            # `mcp_servers.af.tool_timeout_sec` is honoured (5 s timed a 70 s
            # call out; 120 s let it finish); unknown keys are silently ignored.
            "mcp_tool_timeout": Evidence.VERIFIED,
            # --ignore-user-config --ignore-rules authenticate on fresh and
            # resumed threads, keep the AF tool and L1, and start no user
            # config.toml MCP server; the user AGENTS.md still loads (S15).
            "hermetic": Evidence.VERIFIED,
        },
        relies_on=("caller_tools", "l1", "resume", "af_health", "mcp_tool_timeout"),
        # s15 ran on 0.159.1; HTTP MCP, health and timeouts re-run on 0.159.3.
        tested_versions={"codex-cli": "0.159.3"},
        l1_route=(
            "Codex's developer instructions (`-c developer_instructions=…`), "
            "sent with every `codex exec` turn"
        ),
    )

    def __init__(
        self, spec: NativeBackendSpec, *, runtime_manager: Any = None, **_: Any
    ) -> None:
        self.capabilities.require_spec(spec)
        self._spec = spec
        self._runtime = runtime_manager
        self._session_id = ""
        self._l1_text = ""
        self._mcp_url: Optional[str] = None
        self._mcp_token: Optional[str] = None
        self._mcp_server: Any = None
        self._hooks: Any = None
        self._turn_active = False
        self._started_emitted = False
        self._proc: Optional[CliProcess] = None
        # The running turn's events, where AF calls are announced.
        self._events: Optional[asyncio.Queue] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    async def open(self, request: SessionOpenRequest) -> None:
        self._l1_text = request.l1_text
        self._session_id = request.session_id or ""
        self._hooks = request.hooks
        if request.resume:
            self._session_id = request.session_id
            self._started_emitted = True  # a resumed process uses `exec resume`
        if request.tools and self._runtime is not None:
            server = await self._runtime.ensure_http_server()
            url, token, _headers = await server.register(
                [self._announced(spec) for spec in request.tools],
                turn_active=lambda: self._turn_active,
            )
            self._mcp_server = server
            self._mcp_url = url
            self._mcp_token = token

    def _announced(self, spec: BridgeToolSpec) -> BridgeToolSpec:
        """``spec`` with a handler that announces each call before it runs:
        to the hooks as the AF tool use it runs next, and to the turn as an
        AF message of its output item."""

        async def handler(args: dict[str, Any]) -> Any:
            call_id, item_id = _call_ids()
            hooks = self._hooks
            if hooks is not None:
                reason = await hooks.before_af_tool(spec.name, call_id, None)
                if reason:
                    return BridgeResult(reason, True)
            # A call without an output item is a message of its own, so no
            # later question joins its widget.
            self._emit(
                MessageEnd(
                    message_id=item_id or call_id,
                    text="",
                    tool_use_ids=(call_id,),
                    af_tool_use_ids=(call_id,),
                )
            )
            try:
                return await spec.handler(args)
            finally:
                if hooks is not None:
                    await hooks.after_af_tool(spec.name, call_id)

        return dataclasses.replace(spec, handler=handler)

    def _emit(self, event: VendorEvent) -> None:
        if self._events is not None:
            self._events.put_nowait(event)

    async def run_turn(self, request: TurnRequest) -> AsyncIterator[VendorEvent]:
        # The local MCP server refuses this session's token outside this window.
        self._turn_active = True
        try:
            async with contextlib.aclosing(self._turn(request)) as events:
                async for event in events:
                    yield event
        finally:
            self._turn_active = False

    async def _turn(self, request: TurnRequest) -> AsyncIterator[VendorEvent]:
        argv = self._argv(request)
        env = {_TOKEN_ENV: self._mcp_token} if self._mcp_token else None
        self._proc = CliProcess(
            argv, cwd=self._spec.cwd or os.getcwd(), env=env, stdin_devnull=True
        )
        try:
            await self._proc.start()
        except Exception as exc:
            yield VendorError(message=f"spawn failed: {exc}", submitted=False)
            return
        saw_result = False
        async with contextlib.aclosing(self._merged_events(self._proc)) as events:
            async for event in events:
                if isinstance(event, TurnEnd):
                    saw_result = True
                yield event
        rc = await self._proc.wait()
        if not saw_result:
            stderr = self._proc.stderr_text
            if self._started_emitted and _MISSING_SESSION in stderr:
                yield VendorError(
                    message="codex has no rollout for the thread to resume",
                    submitted=False,
                    session_missing=True,
                )
                return
            if _AF_FAILED in stderr:
                yield VendorError(message=self._af_failed(stderr), submitted=False)
                return
            yield VendorError(message=f"codex exited rc={rc}: {stderr[-400:]}")

    async def _merged_events(self, proc: CliProcess) -> AsyncIterator[VendorEvent]:
        """The process's events with the AF call announcements merged in, in
        the order they happen."""
        queue: asyncio.Queue = asyncio.Queue()
        self._events = queue
        pump = asyncio.ensure_future(self._pump(proc, queue))
        try:
            while (event := await queue.get()) is not _EXITED:
                yield event
            await pump  # surfaces a failure to read the process
        finally:
            self._events = None
            if not pump.done():
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump

    async def _pump(self, proc: CliProcess, queue: asyncio.Queue) -> None:
        try:
            async with contextlib.aclosing(proc.json_lines()) as lines:
                async for obj in lines:
                    for event in self._map(obj):
                        queue.put_nowait(event)
        finally:
            queue.put_nowait(_EXITED)

    @staticmethod
    def _af_failed(stderr: str) -> str:
        detail = stderr[stderr.index(_AF_FAILED) :].splitlines()[0][:300]
        return (
            "codex could not connect to the AgentFoundation MCP server 'af', so the "
            f"turn was not run ({detail}). Check that the AF host's local MCP "
            "endpoint is reachable from codex (sandbox / network policy)."
        )

    def _argv(self, request: TurnRequest) -> list[str]:
        spec = self._spec
        argv = [find_codex_binary(spec.cli_path) or CODEX_BINARY, "exec"]
        if self._session_id and self._started_emitted:
            argv += ["resume", self._session_id]
        argv += ["--json", "--skip-git-repo-check"]
        if spec.environment == "hermetic":
            argv += ["--ignore-user-config", "--ignore-rules"]
        servers = dict(spec.extra_mcp_servers)
        if self._mcp_url:
            # The URL carries no secret; the bearer token is read from the env.
            # `required`: codex refuses to start the thread (no events, rc 1,
            # stderr names the server) when `af` cannot be initialized — its
            # health check, verified with codex-cli 0.159.3.
            servers["af"] = {
                "url": self._mcp_url,
                "bearer_token_env_var": _TOKEN_ENV,
                "required": True,
                "tool_timeout_sec": math.ceil(spec.mcp_tool_timeout_ms / 1000),
            }
        for name, config in servers.items():
            for key, value in config.items():
                argv += ["-c", f"mcp_servers.{name}.{key}={_toml_value(value)}"]
        # A JSON string literal is a valid TOML basic string (same escapes), so
        # json.dumps handles the multi-line L1 text with quotes/backticks safely.
        argv += ["-c", f"developer_instructions={json.dumps(self._l1_text)}"]
        if spec.effort:
            argv += ["-c", f"model_reasoning_effort={_toml_value(spec.effort)}"]
        if spec.model:
            argv += ["-m", resolve_model_tag(spec.model)]
        if spec.permission_mode == "bypassPermissions":
            argv.append("--dangerously-bypass-approvals-and-sandbox")
        argv.append(request.text)
        return argv

    def _map(self, obj: dict[str, Any]) -> list[VendorEvent]:
        kind = obj.get("type")
        if kind == "thread.started":
            return self._session_started(obj.get("thread_id", ""))
        if kind == "item.completed":
            return self._on_item(obj.get("item") or {})
        if kind == "turn.completed":
            return [
                TurnEnd(
                    session_id=self._session_id,
                    stop_reason="completed",
                    is_error=False,
                    num_turns=0,
                    total_cost_usd=None,
                    usage=dict(obj.get("usage") or {}),
                    result_text="",
                    errors=(),
                )
            ]
        if kind in ("turn.failed", "error"):
            message = (
                obj.get("message")
                or (obj.get("error") or {}).get("message")
                or "codex turn failed"
            )
            return [VendorError(message=str(message), submitted=True)]
        return []

    def _on_item(self, item: dict[str, Any]) -> list[VendorEvent]:
        itype = item.get("type")
        if itype == "agent_message":
            mid = item.get("id") or "m"
            text = item.get("text", "") or ""
            return [
                TextDelta(message_id=mid, text=text),
                MessageEnd(
                    message_id=mid,
                    text=text,
                    tool_use_ids=(),
                    af_tool_use_ids=(),
                    message_uuid=mid,
                ),
            ]
        # An `mcp_tool_call` item arrives once its call finished; AF calls were
        # announced from their MCP requests, which name their output item.
        return []

    def _session_started(self, session_id: str) -> list[VendorEvent]:
        if session_id and not self._started_emitted:
            self._started_emitted = True
            self._session_id = session_id
            return [SessionStarted(session_id=session_id)]
        if session_id and not self._session_id:
            self._session_id = session_id
        return []

    async def interrupt(self) -> None:
        if self._proc is not None:
            await self._proc.kill()

    async def set_model(self, model: str) -> None:
        self._spec.model = model

    async def close(self) -> None:
        if self._proc is not None:
            await self._proc.kill()
        if self._mcp_token and self._mcp_server is not None:
            await self._mcp_server.unregister(self._mcp_token)
            self._mcp_token = None

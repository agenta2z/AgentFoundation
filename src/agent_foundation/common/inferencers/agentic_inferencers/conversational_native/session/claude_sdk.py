"""Claude Code backend over the Claude Agent SDK (``ClaudeSDKClient``).

Keeps Claude Code's own session, loop, built-in tools and compaction. The AF
layer reaches the model through:

* the appended system prompt (L1), via ``system_prompt`` preset + append-file;
* an in-process MCP server ``af`` carrying the AF tool bridge;
* a ``UserPromptSubmit`` hook that injects per-turn context (L2) — a turn
  whose ``init`` arrives without it having run is stopped (hooks dropped by
  policy, see ``claude_hooks``);
* ``PreToolUse`` / ``PostToolUse`` hooks that gate AF tools and end the turn.

Connection lifecycle follows the proven Future/Event/Task pattern (see
``ClaudeCodeSdkInferencer.aconnect``): ``connect()`` and ``disconnect()`` run in
one long-lived task that holds the client's anyio task group open, while
``query()`` / ``receive_response()`` run from the actor's turn task.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.af_hook import (
    fail_closed_output,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
    VendorSessionMissing,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    Compaction,
    MessageEnd,
    SessionStarted,
    TextDelta,
    TurnEnd,
    VendorError,
    VendorEvent,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BackendCapabilities,
    CallerTools,
    Evidence,
    InterruptNotAcknowledged,
    L2Channel,
    NativeBackendSpec,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.claude_common import (
    af_init_problem,
    af_status_problem,
    af_unusable,
    blocked_turn_reason,
    LOCAL_COMMANDS,
    SYNTHETIC_MODEL,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.claude_hooks import (
    PromptHookWatch,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.claude_sessions import (
    claude_fork_message_map,
    fork_claude_session,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.common import (
    build_permission_effort_kwargs,
    find_claude_binary,
    resolve_disable_osx_sandbox,
    resolve_model_tag,
)

logger: logging.Logger = logging.getLogger(__name__)

_AF_PREFIX = "mcp__af__"
_MISSING_SESSION = "No conversation found with session ID"
_HEALTH_TIMEOUT_S = 15.0
_FAILED_MCP = ("failed", "needs-auth", "disabled")

HookCallback = Callable[[dict, Optional[str], Any], Awaitable[dict]]


def _fail_closed(callback: HookCallback) -> HookCallback:
    """A hook callback that answers like the CLI relay does when AF cannot
    decide: Claude Code treats a failed SDK hook callback as a non-blocking
    error and runs the tool anyway (verified with claude 2.1.288)."""

    async def hook(input_data: dict, tool_use_id: Optional[str], context: Any) -> dict:
        try:
            return await callback(input_data, tool_use_id, context)
        except Exception as exc:
            logger.warning(
                "AF %s hook failed: %s", input_data.get("hook_event_name"), exc
            )
            output = fail_closed_output(input_data, f"{type(exc).__name__}: {exc}")
            return {
                ("continue_" if key == "continue" else key): value
                for key, value in (output or {}).items()
            }

    return hook


class ClaudeSdkBackend:
    capabilities = BackendCapabilities(
        kind="claude_sdk",
        caller_tools=CallerTools.INPROCESS,
        l2_channels=(L2Channel.HOOK, L2Channel.ENVELOPE),
        pinned_session_id=True,
        exact_fork=True,
        turn_stop_hook=True,
        subagent_attribution=True,
        compaction_signal=True,
        persistent_process=True,
        slash_passthrough=LOCAL_COMMANDS,
        owns_result_spill=True,
        environments=("inherit", "hermetic"),
        evidence={
            # Verified end-to-end against claude 2.1.289 via
            # scripts/native_spikes/e2e_research_sop.py (chat/sop/resume).
            "caller_tools": Evidence.VERIFIED,
            "turn_stop_hook": Evidence.VERIFIED,
            "l2_hook": Evidence.VERIFIED,
            "resume": Evidence.VERIFIED,
            "exact_fork": Evidence.VERIFIED,  # s13_claude_sdk_fork.py
            "compaction_signal": Evidence.VERIFIED,  # s1_s3_s5_s8_claude.py
            # get_mcp_status lists the in-process `af` (connected, its tools)
            # ~0.3 s after connect(); every turn's init lists it (`source: sdk`)
            # with its mcp__af__ tools. A raising hook callback fails open in
            # Claude Code, hence the fail-closed callbacks.
            "af_health": Evidence.VERIFIED,
            # MCP_TOOL_TIMEOUT applies to the in-process server: 2 s timed an
            # 8 s call out, 20 s let it finish.
            "mcp_tool_timeout": Evidence.VERIFIED,
            # A Task subagent's AF call carries agent_id in PreToolUse, the
            # main thread's has none (S12, s2_s4_s12_claude_cli_hooks.py).
            "subagent_attribution": Evidence.VERIFIED,
            # setting-sources "" + strict-mcp-config authenticate and load no
            # user or project CLAUDE.md, user hook or user MCP server (S8,
            # s1_s3_s5_s8_claude.py); AF's own hooks still run (H, ibid.).
            "hermetic": Evidence.VERIFIED,
        },
        relies_on=(
            "caller_tools",
            "l2_hook",
            "turn_stop_hook",
            "subagent_attribution",
            "resume",
            "exact_fork",
            "compaction_signal",
            "af_health",
            "mcp_tool_timeout",
        ),
        # The SDK drives the system `claude` binary (cli_path pinned).
        tested_versions={"claude-agent-sdk": "0.1.58", "claude": "2.1.289"},
        l1_route=(
            "appended to Claude Code's system prompt (preset `claude_code` + "
            "`append-system-prompt-file`), read when the SDK client starts"
        ),
        l2_routes={
            L2Channel.HOOK: (
                "the UserPromptSubmit hook callback's additionalContext (not user text)"
            )
        },
    )

    def __init__(self, spec: NativeBackendSpec, **_runtime: Any) -> None:
        self.capabilities.require_spec(spec)
        self._spec = spec
        self._client: Any = None
        self._disconnect_event: Optional[asyncio.Event] = None
        self._conn_task: Optional[asyncio.Task] = None
        self._session_id = ""
        self._hooks: Any = None
        self._session_started_emitted = False
        self._stream_message_id: Optional[str] = None
        self._blocked_reason: Optional[str] = None
        self._expects_af = False
        self._stop_requested = False
        self._prompt_hook = PromptHookWatch(self.capabilities.slash_passthrough)
        self._stderr: collections.deque[str] = collections.deque(maxlen=200)

    @property
    def session_id(self) -> str:
        return self._session_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self, request: SessionOpenRequest) -> None:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        self._hooks = request.hooks
        resume_id, resume = request.session_id, request.resume
        if request.fork_from is not None:
            source, boundary = request.fork_from
            # The fork is an existing transcript under a new id: resume it.
            resume_id, resume = await self._fork(source, boundary), True
        self._session_id = resume_id or ""
        options = self._build_options(
            ClaudeAgentOptions, request, resume_id, resume=resume
        )
        client = ClaudeSDKClient(options=options)
        await self._connect(client)
        self._client = client
        self._expects_af = bool(request.tools)
        if request.tools:
            await self._check_af_connected()

    async def _connect(self, client: Any) -> None:
        loop = asyncio.get_running_loop()
        connected: asyncio.Future = loop.create_future()
        self._disconnect_event = asyncio.Event()

        async def _hold_connection() -> None:
            try:
                await client.connect()
                connected.set_result(None)
            except Exception as exc:  # surfaced to open()
                connected.set_exception(exc)
                return
            await self._disconnect_event.wait()
            try:
                await client.disconnect()
            except Exception as exc:
                logger.warning("Claude SDK disconnect failed: %s", exc)

        self._conn_task = loop.create_task(_hold_connection())
        try:
            await connected
        except Exception as exc:
            if self._stderr_says_session_missing():
                raise VendorSessionMissing(
                    "Claude Code has no transcript for the session to resume"
                ) from exc
            raise

    async def _check_af_connected(self) -> None:
        """Fail the open unless Claude Code reports ``af`` connected with its
        tools (a managed ``allowManagedMcpServersOnly`` policy shows up as a
        failed or absent server). Every turn's ``init`` is checked again."""
        problem = await self._af_status_problem()
        if problem is None:
            return
        await self.close()
        raise NativeCapabilityError(problem)

    async def _af_status_problem(self) -> Optional[str]:
        # SDK 0.1.58 + claude 2.1.288: the in-process server is absent from
        # get_mcp_status right after connect() and listed as connected a few
        # hundred ms later, so absence and `pending` are polled past.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _HEALTH_TIMEOUT_S
        while True:
            try:
                response = await self._client.get_mcp_status()
            except Exception as exc:
                return af_unusable("unverifiable", f"get_mcp_status failed: {exc}")
            servers = {s.get("name"): s for s in (response or {}).get("mcpServers", [])}
            server = servers.get("af")
            problem = af_status_problem(server)
            failed = server is not None and server.get("status") in _FAILED_MCP
            if problem is None or failed or loop.time() >= deadline:
                return problem
            await asyncio.sleep(0.2)

    async def _fork(self, source: str, boundary: str) -> str:
        return await fork_claude_session(source, boundary)

    def fork_message_map(self, source: str, forked: str) -> dict[str, str]:
        return claude_fork_message_map(source, forked)

    def _build_options(
        self,
        options_cls: Any,
        request: SessionOpenRequest,
        resume_id: str,
        *,
        resume: bool,
    ) -> Any:
        spec = self._spec
        sdk_kwargs, extra_args = build_permission_effort_kwargs(
            spec.permission_mode or None,
            spec.effort,
            resolve_disable_osx_sandbox(spec.disable_osx_sandbox),
        )
        extra_args["append-system-prompt-file"] = request.l1_path
        if spec.environment == "hermetic":
            extra_args["setting-sources"] = ""
            extra_args["strict-mcp-config"] = None
        model = request.model or spec.model
        kwargs: dict[str, Any] = dict(
            system_prompt={"type": "preset", "preset": "claude_code"},
            cwd=request.cwd or spec.cwd or None,
            model=resolve_model_tag(model) if model else None,
            disallowed_tools=["AskUserQuestion", "EnterPlanMode", "ExitPlanMode"],
            include_partial_messages=True,
            hooks=self._build_hooks(),
            env={"MCP_TOOL_TIMEOUT": str(spec.mcp_tool_timeout_ms)},
            extra_args=extra_args,
            cli_path=find_claude_binary(spec.cli_path),
            stderr=self._stderr.append,
            **sdk_kwargs,
        )
        servers = dict(spec.extra_mcp_servers)
        if request.tools:
            servers["af"] = self._mcp_server(request.tools, request.result_max_chars)
            kwargs["allowed_tools"] = ["mcp__af"]
        if servers:
            kwargs["mcp_servers"] = servers
        # A fresh session pins its id; a resume continues an existing one.
        if resume and resume_id:
            kwargs["resume"] = resume_id
        elif resume_id:
            kwargs["session_id"] = resume_id
        return options_cls(**kwargs)

    def _mcp_server(self, tools: list, max_chars: int) -> Any:
        from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool
        from mcp.types import ToolAnnotations

        # Claude Code spills results above maxResultSizeChars to a file itself.
        annotations = (
            ToolAnnotations(maxResultSizeChars=max_chars) if max_chars else None
        )
        sdk_tools = [self._wrap_tool(sdk_tool, spec, annotations) for spec in tools]
        return create_sdk_mcp_server(name="af", version="1.0.0", tools=sdk_tools)

    @staticmethod
    def _wrap_tool(sdk_tool: Any, spec: Any, annotations: Any) -> Any:
        @sdk_tool(
            spec.name, spec.description, spec.input_schema, annotations=annotations
        )
        async def _handler(args: dict[str, Any]) -> dict[str, Any]:
            result = await spec.handler(args)
            text = getattr(result, "text", str(result))
            is_error = bool(getattr(result, "is_error", False))
            return {"content": [{"type": "text", "text": text}], "is_error": is_error}

        return _handler

    def _stderr_says_session_missing(self) -> bool:
        return any(_MISSING_SESSION in line for line in self._stderr)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _build_hooks(self) -> dict[str, list]:
        from claude_agent_sdk import HookMatcher

        return {
            "UserPromptSubmit": [
                HookMatcher(hooks=[_fail_closed(self._on_user_prompt)])
            ],
            "PreToolUse": [
                HookMatcher(
                    matcher=f"{_AF_PREFIX}.*", hooks=[_fail_closed(self._on_pre_tool)]
                )
            ],
            "PostToolUse": [
                HookMatcher(
                    matcher=f"{_AF_PREFIX}.*", hooks=[_fail_closed(self._on_post_tool)]
                )
            ],
            "PostToolUseFailure": [
                HookMatcher(
                    matcher=f"{_AF_PREFIX}.*",
                    hooks=[_fail_closed(self._on_post_tool_failure)],
                )
            ],
            "PreCompact": [HookMatcher(hooks=[_fail_closed(self._on_pre_compact)])],
        }

    async def _on_user_prompt(
        self, input_data: dict, tool_use_id: Optional[str], context: Any
    ) -> dict:
        self._prompt_hook.fired()
        l2 = self._hooks.l2_for_turn()
        if not l2:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": l2,
            }
        }

    async def _on_pre_tool(
        self, input_data: dict, tool_use_id: Optional[str], context: Any
    ) -> dict:
        name = input_data.get("tool_name", "")
        if not name.startswith(_AF_PREFIX):
            return {}
        reason = await self._hooks.before_af_tool(
            name,
            input_data.get("tool_use_id") or tool_use_id or "",
            input_data.get("agent_id"),
        )
        if reason:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        return {}

    async def _on_post_tool(
        self, input_data: dict, tool_use_id: Optional[str], context: Any
    ) -> dict:
        name = input_data.get("tool_name", "")
        if not name.startswith(_AF_PREFIX):
            return {}
        stop = await self._hooks.after_af_tool(
            name, input_data.get("tool_use_id") or tool_use_id or ""
        )
        if stop:
            return {
                "continue_": False,
                "stopReason": "AgentFoundation ended the turn (a question is pending "
                "or a background task started).",
            }
        return {}

    async def _on_post_tool_failure(
        self, input_data: dict, tool_use_id: Optional[str], context: Any
    ) -> dict:
        """An AF call that returned an error ends here: Claude Code runs
        ``PostToolUseFailure`` instead of ``PostToolUse`` for it and ignores
        ``continue: false`` from this hook (claude 2.1.288), so the call only
        counts as finished; a stop it makes due is delivered by a later AF
        call's ``PostToolUse``, if the turn makes one."""
        name = input_data.get("tool_name", "")
        if name.startswith(_AF_PREFIX):
            await self._hooks.after_af_tool(
                name, input_data.get("tool_use_id") or tool_use_id or ""
            )
        return {}

    async def _on_pre_compact(
        self, input_data: dict, tool_use_id: Optional[str], context: Any
    ) -> dict:
        self._hooks.on_compaction()
        return {}

    # ------------------------------------------------------------------
    # Turn
    # ------------------------------------------------------------------

    async def run_turn(self, request: TurnRequest) -> AsyncIterator[VendorEvent]:
        if self._client is None:
            yield VendorError(message="Claude SDK session is not open", submitted=False)
            return
        self._blocked_reason = None
        self._stop_requested = False
        self._prompt_hook.start(request)
        try:
            await self._client.query(
                request.text, session_id=self._session_id or "default"
            )
        except Exception as exc:
            yield VendorError(message=f"query failed: {exc}", submitted=False)
            return
        stopping: Optional[asyncio.Future] = None
        try:
            async for message in self._client.receive_response():
                for event in self._map(message):
                    if self._stop_requested and isinstance(
                        event, (TextDelta, MessageEnd)
                    ):
                        continue  # output of a turn being stopped
                    yield event
                if self._stop_requested and stopping is None:
                    # Init precedes the first model request: stop the turn
                    # before the model acts without AF's tools or hooks, still
                    # draining.
                    stopping = asyncio.ensure_future(self._client.interrupt())
        finally:
            if stopping is not None:
                await asyncio.gather(stopping, return_exceptions=True)

    def _map(self, message: Any) -> list[VendorEvent]:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            StreamEvent,
            SystemMessage,
        )

        if isinstance(message, StreamEvent):
            return self._map_stream_event(message)
        if isinstance(message, AssistantMessage):
            return self._map_assistant(message)
        if isinstance(message, SystemMessage):
            return self._map_system(message)
        if isinstance(message, ResultMessage):
            return self._map_result(message)
        return []

    def _map_system(self, message: Any) -> list[VendorEvent]:
        subtype = (message.subtype or "").lower()
        if "compact" in subtype:
            return [Compaction()]
        self._blocked_reason = (
            blocked_turn_reason(message.data or {}) or self._blocked_reason
        )
        # As the CLI reports it: the session exists from init on, so a turn
        # that ends before any model output leaves a session to resume (a
        # pinned id cannot be opened as new again).
        events = self._session_started((message.data or {}).get("session_id", ""))
        if subtype != "init":
            return events
        problem = af_init_problem(message.data or {}) if self._expects_af else None
        # Claude Code awaits the UserPromptSubmit callback before it emits init
        # (claude 2.1.288): a callback that has not run by now was dropped.
        problem = problem or self._prompt_hook.problem()
        if problem:
            # The prompt passed the hooks, so it is already in the transcript.
            self._stop_requested = True  # run_turn interrupts the turn
            events.append(VendorError(message=problem, submitted=True))
        return events

    def _model_started(self, *, model_output: bool = True) -> list[VendorEvent]:
        """The turn's first output: the hook check of a turn whose init was
        not seen, and of a ``/…`` prompt (checked only once the model starts;
        ``model_output`` is false for a local command's output)."""
        problem = self._prompt_hook.problem(model_output=model_output)
        if not problem:
            return []
        self._stop_requested = True
        return [VendorError(message=problem, submitted=True)]

    def _map_stream_event(self, message: Any) -> list[VendorEvent]:
        """Token streaming (``include_partial_messages``): text deltas of the
        main thread, keyed by the API message id."""
        if message.parent_tool_use_id:
            return []
        event = message.event or {}
        kind = event.get("type")
        if kind == "message_start":
            self._stream_message_id = (event.get("message") or {}).get("id")
            return self._model_started()
        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [
                    TextDelta(
                        message_id=self._stream_message_id or "m", text=delta["text"]
                    )
                ]
        return []

    def _map_assistant(self, message: Any) -> list[VendorEvent]:
        from claude_agent_sdk import TextBlock, ToolUseBlock

        if message.parent_tool_use_id:
            return []  # subagent output: no main-thread round
        stopped = self._model_started(model_output=message.model != SYNTHETIC_MODEL)
        if stopped:
            return stopped
        events = self._session_started(getattr(message, "session_id", ""))
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        tool_ids, af_ids = [], []
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                tool_ids.append(block.id)
                if block.name.startswith(_AF_PREFIX):
                    af_ids.append(block.id)
        # Text already streamed as deltas is not pushed again by the driver.
        events.append(
            MessageEnd(
                message_id=message.message_id
                or self._stream_message_id
                or message.uuid
                or "m",
                text=text,
                tool_use_ids=tuple(tool_ids),
                af_tool_use_ids=tuple(af_ids),
                message_uuid=message.uuid,
            )
        )
        return events

    def _map_result(self, message: Any) -> list[VendorEvent]:
        if (
            message.is_error
            and not message.num_turns
            and self._stderr_says_session_missing()
        ):
            return [
                VendorError(
                    message="Claude Code has no transcript for the session to resume",
                    submitted=False,
                    session_missing=True,
                )
            ]
        if self._blocked_reason and not message.num_turns:
            return [
                VendorError(
                    message=f"Claude Code did not run the turn: {self._blocked_reason}",
                    submitted=False,
                )
            ]
        events = self._session_started(message.session_id)
        events.append(
            TurnEnd(
                session_id=message.session_id,
                stop_reason=message.stop_reason,
                is_error=message.is_error,
                num_turns=message.num_turns,
                total_cost_usd=message.total_cost_usd,
                usage=dict(message.usage or {}),
                result_text=message.result or "",
                errors=tuple(message.errors or ()),
            )
        )
        return events

    def _session_started(self, session_id: str) -> list[VendorEvent]:
        if session_id and not self._session_started_emitted:
            self._session_started_emitted = True
            self._session_id = session_id
            return [SessionStarted(session_id=session_id)]
        return []

    async def interrupt(self) -> None:
        if self._client is None:
            return  # no live client: no turn can be running
        try:
            await self._client.interrupt()
        except Exception as exc:
            raise InterruptNotAcknowledged(
                f"Claude SDK interrupt failed: {exc}"
            ) from exc

    async def set_model(self, model: str) -> None:
        """Switch the live client's model; raises when Claude Code refuses, so
        the caller can reopen the session on the new model instead."""
        self._spec.model = model
        if self._client is not None:
            await self._client.set_model(resolve_model_tag(model))

    async def close(self) -> None:
        if self._disconnect_event is not None:
            self._disconnect_event.set()
        if self._conn_task is not None:
            try:
                await self._conn_task
            except Exception as exc:
                logger.warning("Claude SDK connection task ended with: %s", exc)
            self._conn_task = None
        self._client = None

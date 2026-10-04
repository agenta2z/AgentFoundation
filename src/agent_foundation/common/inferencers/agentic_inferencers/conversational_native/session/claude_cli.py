"""Claude Code backend over the ``claude -p`` CLI (one process per turn).

Same contract as the SDK backend, different transport. AF tools are served by
the shared localhost HTTP MCP server (S11-verified) via ``--mcp-config``. The
SDK's Python hooks become Claude Code command hooks (``--settings``) that run
the stdlib relay ``bridge/af_hook.py``, which forwards each hook to the AF
process over the same server: ``UserPromptSubmit`` carries L2
(``additionalContext``), ``PreToolUse`` denies AF tools to subagents
(``agent_id``), ``PostToolUse`` ends the turn (``continue: false``),
``PostToolUseFailure`` finishes an AF call that returned an error and
``PreCompact`` reports compaction (S2/S4/S12 verified, also under hermetic
settings). The relay fails closed when the AF process does not answer, and a
prompt a hook blocked fails the turn; a turn whose model starts although the
``UserPromptSubmit`` hook never ran (hooks dropped by policy) is killed. Every
turn's ``system``/``init`` must show ``af`` connected with its tools, else the
process is killed before the model acts. Reads ``--output-format stream-json``
and maps it to VendorEvents.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge import (
    af_hook,
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
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.private_files import (
    write_private_file,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BackendCapabilities,
    CallerTools,
    Evidence,
    L2Channel,
    NativeBackendSpec,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.claude_common import (
    af_init_problem,
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
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.cli_runner import (
    CliProcess,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.common import (
    CLAUDE_BINARY,
    DANGEROUSLY_DISABLE_OSX_SANDBOX,
    find_claude_binary,
    resolve_disable_osx_sandbox,
    resolve_model_tag,
)

logger: logging.Logger = logging.getLogger(__name__)

_AF_PREFIX = "mcp__af__"
_DISALLOWED = ["AskUserQuestion", "EnterPlanMode", "ExitPlanMode"]
_MISSING_SESSION = "No conversation found with session ID"
_STOP_REASON = "AgentFoundation ended the turn (a question is pending or a background task started)."
# Claude Code treats a hook that outlives its timeout as a non-blocking error
# (the tool runs, the turn goes on — verified with 2.1.288), so the relay must
# give its fail-closed answer well before this.
_HOOK_TIMEOUT_S = int(af_hook.TIMEOUT_S * 2)


class ClaudeCliBackend:
    capabilities = BackendCapabilities(
        kind="claude_cli",
        caller_tools=CallerTools.HTTP,
        l2_channels=(L2Channel.HOOK, L2Channel.ENVELOPE),
        pinned_session_id=True,
        exact_fork=True,
        turn_stop_hook=True,
        subagent_attribution=True,
        compaction_signal=True,
        persistent_process=False,
        slash_passthrough=LOCAL_COMMANDS,
        environments=("inherit", "hermetic"),
        evidence={
            # e2e_research_sop.py --backend claude_cli; s2_s4_s12_claude_cli_hooks.py
            "caller_tools": Evidence.VERIFIED,
            "resume": Evidence.VERIFIED,
            "l2_hook": Evidence.VERIFIED,
            "turn_stop_hook": Evidence.VERIFIED,
            "subagent_attribution": Evidence.VERIFIED,
            # s13_claude_sdk_fork.py / s1_s3_s5_s8_claude.py --kind claude_cli
            "exact_fork": Evidence.VERIFIED,
            "compaction_signal": Evidence.VERIFIED,
            # `claude -p` waits for MCP servers (up to MCP_TIMEOUT) before its
            # init, which shows `af` connected or failed (refused, bad bearer,
            # silent server) and names every mcp__af__ tool; a user-level
            # allowManagedMcpServersOnly is ignored (managed settings only).
            "af_health": Evidence.VERIFIED,
            # The hook relay fails closed (deny / stop / block) on a dead host.
            "hook_fail_closed": Evidence.VERIFIED,
            # MCP_TOOL_TIMEOUT: 2 s timed an 8 s HTTP MCP call out, 20 s let it
            # finish.
            "mcp_tool_timeout": Evidence.VERIFIED,
            # --setting-sources "" --strict-mcp-config authenticate and load no
            # user or project CLAUDE.md, user hook or user MCP server (S8,
            # s1_s3_s5_s8_claude.py); the --settings hooks still run (H,
            # s2_s4_s12_claude_cli_hooks.py).
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
            "hook_fail_closed",
            "mcp_tool_timeout",
        ),
        tested_versions={"claude": "2.1.289"},
        l1_route=(
            "appended to Claude Code's system prompt "
            "(`--append-system-prompt-file`), read by every `claude -p` process"
        ),
        l2_routes={
            L2Channel.HOOK: (
                "the UserPromptSubmit command hook (`--settings`) as "
                "additionalContext (not user text)"
            )
        },
    )

    def __init__(
        self, spec: NativeBackendSpec, *, runtime_manager: Any = None, **_: Any
    ) -> None:
        self.capabilities.require_spec(spec)
        self._spec = spec
        self._runtime = runtime_manager
        self._session_id = ""
        self._l1_path = ""
        self._private_dir = ""
        self._mcp_config_path: Optional[str] = None
        self._settings_path: Optional[str] = None
        self._mcp_token: Optional[str] = None
        self._mcp_server: Any = None
        self._turn_active = False
        self._hook_url = ""
        self._hooks: Any = None
        self._started_emitted = False
        self._stream_message_id: Optional[str] = None
        self._blocked_reason: Optional[str] = None
        self._expects_af = False
        self._prompt_hook = PromptHookWatch(self.capabilities.slash_passthrough)
        self._proc: Optional[CliProcess] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    async def open(self, request: SessionOpenRequest) -> None:
        self._l1_path = request.l1_path
        self._private_dir = os.path.dirname(request.l1_path) or os.getcwd()
        self._hooks = request.hooks
        self._session_id = request.session_id or ""
        if request.fork_from is not None:
            # The fork is an existing transcript under a new id: resume it.
            self._session_id = await fork_claude_session(*request.fork_from)
            self._started_emitted = True
        elif request.resume:
            self._started_emitted = True  # this process continues an existing session
        servers = dict(self._spec.extra_mcp_servers)
        if self._runtime is not None:
            server = await self._runtime.ensure_http_server()
            url, token, headers = await server.register(
                request.tools,
                hook_handler=self._on_hook,
                turn_active=lambda: self._turn_active,
                result_max_chars=request.result_max_chars,
            )
            self._mcp_server = server
            self._mcp_token = token
            self._hook_url = f"{server.base_url}/hook"
            if request.tools:
                servers["af"] = {"type": "http", "url": url, "headers": headers}
                self._expects_af = True
            self._settings_path = self._write_private("settings.json", self._settings())
        if servers:
            self._mcp_config_path = self._write_private(
                "mcp_config.json", json.dumps({"mcpServers": servers})
            )

    # ------------------------------------------------------------------
    # Hooks (relayed from the claude process by bridge/af_hook.py)
    # ------------------------------------------------------------------

    def _settings(self) -> str:
        relay = Path(__file__).resolve().parent.parent / "bridge" / "af_hook.py"
        python = (
            self._spec.extra.get("relay_python")
            or shutil.which("python3")
            or sys.executable
        )
        command = [
            {
                "type": "command",
                "command": f"{shlex.quote(python)} {shlex.quote(str(relay))}",
                "timeout": _HOOK_TIMEOUT_S,
            }
        ]
        af_only = [{"matcher": f"{_AF_PREFIX}.*", "hooks": command}]
        return json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [{"hooks": command}],
                    "PreToolUse": af_only,
                    "PostToolUse": af_only,
                    "PostToolUseFailure": af_only,
                    "PreCompact": [{"hooks": command}],
                }
            }
        )

    async def _on_hook(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = payload.get("hook_event_name", "")
        if event == "UserPromptSubmit":
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
        if event == "PreCompact":
            self._hooks.on_compaction()
            return {}
        name = str(payload.get("tool_name", ""))
        if not name.startswith(_AF_PREFIX):
            return {}
        tool_use_id = str(payload.get("tool_use_id") or "")
        if event == "PreToolUse":
            reason = await self._hooks.before_af_tool(
                name, tool_use_id, payload.get("agent_id")
            )
            if not reason:
                return {}
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        if event == "PostToolUseFailure":
            # An AF call that returned an error: Claude Code runs this instead
            # of PostToolUse and ignores ``continue: false`` from it (claude
            # 2.1.288); the call only counts as finished.
            await self._hooks.after_af_tool(name, tool_use_id)
            return {}
        if event == "PostToolUse" and await self._hooks.after_af_tool(
            name, tool_use_id
        ):
            return {"continue": False, "stopReason": _STOP_REASON}
        return {}

    # ------------------------------------------------------------------
    # Turn
    # ------------------------------------------------------------------

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
        self._blocked_reason = None
        self._prompt_hook.start(
            request, hooks_registered=self._settings_path is not None
        )
        argv = self._argv(request)
        env = {"MCP_TOOL_TIMEOUT": str(self._spec.mcp_tool_timeout_ms)}
        if self._mcp_token:
            env.update(AF_HOOK_URL=self._hook_url, AF_HOOK_TOKEN=self._mcp_token)
        self._proc = CliProcess(
            argv, cwd=self._spec.cwd or os.getcwd(), env=env, stdin_devnull=True
        )
        try:
            await self._proc.start()
        except Exception as exc:
            yield VendorError(message=f"spawn failed: {exc}", submitted=False)
            return
        result: Optional[TurnEnd] = None
        async with contextlib.aclosing(self._proc.json_lines()) as lines:
            async for obj in lines:
                for event in self._map(obj):
                    if isinstance(event, TurnEnd):
                        result = event  # emitted after exit (needs stderr)
                        continue
                    yield event
                    if isinstance(event, VendorError):
                        await self._proc.kill()
                        return
        rc = await self._proc.wait()
        if result is None:
            yield VendorError(
                message=f"claude exited rc={rc}: {self._proc.stderr_text[-400:]}"
            )
        elif result.is_error and not result.num_turns and self._missing_session():
            yield VendorError(
                message="Claude Code has no transcript for the session to resume",
                submitted=False,
                session_missing=True,
            )
        else:
            yield result

    def _missing_session(self) -> bool:
        return (
            self._started_emitted
            and self._proc is not None
            and _MISSING_SESSION in self._proc.stderr_text
        )

    def _argv(self, request: TurnRequest) -> list[str]:
        spec = self._spec
        argv = [
            find_claude_binary(spec.cli_path) or CLAUDE_BINARY,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt-file",
            self._l1_path,
            "--disallowedTools",
            ",".join(_DISALLOWED),
            "--permission-mode",
            spec.permission_mode or "bypassPermissions",
        ]
        if spec.effort:
            argv += ["--effort", spec.effort]
        if resolve_disable_osx_sandbox(spec.disable_osx_sandbox):
            argv.append(f"--{DANGEROUSLY_DISABLE_OSX_SANDBOX}")
        if spec.model:
            argv += ["--model", resolve_model_tag(spec.model)]
        if self._settings_path:
            argv += ["--settings", self._settings_path]
        if self._mcp_config_path:
            argv += ["--mcp-config", self._mcp_config_path, "--allowedTools", "mcp__af"]
        if spec.environment == "hermetic":
            # Only our settings and MCP servers; `inherit` keeps the user's own.
            argv += ["--setting-sources", "", "--strict-mcp-config"]
        if self._session_id and self._started_emitted:
            argv += ["--resume", self._session_id]
        elif self._session_id:
            argv += ["--session-id", self._session_id]
        argv.append(request.text)
        return argv

    # ------------------------------------------------------------------
    # stream-json -> VendorEvents
    # ------------------------------------------------------------------

    def _map(self, obj: dict[str, Any]) -> list[VendorEvent]:
        kind = obj.get("type")
        if obj.get("parent_tool_use_id"):
            return []  # subagent output: no main-thread round
        if kind == "system":
            return self._map_system(obj)
        if kind == "stream_event":
            return self._map_stream_event(obj.get("event") or {})
        if kind == "assistant":
            return self._map_assistant(obj)
        if kind == "result":
            return self._map_result(obj)
        return []

    def _map_system(self, obj: dict[str, Any]) -> list[VendorEvent]:
        subtype = str(obj.get("subtype", "")).lower()
        if "compact" in subtype:
            return [Compaction()]
        self._blocked_reason = blocked_turn_reason(obj) or self._blocked_reason
        events = self._session_started(obj.get("session_id", ""))
        problem = (
            af_init_problem(obj) if subtype == "init" and self._expects_af else None
        )
        if problem:
            # Init precedes the first model request: run_turn kills the
            # process before the model can act without AF's tools. The prompt
            # already passed the UserPromptSubmit hooks, so it may be recorded.
            events.append(VendorError(message=problem, submitted=True))
        return events

    def _model_started(self, *, model_output: bool = True) -> list[VendorEvent]:
        """Checked at the turn's first output, not at init: a relay that
        cannot reach AF blocks the prompt, which Claude Code reports only
        after init, while a hook that never ran lets the model start
        (``model_output`` is false for a local command's output). The prompt
        passed the hooks, so it is already in the transcript; run_turn kills
        the process."""
        problem = self._prompt_hook.problem(model_output=model_output)
        return [VendorError(message=problem, submitted=True)] if problem else []

    def _map_stream_event(self, event: dict[str, Any]) -> list[VendorEvent]:
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

    def _map_assistant(self, obj: dict[str, Any]) -> list[VendorEvent]:
        model = (obj.get("message") or {}).get("model")
        stopped = self._model_started(model_output=model != SYNTHETIC_MODEL)
        if stopped:
            return stopped
        events = self._session_started(obj.get("session_id", ""))
        message = obj.get("message", {}) or {}
        content = message.get("content", []) or []
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        tool_ids, af_ids = [], []
        for block in content:
            if block.get("type") == "tool_use":
                tid = block.get("id", "")
                tool_ids.append(tid)
                if str(block.get("name", "")).startswith(_AF_PREFIX):
                    af_ids.append(tid)
        # Text streamed as deltas is not pushed again by the driver.
        events.append(
            MessageEnd(
                message_id=message.get("id")
                or self._stream_message_id
                or obj.get("uuid")
                or "m",
                text=text,
                tool_use_ids=tuple(tool_ids),
                af_tool_use_ids=tuple(af_ids),
                message_uuid=obj.get("uuid"),
            )
        )
        return events

    def _map_result(self, obj: dict[str, Any]) -> list[VendorEvent]:
        events = self._session_started(obj.get("session_id", ""))
        if self._blocked_reason and not obj.get("num_turns"):
            # The model never saw the prompt (e.g. the L2 hook could not reach
            # AF): a failed turn, not a reply made of Claude Code's notice.
            events.append(
                VendorError(
                    message=f"Claude Code did not run the turn: {self._blocked_reason}",
                    submitted=False,
                )
            )
            return events
        events.append(
            TurnEnd(
                session_id=obj.get("session_id", "") or self._session_id,
                stop_reason=obj.get("stop_reason"),
                is_error=bool(obj.get("is_error", False)),
                num_turns=int(obj.get("num_turns", 0) or 0),
                total_cost_usd=obj.get("total_cost_usd"),
                usage=dict(obj.get("usage") or {}),
                result_text=obj.get("result", "") or "",
                errors=tuple(obj.get("errors") or ()),
            )
        )
        return events

    def _session_started(self, session_id: str) -> list[VendorEvent]:
        if session_id and not self._started_emitted:
            self._started_emitted = True
            self._session_id = session_id
            return [SessionStarted(session_id=session_id)]
        return []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def fork_message_map(self, source: str, forked: str) -> dict[str, str]:
        return claude_fork_message_map(source, forked)

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

    def _write_private(self, name: str, text: str) -> str:
        return str(write_private_file(os.path.join(self._private_dir, name), text))

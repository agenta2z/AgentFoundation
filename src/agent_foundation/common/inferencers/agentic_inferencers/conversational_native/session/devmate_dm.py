"""Devmate backend over the ``dm`` CLI (one process per turn).

Same contract as the other per-turn CLI backends, different transport. dm-core's
``--mcp-servers`` supports only unix-socket and stdio MCP (no HTTP). Its *socket*
transport stalls real-model turns, so AF tools are served by the shared in-process
``LocalMcpSocketServer`` (newline-delimited JSON-RPC over a 0600 unix socket) and
reached via dm's *stdio* transport through a stdlib relay
(``bridge/mcp_stdio_relay.py``) that dm spawns and that forwards its stdio MCP to
our socket. Real-model tool calls verified e2e by scripts/native_spikes/s14.

dm-core MCP stall + the fix (see s14 for the full diagnosis): attaching ANY MCP
server stalls a real-model dm turn unless the model promptly calls a tool, because
dm's MCP *warmup* conflicts with the agent's MCP connection. ``DM_CORE_WORKFLOW=1``
(dm's programmatic "workflow mode") disables warmup and uses only the explicit
``--mcp-servers`` — eliminating the stall — but requires explicit CAT credentials
via ``--cats-file`` (``{tokens:{plugboard_cat, ai_gateway_cat, …}, fbid}``;
provisioned by devinfra/the Rust adapter). So AF tools over dm are fully usable
only with a cats-file (see ``resolve_cats_file``); without it, tool turns hit the
dm-core warmup stall.

L1 is appended inline (dm has no ``--append-system-prompt-file``) and re-applied
every turn (dm does not persist the append across ``--resume``); L2 rides the user
turn as a labelled ``<af_context>`` envelope; the turn ends on the ``AF_END_TURN``
directive the bridge puts at the head of a widget/async tool result. Reads
``--output-format devmate-sdk-events`` and maps the ``{event:{<kind>:…}}`` union
to VendorEvents.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
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
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.common import (
    DM_BINARY,
    find_dm_binary,
    resolve_dm_model,
)

logger: logging.Logger = logging.getLogger(__name__)

_AF_PREFIX = "mcp__af__"
_OK_EXITS = ("COMPLETE", "COMPLETED", "")


class DevmateDmBackend:
    capabilities = BackendCapabilities(
        kind="devmate_dm",
        caller_tools=CallerTools.SOCKET,
        l2_channels=(L2Channel.ENVELOPE,),
        pinned_session_id=True,  # --session-id <uuid> is accepted and reflected
        exact_fork=False,  # --fork clones a whole session, not up-to-message
        turn_stop_hook=False,  # directive-only (AF_END_TURN); no continue:false hook
        subagent_attribution=False,  # inner_session events are dropped instead
        compaction_signal=False,
        persistent_process=False,
        slash_passthrough=(),
        evidence={  # stdio-relay MCP + real-model tool call verified by scripts/native_spikes/s14
            "caller_tools": Evidence.VERIFIED,
            # --append-system-prompt reaches the model request: in dm's logged
            # request (DM_DEBUG_EVENTS=1, scripted model, outside workflow
            # mode; dm-core's prompt builder has no workflow branch) an L1
            # sentinel sat in the SYSTEM message once, on the first turn, a
            # --resume turn and a restarted process, whose new L1 replaced the
            # old (dm_l1_scripted_20261003.log in ~/native_spike_logs).
            "l1": Evidence.VERIFIED,
            # A turn run with --resume <pinned id> carries the earlier turns:
            # dm's scripted model, picking its reply by the number of tool
            # results in the model request, answered a second turn and a
            # restarted process from turn 1's history, a fresh id did not
            # (dm_resume_scripted_20261003.log in ~/native_spike_logs).
            "resume": Evidence.VERIFIED,
            # dm-core reads a per-server `toolCallTimeoutMs` from --mcp-servers
            # (default $MCP_TOOL_TIMEOUT, else 24 h): 2 s timed an 8 s call out,
            # 20 s let it finish (scripted model).
            "mcp_tool_timeout": Evidence.VERIFIED,
            # dm-core attaches --mcp-servers to root agents only
            # (`includeUserMcpServers: !parentAgentId`): a scripted subagent's
            # mcp__af__ call never reached the AF server (native harness).
            "subagent_deny": Evidence.VERIFIED,
            # --additional-hooks-file runs outside workflow mode, but workflow
            # mode (DM_CORE_WORKFLOW=1, needed for AF tool turns) loads no user
            # or additional hooks: no hook-based L2, turn stop or deny.
            "hooks": Evidence.UNSUPPORTED,
            # No counterpart of Claude's --setting-sources or Codex's
            # --ignore-user-config: rules and skills load in both modes (no
            # workflow-mode branch in dm-core), ~/.llms hooks and
            # ~/.devmate/mcp.json outside workflow mode.
            "hermetic": Evidence.UNSUPPORTED,
        },
        relies_on=(
            "caller_tools",
            "l1",
            "resume",
            "subagent_deny",
            "mcp_tool_timeout",
        ),
        # `dm` resolves its build from the cwd: master outside an fbsource
        # checkout, the checkout's pinned build inside one (2026.09.01-0146 in
        # the checkout used for these tests).
        tested_versions={"dm": "2026.10.03-0249"},
        l1_route=(
            "appended to dm's system prompt (`--append-system-prompt`), re-sent "
            "with every `dm -p` turn"
        ),
    )

    def __init__(
        self, spec: NativeBackendSpec, *, runtime_manager: Any = None, **_: Any
    ) -> None:
        if spec.environment == "hermetic":
            raise NativeCapabilityError(
                "Devmate dm has no hermetic mode: it always loads the user's "
                "rules and skills (and, outside workflow mode, ~/.llms hooks and "
                "~/.devmate/mcp.json). Use environment: inherit."
            )
        self.capabilities.require_spec(spec)
        self._spec = spec
        self._runtime = runtime_manager
        self._session_id = ""
        self._l1_text = ""
        self._mcp_servers_json: Optional[str] = None
        self._mcp_token: Optional[str] = None
        self._mcp_server: Any = None
        self._cats_file: Optional[str] = None
        self._started_emitted = False
        self._proc: Optional[CliProcess] = None
        self._reset_turn_state()

    @property
    def session_id(self) -> str:
        return self._session_id

    async def open(self, request: SessionOpenRequest) -> None:
        self._l1_text = request.l1_text
        self._session_id = request.session_id or ""
        self._cats_file = resolve_cats_file(self._spec.extra)
        if request.resume:
            self._session_id = request.session_id
            self._started_emitted = True  # a resumed process uses --resume from turn 1
        if request.tools and self._runtime is not None:
            server = await self._runtime.ensure_socket_server()
            socket_path, token = await server.register(request.tools)
            self._mcp_server = server
            self._mcp_token = token
            # dm's `socket` MCP transport stalls real-model turns (dm-core bug,
            # see scripts/native_spikes/s14); its `stdio` transport works. So dm
            # spawns a stdlib relay that bridges its stdio MCP <-> our in-process
            # unix socket. Relay + real-model tool call verified by s14.
            self._mcp_servers_json = json.dumps(
                [
                    {
                        "type": "stdio",
                        "name": "af",
                        "command": self._relay_python(),
                        "args": [self._relay_path(), socket_path],
                        # dm-core's per-server `tools/call` timeout (ms).
                        "toolCallTimeoutMs": self._spec.mcp_tool_timeout_ms,
                    }
                ]
                + self._extra_servers()
            )
        elif self._spec.extra_mcp_servers:
            self._mcp_servers_json = json.dumps(self._extra_servers())

    def _extra_servers(self) -> list[dict[str, Any]]:
        return [
            {"name": name, **config}
            for name, config in self._spec.extra_mcp_servers.items()
        ]

    @staticmethod
    def _relay_path() -> str:
        return str(
            Path(__file__).resolve().parent.parent / "bridge" / "mcp_stdio_relay.py"
        )

    def _relay_python(self) -> str:
        # The relay is stdlib-only, so any python3 runs it. Prefer a plain
        # interpreter (robust under a packaged/PAR host where sys.executable is
        # not a bare python); allow an explicit override.
        return (
            self._spec.extra.get("relay_python")
            or shutil.which("python3")
            or sys.executable
            or "python3"
        )

    async def run_turn(self, request: TurnRequest) -> AsyncIterator[VendorEvent]:
        self._reset_turn_state()
        argv = self._argv(request)
        # Workflow mode (explicit CAT creds) disables dm's MCP warmup, which
        # otherwise stalls a real-model turn that doesn't promptly call a tool
        # (dm-core bug, see scripts/native_spikes/s14). It is the correct mode for
        # programmatic driving but requires a cats-file; without one dm-core
        # throws "Workflow engine requires explicit CAT credentials".
        env = {"DM_CORE_WORKFLOW": "1"} if self._cats_file else None
        self._proc = CliProcess(
            argv, cwd=self._spec.cwd or os.getcwd(), env=env, stdin_devnull=True
        )
        try:
            await self._proc.start()
        except Exception as exc:
            yield VendorError(message=f"spawn failed: {exc}", submitted=False)
            return
        saw_result = False
        async for obj in self._proc.json_lines():
            for event in self._map(obj):
                if isinstance(event, TurnEnd):
                    saw_result = True
                yield event
        rc = await self._proc.wait()
        if not saw_result:
            yield VendorError(
                message=f"dm exited rc={rc}: {self._proc.stderr_text[-400:]}"
            )

    def _argv(self, request: TurnRequest) -> list[str]:
        spec = self._spec
        argv = [
            find_dm_binary(spec.cli_path) or DM_BINARY,
            "-p",
            "--output-format",
            "devmate-sdk-events",
            "--agent-harness",
            str(spec.extra.get("agent_harness", "native")),
            "--append-system-prompt",
            self._l1_text,
        ]
        if spec.model:
            argv += ["--model", resolve_dm_model(spec.model)]
        if self._mcp_servers_json:
            argv += ["--mcp-servers", self._mcp_servers_json]
        if self._cats_file:
            argv += ["--cats-file", self._cats_file]
        if self._session_id and self._started_emitted:
            argv += ["--resume", self._session_id]
        elif self._session_id:
            argv += ["--session-id", self._session_id]
        argv.append(request.text)
        return argv

    # -- event mapping ------------------------------------------------------

    def _reset_turn_state(self) -> None:
        self._cur_step: Optional[str] = None
        self._cur_text = ""
        self._cur_finish = ""
        self._cur_tool_ids: list[str] = []
        self._cur_af_ids: list[str] = []

    def _map(self, obj: dict[str, Any]) -> list[VendorEvent]:
        ev = obj.get("event")
        if not isinstance(ev, dict) or not ev:
            return []
        kind = next(iter(ev))
        body = ev.get(kind) or {}
        handler = getattr(self, f"_on_{kind}", None)
        return handler(body) if handler is not None else []

    def _on_session_start(self, body: dict[str, Any]) -> list[VendorEvent]:
        return self._session_started((body.get("session") or {}).get("id", ""))

    def _on_session_update(self, body: dict[str, Any]) -> list[VendorEvent]:
        return self._session_started((body.get("session") or {}).get("id", ""))

    def _on_step_start(self, body: dict[str, Any]) -> list[VendorEvent]:
        self._reset_turn_state()
        self._cur_step = (body.get("step") or {}).get("id")
        return []

    def _on_action_end(self, body: dict[str, Any]) -> list[VendorEvent]:
        action = body.get("action") or {}
        variant = action.get("variant") or {}
        vkind = next(iter(variant), "") if variant else ""
        out = action.get("output") or {}
        if vkind == "llm_action":
            text = out.get("info") or ""
            if text:
                self._cur_text += text
                mid = action.get("step_id") or self._cur_step or "m"
                return [TextDelta(message_id=mid, text=text)]
        elif vkind == "tool_use_action":
            tv = variant.get("tool_use_action") or {}
            tid = tv.get("tool_use_id") or action.get("id") or ""
            name = tv.get("tool_name") or ""
            if tid:
                self._cur_tool_ids.append(tid)
                if name.startswith(_AF_PREFIX):
                    self._cur_af_ids.append(tid)
        elif vkind == "agent_finished_action":
            data = out.get("data") or {}
            self._cur_finish = (
                data.get("message") or out.get("info") or self._cur_finish
            )
        return []

    def _on_step_end(self, body: dict[str, Any]) -> list[VendorEvent]:
        mid = (body.get("step") or {}).get("id") or self._cur_step or "m"
        text = self._cur_text or self._cur_finish
        if not text and not self._cur_tool_ids:
            self._reset_turn_state()
            return []
        end = MessageEnd(
            message_id=mid,
            text=text,
            tool_use_ids=tuple(self._cur_tool_ids),
            af_tool_use_ids=tuple(self._cur_af_ids),
            message_uuid=mid,
        )
        self._reset_turn_state()
        return [end]

    def _on_session_end(self, body: dict[str, Any]) -> list[VendorEvent]:
        session = body.get("session") or {}
        sid = session.get("id", "") or self._session_id
        exit_code = session.get("exit_code") or ""
        events = self._session_started(sid)
        events.append(
            TurnEnd(
                session_id=sid,
                stop_reason=str(exit_code),
                is_error=exit_code not in _OK_EXITS,
                num_turns=0,
                total_cost_usd=None,
                usage={},
                result_text=session.get("exit_message", "") or self._cur_finish,
                errors=(),
            )
        )
        return events

    def _on_session_error(self, body: dict[str, Any]) -> list[VendorEvent]:
        session = body.get("session") or {}
        message = (
            session.get("exit_message") or body.get("message") or "dm session error"
        )
        return [VendorError(message=str(message), submitted=True)]

    def _session_started(self, session_id: str) -> list[VendorEvent]:
        if session_id and not self._started_emitted:
            self._started_emitted = True
            self._session_id = session_id
            return [SessionStarted(session_id=session_id)]
        if session_id and self._session_id and session_id != self._session_id:
            # dm does not fail a resume of a session it no longer has: it starts
            # a new one with a different id (memory of earlier turns is gone).
            self._session_id = session_id
            return [SessionStarted(session_id=session_id, replaced=True)]
        if session_id and not self._session_id:
            self._session_id = session_id
        return []

    # -- lifecycle ----------------------------------------------------------

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


_PREMINTED_CATS = "/tmp/devinfra/user_cats/cats"


def resolve_cats_file(extra: dict[str, Any]) -> Optional[str]:
    """The CAT-credentials file for dm's workflow mode: the backend's
    ``cats_file`` option, else ``$DM_CATS_FILE``, ``$PREMINTED_CATS_FILE`` or
    devinfra's pre-minted file when present; None when there is none."""
    if extra.get("cats_file"):
        return extra["cats_file"]
    for candidate in (
        os.environ.get("DM_CATS_FILE"),
        os.environ.get("PREMINTED_CATS_FILE"),
        _PREMINTED_CATS,
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None

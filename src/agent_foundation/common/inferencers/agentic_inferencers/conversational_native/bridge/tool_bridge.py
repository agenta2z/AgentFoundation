"""AFToolBridge — AgentFoundation tools as native (MCP) tools.

Handlers run in tasks the vendor SDK / MCP server spawns, so they never rely on
context variables: they look up the active ``TurnScope`` on the host and bind
its run context explicitly. Execution reuses ``tool_dispatch.execute``, the
widget lifecycle of ``widget_core`` and the ``SOPController`` API the SOP slash
commands delegate to (with typed arguments, never a re-parsed command line),
so tool semantics are identical to the text-protocol orchestrator.

A vendor need not cancel a running handler when its turn ends (e.g. a client
disconnect), so a tool's result is applied to the host — context updates,
phase completion, the recorded action, the state update — only if the call's
turn is still the active one when the tool returns. Nor is the result of a
call the vendor already gave up on (its tool-call timeout passed and the model
was told the call failed): the next turn's host notice reports what happened.
Every applied call persists the session record (e.g. a queued widget), not
only the turn's end.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence

import jsonschema
from agent_foundation.common.inferencers.agentic_inferencers.conversational import (
    widget_core,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_response_parser import (
    tool_invocation_to_conversation_tool,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
    SopCommandWording,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_dispatch import (
    apply_outcome,
    execute,
    ToolOutcome,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.widget_core import (
    WidgetHost,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.schema import (
    action_tool_schema,
    assert_unique,
    canonical_arguments,
    executor_arguments,
    MCP_PREFIX,
    mcp_tool_name,
    SOP_COMMAND_SCHEMAS,
    TOOL_ARGUMENT_FORM,
    TOOL_ARGUMENT_FORM_SCHEMA,
    WIDGET_SCHEMAS,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.context.composer import (
    neutralize_host_tags,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.context.text_units import (
    utf16_head,
    utf16_len,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.private_files import (
    write_private_file,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BridgeToolSpec,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.turn import (
    QueuedWidget,
    StopCause,
    TurnScope,
)
from agent_foundation.common.inferencers.run_context import enter_run, exit_run
from agent_foundation.resources.tools.formatters.markdown import ToolMarkdownFormatter

logger: logging.Logger = logging.getLogger(__name__)

END_TURN_MARKER = "AF_END_TURN"
_WIDGET_QUEUED = (
    f"{END_TURN_MARKER} — question queued for the user. It is shown when you end "
    "your turn; the answer arrives as the next message. End your turn now and "
    "make no further tool calls."
)
_ASYNC_STARTED = (
    f"{END_TURN_MARKER} — the tool started in the background. End your turn now; "
    "its outcome arrives later as a host notice."
)
_STOP_REASONS = {
    StopCause.WIDGET_QUEUED: "a question for the user is pending",
    StopCause.ASYNC_DISPATCH: "a background task started",
    StopCause.DASHBOARD_HANDOFF: "a dashboard took over the work",
    StopCause.PAUSE: "the host paused the conversation",
}
_LATER_WIDGET = (
    f"{END_TURN_MARKER} — not queued: a question you asked in an earlier message "
    "is already pending. End your turn now; ask this one after the user has "
    "answered."
)
_TURN_ENDED = "The turn this call belonged to has ended."
_TIMED_OUT = (
    "This call outlasted the agent's tool-call timeout, so AgentFoundation did "
    "not apply its result; the next turn's host notice says what happened."
)
# How long a widget call waits for the vendor event that tells which
# assistant message its tool use belongs to.
_MESSAGE_WAIT_S = 2.0
# A result must reach the vendor before its tool-call timeout: a call counts
# as timed out this much earlier (at most a tenth of the timeout).
_DEADLINE_MARGIN_S = 1.0
_ENDED_WHILE_RUNNING = (
    "The turn this call belonged to ended while the tool ran, so AgentFoundation "
    "did not apply its result."
)

# SOP-control tool → its slash command, the tool name phase-completion checks
# see for the same operation in the text protocol.
_SOP_COMMANDS = {
    "enter_sop": "sop",
    "resume_sop": "resume_sop",
    "pause_sop": "pause_sop",
    "exit_sop": "exit_sop",
    "sop_status": "status",
}


@dataclass(frozen=True)
class BridgeResult:
    text: str
    is_error: bool = False


@dataclass(frozen=True)
class _Call:
    """One bridge call. ``tool_use`` is the AF tool use the vendor's pre-tool
    hook announced for it, or None where a backend cannot attribute calls;
    ``deadline`` (event-loop time) is when the vendor stops waiting for it."""

    tool_use: Optional[str]
    deadline: Optional[float] = None

    def expired(self) -> bool:
        return (
            self.deadline is not None
            and asyncio.get_running_loop().time() >= self.deadline
        )


@dataclass(frozen=True)
class _Outcome:
    """What a tool body produced. ``apply`` applies it to the host and is
    called only while the result still counts; ``text`` is the output of work
    the body already did, None when ``apply`` does all of it."""

    apply: Callable[[], BridgeResult]
    text: Optional[str] = None


class BridgeHost(WidgetHost, Protocol):
    """What the bridge needs from the native orchestrator."""

    native_tool_result_max_chars: int
    sop_control_tools: Sequence[str]
    expose_tool_argument_form: bool
    _paused: bool

    @property
    def current_turn(self) -> Optional[TurnScope]: ...

    @property
    def yolo_mode(self) -> bool: ...

    def vendor_owns_result_spill(self) -> bool: ...

    def _on_async_tool_done(self, tool_name: str, result: Any) -> None: ...

    def _enter_sop(self, name: str, *, yolo: bool = False) -> tuple: ...

    def _reload_sop_definition(self, state: Any) -> None: ...

    def _set_yolo_mode(self, value: bool) -> None: ...

    def _check_phase_completion(self, tool_name: str = "") -> None: ...

    def _consume_pending_followup(self) -> Optional[str]: ...

    def sop_command_wording(self) -> SopCommandWording: ...

    def get_messages(self) -> list: ...

    async def answer_widgets_autonomously(
        self, tools: list, then_run: list, *, is_live: Callable[[], bool]
    ) -> str: ...

    def vendor_tool_timeout_s(self) -> Optional[float]: ...

    def report_late_tool_result(
        self, tool: str, *, ran: bool, text: Optional[str], timeout_s: float
    ) -> None: ...

    def sop_fingerprint(self) -> tuple: ...

    def render_state_update(self) -> str: ...

    def spill_dir(self) -> Path: ...

    def record_action(self, tool_name: str, text: str) -> None: ...

    def persist_record(self) -> None: ...


def _argument_error(schema: dict[str, Any], args: dict[str, Any]) -> Optional[str]:
    """Validate a call against the tool's JSON schema; required strings must
    also be non-empty. Returns an error for the agent, or ``None``."""
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as exc:
        where = "/".join(str(p) for p in exc.absolute_path)
        return f"Invalid arguments{f' at {where}' if where else ''}: {exc.message}"
    missing = [k for k in schema.get("required", []) if args.get(k) in (None, "")]
    if missing:
        return f"Missing required argument(s): {', '.join(missing)}"
    return None


class AFToolBridge:
    def __init__(self, host: BridgeHost, lock: Any) -> None:
        self._host = host
        self._lock = lock
        self._handlers: dict[str, Any] = {}

    async def call(self, name: str, args: dict[str, Any]) -> BridgeResult:
        """Run tool ``name`` (the unprefixed MCP name) for the active turn."""
        handler = self._handlers.get(name)
        if handler is None:
            self.manifest()  # the registry may have changed since the last build
            handler = self._handlers.get(name)
        if handler is None:
            return BridgeResult(f"Unknown AgentFoundation tool: {name}", True)
        return await handler(args or {})

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def manifest(self) -> list[BridgeToolSpec]:
        specs: list[BridgeToolSpec] = []
        formatter = ToolMarkdownFormatter()
        widgets = dict(WIDGET_SCHEMAS)
        if self._host.expose_tool_argument_form:
            widgets[TOOL_ARGUMENT_FORM] = TOOL_ARGUMENT_FORM_SCHEMA[1]
        for tool in self._host.tool_registry.values():
            if not getattr(tool, "agent_enabled", True):
                continue
            if tool.tool_type == "Action":
                specs.append(
                    self._spec(
                        mcp_tool_name(tool.name),
                        formatter.format_tool(tool),
                        action_tool_schema(tool),
                        self._action_handler(tool.name),
                    )
                )
            elif tool.tool_type == "Conversation" and tool.name in widgets:
                specs.append(
                    self._spec(
                        tool.name,
                        (tool.description or "")
                        + (f"\n\n{tool.usage_guidance}" if tool.usage_guidance else ""),
                        widgets[tool.name],
                        self._widget_handler(tool.name, widgets[tool.name]),
                    )
                )
        if TOOL_ARGUMENT_FORM in widgets and all(
            s.name != TOOL_ARGUMENT_FORM for s in specs
        ):
            description, schema = TOOL_ARGUMENT_FORM_SCHEMA
            specs.append(
                self._spec(
                    TOOL_ARGUMENT_FORM,
                    description,
                    schema,
                    self._widget_handler(TOOL_ARGUMENT_FORM, schema),
                )
            )
        for name in self._host.sop_control_tools:
            description, schema = SOP_COMMAND_SCHEMAS[name]
            specs.append(self._spec(name, description, schema, self._sop_handler(name)))
        assert_unique(s.name for s in specs)
        self._handlers = {s.name: s.handler for s in specs}
        return specs

    @staticmethod
    def _spec(name, description, schema, handler) -> BridgeToolSpec:
        return BridgeToolSpec(
            name=name, description=description, input_schema=schema, handler=handler
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _action_handler(self, tool_name: str):
        async def handler(args: dict[str, Any]) -> BridgeResult:
            tool = self._host.tool_registry[tool_name]
            try:
                args = canonical_arguments(tool, args or {})
                arguments = executor_arguments(tool, args)
            except ValueError as exc:
                return BridgeResult(f"Invalid arguments: {exc}", True)
            return await self._run(
                tool_name,
                widget=False,
                schema=action_tool_schema(tool),
                args=args,
                body=lambda turn, call: self._call_action(turn, tool_name, arguments),
            )

        return handler

    def _sop_handler(self, tool: str):
        async def handler(args: dict[str, Any]) -> BridgeResult:
            return await self._run(
                tool,
                widget=False,
                schema=SOP_COMMAND_SCHEMAS[tool][1],
                args=args,
                body=lambda turn, call: self._call_sop(turn, tool, args),
            )

        return handler

    def _widget_handler(self, tool_type: str, schema: dict[str, Any]):
        async def handler(args: dict[str, Any]) -> BridgeResult:
            return await self._run(
                tool_type,
                widget=True,
                schema=schema,
                args=args,
                body=lambda turn, call: self._call_widget(turn, call, tool_type, args),
            )

        return handler

    async def _refusal(
        self, turn: Optional[TurnScope], widget: bool, call: Optional[_Call]
    ) -> Optional[BridgeResult]:
        """Why ``call`` may not run now, or None. A call refused because its
        turn is ending is not a failure: its result says it did not run and
        asks to end the turn, like every end-turn result, and is not an error
        — Claude Code runs ``PostToolUseFailure`` for an error result and
        ignores a stop from it, so only ``PostToolUse`` can end the turn when
        the refused call is its message's last AF call."""
        if turn is None or call is None:
            return BridgeResult("No active AgentFoundation turn; tool refused.", True)
        cause = turn.stop_cause
        if cause is None:
            return None
        if not widget or cause is not StopCause.WIDGET_QUEUED:
            return BridgeResult(
                f"{END_TURN_MARKER} — not run: the host already ended this turn "
                f"({_STOP_REASONS[cause]}). End your turn now and make no further "
                "tool calls."
            )
        if call.tool_use is not None and turn.widget_tool_use is not None:
            await turn.wait_known(call.tool_use, _MESSAGE_WAIT_S)
        if turn.accepts_widget(call.tool_use):
            return None
        return BridgeResult(_LATER_WIDGET)

    async def _run(
        self,
        name: str,
        *,
        widget: bool,
        schema: dict[str, Any],
        args: dict[str, Any],
        body: Callable[[TurnScope, _Call], Awaitable[_Outcome]],
    ) -> BridgeResult:
        turn = self._host.current_turn
        call = self._new_call(turn)
        refused = await self._refusal(turn, widget, call)
        if refused is not None:
            return refused
        error = _argument_error(schema, args or {})
        if error:
            return BridgeResult(error, True)
        token = enter_run(turn.run_ctx)
        try:
            async with self._lock:
                # Re-check under the lock: a call queued behind another may find
                # the turn over or its gate closed by the time it runs.
                if self._host.current_turn is not turn:
                    return BridgeResult(_TURN_ENDED, True)
                refused = await self._refusal(turn, widget, call)
                if refused is not None:
                    return refused
                if call.expired():
                    return self._timed_out(name, None)
                before = self._host.sop_fingerprint()
                outcome = await body(turn, call)
                # The vendor told the model this call failed: applying it now
                # would change state behind the model's back.
                if call.expired():
                    return self._timed_out(name, outcome)
                if self._host.current_turn is not turn:
                    logger.warning(
                        "AF tool %s returned after its turn ended; its result "
                        "is not applied",
                        name,
                    )
                    return BridgeResult(_ENDED_WHILE_RUNNING, True)
                done = outcome.apply()
                result = BridgeResult(
                    self._finish(turn, done.text, before), done.is_error
                )
                self._host.persist_record()
        except Exception as exc:  # reported to the agent, never raised into the vendor
            logger.exception("AF tool call failed")
            return BridgeResult(f"Error: {exc}", True)
        finally:
            exit_run(token)
        return result

    def _new_call(self, turn: Optional[TurnScope]) -> Optional[_Call]:
        if turn is None:
            return None
        timeout = self._host.vendor_tool_timeout_s()
        deadline = None
        if timeout:
            margin = min(_DEADLINE_MARGIN_S, timeout / 10)
            deadline = asyncio.get_running_loop().time() + timeout - margin
        return _Call(tool_use=turn.current_af_tool_use, deadline=deadline)

    def _timed_out(self, name: str, outcome: Optional[_Outcome]) -> BridgeResult:
        """A call whose vendor timeout passed: nothing more of it is applied,
        and the next turn's notice tells the model what did happen."""
        ran = outcome is not None and outcome.text is not None
        logger.warning(
            "AF tool %s %s after the vendor's tool-call timeout; its result is "
            "not applied",
            name,
            "finished" if ran else "was due to run",
        )
        self._host.report_late_tool_result(
            MCP_PREFIX + mcp_tool_name(name),
            ran=ran,
            text=outcome.text if ran else None,
            timeout_s=self._host.vendor_tool_timeout_s() or 0.0,
        )
        return BridgeResult(_TIMED_OUT, True)

    async def _call_action(
        self, turn: TurnScope, tool_name: str, args: dict
    ) -> _Outcome:
        outcome = await execute(
            self._host,
            tool_name,
            dict(args),
            run_ctx=turn.run_ctx,
            on_async_done=self._host._on_async_tool_done,
        )
        return _Outcome(
            apply=lambda: self._apply_action(turn, tool_name, outcome),
            text=outcome.text,
        )

    def _apply_action(
        self, turn: TurnScope, tool_name: str, outcome: ToolOutcome
    ) -> BridgeResult:
        outcome = apply_outcome(self._host, outcome)
        self._host.record_action(tool_name, outcome.text)
        if outcome.is_async:
            turn.request_stop(StopCause.ASYNC_DISPATCH)
            return BridgeResult(f"{_ASYNC_STARTED}\n\n{outcome.text}")
        return BridgeResult(outcome.text, outcome.error is not None)

    async def _call_sop(self, turn: TurnScope, tool: str, args: dict) -> _Outcome:
        return _Outcome(apply=lambda: self._apply_sop(tool, args))

    def _apply_sop(self, tool: str, args: dict) -> BridgeResult:
        host = self._host
        controller = host.sop_controller
        if tool == "enter_sop":
            text = controller.enter(
                str(args["name"]).strip(),
                yolo=bool(args.get("yolo")),
                fresh=bool(args.get("fresh")),
                request=str(args.get("request") or ""),
                build_state=host._enter_sop,
                yolo_mode_setter=host._set_yolo_mode,
                wording=host.sop_command_wording(),
            )
        elif tool == "resume_sop":
            text = controller.resume(
                str(args.get("name") or "").strip(),
                request=str(args.get("request") or ""),
                reload=host._reload_sop_definition,
            )
        elif tool == "pause_sop":
            text = controller.cmd_pause_sop()
        elif tool == "exit_sop":
            text = controller.cmd_exit_sop(wording=host.sop_command_wording())
        else:
            text = controller.cmd_status_summary(len(host.get_messages()))
        host._check_phase_completion(tool_name=_SOP_COMMANDS[tool])
        # The tool result already states the follow-up ("Starting on: …"); it
        # must not linger as a pending follow-up for a later command.
        host._consume_pending_followup()
        host.record_action(tool, text)
        return BridgeResult(text)

    async def _call_widget(
        self, turn: TurnScope, call: _Call, tool_type: str, args: dict
    ) -> _Outcome:
        args = dict(args)
        output = args.pop("output", None) or []
        if isinstance(output, str):
            output = [output]
        then_run = args.pop("then_run", None)
        tool = tool_invocation_to_conversation_tool(
            {"name": tool_type, "arguments": args, "output": list(output)}
        )
        widget_core.prepare(self._host, [tool])
        actions = [then_run] if isinstance(then_run, dict) else []
        if self._host.yolo_mode:
            text = await self._host.answer_widgets_autonomously(
                [tool],
                actions,
                is_live=lambda: self._host.current_turn is turn and not call.expired(),
            )
            return _Outcome(apply=lambda: BridgeResult(text), text=text)
        widget = QueuedWidget(
            tool=tool, then_run=actions, round_context=turn.round_context
        )
        return _Outcome(apply=lambda: self._queue_widget(turn, call, widget))

    @staticmethod
    def _queue_widget(
        turn: TurnScope, call: _Call, widget: QueuedWidget
    ) -> BridgeResult:
        turn.queue_widget(widget, call.tool_use)
        return BridgeResult(_WIDGET_QUEUED)

    # ------------------------------------------------------------------
    # Result shaping
    # ------------------------------------------------------------------

    def _finish(self, turn: TurnScope, text: str, before: tuple) -> str:
        output = neutralize_host_tags(str(text or ""))
        state = ""
        if self._host.sop_fingerprint() != before:
            state = "\n\n" + self._host.render_state_update()
        if self._host._paused:
            turn.request_stop(StopCause.PAUSE)
        directive = ""
        if turn.gate_closed and not output.startswith(END_TURN_MARKER):
            directive = f"{END_TURN_MARKER} — end your turn after this result.\n\n"
        return self._sized(directive, output, state)

    def _sized(self, directive: str, output: str, state: str) -> str:
        """``directive + output + state`` within ``native_tool_result_max_chars``
        UTF-16 units (how Claude Code measures ``maxResultSizeChars``): the
        end-turn directive leads and the state update (L3) ends the result,
        both whole, and the tool output gets the room left, cut where it says
        so and spilled whole to a private file it names. Only a state update
        that does not fit beside that cut is cut and spilled too.

        A vendor that spills oversized results itself shows the model only
        their head (Claude Code: a 2,000-character preview and the file path),
        which keeps the directive but hides a state update at the end; so it
        gets a result whole only when it carries no state update."""
        limit = self._host.native_tool_result_max_chars
        whole = directive + output + state
        if not limit or utf16_len(whole) <= limit:
            return whole
        if not state and self._host.vendor_owns_result_spill():
            return whole
        room = limit - utf16_len(directive)
        output_path = self._spill_path("tool_result")
        output_mark = f"\n... (truncated; full output: {output_path})"
        state_room = room - min(utf16_len(output), utf16_len(output_mark))
        if utf16_len(state) > state_room:
            state_path = self._spill_path("state_update")
            state_mark = f"\n... (truncated; full state update: {state_path})"
            write_private_file(state_path, state.lstrip("\n"))
            state = utf16_head(state, state_room - utf16_len(state_mark)) + state_mark
        room -= utf16_len(state)
        if utf16_len(output) > room:
            write_private_file(output_path, output)
            output = utf16_head(output, room - utf16_len(output_mark)) + output_mark
        return directive + output + state

    def _spill_path(self, kind: str) -> Path:
        directory = self._host.spill_dir()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        return directory / f"{kind}_{uuid.uuid4().hex[:12]}.txt"

"""Per-turn state shared by the turn driver, the session actor and the
AgentFoundation tool bridge."""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class TurnOrigin(str, enum.Enum):
    """Why a vendor turn is being submitted (surfaced to the model in L2)."""

    USER = "user"
    WIDGET_ANSWER = "widget_answer"
    TOOL_COMPLETION = "tool_completion"
    HOST_EVENT = "host_event"
    VALIDATION_RETRY = "validation_retry"
    RESUMED_TURN = "resumed_turn"


class SubmissionState(str, enum.Enum):
    """Durable submission status of the last vendor turn."""

    PREPARED = "prepared"  # definitely not submitted
    SUBMITTED = "submitted"  # may have been accepted by the vendor
    COMMITTED = "committed"  # the vendor finished the turn
    INTERRUPTED = "interrupted"  # cancelled, vendor acknowledged
    UNCERTAIN = "uncertain"  # cancelled or failed without acknowledgement


class StopCause(str, enum.Enum):
    """Why the current vendor turn must end after the in-flight AF calls."""

    WIDGET_QUEUED = "widget_queued"
    ASYNC_DISPATCH = "async_dispatch"
    DASHBOARD_HANDOFF = "dashboard_handoff"
    PAUSE = "pause"


@dataclass
class QueuedWidget:
    """A conversation tool the agent asked for during the turn; presented to
    the user after the vendor turn ends."""

    tool: Any  # ConversationTool
    then_run: list[dict[str, Any]] = field(default_factory=list)
    round_context: Optional[dict[str, Any]] = None


@dataclass
class TurnScope:
    """Everything the bridge and hooks need about the active vendor turn.

    Bridge handlers run in tasks the vendor SDK/server spawns, so they never
    read turn state from context variables; they look it up here.
    """

    run_ctx: Any
    interactive: Any
    turn_number: int
    origin: TurnOrigin
    l2_text: str = ""
    stop_cause: Optional[StopCause] = None
    pending_widgets: list[QueuedWidget] = field(default_factory=list)
    open_af_tool_uses: set[str] = field(default_factory=set)
    done_af_tool_uses: set[str] = field(default_factory=set)
    known_af_tool_uses: set[str] = field(default_factory=set)
    # Assistant message of each AF tool use, where message events carry ids.
    af_tool_messages: dict[str, str] = field(default_factory=dict)
    # The AF tool use the vendor announced last (Claude Code's pre-tool hook,
    # Codex's MCP request); None on backends that announce none (Devmate:
    # their calls cannot be attributed).
    current_af_tool_use: Optional[str] = None
    # The announced tool use of the call that queued the first widget.
    widget_tool_use: Optional[str] = None
    # AF message events observed since the first widget was queued.
    af_messages_since_widget: int = 0
    # The vendor was told to stop after the last AF call of a message.
    stop_signalled: bool = False
    round_context: Optional[dict[str, Any]] = None
    completed_actions: list[Any] = field(default_factory=list)
    compacted: bool = False
    # Outbox entries [0, outbox_upto) were delivered with this turn's L2.
    outbox_upto: int = 0
    # SOP-state hash of the L2 sent with this turn ("" when none was sent).
    l2_hash: str = ""
    # SOP-state hash the model learned from an L3 update during this turn.
    l3_state_hash: str = ""
    # Set by the first event the vendor produced for this turn.
    accepted: bool = False
    _af_ids_changed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def gate_closed(self) -> bool:
        return self.stop_cause is not None

    def request_stop(self, cause: StopCause) -> None:
        if self.stop_cause is None:
            self.stop_cause = cause

    def queue_widget(self, widget: QueuedWidget, tool_use: Optional[str]) -> None:
        """Queue a widget for the end of the turn; the first one decides which
        assistant message later widget calls must come from."""
        if self.stop_cause is None:
            self.widget_tool_use = tool_use
            self.af_messages_since_widget = 0
        self.pending_widgets.append(widget)
        self.request_stop(StopCause.WIDGET_QUEUED)

    def accepts_widget(self, tool_use: Optional[str]) -> bool:
        """Whether a widget call (``tool_use``: its announced tool use) may
        join the compound widget: only calls from the assistant message that
        queued the first widget do; any call after another stop cause is
        refused.

        * Announced calls compare the messages their tool uses belong to.
          Claude Code announces a call in its pre-tool hook, and its message
          events carry the API message id of each tool use; Codex's MCP
          request names the call and the model output item (the code-mode
          ``exec`` call) that made it, announced as that call's message.
          Where message events carry no ids, announced calls join until the
          vendor was told to stop after the last AF call of that message.
        * Calls a backend cannot attribute (Devmate: no pre-tool hook) join
          until the vendor reports any AF message after the first widget was
          queued: dm reports a step (one model response) only after all of
          its tool calls ran, so a later call belongs to a later step.
        """
        if self.stop_cause is None:
            return True
        if self.stop_cause is not StopCause.WIDGET_QUEUED:
            return False
        if self.widget_tool_use is None or tool_use is None:
            return self.af_messages_since_widget == 0
        first = self.af_tool_messages.get(self.widget_tool_use)
        this = self.af_tool_messages.get(tool_use)
        if first is not None and this is not None:
            return first == this
        return not self.stop_signalled

    def stop_now(self) -> bool:
        """True once a stop was requested and no AF call of the current
        assistant message is still running."""
        return self.stop_cause is not None and not self.open_af_tool_uses

    def note_af_call(self, tool_use_id: str) -> None:
        """The vendor announced the AF call it runs next (a pre-tool hook, or
        the call's own MCP request)."""
        self.current_af_tool_use = tool_use_id
        self.open_af_tool_uses.add(tool_use_id)

    def finish_af_call(self, tool_use_id: str) -> bool:
        """An AF call finished; returns whether the vendor must stop now
        (``stop_now``), which is then recorded as signalled."""
        self.open_af_tool_uses.discard(tool_use_id)
        self.done_af_tool_uses.add(tool_use_id)
        if self.current_af_tool_use == tool_use_id:
            self.current_af_tool_use = None
        stop = self.stop_now()
        self.stop_signalled = self.stop_signalled or stop
        return stop

    def note_af_message(
        self, tool_use_ids: tuple[str, ...], message_id: Optional[str] = None
    ) -> None:
        """Record AF tool calls of an assistant message and wake waiters. A
        vendor may report one message in several events (Claude Code: one per
        content block, sharing ``message_id``)."""
        self.known_af_tool_uses.update(tool_use_ids)
        self.open_af_tool_uses.update(set(tool_use_ids) - self.done_af_tool_uses)
        if message_id:
            self.af_tool_messages.update(dict.fromkeys(tool_use_ids, message_id))
        if self.stop_cause is StopCause.WIDGET_QUEUED:
            self.af_messages_since_widget += 1
        changed, self._af_ids_changed = self._af_ids_changed, asyncio.Event()
        changed.set()

    async def wait_known(self, tool_use_id: str, timeout: float) -> None:
        """Wait (bounded) until the message carrying ``tool_use_id`` was seen,
        so a stop decision covers every AF call of that message."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while tool_use_id not in self.known_af_tool_uses:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(self._af_ids_changed.wait(), remaining)
            except asyncio.TimeoutError:
                return

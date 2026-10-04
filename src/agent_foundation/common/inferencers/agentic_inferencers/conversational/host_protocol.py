"""Contracts between conversation hosts and conversational orchestrators.

Two orchestrators implement ``ConversationalHost``:

* ``ConversationalInferencer`` — AgentFoundation owns the agent loop and talks
  to the model through a text protocol.
* ``NativeConversationalInferencer`` — a vendor agent (Claude Code, Devmate,
  Codex, Metamate) owns the loop and the session; AgentFoundation contributes
  instructions, SOP state and tools.

Hosts (OpenStartup, the SOP CLI, the flow-node adapter, programmatic callers)
depend on ``ConversationalHost``. A capability beyond it is used only after
its ``supports_*`` flag says the host has it — never through private
attributes or the concrete class:

========================  ======================  ===  ======
flag                      capability protocol     CI   native
========================  ======================  ===  ======
supports_widget_recovery  SupportsWidgetRecovery  yes  yes
supports_round_resume     SupportsRoundResume     yes  no
supports_inbox            SupportsInbox           yes  yes
supports_prompt_manifest  SupportsPromptManifest  yes  yes
supports_flow_node        SupportsFlowNode        yes  no
supports_rewind           SupportsRewind          no   yes
========================  ======================  ===  ======

``ConversationHostState`` lists the attributes the shared mixins
(``ConversationStateMixin``, ``SOPCommandsMixin``, ``ToolDispatchMixin``,
``ConversationToolsMixin``) read and write; every orchestrator declares them.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    WidgetMailboxes,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_dispatch import (
    AsyncToolTasks,
)

# Host callbacks of ``run_agentic_loop``; a failing callback is logged and the
# turn continues.
OnNewTurn = Callable[[int, str], Awaitable[Optional[int]]]
OnPromptRendered = Callable[[Any, str], Awaitable[None]]
OnTurnComplete = Callable[[int], Awaitable[None]]
OnRoundStart = Callable[[int, int], Awaitable[Optional[dict[str, Any]]]]
OnRoundComplete = Callable[[Any, int, int, str, str, str, Any], Awaitable[None]]


@runtime_checkable
class ConversationalHost(Protocol):
    """Host-facing core surface shared by both orchestrators."""

    supports_widget_recovery: bool
    supports_round_resume: bool
    supports_inbox: bool
    supports_prompt_manifest: bool
    supports_flow_node: bool
    supports_rewind: bool

    prior_context: dict[str, Any]
    tool_registry: dict[str, Any]
    interactive: Any
    sop_controller: Any
    # The active SOP's state (``None`` outside an SOP).
    sop_state: Any
    # Paused or exited SOPs, most recent first; reading gives a copy.
    suspended_sops: list
    # Extra SOP discovery directories; reading gives a copy.
    extra_sop_dirs: list
    # The host's tool dispatcher; reads fall back to the tool executor.
    tool_dispatcher: Any
    # Where the host wants this turn's streaming artifacts.
    cache_folder: Optional[str]

    async def run_agentic_loop(
        self,
        content: str,
        *,
        run_context: Any = None,
        interactive: Any = None,
        session_id: str = "",
        turn_number: int = 0,
        origin: str = "user",
        on_new_turn: Optional[OnNewTurn] = None,
        on_prompt_rendered: Optional[OnPromptRendered] = None,
        on_turn_complete: Optional[OnTurnComplete] = None,
        on_round_start: Optional[OnRoundStart] = None,
        on_round_complete: Optional[OnRoundComplete] = None,
    ) -> AgenticResult:
        """Run one host turn for ``content``.

        ``origin`` says why the turn runs: ``"user"``, ``"tool_completion"``
        (a background tool finished) or ``"host_event"``. Callbacks:

        * ``on_new_turn(turn_number, user_input)`` when a host turn starts —
          the call's first, and one per answered widget batch; a returned
          number renumbers the turn.
        * ``on_round_start(iteration, turn_number)`` before a round (one model
          message); a returned dict is the round's context, and its
          ``cache_folder`` (if any) becomes ``cache_folder``.
        * ``on_prompt_rendered(host, response_text)`` once the model answered
          what ``last_prompt_data()`` describes (after each model call of the
          text protocol, after each vendor turn of a native host).
        * ``on_round_complete(host, iteration, turn_number, raw_text,
          clean_text, display_text, conversation_response)`` when a round's
          message is complete, before its widgets are shown.
        * ``on_turn_complete(iterations)`` when the turn's rounds are done.
        """
        ...

    def accepts_command(self, text: str) -> bool:
        """Whether ``text`` is a slash command the orchestrator handles."""
        ...

    def set_prior_context(self, ctx: dict[str, Any]) -> None: ...

    def update_prior_context(self, **kwargs: Any) -> None: ...

    def set_session_variables(
        self, variables: dict[str, Any], *, tool_type: Optional[str] = None
    ) -> None: ...

    def get_messages(self) -> list[dict[str, Any]]:
        """The transcript mirror (a copy)."""
        ...

    def set_messages(self, messages: list) -> None: ...

    def add_message(self, role: str, content: str) -> None: ...

    def next_required_tools(self) -> set[str]:
        """Required tools of the next available SOP phase."""
        ...

    def check_phase_completion(self, tool_name: str = "") -> None:
        """Advance the active SOP if its current phase is now complete (e.g.
        after the host recorded a phase output itself)."""
        ...

    def export_state(self, *, turn_number: int = 0, iteration: int = 0) -> dict:
        """A JSON-safe snapshot of the conversation at a round."""
        ...

    def restore_state(self, state: dict, *, reattach_sop: bool = True) -> None:
        """Restore an ``export_state`` snapshot; ``reattach_sop=False`` keeps
        the SOP state the host already attached."""
        ...

    @property
    def effective_cwd(self) -> str: ...

    def enable_debug_mode(self) -> None: ...

    def reset_for_flow_invocation(self) -> None:
        """Start a fresh conversation (transcript, dynamic context and, for a
        native host, the vendor session)."""
        ...

    async def aclose(self) -> None:
        """Release the orchestrator's resources (idempotent)."""
        ...


@runtime_checkable
class SupportsWidgetRecovery(Protocol):
    """Answers a widget shown before a reconnect or restart."""

    supports_widget_recovery: bool

    def set_pending_widget_answer(self, payload: dict) -> None:
        """Re-arm a persisted widget's answer: the next ``run_agentic_loop``
        decodes it against that exact widget, without asking the model again."""
        ...


@runtime_checkable
class SupportsRoundResume(Protocol):
    """Resumes a turn at a round: after ``restore_state(export_state(
    turn_number=t, iteration=i))`` the next ``run_agentic_loop`` continues turn
    ``t`` at round ``i`` with the content that round rendered with. A host
    without it re-runs the whole turn with the turn's original user input."""

    supports_round_resume: bool


@runtime_checkable
class SupportsInbox(Protocol):
    """Long-lived inbox mode: one host turn per queued item until shutdown."""

    supports_inbox: bool

    def enable_inbox(
        self,
        interactive: Any = None,
        *,
        auto_shutdown_on_sop_complete: bool = False,
        maxsize: int = 0,
        on_new_turn: Optional[OnNewTurn] = None,
        on_prompt_rendered: Optional[OnPromptRendered] = None,
        on_turn_complete: Optional[OnTurnComplete] = None,
    ) -> None: ...

    def inbox_put(self, item: object) -> None: ...

    def inbox_put_user(self, content: str, source: str = "user") -> None: ...

    def request_shutdown(self) -> None: ...

    @property
    def shutdown_requested(self) -> bool: ...

    async def run(self, *, run_context: Any = None) -> Optional[AgenticResult]: ...


@runtime_checkable
class SupportsPromptManifest(Protocol):
    """Describes what the model was last given ("View Prompt")."""

    supports_prompt_manifest: bool

    def last_prompt_data(self) -> dict[str, Any]:
        """``rendered_prompt``, ``template_source``, ``template_feed`` and
        ``template_config`` of the last model input."""
        ...


@runtime_checkable
class SupportsFlowNode(Protocol):
    """Runs as a flow node: the transcript mirror and the dynamic context
    are the whole conversation, so a flow checkpoints and restores them
    (``get_messages``/``set_messages`` and ``dynamic_context``)."""

    supports_flow_node: bool
    dynamic_context: Any


@runtime_checkable
class SupportsRewind(Protocol):
    """Rewinds its own agent session to a turn boundary; the host calls it
    before it truncates its history, and truncates nothing if it fails."""

    supports_rewind: bool

    async def rewind_to(self, turn_number: int) -> None: ...


class ConversationHostState(Protocol):
    """Attributes the shared conversational mixins rely on."""

    prior_context: dict[str, Any]
    tool_registry: dict[str, Any]
    tool_executor: Any
    prompt_renderer: Any
    interactive: Any
    handler_registry: Any
    sop_controller: Any
    dashboard_coordinator: Any
    workflow_manager: Any
    yolo_mode: bool
    allowed_sops: list
    disallowed_sops: list
    extra_sop_dirs: list
    mailboxes: WidgetMailboxes
    async_tool_tasks: AsyncToolTasks
    _messages: list
    _dynamic_context: Any
    _last_template_feed: dict[str, Any]
    _commands: Any
    _inbox: Any

    def _workspace_root(self) -> str: ...

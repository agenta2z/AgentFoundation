"""NativeConversationalInferencer — the vendor agent runs the conversation.

Claude Code / Devmate / Codex / Metamate keep their own session, transcript,
agent loop, tool calling and compaction. AgentFoundation contributes, through
documented vendor channels:

* L1 session instructions (appended to the vendor system prompt, once),
* L2 per-turn SOP context and L3 state updates in AF tool results,
* its tools (action tools, SOP control, conversation widgets) over MCP,

and keeps its domain state (SOPs, widgets, host record) and the host contract
(``host_protocol.ConversationalHost``). Shared behavior (SOP commands, tool
dispatch, the widget lifecycle) comes from the same mixins and functional
cores (``tool_dispatch``, ``sop_feed``, ``widget_core``) the text-protocol
``ConversationalInferencer`` uses.
"""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational import (
    widget_core,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    command,
    CommandRegistry,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticDynamicContext,
    AgenticResult,
    CompletedAction,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_state import (
    ConversationStateMixin,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_host import (
    ConversationToolsMixin,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.dashboard_coordinator import (
    DashboardCoordinator,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    WidgetMailboxes,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_registry import (
    ConversationToolHandlerRegistry,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers import (
    default_registry,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.host_protocol import (
    OnNewTurn,
    OnPromptRendered,
    OnRoundComplete,
    OnRoundStart,
    OnTurnComplete,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox import (
    ToolCompletion,
    UserMessage,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox_driver import (
    inbox_item_content,
    InboxDriver,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocol_text import (
    TOOL_RESULTS_PREFIX,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_commands import (
    SOPCommandsMixin,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
    SOPController,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_dispatch import (
    AsyncToolTasks,
    ToolDispatchMixin,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.schema import (
    SOP_COMMAND_SCHEMAS,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    AFToolBridge,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.context.composer import (
    neutralize_host_tags,
    TurnComposer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    NativeBackendSpec,
    NativeSessionBackend,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.factory import (
    normalize_spec,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.record import (
    InMemoryRecordStore,
    NativeSessionRecord,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.runtime import (
    NativeRuntimeManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.turn import (
    StopCause,
    TurnScope,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.turn_loop import (
    NativeTurnLoopMixin,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import active_run_context
from agent_foundation.ui.interactive_base import InteractiveBase
from attr import attrib, attrs

logger: logging.Logger = logging.getLogger(__name__)

# The largest ``maxResultSizeChars`` Claude Code honours for an MCP tool
# (claude 2.1.288); it spills a larger result to a file and shows the model
# only its head, which would hide the state update ending an AF tool result.
_VENDOR_RESULT_SIZE_CAP = {"claude_sdk": 500_000, "claude_cli": 500_000}


@attrs(slots=False)
class NativeConversationalInferencer(
    ConversationStateMixin,
    SOPCommandsMixin,
    ToolDispatchMixin,
    ConversationToolsMixin,
    NativeTurnLoopMixin,
    InferencerBase,
):
    """Drop-in host-facing sibling of ``ConversationalInferencer`` whose agent
    loop is owned by a vendor agent session."""

    _SUPPORTS_BTA_FANOUT = False

    supports_round_resume = False
    supports_widget_recovery = True
    supports_inbox = True
    supports_prompt_manifest = True
    # The vendor session owns the transcript a flow node checkpoints (D14).
    supports_flow_node = False
    # rewind_to(turn): the host rewinds the vendor session before truncating.
    supports_rewind = True

    # --- Backend ------------------------------------------------------------
    backend: Any = attrib(kw_only=True)  # NativeBackendSpec | dict
    backend_factory: Optional[Callable[..., NativeSessionBackend]] = attrib(
        default=None, kw_only=True
    )
    runtime_manager: Optional[NativeRuntimeManager] = attrib(default=None, kw_only=True)
    record_store: Any = attrib(factory=InMemoryRecordStore, kw_only=True)
    conversation_key: Optional[str] = attrib(default=None, kw_only=True)
    principal: str = attrib(factory=getpass.getuser, kw_only=True)

    # --- Host-provided domain wiring (same names as ConversationalInferencer) ---
    interactive: Optional[InteractiveBase] = attrib(default=None, kw_only=True)
    tool_registry: dict[str, Any] = attrib(factory=dict, kw_only=True)
    tool_executor: Any = attrib(default=None, kw_only=True)
    prompt_renderer: Any = attrib(default=None, kw_only=True)
    prior_context: dict[str, Any] = attrib(factory=dict, kw_only=True)
    workflow_manager: Any = attrib(default=None, kw_only=True)
    handler_registry: ConversationToolHandlerRegistry = attrib(
        factory=default_registry, kw_only=True
    )
    # Init kwargs ``extra_sop_dirs``, ``allowed_sops`` and ``disallowed_sops``
    # only seed the SOP controller, which owns the lists: read and assign
    # them under those names.
    _initial_extra_sop_dirs: list = attrib(
        factory=list, kw_only=True, alias="extra_sop_dirs", repr=False
    )
    _initial_allowed_sops: list = attrib(
        factory=list, kw_only=True, alias="allowed_sops", repr=False
    )
    _initial_disallowed_sops: list = attrib(
        factory=list, kw_only=True, alias="disallowed_sops", repr=False
    )
    # SOP-control tools offered to the agent; [] leaves SOP control to the
    # user's slash commands.
    sop_control_tools: list = attrib(
        factory=lambda: list(SOP_COMMAND_SCHEMAS), kw_only=True
    )
    # Offer the ``tool_argument_form`` question (no tool.json describes it).
    expose_tool_argument_form: bool = attrib(default=False, kw_only=True)

    # --- Policies -----------------------------------------------------------
    # Autonomous mode outside SOPs (inside an SOP its own yolo flag governs).
    session_yolo: bool = attrib(default=False, kw_only=True)
    soft_max_iterations: Optional[int] = attrib(default=30, kw_only=True)
    max_vendor_turns_per_call: int = attrib(default=50, kw_only=True)
    native_tool_result_max_chars: int = attrib(default=16000, kw_only=True)
    on_l1_drift: str = attrib(default="notice", kw_only=True)  # notice | rotate | fail
    on_session_loss: str = attrib(default="recap", kw_only=True)  # recap | fresh | fail
    rewind_on_repeat_turn: bool = attrib(default=False, kw_only=True)
    # A rewind the backend cannot fork exactly: fail (RewindUnsupported) or
    # continue in a fresh session with a recap.
    on_rewind_unsupported: str = attrib(default="fail", kw_only=True)  # fail | recap
    recap_max_chars: int = attrib(default=30000, kw_only=True)
    # How long to wait for the vendor to acknowledge an interrupt and drain.
    vendor_drain_timeout_s: float = attrib(default=30.0, kw_only=True)
    # A vendor turn with no events for this long while no AF tool runs is
    # interrupted (built-in tools such as long shell commands also produce no
    # events, so this is generous); <= 0 disables the watchdog.
    vendor_stall_timeout_s: float = attrib(default=1800.0, kw_only=True)
    # Idle close for a private runtime manager (a host-injected manager keeps
    # its own setting).
    idle_close_seconds: float = attrib(default=1800.0, kw_only=True)
    # Directory for this conversation's private files (L1, spills, CLI config);
    # defaults to a per-conversation temp directory.
    native_session_dir: Optional[str] = attrib(default=None, kw_only=True)
    # The host backgrounds async tools itself and reports their completion
    # through its own channel (OpenStartup's dispatcher): no outbox notice.
    host_manages_async_results: bool = attrib(default=False, kw_only=True)

    # --- Internal state ------------------------------------------------------
    dashboard_coordinator: Optional[DashboardCoordinator] = attrib(
        default=None, init=False
    )
    sop_controller: Optional[SOPController] = attrib(default=None, init=False)
    _messages: list[dict[str, Any]] = attrib(factory=list, init=False)
    _dynamic_context: AgenticDynamicContext = attrib(
        factory=AgenticDynamicContext, init=False
    )
    _last_rendered_prompt: str = attrib(default="", init=False)
    _last_template_source: str = attrib(default="", init=False)
    _last_template_feed: dict[str, Any] = attrib(factory=dict, init=False)
    _last_template_config: dict[str, Any] = attrib(factory=dict, init=False)
    mailboxes: WidgetMailboxes = attrib(factory=WidgetMailboxes, init=False)
    async_tool_tasks: AsyncToolTasks = attrib(factory=AsyncToolTasks, init=False)
    _pending_widget_result: Optional[dict[str, Any]] = attrib(default=None, init=False)
    _record: Optional[NativeSessionRecord] = attrib(default=None, init=False)
    _current_turn: Optional[TurnScope] = attrib(default=None, init=False)
    # The latest turn whose turn context the vendor's prompt hook took.
    _prompt_hook_turn: Optional[TurnScope] = attrib(default=None, init=False)
    _lease_key: Optional[tuple] = attrib(default=None, init=False)
    _owns_runtime: bool = attrib(default=False, init=False)
    # Notice id -> text kept in memory only (the durable record stores refs).
    _notice_bodies: dict[int, str] = attrib(factory=dict, init=False)
    _cache_folder: Optional[str] = attrib(default=None, init=False)
    # The round a restored blob was captured at (a re-armed widget's round).
    _restored_iteration: Optional[int] = attrib(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        self._validate_bta_inferencer_spec()
        # A turn that reached the vendor must never be re-sent by the generic
        # retry/fallback machinery (no automatic replay).
        self.max_retry = 0
        self.backend = normalize_spec(self.backend)
        if not self.backend.cwd:
            self.backend.cwd = os.getcwd()
        self._validate_backend()
        unknown = sorted(set(self.sop_control_tools) - set(SOP_COMMAND_SCHEMAS))
        if unknown:
            raise ValueError(
                f"Unknown SOP-control tool(s) {unknown!r}; available: "
                f"{list(SOP_COMMAND_SCHEMAS)!r}"
            )
        if self.conversation_key is None:
            self.conversation_key = uuid.uuid4().hex
        if self.runtime_manager is None:
            self.runtime_manager = NativeRuntimeManager(
                idle_close_seconds=self.idle_close_seconds
            )
            self._owns_runtime = True
        if self.prompt_renderer is None:
            self.prompt_renderer = self._default_prompt_renderer()
        native_tm = self.prompt_renderer.template_manager.switch(
            active_template_root_space="conversation_native",
            active_template_type="main",
        )
        self._composer = TurnComposer(native_tm, self.prompt_renderer.render_string)
        self._commands = CommandRegistry(self)
        self._inbox_driver = InboxDriver(
            self._run_inbox_turn, content_for_item=self._inbox_item_content, log=logger
        )
        self.sop_controller = SOPController(
            extra_sop_dirs=list(self._initial_extra_sop_dirs),
            allowed_sops=list(self._initial_allowed_sops),
            disallowed_sops=list(self._initial_disallowed_sops),
            prompt_renderer_ref=self.prompt_renderer,
            tool_registry=self.tool_registry,
            workflow_manager=self.workflow_manager,
            prior_context_reader=lambda: self.prior_context,
            add_message=self.add_message,
            request_shutdown=self.request_shutdown,
            resolve_tool_name=self._resolve_tool_name,
        )
        self.dashboard_coordinator = DashboardCoordinator(
            tool_registry=self.tool_registry,
            tool_dispatcher=getattr(self, "_tool_dispatcher", None)
            or self.tool_executor,
            prior_context_reader=lambda: self.prior_context,
        )
        missing = [
            t for t in self._required_widget_types() if t not in self.handler_registry
        ]
        if missing:
            raise ValueError(f"handler_registry missing handlers for: {missing!r}")
        self._bridge = AFToolBridge(self, asyncio.Lock())

    def _validate_backend(self) -> None:
        """Fail at construction — before any vendor session or user turn —
        when the backend kind is unknown, cannot honour the configured
        environment, relies on a capability whose evidence the configuration
        does not accept, lacks a qualified channel for per-turn context under
        this configuration, or would cut AF tool results below the
        configured result size."""
        caps = self._backend_caps()
        caps.require_spec(self.backend)
        if caps.preferred_l2_channel(self.backend.l2_envelope_allowed) is None:
            raise NativeCapabilityError(
                f"{self.backend.kind} has no qualified channel for per-turn context "
                f"(supports {[c.value for c in caps.l2_channels]}); set "
                "l2_envelope_allowed: true to opt into a labelled envelope."
            )
        cap = _VENDOR_RESULT_SIZE_CAP.get(caps.kind)
        if cap is not None and self.native_tool_result_max_chars > cap:
            raise ValueError(
                f"native_tool_result_max_chars={self.native_tool_result_max_chars:,} "
                f"exceeds the {cap:,}-character tool result size Claude Code "
                f"honours ({caps.kind}): it would spill larger results itself and "
                f"show only their head. Set it to at most {cap:,}."
            )

    @property
    def bridge(self) -> AFToolBridge:
        return self._bridge

    @property
    def _extra_sop_dirs(self) -> list:
        """The SOP controller's live list (legacy name of ``extra_sop_dirs``)."""
        return self.sop_controller.extra_sop_dirs

    @_extra_sop_dirs.setter
    def _extra_sop_dirs(self, dirs: Any) -> None:
        self.extra_sop_dirs = dirs

    @staticmethod
    def _required_widget_types() -> list:
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
            ConversationToolType,
        )

        return list(ConversationToolType)

    @staticmethod
    def _default_prompt_renderer() -> Any:
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.template_manager_renderer import (
            TemplateManagerPromptRenderer,
        )
        from agent_foundation.resources import PROMPT_TEMPLATES_ROOT
        from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
            TemplateManager,
        )

        return TemplateManagerPromptRenderer(
            template_manager=TemplateManager(
                templates=str(PROMPT_TEMPLATES_ROOT),
                active_template_root_space="conversation_native",
                active_template_type="main",
            ),
            template_key="session",
        )

    # --- Mode -----------------------------------------------------------------

    @property
    def yolo_mode(self) -> bool:
        """Autonomous mode: the active SOP's own flag, else the session flag."""
        state = self.sop_state
        return bool(state.yolo_mode) if state is not None else self.session_yolo

    @yolo_mode.setter
    def yolo_mode(self, value: bool) -> None:
        # SOP entry writes this when entering a yolo SOP; that SOP's state
        # already carries the flag, so only an SOP-less write changes the mode.
        if self.sop_state is None:
            self.session_yolo = bool(value)

    @property
    def spec(self) -> NativeBackendSpec:
        return self.backend

    def _workspace_root(self) -> str:
        return self.backend.cwd

    @property
    def effective_cwd(self) -> str:
        return self.backend.cwd

    @property
    def cache_folder(self) -> Optional[str]:
        """Host-assigned streaming-cache folder; the native orchestrator writes
        no streaming cache of its own."""
        return self._cache_folder

    @cache_folder.setter
    def cache_folder(self, value: Optional[str]) -> None:
        self._cache_folder = value

    # --- Inbox mode (SupportsInbox, shared InboxDriver) -----------------------

    def enable_inbox(
        self,
        interactive: Any = None,
        *,
        auto_shutdown_on_sop_complete: bool = False,
        maxsize: int = 0,
        on_new_turn: Any = None,
        on_prompt_rendered: Any = None,
        on_turn_complete: Any = None,
    ) -> None:
        """Enable the inbox event loop. Must be called before run()."""
        self._inbox_driver.enable(
            interactive=interactive,
            maxsize=maxsize,
            on_new_turn=on_new_turn,
            on_prompt_rendered=on_prompt_rendered,
            on_turn_complete=on_turn_complete,
        )
        self._auto_shutdown_on_sop_complete = auto_shutdown_on_sop_complete

    def inbox_put(self, item: Any) -> None:
        self._inbox_driver.put(item)

    def inbox_put_user(self, content: str, source: str = "user") -> None:
        self.inbox_put(UserMessage(content=content, source=source))

    def request_shutdown(self) -> None:
        self._inbox_driver.request_shutdown()

    @property
    def shutdown_requested(self) -> bool:
        return self._inbox_driver.shutdown_requested

    @property
    def _inbox(self) -> Any:
        # Read by the shared tool dispatch: an async completion wakes the inbox.
        return self._inbox_driver.queue

    async def run(self, *, run_context: Any = None) -> Optional[Any]:
        """Drain the inbox, one host turn per item, until shutdown."""
        return await self._inbox_driver.run(run_context=run_context)

    async def _run_inbox_turn(
        self,
        content: str,
        *,
        origin: str,
        interactive: Any,
        turn_number: int,
        on_new_turn: Any,
        on_prompt_rendered: Any,
        on_turn_complete: Any,
        run_context: Any,
    ) -> Any:
        return await self.run_agentic_loop(
            content,
            origin=origin,
            interactive=interactive,
            turn_number=turn_number,
            on_new_turn=on_new_turn,
            on_prompt_rendered=on_prompt_rendered,
            on_turn_complete=on_turn_complete,
            run_context=run_context,
        )

    @staticmethod
    def _inbox_item_content(item: object) -> Optional[str]:
        """A completed background tool resumes the agent; its result reaches the
        model as a host notice in that turn's context, not as user text."""
        if isinstance(item, ToolCompletion):
            name = item.tool_name or "a background tool"
            return f"The background tool {name} finished; continue with its result."
        return inbox_item_content(item)

    def request_pause(self) -> None:
        """Cooperative pause: the current vendor turn is asked to end after
        its in-flight AF calls; the host call returns a ``PausedResult``."""
        self._paused = True
        if self._current_turn is not None:
            self._current_turn.request_stop(StopCause.PAUSE)

    # =========================================================================
    # Host protocol
    # =========================================================================

    async def run_agentic_loop(
        self,
        content: str,
        *,
        run_context: Any = None,
        interactive: Optional[InteractiveBase] = None,
        session_id: str = "",
        turn_number: int = 0,
        origin: str = "user",
        on_new_turn: Optional[OnNewTurn] = None,
        on_prompt_rendered: Optional[OnPromptRendered] = None,
        on_turn_complete: Optional[OnTurnComplete] = None,
        on_round_start: Optional[OnRoundStart] = None,
        on_round_complete: Optional[OnRoundComplete] = None,
    ) -> AgenticResult:
        """One host turn: one vendor turn, plus one per answered widget batch
        (``ConversationalHost.run_agentic_loop``)."""
        return await super().run_agentic_loop(
            content,
            run_context=run_context,
            interactive=interactive,
            session_id=session_id,
            turn_number=turn_number,
            origin=origin,
            on_new_turn=on_new_turn,
            on_prompt_rendered=on_prompt_rendered,
            on_turn_complete=on_turn_complete,
            on_round_start=on_round_start,
            on_round_complete=on_round_complete,
        )

    def export_state(self, *, turn_number: int = 0, iteration: int = 0) -> dict:
        return self._conversation_blob(turn_number=turn_number, iteration=iteration)

    def _conversation_blob(self, *, turn_number: int = 0, iteration: int = 0) -> dict:
        blob = {
            "schema": "native/v1",
            "messages": list(self._messages),
            "prior_context": self._json_safe_prior_context(),
            "content": "",
            "turn_number": turn_number,
            "iteration": iteration,
            "native": self._record.to_dict() if self._record is not None else None,
        }
        blob.update(self.sop_controller.serialize())
        return blob

    def restore_state(self, state: dict, *, reattach_sop: bool = True) -> None:
        self._messages = list(state.get("messages", []))
        self.prior_context = dict(state.get("prior_context", {}))
        self.mailboxes.clear()
        self.sop_controller.restore(state, reattach_sop=reattach_sop)
        self._paused = False  # a restored conversation runs again (CI parity)
        iteration = state.get("iteration")
        self._restored_iteration = iteration if isinstance(iteration, int) else None
        native = state.get("native")
        if native:
            restored = NativeSessionRecord.from_dict(native)
            stored = self.record_store.load(self.conversation_key)
            self._record = restored if restored.newer_than(stored) else stored

    def accepts_command(self, text: str) -> bool:
        """Our commands, plus the vendor commands the backend passes through
        verbatim (e.g. Claude Code's ``/compact``)."""
        return super().accepts_command(text) or self._is_vendor_command(text)

    def set_pending_widget_answer(self, payload: dict) -> None:
        self._pending_widget_result = payload

    def last_prompt_data(self) -> dict[str, Any]:
        return {
            "rendered_prompt": self._last_rendered_prompt,
            "template_source": self._last_template_source,
            "template_feed": self._last_template_feed,
            "template_config": self._last_template_config,
        }

    def reset_for_flow_invocation(self) -> None:
        self._messages = []
        self._dynamic_context = AgenticDynamicContext()
        self._rotate_session("flow invocation reset")

    async def __aenter__(self) -> "NativeConversationalInferencer":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release this inferencer's lease on its live vendor session; a
        private runtime manager (none injected) is shut down with it."""
        if self._lease_key is not None:
            await self.runtime_manager.release(self._lease_key)
            self._lease_key = None
        if self._owns_runtime:
            await self.runtime_manager.aclose_all()

    # =========================================================================
    # Durable record
    # =========================================================================

    def _load_record(self) -> NativeSessionRecord:
        stored = self.record_store.load(self.conversation_key)
        if self._record is None or (
            stored is not None and not self._record.newer_than(stored)
        ):
            self._record = stored
        if self._record is None:
            self._record = NativeSessionRecord(
                conversation_key=self.conversation_key,
                backend=self.backend.kind,
                cwd=self.backend.cwd,
                model=self.backend.model,
                principal=self.principal,
                permission_fingerprint=self._permission_fingerprint(),
            )
        return self._record

    def _save_record(self) -> None:
        if self._record is not None:
            self.record_store.save(self._record)

    def _permission_fingerprint(self) -> str:
        """The permission policy a vendor session was started under. The AF tool
        set is not part of it: a changed tool set is instruction drift (the L1
        core hash covers the tool manifest), handled per ``on_l1_drift``, not a
        permission change."""
        policy = {
            "permission_mode": self.backend.permission_mode,
            "environment": self.backend.environment,
        }
        return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    def _rotate_session(self, reason: str) -> None:
        record = self._record
        if record is None:
            return
        logger.info("Rotating native session %s: %s", self.conversation_key, reason)
        record.rotate()
        self._save_record()

    def _queue_notice(
        self, notice_type: str, body: Optional[str] = None, **ref: Any
    ) -> None:
        """Queue a one-shot notice for the next L2. The record keeps only a
        typed reference; ``body`` (if any) stays in memory and is rendered at
        delivery."""
        record = self._load_record()
        notice_id = record.next_notice_id
        record.add_notice(notice_type, **ref)
        if body is not None:
            self._notice_bodies[notice_id] = body
        self._save_record()

    def session_dir(self) -> Path:
        base = self.native_session_dir or self.prior_context.get("native_session_dir")
        path = (
            Path(base)
            if base
            else Path(tempfile.gettempdir()) / "af_native" / self.conversation_key
        )
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path

    def spill_dir(self) -> Path:
        return self.session_dir() / "tool_results"

    # =========================================================================
    # Tool-dispatch seams (ToolDispatchMixin)
    # =========================================================================

    def _on_command_followup(self, followup: Optional[str]) -> None:
        """A command invoked as a native tool already reports its follow-up
        ("Entered SOP 'X'. Starting on: …") in the tool result."""

    def _publish_async_completion(self, canonical: str, result: Any) -> None:
        text = getattr(result, "result", None)
        if text is None:
            return
        self.add_message("user", f"{TOOL_RESULTS_PREFIX}\n{canonical}: {text}")
        if not self.host_manages_async_results:
            self._queue_notice("tool_completion", body=str(text), tool=canonical)

    # =========================================================================
    # AFToolBridge host
    # =========================================================================

    @property
    def current_turn(self) -> Optional[TurnScope]:
        return self._current_turn

    def sop_fingerprint(self) -> tuple:
        state = self.sop_state
        active = (
            (
                state.sop_name,
                state.current_phase,
                str(state.phase_status),
                tuple(state.completed_phase_ids()),
            )
            if state is not None
            else None
        )
        return active, tuple(s.sop_name for s in self._suspended_sops)

    def render_state_update(self) -> str:
        """L3: the full active-SOP block when the SOP identity differs from
        that of the state the agent last received, else only the active SOP's
        status and next step (``TurnComposer.state_update``). Either way the
        agent then holds the complete current state, so the turn records that
        state's L2 hash as delivered."""
        record = self._load_record()
        feed = self._l2_feed(origin="", notices=[], generation=record.l2_generation)
        text = self._composer.state_update(feed)
        if self._current_turn is not None:
            self._current_turn.l3_state_hash = self._composer.turn_context(feed)[1]
        return text

    def persist_record(self) -> None:
        """Save the record after a state-changing bridge call, with what the
        running turn changed: a queued widget is durable before the turn
        ends, so a crash leaves ``pending_widget`` beside the turn's
        ``submitted`` state."""
        record = self._load_record()
        turn = self._current_turn
        if turn is not None and turn.pending_widgets:
            record.pending_widget = True
        self._save_record()

    def vendor_owns_result_spill(self) -> bool:
        return bool(self._backend_caps().owns_result_spill)

    def record_action(self, tool_name: str, text: str) -> None:
        summary = str(text)[:200]
        self._dynamic_context.add_action(tool_name, summary)
        if self._current_turn is not None:
            self._current_turn.completed_actions.append(
                CompletedAction(tool=tool_name, summary=summary)
            )

    async def answer_widgets_autonomously(
        self, tools: list, then_run: list, *, is_live: Callable[[], bool]
    ) -> str:
        """Yolo: answer conversation tools inline, exactly as the text-protocol
        orchestrator's yolo branch does, then run any ``then_run`` actions;
        each action's result applies only while ``is_live()`` holds when it
        returns (the calling turn is active and its call has not timed out)."""
        collected = await widget_core.synthesize_yolo(self, tools) or {}
        widget_core.record_yolo_answer(self, tools)
        answer = await self._after_widget_answer(
            tools, then_run, collected, is_live=is_live
        )
        if answer.dashboard_handoff and self._current_turn is not None:
            self._current_turn.request_stop(StopCause.DASHBOARD_HANDOFF)
        return f"Answered autonomously:\n{answer.message()}"

    def vendor_tool_timeout_s(self) -> Optional[float]:
        """How long the vendor waits for an AF tool call before it tells the
        model the call failed: every tool-capable backend applies the spec's
        ``mcp_tool_timeout_ms`` (Codex rounds it up to whole seconds)."""
        timeout_ms = self.backend.mcp_tool_timeout_ms
        return timeout_ms / 1000 if timeout_ms and timeout_ms > 0 else None

    def report_late_tool_result(
        self, tool: str, *, ran: bool, text: Optional[str], timeout_s: float
    ) -> None:
        """Queue the notice for an AF call the vendor gave up on (its result
        was not applied); the tool's output stays in memory only."""
        self._queue_notice(
            "late_tool_result", body=text, tool=tool, ran=ran, timeout_s=timeout_s
        )

    def _notice_text(self, entry: dict, current: str) -> str:
        if entry.get("type") == "late_tool_result":
            return self._late_tool_result_text(entry)
        return super()._notice_text(entry, current)

    def _late_tool_result_text(self, entry: dict) -> str:
        tool = f"`{entry['tool']}`" if entry.get("tool") else "An AgentFoundation tool"
        timeout_s = entry.get("timeout_s")
        timeout = (
            f"your {timeout_s:g} s tool-call timeout"
            if isinstance(timeout_s, (int, float)) and timeout_s > 0
            else "your tool-call timeout"
        )
        if not entry.get("ran"):
            return (
                f"{tool} was not run: {timeout} had passed when AgentFoundation "
                "got to it, and you were told the call failed. Call it again if "
                "you still need it."
            )
        body = self._notice_bodies.get(entry.get("id"))
        result = (
            f"Its result:\n{neutralize_host_tags(body)}"
            if body is not None
            else "Its result was not retained."
        )
        return (
            f"{tool} finished after {timeout}, after you were told the call "
            "failed. Results that arrived after the timeout were not applied; "
            f"call it again if you still need it. {result}"
        )

    # =========================================================================
    # SessionHooks (called from the backend's vendor hooks)
    # =========================================================================

    def l2_for_turn(self) -> str:
        """The running turn's L2, for the vendor's prompt hook (Claude Code's
        ``UserPromptSubmit``), which calls this exactly when it runs."""
        turn = self._current_turn
        if turn is None:
            return ""
        self._prompt_hook_turn = turn
        return turn.l2_text

    async def before_af_tool(
        self, tool_name: str, tool_use_id: str, agent_id: Optional[str]
    ) -> Optional[str]:
        turn = self._current_turn
        if turn is None:
            return "No active AgentFoundation turn."
        if agent_id:
            return "AgentFoundation tools are available to the main agent only."
        turn.note_af_call(tool_use_id)
        return None

    async def after_af_tool(self, tool_name: str, tool_use_id: str) -> bool:
        turn = self._current_turn
        if turn is None:
            return False
        # Calls of the same assistant message are known once its message event
        # was processed; wait briefly so a compound widget is not cut short.
        await turn.wait_known(tool_use_id, timeout=2.0)
        return turn.finish_af_call(tool_use_id)

    def on_compaction(self) -> None:
        if self._current_turn is not None:
            self._current_turn.compacted = True
        record = self._load_record()
        record.l2_hash = ""
        self._save_record()

    # =========================================================================
    # Commands that differ from the text-protocol orchestrator
    # =========================================================================

    @command(
        "clear", description="Clear the conversation and start a new agent session"
    )
    async def _cmd_clear(self) -> str:
        self._messages = []
        self._load_record()
        self._rotate_session("/clear")
        return "Conversation cleared; the next message starts a new agent session."

    @command("new", description="Start a new agent session (keeps SOP state)")
    async def _cmd_new(self) -> str:
        self._load_record()
        self._rotate_session("/new")
        return "The next message starts a new agent session."

    @command(
        "model",
        description="Change the agent's model",
        aliases=("set_model",),
        requires_args=True,
    )
    async def _cmd_set_model(self, model_name: str = "") -> str:
        """Record the model; the next turn applies it to the session like any
        other model change (``_switch_model``), including a switch the vendor
        refuses."""
        if not model_name:
            return f"Current model: {self.backend.model or 'default'}. Usage: /model <name>"
        self.backend.model = model_name
        self.prior_context["model_name"] = model_name
        return f"Model set to {model_name}; it applies from your next message."

    @command(
        "root",
        description="Set the session root directory",
        aliases=("set_session_root",),
        requires_args=True,
    )
    async def _cmd_set_session_root(self, path: str = "") -> str:
        """A new root is a user-requested new agent session there (plan §7.3,
        like ``/new``): the command rotates the session itself, so it is not a
        session loss and ``on_session_loss: fail`` does not refuse it; under
        ``recap`` the conversation is carried over as a recap. A session that
        another principal started is not rotated: the next turn's identity
        check refuses it, recap included."""
        if not path:
            return f"Current session root: {self.backend.cwd}. Usage: /root <path>"
        self.prior_context["session_root_path"] = path
        if path == self.backend.cwd:
            return f"The session root is already {path}; the agent session continues."
        self.backend.cwd = path
        record = self._load_record()
        if record.started and record.principal in ("", self.principal):
            self._rotate_session("/root")
            if self.on_session_loss == "recap":
                self._queue_notice("recap")
        return f"Session root set to {path}; the next message starts a new agent session there."

    # =========================================================================
    # InferencerBase entry points
    # =========================================================================

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        raise NotImplementedError(
            "NativeConversationalInferencer is async-only; use ainfer()/run_agentic_loop()."
        )

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> str:
        """One host turn without host callbacks; returns the final text."""
        result = await self.run_agentic_loop(
            str(inference_input), run_context=active_run_context()
        )
        return result.text

    async def _ainfer_recovery(
        self,
        inference_input: Any,
        last_exception: Optional[BaseException] = None,
        last_partial_output: Any = None,
        inference_config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Never re-run a native turn: it may already have been accepted by the
        vendor (submission safety, no automatic replay)."""
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("NativeConversationalInferencer does not retry turns.")

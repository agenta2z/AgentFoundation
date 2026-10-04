"""ConversationalInferencer — self-contained agentic unit.

Owns the full agentic loop: render prompt → call LLM → parse tool calls →
execute tools → accumulate context → loop. The server layer becomes a thin
I/O adapter that sets prior_context, tool_executor, and syncs messages.

Key components (via composition/protocols):
  - base_inferencer: StreamingInferencerBase for actual LLM calls
  - tool_registry + tool_executor: tool definitions + execution dispatch
  - prompt_renderer: Jinja2 template rendering
  - prior_context: fixed static context (session_root_path, workflow state)
  - _dynamic_context: accumulated completed actions with compression
  - context_compressor: optional LLM-based context compression
  - context_budget: per-section character limits

Uses @attrs to match InferencerBase hierarchy.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational import (
    widget_core,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticDynamicContext,
    AgenticResult,
    CompletedAction,
    ContextBudget,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_response_parser import (
    ConversationResponse,
    display_text,
    parse_conversation_response,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_state import (
    ConversationStateMixin,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_host import (  # noqa: F401 — module-level helpers stay importable from here
    _build_input_mode,
    _choice_option_from,
    _record_hitl_checkpoint,
    ConversationToolsMixin,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    finalize_input_value,
    group_and_validate,
    GroupValidationError,
)

# Phase L: decode_tool_bindings is no longer imported here — yolo path (Phase G)
# and _apply_widget_answer (Phase F) both route through the handler registry
# instead. The function itself remains in conversation_tool_runtime for the
# compound-decode helper and legacy tests to import directly.
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationToolType,
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
    UserMessage,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox_driver import (
    inbox_item_content,
    InboxDriver,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocol_text import (
    CONTINUE_AFTER_TOOLS,
    TOOL_RESULT_HEADER,
    TOOL_RESULTS_PREFIX,
    WIDGET_RESPONSE_PREFIX,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_commands import (
    SOPCommandsMixin,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
    SOPController,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_feed import (
    resolve_feed,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_sections import (
    render_sop_sections,
    sections_used_by,
)

# Phase I: DashboardAwareToolExecutor / HubAwareToolExecutor are no longer
# imported here — they moved to dashboard_coordinator.py where they belong.
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_call_parser import (
    parse_llm_response,
    ParsedToolCall,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_dispatch import (  # noqa: F401 — module-level helper stays importable from here
    _run_tool_executor,
    AsyncToolTasks,
    ToolDispatchMixin,
    ToolOutcome,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_input_collector import (
    collect_human_inputs,
    has_human_input_sentinel,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import active_run_context
from agent_foundation.resources.tools.formatters.markdown import ToolMarkdownFormatter
from agent_foundation.resources.tools.models import ToolDefinition
from agent_foundation.ui.interactive_base import InteractiveBase
from attr import attrib, attrs
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    DIRECTIVE_REQUIRES_USER_INPUT,
)

logger = logging.getLogger(__name__)

# Maximum conversation loop iterations for standalone run_conversation()
_MAX_CONVERSATION_ITERATIONS = 20

# Safety ceiling used when ``max_iterations`` is <= 0 (or None/False), which
# means "no fixed cap — run until the model stops on its own" (final answer /
# async dispatch / user handback). The agentic loop has NO other backstop (no
# no-progress or wall-clock/token guard), so a pathological autonomous loop
# (especially in yolo mode) could otherwise spin forever. This ceiling is a
# last-resort guard against a literal infinite loop, NOT a normal tuning knob —
# realistic turns stop far below it. Tune/raise only if a legitimate autonomous
# task genuinely needs more rounds.
_UNBOUNDED_ITERATION_CEILING = 1000

# Protocol-level message markers used in the agentic loop conversation history.
# These strings are part of the LLM-facing protocol — changing them may affect
# prompt comprehension. Keep them short and bracketed for easy parsing.
_WIDGET_RESPONSE_PREFIX = WIDGET_RESPONSE_PREFIX
_TOOL_RESULT_HEADER = TOOL_RESULT_HEADER  # .format(tool_name)
_TOOL_RESULTS_PREFIX = TOOL_RESULTS_PREFIX
_CONTINUE_AFTER_TOOLS = CONTINUE_AFTER_TOOLS

# Conversation role labels surfaced to the template as ``{{ user_role }}`` /
# ``{{ agent_role }}``. ``assistant`` matches the role used for the model's own
# messages in history (see ``add_message``), so a self-continuation CurrentTurn
# renders consistently with prior model turns. Single source of truth for the
# CurrentTurn tag AND the Decision Procedure's 1a/1b branches — change here if
# the model-side role name ever differs.
_USER_ROLE = "user"
_AGENT_ROLE = "assistant"


@attrs(slots=False)
class ConversationalInferencer(
    ConversationStateMixin,
    SOPCommandsMixin,
    ToolDispatchMixin,
    ConversationToolsMixin,
    InferencerBase,
):
    """Self-contained agentic inferencer with tool execution, context management,
    and prompt rendering.

    In server context, message_handlers calls run_agentic_loop() which owns the
    full render→infer→parse→execute→loop cycle.

    For standalone use, run_conversation() provides a simpler convenience loop
    (conversation tools only, no action tools).
    """

    # Skips InferencerBase.__attrs_post_init__, so it validates bta_inferencer itself.
    _SUPPORTS_BTA_FANOUT = False

    # Host-protocol capabilities (see ``host_protocol.ConversationalHost``).
    supports_round_resume = True
    supports_widget_recovery = True
    supports_inbox = True
    supports_prompt_manifest = True
    supports_flow_node = True
    supports_rewind = False

    # --- Core composition ---
    base_inferencer: InferencerBase = attrib(kw_only=True)
    interactive: Optional[InteractiveBase] = attrib(default=None, kw_only=True)
    # Legacy: used only by _ainfer()/run_conversation() (standalone path).
    # Server path uses _messages via run_agentic_loop(). The two are separate.
    conversation_history: list[dict[str, str]] = attrib(factory=list, init=False)

    # --- Agentic loop components ---
    tool_registry: dict[str, ToolDefinition] = attrib(factory=dict, kw_only=True)
    tool_executor: Any = attrib(default=None, kw_only=True)  # ToolExecutorCallable
    prompt_renderer: Any = attrib(default=None, kw_only=True)  # PromptRenderer
    context_compressor: Any = attrib(
        default=None, kw_only=True
    )  # ContextCompressorCallable
    prior_context: dict[str, Any] = attrib(factory=dict, kw_only=True)

    # --- Workflow integration ---
    workflow_manager: Any = attrib(default=None, kw_only=True)  # WorkflowManager
    yolo_mode: bool = attrib(default=False, kw_only=True)

    # --- Configuration ---
    compression_threshold: int = attrib(default=8000, kw_only=True)
    context_budget: ContextBudget = attrib(factory=ContextBudget, kw_only=True)
    # Hard bound on agentic-loop rounds per message. <= 0 / None / False means
    # "no fixed cap" (run until the model stops on its own, bounded only by the
    # internal _UNBOUNDED_ITERATION_CEILING safety guard); see run_agentic_loop.
    max_iterations: int = attrib(default=5, kw_only=True)
    # Soft, prompt-level self-governance threshold (rounds). When set (truthy),
    # it is injected into the prompt template as ``{{ soft_max_iterations }}`` so
    # the model is instructed to stop / use the confirmation tool if it has made
    # no meaningful progress after this many consecutive autonomous rounds. This
    # is guidance only (the model self-assesses progress from the conversation
    # context) — the enforced bound is max_iterations / the safety ceiling.
    # ``None`` disables the instruction. Typically paired with max_iterations<=0.
    soft_max_iterations: Optional[int] = attrib(default=None, kw_only=True)
    max_tool_result_chars: int = attrib(default=4000, kw_only=True)
    # Every round's rendered prompt carries the whole conversation, so a
    # continued vendor conversation holds the history twice. On, each round's
    # base call starts a fresh one (the base's ``areset_conversation`` under the
    # round's agent context). Off (the default) keeps the vendor's conversation
    # across rounds: the only carrier of the vendor's own built-in tool results
    # between rounds, at the cost of that duplication.
    fresh_vendor_session_per_round: bool = attrib(default=False, kw_only=True)
    # Extra directories to discover SOPs from (e.g. server/OpenTeam-provided
    # SOPs). Set by the host; mirrors session_context's extra_sop_dirs so the
    # /sop command and the Available-SOPs prompt list see the same SOPs the
    # executor does. Init kwarg ``extra_sop_dirs`` only seeds the SOP
    # controller, which owns the list: read and assign ``extra_sop_dirs``.
    _initial_extra_sop_dirs: list = attrib(
        factory=list, kw_only=True, alias="extra_sop_dirs", repr=False
    )

    # ─── SOP discovery filters (YAML-configurable) ───────────────────
    #
    # These two lists work together to control which SOPs the LLM sees in
    # the rendered "Available SOPs" prompt section. They are purely a
    # presentation filter — SOPs hidden here remain loadable via
    # ``/sop <name>`` explicitly. The init kwargs ``allowed_sops`` /
    # ``disallowed_sops`` make them YAML-configurable via ``default.yaml``
    # directly, in addition to constructor kwargs and the factory
    # passthrough. They only seed the SOP controller, which owns the lists:
    # read, mutate and assign ``allowed_sops`` / ``disallowed_sops``.
    #
    # Precedence (matches iptables / AWS IAM / k8s NetworkPolicy):
    #   1. If ``allowed_sops`` is non-empty, ONLY those names pass the
    #      whitelist (everything else is hidden).
    #   2. ``disallowed_sops`` then filters the survivors of step 1.
    #
    # **CRITICAL SEMANTIC — empty list = UNCONSTRAINED (not "deny all"):**
    #   * ``allowed_sops = []``      → filter is SKIPPED; every SOP passes
    #                                  the whitelist step. This is NOT a
    #                                  "deny all" allow-list — empty means
    #                                  "no whitelist restriction at all".
    #   * ``disallowed_sops = []``   → filter is SKIPPED; nothing dropped
    #                                  by the denylist step.
    #   * Both empty (the defaults)  → ALL discovered SOPs are visible.
    #
    # This matches the conventional semantic for AWS IAM (no allow rule =
    # all allowed when no deny rule applies), Kubernetes NetworkPolicy
    # (empty selector = "no restriction"), iptables (empty chain passes
    # everything), and AF's own ``allowed_tools`` attrib on
    # ``claude_code_*_inferencer``. The alternative — empty list = deny
    # all — would be a footgun: the default state (empty) would silently
    # hide every SOP. ``test_sop_discovery_filters.py`` locks this in.
    #
    # Naming mirrors the existing AF convention used by
    # ``claude_code_*_inferencer.allowed_tools`` (verified) — single verb
    # root ("allow") with prefix variation for the antonym, rather than
    # mixing different verbs (allow/exclude).
    #
    # Defaults: both empty → no filtering → backward-compatible behavior.
    _initial_allowed_sops: list = attrib(
        factory=list, kw_only=True, alias="allowed_sops", repr=False
    )
    _initial_disallowed_sops: list = attrib(
        factory=list, kw_only=True, alias="disallowed_sops", repr=False
    )

    # --- Handler-registry composition (Phase C) ---
    # Auto-populated to `default_registry()` in __attrs_post_init__ if not
    # explicitly injected. Every ConversationToolType MUST have a registered
    # handler; missing → fail-fast at construction (validated in post-init).
    handler_registry: ConversationToolHandlerRegistry = attrib(
        factory=default_registry, kw_only=True
    )

    # --- Dashboard/hub coordinator (Phase I) ---
    # Auto-constructed in __attrs_post_init__ using narrow context (tool_registry,
    # tool_dispatcher via getattr, prior_context reader). Handles the three
    # dashboard-related concerns previously inline on the CI: normalize_directives
    # (pre-fork), build_seed, and maybe_open (post-fork).
    dashboard_coordinator: Optional[DashboardCoordinator] = attrib(
        default=None, init=False
    )

    # --- SOP controller (Phase K) ---
    # Auto-constructed in __attrs_post_init__ (BEFORE any assignment through the
    # K3b `_paused` property shim). Owns sop_state, _suspended_sops, _paused,
    # _pending_followup, _auto_shutdown_on_sop_complete as attribs.
    # CI accesses them via forwarding @properties so all existing methods keep
    # working; new code should prefer sop_controller.is_paused etc.
    sop_controller: Optional[SOPController] = attrib(default=None, init=False)

    # --- Widget→loop typed mailboxes (Phase D) ---
    # Written by decode-time effects (OverrideNextActionToolArgs /
    # SetTurnVariables / DashboardDirectiveEffect), read by loop-frame
    # consumers on the next iteration boundary (_continue_after_widget /
    # _build_dashboard_seed) through the `_next_*` forwarding properties.
    # Cleared on consume — each effect raises HandlerResultMergeConflict on
    # double-set within one round (see effects/*.py).
    mailboxes: WidgetMailboxes = attrib(factory=WidgetMailboxes, init=False)

    # --- Internal state (init=False) ---
    _dynamic_context: AgenticDynamicContext = attrib(
        factory=AgenticDynamicContext, init=False
    )
    _messages: list[dict[str, str]] = attrib(factory=list, init=False)
    _last_rendered_prompt: str = attrib(default="", init=False)
    _last_template_source: str = attrib(default="", init=False)
    _last_template_feed: dict[str, Any] = attrib(factory=dict, init=False)
    _last_template_config: dict[str, Any] = attrib(factory=dict, init=False)
    async_tool_tasks: AsyncToolTasks = attrib(factory=AsyncToolTasks, init=False)

    def __attrs_post_init__(self) -> None:
        self._validate_bta_inferencer_spec()
        if self.prompt_renderer is None:
            from agent_foundation.common.inferencers.agentic_inferencers.conversational.template_manager_renderer import (
                TemplateManagerPromptRenderer,
            )
            from agent_foundation.resources import PROMPT_TEMPLATES_ROOT
            from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
                TemplateManager,
            )

            self.prompt_renderer = TemplateManagerPromptRenderer(
                template_manager=TemplateManager(
                    templates=str(PROMPT_TEMPLATES_ROOT),
                    active_template_root_space="conversation",
                    active_template_type="main",
                ),
                template_key="initial",
            )

        from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
            CommandRegistry,
        )

        self._commands = CommandRegistry(self)

        # Phase K: auto-construct SOPController BEFORE any assignment that
        # would go through the K3b `_paused` property shim (which forwards to
        # sop_controller). MUST come before the redundant K3a-deleted lines.
        # Narrow callbacks per Design Principle #4 — no CI back-ref.
        if self.sop_controller is None:
            self.sop_controller = SOPController(
                extra_sop_dirs=list(self._initial_extra_sop_dirs),
                allowed_sops=list(self._initial_allowed_sops),
                disallowed_sops=list(self._initial_disallowed_sops),
                prompt_renderer_ref=self.prompt_renderer,
                tool_registry=self.tool_registry,
                workflow_manager=getattr(self, "workflow_manager", None),
                prior_context_reader=lambda: self.prior_context,
                add_message=self.add_message,
                request_shutdown=self.request_shutdown,
                resolve_tool_name=self._resolve_tool_name,
            )

        # Phase I: auto-construct DashboardCoordinator with narrow context.
        # Framework tier no longer knows about HubAwareToolExecutor /
        # DashboardAwareToolExecutor by name — those imports live in the
        # coordinator module.
        if self.dashboard_coordinator is None:
            self.dashboard_coordinator = DashboardCoordinator(
                tool_registry=self.tool_registry,
                tool_dispatcher=getattr(self, "_tool_dispatcher", None)
                or self.tool_executor,
                prior_context_reader=lambda: self.prior_context,
            )

        # Phase C: fail-fast if any ConversationToolType lacks a handler. The
        # default_registry() factory covers all 6 framework tool types; a
        # user-injected registry may be incomplete, in which case we surface
        # the error at construction (not at first widget dispatch).
        _missing = [t for t in ConversationToolType if t not in self.handler_registry]
        if _missing:
            raise ValueError(
                f"handler_registry missing handlers for: {_missing!r}. "
                f"Registered: {self.handler_registry.list_registered()!r}"
            )

        # Phase K3a: the previous `self._paused = False`, `self.sop_state = None`,
        # and `self._suspended_sops = []` assignments are DELETED here — those
        # fields moved to SOPController with matching defaults. The properties
        # forward all reads/writes through `self.sop_controller`.

        # Inbox event loop (opt-in via enable_inbox)
        self._inbox_driver = InboxDriver(self._run_inbox_turn, log=logger)
        self._auto_shutdown_on_sop_complete = False

    @property
    def supports_prompt_rendering(self) -> bool:
        return self.prompt_renderer is not None

    @property
    def _extra_sop_dirs(self) -> list:
        """The SOP controller's live list (legacy name of ``extra_sop_dirs``)."""
        return self.sop_controller.extra_sop_dirs

    @_extra_sop_dirs.setter
    def _extra_sop_dirs(self, dirs: Any) -> None:
        self.extra_sop_dirs = dirs

    @property
    def _active_async_task(self) -> Any:
        """The most recently started background tool run (all runs are held
        by ``async_tool_tasks`` until they finish)."""
        return self.async_tool_tasks.latest

    @_active_async_task.setter
    def _active_async_task(self, task: Any) -> None:
        self.async_tool_tasks.latest = task

    def _workspace_root(self) -> str:
        return getattr(self.base_inferencer, "effective_cwd", "")

    # =========================================================================
    # Agentic Loop
    # =========================================================================

    async def run_agentic_loop(
        self,
        content: str,
        *,
        run_context=None,
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
        """Public conversational host entrypoint (``ConversationalHost``).

        Installs the RunContext bridge for the turn (a host may mint and pass a
        root ``run_context``; ``None`` installs a legacy root -> byte-identical),
        then delegates to the implementation. The internal ``base_inferencer``
        calls thread ``ctx.child("agent")`` (M3), so the run-state separation is
        active across the turn.

        ``origin`` (why the turn runs: user / tool_completion / host_event) is
        part of the shared host contract; the text-protocol loop does not use it.
        """
        from agent_foundation.common.inferencers.run_context import enter_run, exit_run

        _rc_token = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            return await self._run_agentic_loop_impl(
                content,
                interactive=interactive,
                session_id=session_id,
                turn_number=turn_number,
                on_new_turn=on_new_turn,
                on_prompt_rendered=on_prompt_rendered,
                on_turn_complete=on_turn_complete,
                on_round_start=on_round_start,
                on_round_complete=on_round_complete,
            )
        finally:
            exit_run(_rc_token)

    async def _run_agentic_loop_impl(
        self,
        content: str,
        *,
        interactive: Optional[InteractiveBase] = None,
        session_id: str = "",
        turn_number: int = 0,
        on_new_turn: Optional[Any] = None,
        on_prompt_rendered: Optional[Any] = None,
        on_turn_complete: Optional[Any] = None,
        on_round_start: Optional[Any] = None,
        on_round_complete: Optional[Any] = None,
    ) -> AgenticResult:
        """Implementation of :meth:`run_agentic_loop` (body unchanged).

        When interactive + session_id are provided AND base_inferencer supports
        ainfer_streaming(), uses stream_token_batches() for token-by-token delivery.
        Otherwise falls back to non-streaming ainfer().

        NOTE (V2 TODO): Passing interactive + session_id creates a transport
        coupling between the framework-layer inferencer and the server-layer
        InteractiveBase. Consider introducing a StreamingCallback protocol
        to decouple them in a future iteration.
        """
        # Command dispatch: slash-commands bypass the LLM entirely
        if content and self._commands.is_command(content):
            from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
                UnknownCommand,
            )

            try:
                response = await self._commands.dispatch(content)
            except UnknownCommand:
                pass  # fall through to agentic loop
            else:
                self.add_message("user", content)
                self.add_message("assistant", response)
                # A command (e.g. /sop, /resume_sop) may seed an initial request
                # to act on within the just-entered SOP. If so, persist it as the
                # current user turn and fall through into the agentic loop so the
                # SOP starts working immediately ("enter and act"). Otherwise the
                # command is terminal — return its acknowledgement.
                followup = self._consume_pending_followup()
                if followup is None:
                    return AgenticResult(
                        text=response,
                        completed_actions=[],
                        iterations_used=0,
                    )
                self.add_message("user", followup)
                content = followup

        loop_actions: list[CompletedAction] = []
        # Resolve interactive: prefer per-call arg, fallback to self.interactive
        effective_interactive = interactive or self.interactive
        can_stream = (
            effective_interactive is not None
            and hasattr(effective_interactive, "stream_token_batches")
            and hasattr(self.base_inferencer, "ainfer_streaming")
        )
        last_raw_response = ""
        last_boundary_turn: int | None = None  # track last sent turn_boundary

        # M9/D7/§2.8: rehydrate a paused conversation from the resumed RunStateStore.
        self._rehydrate_from_resumed_store()

        # Consume any pending resume state from a prior _restore_pause_state
        _resume = getattr(self, "_pending_resume_state", None)
        start_iteration = 0
        if _resume is not None:
            start_iteration = _resume.get("iteration", 0)
            self._pending_resume_state = None

        # max_iterations <= 0 (or None/False) means "no fixed cap": run until
        # the model stops on its own (final answer / async tool dispatch / user
        # handback). We still bound by a high safety ceiling because the loop has
        # no other backstop — an unbounded autonomous loop (esp. yolo) could
        # otherwise spin forever. `not self.max_iterations` covers None/False/0
        # (and short-circuits before the `<= 0` comparison so None is safe).
        if not self.max_iterations or self.max_iterations <= 0:
            effective_max = _UNBOUNDED_ITERATION_CEILING
        else:
            effective_max = self.max_iterations
        # SOP needs more iterations than the default 5
        if self.sop_state and self.sop_state.sop:
            n_phases = len(self.sop_state.sop.phases)
            effective_max = max(effective_max, n_phases * 3)

        # Local helpers to reduce callback boilerplate
        async def _fire_turn_complete(turn_num: int) -> None:
            if on_turn_complete:
                try:
                    await on_turn_complete(turn_num)
                except Exception as _e:
                    logger.warning("[agentic_loop] on_turn_complete error: %s", _e)

        async def _fire_new_turn(turn_num: int, user_input: str) -> int:
            if on_new_turn:
                try:
                    new = await on_new_turn(turn_num, user_input)
                    if new is not None:
                        return new
                except Exception as _e:
                    logger.warning("[agentic_loop] on_new_turn error: %s", _e)
            return turn_num

        async def _fire_round_start(iter_idx: int, turn_num: int) -> None:
            """Fire the per-round-START hook (every LLM call, incl. action-tool
            continuations). The hook is generic: it returns an opaque round
            context dict; if that dict carries ``cache_folder`` the inferencer
            points its cache there (so streaming files land in the round dir),
            and if the active interactive exposes ``set_round_context`` the
            context is handed to it so WS events can carry the round identity.
            Identity/persistence semantics live entirely server-side."""
            if not on_round_start:
                return
            try:
                ctx = await on_round_start(iter_idx, turn_num)
            except Exception as _e:
                logger.warning("[agentic_loop] on_round_start error: %s", _e)
                return
            if not ctx:
                return
            try:
                cache_folder = (
                    ctx.get("cache_folder") if isinstance(ctx, dict) else None
                )
                if cache_folder:
                    self.cache_folder = cache_folder
                if effective_interactive is not None and hasattr(
                    effective_interactive, "set_round_context"
                ):
                    effective_interactive.set_round_context(ctx)
            except Exception as _e:
                logger.warning("[agentic_loop] set_round_context error: %s", _e)

        async def _fire_round_complete(
            iter_idx: int,
            turn_num: int,
            raw_resp: str,
            clean_resp: str,
            conv_resp: ConversationResponse,
        ) -> None:
            """Fire the per-round-COMPLETE hook AFTER the response is parsed but
            BEFORE any tool/widget dispatch, so a round's user-facing preamble is
            persisted + closed before a widget's ``pending_input``. Passes the
            display-clean text (tool/markup stripped) plus the raw + clean
            response and the parsed conversation response; the server decides
            whether to commit a bubble (only when display text is non-empty)."""
            if not on_round_complete:
                return
            try:
                dtext = display_text(clean_resp)
            except Exception as _e:
                logger.warning("[agentic_loop] display_text error: %s", _e)
                dtext = ""
            try:
                await on_round_complete(
                    self, iter_idx, turn_num, raw_resp, clean_resp, dtext, conv_resp
                )
            except Exception as _e:
                logger.warning("[agentic_loop] on_round_complete error: %s", _e)

        def _record_then_run(name: str, outcome: ToolOutcome) -> None:
            if outcome.is_command:
                self._on_command_followup(outcome.followup_text)
            summary = outcome.text[:200]
            loop_actions.append(CompletedAction(tool=name, summary=summary))
            self._dynamic_context.add_action(name, summary)

        async def _continue_after_widget(
            conv_tools, action_tools, collected, *, text="", raw=None
        ):
            """Shared post-``aget_input`` continuation (the loop tail) around
            ``widget_core``'s after-answer steps: widget-response user message,
            new-turn boundary, bundled action-tools, turn-complete. Called by
            BOTH the live conversation-tool fork AND pending-widget RECOVERY so
            a widget answer produces identical side-effects either way. Returns
            an AgenticResult if the turn should END (dashboard handoff or an
            async action-tool was dispatched), else None (the loop should
            ``continue``). Defined here so it closes over the loop-local
            ``_fire_new_turn``/``_fire_turn_complete`` hooks,
            ``iteration``/``loop_actions``, and ``turn_number``/``content``.
            """
            nonlocal turn_number, content
            accepted = await widget_core.accept_answer(self, conv_tools, collected)
            self.add_message("user", accepted.response_text)
            content = accepted.response_text

            # R2 (deterministic quiet). A dashboard handoff (proposal-selection
            # --experiment-hub) ends the turn QUIETLY right after the phase
            # advance (the 2b→3 advance MUST happen) and BEFORE _fire_new_turn /
            # the Phase-3 render: Phase 3 never renders (the hub owns the work —
            # R3 keeps a later render safe as belt-and-suspenders). Reuses the
            # exact terminal AgenticResult shape used for async-action dispatch.
            if accepted.dashboard_handoff:
                await _fire_turn_complete(iteration + 1)
                return AgenticResult(
                    text=text or "",
                    raw_response=raw,
                    completed_actions=loop_actions,
                    iterations_used=iteration + 1,
                    last_rendered_prompt=self._last_rendered_prompt,
                    last_template_source=self._last_template_source,
                    last_template_feed=self._last_template_feed,
                    last_template_config=self._last_template_config,
                )

            # Notify server of new turn boundary (new turn dir + stream_start).
            turn_number = await _fire_new_turn(turn_number, accepted.response_text)

            # Execute any action tools from the same ToolsToInvoke block,
            # resolving __var__ placeholders with the collected user inputs.
            then_run_results: tuple = ()
            if widget_core.runs_then_run(self, action_tools):
                if accepted.turn_variables:
                    widget_core.apply_turn_variables(self, accepted.turn_variables)
                    self.add_message(
                        "user", widget_core.turn_variables_text(accepted.turn_variables)
                    )
                then_run_results = await widget_core.run_then_run(
                    self,
                    action_tools,
                    accepted,
                    run_ctx=active_run_context(),
                    on_async_done=self._on_async_tool_done,
                    on_applied=_record_then_run,
                )
                self.add_message(
                    "user",
                    f"{_TOOL_RESULTS_PREFIX}\n"
                    + widget_core.then_run_text(then_run_results),
                )

            # Update content so the next iteration's <CurrentTurn> shows a
            # continuation prompt instead of re-feeding the widget response.
            content = _CONTINUE_AFTER_TOOLS
            if any(r.outcome.is_async for r in then_run_results):
                await _fire_turn_complete(iteration + 1)
                return AgenticResult(
                    text=text or "",
                    raw_response=raw,
                    completed_actions=loop_actions,
                    iterations_used=iteration + 1,
                    last_rendered_prompt=self._last_rendered_prompt,
                    last_template_source=self._last_template_source,
                    last_template_feed=self._last_template_feed,
                    last_template_config=self._last_template_config,
                )
            await _fire_turn_complete(iteration + 1)
            return None

        # Initialize first turn BEFORE the loop so cache_folder is set
        # before the first LLM call (streaming files land in turn_001/).
        if start_iteration == 0:
            turn_number = await _fire_new_turn(turn_number, content)

        for iteration in range(start_iteration, effective_max):
            # Cooperative pause check at iteration boundary
            if self._paused:
                from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
                    PausedResult,
                )

                return PausedResult(
                    pause_state=self._serialize_pause_state(
                        turn_number=turn_number,
                        iteration=iteration,
                    ),
                    text=last_raw_response or "",
                    completed_actions=loop_actions,
                    iterations_used=iteration + 1,
                )
            # Signal turn boundary ONLY when the server turn number has
            # changed (i.e., _on_new_turn created a new turn directory).
            # This keeps frontend turn numbers in sync with server turn
            # directories so "View Prompt" maps correctly.
            if (
                iteration > 0
                and can_stream
                and effective_interactive is not None
                and turn_number != last_boundary_turn
            ):
                if hasattr(effective_interactive, "send_turn_boundary"):
                    await effective_interactive.send_turn_boundary(
                        session_id,
                        turn_number=turn_number,
                        cache_folder=getattr(self, "cache_folder", ""),
                    )
                    last_boundary_turn = turn_number

            # Pending-widget RECOVERY (Layer 2, Piece 3). A reconnect/restart
            # re-armed an unanswered widget by setting _pending_widget_result and
            # re-entering the loop at the widget's iteration. Instead of
            # re-rendering / re-calling the LLM (which could emit a DIFFERENT
            # widget → the answer would bind to the wrong variable), decode the
            # persisted answer against the EXACT persisted widget and run the SAME
            # continuation the live path runs — fully deterministic, no inference.
            # Gated on _pending_widget_result, so the live path is unaffected.
            _pwr = getattr(self, "_pending_widget_result", None)
            if _pwr is not None:
                self._pending_widget_result = None
                _recovered = await widget_core.recover(self, _pwr)
                if _recovered.bindings is None:
                    # No usable answer — end the turn with the widget still
                    # pending (mirrors the live ``collected is None`` path).
                    await _fire_turn_complete(iteration + 1)
                    return AgenticResult(
                        text="",
                        raw_response=last_raw_response,
                        completed_actions=loop_actions,
                        iterations_used=iteration + 1,
                        has_conversation_tool=True,
                    )
                _res = await _continue_after_widget(
                    _recovered.tools, _recovered.then_run, _recovered.bindings
                )
                if _res is not None:
                    return _res
                continue

            # Record the content this round renders with, so a per-round snapshot
            # (_conversation_blob, written by the on_round_start hook) is
            # self-contained: the resumed loop renders round Y with the
            # caller-supplied `content`, so the value ENTERING this round must be
            # captured before the snapshot is taken.
            self._render_content = content

            # 0. Per-round START hook (every LLM call incl. continuations).
            # Mints the round's identity server-side and points cache_folder at
            # this round's dir BEFORE rendering/streaming, so per-round bubbles
            # and artifacts are correct. Generic + duck-typed (no-op if unwired).
            await _fire_round_start(iteration, turn_number)

            # 1. Compress dynamic context if needed
            await self._compress_context_if_needed()

            # 2. Render prompt
            rendered = self._render_prompt(content)
            self._last_rendered_prompt = rendered

            # 3. Call LLM (streaming or non-streaming)
            # The rendered prompt is self-contained: it includes the system
            # role text, tools, conversation history, and the current user
            # message. We send it as a single user message with no separate
            # system_prompt, so what gets logged == what gets sent.
            _agent_ctx = self._rc_child("agent")
            try:
                if self.fresh_vendor_session_per_round and isinstance(
                    self.base_inferencer, InferencerBase
                ):
                    await self.base_inferencer.areset_conversation(
                        run_context=_agent_ctx
                    )
                if can_stream:
                    # Clear any prior system_prompt/messages on the base
                    # inferencer so the rendered prompt is the sole input.
                    self.base_inferencer.system_prompt = ""

                    async def token_gen():
                        async with contextlib.aclosing(
                            self.base_inferencer.ainfer_streaming(
                                rendered, run_context=_agent_ctx
                            )
                        ) as stream:
                            async for chunk in stream:
                                yield chunk, {"turn_number": turn_number}

                    async with contextlib.aclosing(token_gen()) as tokens:
                        raw_response = await effective_interactive.stream_token_batches(
                            tokens,
                            session_id,
                            send_stream_end=False,
                            turn_number=turn_number,
                        )
                else:
                    raw_response = await self.base_inferencer.ainfer(
                        rendered, run_context=_agent_ctx
                    )
                    if not isinstance(raw_response, str):
                        raw_response = str(raw_response)
            except Exception as e:
                logger.error("Inferencer error in agentic loop: %s", e)
                raise
            last_raw_response = raw_response

            # Get clean final output if base inferencer has one (e.g., --output-file).
            # CLI-based inferencers (streams_differ_from_final_output=True) publish the
            # clean text from --output-file or trailing JSON schema output with their
            # call's outcome. API-based inferencers have none (stream IS the output).
            clean_response = raw_response
            _final = None  # default: no separate clean output
            if getattr(self.base_inferencer, "streams_differ_from_final_output", False):
                _final = self._final_output_at(self.base_inferencer, _agent_ctx)
                if _final:
                    clean_response = _final
                    logger.debug(
                        "[ConversationalInferencer] Using clean final output "
                        "(%d chars) instead of noisy stream (%d chars) for parsing",
                        len(clean_response),
                        len(raw_response),
                    )
                    # Notify interactive so it can send stream_correction to frontend
                    # and store clean output for message_end.
                    if effective_interactive and hasattr(
                        effective_interactive, "on_clean_output_available"
                    ):
                        try:
                            await effective_interactive.on_clean_output_available(
                                clean_response
                            )
                        except Exception as _e:
                            logger.warning(
                                "[ConversationalInferencer] on_clean_output_available "
                                "failed: %s",
                                _e,
                            )

            # Flush prompt + response artifacts to disk so "View Prompt"
            # works even while waiting for user input (confirmation, etc.).
            if on_prompt_rendered:
                try:
                    await on_prompt_rendered(self, raw_response)
                except Exception as _pr_err:
                    logger.warning(
                        "[agentic_loop] on_prompt_rendered error: %s", _pr_err
                    )

            # Add CLEAN output to conversation history so subsequent turns
            # include exact LLM text (not noisy TUI stdout).
            self.add_message("assistant", clean_response)

            # 4. Check for conversation tools using CLEAN output (intact code fences)
            logger.info(
                "[agentic_loop] clean_response: source=%s, length=%d",
                "output_file" if _final else "raw_stream",
                len(clean_response),
            )
            conv_response = parse_conversation_response(clean_response)

            # Per-round COMPLETE hook — fires for EVERY round (incl. action-tool
            # continuations) AFTER parse and BEFORE any tool/widget dispatch, so a
            # round's preamble bubble is committed + message_end'd before a widget's
            # pending_input. The server commits a bubble only when display text is
            # non-empty (empty/pure-tool rounds → no bubble, balanced terminal).
            await _fire_round_complete(
                iteration, turn_number, raw_response, clean_response, conv_response
            )

            if conv_response.has_conversation_tool:
                logger.info(
                    "[ConversationalInferencer] conversation tool: type=%s prompt=%.80s metadata=%s",
                    conv_response.conversation_tool.tool_type,
                    conv_response.conversation_tool.prompt,
                    conv_response.conversation_tool.metadata,
                )
            else:
                logger.info(
                    "[ConversationalInferencer] no conversation tool found (text_len=%d)",
                    len(conv_response.text),
                )

            if conv_response.has_conversation_tool:
                # Proposal payloads and dashboard handoffs, BEFORE either the
                # yolo or the interactive branch consumes the tools (so yolo
                # select_all sees the per-proposal choices, and the widget
                # relabel and the post-fork opener see the handoff).
                widget_core.prepare(self, conv_response.conversation_tools)
                # Validate parallel_group grouping ONCE here — the single shared
                # pre-branch point both the yolo and interactive paths funnel
                # through — so neither can bypass the guardrails. Fail CLOSED via a
                # controlled self-continuation (NOT an uncaught raise that aborts
                # the turn): inject feedback as a tool-result-style message and keep
                # content == _CONTINUE_AFTER_TOOLS so the next round is classified as
                # a self-continuation (not user input), telling the model to emit
                # dependent groups in later rounds.
                try:
                    group_and_validate(conv_response.conversation_tools)
                except GroupValidationError as _gve:
                    logger.info(
                        "[agentic_loop] parallel_group validation failed: %s", _gve
                    )
                    self.add_message(
                        "user",
                        f"{_TOOL_RESULTS_PREFIX}\n[parallel_group validation] {_gve} "
                        "Re-emit the conversation tools fixing this — put independent "
                        "questions in the SAME parallel_group, and ask any dependent "
                        "question in a LATER assistant round.",
                    )
                    content = _CONTINUE_AFTER_TOOLS
                    await _fire_turn_complete(iteration + 1)
                    continue
                if self.yolo_mode:
                    collected = await widget_core.synthesize_yolo(
                        self, conv_response.conversation_tools
                    )
                    synthetic_summary = str(collected)
                    self._messages.append(
                        {
                            "role": "user",
                            "content": f"[Synthetic auto-advance] {synthetic_summary}",
                            "synthetic": True,
                        }
                    )
                    # In yolo mode, any conversation tool response is
                    # auto-approved: requires_user_input phases advance.
                    widget_core.record_yolo_answer(
                        self, conv_response.conversation_tools
                    )
                elif effective_interactive:
                    collected = await widget_core.present_and_collect(
                        self,
                        conv_response.conversation_tools,
                        conv_response.text,
                        interactive=effective_interactive,
                        then_run=conv_response.action_tools,
                        turn_number=turn_number,
                        iteration=iteration,
                    )
                else:
                    collected = None
                if collected is None:
                    await _fire_turn_complete(iteration + 1)
                    return AgenticResult(
                        text=conv_response.text,
                        raw_response=raw_response,
                        completed_actions=loop_actions,
                        iterations_used=iteration + 1,
                        has_conversation_tool=True,
                        conversation_tool=conv_response.conversation_tool,
                        last_rendered_prompt=self._last_rendered_prompt,
                        last_template_source=self._last_template_source,
                        last_template_feed=self._last_template_feed,
                        last_template_config=self._last_template_config,
                    )
                # Post-widget continuation (dashboard open → widget-response
                # message → phase-completion → new-turn boundary → bundled
                # action-tools → turn-complete). Extracted into
                # _continue_after_widget so the LIVE path and pending-widget
                # RECOVERY run the exact same tail (no drift). Returns an
                # AgenticResult if an async action-tool ended the turn, else None.
                _res = await _continue_after_widget(
                    conv_response.conversation_tools,
                    conv_response.action_tools,
                    collected,
                    text=conv_response.text or "",
                    raw=last_raw_response,
                )
                if _res is not None:
                    return _res
                continue

            # 5a. Execute action tools from ToolsToInvoke (if any)
            if conv_response.action_tools and self.tool_executor:
                tool_results: list[str] = []
                for at in conv_response.action_tools:
                    tc = ParsedToolCall(
                        name=at.get("name", ""),
                        arguments=at.get("arguments", {}),
                        raw=str(at),
                    )
                    result_text = await self._execute_tool_call(tc)
                    summary = result_text[:200]
                    action = CompletedAction(tool=tc.name, summary=summary)
                    loop_actions.append(action)
                    self._dynamic_context.add_action(tc.name, summary)
                    tool_results.append(
                        f"{_TOOL_RESULT_HEADER.format(tc.name)}\n{result_text}"
                    )

                combined = "\n\n".join(tool_results)
                if len(combined) > self.max_tool_result_chars:
                    combined = (
                        combined[: self.max_tool_result_chars] + "\n... (truncated)"
                    )
                self.add_message("user", f"{_TOOL_RESULTS_PREFIX}\n{combined}")
                content = _CONTINUE_AFTER_TOOLS
                if getattr(self, "_async_tool_dispatched", False):
                    self._async_tool_dispatched = False
                    await _fire_turn_complete(iteration + 1)
                    return AgenticResult(
                        text=conv_response.text or "",
                        raw_response=last_raw_response,
                        completed_actions=loop_actions,
                        iterations_used=iteration + 1,
                        last_rendered_prompt=self._last_rendered_prompt,
                        last_template_source=self._last_template_source,
                        last_template_feed=self._last_template_feed,
                        last_template_config=self._last_template_config,
                    )
                await _fire_turn_complete(iteration + 1)
                continue

            # 5b. Parse for action tool calls (legacy XML format)
            parsed = parse_llm_response(raw_response, self._valid_tool_names)
            if not parsed.has_tool_calls:
                await _fire_turn_complete(iteration + 1)
                return AgenticResult(
                    text=parsed.text,
                    raw_response=raw_response,
                    completed_actions=loop_actions,
                    iterations_used=iteration + 1,
                    last_rendered_prompt=self._last_rendered_prompt,
                    last_template_source=self._last_template_source,
                    last_template_feed=self._last_template_feed,
                    last_template_config=self._last_template_config,
                )

            # 6. Execute tools
            tool_results: list[str] = []
            for tc in parsed.tool_calls:
                # Collect __human_input__ values if present
                if has_human_input_sentinel(tc.arguments) and effective_interactive:
                    tool_def = self.tool_registry.get(self._resolve_tool_name(tc.name))
                    tc.arguments = await collect_human_inputs(
                        tc.arguments, tool_def, effective_interactive
                    )
                result_text = await self._execute_tool_call(tc)
                summary = result_text[:200]
                action = CompletedAction(tool=tc.name, summary=summary)
                loop_actions.append(action)
                self._dynamic_context.add_action(tc.name, summary)
                tool_results.append(
                    f"{_TOOL_RESULT_HEADER.format(tc.name)}\n{result_text}"
                )

            combined = "\n\n".join(tool_results)
            if len(combined) > self.max_tool_result_chars:
                combined = combined[: self.max_tool_result_chars] + "\n... (truncated)"

            if parsed.text:
                self.add_message("assistant", parsed.text)
            self.add_message("user", f"{_TOOL_RESULTS_PREFIX}\n{combined}")
            content = _CONTINUE_AFTER_TOOLS
            if getattr(self, "_async_tool_dispatched", False):
                self._async_tool_dispatched = False
                await _fire_turn_complete(iteration + 1)
                return AgenticResult(
                    text=parsed.text or "",
                    raw_response=last_raw_response,
                    completed_actions=loop_actions,
                    iterations_used=iteration + 1,
                    last_rendered_prompt=self._last_rendered_prompt,
                    last_template_source=self._last_template_source,
                    last_template_feed=self._last_template_feed,
                    last_template_config=self._last_template_config,
                )
            await _fire_turn_complete(iteration + 1)

        # Exhausted the effective cap (the configured max_iterations, the SOP
        # phase-derived bound, or the unbounded safety ceiling) — return last
        # raw response. Use effective_max (not self.max_iterations) so the
        # reported count is correct when max_iterations is <=0/None (unbounded).
        await _fire_turn_complete(effective_max)
        return AgenticResult(
            text=last_raw_response,
            raw_response=last_raw_response,
            completed_actions=loop_actions,
            iterations_used=effective_max,
            exhausted_max_iterations=True,
            last_rendered_prompt=self._last_rendered_prompt,
            last_template_source=self._last_template_source,
            last_template_feed=self._last_template_feed,
            last_template_config=self._last_template_config,
        )

    # =========================================================================

    def reset_for_flow_invocation(self) -> None:
        """Reset state for a fresh flow invocation.

        Clears conversation state to prevent leakage between flow phases
        or worker nodes. Called by ConversationalFlowNodeAdapter before
        each invocation.
        """
        self._messages = []
        self.reset_dynamic_context()  # delegates to existing method
        self.conversation_history = []

    # Alias for LWI's reset_sessions_per_iteration which calls reset_session()
    reset_session = reset_for_flow_invocation

    # =========================================================================
    # Pause / Resume
    # =========================================================================

    def _conversation_blob(
        self,
        *,
        turn_number: int = 0,
        iteration: int = 0,
    ) -> dict:
        """Build a serializable snapshot of CI conversation state (PURE).

        Captures everything needed to resume the loop at a given round: the full
        message history (INCLUDING the tool-result / widget / synthetic user
        turns that live ONLY in ``self._messages`` and are never persisted to the
        host's ``session_state``), ``prior_context``, the SOP + suspended-SOP
        stack, the dynamic context, and the ``content`` the current round renders
        with. PURE: unlike ``_serialize_pause_state`` it does NOT mirror into the
        active RunContext node, so a caller that snapshots EVERY round does not
        pollute ``run_state/store.json`` and thereby trip
        ``_rehydrate_from_resumed_store`` on the next fresh turn.
        """
        blob = {
            "messages": list(self._messages),
            "prior_context": self._json_safe_prior_context(),
            "dynamic_context": (
                self._dynamic_context.to_dict()
                if hasattr(self, "_dynamic_context")
                else None
            ),
            "content": getattr(self, "_render_content", ""),
            "turn_number": turn_number,
            "iteration": iteration,
        }
        # Phase K6: SOP portion delegates to SOPController.serialize() —
        # byte-identical to the pre-extraction emission of `sop_state` +
        # `suspended_sops` keys.
        if self.sop_controller is not None:
            blob.update(self.sop_controller.serialize())
        else:
            blob["sop_state"] = None
            blob["suspended_sops"] = []
        return blob

    def _serialize_pause_state(
        self,
        *,
        turn_number: int = 0,
        iteration: int = 0,
    ) -> dict:
        """Capture CI state for pause (pure blob + ctx-node mirror)."""
        blob = self._conversation_blob(turn_number=turn_number, iteration=iteration)
        # D7/§2.8: mirror the pause blob into the active context node so it lands
        # in the persisted Tier-1 RunStateStore (the durable resume artifact).
        # Additive — the returned blob is unchanged (byte-identical without a ctx).
        try:
            from agent_foundation.common.inferencers.run_context import (
                active_run_context,
            )

            _ctx = active_run_context()
            if _ctx is not None:
                _ctx.node().conversation = blob
        except Exception:  # pragma: no cover - mirroring is best-effort
            pass
        return blob

    def _rehydrate_from_resumed_store(self) -> None:
        """M9/D7/§2.8: if the active context node carries a conversation blob loaded
        from a resumed ``RunStateStore``, restore it — the READ side of
        ``_serialize_pause_state`` that wires the host's ``RunStateStore.load`` into
        the CI. No-op when a resume is already pending in-process or no blob exists;
        best-effort so a fresh run is never blocked on restore."""
        if getattr(self, "_pending_resume_state", None) is not None:
            return
        try:
            from agent_foundation.common.inferencers.run_context import (
                active_run_context,
            )

            _ctx = active_run_context()
            if _ctx is not None and _ctx.node().conversation:
                self._restore_pause_state(_ctx.node().conversation)
        except Exception:  # pragma: no cover - resume is best-effort
            pass

    def _restore_pause_state(self, state: dict, *, reattach_sop: bool = True) -> None:
        """Restore CI state from a serialized pause / round-entry snapshot.

        D7/§2.8: when the active context node carries a rehydrated conversation
        blob (loaded from a resumed RunStateStore), prefer it over ``state``.

        ``reattach_sop`` (default True): when False, restore ``_messages`` /
        ``prior_context`` / ``_dynamic_context`` and set the pending-resume marker,
        but do NOT touch ``sop_state`` / ``suspended_sops``. The round-resume host
        (OpenStartup) reattaches the SOP via its extra-dirs-aware factory
        (``_restore_sop_state``) on the rebuilt CI BEFORE calling this, so the
        SOP it attached must not be replaced here.
        The gate applies regardless of whether ``state`` came from the argument or
        the ctx-node preference above.
        """
        try:
            from agent_foundation.common.inferencers.run_context import (
                active_run_context,
            )

            _ctx = active_run_context()
            if _ctx is not None and _ctx.node().conversation:
                state = _ctx.node().conversation
        except Exception:  # pragma: no cover
            pass

        self._messages = state["messages"]
        self.prior_context = dict(state.get("prior_context", {}))

        # Phase D4: reset the three _next_* mailboxes BEFORE any handler-replay
        # logic can run (via _collect_widget_response / pending-widget recovery).
        # Without this reset, replaying a persisted (tool, raw_answer) through
        # the registry would raise HandlerResultMergeConflict when a per-effect
        # .apply() finds the mailbox already populated from the pre-reconnect
        # turn. Design Principle #14.
        self._next_action_tool_overrides = None
        self._next_turn_variables = None
        self._next_dashboard_directives = None

        # Phase K6: SOP restore delegates to SOPController. Preserves the
        # reattach_sop=False cross-repo contract — controller.restore()'s
        # early-return matches OpenStartup's round-resume host expectations.
        if self.sop_controller is not None:
            self.sop_controller.restore(state, reattach_sop=reattach_sop)

        if state.get("dynamic_context") is not None and hasattr(
            self, "_dynamic_context"
        ):
            self._dynamic_context = self._dynamic_context.__class__.from_dict(
                state["dynamic_context"]
            )
        self._pending_resume_state = {
            "turn_number": state.get("turn_number", 0),
            "iteration": state.get("iteration", 0),
        }
        # Unconditional _paused reset (matches pre-migration :1497 semantic —
        # ALWAYS reset on restore, independent of the reattach_sop flag).
        # Goes through K3b's forwarding property shim.
        self._paused = False

    # =========================================================================
    # Host protocol (public names for the private state API above)
    # =========================================================================

    def export_state(self, *, turn_number: int = 0, iteration: int = 0) -> dict:
        return self._conversation_blob(turn_number=turn_number, iteration=iteration)

    def restore_state(self, state: dict, *, reattach_sop: bool = True) -> None:
        self._restore_pause_state(state, reattach_sop=reattach_sop)

    def set_pending_widget_answer(self, payload: dict) -> None:
        """Re-arm a persisted widget's answer; the next loop entry decodes it
        against that exact widget without re-inference."""
        self._pending_widget_result = payload

    def last_prompt_data(self) -> dict[str, Any]:
        return {
            "rendered_prompt": self._last_rendered_prompt,
            "template_source": self._last_template_source,
            "template_feed": self._last_template_feed,
            "template_config": self._last_template_config,
        }

    async def aclose(self) -> None:
        """Nothing to release: each round's backend call owns its resources."""

    # =========================================================================
    # Inbox Event Loop
    # =========================================================================

    def enable_inbox(
        self,
        interactive=None,
        *,
        auto_shutdown_on_sop_complete: bool = False,
        maxsize: int = 0,
        on_new_turn=None,
        on_prompt_rendered=None,
        on_turn_complete=None,
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

    def inbox_put(self, item) -> None:
        """Non-blocking enqueue. Raises RuntimeError if inbox not enabled."""
        self._inbox_driver.put(item)

    def inbox_put_user(self, content: str, source: str = "user") -> None:
        """Convenience: enqueue a UserMessage."""
        self.inbox_put(UserMessage(content=content, source=source))

    def request_shutdown(self) -> None:
        """Cooperative termination. Current run_agentic_loop finishes, then run() returns."""
        self._inbox_driver.request_shutdown()

    @property
    def shutdown_requested(self) -> bool:
        return self._inbox_driver.shutdown_requested

    @property
    def _shutdown_requested(self) -> bool:
        return self._inbox_driver.shutdown_requested

    @_shutdown_requested.setter
    def _shutdown_requested(self, value: bool) -> None:
        self._inbox_driver.shutdown_requested = value

    @property
    def _inbox(self):
        return self._inbox_driver.queue

    @_inbox.setter
    def _inbox(self, value) -> None:
        self._inbox_driver.queue = value

    def _next_turn_number(self) -> int:
        return self._inbox_driver.next_turn_number()

    def _content_for_item(self, item) -> str | None:
        return inbox_item_content(item)

    async def _run_inbox_turn(
        self,
        content: str,
        *,
        origin: str,
        interactive,
        turn_number: int,
        on_new_turn,
        on_prompt_rendered,
        on_turn_complete,
        run_context,
    ) -> AgenticResult:
        return await self.run_agentic_loop(
            content=content,
            origin=origin,
            interactive=interactive,
            turn_number=turn_number,
            on_new_turn=on_new_turn,
            on_prompt_rendered=on_prompt_rendered,
            on_turn_complete=on_turn_complete,
            run_context=run_context,
        )

    async def run(self, *, run_context=None) -> AgenticResult | None:
        """Long-lived event loop. Drains inbox, calls run_agentic_loop per item.

        Returns when request_shutdown() is called.

        ``run_context`` (§9.4 / G3): the SOP CLI host mints a session root and
        passes it here; each inbox item runs under its own ``ctx.child(turn_N)``
        so per-turn run-state is isolated. ``None`` (default) -> byte-identical.
        """
        return await self._inbox_driver.run(run_context=run_context)

    # =========================================================================
    # Prompt Rendering
    # =========================================================================

    def _render_prompt(self, current_message: str) -> str:
        """Build template variables and render via prompt_renderer."""
        # Phase J1: pre-render SOP auto-discover (hoisted from mid-body).
        # `_render_prompt` proper is now a (mostly) pure reader over sop_state.
        self._ensure_sop_state_for_render()
        # Format tools — separate action tools from conversation tools
        formatter = ToolMarkdownFormatter()
        tools_list = list(self.tool_registry.values())
        # Exclude user-only tools (agent_enabled=False) from LLM prompt
        agent_tools = [t for t in tools_list if getattr(t, "agent_enabled", True)]
        action_tools = [t for t in agent_tools if t.tool_type != "Conversation"]
        available_tools = formatter.format_all(action_tools)

        # Append commands to action tools — indistinguishable to the LLM
        commands_text = self._commands.render_for_prompt()
        if commands_text:
            available_tools = (
                f"{available_tools}\n\n{commands_text}"
                if available_tools
                else commands_text
            )

        # Build conversation history (exclude last user msg to avoid duplication)
        messages = list(self._messages)
        if (
            messages
            and messages[-1].get("role") == "user"
            and messages[-1].get("content") == current_message
        ):
            messages = messages[:-1]

        # Build completed_actions for template, respecting dynamic_context_max budget
        all_actions = [
            {"tool": a.tool, "summary": a.summary}
            for a in self._dynamic_context.completed_actions
        ]
        actions_text = "\n".join(f"- {a['tool']}: {a['summary']}" for a in all_actions)
        if len(actions_text) > self.context_budget.dynamic_context_max:
            # Keep most recent actions that fit within budget
            truncated: list[dict[str, str]] = []
            total = 0
            for action in reversed(all_actions):
                line = f"- {action['tool']}: {action['summary']}"
                if total + len(line) + 1 > self.context_budget.dynamic_context_max:
                    break
                truncated.insert(0, action)
                total += len(line) + 1
            all_actions = truncated

        # Render conversation tools
        conv_tools = [t for t in agent_tools if t.tool_type == "Conversation"]
        conversation_tools_text = ""
        if conv_tools:
            conversation_tools_text = formatter._format_conversation_tools(conv_tools)

        # Template variable defaults from .variables.yaml (lowest priority)
        template_vars = getattr(self.prompt_renderer, "template_variables", {}) or {}

        sop_feed = self._build_sop_feed(catalog_mode="when_idle")
        sop = sop_feed["sop"]
        nextstep_guidance = sop_feed["sop_nextstep_guidance"]
        available_sops = sop_feed["available_sops"]
        paused_sop = sop_feed["paused_sop"]
        inprogress_sops = sop_feed["inprogress_sops"]

        # Build feed using build_feed — merges dicts + FeedBase objects
        from rich_python_utils.common_objects.feed_base import build_feed

        # CurrentTurn role tells the model who is driving this round: a genuine
        # user message (_USER_ROLE) vs. the model continuing its own work after
        # tool results (_AGENT_ROLE — content is the _CONTINUE_AFTER_TOOLS
        # nudge). Surfaced to the template (also as user_role/agent_role below)
        # to drive the 1a/1b split in the Decision Procedure.
        current_turn_role = (
            _AGENT_ROLE if current_message == _CONTINUE_AFTER_TOOLS else _USER_ROLE
        )

        feed = build_feed(
            template_vars,
            self.prior_context,
            self.sop_state,
            {
                "session_root_path": getattr(self.base_inferencer, "effective_cwd", ""),
                "sop_nextstep_guidance": nextstep_guidance,
                "available_sops": available_sops,
                "paused_sop": paused_sop,
                "inprogress_sops": inprogress_sops,
                # Explicit "is an SOP currently active?" flag for the template's
                # Decision Procedure branch. Robust signal (the active SOP object
                # itself), unlike sop_description (empty when the SOP has no
                # description) or inprogress_sops (those are SUSPENDED, not active).
                "sop_active": sop is not None,
                "action_tools": available_tools,
                "completed_actions": all_actions,
                "conversation_history": messages,
                "current_turn": {"role": current_turn_role, "content": current_message},
                "conversation_tools": conversation_tools_text,
                # Soft self-governance threshold surfaced to the template. None
                # when unset -> the template's instruction block is skipped.
                "soft_max_iterations": self.soft_max_iterations,
                # Role labels for the 1a (user) / 1b (self-continuation) split.
                "user_role": _USER_ROLE,
                "agent_role": _AGENT_ROLE,
            },
        )

        # Same Jinja2 environment as the main template, so behaviour is identical.
        if hasattr(self.prompt_renderer, "render_string"):
            feed = resolve_feed(feed, self.prompt_renderer.render_string)

        # Identity/SOP sections are shared with the native orchestrator's
        # templates. Only the sections the active template references are
        # rendered, so a missing or foreign main template keeps failing (or
        # rendering) exactly as before.
        template_manager = getattr(self.prompt_renderer, "template_manager", None)
        if template_manager is not None:
            sections = sections_used_by(
                template_manager.get_raw_template(self.prompt_renderer.template_key)
            )
            if sections:
                feed.update(render_sop_sections(template_manager, feed, sections))

        self._last_template_feed = dict(feed)
        self._last_template_source = self.prompt_renderer.template_source
        self._last_template_config = (
            getattr(self.prompt_renderer, "template_config", {}) or {}
        )
        rendered = self.prompt_renderer.render(feed)

        # ── Non-empty rendered-prompt postcondition ──────────────────────
        # A non-empty rendered prompt is a hard invariant of this method:
        # downstream LLM backends (including the rovodev CLI inferencer)
        # will silently hang if handed an empty prompt and there is no
        # other layer in the stack that distinguishes "intentional empty"
        # from "broken template lookup". We fail loudly here with full
        # diagnostic context so the bug surfaces at the producer rather
        # than as a 120-second backend watchdog timeout downstream.
        #
        # Reference incident: OpenStartup production session
        # ``server_20260615_194631_8e0863a8`` turn_002 — a misconfigured
        # ``TemplateManager(templates=...)`` returned ``""`` for
        # ``conversation/main/initial``, which then propagated all the way
        # to rovodev as an empty argv and hung for 120s with zero error
        # output. Fixed at the factory layer (registering AF templates as
        # a fallback root); this guard prevents the same class of silent
        # regression from recurring with any other future renderer
        # misconfiguration.
        if not (rendered and rendered.strip()):
            raise RuntimeError(
                f"ConversationalInferencer._render_prompt: "
                f"prompt_renderer.render(feed) returned empty output. "
                f"renderer={type(self.prompt_renderer).__name__}, "
                f"template_source={self._last_template_source!r}, "
                f"template_key={getattr(self.prompt_renderer, 'template_key', None)!r}. "
                f"This usually means the configured templates directory "
                f"does not contain the requested template file (e.g. "
                f"``conversation/main/<template_key>.jinja2``). "
                f"Check ``TemplateManager(templates=...)`` configuration "
                f"in the caller that constructed this renderer; if it "
                f"points at a partial templates root, add a fallback "
                f"root via ``templates=[primary, fallback, ...]`` or "
                f"``add_template_root(...)``. Setting "
                f"``TemplateManager(strict_lookup=True)`` on the "
                f"underlying manager will surface the same problem one "
                f"layer deeper with the exact failed lookup key."
            )

        return rendered

    # =========================================================================
    # Context Compression
    # =========================================================================

    async def _compress_context_if_needed(self) -> None:
        if self.context_compressor is None:
            return
        if self._dynamic_context.total_chars() < self.compression_threshold:
            return
        compressed = await self.context_compressor(
            self._dynamic_context.to_text(),
            self.context_budget.dynamic_context_max,
            run_context=self._rc_child("context_compression"),
        )
        self._dynamic_context.compress(compressed)

    # =========================================================================
    # Single-step inference (kept for backward compat / standalone use)
    # =========================================================================

    def _infer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **_inference_args,
    ) -> ConversationResponse:
        """Sync single-step inference with conversation tool parsing."""
        raw = self.base_inferencer.infer(
            inference_input,
            inference_config,
            run_context=self._rc_child("agent"),
            **_inference_args,
        )
        raw_str = str(raw) if not isinstance(raw, str) else raw
        return parse_conversation_response(raw_str)

    async def _ainfer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **_inference_args,
    ) -> ConversationResponse:
        """Async single-step inference with conversation tool parsing."""
        if isinstance(inference_input, str):
            self.conversation_history.append(
                {"role": "user", "content": inference_input}
            )

        raw = await self.base_inferencer.ainfer(
            inference_input,
            inference_config,
            run_context=self._rc_child("agent"),
            **_inference_args,
        )
        raw_str = str(raw) if not isinstance(raw, str) else raw

        self.conversation_history.append({"role": "assistant", "content": raw_str})

        return parse_conversation_response(raw_str)

    async def run_conversation(
        self,
        initial_input: str,
        inference_config: Any = None,
        **inference_args,
    ) -> str:
        """Convenience loop for standalone use (outside server context).

        .. deprecated::
            Use run_agentic_loop() for new code. This method is kept for
            backward compatibility with standalone/CLI callers that only
            need conversation tool handling (no action tools).

        Calls _ainfer() in a loop, handling conversation tools internally.
        Uses self.conversation_history (not self._messages).
        """
        current_input = initial_input

        for iteration in range(_MAX_CONVERSATION_ITERATIONS):
            response = await self._ainfer(
                current_input, inference_config, **inference_args
            )

            if not response.has_conversation_tool:
                return response.text

            if self.interactive is None:
                logger.warning(
                    "Conversation tool requested but no interactive transport"
                )
                return response.text

            user_response = await self._handle_conversation_tool(
                response.conversation_tool, response.text
            )

            if user_response is None:
                return response.text

            current_input = user_response

        logger.warning(
            "Conversation loop exhausted after %d iterations",
            _MAX_CONVERSATION_ITERATIONS,
        )
        return response.text

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    # --- Streaming delegation to base_inferencer ---

    @property
    def system_prompt(self) -> str:
        return getattr(self.base_inferencer, "system_prompt", "")

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        if hasattr(self.base_inferencer, "system_prompt"):
            self.base_inferencer.system_prompt = value

    @property
    def cache_folder(self) -> str | None:
        return getattr(self.base_inferencer, "cache_folder", None)

    @cache_folder.setter
    def cache_folder(self, value: str) -> None:
        if hasattr(self.base_inferencer, "cache_folder"):
            self.base_inferencer.cache_folder = value

    async def ainfer_streaming(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context=None,
        **kwargs: Any,
    ):
        """Delegate streaming to the base inferencer under an "agent" child context.

        §2.10: ``run_context`` is keyword-only so it never lands in ``**kwargs`` and
        double-binds when forwarded (which would raise "multiple values for
        run_context"). The agent child is derived from the explicit ``run_context``
        when given, else the active context; the base call installs it as its bridge.
        """
        _agent_ctx = (
            run_context.child("agent")
            if run_context is not None
            else self._rc_child("agent")
        )
        if hasattr(self.base_inferencer, "ainfer_streaming"):
            async with contextlib.aclosing(
                self.base_inferencer.ainfer_streaming(
                    inference_input, inference_config, run_context=_agent_ctx, **kwargs
                )
            ) as stream:
                async for chunk in stream:
                    yield chunk
        else:
            result = await self.base_inferencer.ainfer(
                inference_input, inference_config, run_context=_agent_ctx, **kwargs
            )
            yield str(result) if not isinstance(result, str) else result

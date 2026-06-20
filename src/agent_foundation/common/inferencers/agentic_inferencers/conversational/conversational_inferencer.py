

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

import logging
from typing import Any, Optional

from attr import attrib, attrs
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
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    GroupValidationError,
    decode_compound_bindings,
    decode_tool_bindings,
    finalize_input_value,
    group_and_validate,
    render_templated_fields,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_call_parser import (
    ParsedToolCall,
    parse_llm_response,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_input_collector import (
    collect_human_inputs,
    has_human_input_sentinel,
)
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)
from agent_foundation.ui.input_modes import (
    ChoiceOption,
    InputMode,
    InputModeConfig,
    multiple_choices,
    single_choice,
)
from agent_foundation.ui.interactive_base import (
    InteractionFlags,
    InteractiveBase,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    command,
)
from agent_foundation.resources.tools.formatters.markdown import ToolMarkdownFormatter
from agent_foundation.resources.tools.models import ToolDefinition
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    DIRECTIVE_REQUIRES_USER_INPUT,
    SOPManager,
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
_WIDGET_RESPONSE_PREFIX = "[Collected from conversation widget]"
_TOOL_RESULT_HEADER = "[Tool Result: {}]"  # .format(tool_name)
_TOOL_RESULTS_PREFIX = "[Tool execution results]"
_CONTINUE_AFTER_TOOLS = "Continue based on the tool execution results above."

# Conversation role labels surfaced to the template as ``{{ user_role }}`` /
# ``{{ agent_role }}``. ``assistant`` matches the role used for the model's own
# messages in history (see ``add_message``), so a self-continuation CurrentTurn
# renders consistently with prior model turns. Single source of truth for the
# CurrentTurn tag AND the Decision Procedure's 1a/1b branches — change here if
# the model-side role name ever differs.
_USER_ROLE = "user"
_AGENT_ROLE = "assistant"


def _now_iso() -> str:
    """UTC timestamp for SOP suspension ordering/display."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


@attrs(slots=False)
class ConversationalInferencer(InferencerBase):
    """Self-contained agentic inferencer with tool execution, context management,
    and prompt rendering.

    In server context, message_handlers calls run_agentic_loop() which owns the
    full render→infer→parse→execute→loop cycle.

    For standalone use, run_conversation() provides a simpler convenience loop
    (conversation tools only, no action tools).
    """

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
    # Extra directories to discover SOPs from (e.g. server/OpenTeam-provided
    # SOPs). Set at construction by the host; mirrors session_context's
    # extra_sop_dirs so the /sop command and the Available-SOPs prompt list see
    # the same SOPs the executor does. Init kwarg: ``extra_sop_dirs``.
    _extra_sop_dirs: list = attrib(factory=list, kw_only=True)

    # ─── SOP discovery filters (YAML-configurable) ───────────────────
    #
    # These two lists work together to control which SOPs the LLM sees in
    # the rendered "Available SOPs" prompt section. They are purely a
    # presentation filter — SOPs hidden here remain loadable via
    # ``/sop <name>`` explicitly. The bare (no leading underscore) attrib
    # names make them YAML-configurable via ``default.yaml`` directly, in
    # addition to constructor kwargs and the factory passthrough.
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
    allowed_sops: list = attrib(factory=list, kw_only=True)
    disallowed_sops: list = attrib(factory=list, kw_only=True)

    # --- Internal state (init=False) ---
    _dynamic_context: AgenticDynamicContext = attrib(
        factory=AgenticDynamicContext, init=False
    )
    _messages: list[dict[str, str]] = attrib(factory=list, init=False)
    _last_rendered_prompt: str = attrib(default="", init=False)
    _last_template_source: str = attrib(default="", init=False)
    _last_template_feed: dict[str, Any] = attrib(factory=dict, init=False)
    _last_template_config: dict[str, Any] = attrib(factory=dict, init=False)

    def __attrs_post_init__(self) -> None:
        if self.prompt_renderer is None:
            from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
                TemplateManager,
            )
            from agent_foundation.common.inferencers.agentic_inferencers.conversational.template_manager_renderer import (
                TemplateManagerPromptRenderer,
            )
            from agent_foundation.resources import PROMPT_TEMPLATES_ROOT

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
        self._paused = False
        self.sop_state = None  # SOPState | None — the ONE active SOP
        # Paused + exited SOPs, most-recent-first. Resumable via /resume_sop.
        self._suspended_sops: list = []

        # Inbox event loop fields (opt-in via enable_inbox)
        self._inbox = None  # asyncio.Queue[InboxItem] | None
        self._shutdown_requested = False
        self._default_interactive = None
        self._auto_shutdown_on_sop_complete = False
        self._turn_counter = 0
        self._running = False

    @property
    def supports_prompt_rendering(self) -> bool:
        return self.prompt_renderer is not None

    # =========================================================================
    # Agentic Loop
    # =========================================================================

    async def run_agentic_loop(
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
        """Main entry point. Replaces ConversationRouter._agentic_loop().

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
                cache_folder = ctx.get("cache_folder") if isinstance(ctx, dict) else None
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
                        turn_number=turn_number, iteration=iteration,
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
            try:
                if can_stream:
                    # Clear any prior system_prompt/messages on the base
                    # inferencer so the rendered prompt is the sole input.
                    self.base_inferencer.system_prompt = ""

                    async def token_gen():
                        async for chunk in self.base_inferencer.ainfer_streaming(
                            rendered
                        ):
                            yield chunk, {"turn_number": turn_number}

                    raw_response = await effective_interactive.stream_token_batches(
                        token_gen(),
                        session_id,
                        send_stream_end=False,
                        turn_number=turn_number,
                    )
                else:
                    raw_response = await self.base_inferencer.ainfer(rendered)
                    if not isinstance(raw_response, str):
                        raw_response = str(raw_response)
            except Exception as e:
                logger.error("Inferencer error in agentic loop: %s", e)
                raise
            last_raw_response = raw_response

            # Get clean final output if base inferencer has one (e.g., --output-file).
            # CLI-based inferencers (streams_differ_from_final_output=True) return the
            # clean text from --output-file or trailing JSON schema output.
            # API-based inferencers return None (stream IS the final output).
            clean_response = raw_response
            _final = None  # default: no separate clean output
            if getattr(self.base_inferencer, "streams_differ_from_final_output", False):
                _final = self.base_inferencer.get_final_output()
                if _final:
                    clean_response = _final
                    logger.debug(
                        "[ConversationalInferencer] Using clean final output "
                        "(%d chars) instead of noisy stream (%d chars) for parsing",
                        len(clean_response), len(raw_response),
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
                                "failed: %s", _e,
                            )

            # Flush prompt + response artifacts to disk so "View Prompt"
            # works even while waiting for user input (confirmation, etc.).
            if on_prompt_rendered:
                try:
                    await on_prompt_rendered(self, raw_response)
                except Exception as _pr_err:
                    logger.warning("[agentic_loop] on_prompt_rendered error: %s", _pr_err)

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
                # Enrich proposal_selection tools (resolve proposals_path →
                # proposals + selectable choices) BEFORE either the yolo or the
                # interactive branch consumes them, so yolo select_all sees the
                # per-proposal choices and the widget gets its payload.
                for _ps_tool in conv_response.conversation_tools:
                    if _ps_tool.tool_type == ConversationToolType.PROPOSAL_SELECTION:
                        self._enrich_proposal_selection(_ps_tool)
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
                    collected = self._synthesize_yolo_collected(
                        conv_response.conversation_tools,
                    )
                    synthetic_summary = str(collected)
                    self._messages.append({
                        "role": "user",
                        "content": f"[Synthetic auto-advance] {synthetic_summary}",
                        "synthetic": True,
                    })
                    # In yolo mode, any conversation tool response is auto-approved.
                    # Set user input gate so requires_user_input phases advance.
                    if self.sop_state:
                        self.sop_state.user_input_gate_passed = True
                    self._check_phase_completion()
                elif effective_interactive:
                    collected = await self._handle_conversation_tools(
                        conv_response.conversation_tools,
                        conv_response.text,
                        interactive_override=effective_interactive,
                        action_tools=conv_response.action_tools,
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
                # Combine all collected inputs as the user message
                if isinstance(collected, dict):
                    parts = [f"{k}: {v}" for k, v in collected.items() if v]
                    user_input = (
                        f"{_WIDGET_RESPONSE_PREFIX}\n"
                        + ("\n".join(parts) if parts else str(collected))
                    )
                else:
                    user_input = f"{_WIDGET_RESPONSE_PREFIX}\n{collected}"
                self.add_message("user", user_input)
                content = user_input
                self._check_phase_completion()

                # Notify server of new turn boundary so it can start
                # a new turn directory and send stream_start/stream_end
                turn_number = await _fire_new_turn(turn_number, user_input)

                # Execute any action tools from the same ToolsToInvoke block,
                # resolving __var__ placeholders with the collected user inputs.
                if conv_response.action_tools and self.tool_executor:
                    # Apply any param_overrides from confirmation widget
                    param_overrides = getattr(self, "_pending_param_overrides", None)
                    if param_overrides:
                        self._pending_param_overrides = None

                    # Apply any generic variables from widget response.
                    # Uses prompt_renderer.variable_manager.set() directly —
                    # ConversationalInferencer does NOT have a _set_variable()
                    # method (that method exists on SessionToolExecutor).
                    pending_vars = getattr(self, "_pending_variables", None)
                    if pending_vars:
                        self._pending_variables = None
                        if self.prompt_renderer:
                            vm = getattr(self.prompt_renderer, "variable_manager", None)
                            if vm is not None and hasattr(vm, "set"):
                                for vk, vv in pending_vars.items():
                                    vm.set(vk, vv)
                        # Append to the synthesized user turn so LLM sees them
                        var_lines = [f"[{k}]: {v}" for k, v in pending_vars.items()]
                        self.add_message("user", "\n".join(var_lines))

                    action_tool_results: list[str] = []
                    for at in conv_response.action_tools:
                        resolved_args = {}
                        for k, v in at.get("arguments", {}).items():
                            if isinstance(v, str) and v.startswith("__") and v.endswith("__"):
                                var_name = v[2:-2]
                                if isinstance(collected, dict) and var_name in collected:
                                    resolved_args[k] = collected[var_name]
                                else:
                                    resolved_args[k] = v
                            else:
                                resolved_args[k] = v
                        # Merge user-configured param overrides from confirmation UI
                        if param_overrides:
                            resolved_args.update(param_overrides)
                        tc = ParsedToolCall(
                            name=at.get("name", ""),
                            arguments=resolved_args,
                            raw=str(at),
                        )
                        result_text = await self._execute_tool_call(tc)
                        summary = result_text[:200]
                        action = CompletedAction(tool=tc.name, summary=summary)
                        loop_actions.append(action)
                        self._dynamic_context.add_action(tc.name, summary)
                        action_tool_results.append(
                            f"{_TOOL_RESULT_HEADER.format(tc.name)}\n{result_text}"
                        )
                    # Add tool results to conversation so the LLM sees them
                    combined_results = "\n\n".join(action_tool_results)
                    self.add_message(
                        "user", f"{_TOOL_RESULTS_PREFIX}\n{combined_results}"
                    )

                # Update content so the next iteration's <CurrentTurn> shows
                # a continuation prompt instead of re-feeding the widget response.
                content = _CONTINUE_AFTER_TOOLS
                if getattr(self, '_async_tool_dispatched', False):
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
                    combined = combined[: self.max_tool_result_chars] + "\n... (truncated)"
                self.add_message("user", f"{_TOOL_RESULTS_PREFIX}\n{combined}")
                content = _CONTINUE_AFTER_TOOLS
                if getattr(self, '_async_tool_dispatched', False):
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
            if getattr(self, '_async_tool_dispatched', False):
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

    def set_prior_context(self, ctx: dict[str, Any]) -> None:
        self.prior_context = dict(ctx)

    def update_prior_context(self, **kwargs: Any) -> None:
        if "sop_state" in kwargs:
            self.sop_state = kwargs.pop("sop_state")
            if self.sop_state and self.sop_state.yolo_mode:
                self.yolo_mode = True
        self.prior_context.update(kwargs)

    def set_session_variables(self, variables: dict[str, Any]) -> None:
        """Store variables in both variable_manager and prior_context.

        Used by conversation tool handlers to persist user-provided values
        (e.g., workflow_target_path, strategy) so they're available in both
        template rendering (variable_manager) and SOP guidance (prior_context).
        """
        for name, value in variables.items():
            self.prior_context[name] = value
            if self.prompt_renderer:
                vm = getattr(self.prompt_renderer, "variable_manager", None)
                if vm and hasattr(vm, "set"):
                    vm.set(name, value)

    def _session_root(self) -> str:
        """Best-effort session root for path re-join (used by the finalizer)."""
        root = self.prior_context.get("session_root_path", "") if self.prior_context else ""
        return root or getattr(self.base_inferencer, "effective_cwd", "") or ""

    def _make_field_renderer(self):
        """Return a ``str -> str`` renderer bound to the live session context, or
        ``None`` if no renderer is available. Used to resolve a templated tool
        ``prefix`` (e.g. an echoed ``{{ session_root_path }}``) after parsing."""
        pr = self.prompt_renderer
        if pr is None or not hasattr(pr, "render_string"):
            return None
        ctx: dict[str, Any] = dict(getattr(self, "_last_template_feed", {}) or {})
        ctx.update(self.prior_context or {})
        return lambda s: pr.render_string(s, ctx)

    def set_messages(self, messages: list) -> None:
        """Set conversation messages for prompt rendering.

        Messages are incorporated into the rendered prompt by _render_prompt().
        We do NOT delegate to base_inferencer.set_messages() because that would
        set _messages_override on PlugboardApiInferencer, causing
        ainfer_streaming() to ignore the rendered prompt.
        """
        self._messages = list(messages)

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def _consume_pending_followup(self) -> Optional[str]:
        """Pop a one-shot initial request seeded by a command.

        ``/sop <name> <request>`` and ``/resume_sop <name> <request>`` stash the
        free-text request here so the loop can act on it as the first user turn
        of the just-entered/resumed SOP (instead of merely entering and idling).
        Returns the request once, then clears it.
        """
        followup = getattr(self, "_pending_followup", None)
        if followup:
            self._pending_followup = None
        return followup

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    @property
    def dynamic_context(self) -> AgenticDynamicContext:
        return self._dynamic_context

    def reset_dynamic_context(self) -> None:
        self._dynamic_context = AgenticDynamicContext()

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

    def _serialize_pause_state(
        self, *, turn_number: int = 0, iteration: int = 0,
    ) -> dict:
        """Capture CI state for pause."""
        return {
            "messages": list(self._messages),
            "prior_context": dict(self.prior_context),
            "sop_state": self.sop_state.to_dict() if self.sop_state else None,
            "suspended_sops": [s.to_dict() for s in self._suspended_sops],
            "dynamic_context": (
                self._dynamic_context.to_dict()
                if hasattr(self, "_dynamic_context") else None
            ),
            "turn_number": turn_number,
            "iteration": iteration,
        }

    def _restore_pause_state(self, state: dict) -> None:
        """Restore CI state from a serialized pause snapshot."""
        from agent_foundation.common.workflow.sop_state import SOPState

        self._messages = state["messages"]
        self.prior_context = dict(state.get("prior_context", {}))

        sop_dict = state.get("sop_state")
        if sop_dict:
            self.sop_state = SOPState.from_dict(sop_dict)
            self._reload_sop_definition(self.sop_state)
        else:
            self.sop_state = None

        self._suspended_sops = []
        for sw_dict in state.get("suspended_sops", []):
            sw = SOPState.from_dict(sw_dict)
            self._reload_sop_definition(sw)
            self._suspended_sops.append(sw)

        if state.get("dynamic_context") is not None and hasattr(self, "_dynamic_context"):
            self._dynamic_context = self._dynamic_context.__class__.from_dict(
                state["dynamic_context"]
            )
        self._pending_resume_state = {
            "turn_number": state.get("turn_number", 0),
            "iteration": state.get("iteration", 0),
        }
        self._paused = False

    # =========================================================================
    # Backslash Commands (Model A)
    # =========================================================================

    @command("help", description="List available commands", aliases=("?",))
    async def _cmd_help(self) -> str:
        lines = ["Available commands:"]
        for meta in self._commands.list_commands():
            aliases = f" (aliases: {', '.join('/' + a for a in meta.aliases)})" if meta.aliases else ""
            lines.append(f"  /{meta.name}{aliases} — {meta.description}")
        return "\n".join(lines)

    @command("status", description="Show SOP state and session info", aliases=("s",))
    async def _cmd_status(self) -> str:
        n_susp = len(self._suspended_sops)
        susp_note = f" Suspended SOPs: {n_susp}." if n_susp else ""
        if not self.sop_state:
            return (
                f"No active SOP. Messages: {len(self._messages)}. "
                f"Paused: {self._paused}.{susp_note}"
            )
        s = self.sop_state
        completed = [
            c.phase if hasattr(c, "phase") else str(c) for c in s.completed_phases
        ]
        return (
            f"SOP: {s.sop_name}\n"
            f"Phase: {s.current_phase} ({s.phase_status})\n"
            f"Completed: {completed}\n"
            f"Messages: {len(self._messages)}. Paused: {self._paused}.{susp_note}"
        )

    @command("clear", description="Clear conversation history")
    async def _cmd_clear(self) -> str:
        self._messages = []
        return "Conversation history cleared."

    @command("sop",
             aliases=("enter_sop",),
             description=(
                 "Enter an SOP, optionally with an initial request to start on. "
                 "Usage: /enter_sop <name> [--yolo] [--fresh] [request...]"
             ),
             requires_args=True)
    async def _cmd_sop(self, args: str = "") -> str:
        tokens = args.split()
        if not tokens:
            return "Usage: /sop <name> [--yolo] [--fresh] [request...]"
        name = tokens[0]
        rest = tokens[1:]
        # Only known flags are flags; everything else is the free-text request
        # (order-independent, so the request can come before or after a flag).
        _KNOWN_FLAGS = {"--yolo", "--fresh"}
        yolo = "--yolo" in rest
        fresh = "--fresh" in rest
        request = " ".join(t for t in rest if t not in _KNOWN_FLAGS).strip()

        # Same-name resume-vs-fresh detection (deterministic, no LLM guesswork).
        suspended = next(
            (s for s in self._suspended_sops if s.sop_name == name), None
        )
        if suspended is not None and not fresh:
            return (
                f"You have an in-progress '{name}' ({suspended.sop_status}, "
                f"{suspended.suspension_label.lower()}). "
                f"Use /resume_sop {name} to resume, or "
                f"/sop {name} --fresh to start over."
            )

        state, error = self._enter_sop(name, yolo=yolo)
        if error:
            return error
        # At most one active SOP — auto-pause the current one, if any.
        if self.sop_state is not None:
            self.sop_state.suspension_reason = "paused"
            self.sop_state.suspended_at = _now_iso()
            self._suspended_sops.insert(0, self.sop_state)
        self.sop_state = state
        if state.yolo_mode:
            self.yolo_mode = True
        if request:
            self._pending_followup = request
            return f"Entered SOP '{name}'. Starting on: {request}"
        return f"Entered SOP '{name}'."

    @command("pause_sop", description="Pause the active SOP for a short ad-hoc diversion",
             requires_active_sop=True)
    async def _cmd_pause_sop(self) -> str:
        s = self.sop_state
        s.suspension_reason = "paused"
        s.suspended_at = _now_iso()
        self._suspended_sops.insert(0, s)
        self.sop_state = None
        return (
            f"SOP '{s.sop_name}' paused at {s.sop_status}. "
            f"I'll remind you to resume."
        )

    @command("exit_sop", description="Exit the active SOP (resumable later)",
             requires_active_sop=True)
    async def _cmd_exit_sop(self) -> str:
        s = self.sop_state
        s.suspension_reason = "exited"
        s.suspended_at = _now_iso()
        self._suspended_sops.insert(0, s)
        self.sop_state = None
        return (
            f"Exited SOP '{s.sop_name}' ({s.sop_status}). "
            f"Resume anytime with /resume_sop {s.sop_name}."
        )

    @command("resume_sop",
             description=(
                 "Resume a paused or exited SOP (optionally by name), optionally "
                 "with a request to continue on. Usage: /resume_sop [name] [request...]"
             ),
             requires_args=True)
    async def _cmd_resume_sop(self, args: str = "") -> str:
        if not self._suspended_sops:
            return "No suspended SOPs to resume."
        tokens = args.split()
        # Extract a request ONLY when the first token unambiguously names a
        # suspended SOP — then the remainder is the free-text request to continue
        # on. Otherwise preserve the original contract: the whole arg is the
        # target name (so a mistyped/unknown name still yields a clear error
        # rather than silently resuming the most-recent SOP).
        target = ""
        request = ""
        if tokens and any(s.sop_name == tokens[0] for s in self._suspended_sops):
            target = tokens[0]
            request = " ".join(tokens[1:]).strip()
        else:
            target = args.strip()
        if target:
            match = next(
                (s for s in self._suspended_sops if s.sop_name == target), None
            )
            if match is None:
                avail = ", ".join(s.sop_name for s in self._suspended_sops)
                return f"No suspended SOP named '{target}'. In-progress: {avail}"
        else:
            match = self._suspended_sops[0]  # most recent
        # Auto-pause the active SOP, if any.
        if self.sop_state is not None:
            self.sop_state.suspension_reason = "paused"
            self.sop_state.suspended_at = _now_iso()
            self._suspended_sops.insert(0, self.sop_state)
        self._suspended_sops.remove(match)
        match.suspension_reason = ""
        match.suspended_at = ""
        self._reload_sop_definition(match)  # no-op if .sop already attached
        self.sop_state = match
        if request:
            self._pending_followup = request
            return (
                f"Resumed SOP '{match.sop_name}' at {match.sop_status}. "
                f"Continuing on: {request}"
            )
        return f"Resumed SOP '{match.sop_name}' at {match.sop_status}."

    def _enter_sop(self, name: str, *, yolo: bool = False):
        """Build an SOPState for ``name`` via the shared loader.

        Returns ``(SOPState, None)`` or ``(None, error_message)``.
        """
        from agent_foundation.resources.tools.sop.executor import build_sop_state

        return build_sop_state(
            name, yolo=yolo, extra_sop_dirs=self._extra_sop_dirs or None
        )

    def _reload_sop_definition(self, state) -> None:
        """Reattach the SOP definition object after deserialization.

        Safe no-op when the definition is already attached (e.g. an in-memory
        suspended SOP being resumed in the same process).
        """
        if state.sop_name and state.sop is None:
            from agent_foundation.resources.sops.registry import load_sop

            state.sop = load_sop(state.sop_name).sop

    def _format_suspended_sops(self) -> tuple[str, str]:
        """Render the (paused_sop nudge, inprogress_sops list) prompt strings.

        Rendering-only: the most-recent paused SOP becomes the singular active
        reminder; every other suspended SOP (older paused + all exited) is
        listed passively. ``suspension_reason`` is never mutated here, so the
        user's true pause/exit intent is preserved in state.
        """
        if not self._suspended_sops:
            return "", ""
        most_recent_paused = next(
            (s for s in self._suspended_sops if s.suspension_reason == "paused"),
            None,
        )
        paused_sop = (
            f"{most_recent_paused.sop_name} ({most_recent_paused.sop_status})"
            if most_recent_paused is not None
            else ""
        )
        others = [s for s in self._suspended_sops if s is not most_recent_paused]
        inprogress_sops = "\n".join(
            f"- **{s.sop_name}** ({s.sop_status})" for s in others
        )
        return paused_sop, inprogress_sops

    @command("model", description="Change the LLM model",
             aliases=("set_model",), requires_args=True)
    async def _cmd_set_model(self, model_name: str = "") -> str:
        if not model_name:
            current = self.prior_context.get("model_name", "default")
            return f"Current model: {current}. Usage: /model <name>"
        self.prior_context["model_name"] = model_name
        return f"Model set to {model_name}."

    @command("root", description="Set the session root directory",
             aliases=("set_session_root",), requires_args=True)
    async def _cmd_set_session_root(self, path: str = "") -> str:
        if not path:
            current = self.prior_context.get("session_root_path", "not set")
            return f"Current session root: {current}. Usage: /root <path>"
        self.prior_context["session_root_path"] = path
        return f"Session root set to {path}."

    @command("target", description="Set the workflow target path",
             aliases=("set_workflow_target_path",), requires_args=True)
    async def _cmd_set_target(self, path: str = "") -> str:
        if not path:
            current = self.prior_context.get("workflow_target_path", "not set")
            return f"Current target path: {current}. Usage: /target <path>"
        self.prior_context["workflow_target_path"] = path
        return f"Target path set to {path}."

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
        import asyncio as _asyncio

        if self._inbox is not None:
            raise RuntimeError("Inbox already enabled")
        self._inbox = _asyncio.Queue(maxsize=maxsize)
        self._default_interactive = interactive
        self._auto_shutdown_on_sop_complete = auto_shutdown_on_sop_complete
        self._on_new_turn = on_new_turn
        self._on_prompt_rendered = on_prompt_rendered
        self._on_turn_complete = on_turn_complete

    def inbox_put(self, item) -> None:
        """Non-blocking enqueue. Raises RuntimeError if inbox not enabled."""
        if self._inbox is None:
            raise RuntimeError("Inbox not enabled; call enable_inbox() first")
        self._inbox.put_nowait(item)

    def inbox_put_user(self, content: str, source: str = "user") -> None:
        """Convenience: enqueue a UserMessage."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox import (
            UserMessage,
        )
        self.inbox_put(UserMessage(content=content, source=source))

    def request_shutdown(self) -> None:
        """Cooperative termination. Current run_agentic_loop finishes, then run() returns."""
        self._shutdown_requested = True

    def _next_turn_number(self) -> int:
        self._turn_counter += 1
        return self._turn_counter

    def _content_for_item(self, item) -> str | None:
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox import (
            UserMessage,
            ToolCompletion,
            SyntheticContinue,
            _SYNTHETIC_CONTINUE,
        )
        if isinstance(item, UserMessage):
            return item.content
        if isinstance(item, ToolCompletion):
            return _CONTINUE_AFTER_TOOLS
        if isinstance(item, SyntheticContinue):
            return _SYNTHETIC_CONTINUE
        return None

    async def run(self) -> AgenticResult | None:
        """Long-lived event loop. Drains inbox, calls run_agentic_loop per item.

        Returns when request_shutdown() is called.
        """
        if self._inbox is None:
            raise RuntimeError("Inbox not enabled; call enable_inbox() first")
        if self._running:
            raise RuntimeError("run() is already executing")
        self._running = True
        try:
            last_result: AgenticResult | None = None
            while not self._shutdown_requested:
                item = await self._inbox.get()
                try:
                    content = self._content_for_item(item)
                    if content is None:
                        continue
                    last_result = await self.run_agentic_loop(
                        content=content,
                        interactive=self._default_interactive,
                        turn_number=self._next_turn_number(),
                        on_new_turn=self._on_new_turn,
                        on_prompt_rendered=self._on_prompt_rendered,
                        on_turn_complete=self._on_turn_complete,
                    )
                except Exception as e:
                    logger.exception("Inbox item %r failed: %s", item, e)
                finally:
                    self._inbox.task_done()
            return last_result
        finally:
            self._running = False

    # =========================================================================
    # Phase Completion Detection (Model A)
    # =========================================================================

    def _check_phase_completion(self, tool_name: str = "") -> None:
        """Detect SOP phase completion and advance to next phase.

        Called after _execute_tool_call() applies context_updates.
        Three detection strategies:
          1. All-must-tools: ALL tools declared in ``Tools[__must__]`` for
             this phase have been executed (not just any single one).
          2. User input: user_input_gate_passed + DIRECTIVE_REQUIRES_USER_INPUT
          3. All-outputs-present: every declared output in phase_outputs
        """
        if not self.sop_state or not self.sop_state.sop:
            return

        from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus

        s = self.sop_state
        sop = s.sop
        current = s.current_phase
        if not current:
            return

        from rich_python_utils.common_objects.workflow.stategraph import StateGraphTracker

        completed_ids = [
            r.phase if hasattr(r, "phase") else str(r)
            for r in s.completed_phases
        ]
        if current in completed_ids:
            return

        phase = None
        for p in sop.phases:
            if p.id == current:
                phase = p
                break
        if phase is None:
            return

        detected = False

        if tool_name and s.tool_phase_map.get(tool_name) == current:
            executed = s.phase_executed_tools.setdefault(current, set())
            executed.add(tool_name)
            required = s.phase_required_tools.get(current, set())
            if not required or required <= executed:
                detected = True

        if not detected and s.user_input_gate_passed:
            if DIRECTIVE_REQUIRES_USER_INPUT in " ".join(getattr(phase, "directives", [])):
                detected = True

        if not detected and hasattr(phase, "outputs") and phase.outputs:
            if all(o in s.phase_outputs for o in phase.outputs):
                detected = True

        if not detected:
            return

        completed_ids.append(current)
        s.completed_phases = completed_ids

        tracker = StateGraphTracker(
            graph=sop,
            current_state=None,
            state_status=PhaseStatus.COMPLETED,
            completed_states=completed_ids,
            state_outputs=s.phase_outputs,
            goto_counts=s.goto_counts,
        )
        available = tracker.get_available_next()
        if available:
            s.current_phase = available[0].id
            s.phase_status = PhaseStatus.RUNNING
        else:
            s.current_phase = None
            s.phase_status = PhaseStatus.COMPLETED

        s.user_input_gate_passed = False
        logger.info("SOP phase %s completed; next=%s", current, s.current_phase)

        # Auto-shutdown bridge: signal run() to exit when SOP finishes
        if (
            self._auto_shutdown_on_sop_complete
            and s.current_phase is None
            and s.phase_status == PhaseStatus.COMPLETED
        ):
            self.request_shutdown()

    # =========================================================================
    # Prompt Rendering
    # =========================================================================

    def _render_prompt(self, current_message: str) -> str:
        """Build template variables and render via prompt_renderer."""
        # Format tools — separate action tools from conversation tools
        formatter = ToolMarkdownFormatter()
        tools_list = list(self.tool_registry.values())
        # Exclude user-only tools (agent_enabled=False) from LLM prompt
        agent_tools = [t for t in tools_list if getattr(t, 'agent_enabled', True)]
        action_tools = [t for t in agent_tools if t.tool_type != "Conversation"]
        available_tools = formatter.format_all(action_tools)

        # Append commands to action tools — indistinguishable to the LLM
        commands_text = self._commands.render_for_prompt()
        if commands_text:
            available_tools = f"{available_tools}\n\n{commands_text}" if available_tools else commands_text

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
        template_vars = getattr(
            self.prompt_renderer, "template_variables", {}
        ) or {}

        # Evaluate SOP to generate nextstep guidance
        nextstep_guidance = ""
        sop = self.sop_state.sop if self.sop_state else None

        # Legacy auto-discover: only if no SOPState is active
        if sop is None and self.sop_state is None:
            sop_path = getattr(self.prompt_renderer, "find_sop_file", lambda: None)()
            if sop_path is not None:
                from pathlib import Path as _Path
                from agent_foundation.common.workflow.sop_state import SOPState as _SOPState
                loaded_sop = SOPManager.load(sop_path)
                self.sop_state = _SOPState(
                    sop=loaded_sop,
                    sop_name=loaded_sop.name or _Path(sop_path).stem,
                    tool_phase_map=(
                        loaded_sop.tool_to_phase_map
                        if hasattr(loaded_sop, "tool_to_phase_map") else {}
                    ),
                )
                sop = loaded_sop

        if sop is not None and self.sop_state is not None:
            try:
                from rich_python_utils.common_objects.workflow.stategraph import (
                    StateGraphTracker,
                )

                s = self.sop_state
                completed = [
                    r.phase if hasattr(r, "phase") else str(r)
                    for r in s.completed_phases
                ]
                tracker = StateGraphTracker(
                    graph=sop,
                    current_state=None,
                    state_status="idle",
                    completed_states=completed,
                    state_outputs=s.phase_outputs,
                    goto_counts=s.goto_counts,
                )

                if s.user_input_gate_passed:
                    from rich_python_utils.string_utils.formatting.template_manager.sop_manager import SOPPhase
                    for node in tracker.get_available_next():
                        if not isinstance(node, SOPPhase):
                            continue
                        has_tools = any(
                            sub.name.lower() in ("tools", "command")
                            for sub in getattr(node, "subsections", [])
                        )
                        if not has_tools and "requires user input" in " ".join(
                            getattr(node, "directives", [])
                        ):
                            tracker.completed_states.add(node.id)
                            s.user_input_gate_passed = False
                            break

                nextstep_guidance = SOPManager.render_guidance(
                    tracker, sop, context=dict(self.prior_context),
                )
            except Exception as e:
                logger.warning("SOP evaluation failed: %s", e)

        # When no SOP is active, show the list of available SOPs so the LLM
        # can discover and suggest them to the user. Use the host-provided
        # extra dirs so server/OpenTeam SOPs are discoverable too.
        available_sops = ""
        if self.sop_state is None:
            try:
                from agent_foundation.resources.sops.registry import (
                    load_all_sops,
                    format_all_sops,
                )
                sops = load_all_sops(extra_dirs=self._extra_sop_dirs or None)
                # Apply allow-then-deny discovery filters (precedence: same
                # as iptables / AWS IAM / k8s NetworkPolicy).
                #
                # Step 1: if a non-empty whitelist is set, keep ONLY those.
                # Step 2: drop anything in the denylist from the survivors.
                #
                # Purely cosmetic — hidden SOPs remain loadable via
                # /sop <name> explicitly; this only controls what the LLM
                # surfaces unprompted in the "Available SOPs" section.
                if sops and self.allowed_sops:
                    whitelist = set(self.allowed_sops)
                    sops = {n: i for n, i in sops.items() if n in whitelist}
                if sops and self.disallowed_sops:
                    denylist = set(self.disallowed_sops)
                    sops = {n: i for n, i in sops.items() if n not in denylist}
                if sops:
                    available_sops = format_all_sops(sops)
            except ImportError:
                pass

        # Suspended-SOP prompt sections (driven from CI state, replacing the
        # never-populated WorkflowManager-fed `active_sops`).
        paused_sop, inprogress_sops = self._format_suspended_sops()

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

        # Resolve feed values that are themselves templates (e.g., SOP guidance
        # containing {{ session_root_path }}).  Uses the same Jinja2 Environment
        # as the main template so behaviour is identical.
        if hasattr(self.prompt_renderer, "render_string"):
            try:
                from rich_python_utils.string_utils.formatting.common import (
                    resolve_templated_feed,
                )
                from rich_python_utils.string_utils.formatting.jinja2_format import (
                    extract_variables as jinja2_extract_variables,
                )

                feed = resolve_templated_feed(
                    feed,
                    extract_variables=jinja2_extract_variables,
                    render_template=self.prompt_renderer.render_string,
                )
            except ValueError as e:
                logger.warning("Feed self-resolution failed: %s", e)

        self._last_template_feed = dict(feed)
        self._last_template_source = self.prompt_renderer.template_source
        self._last_template_config = getattr(
            self.prompt_renderer, "template_config", {}
        ) or {}
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
    # Tool Execution
    # =========================================================================

    async def _execute_tool_call(self, tool_call: Any) -> str:
        """Execute a tool call and apply context_updates from the result.

        Tools marked asynchronous=True in the tool registry are launched as
        background asyncio tasks (fire-and-forget) so the conversation turn
        completes immediately. The tool sends task_status notifications to
        the frontend independently.
        """
        import asyncio

        canonical = self._resolve_tool_name(tool_call.name)

        # Commands can be invoked as tools by the LLM (e.g., "set_model")
        if self._commands.is_command_name(canonical):
            result = await self._commands.dispatch_as_tool(
                canonical, tool_call.arguments or {},
            )
            self._check_phase_completion(tool_name=canonical)
            # A command (e.g. /sop, /resume_sop) may seed an initial request for
            # the just-entered SOP. Surface it as a user turn so the loop's next
            # self-continuation acts on the concrete goal, not just the SOP guidance.
            followup = self._consume_pending_followup()
            if followup:
                self.add_message("user", followup)
            return result

        if self.tool_executor is None:
            return f"No tool executor configured for: {canonical}"

        # Check if this tool should run asynchronously (fire-and-forget)
        tool_def = self.tool_registry.get(canonical)
        is_async = tool_def and getattr(tool_def, "asynchronous", False)

        if is_async:
            executor = self.tool_executor

            if self.sop_state:
                from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus
                tool_map = self.sop_state.tool_phase_map
                sop_phase = tool_map.get(canonical)
                if sop_phase:
                    self.sop_state.current_phase = sop_phase
                    self.sop_state.phase_status = PhaseStatus.RUNNING

            async def _run_async() -> None:
                try:
                    result = await executor(canonical, tool_call.arguments)
                    if hasattr(result, "context_updates") and result.context_updates:
                        self.update_prior_context(**result.context_updates)
                    if hasattr(result, "result"):
                        self.add_message(
                            "user",
                            f"{_TOOL_RESULTS_PREFIX}\n{canonical}: {result.result}",
                        )
                    self._check_phase_completion(tool_name=canonical)
                    # Wake the event loop (if inbox enabled)
                    if self._inbox is not None:
                        from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox import (
                            ToolCompletion,
                        )
                        try:
                            self._inbox.put_nowait(ToolCompletion(tool_name=canonical))
                        except Exception:
                            logger.warning("Inbox put failed for tool %s", canonical)
                except Exception as e:
                    logger.error("Async tool %s failed: %s", canonical, e)

            self._active_async_task = asyncio.create_task(_run_async())
            self._async_tool_dispatched = True
            return (
                f"Tool '{canonical}' launched asynchronously. "
                f"Check the task panel for progress and results."
            )

        try:
            result = await self.tool_executor(canonical, tool_call.arguments)
            # result is ToolExecutionResult — apply context_updates to prior_context
            if hasattr(result, "context_updates") and result.context_updates:
                self.update_prior_context(**result.context_updates)
            self._check_phase_completion(tool_name=canonical)
            if hasattr(result, "result"):
                return result.result
            return str(result)
        except Exception as e:
            logger.error("Tool execution error for %s: %s", canonical, e)
            return f"Error executing {canonical}: {e}"

    def _resolve_tool_name(self, name: str) -> str:
        """Resolve a tool name or alias to the canonical tool name.

        Strips a leading ``/`` first — the LLM sometimes emits an action
        name like ``/sop`` copying the prompt's slash-command prose, and
        command/registry keys are never slash-prefixed. Then matches
        against each tool's ``name``, ``aliases``, and ``preferred_prompt_alias``.
        """
        if name.startswith("/"):
            name = name[1:]
        if name in self.tool_registry:
            return name
        normalized = name.replace("-", "_")
        for tool in self.tool_registry.values():
            if (
                name in getattr(tool, "aliases", [])
                or normalized == tool.name
                or normalized == getattr(tool, "preferred_prompt_alias", "")
            ):
                return tool.name
        if normalized in self.tool_registry:
            return normalized
        return name

    @property
    def _valid_tool_names(self) -> set[str]:
        """Set of valid tool names including aliases and preferred prompt aliases."""
        names: set[str] = set()
        for tool in self.tool_registry.values():
            names.add(tool.name)
            for alias in getattr(tool, "aliases", []):
                names.add(alias)
            preferred = getattr(tool, "preferred_prompt_alias", "")
            if preferred:
                names.add(preferred)
        return names

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
            inference_input, inference_config, **_inference_args
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
            inference_input, inference_config, **_inference_args
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

    def _synthesize_yolo_collected(
        self, tools: list,
    ) -> dict[str, str] | str | None:
        """Synthesize responses for conversation tools in yolo mode.

        Uses per-tool yolo_default from tool.json, with per-SOP overrides
        from sop.config.json yolo_overrides. Falls back to "Follow your
        best judgment." for unconfigured tools.
        """
        if not tools:
            return None

        # Synthesize each tool into a UI-shaped response, then route it through
        # the SAME decode/finalize/publish path as a real user response. This
        # keys bindings by the declared output_vars (NOT tool_type), produces
        # composite mode+nested bindings, finalizes paths, and persists — exactly
        # like interactive mode (one source of truth).
        collected: dict[str, str] = {}
        for tool in tools:
            response = self._synthesize_yolo_response(tool)
            collected.update(
                decode_tool_bindings(
                    tool, response, session_root=self._session_root()
                )
            )
        if collected:
            self.set_session_variables(collected)
        return collected

    def _synthesize_yolo_response(self, tool):
        """Build a UI-shaped response for a tool under yolo, from its yolo spec.

        Path inputs never fabricate prose as a path — they default to the
        resolved session root (a valid, autonomous "investigate everything"
        target) rather than the generic free-text default.
        """
        spec = self._resolve_yolo_spec(tool)
        mode = spec.get("mode", "fixed")
        choices = getattr(tool, "choices", []) or []
        if mode == "first_choice" and choices:
            return {"choice_index": 0}
        if mode == "select_all" and choices:
            if tool.tool_type == ConversationToolType.PROPOSAL_SELECTION:
                return {"selected_proposals": [getattr(c, "value", "") for c in choices]}
            return {"content": ",".join(getattr(c, "value", "") or "" for c in choices)}
        if mode == "confirm":
            return {"choice": "yes"}
        if mode == "decline":
            return {"choice": "no"}
        # fixed / none / fallback → a free-text value.
        value = spec.get("value", "Follow your best judgment.")
        if tool.expected_input_type == "path":
            # A path input cannot meaningfully be a prose sentence under yolo;
            # default to the resolved session root.
            value = self._session_root() or value
        return {"content": value}

    def _synthesize_single_yolo(self, tool) -> str:
        """Synthesize a single conversation tool response for yolo mode."""
        # Resolve yolo spec: per-SOP override → tool.json default → builtin
        spec = self._resolve_yolo_spec(tool)
        mode = spec.get("mode", "fixed")

        if mode == "fixed":
            return spec.get("value", "Follow your best judgment.")
        elif mode == "select_all":
            choices = getattr(tool, "choices", []) or []
            if isinstance(choices, list) and choices:
                values = []
                for c in choices:
                    if isinstance(c, dict):
                        values.append(c.get("value", c.get("label", "")))
                    else:
                        values.append(str(c))
                return ", ".join(values)
            return "Follow your best judgment."
        elif mode == "first_choice":
            choices = getattr(tool, "choices", []) or []
            if isinstance(choices, list) and choices:
                c = choices[0]
                if isinstance(c, dict):
                    return c.get("value", c.get("label", ""))
                return str(c)
            return "Follow your best judgment."
        elif mode == "confirm":
            return "yes"
        elif mode == "decline":
            return "no"
        elif mode == "none":
            return "Follow your best judgment."
        else:
            return spec.get("value", "Follow your best judgment.")

    def _resolve_yolo_spec(self, tool) -> dict:
        """Resolution order: per-SOP override → tool.json default → builtin."""
        tool_type = getattr(tool, "tool_type", "")

        # Check per-SOP yolo_overrides
        sop_instance_id = self.prior_context.get("sop_instance_id")
        if sop_instance_id and hasattr(self, "workflow_manager") and self.workflow_manager:
            try:
                instance = self.workflow_manager.active_instances.get(sop_instance_id)
                if instance:
                    definition = self.workflow_manager.registry.get(instance.definition_id)
                    if hasattr(definition, "frontmatter"):
                        overrides = definition.frontmatter.get("yolo_overrides", {})
                        if tool_type in overrides:
                            return overrides[tool_type]
            except Exception:
                pass

        # Check tool.json yolo_default
        tool_name = getattr(tool, "tool_type", "") or getattr(tool, "name", "")
        tool_def = self.tool_registry.get(tool_name)
        if tool_def and getattr(tool_def, "yolo_default", None):
            return tool_def.yolo_default

        # Builtin fallback
        return {"mode": "fixed", "value": "Follow your best judgment."}

    def _resolve_proposals_source(self, tool: ConversationTool) -> "Optional[dict]":
        """Resolve proposal data for a ``proposal_selection`` tool.

        Priority:
          1. ``tool.metadata["proposals"]`` already present (dict) — use as-is.
          2. ``tool.metadata["proposals_path"]`` → AF ``parse_proposal_file``.
             This is the AF-native path: the SOP body passes proposals_path
             (typically via Jinja ``{{ workspace_path__research_propose }}``).
          3. A host-registered :class:`ProposalParser` (e.g. RankEvolve), fed a
             workspace discovered from ``prior_context``.

        Returns a plain ``ProposalIndex.to_dict()``-shaped dict, or ``None``.
        """
        meta = tool.metadata or {}
        existing = meta.get("proposals")
        if isinstance(existing, dict) and existing:
            return existing

        path = meta.get("proposals_path")
        if path:
            try:
                from pathlib import Path as _P

                from agent_foundation.common.data_models.proposal.parser import (
                    parse_proposal_file,
                )

                index = parse_proposal_file(_P(str(path)))
                if index is not None:
                    return index.to_dict()
                logger.warning(
                    "[proposal_selection] proposals_path did not parse: %s", path
                )
            except Exception as exc:  # noqa: BLE001 — enrichment is best-effort
                logger.warning(
                    "[proposal_selection] failed to parse proposals_path %s: %s",
                    path, exc,
                )

        # Host-provided parser fallback (e.g. RankEvolve registers parse_proposals).
        try:
            from agent_foundation.common.data_models.proposal.parsers import (
                get_proposal_parser,
            )

            parser = get_proposal_parser()
            if parser is not None:
                workspace = (
                    meta.get("workspace")
                    or self.prior_context.get("workspace_path__research_propose")
                    or self.prior_context.get("workspace_path")
                )
                if workspace:
                    data = parser.parse(str(workspace))
                    if data is not None:
                        return data.to_dict() if hasattr(data, "to_dict") else data
        except Exception as exc:  # noqa: BLE001
            logger.info("[proposal_selection] registered parser failed: %s", exc)

        return None

    def _enrich_proposal_selection(self, tool: ConversationTool) -> None:
        """Populate proposals + choices + output var for a proposal_selection tool.

        Attaches the resolved proposal payload to ``tool.metadata["proposals"]``
        for the rich widget, derives one selectable choice per proposal id (so
        selection flows through AF's multiple-choice machinery, including yolo
        ``select_all``), and defaults the output variable to
        ``selected_proposal_ids`` when the SOP author omitted it.
        """
        proposals = self._resolve_proposals_source(tool)
        if not proposals:
            return
        if tool.metadata is None:
            tool.metadata = {}
        tool.metadata["proposals"] = proposals

        if not tool.choices:
            choices: list[ChoiceItem] = []
            for group in proposals.get("groups", []):
                for p in group.get("proposals", []):
                    pid = str(p.get("id", "")).strip()
                    if not pid:
                        continue
                    title = p.get("title", "") or pid
                    bits = [b for b in (p.get("impact"), p.get("complexity")) if b]
                    suffix = f" ({', '.join(bits)})" if bits else ""
                    choices.append(
                        ChoiceItem(
                            label=f"{pid}: {title}{suffix}",
                            value=pid,
                            description=p.get("summary", "") or "",
                        )
                    )
            tool.choices = choices
            tool.metadata.setdefault("proposals_count", len(choices))

        tool.show_select_all = True
        if not tool.output_vars:
            tool.output_vars = ["selected_proposal_ids"]

    async def _handle_conversation_tool(
        self,
        tool: ConversationTool,
        assistant_text: str,
        interactive_override: Optional[InteractiveBase] = None,
    ) -> Optional[str]:
        """Handle a single conversation tool by collecting user input.

        Enriches the input_mode with variable content metadata (for UI display)
        and processes the response with choice_index->value mapping and
        variable override application.
        """
        active_interactive = interactive_override or self.interactive
        if active_interactive is None:
            return None

        # Nested (composite-choice) bindings captured this call, so the caller
        # can include them in the collected dict — not only session variables.
        self._last_conv_nested_bindings = {}

        # Resolve any templated prefix (e.g. echoed "{{ session_root_path }}")
        # before building the UI config / finalising values.
        render_templated_fields(tool, self._make_field_renderer())
        input_mode = _build_input_mode(tool)

        # Enrich with variable content for UI display (editable text block)
        if self.prompt_renderer:
            try:
                var_name = tool.output_vars[0] if tool.output_vars else None
                vm = self.prompt_renderer.variable_manager

                # If output_vars is set, resolve directly
                if var_name:
                    content = vm.get_effective_value(var_name, skip_overrides=True)
                    if isinstance(content, dict):
                        input_mode.metadata["variable_content"] = {
                            k: str(v).strip() for k, v in content.items()
                        }
                        input_mode.metadata["variable_name"] = var_name
                # Otherwise, try to auto-detect by matching choice values
                # against known alias-target dicts in the variable manager
                elif tool.tool_type == "single_choice" and tool.choices:
                    choice_values = [
                        c.get("value", "").lower().replace(" ", "_").replace("-", "_")
                        for c in tool.choices if c.get("value")
                    ]
                    for alias in getattr(vm, "_scoped_aliases", {}).values():
                        try:
                            candidate = vm.get_effective_value(alias, skip_overrides=True)
                            if isinstance(candidate, dict):
                                norm_keys = {
                                    k.lower().replace(" ", "_").replace("-", "_"): k
                                    for k in candidate
                                }
                                if choice_values and all(
                                    v in norm_keys for v in choice_values
                                ):
                                    input_mode.metadata["variable_content"] = {
                                        k: str(v).strip()
                                        for k, v in candidate.items()
                                    }
                                    input_mode.metadata["variable_name"] = alias
                                    break
                        except Exception:
                            continue
            except Exception:
                pass  # Non-critical — widget works without enrichment

        # Pass prompt_data inline so the UI's "View Prompt" button on the
        # widget preamble has the rendered prompt available without a REST
        # round-trip. Server-side transports (e.g. WebSocketInteractive) read
        # this kwarg via **kwargs; transports that don't care simply ignore it.
        _prompt_data = {
            "template_source": getattr(self, "_last_template_source", "") or "",
            "template_feed": getattr(self, "_last_template_feed", {}) or {},
            "rendered_prompt": getattr(self, "_last_rendered_prompt", "") or "",
            "template_config": getattr(self, "_last_template_config", {}) or {},
        }
        await active_interactive.asend_response(
            assistant_text,
            flag=InteractionFlags.PendingInput,
            input_mode=input_mode,
            prompt_data=_prompt_data,
        )

        user_input = await active_interactive.aget_input()
        if user_input is None:
            return None

        # Extract the response payload
        if isinstance(user_input, dict):
            response = user_input.get(
                "user_input", user_input.get("content", user_input)
            )
        else:
            return str(user_input)

        # Process structured widget response (dict with choice_index)
        if isinstance(response, dict):
            # Multi-select capture for proposal_selection: collect the list of
            # selected proposal ids and persist it as a comma-joined string
            # under the output variable(s) — e.g. selected_proposal_ids, which
            # Phase 4 consumes as `task --proposal-ids P1,P3`. Scoped strictly to
            # PROPOSAL_SELECTION so no other tool's decode path is affected.
            selected_list = None
            if tool.tool_type == ConversationToolType.PROPOSAL_SELECTION:
                if isinstance(response.get("selected_proposals"), list):
                    selected_list = response["selected_proposals"]
                elif isinstance(response.get("selected"), list):
                    selected_list = response["selected"]
                elif isinstance(response.get("choice_indices"), list) and tool.choices:
                    selected_list = [
                        tool.choices[i].value
                        for i in response["choice_indices"]
                        if isinstance(i, int) and 0 <= i < len(tool.choices)
                    ]
            if selected_list is not None:
                joined = ",".join(str(s) for s in selected_list)
                if tool.output_vars:
                    self.set_session_variables(
                        {var: joined for var in tool.output_vars}
                    )
                return joined

            # Handle confirmation widget response with param_overrides
            if "choice" in response:
                choice_value = response["choice"]
                param_overrides = response.get("param_overrides")
                if param_overrides:
                    self._pending_param_overrides = param_overrides
                variables = response.get("variables")
                if variables and isinstance(variables, dict):
                    self._pending_variables = variables
                return choice_value

            # Map choice_index -> choice value
            choice_idx = response.get("choice_index")
            if (
                choice_idx is not None
                and tool.choices
                and 0 <= choice_idx < len(tool.choices)
            ):
                choice_value = tool.choices[choice_idx].value
            else:
                choice_value = response.get("custom_text", str(response))

            # Apply variable override if user edited the content
            variable_override = response.get("variable_override")
            if variable_override:
                self.set_session_variables(variable_override)
            elif tool.output_vars:
                self.set_session_variables(
                    {var: choice_value for var in tool.output_vars}
                )

            # Composite choice: bind the selected choice's embedded input value
            # to its own variable (distinct from the mode var bound above).
            if (
                choice_idx is not None
                and tool.choices
                and 0 <= choice_idx < len(tool.choices)
                and tool.choices[choice_idx].has_input
            ):
                spec = tool.choices[choice_idx].input
                inputs = response.get("inputs")
                raw = None
                if isinstance(inputs, dict):
                    raw = inputs.get(spec.name, next(iter(inputs.values()), None))
                elif "content" in response:
                    raw = response.get("content")
                nested_val = finalize_input_value(
                    raw,
                    expected_input_type=spec.expected_input_type,
                    prefix=spec.prefix,
                    allow_multiple_input=spec.allow_multiple_input,
                    serialization=spec.serialization,
                    session_root=self._session_root(),
                )
                if spec.name:
                    self.set_session_variables({spec.name: nested_val})
                    self._last_conv_nested_bindings = {spec.name: nested_val}

            return choice_value

        # Clarification / typed free-text (incl. path): finalize (path re-join +
        # serialise) then persist to the declared output variable(s).
        final = finalize_input_value(
            response,
            expected_input_type=tool.expected_input_type,
            prefix=tool.prefix,
            allow_multiple_input=tool.allow_multiple_input,
            serialization=tool.serialization,
            session_root=self._session_root(),
        )
        if tool.output_vars and final:
            self.set_session_variables({v: final for v in tool.output_vars})
        return final

    async def _handle_conversation_tools(
        self,
        tools: list[ConversationTool],
        assistant_text: str,
        interactive_override: Optional[InteractiveBase] = None,
        action_tools: Optional[list[dict]] = None,
    ) -> Optional[dict[str, str]]:
        """Handle conversation tools by presenting a compound widget.

        For a single tool, delegates to _handle_conversation_tool().
        For multiple tools, bundles all into one compound pending_input
        so the frontend renders them as a tabbed multi-input widget.

        Returns a dict mapping output variable names to user values,
        or None if input collection fails.
        """
        if not tools:
            return None

        active_interactive = interactive_override or self.interactive
        if active_interactive is None:
            return None

        # Single tool: delegate to simple handler for backward compat
        if len(tools) == 1:
            tool = tools[0]
            # For confirmation tools, enrich with action tool parameters
            # so the frontend can show a config panel
            if (
                tool.tool_type == ConversationToolType.CONFIRMATION
                and action_tools
                and self.tool_registry
            ):
                tool_params = []
                for at in action_tools:
                    tool_name = at.get("name", "")
                    canonical = self._resolve_tool_name(tool_name)
                    tool_def = self.tool_registry.get(canonical)
                    if tool_def:
                        tool_params.extend(
                            p.to_dict() for p in tool_def.parameters
                            if not p.positional
                        )
                if tool_params:
                    # Will be added to input_mode metadata via _handle_conversation_tool
                    tool._tool_params = tool_params
            # Inject view path for generated documentation if available
            if tool.tool_type == ConversationToolType.CONFIRMATION:
                workflow_target_path = self.prior_context.get("workflow_target_path", "")
                if workflow_target_path:
                    from pathlib import Path as _Path

                    target_dir = _Path(workflow_target_path)
                    if target_dir.is_file():
                        target_dir = target_dir.parent
                    docs_index = target_dir / "docs" / "_build" / "html" / "index.html"
                    if docs_index.exists():
                        if not tool.metadata:
                            tool.metadata = {}
                        tool.metadata.setdefault("view", str(docs_index))
            result = await self._handle_conversation_tool(
                tool, assistant_text, interactive_override
            )
            if result is None:
                return None
            # Signal confirmation gate passed for state tracker auto-completion
            if (
                tool.tool_type == ConversationToolType.CONFIRMATION
                and str(result).lower() in ("yes", "proceed")
            ):
                self.update_prior_context(_user_input_gate_passed=True)
                if self.sop_state:
                    self.sop_state.user_input_gate_passed = True
            var_name = tools[0].output_vars[0] if tools[0].output_vars else "input"
            collected = {var_name: result}
            # Include any composite-choice nested binding so the synthesized user
            # message and __var__ action-arg substitution see the entered value,
            # matching the compound/yolo paths (one source of truth).
            collected.update(getattr(self, "_last_conv_nested_bindings", {}) or {})
            return collected

        # Multiple tools: send ALL as a compound widget in one pending_input
        _field_renderer = self._make_field_renderer()
        tool_configs = []
        for tool in tools:
            # Resolve any templated prefix before building UI config.
            render_templated_fields(tool, _field_renderer)
            mode = _build_input_mode(tool)

            # Enrich with variable content for UI display (editable text block)
            if self.prompt_renderer:
                try:
                    var_name = tool.output_vars[0] if tool.output_vars else None
                    vm = self.prompt_renderer.variable_manager

                    if var_name:
                        content = vm.get_effective_value(var_name, skip_overrides=True)
                        if isinstance(content, dict):
                            mode.metadata["variable_content"] = {
                                k: str(v).strip() for k, v in content.items()
                            }
                            mode.metadata["variable_name"] = var_name
                    elif tool.tool_type == "single_choice" and tool.choices:
                        choice_values = [
                            c.get("value", "").lower().replace(" ", "_").replace("-", "_")
                            for c in tool.choices if c.get("value")
                        ]
                        for alias in getattr(vm, "_scoped_aliases", {}).values():
                            try:
                                candidate = vm.get_effective_value(alias, skip_overrides=True)
                                if isinstance(candidate, dict):
                                    norm_keys = {
                                        k.lower().replace(" ", "_").replace("-", "_"): k
                                        for k in candidate
                                    }
                                    if choice_values and all(
                                        v in norm_keys for v in choice_values
                                    ):
                                        mode.metadata["variable_content"] = {
                                            k: str(v).strip()
                                            for k, v in candidate.items()
                                        }
                                        mode.metadata["variable_name"] = alias
                                        break
                            except Exception:
                                continue
                except Exception:
                    pass  # Non-critical — widget works without enrichment

            tool_configs.append({
                "tool_type": tool.tool_type,
                "prompt": tool.prompt,
                "input_mode": mode.to_dict(),
                "output_var": tool.output_vars[0] if tool.output_vars else tool.tool_type,
                "expected_input_type": tool.expected_input_type,
                "prefix": tool.prefix,
            })

        compound_mode = InputModeConfig(
            mode=InputMode.FREE_TEXT,
            prompt=assistant_text,
            metadata={
                "compound": True,
                "tools": tool_configs,
            },
        )
        # See _handle_conversation_tool above for rationale.
        _prompt_data = {
            "template_source": getattr(self, "_last_template_source", "") or "",
            "template_feed": getattr(self, "_last_template_feed", {}) or {},
            "rendered_prompt": getattr(self, "_last_rendered_prompt", "") or "",
            "template_config": getattr(self, "_last_template_config", {}) or {},
        }
        await active_interactive.asend_response(
            assistant_text,
            flag=InteractionFlags.PendingInput,
            input_mode=compound_mode,
            prompt_data=_prompt_data,
        )

        # Wait for ONE response with all collected values
        user_input = await active_interactive.aget_input()
        if user_input is None:
            return None

        # Extract values from compound response
        collected: dict[str, str] = {}
        if isinstance(user_input, dict):
            values = user_input.get("values", user_input.get("user_input", user_input))
            # Unwrap nested "values" dict from compound widget response
            # Frontend sends {user_input: {values: {...}}} which arrives as
            # {user_input: {values: {...}}, session_id: ...}
            if (
                isinstance(values, dict)
                and "values" in values
                and isinstance(values["values"], dict)
            ):
                values = values["values"]
            if isinstance(values, dict):
                # Decode each child payload (read by the tool's primary output
                # key) into distinct bindings — a composite choice yields BOTH
                # its mode var and its nested input var; multi-value publishes
                # via the declared serialization (never str(list)).
                bindings = decode_compound_bindings(
                    tools, values, session_root=self._session_root()
                )
                # Rich-choice editable content override wins, if present.
                variable_override = values.get("variable_override")
                if isinstance(variable_override, dict):
                    bindings.update(variable_override)
                collected.update(bindings)
                if bindings:
                    self.set_session_variables(bindings)
            else:
                # Fallback: single value
                collected["input"] = str(values)
        else:
            collected["input"] = str(user_input)

        return collected

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
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ):
        """Delegate streaming to base inferencer."""
        if hasattr(self.base_inferencer, "ainfer_streaming"):
            async for chunk in self.base_inferencer.ainfer_streaming(
                inference_input, inference_config, **kwargs
            ):
                yield chunk
        else:
            result = await self.base_inferencer.ainfer(
                inference_input, inference_config, **kwargs
            )
            yield str(result) if not isinstance(result, str) else result


def _choice_option_from(c: ChoiceItem) -> ChoiceOption:
    """Build a UI ChoiceOption from a ChoiceItem, preserving the description and
    any embedded typed ``input`` spec (serialised) so composite choices and rich
    descriptions survive into ``InputModeConfig.to_dict()``."""
    return ChoiceOption(
        label=c.label,
        value=c.value,
        description=getattr(c, "description", "") or "",
        input=c.input.to_dict() if getattr(c, "has_input", False) and c.input is not None else None,
    )


def _build_input_mode(tool: ConversationTool) -> InputModeConfig:
    """Build an InputModeConfig from a ConversationTool."""
    logger.info("[_build_input_mode] input: tool_type=%s metadata=%s prompt=%.60s",
                tool.tool_type, tool.metadata, tool.prompt)
    if tool.tool_type == ConversationToolType.SINGLE_CHOICE:
        options = [_choice_option_from(c) for c in tool.choices]
        return single_choice(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
        )

    if tool.tool_type == ConversationToolType.MULTIPLE_CHOICE:
        options = [_choice_option_from(c) for c in tool.choices]
        return multiple_choices(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
            show_select_all=tool.show_select_all,
            select_all_text=tool.select_all_text,
        )

    if tool.tool_type == ConversationToolType.PROPOSAL_SELECTION:
        # Selection flows through the multiple-choice machinery (one option per
        # proposal id), while the full proposal payload rides in metadata for
        # the rich React ProposalSelectionWidget. Hosts without that widget
        # registered degrade to a plain multi-select over the same options.
        options = [
            ChoiceOption(
                label=c.label,
                value=c.value,
                description=getattr(c, "description", "") or "",
            )
            for c in tool.choices
        ]
        ps_metadata: dict[str, Any] = {"widget_type": "proposal_selection"}
        if tool.metadata:
            ps_metadata.update(tool.metadata)
        return InputModeConfig(
            mode=InputMode.MULTIPLE_CHOICE,
            prompt=tool.prompt,
            options=options,
            allow_custom=False,
            metadata=ps_metadata,
            show_select_all=tool.show_select_all,
            select_all_text=tool.select_all_text or "All proposals",
        )

    if tool.tool_type == ConversationToolType.CONFIRMATION:
        metadata: dict[str, Any] = {
            "widget_type": "confirmation",
            "note_variable": "additional_instructions",
        }
        # Pass through any metadata from the tool (e.g., view path)
        if tool.metadata:
            metadata.update(tool.metadata)
        # Include action tool parameters for config UI
        tool_params = getattr(tool, "_tool_params", None)
        if tool_params:
            metadata["tool_params"] = tool_params
        return InputModeConfig(
            mode=InputMode.FREE_TEXT,
            prompt=tool.prompt,
            metadata=metadata,
        )

    # CLARIFICATION and fallback: free text (optionally a typed path input).
    config = InputModeConfig(
        mode=InputMode.FREE_TEXT,
        prompt=tool.prompt,
        expected_input_type=tool.expected_input_type,
        prefix=tool.prefix,
        allow_multiple_input=tool.allow_multiple_input,
    )
    # Route path inputs to the dedicated widget; mirror typed fields into legacy
    # metadata for one migration window (the new UI reads first-class fields).
    if tool.expected_input_type and tool.expected_input_type != "free_text":
        metadata: dict[str, Any] = {
            "expected_input_type": tool.expected_input_type,
            "prefix": tool.prefix,
        }
        if tool.allow_multiple_input:
            metadata["allow_multiple_input"] = True
        if tool.expected_input_type == "path":
            metadata["widget_type"] = "path_input"
        config.metadata = metadata
    logger.info("[_build_input_mode] output (fallback): mode=%s metadata=%s",
                config.mode, config.metadata)
    return config

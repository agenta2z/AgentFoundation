

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
    parse_conversation_response,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
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
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import SOPManager

logger = logging.getLogger(__name__)

# Maximum conversation loop iterations for standalone run_conversation()
_MAX_CONVERSATION_ITERATIONS = 20

# Protocol-level message markers used in the agentic loop conversation history.
# These strings are part of the LLM-facing protocol — changing them may affect
# prompt comprehension. Keep them short and bracketed for easy parsing.
_WIDGET_RESPONSE_PREFIX = "[Collected from conversation widget]"
_TOOL_RESULT_HEADER = "[Tool Result: {}]"  # .format(tool_name)
_TOOL_RESULTS_PREFIX = "[Tool execution results]"
_CONTINUE_AFTER_TOOLS = "Continue based on the tool execution results above."


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
    max_iterations: int = attrib(default=5, kw_only=True)
    max_tool_result_chars: int = attrib(default=4000, kw_only=True)

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
        self.sop_state = None  # SOPState | None — set by /sop executor

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
                return AgenticResult(
                    text=response,
                    completed_actions=[],
                    iterations_used=0,
                )

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

        # SOP needs more iterations than the default 5
        effective_max = self.max_iterations
        if self.sop_state and self.sop_state.sop:
            n_phases = len(self.sop_state.sop.phases)
            effective_max = max(effective_max, n_phases * 3)

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
            if False and on_prompt_rendered:  # DISABLED: debugging hang
                try:
                    await on_prompt_rendered(self, raw_response)
                except Exception:
                    pass

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
                    # Set confirmation gate for phase completion detection
                    for tool in conv_response.conversation_tools:
                        if getattr(tool, "tool_type", "") == "confirmation":
                            if self.sop_state:
                                self.sop_state.confirmation_gate_passed = True
                            break
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
                if on_new_turn:
                    new_turn = await on_new_turn(turn_number, user_input)
                    if new_turn is not None:
                        turn_number = new_turn

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
                # Fire on_turn_complete after all messages for this turn are committed
                if on_turn_complete:
                    try:
                        await on_turn_complete(iteration + 1)
                    except Exception as _tc_err:
                        logger.warning("[agentic_loop] on_turn_complete error: %s", _tc_err)
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
                # Fire on_turn_complete after all messages for this turn are committed
                if on_turn_complete:
                    try:
                        await on_turn_complete(iteration + 1)
                    except Exception as _tc_err:
                        logger.warning("[agentic_loop] on_turn_complete error: %s", _tc_err)
                continue

            # 5b. Parse for action tool calls (legacy XML format)
            parsed = parse_llm_response(raw_response, self._valid_tool_names)
            if not parsed.has_tool_calls:
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
            # Fire on_turn_complete after all messages for this turn are committed
            if on_turn_complete:
                try:
                    await on_turn_complete(iteration + 1)
                except Exception as _tc_err:
                    logger.warning("[agentic_loop] on_turn_complete error: %s", _tc_err)

        # Exhausted max iterations — return last raw response
        return AgenticResult(
            text=last_raw_response,
            raw_response=last_raw_response,
            completed_actions=loop_actions,
            iterations_used=self.max_iterations,
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
            if self.sop_state.sop_name:
                from agent_foundation.resources.sops.registry import load_sop
                sop_info = load_sop(self.sop_state.sop_name)
                self.sop_state.sop = sop_info.sop
        else:
            self.sop_state = None

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
        if not self.sop_state:
            return f"No active SOP. Messages: {len(self._messages)}. Paused: {self._paused}."
        s = self.sop_state
        completed = [
            c.phase if hasattr(c, "phase") else str(c) for c in s.completed_phases
        ]
        return (
            f"SOP: {s.sop_name}\n"
            f"Phase: {s.current_phase} ({s.phase_status})\n"
            f"Completed: {completed}\n"
            f"Messages: {len(self._messages)}. Paused: {self._paused}."
        )

    @command("clear", description="Clear conversation history")
    async def _cmd_clear(self) -> str:
        self._messages = []
        return "Conversation history cleared."

    @command("pause", description="Pause SOP execution", requires_active_sop=True)
    async def _cmd_pause(self) -> str:
        self._paused = True
        return "SOP paused. Use /resume to continue."

    @command("resume", description="Resume paused SOP")
    async def _cmd_resume(self) -> str:
        if not self._paused and not getattr(self, "_pending_resume_state", None):
            return "Nothing to resume — session is not paused."
        self._paused = False
        return "SOP resumed."

    @command("exit_sop", description="Exit the active SOP", aliases=("exit",),
             requires_active_sop=True)
    async def _cmd_exit_sop(self) -> str:
        sop_name = self.sop_state.sop_name if self.sop_state else "unknown"
        self.sop_state = None
        return f"Exited SOP: {sop_name}."

    # =========================================================================
    # Phase Completion Detection (Model A)
    # =========================================================================

    def _check_phase_completion(self, tool_name: str = "") -> None:
        """Detect SOP phase completion and advance to next phase.

        Called after _execute_tool_call() applies context_updates.
        Three detection strategies:
          1. Tool-mapped: tool from tool_phase_map completed
          2. Confirmation: confirmation_gate_passed + requires_confirmation
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
            detected = True

        if not detected and s.confirmation_gate_passed:
            if "requires confirmation" in " ".join(getattr(phase, "directives", [])):
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

        s.confirmation_gate_passed = False
        logger.info("SOP phase %s completed; next=%s", current, s.current_phase)

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

                if s.confirmation_gate_passed:
                    from rich_python_utils.string_utils.formatting.template_manager.sop_manager import SOPPhase
                    for node in tracker.get_available_next():
                        if not isinstance(node, SOPPhase):
                            continue
                        has_tools = any(
                            sub.name.lower() in ("tools", "command")
                            for sub in getattr(node, "subsections", [])
                        )
                        if not has_tools and "requires confirmation" in " ".join(
                            getattr(node, "directives", [])
                        ):
                            tracker.completed_states.add(node.id)
                            s.confirmation_gate_passed = False
                            break

                nextstep_guidance = SOPManager.render_guidance(
                    tracker, sop, context=dict(self.prior_context),
                )
            except Exception as e:
                logger.warning("SOP evaluation failed: %s", e)

        # Build feed using build_feed — merges dicts + FeedBase objects
        from rich_python_utils.common_objects.feed_base import build_feed

        feed = build_feed(
            template_vars,
            self.prior_context,
            self.sop_state,
            {
                "workflow_nextstep_guidance": nextstep_guidance,
                "action_tools": available_tools,
                "completed_actions": all_actions,
                "conversation_history": messages,
                "current_turn": {"role": "user", "content": current_message},
                "conversation_tools": conversation_tools_text,
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
        return self.prompt_renderer.render(feed)

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
                    self._check_phase_completion(tool_name=canonical)
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
        """Resolve a tool name or alias to the canonical tool name."""
        if name in self.tool_registry:
            return name
        for tool in self.tool_registry.values():
            if (
                name in getattr(tool, "aliases", [])
                or name.replace("-", "_") == tool.name
            ):
                return tool.name
        normalized = name.replace("-", "_")
        if normalized in self.tool_registry:
            return normalized
        return name

    @property
    def _valid_tool_names(self) -> set[str]:
        """Set of valid tool names including aliases."""
        names: set[str] = set()
        for tool in self.tool_registry.values():
            names.add(tool.name)
            for alias in getattr(tool, "aliases", []):
                names.add(alias)
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

        if len(tools) == 1:
            return self._synthesize_single_yolo(tools[0])

        collected: dict[str, str] = {}
        for tool in tools:
            var_name = getattr(tool, "output_variable", None) or tool.tool_type
            collected[var_name] = self._synthesize_single_yolo(tool)
        return collected

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
            if variable_override and self.prompt_renderer:
                vm = self.prompt_renderer.variable_manager
                for vname, edited_content in variable_override.items():
                    vm.set(vname, edited_content)
            elif tool.output_vars and self.prompt_renderer:
                # No override — apply choice value (triggers sub-key resolution)
                vm = self.prompt_renderer.variable_manager
                vm.set(tool.output_vars[0], choice_value)

            return choice_value

        return str(response)

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
                target_path = self.prior_context.get("workflow_target_path", "")
                if target_path:
                    from pathlib import Path as _Path

                    target_dir = _Path(target_path)
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
                self.update_prior_context(_confirmation_gate_passed=True)
                if self.sop_state:
                    self.sop_state.confirmation_gate_passed = True
            var_name = tools[0].output_vars[0] if tools[0].output_vars else "input"
            return {var_name: result}

        # Multiple tools: send ALL as a compound widget in one pending_input
        tool_configs = []
        for tool in tools:
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
                # Extract variable_override if present
                variable_override = values.get("variable_override")
                for tool in tools:
                    var = tool.output_vars[0] if tool.output_vars else tool.tool_type
                    raw_value = values.get(var, "")
                    collected[var] = str(raw_value)

                    # Apply variable override or choice value to template system
                    if (
                        variable_override
                        and isinstance(variable_override, dict)
                        and var in variable_override
                        and self.prompt_renderer
                    ):
                        vm = self.prompt_renderer.variable_manager
                        vm.set(var, variable_override[var])
                    elif tool.output_vars and self.prompt_renderer and raw_value:
                        vm = self.prompt_renderer.variable_manager
                        vm.set(tool.output_vars[0], str(raw_value))
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


def _build_input_mode(tool: ConversationTool) -> InputModeConfig:
    """Build an InputModeConfig from a ConversationTool."""
    logger.info("[_build_input_mode] input: tool_type=%s metadata=%s prompt=%.60s",
                tool.tool_type, tool.metadata, tool.prompt)
    if tool.tool_type == ConversationToolType.SINGLE_CHOICE:
        options = [ChoiceOption(label=c.label, value=c.value) for c in tool.choices]
        return single_choice(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
        )

    if tool.tool_type == ConversationToolType.MULTIPLE_CHOICE:
        options = [ChoiceOption(label=c.label, value=c.value) for c in tool.choices]
        return multiple_choices(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
            show_select_all=tool.show_select_all,
            select_all_text=tool.select_all_text,
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

    # CLARIFICATION and fallback: free text
    config = InputModeConfig(
        mode=InputMode.FREE_TEXT,
        prompt=tool.prompt,
    )
    # Pass expected_input_type and prefix to frontend for path autocomplete
    if tool.expected_input_type and tool.expected_input_type != "free_text":
        config.metadata = {
            "expected_input_type": tool.expected_input_type,
            "prefix": tool.prefix,
        }
    logger.info("[_build_input_mode] output (fallback): mode=%s metadata=%s",
                config.mode, config.metadata)
    return config

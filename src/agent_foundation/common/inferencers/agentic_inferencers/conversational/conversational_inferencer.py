

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

        for iteration in range(self.max_iterations):
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

            # Flush prompt + response artifacts to disk so "View Prompt"
            # works even while waiting for user input (confirmation, etc.).
            if False and on_prompt_rendered:  # DISABLED: debugging hang
                try:
                    await on_prompt_rendered(self, raw_response)
                except Exception:
                    pass

            # Add assistant response to conversation history so subsequent
            # turns include it in the rendered prompt.
            self.add_message("assistant", raw_response)

            # 4. Check for conversation tools
            conv_response = parse_conversation_response(raw_response)

            if conv_response.has_conversation_tool and effective_interactive:
                collected = await self._handle_conversation_tools(
                    conv_response.conversation_tools,
                    conv_response.text,
                    interactive_override=effective_interactive,
                    action_tools=conv_response.action_tools,
                )
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

    # =========================================================================
    # Prompt Rendering
    # =========================================================================

    def _render_prompt(self, current_message: str) -> str:
        """Build template variables and render via prompt_renderer."""
        if not self.prompt_renderer:
            return self._render_fallback_prompt(current_message)

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
        sop_path = getattr(self.prompt_renderer, "find_sop_file", lambda: None)()
        if sop_path is not None:
            try:
                from rich_python_utils.common_objects.workflow.stategraph import (
                    StateGraphTracker,
                )

                sop = SOPManager.load(sop_path)
                # Store SOP for confirmation gate checks in _execute_tool_call
                self.prior_context["_sop"] = sop

                # Extract and store tool-to-phase mapping from SOP
                if hasattr(sop, 'tool_to_phase_map'):
                    tool_map = sop.tool_to_phase_map
                    if tool_map:
                        self.prior_context["tool_phase_map"] = tool_map

                # Validate SOP phase IDs match workflow_description (single source of truth)
                workflow_desc_phases = self.prior_context.get("workflow_description", "")
                if workflow_desc_phases:
                    from agent_foundation.server.workflow_context import _WORKFLOW_DESC_PHASE_RE
                    desc_phase_ids = {
                        m.group(1) for m in _WORKFLOW_DESC_PHASE_RE.finditer(workflow_desc_phases)
                    }
                    sop_phase_ids = set(sop.phase_ids) if hasattr(sop, 'phase_ids') else set()
                    if desc_phase_ids and sop_phase_ids and desc_phase_ids != sop_phase_ids:
                        logger.warning(
                            "SOP phase IDs %s do not match workflow_description phase IDs %s. "
                            "workflow_description is the single source of truth for phase definitions.",
                            sop_phase_ids, desc_phase_ids,
                        )

                # Build tracker from prior_context state
                completed = [
                    r.phase if hasattr(r, "phase") else str(r)
                    for r in self.prior_context.get("completed_phases", [])
                ]
                tracker = StateGraphTracker(
                    graph=sop,
                    current_state=self.prior_context.get("current_phase"),
                    state_status=self.prior_context.get("phase_status", "idle"),
                    completed_states=completed,
                    state_outputs=self.prior_context.get("phase_outputs", {}),
                    goto_counts=self.prior_context.get("goto_counts", {}),
                )

                # Auto-complete confirmation-gate phases (no tools, no outputs)
                # after user confirmed via a confirmation widget
                if self.prior_context.pop("_confirmation_gate_passed", False):
                    from rich_python_utils.string_utils.formatting.template_manager.sop_manager import SOPPhase
                    for node in tracker.get_available_next():
                        if not isinstance(node, SOPPhase):
                            continue
                        has_tools = any(
                            s.name.lower() in ("tools", "command")
                            for s in getattr(node, "subsections", [])
                        )
                        if not has_tools and "requires confirmation" in " ".join(
                            getattr(node, "directives", [])
                        ):
                            tracker.completed_states.add(node.id)
                            self.prior_context.setdefault(
                                "_completed_gate_phases", []
                            ).append(node.id)
                            break

                nextstep_guidance = SOPManager.render_guidance(
                    tracker, sop, context=dict(self.prior_context),
                )
            except Exception as e:
                logger.warning("SOP evaluation failed: %s", e)

        feed = {
            **template_vars,
            "workflow_nextstep_guidance": nextstep_guidance,
            "action_tools": available_tools,
            **self.prior_context,
            "completed_actions": all_actions,
            "conversation_history": messages,
            "current_turn": {"role": "user", "content": current_message},
            "conversation_tools": conversation_tools_text,
        }

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

    def _render_fallback_prompt(self, current_message: str) -> str:
        """Fallback prompt when prompt_renderer is None."""
        formatter = ToolMarkdownFormatter()
        tools_list = [t for t in self.tool_registry.values() if getattr(t, 'agent_enabled', True)]
        available_tools = formatter.format_all(tools_list)
        parts = [
            "You are an AI assistant with tool-use capabilities.",
            "",
            "## Available Tools",
            available_tools,
            "",
            'To invoke a tool: <tool_call>{"name": "...", "arguments": {...}}</tool_call>',
            "",
        ]
        if self._messages:
            parts.append("## Conversation")
            for msg in self._messages:
                if (
                    msg == self._messages[-1]
                    and msg.get("role") == "user"
                    and msg.get("content") == current_message
                ):
                    continue
                parts.append(f"<{msg['role']}>{msg['content']}</{msg['role']}>")
        parts.append(f"\n<user>{current_message}</user>")
        return "\n".join(parts)

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

            # Update prior_context immediately so the next iteration's prompt
            # sees the correct SOP phase as "running", preventing the LLM from
            # retrying with stale "error" status.
            # Use SOP-derived tool-to-phase mapping (populated from SOP at render time)
            tool_map = self.prior_context.get("tool_phase_map", {})
            sop_phase = tool_map.get(canonical, canonical)
            self.prior_context["current_phase"] = sop_phase
            self.prior_context["phase_status"] = "running"
            # Build workflow_status with phase name from workflow_description
            phase_name = sop_phase
            try:
                from agent_foundation.server.workflow_context import _WORKFLOW_DESC_PHASE_RE
                wd = self.prior_context.get("workflow_description", "")
                if wd:
                    for m in _WORKFLOW_DESC_PHASE_RE.finditer(wd):
                        if m.group(1) == sop_phase:
                            phase_name = m.group(2).strip()
                            break
            except Exception:
                pass
            self.prior_context["workflow_status"] = (
                f"Current phase: Phase {sop_phase} — {phase_name} (running)\n"
                f"  Active task: {canonical} — {str(tool_call.arguments.get('target', ''))[:80]}"
            )

            async def _run_async() -> None:
                try:
                    result = await executor(canonical, tool_call.arguments)
                    if hasattr(result, "context_updates") and result.context_updates:
                        self.update_prior_context(**result.context_updates)
                except Exception as e:
                    logger.error("Async tool %s failed: %s", canonical, e)

            # Save strong reference to prevent GC of the background task.
            # asyncio._all_tasks is a WeakSet, so without a strong reference
            # the task could theoretically be collected during long I/O waits.
            self._active_async_task = asyncio.create_task(_run_async())
            return (
                f"Tool '{canonical}' launched asynchronously. "
                f"Check the task panel for progress and results."
            )

        try:
            result = await self.tool_executor(canonical, tool_call.arguments)
            # result is ToolExecutionResult — apply context_updates to prior_context
            if hasattr(result, "context_updates") and result.context_updates:
                self.update_prior_context(**result.context_updates)
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

        await active_interactive.asend_response(
            assistant_text,
            flag=InteractionFlags.PendingInput,
            input_mode=input_mode,
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
        await active_interactive.asend_response(
            assistant_text,
            flag=InteractionFlags.PendingInput,
            input_mode=compound_mode,
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
    if tool.tool_type == ConversationToolType.SINGLE_CHOICE:
        options = [ChoiceOption(label=c.label, value=c.value, description=getattr(c, 'description', '')) for c in tool.choices]
        return single_choice(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
        )

    if tool.tool_type == ConversationToolType.MULTIPLE_CHOICE:
        options = [ChoiceOption(label=c.label, value=c.value, description=getattr(c, 'description', '')) for c in tool.choices]
        return multiple_choices(
            options,
            allow_custom=tool.allow_custom,
            prompt=tool.prompt,
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
    return config

"""ConversationToolsMixin — the conversation-tool (widget) methods of the
conversational orchestrators, as delegators to ``widget_core``.

The orchestrators' turn loops call ``widget_core`` directly; these methods keep
the orchestrators' widget method names (and the loop-frame names of the widget
mailboxes) for their callers. They are delegators, so calling one unbound on an
object that provides the methods it uses keeps working.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational import (
    widget_core,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    HandlerContext,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.widget_core import (  # noqa: F401 — module-level helpers stay importable from here
    _build_input_mode,
    _choice_option_from,
    _record_hitl_checkpoint,
)
from agent_foundation.ui.interactive_base import InteractiveBase

logger: logging.Logger = logging.getLogger(__name__)


class ConversationToolsMixin:
    """Present, persist, decode and auto-answer conversation tools."""

    # The orchestrator's ``mailboxes`` under their loop-frame names.

    @property
    def _next_action_tool_overrides(self) -> Optional[dict[str, Any]]:
        return self.mailboxes.action_overrides

    @_next_action_tool_overrides.setter
    def _next_action_tool_overrides(self, value: Optional[dict[str, Any]]) -> None:
        self.mailboxes.action_overrides = value

    @property
    def _next_turn_variables(self) -> Optional[dict[str, str]]:
        return self.mailboxes.turn_variables

    @_next_turn_variables.setter
    def _next_turn_variables(self, value: Optional[dict[str, str]]) -> None:
        self.mailboxes.turn_variables = value

    @property
    def _next_dashboard_directives(self) -> Optional[dict[str, Any]]:
        return self.mailboxes.dashboard_directives

    @_next_dashboard_directives.setter
    def _next_dashboard_directives(self, value: Optional[dict[str, Any]]) -> None:
        self.mailboxes.dashboard_directives = value

    async def _synthesize_yolo_collected(
        self,
        tools: list,
    ) -> dict[str, str] | str | None:
        return await widget_core.synthesize_yolo(self, tools)

    def _synthesize_yolo_response(self, tool):
        return widget_core.yolo_response(self, tool)

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
        return widget_core.yolo_spec(self, tool)

    def _resolve_proposals_source(self, tool: ConversationTool) -> "Optional[dict]":
        return widget_core.resolve_proposals_source(self, tool)

    def _enrich_proposal_selection(self, tool: ConversationTool) -> None:
        widget_core.enrich_proposal_selection(self, tool)

    # Phase I: `_normalize_dashboard_directives`, `_maybe_open_dashboard`, and
    # `_build_dashboard_seed` moved to DashboardCoordinator.

    def _build_handler_context(
        self,
        *,
        action_tools: Optional[list[dict[str, Any]]] = None,
        active_interactive: Optional[InteractiveBase] = None,
    ) -> HandlerContext:
        return widget_core.handler_context(
            self, action_tools=action_tools, interactive=active_interactive
        )

    def _persist_pending_widget(
        self,
        interactive: Any,
        tools: list,
        action_tools: Optional[list],
        turn_number: Optional[int],
        iteration: Optional[int],
    ) -> None:
        widget_core.persist_pending(
            self,
            interactive,
            tools,
            action_tools,
            turn_number=turn_number,
            iteration=iteration,
        )

    async def _handle_conversation_tool(
        self,
        tool: ConversationTool,
        assistant_text: str,
        interactive_override: Optional[InteractiveBase] = None,
        *,
        turn_number: Optional[int] = None,
        iteration: Optional[int] = None,
        action_tools: Optional[list] = None,
    ) -> Optional[str]:
        """Show one conversation tool (without handler enrichment) and return
        its decoded answer text (see ``_apply_widget_answer``)."""
        active_interactive = interactive_override or self.interactive
        if active_interactive is None:
            return None
        user_input = await widget_core.show(
            self,
            [tool],
            assistant_text,
            interactive=active_interactive,
            then_run=action_tools,
            turn_number=turn_number,
            iteration=iteration,
        )
        return await self._apply_widget_answer(tool, user_input)

    async def _apply_widget_answer(
        self, tool: ConversationTool, user_input: Any
    ) -> Optional[str]:
        """``widget_core.decode_answer`` under ``_build_handler_context()``:
        returns the decoded text and leaves the composite bindings in
        ``_last_handler_bindings`` (``None`` when there are none)."""
        decoded = await widget_core.decode_answer(
            self, tool, user_input, context=self._build_handler_context()
        )
        self._last_handler_bindings = (
            dict(decoded.bindings) if decoded is not None and decoded.bindings else None
        )
        return decoded.text if decoded is not None else None

    @staticmethod
    def _is_affirmative_response(value: Any) -> bool:
        """Whether a confirmation reply means "go ahead" (vs. a decline)."""
        return str(value).strip().lower() in ("yes", "proceed")

    def _record_answered_required_conv_tools(self, tools: list) -> None:
        # Delegated to SOPController (Phase K).
        self.sop_controller.record_answered_required_conv_tools(tools)

    def _open_user_input_gate_if_satisfied(
        self, tools: list, collected: Optional[dict]
    ) -> None:
        # Delegated to SOPController (Phase K).
        self.sop_controller.open_user_input_gate_if_satisfied(tools, collected)

    async def _handle_conversation_tools(
        self,
        tools: list[ConversationTool],
        assistant_text: str,
        interactive_override: Optional[InteractiveBase] = None,
        action_tools: Optional[list[dict]] = None,
        *,
        turn_number: Optional[int] = None,
        iteration: Optional[int] = None,
    ) -> Optional[dict[str, str]]:
        return await widget_core.present_and_collect(
            self,
            tools,
            assistant_text,
            interactive=interactive_override or self.interactive,
            then_run=action_tools,
            turn_number=turn_number,
            iteration=iteration,
        )

    def _decode_compound_response(
        self, tools: list[ConversationTool], user_input: Any
    ) -> Optional[dict[str, str]]:
        return widget_core.decode_compound(self, tools, user_input)

    async def _collect_widget_response(
        self,
        tools: list[ConversationTool],
        action_tools: Optional[list[dict]],
        user_input: Any,
    ) -> Optional[dict[str, str]]:
        """``widget_core.decode`` composed from this object's own decode
        methods: one tool → ``_apply_widget_answer`` (+ its
        ``_last_handler_bindings``) and ``_open_user_input_gate_if_satisfied``;
        several → ``_decode_compound_response``."""
        if not tools:
            return None
        if len(tools) == 1:
            result = await self._apply_widget_answer(tools[0], user_input)
            if result is None:
                return None
            var_name = tools[0].output_vars[0] if tools[0].output_vars else "input"
            collected: dict[str, str] = {var_name: result}
            _bindings = getattr(self, "_last_handler_bindings", None) or {}
            self._last_handler_bindings = None
            if _bindings:
                collected.update(_bindings)
            self._open_user_input_gate_if_satisfied(tools, collected)
            return collected
        return self._decode_compound_response(tools, user_input)

"""ConversationStateMixin — host-facing conversation state shared by the
conversational orchestrators (``ConversationalInferencer`` and the native
sibling in ``conversational_native``).

Stateless: every method reads/writes attributes the host class declares
(see ``host_protocol.ConversationHostState``). Method bodies were moved
verbatim from ``ConversationalInferencer``; the only seam is
``_workspace_root()``, which each orchestrator answers for its own backend.
The SOP-feed methods keep their names as delegators to ``sop_feed``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticDynamicContext,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_feed import (
    build_sop_feed,
    filtered_sops,
    prepare_sop_for_turn,
)

logger: logging.Logger = logging.getLogger(__name__)


class ConversationStateMixin:
    """Prior-context, transcript-mirror, SOP-forwarding and phase API."""

    def _workspace_root(self) -> str:
        """Working directory of the backend agent (fallback session root)."""
        return ""

    # =========================================================================
    # Phase K SOP-field forwarders
    # =========================================================================
    # These @property/setters forward the 5 SOP-owned fields through
    # `self.sop_controller`. Existing methods keep their bodies unchanged
    # (accessing `self.sop_state`, `self._paused`, etc. transparently).
    # New code should prefer the direct controller API (e.g.
    # `self.sop_controller.is_paused`) to avoid the forwarding layer.
    #
    # `_paused` in particular is the K3b shim that preserves AC-PR1's external
    # write contract: `ci._paused = True` from a test still works.

    @property
    def sop_state(self):
        return self.sop_controller.sop_state if self.sop_controller else None

    @sop_state.setter
    def sop_state(self, value) -> None:
        if self.sop_controller is not None:
            self.sop_controller.sop_state = value

    @property
    def _suspended_sops(self) -> list:
        return self.sop_controller._suspended_sops if self.sop_controller else []

    @_suspended_sops.setter
    def _suspended_sops(self, value: list) -> None:
        if self.sop_controller is not None:
            self.sop_controller._suspended_sops = value

    @property
    def _paused(self) -> bool:
        return self.sop_controller.is_paused if self.sop_controller else False

    @_paused.setter
    def _paused(self, value: bool) -> None:
        # K3b shim: preserves the AC-PR1 external-write contract
        # (`_docs/_plan/workflows_and_sop/sop_model_a_..._plan.md:543`).
        # Silent no-op if sop_controller not yet initialized — that only
        # happens transiently during __attrs_post_init__ before K's
        # auto-construct runs.
        if self.sop_controller is not None:
            self.sop_controller.is_paused = value

    @property
    def _pending_followup(self):
        return self.sop_controller._pending_followup if self.sop_controller else None

    @_pending_followup.setter
    def _pending_followup(self, value) -> None:
        if self.sop_controller is not None:
            self.sop_controller._pending_followup = value

    @property
    def _auto_shutdown_on_sop_complete(self) -> bool:
        return (
            self.sop_controller._auto_shutdown_on_sop_complete
            if self.sop_controller
            else False
        )

    @_auto_shutdown_on_sop_complete.setter
    def _auto_shutdown_on_sop_complete(self, value: bool) -> None:
        if self.sop_controller is not None:
            self.sop_controller._auto_shutdown_on_sop_complete = value

    # --- Public host-protocol names (ConversationalHost) ---------------------

    @property
    def suspended_sops(self) -> list:
        """Paused/exited SOPs, most recent first (a copy)."""
        return list(self._suspended_sops)

    @suspended_sops.setter
    def suspended_sops(self, states: Any) -> None:
        self._suspended_sops = list(states or [])

    @property
    def extra_sop_dirs(self) -> list:
        """Extra SOP discovery directories (a copy); the SOP controller owns them."""
        return list(self.sop_controller.extra_sop_dirs)

    @extra_sop_dirs.setter
    def extra_sop_dirs(self, dirs: Any) -> None:
        self.sop_controller.extra_sop_dirs = list(dirs or [])

    @property
    def allowed_sops(self) -> list:
        """SOP catalog allow-list (empty: no restriction): the SOP
        controller's own list, so an in-place change applies too."""
        return self.sop_controller.allowed_sops

    @allowed_sops.setter
    def allowed_sops(self, names: Any) -> None:
        self.sop_controller.allowed_sops = list(names or [])

    @property
    def disallowed_sops(self) -> list:
        """SOP catalog deny-list, applied after ``allowed_sops``: the SOP
        controller's own list, so an in-place change applies too."""
        return self.sop_controller.disallowed_sops

    @disallowed_sops.setter
    def disallowed_sops(self, names: Any) -> None:
        self.sop_controller.disallowed_sops = list(names or [])

    @property
    def tool_dispatcher(self) -> Any:
        """The host's tool dispatcher (falls back to the tool executor)."""
        return getattr(self, "_tool_dispatcher", None) or self.tool_executor

    @tool_dispatcher.setter
    def tool_dispatcher(self, dispatcher: Any) -> None:
        """Dashboards and the experiment hub open through this dispatcher."""
        self._tool_dispatcher = dispatcher
        if self.dashboard_coordinator is not None:
            self.dashboard_coordinator.tool_dispatcher = self.tool_dispatcher

    def accepts_command(self, text: str) -> bool:
        """Whether ``text`` is a slash command this orchestrator handles
        itself; hosts send such input to ``run_agentic_loop`` rather than to
        their own command handling."""
        return bool(text) and self._commands.is_command(text)

    def set_prior_context(self, ctx: dict[str, Any]) -> None:
        self.prior_context = dict(ctx)

    def update_prior_context(self, **kwargs: Any) -> None:
        if "sop_state" in kwargs:
            self.sop_state = kwargs.pop("sop_state")
            if self.sop_state and self.sop_state.yolo_mode:
                self.yolo_mode = True
        self.prior_context.update(kwargs)

    def set_session_variables(
        self,
        variables: dict[str, Any],
        *,
        tool_type: str | None = None,
    ) -> None:
        """Store variables in both variable_manager and prior_context.

        Used by conversation tool handlers to persist user-provided values
        (e.g., workflow_target_path, strategy) so they're available in both
        template rendering (variable_manager) and SOP guidance (prior_context).

        A1.b (v3): if ``tool_type`` is provided (str form; callers holding an
        enum pass ``.value``), ALSO publish tool-namespaced aliases so SOPs
        can reference ``{{ <tool_type>__<var> }}`` alongside the bare form,
        matching the ``<producing_tool>__<output>`` convention adopted for
        action tools in tool_dispatcher.py:687-707.
        """
        vm = None
        if self.prompt_renderer:
            vm = getattr(self.prompt_renderer, "variable_manager", None)

        def _publish(name: str, value: Any) -> None:
            self.prior_context[name] = value
            if vm is not None and hasattr(vm, "set"):
                vm.set(name, value)

        # Accept enum (ConversationToolType) or str; fall back to str().
        safe_tool_type = (
            str(getattr(tool_type, "value", tool_type)).replace("-", "_")
            if tool_type
            else None
        )
        for name, value in variables.items():
            _publish(name, value)
            if safe_tool_type and isinstance(name, str) and "__" not in name:
                _publish(f"{safe_tool_type}__{name}", value)
                # One-off SOP spelling alias (plural) matching the current
                # model_optimization SOP wording. Extend into a helper if a
                # second widget ever needs its own singular/plural bridge.
                if (
                    safe_tool_type == "proposal_selection"
                    and name == "selected_proposal_ids"
                ):
                    _publish("proposal_selection__selected_proposals_ids", value)

    def _session_root(self) -> str:
        """Best-effort session root for path re-join (used by the finalizer)."""
        root = (
            self.prior_context.get("session_root_path", "")
            if self.prior_context
            else ""
        )
        return root or self._workspace_root() or ""

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
        # Delegated to SOPController (Phase K).
        return self.sop_controller.consume_pending_followup()

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    @property
    def dynamic_context(self) -> AgenticDynamicContext:
        return self._dynamic_context

    @dynamic_context.setter
    def dynamic_context(self, context: AgenticDynamicContext) -> None:
        self._dynamic_context = context

    def reset_dynamic_context(self) -> None:
        self._dynamic_context = AgenticDynamicContext()

    def _json_safe_prior_context(self) -> dict:
        """§2.8/J7: ``prior_context`` is loosely typed — ``set_prior_context`` /
        ``update_prior_context`` accept arbitrary caller values — so a stray callable,
        live handle, or exception would make the pause blob non-serializable (Tier-1
        ``to_json`` fails on resume) or persist garbage. Drop any non-JSON value (with
        a warning) so the blob always round-trips. The CI's own typed state
        (``sop_state`` / ``dynamic_context``) serializes separately via ``to_dict`` and
        is unaffected."""
        import json

        safe: dict = {}
        for key, value in self.prior_context.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                logger.warning(
                    "Dropping non-JSON-serializable prior_context[%r] (%s) from pause "
                    "state; prior_context must hold JSON-serializable values.",
                    key,
                    type(value).__name__,
                )
                continue
            safe[key] = value
        return safe

    def _enter_sop(self, name: str, *, yolo: bool = False):
        return self.sop_controller.enter_sop(name, yolo=yolo)

    def _reload_sop_definition(self, state) -> None:
        self.sop_controller.reload_sop_definition(state)

    def _format_suspended_sops(self) -> tuple[str, str]:
        return self.sop_controller.format_suspended_sops()

    def _mark_async_tool_phase_running(self, canonical: str) -> None:
        # Delegated to SOPController (Phase K).
        self.sop_controller.mark_async_tool_phase_running(canonical)

    def check_phase_completion(self, tool_name: str = "") -> None:
        """Advance the active SOP if its current phase is now complete."""
        self.sop_controller.check_phase_completion(tool_name)

    _check_phase_completion = check_phase_completion

    def next_required_tools(self) -> set[str]:
        """Return the set of required tool names for the NEXT available SOP phase.

        Used by the OpenTeam dispatcher (via its ``tool_dispatcher``'s
        inferencer back-ref) to attach a SOP-derived `next_step_tool` field to
        `task_completed` WS events. Thin cross-repo delegator per K5b — the
        actual implementation lives on `SOPController`.
        """
        return self.sop_controller.next_required_tools()

    def _ensure_sop_state_for_render(self) -> None:
        """Pre-render SOP preparation (``sop_feed.prepare_sop_for_turn``)."""
        if self.sop_controller is not None:
            prepare_sop_for_turn(
                self.sop_controller, prompt_renderer=self.prompt_renderer
            )

    def _build_sop_feed(self, *, catalog_mode: str = "when_idle") -> dict[str, Any]:
        """``sop_feed.build_sop_feed`` as a dict (plus the active ``sop``).
        Callers run ``_ensure_sop_state_for_render()`` first."""
        feed = build_sop_feed(
            self.sop_controller,
            self.prior_context,
            catalog_mode=catalog_mode,
            extra_sop_dirs=self.extra_sop_dirs,
            allowed=self.allowed_sops,
            disallowed=self.disallowed_sops,
        )
        return {"sop": feed.sop, **feed.template_values()}

    def _filtered_sops(self) -> dict[str, Any]:
        """The SOP catalog after this host's allow/deny filters."""
        return filtered_sops(
            extra_sop_dirs=self.extra_sop_dirs,
            allowed=self.allowed_sops,
            disallowed=self.disallowed_sops,
        )

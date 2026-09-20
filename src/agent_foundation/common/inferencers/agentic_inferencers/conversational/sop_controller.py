# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""SOPController — SOP orchestration state + behavior, extracted from CI.

Owns the 5 SOP state fields (`sop_state`, `_suspended_sops`, `_paused`,
`_pending_followup`, `_auto_shutdown_on_sop_complete`) as attribs, plus the
14 methods that manipulate them (previously ~445 LOC of SOP logic inline
on ``ConversationalInferencer``).

Narrow-context DI (Design Principle #4): construction takes callable hooks
for CI-owned surfaces (``add_message``, ``request_shutdown``,
``resolve_tool_name``, ``prior_context_reader``) plus data references
(``tool_registry``, ``workflow_manager``, ``prompt_renderer_ref``).
NEVER a full CI back-reference.

``serialize()``/``restore()`` keep the pre-extraction ``_conversation_blob``
emission byte-identical for the two SOP-owned keys (``sop_state``,
``suspended_sops``). ``restore(reattach_sop=False)`` preserves the
OpenStartup round-resume host contract.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

from attr import attrib, attrs

logger: logging.Logger = logging.getLogger(__name__)

# Extracted here so CI can import it too (both writes to
# `sop_state.user_input_gate_passed`).
DIRECTIVE_REQUIRES_USER_INPUT_LOCAL = "requires user input"


def _now_iso_local() -> str:
    """Local copy of `_now_iso` to avoid a CI back-import."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@attrs(kw_only=True, slots=False)
class SOPController:
    # Narrow init-time context.
    extra_sop_dirs: list = attrib(factory=list)
    allowed_sops: list = attrib(factory=list)
    disallowed_sops: list = attrib(factory=list)
    prompt_renderer_ref: Any = attrib(default=None)
    tool_registry: Optional[dict] = attrib(default=None)
    workflow_manager: Any = attrib(default=None)
    # Narrow callbacks — replaces CI back-ref.
    prior_context_reader: Optional[Callable[[], Mapping[str, Any]]] = attrib(
        default=None
    )
    add_message: Optional[Callable[[str, str], None]] = attrib(default=None)
    request_shutdown: Optional[Callable[[], None]] = attrib(default=None)
    resolve_tool_name: Optional[Callable[[str], str]] = attrib(default=None)

    # State (all init=False; defaults match pre-extraction __attrs_post_init__).
    sop_state: Any = attrib(default=None, init=False)
    _suspended_sops: list = attrib(factory=list, init=False)
    _paused: bool = attrib(default=False, init=False)
    _pending_followup: Optional[str] = attrib(default=None, init=False)
    _auto_shutdown_on_sop_complete: bool = attrib(default=False, init=False)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    @is_paused.setter
    def is_paused(self, value: bool) -> None:
        self._paused = value

    @property
    def is_active(self) -> bool:
        return self.sop_state is not None

    def current_sop(self) -> Any:
        return self.sop_state.sop if self.sop_state is not None else None

    # ------------------------------------------------------------------
    # Serialization boundary (K6 — byte-identical to pre-extraction).
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        return {
            "sop_state": self.sop_state.to_dict() if self.sop_state else None,
            "suspended_sops": [s.to_dict() for s in self._suspended_sops],
        }

    def restore(self, state: dict, *, reattach_sop: bool = True) -> None:
        """Restore SOP state; ``reattach_sop=False`` preserves OpenStartup host contract."""
        if not reattach_sop:
            return
        from agent_foundation.common.workflow.sop_state import SOPState

        sop_state_dict = state.get("sop_state")
        self.sop_state = SOPState.from_dict(sop_state_dict) if sop_state_dict else None
        if self.sop_state:
            self.reload_sop_definition(self.sop_state)
        self._suspended_sops = []
        for s in state.get("suspended_sops", []):
            restored = SOPState.from_dict(s)
            self.reload_sop_definition(restored)
            self._suspended_sops.append(restored)

    # ------------------------------------------------------------------
    # Phase K method migrations (from CI, ~445 LOC of SOP logic)
    # ------------------------------------------------------------------

    def reload_sop_definition(self, state: Any) -> None:
        """Reattach the SOP definition object after deserialization.

        Safe no-op when the definition is already attached (e.g. an in-memory
        suspended SOP being resumed in the same process).
        """
        if state.sop_name and state.sop is None:
            from agent_foundation.resources.sops.registry import load_sop

            state.sop = load_sop(state.sop_name).sop
            state.phase_required_tools = state.sop.phase_required_tools
            state.tool_phase_map = state.sop.tool_to_phase_map

    def enter_sop(self, name: str, *, yolo: bool = False):
        """Build an SOPState for ``name`` via the shared loader.

        Returns ``(SOPState, None)`` or ``(None, error_message)``.
        """
        from agent_foundation.resources.tools.sop.executor import build_sop_state

        return build_sop_state(
            name, yolo=yolo, extra_sop_dirs=self.extra_sop_dirs or None
        )

    def format_suspended_sops(self) -> tuple[str, str]:
        """Render the (paused_sop nudge, inprogress_sops list) prompt strings."""
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

    def consume_gate_for_no_tools_requires_input_phase(self) -> bool:
        """Phase J2: hoisted from `_render_prompt`.

        When the user-input gate is open AND the next available SOP phase is
        a "requires user input" phase with NO tools, consume the gate
        (``user_input_gate_passed = False``) AND mark that phase completed
        (append to ``completed_phases``). The persistent completion mirrors
        what the pre-refactor render-body's local `tracker.completed_states.add`
        achieved semantically — subsequent renders see the phase as done.

        Called as an explicit pre-render step from CI's
        ``_ensure_sop_state_for_render``. ``_render_prompt`` then only reads.
        Returns True if the gate was consumed + phase advanced.
        """
        if not self.sop_state or not self.sop_state.sop:
            return False
        s = self.sop_state
        if not s.user_input_gate_passed:
            return False
        try:
            from rich_python_utils.common_objects.workflow.stategraph import (
                StateGraphTracker,
            )
            from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
                SOPPhase,
            )

            completed = s.completed_phase_ids()
            tracker = StateGraphTracker(
                graph=s.sop,
                current_state=None,
                state_status="idle",
                completed_states=completed,
                state_outputs=s.phase_outputs,
                goto_counts=s.goto_counts,
            )
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
                    completed.append(node.id)
                    s.completed_phases = completed
                    s.user_input_gate_passed = False
                    return True
        except Exception as e:  # noqa: BLE001 — best-effort pre-render
            logger.debug("consume_gate_for_no_tools_requires_input_phase failed: %s", e)
        return False

    def consume_pending_followup(self) -> Optional[str]:
        """Return + clear ``_pending_followup`` (one-shot injection)."""
        followup = self._pending_followup
        self._pending_followup = None
        return followup

    def next_required_tools(self) -> set[str]:
        """Return the set of required tool names for the NEXT available SOP phase.

        Used by the OpenTeam dispatcher (via CI back-ref) to attach a
        SOP-derived ``next_step_tool`` field to ``task_completed`` events.
        Best-effort — returns an empty set on any error.
        """
        if not self.sop_state or not self.sop_state.sop:
            return set()
        try:
            from rich_python_utils.common_objects.workflow.common.phase_status import (
                PhaseStatus,
            )
            from rich_python_utils.common_objects.workflow.stategraph import (
                StateGraphTracker,
            )

            s = self.sop_state
            completed_ids = s.completed_phase_ids()
            tracker = StateGraphTracker(
                graph=s.sop,
                current_state=s.current_phase,
                state_status=s.phase_status or PhaseStatus.RUNNING,
                completed_states=completed_ids,
                state_outputs=s.phase_outputs,
                goto_counts=s.goto_counts,
            )
            available = tracker.get_available_next()
            if not available:
                return set()
            next_phase = available[0]
            return set(s.phase_required_tools.get(next_phase.id, set()))
        except Exception as e:  # noqa: BLE001 — best-effort inspector
            logger.debug("next_required_tools failed: %s", e)
            return set()

    def mark_async_tool_phase_running(self, canonical: str) -> None:
        """Reflect a just-dispatched async tool's SOP phase as RUNNING.

        FORWARD-ONLY: never move ``current_phase`` BACKWARD into an
        already-completed phase.
        """
        if not self.sop_state:
            return
        from rich_python_utils.common_objects.workflow.common.phase_status import (
            PhaseStatus,
        )

        sop_phase = self.sop_state.tool_phase_map.get(canonical)
        if sop_phase and sop_phase not in self.sop_state.completed_phase_ids():
            self.sop_state.current_phase = sop_phase
            self.sop_state.phase_status = PhaseStatus.RUNNING

    def open_user_input_gate_if_satisfied(
        self, tools: list, collected: Optional[dict]
    ) -> None:
        """Open the user-input gate when a widget response satisfies a
        ``requires_user_input`` SOP phase, so ``check_phase_completion`` can
        advance it.

        A phase that asks for input is satisfied once the user supplies it.
        The lone exception is a CONFIRMATION the user *declined* — a rejection
        is not satisfaction, so the phase must not advance on it. Every other
        tool type, and an affirmative confirmation, satisfies the gate.
        """
        if not self.sop_state or not collected:
            return
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
            ConversationToolType,
        )

        for tool in tools:
            if tool.tool_type != ConversationToolType.CONFIRMATION:
                continue
            var = tool.output_vars[0] if tool.output_vars else None
            value = collected.get(var) if var else None
            if value is None and len(collected) == 1:
                value = next(iter(collected.values()))
            # `_is_affirmative_response`: inlined for encapsulation.
            if str(value).strip().lower() not in ("yes", "proceed"):
                return
        # Gate genuinely satisfied. Record required conv tools per phase, mirror
        # into prior_context, and set the flag.
        self.record_answered_required_conv_tools(tools)
        if self.prior_context_reader is not None:
            pc = self.prior_context_reader()
            if pc is not None:
                # prior_context_reader returns the LIVE dict (per CI's lambda).
                pc["_user_input_gate_passed"] = True
        self.sop_state.user_input_gate_passed = True

    def record_answered_required_conv_tools(self, tools: list) -> None:
        """Record answered CONVERSATION tools that are required for the current
        SOP phase — required for Strategy 2's ``required <= executed`` guard
        in ``check_phase_completion`` when the required tools are conversation
        tools (0a/0b/1b/2b in model_optimization).

        Match against the CURRENT phase's required-tool SET (authoritative) —
        NOT tool_phase_map (which maps a tool name to only one phase).
        """
        if not self.sop_state:
            return
        cur = self.sop_state.current_phase
        if not cur:
            return
        required = self.sop_state.phase_required_tools.get(cur, set()) or set()
        if not required:
            return
        for tool in tools:
            name = getattr(tool.tool_type, "value", tool.tool_type)
            if name in required:
                self.sop_state.phase_executed_tools.setdefault(cur, set()).add(name)

    def record_yolo_answer(self, tools: list) -> None:
        """Yolo-path parity with the interactive gate (per plan §K4).

        Records answered required conv tools + opens the user-input gate +
        triggers phase-completion check. Encapsulates the previously-inline
        3-line ``if self.sop_state:`` block at the CI's yolo branch.
        """
        if not self.sop_state:
            return
        self.record_answered_required_conv_tools(tools)
        self.sop_state.user_input_gate_passed = True
        self.check_phase_completion()

    def check_phase_completion(self, tool_name: str = "") -> None:
        """Detect SOP phase completion and advance to next phase.

        Three detection strategies:
          1. All-required-tools: ALL tools declared in Tools[__required__]
             for this phase have been executed
          2. User input: user_input_gate_passed + DIRECTIVE_REQUIRES_USER_INPUT
          3. All-outputs-present: every declared output in phase_outputs
        """
        if not self.sop_state or not self.sop_state.sop:
            return

        from rich_python_utils.common_objects.workflow.common.phase_status import (
            PhaseStatus,
        )
        from rich_python_utils.common_objects.workflow.stategraph import (
            StateGraphTracker,
        )

        s = self.sop_state
        sop = s.sop
        current = s.current_phase
        if not current:
            return

        completed_ids = s.completed_phase_ids()
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
            if DIRECTIVE_REQUIRES_USER_INPUT_LOCAL in " ".join(
                getattr(phase, "directives", [])
            ):
                _required = s.phase_required_tools.get(current, set()) or set()
                _executed = s.phase_executed_tools.get(current, set()) or set()
                if _required <= _executed:
                    detected = True

        if not detected and hasattr(phase, "outputs") and phase.outputs:
            if all(o in s.phase_outputs for o in phase.outputs):
                _required = s.phase_required_tools.get(current, set()) or set()
                _executed = s.phase_executed_tools.get(current, set()) or set()
                if _required <= _executed:
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

        # Auto-shutdown bridge
        if (
            self._auto_shutdown_on_sop_complete
            and s.current_phase is None
            and s.phase_status == PhaseStatus.COMPLETED
            and self.request_shutdown is not None
        ):
            self.request_shutdown()

    # ------------------------------------------------------------------
    # Command bodies (moved from CI @command decorators — CI keeps thin
    # delegators to preserve CommandRegistry MRO scan)
    # ------------------------------------------------------------------

    def cmd_status_summary(self, messages_count: int) -> str:
        """SOP-portion of `/status` — combine with CI-portion at the caller."""
        n_susp = len(self._suspended_sops)
        susp_note = f" Suspended SOPs: {n_susp}." if n_susp else ""
        if not self.sop_state:
            return (
                f"No active SOP. Messages: {messages_count}. "
                f"Paused: {self._paused}.{susp_note}"
            )
        s = self.sop_state
        completed = s.completed_phase_ids()
        return (
            f"SOP: {s.sop_name}\n"
            f"Phase: {s.current_phase} ({s.phase_status})\n"
            f"Completed: {completed}\n"
            f"Messages: {messages_count}. Paused: {self._paused}.{susp_note}"
        )

    def cmd_sop(
        self,
        args: str = "",
        *,
        yolo_mode_setter: Optional[Callable[[bool], None]] = None,
    ) -> str:
        """Enter an SOP; auto-pause active SOP if any. Returns confirmation text."""
        tokens = args.split()
        if not tokens:
            return "Usage: /sop <name> [--yolo] [--fresh] [request...]"
        name = tokens[0]
        rest = tokens[1:]
        _KNOWN_FLAGS = {"--yolo", "--fresh"}
        yolo = "--yolo" in rest
        fresh = "--fresh" in rest
        request = " ".join(t for t in rest if t not in _KNOWN_FLAGS).strip()

        suspended = next((s for s in self._suspended_sops if s.sop_name == name), None)
        if suspended is not None and not fresh:
            return (
                f"You have an in-progress '{name}' ({suspended.sop_status}, "
                f"{suspended.suspension_label.lower()}). "
                f"Use /resume_sop {name} to resume, or "
                f"/sop {name} --fresh to start over."
            )

        state, error = self.enter_sop(name, yolo=yolo)
        if error:
            return error
        if self.sop_state is not None:
            self.sop_state.suspension_reason = "paused"
            self.sop_state.suspended_at = _now_iso_local()
            self._suspended_sops.insert(0, self.sop_state)
        self.sop_state = state
        if state.yolo_mode and yolo_mode_setter is not None:
            yolo_mode_setter(True)
        if request:
            self._pending_followup = request
            return f"Entered SOP '{name}'. Starting on: {request}"
        return f"Entered SOP '{name}'."

    def cmd_pause_sop(self) -> str:
        s = self.sop_state
        s.suspension_reason = "paused"
        s.suspended_at = _now_iso_local()
        self._suspended_sops.insert(0, s)
        self.sop_state = None
        return (
            f"SOP '{s.sop_name}' paused at {s.sop_status}. I'll remind you to resume."
        )

    def cmd_exit_sop(self) -> str:
        s = self.sop_state
        s.suspension_reason = "exited"
        s.suspended_at = _now_iso_local()
        self._suspended_sops.insert(0, s)
        self.sop_state = None
        return (
            f"Exited SOP '{s.sop_name}' ({s.sop_status}). "
            f"Resume anytime with /resume_sop {s.sop_name}."
        )

    def cmd_resume_sop(self, args: str = "") -> str:
        if not self._suspended_sops:
            return "No suspended SOPs to resume."
        tokens = args.split()
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
            match = self._suspended_sops[0]
        if self.sop_state is not None:
            self.sop_state.suspension_reason = "paused"
            self.sop_state.suspended_at = _now_iso_local()
            self._suspended_sops.insert(0, self.sop_state)
        self._suspended_sops.remove(match)
        match.suspension_reason = ""
        match.suspended_at = ""
        self.reload_sop_definition(match)
        self.sop_state = match
        if request:
            self._pending_followup = request
            return (
                f"Resumed SOP '{match.sop_name}' at {match.sop_status}. "
                f"Continuing on: {request}"
            )
        return f"Resumed SOP '{match.sop_name}' at {match.sop_status}."

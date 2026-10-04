"""Conversation widgets as a functional core shared by the conversational
orchestrators.

A widget is one batch of the model's conversation-tool calls, shown to the
user as one input (a compound, tabbed input when the batch has several tools).
Its lifecycle:

* ``prepare`` — attach proposal payloads and dashboard handoffs to the tools;
  ``conversation_tool_runtime.group_and_validate`` then checks that the batch
  can be one widget.
* ``present_and_collect`` — show the widget through an ``InteractiveBase``
  (persisted durably, so a restart can show it again), wait for the answer and
  ``decode`` it. Under yolo ``synthesize_yolo`` answers instead, and
  ``record_yolo_answer`` lets the SOP advance as it would on a user's answer.
* ``after_answer`` — what follows every answer: the dashboard opens, the
  answer's mailboxes are taken and cleared, the answer text is built, the SOP
  checks phase completion, and — unless the answer hands the work to a
  dashboard — the widget's bundled ``then_run`` actions run with
  ``substitute_vars`` and the user's parameter overrides. It is
  ``accept_answer`` + ``apply_turn_variables`` + ``run_then_run``; an
  orchestrator that interleaves bookkeeping of its own between those steps
  composes them itself.
* ``recover`` — decode a re-armed answer against the exact persisted widget,
  without re-inference.

Every function takes the orchestrator as a ``WidgetHost``. Decoded bindings
are returned to the caller, never left on the host.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    decode_compound_bindings,
    render_templated_fields,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    EffectTarget,
    HandlerContext,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_registry import (
    ConversationToolHandlerRegistry,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocol_text import (
    TOOL_RESULT_HEADER,
    TOOL_RESULTS_PREFIX,
    WIDGET_RESPONSE_PREFIX,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_dispatch import (
    apply_outcome,
    AsyncDoneCallback,
    execute,
    ToolDispatchHost,
    ToolOutcome,
)
from agent_foundation.ui.input_modes import ChoiceOption, InputMode, InputModeConfig
from agent_foundation.ui.interactive_base import InteractionFlags, InteractiveBase

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_YOLO_VALUE = "Follow your best judgment."


class WidgetHost(EffectTarget, ToolDispatchHost, Protocol):
    """The orchestrator surface the widget functions use; both conversational
    orchestrators implement it."""

    handler_registry: ConversationToolHandlerRegistry
    interactive: Optional[InteractiveBase]
    dashboard_coordinator: Any  # DashboardCoordinator | None
    workflow_manager: Any

    def _session_root(self) -> str: ...

    def _make_field_renderer(self) -> Optional[Callable[[str], str]]: ...

    def _resolve_tool_name(self, name: str) -> str: ...

    def last_prompt_data(self) -> dict[str, Any]: ...

    def export_state(self, *, turn_number: int = 0, iteration: int = 0) -> dict: ...


@dataclass(frozen=True)
class Decoded:
    """One tool's decoded answer: its text (``None``: no usable answer) and
    the composite bindings its handler decoded."""

    text: Optional[str]
    bindings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptedAnswer:
    """An answer after ``accept_answer``: the text that reports it, its
    bindings, whether it hands the work to a dashboard, and what its effects
    left for the widget's bundled actions."""

    response_text: str
    bindings: Any
    dashboard_handoff: bool
    action_overrides: Optional[Mapping[str, Any]] = None
    turn_variables: Optional[Mapping[str, str]] = None


@dataclass(frozen=True)
class ThenRunResult:
    """One bundled action, run and applied; ``name`` as the widget call gave it."""

    name: str
    outcome: ToolOutcome


@dataclass(frozen=True)
class AfterAnswer:
    """Everything that followed one answer.

    ``turn_variables`` are the ones published for the bundled actions (empty
    when none ran); ``async_dispatched`` is set when one of those actions
    started in the background."""

    dashboard_handoff: bool
    response_text: str
    bindings: Any
    turn_variables: Mapping[str, str] = field(default_factory=dict)
    then_run_results: tuple[ThenRunResult, ...] = ()
    async_dispatched: bool = False

    def message(self) -> str:
        """The answer as one message: its text, the turn variables, then the
        bundled actions' results."""
        text = self.response_text
        if self.turn_variables:
            text += "\n" + turn_variables_text(self.turn_variables)
        if self.then_run_results:
            text += f"\n\n{TOOL_RESULTS_PREFIX}\n" + then_run_text(
                self.then_run_results
            )
        return text


@dataclass(frozen=True)
class RecoveredAnswer:
    """A re-armed widget and its decoded answer (``bindings`` is ``None`` when
    the persisted answer is unusable; the widget then stays pending)."""

    tools: Sequence[ConversationTool]
    then_run: Sequence[Mapping[str, Any]]
    bindings: Optional[dict[str, str]]


# ---------------------------------------------------------------------------
# Handler plumbing
# ---------------------------------------------------------------------------


def _record_hitl_checkpoint(user_input) -> None:
    """§2.11: record a HITL decision (approve/reject + user input) into the active
    context node's Tier-1 ``checkpoints`` so resume can rehydrate it. Module-level
    (not a method) so it works regardless of the calling object's class. Additive —
    no-op without an active context; never affects control flow."""
    try:
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is None:
            return
        node = ctx.node()
        if (
            isinstance(user_input, (str, int, float, bool, dict, list))
            or user_input is None
        ):
            _payload_input = user_input
        else:
            _payload_input = str(user_input)
        node.checkpoints[f"hitl_{len(node.checkpoints)}"] = {
            "approved": user_input is not None,
            "user_input": _payload_input,
            "timestamp": time.time(),
        }
    except Exception:  # pragma: no cover - best-effort persistence
        pass


def _choice_option_from(c: ChoiceItem) -> ChoiceOption:
    """Build a UI ChoiceOption from a ChoiceItem, preserving the description and
    any embedded typed ``input`` spec (serialised) so composite choices and rich
    descriptions survive into ``InputModeConfig.to_dict()``."""
    return ChoiceOption(
        label=c.label,
        value=c.value,
        description=getattr(c, "description", "") or "",
        input=c.input.to_dict()
        if getattr(c, "has_input", False) and c.input is not None
        else None,
    )


def _build_input_mode(tool: ConversationTool, ctx: HandlerContext) -> InputModeConfig:
    """Dispatch through the handler registry.

    Handlers are registered per ConversationToolType in handlers/__init__.py.
    Adding a new tool type = new handler file + registry.register() call; NO
    edits to this function.

    Requires ``ctx.handler_registry`` to be set — ``handler_context`` populates
    it. The ``require`` lookup raises with a helpful message on any
    unregistered tool_type (this is also validated fail-fast in
    ``__attrs_post_init__``).
    """
    if ctx.handler_registry is None:
        raise RuntimeError(
            "_build_input_mode called with ctx.handler_registry=None — "
            "callers must construct ctx via widget_core.handler_context()"
        )
    handler = ctx.handler_registry.require(tool.tool_type)
    return handler.build_input_mode(tool, ctx)


def handler_context(
    host: WidgetHost,
    *,
    action_tools: Optional[list[dict[str, Any]]] = None,
    interactive: Optional[InteractiveBase] = None,
) -> HandlerContext:
    """Construct a HandlerContext with the MINIMAL surface per §A2.

    Do not add ``sop_state`` / ``tool_dispatcher`` / ``variable_manager`` /
    ``yolo_response`` fields here — they were audited-out (see plan §A2).
    SOP is loop-frame; dashboard-open is loop-frame; publishing is via
    ``PublishSessionVariablesEffect``; yolo response IS the ``response``
    arg to ``handle_response``. Adding preemptive fields violates the
    anti-refattening principle (DP #4).
    """
    return HandlerContext(
        prior_context=MappingProxyType(host.prior_context),
        prompt_renderer=host.prompt_renderer,
        tool_executor=host.tool_executor,
        interactive=interactive or host.interactive,
        action_tools=action_tools,
        tool_registry=host.tool_registry,
        resolve_tool_name=host._resolve_tool_name,
        session_root=host._session_root(),
        handler_registry=host.handler_registry,
    )


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def prepare(host: WidgetHost, tools: Sequence[ConversationTool]) -> None:
    """Make the model's widget calls ready for either answer path (user or
    yolo): ``proposal_selection`` tools get their proposals and one choice per
    proposal (so yolo ``select_all`` sees them too), then a dashboard flag
    (``experiment_hub`` / ``host_dashboard``) on any tool becomes
    ``metadata.open_dashboard`` (+ ``submit_label``) when that dashboard
    embeds the tool's widget, which the widget relabel, the dashboard opener
    and the handoff decision read."""
    for tool in tools:
        if tool.tool_type == ConversationToolType.PROPOSAL_SELECTION:
            enrich_proposal_selection(host, tool)
    if host.dashboard_coordinator is not None:
        host.dashboard_coordinator.normalize_directives(list(tools))


def resolve_proposals_source(
    host: WidgetHost, tool: ConversationTool
) -> Optional[dict]:
    """Resolve proposal data for a ``proposal_selection`` tool.

    Priority:
      1. ``tool.metadata["proposals"]`` already present (dict) — use as-is.
      2. ``tool.metadata["proposals_path"]`` → AF ``parse_proposal_file``.
         This is the AF-native path: the SOP body passes proposals_path
         (typically via Jinja ``{{ workspace_path__research_propose }}``).
      3. A host-registered :class:`ProposalParser` (e.g. RankEvolve), fed a
         workspace discovered from ``host.prior_context``.

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
                path,
                exc,
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
                or host.prior_context.get("workspace_path__research_propose")
                or host.prior_context.get("workspace_path")
            )
            if workspace:
                data = parser.parse(str(workspace))
                if data is not None:
                    return data.to_dict() if hasattr(data, "to_dict") else data
    except Exception as exc:  # noqa: BLE001
        logger.info("[proposal_selection] registered parser failed: %s", exc)

    return None


def enrich_proposal_selection(host: WidgetHost, tool: ConversationTool) -> None:
    """Populate proposals + choices + output var for a proposal_selection tool.

    Attaches the resolved proposal payload to ``tool.metadata["proposals"]``
    for the rich widget, derives one selectable choice per proposal id (so
    selection flows through AF's multiple-choice machinery, including yolo
    ``select_all``), and defaults the output variable to
    ``selected_proposal_ids`` when the SOP author omitted it.
    """
    proposals = resolve_proposals_source(host, tool)
    if not proposals:
        return
    if tool.metadata is None:
        tool.metadata = {}
    tool.metadata["proposals"] = proposals

    # Attach `proposal_file_abs` per proposal so the widget can lazy-fetch
    # the full per-proposal `.md` doc via `GET /api/view/<abs>` on expand.
    # Only meaningful when `proposals_path` is set (the AF-native SOP path);
    # when proposals arrived via a host-registered parser or LLM-inline
    # (no `.json` file path), the widget gracefully falls back to inline
    # detail fields — matching today's behavior for those sources.
    proposals_json_path = tool.metadata.get("proposals_path")
    if proposals_json_path:
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        attach_proposal_file_abs(proposals, proposals_json_path)

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


# ---------------------------------------------------------------------------
# present_and_collect
# ---------------------------------------------------------------------------


async def present_and_collect(
    host: WidgetHost,
    tools: Sequence[ConversationTool],
    preamble: str,
    *,
    interactive: Optional[InteractiveBase],
    then_run: Optional[list[dict[str, Any]]] = None,
    turn_number: Optional[int] = None,
    iteration: Optional[int] = None,
) -> Optional[dict[str, str]]:
    """Show ``tools`` as one widget (a compound widget for several tools)
    under ``preamble``, wait for the answer and ``decode`` it.

    Each tool's handler first enriches it (``enrich_before_send``; a
    confirmation's parameter panel comes from the bundled ``then_run``
    actions). Returns the answer's bindings, or ``None`` without an
    interactive or a usable answer.
    """
    if not tools or interactive is None:
        return None
    await enrich(host, tools, then_run)
    answer = await show(
        host,
        tools,
        preamble,
        interactive=interactive,
        then_run=then_run,
        turn_number=turn_number,
        iteration=iteration,
    )
    return await decode(host, tools, answer)


async def enrich(
    host: WidgetHost,
    tools: Sequence[ConversationTool],
    then_run: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Let each tool's registered handler enrich it before it is shown
    (handlers without an override are no-ops)."""
    ctx = handler_context(host, action_tools=then_run)
    for tool in tools:
        handler = host.handler_registry.get(tool.tool_type)
        if handler is not None:
            await handler.enrich_before_send(tool, ctx)


async def show(
    host: WidgetHost,
    tools: Sequence[ConversationTool],
    preamble: str,
    *,
    interactive: InteractiveBase,
    then_run: Optional[list[dict[str, Any]]] = None,
    turn_number: Optional[int] = None,
    iteration: Optional[int] = None,
) -> Any:
    """Send ``tools`` as one pending input (compound when there are several,
    so the frontend renders a tabbed multi-input widget), persist it, and
    return the raw answer."""
    renderer = host._make_field_renderer()
    if len(tools) == 1:
        tool = tools[0]
        # Resolve any templated prefix (e.g. echoed "{{ session_root_path }}")
        # before building the UI config / finalising values.
        render_templated_fields(tool, renderer)
        input_mode = _build_input_mode(tool, handler_context(host))
        _add_variable_content(host, tool, input_mode)
    else:
        ctx = handler_context(host, action_tools=then_run)
        tool_configs = []
        for tool in tools:
            render_templated_fields(tool, renderer)
            mode = _build_input_mode(tool, ctx)
            _add_variable_content(host, tool, mode)
            tool_configs.append(
                {
                    "tool_type": tool.tool_type,
                    "prompt": tool.prompt,
                    "input_mode": mode.to_dict(),
                    "output_var": tool.output_vars[0]
                    if tool.output_vars
                    else tool.tool_type,
                    "expected_input_type": tool.expected_input_type,
                    "prefix": tool.prefix,
                }
            )
        input_mode = InputModeConfig(
            mode=InputMode.FREE_TEXT,
            prompt=preamble,
            metadata={
                "compound": True,
                "tools": tool_configs,
            },
        )
    await interactive.asend_response(
        preamble,
        flag=InteractionFlags.PendingInput,
        input_mode=input_mode,
        prompt_data=_prompt_data(host),
    )
    # Persisted before blocking below, so a disconnect/restart while the
    # widget is pending can re-display and re-arm it.
    persist_pending(
        host,
        interactive,
        list(tools),
        then_run,
        turn_number=turn_number,
        iteration=iteration,
    )
    return await interactive.aget_input()


def _prompt_data(host: WidgetHost) -> dict[str, Any]:
    """The rendered prompt, inline with the widget, so the UI's "View Prompt"
    button on the widget preamble needs no REST round-trip. Server-side
    transports (e.g. WebSocketInteractive) read the ``prompt_data`` kwarg;
    transports that don't care ignore it."""
    data = host.last_prompt_data()
    return {
        "template_source": data.get("template_source") or "",
        "template_feed": data.get("template_feed") or {},
        "rendered_prompt": data.get("rendered_prompt") or "",
        "template_config": data.get("template_config") or {},
    }


def _add_variable_content(
    host: WidgetHost, tool: ConversationTool, input_mode: InputModeConfig
) -> None:
    """Enrich the input mode with variable content for UI display (an
    editable text block)."""
    if not host.prompt_renderer:
        return
    try:
        var_name = tool.output_vars[0] if tool.output_vars else None
        vm = host.prompt_renderer.variable_manager

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
                for c in tool.choices
                if c.get("value")
            ]
            for alias in getattr(vm, "_scoped_aliases", {}).values():
                try:
                    candidate = vm.get_effective_value(alias, skip_overrides=True)
                    if isinstance(candidate, dict):
                        norm_keys = {
                            k.lower().replace(" ", "_").replace("-", "_"): k
                            for k in candidate
                        }
                        if choice_values and all(v in norm_keys for v in choice_values):
                            input_mode.metadata["variable_content"] = {
                                k: str(v).strip() for k, v in candidate.items()
                            }
                            input_mode.metadata["variable_name"] = alias
                            break
                except Exception:
                    continue
    except Exception:
        pass  # Non-critical — widget works without enrichment


def persist_pending(
    host: WidgetHost,
    interactive: Any,
    tools: list,
    then_run: Optional[list],
    *,
    turn_number: Optional[int],
    iteration: Optional[int],
) -> None:
    """After a widget is emitted (``asend_response``) and BEFORE blocking on
    ``aget_input``, durably persist it (Layer 2, Piece 1): the marker
    (session_state.json) + the emit-point continuation blob (sidecar). A
    disconnect/restart while the widget is pending can then re-display AND
    re-arm it. No-op unless the transport supports persistence and the
    (turn, iteration) needed for the emit-point blob is known."""
    if (
        turn_number is None
        or iteration is None
        or not hasattr(interactive, "persist_pending_widget")
    ):
        return
    try:
        blob = host.export_state(turn_number=turn_number, iteration=iteration)
        interactive.persist_pending_widget(
            tools=tools, action_tools=then_run, blob=blob
        )
    except Exception as e:  # best-effort — never break the live turn
        logger.warning("persist pending widget failed: %s", e)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def _answer_var(tool: ConversationTool) -> str:
    return tool.output_vars[0] if tool.output_vars else "input"


async def decode(
    host: WidgetHost, tools: Sequence[ConversationTool], user_input: Any
) -> Optional[dict[str, str]]:
    """Turn a widget's raw answer into its bindings, publish them and open
    the SOP's user-input gate (a declined confirmation is withheld there).
    Deterministic: no model call. ``None`` when there is no usable answer."""
    if not tools:
        return None
    if len(tools) > 1:
        return decode_compound(host, tools, user_input)
    decoded = await decode_answer(host, tools[0], user_input)
    if decoded is None or decoded.text is None:
        return None
    collected: dict[str, str] = {_answer_var(tools[0]): decoded.text}
    collected.update(decoded.bindings)
    host.sop_controller.open_user_input_gate_if_satisfied(list(tools), collected)
    return collected


async def decode_answer(
    host: WidgetHost,
    tool: ConversationTool,
    user_input: Any,
    *,
    context: Optional[HandlerContext] = None,
) -> Optional[Decoded]:
    """Apply one tool's raw answer DETERMINISTICALLY (no LLM) via the handler
    registry: the handler decodes it and its effects are applied to ``host``.
    ``context`` defaults to ``handler_context(host)``. ``None`` for a missing
    answer.

    Handlers are pure functions of ``(tool, response, ctx)`` per Design
    Principle #13. The HITL checkpoint is recorded here, before dispatch —
    cross-cutting infrastructure, not tool-type-specific decode.
    """
    _record_hitl_checkpoint(user_input)
    if user_input is None:
        return None

    # Extract the response payload. Wrap bare strings so handlers see a
    # uniform dict contract.
    if isinstance(user_input, dict):
        response = user_input.get("user_input", user_input.get("content", user_input))
    else:
        response = user_input
    if not isinstance(response, dict):
        response = {"content": response}

    handler = host.handler_registry.require(tool.tool_type)
    ctx = context if context is not None else handler_context(host)
    result = await handler.handle_response(tool, response, ctx)
    for effect in result.effects:
        await effect.apply(host)
    return Decoded(text=result.text or None, bindings=dict(result.bindings or {}))


def decode_compound(
    host: WidgetHost, tools: Sequence[ConversationTool], user_input: Any
) -> Optional[dict[str, str]]:
    """Decode a COMPOUND (multi-tool / tabbed) widget's raw answer into the
    ``collected`` dict, publish each child tool's output vars, and open the
    user-input gate."""
    _record_hitl_checkpoint(user_input)  # §2.11: persist HITL decision (Tier-1)
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
                list(tools), values, session_root=host._session_root()
            )
            # Rich-choice editable content override wins, if present.
            variable_override = values.get("variable_override")
            if isinstance(variable_override, dict):
                bindings.update(variable_override)
            collected.update(bindings)
            if bindings:
                # A1.b (v3): per-tool namespaced publish for compound
                # widgets. Iterate child tools and publish each's subset of
                # bindings with its own tool_type so `<tool_type>__<var>`
                # aliases carry the correct producing tool. Any binding
                # not claimed by a child tool's output_vars still lands
                # via the residual aggregate write below.
                claimed: set[str] = set()
                for child_tool in tools:
                    out_vars = getattr(child_tool, "output_vars", None) or []
                    subset = {v: bindings[v] for v in out_vars if v in bindings}
                    if subset:
                        host.set_session_variables(
                            subset,
                            tool_type=getattr(child_tool, "tool_type", None),
                        )
                        claimed.update(subset.keys())
                residual = {k: v for k, v in bindings.items() if k not in claimed}
                if residual:
                    host.set_session_variables(residual)
        else:
            # Fallback: single value
            collected["input"] = str(values)
    else:
        collected["input"] = str(user_input)

    host.sop_controller.open_user_input_gate_if_satisfied(list(tools), collected)
    return collected


# ---------------------------------------------------------------------------
# synthesize_yolo
# ---------------------------------------------------------------------------


async def synthesize_yolo(
    host: WidgetHost, tools: Sequence[ConversationTool]
) -> Optional[dict[str, str]]:
    """Answer conversation tools autonomously (yolo).

    The synthetic responses go through the SAME handler registry as a user's
    answer — HITL checkpoint, dashboard directives and nested bindings
    included; handlers never branch on interactive-vs-yolo, the response dict
    IS the contract (Design Principles #3 and #13). So yolo ``select_all`` on
    an ``--experiment-hub`` proposal_selection captures its directive too.
    """
    if not tools:
        return None

    collected: dict[str, str] = {}
    for tool in tools:
        handler = host.handler_registry.require(tool.tool_type)
        result = await handler.handle_response(
            tool, yolo_response(host, tool), handler_context(host)
        )
        for effect in result.effects:
            await effect.apply(host)
        if tool.output_vars and result.text:
            collected[tool.output_vars[0]] = result.text
        # Composite nested bindings (§A1.a — decode RESULT, not effect).
        if result.bindings:
            collected.update(result.bindings)
    return collected


def record_yolo_answer(host: WidgetHost, tools: Sequence[ConversationTool]) -> None:
    """An autonomous answer satisfies the SOP as a user's would: the answered
    required conversation tools are recorded, the user-input gate opens, and
    the phase check runs."""
    host.sop_controller.record_yolo_answer(list(tools))


def yolo_response(host: WidgetHost, tool: ConversationTool) -> dict[str, Any]:
    """Build a UI-shaped response for a tool under yolo, from its yolo spec.

    A free-text answer is the widget's ``default`` when the model prefilled
    one (SOPs tell it to put a target the user already named there), else the
    resolved session root for a path input (a valid "investigate everything"
    target, never prose as a path), else the spec's value. A confirmation's
    answer is a decision, so its ``default`` is not used.
    """
    spec = yolo_spec(host, tool)
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
    value = spec.get("value", _DEFAULT_YOLO_VALUE)
    prefill = _prefilled_answer(tool)
    if prefill:
        value = prefill
    elif tool.expected_input_type == "path":
        value = host._session_root() or value
    return {"content": value}


def _prefilled_answer(tool: ConversationTool) -> str:
    if tool.tool_type == ConversationToolType.CONFIRMATION:
        return ""
    default = (tool.metadata or {}).get("default")
    return default.strip() if isinstance(default, str) else ""


def yolo_spec(host: WidgetHost, tool: Any) -> dict:
    """Resolution order: per-SOP override → tool.json default → builtin."""
    tool_type = getattr(tool, "tool_type", "")

    # Check per-SOP yolo_overrides
    sop_instance_id = host.prior_context.get("sop_instance_id")
    workflow_manager = getattr(host, "workflow_manager", None)
    if sop_instance_id and workflow_manager:
        try:
            instance = workflow_manager.active_instances.get(sop_instance_id)
            if instance:
                definition = workflow_manager.registry.get(instance.definition_id)
                if hasattr(definition, "frontmatter"):
                    overrides = definition.frontmatter.get("yolo_overrides", {})
                    if tool_type in overrides:
                        return overrides[tool_type]
        except Exception:
            pass

    # Check tool.json yolo_default
    tool_name = getattr(tool, "tool_type", "") or getattr(tool, "name", "")
    tool_def = host.tool_registry.get(tool_name)
    if tool_def and getattr(tool_def, "yolo_default", None):
        return tool_def.yolo_default

    # Builtin fallback
    return {"mode": "fixed", "value": _DEFAULT_YOLO_VALUE}


# ---------------------------------------------------------------------------
# after_answer
# ---------------------------------------------------------------------------


def answer_text(bindings: Any) -> str:
    """The message that reports an answer to the model."""
    if isinstance(bindings, dict):
        parts = [f"{k}: {v}" for k, v in bindings.items() if v]
        return f"{WIDGET_RESPONSE_PREFIX}\n" + (
            "\n".join(parts) if parts else str(bindings)
        )
    return f"{WIDGET_RESPONSE_PREFIX}\n{bindings}"


def is_dashboard_handoff(tools: Sequence[ConversationTool]) -> bool:
    """Whether answering ``tools`` hands the work to a dashboard (e.g.
    ``proposal_selection --experiment-hub``): a tool carries
    ``metadata.open_dashboard`` (set by ``prepare``; it survives
    ``to_dict``/``from_dict``, so this holds for recovered widgets too)."""
    return any(
        isinstance(getattr(t, "metadata", None), dict)
        and t.metadata.get("open_dashboard")
        for t in tools
    )


async def accept_answer(
    host: WidgetHost, tools: Sequence[ConversationTool], bindings: Any
) -> AcceptedAnswer:
    """The first steps after every answer.

    The dashboard opens (awaited, so its subtab is active before the turn
    advances; the selection is already published, so the SOP contract holds
    even if the open no-ops). All three mailboxes are taken and cleared: the
    parameter overrides and turn variables belong to this answer's bundled
    actions only, and a mailbox left set would make the next answer's effect
    raise ``HandlerResultMergeConflict``. Then the SOP checks phase completion
    — before the handoff decision, so e.g. a 2b→3 advance still happens.
    """
    mailboxes = host.mailboxes
    if host.dashboard_coordinator is not None:
        await host.dashboard_coordinator.maybe_open(
            list(tools),
            bindings,
            next_dashboard_directives=mailboxes.dashboard_directives,
        )
    overrides, turn_variables = mailboxes.action_overrides, mailboxes.turn_variables
    mailboxes.clear()
    text = answer_text(bindings)
    host.sop_controller.check_phase_completion()
    return AcceptedAnswer(
        response_text=text,
        bindings=bindings,
        dashboard_handoff=is_dashboard_handoff(tools),
        action_overrides=overrides,
        turn_variables=turn_variables,
    )


def runs_then_run(host: WidgetHost, then_run: Sequence[Mapping[str, Any]]) -> bool:
    """Whether an answer's bundled actions run (there are some, and a tool
    executor to run them)."""
    return bool(then_run) and bool(host.tool_executor)


def apply_turn_variables(host: WidgetHost, variables: Mapping[str, str]) -> None:
    """Publish an answer's turn variables (e.g. a confirmation's edited
    variables) for its bundled actions: into ``prior_context`` and the prompt
    renderer's variable manager."""
    vm = (
        getattr(host.prompt_renderer, "variable_manager", None)
        if host.prompt_renderer
        else None
    )
    for key, value in variables.items():
        host.prior_context[key] = value
        if vm is not None and hasattr(vm, "set"):
            vm.set(key, value)


def turn_variables_text(variables: Mapping[str, str]) -> str:
    return "\n".join(f"[{k}]: {v}" for k, v in variables.items())


def substitute_vars(arguments: Optional[Mapping[str, Any]], bindings: Any) -> dict:
    """A bundled action's arguments with each ``"__name__"`` placeholder
    replaced by the answer's binding ``name`` (left as is when the answer has
    no such binding)."""
    resolved = {}
    for key, value in (arguments or {}).items():
        if isinstance(value, str) and value.startswith("__") and value.endswith("__"):
            name = value[2:-2]
            if isinstance(bindings, dict) and name in bindings:
                value = bindings[name]
        resolved[key] = value
    return resolved


async def run_then_run(
    host: WidgetHost,
    then_run: Sequence[Mapping[str, Any]],
    accepted: AcceptedAnswer,
    *,
    run_ctx: Any = None,
    on_async_done: Optional[AsyncDoneCallback] = None,
    on_applied: Optional[Callable[[str, ToolOutcome], None]] = None,
    is_live: Optional[Callable[[], bool]] = None,
) -> tuple[ThenRunResult, ...]:
    """Run an answer's bundled actions in order: ``substitute_vars`` from the
    answer's bindings, then the user's parameter overrides, through
    ``tool_dispatch.execute`` under ``run_ctx``.

    Each result is applied (``apply_outcome``) and reported to ``on_applied``
    only while ``is_live()`` holds when the action returns (e.g. its turn is
    still the active one); otherwise it and the remaining actions are dropped.
    """
    results: list[ThenRunResult] = []
    for action in then_run:
        name = action.get("name", "")
        arguments = substitute_vars(action.get("arguments"), accepted.bindings)
        if accepted.action_overrides:
            arguments.update(accepted.action_overrides)
        outcome = await execute(
            host, name, arguments, run_ctx=run_ctx, on_async_done=on_async_done
        )
        if is_live is not None and not is_live():
            logger.warning(
                "then_run tool %s returned after its turn ended; its result "
                "is not applied",
                name,
            )
            break
        outcome = apply_outcome(host, outcome)
        if on_applied is not None:
            on_applied(name, outcome)
        results.append(ThenRunResult(name=name, outcome=outcome))
    return tuple(results)


def then_run_text(results: Sequence[ThenRunResult]) -> str:
    return "\n\n".join(
        f"{TOOL_RESULT_HEADER.format(r.name)}\n{r.outcome.text}" for r in results
    )


async def after_answer(
    host: WidgetHost,
    tools: Sequence[ConversationTool],
    then_run: Sequence[Mapping[str, Any]],
    bindings: Any,
    *,
    run_ctx: Any = None,
    on_async_done: Optional[AsyncDoneCallback] = None,
    on_applied: Optional[Callable[[str, ToolOutcome], None]] = None,
    is_live: Optional[Callable[[], bool]] = None,
) -> AfterAnswer:
    """Everything that follows an answer (live, autonomous or recovered):
    ``accept_answer``; then, unless the answer hands the work to a dashboard
    (which ends the turn quietly) and if the bundled actions run, the turn
    variables are published and ``run_then_run`` runs them."""
    accepted = await accept_answer(host, tools, bindings)
    if accepted.dashboard_handoff or not runs_then_run(host, then_run):
        return AfterAnswer(
            dashboard_handoff=accepted.dashboard_handoff,
            response_text=accepted.response_text,
            bindings=bindings,
        )
    variables = dict(accepted.turn_variables or {})
    apply_turn_variables(host, variables)
    results = await run_then_run(
        host,
        then_run,
        accepted,
        run_ctx=run_ctx,
        on_async_done=on_async_done,
        on_applied=on_applied,
        is_live=is_live,
    )
    return AfterAnswer(
        dashboard_handoff=False,
        response_text=accepted.response_text,
        bindings=bindings,
        turn_variables=variables,
        then_run_results=results,
        async_dispatched=any(r.outcome.is_async for r in results),
    )


# ---------------------------------------------------------------------------
# recover
# ---------------------------------------------------------------------------


async def recover(host: WidgetHost, pending: Mapping[str, Any]) -> RecoveredAnswer:
    """Decode a re-armed widget answer (a host restart or reconnect while the
    widget was pending) against the EXACT persisted widget: no re-inference,
    which could emit a different widget and bind the answer to the wrong
    variable. ``pending`` holds ``tools``, ``action_tools`` (its bundled
    actions) and the answer's ``raw_value``."""
    tools = pending.get("tools") or []
    then_run = pending.get("action_tools") or []
    return RecoveredAnswer(
        tools=tools,
        then_run=then_run,
        bindings=await decode(host, tools, pending.get("raw_value")),
    )

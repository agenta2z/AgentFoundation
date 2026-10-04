"""Per-turn SOP preparation and the SOP-derived prompt values, shared by the
conversational orchestrators.

``prepare_sop_for_turn`` holds the only render-time SOP mutations. The classic
orchestrator runs it before every render; the native one before every vendor
turn, even when no turn context is sent, because a "requires user input" phase
without tools advances only there. ``build_sop_feed`` is then a pure reader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent_foundation.common.workflow.sop_state import SOPState
from agent_foundation.resources.sops.registry import format_all_sops, load_all_sops
from rich_python_utils.common_objects.workflow.stategraph import StateGraphTracker
from rich_python_utils.string_utils.formatting.common import resolve_templated_feed
from rich_python_utils.string_utils.formatting.jinja2_format import extract_variables
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOPManager,
)

logger: logging.Logger = logging.getLogger(__name__)

CATALOG_MODES: tuple[str, ...] = ("when_idle", "always")


@dataclass(frozen=True)
class SopFeed:
    """SOP-derived prompt values for one render."""

    # The active SOP's definition (None when no SOP, or its definition, is loaded).
    sop: Any = None
    sop_nextstep_guidance: str = ""
    available_sops: str = ""
    paused_sop: str = ""
    inprogress_sops: str = ""

    @property
    def sop_active(self) -> bool:
        return self.sop is not None

    def template_values(self) -> dict[str, Any]:
        """The values the conversation templates consume."""
        return {
            "sop_nextstep_guidance": self.sop_nextstep_guidance,
            "available_sops": self.available_sops,
            "paused_sop": self.paused_sop,
            "inprogress_sops": self.inprogress_sops,
            "sop_active": self.sop_active,
        }


def prepare_sop_for_turn(controller: Any, *, prompt_renderer: Any = None) -> None:
    """With no active SOP, adopt the SOP file co-located with the prompt
    template, if the renderer finds one (legacy auto-discover); then let a
    satisfied user-input gate complete a no-tools "requires user input" phase.
    """
    if controller.sop_state is None:
        find_sop_file = getattr(prompt_renderer, "find_sop_file", None)
        sop_path = find_sop_file() if find_sop_file is not None else None
        if sop_path is not None:
            loaded = SOPManager.load(sop_path)
            controller.sop_state = SOPState(
                sop=loaded,
                sop_name=loaded.name or Path(sop_path).stem,
                tool_phase_map=getattr(loaded, "tool_to_phase_map", {}),
            )
    controller.consume_gate_for_no_tools_requires_input_phase()


def filtered_sops(
    *,
    extra_sop_dirs: Sequence[str | Path] = (),
    allowed: Sequence[str] = (),
    disallowed: Sequence[str] = (),
) -> dict[str, Any]:
    """Discoverable SOPs after the allow-then-deny filters (precedence as in
    iptables / AWS IAM / k8s NetworkPolicy): a non-empty ``allowed`` keeps only
    those names, then ``disallowed`` drops names from the survivors; an empty
    list does not filter.

    Purely presentational: hidden SOPs stay enterable by name.
    """
    sops = load_all_sops(extra_dirs=list(extra_sop_dirs) or None)
    if sops and allowed:
        allowed_names = set(allowed)
        sops = {n: i for n, i in sops.items() if n in allowed_names}
    if sops and disallowed:
        denied = set(disallowed)
        sops = {n: i for n, i in sops.items() if n not in denied}
    return sops


def build_sop_feed(
    controller: Any,
    prior_context: Mapping[str, Any],
    *,
    catalog_mode: str = "when_idle",
    extra_sop_dirs: Sequence[str | Path] = (),
    allowed: Sequence[str] = (),
    disallowed: Sequence[str] = (),
) -> SopFeed:
    """SOP values for a render: next-step guidance of the active SOP, the
    filtered catalog, and the paused / in-progress lists.

    ``catalog_mode="when_idle"`` lists the catalog only while no SOP is active
    (the classic prompt); ``"always"`` lists it regardless (the native session
    instructions). Run ``prepare_sop_for_turn`` first.
    """
    if catalog_mode not in CATALOG_MODES:
        raise ValueError(
            f"catalog_mode must be one of {CATALOG_MODES}, got {catalog_mode!r}"
        )
    state = controller.sop_state
    sop = state.sop if state is not None else None
    guidance = _nextstep_guidance(state, prior_context) if sop is not None else ""
    available = ""
    if catalog_mode == "always" or state is None:
        sops = filtered_sops(
            extra_sop_dirs=extra_sop_dirs, allowed=allowed, disallowed=disallowed
        )
        if sops:
            available = format_all_sops(sops)
    paused, inprogress = controller.format_suspended_sops()
    return SopFeed(
        sop=sop,
        sop_nextstep_guidance=guidance,
        available_sops=available,
        paused_sop=paused,
        inprogress_sops=inprogress,
    )


def resolve_feed(
    feed: dict[str, Any], render_string: Callable[[str, dict], str]
) -> dict[str, Any]:
    """Resolve feed values that are themselves templates (e.g. SOP guidance
    containing ``{{ session_root_path }}``) against the complete feed, with the
    main template's renderer. A cyclic reference leaves the feed unresolved."""
    try:
        return resolve_templated_feed(
            feed, extract_variables=extract_variables, render_template=render_string
        )
    except ValueError as e:
        logger.warning("Feed self-resolution failed: %s", e)
        return feed


def _nextstep_guidance(state: Any, prior_context: Mapping[str, Any]) -> str:
    try:
        tracker = StateGraphTracker(
            graph=state.sop,
            current_state=None,
            state_status="idle",
            completed_states=state.completed_phase_ids(),
            state_outputs=state.phase_outputs,
            goto_counts=state.goto_counts,
        )
        return SOPManager.render_guidance(
            tracker, state.sop, context=dict(prior_context)
        )
    except Exception as e:  # noqa: BLE001 — a broken SOP must not break the turn
        logger.warning("SOP evaluation failed: %s", e)
        return ""

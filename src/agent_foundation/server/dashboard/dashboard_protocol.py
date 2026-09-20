"""Dashboard protocol — manifest + scenarioState schema (the Python mirror of the
JS dashboard manifest) and the WebSocket event shapes for the generic Dashboard
framework.

Transport-agnostic: this module defines data + payload builders only. The host
app (OpenTeam) owns the actual WebSocket transport and emits these payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# History-message role — UI-only, filtered from LLM history exactly like task_ref.
DASHBOARD_REF_ROLE: str = "dashboard_ref"

# WebSocket event types (server -> client).
WS_DASHBOARD_OPEN: str = "dashboard_open"
WS_DASHBOARD_STATUS: str = "dashboard_status"
WS_DASHBOARD_EVENT: str = "dashboard_event"

# `dashboard_event.event_type` values — the single generic live-update family
# (replaces RankEvolve's bespoke submission_state/setup_completed/... events).
EVENT_RUN_PROGRESS: str = "run_progress"
EVENT_RUN_COMPLETED: str = "run_completed"
EVENT_SUBMISSION_STATE: str = "submission_state"
EVENT_SETUP_COMPLETED: str = "setup_completed"
EVENT_COMBO_CHANGED: str = "combo_changed"
EVENT_VERDICT_READY: str = "verdict_ready"
EVENT_AUTOPILOT_TICK: str = "autopilot_tick"
# Generic terminal "the user marked the evolution cycle done" signal. The hub
# stays SOP-agnostic — it just says "I'm done"; the HOST maps this to a real
# Phase-3 completion (writes the declared Phase-3 output + _check_phase_completion).
EVENT_EVOLUTION_COMPLETE: str = "evolution_complete"

DASHBOARD_EVENT_TYPES: tuple[str, ...] = (
    EVENT_RUN_PROGRESS,
    EVENT_RUN_COMPLETED,
    EVENT_SUBMISSION_STATE,
    EVENT_SETUP_COMPLETED,
    EVENT_COMBO_CHANGED,
    EVENT_VERDICT_READY,
    EVENT_AUTOPILOT_TICK,
    EVENT_EVOLUTION_COMPLETE,
)


@dataclass
class ViewSpec:
    """One tab of a dashboard. ``type`` resolves to a React component via the JS
    ViewRegistry; ``config`` is opaque per-view configuration shipped to the
    client (e.g. a widget_host's hosted-widget list)."""

    type: str
    label: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.label:
            d["label"] = self.label
        if self.config:
            d["config"] = self.config
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViewSpec:
        return cls(
            type=data["type"],
            label=data.get("label", ""),
            config=dict(data.get("config", {}) or {}),
        )


@dataclass
class DashboardManifest:
    """The canonical tab structure of a dashboard. Source of truth = a Dashboard
    tool's ``dashboard_config.view_manifest``; shipped to the client in the
    ``dashboard_open`` payload (so the FE never re-declares the tab list)."""

    id: str
    label: str = ""
    icon: str = ""
    views: list[ViewSpec] = field(default_factory=list)
    pipeline: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "icon": self.icon,
            "views": [v.to_dict() for v in self.views],
            "pipeline": list(self.pipeline),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DashboardManifest:
        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            icon=data.get("icon", ""),
            views=[ViewSpec.from_dict(v) for v in data.get("views", [])],
            pipeline=list(data.get("pipeline", []) or []),
        )

    @classmethod
    def from_dashboard_config(
        cls, dashboard_id: str, dashboard_config: dict[str, Any] | None
    ) -> DashboardManifest:
        """Build a manifest from a tool's ``dashboard_config``. ``view_manifest``
        entries may be plain view-type strings or ``{type,label,config}`` dicts."""
        cfg = dashboard_config or {}
        views: list[ViewSpec] = []
        for v in cfg.get("view_manifest", []) or []:
            if isinstance(v, str):
                views.append(ViewSpec(type=v))
            elif isinstance(v, dict):
                views.append(ViewSpec.from_dict(v))
        return cls(
            id=dashboard_id,
            label=cfg.get("label", ""),
            icon=cfg.get("icon", ""),
            views=views,
            pipeline=list(cfg.get("pipeline", []) or []),
        )


def build_dashboard_open(
    *, hub_id: str, manifest: DashboardManifest, seed: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a ``dashboard_open`` WS payload (server -> client)."""
    return {
        "type": WS_DASHBOARD_OPEN,
        "hub_id": hub_id,
        "manifest": manifest.to_dict(),
        "seed": seed or {},
    }


def build_dashboard_status(*, hub_id: str, status: str, **extra: Any) -> dict[str, Any]:
    """Build a ``dashboard_status`` WS payload (server -> client)."""
    return {"type": WS_DASHBOARD_STATUS, "hub_id": hub_id, "status": status, **extra}


def build_dashboard_event(
    *, hub_id: str, event_type: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a generic ``dashboard_event`` WS payload (server -> client)."""
    return {
        "type": WS_DASHBOARD_EVENT,
        "hub_id": hub_id,
        "event_type": event_type,
        "payload": payload or {},
    }

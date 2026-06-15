"""SOPState — structured SOP runtime state.

Replaces loose keys in prior_context with a typed, serializable object.
Inherits FeedBase for dict protocol support and template feed integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich_python_utils.common_objects.feed_base import FeedBase
from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus


@dataclass
class SOPState(FeedBase):
    """Complete SOP runtime state."""

    sop_name: str = ""
    current_phase: str | None = None
    phase_status: PhaseStatus = PhaseStatus.IDLE
    completed_phases: list = field(default_factory=list)
    phase_outputs: dict = field(default_factory=dict)
    goto_counts: dict = field(default_factory=dict)
    tool_phase_map: dict = field(default_factory=dict)
    phase_required_tools: dict = field(default_factory=dict)
    phase_executed_tools: dict = field(default_factory=dict)
    yolo_mode: bool = False
    instance_id: str = ""
    sop_description: str = ""
    user_input_gate_passed: bool = False

    # Lifecycle: set when this SOP is moved off the active slot into the
    # suspended bag. "" while active; "paused" (ad-hoc diversion, LLM nudges
    # the user to resume) or "exited" (longer interruption, passive list).
    suspension_reason: str = ""
    suspended_at: str = ""

    sop: Any = field(default=None, repr=False)

    @property
    def suspension_label(self) -> str:
        """Human-readable suspension state for prompt rendering."""
        return {"paused": "Paused", "exited": "Exited"}.get(self.suspension_reason, "")

    @property
    def sop_status(self) -> str:
        """Human-readable status for the template."""
        n_completed = len(self.completed_phases)
        n_total = len(self.sop.phases) if self.sop and hasattr(self.sop, "phases") else 0
        phase_name = ""
        if self.sop and hasattr(self.sop, "phases") and self.current_phase:
            for p in self.sop.phases:
                if p.id == self.current_phase:
                    phase_name = p.name
                    break
        if self.phase_status == "completed":
            return f"Completed ({n_completed}/{n_total} phases done)"
        if self.current_phase and phase_name:
            return f"Phase {self.current_phase} of {n_total}: {phase_name} ({self.phase_status})"
        if self.current_phase:
            return f"Phase {self.current_phase} of {n_total} ({self.phase_status})"
        return f"{n_completed}/{n_total} phases completed"

    def to_feed(self) -> dict[str, Any]:
        """Template-visible keys only. Excludes sop (non-serializable)."""
        return {
            "sop_description": self.sop_description,
            "sop_status": self.sop_status,
            "current_phase": self.current_phase,
            "phase_status": self.phase_status,
            "sop_name": self.sop_name,
            "sop_instance_id": self.instance_id,
            "sop_yolo_mode": self.yolo_mode,
            "completed_phases": self.completed_phases,
            "phase_outputs": self.phase_outputs,
            "tool_phase_map": self.tool_phase_map,
            "goto_counts": self.goto_counts,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot — excludes sop (reload by name on resume)."""
        d = super().to_dict()
        d.pop("sop", None)
        d["completed_phases"] = [
            r.to_dict() if hasattr(r, "to_dict") and callable(r.to_dict)
            else ({"phase": r.phase} if hasattr(r, "phase") else str(r))
            for r in self.completed_phases
        ]
        d["phase_required_tools"] = {
            k: sorted(v) for k, v in self.phase_required_tools.items()
        }
        d["phase_executed_tools"] = {
            k: sorted(v) for k, v in self.phase_executed_tools.items()
        }
        return d

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

    def __post_init__(self) -> None:
        # Runtime invariant: phase_required_tools / phase_executed_tools are
        # dict[str, set[str]]. build_sop_state() and _check_phase_completion()
        # populate them with sets, but to_dict() serializes each set as a sorted
        # list (JSON has no sets) and from_dict() restores them verbatim as
        # lists. Coerce back to sets on construction so the invariant holds on
        # BOTH fresh build and resume-from-disk — otherwise the subset guards in
        # _check_phase_completion (`required <= executed`) become `list <= set`
        # (TypeError) and `.add()` on a restored list fails, breaking every
        # resumed SOP that carries required/executed tools.
        self.phase_required_tools = {
            k: set(v) for k, v in self.phase_required_tools.items()
        }
        self.phase_executed_tools = {
            k: set(v) for k, v in self.phase_executed_tools.items()
        }

    @property
    def suspension_label(self) -> str:
        """Human-readable suspension state for prompt rendering."""
        return {"paused": "Paused", "exited": "Exited"}.get(self.suspension_reason, "")

    @property
    def sop_status(self) -> str:
        """Human-readable status for the template."""
        n_completed = len(self.completed_phases)
        n_total = (
            len(self.sop.phases) if self.sop and hasattr(self.sop, "phases") else 0
        )
        phase_name = ""
        if self.sop and hasattr(self.sop, "phases") and self.current_phase:
            for p in self.sop.phases:
                if p.id == self.current_phase:
                    phase_name = p.name
                    break
        if self.phase_status == "completed":
            return f"Completed ({n_completed}/{n_total} phases done)"
        # Surface a status suffix only for non-normal states (e.g. error); the
        # active phase is ~always "running" (noise) and "paused" is shown via
        # suspension_label elsewhere.
        _suffix = (
            f" — {self.phase_status}"
            if self.phase_status not in (PhaseStatus.RUNNING, PhaseStatus.IDLE)
            else ""
        )
        if self.current_phase and phase_name:
            return f"Current Phase ({self.current_phase} of {n_total}): {phase_name}{_suffix}"
        if self.current_phase:
            return f"Current Phase ({self.current_phase} of {n_total}){_suffix}"
        return f"{n_completed}/{n_total} phases completed"

    @property
    def sop_outline(self) -> str:
        """Ordered phase outline for the prompt: every phase in authored order,
        with a check mark for completed phases and an arrow for the current one.
        Empty when no SOP graph is attached (byte-identical for non-SOP paths).
        No hard next-phase annotation -- non-linear goto SOPs have no single
        linear next; the current marker plus the ordered list convey it."""
        if not (self.sop and hasattr(self.sop, "phases") and self.sop.phases):
            return ""
        completed_ids = set()
        for r in self.completed_phases:
            pid = getattr(r, "phase", None)
            if pid is None and isinstance(r, dict):
                pid = r.get("phase")
            if pid is not None:
                completed_ids.add(pid)
        lines = []
        for p in self.sop.phases:
            if p.id == self.current_phase:
                marker = " \u25b6 (current)"
            elif p.id in completed_ids:
                marker = " \u2713 (done)"
            else:
                marker = ""
            lines.append(f"{p.id} {p.name}{marker}")
        return "\n".join(lines)

    def completed_phase_ids(self) -> list[str]:
        """Normalized ids of completed phases.

        ``completed_phases`` entries may be phase-record objects (carrying a
        ``.phase`` attr) or bare id strings. This is the single source of truth
        for that id list — used by the phase-completion advancer AND the async
        dispatch writer's forward-only guard so the two can never drift (the
        normalization was previously duplicated inline at four call sites).
        """
        return [
            r.phase if hasattr(r, "phase") else str(r) for r in self.completed_phases
        ]

    def to_feed(self) -> dict[str, Any]:
        """Template-visible keys only. Excludes sop (non-serializable)."""
        return {
            "sop_description": self.sop_description,
            "sop_status": self.sop_status,
            "sop_outline": self.sop_outline,
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
            r.to_dict()
            if hasattr(r, "to_dict") and callable(r.to_dict)
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

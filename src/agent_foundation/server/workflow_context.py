
# pyre-strict
"""Workflow context — session-level workflow state for prompt injection."""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# Regex to parse phase definitions from workflow_description.
# Matches: **Phase <id> — <name>**: (with flexible whitespace and dash variants)
# Example: **Phase 1 — Codebase Investigation**: ...
_WORKFLOW_DESC_PHASE_RE = re.compile(
    r"\*\*\s*Phase\s+(\w+)\s*[—–\-]+\s*(.+?)\s*\*\*\s*:",
)

# Strategy -> _variables/ filename mapping.
# Currently only "default" exists. Strategy-specific descriptions
# will be handled as a separate concept downstream.
STRATEGY_FILE_MAP: dict[str, str] = {
    "default": "default",
}


def load_workflow_description(strategy: str, templates_dir: str = "") -> str:
    """Load a versioned workflow description from _variables/ files.

    Follows the same pattern as PTI's _load_analysis_request_template().
    If the file doesn't exist, returns an empty string (template renders empty).
    """
    filename = STRATEGY_FILE_MAP.get(strategy, "default")
    rel_parts = ("conversation", "main", "_variables", "workflow_description", f"{filename}.jinja2")

    # Try importlib.resources first (works with Buck link-trees)
    if not templates_dir:
        try:
            from importlib import resources

            # TODO: migrate prompt_templates resource package
            pkg = resources.files("rankevolve.src.resources.prompt_templates")
            resource = pkg.joinpath(*rel_parts)
            return resource.read_text(encoding="utf-8")
        except Exception:
            pass

    # Fallback: filesystem path
    if not templates_dir:
        templates_dir = str(
            os.path.join(os.path.dirname(__file__), "..", "resources", "prompt_templates")
        )
    var_path = os.path.join(templates_dir, *rel_parts)
    if os.path.isfile(var_path):
        with open(var_path) as f:
            return f.read()
    logger.warning("Workflow description file not found: %s", var_path)
    return ""


@dataclass
class WorkflowPhaseRecord:
    """Record of a completed workflow phase."""

    phase: str  # "understand_codebase", "research_propose", "task"
    status: str = "completed"  # "completed" | "error" | "cancelled"
    summary: str = ""
    workspace_path: str = ""
    task_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowPhaseRecord:
        return cls(
            phase=data["phase"],
            status=data.get("status", "completed"),
            summary=data.get("summary", ""),
            workspace_path=data.get("workspace_path", ""),
            task_id=data.get("task_id", ""),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class WorkflowContext:
    """Session-level workflow state — tracked across turns, persisted, injected into prompts.

    Supports two modes for session_context values:
    - Full string: Dynamic values computed at access time (workflow_status via to_status_text())
    - Versioned variable: workflow_description loaded from _variables/ at strategy selection time
    """

    # Strategy selection (determines which workflow_description version to load)
    strategy: str = "default"
    workflow_description: str = ""

    # Dynamic flow state
    current_phase: str = "idle"
    phase_status: str = "idle"  # "idle" | "running" | "completed" | "error"
    completed_phases: list[WorkflowPhaseRecord] = field(default_factory=list)
    active_task_summary: str = ""
    active_workspace: str = ""
    iteration_count: int = 0

    # SOP phase output tracking — tool executors write here when phase outputs
    # are produced (e.g., workflow_target_path, strategy). Used by the
    # StateGraphTracker to determine phase completion.
    phase_outputs: dict[str, Any] = field(default_factory=dict)

    # Tool name → SOP phase ID mapping, extracted from the SOP.
    # Populated by the conversational inferencer when the SOP is loaded.
    # Used by tool executors to determine which SOP phase a tool belongs to.
    tool_phase_map: dict[str, str] = field(default_factory=dict)

    # Optional StateGraphTracker for SOP-driven state management
    state_tracker: Any = None  # StateGraphTracker | None

    def __post_init__(self) -> None:
        """Load default workflow description if not already set."""
        if not self.workflow_description:
            self.workflow_description = load_workflow_description(self.strategy)

    def set_strategy(self, strategy: str, templates_dir: str = "") -> None:
        """Set evolution strategy and load the corresponding workflow description."""
        self.strategy = strategy
        self.workflow_description = load_workflow_description(strategy, templates_dir)

    @property
    def phase_names(self) -> dict[str, str]:
        """Parse phase ID → name mapping from workflow_description.

        The workflow_description is the single source of truth for phase
        definitions. Phase entries are matched by the pattern:
            **Phase <id> — <name>**:
        e.g., **Phase 1 — Codebase Investigation**: ...

        Returns:
            Dict mapping phase ID strings to display names,
            e.g. {"0": "Setup", "1": "Codebase Investigation", ...}
        """
        if not self.workflow_description:
            return {}
        return {
            m.group(1): m.group(2).strip()
            for m in _WORKFLOW_DESC_PHASE_RE.finditer(self.workflow_description)
        }

    def start_phase(self, phase: str, summary: str = "") -> None:
        """Start a phase — updates both legacy fields and state_tracker."""
        self.current_phase = phase
        self.phase_status = "running"
        self.active_task_summary = summary[:80] if summary else ""
        if self.state_tracker is not None:
            self.state_tracker.start(phase)

    def complete_phase(
        self,
        phase: str,
        summary: str = "",
        workspace_path: str = "",
        task_id: str = "",
        **outputs,
    ) -> None:
        """Complete a phase — updates legacy fields, state_tracker, and records outputs."""
        self.phase_status = "completed"
        self.completed_phases.append(
            WorkflowPhaseRecord(
                phase=phase,
                status="completed",
                summary=summary[:80] if summary else "",
                workspace_path=workspace_path,
                task_id=task_id,
            )
        )
        self.active_task_summary = ""
        if self.state_tracker is not None:
            self.state_tracker.complete(phase, **outputs)

    def fail_phase(
        self, phase: str, error: str = "", task_id: str = ""
    ) -> None:
        """Fail a phase — updates legacy fields and state_tracker."""
        self.phase_status = "error"
        self.completed_phases.append(
            WorkflowPhaseRecord(
                phase=phase,
                status="error",
                summary=error[:80] if error else "",
                task_id=task_id,
            )
        )
        self.active_task_summary = ""
        if self.state_tracker is not None:
            self.state_tracker.fail(phase, error)

    def to_status_text(self, phase_names: dict[str, str] | None = None) -> str:
        """Render human-readable status for {{ workflow_status }} in templates.

        Renders completed phases first (with their outputs), then the current
        phase. This structure matches the SOP phase ordering and is easy for
        the LLM to understand.

        Args:
            phase_names: Optional override for phase ID → display name mapping.
                If None, uses self.phase_names (parsed from workflow_description).
        """
        if (
            self.current_phase == "idle"
            and self.phase_status == "idle"
            and not self.completed_phases
        ):
            return "No workflow actions taken yet."
        pn = phase_names if phase_names is not None else self.phase_names
        lines: list[str] = []

        # 1. Completed phases first (with their outputs)
        if self.completed_phases:
            lines.append("Completed phases:")
            for rec in self.completed_phases:
                rec_name = pn.get(rec.phase, rec.phase)
                status_label = "completed" if rec.status == "completed" else rec.status
                lines.append(f"  - Phase {rec.phase} — {rec_name} ({status_label}): {rec.summary}")
            # Show accumulated phase outputs as key-value pairs
            if self.phase_outputs:
                for key, value in self.phase_outputs.items():
                    lines.append(f"    {key}: {value}")
            lines.append("")

        # 2. Current phase (skip if idle — all info is in completed phases above)
        if self.current_phase != "idle":
            cur_name = pn.get(self.current_phase, self.current_phase)
            lines.append(f"Current phase: Phase {self.current_phase} — {cur_name} ({self.phase_status})")
            if self.active_task_summary:
                lines.append(f"  Active task: {self.active_task_summary}")
            if self.active_workspace:
                lines.append(f"  Workspace: {self.active_workspace}")

        if self.iteration_count > 0:
            lines.append(f"Evolve iterations: {self.iteration_count}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "workflow_description": self.workflow_description,
            "current_phase": self.current_phase,
            "phase_status": self.phase_status,
            "completed_phases": [r.to_dict() for r in self.completed_phases],
            "active_task_summary": self.active_task_summary,
            "active_workspace": self.active_workspace,
            "iteration_count": self.iteration_count,
            "phase_outputs": dict(self.phase_outputs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowContext:
        return cls(
            strategy=data.get("strategy", "default"),
            workflow_description=data.get("workflow_description", ""),
            current_phase=data.get("current_phase", "idle"),
            phase_status=data.get("phase_status", "idle"),
            completed_phases=[
                WorkflowPhaseRecord.from_dict(r)
                for r in data.get("completed_phases", [])
            ],
            active_task_summary=data.get("active_task_summary", ""),
            active_workspace=data.get("active_workspace", ""),
            iteration_count=data.get("iteration_count", 0),
            phase_outputs=data.get("phase_outputs", {}),
        )

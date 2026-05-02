
# pyre-strict
"""Workflow context — session-level workflow state for prompt injection."""
from __future__ import annotations

import dataclasses
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from agent_foundation.common.workflow_constants import _WORKFLOW_DESC_PHASE_RE  # noqa: E402

logger: logging.Logger = logging.getLogger(__name__)

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

    # Task queue — tracks async tool invocations.
    # Each entry: {task_id, tool_name, request, title, args, status, created_at,
    #              workspace, result_summary, hypothesis_id, phase}
    # status: "queued" | "running" | "completed" | "error"
    task_queue: list[dict[str, Any]] = field(default_factory=list)
    max_parallel_tasks: int = 1

    # Active multi-task hub ID — when set, /task commands are routed to this hub's queue
    active_multi_task_id: str | None = None

    # Multi-task hubs that have been explicitly closed. Lifecycle marker so we can
    # distinguish "all current runs complete" (hub still open for more) from
    # "user explicitly closed the hub". Used by resume to decide whether to
    # re-hydrate the hub UI.
    closed_multi_task_ids: set[str] = field(default_factory=set)

    # Tool names that bypass the parallelism cap — these entries do NOT count
    # against ``max_parallel_tasks`` and a queued entry with one of these names
    # is allowed to start even while the cap is reached.
    bypass_cap_tools: set[str] = field(default_factory=set)

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
        if outputs:
            self.phase_outputs.update(outputs)
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

    # -- Task queue methods --------------------------------------------------

    def enqueue_task(
        self,
        task_id: str,
        tool_name: str,
        request: str,
        title: str,
        args: dict[str, Any] | None = None,
        hypothesis_id: str = "",
        phase: str = "",
    ) -> dict[str, Any]:
        """Add a task to the queue with status='queued'."""
        from datetime import datetime, timezone

        entry: dict[str, Any] = {
            "task_id": task_id,
            "tool_name": tool_name,
            "request": request,
            "title": title,
            "args": args or {},
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workspace": "",
            "result_summary": "",
            "hypothesis_id": hypothesis_id,
            "phase": phase,
        }
        self.task_queue.append(entry)
        return entry

    def get_next_runnable(self) -> dict[str, Any] | None:
        """Return the first queued task if a slot is available.

        Queue-isolation rule: entries whose ``tool_name`` is listed in
        ``bypass_cap_tools`` do NOT count against the concurrency cap,
        AND a queued bypass-cap entry is allowed to start even while
        the cap is reached. Without this rule, a single long-lived
        bypass task would block all regular work for its duration.
        """
        for entry in self.task_queue:
            if entry["status"] != "queued":
                continue
            if entry.get("tool_name") in self.bypass_cap_tools:
                return entry
            if self.running_count >= self.max_parallel_tasks:
                # Regular tasks are at cap — but a SUBSEQUENT bypass-cap
                # entry is still eligible. Keep scanning.
                continue
            return entry
        return None

    @property
    def running_count(self) -> int:
        """Count of running tasks that respect the parallelism cap.

        Excludes entries whose ``tool_name`` is in ``bypass_cap_tools``
        — they bypass the cap (see ``get_next_runnable``)."""
        return sum(
            1
            for e in self.task_queue
            if e["status"] == "running"
            and e.get("tool_name") not in self.bypass_cap_tools
        )

    def get_entry(self, task_id: str) -> dict[str, Any] | None:
        """Return the queue entry dict for task_id, or None if not found."""
        for e in self.task_queue:
            if e["task_id"] == task_id:
                return e
        return None

    def update_entry(self, task_id: str, **fields: Any) -> None:
        """Update arbitrary fields on a task_queue entry."""
        entry = self.get_entry(task_id)
        if entry is not None:
            entry.update(fields)

    def mark_running(self, task_id: str, workspace: str | None = None) -> None:
        """Mark entry running. workspace=None preserves the existing workspace
        value (don't wipe). Pass an explicit string (including empty "") to set."""
        for e in self.task_queue:
            if e["task_id"] == task_id:
                e["status"] = "running"
                if workspace is not None:
                    e["workspace"] = workspace
                break

    def mark_completed(self, task_id: str, summary: str = "") -> None:
        for e in self.task_queue:
            if e["task_id"] == task_id:
                e["status"] = "completed"
                e["result_summary"] = summary
                break

    def mark_error(self, task_id: str, error: str = "") -> None:
        for e in self.task_queue:
            if e["task_id"] == task_id:
                e["status"] = "error"
                e["result_summary"] = error
                break

    def close_multi_task(self, multi_task_id: str) -> None:
        """Explicitly close a multi-task hub. Records the close in
        closed_multi_task_ids and clears active_multi_task_id if it matches."""
        self.closed_multi_task_ids.add(multi_task_id)
        if self.active_multi_task_id == multi_task_id:
            self.active_multi_task_id = None

    def is_phase_complete(self, phase: str) -> bool:
        """True if ALL queued tasks for this phase are completed or errored."""
        phase_tasks = [e for e in self.task_queue if e["phase"] == phase]
        if not phase_tasks:
            return True  # no tasks -> vacuously complete
        return all(e["status"] in ("completed", "error") for e in phase_tasks)

    def get_queue_summary(self, phase: str = "") -> str:
        """Human-readable queue status like '3/8 completed, 1 running, 4 queued'."""
        tasks = [e for e in self.task_queue if not phase or e["phase"] == phase]
        if not tasks:
            return ""
        by_status: dict[str, int] = {}
        for e in tasks:
            by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        total = len(tasks)
        parts = []
        if by_status.get("completed", 0):
            parts.append(f"{by_status['completed']}/{total} completed")
        if by_status.get("running", 0):
            parts.append(f"{by_status['running']} running")
        if by_status.get("queued", 0):
            parts.append(f"{by_status['queued']} queued")
        if by_status.get("error", 0):
            parts.append(f"{by_status['error']} failed")
        return ", ".join(parts) if parts else ""

    def to_status_text(
        self,
        phase_names: dict[str, str] | None = None,
        sop_obj: Any = None,
    ) -> str:
        """Render human-readable status for {{ workflow_status }} in templates.

        Renders completed phases first (with their outputs), then the current
        phase. This structure matches the SOP phase ordering and is easy for
        the LLM to understand.

        Args:
            phase_names: Optional override for phase ID → display name mapping.
                If None, uses self.phase_names (parsed from workflow_description).
            sop_obj: Optional ``SOP`` instance. When provided, an extra
                "Next pending" line is appended for the next phase whose
                dependencies are all satisfied. If that next pending phase
                carries the ``"requires confirmation"`` directive, an
                explicit ``REQUIRES USER CONFIRMATION`` instruction is also
                rendered — addresses the structural gap where gate phases
                never become ``current_phase`` (they have no tool to fire
                ``start_phase()``) and are otherwise invisible in the status
                block.
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
            lines.append(
                f"Current phase: Phase {self.current_phase} — {cur_name} ({self.phase_status})"
            )
            if self.active_task_summary:
                lines.append(f"  Active task: {self.active_task_summary}")
            if self.active_workspace:
                lines.append(f"  Task workspace path: {self.active_workspace}")

        # Next pending phase (esp. gate phases that have no tool to
        # become current_phase). Only when sop_obj is supplied — preserves
        # backward compat with all existing callers that pass no SOP.
        if sop_obj is not None:
            try:
                completed_ids = {r.phase for r in self.completed_phases}
                next_pending = sop_obj.get_next_pending_phase(completed_ids)
                # Skip when next pending IS current_phase (already shown above)
                # OR when current phase is mid-run (LLM is in the middle of it).
                if (
                    next_pending is not None
                    and next_pending.id != self.current_phase
                    and self.phase_status != "running"
                ):
                    np_name = next_pending.name or pn.get(
                        next_pending.id, next_pending.id
                    )
                    lines.append(f"Next pending: Phase {next_pending.id} — {np_name}")
                    requires_confirmation = "requires confirmation" in (
                        getattr(next_pending, "directives", []) or []
                    )
                    if requires_confirmation:
                        lines.append(
                            "  REQUIRES USER CONFIRMATION — emit a "
                            "`confirmation` conversation tool with the "
                            "`view` parameter pointing to the prior "
                            "phase's output before proceeding."
                        )
            except Exception as e:
                logger.warning("to_status_text next-pending render failed: %s", e)

        if self.iteration_count > 0:
            lines.append(f"Iterations: {self.iteration_count}")

        # Task queue status
        if self.task_queue:
            summary = self.get_queue_summary()
            if summary:
                lines.append(f"Task queue: {summary}")
            running = [e for e in self.task_queue if e["status"] == "running"]
            for e in running:
                label = e.get("hypothesis_id") or e.get("task_id", "")
                lines.append(f"  Running: {label} — {e.get('title', '')}")
            next_q = next(
                (e for e in self.task_queue if e["status"] == "queued"), None
            )
            if next_q:
                label = next_q.get("hypothesis_id") or next_q.get("task_id", "")
                lines.append(f"  Next: {label} — {next_q.get('title', '')}")

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
            "task_queue": list(self.task_queue),
            "max_parallel_tasks": self.max_parallel_tasks,
            "active_multi_task_id": self.active_multi_task_id,
            "closed_multi_task_ids": sorted(self.closed_multi_task_ids),
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
            task_queue=data.get("task_queue", []),
            max_parallel_tasks=data.get("max_parallel_tasks", 1),
            active_multi_task_id=data.get("active_multi_task_id"),
            closed_multi_task_ids=set(data.get("closed_multi_task_ids", [])),
        )

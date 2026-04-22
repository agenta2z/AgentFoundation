"""StateManager — read/write EmployeeRuntimeState from state.json."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_foundation.employees.models.enums import EmployeeStatus

logger = logging.getLogger(__name__)


@dataclass
class EmployeeRuntimeState:
    """Ephemeral runtime state for an AIEmployee.

    Stored in state.json — NOT in employee.yaml.
    Frequently updated; not version-controlled (gitignored).

    Maps to the OpenStartup dashboard's employee metrics/status display.
    """
    employee_id: str
    status: EmployeeStatus = EmployeeStatus.active
    current_task_id: str | None = None
    task_queue: list[dict[str, Any]] = field(default_factory=list)
    active_correspondence: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    pending_reason: dict[str, Any] | None = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dict(cls, data: dict) -> "EmployeeRuntimeState":
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        elif last_updated is None:
            last_updated = datetime.now(timezone.utc)

        try:
            status = EmployeeStatus(data.get("status", "active"))
        except ValueError:
            status = EmployeeStatus.active

        return cls(
            employee_id=data["employee_id"],
            status=status,
            current_task_id=data.get("current_task_id"),
            task_queue=data.get("task_queue", []),
            active_correspondence=data.get("active_correspondence", []),
            metrics=data.get("metrics", {}),
            pending_reason=data.get("pending_reason"),
            last_updated=last_updated,
        )

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "task_queue": self.task_queue,
            "active_correspondence": self.active_correspondence,
            "metrics": self.metrics,
            "pending_reason": self.pending_reason,
            "last_updated": self.last_updated.isoformat(),
        }


class StateManager:
    """Thread-safe reader/writer for EmployeeRuntimeState (state.json).

    Usage::

        sm = StateManager(data_dir=Path("_data/employees"))
        state = sm.load("alice_pm")
        sm.update_status("alice_pm", EmployeeStatus.active)
        sm.record_metric("alice_pm", "issues_resolved", 42)
    """

    _STATE_FILENAME = "state.json"

    def __init__(self, employees_dir: Path) -> None:
        self._employees_dir = Path(employees_dir)
        self._lock = threading.Lock()

    def _state_path(self, employee_id: str) -> Path:
        return self._employees_dir / employee_id / self._STATE_FILENAME

    def load(self, employee_id: str) -> EmployeeRuntimeState:
        """Load state.json for an employee. Returns a default state if not found."""
        path = self._state_path(employee_id)
        if not path.exists():
            return EmployeeRuntimeState(employee_id=employee_id)
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            return EmployeeRuntimeState.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("[StateManager] Failed to load state for '%s': %s", employee_id, e)
            return EmployeeRuntimeState(employee_id=employee_id)

    def save(self, state: EmployeeRuntimeState) -> None:
        """Write state to state.json."""
        path = self._state_path(state.employee_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        state.last_updated = datetime.now(timezone.utc)
        with self._lock:
            with path.open("w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

    def update_status(
        self,
        employee_id: str,
        status: EmployeeStatus,
        pending_reason: dict | None = None,
    ) -> None:
        """Update employee status and optionally set pending_reason."""
        state = self.load(employee_id)
        state.status = status
        state.pending_reason = pending_reason
        self.save(state)

    def record_metric(self, employee_id: str, key: str, value: Any) -> None:
        """Update a single metric value."""
        state = self.load(employee_id)
        state.metrics[key] = value
        self.save(state)

    def increment_metric(self, employee_id: str, key: str, delta: int = 1) -> None:
        """Increment a numeric metric."""
        state = self.load(employee_id)
        state.metrics[key] = int(state.metrics.get(key, 0)) + delta
        self.save(state)

    def set_current_task(self, employee_id: str, task_id: str | None) -> None:
        """Update the current task id."""
        state = self.load(employee_id)
        state.current_task_id = task_id
        self.save(state)

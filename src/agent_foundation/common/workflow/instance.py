"""WorkflowInstance — a live workflow execution. One per enter_workflow call."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from rich_python_utils.common_objects.workflow.stategraph import StateGraphTracker
from rich_python_utils.common_objects.workflow.workgraph import WorkGraph


@dataclass
class WorkflowInstance:
    """Per-session runtime state for one workflow execution.

    Owns the WorkGraph + StateGraphTracker pair. Runtime objects are
    reconstructed on resume from workspace state.
    """

    instance_id: str
    definition_id: str
    status: Literal["active", "suspended", "completed", "aborted"] = "active"
    yolo_mode: bool = False
    workspace: Path = field(default_factory=lambda: Path("."))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: datetime = field(default_factory=datetime.utcnow)

    graph: Optional[WorkGraph] = field(default=None, repr=False)
    tracker: Optional[StateGraphTracker] = field(default=None, repr=False)
    _graph_task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_persistent_dict(self) -> dict[str, Any]:
        """Serializable shape for session JSON. Graph state lives in workspace files."""
        return {
            "instance_id": self.instance_id,
            "definition_id": self.definition_id,
            "status": self.status,
            "yolo_mode": self.yolo_mode,
            "workspace": str(self.workspace),
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "tracker_state": self.tracker.to_dict() if self.tracker else None,
        }

    @classmethod
    def from_persistent_dict(cls, data: dict[str, Any]) -> WorkflowInstance:
        """Reconstruct from session JSON. WorkGraph is rebuilt lazily on resume."""
        return cls(
            instance_id=data["instance_id"],
            definition_id=data["definition_id"],
            status=data.get("status", "suspended"),
            yolo_mode=data.get("yolo_mode", False),
            workspace=Path(data.get("workspace", ".")),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active_at=datetime.fromisoformat(data["last_active_at"]),
        )

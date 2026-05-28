"""WorkflowManager — per-session orchestrator for workflow lifecycle.

Handles enter/exit/resume of workflow instances. Delegates execution
to WorkGraph. Provides prompt rendering sections.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOPManager,
)

from agent_foundation.common.workflow.definition import WorkflowDefinition
from agent_foundation.common.workflow.instance import WorkflowInstance
from agent_foundation.common.workflow.registry import WorkflowRegistry
from agent_foundation.common.workflow.sop_workgraph import build_sop_workgraph

logger = logging.getLogger(__name__)


class WorkflowInstanceNotFound(KeyError):
    pass


class WorkflowInstanceNotResumable(ValueError):
    pass


class WorkflowManager:
    """Per-session orchestrator for enter/exit/resume/list of workflow instances."""

    def __init__(
        self,
        registry: WorkflowRegistry,
        session_workspace: Path,
        inferencer_factory: Optional[Callable] = None,
        graph_reporter: Any = None,
    ):
        self.registry = registry
        self.session_workspace = session_workspace
        self.inferencer_factory = inferencer_factory
        self.graph_reporter = graph_reporter
        self.active_instances: dict[str, WorkflowInstance] = {}
        self.focused_instance_id: Optional[str] = None

    async def enter_workflow(
        self, definition_id: str, *, yolo_mode: bool = False
    ) -> str:
        """Create and start a new WorkflowInstance. Returns instance_id."""
        definition = self.registry.get(definition_id)
        instance_id = uuid.uuid4().hex[:8]

        instance = WorkflowInstance(
            instance_id=instance_id,
            definition_id=definition_id,
            yolo_mode=yolo_mode,
            workspace=self.session_workspace / "workflows" / definition_id / instance_id,
            created_at=datetime.now(UTC),
            last_active_at=datetime.now(UTC),
        )
        instance.workspace.mkdir(parents=True, exist_ok=True)

        graph, tracker = build_sop_workgraph(
            sop=definition.sop,
            inferencer_factory=self.inferencer_factory,
            workspace=instance.workspace,
            yolo_mode=yolo_mode,
            graph_reporter=self.graph_reporter,
            max_goto_iterations=definition.frontmatter.get("max_goto_iterations", 10),
            max_total_nodes=definition.frontmatter.get("max_total_nodes", 500),
        )
        instance.graph = graph
        instance.tracker = tracker
        instance._graph_task = asyncio.create_task(graph.arun())

        self.active_instances[instance_id] = instance
        self.focused_instance_id = instance_id
        logger.info("Workflow %s entered: instance %s", definition_id, instance_id)
        return instance_id

    async def exit_workflow(self, instance_id: Optional[str] = None) -> None:
        """Suspend the focused (or specified) instance."""
        instance = self._resolve_instance(instance_id)
        instance.status = "suspended"
        instance.last_active_at = datetime.now(UTC)
        if instance._graph_task and not instance._graph_task.done():
            instance._graph_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await instance._graph_task
        if self.focused_instance_id == instance.instance_id:
            self.focused_instance_id = None
        logger.info("Workflow instance %s suspended", instance.instance_id)

    async def resume_workflow(self, instance_id: str) -> None:
        """Resume a suspended instance."""
        instance = self.active_instances.get(instance_id)
        if not instance:
            raise WorkflowInstanceNotFound(instance_id)
        if instance.status != "suspended":
            raise WorkflowInstanceNotResumable(
                f"Instance {instance_id} is {instance.status}, not suspended"
            )

        definition = self.registry.get(instance.definition_id)
        graph, tracker = build_sop_workgraph(
            sop=definition.sop,
            inferencer_factory=self.inferencer_factory,
            workspace=instance.workspace,
            yolo_mode=instance.yolo_mode,
            graph_reporter=self.graph_reporter,
        )

        if instance.tracker:
            saved = instance.tracker.to_dict()
            tracker.current_state = saved.get("current_state")
            tracker.state_status = saved.get("state_status", "idle")
            tracker.completed_states = saved.get("completed_states", [])
            tracker.state_outputs = saved.get("state_outputs", {})
            tracker.goto_counts = saved.get("goto_counts", {})
            tracker.foreach_state = saved.get("foreach_state", {})

        instance.graph = graph
        instance.tracker = tracker
        instance.status = "active"
        instance.last_active_at = datetime.now(UTC)
        instance._graph_task = asyncio.create_task(graph.arun())
        self.focused_instance_id = instance_id
        logger.info("Workflow instance %s resumed", instance_id)

    def get_focused_instance(self) -> Optional[WorkflowInstance]:
        if self.focused_instance_id:
            return self.active_instances.get(self.focused_instance_id)
        return None

    def list_instances(self) -> list[WorkflowInstance]:
        return list(self.active_instances.values())

    def render_prompt_sections(self) -> dict[str, str]:
        """Render prompt sections for SOP-aware prompts."""
        sections: dict[str, str] = {}

        # Available SOPs (from SOPRegistry if available, else legacy WorkflowRegistry)
        try:
            from agent_foundation.resources.sops.registry import format_all_sops, load_all_sops
            sops = load_all_sops(extra_dirs=getattr(self, "_extra_sop_dirs", None))
            if sops:
                sections["available_sops"] = format_all_sops(sops)
        except ImportError:
            pass

        if "available_sops" not in sections:
            available = self.registry.list_all()
            if available:
                lines = []
                for wf in available:
                    desc = wf.description[:120] if wf.description else ""
                    lines.append(f"- **{wf.name}** (`{wf.workflow_id}`): {desc}")
                sections["available_workflows"] = "\n".join(lines)

        # Active SOPs / instances
        instances = self.list_instances()
        if instances:
            lines = []
            for inst in instances:
                phase = ""
                if inst.tracker and inst.tracker.current_state:
                    phase = f" (phase {inst.tracker.current_state})"
                lines.append(
                    f"- `{inst.instance_id}` [{inst.definition_id}] "
                    f"status={inst.status}{phase}"
                )
            sections["active_sops"] = "\n".join(lines)

        # Focused instance context
        focused = self.get_focused_instance()
        if focused:
            definition = self.registry.get(focused.definition_id)
            sections["workflow_description"] = definition.raw_markdown
            if focused.tracker:
                sections["workflow_nextstep_guidance"] = SOPManager.render_guidance(
                    focused.tracker, definition.sop
                )
                sections["workflow_status"] = (
                    f"Instance {focused.instance_id} | "
                    f"Status: {focused.status} | "
                    f"Current: {focused.tracker.current_state or 'idle'}"
                )

        return sections

    def to_dict(self) -> dict[str, Any]:
        return {
            "focused_instance_id": self.focused_instance_id,
            "instances": {
                iid: inst.to_persistent_dict()
                for iid, inst in self.active_instances.items()
            },
        }

    def _resolve_instance(self, instance_id: Optional[str] = None) -> WorkflowInstance:
        iid = instance_id or self.focused_instance_id
        if not iid or iid not in self.active_instances:
            raise WorkflowInstanceNotFound(iid or "(no focused instance)")
        return self.active_instances[iid]

"""SOP tool executor — runs an SOP end-to-end via WorkflowManager.

Pipeline:
1. Parse arguments (workflow name, yolo flag, model, params)
2. Resolve workflow via WorkflowRegistry
3. Allocate workspace
4. Build inferencer_factory
5. Create WorkflowInstance + WorkGraph via WorkflowManager
6. Run WorkGraph
7. Return ToolExecutionResult with artifacts
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def execute(arguments: dict[str, Any], session_context: dict[str, Any]) -> Any:
    """Main entry point for the /sop tool."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
        ToolExecutionResult,
    )
    from agent_foundation.common.workflow.registry import WorkflowRegistry
    from agent_foundation.common.workflow.manager import WorkflowManager

    workflow_name = arguments.get("workflow", "")
    yolo_mode = arguments.get("yolo", False)
    model_override = arguments.get("model")
    params_json = arguments.get("params", "{}")
    max_concurrency = arguments.get("max_concurrency", 1)

    try:
        params = json.loads(params_json) if isinstance(params_json, str) else params_json
    except json.JSONDecodeError:
        params = {}

    # Resolve workflow
    registry = WorkflowRegistry()
    registry.load_all()

    try:
        definition = registry.get(workflow_name)
    except KeyError:
        available = [d.workflow_id for d in registry.list_all()]
        return ToolExecutionResult(
            result=f"Workflow '{workflow_name}' not found. Available: {available}",
        )

    # Allocate workspace
    session_root = session_context.get("session_root", "")
    if session_root:
        workspace = Path(session_root) / "workflows" / workflow_name
    else:
        workspace = Path.home() / ".agent_foundation" / "workflows" / workflow_name
    workspace.mkdir(parents=True, exist_ok=True)

    # Build a simple inferencer factory
    def inferencer_factory(**kwargs):
        """Create a ConversationalInferencer for a single phase."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
            ConversationalInferencer,
        )

        phase = kwargs.get("phase")
        phase_workspace = kwargs.get("workspace", workspace)

        base_inferencer = session_context.get("base_inferencer")
        if base_inferencer is None:
            logger.warning("No base_inferencer in session_context; SOP phase will fail")
            return None

        return ConversationalInferencer(
            base_inferencer=base_inferencer,
            tool_registry=session_context.get("tool_registry", {}),
            tool_executor=session_context.get("tool_executor"),
            prompt_renderer=session_context.get("prompt_renderer"),
            prior_context={
                **session_context.get("prior_context", {}),
                **params,
                "active_sop_phase": phase.id if phase else "",
                "active_sop_phase_name": phase.name if phase else "",
            },
            yolo_mode=yolo_mode,
        )

    # Create manager and enter workflow
    manager = WorkflowManager(
        registry=registry,
        session_workspace=workspace,
        inferencer_factory=inferencer_factory,
    )

    try:
        instance_id = await manager.enter_workflow(
            workflow_name, yolo_mode=yolo_mode,
        )
        instance = manager.active_instances[instance_id]

        if instance._graph_task:
            await instance._graph_task

        status = instance.tracker.status if instance.tracker else "unknown"
        completed = list(instance.tracker.completed_states) if instance.tracker else []

        return ToolExecutionResult(
            result=(
                f"SOP '{workflow_name}' completed. "
                f"Status: {status}. "
                f"Completed phases: {completed}. "
                f"Workspace: {workspace}"
            ),
            context_updates={
                "workflow_instance_id": instance_id,
                "workflow_status": status,
                "workspace_path": str(workspace),
            },
        )
    except Exception as e:
        logger.error("SOP execution failed: %s", e, exc_info=True)
        return ToolExecutionResult(
            result=f"SOP execution failed: {e}",
        )

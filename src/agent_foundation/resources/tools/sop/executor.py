"""SOP tool executor — runs an SOP end-to-end via WorkflowManager.

Pipeline:
1. Parse arguments (workflow name, yolo flag, model, params)
2. Resolve workflow via SOPRegistry (supports extra_dirs for OpenStartup SOPs)
3. Allocate workspace
4. Build inferencer_factory (from session_context OR from scratch)
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
    from agent_foundation.resources.sops.registry import load_all_sops, SOPNotFound
    from agent_foundation.common.workflow.manager import WorkflowManager
    from agent_foundation.common.workflow.registry import WorkflowRegistry

    workflow_name = arguments.get("workflow", "")
    yolo_mode = arguments.get("yolo", False)
    model_override = arguments.get("model")
    params_json = arguments.get("params", "{}")
    max_concurrency = arguments.get("max_concurrency", 1)

    try:
        params = json.loads(params_json) if isinstance(params_json, str) else params_json
    except json.JSONDecodeError:
        params = {}

    # Resolve workflow via SOPRegistry (includes extra_dirs from session_context)
    extra_sop_dirs = session_context.get("extra_sop_dirs", [])
    sops = load_all_sops(extra_dirs=extra_sop_dirs or None)

    if workflow_name not in sops:
        available = list(sops.keys())
        return ToolExecutionResult(
            result=f"SOP '{workflow_name}' not found. Available: {available}",
        )

    sop_info = sops[workflow_name]

    # Also build a WorkflowRegistry that knows about this SOP
    # (WorkflowManager requires a WorkflowRegistry)
    registry = WorkflowRegistry()
    registry.load_all()
    # Ensure the SOP is in the registry's definitions
    if workflow_name not in registry._definitions:
        from agent_foundation.common.workflow.definition import WorkflowDefinition
        registry._definitions[workflow_name] = WorkflowDefinition(
            workflow_id=sop_info.name,
            name=sop_info.display_name,
            description=sop_info.description,
            source_path=sop_info.body_path,
            sop=sop_info.sop,
            raw_markdown=sop_info.body,
            frontmatter=sop_info.config,
            available_tools=sop_info.requires_tools,
            keywords=sop_info.keywords,
            example_requests=sop_info.example_requests,
        )

    # Allocate workspace
    session_root = session_context.get("session_root", "")
    if session_root:
        workspace = Path(session_root) / "sops" / workflow_name
    else:
        workspace = Path.home() / ".agent_foundation" / "sops" / workflow_name
    workspace.mkdir(parents=True, exist_ok=True)

    # Build inferencer factory
    def inferencer_factory(**kwargs):
        """Create a ConversationalInferencer for a single SOP phase."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
            ConversationalInferencer,
        )

        phase = kwargs.get("phase")

        # Try session_context first (when called from ToolDispatcher)
        base_inferencer = session_context.get("base_inferencer")
        tool_registry = session_context.get("tool_registry", {})
        tool_executor = session_context.get("tool_executor")
        prompt_renderer = session_context.get("prompt_renderer")

        # Fallback: build a base inferencer from scratch
        if base_inferencer is None:
            base_inferencer = _build_base_inferencer(model_override, session_context)
            if base_inferencer is None:
                logger.error(
                    "Cannot create base_inferencer for SOP phase %s. "
                    "Provide base_inferencer in session_context or ensure "
                    "a CLI backend is available.",
                    phase.id if phase else "?",
                )
                return None

        # Fallback: load tool registry if not provided
        if not tool_registry:
            try:
                from agent_foundation.resources.tools.registry import load_all_tools
                tool_registry = load_all_tools(
                    extra_dirs=session_context.get("extra_tool_dirs")
                )
            except Exception as e:
                logger.warning("Failed to load tool registry: %s", e)

        return ConversationalInferencer(
            base_inferencer=base_inferencer,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            prompt_renderer=prompt_renderer,
            prior_context={
                **session_context.get("prior_context", {}),
                **params,
                "active_sop_phase": phase.id if phase else "",
                "active_sop_phase_name": phase.name if phase else "",
                "sop_name": workflow_name,
                "sop_instance_id": session_context.get("sop_instance_id", ""),
                "session_root_path": session_root or str(workspace),
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


def _build_base_inferencer(model_override: str | None, session_context: dict) -> Any:
    """Try to build a base inferencer from available CLI backends."""
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
            RovoDevCliInferencer,
        )
        model = model_override or "sonnet"
        return RovoDevCliInferencer(model_name=model, yolo=True)
    except ImportError:
        pass

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
            ClaudeCodeCliInferencer,
        )
        model = model_override or "opus"
        return ClaudeCodeCliInferencer(model_name=model)
    except ImportError:
        pass

    logger.error("No CLI backend available (rovodev or claude_code). Cannot create base_inferencer.")
    return None

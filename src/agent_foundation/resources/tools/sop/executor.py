"""SOP tool executor — initializes SOP state in the CI's sop_state.

The CI's own agentic loop handles execution: SOP guidance rotates in the
prompt as phases complete via _check_phase_completion().
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


async def execute(arguments: dict[str, Any], session_context: dict[str, Any]) -> Any:
    """Enter an SOP by loading it and returning SOPState via context_updates."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
        ToolExecutionResult,
    )
    from agent_foundation.common.workflow.sop_state import SOPState
    from agent_foundation.resources.sops.registry import load_all_sops
    from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus

    workflow_name = arguments.get("workflow", "")
    yolo_mode = arguments.get("yolo", False)

    extra_sop_dirs = session_context.get("extra_sop_dirs", [])
    sops = load_all_sops(extra_dirs=extra_sop_dirs or None)

    if workflow_name not in sops:
        available = list(sops.keys())
        return ToolExecutionResult(
            result=f"SOP '{workflow_name}' not found. Available: {available}",
        )

    sop_info = sops[workflow_name]
    sop = sop_info.sop
    initial_phase_id = sop.phases[0].id if sop.phases else None

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    instance_id = f"{workflow_name}__{ts}__{uuid4().hex[:8]}"

    sop_state = SOPState(
        sop_name=workflow_name,
        sop=sop,
        current_phase=initial_phase_id,
        phase_status=PhaseStatus.IDLE,
        tool_phase_map=sop.tool_to_phase_map,
        yolo_mode=yolo_mode,
        instance_id=instance_id,
        workflow_description=sop_info.body or sop_info.description,
    )

    desc_preview = (sop_info.description or "")[:80]
    return ToolExecutionResult(
        result=f"Entered SOP: {workflow_name} ({desc_preview}...)",
        context_updates={"sop_state": sop_state},
    )

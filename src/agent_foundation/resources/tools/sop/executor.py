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


def build_sop_state(
    name: str,
    *,
    yolo: bool = False,
    extra_sop_dirs: Any = None,
) -> tuple[Any, str | None]:
    """Load an SOP by name and construct its SOPState.

    Single source of truth for SOP entry — shared by the ``/sop`` command
    (``ConversationalInferencer._enter_sop``) and this tool executor (used by
    the CLI and any programmatic caller).

    Returns ``(sop_state, None)`` on success, or ``(None, error_message)`` if
    the SOP is not found.
    """
    from agent_foundation.common.workflow.sop_state import SOPState
    from agent_foundation.resources.sops.registry import load_all_sops
    from rich_python_utils.common_objects.workflow.common.phase_status import (
        PhaseStatus,
    )

    sops = load_all_sops(extra_dirs=extra_sop_dirs or None)
    if name not in sops:
        return None, f"SOP '{name}' not found. Available: {list(sops.keys())}"

    sop_info = sops[name]
    sop = sop_info.sop
    initial_phase_id = sop.phases[0].id if sop.phases else None

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    instance_id = f"{name}__{ts}__{uuid4().hex[:8]}"

    sop_state = SOPState(
        sop_name=name,
        sop=sop,
        current_phase=initial_phase_id,
        phase_status=PhaseStatus.IDLE,
        tool_phase_map=sop.tool_to_phase_map,
        phase_required_tools=sop.phase_required_tools,
        yolo_mode=yolo,
        instance_id=instance_id,
        sop_description=sop_info.description or sop_info.body,
    )
    return sop_state, None


async def execute(arguments: dict[str, Any], session_context: dict[str, Any]) -> Any:
    """Enter an SOP by loading it and returning SOPState via context_updates."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
        ToolExecutionResult,
    )

    workflow_name = arguments.get("workflow", "")
    yolo_mode = arguments.get("yolo", False)
    extra_sop_dirs = session_context.get("extra_sop_dirs", [])

    sop_state, error = build_sop_state(
        workflow_name, yolo=yolo_mode, extra_sop_dirs=extra_sop_dirs
    )
    if error:
        return ToolExecutionResult(result=error)

    desc_preview = (sop_state.sop_description or "")[:80]
    return ToolExecutionResult(
        result=f"Entered SOP: {workflow_name} ({desc_preview}...)",
        context_updates={"sop_state": sop_state},
    )

"""Factory for building a WorkGraph from an SOP definition.

Creates a WorkGraph + StateGraphTracker pair. The WorkGraph starts with
[__initial__] phase node(s) and grows dynamically via GraphExpansionResult
as each SOPWorkGraphNode completes and queries the tracker for next phases.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from rich_python_utils.common_objects.workflow.stategraph import (
    StateGraphTracker,
)
from rich_python_utils.common_objects.workflow.workgraph import WorkGraph
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOP,
)

from agent_foundation.common.workflow.sop_workgraph_node import SOPWorkGraphNode

logger = logging.getLogger(__name__)


class InvalidSOPError(ValueError):
    """Raised when an SOP definition is invalid."""


def build_sop_workgraph(
    sop: SOP,
    inferencer_factory: Callable,
    workspace: Path,
    *,
    yolo_mode: bool = False,
    max_concurrency: int = 1,
    max_expansion_depth: int = 200,
    max_total_nodes: int = 500,
    max_goto_iterations: int = 10,
    graph_reporter: Any = None,
) -> tuple[WorkGraph, StateGraphTracker]:
    """Construct a WorkGraph + StateGraphTracker pair for an SOP.

    Returns both because callers need the tracker for snapshots/persistence.
    Raises InvalidSOPError if no [__initial__] phase is declared.
    """
    tracker = StateGraphTracker(
        graph=sop,
        max_goto_iterations=max_goto_iterations,
    )
    tracker_lock = asyncio.Lock()

    initial_phases = [
        p for p in sop.phases
        if "initial" in getattr(p, "directives", [])
    ]
    if not initial_phases:
        raise InvalidSOPError(
            "SOP has no [__initial__] phase. "
            "Mark at least one phase with [__initial__] to designate the entry point."
        )

    start_nodes = [
        SOPWorkGraphNode(
            phase=p,
            sop=sop,
            tracker=tracker,
            tracker_lock=tracker_lock,
            inferencer_factory=inferencer_factory,
            workspace=workspace / f"phase_{p.id}_iter_0",
            yolo_mode=yolo_mode,
        )
        for p in initial_phases
    ]

    graph = WorkGraph(
        start_nodes=start_nodes,
        use_async=True,
        max_concurrency=max_concurrency,
        max_expansion_depth=max_expansion_depth,
        max_total_nodes=max_total_nodes,
    )

    if graph_reporter:
        graph.set_graph_event_callback(graph_reporter.on_graph_topology)

    return graph, tracker

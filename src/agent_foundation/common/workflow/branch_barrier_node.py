"""BranchBarrierNode — convergence node for __branch__ fan-out.

Mirrors BTA's aggregator-after-workers pattern. Uses WorkGraph's existing
multi-parent Queue merge (_merge_upstream_inputs) to wait for ALL branch
leaves. No per-sibling counter, no race conditions.

Uses value=self._barrier_aggregate (ActionNode pattern at action_node.py:229).
Does NOT override _run/_arun.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from attr import attrib, attrs

from rich_python_utils.common_objects.workflow.stategraph import (
    StateGraphTracker,
)
from rich_python_utils.common_objects.workflow.workgraph import (
    WorkGraphNode,
)
from rich_python_utils.common_objects.workflow.common.expansion import (
    GraphExpansionResult,
    SubgraphSpec,
)
from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
    StepResultSaveOptions,
)
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOP,
)

logger = logging.getLogger(__name__)


@attrs(slots=False, kw_only=True)
class BranchBarrierNode(WorkGraphNode):
    """Convergence node for __branch__ — gathers all sibling outputs once.

    tracker.complete() is called EXACTLY ONCE here, regardless of N leaves.
    """

    phase_id: str = attrib()
    tracker: StateGraphTracker = attrib()
    tracker_lock: asyncio.Lock = attrib()
    sop: SOP = attrib()
    inferencer_factory: Callable = attrib()
    workspace: Path = attrib()
    expected_count: int = attrib()
    yolo_mode: bool = attrib(default=False)

    def __attrs_post_init__(self) -> None:
        self.value = self._barrier_aggregate
        self.name = f"sop_branch_barrier_{self.phase_id}"
        self.enable_result_save = StepResultSaveOptions.Always

    async def _barrier_aggregate(self, *leaf_results, **_) -> Any:
        """Aggregate all branch leaf outputs into one phase completion."""
        merged_outputs = {"branch_results": list(leaf_results)}

        async with self.tracker_lock:
            self.tracker.complete(self.phase_id, **merged_outputs)
            available = self.tracker.get_available_next()
            pending_threads = self.tracker.get_pending_thread_spawns()

        next_nodes = self._build_successors(available, pending_threads)
        if not next_nodes:
            return merged_outputs

        return GraphExpansionResult(
            result=merged_outputs,
            subgraph=SubgraphSpec(nodes=next_nodes, entry_nodes=next_nodes),
            expansion_id=f"sop_branch_complete_{self.phase_id}",
        )

    def _build_successors(self, available, pending_threads) -> list:
        from agent_foundation.common.workflow.sop_workgraph_node import (
            SOPWorkGraphNode,
            _parse_duration_seconds,
        )

        next_nodes = []
        for phase in available:
            if phase.branch:
                items = self.tracker.get_branch_items(phase.id)
                if items:
                    leaves = []
                    for i, item in enumerate(items):
                        node = SOPWorkGraphNode(
                            phase=phase, sop=self.sop, tracker=self.tracker,
                            tracker_lock=self.tracker_lock,
                            inferencer_factory=self.inferencer_factory,
                            workspace=self.workspace.parent / f"phase_{phase.id}_iter_{i}",
                            branch_item=item, iteration=i, yolo_mode=self.yolo_mode,
                            is_branch_leaf=True,
                        )
                        leaves.append(node)
                    barrier = BranchBarrierNode(
                        phase_id=phase.id, tracker=self.tracker,
                        tracker_lock=self.tracker_lock, sop=self.sop,
                        inferencer_factory=self.inferencer_factory,
                        workspace=self.workspace.parent / f"phase_{phase.id}_barrier",
                        expected_count=len(leaves), yolo_mode=self.yolo_mode,
                    )
                    for leaf in leaves:
                        leaf.add_next(barrier)
                    next_nodes.extend(leaves + [barrier])
                    continue
            next_nodes.append(SOPWorkGraphNode(
                phase=phase, sop=self.sop, tracker=self.tracker,
                tracker_lock=self.tracker_lock,
                inferencer_factory=self.inferencer_factory,
                workspace=self.workspace.parent / f"phase_{phase.id}_iter_0",
                iteration=0, yolo_mode=self.yolo_mode,
            ))

        for spawn in pending_threads:
            target_phase = self.sop.get_phase(spawn.target_phase)
            if target_phase:
                iter_num = self.tracker.goto_counts.get(
                    f"{spawn.source_phase}->{spawn.target_phase}", 0
                )
                node = SOPWorkGraphNode(
                    phase=target_phase, sop=self.sop, tracker=self.tracker,
                    tracker_lock=self.tracker_lock,
                    inferencer_factory=self.inferencer_factory,
                    workspace=self.workspace.parent / f"phase_{target_phase.id}_iter_{iter_num}",
                    iteration=iter_num, yolo_mode=self.yolo_mode,
                )
                if spawn.wait_duration:
                    node.min_repeat_wait = _parse_duration_seconds(spawn.wait_duration)
                next_nodes.append(node)

        return next_nodes

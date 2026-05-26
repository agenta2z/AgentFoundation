"""SOPWorkGraphNode — bridge between StateGraph (blueprint) and WorkGraph (runtime).

Each instance runs one SOP phase via a ConversationalInferencer.
On completion, queries StateGraphTracker for what's next and creates
successor nodes via GraphExpansionResult.

Production precedent: BTA's breakdown_node = WorkGraphNode(...) at
breakdown_then_aggregate_inferencer.py:1226 wrapping ConversationalFlowNodeAdapter.

Uses value=self._execute_phase (ActionNode pattern at action_node.py:229).
Does NOT override _run/_arun — those overrides would be dead code because
WorkGraph dispatches through self.value via async_execute_with_retry.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional

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
    SOPPhase,
)

logger = logging.getLogger(__name__)


@attrs(slots=False, kw_only=True)
class SOPWorkGraphNode(WorkGraphNode):
    """A WorkGraphNode that runs one SOP phase via a ConversationalInferencer."""

    phase: SOPPhase = attrib()
    sop: SOP = attrib()
    tracker: StateGraphTracker = attrib()
    tracker_lock: asyncio.Lock = attrib()
    inferencer_factory: Callable = attrib()
    workspace: Path = attrib()

    branch_item: Any = attrib(default=None)
    is_branch_leaf: bool = attrib(default=False)
    iteration: int = attrib(default=0)
    min_repeat_wait: Optional[float] = attrib(default=None)
    yolo_mode: bool = attrib(default=False)

    def __attrs_post_init__(self) -> None:
        self.value = self._execute_phase

        node_name = f"sop_phase_{self.phase.id}_iter{self.iteration}"
        if self.branch_item is not None:
            node_name += f"_branch{hash(self.branch_item) & 0xFFFF:04x}"
        self.name = node_name

        self.enable_result_save = StepResultSaveOptions.SkipResumable

    async def _execute_phase(self, *upstream_results, **_) -> Any:
        """Execute one SOP phase. Invoked by WorkGraph._arun via self.value."""
        if self.min_repeat_wait:
            await asyncio.sleep(self.min_repeat_wait)

        inferencer = self.inferencer_factory(
            phase=self.phase,
            sop=self.sop,
            branch_item=self.branch_item,
            yolo_mode=self.yolo_mode,
            workspace=self.workspace,
            upstream_results=upstream_results,
        )

        result = await inferencer.run_agentic_loop(
            content=self._build_seed_message(),
        )

        if self.is_branch_leaf:
            return result

        artifacts = self._extract_declared_outputs(result)
        async with self.tracker_lock:
            self.tracker.complete(self.phase.id, **artifacts)

        return await self._build_successor_expansion(artifacts)

    async def _build_successor_expansion(self, artifacts) -> Any:
        """Query tracker for next phases; build successor nodes."""
        async with self.tracker_lock:
            available = self.tracker.get_available_next()
            pending_threads = self.tracker.get_pending_thread_spawns()

        next_nodes = []

        for phase in available:
            if phase.branch:
                next_nodes.extend(self._build_branch_leaves_and_barrier(phase))
            else:
                next_nodes.append(self._create_child_node(phase, iteration=0))

        for spawn in pending_threads:
            target_phase = self.sop.get_phase(spawn.target_phase)
            if target_phase:
                iter_num = self.tracker.goto_counts.get(
                    f"{spawn.source_phase}->{spawn.target_phase}", 0
                )
                node = self._create_child_node(target_phase, iteration=iter_num)
                if spawn.wait_duration:
                    node.min_repeat_wait = _parse_duration_seconds(spawn.wait_duration)
                next_nodes.append(node)

        if not next_nodes:
            return artifacts

        return GraphExpansionResult(
            result=artifacts,
            subgraph=SubgraphSpec(nodes=next_nodes, entry_nodes=next_nodes),
            expansion_id=f"sop_phase_{self.phase.id}_iter_{self.iteration}",
            seed={"phase_id": self.phase.id},
        )

    def _create_child_node(
        self, phase, *, iteration=0, branch_item=None, is_branch_leaf=False,
    ):
        child_workspace = self.workspace.parent / f"phase_{phase.id}_iter_{iteration}"
        return SOPWorkGraphNode(
            phase=phase,
            sop=self.sop,
            tracker=self.tracker,
            tracker_lock=self.tracker_lock,
            inferencer_factory=self.inferencer_factory,
            workspace=child_workspace,
            branch_item=branch_item,
            iteration=iteration,
            yolo_mode=self.yolo_mode,
            is_branch_leaf=is_branch_leaf,
        )

    def _build_branch_leaves_and_barrier(self, phase: SOPPhase) -> list:
        """Build N branch leaves + 1 BranchBarrierNode."""
        from agent_foundation.common.workflow.branch_barrier_node import BranchBarrierNode

        items = self.tracker.get_branch_items(phase.id)
        if not items:
            return [self._create_child_node(phase, iteration=0)]

        leaves = [
            self._create_child_node(
                phase, iteration=i, branch_item=item, is_branch_leaf=True,
            )
            for i, item in enumerate(items)
        ]
        barrier = BranchBarrierNode(
            phase_id=phase.id,
            tracker=self.tracker,
            tracker_lock=self.tracker_lock,
            sop=self.sop,
            inferencer_factory=self.inferencer_factory,
            workspace=self.workspace.parent / f"phase_{phase.id}_barrier",
            expected_count=len(leaves),
            yolo_mode=self.yolo_mode,
        )
        for leaf in leaves:
            leaf.add_next(barrier)
        return leaves + [barrier]

    def _build_seed_message(self) -> str:
        parts = [f"Execute Phase {self.phase.id}"]
        if self.phase.name:
            parts[0] += f" — {self.phase.name}"
        if self.phase.description:
            parts.append(self.phase.description[:500])
        if self.branch_item is not None:
            parts.append(f"Branch item: {self.branch_item}")
        return "\n\n".join(parts)

    def _extract_declared_outputs(self, result) -> dict[str, Any]:
        outputs = {}
        if hasattr(result, "text") and result.text:
            for out_name in self.phase.outputs:
                outputs[out_name] = result.text
        return outputs


def _parse_duration_seconds(duration_str: str) -> float:
    """Parse '1h', '30m', '10s', '2d' into seconds."""
    if not duration_str:
        return 0.0
    unit = duration_str[-1].lower()
    value = float(duration_str[:-1])
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit, 1)
    return value * multiplier

"""Graph execution event protocol for UI visualization.

Events emitted by WorkGraph-based inferencers (BTA, LinearWorkflow, etc.)
during execution. Consumed by interactive adapters (WebSocket, terminal, etc.)
to render real-time graph visualization.

Design decisions:
- No new GraphNode class — reuse WorkGraph.to_serializable_obj() + transient attrs
- NodeStatusEvent.timestamp uses time.time() for elapsed-time display in UI
- All graph event sends should be wrapped in try/except to prevent visualization
  failures from aborting the underlying computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeStatus(StrEnum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    ERROR     = "error"
    SKIPPED   = "skipped"   # resumed from checkpoint


@dataclass
class GraphTopologyEvent:
    """Sent once when graph is built. Reuses WorkGraph.to_serializable_obj()."""
    nodes: list[dict]   # [{id, label, group, status, is_container?}, ...]
    edges: list[dict]   # [{"source": name, "target": name}, ...]
    layout: str = "horizontal"
    parent_node_id: str = ""   # non-empty when this is a sub-graph of a parent node
    version: int = 0           # monotonic counter for UI change detection

    @classmethod
    def from_work_graph(
        cls,
        workgraph: Any,
        breakdown_node_name: str = "",
        layout: str = "horizontal",
    ) -> "GraphTopologyEvent":
        """Build topology from existing WorkGraph — no new GraphNode class needed.

        Label priority:
          1. node._viz_label  — set by BTA at node creation with actual subtask description
          2. node.name        — fallback (e.g. "worker_0")

        Group is read from the live node object (WorkGraphNode.to_serializable_obj()
        does NOT include 'group' — only {_type, _module, name, value_ref,
        next_names, previous_names, config}).

        NOTE: _all_nodes() is private (underscore prefix) in WorkGraph.
        """
        serialized = workgraph.to_serializable_obj()
        # Build name→node map to access transient attrs (_viz_label, group)
        node_map = {n.name: n for n in workgraph._all_nodes()}
        nodes = []
        edges = []
        for n in serialized.get("nodes", []):
            real_node = node_map.get(n["name"])
            viz_label = getattr(real_node, "_viz_label", None) or n["name"]
            # group NOT in serialized dict — read from live object
            node_group = getattr(real_node, "group", None) if real_node else None
            nodes.append({
                "id": n["name"],
                "label": viz_label,
                "group": node_group,
                "status": (
                    NodeStatus.COMPLETED if n["name"] == breakdown_node_name
                    else NodeStatus.PENDING
                ),
                "is_container": getattr(real_node, "_is_container", False) if real_node else False,
            })
            for target in n.get("next_names", []):
                edges.append({"source": n["name"], "target": target})
        return cls(nodes=nodes, edges=edges, layout=layout)


@dataclass
class NodeStatusEvent:
    """Sent when a node's lifecycle status changes."""
    node_id: str
    status: str            # NodeStatus value
    label: str = ""        # optional dynamic label update
    error: str = ""
    timestamp: float = field(default_factory=time.time)  # time.time() — for elapsed-time UI
    output_path: str = ""  # absolute path to node's output file (served via /api/view/{path})
                           # Set by BTA _async_status_cb on completion so UI can fetch content


@dataclass
class NodeStreamEvent:
    """Sent when a node produces streaming content."""
    node_id: str
    content: str           # text chunk
    is_final: bool = False # True on last chunk — signals end of streaming for this node


@dataclass
class GraphReconcileEvent:
    """Sent once after WorkGraph completes — final truth for all node statuses.

    The frontend reconciles any stale UI state (e.g., a node stuck on
    "running" due to a lost WebSocket event) and logs every gap as a
    console warning for observability.
    """
    node_statuses: dict    # {node_id: status_str}

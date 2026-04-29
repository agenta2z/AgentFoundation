# Plan: Generic Graph Visualization for WorkGraph/Workflow Nodes
## Final Consolidated Plan (v3 — integrates all agent discussions)

---

## Context

The TaskPanel currently shows only "create_role is running..." with no pipeline visibility.
The `create_role` tool uses `BreakdownThenAggregateInferencer` (BTA): breakdown → N workers → aggregator.

Goal: Generic visualization in AgentFoundation, BTA as first consumer, OpenStartup TaskPanel as first UI.

---

## Key Design Decisions (resolved from multi-agent discussion)

| Decision | Resolution | Rationale |
|---|---|---|
| **No new `GraphNode` class** | Reuse `WorkGraph.to_serializable_obj()` + `Node.name/.next/.group` | Already exists, no duplication |
| **No `detail_view` on nodes** | Nodes deliver content via `node_stream` events — UI renders whatever arrives | Keep graph definition structural only |
| **Status as transient attr** | `node._viz_status` on existing `WorkGraphNode` (slots=False, safe) | No new class needed |
| **`_build_diamond_graph` guard** | Called from 6+ paths — emit topology only once via `_graph_topology_emitted` flag | Prevent duplicate WS messages |
| **`NodeStreamInteractive` per worker** | Wrap the real interactive with `node_id` injection at `stream_token_batches()` level | Workers stream via acli → intercepted here |
| **Protocol-based `graph_reporter`** | BTA never imports WebSocket code — calls protocol methods | Clean layer separation |

---

## Architecture

```
Server (Python)                       WS message           Client (React)
─────────────────────────────────     ──────────────────   ──────────────────────────────────
BTA._ainfer()
  ├─ _build_diamond_graph()  ──────→  graph_topology   →  tasks[id].graph = {nodes, edges}
  ├─ WorkGraph propagates callbacks
  ├─ worker_0 starts         ──────→  node_status      →  node "worker_0" → running
  ├─ worker_1 starts         ──────→  node_status      →  node "worker_1" → running
  ├─ worker_0 token          ──────→  node_stream      →  tasks[id].nodeStreams[worker_0] +=
  ├─ worker_0 done           ──────→  node_status      →  node "worker_0" → completed
  └─ aggregator starts       ──────→  node_status      →  node "aggregator" → running
```

---

## Phase 1: Generic Graph Event Protocol (AgentFoundation)

### 1a. Event types — minimal, reuse existing WorkGraph serialization

**File: `AgentFoundation/src/agent_foundation/common/inferencers/graph_events.py`** (NEW)

`WorkGraph.to_serializable_obj()` (workgraph.py:1164) already traverses nodes and returns
`{name, next_names, previous_names, group, ...}`. Reuse it — no new `GraphNode` class.
`Node` uses `@attrs(slots=False)` so transient attrs on instances are safe.

```python
from dataclasses import dataclass
from enum import StrEnum

class NodeStatus(StrEnum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    ERROR     = "error"
    SKIPPED   = "skipped"   # resumed from checkpoint

@dataclass
class GraphTopologyEvent:
    """Sent once when graph is built. Reuses WorkGraph.to_serializable_obj()."""
    nodes: list[dict]   # [{id, label, group, status}, ...]
    edges: list[dict]   # [{"source": name, "target": name}, ...]
    layout: str = "horizontal"

    @classmethod
    def from_work_graph(cls, workgraph, breakdown_node_name: str = "",
                        layout: str = "horizontal") -> "GraphTopologyEvent":
        """Build topology from existing WorkGraph — no new GraphNode class needed.
        
        Label priority:
          1. node._viz_label  — set by BTA at node creation with the actual subtask description
          2. node.name        — fallback (e.g. "worker_0")
        
        This means workers show "Research core responsibilities" not "Worker 0".
        """
        serialized = workgraph.to_serializable_obj()
        # Build a quick name→node map to access transient attrs like _viz_label
        # NOTE: the method is _all_nodes() (private, underscore prefix), NOT get_all_nodes()
        node_map = {n.name: n for n in workgraph._all_nodes()}
        nodes = []
        edges = []
        for n in serialized.get("nodes", []):
            real_node = node_map.get(n["name"])
            viz_label = getattr(real_node, "_viz_label", None) or n["name"]
            # NOTE: WorkGraphNode.to_serializable_obj() does NOT include 'group'.
            # Read it directly from the live node object via node_map.
            node_group = getattr(real_node, "group", None) if real_node else None
            nodes.append({
                "id": n["name"],
                "label": viz_label,
                "group": node_group,
                "status": NodeStatus.COMPLETED if n["name"] == breakdown_node_name
                          else NodeStatus.PENDING,
            })
            for target in n.get("next_names", []):
                edges.append({"source": n["name"], "target": target})
        return cls(nodes=nodes, edges=edges, layout=layout)

@dataclass
class NodeStatusEvent:
    node_id: str
    status: str
    label: str = ""
    error: str = ""
    timestamp: float = 0.0  # time.time() — set by emitter; used for elapsed-time display in UI

@dataclass
class NodeStreamEvent:
    node_id: str
    content: str
    is_final: bool = False
```

### 1b. WorkGraphNode — automatic status events (all WorkGraphs)

**File: `RichPythonUtils/src/.../workflow/workgraph.py`**

Add `_graph_event_callback` to `WorkGraphNode` (optional, default None).
When set, nodes emit RUNNING/COMPLETED/ERROR status events during `__call__`.
`WorkGraph` exposes `set_graph_event_callback()` to propagate to all nodes.

```python
# WorkGraphNode — add attribute:
_graph_event_callback: Optional[Callable] = attrib(default=None, repr=False, kw_only=True)

# WorkGraphNode._run() (line ~560) and _arun() (line ~821) — wrap existing execution logic:
if self._graph_event_callback:
    self._graph_event_callback(NodeStatusEvent(self.name, NodeStatus.RUNNING))
try:
    result = <existing_execution>
    if self._graph_event_callback:
        self._graph_event_callback(NodeStatusEvent(self.name, NodeStatus.COMPLETED))
    return result
except Exception as e:
    if self._graph_event_callback:
        self._graph_event_callback(NodeStatusEvent(self.name, NodeStatus.ERROR, error=str(e)))
    raise

# WorkGraph — add method to propagate callback to all reachable nodes:
def set_graph_event_callback(self, callback: Callable) -> None:
    for node in self._all_nodes():  # private method at workgraph.py:1239
        node._graph_event_callback = callback
```

### 1c. BTA — emit topology + inject per-worker interactives

**File: `AgentFoundation/.../flow_inferencers/breakdown_then_aggregate_inferencer.py`**

**CRITICAL: `_build_diamond_graph()` is called from 6+ paths (lines 901, 926, 971, 998,
1030, 1121). Guard with `_graph_topology_emitted` flag.**

**Worker labels**: At node creation, set `node._viz_label = query_str[:80]` so the UI shows
the actual subtask description (e.g. "Research core coordination responsibilities") not
"worker_0". `query_str` is available as `q` in `_build_diamond_graph`'s inner loop.
`WorkGraphNode` uses `@attrs(slots=False)` so transient attrs are safe.

```python
# In _build_diamond_graph, when creating each worker node:
node = WorkGraphNode(name=f"worker_{i}", value=..., group=worker_group, ...)
# Store subtask description as display label — read by GraphTopologyEvent.from_work_graph()
node._viz_label = (query_str[:80] if isinstance(query_str, str)
                   else query_str.get("description", f"worker_{i}")[:80])
worker_nodes.append(node)
```

```python
# New attrs:
graph_reporter: Optional[Any] = attrib(default=None, kw_only=True)
_graph_topology_emitted: bool = attrib(default=False, init=False, repr=False)
# NOTE: All graph event emissions should be wrapped in try/except to prevent
# visualization failures from aborting the BTA computation:
#   try: await self.graph_reporter.on_graph_topology(...)
#   except Exception as e: _logger.warning("graph event failed: %s", e)

# In _build_diamond_graph(), at the TOP (reset flag for reused BTA instances):
# If the same BTA instance is reused (retry, interactive re-run), the flag must reset
# so topology is re-emitted with the new sub-queries.
self._graph_topology_emitted = False  # reset at start of each _build_diamond_graph call

# AFTER self.start_nodes = worker_nodes:
if self.graph_reporter and not self._graph_topology_emitted:
    self._graph_topology_emitted = True
    # Store for async emit in _ainfer() — _build_diamond_graph is sync
    # Build topology: from_work_graph covers workers + aggregator.
    # Breakdown is NOT a WorkGraphNode — it runs before _build_diamond_graph.
    # Manually prepend a virtual breakdown node to the topology.
    worker_agg_topology = GraphTopologyEvent.from_work_graph(self)
    breakdown_node = {"id": "breakdown", "label": "Breakdown", "group": None,
                      "status": NodeStatus.COMPLETED}
    worker_agg_topology.nodes.insert(0, breakdown_node)
    for wn in self.start_nodes:  # workers — add breakdown→worker edges
        worker_agg_topology.edges.insert(0, {"source": "breakdown", "target": wn.name})
    self._pending_topology = worker_agg_topology

# In _ainfer(), BEFORE asyncio.gather(worker tasks):
# Use getattr + None check — safer than hasattr (guards against exception mid-del)
pending = getattr(self, '_pending_topology', None)
if self.graph_reporter and pending is not None:
    self._pending_topology = None  # clear first (re-entrant safety)
    await self.graph_reporter.on_graph_topology(pending)
    # Propagate async status callbacks to all WorkGraph nodes.
    # BTA uses _arun() (use_async=True), so the callback is awaited in _arun().
    reporter = self.graph_reporter
    async def _async_status_cb(event):
        await reporter.on_node_status(
            event.node_id, event.status, getattr(event, 'error', '')
        )
    self.set_graph_event_callback(_async_status_cb)
    # IMPORTANT: WorkGraphNode._arun() must await the async callback properly.
    # In _arun() (workgraph.py ~line 821), use call_maybe_async (already imported at line 26):
    #   result = self._graph_event_callback(NodeStatusEvent(...))
    #   if asyncio.iscoroutine(result): await result
    # OR simply: await call_maybe_async(self._graph_event_callback, event)
    # In _run() (sync path, line 560), the async callback would return an unawaited coroutine
    # — skip callbacks in _run() or check with inspect.iscoroutinefunction().
    # Give each worker inferencer its own NodeStreamInteractive for per-node streaming.
    # IMPORTANT: The raw inferencer objects (w) are captured in closures inside _make_worker_fn
    # and are NOT accessible via start_nodes (those are WorkGraphNode objects, not inferencers).
    # Solution: In _build_diamond_graph(), collect worker inferencers in a list BEFORE calling
    # _make_worker_fn(), then assign their interactive IMMEDIATELY (before the closure is built):
    #
    #   worker_inferencers = []  # collect all w objects
    #   # Inside the worker creation loop, BEFORE _make_worker_fn(w, q, ...):
    #   if self.graph_reporter:
    #       w.interactive = self.graph_reporter.node_interactive(f"worker_{i}")
    #   worker_inferencers.append(w)
    #   node = WorkGraphNode(name=f"worker_{i}", value=_make_worker_fn(w, q, ...), ...)
    #
    # w.ainfer(q) reads w.interactive at call time (closure captures reference to w),
    # so setting w.interactive BEFORE _make_worker_fn is built guarantees correct routing.
    # The graph_reporter is available because it's an instance attribute set in _ainfer().
```

### 1d. NodeStreamInteractive + WebSocketGraphReporter

**File: `AgentFoundation/src/agent_foundation/ui/graph_interactive_adapter.py`** (NEW)

The CRITICAL piece for concurrent per-node token streaming.
Workers stream via `acli` → `WebSocketInteractive.stream_token_batches()`.
We intercept at `stream_token_batches()` level by giving each worker a wrapped interactive.

```python
class NodeStreamInteractive:
    """Wraps WebSocketInteractive — tags all streaming with a specific node_id.
    
    Each BTA worker gets its own instance. All stream_token_batches() calls are
    intercepted and emitted as node_stream WS messages tagged with node_id.
    Delegates all other interactive methods (send_pending_input, etc.) to
    the real WebSocketInteractive.
    """
    def __init__(self, ws_interactive, task_id: str, node_id: str):
        self._ws = ws_interactive
        self._task_id = task_id
        self._node_id = node_id

    async def stream_token_batches(
        self,
        token_stream,               # AsyncIterator[tuple[str, dict]] — yields (chunk, metadata)
        session_id: str = "",
        batch_interval_ms: float = 50.0,
        task_id=None,
        send_stream_end: bool = True,
        turn_number=None,
        **kwargs,
    ) -> str:
        """Intercept + re-emit tokens with node_id, then delegate to real WS for batching.

        IMPORTANT: Real stream_token_batches() (websocket_interactive.py:54) takes
        (token_stream: AsyncIterator[tuple[str, dict]], session_id, ...) where
        the stream yields (chunk_str, metadata_dict) TUPLES — NOT plain strings.
        """
        async def _tagged_stream():
            async for chunk, metadata in token_stream:
                await self._ws.send_graph_event(
                    NodeStreamEvent(node_id=self._node_id, content=chunk),
                    task_id=self._task_id
                )
                yield chunk, metadata

        result = await self._ws.stream_token_batches(
            _tagged_stream(), session_id, batch_interval_ms,
            task_id, send_stream_end, turn_number
        )
        await self._ws.send_graph_event(
            NodeStreamEvent(node_id=self._node_id, content="", is_final=True),
            task_id=self._task_id
        )
        return result

    def __getattr__(self, name):
        # Delegate everything else to the real interactive
        return getattr(self._ws, name)


class WebSocketGraphReporter:
    """Bridges BTA graph_reporter protocol → WebSocket send_graph_event().
    BTA never imports WebSocket code — it calls this via the protocol."""

    def __init__(self, ws_interactive, task_id: str):
        self._ws = ws_interactive
        self._task_id = task_id

    async def on_graph_topology(self, event) -> None:
        await self._ws.send_graph_event(event, task_id=self._task_id)

    async def on_node_status(self, node_id: str, status: str, error: str = "") -> None:
        from agent_foundation.common.inferencers.graph_events import NodeStatusEvent
        await self._ws.send_graph_event(
            NodeStatusEvent(node_id=node_id, status=status, error=error),
            task_id=self._task_id
        )

    def node_interactive(self, node_id: str) -> "NodeStreamInteractive":
        """Returns a per-node interactive wrapper for token streaming."""
        return NodeStreamInteractive(self._ws, self._task_id, node_id)
```

---

## Phase 2: Server Wiring (OpenStartup)

### 2a. `send_graph_event()` in websocket_interactive.py

```python
async def send_graph_event(self, event: Any, task_id: str = "") -> None:
    from agent_foundation.common.inferencers.graph_events import (
        GraphTopologyEvent, NodeStatusEvent, NodeStreamEvent
    )
    if isinstance(event, GraphTopologyEvent):
        msg = {"type": "graph_topology", "task_id": task_id,
               "nodes": event.nodes, "edges": event.edges, "layout": event.layout}
    elif isinstance(event, NodeStatusEvent):
        msg = {"type": "node_status", "task_id": task_id,
               "node_id": event.node_id, "status": event.status,
               "label": event.label, "error": event.error}
    elif isinstance(event, NodeStreamEvent):
        msg = {"type": "node_stream", "task_id": task_id,
               "node_id": event.node_id, "content": event.content,
               "is_final": event.is_final}
    else:
        return
    await self._send(msg)
```

### 2b. tool_dispatcher.py — pass interactive + task_id in task_context

The interactive and task_id are already in task_context. Add a convenience reference:

```python
task_context = {
    **self._session_context,
    "task_id": task_id,
    "working_dir": task_working_dir,
    "interactive": interactive_ref,   # already present — used by executor
}
```

### 2c. create_role/executor.py — attach WebSocketGraphReporter to BTA

```python
from agent_foundation.ui.graph_interactive_adapter import WebSocketGraphReporter

# After instantiate(cfg):
interactive = session_context.get("interactive")
task_id = session_context.get("task_id", "")
if interactive and task_id:
    inferencer.graph_reporter = WebSocketGraphReporter(interactive, task_id)
```

---

## Phase 3: UI Components (OpenStartup)

### 3a. useManagerChat.js — handle 3 new message types

```javascript
case 'graph_topology': {
  const tid = data.task_id;
  if (!tid) break;
  setTasks(prev => ({...prev, [tid]: {
    ...prev[tid],
    graph: { nodes: data.nodes, edges: data.edges, layout: data.layout },
    nodeStreams: {},
  }}));
  break;
}
case 'node_status': {
  const tid = data.task_id;
  if (!tid) break;
  // NOTE: Do NOT read `tasks` directly here — handleServerMessage is useCallback([])
  // so `tasks` would be a stale closure. Always use the `prev` pattern inside setTasks.
  setTasks(prev => {
    const task = prev[tid];
    if (!task?.graph) return prev;  // check via prev, not stale `tasks`
    const nodes = task.graph.nodes.map(n =>
      n.id === data.node_id
        ? {...n, status: data.status, label: data.label || n.label, error: data.error}
        : n
    );
    return {...prev, [tid]: {...task, graph: {...task.graph, nodes}}};
  });
  break;
}
case 'node_stream': {
  const tid = data.task_id;
  if (!tid) break;
  setTasks(prev => {
    const task = prev[tid];
    if (!task) return prev;
    const nodeStreams = {...(task.nodeStreams || {})};
    nodeStreams[data.node_id] = (nodeStreams[data.node_id] || '') + data.content;
    return {...prev, [tid]: {...task, nodeStreams}};
  });
  break;
}
```

### 3b. GraphFlowView.js (NEW) — horizontal flow visualization

Pure SVG/CSS — no external library. Topological depth layout:

```
[Breakdown ✅] ──→ [Worker 1 🔄] ──→ [Aggregator ⏳]
               └→  [Worker 2 ✅]
               └→  [Worker 3 🔄]
```

- Group nodes by BFS depth from source nodes
- x = depth × COLUMN_WIDTH, y = position_in_group × ROW_HEIGHT (vertically centered)
- Nodes: MUI Box with status color + icon + label
- Edges: SVG `<path>` bezier curves between node centers
- Click → `onNodeClick(node.id)` — updates `selectedNodeId`
- Auto-switch: when a node becomes `running`, auto-select it (overridable by click)

### 3c. NodeDetailPanel.js (NEW) — selected node content

- Header: node label + status chip + timing (from Phase 3f — `timestamp` on `NodeStatusEvent`)
- Body: `MarkdownRenderer` with `nodeStreams[selectedNodeId]`
- Blinking cursor while `node.status === 'running'`
- Auto-scrolls as content arrives
- Works for BOTH live streaming AND completed nodes (review final output)

### 3d. TaskPanel.js — split-pane layout

```jsx
{task.graph ? (
  <Box sx={{ display:'flex', flexDirection:'column', height:'100%' }}>
    <Box sx={{ height:'40%', borderBottom:'1px solid', borderColor:'divider', overflow:'auto', flexShrink:0 }}>
      <GraphFlowView
        nodes={task.graph.nodes}
        edges={task.graph.edges}
        selectedNodeId={selectedNodeId}
        onNodeClick={setSelectedNodeId}
      />
    </Box>
    <Box sx={{ flex:1, overflow:'auto' }}>
      <NodeDetailPanel
        node={task.graph.nodes.find(n => n.id === selectedNodeId)}
        content={task.nodeStreams?.[selectedNodeId] || ''}
        isStreaming={task.graph.nodes.find(n => n.id === selectedNodeId)?.status === 'running'}
      />
    </Box>
  </Box>
) : (
  /* Existing streaming view — no regression for non-graph tasks */
  <MarkdownRenderer content={task.streamContent} />
)}
```

---

## Phase 3e: Auto-switch to running node

In `useManagerChat.js` `node_status` handler, track auto-selection separately from manual clicks:

```javascript
case 'node_status': {
  // Merge status update + auto-select into ONE setTasks call to avoid stale prev issues
  setTasks(prev => {
    const task = prev[tid];
    if (!task?.graph) return prev;
    const nodes = task.graph.nodes.map(n =>
      n.id === data.node_id ? {...n, status: data.status} : n
    );
    const update = {...task, graph: {...task.graph, nodes}};
    if (data.status === 'running') update.autoSelectedNodeId = data.node_id;
    return {...prev, [tid]: update};
  });
  break;
}
```

In `TaskPanel.js`, distinguish auto-selection from user click:
```jsx
const [userSelectedNodeId, setUserSelectedNodeId] = useState(null);
const effectiveNodeId = userSelectedNodeId || task.autoSelectedNodeId || task.graph?.nodes[0]?.id;
// onNodeClick sets userSelectedNodeId (overrides auto-switch)
```

## Phase 3f: Node timing

`NodeStatusEvent` already includes `timestamp: float` (defined in Phase 1a / graph_events.py).
The `WorkGraphNode._run()`/`_arun()` callback must set it with `time.time()`:

```python
import time
self._graph_event_callback(NodeStatusEvent(
    node_id=self.name, status=NodeStatus.RUNNING, timestamp=time.time()
))
```

In `NodeDetailPanel.js`, show elapsed time when running:
```jsx
{node.status === 'running' && node.startedAt && (
  <Typography variant="caption" sx={{ color: 'text.disabled' }}>
    {Math.round((Date.now()/1000 - node.startedAt))}s elapsed
  </Typography>
)}
```

Store `startedAt`/`completedAt` from `timestamp` in `useManagerChat.js` node_status handler.

## Phase 3g: Collapse graph when all nodes complete

In `TaskPanel.js`:
```jsx
const allComplete = task.graph.nodes.every(n => ['completed', 'error', 'skipped'].includes(n.status));
const [graphCollapsed, setGraphCollapsed] = useState(false);

useEffect(() => {
  if (allComplete) setGraphCollapsed(true);
}, [allComplete]);

<Box sx={{ height: graphCollapsed ? 48 : '40%', transition: 'height 0.3s ease', overflow: 'hidden', flexShrink: 0 }}>
  {graphCollapsed ? (
    <Box onClick={() => setGraphCollapsed(false)}
         sx={{ cursor: 'pointer', px: 2, py: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
      <CheckCircleIcon sx={{ color: 'success.main', fontSize: 16 }} />
      <Typography variant="caption">All {task.graph.nodes.length} nodes completed — click to expand</Typography>
    </Box>
  ) : (
    <GraphFlowView ... />
  )}
</Box>
```

## Phase 4: Polish

1. **Resizable splitter** between graph and detail panels
2. **`useInferencerGraph` hook** in AF's React library for reuse outside OpenStartup

---

## File Summary

### AgentFoundation
| File | Change |
|---|---|
| `common/inferencers/graph_events.py` | **NEW** — NodeStatus, GraphTopologyEvent.from_work_graph(), NodeStatusEvent, NodeStreamEvent |
| `ui/graph_interactive_adapter.py` | **NEW** — NodeStreamInteractive, WebSocketGraphReporter |
| `.../breakdown_then_aggregate_inferencer.py` | Add `graph_reporter`, `_graph_topology_emitted`; emit in `_build_diamond_graph` + `_ainfer` |

### RichPythonUtils
| File | Change |
|---|---|
| `.../workflow/workgraph.py` | Add `_graph_event_callback` to WorkGraphNode + `set_graph_event_callback()` to WorkGraph |

### OpenStartup Server
| File | Change |
|---|---|
| `services/websocket_interactive.py` | Add `send_graph_event()` |
| `resources/tools/create_role/executor.py` | Attach `WebSocketGraphReporter` to BTA |

### OpenStartup UI
| File | Change |
|---|---|
| `ui/src/hooks/useManagerChat.js` | Handle `graph_topology`, `node_status`, `node_stream` |
| `ui/src/components/chat/GraphFlowView.js` | **NEW** |
| `ui/src/components/chat/NodeDetailPanel.js` | **NEW** |
| `ui/src/components/chat/TaskPanel.js` | Split-pane when `task.graph` exists |

---

## Wire Protocol

```json
{"type":"graph_topology","task_id":"task-xxx",
 "nodes":[
   {"id":"breakdown","label":"Breakdown","group":null,"status":"completed"},
   {"id":"worker_0","label":"Research core coordination responsibilities","group":"workers","status":"pending"},
   {"id":"worker_1","label":"Analyse planning & reporting scope","group":"workers","status":"pending"},
   {"id":"aggregator","label":"Aggregator","group":null,"status":"pending"}
 ],
 "edges":[
   {"source":"breakdown","target":"worker_0"},
   {"source":"breakdown","target":"worker_1"},
   {"source":"worker_0","target":"aggregator"},
   {"source":"worker_1","target":"aggregator"}
 ],
 "layout":"horizontal"}

{"type":"node_status","task_id":"task-xxx","node_id":"worker_0","status":"running","label":"","error":""}
{"type":"node_status","task_id":"task-xxx","node_id":"worker_0","status":"completed","label":"","error":""}

{"type":"node_stream","task_id":"task-xxx","node_id":"worker_0","content":"Researching...","is_final":false}
{"type":"node_stream","task_id":"task-xxx","node_id":"worker_0","content":"","is_final":true}
```

---

## Verification

1. Trigger `create_role` → diamond graph appears in TaskPanel top after breakdown
2. Workers activate: pending (gray) → running (blue+spinner) in real-time
3. Click running worker → bottom panel streams its live research output
4. Click completed worker → shows final facet output (review mode)
5. Aggregator starts → auto-switches to aggregator in detail panel
6. All complete → all nodes green; `role_document.md` produced
7. Non-BTA task → `task.graph` is null → existing streaming text view (no regression)
8. Existing orchestrator token streaming completely unaffected

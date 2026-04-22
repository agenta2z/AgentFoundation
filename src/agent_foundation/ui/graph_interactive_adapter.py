"""Graph interactive adapter — per-node streaming router for concurrent BTA workers.

The BreakdownThenAggregateInferencer runs N workers in parallel. Each worker needs
to stream tokens tagged with its own node_id for graph visualization. This module
provides:

  NodeStreamInteractive — wraps WebSocketInteractive, intercepts stream_token_batches()
                          to tag tokens with node_id, then delegates to the real WS.

  WebSocketGraphReporter — concrete graph_reporter implementation that bridges BTA
                           graph events to WebSocket send_graph_event() calls.
                           BTA never imports WebSocket code — it calls this via protocol.

Usage in executor.py:
    from agent_foundation.ui.graph_interactive_adapter import WebSocketGraphReporter
    if interactive and task_id:
        inferencer.graph_reporter = WebSocketGraphReporter(interactive, task_id)
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

_logger = logging.getLogger(__name__)


class NodeStreamInteractive:
    """Wraps WebSocketInteractive — tags all streaming with a specific node_id.

    Each BTA worker gets its own instance. When the worker's RovoDevCliInferencer
    calls stream_token_batches(), this wrapper intercepts each (chunk, metadata)
    tuple, emits a node_stream WS event tagged with node_id, then yields the tuple
    to the real WS for batched sending.

    Uses __getattr__ delegation so all other interactive methods (asend_response,
    aget_input, send_turn_boundary, etc.) pass through to the real WS unchanged.

    IMPORTANT: Real WebSocketInteractive.stream_token_batches() signature is:
        (token_stream: AsyncIterator[tuple[str, dict]], session_id: str,
         batch_interval_ms=50.0, task_id=None, send_stream_end=True, turn_number=None)
    The stream yields (chunk_str, metadata_dict) TUPLES — not plain strings.
    """

    def __init__(self, ws_interactive: Any, task_id: str, node_id: str) -> None:
        self._ws = ws_interactive
        self._task_id = task_id
        self._node_id = node_id

    async def stream_token_batches(
        self,
        token_stream: Any,              # AsyncIterator[tuple[str, dict]]
        session_id: str = "",
        batch_interval_ms: float = 50.0,
        task_id: Any = None,
        send_stream_end: bool = True,
        turn_number: Any = None,
        **kwargs: Any,
    ) -> str:
        """Intercept tokens, emit node_stream events, delegate to real WS for batching."""
        from agent_foundation.common.inferencers.graph_events import NodeStreamEvent

        async def _tagged_stream():
            async for chunk, metadata in token_stream:
                try:
                    await self._ws.send_graph_event(
                        NodeStreamEvent(node_id=self._node_id, content=chunk),
                        task_id=self._task_id,
                    )
                except Exception as exc:
                    _logger.warning("[NodeStreamInteractive] send_graph_event failed: %s", exc)
                yield chunk, metadata

        result = await self._ws.stream_token_batches(
            _tagged_stream(), session_id, batch_interval_ms,
            task_id, send_stream_end, turn_number,
        )
        try:
            await self._ws.send_graph_event(
                NodeStreamEvent(node_id=self._node_id, content="", is_final=True),
                task_id=self._task_id,
            )
        except Exception as exc:
            _logger.warning("[NodeStreamInteractive] final node_stream event failed: %s", exc)
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes/methods to the real WS interactive."""
        return getattr(self._ws, name)


class WebSocketGraphReporter:
    """Bridges BTA graph_reporter protocol → WebSocket send_graph_event().

    BTA uses protocol-based graph_reporter so it never imports WebSocket code.
    This class is the concrete implementation that the executor wires up.

    All graph event sends are wrapped in try/except to prevent visualization
    failures from aborting the BTA computation.
    """

    def __init__(self, ws_interactive: Any, task_id: str, max_msg_per_sec: int = 30) -> None:
        self._ws = ws_interactive
        self._task_id = task_id
        self._max_msg_per_sec = max_msg_per_sec
        self._send_times: list[float] = []

    def _check_rate(self) -> bool:
        """Returns True if sending is within rate limit."""
        now = _time.monotonic()
        self._send_times = [t for t in self._send_times if now - t < 1.0]
        if len(self._send_times) >= self._max_msg_per_sec:
            return False
        self._send_times.append(now)
        return True

    async def on_graph_topology(self, event: Any) -> None:
        """Send graph topology event to the frontend."""
        try:
            await self._ws.send_graph_event(event, task_id=self._task_id)
        except Exception as exc:
            _logger.warning("[WebSocketGraphReporter] on_graph_topology failed: %s", exc)

    async def on_node_status(self, node_id: str, status: str, error: str = "", output_path: str = "") -> None:
        """Send a node lifecycle status change to the frontend.

        output_path: absolute path to the node's primary output file (e.g. facet.md).
        The UI fetches this via GET /api/view/{output_path} to show content in NodeDetailPanel.
        Workers → children/worker_N/outputs/facet.md
        Breakdown → checkpoints/breakdown_result.json
        Aggregator → children/aggregator/outputs/role_document.md
        """
        from agent_foundation.common.inferencers.graph_events import NodeStatusEvent
        try:
            await self._ws.send_graph_event(
                NodeStatusEvent(node_id=node_id, status=status, error=error, output_path=output_path),
                task_id=self._task_id,
            )
        except Exception as exc:
            _logger.warning("[WebSocketGraphReporter] on_node_status failed: %s", exc)

    async def on_graph_reconcile(self, node_statuses: dict) -> None:
        """Send final node statuses after graph completes — frontend reconciles gaps."""
        from agent_foundation.common.inferencers.graph_events import GraphReconcileEvent
        try:
            await self._ws.send_graph_event(
                GraphReconcileEvent(node_statuses=node_statuses),
                task_id=self._task_id,
            )
        except Exception as exc:
            _logger.warning("[WebSocketGraphReporter] on_graph_reconcile failed: %s", exc)

    async def on_node_stream(self, node_id: str, content: str, is_final: bool = True) -> None:
        """Emit content for a node (its output text) as a node_stream event.

        is_final events always pass (never throttled). Regular stream chunks
        are rate-limited — excess chunks are silently dropped (the per-node
        node_stream_observer already coalesces at 200ms, so dropped chunks
        here means the global rate is truly exceeded).
        """
        if not is_final and not self._check_rate():
            return
        from agent_foundation.common.inferencers.graph_events import NodeStreamEvent
        try:
            await self._ws.send_graph_event(
                NodeStreamEvent(node_id=node_id, content=content, is_final=is_final),
                task_id=self._task_id,
            )
        except Exception as exc:
            _logger.warning("[WebSocketGraphReporter] on_node_stream failed: %s", exc)

    def node_stream_observer(self, node_id: str, flush_interval_ms: float = 200.0):
        """Returns a batching stream observer for a node.

        Accumulates chunks and flushes every flush_interval_ms as one node_stream
        event. This prevents flooding the WebSocket with per-token messages.
        The main conversation uses stream_token_batches(50ms) for the same purpose.

        Set on the inferencer before ainfer():
            worker.stream_observer = reporter.node_stream_observer(f"worker_{i}")

        Args:
            node_id: The graph node ID (e.g., "worker_0", "breakdown", "aggregator").
            flush_interval_ms: Minimum interval between WS sends. Default 200ms.

        Returns:
            An async callable that accepts (chunk: str) and batches before sending.
        """
        from agent_foundation.common.inferencers.graph_events import NodeStreamEvent

        _batch: list[str] = []
        _last_flush = [_time.monotonic()]
        _ws = self._ws
        _task_id = self._task_id

        async def _observer(chunk: str) -> None:
            _batch.append(chunk)
            now = _time.monotonic()
            if (now - _last_flush[0]) * 1000 >= flush_interval_ms:
                content = "".join(_batch)
                _batch.clear()
                _last_flush[0] = now
                try:
                    await _ws.send_graph_event(
                        NodeStreamEvent(node_id=node_id, content=content),
                        task_id=_task_id,
                    )
                except Exception as exc:
                    _logger.warning(
                        "[WebSocketGraphReporter] node_stream_observer flush failed: %s", exc
                    )

        return _observer

    def node_interactive(self, node_id: str) -> "NodeStreamInteractive":
        """Returns a per-node interactive wrapper for tagged token streaming.

        Set this on the worker inferencer BEFORE _make_worker_fn() captures it:
            worker.interactive = reporter.node_interactive(f"worker_{i}")
        """
        return NodeStreamInteractive(self._ws, self._task_id, node_id)

    def child_reporter(self, parent_node_id: str) -> "NamespacedGraphReporter":
        """Returns a scoped reporter for a nested BTA.

        The child reporter prefixes all node IDs with ``parent_node_id/``
        and tags sub-graph topology events with ``parent_node_id`` so the
        UI can splice them under the correct container node.
        """
        return NamespacedGraphReporter(self, parent_node_id)


class NamespacedGraphReporter:
    """Wraps a parent reporter; prefixes all node_ids with ``parent_node_id/``.

    Used to give inner BTAs their own reporter that automatically scopes
    events into a sub-graph identified by ``parent_node_id``.

    Composable: ``child_reporter()`` produces a deeper-nested reporter
    with composed prefixes (e.g., ``worker_0/worker_2/worker_1``).
    """

    def __init__(self, parent: "WebSocketGraphReporter | NamespacedGraphReporter", parent_node_id: str) -> None:
        self._parent = parent
        self._parent_node_id = parent_node_id

    def _qualify(self, node_id: str) -> str:
        return f"{self._parent_node_id}/{node_id}" if self._parent_node_id else node_id

    async def on_graph_topology(self, event: Any) -> None:
        from dataclasses import replace as _replace
        nodes = [{**n, "id": self._qualify(n["id"])} for n in event.nodes]
        edges = [{"source": self._qualify(e["source"]),
                  "target": self._qualify(e["target"])} for e in event.edges]
        new_evt = _replace(event, nodes=nodes, edges=edges,
                           parent_node_id=self._parent_node_id)
        await self._parent.on_graph_topology(new_evt)

    async def on_node_status(self, node_id: str, status: str, error: str = "", output_path: str = "") -> None:
        await self._parent.on_node_status(self._qualify(node_id), status, error=error, output_path=output_path)

    async def on_node_stream(self, node_id: str, content: str, is_final: bool = True) -> None:
        await self._parent.on_node_stream(self._qualify(node_id), content, is_final=is_final)

    def node_stream_observer(self, node_id: str, flush_interval_ms: float = 200.0):
        return self._parent.node_stream_observer(self._qualify(node_id), flush_interval_ms=flush_interval_ms)

    def node_interactive(self, node_id: str) -> "NodeStreamInteractive":
        return self._parent.node_interactive(self._qualify(node_id))

    def child_reporter(self, parent_node_id: str) -> "NamespacedGraphReporter":
        return NamespacedGraphReporter(self._parent, self._qualify(parent_node_id))

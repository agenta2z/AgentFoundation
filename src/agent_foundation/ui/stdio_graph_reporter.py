"""StdioGraphReporter — duck-typed peer of WebSocketGraphReporter.

Emits the same 4 event types as NDJSON on a writeable text stream
(typically fd=3 of a child process). Designed for subprocess-based UIs
that launch openteam-task as a child and want live topology events.

Same 7-member surface as WebSocketGraphReporter (4 async + 3 factory).
All sends try/except wrapped — visualization NEVER aborts computation.
asyncio.Lock serializes concurrent BTA worker writes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, IO, Optional

from agent_foundation.ui.graph_interactive_adapter import NamespacedGraphReporter

_logger = logging.getLogger(__name__)

_ENV_FD = "ROVODEV_TUI_GRAPH_FD"
_MAX_LINE_BYTES = 4000


def _activated_fd() -> Optional[int]:
    raw = os.environ.get(_ENV_FD)
    if not raw:
        return None
    try:
        fd = int(raw)
    except ValueError:
        return None
    try:
        os.fstat(fd)
    except OSError:
        return None
    return fd


def _serialize(event: Any, task_id: str) -> dict:
    from agent_foundation.common.inferencers.graph_events import (
        GraphTopologyEvent, NodeStatusEvent, NodeStreamEvent, GraphReconcileEvent,
    )
    if isinstance(event, GraphTopologyEvent):
        msg = {
            "type": "graph_topology", "task_id": task_id,
            "nodes": event.nodes, "edges": event.edges, "layout": event.layout,
        }
        if event.parent_node_id:
            msg["parent_node_id"] = event.parent_node_id
        if event.version:
            msg["version"] = event.version
        return msg
    if isinstance(event, NodeStatusEvent):
        return {
            "type": "node_status", "task_id": task_id,
            "node_id": event.node_id, "status": event.status,
            "label": event.label, "error": event.error,
            "timestamp": event.timestamp, "output_path": event.output_path,
        }
    if isinstance(event, NodeStreamEvent):
        return {
            "type": "node_stream", "task_id": task_id,
            "node_id": event.node_id, "content": event.content,
            "is_final": event.is_final,
        }
    if isinstance(event, GraphReconcileEvent):
        return {
            "type": "graph_reconcile", "task_id": task_id,
            "nodes": event.node_statuses,
        }
    if is_dataclass(event):
        d = asdict(event)
        d.setdefault("type", type(event).__name__)
        d.setdefault("task_id", task_id)
        return d
    raise TypeError(f"Cannot serialize event of type {type(event).__name__}")


class StdioGraphReporter:
    """NDJSON graph reporter — writes one event per line to a text stream."""

    _FROM_ENV_CACHE: dict[int, "StdioGraphReporter"] = {}

    def __init__(self, task_id: str, stream: IO[str], *,
                 max_msg_per_sec: int = 30) -> None:
        self._task_id = task_id
        self._stream = stream
        self._max_msg_per_sec = max_msg_per_sec
        self._send_times: list[float] = []
        self._lock = asyncio.Lock()
        self._fd_identity: Optional[tuple[int, int]] = None

    @classmethod
    def from_env(cls, task_id: str = "") -> Optional["StdioGraphReporter"]:
        """Construct from ROVODEV_TUI_GRAPH_FD env var; returns None if absent.

        Idempotent: caches per fd with inode-based invalidation to handle
        fd recycling across test runs / process restarts.
        """
        fd = _activated_fd()
        if fd is None:
            return None
        if fd in cls._FROM_ENV_CACHE:
            cached = cls._FROM_ENV_CACHE[fd]
            try:
                st = os.fstat(fd)
                identity = (st.st_dev, st.st_ino)
            except OSError:
                identity = None
            if identity is None or identity != cached._fd_identity or cached._stream.closed:
                del cls._FROM_ENV_CACHE[fd]
            else:
                return cached
        try:
            stream = os.fdopen(fd, "w", buffering=1, encoding="utf-8")
        except OSError as exc:
            _logger.warning("[StdioGraphReporter.from_env] fdopen(%d) failed: %s", fd, exc)
            return None
        instance = cls(task_id=task_id or f"task-{os.getpid()}", stream=stream)
        try:
            st = os.fstat(fd)
            instance._fd_identity = (st.st_dev, st.st_ino)
        except OSError:
            instance._fd_identity = None
        cls._FROM_ENV_CACHE[fd] = instance
        return instance

    async def _emit(self, msg: dict) -> None:
        try:
            line = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            _logger.warning("[StdioGraphReporter] json.dumps failed: %s", exc)
            return
        async with self._lock:
            try:
                if len(line.encode("utf-8")) > _MAX_LINE_BYTES and msg.get("type") == "node_stream":
                    self._write_chunked_stream(msg)
                else:
                    self._stream.write(line + "\n")
                    self._stream.flush()
            except BrokenPipeError:
                pass
            except OSError as exc:
                _logger.debug("[StdioGraphReporter] write failed: %s", exc)

    def _write_chunked_stream(self, msg: dict) -> None:
        content = msg.get("content", "")
        is_final = msg.get("is_final", False)
        chunk_size = 3000
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        for i, chunk in enumerate(chunks):
            sub = dict(msg)
            sub["content"] = chunk
            sub["continuation"] = True
            sub["is_final"] = is_final and (i == len(chunks) - 1)
            try:
                self._stream.write(json.dumps(sub, separators=(",", ":"), ensure_ascii=False) + "\n")
            except (BrokenPipeError, OSError):
                return
        try:
            self._stream.flush()
        except (BrokenPipeError, OSError):
            pass

    def _check_rate(self) -> bool:
        now = time.monotonic()
        self._send_times = [t for t in self._send_times if now - t < 1.0]
        if len(self._send_times) >= self._max_msg_per_sec:
            return False
        self._send_times.append(now)
        return True

    # ── graph_reporter protocol (4 async methods) ────────────────────────

    async def on_graph_topology(self, event: Any) -> None:
        try:
            await self._emit(_serialize(event, self._task_id))
        except Exception as exc:
            _logger.warning("[StdioGraphReporter] on_graph_topology failed: %s", exc)

    async def on_node_status(self, node_id: str, status: str,
                             error: str = "", output_path: str = "") -> None:
        from agent_foundation.common.inferencers.graph_events import NodeStatusEvent
        try:
            await self._emit(_serialize(
                NodeStatusEvent(node_id=node_id, status=status,
                                error=error, output_path=output_path),
                self._task_id,
            ))
        except Exception as exc:
            _logger.warning("[StdioGraphReporter] on_node_status failed: %s", exc)

    async def on_node_stream(self, node_id: str, content: str,
                             is_final: bool = True) -> None:
        if not is_final and not self._check_rate():
            return
        from agent_foundation.common.inferencers.graph_events import NodeStreamEvent
        try:
            await self._emit(_serialize(
                NodeStreamEvent(node_id=node_id, content=content, is_final=is_final),
                self._task_id,
            ))
        except Exception as exc:
            _logger.warning("[StdioGraphReporter] on_node_stream failed: %s", exc)

    async def on_graph_reconcile(self, node_statuses: dict) -> None:
        from agent_foundation.common.inferencers.graph_events import GraphReconcileEvent
        try:
            await self._emit(_serialize(
                GraphReconcileEvent(node_statuses=node_statuses),
                self._task_id,
            ))
        except Exception as exc:
            _logger.warning("[StdioGraphReporter] on_graph_reconcile failed: %s", exc)

    # ── factory methods (3) ──────────────────────────────────────────────

    def node_stream_observer(self, node_id: str,
                             flush_interval_ms: float = 200.0) -> Callable:
        _batch: list[str] = []
        _last_flush = [time.monotonic()]

        async def _observer(chunk: str) -> None:
            _batch.append(chunk)
            now = time.monotonic()
            if (now - _last_flush[0]) * 1000 >= flush_interval_ms:
                content = "".join(_batch)
                _batch.clear()
                _last_flush[0] = now
                if content:
                    await self.on_node_stream(node_id, content, is_final=False)

        return _observer

    def node_interactive(self, node_id: str) -> Any:
        return _StdioNodeInteractive(self, node_id)

    def child_reporter(self, parent_node_id: str) -> NamespacedGraphReporter:
        return NamespacedGraphReporter(self, parent_node_id)


class _StdioNodeInteractive:
    """Stub for BTA's worker.interactive when no parent WS exists."""

    _ASYNC_NOOP_NAMES = frozenset({
        "send_graph_event", "on_clean_output_available", "stream_token_batches",
        "send_task_status", "send_turn_boundary", "asend_response", "aget_input",
    })

    def __init__(self, parent: StdioGraphReporter, node_id: str) -> None:
        self._parent = parent
        self._node_id = node_id

    async def stream_token_batches(
        self, token_stream: Any, session_id: str = "",
        batch_interval_ms: float = 50.0, task_id: Any = None,
        send_stream_end: bool = True, turn_number: Any = None, **kwargs: Any,
    ) -> str:
        out: list[str] = []
        async for chunk, _meta in token_stream:
            out.append(chunk)
            try:
                await self._parent.on_node_stream(self._node_id, chunk, is_final=False)
            except Exception:
                pass
        try:
            await self._parent.on_node_stream(self._node_id, "", is_final=True)
        except Exception:
            pass
        return "".join(out)

    def __getattr__(self, name: str) -> Any:
        if name in self._ASYNC_NOOP_NAMES:
            async def _async_noop(*args: Any, **kwargs: Any) -> Any:
                return None
            return _async_noop

        def _sync_noop(*args: Any, **kwargs: Any) -> Any:
            return None
        return _sync_noop

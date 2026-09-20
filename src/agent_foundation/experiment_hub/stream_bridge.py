# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Adapter bridging push-based StreamCallback to pull-based AsyncIterator.

Ported faithfully from RankEvolve's ``server.stream_bridge`` (no rankevolve
deps — pure asyncio/stdlib).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator
from typing import Any


logger: logging.Logger = logging.getLogger(__name__)


class StreamBridgeAdapter:
    """Converts a push-based StreamCallback into a pull-based
    AsyncIterator[tuple[str, dict]] that the display layer can iterate.

    Usage:
        adapter = StreamBridgeAdapter()

        async def producer():
            await orchestrator.run(..., stream_callback=adapter.callback)
            await adapter.close()

        async def consumer():
            async for chunk, metadata in adapter:
                display.update(chunk)

        await asyncio.gather(producer(), consumer())

    Design notes:
    - Uses asyncio.Queue (not locks) — idiomatic for single event loop
    - Backpressure: unbounded queue; the display's refresh throttles redraws
    - close() is idempotent — safe to call multiple times
    - The adapter handles chunks from multiple sequential phases without reset
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        self._closed: bool = False
        self._token_count: int = 0
        self._start_time: float = time.monotonic()

    async def callback(self, chunk: str, metadata: dict[str, Any]) -> None:
        """StreamCallback-compatible push interface.

        Filters: only enqueues non-empty text chunks for display.
        Empty chunks (metadata-only events like "step", "tool_use", "error")
        are skipped.
        """
        if chunk:
            self._token_count += 1
            if self._token_count == 1:
                elapsed = time.monotonic() - self._start_time
                msg = (
                    f"[stream_bridge] First token received after {elapsed:.1f}s "
                    f"(phase={metadata.get('phase', '?')}, "
                    f"agent={metadata.get('agent_id', '?')}, "
                    f"chunk_len={len(chunk)})\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                logger.info(
                    "First streaming token received (phase=%s, agent=%s)",
                    metadata.get("phase", "?"),
                    metadata.get("agent_id", "?"),
                )
            elif self._token_count % 100 == 0:
                sys.stderr.write(
                    f"[stream_bridge] Tokens enqueued: {self._token_count}, "
                    f"queue_size={self._queue.qsize()}\n"
                )
                sys.stderr.flush()
            await self._queue.put((chunk, metadata))
            # Yield to event loop so the consumer can process items.
            # Without this, the producer (reading from subprocess stdout
            # in a tight loop) starves the consumer coroutine.
            await asyncio.sleep(0)

    async def close(self) -> None:
        """Signal end-of-stream. Must be called when producer finishes.
        Idempotent — safe to call multiple times."""
        if not self._closed:
            self._closed = True
            elapsed = time.monotonic() - self._start_time
            msg = (
                f"[stream_bridge] Closing after {elapsed:.1f}s. "
                f"Total tokens: {self._token_count}\n"
            )
            sys.stderr.write(msg)
            sys.stderr.flush()
            logger.info(
                "Stream bridge closing. Total tokens received: %d",
                self._token_count,
            )
            await self._queue.put(None)

    def __aiter__(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        return self

    async def __anext__(self) -> tuple[str, dict[str, Any]]:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

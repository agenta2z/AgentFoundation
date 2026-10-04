"""Closing httpx's streamed line iterator for good."""

from __future__ import annotations

from contextlib import aclosing, asynccontextmanager, suppress
from typing import AsyncIterator

import httpx


@asynccontextmanager
async def closing_lines(
    response: httpx.Response,
) -> AsyncIterator[AsyncIterator[str]]:
    """``response.aiter_lines()``, finished however the block exits.

    The line iterator sits on a chain of nested async generators (text, bytes,
    raw, then the transport's own) that httpx never closes. A chain left
    mid-stream (a ``break`` or ``return``, a closed or cancelled consumer) is
    reclaimed only by the cyclic garbage collector, whose asyncio finalizer then
    schedules one ``aclose()`` task per generator at an arbitrary later point
    (``later.unittest`` reports those as leaked tasks). On exit the response is
    closed first, so resuming the chain ends it at once: what is already
    buffered is read and dropped, then the closed transport ends the stream or
    fails it, and that failure is dropped too.
    """
    lines = response.aiter_lines()
    async with aclosing(lines):
        try:
            yield lines
        finally:
            await response.aclose()
            with suppress(httpx.HTTPError, httpx.StreamError):
                async for _ in lines:
                    pass

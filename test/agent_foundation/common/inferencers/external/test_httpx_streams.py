"""The RovoChat and RovoDev serve streams leave nothing to the garbage collector.

httpx's line iterator sits on nested async generators that httpx never closes.
Stopping mid-body (RovoChat at its final response, RovoDev serve at the end of
the run, a consumer that stops early or is cancelled) must not leave them
suspended for asyncio's finalizer to close from tasks at an arbitrary later
point (``later.unittest`` reports those tasks as leaked).
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import tempfile
from contextlib import aclosing
from typing import AsyncIterator
from unittest import mock

import httpx
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.auth import (
    RovoChatAuth,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.client import (
    RovoChatClient,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.rovochat_inferencer import (
    RovoChatInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.types import (
    RovoChatConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
    rovodev_serve_inferencer as serve_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_serve_inferencer import (
    RovoDevServeInferencer,
)
from later.unittest import TestCase


class _Body(httpx.AsyncByteStream):
    """A streamed body, one part per chunk; once closed, reading on fails the
    way a closed connection does."""

    def __init__(self, parts: list[str]) -> None:
        self._parts = [part.encode() for part in parts]
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self._parts:
            if self.closed:
                raise httpx.ReadError("connection closed")
            yield part

    async def aclose(self) -> None:
        self.closed = True


class _AbandonedGenerators:
    """The async generators first iterated inside the block that the garbage
    collector later found suspended (handed to asyncio's finalizer)."""

    def __enter__(self) -> list[str]:
        self._names: list[str] = []
        self._hooks = sys.get_asyncgen_hooks()

        def finalizer(agen) -> None:
            self._names.append(agen.__qualname__)
            self._hooks.finalizer(agen)

        sys.set_asyncgen_hooks(self._hooks.firstiter, finalizer)
        return self._names

    def __exit__(self, *exc_info) -> None:
        gc.collect()
        sys.set_asyncgen_hooks(*self._hooks)


class _StreamTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.bodies: list[_Body] = []
        real_client = httpx.AsyncClient

        def client(*args, **kwargs) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(self.handle)
            return real_client(*args, **kwargs)

        self.enterContext(mock.patch.object(httpx, "AsyncClient", client))

    def handle(self, request: httpx.Request) -> httpx.Response:
        raise NotImplementedError

    def stream(self, parts: list[str], content_type: str) -> httpx.Response:
        body = _Body(parts)
        self.bodies.append(body)
        return httpx.Response(200, stream=body, headers={"Content-Type": content_type})

    def assert_closed(self) -> None:
        self.assertTrue(self.bodies)
        self.assertTrue(all(body.closed for body in self.bodies))


class RovoChatStreamTest(_StreamTestCase):
    def handle(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/stream"):
            return httpx.Response(200, json={"id": "conv-1"})
        events = [
            {"type": "RECONNECT_SUPPORTED"},
            {"type": "ANSWER_PART", "message": {"content": "Hel"}},
            {"type": "ANSWER_PART", "message": {"content": "Hello"}},
            {"type": "FINAL_RESPONSE", "message": {"content": "Hello."}},
        ]
        return self.stream(
            [json.dumps(event) + "\n" for event in events], "application/x-ndjson"
        )

    def client(self) -> RovoChatClient:
        return RovoChatClient(
            config=RovoChatConfig(base_url="https://rovo.example.test"),
            auth=RovoChatAuth(uct_token="uct"),
        )

    async def test_the_inferencer_stopping_at_the_final_response(self) -> None:
        inf = RovoChatInferencer(
            base_url="https://rovo.example.test", cloud_id="cloud-1", uct_token="uct"
        )
        with _AbandonedGenerators() as abandoned:
            reply = await inf.ainfer("hi")

        self.assertIn("Hello.", str(reply))
        self.assertEqual(abandoned, [])
        self.assert_closed()

    async def test_a_consumer_that_stops_early(self) -> None:
        with _AbandonedGenerators() as abandoned:
            async with aclosing(self.client().send_message_stream("c", "hi")) as events:
                async for event in events:
                    break

        self.assertEqual(event.event_type, "RECONNECT_SUPPORTED")
        self.assertEqual(abandoned, [])
        self.assert_closed()

    async def test_a_consumer_cancelled_between_events(self) -> None:
        first = asyncio.Event()

        async def consume() -> None:
            async with aclosing(self.client().send_message_stream("c", "hi")) as events:
                async for _ in events:
                    first.set()
                    await asyncio.Event().wait()

        with _AbandonedGenerators() as abandoned:
            task = asyncio.create_task(consume())
            await first.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(abandoned, [])
        self.assert_closed()


class _FakeServer:
    """The ``acli rovodev serve`` process: running until signalled."""

    pid = 4242
    stderr = None
    returncode = None

    def send_signal(self, sig: int) -> None:
        self.returncode = -sig

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode


class RovoDevServeStreamTest(_StreamTestCase):
    def setUp(self) -> None:
        super().setUp()

        async def spawn(*cmd, **kwargs) -> _FakeServer:
            return _FakeServer()

        for target, name, value in (
            (serve_module, "find_acli_binary", lambda path: "acli"),
            (serve_module, "find_available_port", lambda: 8001),
            (asyncio, "create_subprocess_exec", spawn),
        ):
            self.enterContext(mock.patch.object(target, name, value))
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.inf = RovoDevServeInferencer(acli_path="acli", target_path=tmp)

    async def asyncTearDown(self) -> None:
        await self.inf.adisconnect()
        await super().asyncTearDown()

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v3/stream_chat":
            return httpx.Response(200, json={"status": "ok"})
        return self.stream(
            [
                "event: tool_call_start\ndata: {}\n\n",
                'event: text_delta\ndata: {"delta": "Hel"}\n\n',
                'event: text_delta\ndata: {"delta": "lo."}\n\n',
                "event: agent_run_end\ndata: {}\n\n",
            ],
            "text/event-stream",
        )

    async def test_the_end_of_the_run(self) -> None:
        with _AbandonedGenerators() as abandoned:
            reply = await self.inf.ainfer("hi")

        self.assertIn("Hello.", str(reply))
        self.assertEqual(abandoned, [])
        self.assert_closed()

    async def test_a_consumer_that_stops_early(self) -> None:
        with _AbandonedGenerators() as abandoned:
            async with aclosing(self.inf.ainfer_streaming("hi")) as chunks:
                async for chunk in chunks:
                    break

        self.assertEqual(chunk, "Hel")
        self.assertEqual(abandoned, [])
        self.assert_closed()

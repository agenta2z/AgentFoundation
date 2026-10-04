"""Closing or failing a conversational stream closes the leaf stream it consumes.

``async for`` never closes the generator it iterates: when the consumer is
closed or the delivery sink raises, the leaf stream stays suspended until GC
finalizes it, from a different Context. Each layer that creates a stream
therefore closes it (``contextlib.aclosing``), so the leaf's ``finally`` runs
synchronously in the consumer's own Context.
"""

import asyncio

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.ui.graph_interactive_adapter import NodeStreamInteractive
from attr import attrs


@attrs(slots=False)
class _RecordingStreamBase(InferencerBase):
    """Streams two chunks and records when its stream is finalized."""

    def _setup_record(self):
        self.stream_closed = False

    def _infer(self, inp, cfg=None, **kw):
        return "unused"

    async def _ainfer(self, inp, cfg=None, **kw):
        return "unused"

    async def ainfer_streaming(self, inp, cfg=None, *, run_context=None, **kw):
        try:
            yield "first "
            yield "second"
        finally:
            self.stream_closed = True


class _FailingSinkInteractive:
    """Consumes one token, then fails like a cancelled websocket send."""

    def __init__(self, exc_type):
        self._exc_type = exc_type

    async def stream_token_batches(self, token_stream, session_id, **kwargs):
        async for _chunk, _metadata in token_stream:
            raise self._exc_type("send failed")
        return ""


def _make_ci():
    base = _RecordingStreamBase()
    base._setup_record()
    return base, ConversationalInferencer(base_inferencer=base, max_iterations=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_type", [RuntimeError, asyncio.CancelledError])
async def test_failed_delivery_closes_leaf_stream(exc_type):
    base, ci = _make_ci()

    with pytest.raises(exc_type) as excinfo:
        await ci.run_agentic_loop(
            "go", interactive=_FailingSinkInteractive(exc_type), session_id="s"
        )

    assert excinfo.value is not None
    assert base.stream_closed


@pytest.mark.asyncio
async def test_closing_passthrough_stream_closes_leaf_stream():
    base, ci = _make_ci()

    stream = ci.ainfer_streaming("hi")
    assert await stream.__anext__() == "first "
    await stream.aclose()

    assert base.stream_closed


class _RecordingWs:
    """Real-WS stand-in: fails mid-stream, records its tagged stream."""

    def __init__(self):
        self.tagged = None

    async def send_graph_event(self, event, task_id=None):
        return None

    async def stream_token_batches(self, token_stream, *args, **kwargs):
        self.tagged = token_stream
        async for _chunk, _metadata in token_stream:
            raise RuntimeError("send failed")
        return ""


@pytest.mark.asyncio
async def test_node_stream_interactive_closes_its_tagged_stream():
    ws = _RecordingWs()
    node = NodeStreamInteractive(ws, task_id="t", node_id="n")
    upstream_closed = []

    async def upstream():
        try:
            yield "a", {}
            yield "b", {}
        finally:
            upstream_closed.append(True)

    source = upstream()
    with pytest.raises(RuntimeError):
        await node.stream_token_batches(source, "s")

    assert ws.tagged.ag_frame is None
    assert upstream_closed == []
    await source.aclose()
    assert upstream_closed == [True]

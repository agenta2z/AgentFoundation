"""The streaming entries' invocations: ``framed_agen`` / ``framed_gen`` (plan v8
§5.1 streaming binding rule, P3 c7).

A public streaming entry opens its invocation at the first resumption: it
resolves the ctx once, claims the path, runs ``_init_call_state`` and picks the
fan-out or the pipeline inside the frame. The ctx and the frame are bound only
while the pipeline runs, so the transport sees them and the consumer never does.
Only an exhausted stream publishes an outcome; the claim is held until the
stream is exhausted, closed or finalized.
"""

from __future__ import annotations

import asyncio
import gc
from typing import Any

import pytest
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    ConcurrentInvocationError,
    enter_run,
    exit_run,
    frame_for,
    framed_agen,
    framed_gen,
    InvocationCleanupError,
    NodeOutcomeState,
    open_invocation,
    read_outcome,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrib, attrs

KINDS = ("sync", "async")


class _Owner:
    pass


class _FailingClose:
    def close(self):
        raise OSError("pipe")


@attrs
class _Leaf(StreamingInferencerBase):
    """Records, inside its transport, the ctx and the frame it runs under."""

    seen: list = attrib(factory=list, kw_only=True)
    fail_after: Any = attrib(default=None, kw_only=True)
    publishes: bool = attrib(default=False, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "abc"

    async def _ainfer_streaming(self, prompt, **kwargs):
        try:
            for index, chunk in enumerate(("a", "b", "c")):
                if self.fail_after is not None and index == self.fail_after:
                    raise RuntimeError("transport failed")
                frame = frame_for(self)
                self.seen.append((active_run_context(), frame and frame.entry))
                yield chunk
                await asyncio.sleep(0)
        finally:
            self.seen.append(("finally", active_run_context()))

    def _outcome_for(self, frame):
        return NodeOutcomeState(final_output="done") if self.publishes else None


def _stream(leaf, kind, **kwargs):
    if kind == "sync":
        return leaf.infer_streaming("q", **kwargs)
    return leaf.ainfer_streaming("q", **kwargs)


def _drain(leaf, kind, **kwargs):
    if kind == "sync":
        return list(_stream(leaf, kind, **kwargs))

    async def main():
        return [chunk async for chunk in _stream(leaf, kind, **kwargs)]

    return asyncio.run(main())


def _host():
    return RunContext.root().child("leaf")


# -- binding ------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_the_transport_sees_the_ctx_and_the_frame_the_consumer_does_not(kind):
    leaf, ctx = _Leaf(), _host()
    consumer_view = []
    if kind == "sync":
        for _ in leaf.infer_streaming("q", run_context=ctx):
            consumer_view.append((active_run_context(), frame_for(leaf)))
    else:

        async def main():
            async for _ in leaf.ainfer_streaming("q", run_context=ctx):
                consumer_view.append((active_run_context(), frame_for(leaf)))

        asyncio.run(main())
    entry = "infer_streaming" if kind == "sync" else "ainfer_streaming"
    assert leaf.seen[:3] == [(ctx, entry)] * 3
    assert leaf.seen[3] == ("finally", ctx)
    assert consumer_view == [(None, None)] * 3


def test_each_resumption_binds_in_the_resuming_task():
    leaf, ctx = _Leaf(), _host()

    async def main():
        stream = leaf.ainfer_streaming("q", run_context=ctx)

        async def step():
            return await stream.__anext__(), active_run_context()

        first = await asyncio.create_task(step())
        second = await asyncio.create_task(step())
        await stream.aclose()
        return first, second, active_run_context()

    first, second, after = asyncio.run(main())
    assert (first, second, after) == (("a", None), ("b", None), None)
    assert [seen for seen, _ in leaf.seen[:2]] == [ctx, ctx]


@pytest.mark.parametrize("kind", KINDS)
def test_a_bare_stream_mints_one_root_for_its_whole_life(kind):
    leaf = _Leaf()
    assert "".join(_drain(leaf, kind)) == "abc"
    roots = {id(seen) for seen, _ in leaf.seen[:3]}
    assert len(roots) == 1
    assert leaf.seen[0][0].legacy_mint
    assert active_run_context() is None


# -- claim --------------------------------------------------------------------


def test_an_open_stream_blocks_a_same_path_call_until_it_is_closed():
    leaf, ctx = _Leaf(), _host()

    async def main():
        stream = leaf.ainfer_streaming("q", run_context=ctx)
        assert await stream.__anext__() == "a"
        with pytest.raises(ConcurrentInvocationError, match="aclosing"):
            await _Leaf().ainfer("q", run_context=ctx)
        await stream.aclose()
        return await _Leaf().ainfer("q", run_context=ctx)

    assert asyncio.run(main()) == "abc"


def test_an_open_sync_stream_blocks_a_same_path_call_until_it_is_closed():
    leaf, ctx = _Leaf(), _host()
    stream = leaf.infer_streaming("q", run_context=ctx)
    assert next(stream) == "a"
    with pytest.raises(ConcurrentInvocationError, match="closing"):
        _Leaf().infer("q", run_context=ctx)
    stream.close()
    assert _Leaf().infer("q", run_context=ctx) == "abc"


@pytest.mark.parametrize("kind", KINDS)
def test_a_stream_never_iterated_holds_no_claim(kind):
    leaf, ctx = _Leaf(), _host()
    stream = _stream(leaf, kind, run_context=ctx)
    assert _Leaf().infer("q", run_context=ctx) == "abc"
    assert leaf.seen == []
    del stream


def test_an_abandoned_stream_releases_its_claim_when_finalized():
    leaf, ctx = _Leaf(), _host()

    async def main():
        stream = leaf.ainfer_streaming("q", run_context=ctx)
        async for _ in stream:
            break
        with pytest.raises(ConcurrentInvocationError):
            await _Leaf().ainfer("q", run_context=ctx)
        del stream
        gc.collect()
        for _ in range(20):
            await asyncio.sleep(0)
        return await _Leaf().ainfer("q", run_context=ctx)

    assert asyncio.run(main()) == "abc"


@pytest.mark.parametrize("kind", KINDS)
def test_a_rejected_stream_runs_nothing(kind):
    inputs = []
    leaf, ctx = _Leaf(state_factory=lambda x: inputs.append(x) or {}), _host()
    token = enter_run(ctx)
    try:
        with open_invocation(_Owner()):
            with pytest.raises(ConcurrentInvocationError):
                _drain(leaf, kind, run_context=ctx)
    finally:
        exit_run(token)
    assert leaf.seen == []
    assert inputs == []


@pytest.mark.parametrize("kind", KINDS)
def test_closing_a_stream_early_closes_its_transport_before_returning(kind):
    leaf, ctx = _Leaf(), _host()
    stream = _stream(leaf, kind, run_context=ctx)
    if kind == "sync":
        assert next(stream) == "a"
        stream.close()
    else:

        async def main():
            assert await stream.__anext__() == "a"
            await stream.aclose()
            return list(leaf.seen)

        leaf.seen = asyncio.run(main())
    assert leaf.seen[-1] == ("finally", ctx)


# -- per-call state and outcome -----------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_stream_initializes_its_call_state_inside_the_claim(kind):
    inputs = []
    leaf, ctx = _Leaf(state_factory=lambda x: inputs.append(x) or {}), _host()
    _drain(leaf, kind, run_context=ctx)
    assert inputs == ["q"]
    assert ctx.store.peek(ctx.path).call == {}


@pytest.mark.parametrize("kind", KINDS)
def test_only_an_exhausted_stream_publishes_its_outcome(kind):
    leaf, ctx = _Leaf(publishes=True), _host()
    _drain(leaf, kind, run_context=ctx)
    assert read_outcome(ctx).final_output == "done"

    stream = _stream(leaf, kind, run_context=ctx)
    if kind == "sync":
        next(stream)
        stream.close()
    else:

        async def main():
            await stream.__anext__()
            await stream.aclose()

        asyncio.run(main())
    assert read_outcome(ctx) is None


@pytest.mark.parametrize("kind", KINDS)
def test_a_failed_stream_publishes_nothing_and_raises_unchanged(kind):
    leaf, ctx = _Leaf(publishes=True, fail_after=1), _host()
    with pytest.raises(RuntimeError, match="transport failed"):
        _drain(leaf, kind, run_context=ctx)
    assert read_outcome(ctx) is None
    assert _Leaf().infer("q", run_context=ctx) == "abc"


# -- the wrappers on their own --------------------------------------------------


def test_framed_agen_closes_the_ledger_after_the_stream_and_raises_cleanup():
    owner, ctx = _Owner(), _host()

    async def inner():
        frame_for(owner).ledger.register(_FailingClose(), "pipe")
        yield 1

    async def main():
        stream = framed_agen(owner, "ainfer_streaming", ctx, inner)
        return [item async for item in stream]

    with pytest.raises(InvocationCleanupError, match="pipe: OSError: pipe"):
        asyncio.run(main())


def test_framed_gen_closes_its_inner_stream_on_early_close():
    owner, ctx, events = _Owner(), _host(), []

    def inner():
        try:
            yield 1
            yield 2
        finally:
            events.append(("inner finally", active_run_context()))

    stream = framed_gen(owner, "infer_streaming", ctx, inner)
    assert next(stream) == 1
    stream.close()
    assert events == [("inner finally", ctx)]
    assert active_run_context() is None


def test_framed_agen_closes_its_inner_stream_on_early_close():
    owner, ctx, events = _Owner(), _host(), []

    async def inner():
        try:
            yield 1
            yield 2
        finally:
            events.append(("inner finally", active_run_context()))

    async def main():
        stream = framed_agen(owner, "ainfer_streaming", ctx, inner)
        assert await stream.__anext__() == 1
        await stream.aclose()

    asyncio.run(main())
    assert events == [("inner finally", ctx)]

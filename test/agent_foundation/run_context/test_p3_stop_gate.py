"""P3 stop gate (plan v8 §11 P3): every entry gets a correct, unique frame.

The stub records, inside its transport, the frame ``frame_for(self)`` resolves:
its owner, entry, ctx path, mode and invocation id. Each test drives one entry
kind and checks that the transport ran inside a frame of its own invocation, at
the right path, distinct from every other invocation's. The streaming entries,
the CLI adapters and OpenClaw are covered by ``test_streaming_entry_contract.py``,
and abandoned streams and lazy iterators by ``test_framed_streams.py`` and
``test_lazy_infer_iterator.py``.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    ConcurrentInvocationError,
    frame_for,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrib, attrs


def _view(owner):
    frame = frame_for(owner)
    ctx = active_run_context()
    if frame is None:
        return None
    return {
        "entry": frame.entry,
        "path": frame.ctx.path if frame.ctx is not None else None,
        "mode": frame.mode,
        "ctx_is_frame_ctx": frame.ctx is ctx,
        "id": frame.invocation_id,
        "parent_owner": None if frame.parent is None else frame.parent.owner,
    }


@attrs
class _Leaf(InferencerBase):
    seen: list = attrib(factory=list, kw_only=True)
    lock: threading.Lock = attrib(factory=threading.Lock, kw_only=True)
    gate: object = attrib(default=None, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        with self.lock:
            self.seen.append((inference_input, _view(self)))
        return inference_input

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        with self.lock:
            self.seen.append((inference_input, _view(self)))
        if self.gate is not None:
            await self.gate.wait()
        return inference_input


@attrs
class _Parent(InferencerBase):
    """Calls its child at a child slot from inside its own invocation."""

    child: _Leaf = attrib(factory=_Leaf, kw_only=True)
    seen: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append(_view(self))
        return self.child.infer(inference_input, run_context=self._rc_child("child"))

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append(_view(self))
        a, b = await asyncio.gather(
            self.child.ainfer("a", run_context=self._rc_child("a")),
            self.child.ainfer("b", run_context=self._rc_child("b")),
        )
        return a + b


@attrs
class _Session(StreamingInferencerBase):
    seen: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append((kwargs.get("new_session"), _view(self)))
        return "ok"

    async def _ainfer_streaming(self, prompt, **kwargs):
        self.seen.append((kwargs.get("new_session"), _view(self)))
        yield "ok"


def _root():
    return RunContext.root()


def _ids(views):
    return [v["id"] for v in views]


# -- nested calls and copied-context siblings ------------------------------------


def test_a_nested_child_call_opens_its_own_frame_at_the_child_path():
    parent = _Parent()
    assert parent.infer("q", run_context=_root()) == "q"
    (pview,) = parent.seen
    ((_, cview),) = parent.child.seen
    assert (pview["path"], cview["path"]) == ("/", "/child")
    assert cview["parent_owner"] is parent
    assert pview["id"] != cview["id"]


def test_copied_context_sibling_tasks_open_distinct_frames_under_the_parent():
    parent = _Parent()
    assert asyncio.run(parent.ainfer("q", run_context=_root())) == "ab"
    views = dict(parent.child.seen)
    assert {views["a"]["path"], views["b"]["path"]} == {"/a", "/b"}
    assert views["a"]["parent_owner"] is views["b"]["parent_owner"] is parent
    assert len({views["a"]["id"], views["b"]["id"], parent.seen[0]["id"]}) == 3


# -- parallel entries -----------------------------------------------------------


# One item at a time: a host parallel entry on an uncertified class refuses
# overlapping items (the P6 single-flight guard, test_single_flight_guard.py).


def test_parallel_infer_items_each_open_a_frame_at_their_own_path():
    leaf = _Leaf()
    out = leaf.parallel_infer(["x", "y", "z"], num_workers=1, run_context=_root())
    assert out == ["x", "y", "z"]
    views = dict(leaf.seen)
    assert {views[k]["path"] for k in "xyz"} == {
        "/parallel_0",
        "/parallel_1",
        "/parallel_2",
    }
    assert all(
        views[k]["entry"] == "infer" and views[k]["mode"] == "host" for k in "xyz"
    )
    assert len(set(_ids(views.values()))) == 3


def test_aparallel_infer_items_each_open_a_frame_at_their_own_path():
    leaf = _Leaf()
    out = asyncio.run(
        leaf.aparallel_infer(["x", "y"], max_concurrency=1, run_context=_root())
    )
    assert out == ["x", "y"]
    views = dict(leaf.seen)
    assert {views["x"]["path"], views["y"]["path"]} == {"/parallel_0", "/parallel_1"}
    assert {views["x"]["entry"], views["y"]["entry"]} == {"ainfer"}
    assert views["x"]["id"] != views["y"]["id"]


# -- adapters over the public entries --------------------------------------------


@pytest.mark.parametrize(
    "call",
    (
        lambda leaf, ctx: leaf("q", run_context=ctx),
        lambda leaf, ctx: list(leaf.iter_infer("q", run_context=ctx)),
        lambda leaf, ctx: leaf.infer(iter(["q"]), run_context=ctx),
    ),
    ids=("__call__", "iter_infer", "infer_iterator"),
)
def test_sync_adapters_reach_one_frame_per_item(call):
    leaf, ctx = _Leaf(), _root()
    result = call(leaf, ctx)
    list(result) if hasattr(result, "__next__") else result
    ((item, view),) = leaf.seen
    assert (item, view["entry"], view["path"], view["ctx_is_frame_ctx"]) == (
        "q",
        "infer",
        "/",
        True,
    )


@pytest.mark.parametrize(
    "helper", ("new_session", "resume_session", "anew_session", "aresume_session")
)
def test_session_helpers_run_inside_their_inner_call_frame(helper):
    inst, ctx = _Session(), _root()
    if helper == "new_session":
        inst.new_session("q", run_context=ctx)
    elif helper == "resume_session":
        inst.resume_session("sid", "q", run_context=ctx)
    elif helper == "anew_session":
        asyncio.run(inst.anew_session("q", run_context=ctx))
    else:
        asyncio.run(inst.aresume_session("sid", "q", run_context=ctx))
    ((_, view),) = inst.seen
    assert view["path"] == "/" and view["mode"] == "host"
    assert view["entry"] == ("ainfer" if helper.startswith("a") else "infer")
    assert frame_for(inst) is None


# -- cancellation and sync-from-async ---------------------------------------------


def test_a_cancelled_call_releases_its_claim():
    ctx = _root()

    async def main():
        gate = asyncio.Event()
        leaf = _Leaf(gate=gate)
        task = asyncio.create_task(leaf.ainfer("slow", run_context=ctx))
        while not leaf.seen:
            await asyncio.sleep(0)
        with pytest.raises(ConcurrentInvocationError):
            await _Leaf().ainfer("rejected", run_context=ctx)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await _Leaf().ainfer("after", run_context=ctx)

    assert asyncio.run(main()) == "after"


def test_a_sync_call_from_inside_a_running_loop_opens_its_frame():
    leaf, ctx = _Leaf(), _root()

    async def main():
        return leaf.infer("q", run_context=ctx)

    assert asyncio.run(main()) == "q"
    ((_, view),) = leaf.seen
    assert (view["entry"], view["path"], view["ctx_is_frame_ctx"]) == (
        "infer",
        "/",
        True,
    )


def test_bare_entries_open_legacy_or_no_ctx_frames():
    leaf = _Leaf()
    leaf.infer("q")
    session = _Session()
    session.new_session("q")
    assert leaf.seen[0][1]["mode"] == "legacy"
    assert session.seen[0][1]["mode"] == "legacy"

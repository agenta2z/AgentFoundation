"""Characterization goldens: RunContext ContextVar and thread lifetimes of the
streaming entrypoints and of lazy ``infer(iterator)`` (plan bugs B29 / B34).

What is pinned:

* The streaming templates bind the leaf's context (and its invocation frame) only
  while the pipeline runs, around each resumption (``framed_agen`` /
  ``framed_gen``, B29 fixed in P3 c7): the CONSUMER sees its own context between
  yields, with or without an explicit ``run_context``.
* A stream abandoned with ``break`` (no ``aclose``) leaves nothing active in the
  consumer task: a later bare call there mints its own root, and the finalizer
  (a GC-scheduled task or ``shutdown_asyncgens``) closes the stream without a
  ``ContextVar.reset`` error. Closing the stream closes its whole chain (pipeline,
  filters, transport) by ownership, so the GC finalizer runs the transport's
  cleanup under the stream's context; at loop shutdown asyncio closes every live
  generator itself, so the transport may be closed directly, outside it.
* The sync ``infer_streaming`` bridge (``iterate_async_in_thread``) owns its loop
  and thread: closing the generator cancels the transport (it sees
  ``CancelledError`` and runs its ``finally``) and joins the thread before
  ``close()`` returns.
* Lazy ``infer(iterator)`` resolves its context once, before returning, and binds
  it only around each ``next()`` (``ctx_bound_gen``, B29 extended, fixed in P3 c9):
  the caller sees its own context between items, and an abandoned or unstarted
  iterator leaves nothing active.
* ``state_factory`` runs inside the first item's invocation, at the first
  ``next``, and receives that item (B34 fixed; populate-once per node).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import gc
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import bridge, RunContext
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrib, attrs

from ._golden import check_golden, Normalizer

_DRAIN_STEPS = 200


class _Recorder:
    """Chronological event log; RunContexts are projected to stable labels.

    Keeps a reference to every labelled context so ``id()`` values are never
    reused within one recorder.
    """

    def __init__(self, **named: Any) -> None:
        self.events: list[dict[str, Any]] = []
        self._keep: list[Any] = list(named.values())
        self._labels: dict[int, str] = {id(v): k for k, v in named.items()}
        self._auto = 0

    def ctx(self, ctx: RunContext | None) -> dict[str, Any] | None:
        if ctx is None:
            return None
        if id(ctx) not in self._labels:
            self._keep.append(ctx)
            self._auto += 1
            self._labels[id(ctx)] = f"ctx{self._auto}"
        return {
            "id": self._labels[id(ctx)],
            "path": ctx.path,
            "legacy_mint": ctx.legacy_mint,
        }

    def note(self, where: str, what: str, value: Any = None) -> None:
        active = self.ctx(bridge.active_run_context())
        self.events.append(
            {"where": where, "what": what, "value": value, "ctx": active}
        )

    def saw(self, where: str, what: str) -> bool:
        return any(e["where"] == where and e["what"] == what for e in self.events)


@attrs
class _Chunks(StreamingInferencerBase):
    rec: Any = attrib(default=None, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kw):
        return "unused"

    async def _ainfer_streaming(self, prompt, **kwargs):
        try:
            for chunk in ("a", "b", "c"):
                self.rec.note("leaf", "yield", chunk)
                yield chunk
                await asyncio.sleep(0)
        except BaseException as exc:
            self.rec.note("leaf", "raised", type(exc).__name__)
            raise
        finally:
            self.rec.note("leaf", "finally")


@attrs
class _Blocking(StreamingInferencerBase):
    """Yields ``a``, then blocks on a loop-local event until :meth:`release`."""

    rec: Any = attrib(default=None, kw_only=True)
    waiting: threading.Event = attrib(factory=threading.Event, kw_only=True)
    thread: threading.Thread | None = attrib(default=None, init=False)
    _loop: Any = attrib(default=None, init=False)
    _release: Any = attrib(default=None, init=False)

    def _infer(self, inference_input, inference_config=None, **kw):
        return "unused"

    async def _ainfer_streaming(self, prompt, **kwargs):
        self.thread = threading.current_thread()
        self._loop = asyncio.get_running_loop()
        self._release = asyncio.Event()
        try:
            self.rec.note("leaf", "yield", "a")
            yield "a"
            self.waiting.set()
            await self._release.wait()
            self.rec.note("leaf", "yield", "b")
            yield "b"
        except BaseException as exc:
            self.rec.note("leaf", "raised", type(exc).__name__)
            raise
        finally:
            self.rec.note("leaf", "finally")

    def release(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._release.set)


@attrs
class _Plain(InferencerBase):
    rec: Any = attrib(default=None, kw_only=True)
    probe_name: str = attrib(default="plain", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kw):
        self.rec.note(self.probe_name, "_infer", inference_input)
        return inference_input

    async def _ainfer(self, inference_input, inference_config=None, **kw):
        self.rec.note(self.probe_name, "_ainfer", inference_input)
        return inference_input


def _exc(exc: BaseException | None) -> dict[str, str] | None:
    if exc is None:
        return None
    return {"type": type(exc).__name__, "message": str(exc)}


def _loop_error(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": context.get("message"),
        "exception": _exc(context.get("exception")),
    }


def _capture_loop_errors(errors: list[dict[str, Any]]) -> None:
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: errors.append(_loop_error(context))
    )


@contextlib.contextmanager
def _unraisable() -> Iterator[list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    previous = sys.unraisablehook
    gc.collect()
    sys.unraisablehook = lambda u: seen.append(
        {"exception": _exc(u.exc_value), "err_msg": u.err_msg}
    )
    try:
        yield seen
    finally:
        gc.collect()
        sys.unraisablehook = previous


async def _drain(done: Callable[[], bool]) -> None:
    for _ in range(_DRAIN_STEPS):
        if done():
            break
        await asyncio.sleep(0)
    gc.collect()
    for _ in range(3):
        await asyncio.sleep(0)


def _norm(data: Any) -> Any:
    return Normalizer().value(data)


# --- 1. between yields ------------------------------------------------------


async def _consume(leaf: _Chunks, rec: _Recorder, **kwargs: Any) -> None:
    rec.note("consumer", "before")
    async for chunk in leaf.ainfer_streaming("x", **kwargs):
        rec.note("consumer", "received", chunk)
    rec.note("consumer", "exhausted")


def test_async_stream_ctx_visible_between_yields():
    host = RunContext.root(workspace=None)
    out: dict[str, Any] = {}
    for mode, kwargs in (("bare", {}), ("explicit_host", {"run_context": host})):
        rec = _Recorder(host=host)
        asyncio.run(_consume(_Chunks(rec=rec), rec, **kwargs))
        out[mode] = rec.events
    out["test_thread_ctx_after"] = _Recorder().ctx(bridge.active_run_context())
    check_golden("lifetimes/async_between_yields", _norm(out))


# --- 2. abandoned async stream ----------------------------------------------


async def _abandon_then_gc(rec: _Recorder, errors: list[dict[str, Any]]) -> None:
    _capture_loop_errors(errors)
    gen = _Chunks(rec=rec).ainfer_streaming("x")
    async for chunk in gen:
        rec.note("consumer", "received", chunk)
        break
    rec.note("consumer", "after_break")
    await _Plain(rec=rec, probe_name="other").ainfer("y")
    rec.note("consumer", "after_other_ainfer")
    del gen
    gc.collect()
    await _drain(lambda: bool(errors) and rec.saw("leaf", "finally"))
    rec.note("consumer", "after_finalize")


async def _abandon_until_shutdown(
    rec: _Recorder, errors: list[dict[str, Any]], keep: list[Any]
) -> None:
    _capture_loop_errors(errors)
    gen = _Chunks(rec=rec).ainfer_streaming("x")
    keep.append(gen)
    async for chunk in gen:
        rec.note("consumer", "received", chunk)
        break
    rec.note("consumer", "after_break")


def test_abandoned_async_stream_gc_finalizer():
    rec, errors = _Recorder(), []
    with _unraisable() as unraisable:
        asyncio.run(_abandon_then_gc(rec, errors))
    rec.note("test_thread", "after_run")
    data = {"events": rec.events, "loop_errors": errors, "unraisable": unraisable}
    check_golden("lifetimes/async_abandoned_gc", _norm(data))


def _shutdown_order_independent(events: list[dict[str, Any]]) -> None:
    """``shutdown_asyncgens`` closes every live async generator concurrently, in
    no fixed order: the transport is closed either through the stream's chain
    (under the stream's ctx) or directly by asyncio (under none). Pin that it is
    one of the two, not which."""
    for event in events:
        if event["where"] == "leaf" and event["what"] in ("raised", "finally"):
            assert event["ctx"] is None or event["ctx"]["id"] == "ctx1", event
            event["ctx"] = "<stream ctx, or none if asyncio closed it first>"


def test_abandoned_async_stream_loop_shutdown_finalizer():
    rec, errors, keep = _Recorder(), [], []
    with _unraisable() as unraisable:
        asyncio.run(_abandon_until_shutdown(rec, errors, keep))
        rec.note("test_thread", "after_run")
        keep.clear()
    _shutdown_order_independent(rec.events)
    data = {"events": rec.events, "loop_errors": errors, "unraisable": unraisable}
    check_golden("lifetimes/async_abandoned_loop_shutdown", _norm(data))


# --- 3. abandoned sync stream -----------------------------------------------


def _alive_after(thread: threading.Thread, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while thread in threading.enumerate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return thread in threading.enumerate()


def _abandon_sync_stream(leaf: _Blocking, rec: _Recorder) -> dict[str, Any]:
    gen = leaf.infer_streaming("x")
    started_before_next = leaf.thread is not None
    rec.note("caller", "received", next(gen))
    assert leaf.waiting.wait(5.0), "transport never reached its blocking await"
    gen.close()
    rec.note("caller", "after_close")
    thread = leaf.thread
    return {
        "thread_started_before_first_next": started_before_next,
        "thread_is_caller": thread is threading.current_thread(),
        "thread_daemon": thread.daemon,
        "thread_target": thread.name.rsplit(" ", 1)[-1],
        "alive_after_close_and_0_5s": _alive_after(thread, 0.5),
        "leaf_raised_before_release": rec.saw("leaf", "raised"),
        "leaf_finally_before_release": rec.saw("leaf", "finally"),
    }


def test_abandoned_sync_stream_stops_its_bridge_thread():
    rec = _Recorder()
    leaf = _Blocking(rec=rec)
    try:
        at_close = _abandon_sync_stream(leaf, rec)
    finally:
        leaf.release()
        if leaf.thread is not None:
            leaf.thread.join(5.0)
    data = {
        "at_close": at_close,
        "alive_after_release_and_join": leaf.thread.is_alive(),
        "events": rec.events,
    }
    check_golden("lifetimes/sync_stream_abandoned", _norm(data))


# --- 4. lazy infer(iterator) ------------------------------------------------


def _factory_probe(rec: _Recorder, source: Any) -> Callable[[Any], Any]:
    def factory(inp: Any) -> dict[str, Any]:
        seen = {"type": type(inp).__name__, "is_source_iterator": inp is source}
        rec.note("state_factory", "received", seen)
        return {}

    return factory


def _lazy_abandoned_after_one(rec: _Recorder) -> None:
    source = iter(["i1", "i2", "i3"])
    leaf = _Plain(rec=rec, state_factory=_factory_probe(rec, source))
    rec.note("caller", "before")
    gen = leaf.infer(source)
    rec.note("caller", "returned", type(gen).__name__)
    rec.note("caller", "next", next(gen))
    _Plain(rec=rec, probe_name="other").infer("y")
    rec.note("caller", "after_other_infer")
    del gen
    gc.collect()
    rec.note("caller", "after_abandon", {"source_remaining": list(source)})


def _lazy_unstarted_closed(rec: _Recorder) -> None:
    gen = _Plain(rec=rec).infer(iter(["i1", "i2"]))
    rec.note("caller", "returned", type(gen).__name__)
    gen.close()
    rec.note("caller", "after_close")


def _iter_infer_abandoned_after_one(rec: _Recorder) -> None:
    gen = _Plain(rec=rec).iter_infer(iter(["i1", "i2"]))
    rec.note("caller", "returned", type(gen).__name__)
    rec.note("caller", "next", next(gen))
    gen.close()
    rec.note("caller", "after_close")


def _eager_merger_path(rec: _Recorder) -> None:
    source = iter(["i1", "i2"])
    leaf = _Plain(
        rec=rec,
        post_response_merger="|".join,
        state_factory=_factory_probe(rec, source),
    )
    rec.note("caller", "returned", leaf.infer(source))


def test_lazy_infer_iterator_ctx_lifetime_and_state_factory_input():
    out: dict[str, Any] = {}
    for name, scenario in (
        ("lazy_abandoned_after_one", _lazy_abandoned_after_one),
        ("lazy_unstarted_closed", _lazy_unstarted_closed),
        ("iter_infer_abandoned_after_one", _iter_infer_abandoned_after_one),
        ("eager_merger_path", _eager_merger_path),
    ):
        rec = _Recorder()
        with _unraisable() as unraisable:
            contextvars.copy_context().run(scenario, rec)
        out[name] = {"events": rec.events, "unraisable": unraisable}
    out["test_thread_ctx_after"] = _Recorder().ctx(bridge.active_run_context())
    check_golden("lifetimes/lazy_infer_iterator", _norm(out))

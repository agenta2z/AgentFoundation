"""Lazy ``infer(iterator)`` binds its run context per ``next()`` (plan v8 §5.1,
B29 extended, P3 c9).

``infer`` resolves the ctx once (minting at most once) before it returns; each
``next()`` binds it around that item's own invocation and resets it before the
item is yielded. However the iterator ends (exhausted, ``break``, ``close()``,
garbage collection, ``close()`` from another thread), the consumer keeps its own
context, no path claim survives, and nothing is reported to the unraisable hook.
"""

from __future__ import annotations

import gc
import sys
import threading
from typing import Any

import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    ctx_bound_gen,
    RunContext,
)
from attr import attrib, attrs


@attrs
class _Echo(InferencerBase):
    seen: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append((inference_input, active_run_context()))
        return inference_input

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input)


@pytest.fixture
def unraisable():
    seen: list[Any] = []
    previous = sys.unraisablehook
    gc.collect()
    sys.unraisablehook = seen.append
    try:
        yield seen
    finally:
        gc.collect()
        sys.unraisablehook = previous


def _end_by_break(gen):
    for _ in gen:
        break


def _end_by_close(gen):
    next(gen)
    gen.close()


def _end_by_gc(gen):
    next(gen)


def _end_by_close_from_another_thread(gen):
    next(gen)
    closer = threading.Thread(target=gen.close)
    closer.start()
    closer.join(5.0)


ENDINGS = {
    "break": _end_by_break,
    "close": _end_by_close,
    "gc": _end_by_gc,
    "close_from_thread": _end_by_close_from_another_thread,
}


@pytest.mark.parametrize("ending", sorted(ENDINGS))
@pytest.mark.parametrize("explicit", (False, True))
def test_an_abandoned_lazy_iterator_leaves_the_consumer_context_alone(
    ending, explicit, unraisable
):
    leaf = _Echo()
    ctx = RunContext.root().child("leaf") if explicit else None
    gen = leaf.infer(iter(["i1", "i2", "i3"]), run_context=ctx)
    assert active_run_context() is None
    ENDINGS[ending](gen)
    del gen
    gc.collect()
    assert active_run_context() is None
    assert unraisable == []
    (item, item_ctx), *_ = leaf.seen
    assert item == "i1"
    assert item_ctx is ctx if explicit else item_ctx.legacy_mint
    if explicit:
        assert _Echo().infer("next", run_context=ctx) == "next"


def test_every_item_runs_under_the_one_resolved_ctx_and_the_consumer_never_does():
    leaf = _Echo()
    ctx = RunContext.root().child("leaf")
    between = []
    for _ in leaf.infer(iter(["i1", "i2"]), run_context=ctx):
        between.append(active_run_context())
    assert [seen for _, seen in leaf.seen] == [ctx, ctx]
    assert between == [None, None]


def test_a_bare_lazy_iterator_mints_one_root_for_all_its_items():
    leaf = _Echo()
    assert list(leaf.infer(iter(["i1", "i2"]))) == ["i1", "i2"]
    roots = {id(seen) for _, seen in leaf.seen}
    assert len(roots) == 1


def test_ctx_bound_gen_binds_close_too():
    ctx = RunContext.root()
    events = []

    def inner():
        try:
            yield 1
            yield 2
        finally:
            events.append(active_run_context())

    gen = ctx_bound_gen(ctx, inner())
    assert next(gen) == 1
    assert active_run_context() is None
    gen.close()
    assert events == [ctx]
    assert active_run_context() is None

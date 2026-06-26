"""§9.3/N-Major1: _iter_child_slots + _with_child_ctx slot-aware lifecycle binding."""

import asyncio

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import RunContext, active_run_context, enter_run, exit_run


@attrs
class _Child(InferencerBase):
    seen_path: str = attrib(default=None, init=False)

    def _infer(self, x, inference_config=None, **kw):
        return x

    async def _pre_retry(self, attempt, exception):
        ctx = active_run_context()
        self.seen_path = ctx.path if ctx is not None else None


@attrs
class _Parent(InferencerBase):
    a: _Child = attrib(factory=_Child)
    b: _Child = attrib(factory=_Child)

    def _infer(self, x, inference_config=None, **kw):
        return x

    def _iter_child_slots(self):
        yield ("review", self.a)
        yield ("fix", self.b)


def test_default_iter_child_slots_positional():
    @attrs
    class P(InferencerBase):
        c: _Child = attrib(factory=_Child)
        def _infer(self, x, inference_config=None, **kw): return x
        def _iter_child_inferencers(self):
            yield self.c
    slots = list(P()._iter_child_slots())
    assert slots[0][0] == "child_0"


def test_pre_retry_binds_each_child_slot():
    p = _Parent()
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        asyncio.run(p.pre_retry(1, ValueError("x")))
    finally:
        exit_run(tok)
    # each child's _pre_retry saw ITS OWN slot path
    assert p.a.seen_path == "/review"
    assert p.b.seen_path == "/fix"


def test_pre_retry_without_ctx_is_byte_identical_noop_binding():
    p = _Parent()
    # no active ctx -> _with_child_ctx is a no-op; pre_retry still runs
    asyncio.run(p.pre_retry(1, ValueError("x")))
    assert p.a.seen_path is None and p.b.seen_path is None  # no ctx bound

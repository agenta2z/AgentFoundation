"""M6 regression: a leaf ``adisconnect`` must reclaim EVERY connection-scoped branch.

The Tier-3 refactor moved live handles (SDK clients / subprocesses) from instance
attrs into a path-keyed connection-scoped store. A branch is established under
``ctx.child(slot)`` DURING ``_ainfer`` (active ctx present), but ``adisconnect`` runs
at a lifecycle boundary (``__aexit__`` / host cleanup) where ``active_run_context()``
is ``None``. The per-call Tier-3 property shims read only the active branch / backing,
so reading them at teardown stranded every branch -> SDK client / subprocess leak.

The fix: ``StreamingInferencerBase._iter_live_handle_sets()`` yields a get/set view
over every stored branch + the legacy backing, so a leaf drains them all by stored
path (not by the absent active ctx). NB: orchestrator-side slot-binding of
``adisconnect`` would NOT fix this — it no-ops with no active ctx at the boundary.
"""

import asyncio

from agent_foundation.common.inferencers.run_context import (
    RunContext,
    active_run_context,
    enter_run,
    exit_run,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)


class _Leaf(StreamingInferencerBase):
    def __init__(self):
        self.__dict__["torn"] = []

    def _infer(self, *a, **k):
        return ""

    async def _ainfer_streaming(self, *a, **k):
        if False:
            yield

    async def adisconnect(self):
        # the real claude_code shape: drain ALL branches + backing
        for h in self._iter_live_handle_sets():
            fn = h.get("disconnect_fn")
            if fn:
                await fn()
            h.set("disconnect_fn", None)
            h.set("client", None)


def _establish(leaf, slot):
    """Simulate a leaf connecting under ctx.child(slot) during _ainfer."""
    st = enter_run(active_run_context().child(slot))
    try:

        async def _d(s=slot):
            leaf.__dict__["torn"].append(s)

        leaf._tier3_set("disconnect_fn", _d)
    finally:
        exit_run(st)


def test_adisconnect_drains_every_branch_at_a_no_context_boundary():
    leaf = _Leaf()

    async def main():
        root = RunContext.root(workspace=None)
        tok = enter_run(root)
        for slot in ("review", "fix"):
            _establish(leaf, slot)
        exit_run(tok)
        # lifecycle boundary: NO active context
        assert active_run_context() is None
        await leaf.adisconnect()
        return sorted(leaf.__dict__["torn"])

    torn = asyncio.run(main())
    assert torn == ["fix", "review"]  # both branches reclaimed
    # nothing left live in the connection-scoped store
    store = leaf._get_live_handle_store()
    assert not any(b.get("disconnect_fn") for b in store._by_path.values())


def test_legacy_no_context_connect_is_byte_identical():
    """With no context ever (legacy), the connection lands on the backing and is torn
    down via the backing view — identical to the pre-refactor single-attr teardown."""
    leaf = _Leaf()

    async def main():
        assert active_run_context() is None

        async def _d():
            leaf.__dict__["torn"].append("backing")

        leaf._tier3_set("disconnect_fn", _d)  # no ctx -> backing
        await leaf.adisconnect()

    asyncio.run(main())
    assert leaf.__dict__["torn"] == ["backing"]
    assert leaf.__dict__.get("_disconnect_fn_backing") is None

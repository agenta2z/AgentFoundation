"""Live-handle scope (plan v8 §5.5, B32, P3 c12).

A leaf keys its connection-scoped branches by ``(handle scope, path)``. A host
root's scope is its handle store's ``scope_id``; every legacy-minted root shares
one legacy scope. So two independent host roots, both at ``"/"``, never share a
leaf's session or client; a host keeps cross-turn continuity by building its
turn roots on one handle store; two sequential bare calls keep today's
continuity; and teardown reaches the branches of every scope.
"""

from __future__ import annotations

import asyncio

from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    LEGACY_HANDLE_SCOPE,
    LiveHandleStore,
    mint_root,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrib, attrs


@attrs
class _Leaf(StreamingInferencerBase):
    """Opens a session on its first call in a branch, resumes it afterwards."""

    calls: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._call(inference_input)

    async def _ainfer_streaming(self, prompt, **kwargs):
        yield self._call(prompt)

    def _call(self, prompt):
        resumed = self.active_session_id
        sid = resumed or f"sid-{len(self.calls)}"
        self.calls.append((prompt, resumed))
        self.active_session_id = sid
        self._tier3_set("client", self._tier3_get("client") or f"client-{sid}")
        return sid


def _under(ctx, fn):
    token = enter_run(ctx)
    try:
        return fn()
    finally:
        exit_run(token)


def test_store_scopes_are_unique_and_legacy_roots_share_one():
    a, b = LiveHandleStore(), LiveHandleStore()
    assert a.scope_id != b.scope_id
    assert RunContext.root(handle_store=a).child("x").handle_scope == a.scope_id
    assert mint_root().handle_scope == mint_root().handle_scope == LEGACY_HANDLE_SCOPE
    assert RunContext.root(handle_store=a).live_branch_key == (a.scope_id, "/")


def test_two_independent_host_roots_never_share_a_session_or_client():
    leaf = _Leaf()
    first, second = RunContext.root(), RunContext.root()
    assert leaf.infer("q1", run_context=first) == "sid-0"
    assert leaf.infer("q2", run_context=second) == "sid-1"
    assert leaf.calls == [("q1", None), ("q2", None)]
    assert _under(first, lambda: leaf._tier3_get("client")) == "client-sid-0"
    assert _under(second, lambda: leaf._tier3_get("client")) == "client-sid-1"


def test_one_handle_store_reused_across_turns_keeps_continuity():
    leaf, handles = _Leaf(), LiveHandleStore()
    turn1 = RunContext.root(handle_store=handles)
    turn2 = RunContext.root(handle_store=handles)
    assert leaf.infer("q1", run_context=turn1) == "sid-0"
    assert leaf.infer("q2", run_context=turn2) == "sid-0"
    assert leaf.calls == [("q1", None), ("q2", "sid-0")]


def test_two_sequential_bare_calls_keep_their_continuity():
    leaf = _Leaf()
    assert leaf.infer("q1") == "sid-0"
    assert leaf.infer("q2") == "sid-0"
    assert asyncio.run(leaf.ainfer("q3")) == "sid-0"
    assert leaf.calls == [("q1", None), ("q2", "sid-0"), ("q3", "sid-0")]
    assert _under(mint_root(), lambda: leaf._tier3_get("client")) == "client-sid-0"


def test_teardown_reaches_the_branches_of_every_scope():
    leaf = _Leaf()
    leaf.infer("q1", run_context=RunContext.root())
    leaf.infer("q2", run_context=RunContext.root())
    leaf.infer("q3")
    clients = sorted(
        h.get("client") for h in leaf._iter_live_handle_sets() if h.get("client")
    )
    assert clients == ["client-sid-0", "client-sid-1", "client-sid-2"]

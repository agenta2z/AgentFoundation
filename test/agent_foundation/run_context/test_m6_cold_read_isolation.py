"""M6 (V8 cold-read): a fresh branch must NOT cold-read another branch's live handle.

Regression for the leak where ``_tier3_set`` dual-wrote the instance backing under a
context, so a sibling branch's COLD read (before it set its own handle) fell back to
that polluted backing and saw the other branch's client/session. The fix: under a
context, the handle lives in THIS instance's connection-scoped store keyed by
``ctx.path`` (V8 isolation + V7 continuity); branch writes never touch the instance
backing, so cold reads resolve to None (connect fresh) — not a sibling's handle.
"""

from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    LiveHandleStore,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrs


@attrs
class _Leaf(StreamingInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kw):
        return "x"

    async def _ainfer_streaming(self, prompt, **kwargs):
        yield "x"

    @property
    def client(self):
        return self._tier3_get("client", None)

    @client.setter
    def client(self, value):
        self._tier3_set("client", value)


def test_sibling_branch_cold_read_does_not_leak():
    """Branch A writes its client; branch B cold-reads -> must see None, not A's."""
    leaf = _Leaf()
    root = RunContext.root(workspace=None)
    tok = enter_run(root.child("worker_0"))
    try:
        leaf.client = "client-A"
    finally:
        exit_run(tok)
    # Branch B: a fresh child, reads BEFORE writing -> must NOT inherit A's handle.
    tok = enter_run(root.child("worker_1"))
    try:
        assert leaf.client is None
    finally:
        exit_run(tok)


def test_branch_write_does_not_pollute_instance_backing():
    """A write under a context must not touch the instance backing (the shared base)."""
    leaf = _Leaf()
    root = RunContext.root(workspace=None)
    tok = enter_run(root.child("worker_0"))
    try:
        leaf.client = "client-A"
    finally:
        exit_run(tok)
    assert leaf.__dict__.get("_client_backing") is None  # base untouched


def test_v7_continuity_survives_across_turn_roots_on_one_handle_store():
    """The connection-scoped store is owned by the INSTANCE, so the same logical
    path keeps its handle across per-turn roots that a host builds on ONE handle
    store (V7, README Note B)."""
    leaf = _Leaf()
    handles = LiveHandleStore()
    tok = enter_run(
        RunContext.root(workspace=None, handle_store=handles).child("agent")
    )
    try:
        leaf.client = "persistent-conn"
    finally:
        exit_run(tok)
    # A brand-new root (next turn) on the same handle store, same path.
    tok = enter_run(
        RunContext.root(workspace=None, handle_store=handles).child("agent")
    )
    try:
        assert leaf.client == "persistent-conn"
    finally:
        exit_run(tok)


def test_independent_roots_never_share_a_branch_handle():
    """B32: two independent host roots (both at ``"/"``) are distinct handle
    scopes, so one never sees the other's connection at the same path."""
    leaf = _Leaf()
    tok = enter_run(RunContext.root(workspace=None).child("agent"))
    try:
        leaf.client = "root-a-conn"
    finally:
        exit_run(tok)
    tok = enter_run(RunContext.root(workspace=None).child("agent"))
    try:
        assert leaf.client is None
    finally:
        exit_run(tok)


def test_setup_base_connection_is_inherited_by_branches():
    """A connection established with NO context (setup-time -> instance backing) is a
    SHARED base that branches inherit via the fallback (not a leak — it's the base)."""
    leaf = _Leaf()
    leaf.client = "base"  # no active ctx -> instance backing
    root = RunContext.root(workspace=None)
    tok = enter_run(root.child("worker_0"))
    try:
        assert leaf.client == "base"  # branch inherits the shared base
    finally:
        exit_run(tok)

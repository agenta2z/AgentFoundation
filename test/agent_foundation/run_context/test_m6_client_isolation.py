"""M6-client (V8): connection-scoped live handles isolate across concurrent branches.

The Tier-3 compat helpers (`_tier3_get`/`_tier3_set`) keep one shared instance's
live handle per-branch when run under distinct child contexts — the V8 collision
fix — while remaining byte-identical (instance backing) with no active context.
"""

from attr import attrs

from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    enter_run,
    exit_run,
)


@attrs
class _Leaf(StreamingInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kw):
        return "x"

    async def _ainfer_streaming(self, prompt, **kwargs):
        yield "x"

    # A Tier-3-backed handle via the reusable compat helpers.
    @property
    def client(self):
        return self._tier3_get("client", None)

    @client.setter
    def client(self, value):
        self._tier3_set("client", value)


def test_handle_instance_fallback_without_ctx_is_byte_identical():
    leaf = _Leaf()
    assert leaf.client is None
    leaf.client = "C0"
    assert leaf.client == "C0"
    assert leaf.__dict__["_client_backing"] == "C0"


def test_one_instance_two_branches_isolate_their_handles():
    """V8: same instance, two child branches -> independent clients (no collision)."""
    leaf = _Leaf()
    root = RunContext.root(workspace=None)
    for ctx, val in ((root.child("worker_0"), "client-A"), (root.child("worker_1"), "client-B")):
        tok = enter_run(ctx)
        try:
            leaf.client = val
            assert leaf.client == val
        finally:
            exit_run(tok)
    # Each branch retained its own handle (connection-scoped, per path).
    for ctx, expected in ((root.child("worker_0"), "client-A"), (root.child("worker_1"), "client-B")):
        tok = enter_run(ctx)
        try:
            assert leaf.client == expected
        finally:
            exit_run(tok)


def test_handle_survives_across_calls_in_same_branch():
    """V7: connection-scoped handle persists across calls in one branch (no relaunch)."""
    leaf = _Leaf()
    root = RunContext.root(workspace=None)
    branch = root.child("agent")
    tok = enter_run(branch)
    try:
        leaf.client = "persistent"
    finally:
        exit_run(tok)
    # Re-enter the same branch later -> handle still there.
    tok = enter_run(root.child("agent"))
    try:
        assert leaf.client == "persistent"
    finally:
        exit_run(tok)

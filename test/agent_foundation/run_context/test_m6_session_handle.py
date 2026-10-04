"""M6: Tier-3 session continuity — ``active_session_id`` lives in the instance's
connection-scoped store keyed by ``ctx.path`` (V8 isolation + V7 continuity), with
the instance ``_session_id`` as the legacy/no-ctx fallback. A branch write never
pollutes the instance backing, so sibling cold reads can't see another branch's
session (the V8 cold-read fix — see test_m6_cold_read_isolation)."""

import copy
import pickle

from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.run_context.bridge import mint_root
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    _SessionSlot,
    StreamingInferencerBase,
)
from attr import attrs


@attrs
class _Stream(StreamingInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kw):
        return "x"

    async def _ainfer_streaming(self, prompt, **kwargs):  # minimal concrete impl
        yield "x"


def test_session_id_instance_fallback_without_ctx_is_byte_identical():
    s = _Stream()
    s.active_session_id = "sess1"  # no active ctx -> instance backing only
    assert s._session_id == "sess1"
    assert s.active_session_id == "sess1"


def test_session_set_under_ctx_does_not_pollute_instance_backing():
    s = _Stream()
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        s.active_session_id = "sess2"
        assert s.active_session_id == "sess2"  # readable in-branch
        # write-purity: the instance backing (shared base) is NOT touched by a
        # branch write -> sibling cold reads can't inherit this branch's session.
        assert s._session_id is None
    finally:
        exit_run(tok)
    # The branch write left the instance backing pure (asserted above) -> a sibling
    # branch can never cold-read this session. But a no-ctx PUBLIC read now SURFACES the
    # live connection's session from the connection-scoped store (V7 across-call
    # continuity: ``inf.active_session_id`` stays readable between calls, e.g. by the
    # host), since there is no explicit instance session to prefer.
    assert s._session_id is None  # backing still unpolluted (V8)
    assert s.active_session_id == "sess2"


def test_branch_session_wins_over_instance_base_when_present():
    s = _Stream()
    s._session_id = "instance-old"  # a setup/base session (no ctx)
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        # base is inherited until the branch sets its own
        assert s.active_session_id == "instance-old"
        s.active_session_id = "ctx-new"
        assert s.active_session_id == "ctx-new"  # branch session wins
    finally:
        exit_run(tok)
    # outside the run, falls back to the instance base (unpolluted)
    assert s.active_session_id == "instance-old"


def test_per_branch_session_isolation():
    """Two child branches keep independent live sessions (per-path), and a cold read
    in one branch never sees the other's."""
    s = _Stream()
    root = RunContext.root(workspace=None)
    a, b = root.child("worker_0"), root.child("worker_1")
    for ctx, sid in ((a, "A"), (b, "B")):
        tok = enter_run(ctx)
        try:
            s.active_session_id = sid
        finally:
            exit_run(tok)
    for ctx, expected in ((root.child("worker_0"), "A"), (root.child("worker_1"), "B")):
        tok = enter_run(ctx)
        try:
            assert s.active_session_id == expected
        finally:
            exit_run(tok)


def _under(ctx, fn):
    tok = enter_run(ctx)
    try:
        return fn()
    finally:
        exit_run(tok)


def _reset(s):
    s.active_session_id = None


def test_host_reset_does_not_fall_back_to_instance_backing():
    """B6(a): a host reset leaves the branch with NO session — the getter must not
    resurrect the shared instance backing it was cleared to escape."""
    s = _Stream()
    s._session_id = "shared-base"
    worker = RunContext.root(workspace=None).child("worker_0")
    _under(worker, lambda: _reset(s))
    assert _under(worker, lambda: s.active_session_id) is None
    assert s._session_id == "shared-base"  # the shared backing is untouched


def test_host_reset_is_branch_local():
    s = _Stream()
    s._session_id = "shared-base"
    root = RunContext.root(workspace=None)
    _under(root.child("worker_0"), lambda: _reset(s))
    assert _under(root.child("worker_1"), lambda: s.active_session_id) == "shared-base"
    assert s.active_session_id == "shared-base"  # no-ctx read prefers the backing


def test_host_write_after_reset_replaces_the_tombstone():
    s = _Stream()
    s._session_id = "shared-base"
    worker = RunContext.root(workspace=None).child("worker_0")
    _under(worker, lambda: _reset(s))

    def _set_new():
        s.active_session_id = "fresh"

    _under(worker, _set_new)
    assert _under(worker, lambda: s.active_session_id) == "fresh"


def test_no_ctx_read_ignores_reset_branches():
    """A reset branch holds no session, so it neither surfaces as one nor makes a
    single live branch ambiguous for the between-calls public read."""
    s = _Stream()
    root = RunContext.root(workspace=None)
    _under(root.child("worker_0"), lambda: _reset(s))
    assert s.active_session_id is None

    def _set_live():
        s.active_session_id = "live"

    _under(root.child("worker_1"), _set_live)
    assert s.active_session_id == "live"


def test_legacy_mint_reset_still_clears_the_backing():
    s = _Stream()
    s._session_id = "old"
    _under(mint_root(), lambda: _reset(s))
    assert s._session_id is None
    assert s.active_session_id is None


def test_reset_marker_survives_copy_and_pickle():
    s = _Stream()
    s._session_id = "shared-base"
    worker = RunContext.root(workspace=None).child("worker_0")
    _under(worker, lambda: _reset(s))
    store = s.__dict__["_live_handle_store"]
    for clone in (copy.deepcopy(store), pickle.loads(pickle.dumps(store))):
        assert clone.peek(worker.live_branch_key).get("live_session_id") is (
            _SessionSlot.RESET
        )

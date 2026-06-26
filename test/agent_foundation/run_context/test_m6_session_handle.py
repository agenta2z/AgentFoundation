"""M6: Tier-3 session continuity — ``active_session_id`` lives in the instance's
connection-scoped store keyed by ``ctx.path`` (V8 isolation + V7 continuity), with
the instance ``_session_id`` as the legacy/no-ctx fallback. A branch write never
pollutes the instance backing, so sibling cold reads can't see another branch's
session (the V8 cold-read fix — see test_m6_cold_read_isolation)."""

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

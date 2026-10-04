"""``LiveHandleField`` and the session-scoped helpers (plan v8 §5.5, P3 c11).

``active_session_id`` is a ``LiveHandleField`` over ``_session_scoped_get`` /
``_session_scoped_set``: under a host ctx it is this branch's live slot (a host
reset reads as ``None``, B6(a)); with no ctx or under a legacy root it is the
instance backing, and a reset there also clears every live branch, so the no-ctx
read can no longer surface another branch's session (B6(b)). Session-scoped
state other than the session id (devmate's error counter, P10) uses the same two
helpers. The Tier-3 handle helpers live on ``InferencerBase``.
"""

from __future__ import annotations

import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    mint_root,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    LiveHandleField,
    StreamingInferencerBase,
)
from attr import attrib, attrs


@attrs
class _Leaf(StreamingInferencerBase):
    _error_count: int = attrib(default=0, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "ok"

    async def _ainfer_streaming(self, prompt, **kwargs):
        yield "ok"


@attrs(slots=False)
class _Plain(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "ok"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return "ok"


def _under(ctx, fn):
    token = enter_run(ctx)
    try:
        return fn()
    finally:
        exit_run(token)


def _set_session(leaf, value):
    leaf.active_session_id = value


def test_active_session_id_is_a_live_handle_field():
    field = vars(StreamingInferencerBase)["active_session_id"]
    assert isinstance(field, LiveHandleField)
    assert (field.name, field.backing) == ("live_session_id", "_session_id")
    assert StreamingInferencerBase.active_session_id is field


# -- B6(b): a reset with no host ctx really resets ------------------------------


@pytest.mark.parametrize("reset_under", ("no_ctx", "legacy"))
def test_a_bare_reset_clears_every_live_branch(reset_under):
    leaf, host = _Leaf(), RunContext.root().child("leaf")
    _under(host, lambda: _set_session(leaf, "branch-sid"))
    assert leaf.active_session_id == "branch-sid"
    if reset_under == "no_ctx":
        leaf.active_session_id = None
    else:
        _under(mint_root(), lambda: _set_session(leaf, None))
    assert leaf.active_session_id is None
    assert _under(host, lambda: leaf.active_session_id) is None


def test_reset_session_with_no_ctx_clears_every_live_branch():
    leaf = _Leaf()
    _under(RunContext.root().child("a"), lambda: _set_session(leaf, "a-sid"))
    leaf.reset_session()
    assert leaf.active_session_id is None


def test_a_bare_non_reset_write_leaves_the_branches_alone():
    leaf, host = _Leaf(), RunContext.root().child("leaf")
    _under(host, lambda: _set_session(leaf, "branch-sid"))
    leaf.active_session_id = "bare-sid"
    assert leaf.active_session_id == "bare-sid"
    assert _under(host, lambda: leaf.active_session_id) == "branch-sid"


# -- the host policy ----------------------------------------------------------


def test_host_branches_are_isolated_and_a_host_reset_never_reads_the_backing():
    leaf = _Leaf()
    a, b = RunContext.root().child("a"), RunContext.root().child("b")
    leaf.active_session_id = "backing-sid"
    _under(a, lambda: _set_session(leaf, "a-sid"))
    assert _under(a, lambda: leaf.active_session_id) == "a-sid"
    assert _under(b, lambda: leaf.active_session_id) == "backing-sid"
    _under(a, lambda: _set_session(leaf, None))
    assert _under(a, lambda: leaf.active_session_id) is None
    assert leaf._session_id == "backing-sid"


def test_an_ambiguous_set_of_live_branches_is_never_surfaced_bare():
    leaf = _Leaf()
    _under(RunContext.root().child("a"), lambda: _set_session(leaf, "a-sid"))
    _under(RunContext.root().child("b"), lambda: _set_session(leaf, "b-sid"))
    assert leaf.active_session_id is None


# -- session-scoped state beyond the session id ---------------------------------


def test_session_scoped_state_follows_the_session_policy():
    leaf = _Leaf()
    a, b = RunContext.root().child("a"), RunContext.root().child("b")

    def get():
        return leaf._session_scoped_get("errors", 0, backing="_error_count")

    def put(value):
        leaf._session_scoped_set("errors", value, backing="_error_count")

    _under(a, lambda: put(3))
    assert _under(a, get) == 3
    assert _under(b, get) == 0
    assert leaf._error_count == 0
    put(5)
    assert get() == 5
    assert _under(b, get) == 5
    _under(a, lambda: put(None))
    assert _under(a, get) == 0


def test_session_scoped_state_defaults_its_backing_to_the_underscored_name():
    leaf = _Leaf()
    leaf._session_scoped_set("error_count", 2)
    assert leaf._error_count == 2
    assert leaf._session_scoped_get("error_count", 0) == 2


# -- the lifted Tier-3 helpers --------------------------------------------------


def test_the_tier3_helpers_live_on_every_inferencer():
    plain, host = _Plain(), RunContext.root().child("plain")
    _under(host, lambda: plain._tier3_set("client", "branch-client"))
    plain._tier3_set("client", "bare-client")
    assert _under(host, lambda: plain._tier3_get("client")) == "branch-client"
    assert plain._tier3_get("client") == "bare-client"
    seen = [h.get("client") for h in plain._iter_live_handle_sets()]
    assert seen == ["branch-client", "bare-client"]

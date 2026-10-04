"""CLI session policy in the invocation seam's provider hooks (plan v8 §5.1,
P3 c6).

Each CLI leaf resolves its session in ``_prepare_call`` and adopts the result's
session in ``_conclude_call`` / ``_aconclude_call``, inside the invocation. A stub
transport (``_infer`` / ``_ainfer``) records the session kwargs it receives and
returns a response carrying a session id, so each test pins the policy, not the
CLI:

* bare calls continue the session, and ``new_session`` starts a fresh one;
* a claim-rejected call and a render-only call leave the session untouched (no
  provider runs, so the hooks don't either);
* under an explicit host ctx the policy reads and writes that ctx's live slot;
* only the async transport's stream result feeds the async post-hook.
"""

from __future__ import annotations

import asyncio

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer import (
    CodexCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
    devmate_cli_inferencer as devmate_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer import (
    KiroCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer import (
    OpenClawInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
    rovodev_cli_inferencer as rovodev_module,
)
from agent_foundation.common.inferencers.run_context import (
    ConcurrentInvocationError,
    enter_run,
    exit_run,
    invocation_of,
    NodeOutcomeState,
    open_invocation,
    read_outcome,
    RunContext,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_response import (
    TerminalInferencerResponse,
)

KINDS = ("sync", "async")


class _Owner:
    pass


class _StubTransport:
    """Records the session kwargs of each provider call; the CLI never runs."""

    def _transport(self, kwargs):
        stream_result = self.__dict__.get("stream_result")
        if stream_result is not None:
            # What a real async stream would record in the invocation.
            invocation_of(self).put(type(self)._STREAM_RESULT, stream_result)
        calls = self.__dict__.setdefault("calls", [])
        calls.append((kwargs.get("session_id"), kwargs.get("resume")))
        sid = self.__dict__.get("reply_session", f"s{len(calls)}")
        self.__dict__["last_reply"] = sid
        reply = self.__dict__.get("reply")
        if reply is not None:
            return reply
        return TerminalInferencerResponse(output="ok", session_id=sid, success=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._transport(kwargs)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._transport(kwargs)


class _Claude(_StubTransport, ClaudeCodeCliInferencer):
    pass


class _Codex(_StubTransport, CodexCliInferencer):
    pass


class _Kiro(_StubTransport, KiroCliInferencer):
    pass


class _RovoDev(_StubTransport, rovodev_module.RovoDevCliInferencer):
    pass


def _rovodev(tmp_path, monkeypatch):
    """rovodev reads the session back from its sessions directory; the stub
    directory holds the session of the last transport reply."""
    inst = _RovoDev(target_path=str(tmp_path))
    monkeypatch.setattr(
        rovodev_module,
        "find_latest_session_id",
        lambda workspace_path=None: inst.__dict__.get("last_reply"),
    )
    monkeypatch.setattr(rovodev_module, "ensure_session_metadata", lambda *a, **k: None)
    return inst


class _Devmate(_StubTransport, devmate_module.DevmateCliInferencer):
    def _outcome_for(self, frame):
        return NodeOutcomeState(final_output=str(frame.result))


def _devmate(tmp_path, monkeypatch):
    monkeypatch.setattr(devmate_module, "sync_config_to_target", lambda *a, **k: None)
    (tmp_path / ".sl").mkdir(exist_ok=True)
    return _Devmate(target_path=str(tmp_path))


LEAVES = {
    "claude_code": lambda tmp_path, _mp: _Claude(target_path=str(tmp_path)),
    "devmate": _devmate,
    "codex": lambda tmp_path, _mp: _Codex(target_path=str(tmp_path)),
    "kiro": lambda tmp_path, _mp: _Kiro(target_path=str(tmp_path)),
    "rovodev": _rovodev,
}
STREAM_RESULT_LEAVES = ("claude_code", "codex")


def _call(inst, kind, text="q", **kwargs):
    if kind == "sync":
        return inst.infer(text, **kwargs)
    return asyncio.run(inst.ainfer(text, **kwargs))


def _session_under(inst, ctx):
    token = enter_run(ctx)
    try:
        return inst.active_session_id
    finally:
        exit_run(token)


def _streak_under(inst, ctx):
    token = enter_run(ctx)
    try:
        return inst._error_streak()
    finally:
        exit_run(token)


@pytest.fixture(params=sorted(LEAVES))
def leaf(request, tmp_path, monkeypatch):
    return LEAVES[request.param](tmp_path, monkeypatch)


@pytest.mark.parametrize("kind", KINDS)
def test_bare_calls_continue_the_session(leaf, kind):
    _call(leaf, kind)
    _call(leaf, kind)
    assert leaf.calls == [(None, False), ("s1", True)]
    assert leaf.active_session_id == "s2"


@pytest.mark.parametrize("kind", KINDS)
def test_new_session_starts_a_fresh_session(leaf, kind):
    _call(leaf, kind)
    _call(leaf, kind, new_session=True)
    assert leaf.calls == [(None, False), (None, False)]
    assert leaf.active_session_id == "s2"


@pytest.mark.parametrize("kind", KINDS)
def test_a_claim_rejected_call_leaves_the_session_untouched(leaf, kind):
    ctx = RunContext.root().child("leaf")
    token = enter_run(ctx)
    try:
        leaf.active_session_id = "s0"
        with open_invocation(_Owner()):
            with pytest.raises(ConcurrentInvocationError):
                _call(leaf, kind, run_context=ctx, new_session=True)
    finally:
        exit_run(token)
    assert _session_under(leaf, ctx) == "s0"
    assert "calls" not in leaf.__dict__


@pytest.mark.parametrize("kind", KINDS)
def test_a_render_only_call_leaves_the_session_untouched(leaf, kind):
    leaf.active_session_id = "s0"
    assert _call(leaf, kind, render_only=True, new_session=True) == "q"
    assert leaf.active_session_id == "s0"
    assert "calls" not in leaf.__dict__


@pytest.mark.parametrize("kind", KINDS)
def test_the_policy_uses_the_explicit_host_ctx_slot(leaf, kind):
    ctx = RunContext.root().child("leaf")
    leaf.active_session_id = "backing"
    token = enter_run(ctx)
    try:
        leaf.active_session_id = "slot"
    finally:
        exit_run(token)
    leaf.reply_session = "slot-next"
    _call(leaf, kind, run_context=ctx)
    assert leaf.calls == [("slot", True)]
    assert _session_under(leaf, ctx) == "slot-next"
    assert leaf._session_id == "backing"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", STREAM_RESULT_LEAVES)
def test_only_the_async_post_hook_adopts_the_stream_result_session(
    name, kind, tmp_path, monkeypatch
):
    inst = LEAVES[name](tmp_path, monkeypatch)
    inst.reply_session = None
    inst.stream_result = {"session_id": "streamed"}
    result = _call(inst, kind)
    if kind == "async":
        assert inst.active_session_id == "streamed"
        assert result.session_id == "streamed"
    else:
        assert inst.active_session_id is None
        assert result.session_id is None


@pytest.mark.parametrize("kind", KINDS)
def test_rovodev_sync_captures_the_session_only_after_success(
    kind, tmp_path, monkeypatch
):
    inst = _rovodev(tmp_path, monkeypatch)
    inst.reply = TerminalInferencerResponse(output="", success=False, error="x")
    _call(inst, kind)
    assert inst.active_session_id == {"sync": None, "async": "s1"}[kind]


def test_rovodev_async_wraps_a_plain_result(tmp_path, monkeypatch):
    inst = _rovodev(tmp_path, monkeypatch)
    inst.reply = "noisy stdout"
    result = _call(inst, "async")
    assert isinstance(result, TerminalInferencerResponse)
    assert (result.raw_output, result.success) == ("noisy stdout", True)
    assert inst.active_session_id == "s1"


@pytest.mark.parametrize("host", (False, True))
def test_devmate_promoted_failure_publishes_no_outcome(host, tmp_path, monkeypatch):
    inst = _devmate(tmp_path, monkeypatch)
    ctx = RunContext.root().child("leaf") if host else None
    inst.reply = TerminalInferencerResponse(
        output="", success=False, error="boom", session_id="failed-sid"
    )
    with pytest.raises(devmate_module.InferencerExecutionError, match="boom"):
        _call(inst, "async", run_context=ctx)
    if host:
        assert read_outcome(ctx) is None
        assert _session_under(inst, ctx) == "failed-sid"
        # B33: the error streak is the branch's; the instance counter is untouched.
        assert _streak_under(inst, ctx) == 1
        assert inst._consecutive_error_count == 0
    else:
        assert inst.active_session_id == "failed-sid"
        assert inst._consecutive_error_count == 1

    inst.reply = None
    _call(inst, "async", run_context=ctx)
    if host:
        assert read_outcome(ctx).final_output == "ok"
        assert _streak_under(inst, ctx) == 0
    else:
        assert inst._consecutive_error_count == 0


def test_devmate_sync_infer_does_not_promote_failures(tmp_path, monkeypatch):
    inst = _devmate(tmp_path, monkeypatch)
    inst.reply = TerminalInferencerResponse(
        output="", success=False, error="boom", session_id="failed-sid"
    )
    assert _call(inst, "sync").success is False
    assert inst._consecutive_error_count == 0
    assert inst.active_session_id == "failed-sid"


class _OpenClaw(OpenClawInferencer):
    """OpenClaw adopts the session inside ``_ainfer``; the stub records the
    session the seam resolved."""

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        self.__dict__.setdefault("calls", []).append(kwargs.get("session_id"))
        self.active_session_id = "next"
        return "ok"


@pytest.mark.parametrize("kind", KINDS)
def test_openclaw_resolves_the_session_inside_the_invocation(kind):
    inst = _OpenClaw(auth_token="tok")
    inst.active_session_id = "s0"
    _call(inst, kind)
    _call(inst, kind, new_session=True)
    _call(inst, kind, session_id="explicit")
    assert inst.calls == ["s0", inst.session_id, "explicit"]


@pytest.mark.parametrize("kind", KINDS)
def test_openclaw_claim_rejected_call_leaves_the_session_untouched(kind):
    inst = _OpenClaw(auth_token="tok")
    ctx = RunContext.root().child("leaf")
    token = enter_run(ctx)
    try:
        inst.active_session_id = "s0"
        with open_invocation(_Owner()):
            with pytest.raises(ConcurrentInvocationError):
                _call(inst, kind, run_context=ctx, new_session=True)
    finally:
        exit_run(token)
    assert _session_under(inst, ctx) == "s0"
    assert "calls" not in inst.__dict__


@pytest.mark.parametrize("kind", KINDS)
def test_openclaw_render_only_call_leaves_the_session_untouched(kind):
    inst = _OpenClaw(auth_token="tok")
    inst.active_session_id = "s0"
    assert _call(inst, kind, render_only=True, new_session=True) == "q"
    assert inst.active_session_id == "s0"
    assert "calls" not in inst.__dict__

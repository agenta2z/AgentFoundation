"""The invocation seam in ``_(a)infer_single`` (plan v8 §5.1, P3 c5)."""

from __future__ import annotations

import asyncio

import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    aopen_invocation,
    ConcurrentInvocationError,
    enter_run,
    exit_run,
    frame_for,
    InvocationCleanupError,
    NodeOutcomeState,
    open_invocation,
    read_outcome,
    RunContext,
)
from attr import attrib, attrs

KINDS = ("sync", "async")


class _Owner:
    pass


class _FailingClose:
    def __init__(self, events):
        self._events = events

    def close(self):
        self._events.append("close")
        raise OSError("pipe")


@attrs
class _Recorder(InferencerBase):
    events = attrib(factory=list)
    frames = attrib(factory=list)
    fail_with = attrib(default=None)
    fail_conclude = attrib(default=False)
    leak = attrib(default=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._pipeline(inference_input, kwargs)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._pipeline(inference_input, kwargs)

    def _render_prompt(self, inference_input, extra_feed=None):
        return inference_input

    def _pipeline(self, inference_input, kwargs):
        frame = frame_for(self)
        self.frames.append(frame)
        self.events.append(("pipeline", inference_input, kwargs.get("prepared")))
        if self.leak:
            frame.ledger.register(_FailingClose(self.events), "pipe")
        if self.fail_with is not None:
            raise self.fail_with
        return f"r:{inference_input}"

    def _init_call_state(self, inference_input):
        self.events.append(("init", inference_input))
        super()._init_call_state(inference_input)

    def _prepare_call(self, inference_args):
        self.events.append(("prepare", "extra_feed" in inference_args))
        return {**inference_args, "prepared": True}

    def _conclude_call(self, result):
        self.events.append(("conclude", result))
        if self.fail_conclude:
            raise ValueError("promoted failure")
        return f"{result}!"

    def _outcome_for(self, frame):
        return NodeOutcomeState(final_output=frame.result)


def _root(store=None):
    return RunContext.root(
        workspace=InferencerWorkspace(root="/tmp/run"),
        store=store,
    )


def _call(inf, kind, text="q", **kwargs):
    if kind == "sync":
        return inf.infer(text, **kwargs)
    return asyncio.run(inf.ainfer(text, **kwargs))


def _pipeline_calls(inf):
    return [entry for entry in inf.events if entry[0] == "pipeline"]


# -- order ------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_hooks_run_inside_the_frame_in_seam_order(kind):
    factory_inputs = []

    def state_factory(inference_input):
        factory_inputs.append(inference_input)
        return {}

    inf = _Recorder(state_factory=state_factory)
    ctx = _root()
    assert _call(inf, kind, run_context=ctx, extra_feed={"k": "v"}) == "r:q!"
    assert inf.events == [
        ("init", "q"),
        ("prepare", False),
        ("pipeline", "q", True),
        ("conclude", "r:q"),
    ]
    assert factory_inputs == ["q"]
    assert inf.frames[0].owner is inf
    assert inf.frames[0].closed
    assert frame_for(inf) is None


@pytest.mark.parametrize("kind", KINDS)
def test_a_direct_single_entry_opens_its_own_frame(kind):
    inf = _Recorder()
    if kind == "sync":
        result = inf._infer_single("q")
    else:
        result = asyncio.run(inf._ainfer_single("q"))
    assert result == "r:q!"
    assert inf.frames[0] is not None
    assert inf.frames[0].mode == "no_ctx"
    assert inf.frames[0].entry == ("infer" if kind == "sync" else "ainfer")


# -- claim ------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_rejected_call_initializes_nothing(kind):
    factory_inputs = []
    inf = _Recorder(state_factory=lambda x: factory_inputs.append(x) or {})
    ctx = _root()
    token = enter_run(ctx)
    try:
        if kind == "sync":
            with open_invocation(_Owner()):
                with pytest.raises(ConcurrentInvocationError, match="_Recorder"):
                    inf.infer("q", run_context=ctx)
        else:

            async def main():
                async with aopen_invocation(_Owner()):
                    with pytest.raises(ConcurrentInvocationError, match="_Recorder"):
                        await inf.ainfer("q", run_context=ctx)

            asyncio.run(main())
    finally:
        exit_run(token)
    assert inf.events == []
    assert factory_inputs == []
    node = ctx.store.peek("/")
    assert node is None or node.call is None


# -- outcome ----------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_success_publishes_one_outcome_stamped_with_its_invocation(kind):
    inf = _Recorder()
    ctx = _root()
    _call(inf, kind, run_context=ctx)
    outcome = read_outcome(ctx)
    assert outcome.final_output == "r:q!"
    assert outcome.invocation_id == inf.frames[0].invocation_id
    assert outcome.cleanup_errors == ()


@pytest.mark.parametrize("kind", KINDS)
def test_a_failed_call_never_exposes_the_previous_outcome(kind):
    inf = _Recorder()
    ctx = _root()
    _call(inf, kind, text="a", run_context=ctx)
    assert read_outcome(ctx).final_output == "r:a!"

    inf.fail_with = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        _call(inf, kind, text="b", run_context=ctx)
    assert read_outcome(ctx) is None

    inf.fail_with = None
    _call(inf, kind, text="c", run_context=ctx)
    outcome = read_outcome(ctx)
    assert outcome.final_output == "r:c!"
    assert outcome.invocation_id == inf.frames[-1].invocation_id
    assert len({frame.invocation_id for frame in inf.frames}) == 3


@pytest.mark.parametrize("kind", KINDS)
def test_a_conclude_failure_fails_the_call_and_publishes_nothing(kind):
    inf = _Recorder(fail_conclude=True)
    ctx = _root()
    with pytest.raises(ValueError, match="promoted failure"):
        _call(inf, kind, run_context=ctx)
    assert inf.events[-1] == ("conclude", "r:q")
    assert read_outcome(ctx) is None


def test_bare_calls_publish_nowhere_a_host_can_read():
    inf = _Recorder()
    assert inf.infer("q") == "r:q!"
    assert inf.frames[0].mode == "legacy"


# -- cleanup ----------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_cleanup_failure_after_success_publishes_then_raises_with_the_result(
    kind,
):
    inf = _Recorder(leak=True, max_retry=3)
    ctx = _root()
    with pytest.raises(InvocationCleanupError) as caught:
        _call(inf, kind, run_context=ctx)
    assert caught.value.result == "r:q!"
    assert caught.value.errors == ("pipe: OSError: pipe",)
    outcome = read_outcome(ctx)
    assert outcome.final_output == "r:q!"
    assert outcome.cleanup_errors == ("pipe: OSError: pipe",)
    assert len(_pipeline_calls(inf)) == 1
    assert inf.events.count("close") == 1


@pytest.mark.parametrize("kind", KINDS)
def test_a_cleanup_failure_during_a_failing_call_is_a_note(kind, caplog):
    inf = _Recorder(leak=True, fail_with=RuntimeError("boom"))
    ctx = _root()
    with pytest.raises(RuntimeError, match="boom") as caught:
        _call(inf, kind, run_context=ctx)
    assert caught.value is inf.fail_with
    assert "owned resource failed to close: pipe: OSError: pipe" in (
        caught.value.__notes__
    )
    assert "its owned resources also failed to close" in caplog.text
    assert read_outcome(ctx) is None


# -- provider hooks -----------------------------------------------------------


@attrs
class _Delegating(_Recorder):
    """A call delegated to a per-call BTA: the fan-out runs instead of the
    provider."""

    @property
    def _delegates_execution(self) -> bool:
        return True

    def _should_fan_out(self) -> bool:
        return True

    def _run_fanout(self, inference_input, inference_config, kwargs):
        self.events.append(("fanout", sorted(kwargs)))
        return "FANOUT", {}

    async def _arun_fanout(self, inference_input, inference_config, kwargs):
        self.events.append(("fanout", sorted(kwargs)))
        return "FANOUT", {}


@pytest.mark.parametrize("kind", KINDS)
def test_a_render_only_call_runs_no_provider_hooks(kind):
    inf = _Recorder()
    assert _call(inf, kind, render_only=True) == "q"
    assert inf.events == [("init", "q")]


@pytest.mark.parametrize("kind", KINDS)
def test_a_delegated_call_runs_no_provider_hooks(kind):
    inf = _Delegating()
    assert _call(inf, kind, run_context=_root(), new_session=True) == "FANOUT"
    assert inf.events == [("init", "q"), ("fanout", ["new_session"])]


@attrs
class _AsyncConcluding(_Recorder):
    async def _aconclude_call(self, result):
        await asyncio.sleep(0)
        self.events.append(("aconclude", result))
        return f"{result}?"


@pytest.mark.parametrize("kind", KINDS)
def test_async_entries_run_the_async_post_hook(kind):
    inf = _AsyncConcluding()
    expected = {"sync": "r:q!", "async": "r:q?"}[kind]
    assert _call(inf, kind, run_context=_root()) == expected
    last = {"sync": ("conclude", "r:q"), "async": ("aconclude", "r:q")}[kind]
    assert inf.events[-1] == last


def test_the_async_post_hook_defaults_to_the_sync_one():
    inf = _Recorder()
    assert asyncio.run(inf._aconclude_call("x")) == "x!"
    assert inf.events == [("conclude", "x")]

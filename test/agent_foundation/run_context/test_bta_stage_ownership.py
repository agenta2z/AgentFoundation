"""A BTA call owns the stages it builds and closes them (plan v8 §5.7, P6 c5; B25).

A worker a factory built is owned by its attempt: it closes (``adisconnect``) when the
attempt ends, sync and async, and a failed close surfaces through the invocation's
cleanup errors. A configured instance is borrowed: the call never closes it, and in
host mode never writes its workspace backing either. One borrowed duck-typed instance
can't serve two workers that may run concurrently in host mode. The SDK leaves' sync
bridges close the client they connected inside the loop that created it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    InvocationCleanupError,
    read_outcome,
    RunContext,
    StageOwnershipError,
    UncertifiedConcurrentUseError,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode

KINDS = ("sync", "async")


@attrs
class _Worker(InferencerBase):
    """Records its calls and closes into a shared ``events`` list."""

    label: str = attrib(default="w", kw_only=True)
    events: list = attrib(factory=list, kw_only=True)
    close_fails: bool = attrib(default=False, kw_only=True)
    fail: bool = attrib(default=False, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.events.append(("run", self.label))
        if self.fail:
            raise RuntimeError("worker failed")
        return f"{self.label}:{inference_input}"

    async def adisconnect(self):
        self.events.append(("close", self.label))
        if self.close_fails:
            raise OSError(f"{self.label} close failed")


class _Duck:
    """A duck-typed worker: no invocation of its own."""

    def infer(self, inference_input, **kwargs):
        return f"duck:{inference_input}"

    async def ainfer(self, inference_input, **kwargs):
        return f"duck:{inference_input}"


def _bta(workers, **kwargs):
    defaults = {
        "breakdown_inferencer": _Worker(label="bd"),
        "predefined_sub_queries": ["q0", "q1"],
        "disable_aggregator": True,
        "fallback_mode": FallbackMode.NEVER,
        "max_retry": 0,
    }
    return BreakdownThenAggregateInferencer(
        worker_inferencers=workers, **{**defaults, **kwargs}
    )


def _call(inf, kind, **kwargs):
    if kind == "sync":
        return inf.infer("go", **kwargs)
    return asyncio.run(inf.ainfer("go", **kwargs))


def _host(tmp_path):
    return RunContext.root(workspace=InferencerWorkspace(root=str(tmp_path)))


def _closes(events):
    return sorted(label for kind, label in events if kind == "close")


# -- owned workers ------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_owned_workers_close_when_the_attempt_ends(kind, tmp_path):
    events = []
    built = []

    def factory(sub_query, index):
        built.append(_Worker(label=f"w{index}", events=events))
        return built[-1]

    _call(_bta(factory), kind, run_context=_host(tmp_path))
    assert _closes(events) == ["w0", "w1"]
    assert events.index(("close", "w0")) > events.index(("run", "w0"))


@pytest.mark.parametrize("kind", KINDS)
def test_every_call_closes_its_own_workers(kind, tmp_path):
    """B25: a reused BTA closed only the last call's workers, at a later
    ``adisconnect``."""
    events = []
    count = iter(range(100))

    def factory(sub_query, index):
        return _Worker(label=f"w{next(count)}", events=events)

    bta = _bta(factory)
    _call(bta, kind, run_context=_host(tmp_path / "one"))
    _call(bta, kind, run_context=_host(tmp_path / "two"))
    assert _closes(events) == ["w0", "w1", "w2", "w3"]


def test_a_sync_bta_inside_a_running_loop_closes_its_workers(tmp_path):
    events = []

    def factory(sub_query, index):
        return _Worker(label=f"w{index}", events=events)

    async def main():
        return _bta(factory).infer("go", run_context=_host(tmp_path))

    asyncio.run(main())
    assert _closes(events) == ["w0", "w1"]


def test_a_rerun_closes_the_first_attempts_workers_before_the_second_starts(
    tmp_path,
):
    events = []
    count = iter(range(100))

    class _Interactive:
        answers = ["rerun", "approve"]

        async def asend_response(self, *args, **kwargs):
            pass

        async def aget_input(self):
            return self.answers.pop(0)

    def factory(sub_query, index):
        return _Worker(label=f"w{next(count)}", events=events)

    bta = _bta(
        factory,
        interactive=_Interactive(),
        enable_checkpoint_results_review=True,
        predefined_sub_queries=["q0"],
    )
    _call(bta, "async", run_context=_host(tmp_path))
    assert events == [("run", "w0"), ("close", "w0"), ("run", "w1"), ("close", "w1")]


def test_a_factory_returning_one_object_twice_raises(tmp_path):
    shared = _Worker()
    bta = _bta(lambda sub_query, index: shared)
    with pytest.raises(StageOwnershipError, match="same _Worker"):
        _call(bta, "async", run_context=_host(tmp_path))


# -- cleanup failures -----------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_failed_close_after_success_publishes_then_raises_with_the_result(
    kind, tmp_path
):
    events = []

    def factory(sub_query, index):
        return _Worker(label=f"w{index}", events=events, close_fails=index == 0)

    ctx = _host(tmp_path)
    with pytest.raises(InvocationCleanupError) as info:
        _call(_bta(factory), kind, run_context=ctx)
    assert info.value.result == ("w0:q0", "w1:q1")
    assert info.value.errors == ("worker_00: OSError: w0 close failed",)
    assert read_outcome(ctx).cleanup_errors == info.value.errors
    assert _closes(events) == ["w0", "w1"]


def test_a_failed_close_during_a_failing_call_becomes_a_note(tmp_path):
    def factory(sub_query, index):
        return _Worker(label=f"w{index}", close_fails=True, fail=index == 1)

    with pytest.raises(Exception, match="worker failed") as info:
        _call(_bta(factory), "async", run_context=_host(tmp_path))
    notes = getattr(info.value, "__notes__", [])
    assert any("w0 close failed" in note for note in notes)


# -- borrowed stages ------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_borrowed_workers_are_never_closed_by_the_call(kind, tmp_path):
    events = []
    workers = [_Worker(label="a", events=events), _Worker(label="b", events=events)]
    _call(_bta(workers), kind, run_context=_host(tmp_path))
    assert _closes(events) == []


def test_a_host_call_binds_a_borrowed_worker_without_writing_its_backing(tmp_path):
    borrowed = _Worker(label="a")
    owned = []

    _call(_bta([borrowed]), "async", run_context=_host(tmp_path / "host"))
    assert borrowed.__dict__.get("_InferencerBase__workspace") is None

    def factory(sub_query, index):
        owned.append(_Worker())
        return owned[-1]

    _call(_bta(factory), "async", run_context=_host(tmp_path / "owned"))
    assert owned[0].__dict__.get("_InferencerBase__workspace") is not None


def test_a_bare_call_still_binds_a_borrowed_worker(tmp_path):
    borrowed = _Worker(label="a")
    bta = _bta(
        [borrowed],
        workspace=InferencerWorkspace(root=str(tmp_path)),
        predefined_sub_queries=["q0"],
    )
    _call(bta, "async")
    assert borrowed._workspace.root.endswith("worker_00")


def test_one_borrowed_duck_typed_worker_cannot_serve_concurrent_host_workers(
    tmp_path,
):
    duck = _Duck()
    with pytest.raises(UncertifiedConcurrentUseError, match="_Duck"):
        _call(_bta([duck]), "async", run_context=_host(tmp_path))


@pytest.mark.parametrize(
    "case", ("sync", "max_concurrency_1", "bare", "distinct_instances")
)
def test_a_duck_typed_worker_may_serve_sequential_or_bare_workers(case, tmp_path):
    workers: Any = [_Duck()]
    kwargs = {}
    run_context = _host(tmp_path)
    kind = "async"
    if case == "sync":
        kind = "sync"
    elif case == "max_concurrency_1":
        kwargs["max_concurrency"] = 1
    elif case == "bare":
        run_context = None
        kwargs["workspace"] = InferencerWorkspace(root=str(tmp_path))
    else:
        workers = [_Duck(), _Duck()]
    result = _call(_bta(workers, **kwargs), kind, run_context=run_context)
    assert result == ("duck:q0", "duck:q1")


# -- SDK leaves close per-call clients in their own loop -----------------------------------


def test_the_claude_sdk_sync_bridge_closes_its_client_in_the_loop_it_connected(
    tmp_path,
):
    loops = {}
    inf = ClaudeCodeSdkInferencer(target_path=str(tmp_path))

    async def fake_ainfer(inference_input, inference_config=None, **kwargs):
        loops["connected"] = asyncio.get_running_loop()

        async def disconnect():
            loops["closed"] = asyncio.get_running_loop()

        inf._client = object()
        inf._connected_loop = loops["connected"]
        inf._disconnect_fn = disconnect
        return "ok"

    inf._ainfer = fake_ainfer
    assert inf._infer("go") == "ok"
    assert loops["closed"] is loops["connected"]
    assert (inf._client, inf._disconnect_fn, inf._connected_loop) == (None, None, None)
    asyncio.run(inf.adisconnect())


# -- the breakdown stage under a host ctx (P9) -------------------------------------------


def test_a_host_bta_binds_its_breakdown_per_call_without_writing_it(tmp_path):
    """Nested BTAs that a factory builds around one shared breakdown: each call's
    breakdown runs in that call's ``breakdown`` workspace and the shared stage's
    backing is never written."""

    @attrs
    class _Breakdown(InferencerBase):
        roots: list = attrib(factory=list, kw_only=True)

        def _infer(self, inference_input, inference_config=None, **kwargs):
            self.roots.append(self._workspace.root)
            return "1. n0"

    shared = _Breakdown()

    def nested(sub_query, index):
        return BreakdownThenAggregateInferencer(
            breakdown_inferencer=shared,
            worker_inferencers=lambda sub_query, index: _Worker(),
            disable_aggregator=True,
        )

    outer = _bta(nested, predefined_sub_queries=["q0"])
    _call(outer, "async", run_context=_host(tmp_path))
    assert shared.roots == [
        str(tmp_path / "children" / "worker_00" / "children" / "breakdown")
    ]
    assert shared.__dict__.get("_InferencerBase__workspace") is None

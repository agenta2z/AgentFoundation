"""The single-flight guard (plan v8 §5.1, P6 c7): in host mode, one instance of a
class that is not host-pure certified runs one invocation at a time in the process.

The guard is held from frame open (a stream's: its first resumption) to release,
whatever the path, stage role or depth; an overlapping invocation raises
``UncertifiedConcurrentUseError`` before it runs anything, and gives back its path
claim. Sequential reuse, bare and legacy calls, and certified classes are
unaffected. Certification is read from the class itself (``vars(type(owner))``), so a
subclass never inherits it. A host parallel entry refuses overlapping items of an
uncertified class before any item runs.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    host_pure_certified,
    open_invocation,
    RunContext,
    UncertifiedConcurrentUseError,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode


class _Gate:
    """Holds the call whose input is ``"a"`` in flight until released."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()


@attrs
class _Gated(InferencerBase):
    """A call for ``"a"`` blocks on ``gate`` (when set); records call-state inits."""

    gate: _Gate = attrib(default=None, kw_only=True)
    inits: list = attrib(factory=list, kw_only=True)

    def _init_call_state(self, inference_input):
        self.inits.append(inference_input)
        return super()._init_call_state(inference_input)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"r:{inference_input}"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        if self.gate is not None and inference_input == "a":
            self.gate.entered.set()
            await self.gate.release.wait()
        return f"r:{inference_input}"


@attrs
class _SubGated(_Gated):
    pass


@attrs
class _Stream(StreamingInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "s"

    async def _ainfer_streaming(self, prompt, **kwargs):
        for chunk in ("a", "b"):
            yield chunk


def _host():
    return RunContext.root()


async def _overlap(first_call, second_call, gate):
    """Start ``first_call`` (input ``"a"``), wait until it blocks on ``gate``, run
    ``second_call`` while it is in flight, then release the first. Returns both
    outcomes, an exception standing for its call."""
    first = asyncio.ensure_future(first_call)
    await asyncio.wait_for(gate.entered.wait(), 5)
    try:
        second = await second_call
    except Exception as exc:
        second = exc
    gate.release.set()
    return await asyncio.wait_for(first, 5), second


def test_two_overlapping_host_calls_on_one_uncertified_leaf(tmp_path):
    async def main():
        gate = _Gate()
        leaf = _Gated(gate=gate)
        first, second = await _overlap(
            leaf.ainfer("a", run_context=_host()),
            leaf.ainfer("b", run_context=_host()),
            gate,
        )
        return leaf, first, second

    leaf, first, second = asyncio.run(main())
    assert first == "r:a"
    assert isinstance(second, UncertifiedConcurrentUseError)
    assert "_Gated" in str(second) and "factory slot" in str(second)
    assert leaf.inits == ["a"]


@pytest.mark.parametrize("mode", ("bare", "legacy_roots"))
def test_overlapping_bare_calls_are_unaffected(mode, tmp_path):
    async def main():
        gate = _Gate()
        leaf = _Gated(gate=gate)
        kwargs = {}
        if mode == "legacy_roots":
            leaf.workspace = InferencerWorkspace(root=str(tmp_path))
        return await _overlap(
            leaf.ainfer("a", **kwargs), leaf.ainfer("b", **kwargs), gate
        )

    assert asyncio.run(main()) == ("r:a", "r:b")


def test_sequential_reuse_and_a_call_after_a_failure_pass():
    leaf = _Gated()
    root = _host()
    assert leaf.infer("a", run_context=root.child("x")) == "r:a"
    assert leaf.infer("b", run_context=root.child("y")) == "r:b"

    @attrs
    class _Fails(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **kwargs):
            raise ValueError("boom")

    failing = _Fails(fallback_mode=FallbackMode.NEVER, max_retry=0)
    for _ in range(2):
        with pytest.raises(ValueError, match="boom"):
            failing.infer("q", run_context=_host())


def test_a_refused_invocation_gives_back_its_path_claim():
    """``leaf`` is in flight at ``/a``; its overlapping invocation at ``/b`` is
    refused and leaves ``/b`` free for another owner."""
    leaf, other, root = _Gated(), _Gated(), _host()
    token = enter_run(root.child("a"))
    try:
        with open_invocation(leaf):
            inner = enter_run(root.child("b"))
            try:
                with pytest.raises(UncertifiedConcurrentUseError):
                    with open_invocation(leaf):
                        pass
                with open_invocation(other):
                    pass
            finally:
                exit_run(inner)
    finally:
        exit_run(token)


def test_a_certified_class_may_overlap_and_a_subclass_does_not_inherit(monkeypatch):
    monkeypatch.setattr(_Gated, "_HOST_PURE_CERTIFIED", True, raising=False)
    assert host_pure_certified(_Gated) and not host_pure_certified(_SubGated)

    async def main(cls):
        gate = _Gate()
        inf = cls(gate=gate)
        return await _overlap(
            inf.ainfer("a", run_context=_host()),
            inf.ainfer("b", run_context=_host()),
            gate,
        )

    assert asyncio.run(main(_Gated)) == ("r:a", "r:b")
    first, second = asyncio.run(main(_SubGated))
    assert first == "r:a" and isinstance(second, UncertifiedConcurrentUseError)


def test_overlapping_streams_are_refused_at_first_resumption():
    stream = _Stream()

    async def main():
        first = stream.ainfer_streaming("q", run_context=_host())
        assert await first.__anext__() == "a"
        second = stream.ainfer_streaming("q", run_context=_host())
        with pytest.raises(UncertifiedConcurrentUseError):
            await second.__anext__()
        assert [chunk async for chunk in first] == ["b"]
        assert [chunk async for chunk in stream.ainfer_streaming("q")] == ["a", "b"]

    asyncio.run(main())


def test_overlapping_sync_calls_from_two_threads_are_refused():
    started, release = threading.Event(), threading.Event()

    @attrs
    class _Blocking(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **kwargs):
            started.set()
            release.wait(5)
            return "done"

    leaf = _Blocking()
    results = {}

    def first():
        results["first"] = leaf.infer("a", run_context=_host())

    thread = threading.Thread(target=first)
    thread.start()
    started.wait(5)
    try:
        with pytest.raises(UncertifiedConcurrentUseError):
            leaf.infer("b", run_context=_host())
    finally:
        release.set()
        thread.join(5)
    assert results == {"first": "done"}


# -- shared stages across BTA calls -----------------------------------------------------


def _bta(breakdown, tmp_path):
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=breakdown,
        worker_inferencers=lambda sub_query, index: _Gated(),
        disable_aggregator=True,
        workspace=InferencerWorkspace(root=str(tmp_path)),
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )


def test_one_borrowed_breakdown_shared_by_two_overlapping_bta_calls_raises(tmp_path):
    async def main():
        gate = _Gate()
        breakdown = _Gated(gate=gate)
        first_bta = _bta(breakdown, tmp_path / "one")
        second_bta = _bta(breakdown, tmp_path / "two")
        return await _overlap(
            first_bta.ainfer("a", run_context=_host()),
            second_bta.ainfer("b", run_context=_host()),
            gate,
        )

    first, second = asyncio.run(main())
    assert isinstance(second, UncertifiedConcurrentUseError)
    assert "_Gated" in str(second)


# -- parallel entries -------------------------------------------------------------------


def test_host_parallel_entries_refuse_overlapping_items_of_an_uncertified_class():
    leaf = _Gated()
    with pytest.raises(UncertifiedConcurrentUseError, match="parallel_infer"):
        leaf.parallel_infer(["a", "b"], num_workers=2, run_context=_host())
    with pytest.raises(UncertifiedConcurrentUseError, match="aparallel_infer"):
        asyncio.run(leaf.aparallel_infer(["a", "b"], run_context=_host()))
    assert leaf.inits == []
    assert leaf.parallel_infer(["a", "b"], num_workers=1, run_context=_host()) == [
        "r:a",
        "r:b",
    ]
    assert leaf.parallel_infer(["a", "b"], num_workers=2) == ["r:a", "r:b"]

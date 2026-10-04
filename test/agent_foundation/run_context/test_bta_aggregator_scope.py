"""A BTA call resolves its aggregator once and never writes the slot (plan v8 §5.7,
P6 c6; B2).

``build_aggregator`` resolves the slot per call (``_BTA_AGGREGATOR``): a factory's
product is owned by the call and closes when the call ends; an instance is borrowed.
The slot keeps its value, so the next call builds its own aggregator instead of
reusing call 1's with its feed and session. Traversals (and so ``pre_retry``) see the
call's aggregator while the call runs, and the slot otherwise.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import BtaCallSummary, RunContext
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode

KINDS = ("sync", "async")


@attrs
class _Agg(InferencerBase):
    """An aggregator that writes its deliverable and records its closes."""

    closed: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        ws = self._workspace
        if ws is not None:
            os.makedirs(ws.outputs_dir, exist_ok=True)
            with open(ws.output_path("output.md"), "w", encoding="utf-8") as f:
                f.write("aggregated")
        return "agg"

    async def adisconnect(self):
        self.closed.append(self)


@attrs
class _Leaf(InferencerBase):
    fail: bool = attrib(default=False, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        if self.fail:
            raise RuntimeError("worker failed")
        return f"w:{inference_input}"


def _bta(aggregator, workers=None, **kwargs):
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Leaf(),
        worker_inferencers=workers or (lambda sub_query, index: _Leaf()),
        aggregator_inferencer=aggregator,
        predefined_sub_queries=["q0", "q1"],
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
        **kwargs,
    )


def _call(inf, kind, **kwargs):
    if kind == "sync":
        return inf.infer("go", **kwargs)
    return asyncio.run(inf.ainfer("go", **kwargs))


def _host(tmp_path):
    return RunContext.root(workspace=InferencerWorkspace(root=str(tmp_path)))


@pytest.mark.parametrize("kind", KINDS)
def test_a_factory_aggregator_is_built_per_call_and_closed_when_it_ends(kind, tmp_path):
    built = []

    def factory():
        built.append(_Agg())
        return built[-1]

    bta = _bta(factory)
    _call(bta, kind, run_context=_host(tmp_path / "one"))
    _call(bta, kind, run_context=_host(tmp_path / "two"))
    assert bta.aggregator_inferencer is factory
    assert len(built) == 2 and built[0] is not built[1]
    assert [agg.closed for agg in built] == [[built[0]], [built[1]]]


@pytest.mark.parametrize("kind", KINDS)
def test_an_instance_aggregator_is_borrowed_and_never_closed_by_the_call(
    kind, tmp_path
):
    aggregator = _Agg()
    bta = _bta(aggregator)
    assert _call(bta, kind, run_context=_host(tmp_path)) == "agg"
    assert bta.aggregator_inferencer is aggregator
    assert aggregator.closed == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_failing_call_leaves_the_slot_alone_and_closes_its_aggregator(kind, tmp_path):
    built = []

    def factory():
        built.append(_Agg())
        return built[-1]

    def workers(sub_query, index):
        return _Leaf(fail=True)

    bta = _bta(factory, workers)
    with pytest.raises(RuntimeError, match="worker failed"):
        _call(bta, kind, run_context=_host(tmp_path))
    assert bta.aggregator_inferencer is factory
    assert all(agg.closed == [agg] for agg in built)


def test_traversals_see_the_calls_aggregator_while_it_runs(tmp_path):
    built, seen = [], []

    def factory():
        built.append(_Agg())
        return built[-1]

    @attrs
    class _Probe(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **kwargs):
            seen.append(
                (
                    list(bta._iter_child_inferencers()),
                    dict(bta._iter_child_slots()).get("aggregator"),
                )
            )
            return "probe"

    bta = _bta(factory, lambda sub_query, index: _Probe())
    bta.predefined_sub_queries = ["q0"]
    _call(bta, "async", run_context=_host(tmp_path))
    ((children, slot_child),) = seen
    assert children == [built[0]] and slot_child is built[0]
    assert list(bta._iter_child_inferencers()) == [factory]


@attrs(slots=False)
class _ConcludeFailsOnce(BreakdownThenAggregateInferencer):
    failures: list = attrib(factory=lambda: ["once"], kw_only=True)

    def _conclude_attempt(self, result):
        if self.failures:
            self.failures.pop()
            raise RuntimeError("conclude failed once")
        return super()._conclude_attempt(result)


def test_pre_retry_archives_the_calls_aggregator_workspace(tmp_path):
    """The slot write used to be what let ``pre_retry`` reach a built aggregator."""
    built = []

    def factory():
        built.append(_Agg())
        return built[-1]

    bta = _ConcludeFailsOnce(
        breakdown_inferencer=_Leaf(),
        worker_inferencers=lambda sub_query, index: _Leaf(),
        aggregator_inferencer=factory,
        predefined_sub_queries=["q0"],
        fallback_mode=FallbackMode.NEVER,
        max_retry=2,
    )
    asyncio.run(bta.ainfer("go", run_context=_host(tmp_path)))
    assert len(built) == 1
    attempts = tmp_path / "children" / "aggregator" / ".attempts"
    assert any(path.name == "output.md" for path in attempts.rglob("*"))


def test_summary_at_reads_the_outcome_under_a_ctx_and_the_getter_without_one(
    tmp_path,
):
    bta = _bta(_Agg(), workspace=InferencerWorkspace(root=str(tmp_path / "bare")))
    _call(bta, "sync")
    assert InferencerBase._summary_at(bta, None) is bta.last_call_summary
    assert bta.last_call_summary.worker_count == 2

    ctx = _host(tmp_path / "host")
    host_bta = _bta(_Agg())
    _call(host_bta, "sync", run_context=ctx)
    summary = InferencerBase._summary_at(host_bta, ctx)
    assert isinstance(summary, BtaCallSummary) and summary.worker_count == 2
    assert InferencerBase._summary_at(host_bta, ctx.child("elsewhere")) is None

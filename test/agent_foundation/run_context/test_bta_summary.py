"""A BTA run ends by freezing a ``BtaCallSummary`` (plan v8 §5.7, P6 c3).

The summary is the run's last statement: discarded when an attempt opens, published
after ``_conclude_attempt``. Finalize reads it instead of live stages; without one
(the result came from an external fallback or a non-exception default) finalize takes
the leaf path (B36). A host BTA publishes it as its node's outcome; outside a host
ctx it also lands on the ``last_call_summary`` compat getter.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    BtaCallSummary,
    decode_state,
    encode_state,
    open_invocation,
    publish_result,
    read_outcome,
    read_result,
    RunContext,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode

KINDS = ("sync", "async")


@attrs
class _Leaf(InferencerBase):
    response: str = attrib(default="ok", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"{self.response}:{inference_input}"


def _bta(**kwargs):
    defaults = {
        "breakdown_inferencer": _Leaf(),
        "worker_inferencers": lambda sub_query, index: _Leaf(response=f"w{index}"),
        "aggregator_inferencer": _Leaf(response="agg", output_path="report.md"),
        "predefined_sub_queries": ["q0", "q1"],
    }
    bta = BreakdownThenAggregateInferencer(**{**defaults, **kwargs})
    bta.name = "sum"
    return bta


def _call(inf, kind, **kwargs):
    if kind == "sync":
        return inf.infer("go", **kwargs)
    return asyncio.run(inf.ainfer("go", **kwargs))


def _expected(ws_root, *, disable_aggregator=False):
    ws = InferencerWorkspace(root=str(ws_root))
    return BtaCallSummary(
        worker_child_names=["worker_00", "worker_01"],
        worker_workspace_roots=[ws.child("worker_00").root, ws.child("worker_01").root],
        aggregator_output_name="report.md",
        aggregator_workspace_root=ws.child("aggregator").root,
        disable_aggregator=disable_aggregator,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_a_host_run_publishes_its_summary_as_the_node_outcome(kind, tmp_path):
    bta = _bta()
    ctx = RunContext.root(workspace=InferencerWorkspace(root=str(tmp_path)))
    assert _call(bta, kind, run_context=ctx) == "agg:go"
    summary = read_outcome(ctx).summary
    assert summary == _expected(tmp_path)
    assert summary.worker_count == 2
    assert bta.last_call_summary is None


@pytest.mark.parametrize("kind", KINDS)
def test_a_bare_run_lands_on_the_compat_getter(kind, tmp_path):
    bta = _bta(workspace=InferencerWorkspace(root=str(tmp_path)))
    _call(bta, kind)
    assert bta.last_call_summary == _expected(tmp_path)


def test_without_a_workspace_the_roots_are_none(tmp_path):
    bta = _bta(
        disable_aggregator=True,
        aggregator_inferencer=None,
        checkpoint_dir=str(tmp_path),
    )
    bta.infer("go")
    summary = bta.last_call_summary
    assert summary.worker_child_names == ("worker_00", "worker_01")
    assert summary.worker_workspace_roots == (None, None)
    assert summary.aggregator_workspace_root is None
    assert summary.disable_aggregator is True


@attrs(slots=False)
class _ConcludeFails(BreakdownThenAggregateInferencer):
    def _conclude_attempt(self, result):
        raise RuntimeError("conclude failed")


@pytest.mark.parametrize("kind", KINDS)
def test_a_conclude_hook_that_raises_leaves_no_summary(kind, tmp_path):
    bta = _ConcludeFails(
        breakdown_inferencer=_Leaf(),
        worker_inferencers=lambda sub_query, index: _Leaf(),
        predefined_sub_queries=["q0"],
        disable_aggregator=True,
        max_retry=0,
        fallback_mode=FallbackMode.NEVER,
    )
    ctx = RunContext.root(workspace=InferencerWorkspace(root=str(tmp_path)))
    with pytest.raises(RuntimeError, match="conclude failed"):
        _call(bta, kind, run_context=ctx)
    assert read_outcome(ctx) is None


def test_opening_an_attempt_withdraws_a_published_summary():
    bta = _bta()
    with open_invocation(bta):
        publish_result(bta, bta._BTA_SUMMARY, BtaCallSummary())
        bta._open_attempt("go", use_async=False)
        assert read_result(bta, bta._BTA_SUMMARY) is None
    assert bta.last_call_summary is None


def test_finalize_without_a_summary_takes_the_leaf_path(tmp_path):
    bta = _bta(workspace=InferencerWorkspace(root=str(tmp_path)))
    with open_invocation(bta):
        assert bta._finalize_output("<Response>substitute</Response>") == "substitute"
    with open(bta.resolve_output_path(), encoding="utf-8") as f:
        assert f.read() == "substitute"


def test_the_summary_round_trips_through_the_state_codec():
    summary = _expected("/ws")
    assert decode_state(json.loads(json.dumps(encode_state(summary)))) == summary

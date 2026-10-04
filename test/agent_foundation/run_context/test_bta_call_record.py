"""A BTA call reads its paths from one record, taken when the call starts (plan v8
§5.7, P6 c1).

``_bta_call`` snapshots the workspace the call resolves under its own ctx, its
``checkpoint_dir``, and the checkpoint root they imply. Every BTA path site reads
the record, so a read inside a child's call, where the active ctx carries the
child's ``workspace_override``, still names the BTA's own paths. The precedence is
today's: a configured backing workspace wins over the host ctx's workspace, and a
``checkpoint_dir``-only BTA keeps its layout.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    NoInvocationError,
    open_invocation,
    RunContext,
)
from attr import attrib, attrs

KINDS = ("sync", "async")


@attrs
class _Leaf(InferencerBase):
    response: str = attrib(default="ok", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"{self.response}:{inference_input}"


@attrs
class _Probe(InferencerBase):
    """A worker that, inside its own call, reads its BTA's result path and the
    BTA's ambient workspace."""

    bta: Any = attrib(default=None, kw_only=True)
    seen: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append((self.bta._get_result_path("probe"), self.bta._workspace.root))
        return "probe"


def _workers(sub_query, index):
    return _Leaf(response=f"w{index}")


def _bta(workers, **kwargs):
    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Leaf(response="bd"),
        worker_inferencers=workers,
        aggregator_inferencer=_Leaf(response="agg"),
        predefined_sub_queries=["q0"],
        enable_result_save=True,
        **kwargs,
    )
    bta.name = "rec"
    return bta


def _call(inf, kind, **kwargs):
    if kind == "sync":
        return inf.infer("go", **kwargs)
    return asyncio.run(inf.ainfer("go", **kwargs))


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def _files(root):
    root = str(root)
    return {
        os.path.relpath(os.path.join(dirpath, name), root)
        for dirpath, _dirs, names in os.walk(root)
        for name in names
    }


def _checkpoints(ws_root):
    return _files(os.path.join(ws_root, "checkpoints"))


@pytest.mark.parametrize("kind", KINDS)
def test_sequential_host_calls_under_two_ctx_workspaces_write_two_trees(kind, tmp_path):
    bta = _bta(_workers)
    _call(bta, kind, run_context=_host(tmp_path / "one"))
    _call(bta, kind, run_context=_host(tmp_path / "two"))
    one, two = _checkpoints(tmp_path / "one"), _checkpoints(tmp_path / "two")
    assert one == two
    assert {
        "rec_result/main.pkl",
        "aggregator_result/rec.aggregator_result/main.pkl",
        "breakdown/__graph_expansion__breakdown_result/main.pkl",
    } <= one


@pytest.mark.parametrize("kind", KINDS)
def test_a_configured_backing_workspace_wins_over_the_ctx_workspace(kind, tmp_path):
    backing = InferencerWorkspace(root=str(tmp_path / "backing"))
    bta = _bta(_workers, workspace=backing)
    _call(bta, kind, run_context=_host(tmp_path / "one"))
    _call(bta, kind, run_context=_host(tmp_path / "two"))
    assert "rec_result/main.pkl" in _checkpoints(backing.root)
    assert _checkpoints(tmp_path / "one") == _checkpoints(tmp_path / "two") == set()


@pytest.mark.parametrize("kind", KINDS)
def test_a_read_under_a_childs_ctx_names_the_bta_calls_paths(kind, tmp_path):
    probe = _Probe()
    bta = _bta([probe])
    probe.bta = bta
    ws = InferencerWorkspace(root=str(tmp_path))
    _call(bta, kind, run_context=RunContext.root(workspace=ws))
    ((path, ambient_root),) = probe.seen
    assert path == os.path.join(ws.checkpoints_dir, "probe_result.pkl")
    assert ambient_root == ws.child(bta._worker_child_name(0)).root


@pytest.mark.parametrize("kind", KINDS)
def test_a_checkpoint_dir_only_bta_keeps_its_breakdown_layout(kind, tmp_path):
    """No aggregator: with one configured, a ``checkpoint_dir``-only BTA has no
    aggregator workspace and its finalize refuses to drop the aggregated output."""
    bta = _bta(_workers, checkpoint_dir=str(tmp_path), disable_aggregator=True)
    bta.enable_result_save = False
    assert _call(bta, kind) == "w0:q0"
    assert _files(tmp_path) == {
        ".bta_execution.lock",
        "bta_manifest.json",
        "breakdown/__graph_expansion__breakdown_result/main.pkl",
        "breakdown/__graph_expansion__breakdown_result/manifest.json",
    }


def test_a_checkpoint_dir_only_bta_keeps_its_graph_and_aggregator_paths(tmp_path):
    bta = _bta(_workers, checkpoint_dir=str(tmp_path))
    with open_invocation(bta):
        bta._open_attempt("go", use_async=False)
        spec = bta._build_subgraph_spec(["q0"])
        graph = bta._get_result_path("rec")
    worker, aggregator = spec.nodes
    assert graph == str(tmp_path / "rec_result.pkl")
    assert aggregator._get_result_path(aggregator.name) == str(
        tmp_path / "aggregator_result" / "rec.aggregator_result.pkl"
    )
    with pytest.raises(NotImplementedError):
        worker._get_result_path(worker.name)


def test_the_record_is_taken_once_per_invocation(tmp_path):
    bta = _bta(_workers, checkpoint_dir=str(tmp_path / "first"))
    with open_invocation(bta):
        record = bta._bta_call()
        bta.checkpoint_dir = str(tmp_path / "second")
        assert bta._bta_call() is record
        assert bta._get_result_path("x") == str(tmp_path / "first" / "x_result.pkl")
    with open_invocation(bta):
        assert bta._bta_call().checkpoint_root == str(tmp_path / "second")


def test_with_no_workspace_and_no_checkpoint_dir_there_is_no_result_path():
    bta = _bta(_workers)
    with open_invocation(bta):
        assert bta._bta_call().checkpoint_root is None
        with pytest.raises(NotImplementedError):
            bta._get_result_path("x")


def test_a_path_read_outside_an_invocation_raises():
    with pytest.raises(NoInvocationError):
        _bta(_workers)._get_result_path("x")

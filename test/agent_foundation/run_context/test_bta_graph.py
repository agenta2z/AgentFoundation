"""Each BTA attempt runs its own ``_BtaGraph`` (plan v8 §5.8, P7; B24).

The engine state a run kept on the BTA (``start_nodes``, the expansion limits, node
queues, the event callback) moves onto a per-attempt ``WorkGraph`` that copies the
engine settings the BTA configures and forwards logging and the graph-level result
path to it, so session logs and checkpoints are unchanged (the BTA goldens) while
overlapping runs of one BTA no longer share a graph. With its call state in the frame
and its graph per attempt, BTA (and MFI) are host-pure: one instance serves
overlapping host calls. Since P11 the BTA is no longer a ``WorkGraph`` itself.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import attr
import pytest
import yaml
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
    breakdown_then_aggregate_inferencer as bta_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
    BtaWorkspaceBusyError,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    host_pure_certified,
    RunContext,
)
from attr import attrib, attrs
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.workflow.workgraph import WorkGraph


class _Breakdown:
    """A duck-typed breakdown (one shared object, no invocation of its own); the call
    for ``"A"`` answers last."""

    async def ainfer(self, inference_input, **kwargs):
        await asyncio.sleep(0.05 if inference_input == "A" else 0)
        return f"1. {inference_input}-q0\n2. {inference_input}-q1"

    def infer(self, inference_input, **kwargs):
        return f"1. {inference_input}-q0\n2. {inference_input}-q1"


@attrs
class _Echo(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"w:{inference_input}"


def _bta(**kwargs):
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Breakdown(),
        worker_inferencers=lambda sub_query, index: _Echo(),
        breakdown_format="numbered_list",
        disable_aggregator=True,
        **kwargs,
    )


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def test_every_workgraph_field_is_classified():
    """A new engine field fails here instead of silently taking its default on the
    attempt graph."""
    fields = {f.name for f in attr.fields(WorkGraph)}
    logging = {f.name for f in attr.fields(Debuggable)}
    per_attempt = bta_module._BTA_GRAPH_PER_ATTEMPT
    copied = bta_module._BTA_GRAPH_COPIED
    defaults = bta_module._BTA_GRAPH_ENGINE_DEFAULTS
    groups = (per_attempt, copied, defaults, logging)
    assert sum(len(group) for group in groups) == len(set().union(*groups))
    assert fields == per_attempt | copied | defaults | logging


def test_a_bta_is_not_a_graph_and_carries_only_what_its_attempts_copy():
    """P11: the definition is an inferencer; each attempt runs its own graph."""
    owner = _bta()
    assert not isinstance(owner, WorkGraph)
    for name in bta_module._BTA_GRAPH_COPIED:
        assert hasattr(owner, name), name
    for name in bta_module._BTA_GRAPH_ENGINE_DEFAULTS | {
        "start_nodes",
        "subgraph_registry",
        "use_async",
        "max_expansion_depth",
        "max_total_nodes",
    }:
        assert not hasattr(owner, name), name


_CORE_PROJECTS = Path(__file__).resolve().parents[4]
_BTA_TARGETS = {
    "BTA": BreakdownThenAggregateInferencer,
    "MultiFlow": MultiFlowInferencer,
    "BreakdownThenAggregateInferencer": BreakdownThenAggregateInferencer,
    "MultiFlowInferencer": MultiFlowInferencer,
}


def _bta_nodes(node):
    """Every BTA / MultiFlow node of a parsed config, with its class."""
    if isinstance(node, dict):
        target = str(node.get("_target_", "")).rsplit(".", 1)[-1]
        if target in _BTA_TARGETS:
            yield _BTA_TARGETS[target], node
        for value in node.values():
            yield from _bta_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _bta_nodes(value)


def test_no_shipped_config_passes_a_constructor_argument_the_graph_base_took():
    """The BTA stopped inheriting ``WorkGraph``'s constructor fields. The config
    loader drops a key its target does not accept (with a warning), so a shipped
    YAML relying on one would lose it silently: none passes one to a BTA /
    MultiFlow node."""
    graph_only = {f.alias for f in attr.fields(WorkGraph) if f.init} - {
        f.alias for f in attr.fields(InferencerBase) if f.init
    }
    lost = {}
    nodes = 0
    for root in ("AgentFoundation/src", "OpenStartup/src", "OpenStartup/_dev"):
        for path in sorted((_CORE_PROJECTS / root).rglob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            if "_target_" not in text:
                continue
            for cls, node in _bta_nodes(yaml.safe_load(text)):
                nodes += 1
                accepted = {f.alias for f in attr.fields(cls) if f.init}
                dropped = set(node) & (graph_only - accepted)
                if dropped:
                    rel = path.relative_to(_CORE_PROJECTS)
                    lost[f"{rel}:{node.get('name')}"] = sorted(dropped)
    assert nodes > 10
    assert lost == {}


def test_the_attempt_graph_copies_the_engine_fields_and_forwards_logging(tmp_path):
    owner = _bta(
        enable_result_save=True,
        checkpoint_mode="jsonfy",
        max_concurrency=3,
        group_max_concurrency={"workers": 2},
        name="owner",
        workspace=InferencerWorkspace(root=str(tmp_path)),
    )

    @attrs(slots=True)
    class _Attempt:
        use_async: bool = attrib()

    graph = bta_module._bta_graph(owner, _Attempt(use_async=True))
    for name in bta_module._BTA_GRAPH_COPIED:
        assert getattr(graph, name) == getattr(owner, name), name
    assert graph.group_max_concurrency == {"workers": 2}
    engine_defaults = {f.name: f.default for f in attr.fields(WorkGraph)}
    for name in bta_module._BTA_GRAPH_ENGINE_DEFAULTS:
        default = engine_defaults[name]
        if isinstance(default, attr.Factory):
            default = default.factory()
        assert getattr(graph, name) == default, name
    assert (graph.name, graph.use_async, graph.max_expansion_depth) == (
        "owner",
        True,
        1,
    )
    assert graph.start_nodes == []
    assert sorted(graph.subgraph_registry) == ["bta_diamond", "bta_workers"]
    owner._rebuild_subgraph = lambda: "rebuilt by the owner"
    assert graph.subgraph_registry["bta_workers"]("expansion-id") == (
        "rebuilt by the owner"
    )
    logged = []
    owner.log = lambda *args, **kwargs: logged.append(args)
    graph.log_info("engine record", "Diag")
    assert logged == [("engine record", "Diag")]


@pytest.mark.parametrize("mode", ("bare", "host"))
def test_overlapping_calls_on_one_bta_each_run_their_own_graph(mode, tmp_path):
    """B24: call A's breakdown answers after call B's started; each host call runs
    and returns its own sub-queries. Bare calls have no guard, but they share the
    BTA's checkpoint root, whose lease (P8) refuses the second before it reads or
    writes anything."""
    if mode == "bare":
        bta = _bta(workspace=InferencerWorkspace(root=str(tmp_path)))
    else:
        bta = _bta()

    def kwargs(label):
        return {} if mode == "bare" else {"run_context": _host(tmp_path / label)}

    async def main():
        return await asyncio.gather(
            bta.ainfer("A", **kwargs("a")),
            bta.ainfer("B", **kwargs("b")),
            return_exceptions=True,
        )

    first, second = asyncio.run(main())
    assert first == ("w:A-q0", "w:A-q1")
    if mode == "bare":
        assert isinstance(second, BtaWorkspaceBusyError)
    else:
        assert second == ("w:B-q0", "w:B-q1")


def test_distinct_btas_run_concurrently_under_distinct_ctxs(tmp_path):
    first, second = _bta(), _bta()

    async def main():
        return await asyncio.gather(
            first.ainfer("A", run_context=_host(tmp_path / "a")),
            second.ainfer("B", run_context=_host(tmp_path / "b")),
        )

    assert asyncio.run(main()) == [("w:A-q0", "w:A-q1"), ("w:B-q0", "w:B-q1")]


def test_bta_and_mfi_are_certified_and_subclasses_are_not():
    @attrs(slots=False)
    class _Sub(BreakdownThenAggregateInferencer):
        pass

    assert host_pure_certified(BreakdownThenAggregateInferencer)
    assert host_pure_certified(MultiFlowInferencer)
    assert not host_pure_certified(_Sub)

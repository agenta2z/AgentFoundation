"""Per-call values a parent assigns to a stage travel as invocation keywords (plan
v8 §5.9, B18, P5 c4).

A class declares the keywords it accepts (``_INVOCATION_KEYWORDS``, merged along
the MRO). Every public entry pops them from its call kwargs into its invocation
frame before anything else, so they never reach a transport; ``_effective(name)``
returns the per-call value when one was passed (``None`` included), else the
configured instance value. BTA hands its stages their graph-reporter observer and
interactive handler this way instead of writing them onto the stage: a borrowed
stage keeps its configuration, and two calls never share one call's observer. A
duck-typed stage keeps the attribute protocol. A nested BTA worker gets its node
name the same way (``bta_node_name``, P7; B31), so two workers sharing one nested
BTA each run under their own name and its ``name`` never changes.
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType
from typing import Any

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.bta_checkpoints import (
    LEASE_FILE,
    MANIFEST_FILE,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.mock_inferencers.mock_bta_components import (
    MockWorker,
)
from agent_foundation.common.inferencers.run_context import (
    open_invocation,
    RunContext,
    RuntimeBindings,
    RuntimeKey,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrib, attrs

KINDS = ("sync", "async")


@attrs
class _Leaf(StreamingInferencerBase):
    """Streams two chunks; records the kwargs its transport received."""

    transport_kwargs: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.transport_kwargs.append(sorted(kwargs))
        return f"leaf:{inference_input}"

    async def _ainfer_streaming(self, prompt, **kwargs):
        self.transport_kwargs.append(sorted(kwargs))
        for chunk in ("a", "b"):
            yield chunk


@attrs
class _Interactive(InferencerBase):
    """A stage that reads ``interactive`` per call, like PTI and Conversational."""

    interactive: Any = attrib(default=None, kw_only=True)
    seen: list = attrib(factory=list, kw_only=True)
    _INVOCATION_KEYWORDS = MappingProxyType(
        {"interactive": RuntimeKey("tests._Interactive.interactive")}
    )

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append(self._effective("interactive"))
        return f"ok:{inference_input}"


class _Observer:
    def __init__(self, label):
        self.label, self.chunks = label, []

    async def __call__(self, chunk):
        self.chunks.append(chunk)


class _Reporter:
    """A graph reporter whose per-node observers and interactive handles are
    labelled with this reporter's name."""

    def __init__(self, name):
        self.name, self.observers = name, {}

    def node_stream_observer(self, node_id, flush_interval_ms=200.0):
        return self.observers.setdefault(node_id, _Observer(f"{self.name}:{node_id}"))

    def node_interactive(self, node_id):
        return f"{self.name}:{node_id}:interactive"

    async def on_graph_topology(self, event):
        pass

    async def on_node_status(self, node_id, status, error="", output_path=""):
        pass

    async def on_node_stream(self, node_id, content, is_final=True):
        pass

    async def on_graph_reconcile(self, statuses):
        pass

    def child_reporter(self, node_id):
        return self


def _call(inf, kind, text="q", **kwargs):
    if kind == "sync":
        return inf.infer(text, **kwargs)
    return asyncio.run(inf.ainfer(text, **kwargs))


# -- declaration and resolution -----------------------------------------------


def test_declarations_merge_along_the_mro():
    assert set(StreamingInferencerBase._invocation_keywords()) == {"stream_observer"}
    assert set(BreakdownThenAggregateInferencer._invocation_keywords()) == {
        "interactive",
        "bta_node_name",
    }
    assert InferencerBase._invocation_keywords() == {}


def test_effective_prefers_a_passed_value_even_none_else_the_configured_one():
    inf = _Interactive(interactive="configured")
    key = _Interactive._INVOCATION_KEYWORDS["interactive"]
    assert inf._effective("interactive") == "configured"
    with open_invocation(inf) as frame:
        assert inf._effective("interactive") == "configured"
        frame.put(key, "per-call")
        assert inf._effective("interactive") == "per-call"
        frame.put(key, None)
        assert inf._effective("interactive") is None
    assert inf.interactive == "configured"


@pytest.mark.parametrize("kind", KINDS)
def test_an_entry_pops_a_declared_keyword_into_its_frame(kind):
    inf = _Interactive(interactive="configured")
    _call(inf, kind, interactive="per-call")
    _call(inf, kind)
    assert inf.seen == ["per-call", "configured"]
    assert inf.interactive == "configured"


@pytest.mark.parametrize("kind", KINDS)
def test_a_per_call_observer_streams_the_call_and_never_reaches_the_transport(kind):
    configured, per_call = _Observer("configured"), _Observer("per-call")
    leaf = _Leaf(stream_observer=configured)
    if kind == "sync":
        assert leaf.infer("q", stream_observer=per_call) == "leaf:q"
        assert leaf.transport_kwargs == [[]]
    else:
        assert asyncio.run(leaf.ainfer("q", stream_observer=per_call)) == "ab"
        assert per_call.chunks == ["a", "b"]
        assert leaf.transport_kwargs == [[]]
    assert configured.chunks == []
    assert leaf.stream_observer is configured


@pytest.mark.parametrize("entry", ("ainfer_streaming", "infer_streaming"))
def test_streaming_entries_pop_declared_keywords_too(entry):
    leaf = _Leaf()
    if entry == "infer_streaming":
        chunks = list(leaf.infer_streaming("q", stream_observer=_Observer("x")))
    else:

        async def main():
            return [
                c
                async for c in leaf.ainfer_streaming(
                    "q", stream_observer=_Observer("x")
                )
            ]

        chunks = asyncio.run(main())
    assert chunks == ["a", "b"]
    assert leaf.transport_kwargs == [[]]


# -- BTA dispatch ---------------------------------------------------------------


def _bta(tmp_path, worker, **kwargs):
    """A one-worker BTA borrowing ``worker`` (a static worker list is shared, never
    cloned)."""
    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Leaf(),
        worker_inferencers=[worker],
        predefined_sub_queries=["s0"],
        disable_aggregator=True,
        **kwargs,
    )
    bta.name = "kw"
    return bta


def _host(tmp_path, reporter):
    return RunContext.root(
        workspace=InferencerWorkspace(root=str(tmp_path)),
        runtime=RuntimeBindings(graph_reporter=reporter),
    )


def test_a_borrowed_leaf_gets_each_bta_calls_own_observer(tmp_path):
    configured = _Observer("configured")
    leaf = _Leaf(stream_observer=configured)
    first, second = _Reporter("first"), _Reporter("second")
    asyncio.run(
        _bta(tmp_path, leaf).ainfer("go", run_context=_host(tmp_path / "1", first))
    )
    asyncio.run(
        _bta(tmp_path, leaf).ainfer("go", run_context=_host(tmp_path / "2", second))
    )
    (first_obs,) = first.observers.values()
    (second_obs,) = second.observers.values()
    assert first_obs.chunks == ["a", "b"] and second_obs.chunks == ["a", "b"]
    assert configured.chunks == []
    assert leaf.stream_observer is configured
    assert "interactive" not in vars(leaf)


def test_a_stage_that_reads_interactive_gets_it_per_call_and_keeps_its_own(tmp_path):
    stage = _Interactive(interactive="configured")
    reporter = _Reporter("r")
    asyncio.run(
        _bta(tmp_path, stage).ainfer("go", run_context=_host(tmp_path, reporter))
    )
    assert stage.seen == ["r:kw.worker_00:interactive"]
    assert stage.interactive == "configured"


def test_a_duck_typed_stage_keeps_the_attribute_protocol(tmp_path):
    stage = MockWorker()
    reporter = _Reporter("r")
    asyncio.run(
        _bta(tmp_path, stage).ainfer("go", run_context=_host(tmp_path, reporter))
    )
    assert isinstance(stage.stream_observer, _Observer)
    assert stage.interactive is not None


# -- nested BTA node name (B31) ---------------------------------------------------


def _nested(seen):
    """A nested BTA whose worker factory records, per call, the node name it runs
    under and the name its log records carry."""

    def worker(sub_query, index):
        seen.append((nested._effective("bta_node_name"), nested._log_display_name()))
        return _Leaf()

    nested = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Leaf(),
        worker_inferencers=worker,
        predefined_sub_queries=["n0"],
        disable_aggregator=True,
        enable_result_save=True,
    )
    nested.name = "nested"
    return nested


def _graph_results(ws_root):
    return sorted(
        p.name
        for p in (ws_root / "checkpoints").iterdir()
        if p.name not in ("breakdown", LEASE_FILE, MANIFEST_FILE)
    )


@pytest.mark.parametrize("kind", KINDS)
def test_two_workers_sharing_a_nested_bta_each_run_under_their_own_name(kind, tmp_path):
    seen = []
    nested = _nested(seen)
    outer = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Leaf(),
        worker_inferencers=[nested, nested],
        predefined_sub_queries=["s0", "s1"],
        disable_aggregator=True,
    )
    outer.name = "kw"
    _call(outer, kind, "go", run_context=_host(tmp_path, None))
    assert sorted(seen) == [
        ("kw.worker_00", "kw.worker_00"),
        ("kw.worker_01", "kw.worker_01"),
    ]
    for worker in ("worker_00", "worker_01"):
        root = tmp_path / "children" / worker
        assert _graph_results(root) == [f"kw.{worker}_result"]
        inner = root / "children" / "worker_00"
        assert _graph_results(inner) == [f"kw.{worker}.worker_00_result"]
    assert nested.name == "nested"
    seen.clear()
    _call(nested, kind, "direct", run_context=_host(tmp_path / "direct", None))
    assert seen == [("nested", "nested")]


@pytest.mark.parametrize("own_reporter", (False, True))
def test_under_a_reporter_a_nested_bta_runs_nameless_unless_it_has_its_own(
    own_reporter, tmp_path
):
    """The ctx path identifies a nested BTA's nodes under a reporter, so it runs
    without the prefix name, as before; one with its own reporter keeps it."""
    seen = []
    nested = _nested(seen)
    if own_reporter:
        nested.graph_reporter = _Reporter("own")
    asyncio.run(
        _bta(tmp_path, nested).ainfer("go", run_context=_host(tmp_path, _Reporter("r")))
    )
    name = "kw.worker_00" if own_reporter else None
    assert seen == [(name, name)]
    assert nested.name == "nested"

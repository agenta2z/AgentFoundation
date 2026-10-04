"""Characterization goldens for ``BreakdownThenAggregateInferencer`` and ``MultiFlowInferencer``.

The invocation-scoped runtime refactor rewrites how BTA and MFI thread run contexts,
checkpoints, graph events and output finalization. These goldens pin what a caller
observes TODAY on pure stubs, so each refactor commit either reproduces them exactly or
changes a golden in a reviewed commit that names the behaviour it changes:

  * **Fresh runs**: sync, async, and async under a host ``RunContext`` with a recording
    graph reporter, in both ``checkpoint_mode`` values. Pinned: the return value, the
    stub call order, the workspace tree, the session-log type sequences, the ordered
    graph-event stream and the host-store node paths.
  * **Partial resume**: run 1 completes; the checkpoints of workers 0 and 2 and the
    top-level result are deleted and the breakdown is promoted; a fresh instance then
    re-executes exactly those two workers.
  * **No aggregator**, the **MFI** post-processing chain, and the MFI interactive
    "rerun" review (one post-processing pass per call since P6 c2).

Known defects are pinned as they behave now, never fixed here; the test docstring names
the bug. B36 (finalize after fallback) lives in ``test_golden_bta_finalize.py``.

Stub call order is recorded as-is on the async paths too: the stubs do no I/O, so the
single-threaded event loop schedules workers in a fixed order.
"""

import asyncio
import json
import os
import re
import shutil

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext, RuntimeBindings
from attr import attrib, attrs

from ._golden import check_golden, Normalizer, session_log_types, workspace_tree

MODES = ("pickle", "jsonfy")
KINDS = ("sync", "async")
SUB_QUERIES = ("q0", "q1", "q2")
QUERY = "hello"
MANIFEST_REL = "artifacts/aggregation_report_manifest.json"

CALLS = []


@attrs
class Stub(InferencerBase):
    """Leaf answering ``<response>:<input>``; appends ``response`` to ``CALLS``."""

    _response = attrib(default="mock")
    _fail = attrib(default=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        CALLS.append(self._response)
        if self._fail:
            raise ValueError(f"{self._response} failed")
        return f"{self._response}:{inference_input}"


@pytest.fixture(autouse=True)
def _fresh_calls():
    CALLS.clear()
    yield
    CALLS.clear()


def bind(inf, ws_root, name):
    ws = InferencerWorkspace(root=str(ws_root))
    ws.ensure_dirs()
    inf._workspace = ws
    inf.name = name
    return inf


def indexed_worker(sub_query, index):
    """A module-level worker factory, so a resume can verify the BTA's identity."""
    return Stub(response=f"w{index}")


def make_bta(ws_root, mode="pickle", **overrides):
    kwargs = {
        "breakdown_inferencer": Stub(response="bd"),
        "worker_inferencers": indexed_worker,
        "aggregator_inferencer": Stub(response="agg"),
        "checkpoint_mode": mode,
        "enable_result_save": True,
        "resume_with_saved_results": True,
        "predefined_sub_queries": list(SUB_QUERIES),
    }
    kwargs.update(overrides)
    return bind(BreakdownThenAggregateInferencer(**kwargs), ws_root, "gbta")


def run(inf, kind, **kwargs):
    if kind == "sync":
        return inf.infer(QUERY, **kwargs)
    return asyncio.run(inf.ainfer(QUERY, **kwargs))


def chronological_sessions(ws_root, norm):
    """Session-log type sequences grouped by id-free path, oldest log first."""
    grouped = {}
    for path, types in session_log_types(ws_root, norm).items():
        grouped.setdefault(re.sub(r"<ID\d+>", "<ID>", path), []).append(types)
    return grouped


def snapshot(ws_root, norm, **records):
    data = {key: norm.value(val) for key, val in records.items()}
    data["tree"] = workspace_tree(ws_root, norm)
    data["session_chronological"] = chronological_sessions(ws_root, norm)
    sort_manifest_contributors(data["tree"])
    return data


def sort_manifest_contributors(tree):
    """The manifest lists contributors in ``os.listdir`` order, which follows the
    random logger ids; that order is not part of the contract."""
    manifest = tree.get(MANIFEST_REL)
    if isinstance(manifest, dict) and "json" in manifest:
        manifest["json"]["contributors"].sort(key=lambda c: (c["path"], c["category"]))


def golden_name(stem, kind, mode):
    return f"bta/{stem}_{mode}" if kind == "sync" else f"bta/{stem}_async_{mode}"


def drop_partial_results(ws_root):
    for index in (0, 2):
        shutil.rmtree(
            os.path.join(ws_root, "children", f"worker_0{index}", "checkpoints")
        )
    for rel in ("gbta_result.json", "gbta_result.json.types.json"):
        path = os.path.join(ws_root, "checkpoints", rel)
        if os.path.exists(path):
            os.remove(path)
    for rel in ("gbta_result", "aggregator_result"):
        shutil.rmtree(os.path.join(ws_root, "checkpoints", rel), ignore_errors=True)


def promote_breakdown(ws_root):
    path = InferencerWorkspace(root=str(ws_root)).checkpoint_path(
        os.path.join("breakdown", "decomposed_subtasks.json")
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"subtasks": [{"description": q} for q in SUB_QUERIES]}, f)


def worker_indices(calls):
    return sorted(int(call[1:]) for call in calls if call.startswith("w"))


class RecordingReporter:
    """Graph reporter appending every callback, in arrival order, to ``events``."""

    def __init__(self):
        self.events = []

    async def on_graph_topology(self, event):
        nodes = [[node["id"], str(node["status"])] for node in event.nodes]
        edges = [[edge["source"], edge["target"]] for edge in event.edges]
        self.events.append({"kind": "topology", "nodes": nodes, "edges": edges})

    async def on_node_status(self, node_id, status, error="", output_path=""):
        self.events.append(
            {
                "kind": "status",
                "node": node_id,
                "status": str(status),
                "error": str(error),
                "output_path": output_path,
            }
        )

    async def on_node_stream(self, node_id, content, is_final=True):
        self.events.append(
            {"kind": "stream", "node": node_id, "content": content, "final": is_final}
        )

    async def on_graph_reconcile(self, statuses):
        self.events.append({"kind": "reconcile", "statuses": dict(statuses)})

    def child_reporter(self, node_id):
        self.events.append({"kind": "child_reporter", "node": node_id})
        return self

    def node_stream_observer(self, node_id, flush_interval_ms=200.0):
        self.events.append({"kind": "stream_observer", "node": node_id})
        return None

    def node_interactive(self, node_id):
        self.events.append({"kind": "node_interactive", "node": node_id})
        return None


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_fresh(tmp_path, kind, mode):
    """Fresh run: the breakdown stub is never called (predefined sub-queries), only
    the top-level result honours ``checkpoint_mode`` (node checkpoints stay ``.pkl``)."""
    out = run(make_bta(tmp_path, mode), kind)
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(tmp_path, norm, result=out, calls=CALLS)
    check_golden(f"bta/fresh_{kind}_{mode}", data)


@pytest.mark.parametrize("mode", MODES)
def test_fresh_async_host(tmp_path, mode):
    """Async run under a host ``RunContext``. Pinned oddities: ``breakdown`` reports
    ``completed`` twice, the aggregator streams as ``aggregator`` but reports status as
    ``gbta.aggregator``, and the host store holds only the root node."""
    reporter = RecordingReporter()
    root = RunContext.root(
        workspace=InferencerWorkspace(root=str(tmp_path)),
        runtime=RuntimeBindings(graph_reporter=reporter),
    )
    out = run(make_bta(tmp_path, mode), "async", run_context=root)
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(
        tmp_path,
        norm,
        result=out,
        calls=CALLS,
        events=reporter.events,
        store_paths_sorted=sorted(root._store._nodes),
    )
    check_golden(f"bta/fresh_async_host_{mode}", data)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_resume_partial(tmp_path, kind, mode):
    """Partial resume re-executes workers 0 and 2 plus the aggregator, and reloads
    worker 1 from its checkpoint. B35 (fixed in P8): the workers are rebuilt from
    the plan committed in run 1's manifest, so they get run 1's sub-queries; they
    were rebuilt from the promoted breakdown before, whose subtasks parse into
    other queries (``**Description**: q0``)."""
    first = run(make_bta(tmp_path, mode), kind)
    first_calls = list(CALLS)
    drop_partial_results(tmp_path)
    promote_breakdown(tmp_path)
    CALLS.clear()
    second = run(make_bta(tmp_path, mode), kind)
    assert worker_indices(CALLS) == [0, 2]
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(
        tmp_path,
        norm,
        results=[first, second],
        calls=[first_calls, list(CALLS)],
        rerun_worker_indices_sorted=worker_indices(CALLS),
    )
    check_golden(golden_name("resume_partial", kind, mode), data)


@pytest.mark.parametrize("kind", KINDS)
def test_resume_without_promoted_breakdown(tmp_path, kind):
    """B37 (fixed in P8): a run of predefined sub-queries promotes no breakdown, so
    its resume rebuilds the workers from the committed plan in its manifest (it fed
    ``None`` into ``_build_subgraph_spec`` before); only the dropped results are
    computed again."""
    first = run(make_bta(tmp_path), kind)
    drop_partial_results(tmp_path)
    CALLS.clear()
    second = run(make_bta(tmp_path), kind)
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(
        tmp_path, norm, first_result=first, second_result=second, second_calls=CALLS
    )
    suffix = "" if kind == "sync" else "_async"
    check_golden(f"bta/resume_without_promoted_breakdown{suffix}", data)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("aggregator", ["configured", "unset"])
def test_no_aggregator(tmp_path, aggregator, mode):
    """``disable_aggregator=True`` with workers that write deliverables. With an
    aggregator still configured, ``_finalize_output`` resolves the unused aggregator
    workspace and skips the ``outputs/workers/`` symlinks; with none they appear. Either
    way ``outputs/aggregation_report.md`` holds only the last worker's answer."""
    overrides = {
        "disable_aggregator": True,
        "worker_inferencers": lambda sub_query, index: Stub(
            response=f"w{index}", output_path="answer.md"
        ),
    }
    if aggregator == "unset":
        overrides["aggregator_inferencer"] = None
    out = run(make_bta(tmp_path, mode, **overrides), "sync")
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(tmp_path, norm, result=out, calls=CALLS)
    suffix = "" if aggregator == "configured" else "_unset"
    check_golden(f"bta/no_aggregator{suffix}_{mode}", data)


class ScriptedInteractive:
    """Answers checkpoint reviews from a script and keeps every prompt shown."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    async def asend_response(self, prompt, flag=None, input_mode=None, **kwargs):
        self.prompts.append(prompt)

    async def aget_input(self):
        return self.answers.pop(0)


def make_mfi(ws_root, mode, parsed, winners, **extra):
    def parser(raw):
        parsed.append(raw)
        return f"parsed<{raw}>"

    def winner(raw):
        winners.append(raw)
        return 1

    flows = [
        {
            "input": f"task{i}",
            "initial_inferencer": Stub(response=f"f{i}i"),
            "followup_inferencer": Stub(response=f"f{i}f"),
            "end_condition": lambda state, out: True,
            "max_dynamic_steps": 1,
        }
        for i in range(2)
    ]
    mfi = MultiFlowInferencer(
        flow_configs=flows,
        aggregator_inferencer=Stub(response="agg"),
        response_parser=parser,
        winner_parser=winner,
        checkpoint_mode=mode,
        enable_result_save=True,
        **extra,
    )
    return bind(mfi, ws_root, "gmfi")


@pytest.mark.parametrize("mode", MODES)
def test_mfi(tmp_path, mode):
    """Two flows plus an aggregator; ``response_parser`` and ``winner_parser`` each see
    the aggregator output once."""
    parsed, winners = [], []
    mfi = make_mfi(tmp_path, mode, parsed, winners)
    out = run(mfi, "sync")
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(
        tmp_path,
        norm,
        result=out,
        calls=CALLS,
        parser_inputs=parsed,
        winner_parser_inputs=winners,
        winner_flow_idx=mfi.get_winner_flow_idx(),
    )
    check_golden(f"bta/mfi_{mode}", data)


def test_mfi_interactive_rerun(tmp_path):
    """A "rerun" review runs a second attempt of the same call (the flows and the
    aggregator run twice); MFI post-processes only the returned attempt's result, so
    both parsers run once and the answer is ``parsed<...>``. Until P6 c2 the rerun
    re-entered ``MFI._ainfer`` and the outer call parsed the parsed result again."""
    parsed, winners = [], []
    interactive = ScriptedInteractive(["rerun", "approve"])
    mfi = make_mfi(
        tmp_path,
        "pickle",
        parsed,
        winners,
        interactive=interactive,
        enable_checkpoint_results_review=True,
    )
    out = run(mfi, "async")
    assert len(parsed) == 1
    norm = Normalizer({"<WS>": tmp_path})
    data = snapshot(
        tmp_path,
        norm,
        result=out,
        calls=CALLS,
        parser_inputs=parsed,
        winner_parser_inputs=winners,
        review_prompts=interactive.prompts,
    )
    check_golden("bta/mfi_interactive_rerun", data)

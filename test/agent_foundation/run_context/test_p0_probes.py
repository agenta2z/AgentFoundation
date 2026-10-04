"""P0 probes for the bugs the refactor plan must confirm before it fixes them.

Each test asserts the CORRECT behaviour. A bug that reproduces today is marked
``xfail(strict=True)`` with the exception its reproduction raises, so the test flips
to XPASS (and fails the suite) in the phase that fixes it. A probe that did not
reproduce is a plain passing test pinning the behaviour observed.
"""

import asyncio
import json
import os
import shutil

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    RunContext,
)
from attr import attrib, attrs

KINDS = ("sync", "async")
QUERY = "hello"
SUB_QUERIES = ["q0", "q1", "q2", "q3"]
CALLS = []
NAMES = []
ATTEMPTS = []


@pytest.fixture(autouse=True)
def _fresh_records():
    for record in (CALLS, NAMES, ATTEMPTS):
        record.clear()
    yield
    for record in (CALLS, NAMES, ATTEMPTS):
        record.clear()


@attrs
class Stub(InferencerBase):
    _response = attrib(default="mock")
    _fail = attrib(default=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        ctx = active_run_context()
        CALLS.append((self._response, str(inference_input), ctx.path if ctx else None))
        if self._fail:
            raise ValueError(f"{self._response} failed")
        return f"{self._response}:{inference_input}"


@attrs
class DecomposingBreakdown(InferencerBase):
    """Writes the decomposition the way the extraction registry does, so the real
    ``_promote_child_checkpoints`` publishes it into the BTA's checkpoints."""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        path = self._workspace.output_path("decomposed_subtasks.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"subtasks": [{"description": q} for q in SUB_QUERIES]}, f)
        return list(SUB_QUERIES)


@attrs
class WritingFailingAggregator(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        path = self._workspace.output_path("aggregation_report.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("failed attempt report")
        raise ValueError("aggregator failed")


@attrs
class NameRecordingBTA(BreakdownThenAggregateInferencer):
    """Records the node name each call runs under."""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        NAMES.append(self._effective("bta_node_name"))
        return super()._infer(inference_input, inference_config, **kwargs)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        NAMES.append(self._effective("bta_node_name"))
        return await super()._ainfer(inference_input, inference_config, **kwargs)


@attrs
class AttemptCountingBTA(BreakdownThenAggregateInferencer):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        ATTEMPTS.append(inference_input)
        return super()._infer(inference_input, inference_config, **kwargs)


@attrs
class FirstAttemptFailingWorker(InferencerBase):
    _response = attrib(default="w")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        CALLS.append((self._response, str(inference_input), None))
        if len(ATTEMPTS) == 1:
            raise ValueError(f"{self._response} failed in attempt 1")
        return f"{self._response}:{inference_input}"


@attrs
class CallStateRecorder(InferencerBase):
    seen = attrib(factory=list)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"r:{inference_input}"

    def _init_call_state(self, inference_input):
        self.seen.append(inference_input)
        super()._init_call_state(inference_input)


def indexed_worker(sub_query, index):
    """A module-level worker factory: it has a resume identity (a lambda has none)."""
    return Stub(response=f"w{index}")


def first_attempt_failing_worker(sub_query, index):
    return FirstAttemptFailingWorker(response=f"w{index}")


def bind(inf, ws_root, name):
    ws = InferencerWorkspace(root=str(ws_root))
    ws.ensure_dirs()
    inf._workspace = ws
    inf.name = name
    return inf


def run(inf, kind, text=QUERY, **kwargs):
    if kind == "sync":
        return inf.infer(text, **kwargs)
    return asyncio.run(inf.ainfer(text, **kwargs))


def tree(root):
    paths = []
    for dirpath, _, files in os.walk(str(root)):
        rel_dir = os.path.relpath(dirpath, str(root))
        if "logs" not in rel_dir.split(os.sep):
            paths.extend(os.path.normpath(os.path.join(rel_dir, f)) for f in files)
    return sorted(paths)


def read_or_none(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def drop_after_worker_0(ws_root):
    """Leave the tree as a crash right after worker 0 persisted would."""
    for rel in (
        "children/worker_01/checkpoints",
        "checkpoints/aggregator_result",
        "checkpoints/gbta_result",
    ):
        shutil.rmtree(os.path.join(ws_root, rel), ignore_errors=True)
    for rel in ("gbta_result.json", "gbta_result.json.types.json"):
        path = os.path.join(ws_root, "checkpoints", rel)
        if os.path.exists(path):
            os.remove(path)


def worker_calls():
    return sorted(call[1] for call in CALLS if call[0].startswith("w"))


def sub_queries_run():
    return [q for q in SUB_QUERIES if any(q in text for text in worker_calls())]


# --- B15: LWI iteration workspaces -------------------------------------------------


def looping_lwi(ws_root):
    count = [0]

    def work(step_input, state):
        count[0] += 1
        if count[0] % 3:
            state["iteration"] += 1
        return f"iter_{state['iteration']}"

    steps = [
        WorkflowStepConfig(name="probe", inferencer=Stub(response="p")),
        WorkflowStepConfig(
            name="work",
            step_fn=work,
            output_state_key="work_output",
            loop_back_to="probe",
            loop_condition=lambda state, result: count[0] % 3 != 0,
            max_loop_iterations=3,
            enable_result_save=False,
        ),
    ]
    return LinearWorkflowInferencer(
        step_configs=steps,
        workspace=InferencerWorkspace(root=str(ws_root)),
        iteration_record_builder=lambda state: {"iteration": state["iteration"]},
        response_builder=lambda state: {
            "input": state["original_input"],
            "iteration": state["iteration"],
        },
    )


def test_b15_iteration_dirs_derive_from_call_start_workspace(tmp_path):
    base = tmp_path / "lwi"
    looping_lwi(base).infer("task-1")
    assert os.path.isdir(base / "iteration_2")
    assert os.path.isdir(base / "iteration_3")


def test_b15_reused_lwi_keeps_its_workspace(tmp_path):
    base = tmp_path / "lwi"
    lwi = looping_lwi(base)
    lwi.infer("task-1")
    assert lwi._workspace.root == str(base)


# --- B31: nested BTA shared by round-robin workers ---------------------------------


def nested_bta(name="nested"):
    nested = NameRecordingBTA(
        breakdown_inferencer=Stub(response="nbd"),
        worker_inferencers=lambda sub_query, index: Stub(response=f"nw{index}"),
        aggregator_inferencer=Stub(response="nagg"),
        predefined_sub_queries=["a", "b"],
        enable_result_save=True,
    )
    nested.name = name
    return nested


def run_outer(ws_root, workers, kind, sub_queries=("q0", "q1")):
    outer = BreakdownThenAggregateInferencer(
        breakdown_inferencer=Stub(response="bd"),
        worker_inferencers=workers,
        aggregator_inferencer=Stub(response="agg"),
        predefined_sub_queries=list(sub_queries),
        enable_result_save=True,
    )
    run(bind(outer, ws_root, "outer"), kind)
    return sorted(NAMES), sorted(CALLS), tree(ws_root)


@pytest.mark.parametrize("kind", KINDS)
def test_b31_shared_nested_bta_uses_each_worker_node_name(tmp_path, kind):
    nested = nested_bta()
    names, calls, files = run_outer(tmp_path / "shared", [nested, nested], kind)
    NAMES.clear()
    CALLS.clear()
    distinct = run_outer(tmp_path / "distinct", [nested_bta(), nested_bta()], kind)
    assert names == ["outer.worker_00", "outer.worker_01"]
    assert (names, calls, files) == distinct


@pytest.mark.parametrize("kind", KINDS)
def test_b31_nested_bta_name_unchanged_after_call(tmp_path, kind):
    nested = nested_bta()
    run_outer(tmp_path, [nested], kind, sub_queries=["q0"])
    assert nested.name == "nested"


# --- B34: per-call initialization --------------------------------------------------


def test_b34_parallel_infer_inits_call_state_per_item():
    inf = CallStateRecorder()
    assert inf.parallel_infer(["a", "b", "c"], num_workers=2) == ["r:a", "r:b", "r:c"]
    assert sorted(inf.seen) == ["a", "b", "c"]


def test_b34_aparallel_infer_inits_call_state_per_item():
    inf = CallStateRecorder()
    assert asyncio.run(inf.aparallel_infer(["a", "b"])) == ["r:a", "r:b"]
    assert sorted(inf.seen) == ["a", "b"]


@pytest.mark.parametrize("kind", KINDS)
def test_b34_iterator_inits_call_state_per_item(kind):
    factory_inputs = []

    def state_factory(inference_input):
        factory_inputs.append(inference_input)
        return {}

    inf = CallStateRecorder(state_factory=state_factory)
    out = run(inf, kind, text=iter(["a", "b"]))
    assert list(out) == ["r:a", "r:b"]
    assert inf.seen == ["a", "b"]
    # Both items run at one ctx path, so they share one node; call state is
    # populated once per node (M4), from the first item rather than the iterator.
    assert factory_inputs == ["a"]


# --- B35: resume keeps truncation and interactive selection ------------------------


class ScriptedSelection:
    async def asend_response(self, *args, **kwargs):
        return None

    async def aget_input(self):
        return "0|2"


def decomposing_bta(ws_root, **overrides):
    breakdown = DecomposingBreakdown(
        expected_extraction=[
            {
                "label": "decomposed_subtasks",
                "source": "response",
                "persist_to": "decomposed_subtasks.json",
                "checkpoint_scope": "parent",
            }
        ]
    )
    kwargs = {
        "breakdown_inferencer": breakdown,
        "worker_inferencers": indexed_worker,
        "aggregator_inferencer": Stub(response="agg"),
        "enable_result_save": True,
        "resume_with_saved_results": True,
    }
    kwargs.update(overrides)
    return bind(BreakdownThenAggregateInferencer(**kwargs), ws_root, "gbta")


@pytest.mark.parametrize("kind", KINDS)
def test_b35_resume_keeps_max_breakdown_truncation(tmp_path, kind):
    run(decomposing_bta(tmp_path, max_breakdown=2), kind)
    assert sub_queries_run() == ["q0", "q1"]
    drop_after_worker_0(tmp_path)
    CALLS.clear()
    assert run(decomposing_bta(tmp_path, max_breakdown=2), kind) == "agg:hello"
    assert sub_queries_run() == ["q1"]


def test_b35_resume_keeps_interactive_selection(tmp_path):
    selection = {
        "interactive": ScriptedSelection(),
        "enable_checkpoint_sub_query_selection": True,
    }
    run(decomposing_bta(tmp_path, **selection), "async")
    assert sub_queries_run() == ["q0", "q2"]
    drop_after_worker_0(tmp_path)
    CALLS.clear()
    assert run(decomposing_bta(tmp_path, **selection), "async") == "agg:hello"
    assert sub_queries_run() == ["q2"]


# --- B36: finalize after a result no BTA run produced ------------------------------


FALLBACKS = {"external": "fb:hello", "default": "default answer"}


def fallback_config(fallback):
    if fallback == "external":
        return {"fallback_inferencer": Stub(response="fb")}
    return {"default_return_or_raise": "default answer"}


def fallback_bta(ws_root, fallback, **overrides):
    kwargs = {
        "breakdown_inferencer": Stub(response="bd"),
        "worker_inferencers": lambda sub_query, index: Stub(response=f"w{index}"),
        "aggregator_inferencer": Stub(response="agg"),
        "enable_result_save": True,
        "resume_with_saved_results": True,
        "predefined_sub_queries": ["q0", "q1", "q2"],
        "max_retry": 1,
        **fallback_config(fallback),
    }
    kwargs.update(overrides)
    return bind(BreakdownThenAggregateInferencer(**kwargs), ws_root, "gbta")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fallback", sorted(FALLBACKS))
def test_b36_substitute_result_is_canonical_output(tmp_path, fallback, kind):
    def failing_worker(sub_query, index):
        return Stub(response=f"w{index}", fail=True)

    bta = fallback_bta(tmp_path, fallback, worker_inferencers=failing_worker)
    expected = FALLBACKS[fallback]
    assert run(bta, kind) == expected
    assert read_or_none(bta.resolve_output_path()) == expected


@pytest.mark.parametrize("fallback", sorted(FALLBACKS))
def test_b36_failed_aggregator_output_is_not_canonical(tmp_path, fallback):
    bta = fallback_bta(
        tmp_path, fallback, aggregator_inferencer=WritingFailingAggregator()
    )
    expected = FALLBACKS[fallback]
    assert run(bta, "sync") == expected
    assert read_or_none(bta.resolve_output_path()) == expected


# --- B37: resume without a promoted breakdown --------------------------------------


def predefined_bta(ws_root, cls=BreakdownThenAggregateInferencer, **overrides):
    kwargs = {
        "breakdown_inferencer": Stub(response="bd"),
        "worker_inferencers": indexed_worker,
        "aggregator_inferencer": Stub(response="agg"),
        "enable_result_save": True,
        "resume_with_saved_results": True,
        "predefined_sub_queries": ["q0", "q1", "q2"],
    }
    kwargs.update(overrides)
    return bind(cls(**kwargs), ws_root, "gbta")


@pytest.mark.parametrize("kind", KINDS)
def test_b37_resume_without_promoted_breakdown_completes(tmp_path, kind):
    run(predefined_bta(tmp_path), kind)
    drop_after_worker_0(tmp_path)
    CALLS.clear()
    assert run(predefined_bta(tmp_path), kind) == "agg:hello"
    assert sub_queries_run() == ["q1"]


def test_b37_sync_retry_after_worker_failure_completes(tmp_path):
    bta = predefined_bta(
        tmp_path,
        cls=AttemptCountingBTA,
        worker_inferencers=first_attempt_failing_worker,
        max_retry=2,
    )
    assert bta.infer(QUERY) == "agg:hello"


# --- B7: PlanThenImplementInferencer reuse -----------------------------------------


def stub_pti():
    return PlanThenImplementInferencer(
        planner_inferencer=Stub(response="plan"),
        executor_inferencer=Stub(response="impl"),
        analyzer_inferencer=None,
    )


def child_inputs():
    return [(call[0], call[1].split("\n")[0]) for call in CALLS]


def test_pti_bare_reuse_without_workspace_runs_each_call():
    pti = stub_pti()
    pti.infer("task-1")
    CALLS.clear()
    result = pti.infer("task-2")
    assert child_inputs() == [("plan", "task-2"), ("impl", "task-2")]
    assert str(result).startswith("impl:task-2")
    assert not [name for name in vars(pti) if name.startswith("_current_")]
    assert pti._workspace is None


def test_pti_second_root_call_runs_in_its_own_workspace(tmp_path):
    pti = stub_pti()
    for index in (1, 2):
        root = RunContext.root(
            workspace=InferencerWorkspace(root=str(tmp_path / f"root{index}"))
        )
        CALLS.clear()
        result = pti.infer(f"task-{index}", run_context=root)
    assert child_inputs() == [("plan", "task-2"), ("impl", "task-2")]
    assert str(result).startswith("impl:task-2")
    assert os.path.isfile(tmp_path / "root2" / "checkpoints" / "final_result.json")

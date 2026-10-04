"""Characterization goldens for where PTI and LWI children run and write.

Pins today's locations before the invocation-scoped refactor moves run state off
the instances. Every step child is a local :class:`_Probe` that records, from
inside its own call, the active ``RunContext`` path, the context workspace, the
workspace it resolves (``self._workspace``), ``legacy_mint`` and its output
path. Each golden holds, per call:

* the orchestrator result and the probe records in execution order;
* the sorted node paths in the host store (host calls only);
* the orchestrator's backing workspace and output path after the call;

plus the full normalized ``workspace_tree`` of ``tmp_path`` at the end.

Scenarios: PTI, LWI with static steps and LWI in dynamic mode, each as a bare
call then host calls on the same instance. Two looping LWI scenarios pin how
``_setup_iteration`` roots each iteration (B15, fixed in P9): one instance called
twice bare, and one under two host roots. The last scenario pins a looping LWI
with a workspace and the default ``_record_iteration`` snapshot.
"""

import json
from typing import Any, Dict, Optional

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

from ._golden import check_golden, Normalizer, workspace_tree


class _Recorder(list):
    def __deepcopy__(self, memo):
        return self


@attrs(slots=False)
class _Probe(InferencerBase):
    label = attrib(default="x")
    recorder = attrib(factory=_Recorder)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        ctx = active_run_context()
        ws = self._workspace
        self.recorder.append(
            {
                "label": self.label,
                "ctx_path": ctx.path if ctx else None,
                "ctx_workspace": str(ctx.workspace.root)
                if ctx and ctx.workspace
                else None,
                "workspace": str(ws.root) if ws else None,
                "legacy_mint": ctx.legacy_mint if ctx else None,
                "output_path": self.output_path,
                "resolved_output_path": self.resolve_output_path(),
            }
        )
        return f"{self.label}:{str(inference_input)[:24]}"


def _plain(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return {"type": type(value).__name__, "str": str(value)}
    return value


def _backing_root(inf) -> Optional[str]:
    ws = inf.__dict__.get("_InferencerBase__workspace")
    return str(ws.root) if ws is not None else None


def _call(inf, rec: _Recorder, text: str, host: Optional[str] = None) -> Dict[str, Any]:
    root = RunContext.root(workspace=InferencerWorkspace(root=host)) if host else None
    try:
        result: Any = inf.infer(text, run_context=root)
    except Exception as exc:
        result = {"error": type(exc).__name__}
    out = {"result": _plain(result), "children": list(rec)}
    rec.clear()
    if root is not None:
        out["store_paths"] = sorted(root._store._nodes)
    out["after"] = {
        "backing_workspace": _backing_root(inf),
        "output_path": inf.output_path,
        "resolved_output_path": inf.resolve_output_path(),
    }
    return out


def _golden(name: str, tmp_path, calls: Dict[str, Any]) -> None:
    norm = Normalizer({"<WS>": tmp_path})
    data = norm.value(calls)
    data["tree"] = workspace_tree(tmp_path, norm)
    check_golden(f"locations/{name}", data)


def _pti(rec: _Recorder) -> PlanThenImplementInferencer:
    return PlanThenImplementInferencer(
        planner_inferencer=_Probe(label="plan", recorder=rec),
        executor_inferencer=_Probe(label="impl", recorder=rec),
        analyzer_inferencer=None,
    )


def _looping_lwi(rec: _Recorder, workspace=None, record_builder=None):
    count = [0]

    def work(step_input, state):
        count[0] += 1
        if count[0] % 3:
            state["iteration"] += 1
        return f"iter_{state['iteration']}"

    steps = [
        WorkflowStepConfig(name="probe", inferencer=_Probe(label="p", recorder=rec)),
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
        step_configs=steps, workspace=workspace, iteration_record_builder=record_builder
    )


def _iteration_only(state) -> Dict[str, Any]:
    return {"iteration": state["iteration"]}


def test_pti_locations(tmp_path):
    """B7 (fixed in P9): a host call re-roots PTI for that invocation only, so host
    call 2 runs in host2 instead of replaying host1, and no backing is left."""
    rec = _Recorder()
    pti = _pti(rec)
    calls = {
        "bare": _call(pti, rec, "task-1"),
        "host1": _call(pti, rec, "task-2", str(tmp_path / "host1")),
        "host2": _call(pti, rec, "task-3", str(tmp_path / "host2")),
    }
    _golden("pti", tmp_path, calls)


def test_lwi_static_locations(tmp_path):
    rec = _Recorder()
    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=_Probe(label="one", recorder=rec)),
            WorkflowStepConfig(name="s2", inferencer=_Probe(label="two", recorder=rec)),
        ]
    )
    calls = {
        "bare": _call(lwi, rec, "task-1"),
        "host1": _call(lwi, rec, "task-2", str(tmp_path / "host1")),
        "host2": _call(lwi, rec, "task-3", str(tmp_path / "host2")),
    }
    _golden("lwi_static", tmp_path, calls)


def test_lwi_dynamic_locations(tmp_path):
    """Dynamic steps run under ctx paths ``/step_N`` while their workspaces are
    ``children/initial`` and ``children/roundNN``."""
    rec = _Recorder()
    lwi = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=_Probe(label="init", recorder=rec),
        default_followup_inferencer=_Probe(label="fu", recorder=rec),
        end_condition=lambda state, _: len(state["dynamic_step_results"]) >= 3,
        max_dynamic_steps=3,
        output_path="output.md",
    )
    calls = {
        "bare": _call(lwi, rec, "task-1"),
        "host1": _call(lwi, rec, "task-2", str(tmp_path / "host1")),
        "host2": _call(lwi, rec, "task-3", str(tmp_path / "host2")),
    }
    _golden("lwi_dynamic", tmp_path, calls)


def test_lwi_reused_bare_b15(tmp_path):
    """B15 (fixed in P9): every iteration derives from the call's own workspace
    (``lwi/iteration_3``, not ``lwi/iteration_2/iteration_3``) and re-roots only
    that invocation; the workflow's checkpoints and final result stay at ``lwi``,
    so bare call 2 starts there and returns call 1's cached final result."""
    rec = _Recorder()
    ws = InferencerWorkspace(root=str(tmp_path / "lwi"))
    lwi = _looping_lwi(rec, workspace=ws, record_builder=_iteration_only)
    calls = {
        "bare1": _call(lwi, rec, "task-1"),
        "bare2": _call(lwi, rec, "task-2"),
    }
    _golden("lwi_reused_bare_b15", tmp_path, calls)


def test_lwi_loop_host_sticky(tmp_path):
    """B15 under host roots (fixed in P9): no iteration is written into the
    backing, so host2 runs in its own root instead of replaying host1."""
    rec = _Recorder()
    lwi = _looping_lwi(rec, record_builder=_iteration_only)
    calls = {
        "host1": _call(lwi, rec, "task-1", str(tmp_path / "host1")),
        "host2": _call(lwi, rec, "task-2", str(tmp_path / "host2")),
    }
    _golden("lwi_loop_host_sticky", tmp_path, calls)


def test_lwi_loop_default_record_builder(tmp_path):
    """The default ``_record_iteration`` snapshot excludes ``iteration_records``,
    so the loop checkpoint and final-result saves serialize without a cycle."""
    rec = _Recorder()
    lwi = _looping_lwi(rec, workspace=InferencerWorkspace(root=str(tmp_path / "lwi")))
    _golden(
        "lwi_loop_default_record_builder",
        tmp_path,
        {"bare1": _call(lwi, rec, "task-1")},
    )

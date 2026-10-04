"""LWI iterations re-root one invocation from its own start (plan v8 §5.13, P9 c5).

Each iteration's workspace derives from the workspace the call was rooted at and
re-roots only that invocation (B15): iteration directories are flat
(``iteration_3``, not ``iteration_2/iteration_3``) and the instance keeps its
workspace. The workflow's own checkpoints and final result stay at that root, so a
resume finds the latest loop checkpoint; after the steps the invocation is back at
its root. A subclass's own two-argument ``_get_iteration_workspace`` (PTI) no longer
breaks the second meta-iteration.
"""

from __future__ import annotations

import json
import os

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode


@attrs
class _Probe(InferencerBase):
    """Records the iteration it runs in; crashes once in ``crash_in``."""

    seen: list = attrib(factory=list, kw_only=True)
    crash_in: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append(inference_input)
        if inference_input in self.crash_in:
            self.crash_in.remove(inference_input)
            raise RuntimeError(f"crash in iteration {inference_input}")
        return f"ran {inference_input}"


def _advance(step_input, state):
    if state["iteration"] < 3:
        state["iteration"] += 1
    return state["iteration"]


def _looping(root, probe):
    return LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(
                name="probe",
                inferencer=probe,
                input_builder=lambda state: state["iteration"],
            ),
            WorkflowStepConfig(
                name="advance",
                step_fn=_advance,
                loop_back_to="probe",
                loop_condition=lambda state, result: result < 3
                or state.get("_prev_iteration") != 3,
                enable_result_save=False,
            ),
        ],
        workspace=InferencerWorkspace(root=str(root)),
        iteration_record_builder=lambda state: {"iteration": state["iteration"]},
        response_builder=lambda state: state["iteration"],
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )


def _no_retry_probe(**kwargs):
    return _Probe(fallback_mode=FallbackMode.NEVER, max_retry=0, **kwargs)


def test_iterations_are_flat_and_the_checkpoints_stay_at_the_call_root(tmp_path):
    probe = _no_retry_probe()
    lwi = _looping(tmp_path, probe)
    assert lwi.infer("go") == 3
    assert probe.seen == [1, 2, 3]
    assert sorted(p.name for p in tmp_path.glob("iteration_*")) == [
        "iteration_2",
        "iteration_3",
    ]
    assert not list(tmp_path.glob("iteration_*/iteration_*"))
    assert not list(tmp_path.glob("iteration_*/checkpoints/*"))
    final = json.loads((tmp_path / "checkpoints" / "final_result.json").read_text())
    assert final["iteration"] == 3
    assert lwi._workspace.root == str(tmp_path)


def test_a_fresh_instance_resumes_a_crashed_run_in_its_last_iteration(tmp_path):
    crashed = _no_retry_probe(crash_in=[3])
    with pytest.raises(RuntimeError, match="crash in iteration 3"):
        _looping(tmp_path, crashed).infer("go")
    assert crashed.seen == [1, 2, 3]
    resumed = _no_retry_probe()
    assert _looping(tmp_path, resumed).infer("go") == 3
    assert resumed.seen == [3]


@attrs
class _Fixed(InferencerBase):
    response: str = attrib(default="", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.response


@attrs
class _Executor(InferencerBase):
    """Leaves a result file in the PTI's current iteration, so analysis runs."""

    pti: object = attrib(default=None, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        results = os.path.join(
            self.pti._current_iteration_workspace, "outputs", "benchmarks"
        )
        os.makedirs(results, exist_ok=True)
        with open(os.path.join(results, "result.txt"), "w") as f:
            f.write("ok")
        return "implemented"


@attrs
class _Analyzer(InferencerBase):
    decisions: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return json.dumps({"should_continue": self.decisions.pop(0)})


def test_a_multi_iteration_pti_runs_every_meta_iteration(tmp_path):
    executor = _Executor()
    pti = PlanThenImplementInferencer(
        planner_inferencer=_Fixed(response="## Plan\n1. Step one"),
        executor_inferencer=executor,
        analyzer_inferencer=_Analyzer(decisions=[True, True, False]),
        enable_analysis=True,
        enable_multiple_iterations=True,
        max_meta_iterations=3,
        workspace=InferencerWorkspace(root=str(tmp_path)),
    )
    executor.pti = pti
    result = pti.infer("task")
    assert [record.iteration for record in result.iteration_history] == [1, 2, 3]
    assert result.total_meta_iterations == 3
    followups = tmp_path / "followup_iterations"
    assert sorted(p.name for p in followups.iterdir()) == ["iteration_2", "iteration_3"]
    assert (followups / "iteration_3" / "outputs" / "benchmarks").is_dir()
    assert not list(followups.glob("iteration_*/followup_iterations"))

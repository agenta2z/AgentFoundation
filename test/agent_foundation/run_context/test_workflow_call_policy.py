"""Workflow checkpoint settings are per call (plan v8 §13, P9 c7: the save-flag subphase).

A workflow decides its own checkpoint policy per call (LWI's auto-enabled
checkpointing, Dual's per-call policy, PTI's, dynamic mode's expansion budget) and
keeps it in the invocation; the engine reads it through the configured fields. Under
a host ctx a parent workflow hands its child workflows their settings
(``enable_result_save``, ``resume_with_saved_results``, ``_result_root_override``)
for the calls they make within its invocation instead of writing them; without one
it writes them, as before. With that, LWI, Dual, MFDual and PTI are host-pure: one
instance serves overlapping host calls.
"""

from __future__ import annotations

import asyncio

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    host_pure_certified,
    open_invocation,
    RunContext,
)
from attr import attrib, attrs
from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
    StepResultSaveOptions,
)

_SETTINGS = ("enable_result_save", "resume_with_saved_results", "_result_root_override")


@attrs
class _Seeing(InferencerBase):
    """Records ``target``'s checkpoint settings as seen from inside its own call."""

    target: object = attrib(default=None, kw_only=True)
    seen: list = attrib(factory=list, kw_only=True)
    response: str = attrib(default="ok", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        if self.target is not None:
            self.seen.append({name: getattr(self.target, name) for name in _SETTINGS})
        return self.response


@attrs
class _Echo(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return str(inference_input).splitlines()[0]


@attrs
class _SlowFirst(InferencerBase):
    """Answers ``response``; the call whose input mentions ``slow`` yields to the
    loop first, so the other overlapping call starts while it is in flight. It
    keeps no state, so a test may certify it for overlapping use."""

    response: str = attrib(default="ok", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        if "slow" in str(inference_input):
            await asyncio.sleep(0.05)
        return self.response


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def _configured(inf):
    return {name: vars(inf)[name] for name in _SETTINGS}


def test_an_lwi_decides_its_checkpointing_per_call_and_keeps_its_configuration(
    tmp_path,
):
    step = _Seeing()
    lwi = LinearWorkflowInferencer(
        step_configs=[WorkflowStepConfig(name="s1", inferencer=step)]
    )
    step.target = lwi
    before = _configured(lwi)
    lwi.infer("go", run_context=_host(tmp_path))
    assert step.seen == [
        {
            "enable_result_save": StepResultSaveOptions.Always,
            "resume_with_saved_results": True,
            "_result_root_override": None,
        }
    ]
    assert _configured(lwi) == before


def test_a_dynamic_lwi_keeps_its_expansion_budget_per_call(tmp_path):
    lwi = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=_Seeing(),
        default_followup_inferencer=_Seeing(),
        end_condition=lambda state, result: len(state["dynamic_step_results"]) >= 2,
        max_dynamic_steps=2,
    )
    before = vars(lwi)["max_expansion_events"]
    lwi.infer("go", run_context=_host(tmp_path))
    assert vars(lwi)["max_expansion_events"] == before


def _dual(**kwargs):
    return DualInferencer(
        base_inferencer=kwargs.pop("base", None) or _Seeing(response="proposal"),
        review_inferencer=_Seeing(
            response="## Review\nSeverity: COSMETIC\nApproved: true"
        ),
        fixer_inferencer=None,
        consensus_config=ConsensusConfig(),
        **kwargs,
    )


def _pti_with_dual_planner():
    seeing = _Seeing(response="## Plan\n1. Step one")
    planner = _dual(base=seeing)
    seeing.target = planner
    pti = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=_Seeing(),
        analyzer_inferencer=None,
    )
    return pti, planner, seeing


def test_a_handed_setting_lasts_for_the_childs_calls_within_the_parents():
    parent = LinearWorkflowInferencer(step_configs=[])
    child = LinearWorkflowInferencer(step_configs=[])
    before = _configured(child)
    root = RunContext.root()
    token = enter_run(root)
    try:
        with open_invocation(parent):
            parent._configure_child_workflow(
                child,
                _result_root_override="/parent/checkpoints/child",
                enable_result_save=StepResultSaveOptions.Always,
                resume_with_saved_results=True,
                checkpoint_mode="jsonfy",
            )
            assert _configured(child) == before
            inner = enter_run(root.child("child"))
            try:
                with open_invocation(child):
                    assert child._result_root_override == "/parent/checkpoints/child"
                    assert child.enable_result_save == StepResultSaveOptions.Always
                    child._set_call_policy(enable_result_save=False)
                    assert child.enable_result_save is False
                    assert child.resume_with_saved_results is True
            finally:
                exit_run(inner)
            assert child._result_root_override is None
    finally:
        exit_run(token)
    assert _configured(child) == before


def test_without_a_host_ctx_the_settings_are_written_onto_the_child():
    parent = LinearWorkflowInferencer(step_configs=[])
    child = LinearWorkflowInferencer(step_configs=[])
    with open_invocation(parent):
        parent._configure_child_workflow(
            child,
            _result_root_override="/parent/checkpoints/child",
            enable_result_save=StepResultSaveOptions.Always,
            resume_with_saved_results=True,
        )
    assert _configured(child) == {
        "enable_result_save": StepResultSaveOptions.Always,
        "resume_with_saved_results": True,
        "_result_root_override": "/parent/checkpoints/child",
    }


def test_a_host_pti_leaves_its_child_workflow_unwritten(tmp_path):
    pti, planner, _ = _pti_with_dual_planner()
    before = _configured(planner)
    pti.infer("task", run_context=_host(tmp_path))
    assert _configured(planner) == before


def test_a_bare_pti_still_writes_its_child_workflow_settings(tmp_path):
    pti, planner, _ = _pti_with_dual_planner()
    pti._workspace = InferencerWorkspace(root=str(tmp_path))
    pti.infer("task")
    assert vars(planner)["_result_root_override"] is None
    assert vars(planner)["enable_result_save"] == StepResultSaveOptions.Always
    assert vars(planner)["resume_with_saved_results"] is True


def test_the_workflow_family_is_certified():
    for cls in (
        LinearWorkflowInferencer,
        DualInferencer,
        MultiFlowDualInferencer,
        PlanThenImplementInferencer,
    ):
        assert host_pure_certified(cls), cls.__name__


def test_overlapping_host_calls_on_one_pti_each_run_in_their_own_root(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_SlowFirst, "_HOST_PURE_CERTIFIED", True, raising=False)
    monkeypatch.setattr(_Echo, "_HOST_PURE_CERTIFIED", True, raising=False)
    pti = PlanThenImplementInferencer(
        planner_inferencer=_SlowFirst(response="## Plan\n1. Step one"),
        executor_inferencer=_Echo(),
        analyzer_inferencer=None,
    )

    async def main():
        return await asyncio.gather(
            pti.ainfer("slow task-a", run_context=_host(tmp_path / "a")),
            pti.ainfer("task-b", run_context=_host(tmp_path / "b")),
        )

    first, second = asyncio.run(main())
    assert "task-a" in str(first) and "task-b" in str(second)
    for root in ("a", "b"):
        assert (tmp_path / root / "checkpoints" / "final_result.json").is_file()


def test_overlapping_host_calls_on_one_dual_each_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(_SlowFirst, "_HOST_PURE_CERTIFIED", True, raising=False)
    monkeypatch.setattr(_Seeing, "_HOST_PURE_CERTIFIED", True, raising=False)
    dual = _dual(base=_SlowFirst(response="proposal"))

    async def main():
        return await asyncio.gather(
            dual.ainfer("slow one", run_context=_host(tmp_path / "one")),
            dual.ainfer("two", run_context=_host(tmp_path / "two")),
        )

    results = asyncio.run(main())
    assert [str(result) for result in results] == ["proposal", "proposal"]

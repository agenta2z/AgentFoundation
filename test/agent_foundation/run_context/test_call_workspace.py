"""A workflow re-roots itself for one invocation (plan v8 §5.13, P9 c4; B7 part 2).

``_set_call_workspace`` records the workspace in the invocation frame instead of
writing the instance backing: ``_workspace`` returns it until that invocation ends,
for that instance only. PTI uses it for its per-call base workspace under a host ctx,
so a second host call runs in its own root instead of replaying the first.
"""

from __future__ import annotations

import os

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import open_invocation, RunContext
from attr import attrs


@attrs
class _Leaf(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"{inference_input}+"


def test_a_call_workspace_lasts_for_its_invocation_and_leaves_the_backing(tmp_path):
    configured = InferencerWorkspace(root=str(tmp_path / "configured"))
    call = InferencerWorkspace(root=str(tmp_path / "call"))
    inf, other = _Leaf(workspace=configured), _Leaf(workspace=configured)
    with open_invocation(inf):
        inf._set_call_workspace(call)
        assert inf._workspace is call
        assert other._workspace is configured
        inf._set_call_workspace(None)
        assert inf._workspace is None
    assert inf._workspace is configured
    assert vars(inf)["_InferencerBase__workspace"] is configured
    with open_invocation(inf):
        assert inf._workspace is configured


def _pti():
    return PlanThenImplementInferencer(
        planner_inferencer=_Leaf(),
        executor_inferencer=_Leaf(),
        analyzer_inferencer=None,
    )


def test_each_host_call_of_one_pti_runs_and_logs_in_its_own_root(tmp_path):
    pti = _pti()
    for index in (1, 2):
        root = RunContext.root(
            workspace=InferencerWorkspace(root=str(tmp_path / f"root{index}"))
        )
        result = pti.infer(f"task-{index}", run_context=root)
        assert str(result).endswith(f"task-{index}++")
    for index in (1, 2):
        sessions = tmp_path / f"root{index}" / "logs" / "session"
        assert [p.name for p in sessions.iterdir() if p.suffix == ".jsonl"]
        assert (
            tmp_path / f"root{index}" / "checkpoints" / "final_result.json"
        ).is_file()
    assert vars(pti)["_InferencerBase__workspace"] is None


def test_a_bare_call_still_sets_the_resume_workspace_backing(tmp_path):
    resume = str(tmp_path / "resume")
    os.makedirs(resume)
    pti = PlanThenImplementInferencer(
        planner_inferencer=_Leaf(),
        executor_inferencer=_Leaf(),
        analyzer_inferencer=None,
        resume_workspace=resume,
    )
    pti.infer("task")
    assert pti._workspace.root == resume

"""MFI never mutates its definition (flow_configs / predefined_sub_queries) to
propagate the runtime input (M5 §2.4; B20, P9): each attempt derives the flow inputs
(``_BtaAttempt.effective_sub_queries``), in every mode, and the flows' followup
builders read their own entry."""

import copy

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    open_invocation,
    RunContext,
)
from attr import attrib, attrs


def _mfi(**kwargs):
    return MultiFlowInferencer(
        flow_configs=[{"input": "PLACEHOLDER_A"}, {"input": "PLACEHOLDER_B"}],
        propagate_runtime_input=True,
        disable_aggregator=True,
        **kwargs,
    )


@pytest.mark.parametrize("with_ctx", (False, True))
def test_an_attempt_derives_the_flow_inputs_and_the_definition_stays(with_ctx):
    mfi = _mfi()
    before = (
        copy.deepcopy(mfi.flow_configs),
        copy.deepcopy(mfi.predefined_sub_queries),
    )
    token = enter_run(RunContext.root(workspace=None)) if with_ctx else None
    try:
        with open_invocation(mfi):
            attempt = mfi._open_attempt("RUNTIME_INPUT", use_async=True)
            assert attempt.effective_sub_queries == ["RUNTIME_INPUT", "RUNTIME_INPUT"]
            assert mfi._get_effective_predefined_sub_queries() == [
                "RUNTIME_INPUT",
                "RUNTIME_INPUT",
            ]
            assert mfi._flow_input(1, mfi.flow_configs[1]) == "RUNTIME_INPUT"
    finally:
        if token is not None:
            exit_run(token)
    assert (mfi.flow_configs, mfi.predefined_sub_queries) == before


def test_without_propagation_the_configured_inputs_run():
    mfi = MultiFlowInferencer(
        flow_configs=[{"input": "A"}, {"input": "B"}], disable_aggregator=True
    )
    with open_invocation(mfi):
        mfi._open_attempt("RUNTIME_INPUT", use_async=True)
        assert mfi._get_effective_predefined_sub_queries() == ["A", "B"]
        assert mfi._flow_input(0, mfi.flow_configs[0]) == "A"


@attrs
class _Recorder(InferencerBase):
    seen: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append(str(inference_input))
        return "out"


def test_every_step_of_every_flow_gets_the_runtime_input(tmp_path):
    """A followup step that receives the injected upstream artifacts gets the
    flow's runtime input; it got the placeholder under any ctx before (E12)."""
    steps = [_Recorder() for _ in range(4)]
    flows = [
        {
            "initial_inferencer": steps[2 * i],
            "followup_inferencer": steps[2 * i + 1],
            "end_condition": lambda state, out: state["dynamic_step_count"] >= 2,
            "max_dynamic_steps": 2,
        }
        for i in range(2)
    ]
    mfi = MultiFlowInferencer(
        flow_configs=flows,
        propagate_runtime_input=True,
        inject_upstream_artifacts=True,
        disable_aggregator=True,
    )
    root = RunContext.root(workspace=InferencerWorkspace(root=str(tmp_path)))
    mfi.infer("RUNTIME TASK", run_context=root)
    assert [step.seen for step in steps] == [["RUNTIME TASK"]] * 4
    assert [cfg["input"] for cfg in mfi.flow_configs] == ["", ""]

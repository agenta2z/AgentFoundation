"""The workflow resume marker lives in the invocation (plan v8 §13, P9 c1).

The workflow engine sets ``_step_was_previously_attempted`` / ``_previous_attempt_info``
when a run resumes into a step that a crashed run had started, and the step reads them
from inside its own child call (PTI's executor input). They are components of the
workflow's invocation frame behind the same properties: the step still sees them, the
instance is never written, and a later invocation starts without them.
"""

from __future__ import annotations

import asyncio

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    NoInvocationError,
    open_invocation,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode


@attrs
class _Step(InferencerBase):
    """Records the workflow's resume marker as seen from inside its own call."""

    workflow: object = attrib(default=None, kw_only=True)
    seen: list = attrib(factory=list, kw_only=True)
    crashes: int = attrib(default=0, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        info = self.workflow._previous_attempt_info
        self.seen.append(
            (self.workflow._step_was_previously_attempted, info and info["step_name"])
        )
        if self.crashes:
            self.crashes -= 1
            raise RuntimeError("crash")
        return f"{inference_input}+"


def _workflow(tmp_path, crashing):
    steps = [_Step(), crashing, _Step()]
    lwi = LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="s1", inferencer=steps[0]),
            WorkflowStepConfig(name="s2", inferencer=steps[1]),
            # A loop makes the engine checkpoint the loop position and the marker.
            WorkflowStepConfig(
                name="s3",
                inferencer=steps[2],
                loop_back_to="s1",
                loop_condition=lambda state, result: False,
            ),
        ],
        workspace=InferencerWorkspace(root=str(tmp_path)),
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )
    for step in steps:
        step.workflow = lwi
    return lwi


def test_a_resumed_step_reads_the_marker_and_the_instance_is_never_written(tmp_path):
    crashing = _Step(crashes=1, fallback_mode=FallbackMode.NEVER, max_retry=0)
    lwi = _workflow(tmp_path, crashing)
    with pytest.raises(RuntimeError, match="crash"):
        asyncio.run(lwi.ainfer("go"))
    asyncio.run(lwi.ainfer("go"))
    assert crashing.seen == [(False, None), (True, "s2")]
    assert not [name for name in vars(lwi) if "attempt" in name and "backing" in name]


def test_each_invocation_starts_without_a_marker():
    lwi = LinearWorkflowInferencer(step_configs=[])
    with open_invocation(lwi):
        lwi._step_was_previously_attempted = True
        lwi._previous_attempt_info = {"step_name": "s2"}
    with open_invocation(lwi):
        assert lwi._step_was_previously_attempted is False
        assert lwi._previous_attempt_info is None


def test_the_marker_is_read_and_written_only_inside_an_invocation():
    lwi = LinearWorkflowInferencer(step_configs=[])
    with pytest.raises(NoInvocationError):
        lwi._step_was_previously_attempted
    with pytest.raises(NoInvocationError):
        lwi._previous_attempt_info = {"step_name": "s2"}

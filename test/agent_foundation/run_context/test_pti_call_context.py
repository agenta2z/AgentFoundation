"""PTI's per-call context lives in the invocation (plan v8 §13, P9 c3; B7 part).

``_current_base_workspace``, ``_current_iteration_workspace``,
``_current_inference_config`` and ``_current_inference_args`` are fields of one
``_PtiCall`` component of the PTI's invocation frame: the step methods (and the
children they call) read the call's own values, the instance is never written, and
each invocation starts empty.
"""

from __future__ import annotations

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    NoInvocationError,
    open_invocation,
)
from attr import attrib, attrs


@attrs
class _Planner(InferencerBase):
    """Records the PTI call context it sees from inside its own call."""

    pti: object = attrib(default=None, kw_only=True)
    seen: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append(dict(self.pti._current_inference_config))
        return "## Plan\n1. Step one"


@attrs
class _Executor(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "Implementation complete."


def _pti():
    planner = _Planner()
    pti = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=_Executor(),
        analyzer_inferencer=None,
    )
    planner.pti = pti
    return pti, planner


def test_each_call_runs_with_its_own_context_and_never_writes_the_instance():
    pti, planner = _pti()
    pti.infer("task-1", inference_config={"call": 1})
    pti.infer("task-2", inference_config={"call": 2})
    assert planner.seen == [{"call": 1}, {"call": 2}]
    assert not [name for name in vars(pti) if name.startswith("_current_")]


def test_each_invocation_starts_with_an_empty_context():
    pti, _ = _pti()
    with open_invocation(pti):
        pti._current_base_workspace = "/tmp/base"
        pti._current_inference_args = {"a": 1}
    with open_invocation(pti):
        assert pti._current_base_workspace is None
        assert pti._current_inference_args is None


def test_the_context_is_read_and_written_only_inside_an_invocation():
    pti, _ = _pti()
    with pytest.raises(NoInvocationError):
        pti._current_iteration_workspace
    with pytest.raises(NoInvocationError):
        pti._current_inference_config = {}

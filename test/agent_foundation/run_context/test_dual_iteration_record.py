"""Dual's iteration record lives in the invocation (plan v8 §13, P9 c2).

The review step builds the round's ``ConsensusIterationRecord``; the fix step of the
same run completes it and the finalize safety net checks it. It is a component of the
Dual's invocation frame: a run never writes it onto the instance, and the next run
starts without the previous run's record.
"""

from __future__ import annotations

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    NoInvocationError,
    open_invocation,
)
from attr import attrib, attrs


@attrs
class _Fixed(InferencerBase):
    response: str = attrib(default="", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.response


@attrs(slots=False)
class _RecordingDual(DualInferencer):
    """Records the iteration record each finalize sees."""

    seen: list = attrib(factory=list, kw_only=True)

    def _finalize_output(self, response):
        self.seen.append(self._last_iteration_record)
        return super()._finalize_output(response)


def _dual():
    return _RecordingDual(
        base_inferencer=_Fixed(response="proposal"),
        review_inferencer=_Fixed(
            response="## Review\nSeverity: COSMETIC\nApproved: true"
        ),
        fixer_inferencer=None,
        consensus_config=ConsensusConfig(),
    )


def test_a_run_finalizes_with_its_own_record_and_never_writes_the_instance():
    dual = _dual()
    dual.infer("first")
    dual.infer("second")
    first, second = dual.seen
    assert first is not None and second is not None and first is not second
    assert first.consensus_reached and second.consensus_reached
    assert "_last_iteration_record" not in vars(dual)


def test_each_invocation_starts_without_a_record():
    dual = _dual()
    with open_invocation(dual):
        dual._last_iteration_record = object()
    with open_invocation(dual):
        assert dual._last_iteration_record is None


def test_the_record_is_read_and_written_only_inside_an_invocation():
    dual = _dual()
    with pytest.raises(NoInvocationError):
        dual._last_iteration_record
    with pytest.raises(NoInvocationError):
        dual._last_iteration_record = object()

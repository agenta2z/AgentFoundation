"""Tests for max_iterations <= 0 / False / None meaning "no fixed cap".

Verifies that a non-positive (or falsy) ``max_iterations`` no longer means
"run zero rounds" (the old `range(0, 0)` bug) but instead "run until the model
stops on its own", while a high safety ceiling (`_UNBOUNDED_ITERATION_CEILING`)
still bounds a pathological runaway loop. Also confirms a positive cap still
bounds as before (regression guard).
"""

import pytest
from attr import attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational import (
    conversational_inferencer as ci_mod,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


_FINAL_ANSWER = "Here is my final answer."
_LOOPING_RESPONSE = (
    "```json ToolsToInvoke\n"
    '{"type": "conversation", "name": "clarification", '
    '"arguments": {"prompt": "pick?"}}\n'
    "```"
)


@attrs(slots=False)
class _FinalAnswerBase(InferencerBase):
    """Always returns a plain final answer (no tool calls) -> CI stops itself."""

    def _infer(self, inp, cfg=None, **kw):
        return _FINAL_ANSWER

    async def _ainfer(self, inp, cfg=None, **kw):
        return _FINAL_ANSWER


@attrs(slots=False)
class _LoopingBase(InferencerBase):
    """Always emits a conversation tool. With yolo_mode it is auto-resolved and
    the loop continues every round -> only the cap / ceiling can stop it."""

    def _infer(self, inp, cfg=None, **kw):
        return _LOOPING_RESPONSE

    async def _ainfer(self, inp, cfg=None, **kw):
        return _LOOPING_RESPONSE


class TestUnboundedSemantics:

    @pytest.mark.asyncio
    async def test_zero_runs_and_stops_on_final_answer(self):
        ci = ConversationalInferencer(base_inferencer=_FinalAnswerBase(), max_iterations=0)
        result = await ci.run_agentic_loop("hello")
        # Ran (NOT the old range(0,0) -> zero-iteration do-nothing) and the CI
        # stopped on its own when the model produced a final answer.
        assert result.iterations_used == 1
        assert not result.exhausted_max_iterations
        assert "final answer" in result.text.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cap", [0, -1, False, None])
    async def test_nonpositive_or_falsy_caps_are_unbounded_not_zero(self, cap):
        ci = ConversationalInferencer(base_inferencer=_FinalAnswerBase(), max_iterations=cap)
        result = await ci.run_agentic_loop("hello")
        assert result.iterations_used == 1
        assert not result.exhausted_max_iterations

    @pytest.mark.asyncio
    async def test_safety_ceiling_bounds_runaway(self, monkeypatch):
        # Shrink the ceiling so the test is fast; a never-stopping base loops to it.
        monkeypatch.setattr(ci_mod, "_UNBOUNDED_ITERATION_CEILING", 3)
        ci = ConversationalInferencer(
            base_inferencer=_LoopingBase(), max_iterations=0, yolo_mode=True,
        )
        result = await ci.run_agentic_loop("go")
        assert result.exhausted_max_iterations
        assert result.iterations_used == 3  # bounded by the safety ceiling

    @pytest.mark.asyncio
    async def test_positive_cap_still_bounds(self):
        # Regression: a positive cap behaves exactly as before.
        ci = ConversationalInferencer(
            base_inferencer=_LoopingBase(), max_iterations=2, yolo_mode=True,
        )
        result = await ci.run_agentic_loop("go")
        assert result.exhausted_max_iterations
        assert result.iterations_used == 2

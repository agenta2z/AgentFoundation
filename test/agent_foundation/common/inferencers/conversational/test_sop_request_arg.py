"""Tests for ``/sop <name> <request>`` and ``/resume_sop <name> <request>``.

The SOP-entry commands accept an optional free-text *request* that becomes the
first user turn of the just-entered/resumed SOP, so the agent starts acting on
the concrete goal instead of merely entering and idling. Two invocation paths:

  * Path A — the user types ``/sop X req`` directly: the command short-circuit
    enters the SOP, then *falls through* and runs an agentic turn on ``req``.
  * Path B — the LLM invokes ``sop`` as a tool: the seed is surfaced into
    history so the loop's self-continuation acts on it.
"""

import types

import pytest
from attr import attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.workflow.sop_state import SOPState


_FINAL_ANSWER = "Here is my final answer."


@attrs(slots=False)
class _FinalAnswerBase(InferencerBase):
    """Returns a plain final answer (no tool calls) -> CI stops after one round."""

    def _infer(self, inp, cfg=None, **kw):
        return _FINAL_ANSWER

    async def _ainfer(self, inp, cfg=None, **kw):
        return _FINAL_ANSWER


class _FakeSOP:
    phases: list = []


def _make_ci(**kw):
    ci = ConversationalInferencer(base_inferencer=_FinalAnswerBase(), **kw)
    # Avoid the real SOP registry: enter/reload are stubbed to no-ops.
    ci._enter_sop = lambda name, *, yolo=False: (
        SOPState(sop=_FakeSOP(), sop_name=name, yolo_mode=yolo),
        None,
    )
    ci._reload_sop_definition = lambda m: None
    return ci


class TestArgParsing:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "args",
        [
            "role_creation hire an MLE --yolo",
            "role_creation --yolo hire an MLE",  # order-independent
        ],
    )
    async def test_sop_splits_name_flags_and_request(self, args):
        ci = _make_ci()
        result = await ci._cmd_sop(args)
        assert ci.sop_state.sop_name == "role_creation"
        assert ci._consume_pending_followup() == "hire an MLE"
        assert "Starting on: hire an MLE" in result

    @pytest.mark.asyncio
    async def test_sop_without_request_sets_no_followup(self):
        ci = _make_ci()
        result = await ci._cmd_sop("role_creation")
        assert ci._consume_pending_followup() is None
        assert result == "Entered SOP 'role_creation'."

    @pytest.mark.asyncio
    async def test_resume_named_sop_takes_remaining_as_request(self):
        ci = _make_ci()
        susp = SOPState(sop=_FakeSOP(), sop_name="role_creation")
        susp.suspension_reason = "paused"
        ci._suspended_sops = [susp]
        result = await ci._cmd_resume_sop("role_creation continue with the JD draft")
        assert ci.sop_state.sop_name == "role_creation"
        assert ci._consume_pending_followup() == "continue with the JD draft"
        assert "Continuing on: continue with the JD draft" in result

    @pytest.mark.asyncio
    async def test_resume_unrecognized_first_token_is_a_name_lookup(self):
        # Safety: when the first token is NOT a known suspended SOP, the whole
        # arg is treated as a target name (clear error) rather than silently
        # resuming the most-recent SOP with the text as a request.
        ci = _make_ci()
        susp = SOPState(sop=_FakeSOP(), sop_name="role_creation")
        susp.suspension_reason = "paused"
        ci._suspended_sops = [susp]
        result = await ci._cmd_resume_sop("keep going on the JD")
        assert "No suspended SOP named" in result
        assert ci.sop_state is None
        assert ci._consume_pending_followup() is None

    @pytest.mark.asyncio
    async def test_resume_no_arg_resumes_most_recent(self):
        ci = _make_ci()
        susp = SOPState(sop=_FakeSOP(), sop_name="role_creation")
        susp.suspension_reason = "paused"
        ci._suspended_sops = [susp]
        result = await ci._cmd_resume_sop("")
        assert ci.sop_state.sop_name == "role_creation"
        assert ci._consume_pending_followup() is None
        assert "Resumed SOP 'role_creation'" in result


class TestConsumeOnce:

    def test_followup_popped_once_then_none(self):
        ci = _make_ci()
        ci._pending_followup = "do the thing"
        assert ci._consume_pending_followup() == "do the thing"
        assert ci._consume_pending_followup() is None


class TestPathAFollowThrough:
    """User-typed slash command: enter + run a turn on the request."""

    @pytest.mark.asyncio
    async def test_request_runs_an_agentic_turn(self):
        ci = _make_ci()
        result = await ci.run_agentic_loop("/sop role_creation hire an MLE")
        # The loop actually ran (not the old enter-and-return) and produced the
        # model's answer to the seeded request.
        assert result.iterations_used >= 1
        assert "final answer" in result.text.lower()
        assert ci.sop_state.sop_name == "role_creation"
        # The request was persisted as a user turn.
        assert any(
            m["role"] == "user" and m["content"] == "hire an MLE"
            for m in ci.get_messages()
        )

    @pytest.mark.asyncio
    async def test_no_request_is_terminal(self):
        ci = _make_ci()
        result = await ci.run_agentic_loop("/sop role_creation")
        # No seed -> command is terminal, no LLM round.
        assert result.iterations_used == 0
        assert "Entered SOP 'role_creation'." in result.text
        assert ci.sop_state.sop_name == "role_creation"


class TestPathBSeedIntoHistory:
    """LLM-invoked command tool: the seed is surfaced into history."""

    @pytest.mark.asyncio
    async def test_command_tool_seeds_followup_into_history(self):
        ci = _make_ci()
        tool_call = types.SimpleNamespace(
            name="sop", arguments={"args": "role_creation hire an MLE"}
        )
        result = await ci._execute_tool_call(tool_call)
        assert "Entered SOP 'role_creation'." in result
        # Seed surfaced as a user turn for the self-continuation to act on.
        assert any(
            m["role"] == "user" and m["content"] == "hire an MLE"
            for m in ci.get_messages()
        )
        # Consumed (not left dangling for a later turn).
        assert ci._consume_pending_followup() is None

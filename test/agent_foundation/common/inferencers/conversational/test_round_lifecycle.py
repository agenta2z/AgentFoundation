"""Integration tests for run_agentic_loop's per-round lifecycle hooks.

Drives a real ``ConversationalInferencer`` with a scripted FAKE base inferencer
(no network) and asserts the firing contract of ``on_round_start`` /
``on_round_complete``:

  (i)   on_round_start fires once per iteration, INCLUDING an action-tool
        continuation round (round 1 = action tool, round 2 = final text).
  (ii)  on_round_complete fires per round with display_text non-empty for a
        text round and '' for a pure-tool round.
  (iii) on_round_complete fires BEFORE a conversation-tool widget handoff
        (i.e. before aget_input returns).
  (iv)  a GroupValidationError (two grouped tools with a duplicate output var)
        triggers a controlled self-continuation — the loop does NOT raise out,
        a follow-up round occurs.

The base is non-streaming and no interactive exposes ``stream_token_batches``,
so the loop takes the ``ainfer`` path (mirrors test_unbounded_iterations.py).
"""

import pytest
from attr import attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@attrs(slots=False)
class _ScriptedBase(InferencerBase):
    """Returns a pre-scripted sequence of raw responses, one per ainfer call.

    The script is read off ``self._script`` (a list) using ``self._idx``; once
    exhausted it repeats the last entry (so a runaway loop still terminates via
    the iteration cap rather than an IndexError).
    """

    def _setup_script(self, script):
        self._script = list(script)
        self._idx = 0

    def _next(self):
        i = min(self._idx, len(self._script) - 1)
        self._idx += 1
        return self._script[i]

    def _infer(self, inp, cfg=None, **kw):
        return self._next()

    async def _ainfer(self, inp, cfg=None, **kw):
        return self._next()


class _FakeInteractive:
    """Minimal interactive supporting the conversation-tool widget handoff.

    Deliberately does NOT expose ``stream_token_batches`` so the loop uses the
    non-streaming ``ainfer`` path. Records the ordered sequence of lifecycle
    events so the test can assert round-complete precedes aget_input.
    """

    def __init__(self, response, events):
        self._response = response
        self.events = events  # shared ordered log

    async def asend_response(self, response, flag=None, input_mode=None, **kwargs):
        self.events.append("asend_response")

    async def aget_input(self):
        self.events.append("aget_input")
        return self._response


class _ToolResult:
    """Mimics ToolExecutionResult: carries a ``.result`` string (no async)."""

    def __init__(self, result):
        self.result = result
        self.context_updates = None


def _make_executor(result_text="action done"):
    async def _executor(name, arguments):
        return _ToolResult(result_text)
    return _executor


# ---------------------------------------------------------------------------
# Response script fragments
# ---------------------------------------------------------------------------

_ACTION_TOOL_ROUND = (
    "Working on it now.\n"
    "```json ToolsToInvoke\n"
    '{"type": "action", "name": "noop_tool", "arguments": {}}\n'
    "```"
)
_FINAL_TEXT_ROUND = "All finished. Here is the answer."

_CONV_CLARIFY = (
    "```json ToolsToInvoke\n"
    '{"type": "conversation", "name": "clarification", '
    '"arguments": {"prompt": "What is your name?"}, "output": ["user_name"]}\n'
    "```"
)


def _capturing_hooks():
    """Return (on_round_start, on_round_complete, starts, completes)."""
    starts = []
    completes = []

    async def on_round_start(iter_idx, turn_num):
        starts.append((iter_idx, turn_num))
        return None  # no round context

    async def on_round_complete(ci, iter_idx, turn_num, raw, clean, dtext, conv_resp):
        completes.append(
            {
                "iter": iter_idx,
                "dtext": dtext,
                "has_conv_tool": conv_resp.has_conversation_tool,
            }
        )

    return on_round_start, on_round_complete, starts, completes


# ---------------------------------------------------------------------------
# (i) + (ii) action-tool continuation: 2 round-starts + display_text contract
# ---------------------------------------------------------------------------


class TestRoundStartFiresPerIterationIncludingContinuation:
    @pytest.mark.asyncio
    async def test_action_tool_then_final_text(self):
        base = _ScriptedBase()
        base._setup_script([_ACTION_TOOL_ROUND, _FINAL_TEXT_ROUND])
        from agent_foundation.resources.tools.models import ToolDefinition
        ci = ConversationalInferencer(
            base_inferencer=base,
            tool_executor=_make_executor(),
            tool_registry={"noop_tool": ToolDefinition(name="noop_tool")},
            max_iterations=5,
        )
        on_start, on_complete, starts, completes = _capturing_hooks()

        result = await ci.run_agentic_loop(
            "go", on_round_start=on_start, on_round_complete=on_complete
        )

        # (i) on_round_start fired once per iteration: round 0 (action tool) then
        # round 1 (the continuation that produced the final text).
        assert [s[0] for s in starts] == [0, 1]
        assert result.iterations_used == 2

        # (ii) on_round_complete fired per round. Round 0 is an action-tool round
        # whose preamble ("Working on it now.") IS display text (non-empty),
        # while round 1's display text is the final answer.
        assert len(completes) == 2
        assert completes[0]["iter"] == 0
        assert completes[0]["dtext"] == "Working on it now."
        assert completes[1]["iter"] == 1
        assert completes[1]["dtext"] == _FINAL_TEXT_ROUND

    @pytest.mark.asyncio
    async def test_pure_tool_round_has_empty_display_text(self):
        # A round that is ONLY a tool block (no surrounding prose) → dtext == "".
        pure_tool = (
            "```json ToolsToInvoke\n"
            '{"type": "action", "name": "noop_tool", "arguments": {}}\n'
            "```"
        )
        base = _ScriptedBase()
        base._setup_script([pure_tool, _FINAL_TEXT_ROUND])
        from agent_foundation.resources.tools.models import ToolDefinition
        ci = ConversationalInferencer(
            base_inferencer=base,
            tool_executor=_make_executor(),
            tool_registry={"noop_tool": ToolDefinition(name="noop_tool")},
            max_iterations=5,
        )
        on_start, on_complete, starts, completes = _capturing_hooks()

        await ci.run_agentic_loop(
            "go", on_round_start=on_start, on_round_complete=on_complete
        )

        assert completes[0]["dtext"] == ""           # pure-tool round
        assert completes[1]["dtext"] == _FINAL_TEXT_ROUND  # text round


# ---------------------------------------------------------------------------
# (iii) on_round_complete fires BEFORE the widget handoff (aget_input)
# ---------------------------------------------------------------------------


class TestRoundCompletePrecedesWidgetHandoff:
    @pytest.mark.asyncio
    async def test_round_complete_runs_before_aget_input(self):
        base = _ScriptedBase()
        base._setup_script([_CONV_CLARIFY])
        events: list[str] = []
        interactive = _FakeInteractive(response="Ada", events=events)

        ci = ConversationalInferencer(
            base_inferencer=base,
            interactive=interactive,
            max_iterations=1,  # one widget round; collected → loop continues once
        )

        order: list[str] = []

        async def on_round_start(iter_idx, turn_num):
            return None

        async def on_round_complete(ci_, i, t, raw, clean, dtext, conv_resp):
            order.append("round_complete")
            events.append("round_complete")

        await ci.run_agentic_loop(
            "hi",
            interactive=interactive,
            on_round_start=on_round_start,
            on_round_complete=on_round_complete,
        )

        # round_complete must appear before the widget's aget_input.
        assert "round_complete" in events
        assert "aget_input" in events
        assert events.index("round_complete") < events.index("aget_input")
        # And before asend_response too (preamble committed before pending_input).
        assert events.index("round_complete") < events.index("asend_response")


# ---------------------------------------------------------------------------
# (iv) GroupValidationError → controlled continuation (no raise out)
# ---------------------------------------------------------------------------


class TestGroupValidationControlledContinuation:
    @pytest.mark.asyncio
    async def test_duplicate_grouped_output_var_does_not_abort_turn(self):
        # Round 0: two grouped conversation tools sharing the SAME output var →
        # group_and_validate raises GroupValidationError internally. The loop
        # must NOT propagate it; it injects a corrective message and continues to
        # round 1, where the model emits a plain final answer.
        bad_grouped = (
            "```json ToolsToInvoke\n"
            '{"type": "conversation", "name": "clarification", '
            '"arguments": {"prompt": "q1"}, "output": ["dup"], "parallel_group": 1}\n'
            '{"type": "conversation", "name": "clarification", '
            '"arguments": {"prompt": "q2"}, "output": ["dup"], "parallel_group": 1}\n'
            "```"
        )
        base = _ScriptedBase()
        base._setup_script([bad_grouped, _FINAL_TEXT_ROUND])
        # An interactive is present so the conv-tool branch is reachable, but the
        # validation fails BEFORE any widget dispatch, so aget_input is never hit.
        events: list[str] = []
        interactive = _FakeInteractive(response="never used", events=events)

        ci = ConversationalInferencer(
            base_inferencer=base,
            interactive=interactive,
            max_iterations=5,
        )
        on_start, on_complete, starts, completes = _capturing_hooks()

        result = await ci.run_agentic_loop(
            "go",
            interactive=interactive,
            on_round_start=on_start,
            on_round_complete=on_complete,
        )

        # Did NOT raise out, and a follow-up round (round 1) ran after the
        # controlled continuation.
        assert [s[0] for s in starts] == [0, 1]
        assert result.iterations_used == 2
        # The widget was never presented (validation failed first).
        assert "aget_input" not in events
        # Round 0 saw the (invalid) conversation tools; round 1 was final text.
        assert completes[0]["has_conv_tool"] is True
        assert completes[1]["dtext"] == _FINAL_TEXT_ROUND

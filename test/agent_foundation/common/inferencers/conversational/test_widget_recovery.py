"""Unit tests for pending-widget RECOVERY (Layer 2, Piece 3): the deterministic
re-arm that applies a persisted widget answer to the EXACT persisted widget with
NO LLM re-inference, then runs the SAME continuation the live path runs.

``unittest.TestCase`` style so the fbcode ``python_unittest`` runner collects them
(mirrors ``test_phase_progression_guards.py``). Three layers of coverage:

  * ``WidgetRecoveryLoopTest`` — drives a REAL ``ConversationalInferencer``
    through ``run_agentic_loop`` with ``_pending_widget_result`` set (exactly what
    ``ConversationService.resume_conversation_from_widget`` seeds on recovery). A
    scripted fake base inferencer (no network) provides ONLY the post-widget
    continuation round, so a base call count of 1 PROVES the widget round was not
    re-inferred — the correctness property the plan demands (a re-emitted widget
    could differ in output_vars and bind the answer to the wrong variable).
  * ``CollectWidgetResponseTest`` — unbound-method tests of the recovery-specific,
    emit-free decode ``_collect_widget_response`` (single delegates to
    ``_apply_widget_answer`` + wrap + gate; multiple delegates to
    ``_decode_compound_response``): the parity guarantee vs the live decode.
  * ``ConversationToolRoundTripTest`` — the faithful ``to_dict``/``from_dict``
    round-trip (across the JSON boundary the sidecar crosses) that Piece 1
    persistence relies on to reconstruct the exact widget on recovery.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (  # noqa: E501
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (  # noqa: E501
    ConversationalInferencer as CI,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrs

_FINAL_TEXT = "All finished. Here is the answer."


@attrs(slots=False)
class _ScriptedBase(InferencerBase):
    """Returns pre-scripted raw responses, one per (a)infer call, and counts the
    calls on ``_idx`` so a test can prove how many LLM rounds actually ran (once
    exhausted it repeats the last entry so a runaway loop still hits the cap)."""

    def _setup(self, script):
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


class WidgetRecoveryLoopTest(unittest.TestCase):
    """End-to-end recovery through the real ``run_agentic_loop`` — the branch that
    consumes ``_pending_widget_result`` fires BEFORE any render/infer, so no
    interactive and no re-inference of the widget round are involved."""

    def _ci(self, script):
        base = _ScriptedBase()
        base._setup(script)
        return CI(base_inferencer=base, max_iterations=5), base

    def test_recovery_applies_answer_without_reinference(self):
        # Script has ONLY the continuation round: if the widget round were
        # (wrongly) re-inferred, the base would be called twice.
        ci, base = self._ci([_FINAL_TEXT])
        tool = ConversationTool(
            tool_type=ConversationToolType.PROPOSAL_SELECTION,
            prompt="Pick proposals",
            output_vars=["selected_proposals_ids"],
        )
        ci._pending_widget_result = {
            "tools": [tool],
            "action_tools": [],
            "raw_value": {"selected_proposals": ["H1", "H3"]},
        }

        result = asyncio.run(ci.run_agentic_loop(""))

        # (1) DETERMINISM: the widget round was NOT re-inferred — the base ran
        # EXACTLY once, for the post-widget (M+1) continuation only.
        self.assertEqual(base._idx, 1)
        # (2) The answer bound to the EXACT persisted widget's output var and was
        # published to prior_context (bare + tool-namespaced alias).
        self.assertEqual(ci.prior_context.get("selected_proposals_ids"), "H1,H3")
        self.assertEqual(
            ci.prior_context.get("proposal_selection__selected_proposals_ids"),
            "H1,H3",
        )
        # (3) The shared tail ran: the decoded answer is in the synthesized
        # widget-response user message, and the loop reached a terminal answer.
        self.assertTrue(any("H1,H3" in (m.get("content") or "") for m in ci._messages))
        self.assertIn("finished", result.text.lower())
        # (4) The one-shot recovery flag was consumed (no re-fire on iteration 2).
        self.assertIsNone(getattr(ci, "_pending_widget_result", None))

    def test_none_answer_leaves_widget_pending(self):
        # A missing/unusable persisted answer must NOT re-infer and must NOT
        # continue — the turn ends with the widget still pending (mirrors the
        # live ``collected is None`` handback).
        ci, base = self._ci([_FINAL_TEXT])
        tool = ConversationTool(
            tool_type=ConversationToolType.CLARIFICATION,
            prompt="Your name?",
            output_vars=["user_name"],
        )
        ci._pending_widget_result = {
            "tools": [tool],
            "action_tools": [],
            "raw_value": None,
        }

        result = asyncio.run(ci.run_agentic_loop(""))

        self.assertEqual(base._idx, 0)  # no inference at all
        self.assertTrue(result.has_conversation_tool)
        self.assertIsNone(getattr(ci, "_pending_widget_result", None))


class CollectWidgetResponseTest(unittest.IsolatedAsyncioTestCase):
    """`_collect_widget_response` — the emit-free recovery decode. Post-Phase F:
    it's an async method (delegates to async `_apply_widget_answer`). Fake CI
    uses async lambdas via ``asyncio``'s ``coroutine`` protocol.
    """

    def _fake_single(self, applied):
        seen = {}
        f = SimpleNamespace(_last_conv_nested_bindings={})

        async def _apply(tool, ui):
            return applied

        f._apply_widget_answer = _apply
        f._open_user_input_gate_if_satisfied = lambda tools, collected: seen.update(
            gate=collected
        )
        f._seen = seen
        return f

    async def test_single_wraps_and_opens_gate(self):
        f = self._fake_single("H1,H3")
        tool = SimpleNamespace(output_vars=["selected_proposals_ids"])
        out = await CI._collect_widget_response(
            f, [tool], [], {"selected_proposals": ["H1", "H3"]}
        )
        self.assertEqual(out, {"selected_proposals_ids": "H1,H3"})
        # The gate saw the SAME collected dict the live path would open it with.
        self.assertEqual(f._seen.get("gate"), {"selected_proposals_ids": "H1,H3"})

    async def test_single_default_var_when_no_output_vars(self):
        f = self._fake_single("hello")
        tool = SimpleNamespace(output_vars=[])
        out = await CI._collect_widget_response(f, [tool], [], {"user_input": "hello"})
        self.assertEqual(out, {"input": "hello"})

    async def test_single_none_returns_none(self):
        f = SimpleNamespace(_last_conv_nested_bindings={})

        async def _apply(tool, ui):
            return None

        f._apply_widget_answer = _apply
        f._open_user_input_gate_if_satisfied = lambda *a: None
        tool = SimpleNamespace(output_vars=["v"])
        self.assertIsNone(await CI._collect_widget_response(f, [tool], [], None))

    async def test_multiple_delegates_to_compound(self):
        calls = {}
        f = SimpleNamespace()
        f._decode_compound_response = lambda tools, ui: (
            calls.update(n=len(tools), ui=ui) or {"a": "1", "b": "2"}
        )
        tools = [SimpleNamespace(output_vars=["a"]), SimpleNamespace(output_vars=["b"])]
        out = await CI._collect_widget_response(
            f, tools, [], {"values": {"a": "1", "b": "2"}}
        )
        self.assertEqual(out, {"a": "1", "b": "2"})
        self.assertEqual(calls.get("n"), 2)  # compound decode saw both tabs

    async def test_empty_tools_returns_none(self):
        self.assertIsNone(
            await CI._collect_widget_response(SimpleNamespace(), [], [], {"x": 1})
        )


class ConversationToolRoundTripTest(unittest.TestCase):
    """Piece 1 persistence stores each ``ConversationTool`` as a dict in a JSON
    sidecar; RECOVERY revives it with ``from_dict``. The round-trip (across the
    JSON boundary) must preserve every field the decode reads, so the recovered
    widget is byte-for-byte the one the user answered."""

    def _round_trip(self, tool):
        # Cross the exact JSON boundary the sidecar crosses.
        return ConversationTool.from_dict(json.loads(json.dumps(tool.to_dict())))

    def test_proposal_selection_round_trip(self):
        tool = ConversationTool(
            tool_type=ConversationToolType.PROPOSAL_SELECTION,
            prompt="Pick proposals",
            output_vars=["selected_proposals_ids"],
            metadata={"open_dashboard": "experiment_hub"},
        )
        revived = self._round_trip(tool)
        # str-Enum: equality holds whether revived carries the enum or the plain
        # "proposal_selection" the JSON boundary yields.
        self.assertEqual(revived.tool_type, ConversationToolType.PROPOSAL_SELECTION)
        self.assertEqual(revived.output_vars, ["selected_proposals_ids"])
        self.assertEqual(revived.metadata.get("open_dashboard"), "experiment_hub")

    def test_typed_freetext_round_trip_preserves_serialization(self):
        tool = ConversationTool(
            tool_type=ConversationToolType.CLARIFICATION,
            prompt="Which paths?",
            output_vars=["paths"],
            expected_input_type="path",
            prefix="/repo",
            allow_multiple_input=True,
            serialization="json",
        )
        revived = self._round_trip(tool)
        self.assertEqual(revived.expected_input_type, "path")
        self.assertEqual(revived.prefix, "/repo")
        self.assertTrue(revived.allow_multiple_input)
        self.assertEqual(revived.serialization, "json")
        self.assertEqual(revived.output_vars, ["paths"])


if __name__ == "__main__":
    unittest.main()

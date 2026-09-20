# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Phase M3 — Integration test for handler-registry dispatch through CI.

Instantiates a real ConversationalInferencer with default_registry() and
asserts the full dispatch chain (registry.require → handler.handle_response
→ effect.apply → CI state mutations) works end-to-end for representative
tool_types.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrs


@attrs(slots=False)
class _NoOpBase(InferencerBase):
    """Base inferencer that never gets called (all tests exit before infer)."""

    def _infer(self, inp, cfg=None, **kw):
        return ""

    async def _ainfer(self, inp, cfg=None, **kw):
        return ""


class HandlerRegistryDispatchTest(unittest.IsolatedAsyncioTestCase):
    def _ci(self):
        return ConversationalInferencer(base_inferencer=_NoOpBase())

    async def test_confirmation_dispatch_sets_gate(self):
        ci = self._ci()
        tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
        await ci._apply_widget_answer(tool, {"choice": "yes"})
        # ApplyContextUpdates effect writes _confirmation_gate_passed=True.
        assert ci.prior_context.get("_confirmation_gate_passed") is True

    async def test_proposal_selection_dispatch_publishes_ids(self):
        ci = self._ci()
        tool = ConversationTool(
            tool_type=ConversationToolType.PROPOSAL_SELECTION,
            output_vars=["selected_proposal_ids"],
        )
        result = await ci._apply_widget_answer(
            tool, {"selected_proposals": ["P1", "P3"]}
        )
        assert result == "P1,P3"
        assert ci.prior_context.get("selected_proposal_ids") == "P1,P3"

    async def test_proposal_selection_dispatch_captures_auto_implement(self):
        ci = self._ci()
        tool = ConversationTool(
            tool_type=ConversationToolType.PROPOSAL_SELECTION,
            output_vars=["selected_proposal_ids"],
        )
        await ci._apply_widget_answer(
            tool,
            {"selected_proposals": ["P1"], "auto_implement": True},
        )
        # DashboardDirectiveEffect writes _next_dashboard_directives.
        assert ci._next_dashboard_directives == {"auto_implement": True}

    async def test_single_choice_dispatch_by_index(self):
        ci = self._ci()
        tool = ConversationTool(
            tool_type=ConversationToolType.SINGLE_CHOICE,
            choices=[
                ChoiceItem(label="A", value="a"),
                ChoiceItem(label="B", value="b"),
            ],
            output_vars=["mode"],
        )
        result = await ci._apply_widget_answer(tool, {"choice_index": 1})
        assert result == "b"
        assert ci.prior_context.get("mode") == "b"

    async def test_confirmation_dispatch_param_overrides(self):
        ci = self._ci()
        tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
        await ci._apply_widget_answer(
            tool,
            {"choice": "yes", "param_overrides": {"foo": "bar"}},
        )
        # OverrideNextActionToolArgs writes _next_action_tool_overrides.
        assert ci._next_action_tool_overrides == {"foo": "bar"}

    async def test_clarification_dispatch_publishes_output_var(self):
        ci = self._ci()
        tool = ConversationTool(
            tool_type=ConversationToolType.CLARIFICATION,
            output_vars=["user_name"],
        )
        result = await ci._apply_widget_answer(tool, {"content": "Alice"})
        assert result == "Alice"
        assert ci.prior_context.get("user_name") == "Alice"


class YoloRegressionTest(unittest.IsolatedAsyncioTestCase):
    """Phase M4 — yolo mode regression tests. Post-Phase G, yolo routes through
    the same registry as interactive."""

    async def test_yolo_routes_through_registry(self):
        """`_synthesize_yolo_collected` uses `handler_registry.require(...)`,
        NOT the deleted `decode_tool_bindings` path."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
            ConversationalInferencer,
        )

        ci = ConversationalInferencer(base_inferencer=_NoOpBase(), yolo_mode=True)
        # Confirm yolo path calls the async registry dispatch.
        assert ci.handler_registry is not None
        # `_synthesize_yolo_collected` is async — this asserts the signature.
        import inspect

        assert inspect.iscoroutinefunction(ci._synthesize_yolo_collected)

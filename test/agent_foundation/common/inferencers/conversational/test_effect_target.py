"""EffectTarget / WidgetMailboxes: widget effects act on an orchestrator's
``mailboxes`` and ``update_prior_context``; CI's ``_next_*`` names forward to
its mailboxes; an object of the earlier effect-target shape still works."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    ApplyContextUpdates,
    DashboardDirectiveEffect,
    OverrideNextActionToolArgs,
    PublishSessionVariablesEffect,
    SetPromptVariable,
    SetTurnVariables,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    effect_target,
    EffectTarget,
    WidgetMailboxes,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrs
from later.unittest import TestCase


@attrs(slots=False)
class _NoOpBase(InferencerBase):
    def _infer(self, inp, cfg=None, **kw):
        return ""

    async def _ainfer(self, inp, cfg=None, **kw):
        return ""


class _Target:
    """A minimal EffectTarget that records how it is written."""

    def __init__(self) -> None:
        self.prior_context: dict[str, Any] = {}
        self.prompt_renderer = SimpleNamespace(
            set_variable=lambda n, v: self.variables.append((n, v))
        )
        self.mailboxes = WidgetMailboxes()
        self.updates: list[dict[str, Any]] = []
        self.published: list[tuple[dict[str, Any], Optional[str]]] = []
        self.variables: list[tuple[str, Any]] = []

    def update_prior_context(self, **updates: Any) -> None:
        self.updates.append(updates)
        self.prior_context.update(updates)

    def set_session_variables(
        self, variables: dict[str, Any], *, tool_type: Optional[str] = None
    ) -> None:
        self.published.append((variables, tool_type))


class EffectTargetTest(TestCase):
    async def test_effects_act_on_the_target_surface(self) -> None:
        target = _Target()
        self.assertIsInstance(target, EffectTarget)

        await ApplyContextUpdates({"_confirmation_gate_passed": True}).apply(target)
        await OverrideNextActionToolArgs({"depth": "deep"}).apply(target)
        await SetTurnVariables({"topic": "lidar"}).apply(target)
        await DashboardDirectiveEffect({"auto_implement": True}).apply(target)
        await PublishSessionVariablesEffect(
            {"x": "1"}, tool_type="clarification"
        ).apply(target)
        await SetPromptVariable("mode", "fast").apply(target)

        self.assertEqual(target.updates, [{"_confirmation_gate_passed": True}])
        self.assertEqual(
            target.mailboxes,
            WidgetMailboxes(
                action_overrides={"depth": "deep"},
                turn_variables={"topic": "lidar"},
                dashboard_directives={"auto_implement": True},
            ),
        )
        self.assertEqual(target.published, [({"x": "1"}, "clarification")])
        self.assertEqual(target.variables, [("mode", "fast")])

        target.mailboxes.clear()
        self.assertEqual(target.mailboxes, WidgetMailboxes())

    async def test_earlier_shape_target_keeps_its_attributes(self) -> None:
        legacy = SimpleNamespace(
            prior_context={},
            _next_action_tool_overrides=None,
            _next_turn_variables=None,
            _next_dashboard_directives=None,
        )
        self.assertNotIsInstance(legacy, EffectTarget)

        await ApplyContextUpdates({"k": "v"}).apply(legacy)
        await OverrideNextActionToolArgs({"depth": "deep"}).apply(legacy)
        view = effect_target(legacy).mailboxes
        view.clear()

        self.assertEqual(legacy.prior_context, {"k": "v"})
        self.assertIsNone(legacy._next_action_tool_overrides)


class ConversationalInferencerMailboxesTest(TestCase):
    async def test_ci_is_an_effect_target_and_next_names_forward(self) -> None:
        ci = ConversationalInferencer(base_inferencer=_NoOpBase())
        self.assertIs(effect_target(ci), ci)

        tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
        await ci._apply_widget_answer(
            tool,
            {"choice": "yes", "param_overrides": {"foo": "bar"}, "variables": {"v": 2}},
        )

        self.assertEqual(ci.mailboxes.action_overrides, {"foo": "bar"})
        self.assertEqual(ci.mailboxes.turn_variables, {"v": 2})
        self.assertEqual(ci._next_action_tool_overrides, {"foo": "bar"})
        self.assertTrue(ci.prior_context["_confirmation_gate_passed"])

        ci._next_dashboard_directives = {"auto_implement": False}
        self.assertEqual(ci.mailboxes.dashboard_directives, {"auto_implement": False})

        ci.restore_state({"messages": [], "prior_context": {}})
        self.assertEqual(ci.mailboxes, WidgetMailboxes())

    async def test_context_updates_reach_update_prior_context(self) -> None:
        ci = ConversationalInferencer(base_inferencer=_NoOpBase())
        seen: list[dict[str, Any]] = []
        ci.update_prior_context = lambda **updates: seen.append(updates)

        await ApplyContextUpdates({"_confirmation_gate_passed": True}).apply(ci)

        self.assertEqual(seen, [{"_confirmation_gate_passed": True}])

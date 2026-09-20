# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Per-Effect unit tests.

Constructs a minimal mock inferencer (types.SimpleNamespace with the fields the
effect writes to), calls ``await effect.apply(mock)``, asserts the mutation.

Merge-conflict semantics: each effect that raises on double-set has an
explicit test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    ApplyContextUpdates,
    DashboardDirectiveEffect,
    OverrideNextActionToolArgs,
    PublishSessionVariablesEffect,
    SetPromptVariable,
    SetTurnVariables,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    HandlerResultMergeConflict,
)


def _mock_ci_with_next_mailboxes() -> SimpleNamespace:
    """Mock CI carrying the three _next_* mailboxes the effects target."""
    return SimpleNamespace(
        _next_action_tool_overrides=None,
        _next_turn_variables=None,
        _next_dashboard_directives=None,
        prior_context={},
    )


# ============================================================================
# OverrideNextActionToolArgs
# ============================================================================


@pytest.mark.asyncio
async def test_override_next_action_tool_args_sets_mailbox() -> None:
    ci = _mock_ci_with_next_mailboxes()
    await OverrideNextActionToolArgs({"foo": "bar"}).apply(ci)
    assert ci._next_action_tool_overrides == {"foo": "bar"}


@pytest.mark.asyncio
async def test_override_next_action_tool_args_raises_on_double_set() -> None:
    ci = _mock_ci_with_next_mailboxes()
    await OverrideNextActionToolArgs({"foo": "bar"}).apply(ci)
    with pytest.raises(HandlerResultMergeConflict):
        await OverrideNextActionToolArgs({"baz": "qux"}).apply(ci)


# ============================================================================
# SetTurnVariables
# ============================================================================


@pytest.mark.asyncio
async def test_set_turn_variables_sets_mailbox() -> None:
    ci = _mock_ci_with_next_mailboxes()
    await SetTurnVariables({"x": "1"}).apply(ci)
    assert ci._next_turn_variables == {"x": "1"}


@pytest.mark.asyncio
async def test_set_turn_variables_raises_on_double_set() -> None:
    ci = _mock_ci_with_next_mailboxes()
    await SetTurnVariables({"x": "1"}).apply(ci)
    with pytest.raises(HandlerResultMergeConflict):
        await SetTurnVariables({"y": "2"}).apply(ci)


# ============================================================================
# DashboardDirectiveEffect
# ============================================================================


@pytest.mark.asyncio
async def test_dashboard_directive_effect_sets_none_mailbox() -> None:
    ci = _mock_ci_with_next_mailboxes()
    await DashboardDirectiveEffect({"auto_implement": True}).apply(ci)
    assert ci._next_dashboard_directives == {"auto_implement": True}


@pytest.mark.asyncio
async def test_dashboard_directive_effect_merges_disjoint_keys() -> None:
    ci = _mock_ci_with_next_mailboxes()
    await DashboardDirectiveEffect({"auto_implement": True}).apply(ci)
    await DashboardDirectiveEffect({"other_key": "value"}).apply(ci)
    assert ci._next_dashboard_directives == {
        "auto_implement": True,
        "other_key": "value",
    }


@pytest.mark.asyncio
async def test_dashboard_directive_effect_idempotent_same_key_same_value() -> None:
    ci = _mock_ci_with_next_mailboxes()
    await DashboardDirectiveEffect({"auto_implement": True}).apply(ci)
    # Same key with same value should NOT raise (idempotent).
    await DashboardDirectiveEffect({"auto_implement": True}).apply(ci)
    assert ci._next_dashboard_directives == {"auto_implement": True}


@pytest.mark.asyncio
async def test_dashboard_directive_effect_raises_on_same_key_different_value() -> None:
    ci = _mock_ci_with_next_mailboxes()
    await DashboardDirectiveEffect({"auto_implement": True}).apply(ci)
    with pytest.raises(HandlerResultMergeConflict):
        await DashboardDirectiveEffect({"auto_implement": False}).apply(ci)


# ============================================================================
# PublishSessionVariablesEffect
# ============================================================================


@pytest.mark.asyncio
async def test_publish_session_variables_forwards_to_ci_method() -> None:
    calls: list[tuple[dict[str, object], object]] = []

    def _set_session_variables(
        variables: dict[str, object], *, tool_type: object = None
    ) -> None:
        calls.append((variables, tool_type))

    ci = SimpleNamespace(set_session_variables=_set_session_variables)
    await PublishSessionVariablesEffect({"x": "1"}, tool_type="clarification").apply(ci)
    assert calls == [({"x": "1"}, "clarification")]


@pytest.mark.asyncio
async def test_publish_session_variables_no_tool_type_default() -> None:
    calls: list[tuple[dict[str, object], object]] = []

    def _set_session_variables(
        variables: dict[str, object], *, tool_type: object = None
    ) -> None:
        calls.append((variables, tool_type))

    ci = SimpleNamespace(set_session_variables=_set_session_variables)
    await PublishSessionVariablesEffect({"y": "2"}).apply(ci)
    assert calls == [({"y": "2"}, None)]


# ============================================================================
# ApplyContextUpdates + SetPromptVariable — smoke coverage for existing effects
# ============================================================================


@pytest.mark.asyncio
async def test_apply_context_updates_writes_to_prior_context() -> None:
    ci = SimpleNamespace(prior_context={})
    await ApplyContextUpdates({"key": "value"}).apply(ci)
    assert ci.prior_context == {"key": "value"}


@pytest.mark.asyncio
async def test_set_prompt_variable_forwards_to_renderer() -> None:
    calls: list[tuple[str, object]] = []

    class _Renderer:
        def set_variable(self, name: str, value: object) -> None:
            calls.append((name, value))

    ci = SimpleNamespace(prompt_renderer=_Renderer())
    await SetPromptVariable("foo", "bar").apply(ci)
    assert calls == [("foo", "bar")]

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Per-handler unit tests.

Each handler test covers:
- ``build_input_mode`` returns an ``InputModeConfig`` with the expected shape.
- ``handle_response`` emits the expected typed effects + text for a
  representative response payload.
- Handler-specific edge cases (composite input, variable_override,
  auto_implement, param_overrides, etc.).

Uses pytest bare-function style + a minimal ``HandlerContext`` fixture. No
BUCK target required (matches sibling ``test_effects.py`` pattern).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
    InputFieldSpec,
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
    HandlerContext,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers import (
    default_registry,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.clarification import (
    ClarificationHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.confirmation import (
    ConfirmationHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.multiple_choice import (
    MultipleChoiceHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.proposal_selection import (
    ProposalSelectionHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.single_choice import (
    SingleChoiceHandler,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers.tool_argument_form import (
    ToolArgumentFormHandler,
)


def _ctx(prior_context: dict[str, Any] | None = None) -> HandlerContext:
    """Minimal HandlerContext for tests."""
    return HandlerContext(
        prior_context=MappingProxyType(prior_context or {}),
        prompt_renderer=None,
        tool_executor=None,
        interactive=None,
        action_tools=None,
        tool_registry=None,
        resolve_tool_name=None,
    )


# ============================================================================
# Registry contract
# ============================================================================


def test_default_registry_registers_all_six_handlers() -> None:
    reg = default_registry()
    assert ConversationToolType.CLARIFICATION in reg
    assert ConversationToolType.SINGLE_CHOICE in reg
    assert ConversationToolType.MULTIPLE_CHOICE in reg
    assert ConversationToolType.CONFIRMATION in reg
    assert ConversationToolType.PROPOSAL_SELECTION in reg
    assert ConversationToolType.TOOL_ARGUMENT_FORM in reg


def test_registry_require_raises_on_missing() -> None:
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_registry import (
        ConversationToolHandlerRegistry,
    )

    reg = ConversationToolHandlerRegistry()
    with pytest.raises(ValueError, match="No handler registered"):
        reg.require(ConversationToolType.CLARIFICATION)


# ============================================================================
# ClarificationHandler
# ============================================================================


def test_clarification_build_input_mode_free_text() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.CLARIFICATION, prompt="What's your name?"
    )
    cfg = ClarificationHandler().build_input_mode(tool, _ctx())
    assert cfg.prompt == "What's your name?"


@pytest.mark.asyncio
async def test_clarification_handle_response_publishes_output_var() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.CLARIFICATION,
        output_vars=["strategy"],
    )
    result = await ClarificationHandler().handle_response(
        tool, {"content": "hybrid"}, _ctx()
    )
    assert result.text == "hybrid"
    assert len(result.effects) == 1
    assert isinstance(result.effects[0], PublishSessionVariablesEffect)
    assert result.effects[0].variables == {"strategy": "hybrid"}


@pytest.mark.asyncio
async def test_clarification_handle_response_no_output_var_no_effects() -> None:
    tool = ConversationTool(tool_type=ConversationToolType.CLARIFICATION)
    result = await ClarificationHandler().handle_response(
        tool, {"content": "foo"}, _ctx()
    )
    assert result.text == "foo"
    assert result.effects == []


# ============================================================================
# SingleChoiceHandler
# ============================================================================


@pytest.mark.asyncio
async def test_single_choice_handle_response_by_choice_index() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.SINGLE_CHOICE,
        choices=[
            ChoiceItem(label="Yes", value="yes"),
            ChoiceItem(label="No", value="no"),
        ],
        output_vars=["confirmed"],
    )
    result = await SingleChoiceHandler().handle_response(
        tool, {"choice_index": 0}, _ctx()
    )
    assert result.text == "yes"
    assert any(
        isinstance(e, PublishSessionVariablesEffect)
        and e.variables == {"confirmed": "yes"}
        for e in result.effects
    )


@pytest.mark.asyncio
async def test_single_choice_variable_override_wins() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.SINGLE_CHOICE,
        choices=[ChoiceItem(label="One", value="one")],
        output_vars=["default_var"],
    )
    result = await SingleChoiceHandler().handle_response(
        tool,
        {"choice_index": 0, "variable_override": {"custom": "value"}},
        _ctx(),
    )
    # variable_override wins — default_var NOT published.
    assert len(result.effects) == 1
    effect = result.effects[0]
    assert isinstance(effect, PublishSessionVariablesEffect)
    assert effect.variables == {"custom": "value"}


@pytest.mark.asyncio
async def test_single_choice_composite_input_returns_bindings() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.SINGLE_CHOICE,
        choices=[
            ChoiceItem(
                label="Path-based",
                value="path",
                input=InputFieldSpec(name="target_path", expected_input_type="path"),
            ),
        ],
        output_vars=["mode"],
    )
    result = await SingleChoiceHandler().handle_response(
        tool,
        {"choice_index": 0, "inputs": {"target_path": "/tmp/foo"}},
        _ctx(),
    )
    assert result.text == "path"
    # Composite binding is a decode RESULT (§A1.a), not an effect.
    assert "target_path" in result.bindings
    # Also emits a PublishSessionVariablesEffect for the nested value.
    nested_effects = [
        e
        for e in result.effects
        if isinstance(e, PublishSessionVariablesEffect) and "target_path" in e.variables
    ]
    assert len(nested_effects) == 1


# ============================================================================
# ConfirmationHandler
# ============================================================================


@pytest.mark.asyncio
async def test_confirmation_choice_yes_sets_gate() -> None:
    tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
    result = await ConfirmationHandler().handle_response(
        tool, {"choice": "yes"}, _ctx()
    )
    assert result.text == "yes"
    assert any(
        isinstance(e, ApplyContextUpdates)
        and e.updates == {"_confirmation_gate_passed": True}
        for e in result.effects
    )


@pytest.mark.asyncio
async def test_confirmation_choice_no_no_gate() -> None:
    tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
    result = await ConfirmationHandler().handle_response(tool, {"choice": "no"}, _ctx())
    assert result.text == "no"
    assert not any(isinstance(e, ApplyContextUpdates) for e in result.effects)


@pytest.mark.asyncio
async def test_confirmation_param_overrides_emits_effect() -> None:
    tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
    result = await ConfirmationHandler().handle_response(
        tool,
        {"choice": "yes", "param_overrides": {"foo": 1}},
        _ctx(),
    )
    assert any(
        isinstance(e, OverrideNextActionToolArgs) and e.overrides == {"foo": 1}
        for e in result.effects
    )


@pytest.mark.asyncio
async def test_confirmation_variables_emits_effect() -> None:
    tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
    result = await ConfirmationHandler().handle_response(
        tool,
        {"choice": "yes", "variables": {"x": "y"}},
        _ctx(),
    )
    assert any(
        isinstance(e, SetTurnVariables) and e.variables == {"x": "y"}
        for e in result.effects
    )


# ============================================================================
# ProposalSelectionHandler
# ============================================================================


@pytest.mark.asyncio
async def test_proposal_selection_joins_ids_publishes_output_var() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["selected_proposal_ids"],
    )
    result = await ProposalSelectionHandler().handle_response(
        tool, {"selected_proposals": ["P1", "P3", "P7"]}, _ctx()
    )
    assert result.text == "P1,P3,P7"
    publish = [
        e for e in result.effects if isinstance(e, PublishSessionVariablesEffect)
    ]
    assert len(publish) == 1
    assert publish[0].variables == {"selected_proposal_ids": "P1,P3,P7"}


@pytest.mark.asyncio
async def test_proposal_selection_auto_implement_true_emits_directive() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["ids"],
    )
    result = await ProposalSelectionHandler().handle_response(
        tool,
        {"selected_proposals": ["P1"], "auto_implement": True},
        _ctx(),
    )
    directives = [e for e in result.effects if isinstance(e, DashboardDirectiveEffect)]
    assert len(directives) == 1
    assert directives[0].directives == {"auto_implement": True}


@pytest.mark.asyncio
async def test_proposal_selection_auto_implement_false_still_captured() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["ids"],
    )
    result = await ProposalSelectionHandler().handle_response(
        tool,
        {"selected_proposals": ["P1"], "auto_implement": False},
        _ctx(),
    )
    directives = [e for e in result.effects if isinstance(e, DashboardDirectiveEffect)]
    assert len(directives) == 1
    assert directives[0].directives == {"auto_implement": False}


@pytest.mark.asyncio
async def test_proposal_selection_no_auto_implement_no_directive() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["ids"],
    )
    result = await ProposalSelectionHandler().handle_response(
        tool, {"selected_proposals": ["P1"]}, _ctx()
    )
    assert not any(isinstance(e, DashboardDirectiveEffect) for e in result.effects)


@pytest.mark.asyncio
async def test_proposal_selection_fallback_selected_key() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["ids"],
    )
    result = await ProposalSelectionHandler().handle_response(
        tool, {"selected": ["A", "B"]}, _ctx()
    )
    assert result.text == "A,B"


@pytest.mark.asyncio
async def test_proposal_selection_fallback_choice_indices() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        choices=[
            ChoiceItem(label="P1", value="P1"),
            ChoiceItem(label="P2", value="P2"),
        ],
        output_vars=["ids"],
    )
    result = await ProposalSelectionHandler().handle_response(
        tool, {"choice_indices": [0, 1]}, _ctx()
    )
    assert result.text == "P1,P2"


# ============================================================================
# MultipleChoiceHandler + ToolArgumentFormHandler — smoke coverage
# ============================================================================


def test_multiple_choice_build_input_mode() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.MULTIPLE_CHOICE,
        choices=[
            ChoiceItem(label="A", value="a"),
            ChoiceItem(label="B", value="b"),
        ],
    )
    cfg = MultipleChoiceHandler().build_input_mode(tool, _ctx())
    assert cfg is not None


def test_tool_argument_form_build_input_mode() -> None:
    tool = ConversationTool(
        tool_type=ConversationToolType.TOOL_ARGUMENT_FORM,
        tool_name="my_tool",
        fields=[{"name": "x", "label": "X"}],
    )
    cfg = ToolArgumentFormHandler().build_input_mode(tool, _ctx())
    assert cfg is not None

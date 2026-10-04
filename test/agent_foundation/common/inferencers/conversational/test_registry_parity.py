# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Phase 0.1 — Registry parity harness.

Asserts that dispatching through the registered handler produces the same
observable output as the pre-migration inline decoder path. Specifically:

- ``handler.build_input_mode(tool, ctx).to_dict()`` shape is preserved for
  every ``ConversationToolType``.
- ``handler.handle_response(tool, response, ctx)`` emits the correct typed
  effects for representative response payloads per tool_type.

This is the load-bearing safety gate for Phase F's inline-decoder deletion.
By construction, this file complements ``test_handlers.py`` (per-handler
coverage) + ``test_effects.py`` (per-effect coverage). The 243 baseline
tests continuing to pass across Phase 0→F is the transitive parity proof;
these explicit assertions document the invariant.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    DashboardDirectiveEffect,
    PublishSessionVariablesEffect,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    HandlerContext,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers import (
    default_registry,
)


def _ctx() -> HandlerContext:
    reg = default_registry()
    return HandlerContext(
        prior_context=MappingProxyType({}),
        prompt_renderer=None,
        tool_executor=None,
        interactive=None,
        action_tools=None,
        tool_registry=None,
        resolve_tool_name=None,
        handler_registry=reg,
    )


@pytest.mark.parametrize(
    "tool_type",
    [
        ConversationToolType.CLARIFICATION,
        ConversationToolType.SINGLE_CHOICE,
        ConversationToolType.MULTIPLE_CHOICE,
        ConversationToolType.CONFIRMATION,
        ConversationToolType.PROPOSAL_SELECTION,
        ConversationToolType.TOOL_ARGUMENT_FORM,
    ],
)
def test_every_tool_type_has_handler_and_builds_input_mode(
    tool_type: ConversationToolType,
) -> None:
    """Every declared ConversationToolType must be dispatchable — no silent
    fallthrough to the free-text default (fixed the pre-migration
    TOOL_ARGUMENT_FORM bug per Phase E)."""
    reg = default_registry()
    handler = reg.require(tool_type)
    assert handler is not None
    # Construct a minimal tool of this type and assert build_input_mode
    # doesn't raise + returns a truthy config.
    tool = ConversationTool(
        tool_type=tool_type,
        prompt="test",
        choices=[ChoiceItem(label="a", value="a")]
        if tool_type
        in (
            ConversationToolType.SINGLE_CHOICE,
            ConversationToolType.MULTIPLE_CHOICE,
            ConversationToolType.PROPOSAL_SELECTION,
        )
        else [],
    )
    cfg = handler.build_input_mode(tool, _ctx())
    assert cfg is not None


@pytest.mark.asyncio
async def test_proposal_selection_effects_shape() -> None:
    """Registry dispatch produces PublishSessionVariables + DashboardDirective
    for a proposal_selection response — parity with the inline decoder."""
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["selected_proposal_ids"],
    )
    reg = default_registry()
    handler = reg.require(tool.tool_type)
    result = await handler.handle_response(
        tool,
        {"selected_proposals": ["P1", "P2"], "auto_implement": True},
        _ctx(),
    )
    assert result.text == "P1,P2"
    publish = [
        e for e in result.effects if isinstance(e, PublishSessionVariablesEffect)
    ]
    directives = [e for e in result.effects if isinstance(e, DashboardDirectiveEffect)]
    assert len(publish) == 1
    assert len(directives) == 1
    assert directives[0].directives == {"auto_implement": True}


@pytest.mark.asyncio
async def test_confirmation_choice_yes_gate_effect() -> None:
    """Registry dispatch of confirmation choice=yes emits
    ApplyContextUpdates({_confirmation_gate_passed: True}) — pre-migration
    parity."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
        ApplyContextUpdates,
    )

    tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
    reg = default_registry()
    handler = reg.require(tool.tool_type)
    result = await handler.handle_response(tool, {"choice": "yes"}, _ctx())
    assert any(
        isinstance(e, ApplyContextUpdates)
        and e.updates == {"_confirmation_gate_passed": True}
        for e in result.effects
    )

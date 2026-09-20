# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Phase 0.3 — Yolo-path parity harness.

Post-Phase G: yolo synthetic responses route through the SAME handler
registry as interactive dispatch. This test locks the invariant that
handlers cannot observe whether the response came from a user or a
`_synthesize_yolo_response` — the response dict is the contract.

The latent auto_implement bug (yolo select_all previously skipped
DashboardDirectiveEffect) is fixed by construction — the ProposalSelection
handler captures auto_implement in ALL dispatch paths.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects import (
    DashboardDirectiveEffect,
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


@pytest.mark.asyncio
async def test_yolo_synthetic_response_matches_interactive() -> None:
    """A synthetic yolo response and a user-typed response producing the same
    dict content produce identical handler output — the response IS the
    contract (Design Principle #13)."""
    tool = ConversationTool(
        tool_type=ConversationToolType.CLARIFICATION,
        output_vars=["strategy"],
    )
    reg = default_registry()
    handler = reg.require(tool.tool_type)

    user_response = {"content": "holistic"}
    yolo_response = {"content": "holistic"}  # synthesized from tool.json default

    r_user = await handler.handle_response(tool, user_response, _ctx())
    r_yolo = await handler.handle_response(tool, yolo_response, _ctx())

    assert r_user.text == r_yolo.text == "holistic"
    assert len(r_user.effects) == len(r_yolo.effects)


@pytest.mark.asyncio
async def test_yolo_select_all_captures_auto_implement() -> None:
    """LATENT BUG FIX: yolo `select_all` on a proposal_selection tool now
    correctly captures `auto_implement` via DashboardDirectiveEffect.
    Pre-migration, yolo skipped this and the dashboard directive was lost.
    """
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        choices=[
            ChoiceItem(label="P1", value="P1"),
            ChoiceItem(label="P2", value="P2"),
        ],
        output_vars=["selected_proposal_ids"],
    )
    reg = default_registry()
    handler = reg.require(tool.tool_type)
    # Yolo-synthesized select-all response with auto_implement flag.
    yolo_response = {
        "choice_indices": [0, 1],
        "auto_implement": True,
    }
    result = await handler.handle_response(tool, yolo_response, _ctx())
    directives = [e for e in result.effects if isinstance(e, DashboardDirectiveEffect)]
    assert len(directives) == 1
    assert directives[0].directives == {"auto_implement": True}

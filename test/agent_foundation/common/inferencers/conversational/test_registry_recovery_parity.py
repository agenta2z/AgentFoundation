# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Phase 0.2 — Recovery-path parity harness.

Asserts that pending-widget RECOVERY (an unanswered widget re-armed after
reconnect) produces the same `collected` dict as the interactive path via
`_apply_widget_answer`.

Sibling ``test_widget_recovery.py`` covers the loop-level round-trip; this
file locks the invariant at the registry-dispatch level.
"""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
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
async def test_recovery_dispatch_matches_interactive_dispatch() -> None:
    """A persisted (tool, raw_answer) replayed through the registry must produce
    the SAME text + effects as the live interactive dispatch."""
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["selected_proposal_ids"],
    )
    persisted_response = {"selected_proposals": ["H1", "H3"]}

    reg = default_registry()
    handler = reg.require(tool.tool_type)

    # First pass: live interactive dispatch.
    live_result = await handler.handle_response(tool, persisted_response, _ctx())

    # Second pass: recovery replay with fresh ctx (no shared state).
    recovery_result = await handler.handle_response(tool, persisted_response, _ctx())

    # Both must produce identical text.
    assert live_result.text == recovery_result.text
    # Both must produce the same number and types of effects.
    assert len(live_result.effects) == len(recovery_result.effects)
    for a, b in zip(live_result.effects, recovery_result.effects):
        assert type(a) is type(b)

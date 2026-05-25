"""Regression test for the interactive_checkpoint async bug.

Before Phase 0, interactive_checkpoint.py called await interactive.aget_input()
and await interactive.asend_response(), but InteractiveBase didn't define these
methods — causing AttributeError at runtime. This test verifies the fix.
"""
import asyncio
import pytest

from attr import attrs, attrib
from agent_foundation.ui.interactive_base import InteractionFlags
from agent_foundation.ui.rich_interactive_base import RichInteractiveBase
from agent_foundation.ui.interactive_checkpoint import (
    run_checkpoint,
    checkpoint_plan_review,
    checkpoint_results_review,
    CheckpointResult,
)
from agent_foundation.ui.input_modes import ChoiceOption


@attrs
class FakeInteractive(RichInteractiveBase):
    """Interactive stub that returns canned responses for checkpoint testing.

    Extends RichInteractiveBase (not InteractiveBase) because
    interactive_checkpoint passes input_mode= to asend_response, which
    only RichInteractiveBase.send_response() accepts.
    """
    responses: list = attrib(factory=list, kw_only=True)
    _response_idx: int = attrib(default=0, init=False)
    _sent: list = attrib(factory=list, init=False)

    def _get_input(self):
        if self._response_idx < len(self.responses):
            val = self.responses[self._response_idx]
            self._response_idx += 1
            return val
        return None

    def reset_input(self, flag):
        pass

    def _send_response(self, response, flag=InteractionFlags.TurnCompleted):
        self._sent.append(response)


@pytest.mark.asyncio
async def test_run_checkpoint_no_attribute_error():
    """run_checkpoint succeeds without AttributeError — the core regression test."""
    fake = FakeInteractive(responses=["approve"])
    result = await run_checkpoint(
        interactive=fake,
        prompt="Test checkpoint",
        options=[
            ChoiceOption(label="Approve", value="approve"),
            ChoiceOption(label="Reject", value="reject"),
        ],
    )
    assert isinstance(result, CheckpointResult)
    assert result.action == "approve"


@pytest.mark.asyncio
async def test_checkpoint_plan_review_approve():
    """checkpoint_plan_review works end-to-end."""
    fake = FakeInteractive(responses=["approve"])
    result = await checkpoint_plan_review(
        interactive=fake,
        plan_summary="Test plan",
    )
    assert result.action == "approve"


@pytest.mark.asyncio
async def test_checkpoint_results_review_none_interactive():
    """checkpoint_results_review with None interactive returns default."""
    result = await checkpoint_results_review(
        interactive=None,
        results_summary="Test results",
        default_action="approve",
    )
    assert result.action == "approve"


@pytest.mark.asyncio
async def test_run_checkpoint_custom_input():
    """run_checkpoint handles custom text input."""
    fake = FakeInteractive(responses=["custom answer"])
    result = await run_checkpoint(
        interactive=fake,
        prompt="Choose",
        options=[ChoiceOption(label="A", value="a")],
    )
    assert result.action == "custom"
    assert result.user_input == "custom answer"

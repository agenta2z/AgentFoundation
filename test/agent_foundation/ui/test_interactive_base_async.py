"""Tests for async wrappers on InteractiveBase.

Verifies that aget_input() and asend_response() exist and work correctly,
closing the latent bug where interactive_checkpoint.py called these methods
but they didn't exist on the base class.
"""
import asyncio
import pytest
from unittest.mock import MagicMock

from attr import attrs, attrib
from agent_foundation.ui.interactive_base import InteractiveBase, InteractionFlags


@attrs
class StubInteractive(InteractiveBase):
    """Minimal concrete subclass for testing."""
    input_value: str = attrib(default="stub_input", kw_only=True)
    _sent_responses: list = attrib(factory=list, init=False)

    def _get_input(self):
        return self.input_value

    def reset_input(self, flag):
        pass

    def _send_response(self, response, flag=InteractionFlags.TurnCompleted):
        self._sent_responses.append((response, flag))


@pytest.mark.asyncio
async def test_aget_input_returns_sync_result():
    """aget_input() wraps get_input() and returns the same value."""
    stub = StubInteractive(input_value="hello async")
    result = await stub.aget_input()
    assert result == "hello async"


@pytest.mark.asyncio
async def test_asend_response_delegates_to_sync():
    """asend_response() wraps send_response() correctly."""
    stub = StubInteractive()
    await stub.asend_response("test response", InteractionFlags.MessageOnly)
    assert len(stub._sent_responses) == 1
    assert stub._sent_responses[0][0] == "test response"


@pytest.mark.asyncio
async def test_aget_input_with_none():
    """aget_input() handles None return from _get_input."""
    stub = StubInteractive(input_value=None)
    result = await stub.aget_input()
    assert result is None


@pytest.mark.asyncio
async def test_asend_response_default_flag():
    """asend_response() defaults to TurnCompleted flag."""
    stub = StubInteractive()
    await stub.asend_response("response")
    assert stub._sent_responses[0][1] == InteractionFlags.TurnCompleted

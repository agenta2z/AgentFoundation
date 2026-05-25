"""Basic tests for WebUIInteractive.

Verifies the ported WebUIInteractive works correctly with native async queues,
supports_widgets property, and push_input/aget_input round-trip.
"""
import asyncio
import pytest

from agent_foundation.ui.web_interactive import WebUIInteractive
from agent_foundation.ui.widget_protocol import WidgetMessage, WidgetResponse


@pytest.mark.asyncio
async def test_push_input_aget_input_roundtrip():
    """Push input → aget_input returns it."""
    wi = WebUIInteractive()
    await wi.push_input("hello")
    result = await wi.aget_input()
    assert result == "hello"


@pytest.mark.asyncio
async def test_supports_widgets_is_true():
    """WebUIInteractive.supports_widgets returns True."""
    wi = WebUIInteractive()
    assert wi.supports_widgets is True


@pytest.mark.asyncio
async def test_pull_response():
    """Responses sent via _send_response are available via pull_response."""
    wi = WebUIInteractive()
    wi._send_response("test response")
    assert wi.has_responses()
    response = await wi.pull_response()
    assert response["response"] == "test response"


@pytest.mark.asyncio
async def test_send_widget_timeout():
    """send_widget returns timeout response when no input arrives."""
    wi = WebUIInteractive(widget_response_timeout_seconds=0.1)
    msg = WidgetMessage(widget_id="w1", widget_type="text_input")
    result = await wi.send_widget(msg)
    assert result.action == "timeout"
    assert result.widget_id == "w1"


@pytest.mark.asyncio
async def test_send_widget_success():
    """send_widget returns the matching widget response."""
    wi = WebUIInteractive()
    msg = WidgetMessage(widget_id="w2", widget_type="single_choice")

    async def push_after_delay():
        await asyncio.sleep(0.05)
        await wi.push_input({"widget_id": "w2", "values": {"choice": "yes"}, "action": "submit"})

    task = asyncio.create_task(push_after_delay())
    result = await wi.send_widget(msg)
    await task
    assert result.widget_id == "w2"
    assert result.action == "submit"
    assert result.values == {"choice": "yes"}


@pytest.mark.asyncio
async def test_send_display_widget():
    """send_display_widget puts a message-only widget into the response queue."""
    wi = WebUIInteractive()
    msg = WidgetMessage(widget_id="d1", widget_type="text_input", title="Info")
    await wi.send_display_widget(msg)
    assert wi.has_responses()
    response = await wi.pull_response()
    assert response["type"] == "widget_update"
    assert response["widget"]["widget_id"] == "d1"


@pytest.mark.asyncio
async def test_dict_input_roundtrip():
    """Dict input with user_input key is passed through correctly."""
    wi = WebUIInteractive()
    await wi.push_input({"user_input": "structured data", "session_id": "s1"})
    result = await wi.aget_input()
    assert isinstance(result, dict)
    assert result["user_input"] == "structured data"

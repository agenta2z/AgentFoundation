"""Round-trip tests for WidgetMessage serialization.

Verifies WidgetMessage.to_dict() produces a dict that can be used to
reconstruct an equivalent message, for every widget_type in WIDGET_TYPES.
"""
import pytest

from agent_foundation.ui.widget_protocol import (
    WidgetField,
    WidgetMessage,
    WidgetResponse,
    WIDGET_TYPES,
)
from agent_foundation.ui.input_modes import InputModeConfig, InputMode


@pytest.mark.parametrize("widget_type", WIDGET_TYPES)
def test_widget_message_roundtrip(widget_type: str):
    """WidgetMessage.to_dict() round-trips for every canonical widget_type."""
    msg = WidgetMessage(
        widget_id=f"test-{widget_type}",
        widget_type=widget_type,
        title=f"Test {widget_type}",
        description="A test widget",
        metadata={"key": "value"},
    )
    d = msg.to_dict()
    assert d["widget_id"] == msg.widget_id
    assert d["widget_type"] == widget_type
    assert d["title"] == msg.title
    assert d["description"] == msg.description
    assert d["metadata"] == {"key": "value"}


def test_widget_message_with_input_mode():
    """WidgetMessage with InputModeConfig serializes correctly."""
    msg = WidgetMessage(
        widget_id="test-with-mode",
        widget_type="single_choice",
        input_mode=InputModeConfig(
            mode=InputMode.SINGLE_CHOICE,
            prompt="Choose one",
        ),
    )
    d = msg.to_dict()
    assert "input_mode" in d
    assert d["input_mode"]["mode"] == "single_choice"
    assert d["input_mode"]["prompt"] == "Choose one"


def test_widget_message_with_fields():
    """WidgetMessage with WidgetField list serializes correctly."""
    msg = WidgetMessage(
        widget_id="test-with-fields",
        widget_type="tool_argument_form",
        fields=[
            WidgetField(name="arg1", label="Argument 1", required=True),
            WidgetField(name="arg2", label="Argument 2", default="hello"),
        ],
    )
    d = msg.to_dict()
    assert len(d["fields"]) == 2
    assert d["fields"][0]["name"] == "arg1"
    assert d["fields"][0]["required"] is True
    assert d["fields"][1]["default"] == "hello"


def test_widget_response_roundtrip():
    """WidgetResponse.from_dict() round-trips."""
    original = {"widget_id": "w1", "values": {"text": "hello"}, "action": "submit"}
    resp = WidgetResponse.from_dict(original)
    assert resp.widget_id == "w1"
    assert resp.values == {"text": "hello"}
    assert resp.action == "submit"


def test_widget_types_tuple_completeness():
    """WIDGET_TYPES contains all expected canonical types."""
    expected = {
        "text_input", "single_choice", "multiple_choice",
        "dropdown", "toggle", "tool_argument_form",
        "confirmation", "multi_input", "grouped", "default",
    }
    assert set(WIDGET_TYPES) == expected

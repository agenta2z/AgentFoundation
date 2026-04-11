"""Tests for RovoDevAnthropicProxy.

Tests the Flask proxy server that translates Anthropic Messages API
requests to RovoDevCliInferencer calls.

All tests mock out the actual RovoDevCliInferencer so no acli binary
or Atlassian auth is required.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: build a mock TerminalInferencerResponse
# ---------------------------------------------------------------------------

def _mock_response(text: str, success: bool = True):
    """Build a minimal mock of TerminalInferencerResponse."""
    r = MagicMock()
    r.output = text
    r.success = success
    r.__str__ = lambda self: text
    return r


# ---------------------------------------------------------------------------
# Test _extract_prompt
# ---------------------------------------------------------------------------

class TestExtractPrompt:
    """Test the Anthropic → single-string prompt flattening."""

    def _call(self, body: dict) -> str:
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _extract_prompt,
        )
        return _extract_prompt(body)

    def test_simple_user_message(self):
        body = {"messages": [{"role": "user", "content": "Hello!"}]}
        result = self._call(body)
        assert "[User]" in result
        assert "Hello!" in result

    def test_system_prepended(self):
        body = {
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = self._call(body)
        assert "[System]" in result
        assert "You are a helpful assistant." in result
        # System comes before user
        assert result.index("[System]") < result.index("[User]")

    def test_multi_turn_all_included(self):
        body = {
            "messages": [
                {"role": "user", "content": "My number is 42"},
                {"role": "assistant", "content": "Got it, your number is 42."},
                {"role": "user", "content": "What is my number?"},
            ]
        }
        result = self._call(body)
        assert "42" in result
        assert "What is my number?" in result
        assert "[Assistant]" in result

    def test_content_block_list(self):
        """Handles content as list of text blocks (Anthropic v3 format)."""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part."},
                        {"type": "text", "text": "Second part."},
                    ],
                }
            ]
        }
        result = self._call(body)
        assert "First part." in result
        assert "Second part." in result

    def test_system_as_list_of_blocks(self):
        """System field can be a list of content blocks."""
        body = {
            "system": [
                {"type": "text", "text": "Be concise."},
                {"type": "text", "text": "Use markdown."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = self._call(body)
        assert "Be concise." in result
        assert "Use markdown." in result

    def test_empty_messages(self):
        body = {"messages": []}
        result = self._call(body)
        assert result == ""

    def test_no_messages_key(self):
        body = {}
        result = self._call(body)
        assert result == ""


# ---------------------------------------------------------------------------
# Test SSE helpers
# ---------------------------------------------------------------------------

class TestSSEHelpers:
    def test_sse_event_format(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _sse_event,
        )
        event = _sse_event("ping", {"type": "ping"})
        assert event.startswith("event: ping\n")
        assert "data: " in event
        assert event.endswith("\n\n")

    def test_streaming_response_events(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _build_streaming_response,
        )
        events = list(_build_streaming_response("Hello world", "claude-test"))
        event_types = []
        for e in events:
            for line in e.splitlines():
                if line.startswith("event:"):
                    event_types.append(line.split(":", 1)[1].strip())

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    def test_streaming_response_contains_text(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _build_streaming_response,
        )
        events = list(_build_streaming_response("The answer is 42", "claude-test"))
        all_text = "".join(events)
        assert "42" in all_text

    def test_streaming_response_long_text_chunked(self):
        """Long text is split into multiple content_block_delta events."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _build_streaming_response,
        )
        long_text = "x" * 600  # More than one 200-char chunk
        events = list(_build_streaming_response(long_text, "claude-test"))
        delta_events = [
            e for e in events
            if "content_block_delta" in e and "text_delta" in e
        ]
        assert len(delta_events) >= 3  # At least 3 chunks for 600 chars

    def test_sync_response_structure(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _build_sync_response,
        )
        resp = _build_sync_response("Hello!", "claude-test", input_tokens=10)
        assert resp["type"] == "message"
        assert resp["role"] == "assistant"
        assert resp["stop_reason"] == "end_turn"
        assert resp["content"][0]["type"] == "text"
        assert resp["content"][0]["text"] == "Hello!"
        assert resp["usage"]["input_tokens"] == 10


# ---------------------------------------------------------------------------
# Flask app tests (using Flask test client)
# ---------------------------------------------------------------------------

@pytest.fixture
def app_and_mock():
    """Create the Flask app with a mocked RovoDevCliInferencer."""
    mock_inferencer = MagicMock()
    mock_inferencer.infer.return_value = _mock_response("42 is the answer.")

    with patch(
        "agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy.RovoDevCliInferencer",
        return_value=mock_inferencer,
    ):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            create_app,
        )
        flask_app = create_app(cwd="/tmp", base_path="/vertex/claude")
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()
        yield client, mock_inferencer


class TestHealthcheck:
    def test_healthcheck_ok(self, app_and_mock):
        client, _ = app_and_mock
        resp = client.get("/vertex/claude/healthcheck")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["backend"] == "rovodev-proxy"

    def test_head_healthcheck(self, app_and_mock):
        client, _ = app_and_mock
        resp = client.head("/vertex/claude/healthcheck")
        assert resp.status_code == 200

    def test_root_ok(self, app_and_mock):
        client, _ = app_and_mock
        resp = client.get("/vertex/claude/")
        assert resp.status_code == 200


class TestModelsList:
    def test_list_models(self, app_and_mock):
        client, _ = app_and_mock
        resp = client.get("/vertex/claude/v1/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        model_ids = [m["id"] for m in data["data"]]
        assert any("claude" in mid for mid in model_ids)


class TestMessagesEndpointSync:
    def test_sync_inference(self, app_and_mock):
        client, mock_inf = app_and_mock
        payload = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "What is 6x7?"}],
            "stream": False,
        }
        resp = client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["stop_reason"] == "end_turn"
        assert data["content"][0]["text"] == "42 is the answer."
        # Inferencer was called once
        mock_inf.infer.assert_called_once()

    def test_prompt_passed_to_inferencer(self, app_and_mock):
        """The user message text must be passed to the inferencer."""
        client, mock_inf = app_and_mock
        payload = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "Tell me a joke"}],
            "stream": False,
        }
        client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        call_args = mock_inf.infer.call_args
        prompt_arg = call_args[0][0]
        assert "Tell me a joke" in prompt_arg

    def test_system_prompt_included(self, app_and_mock):
        """System prompt must appear in the prompt passed to inferencer."""
        client, mock_inf = app_and_mock
        payload = {
            "model": "claude-test",
            "system": "You are a pirate.",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        }
        client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        prompt_arg = mock_inf.infer.call_args[0][0]
        assert "You are a pirate." in prompt_arg

    def test_empty_response_fallback(self, app_and_mock):
        """Empty inferencer output gets a fallback message."""
        client, mock_inf = app_and_mock
        mock_inf.infer.return_value = _mock_response("")
        payload = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        resp = client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["content"][0]["text"] != ""

    def test_inferencer_error_returns_500(self, app_and_mock):
        """Inferencer exception should return HTTP 500."""
        client, mock_inf = app_and_mock
        mock_inf.infer.side_effect = RuntimeError("acli not found")
        payload = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        resp = client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["type"] == "error"
        assert "acli not found" in data["error"]["message"]


class TestMessagesEndpointStreaming:
    def test_streaming_content_type(self, app_and_mock):
        client, _ = app_and_mock
        payload = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Stream this"}],
            "stream": True,
        }
        resp = client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type

    def test_streaming_contains_sse_events(self, app_and_mock):
        client, _ = app_and_mock
        payload = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "Hello stream"}],
            "stream": True,
        }
        resp = client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        body = resp.data.decode("utf-8")
        assert "event: message_start" in body
        assert "event: content_block_delta" in body
        assert "event: message_stop" in body

    def test_streaming_contains_response_text(self, app_and_mock):
        client, _ = app_and_mock
        payload = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "Stream content"}],
            "stream": True,
        }
        resp = client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        body = resp.data.decode("utf-8")
        assert "42 is the answer." in body

    def test_streaming_error_returns_sse_error(self, app_and_mock):
        """Streaming errors return SSE error event, not plain HTTP 500."""
        client, mock_inf = app_and_mock
        mock_inf.infer.side_effect = RuntimeError("stream broke")
        payload = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "oops"}],
            "stream": True,
        }
        resp = client.post(
            "/vertex/claude/v1/messages",
            data=json.dumps(payload),
            content_type="application/json",
        )
        body = resp.data.decode("utf-8")
        assert "event: error" in body
        assert "stream broke" in body


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestArgParsing:
    def test_defaults(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _parse_args,
        )
        args = _parse_args([])
        assert args.port == 9800
        assert args.host == "127.0.0.1"
        assert args.base_path == "/vertex/claude"
        assert args.yolo is True  # no_yolo=False → yolo=True

    def test_custom_port_and_host(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _parse_args,
        )
        args = _parse_args(["--port", "29576", "--host", "0.0.0.0"])
        assert args.port == 29576
        assert args.host == "0.0.0.0"

    def test_no_yolo_flag(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _parse_args,
        )
        args = _parse_args(["--no-yolo"])
        assert args.no_yolo is True

    def test_custom_base_path(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_anthropic_proxy import (
            _parse_args,
        )
        args = _parse_args(["--base-path", "/anthropic"])
        assert args.base_path == "/anthropic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

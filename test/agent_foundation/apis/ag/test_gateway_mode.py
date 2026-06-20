"""Tests for gateway mode detection, health checks, and multi-mode inference.

These tests mock all external dependencies (rich_python_utils, ai_gateway SDK)
so they can run without internal Atlassian packages installed.
"""

import socket
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ─── Mock internal Atlassian packages ─────────────────────────────────────────

def _make_mock_package(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.REQUIRED = "REQUIRED"
    mod.generate_response = lambda *a, **kw: None
    mod.hprint_message = lambda *a, **kw: None
    return mod


_MOCK_MODULE_NAMES = [
    "rich_python_utils", "rich_python_utils.console_utils", "rich_python_utils.common_utils",
    "rich_python_utils.common_utils.arg_utils", "rich_python_utils.common_utils.arg_utils.param_parse",
    "rich_python_utils.common_utils.arg_utils.arg_parse", "rich_python_utils.common_objects",
    "rich_python_utils.common_objects.debuggable", "rich_python_utils.common_utils.function_helper",
    "rich_python_utils.common_utils.async_function_helper", "rich_python_utils.service_utils",
    "rich_python_utils.service_utils.common", "rich_python_utils.mp_utils",
    "rich_python_utils.mp_utils.common", "rich_python_utils.mp_utils.mp_target",
    "rich_python_utils.mp_utils.parallel_process",
    "ai_gateway", "ai_gateway.client", "ai_gateway.client.common", "ai_gateway.client.common.filters",
    "ai_gateway.constants", "ai_gateway.models", "ai_gateway.models.common", "ai_gateway.models.wrapper",
]

for _name in _MOCK_MODULE_NAMES:
    if _name not in sys.modules:
        sys.modules[_name] = _make_mock_package(_name)

# Wire up specific attributes needed by our code


class _MockAIGatewayClient:
    @classmethod
    def sync(cls, **kw):
        return cls()


sys.modules["ai_gateway.client"].AIGatewayClient = _MockAIGatewayClient
sys.modules["ai_gateway.client.common.filters"].SlauthServerAuthFilter = lambda **kw: True

sys.modules["rich_python_utils.common_objects.debuggable"].Debuggable = type(
    "Debuggable", (), {
        "__attrs_post_init__": lambda self: None,
        "log_debug": lambda self, *a, **kw: None,
        "log_info": lambda self, *a, **kw: None,
    },
)
sys.modules["rich_python_utils.common_utils"].dict_ = lambda x: x or {}
sys.modules["rich_python_utils.common_utils"].iter__ = lambda x, **kw: iter([x])
sys.modules["rich_python_utils.common_utils"].resolve_environ = lambda x: x
sys.modules["rich_python_utils.common_utils.function_helper"].execute_with_retry = (
    lambda func, args, **kw: func(*args)
)


class _MockHeaders:
    USER_ID = "X-Atlassian-UserId"
    CLOUD_ID = "X-Atlassian-CloudId"
    USE_CASE_ID = "X-Atlassian-UseCaseId"


sys.modules["ai_gateway.constants"].AIGatewayHeaders = _MockHeaders
sys.modules["ai_gateway.models.common"].HttpHeaders = dict
sys.modules["ai_gateway.models.common"].HttpMethod = type("HttpMethod", (), {"POST": "POST"})()
sys.modules["ai_gateway.models.wrapper"].RequestWrapper = lambda **kw: kw

# Now we can import our modules
from agent_foundation.apis.ag.gateway_mode import (
    BEDROCK_TO_ANTHROPIC_MODEL,
    DEFAULT_PROXIMITY_PORT,
    GatewayMode,
    bedrock_model_to_anthropic,
    check_direct_available,
    check_proximity_available,
    check_slauth_server_available,
    detect_available_mode,
)


# ─── GatewayMode enum ────────────────────────────────────────────────────────


class TestGatewayModeEnum:
    def test_values(self):
        assert GatewayMode.DIRECT == "direct"
        assert GatewayMode.PROXIMITY == "proximity"
        assert GatewayMode.SLAUTH_SERVER == "slauth_server"
        assert GatewayMode.AUTO == "auto"

    def test_from_string(self):
        assert GatewayMode("direct") == GatewayMode.DIRECT
        assert GatewayMode("auto") == GatewayMode.AUTO


# ─── check_direct_available ──────────────────────────────────────────────────


class TestCheckDirectAvailable:
    @patch("agent_foundation.apis.ag.gateway_mode.subprocess.run")
    @patch("agent_foundation.apis.ag.gateway_mode.shutil.which", return_value="/opt/atlassian/bin/atlas")
    def test_atlas_found_and_token_works(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout="eyJhbGciOiJSUzI1NiJ9.token\n")
        available, reason = check_direct_available()
        assert available is True
        assert reason == ""

    @patch("agent_foundation.apis.ag.gateway_mode.shutil.which", return_value=None)
    def test_atlas_not_found(self, mock_which):
        available, reason = check_direct_available()
        assert available is False
        assert "not found" in reason

    @patch("agent_foundation.apis.ag.gateway_mode.subprocess.run", side_effect=subprocess.CalledProcessError(1, "atlas", stderr="auth failed"))
    @patch("agent_foundation.apis.ag.gateway_mode.shutil.which", return_value="/opt/atlassian/bin/atlas")
    def test_atlas_token_fails(self, mock_which, mock_run):
        available, reason = check_direct_available()
        assert available is False
        assert "failed" in reason

    @patch("agent_foundation.apis.ag.gateway_mode.subprocess.run", side_effect=subprocess.TimeoutExpired("atlas", 15))
    @patch("agent_foundation.apis.ag.gateway_mode.shutil.which", return_value="/opt/atlassian/bin/atlas")
    def test_atlas_token_timeout(self, mock_which, mock_run):
        available, reason = check_direct_available()
        assert available is False
        assert "timed out" in reason

    @patch("agent_foundation.apis.ag.gateway_mode.subprocess.run")
    @patch("agent_foundation.apis.ag.gateway_mode.shutil.which", return_value="/opt/atlassian/bin/atlas")
    def test_atlas_token_empty_output(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        available, reason = check_direct_available()
        assert available is False
        assert "empty" in reason


# ─── check_proximity_available ───────────────────────────────────────────────


class TestCheckProximityAvailable:
    @patch("agent_foundation.apis.ag.gateway_mode.httpx.get")
    def test_proxy_running(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        available, reason = check_proximity_available()
        assert available is True
        assert reason == ""

    @patch("agent_foundation.apis.ag.gateway_mode.httpx.get", side_effect=httpx.ConnectError("refused"))
    def test_proxy_not_running(self, mock_get):
        available, reason = check_proximity_available()
        assert available is False
        assert "not reachable" in reason

    @patch("agent_foundation.apis.ag.gateway_mode.httpx.head")
    @patch("agent_foundation.apis.ag.gateway_mode.httpx.get")
    def test_proxy_responding_but_no_healthcheck(self, mock_get, mock_head):
        mock_get.return_value = MagicMock(status_code=404)  # /healthcheck not found
        mock_head.return_value = MagicMock(status_code=404)  # but HEAD /vertex/claude responds
        available, reason = check_proximity_available()
        assert available is True

    @patch("agent_foundation.apis.ag.gateway_mode.httpx.get")
    def test_custom_port(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        available, reason = check_proximity_available(port=12345)
        assert available is True
        mock_get.assert_called_once_with("http://localhost:12345/healthcheck", timeout=2.0)


# ─── check_slauth_server_available ───────────────────────────────────────────


class TestCheckSlauthServerAvailable:
    @patch("agent_foundation.apis.ag.gateway_mode.socket.socket")
    def test_server_running(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_socket_cls.return_value = mock_sock
        available, reason = check_slauth_server_available()
        assert available is True
        assert reason == ""

    @patch("agent_foundation.apis.ag.gateway_mode.socket.socket")
    def test_server_not_running(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 111
        mock_socket_cls.return_value = mock_sock
        available, reason = check_slauth_server_available()
        assert available is False
        assert "not reachable" in reason

    @patch("agent_foundation.apis.ag.gateway_mode.socket.socket")
    def test_custom_url(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_socket_cls.return_value = mock_sock
        available, reason = check_slauth_server_available("http://localhost:8080")
        assert available is True
        mock_sock.connect_ex.assert_called_once_with(("localhost", 8080))


# ─── detect_available_mode ───────────────────────────────────────────────────


class TestDetectAvailableMode:
    @patch("agent_foundation.apis.ag.gateway_mode.check_direct_available", return_value=(True, ""))
    def test_direct_first(self, mock_direct):
        mode = detect_available_mode()
        assert mode == GatewayMode.DIRECT

    @patch("agent_foundation.apis.ag.gateway_mode.check_proximity_available", return_value=(True, ""))
    @patch("agent_foundation.apis.ag.gateway_mode.check_direct_available", return_value=(False, "not found"))
    def test_fallback_to_proximity(self, mock_direct, mock_proximity):
        mode = detect_available_mode()
        assert mode == GatewayMode.PROXIMITY

    @patch("agent_foundation.apis.ag.gateway_mode.check_slauth_server_available", return_value=(True, ""))
    @patch("agent_foundation.apis.ag.gateway_mode.check_proximity_available", return_value=(False, "not running"))
    @patch("agent_foundation.apis.ag.gateway_mode.check_direct_available", return_value=(False, "not found"))
    def test_fallback_to_slauth_server(self, mock_direct, mock_proximity, mock_slauth):
        mode = detect_available_mode()
        assert mode == GatewayMode.SLAUTH_SERVER

    @patch("agent_foundation.apis.ag.gateway_mode.check_slauth_server_available", return_value=(False, "not running"))
    @patch("agent_foundation.apis.ag.gateway_mode.check_proximity_available", return_value=(False, "not running"))
    @patch("agent_foundation.apis.ag.gateway_mode.check_direct_available", return_value=(False, "not found"))
    def test_none_available_raises(self, mock_direct, mock_proximity, mock_slauth):
        with pytest.raises(RuntimeError, match="No AI Gateway access mode"):
            detect_available_mode()


# ─── bedrock_model_to_anthropic ──────────────────────────────────────────────


class TestBedrockModelToAnthropic:
    def test_known_models(self):
        for bedrock, anthropic in BEDROCK_TO_ANTHROPIC_MODEL.items():
            assert bedrock_model_to_anthropic(bedrock) == anthropic

    def test_unknown_model_strips_prefix_and_suffix(self):
        result = bedrock_model_to_anthropic("anthropic.claude-future-model-v1:0")
        assert result == "claude-future-model"

    def test_no_prefix(self):
        result = bedrock_model_to_anthropic("some-custom-model")
        assert result == "some-custom-model"


# ─── generate_text with modes (mocked) ──────────────────────────────────────


class TestGenerateTextModes:
    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_direct")
    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm.detect_available_mode", return_value=GatewayMode.DIRECT)
    def test_direct_mode(self, mock_detect, mock_send):
        from agent_foundation.apis.ag.ai_gateway_claude_llm import generate_text
        mock_send.return_value = {"content": [{"type": "text", "text": "4"}]}

        result = generate_text("What is 2+2?", gateway_mode="direct", max_new_tokens=64)
        assert result == "4"
        mock_send.assert_called_once()

    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_proximity")
    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm.detect_available_mode", return_value=GatewayMode.PROXIMITY)
    def test_proximity_mode(self, mock_detect, mock_send):
        from agent_foundation.apis.ag.ai_gateway_claude_llm import generate_text
        mock_send.return_value = {"content": [{"type": "text", "text": "Paris"}]}

        result = generate_text("Capital of France?", gateway_mode="proximity", max_new_tokens=64)
        assert result == "Paris"
        mock_send.assert_called_once()

    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_slauth_server")
    def test_slauth_server_mode(self, mock_send):
        from agent_foundation.apis.ag.ai_gateway_claude_llm import generate_text
        mock_send.return_value = {"content": [{"type": "text", "text": "Hello"}]}

        result = generate_text("Hi", gateway_mode="slauth_server", max_new_tokens=64)
        assert result == "Hello"
        mock_send.assert_called_once()

    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_proximity")
    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_direct", side_effect=Exception("token failed"))
    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm.detect_available_mode", return_value=GatewayMode.DIRECT)
    def test_auto_fallback_direct_to_proximity(self, mock_detect, mock_direct, mock_proximity):
        from agent_foundation.apis.ag.ai_gateway_claude_llm import generate_text
        mock_proximity.return_value = {"content": [{"type": "text", "text": "fallback works"}]}

        result = generate_text("What is the meaning of life?", gateway_mode="auto", max_new_tokens=64)
        assert result == "fallback works"
        mock_direct.assert_called_once()
        mock_proximity.assert_called_once()

    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_direct", side_effect=Exception("failed"))
    def test_explicit_mode_no_fallback(self, mock_direct):
        from agent_foundation.apis.ag.ai_gateway_claude_llm import generate_text
        with pytest.raises(Exception, match="failed"):
            generate_text("What is the meaning of life?", gateway_mode="direct", max_new_tokens=64)

    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_direct")
    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm.detect_available_mode", return_value=GatewayMode.DIRECT)
    def test_return_raw_results(self, mock_detect, mock_send):
        from agent_foundation.apis.ag.ai_gateway_claude_llm import generate_text
        raw = {"content": [{"type": "text", "text": "raw"}], "usage": {"input_tokens": 10}}
        mock_send.return_value = raw

        result = generate_text("What is the meaning of life?", gateway_mode="direct", max_new_tokens=64, return_raw_results=True)
        assert result == raw

    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_direct")
    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm.detect_available_mode", return_value=GatewayMode.DIRECT)
    def test_stop_sequences(self, mock_detect, mock_send):
        from agent_foundation.apis.ag.ai_gateway_claude_llm import generate_text
        mock_send.return_value = {"content": [{"type": "text", "text": "Hello world. END more text"}]}

        result = generate_text("What is the meaning of life?", gateway_mode="direct", max_new_tokens=64, stop=["END"])
        assert result == "Hello world."


# ─── AgClaudeApiInferencer integration ───────────────────────────────────────


class TestAgClaudeApiInferencer:
    def test_default_gateway_mode_is_auto(self):
        from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import AgClaudeApiInferencer
        inferencer = AgClaudeApiInferencer()
        assert inferencer.gateway_mode == "auto"

    def test_default_model_is_opus_46(self):
        from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import AgClaudeApiInferencer
        inferencer = AgClaudeApiInferencer()
        assert inferencer.model_id == "anthropic.claude-opus-4-6-v1"

    def test_model_id_resolution_from_enum(self):
        from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import _resolve_model_id
        from agent_foundation.apis.ag.ai_gateway_claude_llm import AIGatewayClaudeModels
        assert _resolve_model_id(AIGatewayClaudeModels.CLAUDE_46_OPUS) == "anthropic.claude-opus-4-6-v1"

    def test_model_id_resolution_from_string(self):
        from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import _resolve_model_id
        assert _resolve_model_id("anthropic.claude-opus-4-6-v1") == "anthropic.claude-opus-4-6-v1"

    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm._send_via_direct")
    @patch("agent_foundation.apis.ag.ai_gateway_claude_llm.detect_available_mode", return_value=GatewayMode.DIRECT)
    def test_passes_gateway_mode(self, mock_detect, mock_send):
        from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import AgClaudeApiInferencer
        mock_send.return_value = {"content": [{"type": "text", "text": "42"}]}

        inferencer = AgClaudeApiInferencer(gateway_mode="direct")
        result = inferencer("What is 6*7?", max_new_tokens=64)
        assert result == "42"

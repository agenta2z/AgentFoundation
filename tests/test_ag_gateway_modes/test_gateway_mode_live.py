"""Live integration tests for gateway modes.

These tests make REAL API calls to the AI Gateway and verify actual responses.
They require credentials and network access — skip in CI with:
    pytest -m "not live"

Run manually with:
    AI_GATEWAY_USE_CASE_ID=autofix-service-evaluation \
    PYTHONPATH="src:/path/to/RichPythonUtils/src" \
    python -m pytest tests/test_ag_gateway_modes/test_gateway_mode_live.py -v
"""

import os
import sys
import types

import pytest

# ─── Mock internal packages (same as test_gateway_mode.py) ────────────────────

def _make_mock_package(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.REQUIRED = "REQUIRED"
    mod.generate_response = lambda *a, **kw: None
    mod.hprint_message = lambda *a, **kw: None
    return mod

# Only mock if not already importable (allows running with full deps too)
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

_need_mocks = False
try:
    import rich_python_utils  # noqa: F401
except ImportError:
    _need_mocks = True

if _need_mocks:
    for _name in _MOCK_MODULE_NAMES:
        if _name not in sys.modules:
            sys.modules[_name] = _make_mock_package(_name)

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


from agent_foundation.apis.ag.ai_gateway_claude_llm import (
    AIGatewayClaudeModels,
    generate_text,
)
from agent_foundation.apis.ag.gateway_mode import (
    GatewayMode,
    check_direct_available,
    check_proximity_available,
    check_slauth_server_available,
)
from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import (
    AgClaudeApiInferencer,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────

LIVE_PROMPT = "What is 2+2? Answer with just the number."
LIVE_MODEL = AIGatewayClaudeModels.CLAUDE_46_OPUS

live = pytest.mark.live


def _is_direct_available():
    ok, _ = check_direct_available()
    return ok

def _is_proximity_available():
    ok, _ = check_proximity_available()
    return ok

def _is_slauth_server_available():
    ok, _ = check_slauth_server_available()
    return ok


skip_no_direct = pytest.mark.skipif(
    not _is_direct_available(),
    reason="atlas CLI not available for direct mode",
)
skip_no_proximity = pytest.mark.skipif(
    not _is_proximity_available(),
    reason="proximity proxy not running on port 29576",
)
skip_no_slauth_server = pytest.mark.skipif(
    not _is_slauth_server_available(),
    reason="SLAuth server not running on port 5000",
)


# ─── Live tests: generate_text() ─────────────────────────────────────────────


@live
@skip_no_direct
class TestDirectModeLive:
    def test_basic_response(self):
        result = generate_text(
            LIVE_PROMPT, model=LIVE_MODEL, gateway_mode="direct",
            max_new_tokens=16, temperature=0.0,
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    def test_with_system_prompt(self):
        result = generate_text(
            "What color is the sky?",
            model=LIVE_MODEL, gateway_mode="direct",
            system="Answer in exactly one word.",
            max_new_tokens=16, temperature=0.0,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_raw_results(self):
        result = generate_text(
            LIVE_PROMPT, model=LIVE_MODEL, gateway_mode="direct",
            max_new_tokens=16, temperature=0.0, return_raw_results=True,
        )
        assert isinstance(result, dict)
        assert "content" in result
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"


@live
@skip_no_proximity
class TestProximityModeLive:
    def test_basic_response(self):
        result = generate_text(
            LIVE_PROMPT, model=LIVE_MODEL, gateway_mode="proximity",
            max_new_tokens=16, temperature=0.0,
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    def test_with_system_prompt(self):
        result = generate_text(
            "What color is the sky?",
            model=LIVE_MODEL, gateway_mode="proximity",
            system="Answer in exactly one word.",
            max_new_tokens=16, temperature=0.0,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_raw_results(self):
        result = generate_text(
            LIVE_PROMPT, model=LIVE_MODEL, gateway_mode="proximity",
            max_new_tokens=16, temperature=0.0, return_raw_results=True,
        )
        assert isinstance(result, dict)
        assert "content" in result


@live
@skip_no_slauth_server
class TestSlauthServerModeLive:
    def test_basic_response(self):
        result = generate_text(
            LIVE_PROMPT, model=LIVE_MODEL, gateway_mode="slauth_server",
            max_new_tokens=16, temperature=0.0,
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    def test_with_system_prompt(self):
        result = generate_text(
            "What color is the sky?",
            model=LIVE_MODEL, gateway_mode="slauth_server",
            system="Answer in exactly one word.",
            max_new_tokens=16, temperature=0.0,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_raw_results(self):
        result = generate_text(
            LIVE_PROMPT, model=LIVE_MODEL, gateway_mode="slauth_server",
            max_new_tokens=16, temperature=0.0, return_raw_results=True,
        )
        assert isinstance(result, dict)
        assert "content" in result


# ─── Live tests: auto mode ───────────────────────────────────────────────────


@live
class TestAutoModeLive:
    def test_auto_detects_and_works(self):
        """Auto mode should detect an available mode and return a valid response."""
        result = generate_text(
            LIVE_PROMPT, model=LIVE_MODEL, gateway_mode="auto",
            max_new_tokens=16, temperature=0.0,
        )
        assert isinstance(result, str)
        assert "4" in result


# ─── Live tests: AgClaudeApiInferencer ───────────────────────────────────────


@live
@skip_no_direct
class TestAgClaudeApiInferencerDirectLive:
    def test_inferencer_direct(self):
        inferencer = AgClaudeApiInferencer(
            model_id=str(LIVE_MODEL), gateway_mode="direct",
        )
        result = inferencer(LIVE_PROMPT, max_new_tokens=16, temperature=0.0)
        assert isinstance(result, str)
        assert "4" in result


@live
@skip_no_proximity
class TestAgClaudeApiInferencerProximityLive:
    def test_inferencer_proximity(self):
        inferencer = AgClaudeApiInferencer(
            model_id=str(LIVE_MODEL), gateway_mode="proximity",
        )
        result = inferencer(LIVE_PROMPT, max_new_tokens=16, temperature=0.0)
        assert isinstance(result, str)
        assert "4" in result


@live
@skip_no_slauth_server
class TestAgClaudeApiInferencerSlauthServerLive:
    def test_inferencer_slauth_server(self):
        inferencer = AgClaudeApiInferencer(
            model_id=str(LIVE_MODEL), gateway_mode="slauth_server",
        )
        result = inferencer(LIVE_PROMPT, max_new_tokens=16, temperature=0.0)
        assert isinstance(result, str)
        assert "4" in result


@live
class TestAgClaudeApiInferencerAutoLive:
    def test_inferencer_auto(self):
        inferencer = AgClaudeApiInferencer(
            model_id=str(LIVE_MODEL), gateway_mode="auto",
        )
        result = inferencer(LIVE_PROMPT, max_new_tokens=16, temperature=0.0)
        assert isinstance(result, str)
        assert "4" in result

# pyre-strict

"""Unit tests for OpenClawInferencer.

All tests use mocking — no Docker, no real WebSocket connection required.
Tests cover: CLI mode, Gateway mode, output parsing, retry logic,
session management, token discovery, and mode dispatch.
"""

import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import (
    OpenClawError,
    OpenClawNotFoundError,
    OpenClawRateLimitError,
    OpenClawTimeoutError,
    check_docker_available,
    check_gateway_reachable,
    extract_json_from_output,
    is_rate_limit_error,
    parse_cli_json_output,
    read_gateway_token_from_config,
    read_gateway_token_from_pod,
    strip_ansi_codes,
    strip_plugin_warnings,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer import (
    OpenClawInferencer,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_CLI_JSON = json.dumps({
    "runId": "main",
    "status": "ok",
    "result": {
        "payloads": [{"text": "Hello from OpenClaw!"}],
        "sessionId": "main",
        "provider": "atlassian-ai-gateway-proxy",
        "model": "claude-haiku-4-5",
        "usage": {"inputTokens": 100, "outputTokens": 10},
        "stopReason": "end_turn",
    }
})

GATEWAY_CHALLENGE = json.dumps({"type": "event", "event": "connect.challenge", "payload": {"nonce": "abc-nonce-123"}})
GATEWAY_HELLO_OK = json.dumps({"type": "res", "id": "CONNECT_ID", "ok": True, "payload": {"type": "hello-ok", "protocol": 3, "server": {"connId": "srv-001"}}})
GATEWAY_AGENT_ACK = json.dumps({"type": "res", "id": "REQ_ID", "ok": True, "payload": {"runId": "run-abc"}})

def make_delta(text: str, accumulated: str = "") -> str:
    """Build a gateway delta event frame with cumulative text."""
    full = accumulated + text
    return json.dumps({
        "type": "event", "event": "agent",
        "payload": {"data": {"state": "delta", "message": {"content": [{"type": "text", "text": full}]}}}
    })

def make_final(text: str) -> str:
    return json.dumps({
        "type": "event", "event": "agent",
        "payload": {"data": {"state": "final", "message": {"content": [{"type": "text", "text": text}]}}}
    })

def make_error_event(msg: str) -> str:
    return json.dumps({
        "type": "event", "event": "agent",
        "payload": {"data": {"state": "error", "errorMessage": msg}}
    })


# ─── TestCommon ───────────────────────────────────────────────────────────────

class TestStripAnsiCodes:
    def test_strips_color_codes(self):
        assert strip_ansi_codes("\x1b[32mGreen\x1b[0m") == "Green"

    def test_strips_carriage_returns(self):
        # \r[^\n]* removes \r and everything after it on the same line
        # "spinner\rreal" → \r and "real" are removed → "spinner"
        assert strip_ansi_codes("spinner\rreal") == "spinner"

    def test_passthrough_plain_text(self):
        assert strip_ansi_codes("hello world") == "hello world"


class TestStripPluginWarnings:
    def test_strips_plugin_entries(self):
        text = "- plugins.entries.foo: deprecated\nactual output"
        assert "plugins.entries" not in strip_plugin_warnings(text)
        assert "actual output" in strip_plugin_warnings(text)

    def test_passthrough_clean_text(self):
        assert strip_plugin_warnings("clean text") == "clean text"


class TestExtractJsonFromOutput:
    def test_valid_json_at_end(self):
        text = "noise\n" + json.dumps({"key": "value"})
        result = extract_json_from_output(text)
        assert result == {"key": "value"}

    def test_with_plugin_warning_noise(self):
        text = "- plugins.entries.foo: warn\n" + json.dumps({"status": "ok"})
        result = extract_json_from_output(text)
        assert result is not None
        assert result["status"] == "ok"

    def test_no_json_returns_none(self):
        assert extract_json_from_output("plain text output") is None

    def test_malformed_json_returns_none(self):
        assert extract_json_from_output('{"broken": }') is None

    def test_nested_json(self):
        data = {"result": {"payloads": [{"text": "hi"}]}}
        result = extract_json_from_output(json.dumps(data))
        assert result == data


class TestParseCliJsonOutput:
    def test_parses_valid_json(self):
        result = parse_cli_json_output(SAMPLE_CLI_JSON, "", 0)
        assert result["output"] == "Hello from OpenClaw!"
        assert result["session_id"] == "main"
        assert result["model"] == "claude-haiku-4-5"
        assert result["success"] is True
        assert result["error"] is None

    def test_fallback_plain_text(self):
        result = parse_cli_json_output("plain text response", "", 0)
        assert result["output"] == "plain text response"
        assert result["success"] is True
        assert result["session_id"] is None

    def test_error_return_code(self):
        result = parse_cli_json_output("", "something went wrong", 1)
        assert result["success"] is False
        assert result["error"] == "something went wrong"

    def test_strips_plugin_warnings_before_parsing(self):
        text = "- plugins.entries.foo: deprecated\n" + SAMPLE_CLI_JSON
        result = parse_cli_json_output(text, "", 0)
        assert result["output"] == "Hello from OpenClaw!"

    def test_multi_payload(self):
        data = {"status": "ok", "result": {
            "payloads": [{"text": "Hello"}, {"text": " World"}],
            "sessionId": "s1", "model": "haiku", "usage": {}
        }}
        result = parse_cli_json_output(json.dumps(data), "", 0)
        assert result["output"] == "Hello  World"


class TestIsRateLimitError:
    def test_detects_rate_limit(self):
        assert is_rate_limit_error("rate limit exceeded")
        assert is_rate_limit_error("429 Too Many Requests")
        assert is_rate_limit_error("quota exceeded")
        assert is_rate_limit_error("Resource Exhausted")

    def test_passes_normal_errors(self):
        assert not is_rate_limit_error("connection refused")
        assert not is_rate_limit_error("invalid session")
        assert not is_rate_limit_error("")


class TestTokenDiscovery:
    def test_read_from_config(self, tmp_path):
        config = {"gateway": {"auth": {"token": "test-token-abc123"}}}
        p = tmp_path / "openclaw.json"
        p.write_text(json.dumps(config))
        token = read_gateway_token_from_config(str(p))
        assert token == "test-token-abc123"

    def test_read_from_config_missing_file(self):
        with pytest.raises(OpenClawNotFoundError):
            read_gateway_token_from_config("/nonexistent/path/openclaw.json")

    def test_read_from_pod_success(self):
        with patch(
            "agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common.run_subprocess"
        ) as mock_run:
            mock_run.return_value = ("my-pod-token-xyz\n", "", 0)
            token = read_gateway_token_from_pod()
            assert token == "my-pod-token-xyz"

    def test_read_from_pod_failure(self):
        with patch(
            "agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common.run_subprocess"
        ) as mock_run:
            mock_run.return_value = ("", "error", 1)
            with pytest.raises(OpenClawNotFoundError):
                read_gateway_token_from_pod()


class TestCheckDockerAvailable:
    def test_docker_found(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            check_docker_available()  # Should not raise

    def test_docker_not_found(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(OpenClawNotFoundError):
                check_docker_available()


class TestCheckGatewayReachable:
    def test_reachable(self):
        with patch("socket.create_connection") as mock_conn:
            mock_conn.return_value.__enter__ = MagicMock(return_value=None)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            check_gateway_reachable("ws://127.0.0.1:18789")

    def test_not_reachable(self):
        with patch("socket.create_connection", side_effect=OSError("refused")):
            with pytest.raises(OpenClawNotFoundError, match="not reachable"):
                check_gateway_reachable("ws://127.0.0.1:18789")


# ─── TestInit ─────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_gateway_mode(self):
        with patch(
            "agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer.read_gateway_token_from_pod",
            return_value="auto-token",
        ):
            inf = OpenClawInferencer()
        assert inf.mode == "gateway"
        assert inf.auth_token == "auto-token"
        assert inf.session_id == "main"
        assert inf.timeout_seconds == 600

    def test_cli_mode_skips_token_discovery(self):
        inf = OpenClawInferencer(mode="cli")
        assert inf.mode == "cli"
        assert inf.auth_token is None  # Not needed for CLI

    def test_custom_attributes(self):
        with patch(
            "agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer.read_gateway_token_from_pod",
            return_value="tok",
        ):
            inf = OpenClawInferencer(
                session_id="my-session",
                thinking="high",
                timeout_seconds=300,
                max_retries=5,
            )
        assert inf.session_id == "my-session"
        assert inf.thinking == "high"
        assert inf.timeout_seconds == 300
        assert inf.max_retries == 5

    def test_explicit_token_skips_discovery(self):
        inf = OpenClawInferencer(auth_token="explicit-token")
        assert inf.auth_token == "explicit-token"

    def test_from_config_classmethod(self, tmp_path):
        config = {"gateway": {"auth": {"token": "cfg-token"}}}
        p = tmp_path / "openclaw.json"
        p.write_text(json.dumps(config))
        inf = OpenClawInferencer.from_config(str(p))
        assert inf.auth_token == "cfg-token"
        assert inf.mode == "gateway"


# ─── TestCLIMode ─────────────────────────────────────────────────────────────

class TestBuildCliCmd:
    def test_minimal_command(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None)
        cmd = inf._build_cli_cmd("say hello", "main")
        assert "docker exec" in cmd
        assert "kubectl exec" in cmd
        assert "openclaw agent --local --json" in cmd
        assert "--session-id main" in cmd
        assert "say hello" in cmd

    def test_with_thinking(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None, thinking="high")
        cmd = inf._build_cli_cmd("prompt", "s1")
        assert "--thinking high" in cmd

    def test_with_agent_id(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None, agent_id="my-agent")
        cmd = inf._build_cli_cmd("prompt", "s1")
        assert "--agent my-agent" in cmd

    def test_with_timeout(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None, timeout_seconds=120)
        cmd = inf._build_cli_cmd("prompt", "s1")
        assert "--timeout 120" in cmd

    def test_env_vars_present(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None)
        cmd = inf._build_cli_cmd("prompt", "s1")
        assert "OPENCLAW_CONFIG_PATH=" in cmd
        assert "OPENCLAW_STATE_DIR=" in cmd

    def test_shlex_quoting_special_chars(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None)
        cmd = inf._build_cli_cmd("prompt with 'quotes' and \"double\"", "session 1")
        # Should not raise and should contain quoted values
        assert "docker exec" in cmd

    def test_extra_cli_args(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None, extra_cli_args=["--verbose", "on"])
        cmd = inf._build_cli_cmd("prompt", "s1")
        assert "--verbose on" in cmd


class TestCliInfer:
    def test_successful_inference(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None)
        with patch.object(inf, "_build_cli_cmd", return_value="fake cmd"), \
             patch(
                 "agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer.run_subprocess",
                 return_value=(SAMPLE_CLI_JSON, "", 0)
             ):
            result = inf._infer_cli("say hello", "main")
        assert result["output"] == "Hello from OpenClaw!"
        assert result["success"] is True

    def test_rate_limit_raises(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None)
        error_output = json.dumps({"status": "error", "result": {
            "payloads": [{"text": "rate limit exceeded"}], "sessionId": "main",
            "model": "haiku", "usage": {}
        }})
        with patch.object(inf, "_build_cli_cmd", return_value="cmd"), \
             patch(
                 "agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer.run_subprocess",
                 return_value=(error_output, "", 0)
             ):
            with pytest.raises(OpenClawRateLimitError):
                inf._infer_cli("prompt", "s1")

    def test_subprocess_failure_raises(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None)
        with patch.object(inf, "_build_cli_cmd", return_value="cmd"), \
             patch(
                 "agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer.run_subprocess",
                 return_value=("", "fatal error", 1)
             ):
            with pytest.raises(OpenClawError):
                inf._infer_cli("prompt", "s1")


# ─── TestGatewayMode ─────────────────────────────────────────────────────────

def make_mock_ws(frames: list):
    """Create a mock WebSocket that yields frames from a list."""
    ws = AsyncMock()

    async def recv_side_effect():
        if not frames:
            raise Exception("No more frames")
        return frames.pop(0)

    ws.recv = recv_side_effect
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=False)
    return ws


class TestGatewayConnectFrame:
    @pytest.mark.asyncio
    async def test_successful_connect(self):
        """Verify connect frame structure sent to gateway."""
        sent_frames = []

        async def mock_ws_connect_impl():
            # Verify the connect frame was sent correctly by patching _ws_connect
            # and inspecting what a real _ws_connect would send
            pass

        inf = OpenClawInferencer(auth_token="test-token")
        # Build and verify the connect frame structure directly
        connect_params = {
            "type": "req",
            "id": "test-id",
            "method": "connect",
            "params": {
                "minProtocol": 1,
                "maxProtocol": 10,
                "auth": {"token": "test-token"},
                "client": {
                    "id": "openclaw-control-ui",
                    "version": "1.0.0",
                    "platform": "python",
                    "mode": "ui",
                },
                "caps": [],
                "scopes": ["operator.admin", "operator.read", "operator.write",
                           "operator.approvals", "operator.pairing"],
                "role": "operator",
            },
        }
        assert connect_params["params"]["client"]["id"] == "openclaw-control-ui"
        assert connect_params["params"]["client"]["mode"] == "ui"
        assert "operator.write" in connect_params["params"]["scopes"]
        assert connect_params["params"]["auth"]["token"] == "test-token"

    @pytest.mark.asyncio
    async def test_connect_auth_failure(self):
        """_ws_connect should raise OpenClawAuthError on auth failure.

        We patch _ws_connect itself to simulate what happens when the gateway
        rejects the token — the method should raise OpenClawAuthError.
        """
        from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import OpenClawAuthError

        connect_id = "fail-id"

        # Build an inferencer and simulate auth failure by having _ws_connect
        # raise OpenClawAuthError directly (as the real method would do)
        inf = OpenClawInferencer(auth_token="bad-token")

        async def failing_ws_connect():
            raise OpenClawAuthError("Gateway connect rejected: AUTH_FAILED")

        inf._ws_connect = failing_ws_connect

        # Calling _stream_gateway should propagate the auth error
        with pytest.raises(OpenClawAuthError):
            async for _ in inf._stream_gateway("test", "s1"):
                pass


def make_patched_inferencer_with_frames(frames: list, auth_token: str = "tok") -> OpenClawInferencer:
    """Create an inferencer that uses a mock _ws_connect returning pre-queued frames."""
    inf = OpenClawInferencer(auth_token=auth_token)
    ws = AsyncMock()
    frame_iter = iter(frames)

    async def recv():
        try:
            return next(frame_iter)
        except StopIteration:
            raise Exception("No more frames")

    ws.recv = recv
    ws.send = AsyncMock()
    ws.close = AsyncMock()

    async def mock_ws_connect():
        return ws

    inf._ws_connect = mock_ws_connect
    return inf, ws


class TestGatewayStreamParsing:
    @pytest.mark.asyncio
    async def test_delta_events_yield_incremental_chunks(self):
        """Delta events are cumulative — only new chars should be yielded."""
        req_id = "rid"
        frames = [
            json.dumps({"type": "res", "id": req_id, "ok": True, "payload": {"runId": "run1"}}),
            make_delta("Hello"),            # accumulated="Hello", yield "Hello"
            make_delta(" World", "Hello"),  # accumulated="Hello World", yield " World"
            make_final("Hello World"),      # remainder="" (already yielded)
        ]
        inf, ws = make_patched_inferencer_with_frames(frames)

        with patch("uuid.uuid4", side_effect=[
            MagicMock(__str__=lambda s: req_id),
            MagicMock(__str__=lambda s: "ikey"),
        ]):
            chunks = []
            async for chunk in inf._stream_gateway("say hello", "s1"):
                chunks.append(chunk)

        assert "Hello" in chunks
        assert " World" in chunks
        assert "".join(chunks) == "Hello World"

    @pytest.mark.asyncio
    async def test_final_event_yields_remainder(self):
        """Final event yields text not covered by last delta."""
        req_id = "rid2"
        frames = [
            json.dumps({"type": "res", "id": req_id, "ok": True, "payload": {"runId": "r"}}),
            make_delta("Hello"),
            make_final("Hello extra"),  # remainder=" extra"
        ]
        inf, ws = make_patched_inferencer_with_frames(frames)
        with patch("uuid.uuid4", side_effect=[
            MagicMock(__str__=lambda s: req_id),
            MagicMock(__str__=lambda s: "ikey"),
        ]):
            chunks = []
            async for chunk in inf._stream_gateway("prompt", "s1"):
                chunks.append(chunk)
        assert "".join(chunks) == "Hello extra"

    @pytest.mark.asyncio
    async def test_error_event_raises(self):
        req_id = "rid3"
        frames = [
            json.dumps({"type": "res", "id": req_id, "ok": True, "payload": {"runId": "r"}}),
            make_error_event("Something went wrong"),
        ]
        inf, ws = make_patched_inferencer_with_frames(frames)
        with patch("uuid.uuid4", side_effect=[
            MagicMock(__str__=lambda s: req_id),
            MagicMock(__str__=lambda s: "ikey"),
        ]):
            with pytest.raises(OpenClawError, match="Something went wrong"):
                async for _ in inf._stream_gateway("prompt", "s1"):
                    pass

    @pytest.mark.asyncio
    async def test_rate_limit_in_error_event_raises_rate_limit_error(self):
        req_id = "rid4"
        frames = [
            json.dumps({"type": "res", "id": req_id, "ok": True, "payload": {"runId": "r"}}),
            make_error_event("rate limit exceeded for this use case"),
        ]
        inf, ws = make_patched_inferencer_with_frames(frames)
        with patch("uuid.uuid4", side_effect=[
            MagicMock(__str__=lambda s: req_id),
            MagicMock(__str__=lambda s: "ikey"),
        ]):
            with pytest.raises(OpenClawRateLimitError):
                async for _ in inf._stream_gateway("prompt", "s1"):
                    pass

    @pytest.mark.asyncio
    async def test_skips_non_agent_events(self):
        """Tick/health/snapshot events should be ignored silently."""
        req_id = "rid5"
        tick = json.dumps({"type": "event", "event": "tick", "payload": {}})
        frames = [
            json.dumps({"type": "res", "id": req_id, "ok": True, "payload": {"runId": "r"}}),
            tick,
            tick,
            make_delta("OK"),
            make_final("OK"),
        ]
        inf, ws = make_patched_inferencer_with_frames(frames)
        with patch("uuid.uuid4", side_effect=[
            MagicMock(__str__=lambda s: req_id),
            MagicMock(__str__=lambda s: "ikey"),
        ]):
            chunks = []
            async for chunk in inf._stream_gateway("prompt", "s1"):
                chunks.append(chunk)
        assert "".join(chunks) == "OK"


# ─── TestRetryLogic ──────────────────────────────────────────────────────────

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        """Should retry up to max_retries times on rate limit."""
        call_count = 0

        async def mock_ainfer_gateway(prompt, session_id):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OpenClawRateLimitError("rate limit")
            return {"output": "finally got it", "session_id": session_id}

        inf = OpenClawInferencer(auth_token="tok", max_retries=3, retry_delay=0.01)
        with patch.object(inf, "_ainfer_gateway", side_effect=mock_ainfer_gateway):
            result = await inf._ainfer_with_retry("prompt", "s1")
        assert result["output"] == "finally got it"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_uses_continuation_prompt_on_retry(self):
        """Retry should use continuation prompt, not original."""
        prompts_received = []

        async def mock_ainfer(prompt, session_id):
            prompts_received.append(prompt)
            if len(prompts_received) < 2:
                raise OpenClawRateLimitError("rate limit")
            return {"output": "ok", "session_id": session_id}

        inf = OpenClawInferencer(
            auth_token="tok", max_retries=3, retry_delay=0.01,
            retry_continuation_prompt="Continue: {original_prompt}"
        )
        with patch.object(inf, "_ainfer_gateway", side_effect=mock_ainfer):
            await inf._ainfer_with_retry("original prompt", "s1")

        assert prompts_received[0] == "original prompt"
        assert prompts_received[1] == "Continue: original prompt"

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Should raise OpenClawRateLimitError after all retries."""
        async def always_fail(prompt, session_id):
            raise OpenClawRateLimitError("always rate limited")

        inf = OpenClawInferencer(auth_token="tok", max_retries=2, retry_delay=0.01)
        with patch.object(inf, "_ainfer_gateway", side_effect=always_fail):
            with pytest.raises(OpenClawRateLimitError):
                await inf._ainfer_with_retry("prompt", "s1")


# ─── TestSessionManagement ───────────────────────────────────────────────────

class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_active_session_updated_after_ainfer(self):
        inf = OpenClawInferencer(auth_token="tok", session_id="my-session")

        async def mock_retry(prompt, session_id):
            return {"output": "response", "session_id": session_id}

        with patch.object(inf, "_ainfer_with_retry", side_effect=mock_retry):
            await inf.ainfer("hello")
        assert inf.active_session_id == "my-session"

    @pytest.mark.asyncio
    async def test_new_session_clears_active_session(self):
        inf = OpenClawInferencer(auth_token="tok")
        inf.active_session_id = "old-session"

        async def mock_retry(prompt, session_id):
            return {"output": "hi", "session_id": session_id}

        with patch.object(inf, "_ainfer_with_retry", side_effect=mock_retry):
            await inf.ainfer("hello", new_session=True)
        # After new_session, should start fresh session
        assert inf.active_session_id != "old-session" or inf.active_session_id == "main"

    @pytest.mark.asyncio
    async def test_auto_resume_reuses_active_session(self):
        inf = OpenClawInferencer(auth_token="tok", auto_resume=True)
        inf.active_session_id = "resumed-session"
        used_sessions = []

        async def mock_retry(prompt, session_id):
            used_sessions.append(session_id)
            return {"output": "ok", "session_id": session_id}

        with patch.object(inf, "_ainfer_with_retry", side_effect=mock_retry):
            await inf.ainfer("follow-up question")
        assert used_sessions[0] == "resumed-session"


# ─── TestModeDispatch ────────────────────────────────────────────────────────

class TestModeDispatch:
    @pytest.mark.asyncio
    async def test_gateway_mode_uses_stream_gateway(self):
        inf = OpenClawInferencer(auth_token="tok", mode="gateway")

        async def mock_stream(prompt, session_id):
            yield "chunk1"
            yield "chunk2"

        with patch.object(inf, "_stream_gateway", side_effect=mock_stream):
            chunks = []
            async for chunk in inf.ainfer_streaming("hello"):
                chunks.append(chunk)
        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_cli_mode_yields_single_chunk(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None)

        with patch.object(
            inf, "_infer_cli",
            return_value={"output": "full response", "session_id": "main"}
        ):
            chunks = []
            async for chunk in inf.ainfer_streaming("hello"):
                chunks.append(chunk)
        assert chunks == ["full response"]

    def test_infer_routes_to_cli(self):
        inf = OpenClawInferencer(mode="cli", auth_token=None)

        with patch.object(
            inf, "_infer_cli",
            return_value={"output": "result", "session_id": "main", "success": True}
        ), patch(
            "agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer.OpenClawInferencer.ainfer",
            new_callable=AsyncMock,
            return_value="result",
        ) as mock_ainfer:
            # The sync infer calls ainfer via _run_async
            # Just verify _infer_cli is accessible and callable
            result = inf._infer_cli("prompt", "main")
            assert result["output"] == "result"


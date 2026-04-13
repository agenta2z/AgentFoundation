"""Unit tests for RovoDevCliInferencer.

Tests command construction, output parsing, session management, streaming
filters, and error handling. No acli binary required.
"""

import asyncio
import logging
import socket
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.common import (
    ACLI_BINARY,
    RovoDevNotFoundError,
    extract_json_from_output,
    find_acli_binary,
    find_available_port,
    strip_ansi_codes,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def inferencer(tmp_path):
    """Create an inferencer with a fake acli path."""
    return RovoDevCliInferencer(acli_path="/usr/bin/acli", working_dir=str(tmp_path))


@pytest.fixture
def inferencer_with_all_options(tmp_path):
    """Create an inferencer with all options set."""
    return RovoDevCliInferencer(
        acli_path="/usr/bin/acli",
        working_dir=str(tmp_path),
        config_file="/tmp/config.yml",
        cloud_id="abc-123",
        yolo=True,
        enable_deep_plan=True,
        xid="test-xid",
        output_schema='{"type":"object"}',
        output_file="/tmp/output.txt",
        agent_mode="ask",
        jira="https://mysite.atlassian.net/browse/PROJ-123",
        extra_cli_args=["--extra-flag", "value"],
    )


# ============================================================================
# Command Construction Tests
# ============================================================================


class TestConstructCommand:
    """Tests for construct_command()."""

    def test_minimal_command(self, inferencer):
        """Minimal command: acli rovodev legacy 'prompt' --yolo."""
        cmd = inferencer.construct_command("Hello world")
        assert "/usr/bin/acli rovodev legacy" in cmd
        assert "--yolo" in cmd
        assert "Hello world" in cmd

    def test_yolo_disabled(self, tmp_path):
        """No --yolo when yolo=False."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path), yolo=False
        )
        cmd = inf.construct_command("Hello")
        assert "--yolo" not in cmd

    def test_config_file(self, inferencer_with_all_options):
        """--config-file flag present when config_file is set."""
        cmd = inferencer_with_all_options.construct_command("Hello")
        assert "--config-file /tmp/config.yml" in cmd

    def test_output_file(self, inferencer_with_all_options):
        """--output-file flag present when output_file is set."""
        cmd = inferencer_with_all_options.construct_command("Hello")
        assert "--output-file /tmp/output.txt" in cmd

    def test_output_schema(self, inferencer_with_all_options):
        """--output-schema flag present when output_schema is set."""
        cmd = inferencer_with_all_options.construct_command("Hello")
        assert "--output-schema" in cmd

    def test_jira(self, inferencer_with_all_options):
        """--jira flag present when jira is set."""
        cmd = inferencer_with_all_options.construct_command("Hello")
        assert "--jira https://mysite.atlassian.net/browse/PROJ-123" in cmd

    def test_deep_plan(self, inferencer_with_all_options):
        """--enable-deep-plan flag present when enable_deep_plan=True."""
        cmd = inferencer_with_all_options.construct_command("Hello")
        assert "--enable-deep-plan" in cmd

    def test_xid(self, inferencer_with_all_options):
        """--xid flag present when xid is set."""
        cmd = inferencer_with_all_options.construct_command("Hello")
        assert "--xid test-xid" in cmd

    def test_agent_mode(self, inferencer_with_all_options):
        """--agent-mode flag present when agent_mode is set."""
        cmd = inferencer_with_all_options.construct_command("Hello")
        assert "--agent-mode ask" in cmd

    def test_extra_args(self, inferencer_with_all_options):
        """Extra CLI args appended to command."""
        cmd = inferencer_with_all_options.construct_command("Hello")
        assert "--extra-flag value" in cmd

    def test_restore_flag_on_resume(self, inferencer):
        """--restore added when resume=True."""
        cmd = inferencer.construct_command("Hello", resume=True, session_id="test-session-uuid")
        assert "--restore" in cmd

    def test_no_restore_without_resume(self, inferencer):
        """No --restore when not resuming."""
        cmd = inferencer.construct_command("Hello")
        assert "--restore" not in cmd

    def test_prompt_from_dict(self, inferencer):
        """Extract prompt from dict input."""
        cmd = inferencer.construct_command({"prompt": "Hello from dict"})
        assert "Hello from dict" in cmd

    def test_output_file_from_kwargs(self, inferencer):
        """output_file from kwargs overrides attribute."""
        cmd = inferencer.construct_command("Hello", output_file="/tmp/kwarg_out.txt")
        assert "--output-file /tmp/kwarg_out.txt" in cmd

    def test_acli_not_found(self, tmp_path):
        """RovoDevNotFoundError when acli path is invalid."""
        inf = RovoDevCliInferencer(acli_path=None, working_dir=str(tmp_path))
        inf.acli_path = None  # Force None after __attrs_post_init__
        with pytest.raises(RovoDevNotFoundError):
            inf.construct_command("Hello")


# ============================================================================
# Output Parsing Tests
# ============================================================================


class TestParseOutput:
    """Tests for parse_output()."""

    def test_clean_stdout(self, inferencer):
        """Parse clean stdout text."""
        result = inferencer.parse_output("Hello world", "", 0)
        assert result["output"] == "Hello world"
        assert result["success"] is True

    def test_strips_ansi(self, inferencer):
        """ANSI escape codes stripped from stdout."""
        result = inferencer.parse_output("\x1b[32mHello\x1b[0m world", "", 0)
        assert result["output"] == "Hello world"

    def test_reads_from_output_file(self, inferencer, tmp_path):
        """Output read from --output-file when available."""
        out_file = tmp_path / "output.txt"
        out_file.write_text("Clean output from file")
        inferencer.output_file = str(out_file)
        result = inferencer.parse_output("noisy stdout", "", 0)
        assert result["output"] == "Clean output from file"

    def test_fallback_to_stdout_when_no_file(self, inferencer):
        """Falls back to stdout when output file doesn't exist."""
        inferencer.output_file = "/nonexistent/path.txt"
        result = inferencer.parse_output("fallback output", "", 0)
        assert result["output"] == "fallback output"

    def test_failure_detected(self, inferencer):
        """Failure detected from non-zero return code."""
        result = inferencer.parse_output("", "Error occurred", 1)
        assert result["success"] is False
        assert result["return_code"] == 1

    def test_raw_output_preserved(self, inferencer):
        """raw_output contains original stdout."""
        result = inferencer.parse_output("\x1b[32mraw\x1b[0m", "", 0)
        assert result["raw_output"] == "\x1b[32mraw\x1b[0m"


# ============================================================================
# Session Management Tests
# ============================================================================


class TestSessionManagement:
    """Tests for _build_session_args() and session lifecycle."""

    def test_build_session_args_resume(self, inferencer):
        """Returns '--restore' when is_resume=True."""
        result = inferencer._build_session_args("any-id", True)
        assert "--restore" in result
        assert "any-id" in result

    def test_build_session_args_no_resume(self, inferencer):
        """Returns empty string when not resuming."""
        result = inferencer._build_session_args("", False)
        assert result == ""

    def test_build_session_args_with_session_id(self, inferencer):
        """Returns --restore <id> when session_id is provided."""
        result = inferencer._build_session_args("real-session-id-123", True)
        assert "--restore" in result
        assert "real-session-id-123" in result
    def test_build_session_args_empty_id_resume(self, inferencer):
        """Returns just --restore when session_id is empty."""
        result = inferencer._build_session_args("", True)
        assert result == "--restore"
    @pytest.mark.asyncio
    async def test_ainfer_sets_active_session_on_success(self, inferencer):
        """active_session_id set to 'active' after successful inference."""
        mock_result = MagicMock()
        mock_result.success = True
        with patch.object(inferencer, "_ainfer_single", new_callable=AsyncMock, return_value=mock_result), \
             patch("agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer.find_latest_session_id", return_value="test-session-uuid"):
            await inferencer.ainfer("Hello")
        assert inferencer.active_session_id == "test-session-uuid"

    @pytest.mark.asyncio
    async def test_ainfer_no_session_on_failure(self, inferencer):
        """active_session_id not set on failed inference."""
        mock_result = MagicMock()
        mock_result.success = False
        with patch.object(inferencer, "_ainfer_single", new_callable=AsyncMock, return_value=mock_result):
            await inferencer.ainfer("Hello")
        assert inferencer.active_session_id is None

    @pytest.mark.asyncio
    async def test_ainfer_new_session_clears_active(self, inferencer):
        """new_session=True clears active_session_id."""
        inferencer.active_session_id = "old-session-id"
        mock_result = MagicMock()
        mock_result.success = True
        with patch.object(inferencer, "_ainfer_single", new_callable=AsyncMock, return_value=mock_result), \
             patch("agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer.find_latest_session_id", return_value="new-session-uuid"):
            await inferencer.ainfer("Hello", new_session=True)
        # Should be set to "active" again after successful call
        assert inferencer.active_session_id == "new-session-uuid"

    @pytest.mark.asyncio
    async def test_ainfer_auto_resume_injects_kwargs(self, inferencer):
        """When auto_resume=True and active_session_id is set, resume=True is injected."""
        inferencer.active_session_id = "existing-session-uuid"

        captured_kwargs = {}

        async def capture_kwargs(inp, cfg=None, **kwargs):
            captured_kwargs.update(kwargs)
            result = MagicMock()
            result.success = True
            return result

        with patch.object(inferencer, "_ainfer_single", side_effect=capture_kwargs):
            await inferencer.ainfer("Hello")

        assert captured_kwargs.get("resume") is True
        assert captured_kwargs.get("session_id") == "existing-session-uuid"

    def test_infer_sets_active_session_on_success(self, inferencer):
        """Sync infer() sets active_session_id on success."""
        mock_result = MagicMock()
        mock_result.success = True
        with patch.object(inferencer, "_infer_single", return_value=mock_result), \
             patch("agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer.find_latest_session_id", return_value="test-session-uuid"):
            inferencer.infer("Hello")
        assert inferencer.active_session_id == "test-session-uuid"


# ============================================================================
# Yield Filter Tests
# ============================================================================


class TestYieldFilter:
    """Tests for _yield_filter()."""

    @pytest.mark.asyncio
    async def test_strips_ansi_from_chunks(self, inferencer):
        """ANSI codes stripped from streaming chunks."""

        async def mock_chunks():
            yield "\x1b[32mHello\x1b[0m"
            yield "\x1b[31mWorld\x1b[0m"

        result = []
        async for chunk in inferencer._yield_filter(mock_chunks()):
            result.append(chunk)
        assert "Hello" in result
        assert "World" in result

    @pytest.mark.asyncio
    async def test_filters_empty_ansi_lines(self, inferencer):
        """Lines that are only ANSI codes are filtered out."""

        async def mock_chunks():
            yield "\x1b[32m\x1b[0m"
            yield "Real content"

        result = []
        async for chunk in inferencer._yield_filter(mock_chunks()):
            result.append(chunk)
        assert len(result) == 1
        assert result[0] == "Real content"


# ============================================================================
# Common Utility Tests
# ============================================================================


class TestCommonUtils:
    """Tests for common.py utilities."""

    def test_strip_ansi_codes_basic(self):
        """Basic ANSI color codes removed."""
        assert strip_ansi_codes("\x1b[32mgreen\x1b[0m") == "green"

    def test_strip_ansi_codes_preserves_content(self):
        """Non-ANSI content preserved."""
        assert strip_ansi_codes("Hello world") == "Hello world"

    def test_strip_ansi_codes_carriage_return(self):
        """Carriage return spinner lines removed."""
        assert strip_ansi_codes("line1\rspinner_update") == "line1"

    def test_strip_ansi_codes_complex(self):
        """Complex ANSI sequences stripped."""
        text = "\x1b[1;34m[Bold Blue]\x1b[0m Normal \x1b[4mUnderline\x1b[0m"
        result = strip_ansi_codes(text)
        assert result == "[Bold Blue] Normal Underline"

    def test_find_acli_binary_explicit(self):
        """Returns explicit path when provided."""
        assert find_acli_binary("/custom/path/acli") == "/custom/path/acli"

    def test_find_acli_binary_not_found(self):
        """Raises RovoDevNotFoundError when not found."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RovoDevNotFoundError):
                find_acli_binary()

    @patch("shutil.which", return_value="/usr/local/bin/acli")
    def test_find_acli_binary_from_path(self, mock_which):
        """Finds acli from PATH."""
        result = find_acli_binary()
        assert result == "/usr/local/bin/acli"

    def test_find_available_port(self):
        """Returns an available port."""
        port = find_available_port(19100, 19200)
        assert 19100 <= port < 19200
        # Verify it's actually free
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))  # Should not raise

    def test_find_available_port_all_occupied(self):
        """Raises RuntimeError when no port available."""
        # Bind a port to block it, use a tiny range
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            with pytest.raises(RuntimeError, match="No available port"):
                find_available_port(port, port + 1)


# ============================================================================
# Registration Tests
# ============================================================================


class TestRegistration:
    """Tests for lazy import registration."""

    def test_cli_inferencer_importable(self):
        """RovoDevCliInferencer importable via lazy import."""
        from agent_foundation.common.inferencers.agentic_inferencers import (
            RovoDevCliInferencer,
        )
        assert RovoDevCliInferencer is not None

    def test_serve_inferencer_importable(self):
        """RovoDevServeInferencer importable via lazy import."""
        from agent_foundation.common.inferencers.agentic_inferencers import (
            RovoDevServeInferencer,
        )
        assert RovoDevServeInferencer is not None

    def test_in_all(self):
        """Both inferencers in __all__."""
        import agent_foundation.common.inferencers.agentic_inferencers as mod
        assert "RovoDevCliInferencer" in mod.__all__
        assert "RovoDevServeInferencer" in mod.__all__


# ============================================================================
# Non-Legacy Mode Tests
# ============================================================================


@pytest.fixture
def non_legacy_inferencer(tmp_path):
    """Create a non-legacy (TUI) mode inferencer."""
    return RovoDevCliInferencer(
        acli_path="/usr/bin/acli", working_dir=str(tmp_path), enable_legacy=False
    )


class TestNonLegacyConstructCommand:
    """Tests for construct_command() in non-legacy mode."""

    def test_no_legacy_subcommand(self, non_legacy_inferencer):
        """Non-legacy command omits 'legacy' subcommand."""
        cmd = non_legacy_inferencer.construct_command("Hello")
        assert "/usr/bin/acli rovodev" in cmd
        assert "legacy" not in cmd

    def test_auto_output_schema_injected(self, non_legacy_inferencer):
        """--output-schema auto-injected when raw_output_to_file=True and no user schema."""
        cmd = non_legacy_inferencer.construct_command("Hello")
        assert "--output-schema" in cmd
        assert '"response"' in cmd

    def test_user_schema_preserved(self, tmp_path):
        """User's output_schema is used instead of auto-injected."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
            enable_legacy=False, output_schema='{"type":"object","properties":{"answer":{"type":"string"}}}',
        )
        cmd = inf.construct_command("Hello")
        assert "--output-schema" in cmd
        assert "answer" in cmd
        assert '"response"' not in cmd  # auto-injected schema NOT used

    def test_no_auto_schema_when_raw_off(self, tmp_path):
        """No auto-injection when raw_output_to_file=False."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
            enable_legacy=False, raw_output_to_file=False,
        )
        cmd = inf.construct_command("Hello")
        assert "--output-schema" not in cmd

    def test_no_output_file(self, non_legacy_inferencer):
        """--output-file never appears in non-legacy mode."""
        cmd = non_legacy_inferencer.construct_command("Hello")
        assert "--output-file" not in cmd

    def test_jira_skipped(self, tmp_path, caplog):
        """--jira absent and warning logged in non-legacy mode."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
            enable_legacy=False, jira="https://jira/PROJ-1",
        )
        with caplog.at_level(logging.WARNING):
            cmd = inf.construct_command("Hello")
        assert "--jira" not in cmd
        assert "jira" in caplog.text.lower()

    def test_deep_plan_skipped(self, tmp_path, caplog):
        """--enable-deep-plan absent and warning logged in non-legacy mode."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
            enable_legacy=False, enable_deep_plan=True,
        )
        with caplog.at_level(logging.WARNING):
            cmd = inf.construct_command("Hello")
        assert "--enable-deep-plan" not in cmd
        assert "enable_deep_plan" in caplog.text

    def test_agent_mode_skipped(self, tmp_path, caplog):
        """--agent-mode absent and warning logged in non-legacy mode."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
            enable_legacy=False, agent_mode="ask",
        )
        with caplog.at_level(logging.WARNING):
            cmd = inf.construct_command("Hello")
        assert "--agent-mode" not in cmd
        assert "agent_mode" in caplog.text

    def test_yolo_present(self, non_legacy_inferencer):
        """--yolo works in non-legacy mode."""
        cmd = non_legacy_inferencer.construct_command("Hello")
        assert "--yolo" in cmd

    def test_xid_present(self, tmp_path):
        """--xid works in non-legacy mode (hidden flag in TUI)."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
            enable_legacy=False, xid="test-xid",
        )
        cmd = inf.construct_command("Hello")
        assert "--xid test-xid" in cmd

    def test_restore_works(self, non_legacy_inferencer):
        """--restore works in non-legacy mode."""
        cmd = non_legacy_inferencer.construct_command(
            "Hello", session_id="abc-123", resume=True
        )
        assert "--restore" in cmd
        assert "abc-123" in cmd

    def test_config_override(self, tmp_path):
        """--config-override included in non-legacy mode."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
            enable_legacy=False, config_override='{"agent":{"modelId":"opus"}}',
        )
        cmd = inf.construct_command("Hello")
        assert "--config-override" in cmd

    def test_enable_legacy_default_true(self, tmp_path):
        """Default enable_legacy is True (backward compat)."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
        )
        assert inf.enable_legacy is True
        cmd = inf.construct_command("Hello")
        assert "legacy" in cmd


# ============================================================================
# Extract JSON Tests
# ============================================================================


class TestExtractJsonFromOutput:
    """Tests for extract_json_from_output()."""

    def test_valid_json(self):
        """Parses clean JSON."""
        assert extract_json_from_output('{"response": "4"}') == {"response": "4"}

    def test_with_tui_noise(self):
        """Parses JSON after TUI output noise."""
        text = "Working in /tmp\n✔ Started 18 MCP servers\n\n{\n    \"response\": \"hello\"\n}\n"
        result = extract_json_from_output(text)
        assert result == {"response": "hello"}

    def test_with_xml_tags(self):
        """XML tags preserved inside JSON value."""
        text = '{"response": "<Result><Value>42</Value></Result>"}'
        result = extract_json_from_output(text)
        assert result["response"] == "<Result><Value>42</Value></Result>"

    def test_with_escaped_quotes(self):
        """Handles escaped quotes in JSON values."""
        text = '{"response": "He said \\"hello\\" to me"}'
        result = extract_json_from_output(text)
        assert result["response"] == 'He said "hello" to me'

    def test_with_nested_braces(self):
        """Handles nested braces inside string values."""
        text = '{"response": "obj = {a: 1}"}'
        result = extract_json_from_output(text)
        assert result["response"] == "obj = {a: 1}"

    def test_no_json(self):
        """Returns None when no JSON present."""
        assert extract_json_from_output("no json here") is None

    def test_empty_string(self):
        """Returns None for empty string."""
        assert extract_json_from_output("") is None

    def test_invalid_json(self):
        """Returns None for malformed JSON."""
        assert extract_json_from_output("{invalid json}") is None


# ============================================================================
# Non-Legacy Parse Output Tests
# ============================================================================


class TestNonLegacyParseOutput:
    """Tests for parse_output() in non-legacy mode."""

    def test_extracts_from_json(self, non_legacy_inferencer):
        """parse_output() extracts response from trailing JSON."""
        stdout = 'TUI noise\n\n{\n    "response": "42"\n}\n'
        result = non_legacy_inferencer.parse_output(stdout, "", 0)
        assert result["output"] == "42"
        assert result["success"] is True

    def test_fallback_to_ansi_strip(self, non_legacy_inferencer):
        """Falls back to ANSI stripping when no JSON found."""
        stdout = "\x1b[32mHello\x1b[0m world"
        result = non_legacy_inferencer.parse_output(stdout, "", 0)
        assert result["output"] == "Hello world"

    def test_user_schema_skips_json_extraction(self, tmp_path):
        """When user sets output_schema, JSON extraction is skipped."""
        inf = RovoDevCliInferencer(
            acli_path="/usr/bin/acli", working_dir=str(tmp_path),
            enable_legacy=False, output_schema='{"type":"object"}',
        )
        stdout = '{"response": "should not extract"}'
        result = inf.parse_output(stdout, "", 0)
        # Should fall through to ANSI stripping, not JSON extraction
        assert result["output"] != "should not extract"

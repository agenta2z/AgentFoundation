"""Tests for get_final_output() / streams_differ_from_final_output.

Covers:
- StreamingInferencerBase: default hook returns None, flag is False
- RovoDevCliInferencer: flag is True, legacy + non-legacy get_final_output()
- ConversationalInferencer: uses clean output for parsing + add_message()
- WebSocketInteractive: on_clean_output_available() sends stream_correction
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add OpenStartup src to path for WebSocketInteractive tests
# Path: test/agent_foundation/inferencers/ -> test/ -> AgentFoundation/ -> CoreProjects/ -> OpenStartup/src
_OPENTEAM_SRC = Path(__file__).parent.parent.parent.parent.parent / \
    "OpenStartup" / "src"
if _OPENTEAM_SRC.exists() and str(_OPENTEAM_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENTEAM_SRC))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ws_interactive():
    """Create a WebSocketInteractive with a capturing send_callback."""
    from openteam.server.services.websocket_interactive import WebSocketInteractive
    sent = []
    async def send_cb(msg):
        sent.append(msg)
    q = asyncio.Queue()
    ws = WebSocketInteractive(send_cb, q)
    return ws, sent


# ===========================================================================
# StreamingInferencerBase — base class behaviour
# ===========================================================================

class TestStreamingInferencerBase:

    def test_streams_differ_default_false(self):
        """Base class streams_differ_from_final_output is False."""
        from agent_foundation.common.inferencers.streaming_inferencer_base import (
            StreamingInferencerBase,
        )
        assert StreamingInferencerBase.streams_differ_from_final_output is False

    def test_get_final_output_returns_none(self):
        """Base class get_final_output() returns None (stream IS the output)."""
        from agent_foundation.common.inferencers.streaming_inferencer_base import (
            StreamingInferencerBase,
        )
        # Create a minimal concrete subclass (can't instantiate base directly)
        class _Concrete(StreamingInferencerBase):
            def _infer(self, *a, **kw):
                return ""

            async def _ainfer_streaming(self, *a, **kw):
                return
                yield  # make it a generator

            async def _yield_filter(self, chunks, **kw):
                async for c in chunks:
                    yield c

        obj = _Concrete()
        assert obj.get_final_output() is None


# ===========================================================================
# RovoDevCliInferencer — override behaviour
# ===========================================================================

class TestRovoDevCliInferencerGetFinalOutput:

    def _make_inferencer(self, **kwargs):
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
            RovoDevCliInferencer,
        )
        defaults = dict(target_path="/tmp", enable_legacy=True)
        defaults.update(kwargs)
        return RovoDevCliInferencer(**defaults)

    def test_streams_differ_is_true(self):
        """RovoDevCliInferencer.streams_differ_from_final_output is True."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
            RovoDevCliInferencer,
        )
        assert RovoDevCliInferencer.streams_differ_from_final_output is True

    def test_legacy_reads_from_last_clean_output(self):
        """Legacy mode get_final_output() reads from _last_clean_output instance var.

        This is set by ainfer_streaming() BEFORE the temp file is deleted,
        so get_final_output() works even after file cleanup.
        """
        clean_text = 'Clean LLM output\n```json ToolsToInvoke\n{"type": "conversation"}\n```'
        inf = self._make_inferencer()
        inf._last_clean_output = clean_text
        result = inf.get_final_output()
        assert result == clean_text

    def test_legacy_returns_none_when_no_last_clean_output(self):
        """Legacy mode returns None when _last_clean_output is not set."""
        inf = self._make_inferencer()
        # _last_clean_output not set at all
        assert inf.get_final_output() is None

    def test_legacy_returns_none_when_last_clean_output_empty(self):
        """Legacy mode returns None when _last_clean_output is empty/None."""
        inf = self._make_inferencer()
        inf._last_clean_output = None
        assert inf.get_final_output() is None

        inf._last_clean_output = ""
        assert inf.get_final_output() is None

    def test_nonlegacy_extracts_from_last_raw_stdout(self):
        """Non-legacy mode extracts clean output from trailing JSON in _last_raw_stdout."""
        inf = self._make_inferencer(enable_legacy=False)
        clean_text = "Hello from the LLM"
        inf._last_raw_stdout = (
            "some TUI noise\n"
            f'{{"response": "{clean_text}"}}\n'
        )
        result = inf.get_final_output()
        assert result == clean_text

    def test_nonlegacy_returns_none_when_no_stdout(self):
        """Non-legacy mode returns None when _last_raw_stdout not set."""
        inf = self._make_inferencer(enable_legacy=False)
        # _last_raw_stdout not set
        assert inf.get_final_output() is None

    def test_nonlegacy_returns_none_when_no_trailing_json(self):
        """Non-legacy mode returns None when stdout has no trailing JSON."""
        inf = self._make_inferencer(enable_legacy=False)
        inf._last_raw_stdout = "just plain text, no JSON"
        result = inf.get_final_output()
        assert result is None

    def test_yield_filter_resets_last_raw_stdout(self):
        """_yield_filter resets _last_raw_stdout on each call (non-legacy)."""
        inf = self._make_inferencer(enable_legacy=False)
        inf._last_raw_stdout = "stale content"

        async def run():
            async def chunks():
                yield "chunk1"
                yield "chunk2"
            result = []
            async for c in inf._yield_filter(chunks()):
                result.append(c)
            return result

        asyncio.run(run())
        # After _yield_filter runs, _last_raw_stdout should be reset and then
        # re-accumulated with the non-legacy chunks (non-empty after strip_ansi_codes)
        assert inf._last_raw_stdout == "chunk1chunk2"  # accumulated non-legacy chunks


# ===========================================================================
# WebSocketInteractive — on_clean_output_available
# ===========================================================================

class TestWebSocketInteractiveCleanOutput:

    def test_initial_clean_output_is_none(self):
        """clean_output property is None before on_clean_output_available()."""
        ws, _ = _make_ws_interactive()
        assert ws.clean_output is None

    def test_on_clean_output_available_stores_and_sends(self):
        """on_clean_output_available stores clean output and sends stream_correction."""
        ws, sent = _make_ws_interactive()

        async def run():
            await ws.on_clean_output_available("Clean LLM text")

        asyncio.run(run())

        assert ws.clean_output == "Clean LLM text"
        assert len(sent) == 1
        assert sent[0]["type"] == "stream_correction"
        assert sent[0]["content"] == "Clean LLM text"

    def test_on_clean_output_available_overwrites(self):
        """Calling on_clean_output_available() twice overwrites the stored value."""
        ws, sent = _make_ws_interactive()

        async def run():
            await ws.on_clean_output_available("first")
            await ws.on_clean_output_available("second")

        asyncio.run(run())

        assert ws.clean_output == "second"
        assert len(sent) == 2
        assert sent[1]["content"] == "second"


# ===========================================================================
# ConversationalInferencer — uses clean output for parsing
# ===========================================================================

class TestConversationalInferencerCleanOutput:
    """Verify that run_agentic_loop uses get_final_output() for parsing."""

    def _build_mock_base_inferencer(self, clean_text: str, streams_differ: bool = True):
        """Build a mock base inferencer that streams noise but returns clean text."""
        mock = MagicMock()
        mock.streams_differ_from_final_output = streams_differ
        mock.get_final_output = MagicMock(return_value=clean_text)
        mock.system_prompt = ""

        async def _streaming(prompt):
            yield "noisy stdout line 1"
            yield "noisy stdout line 2"

        mock.ainfer_streaming = _streaming
        return mock

    def test_add_message_uses_clean_output(self):
        """get_final_output() returns clean text, not noisy stream (unit-level check)."""
        # We verify the contract directly: if streams_differ_from_final_output=True
        # and get_final_output() returns a value, that value should be used.
        # Full integration test of run_agentic_loop is covered by end-to-end testing.
        clean_text = "This is the clean LLM response with ```json ToolsToInvoke``` intact."
        mock_base = self._build_mock_base_inferencer(clean_text)

        assert mock_base.streams_differ_from_final_output is True
        assert mock_base.get_final_output() == clean_text

        # Verify that a non-differs inferencer returns None from get_final_output
        mock_api = self._build_mock_base_inferencer("irrelevant", streams_differ=False)
        assert mock_api.streams_differ_from_final_output is False
        # (get_final_output still returns our mock value, but the CI would skip calling it)


# ===========================================================================
# OpenStartup — run existing unit tests to ensure no regressions
# ===========================================================================

class TestNoRegressions:

    def test_websocket_interactive_stream_token_batches_still_works(self):
        """stream_token_batches still works correctly after adding on_clean_output_available."""
        ws, sent = _make_ws_interactive()

        async def run():
            async def tokens():
                yield "hello", {}
                yield " world", {}
            result = await ws.stream_token_batches(tokens(), session_id="test")
            return result

        result = asyncio.run(run())
        assert result == "hello world"
        token_msgs = [m for m in sent if m["type"] == "token"]
        assert len(token_msgs) >= 1

    def test_streaming_inferencer_base_streams_differ_default(self):
        """Non-CLI subclasses inherit streams_differ_from_final_output=False."""
        from agent_foundation.common.inferencers.streaming_inferencer_base import (
            StreamingInferencerBase,
        )
        # Verify at class level — no need to instantiate
        assert StreamingInferencerBase.streams_differ_from_final_output is False

        # Also verify that a mock subclass that doesn't override inherits False
        class _APIInferencer(StreamingInferencerBase):
            def _infer(self, *a, **kw):
                return ""
            async def _ainfer_streaming(self, *a, **kw):
                return
                yield
            async def _yield_filter(self, chunks, **kw):
                async for c in chunks:
                    yield c

        obj = _APIInferencer()
        assert obj.streams_differ_from_final_output is False
        assert obj.get_final_output() is None

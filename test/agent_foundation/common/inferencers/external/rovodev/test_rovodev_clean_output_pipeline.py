"""Tests for RovoDevCliInferencer clean output pipeline.

Verifies the single-read architecture: _get_clean_output_for_cache() reads
--output-file once and records the content for dual use (cache overwrite +
response output field).  The _ainfer() override wraps the result in a
TerminalInferencerResponse so the logged InferenceResponse has the correct
clean output, not the noisy TUI transcript.

The clean output is a component of the call's invocation (``_CALL_OUTPUT``); the
``_last_clean_output`` field is its bare compat getter, written when a non-host
invocation closes.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
    RovoDevOutput,
)
from agent_foundation.common.inferencers.run_context import (
    aopen_invocation,
    open_invocation,
    publish_result,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_response import (
    TerminalInferencerResponse,
)


def _make_inferencer(tmp_path: Path) -> RovoDevCliInferencer:
    return RovoDevCliInferencer(
        acli_path="/usr/bin/acli",
        target_path=str(tmp_path),
    )


def _ainfer_with_clean_output(inf, clean, noisy):
    """``inf._ainfer`` inside an invocation whose clean output is ``clean``, the
    base ``_ainfer`` returning ``noisy``."""

    async def call():
        async with aopen_invocation(inf):
            publish_result(inf, inf._CALL_OUTPUT, RovoDevOutput(clean=clean))
            return await inf._ainfer("test input")

    with patch.object(
        RovoDevCliInferencer.__mro__[1],
        "_ainfer",
        new_callable=AsyncMock,
        return_value=noisy,
    ):
        return asyncio.run(call())


def _stream_through_pipeline(inf, base_pipeline):
    """Drain ``inf._ainfer_streaming_pipeline`` in an invocation, the base class
    pipeline replaced by ``base_pipeline(inf, kwargs)``; returns the call's
    recorded clean output."""

    async def fake(self, inference_input, inference_config=None, **kwargs):
        async for chunk in base_pipeline(self, kwargs):
            yield chunk

    async def call():
        async with aopen_invocation(inf):
            async for _ in inf._ainfer_streaming_pipeline("q"):
                pass
            return inf._call_output().clean

    with patch.object(
        RovoDevCliInferencer.__mro__[1], "_ainfer_streaming_pipeline", fake
    ):
        return asyncio.run(call())


# ---------------------------------------------------------------------------
# _get_clean_output_for_cache: single read, dual use
# ---------------------------------------------------------------------------


class TestGetCleanOutputForCacheSideEffect(unittest.TestCase):
    """_get_clean_output_for_cache reads --output-file and records it as the
    call's clean output as a side effect (``_last_clean_output`` once a bare
    invocation closes)."""

    def test_sets_last_clean_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            output_file = Path(tmp) / "output.md"
            output_file.write_text("<Response>\nClean LLM output\n</Response>")
            with open_invocation(inf) as frame:
                frame.put(inf._OUTPUT_FILE, str(output_file))
                result = inf._get_clean_output_for_cache()
                self.assertIsNotNone(result)
                self.assertIn("Clean LLM output", result)
                self.assertEqual(inf._call_output().clean, result)
            self.assertEqual(inf._last_clean_output, result)

    def test_return_value_matches_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            output_file = Path(tmp) / "output.md"
            output_file.write_text("The clean content")
            with open_invocation(inf) as frame:
                frame.put(inf._OUTPUT_FILE, str(output_file))
                returned = inf._get_clean_output_for_cache()
                stored = inf._call_output().clean
                self.assertEqual(returned, stored)

    def test_returns_none_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            with open_invocation(inf) as frame:
                frame.put(inf._OUTPUT_FILE, str(Path(tmp) / "nonexistent.md"))
                result = inf._get_clean_output_for_cache()
                self.assertIsNone(result)

    def test_returns_none_when_file_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            output_file = Path(tmp) / "output.md"
            output_file.write_text("   ")
            with open_invocation(inf) as frame:
                frame.put(inf._OUTPUT_FILE, str(output_file))
                result = inf._get_clean_output_for_cache()
                self.assertIsNone(result)

    def test_returns_none_for_non_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = RovoDevCliInferencer(
                acli_path="/usr/bin/acli",
                target_path=str(tmp),
                enable_legacy=False,
            )
            output_file = Path(tmp) / "output.md"
            output_file.write_text("Content")
            with open_invocation(inf) as frame:
                frame.put(inf._OUTPUT_FILE, str(output_file))
                result = inf._get_clean_output_for_cache()
                self.assertIsNone(result)


class TestCallOutputFile(unittest.TestCase):
    """The call's auto ``--output-file`` is a component of its invocation frame;
    without one (a direct hook call) the configured ``output_file`` applies."""

    def test_frame_component_wins_over_configured_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            inf.output_file = "configured.md"
            with open_invocation(inf) as frame:
                frame.put(inf._OUTPUT_FILE, "call.md")
                self.assertEqual(inf._call_output_file(), "call.md")
                frame.discard(inf._OUTPUT_FILE)
                self.assertEqual(inf._call_output_file(), "configured.md")

    def test_frameless_read_uses_configured_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            self.assertIsNone(inf._call_output_file())
            inf.output_file = "configured.md"
            self.assertEqual(inf._call_output_file(), "configured.md")


# ---------------------------------------------------------------------------
# _ainfer override: returns TerminalInferencerResponse when clean output exists
# ---------------------------------------------------------------------------


class TestAinferOverrideWrapsCleanOutput(unittest.TestCase):
    def test_returns_terminal_response_when_clean_output_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            clean = "<Response>\nClean review JSON\n</Response>"
            noisy = "Working in /tmp\nMCP errors...\nTool calls...\nClean review JSON"

            result = _ainfer_with_clean_output(inf, clean, noisy)

            self.assertIsInstance(result, TerminalInferencerResponse)
            self.assertEqual(result.output, clean)
            self.assertEqual(result.raw_output, noisy)
            self.assertTrue(result.success)

    def test_returns_raw_string_when_no_clean_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            noisy = "raw noisy output"

            result = _ainfer_with_clean_output(inf, None, noisy)

            self.assertIsInstance(result, str)
            self.assertEqual(result, noisy)

    def test_returns_raw_string_when_clean_output_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            noisy = "raw output"

            result = _ainfer_with_clean_output(inf, "", noisy)

            self.assertIsInstance(result, str)

    def test_ignores_a_previous_calls_clean_output(self):
        """B23: the instance's last clean output is a bare getter, never this
        call's: a call that recorded none returns its raw output."""
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            inf._last_clean_output = "previous call's output"

            async def call():
                async with aopen_invocation(inf):
                    return await inf._ainfer("test input")

            with patch.object(
                RovoDevCliInferencer.__mro__[1],
                "_ainfer",
                new_callable=AsyncMock,
                return_value="raw output",
            ):
                result = asyncio.run(call())

            self.assertEqual(result, "raw output")


# ---------------------------------------------------------------------------
# ainfer: isinstance handling of TerminalInferencerResponse
# ---------------------------------------------------------------------------


class TestAinferAcceptsWrappedResponse(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_passes_through_terminal_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            expected = TerminalInferencerResponse(
                output="clean", raw_output="noisy", success=True
            )

            with (
                patch.object(
                    inf, "_ainfer", new_callable=AsyncMock, return_value=expected
                ),
                patch(
                    "agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer.find_latest_session_id",
                    return_value=None,
                ),
            ):
                result = self._run(inf.ainfer("test"))

            self.assertIsInstance(result, TerminalInferencerResponse)
            self.assertEqual(result.output, "clean")
            self.assertEqual(result.raw_output, "noisy")

    def test_wraps_plain_string_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))

            async def ainfer(*args, **kwargs):
                inf._record_output(clean="clean from get_final_output")
                return "raw string"

            with (
                patch.object(inf, "_ainfer", side_effect=ainfer),
                patch(
                    "agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer.find_latest_session_id",
                    return_value=None,
                ),
            ):
                result = self._run(inf.ainfer("test"))

            self.assertIsInstance(result, TerminalInferencerResponse)
            self.assertEqual(result.output, "clean from get_final_output")
            self.assertEqual(result.raw_output, "raw string")


# ---------------------------------------------------------------------------
# Single-read verification: subclass finally skips read when already set
# ---------------------------------------------------------------------------


class TestSingleReadArchitecture(unittest.TestCase):
    def test_subclass_finally_skips_read_when_already_set(self):
        """When _get_clean_output_for_cache already recorded the clean output,
        the subclass finally should NOT re-read the file."""

        async def base_pipeline(inf, kwargs):
            output_file = Path(kwargs["output_file"])
            output_file.write_text("Clean content from output file")
            try:
                yield "noisy"
            finally:
                inf._get_clean_output_for_cache()
                output_file.write_text("MODIFIED — should NOT be re-read")

        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            clean = _stream_through_pipeline(inf, base_pipeline)
            self.assertEqual(clean, "Clean content from output file")

    def test_defensive_fallback_reads_when_cache_skipped(self):
        """When _get_clean_output_for_cache was NOT called (e.g., no cache file),
        the subclass finally should read the file as a fallback."""

        async def base_pipeline(inf, kwargs):
            Path(kwargs["output_file"]).write_text("Fallback content")
            yield "noisy"

        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            clean = _stream_through_pipeline(inf, base_pipeline)
            self.assertEqual(clean, "Fallback content")


# ---------------------------------------------------------------------------
# Response field correctness
# ---------------------------------------------------------------------------


class TestResponseFieldCorrectness(unittest.TestCase):
    def test_output_field_is_clean_not_noisy(self):
        """The output field of TerminalInferencerResponse should contain
        the clean --output-file content, not the noisy TUI transcript."""
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            clean = '<Response>\n```json\n{"approve": true}\n```\n</Response>'
            noisy = "Working in /tmp\n[MCP] errors\n" + clean + "\nSession: 46K/1M"

            result = _ainfer_with_clean_output(inf, clean, noisy)

            self.assertIsInstance(result, TerminalInferencerResponse)
            self.assertEqual(result.output, clean)
            self.assertIn("MCP", result.raw_output)
            self.assertIn("<Response>", result.output)
            self.assertNotEqual(result.output, result.raw_output)

    def test_str_of_response_returns_clean_output(self):
        """str(TerminalInferencerResponse) returns output, which downstream
        consumers like DualInferencer._default_parse_review use."""
        resp = TerminalInferencerResponse(
            output="clean content",
            raw_output="noisy content",
        )
        self.assertEqual(str(resp), "clean content")


if __name__ == "__main__":
    unittest.main()

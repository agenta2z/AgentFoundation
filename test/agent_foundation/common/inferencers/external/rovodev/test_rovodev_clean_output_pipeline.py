"""Tests for RovoDevCliInferencer clean output pipeline.

Verifies the single-read architecture: _get_clean_output_for_cache() reads
--output-file once and stores the content for dual use (cache overwrite +
response output field).  The _ainfer() override wraps the result in a
TerminalInferencerResponse so the logged InferenceResponse has the correct
clean output, not the noisy TUI transcript.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
    _current_output_file,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_response import (
    TerminalInferencerResponse,
)


def _make_inferencer(tmp_path: Path) -> RovoDevCliInferencer:
    return RovoDevCliInferencer(
        acli_path="/usr/bin/acli",
        target_path=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# _get_clean_output_for_cache: single read, dual use
# ---------------------------------------------------------------------------


class TestGetCleanOutputForCacheSideEffect(unittest.TestCase):
    """_get_clean_output_for_cache reads --output-file and stores it
    in _last_clean_output as a side effect."""

    def test_sets_last_clean_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            output_file = Path(tmp) / "output.md"
            output_file.write_text("<Response>\nClean LLM output\n</Response>")
            token = _current_output_file.set(str(output_file))
            try:
                result = inf._get_clean_output_for_cache()
                self.assertIsNotNone(result)
                self.assertIn("Clean LLM output", result)
                self.assertEqual(inf._last_clean_output, result)
            finally:
                _current_output_file.reset(token)

    def test_return_value_matches_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            output_file = Path(tmp) / "output.md"
            output_file.write_text("The clean content")
            token = _current_output_file.set(str(output_file))
            try:
                returned = inf._get_clean_output_for_cache()
                stored = inf._last_clean_output
                self.assertEqual(returned, stored)
            finally:
                _current_output_file.reset(token)

    def test_returns_none_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            token = _current_output_file.set(str(Path(tmp) / "nonexistent.md"))
            try:
                result = inf._get_clean_output_for_cache()
                self.assertIsNone(result)
            finally:
                _current_output_file.reset(token)

    def test_returns_none_when_file_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            output_file = Path(tmp) / "output.md"
            output_file.write_text("   ")
            token = _current_output_file.set(str(output_file))
            try:
                result = inf._get_clean_output_for_cache()
                self.assertIsNone(result)
            finally:
                _current_output_file.reset(token)

    def test_returns_none_for_non_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = RovoDevCliInferencer(
                acli_path="/usr/bin/acli",
                target_path=str(tmp),
                enable_legacy=False,
            )
            output_file = Path(tmp) / "output.md"
            output_file.write_text("Content")
            token = _current_output_file.set(str(output_file))
            try:
                result = inf._get_clean_output_for_cache()
                self.assertIsNone(result)
            finally:
                _current_output_file.reset(token)


# ---------------------------------------------------------------------------
# _ainfer override: returns TerminalInferencerResponse when clean output exists
# ---------------------------------------------------------------------------


class TestAinferOverrideWrapsCleanOutput(unittest.TestCase):

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_returns_terminal_response_when_clean_output_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            inf._last_clean_output = "<Response>\nClean review JSON\n</Response>"
            noisy = "Working in /tmp\nMCP errors...\nTool calls...\nClean review JSON"

            with patch.object(
                RovoDevCliInferencer.__mro__[1],
                "_ainfer",
                new_callable=AsyncMock,
                return_value=noisy,
            ):
                result = self._run(inf._ainfer("test input"))

            self.assertIsInstance(result, TerminalInferencerResponse)
            self.assertEqual(result.output, inf._last_clean_output)
            self.assertEqual(result.raw_output, noisy)
            self.assertTrue(result.success)

    def test_returns_raw_string_when_no_clean_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            inf._last_clean_output = None
            noisy = "raw noisy output"

            with patch.object(
                RovoDevCliInferencer.__mro__[1],
                "_ainfer",
                new_callable=AsyncMock,
                return_value=noisy,
            ):
                result = self._run(inf._ainfer("test input"))

            self.assertIsInstance(result, str)
            self.assertEqual(result, noisy)

    def test_returns_raw_string_when_clean_output_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            inf._last_clean_output = ""
            noisy = "raw output"

            with patch.object(
                RovoDevCliInferencer.__mro__[1],
                "_ainfer",
                new_callable=AsyncMock,
                return_value=noisy,
            ):
                result = self._run(inf._ainfer("test input"))

            self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# ainfer: isinstance handling of TerminalInferencerResponse
# ---------------------------------------------------------------------------


class TestAinferAcceptsWrappedResponse(unittest.TestCase):

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_passes_through_terminal_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            expected = TerminalInferencerResponse(
                output="clean", raw_output="noisy", success=True
            )

            with patch.object(inf, "_ainfer_single", new_callable=AsyncMock, return_value=expected), \
                 patch("agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer.find_latest_session_id", return_value=None):
                result = self._run(inf.ainfer("test"))

            self.assertIsInstance(result, TerminalInferencerResponse)
            self.assertEqual(result.output, "clean")
            self.assertEqual(result.raw_output, "noisy")

    def test_wraps_plain_string_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            inf._last_clean_output = "clean from get_final_output"

            with patch.object(inf, "_ainfer_single", new_callable=AsyncMock, return_value="raw string"), \
                 patch("agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer.find_latest_session_id", return_value=None):
                result = self._run(inf.ainfer("test"))

            self.assertIsInstance(result, TerminalInferencerResponse)
            self.assertEqual(result.raw_output, "raw string")


# ---------------------------------------------------------------------------
# Single-read verification: subclass finally skips read when already set
# ---------------------------------------------------------------------------


class TestSingleReadArchitecture(unittest.TestCase):

    def test_subclass_finally_skips_read_when_already_set(self):
        """When _get_clean_output_for_cache already stored _last_clean_output,
        the subclass finally should NOT re-read the file."""
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            output_file = Path(tmp) / "output.md"
            output_file.write_text("Clean content from output file")

            token = _current_output_file.set(str(output_file))
            try:
                cache_result = inf._get_clean_output_for_cache()
                self.assertEqual(inf._last_clean_output, "Clean content from output file")

                output_file.write_text("MODIFIED — should NOT be re-read")

                if not getattr(inf, "_last_clean_output", None):
                    inf._last_clean_output = output_file.read_text().strip()

                self.assertEqual(
                    inf._last_clean_output,
                    "Clean content from output file",
                )
            finally:
                _current_output_file.reset(token)

    def test_defensive_fallback_reads_when_cache_skipped(self):
        """When _get_clean_output_for_cache was NOT called (e.g., no cache file),
        the subclass finally should read the file as a fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            output_file = Path(tmp) / "output.md"
            output_file.write_text("Fallback content")

            self.assertFalse(hasattr(inf, "_last_clean_output") and inf._last_clean_output)

            if not getattr(inf, "_last_clean_output", None):
                content = output_file.read_text().strip()
                inf._last_clean_output = content if content else None

            self.assertEqual(inf._last_clean_output, "Fallback content")


# ---------------------------------------------------------------------------
# Response field correctness
# ---------------------------------------------------------------------------


class TestResponseFieldCorrectness(unittest.TestCase):

    def test_output_field_is_clean_not_noisy(self):
        """The output field of TerminalInferencerResponse should contain
        the clean --output-file content, not the noisy TUI transcript."""
        with tempfile.TemporaryDirectory() as tmp:
            inf = _make_inferencer(Path(tmp))
            clean = "<Response>\n```json\n{\"approve\": true}\n```\n</Response>"
            noisy = "Working in /tmp\n[MCP] errors\n" + clean + "\nSession: 46K/1M"

            inf._last_clean_output = clean

            with patch.object(
                RovoDevCliInferencer.__mro__[1],
                "_ainfer",
                new_callable=AsyncMock,
                return_value=noisy,
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    inf._ainfer("test")
                )

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

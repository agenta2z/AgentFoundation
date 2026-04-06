

"""Unit tests for the dual-timer (tool_use_idle_timeout_seconds) feature
in StreamingInferencerBase.ainfer_streaming().

Tests:
1. Empty-string sentinels switch the timer to tool_use_idle_timeout_seconds.
2. Non-empty text chunks switch back to idle_timeout_seconds.
3. tool_use_idle_timeout_seconds=0 preserves the old single-timer behavior.
4. Empty sentinels are NOT passed downstream (not cached, not accumulated).
"""

import asyncio
import unittest
from typing import Any, AsyncIterator, Optional
from unittest.mock import patch

from attr import attrib, attrs
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)


@attrs
class _FakeStreamingInferencer(StreamingInferencerBase):
    """Minimal concrete subclass for testing.

    _chunks is a list of (chunk, delay) tuples that _ainfer_streaming yields.
    If delay > 0, it sleeps before yielding.
    """

    chunks: list = attrib(factory=list)

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        raise NotImplementedError("Not used in streaming tests")

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        for chunk, delay in self.chunks:
            if delay > 0:
                await asyncio.sleep(delay)
            yield chunk


class TestDualTimerBehavior(unittest.TestCase):
    """Tests for the dual-timer mechanism in ainfer_streaming()."""

    def _collect_chunks(
        self, inferencer: _FakeStreamingInferencer, prompt: str = "test"
    ) -> list[str]:
        """Helper: collect all chunks from ainfer_streaming synchronously."""

        async def _run():
            chunks = []
            async for chunk in inferencer.ainfer_streaming(prompt):
                chunks.append(chunk)
            return chunks

        return asyncio.run(_run())

    def test_empty_sentinels_not_yielded_downstream(self):
        """Empty-string sentinels from _ainfer_streaming must NOT appear in output."""
        inferencer = _FakeStreamingInferencer(
            chunks=[
                ("hello", 0),
                ("", 0),  # sentinel
                ("", 0),  # sentinel
                (" world", 0),
                ("", 0),  # sentinel
            ],
            idle_timeout_seconds=10,
            tool_use_idle_timeout_seconds=60,
        )
        result = self._collect_chunks(inferencer)
        self.assertEqual(result, ["hello", " world"])

    def test_sentinels_not_cached(self):
        """Empty sentinels must not be written to cache files."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            inferencer = _FakeStreamingInferencer(
                chunks=[
                    ("line1", 0),
                    ("", 0),  # sentinel
                    ("line2", 0),
                ],
                idle_timeout_seconds=10,
                tool_use_idle_timeout_seconds=60,
                cache_folder=tmpdir,
            )
            self._collect_chunks(inferencer)

            # Find the cache file
            cache_files = []
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".txt"):
                        cache_files.append(os.path.join(root, f))

            self.assertEqual(len(cache_files), 1)
            content = open(cache_files[0]).read()
            # Cache should contain "line1line2" plus the success marker, no empty strings
            self.assertIn("line1line2", content)
            self.assertIn("STREAM COMPLETED SUCCESSFULLY", content)

    def test_tool_use_timeout_zero_preserves_single_timer(self):
        """When tool_use_idle_timeout_seconds=0, empty sentinels are still filtered
        but the timeout stays at idle_timeout_seconds (single-timer behavior)."""
        inferencer = _FakeStreamingInferencer(
            chunks=[
                ("text", 0),
                ("", 0),  # sentinel - should NOT switch timer since tool_use=0
                ("more", 0),
            ],
            idle_timeout_seconds=10,
            tool_use_idle_timeout_seconds=0,  # disabled
        )
        result = self._collect_chunks(inferencer)
        self.assertEqual(result, ["text", "more"])

    def test_sentinel_switches_to_tool_use_timeout(self):
        """After receiving an empty sentinel, the timer should switch to
        tool_use_idle_timeout_seconds. A delay within that window should succeed."""
        inferencer = _FakeStreamingInferencer(
            chunks=[
                ("start", 0),
                ("", 0),  # switch to tool_use timeout (5s)
                # Sleep 2s — within 5s tool_use timeout, should succeed
                ("end", 2),
            ],
            idle_timeout_seconds=1,  # 1s text timeout
            tool_use_idle_timeout_seconds=5,
        )
        result = self._collect_chunks(inferencer)
        self.assertEqual(result, ["start", "end"])

    def test_text_after_sentinel_switches_back_to_idle_timeout(self):
        """After receiving text following a sentinel, the timer should switch
        back to idle_timeout_seconds. A delay exceeding that should timeout."""
        inferencer = _FakeStreamingInferencer(
            chunks=[
                ("start", 0),
                ("", 0),  # switch to tool_use timeout
                ("mid", 0),  # switch back to idle timeout (1s)
                # Sleep 3s — exceeds 1s idle timeout
                ("late", 3),
            ],
            idle_timeout_seconds=1,
            tool_use_idle_timeout_seconds=10,
        )
        with self.assertRaises(asyncio.TimeoutError):
            self._collect_chunks(inferencer)

    def test_idle_timeout_fires_during_text_phase(self):
        """Without any sentinels, a gap exceeding idle_timeout_seconds should
        raise TimeoutError (original behavior)."""
        inferencer = _FakeStreamingInferencer(
            chunks=[
                ("first", 0),
                ("second", 3),  # 3s gap > 1s idle timeout
            ],
            idle_timeout_seconds=1,
            tool_use_idle_timeout_seconds=60,
        )
        with self.assertRaises(asyncio.TimeoutError):
            self._collect_chunks(inferencer)

    def test_per_call_tool_use_timeout_override(self):
        """Per-call tool_use_idle_timeout_seconds kwarg should override instance."""
        inferencer = _FakeStreamingInferencer(
            chunks=[
                ("", 0),  # sentinel
                # 2s delay — would timeout with instance's 1s idle, but
                # per-call tool_use override of 5s should keep it alive
                ("ok", 2),
            ],
            idle_timeout_seconds=1,
            tool_use_idle_timeout_seconds=1,  # instance: too short
        )

        async def _run():
            chunks = []
            async for chunk in inferencer.ainfer_streaming(
                "test",
                tool_use_idle_timeout_seconds=5,  # override: long enough
            ):
                chunks.append(chunk)
            return chunks

        result = asyncio.run(_run())
        self.assertEqual(result, ["ok"])

    def test_idle_timeout_disabled_with_zero(self):
        """idle_timeout_seconds=0 should disable the timeout entirely."""
        inferencer = _FakeStreamingInferencer(
            chunks=[
                ("a", 0),
                ("b", 0.5),
            ],
            idle_timeout_seconds=0,
            tool_use_idle_timeout_seconds=0,
        )
        result = self._collect_chunks(inferencer)
        self.assertEqual(result, ["a", "b"])

    def test_all_sentinels_produces_empty_output(self):
        """If _ainfer_streaming yields only empty sentinels, output should be empty."""
        inferencer = _FakeStreamingInferencer(
            chunks=[
                ("", 0),
                ("", 0),
                ("", 0),
            ],
            idle_timeout_seconds=10,
            tool_use_idle_timeout_seconds=60,
        )
        result = self._collect_chunks(inferencer)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

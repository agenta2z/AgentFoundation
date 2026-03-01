"""Tests for InferencerBase async methods."""

import asyncio
import unittest
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)


@attrs
class MockInferencer(InferencerBase):
    """Concrete inferencer for testing the base class methods."""

    mock_response: str = attrib(default="mock_response")
    call_count: int = attrib(default=0, init=False)

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        """Simple sync implementation that returns mock_response."""
        self.call_count += 1
        return self.mock_response


@attrs
class AsyncMockInferencer(InferencerBase):
    """Async-native inferencer for testing async methods."""

    mock_response: str = attrib(default="async_mock_response")
    call_count: int = attrib(default=0, init=False)
    delay_seconds: float = attrib(default=0.0)

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        """Sync fallback - should not be used directly in async tests."""
        self.call_count += 1
        return f"sync_{self.mock_response}"

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        """Async implementation with optional delay."""
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        self.call_count += 1
        return self.mock_response


class InferencerBaseAsyncTest(unittest.IsolatedAsyncioTestCase):
    """Test suite for InferencerBase async methods."""

    async def test_ainfer_single_basic(self):
        """Test basic async inference with _ainfer_single."""
        inferencer = AsyncMockInferencer(mock_response="hello")
        result = await inferencer._ainfer_single("test input")
        self.assertEqual(result, "hello")
        self.assertEqual(inferencer.call_count, 1)

    async def test_ainfer_calls_async_implementation(self):
        """Verify ainfer() uses _ainfer() when available."""
        inferencer = AsyncMockInferencer(mock_response="async_result")
        result = await inferencer.ainfer("test input")

        self.assertEqual(result, "async_result")
        self.assertEqual(inferencer.call_count, 1)

    async def test_ainfer_default_wraps_sync(self):
        """Default _ainfer wraps sync _infer for backwards compatibility."""
        inferencer = MockInferencer(mock_response="sync_wrapped")
        result = await inferencer.ainfer("test input")

        self.assertEqual(result, "sync_wrapped")
        self.assertEqual(inferencer.call_count, 1)

    async def test_ainfer_with_preprocessor(self):
        """Test that input_preprocessor is applied in async path."""

        def uppercase_preprocessor(inp):
            return inp.upper()

        inferencer = AsyncMockInferencer(
            mock_response="processed",
            input_preprocessor=uppercase_preprocessor,
        )
        result = await inferencer.ainfer("test")
        self.assertEqual(result, "processed")

    async def test_ainfer_with_postprocessor(self):
        """Test that response_post_processor is applied in async path."""

        def add_suffix(response):
            return f"{response}_processed"

        inferencer = AsyncMockInferencer(
            mock_response="hello",
            response_post_processor=add_suffix,
        )
        result = await inferencer.ainfer("test")
        self.assertEqual(result, "hello_processed")

    async def test_ainfer_iterator_collects_results(self):
        """Test async inference with iterator input collects all results."""
        inferencer = AsyncMockInferencer(mock_response="item")

        inputs = iter(["a", "b", "c"])
        result = await inferencer.ainfer(inputs)

        self.assertEqual(result, ["item", "item", "item"])
        self.assertEqual(inferencer.call_count, 3)

    async def test_ainfer_iterator_with_merger(self):
        """Test async inference with iterator and post_response_merger."""

        def merge_responses(responses):
            return " ".join(responses)

        inferencer = AsyncMockInferencer(
            mock_response="word",
            post_response_merger=merge_responses,
        )

        inputs = iter(["a", "b", "c"])
        result = await inferencer.ainfer(inputs)

        self.assertEqual(result, "word word word")

    async def test_aiter_infer_yields_results(self):
        """Test aiter_infer yields individual results."""
        inferencer = AsyncMockInferencer(mock_response="item")

        results = []
        async for item in inferencer.aiter_infer("test"):
            results.append(item)

        self.assertEqual(results, ["item"])

    async def test_async_context_manager(self):
        """Test async context manager calls aconnect/adisconnect."""
        connect_called = False
        disconnect_called = False

        @attrs
        class LifecycleInferencer(InferencerBase):
            def _infer(self, *args, **kwargs):
                return "result"

            async def aconnect(self, **kwargs):
                nonlocal connect_called
                connect_called = True

            async def adisconnect(self):
                nonlocal disconnect_called
                disconnect_called = True

        async with LifecycleInferencer() as inf:
            self.assertTrue(connect_called)
            result = await inf.ainfer("test")
            self.assertEqual(result, "result")

        self.assertTrue(disconnect_called)

    async def test_ainfer_with_retry(self):
        """Test async inference retry logic."""
        attempt_count = 0

        @attrs
        class FailFirstInferencer(InferencerBase):
            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise RuntimeError("Temporary failure")
                return "success"

        inferencer = FailFirstInferencer(max_retry=3)
        result = await inferencer.ainfer("test")

        self.assertEqual(result, "success")
        self.assertEqual(attempt_count, 3)

    async def test_concurrent_ainfer_calls(self):
        """Test that multiple concurrent ainfer calls work correctly."""
        inferencer = AsyncMockInferencer(mock_response="concurrent", delay_seconds=0.01)

        tasks = [inferencer.ainfer(f"input_{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        self.assertEqual(results, ["concurrent"] * 5)
        self.assertEqual(inferencer.call_count, 5)


if __name__ == "__main__":
    unittest.main()

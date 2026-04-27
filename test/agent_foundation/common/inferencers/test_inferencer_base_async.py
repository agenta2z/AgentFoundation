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


# =============================================================================
# Fix 1a — Recursive pre_retry + _iter_child_inferencers
# =============================================================================


class PreRetryAndChildIterTest(unittest.IsolatedAsyncioTestCase):
    """Tests for the new InferencerBase recursive pre_retry hook and the
    generic _iter_child_inferencers primitive used uniformly by lifecycle
    methods and the retry-time hook."""

    async def test_iter_child_inferencers_default_empty(self):
        """Default base implementation yields nothing."""
        inf = MockInferencer()
        self.assertEqual(list(inf._iter_child_inferencers()), [])

    async def test_pre_retry_default_is_no_op(self):
        """No overrides → forced retry runs cleanly with no error and no
        side effect on the instance (the callback wrapper is not built)."""
        attempts = 0

        @attrs
        class FailOnce(InferencerBase):
            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("transient")
                return "ok"

        inf = FailOnce(max_retry=2)
        result = await inf.ainfer("test")
        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 2)

    async def test_pre_retry_fires_between_attempts(self):
        """Subclass overrides _pre_retry; assert it fires once with the
        right (attempt, exception) on a single retry."""
        recorded = []
        attempts = 0

        @attrs
        class WithPreRetry(InferencerBase):
            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("first")
                return "ok"

            async def _pre_retry(self, attempt, exception):
                recorded.append((attempt, type(exception).__name__))

        inf = WithPreRetry(max_retry=2)
        result = await inf.ainfer("test")
        self.assertEqual(result, "ok")
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0][1], "ConnectionError")

    async def test_pre_retry_propagates_to_children(self):
        """Parent's pre_retry recurses to children declared via
        _iter_child_inferencers. Both children's pre_retry fires."""

        @attrs
        class RecordingChild(InferencerBase):
            calls: list = attrib(factory=list, init=False)

            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                return "child_ok"

            async def _pre_retry(self, attempt, exception):
                self.calls.append((attempt, type(exception).__name__))

        child_a = RecordingChild()
        child_b = RecordingChild()
        attempts = 0

        @attrs
        class Parent(InferencerBase):
            kid_a: Any = attrib(default=None)
            kid_b: Any = attrib(default=None)

            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("p")
                return "parent_ok"

            def _iter_child_inferencers(self):
                if self.kid_a is not None:
                    yield self.kid_a
                if self.kid_b is not None:
                    yield self.kid_b

        parent = Parent(kid_a=child_a, kid_b=child_b, max_retry=2)
        await parent.ainfer("test")
        self.assertEqual(len(child_a.calls), 1)
        self.assertEqual(len(child_b.calls), 1)

    async def test_pre_retry_dedups_shared_children(self):
        """Same child instance referenced from two slots — pre_retry runs
        only once on that instance per propagation."""

        @attrs
        class CountingChild(InferencerBase):
            count: int = attrib(default=0, init=False)

            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                return "ok"

            async def _pre_retry(self, attempt, exception):
                self.count += 1

        shared = CountingChild()
        attempts = 0

        @attrs
        class Parent(InferencerBase):
            slot_a: Any = attrib(default=None)
            slot_b: Any = attrib(default=None)

            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("p")
                return "parent_ok"

            def _iter_child_inferencers(self):
                # Yields shared twice — InferencerBase.pre_retry's id-based
                # _seen set must dedup.
                if self.slot_a is not None:
                    yield self.slot_a
                if self.slot_b is not None:
                    yield self.slot_b

        parent = Parent(slot_a=shared, slot_b=shared, max_retry=2)
        await parent.ainfer("test")
        self.assertEqual(shared.count, 1)

    async def test_pre_retry_handles_cycles(self):
        """Parent's child yields the parent (cycle). No infinite loop;
        each instance runs at most once."""
        parent_calls = []
        child_calls = []
        attempts = 0

        @attrs
        class CycleChild(InferencerBase):
            parent_ref: Any = attrib(default=None)

            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                return "ok"

            async def _pre_retry(self, attempt, exception):
                child_calls.append(1)

            def _iter_child_inferencers(self):
                if self.parent_ref is not None:
                    yield self.parent_ref

        @attrs
        class Parent(InferencerBase):
            kid: Any = attrib(default=None)

            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("p")
                return "parent_ok"

            async def _pre_retry(self, attempt, exception):
                parent_calls.append(1)

            def _iter_child_inferencers(self):
                if self.kid is not None:
                    yield self.kid

        kid = CycleChild()
        parent = Parent(kid=kid, max_retry=2)
        kid.parent_ref = parent  # cycle
        await parent.ainfer("test")
        self.assertEqual(len(parent_calls), 1)
        self.assertEqual(len(child_calls), 1)

    async def test_pre_retry_continues_on_child_failure(self):
        """One child raises in _pre_retry; siblings still run."""
        sibling_calls = []
        attempts = 0

        @attrs
        class GoodChild(InferencerBase):
            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                return "ok"

            async def _pre_retry(self, attempt, exception):
                sibling_calls.append(1)

        @attrs
        class BadChild(InferencerBase):
            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                return "ok"

            async def _pre_retry(self, attempt, exception):
                raise RuntimeError("bad child cleanup")

        bad = BadChild()
        good = GoodChild()

        @attrs
        class Parent(InferencerBase):
            bad_ref: Any = attrib(default=None)
            good_ref: Any = attrib(default=None)

            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("p")
                return "parent_ok"

            def _iter_child_inferencers(self):
                # Bad child first — verify good child still fires after.
                yield self.bad_ref
                yield self.good_ref

        parent = Parent(bad_ref=bad, good_ref=good, max_retry=2)
        result = await parent.ainfer("test")
        self.assertEqual(result, "parent_ok")
        self.assertEqual(len(sibling_calls), 1)

    async def test_pre_retry_fires_before_user_callback(self):
        """Subclass _pre_retry runs first; user-supplied on_retry_callback
        runs after — so user callback sees clean state."""
        order = []
        attempts = 0

        @attrs
        class WithPreRetry(InferencerBase):
            def _infer(self, *args, **kwargs):
                return "sync"

            async def _ainfer(self, *args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("first")
                return "ok"

            async def _pre_retry(self, attempt, exception):
                order.append("pre_retry")

        def user_callback(attempt, exception, inference_args):
            order.append("user_callback")

        inf = WithPreRetry(max_retry=2)
        await inf.ainfer("test", on_retry_callback=user_callback)
        self.assertEqual(order, ["pre_retry", "user_callback"])


if __name__ == "__main__":
    unittest.main()

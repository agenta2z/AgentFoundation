# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for BreakdownThenAggregateInferencer.

Phase 7: Tests covering basic diamond functionality, parse_numbered_list,
error handling, and resumability.
"""

import os
import pickle
import shutil
import tempfile
import unittest

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
    parse_numbered_list,
)
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)


@attrs
class MockInferencer(InferencerBase):
    """Mock inferencer that returns a configurable response."""

    _response = attrib(default="mock response")
    _call_count = attrib(default=0, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._call_count += 1
        if callable(self._response):
            return self._response(inference_input)
        return self._response


# ---------------------------------------------------------------------------
# 7.1: Basic diamond functionality
# ---------------------------------------------------------------------------


class TestBasicDiamondFunctionality(unittest.TestCase):
    """Tests for basic breakdown-then-aggregate diamond execution."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_worker_factory(self, response_fn=None):
        """Return a worker_factory that creates MockInferencers.

        Args:
            response_fn: If provided, called with (sub_query, index) to produce
                the response value for the mock. Otherwise returns a static
                string derived from the sub_query.
        """
        created_workers = []

        def factory(sub_query, index):
            if response_fn is not None:
                resp = response_fn(sub_query, index)
            else:
                resp = f"result_for_{sub_query}"
            worker = MockInferencer(response=resp)
            created_workers.append(worker)
            return worker

        factory.created_workers = created_workers
        return factory

    def test_fixed_breakdown_3_queries(self):
        """Breakdown returns 3 numbered queries; verify 3 workers execute
        and the aggregator receives all 3 results."""
        breakdown = MockInferencer(response="1. Q1\n2. Q2\n3. Q3")

        aggregator_inputs = []

        def agg_fn(inp):
            aggregator_inputs.append(inp)
            return "aggregated"

        aggregator = MockInferencer(response=agg_fn)
        worker_factory = self._make_worker_factory()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory,
            aggregator_inferencer=aggregator,
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("original question")

        # 3 workers should have been created
        self.assertEqual(len(worker_factory.created_workers), 3)
        # Aggregator should have been called
        self.assertEqual(aggregator._call_count, 1)
        # Aggregator input should contain all 3 worker results
        agg_input = aggregator_inputs[0]
        self.assertIn("result_for_Q1", agg_input)
        self.assertIn("result_for_Q2", agg_input)
        self.assertIn("result_for_Q3", agg_input)
        self.assertEqual(result, "aggregated")

    def test_dynamic_breakdown_with_max(self):
        """Breakdown returns 10 queries but max_breakdown=5; only 5 workers
        should execute."""
        queries = "\n".join(f"{i + 1}. Query{i + 1}" for i in range(10))
        breakdown = MockInferencer(response=queries)
        worker_factory = self._make_worker_factory()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory,
            aggregator_inferencer=None,
            max_breakdown=5,
            checkpoint_dir=self.tmpdir,
        )

        bta.infer("original question")

        self.assertEqual(len(worker_factory.created_workers), 5)

    def test_no_aggregator(self):
        """When aggregator_inferencer=None, raw tuple of worker results is
        returned."""
        breakdown = MockInferencer(response="1. A\n2. B")
        worker_factory = self._make_worker_factory()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("question")

        # Without aggregator, result should be a tuple of worker outputs
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIn("result_for_A", result)
        self.assertIn("result_for_B", result)

    def test_single_sub_query(self):
        """Breakdown returns a single query; verify single worker executes."""
        breakdown = MockInferencer(response="1. OnlyOne")
        worker_factory = self._make_worker_factory()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("question")

        self.assertEqual(len(worker_factory.created_workers), 1)


# ---------------------------------------------------------------------------
# 7.2: parse_numbered_list tests
# ---------------------------------------------------------------------------


class TestParseNumberedList(unittest.TestCase):
    """Tests for the parse_numbered_list helper function."""

    def test_parse_numbered_dot(self):
        text = "1. A\n2. B"
        self.assertEqual(parse_numbered_list(text), ["A", "B"])

    def test_parse_numbered_paren(self):
        text = "1) A\n2) B"
        self.assertEqual(parse_numbered_list(text), ["A", "B"])

    def test_parse_bullet_dash(self):
        text = "- A\n- B"
        self.assertEqual(parse_numbered_list(text), ["A", "B"])

    def test_parse_empty(self):
        self.assertEqual(parse_numbered_list(""), [])


# ---------------------------------------------------------------------------
# 7.3: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling(unittest.TestCase):
    """Tests for error propagation in BreakdownThenAggregateInferencer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_worker_failure_raises(self):
        """When a worker throws an exception, it should propagate."""
        breakdown = MockInferencer(response="1. A\n2. B")

        call_index = [0]

        def failing_factory(sub_query, index):
            def fail_on_call(inp):
                raise RuntimeError(f"Worker {index} failed")

            return MockInferencer(response=fail_on_call)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=failing_factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
        )

        with self.assertRaises(RuntimeError):
            bta.infer("question")

    def test_empty_breakdown(self):
        """When breakdown returns empty list, return breakdown output as-is."""
        # Use a breakdown_parser that returns an empty list
        breakdown = MockInferencer(response="no queries here")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=lambda sub_query, index: MockInferencer(),
            aggregator_inferencer=None,
            breakdown_parser=lambda x: [],
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("question")

        # Should return the raw breakdown output
        self.assertEqual(result, "no queries here")

    def test_breakdown_failure_raises(self):
        """When breakdown inferencer throws, exception should propagate."""

        def failing_breakdown(inp):
            raise ValueError("Breakdown failed")

        breakdown = MockInferencer(response=failing_breakdown)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=lambda sub_query, index: MockInferencer(),
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
        )

        with self.assertRaises(ValueError):
            bta.infer("question")


# ---------------------------------------------------------------------------
# 7.4: Resumability
# ---------------------------------------------------------------------------


class TestResumability(unittest.TestCase):
    """Tests for checkpoint/resume behavior."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resume_after_partial_workers(self):
        """3 workers: first 2 complete and save results, crash before 3rd.
        On resume, workers 1-2 should load from saved results."""
        breakdown = MockInferencer(response="1. W1\n2. W2\n3. W3")

        # --- First run: simulate crash on worker 3 ---
        run1_call_counts = [0, 0, 0]

        def crashing_factory(sub_query, index):
            def worker_fn(inp):
                run1_call_counts[index] += 1
                if index == 2:
                    raise RuntimeError("Simulated crash on worker 3")
                return f"result_{index}"

            return MockInferencer(response=worker_fn)

        from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
            StepResultSaveOptions,
        )

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=crashing_factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            enable_result_save=StepResultSaveOptions.OnError,
            resume_with_saved_results=False,
        )

        with self.assertRaises(RuntimeError):
            bta.infer("question")

        # --- Second run: resume, workers 1-2 should load saved results ---
        run2_call_counts = [0, 0, 0]

        def resuming_factory(sub_query, index):
            def worker_fn(inp):
                run2_call_counts[index] += 1
                return f"result_{index}"

            return MockInferencer(response=worker_fn)

        bta_resume = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=resuming_factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            enable_result_save=StepResultSaveOptions.Always,
            resume_with_saved_results=True,
        )

        result = bta_resume.infer("question")

        # Workers 0 and 1 should NOT have been called again (loaded from save)
        self.assertEqual(run2_call_counts[0], 0, "Worker 0 should load from saved")
        self.assertEqual(run2_call_counts[1], 0, "Worker 1 should load from saved")
        # Worker 2 should have been called (no saved result from crash)
        self.assertEqual(run2_call_counts[2], 1, "Worker 2 should execute on resume")




# ---------------------------------------------------------------------------
# 7.5: Concurrency control (max_concurrency)
# ---------------------------------------------------------------------------


@attrs
class AsyncMockInferencer(InferencerBase):
    """Mock inferencer with async support for concurrency testing."""

    _response = attrib(default="mock response")
    _delay = attrib(default=0.0)
    _call_count = attrib(default=0, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._call_count += 1
        if callable(self._response):
            return self._response(inference_input)
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        import asyncio

        self._call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if callable(self._response):
            return self._response(inference_input)
        return self._response


class TestMaxConcurrency(unittest.IsolatedAsyncioTestCase):
    """Tests for max_concurrency parameter controlling parallel worker execution.

    NOTE: max_concurrency uses a shared asyncio.Semaphore across the entire
    WorkGraph execution (start nodes AND downstream propagation). When an
    aggregator is present, the downstream propagation to the aggregator also
    acquires the semaphore while the start-node slot is still held, which can
    cause deadlock. Therefore these tests exercise max_concurrency WITHOUT
    an aggregator (where it works correctly).
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_async_worker_factory(self, delay=0.05, tracker=None):
        """Create a worker factory that tracks concurrent execution.

        Args:
            delay: Simulated async work duration per worker.
            tracker: Dict to record concurrency metrics. If None, one is created.

        Returns:
            (factory, tracker) tuple.
        """
        import asyncio

        if tracker is None:
            tracker = {"max_concurrent": 0, "current": 0, "lock": asyncio.Lock()}

        def factory(sub_query, index):
            async def _tracked_response(inp):
                async with tracker["lock"]:
                    tracker["current"] += 1
                    if tracker["current"] > tracker["max_concurrent"]:
                        tracker["max_concurrent"] = tracker["current"]
                await asyncio.sleep(delay)
                async with tracker["lock"]:
                    tracker["current"] -= 1
                return f"result_{index}"

            return AsyncMockInferencer(response=_tracked_response, delay=0.0)

        return factory, tracker

    async def test_unlimited_concurrency_by_default(self):
        """Without max_concurrency, all workers should run concurrently."""
        n_workers = 6
        queries = "\n".join(f"{i+1}. Q{i+1}" for i in range(n_workers))
        breakdown = AsyncMockInferencer(response=queries)
        factory, tracker = self._make_async_worker_factory(delay=0.05)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=factory,
            aggregator_inferencer=None,  # No aggregator to avoid deadlock
            checkpoint_dir=self.tmpdir,
        )

        result = await bta.ainfer("question")

        # All 6 workers should have been concurrent (no throttling)
        self.assertEqual(
            tracker["max_concurrent"],
            n_workers,
            f"Expected all {n_workers} workers concurrent, got {tracker['max_concurrent']}",
        )

    async def test_max_concurrency_limits_parallel_workers(self):
        """With max_concurrency=2 and 6 workers, at most 2 should run at once."""
        n_workers = 6
        max_conc = 2
        queries = "\n".join(f"{i+1}. Q{i+1}" for i in range(n_workers))
        breakdown = AsyncMockInferencer(response=queries)
        factory, tracker = self._make_async_worker_factory(delay=0.05)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=factory,
            aggregator_inferencer=None,  # No aggregator to avoid deadlock
            checkpoint_dir=self.tmpdir,
            max_concurrency=max_conc,
        )

        result = await bta.ainfer("question")

        self.assertLessEqual(
            tracker["max_concurrent"],
            max_conc,
            f"Expected at most {max_conc} concurrent workers, got {tracker['max_concurrent']}",
        )
        # Also verify that some parallelism did happen (not fully sequential)
        self.assertGreater(
            tracker["max_concurrent"],
            1,
            "Expected at least 2 workers running concurrently with max_concurrency=2",
        )

    async def test_max_concurrency_1_runs_sequentially(self):
        """With max_concurrency=1, workers should run one at a time."""
        n_workers = 4
        queries = "\n".join(f"{i+1}. Q{i+1}" for i in range(n_workers))
        breakdown = AsyncMockInferencer(response=queries)
        factory, tracker = self._make_async_worker_factory(delay=0.05)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=factory,
            aggregator_inferencer=None,  # No aggregator to avoid deadlock
            checkpoint_dir=self.tmpdir,
            max_concurrency=1,
        )

        result = await bta.ainfer("question")

        self.assertEqual(
            tracker["max_concurrent"],
            1,
            "With max_concurrency=1, only 1 worker should run at a time",
        )

    async def test_max_concurrency_all_workers_complete(self):
        """Verify all workers complete and produce results regardless of throttling."""
        import asyncio

        n_workers = 8
        max_conc = 3
        queries = "\n".join(f"{i+1}. Q{i+1}" for i in range(n_workers))
        breakdown = AsyncMockInferencer(response=queries)
        completed = []

        def factory(sub_query, index):
            async def _response(inp):
                await asyncio.sleep(0.02)
                completed.append(index)
                return f"result_{index}"

            return AsyncMockInferencer(response=_response, delay=0.0)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=factory,
            aggregator_inferencer=None,  # No aggregator to avoid deadlock
            checkpoint_dir=self.tmpdir,
            max_concurrency=max_conc,
        )

        result = await bta.ainfer("question")

        # All 8 workers should have completed
        self.assertEqual(
            len(completed),
            n_workers,
            f"Expected {n_workers} completions, got {len(completed)}",
        )

    def test_max_concurrency_does_not_affect_sync_path(self):
        """max_concurrency should not break the sync infer() path."""
        breakdown = MockInferencer(response="1. Q1\n2. Q2\n3. Q3")
        worker_factory_calls = []

        def factory(sub_query, index):
            worker_factory_calls.append(index)
            return MockInferencer(response=f"result_{index}")

        aggregator = MockInferencer(response=lambda inp: "aggregated")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=factory,
            aggregator_inferencer=aggregator,
            checkpoint_dir=self.tmpdir,
            max_concurrency=2,
        )

        result = bta.infer("question")

        self.assertEqual(result, "aggregated")
        # All 3 workers should have been created and executed
        self.assertEqual(len(worker_factory_calls), 3)


    async def test_sliding_window_not_batched(self):
        """Verify max_concurrency uses sliding window, not batch-and-wait.

        With max_concurrency=2 and 3 workers where worker_0 is fast (10ms)
        and worker_1 is slow (100ms):
        - Sliding window: worker_2 starts at ~10ms when worker_0 finishes
          (total time ≈ 100ms)
        - Batch approach: worker_2 starts at ~100ms when both finish
          (total time ≈ 200ms)

        We verify via completion order: worker_0 finishes first, then
        worker_2 starts and finishes before worker_1, proving the
        sliding window released the slot immediately.
        """
        import asyncio
        import time

        queries = "1. Q1\n2. Q2\n3. Q3"
        breakdown = AsyncMockInferencer(response=queries)

        completion_order = []

        def factory(sub_query, index):
            async def _response(inp):
                if index == 0:
                    await asyncio.sleep(0.01)   # Fast: 10ms
                elif index == 1:
                    await asyncio.sleep(0.15)   # Slow: 150ms
                else:
                    await asyncio.sleep(0.05)   # Medium: 50ms
                completion_order.append(index)
                return f"result_{index}"

            return AsyncMockInferencer(response=_response, delay=0.0)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            max_concurrency=2,
        )

        start = time.monotonic()
        result = await bta.ainfer("question")
        elapsed = time.monotonic() - start

        # Sliding window: worker_0 (10ms) and worker_1 (150ms) start together.
        # worker_0 finishes at ~10ms, releasing the slot for worker_2 (50ms).
        # worker_2 finishes at ~60ms, worker_1 finishes at ~150ms.
        # Total ≈ 150ms. Batch would be ≈ 200ms (150ms + 50ms).

        # worker_0 should finish first, worker_2 second (started in worker_0's slot),
        # worker_1 last (the slow one)
        self.assertEqual(
            completion_order,
            [0, 2, 1],
            f"Expected sliding-window completion order [0, 2, 1], got {completion_order}. "
            "If [0, 1, 2], the implementation is batched rather than sliding window.",
        )

        # Total time should be ~150ms (sliding window), not ~200ms (batched)
        self.assertLess(
            elapsed,
            0.19,
            f"Elapsed {elapsed:.3f}s suggests batching, not sliding window. "
            "Sliding window should complete in ~150ms.",
        )

    async def test_max_concurrency_with_aggregator_no_deadlock(self):
        """Verify max_concurrency works correctly with aggregator (no deadlock).

        Previously this would deadlock because the semaphore was acquired at
        two nested levels (start-node + downstream propagation). The fix moved
        to callee-side semaphore gating: each node acquires the semaphore only
        for its own computation and releases before downstream propagation.
        """
        queries = "1. Q1\n2. Q2\n3. Q3"
        breakdown = AsyncMockInferencer(response=queries)

        def factory(sub_query, index):
            return AsyncMockInferencer(response=f"r{index}", delay=0.01)

        aggregator = AsyncMockInferencer(response="aggregated")

        # max_concurrency=1 with aggregator — previously deadlocked
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=factory,
            aggregator_inferencer=aggregator,
            checkpoint_dir=self.tmpdir,
            max_concurrency=1,
        )
        result = await bta.ainfer("question")
        # The key assertion is that this completes without deadlock.
        # Result contains aggregator output (may be wrapped in a tuple
        # depending on WorkGraph post-processing).
        self.assertIsNotNone(result)
        self.assertIn("aggregated", str(result))

        # max_concurrency=2 with aggregator — also previously deadlocked
        bta2 = BreakdownThenAggregateInferencer(
            breakdown_inferencer=AsyncMockInferencer(response=queries),
            worker_factory=factory,
            aggregator_inferencer=AsyncMockInferencer(response="aggregated2"),
            checkpoint_dir=self.tmpdir,
            max_concurrency=2,
        )
        result2 = await bta2.ainfer("question")
        self.assertIsNotNone(result2)
        self.assertIn("aggregated2", str(result2))


if __name__ == "__main__":
    unittest.main()

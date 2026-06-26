

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
from rich_python_utils.common_utils.function_helper import FallbackMode


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

    def _make_worker_inferencers(self, response_fn=None):
        """Return a worker_inferencers that creates MockInferencers.

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
        worker_inferencers = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=worker_inferencers,
            aggregator_inferencer=aggregator,
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("original question")

        # 3 workers should have been created
        self.assertEqual(len(worker_inferencers.created_workers), 3)
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
        worker_inferencers = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=worker_inferencers,
            aggregator_inferencer=None,
            max_breakdown=5,
            checkpoint_dir=self.tmpdir,
        )

        bta.infer("original question")

        self.assertEqual(len(worker_inferencers.created_workers), 5)

    def test_no_aggregator(self):
        """When aggregator_inferencer=None, raw tuple of worker results is
        returned."""
        breakdown = MockInferencer(response="1. A\n2. B")
        worker_inferencers = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=worker_inferencers,
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
        worker_inferencers = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=worker_inferencers,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("question")

        self.assertEqual(len(worker_inferencers.created_workers), 1)


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
            worker_inferencers=failing_factory,
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
            worker_inferencers=lambda sub_query, index: MockInferencer(),
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
            worker_inferencers=lambda sub_query, index: MockInferencer(),
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
        """Crash during workers → resume skips breakdown (loaded from checkpoint).

        Worker nodes explicitly disable result saving (enable_result_save=False)
        so that workers handle their own internal checkpointing. On resume,
        the breakdown checkpoint IS loaded (skipping the breakdown step), but
        all workers re-execute — this is by design, since workers (which could
        be PTI or DualInferencer instances) manage their own resume internally.
        """
        breakdown_call_count = [0]

        def counting_breakdown_fn(inp):
            breakdown_call_count[0] += 1
            return "1. W1\n2. W2\n3. W3"

        breakdown = MockInferencer(response=counting_breakdown_fn)

        # --- First run: simulate crash on worker 3 ---
        def crashing_factory(sub_query, index):
            def worker_fn(inp):
                if index == 2:
                    raise RuntimeError("Simulated crash on worker 3")
                return f"result_{index}"

            return MockInferencer(response=worker_fn)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=crashing_factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=False,
            # FallbackMode.NEVER: no _infer_recovery call, so _infer (and breakdown)
            # runs exactly once. Default ON_FIRST_FAILURE would re-call _infer via
            # _infer_recovery, causing breakdown to run twice.
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )

        with self.assertRaises(RuntimeError):
            bta.infer("question")

        self.assertEqual(breakdown_call_count[0], 1, "Breakdown ran once in first run")

        # --- Second run: resume, breakdown should be skipped (loaded from checkpoint) ---
        run2_worker_calls = [0, 0, 0]

        def resuming_factory(sub_query, index):
            def worker_fn(inp):
                run2_worker_calls[index] += 1
                return f"result_{index}"

            return MockInferencer(response=worker_fn)

        bta_resume = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=resuming_factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=True,
        )

        result = bta_resume.infer("question")

        # Breakdown should NOT run again — loaded from checkpoint
        self.assertEqual(
            breakdown_call_count[0], 1,
            "Breakdown should not re-run on resume (loaded from checkpoint)"
        )
        # All workers re-execute (by design: worker nodes have enable_result_save=False)
        for i in range(3):
            self.assertEqual(
                run2_worker_calls[i], 1,
                f"Worker {i} should execute on resume (workers manage own checkpoints)"
            )




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

    def _make_async_worker_inferencers(self, delay=0.05, tracker=None):
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
        factory, tracker = self._make_async_worker_inferencers(delay=0.05)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
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
        factory, tracker = self._make_async_worker_inferencers(delay=0.05)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
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
        factory, tracker = self._make_async_worker_inferencers(delay=0.05)

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
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
            worker_inferencers=factory,
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
        worker_inferencers_calls = []

        def factory(sub_query, index):
            worker_inferencers_calls.append(index)
            return MockInferencer(response=f"result_{index}")

        aggregator = MockInferencer(response=lambda inp: "aggregated")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
            aggregator_inferencer=aggregator,
            checkpoint_dir=self.tmpdir,
            max_concurrency=2,
        )

        result = bta.infer("question")

        self.assertEqual(result, "aggregated")
        # All 3 workers should have been created and executed
        self.assertEqual(len(worker_inferencers_calls), 3)


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
            worker_inferencers=factory,
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
            worker_inferencers=factory,
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
            worker_inferencers=factory,
            aggregator_inferencer=AsyncMockInferencer(response="aggregated2"),
            checkpoint_dir=self.tmpdir,
            max_concurrency=2,
        )
        result2 = await bta2.ainfer("question")
        self.assertIsNotNone(result2)
        self.assertIn("aggregated2", str(result2))


class TestPredefinedSubQueries(unittest.TestCase):
    """Tests for predefined_sub_queries mode (skip breakdown).

    Covers plan smoke tests A–H:
    A: predefined list → workers get correct queries, no LLM call
    B: single string, max_breakdown=5 → 5 workers with same query
    C: single string, max_breakdown=None, max_concurrency=3 → 3 workers
    D: single string, both None → 1 worker (fallback)
    E: default None → existing behaviour completely unchanged
    F: checkpoint + predefined → checkpoint wins
    G: predefined + breakdown_only=True → warning logged, proceeds normally
    H: predefined_sub_queries=None, breakdown_inferencer=None → clear ValueError
    + list capping, empty list, no-aggregator, structured dict sub_queries
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_worker_inferencers(self):
        """Returns a factory that records which sub_queries each worker received."""
        received_queries = []

        def factory(sub_query, index):
            received_queries.append(sub_query)
            return MockInferencer(response=f"result_for_{sub_query}")

        factory.received_queries = received_queries
        return factory

    # ── Test A: predefined list ────────────────────────────────────────────────

    def test_predefined_list_skips_breakdown(self):
        """Test A: predefined_sub_queries list bypasses breakdown inferencer entirely."""
        breakdown = MockInferencer(response="1. LLM_Q1\n2. LLM_Q2")
        factory = self._make_worker_inferencers()
        aggregator_inputs = []

        def agg_fn(inp):
            aggregator_inputs.append(inp)
            return "aggregated"

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
            aggregator_inferencer=MockInferencer(response=agg_fn),
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries=["Q_alpha", "Q_beta", "Q_gamma"],
        )

        result = bta.infer("original question")

        # breakdown_inferencer must NOT be called
        self.assertEqual(breakdown._call_count, 0, "breakdown_inferencer should not be called")
        # Workers must receive the predefined queries, not LLM output
        self.assertEqual(factory.received_queries, ["Q_alpha", "Q_beta", "Q_gamma"])
        # Aggregator receives results from the 3 predefined workers
        self.assertIn("result_for_Q_alpha", aggregator_inputs[0])
        self.assertIn("result_for_Q_beta", aggregator_inputs[0])
        self.assertIn("result_for_Q_gamma", aggregator_inputs[0])
        self.assertEqual(result, "aggregated")

    def test_predefined_list_no_breakdown_inferencer_needed(self):
        """predefined_sub_queries works even when breakdown_inferencer=None."""
        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries=["Q1", "Q2"],
        )

        result = bta.infer("question")
        self.assertEqual(factory.received_queries, ["Q1", "Q2"])
        self.assertIsNotNone(result)

    # ── Test B: single string + max_breakdown ─────────────────────────────────

    def test_single_string_auto_repeat_uses_max_breakdown(self):
        """Test B: single string query auto-repeated max_breakdown times."""
        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries="repeat me",
            max_breakdown=5,
        )

        bta.infer("question")

        self.assertEqual(len(factory.received_queries), 5)
        self.assertTrue(all(q == "repeat me" for q in factory.received_queries))

    # ── Test C: single string + max_concurrency fallback ──────────────────────

    def test_single_string_auto_repeat_uses_max_concurrency_when_no_max_breakdown(self):
        """Test C: single string falls back to max_concurrency when max_breakdown=None."""
        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries="repeat me",
            max_breakdown=None,
            max_concurrency=3,
        )

        bta.infer("question")

        self.assertEqual(len(factory.received_queries), 3)
        self.assertTrue(all(q == "repeat me" for q in factory.received_queries))

    # ── Test D: single string + both None → 1 worker ─────────────────────────

    def test_single_string_auto_repeat_fallback_to_1(self):
        """Test D: single string with max_breakdown=None, max_concurrency=None → 1 worker."""
        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries="just one",
            max_breakdown=None,
            max_concurrency=None,
        )

        bta.infer("question")

        self.assertEqual(len(factory.received_queries), 1)
        self.assertEqual(factory.received_queries[0], "just one")

    # ── Test E: default None → unchanged behaviour ────────────────────────────

    def test_default_none_runs_normal_breakdown(self):
        """Test E: predefined_sub_queries=None (default) → normal breakdown runs."""
        breakdown = MockInferencer(response="1. LLM_Q1\n2. LLM_Q2")
        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            # predefined_sub_queries not set — default None
        )

        bta.infer("question")

        # breakdown_inferencer MUST be called
        self.assertEqual(breakdown._call_count, 1)
        # Workers receive LLM-produced queries
        self.assertEqual(factory.received_queries, ["LLM_Q1", "LLM_Q2"])

    # ── Test F: checkpoint wins over predefined ────────────────────────────────

    def test_checkpoint_wins_over_predefined_sub_queries(self):
        """Test F: saved checkpoint takes priority over predefined_sub_queries."""
        import json

        # Write a fake breakdown checkpoint
        ckpt_path = os.path.join(self.tmpdir, "breakdown_result.json")
        with open(ckpt_path, "w") as f:
            json.dump({"sub_queries": ["CKPT_Q1", "CKPT_Q2"], "raw_output": ""}, f)

        breakdown = MockInferencer(response="1. LLM_Q1\n2. LLM_Q2")
        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=True,
            predefined_sub_queries=["PREDEFINED_Q1", "PREDEFINED_Q2"],
        )

        bta.infer("question")

        # Checkpoint sub_queries should be used, not predefined ones
        self.assertEqual(factory.received_queries, ["CKPT_Q1", "CKPT_Q2"])
        # Neither breakdown nor predefined should override checkpoint
        self.assertEqual(breakdown._call_count, 0)

    # ── Test G: breakdown_only=True with predefined → warning, proceeds ────────

    def test_breakdown_only_with_predefined_logs_warning_and_proceeds(self):
        """Test G: breakdown_only=True is ignored (with warning) when predefined set."""
        import logging

        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries=["Q1", "Q2"],
            breakdown_only=True,
        )

        with self.assertLogs(
            "agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer",
            level=logging.WARNING,
        ) as cm:
            result = bta.infer("question")

        # Warning should mention breakdown_only being ignored
        self.assertTrue(
            any("breakdown_only" in msg for msg in cm.output),
            f"Expected breakdown_only warning, got: {cm.output}",
        )
        # Should still proceed and run workers (not stop after breakdown)
        self.assertEqual(factory.received_queries, ["Q1", "Q2"])

    # ── Test H: no breakdown_inferencer + no predefined → ValueError ──────────

    def test_no_breakdown_inferencer_no_predefined_raises_valueerror(self):
        """Test H: breakdown_inferencer=None with predefined_sub_queries=None raises ValueError."""
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=lambda sub_query, index: MockInferencer(),
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            # predefined_sub_queries=None (default)
        )

        with self.assertRaises(ValueError) as ctx:
            bta.infer("question")

        self.assertIn("breakdown_inferencer", str(ctx.exception))
        self.assertIn("predefined_sub_queries", str(ctx.exception))

    # ── max_breakdown cap on predefined list ──────────────────────────────────

    def test_max_breakdown_caps_predefined_list(self):
        """max_breakdown cap is applied to predefined list (truncates to N)."""
        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries=["Q1", "Q2", "Q3", "Q4", "Q5"],
            max_breakdown=3,
        )

        bta.infer("question")

        # Only the first 3 should be used
        self.assertEqual(factory.received_queries, ["Q1", "Q2", "Q3"])

    def test_max_breakdown_cap_does_not_double_truncate_single_string(self):
        """max_breakdown=N with single string → exactly N workers, no double-cap."""
        factory = self._make_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries="query",
            max_breakdown=4,
        )

        bta.infer("question")

        # auto-repeat creates 4 (= max_breakdown), cap checks 4 > 4 = False → keeps 4
        self.assertEqual(len(factory.received_queries), 4)
        self.assertTrue(all(q == "query" for q in factory.received_queries))

    # ── Empty predefined list ─────────────────────────────────────────────────

    def test_empty_predefined_list_returns_empty_string(self):
        """Empty predefined_sub_queries=[] returns '' (consistent with checkpoint path)."""
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=lambda sub_query, index: MockInferencer(),
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries=[],
        )

        result = bta.infer("question")
        self.assertEqual(result, "")

    # ── Structured dict sub_queries ───────────────────────────────────────────

    def test_predefined_structured_dict_sub_queries(self):
        """predefined_sub_queries as List[dict] is accepted and workers run.

        _build_diamond_graph extracts the "query" key from each dict before
        passing it to the worker factory — so workers receive the string query,
        not the raw dict. This verifies the dict list is accepted end-to-end.
        """
        received = []

        def factory(sub_query, index):
            received.append(sub_query)
            return MockInferencer(response=f"result_{index}")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries=[
                {"query": "analyse security", "args": {}},
                {"query": "analyse performance", "args": {}},
            ],
        )

        bta.infer("analyse codebase")

        # _build_diamond_graph extracts "query" from each dict before passing to factory
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0], "analyse security")
        self.assertEqual(received[1], "analyse performance")


class TestPredefinedSubQueriesAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for predefined_sub_queries mode in _ainfer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_async_worker_inferencers(self):
        """Returns a factory recording which sub_queries each async worker received."""
        received_queries = []

        def factory(sub_query, index):
            received_queries.append(sub_query)
            return AsyncMockInferencer(response=f"result_for_{sub_query}")

        factory.received_queries = received_queries
        return factory

    async def test_ainfer_predefined_list(self):
        """predefined_sub_queries list works correctly in async path."""
        breakdown = AsyncMockInferencer(response="1. LLM_Q1\n2. LLM_Q2")
        factory = self._make_async_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
            aggregator_inferencer=AsyncMockInferencer(response="aggregated"),
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries=["async_Q1", "async_Q2"],
        )

        result = await bta.ainfer("question")

        self.assertEqual(breakdown._call_count, 0, "breakdown should not be called in async path")
        self.assertEqual(factory.received_queries, ["async_Q1", "async_Q2"])
        self.assertIn("aggregated", str(result))

    async def test_ainfer_single_string_auto_repeat(self):
        """Single string auto-repeat works in async path with max_breakdown."""
        factory = self._make_async_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries="async query",
            max_breakdown=3,
        )

        await bta.ainfer("question")

        self.assertEqual(len(factory.received_queries), 3)
        self.assertTrue(all(q == "async query" for q in factory.received_queries))

    async def test_ainfer_no_breakdown_inferencer_raises_valueerror(self):
        """breakdown_inferencer=None + predefined_sub_queries=None raises ValueError in async."""
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=lambda sub_query, index: AsyncMockInferencer(),
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
        )

        with self.assertRaises(ValueError) as ctx:
            await bta.ainfer("question")

        self.assertIn("breakdown_inferencer", str(ctx.exception))

    async def test_ainfer_checkpoint_wins_over_predefined(self):
        """Checkpoint takes priority over predefined_sub_queries in async path."""
        import json

        ckpt_path = os.path.join(self.tmpdir, "breakdown_result.json")
        with open(ckpt_path, "w") as f:
            json.dump({"sub_queries": ["CKPT_Q1"], "raw_output": ""}, f)

        factory = self._make_async_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=True,
            predefined_sub_queries=["PREDEFINED_Q1", "PREDEFINED_Q2"],
        )

        await bta.ainfer("question")

        self.assertEqual(factory.received_queries, ["CKPT_Q1"])

    async def test_ainfer_breakdown_only_warning_proceeds(self):
        """breakdown_only=True is ignored with warning in async path."""
        import logging

        factory = self._make_async_worker_inferencers()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            worker_inferencers=factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            predefined_sub_queries=["Q1", "Q2"],
            breakdown_only=True,
        )

        with self.assertLogs(
            "agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer",
            level=logging.WARNING,
        ) as cm:
            await bta.ainfer("question")

        self.assertTrue(any("breakdown_only" in msg for msg in cm.output))
        self.assertEqual(factory.received_queries, ["Q1", "Q2"])


# =============================================================================
# Post-mortem fixes: Fix 1c (BTA child) + Fix 4 (narrow retry exceptions)
# =============================================================================


class TestPostMortemFixes(unittest.TestCase):
    """Fix 1c: BTA._iter_child_inferencers yields the aggregator.
    Fix 4: BTA's WorkGraph nodes use the narrow retry exception list."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_bta_iter_child_inferencers_yields_aggregator(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )
        agg = MockInferencer(response="agg")
        bta = BreakdownThenAggregateInferencer(
            predefined_sub_queries=["q1", "q2"],
            worker_inferencers=lambda i: MockInferencer(response=f"w{i}"),
            aggregator_inferencer=agg,
            checkpoint_dir=self.tmpdir,
        )
        children = list(bta._iter_child_inferencers())
        self.assertEqual(len(children), 1)
        self.assertIs(children[0], agg)

    def test_bta_iter_child_inferencers_no_aggregator_yields_nothing(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )
        bta = BreakdownThenAggregateInferencer(
            predefined_sub_queries=["q1"],
            worker_inferencers=lambda i: MockInferencer(response=f"w{i}"),
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
        )
        self.assertEqual(list(bta._iter_child_inferencers()), [])

    def test_transient_retry_exceptions_excludes_programming_errors(self):
        """Fix 4: TRANSIENT_RETRY_EXCEPTIONS covers transient errors but
        NOT programming errors (TypeError/ValueError/AttributeError)."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            TRANSIENT_RETRY_EXCEPTIONS,
        )
        self.assertIn(TimeoutError, TRANSIENT_RETRY_EXCEPTIONS)
        self.assertIn(ConnectionError, TRANSIENT_RETRY_EXCEPTIONS)
        self.assertIn(OSError, TRANSIENT_RETRY_EXCEPTIONS)
        # Programming errors must NOT be subclasses of anything in the
        # transient list. (TypeError, AttributeError, ValueError are NOT
        # subclasses of TimeoutError, ConnectionError, or OSError.)
        self.assertFalse(issubclass(TypeError, TRANSIENT_RETRY_EXCEPTIONS))
        self.assertFalse(issubclass(AttributeError, TRANSIENT_RETRY_EXCEPTIONS))
        self.assertFalse(issubclass(ValueError, TRANSIENT_RETRY_EXCEPTIONS))


if __name__ == "__main__":
    unittest.main()

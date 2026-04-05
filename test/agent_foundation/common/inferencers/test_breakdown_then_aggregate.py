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


if __name__ == "__main__":
    unittest.main()

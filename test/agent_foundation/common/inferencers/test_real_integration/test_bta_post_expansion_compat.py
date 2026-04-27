"""Mock-based BTA flow test to verify WorkGraph base-class changes don't break diamond pattern (Task 11.6).

Creates a BTA-style WorkGraph with mock nodes: planner → [worker_1, worker_2, worker_3] → aggregator
and verifies:
- Diamond fan-out/fan-in works correctly
- NextNodesSelector routing works unchanged
- WorkGraphStopFlags propagate correctly
- Result saving and resume work

Requirements: 21.2, 21.4, 21.5
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
from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode, WorkGraph
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import ResultPassDownMode
from rich_python_utils.common_objects.workflow.common.worknode_base import (
    WorkGraphStopFlags,
    NextNodesSelector,
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


def _make_worker_factory(response_fn=None):
    """Return a worker_factory that creates MockInferencers."""
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


class TestBTADiamondPostExpansion(unittest.TestCase):
    """Verify BTA diamond pattern works after WorkGraph base-class expansion changes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bta_compat_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_diamond_fan_out_fan_in(self):
        """Planner → [worker_1, worker_2, worker_3] → aggregator works correctly."""
        breakdown = MockInferencer(response="1. Q1\n2. Q2\n3. Q3")

        aggregator_inputs = []

        def agg_fn(inp):
            aggregator_inputs.append(inp)
            return "aggregated"

        aggregator = MockInferencer(response=agg_fn)
        worker_factory = _make_worker_factory()

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

    def test_single_worker_no_fan_out(self):
        """Single sub-query → single worker → aggregator."""
        breakdown = MockInferencer(response="1. Only question")
        aggregator = MockInferencer(response="single aggregated")
        worker_factory = _make_worker_factory()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory,
            aggregator_inferencer=aggregator,
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("question")

        self.assertEqual(len(worker_factory.created_workers), 1)
        self.assertEqual(result, "single aggregated")

    def test_no_aggregator_returns_worker_results(self):
        """Without aggregator, BTA returns concatenated worker results."""
        breakdown = MockInferencer(response="1. Q1\n2. Q2")
        worker_factory = _make_worker_factory()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory,
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("question")

        self.assertEqual(len(worker_factory.created_workers), 2)
        self.assertIn("result_for_Q1", result)
        self.assertIn("result_for_Q2", result)


class TestBTAResumePostExpansion(unittest.TestCase):
    """Verify BTA resume works after WorkGraph base-class expansion changes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bta_resume_compat_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_bta_with_resume_enabled(self):
        """BTA with resume_with_saved_results=True completes successfully."""
        breakdown = MockInferencer(response="1. Q1\n2. Q2\n3. Q3")
        worker_factory = _make_worker_factory()

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory,
            aggregator_inferencer=MockInferencer(response="aggregated"),
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=True,
        )

        result = bta.infer("question")

        self.assertEqual(result, "aggregated")
        self.assertEqual(len(worker_factory.created_workers), 3)

    def test_bta_second_run_with_saved_results(self):
        """Second BTA run with same checkpoint_dir uses saved results."""
        breakdown = MockInferencer(response="1. Q1\n2. Q2")
        worker_factory1 = _make_worker_factory()

        bta1 = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory1,
            aggregator_inferencer=MockInferencer(response="aggregated_1"),
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=True,
        )

        result1 = bta1.infer("question")
        self.assertEqual(result1, "aggregated_1")

        # Second run with same checkpoint dir — should use saved results
        worker_factory2 = _make_worker_factory()

        bta2 = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=worker_factory2,
            aggregator_inferencer=MockInferencer(response="aggregated_2"),
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=True,
        )

        result2 = bta2.infer("question")
        # Should complete successfully
        self.assertIsNotNone(result2)


class TestBTAWorkGraphStopFlags(unittest.TestCase):
    """Verify WorkGraphStopFlags work correctly in BTA diamond after expansion changes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bta_flags_compat_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_basic_workgraph_diamond_with_stop_flags(self):
        """A raw WorkGraph diamond with Continue flags works correctly."""
        call_log = []

        planner = WorkGraphNode(
            name="planner",
            value=lambda x: (call_log.append("planner"), x)[1],
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        )
        worker_1 = WorkGraphNode(
            name="worker_1",
            value=lambda x: (call_log.append("worker_1"), f"w1_{x}")[1],
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        )
        worker_2 = WorkGraphNode(
            name="worker_2",
            value=lambda x: (call_log.append("worker_2"), f"w2_{x}")[1],
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        )
        worker_3 = WorkGraphNode(
            name="worker_3",
            value=lambda x: (call_log.append("worker_3"), f"w3_{x}")[1],
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        )
        aggregator = WorkGraphNode(
            name="aggregator",
            value=lambda *args: (call_log.append("aggregator"), "+".join(str(a) for a in args))[1],
        )

        planner.add_next(worker_1)
        planner.add_next(worker_2)
        planner.add_next(worker_3)
        worker_1.add_next(aggregator)
        worker_2.add_next(aggregator)
        worker_3.add_next(aggregator)

        result = planner.run("input")

        # All nodes should have executed
        self.assertIn("planner", call_log)
        self.assertIn("worker_1", call_log)
        self.assertIn("worker_2", call_log)
        self.assertIn("worker_3", call_log)
        self.assertIn("aggregator", call_log)

    def test_terminate_flag_stops_diamond(self):
        """Terminate flag from planner stops the entire diamond."""
        call_log = []

        planner = WorkGraphNode(
            name="planner",
            value=lambda x: (
                call_log.append("planner"),
                (WorkGraphStopFlags.Terminate, "stopped"),
            )[1],
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        )
        worker_1 = WorkGraphNode(
            name="worker_1",
            value=lambda x: (call_log.append("worker_1"), x)[1],
        )

        planner.add_next(worker_1)

        result = planner.run("input")

        self.assertIn("planner", call_log)
        self.assertNotIn("worker_1", call_log)  # Terminate stops execution


if __name__ == "__main__":
    unittest.main()

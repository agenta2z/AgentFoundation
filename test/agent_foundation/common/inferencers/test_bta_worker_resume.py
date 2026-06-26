"""BTA worker per-node resume — verifies WorkGraph checkpoints on worker nodes.

Tests that _build_subgraph_spec assigns _get_result_path to worker nodes and
does NOT set worker_manages_resume, so WorkGraph can save/load worker results
for instant resume on re-runs.
"""

import json
import os
import shutil
import tempfile
import unittest

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


@attrs
class _MockInferencer(InferencerBase):
    _response = attrib(default="mock response")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._response


class TestWorkerNodeResume(unittest.TestCase):
    """Worker nodes get _get_result_path and no worker_manages_resume."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_bta(self, num_queries=3, **kwargs):
        created = []

        def factory(sub_query, index):
            w = _MockInferencer(response=f"result_{index}")
            created.append(w)
            return w

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=_MockInferencer(response="unused"),
            worker_inferencers=factory,
            aggregator_inferencer=_MockInferencer(response="aggregated"),
            checkpoint_mode="jsonfy",
            enable_result_save=True,
            resume_with_saved_results=True,
            predefined_sub_queries=[f"query_{i}" for i in range(num_queries)],
            **kwargs,
        )
        ws = InferencerWorkspace(root=self.tmpdir)
        ws.ensure_dirs()
        bta._workspace = ws
        bta.name = "test_bta"
        bta._created_workers = created
        return bta

    def test_worker_nodes_have_get_result_path(self):
        """Worker nodes must have _get_result_path assigned."""
        bta = self._make_bta()
        bta._cached_sub_queries = ["q1", "q2", "q3"]

        spec = bta._build_subgraph_spec(bta._cached_sub_queries)

        worker_nodes = [n for n in spec.nodes if "worker" in n.name]
        self.assertEqual(len(worker_nodes), 3)
        for node in worker_nodes:
            path = node._get_result_path(node.name)
            self.assertTrue(path.endswith("_result.json"))
            self.assertIn("checkpoints", path)

    def test_worker_nodes_no_worker_manages_resume(self):
        """Worker nodes must NOT have worker_manages_resume=True so WorkGraph
        checkpoint save/load is not short-circuited."""
        bta = self._make_bta()
        bta._cached_sub_queries = ["q1", "q2"]

        spec = bta._build_subgraph_spec(bta._cached_sub_queries)

        worker_nodes = [n for n in spec.nodes if "worker" in n.name]
        for node in worker_nodes:
            self.assertFalse(
                getattr(node, "worker_manages_resume", False),
                f"{node.name} should not have worker_manages_resume=True",
            )

    def test_worker_checkpoint_path_points_to_child_workspace(self):
        """Checkpoint path should be under children/<worker_name>/checkpoints/."""
        bta = self._make_bta(num_queries=2)
        bta._cached_sub_queries = ["q1", "q2"]

        spec = bta._build_subgraph_spec(bta._cached_sub_queries)

        worker_nodes = [n for n in spec.nodes if "worker" in n.name]
        for i, node in enumerate(worker_nodes):
            path = node._get_result_path(node.name)
            expected_dir = os.path.join(
                self.tmpdir, "children", f"worker_{i}", "checkpoints"
            )
            self.assertTrue(
                path.startswith(expected_dir),
                f"Expected path under {expected_dir}, got {path}",
            )

    def test_aggregator_node_has_get_result_path(self):
        """Aggregator node should also have _get_result_path (existing behavior)."""
        bta = self._make_bta()
        bta._cached_sub_queries = ["q1"]

        spec = bta._build_subgraph_spec(bta._cached_sub_queries)

        agg_nodes = [n for n in spec.nodes if "aggregator" in n.name]
        self.assertEqual(len(agg_nodes), 1)
        path = agg_nodes[0]._get_result_path(agg_nodes[0].name)
        self.assertTrue(path.endswith("_result.json"))

    def test_should_save_result_true_for_workers(self):
        """With _get_result_path set and worker_manages_resume=False,
        _should_save_result() should return True."""
        bta = self._make_bta()
        bta._cached_sub_queries = ["q1"]

        spec = bta._build_subgraph_spec(bta._cached_sub_queries)

        worker_nodes = [n for n in spec.nodes if "worker" in n.name]
        for node in worker_nodes:
            self.assertTrue(
                node._should_save_result(),
                f"{node.name}._should_save_result() should be True",
            )

    def test_load_result_attempts_checkpoint(self):
        """load_result() should try to load from checkpoint, not short-circuit."""
        bta = self._make_bta()
        bta._cached_sub_queries = ["q1"]

        spec = bta._build_subgraph_spec(bta._cached_sub_queries)
        node = [n for n in spec.nodes if "worker" in n.name][0]

        loaded, result = node.load_result()
        self.assertFalse(loaded, "No checkpoint file yet — should return False")

        ckpt_path = node._get_result_path(node.name)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        with open(ckpt_path, "w") as f:
            json.dump({"response": "cached worker output"}, f)

        loaded, result = node.load_result()
        self.assertTrue(loaded, "Checkpoint exists — should load it")
        self.assertEqual(result["response"], "cached worker output")


if __name__ == "__main__":
    unittest.main()

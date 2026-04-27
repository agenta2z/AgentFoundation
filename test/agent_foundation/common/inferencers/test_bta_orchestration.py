"""BTA orchestration tests — Layer 1, R4.

Coverage focus: GAPS not covered by existing test_breakdown_then_aggregate.py.

Existing coverage (test_breakdown_then_aggregate.py):
- Basic 3-query diamond, predefined sub_queries bypass, worker exception handling,
  disable_aggregator, parse_numbered_list

Genuine gaps filled here:
- Heterogeneous worker_factory (dict + task_type_arg_name) (R4)
- _parse_json_subtasks parsing variations (R4 / parser correctness)
- expand_todos_to_workers (R4)
"""

import json
import shutil
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
    parse_numbered_list,
)
from test.agent_foundation.common.inferencers._helpers.mock_inferencer import (
    MockInferencer,
)


# ---------------------------------------------------------------------------
# R4: Heterogeneous worker_factory with task_type_arg_name
# ---------------------------------------------------------------------------


class TestHeterogeneousWorkerFactory(unittest.TestCase):
    """Validates: R4 — dict-based worker_factory dispatches by task_type."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dispatches_by_task_type(self):
        """Breakdown produces tasks with task_type ∈ {security, performance};
        each routes to the correct factory."""
        # _parse_json_subtasks expects {"subtasks": [...]} inside ```json fence
        subtasks_obj = {
            "subtasks": [
                {"description": "Check auth", "args": {"task_type": "security"}},
                {"description": "Profile latency", "args": {"task_type": "performance"}},
            ]
        }
        breakdown_json = f"```json\n{json.dumps(subtasks_obj)}\n```"
        breakdown = MockInferencer(response=breakdown_json)
        aggregator = MockInferencer(response="aggregated")

        # Track which factory was used
        used = {"security": 0, "performance": 0, "default": 0}

        def factory_security(sub_query, index):
            used["security"] += 1
            return MockInferencer(response=f"SEC_{index}")

        def factory_performance(sub_query, index):
            used["performance"] += 1
            return MockInferencer(response=f"PERF_{index}")

        def factory_default(sub_query, index):
            used["default"] += 1
            return MockInferencer(response=f"DEF_{index}")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory={
                "security": factory_security,
                "performance": factory_performance,
                "__default__": factory_default,
            },
            task_type_arg_name="task_type",
            aggregator_inferencer=aggregator,
            breakdown_format="json_subtasks",
            checkpoint_dir=self.tmpdir,
        )

        result = bta.infer("review the system")

        # Each typed factory was used exactly once
        self.assertEqual(used["security"], 1)
        self.assertEqual(used["performance"], 1)
        # Default not used (all queries had a task_type)
        self.assertEqual(used["default"], 0)


# ---------------------------------------------------------------------------
# R4: _parse_json_subtasks parsing variations
# ---------------------------------------------------------------------------


class TestJsonSubtasksParsingVariations(unittest.TestCase):
    """Validates: R4 — _parse_json_subtasks accepts the typical JSON formats
    produced by LLMs (raw JSON, markdown-fenced, with whitespace)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_with_breakdown(self, breakdown_response: str) -> int:
        """Helper: run BTA with given breakdown response and return worker count."""
        breakdown = MockInferencer(response=breakdown_response)
        aggregator = MockInferencer(response="agg")
        worker_count = {"n": 0}

        def factory(sub_query, index):
            worker_count["n"] += 1
            return MockInferencer(response=f"w{index}")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=factory,
            aggregator_inferencer=aggregator,
            breakdown_format="json_subtasks",
            checkpoint_dir=self.tmpdir,
        )
        bta.infer("task")
        return worker_count["n"]

    def test_raw_json_subtasks_object(self):
        """Raw JSON object with subtasks key (no markdown fence)."""
        n = self._run_with_breakdown(
            '{"subtasks": [{"description": "a"}, {"description": "b"}]}'
        )
        self.assertEqual(n, 2)

    def test_markdown_fenced_subtasks_object(self):
        """JSON object with subtasks key inside ```json ... ``` markdown fence."""
        n = self._run_with_breakdown(
            '```json\n{"subtasks": [{"description": "a"}, {"description": "b"}, {"description": "c"}]}\n```'
        )
        self.assertEqual(n, 3)


# ---------------------------------------------------------------------------
# R4: expand_todos_to_workers — one subtask with N todos → N workers
# ---------------------------------------------------------------------------


class TestExpandTodosToWorkers(unittest.TestCase):
    """Validates: R4 — expand_todos_to_workers fans out one subtask's
    todos into N independent workers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_one_subtask_with_three_todos_makes_three_workers(self):
        # Single subtask with 3 todos — must be inside {"subtasks": [...]}
        subtasks_obj = {
            "subtasks": [
                {
                    "description": "Implement feature X",
                    "todos": ["Read api.py", "Read models.py", "Read views.py"],
                }
            ]
        }
        breakdown_json = f"```json\n{json.dumps(subtasks_obj)}\n```"
        breakdown = MockInferencer(response=breakdown_json)

        # Each worker captures its query
        worker_inputs = []

        def factory(sub_query, index):
            return MockInferencer(response=f"done_{index}_{sub_query}")

        # Wrap factory to capture queries via closure
        original_factory = factory
        def tracked_factory(sub_query, index):
            worker_inputs.append(sub_query)
            return original_factory(sub_query, index)

        aggregator = MockInferencer(response="agg")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_factory=tracked_factory,
            aggregator_inferencer=aggregator,
            breakdown_format="json_subtasks",
            expand_todos_to_workers=True,
            checkpoint_dir=self.tmpdir,
        )
        bta.infer("task")

        # 3 workers should have been created (one per todo)
        self.assertEqual(
            len(worker_inputs), 3,
            f"Expected 3 workers (one per todo), got {len(worker_inputs)}: {worker_inputs}",
        )


if __name__ == "__main__":
    unittest.main()

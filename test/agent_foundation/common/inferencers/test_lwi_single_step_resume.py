"""A one-step LWI with a workspace runs, and resumes from its saved step.

A workspace turns on ``resume_with_saved_results=True``; the workflow's
backward resume scan must start at the last step, not at index ``True == 1``.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


class SingleStepResumeTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.runs = []

    def single_step_lwi(self):
        def work(step_input, state):
            self.runs.append(step_input)
            return f"done {len(self.runs)}"

        return LinearWorkflowInferencer(
            step_configs=[
                WorkflowStepConfig(name="work", step_fn=work, output_state_key="out")
            ],
            response_builder=lambda state: state["out"],
            workspace=InferencerWorkspace(root=self.root),
        )

    def test_completes(self):
        self.assertEqual(self.single_step_lwi().infer("go"), "done 1")

    def test_second_run_resumes_the_saved_step(self):
        self.single_step_lwi().infer("go")
        self.assertEqual(self.single_step_lwi().infer("go"), "done 1")
        self.assertEqual(len(self.runs), 1)


if __name__ == "__main__":
    unittest.main()

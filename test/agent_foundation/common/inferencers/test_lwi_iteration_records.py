"""The default LWI iteration record snapshots state without the record list.

A record holding ``iteration_records`` would contain the list it is appended
to; the final-result save then recurses on the cycle whenever the LWI has a
workspace.
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


class DefaultIterationRecordTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def looping_lwi(self, captured):
        runs = [0]

        def work(step_input, state):
            runs[0] += 1
            return f"r{runs[0]}"

        def check(step_input, state):
            if runs[0] < 3:
                state["iteration"] += 1
            return "checked"

        def respond(state):
            captured.update(state)
            return state["out"]

        return LinearWorkflowInferencer(
            step_configs=[
                WorkflowStepConfig(name="work", step_fn=work, output_state_key="out"),
                WorkflowStepConfig(
                    name="check",
                    step_fn=check,
                    output_state_key="chk",
                    loop_back_to="work",
                    loop_condition=lambda state, result: runs[0] < 3,
                    max_loop_iterations=5,
                ),
            ],
            response_builder=respond,
            workspace=InferencerWorkspace(root=self.root),
        )

    def test_looping_lwi_with_workspace_completes(self):
        captured = {}
        self.assertEqual(self.looping_lwi(captured).infer("go"), "r3")
        self.assertEqual(len(captured["iteration_records"]), 2)

    def test_records_do_not_contain_the_record_list(self):
        captured = {}
        self.looping_lwi(captured).infer("go")
        for record in captured["iteration_records"]:
            self.assertNotIn("iteration_records", record)
            self.assertIn("out", record)


if __name__ == "__main__":
    unittest.main()

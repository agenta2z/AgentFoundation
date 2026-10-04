"""Each dynamic LWI step runs in its own ``children/initial|round{NN}`` workspace.

The followup instance serves every step >= 1. An LWI given a constructor
workspace pins that instance to ``children/round01`` (legacy propagation); one
whose workspace comes from the run-context leaves it unbacked. Both must write
each round to its own directory and surface the last round to the flow's
``outputs/``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext
from attr import attrib, attrs


@attrs
class _CountingLeaf(InferencerBase):
    """Answers ``<Response>{label} {n}</Response>`` on its n-th call."""

    label = attrib(default="step")
    calls = attrib(default=0, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.calls += 1
        return f"<Response>{self.label} {self.calls}</Response>"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class DynamicStepWorkspaceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    async def run_three_steps(self, *, pinned):
        ws = InferencerWorkspace(root=self.root)
        lwi = LinearWorkflowInferencer(
            workspace=ws if pinned else None,
            dynamic_mode=True,
            default_initial_inferencer=_CountingLeaf(
                label="INITIAL", output_path="output.md"
            ),
            default_followup_inferencer=_CountingLeaf(
                label="FOLLOWUP", output_path="output.md"
            ),
            end_condition=lambda state, _: len(state["dynamic_step_results"]) >= 3,
            max_dynamic_steps=3,
            output_path="output.md",
        )
        await lwi.ainfer("task", run_context=None if pinned else RunContext.root(ws))

    def step_outputs(self):
        children = os.path.join(self.root, "children")
        return {
            name: _read(os.path.join(children, name, "outputs", "output.md"))
            for name in sorted(os.listdir(children))
        }

    def assert_rounds_kept_apart(self):
        self.assertEqual(
            self.step_outputs(),
            {"initial": "INITIAL 1", "round01": "FOLLOWUP 1", "round02": "FOLLOWUP 2"},
        )
        self.assertEqual(
            _read(os.path.join(self.root, "outputs", "output.md")), "FOLLOWUP 2"
        )
        self.assertTrue(
            os.listdir(
                os.path.join(self.root, "children", "round02", "logs", "session")
            )
        )

    async def test_constructor_workspace(self):
        await self.run_three_steps(pinned=True)
        self.assert_rounds_kept_apart()

    async def test_context_workspace(self):
        await self.run_three_steps(pinned=False)
        self.assert_rounds_kept_apart()


if __name__ == "__main__":
    unittest.main()

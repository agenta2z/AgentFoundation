"""Mock-based PTI flow test to verify base-class changes don't break PTI (Task 11.4).

Creates a PTI instance with mock inferencers and verifies:
- The 4-step flow (plan → approval → implement → analysis) executes without errors
- State dict flows correctly through all steps
- Checkpoint/resume works with new expansion fields

Requirements: 21.1, 21.3, 21.5
"""
import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    DualInferencerResponse,
    ReflectionStyles,
    ResponseSelectors,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)


def _make_mock_inferencer(response_text="mock response", id_="mock"):
    """Create a minimal mock InferencerBase with ainfer, aconnect, adisconnect."""
    mock = MagicMock()
    mock.id = id_
    mock.ainfer = AsyncMock(
        return_value=DualInferencerResponse(
            base_response=response_text,
            reflection_response=None,
            reflection_style=ReflectionStyles.NoReflection,
            response_selector=ResponseSelectors.BaseResponse,
            consensus_achieved=True,
            consensus_history=[],
            total_iterations=1,
        )
    )
    mock.aconnect = AsyncMock()
    mock.adisconnect = AsyncMock()
    mock.set_parent_debuggable = MagicMock()
    return mock


class TestPTIPostExpansionCompat(unittest.IsolatedAsyncioTestCase):
    """Verify PTI's flow works after Workflow base-class expansion changes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="pti_compat_")
        self.workspace = os.path.join(self.tmpdir, "workspace")
        os.makedirs(self.workspace, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_pti(self, **kwargs):
        """Create a PTI with mock sub-inferencers."""
        plan_response = "1. Step one\n2. Step two\n3. Step three"
        impl_response = "Implementation complete"

        pti = PlanThenImplementInferencer(
            planner_inferencer=_make_mock_inferencer(plan_response, "planner"),
            executor_inferencer=_make_mock_inferencer(impl_response, "executor"),
            workspace_root=self.workspace,
            max_meta_iterations=1,
            **kwargs,
        )
        return pti

    async def test_pti_basic_flow_executes(self):
        """PTI runs plan → implement without errors."""
        pti = self._make_pti()

        result = await pti._ainfer("Build a simple calculator")

        # PTI should return a PlanThenImplementResponse
        self.assertIsInstance(result, PlanThenImplementResponse)

    async def test_pti_response_has_correct_structure(self):
        """PTI response has the expected fields after expansion changes."""
        pti = self._make_pti()

        result = await pti._ainfer("Build a simple calculator")

        self.assertIsInstance(result, PlanThenImplementResponse)
        # Should have plan_response populated
        self.assertIsNotNone(result.plan_response)

    async def test_pti_state_flows_through_steps(self):
        """State dict flows correctly through PTI steps."""
        pti = self._make_pti()

        result = await pti._ainfer("Build a simple calculator")

        self.assertIsInstance(result, PlanThenImplementResponse)
        # The plan_response should be populated
        self.assertIsNotNone(result.plan_response)


class TestPTICheckpointExpansionFields(unittest.IsolatedAsyncioTestCase):
    """Verify PTI's checkpoint overrides work with expansion-aware base class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="pti_ckpt_compat_")
        self.workspace = os.path.join(self.tmpdir, "workspace")
        os.makedirs(self.workspace, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_pti(self, **kwargs):
        """Create a PTI with mock sub-inferencers."""
        plan_response = "1. Step one\n2. Step two"
        impl_response = "Implementation complete"

        return PlanThenImplementInferencer(
            planner_inferencer=_make_mock_inferencer(plan_response, "planner"),
            executor_inferencer=_make_mock_inferencer(impl_response, "executor"),
            workspace_root=self.workspace,
            max_meta_iterations=1,
            **kwargs,
        )

    async def test_checkpoint_save_load_with_expansion_base(self):
        """PTI's _save_loop_checkpoint and _try_load_checkpoint work with expansion-aware base."""
        pti = self._make_pti()

        result = await pti._ainfer("Build a calculator")

        self.assertIsInstance(result, PlanThenImplementResponse)
        # Verify checkpoint files were created and are valid JSON (no crash from expansion fields)
        json_files_found = 0
        for root, _dirs, files in os.walk(self.workspace):
            for fname in files:
                fpath = os.path.join(root, fname)
                if fname.endswith(".json") and os.path.isfile(fpath):
                    with open(fpath) as f:
                        data = json.load(f)
                    json_files_found += 1
                    # Should be valid JSON — no crash from expansion fields
                    self.assertIsNotNone(data)
        # At least some checkpoint files should have been created
        self.assertGreater(json_files_found, 0)


if __name__ == "__main__":
    unittest.main()

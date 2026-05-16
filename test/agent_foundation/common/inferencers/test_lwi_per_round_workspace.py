"""Tests for Part D — Per-round workspace in LWI dynamic mode (hierarchical layout).

The hierarchical layout uses ``self._workspace.child(f"round{step_index:02d}")``
for step_index >= 2.  Steps 0 and 1 use pre-assigned workspaces from LWI's
``_propagate_workspace_to_children`` override (``children/initial/`` and
``children/round01/``).

Implementation reference:
- agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/linear_workflow_inferencer.py
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock


class _MockWorkspace:
    """Workspace that tracks child() calls to verify nesting (or lack of)."""

    def __init__(self, name: str, parent: "_MockWorkspace" = None):
        self.name = name
        self.parent = parent
        self.children = {}
        self.ensure_dirs_called = False

    @property
    def path(self):
        if self.parent is None:
            return self.name
        return f"{self.parent.path}/children/{self.name}"

    def child(self, child_name: str) -> "_MockWorkspace":
        if child_name not in self.children:
            self.children[child_name] = _MockWorkspace(child_name, parent=self)
        return self.children[child_name]

    def ensure_dirs(self):
        self.ensure_dirs_called = True


class _SimulateHierarchicalStep:
    """Simulate the per-round assignment logic from _build_dynamic_step_wrapper.

    Mirrors the hierarchical layout code in linear_workflow_inferencer.py:
    - step 0 and 1 use pre-assigned workspaces (no action)
    - step 2+ creates children of the LWI root via self._workspace.child()
    """

    @staticmethod
    def execute_step(lwi_workspace, inf_instance, step_index, state=None):
        """Run the Part D code block from the production code.

        Args:
            lwi_workspace: The LWI's own workspace (``self._workspace``).
            inf_instance: The inferencer instance for this step.
            step_index: The dynamic step index.
            state: Optional state dict for consensus_iteration_id.
        """
        if inf_instance is not None and step_index >= 2:
            lwi_ws = lwi_workspace
            if lwi_ws is not None:
                consensus_iter = state.get("consensus_iteration_id", 0) if state else 0
                iter_suffix = f"_iter{consensus_iter}" if consensus_iter > 0 else ""
                child_name = f"round{step_index:02d}{iter_suffix}"
                round_ws = lwi_ws.child(child_name)
                if hasattr(round_ws, "ensure_dirs"):
                    round_ws.ensure_dirs()
                inf_instance._workspace = round_ws
                if hasattr(inf_instance, "reset_session"):
                    inf_instance.reset_session()


class TestPerRoundWorkspace(unittest.TestCase):
    """Tests for Part D per-round workspace logic (hierarchical layout)."""

    def test_step_0_no_workspace_change(self):
        """Step 0 (initial) does NOT trigger per-round workspace logic."""
        lwi_ws = _MockWorkspace("flow_0")
        inf = MagicMock()
        inf._workspace = _MockWorkspace("initial")
        _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=0)
        self.assertEqual(inf._workspace.name, "initial")

    def test_step_1_no_workspace_change(self):
        """Step 1 (first followup) does NOT trigger per-round workspace logic."""
        lwi_ws = _MockWorkspace("flow_0")
        inf = MagicMock()
        inf._workspace = _MockWorkspace("round01")
        _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=1)
        self.assertEqual(inf._workspace.name, "round01")

    def test_step_2_creates_round02_under_lwi_root(self):
        """Step 2 creates round02 as child of the LWI workspace."""
        lwi_ws = _MockWorkspace("flow_0")
        inf = MagicMock()
        inf._workspace = _MockWorkspace("round01")
        _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=2)
        self.assertEqual(inf._workspace.name, "round02")
        self.assertIs(inf._workspace.parent, lwi_ws)
        self.assertTrue(inf._workspace.ensure_dirs_called)

    def test_rounds_are_siblings_under_lwi_root(self):
        """CRITICAL: round02, round03 are siblings under flow_0/, not nested."""
        lwi_ws = _MockWorkspace("flow_0")
        inf = MagicMock()
        inf._workspace = _MockWorkspace("round01")
        # Step 2
        _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=2)
        self.assertEqual(inf._workspace.name, "round02")
        # Step 3
        _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=3)
        self.assertEqual(inf._workspace.name, "round03")
        # Both are direct children of lwi_ws
        self.assertIn("round02", lwi_ws.children)
        self.assertIn("round03", lwi_ws.children)
        self.assertIs(lwi_ws.children["round02"].parent, lwi_ws)
        self.assertIs(lwi_ws.children["round03"].parent, lwi_ws)
        # No nesting
        self.assertEqual(lwi_ws.children["round02"].children, {})

    def test_clean_numbering_no_gaps(self):
        """Hierarchical approach: round02, round03, round04 — no gaps."""
        lwi_ws = _MockWorkspace("flow_0")
        inf = MagicMock()
        inf._workspace = _MockWorkspace("round01")
        for step in (2, 3, 4):
            _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=step)
        self.assertIn("round02", lwi_ws.children)
        self.assertIn("round03", lwi_ws.children)
        self.assertIn("round04", lwi_ws.children)

    def test_no_op_when_inf_instance_is_none(self):
        """Should not raise when inf_instance is None."""
        lwi_ws = _MockWorkspace("flow_0")
        _SimulateHierarchicalStep.execute_step(lwi_ws, None, step_index=2)

    def test_no_op_when_lwi_workspace_is_none(self):
        """Should not raise when LWI workspace is None."""
        inf = MagicMock()
        _SimulateHierarchicalStep.execute_step(None, inf, step_index=2)

    def test_reset_session_called_each_round(self):
        """Each round triggers reset_session for fresh LLM context."""
        lwi_ws = _MockWorkspace("flow_0")
        inf = MagicMock()
        inf._workspace = _MockWorkspace("round01")
        _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=2)
        _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=3)
        self.assertEqual(inf.reset_session.call_count, 2)

    def test_consensus_iteration_suffix(self):
        """Multi-iteration: round02_iter1 for consensus_iteration_id=1."""
        lwi_ws = _MockWorkspace("flow_0")
        inf = MagicMock()
        inf._workspace = _MockWorkspace("round01")
        state = {"consensus_iteration_id": 1}
        _SimulateHierarchicalStep.execute_step(lwi_ws, inf, step_index=2, state=state)
        self.assertEqual(inf._workspace.name, "round02_iter1")
        self.assertIs(inf._workspace.parent, lwi_ws)


class TestActualLWIIntegration(unittest.TestCase):
    """Integration: verify the actual LWI module has the hierarchical logic."""

    def test_hierarchical_logic_present_in_source(self):
        """Sanity check that the actual production code contains the
        hierarchical block with step_index >= 2 guard and self._workspace.child()."""
        import inspect
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
            linear_workflow_inferencer as lwi_module,
        )
        source = inspect.getsource(lwi_module.LinearWorkflowInferencer._build_dynamic_step_wrapper)
        self.assertIn("step_index >= 2", source,
                      "step_index >= 2 guard missing")
        self.assertNotIn("_base_followup_workspace", source,
                         "Stale _base_followup_workspace stash still present")

    def test_lwi_propagation_override_exists(self):
        """LWI must have _propagate_workspace_to_children override."""
        import inspect
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
            linear_workflow_inferencer as lwi_module,
        )
        source = inspect.getsource(lwi_module.LinearWorkflowInferencer._propagate_workspace_to_children)
        self.assertIn("initial", source, "initial child name missing")
        self.assertIn("round01", source, "round01 child name missing")

    def test_lwi_workspace_propagation_skip(self):
        """LWI must skip default_initial_inferencer and default_followup_inferencer."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
            LinearWorkflowInferencer,
        )
        skip = LinearWorkflowInferencer._workspace_propagation_skip
        self.assertIn("default_initial_inferencer", skip)
        self.assertIn("default_followup_inferencer", skip)


if __name__ == "__main__":
    unittest.main()

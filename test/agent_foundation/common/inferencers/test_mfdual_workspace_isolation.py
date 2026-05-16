"""Tests for Part B — Workspace isolation for runtime role swaps.

When MFDual reuses an instance across roles (winner-as-fixer, loser-as-reviewer),
the helper assigns a fresh role-named workspace + resets the session. The
identity guard protects the originally-configured instances from clobbering.

Implementation reference:
- agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_dual_inferencer.py
- Plan: _docs/_plans/mfdual_hygiene_INTEGRATED_plan.md (Part B)
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, sentinel

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)


def _make_workspace_with_child(child_ws):
    ws = MagicMock()
    ws.child.return_value = child_ws
    return ws


def _make_role_workspace():
    """A role-named workspace with ensure_dirs() and a path attribute."""
    ws = MagicMock()
    ws.ensure_dirs = MagicMock()
    return ws


def _make_mfdual_stub():
    """Construct a minimal MFDual stub bypassing attrs init."""
    obj = MultiFlowDualInferencer.__new__(MultiFlowDualInferencer)
    obj._workspace = None
    obj._fixer_inferencer_original = None
    obj._review_inferencer_original = None
    return obj


class TestReassignRoleWorkspace(unittest.TestCase):
    """Tests for _reassign_role_workspace (Phase B1+B2)."""

    def test_no_op_when_inferencer_none(self):
        mfd = _make_mfdual_stub()
        mfd._workspace = MagicMock()
        # Should not raise
        mfd._reassign_role_workspace(None, "fixer_inferencer")

    def test_no_op_when_workspace_none(self):
        mfd = _make_mfdual_stub()
        inferencer = MagicMock()
        mfd._reassign_role_workspace(inferencer, "fixer_inferencer")
        # Workspace should not have been touched
        self.assertNotIn("_workspace", inferencer.__dict__)

    def test_identity_guard_skips_original_fixer(self):
        """When candidate IS the original configured fixer, skip reassignment."""
        mfd = _make_mfdual_stub()
        original = MagicMock(name="original_fixer")
        mfd._fixer_inferencer_original = original
        role_ws = _make_role_workspace()
        mfd._workspace = _make_workspace_with_child(role_ws)
        # Pass the SAME identity → identity guard fires
        mfd._reassign_role_workspace(original, "fixer_inferencer")
        # workspace.child should NOT have been called (skipped)
        mfd._workspace.child.assert_not_called()

    def test_identity_guard_skips_original_reviewer(self):
        mfd = _make_mfdual_stub()
        original = MagicMock(name="original_reviewer")
        mfd._review_inferencer_original = original
        mfd._workspace = MagicMock()
        mfd._reassign_role_workspace(original, "review_inferencer")
        mfd._workspace.child.assert_not_called()

    def test_reassigns_workspace_for_new_inferencer(self):
        """Different identity → switch_role is called with role workspace."""
        mfd = _make_mfdual_stub()
        mfd._fixer_inferencer_original = MagicMock(name="original")
        new_inf = MagicMock(name="new_fixer")
        role_ws = _make_role_workspace()
        mfd._workspace = _make_workspace_with_child(role_ws)
        mfd._reassign_role_workspace(new_inf, "fixer_inferencer")
        # Verify workspace.child was called and switch_role was invoked
        mfd._workspace.child.assert_called_once_with("fixer_inferencer")
        role_ws.ensure_dirs.assert_called_once()
        new_inf.switch_role.assert_called_once()
        call_kwargs = new_inf.switch_role.call_args
        self.assertEqual(call_kwargs.kwargs.get("new_role") or call_kwargs.args[0], "fixer_inferencer")
        self.assertIs(call_kwargs.kwargs["workspace"], role_ws)
        self.assertTrue(call_kwargs.kwargs["output_is_deliverable"])

    def test_calls_reset_session_on_new_inferencer(self):
        """switch_role is called (which internally calls reset_session)."""
        mfd = _make_mfdual_stub()
        mfd._fixer_inferencer_original = MagicMock(name="original")
        new_inf = MagicMock(name="new_fixer")
        role_ws = _make_role_workspace()
        mfd._workspace = _make_workspace_with_child(role_ws)
        mfd._reassign_role_workspace(new_inf, "fixer_inferencer")
        new_inf.switch_role.assert_called_once()

    def test_uses_correct_role_name_for_review(self):
        mfd = _make_mfdual_stub()
        mfd._review_inferencer_original = MagicMock(name="original")
        new_inf = MagicMock(name="new_reviewer")
        role_ws = _make_role_workspace()
        mfd._workspace = _make_workspace_with_child(role_ws)
        mfd._reassign_role_workspace(new_inf, "review_inferencer")
        mfd._workspace.child.assert_called_once_with("review_inferencer")

    def test_handles_inferencer_without_reset_session(self):
        """Doesn't crash when inferencer has no reset_session -- switch_role
        handles this internally via hasattr guard."""
        mfd = _make_mfdual_stub()
        mfd._fixer_inferencer_original = MagicMock(name="original")
        new_inf = MagicMock(spec=["_workspace", "switch_role"])  # no reset_session
        role_ws = _make_role_workspace()
        mfd._workspace = _make_workspace_with_child(role_ws)
        # Should not raise — switch_role is called, which handles reset_session internally
        mfd._reassign_role_workspace(new_inf, "fixer_inferencer")
        new_inf.switch_role.assert_called_once()


if __name__ == "__main__":
    unittest.main()

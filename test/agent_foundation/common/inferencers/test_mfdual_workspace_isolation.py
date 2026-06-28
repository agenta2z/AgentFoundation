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
        # Workspace dir is NOT created at propose time (deferred to the round step).
        # switch_role is invoked for template/session reset only, no workspace arg.
        mfd._workspace.child.assert_not_called()
        new_inf.switch_role.assert_called_once()
        call_kwargs = new_inf.switch_role.call_args
        self.assertEqual(call_kwargs.kwargs.get("new_role") or call_kwargs.args[0], "fixer_inferencer")
        self.assertNotIn("workspace", call_kwargs.kwargs)
        self.assertTrue(call_kwargs.kwargs["output_is_deliverable"])

    def test_role_switch_runs_under_role_child_ctx_not_worker_node(self):
        """§2.7 regression: the role-switch must run under the role's OWN
        ``./review`` / ``./fix`` child ctx — matching the dispatch's
        ``run_context=self._rc_child("review"/"fix")`` — NOT the active worker
        node.

        Before the fix, ``switch_role`` ran with the worker node active (it is
        called from ``_step_propose_impl``, a step_fn that pushes no child ctx),
        so ``_record_role_state`` tagged the worker path with the reviewer
        (runner-up) leaf's creator and then the fixer (winner) leaf — a DIFFERENT
        creator — re-tagged the same path, tripping the collision guard on the
        consensus re-run. Scoping each role-switch to its own child node makes the
        two distinct-creator leaves land on distinct nodes (no collision).
        """
        from agent_foundation.common.inferencers.run_context import (
            RunContext,
            RunStateStore,
            active_run_context,
            enter_run,
            exit_run,
        )

        captured = {}

        def _capture(new_role=None, **kw):
            ctx = active_run_context()
            captured[new_role] = ctx.path if ctx is not None else None

        mfd = _make_mfdual_stub()
        mfd._workspace = _make_workspace_with_child(_make_role_workspace())
        # distinct "originals" so the identity guard does NOT skip our candidates
        mfd._review_inferencer_original = MagicMock(name="orig_reviewer")
        mfd._fixer_inferencer_original = MagicMock(name="orig_fixer")

        reviewer = MagicMock(name="runner_up_leaf")
        reviewer.switch_role.side_effect = _capture
        fixer = MagicMock(name="winner_leaf")
        fixer.switch_role.side_effect = _capture

        worker = (
            RunContext.root(store=RunStateStore())
            .child("propose")
            .child("plan_bta.worker_0")
        )
        token = enter_run(worker)
        try:
            mfd._reassign_role_workspace(reviewer, "review_inferencer")
            mfd._reassign_role_workspace(fixer, "fixer_inferencer")
        finally:
            exit_run(token)

        self.assertEqual(
            captured["review_inferencer"], "/propose/plan_bta.worker_0/review"
        )
        self.assertEqual(
            captured["fixer_inferencer"], "/propose/plan_bta.worker_0/fix"
        )
        # the bug was BOTH landing on the bare worker node:
        self.assertNotEqual(
            captured["review_inferencer"], "/propose/plan_bta.worker_0"
        )

    def test_runtime_reviewer_resolvable_under_review_child_ctx(self):
        """Root-cause regression for the MFDual ``/review`` CollisionError.

        ``_select_reviewer_and_fixer`` writes the runtime-resolved reviewer into
        the WORKER node's scratch via ``_role_set`` (it runs in
        ``_step_propose_impl`` under the worker node). But the §2.7 review
        pre-render reads ``_role_get('review_inferencer')`` while the active ctx is
        the ``./review`` CHILD node. ``_role_get`` is node-scoped (reads
        ``active_run_context().node().scratch`` with NO ancestor walk-up), so under
        ``./review`` it misses the worker-node value and falls back to the STALE
        construction-time ``self.review_inferencer`` placeholder (``flow_configs[1]``
        — the runner-up seed). When that placeholder's class differs from the
        runtime reviewer, the pre-render's ``_effective_role`` claims ``./review``
        with a DIFFERENT creator than the one ``_reassign_role_workspace`` already
        claimed it with (the real reviewer) → ``CollisionError``.

        A reviewer selected for the worker MUST therefore remain resolvable when
        read under the ``./review`` child where the pre-render runs.
        """
        from agent_foundation.common.inferencers.run_context import (
            RunContext,
            RunStateStore,
            enter_run,
            exit_run,
        )

        mfd = _make_mfdual_stub()
        mfd._workspace = _make_workspace_with_child(_make_role_workspace())
        # distinct original → identity guard does NOT skip the runtime reviewer
        mfd._review_inferencer_original = MagicMock(name="orig_reviewer")
        # construction-time STALE placeholder (the runner-up seed) — a different
        # object than the runtime reviewer, mirroring flow_configs[1] != selection.
        mfd.review_inferencer = MagicMock(name="stale_construction_placeholder")

        runtime_reviewer = MagicMock(name="runtime_selected_reviewer")

        worker = (
            RunContext.root(store=RunStateStore())
            .child("propose")
            .child("plan_bta.worker_0")
        )
        tok = enter_run(worker)
        try:
            # selection writes the WORKER node (mirrors _select_reviewer_and_fixer)
            mfd._role_set("review_inferencer", runtime_reviewer, worker)
            # the propose tail reassigns the role's workspace + session
            mfd._reassign_role_workspace(runtime_reviewer, "review_inferencer")
            # the review pre-render reads _role_get UNDER ./review
            review_ctx = mfd._rc_child("review")
            rtok = enter_run(review_ctx)
            try:
                resolved = mfd._role_get("review_inferencer")
            finally:
                exit_run(rtok)
            self.assertIs(
                resolved,
                runtime_reviewer,
                f"_role_get under ./review resolved {resolved!r} instead of the "
                "runtime-selected reviewer — the node-scope fallback to the stale "
                "construction placeholder is the defect that collides at ./review",
            )
        finally:
            exit_run(tok)

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
        mfd._workspace.child.assert_not_called()

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

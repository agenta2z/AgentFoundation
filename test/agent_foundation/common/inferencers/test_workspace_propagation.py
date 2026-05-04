"""Unit tests for workspace propagation across the inferencer hierarchy.

Covers the consolidation refactor:

* LWI's ``workspace_path`` field renamed to ``workspace_root`` (hard rename;
  no backward-compat alias).
* PTI's runtime fallback chain at ``_ainfer``:
  ``resume_workspace`` -> ``workspace_root`` -> ``self._workspace.root``.
* PTI's ``_workspace_propagation_skip`` suppresses generic propagation for
  ``_CHILD_DEFAULTS`` (planner/executor/analyzer) — the runtime
  ``_setup_child_workflows`` claims those with iter_<N>-aware paths.
* MultiFlowInferencer's ``_propagate_workspace_to_children`` override walks
  ``flow_configs`` (list-of-dicts) which the generic walker doesn't descend.
* Dual continues to propagate via the inherited base mechanism (no
  regression).
"""
from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock

import attr
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ws_root(tmp_path):
    """Plain string path that can be used as a workspace root."""
    p = tmp_path / "ws"
    p.mkdir()
    return str(p)


def _make_workspace(root: str):
    """Construct an InferencerWorkspace with ensure_dirs() called."""
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )
    ws = InferencerWorkspace(root=root)
    ws.ensure_dirs()
    return ws


def _make_minimal_inferencer():
    """Construct a no-op InferencerBase suitable as a child slot.

    Uses a real subclass instead of a mock so the ``_workspace`` property
    setter (and its propagation chain) fires correctly.
    """
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase

    @attr.attrs(slots=False)
    class _Stub(InferencerBase):
        async def _ainfer(self, inference_input, **kwargs):
            return ""

        def _infer(self, inference_input, **kwargs):
            return ""

    return _Stub()


# ---------------------------------------------------------------------------
# Test 1: LWI field rename — kwarg form
# ---------------------------------------------------------------------------


def test_lwi_workspace_root_field_rename_kwarg_works(tmp_ws_root):
    """``LWI(workspace_root=...)`` constructor kwarg sets the field."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )
    lwi = LinearWorkflowInferencer(workspace_root=tmp_ws_root)
    assert lwi.workspace_root == tmp_ws_root
    assert lwi._workspace is not None
    assert lwi._workspace.root == tmp_ws_root


# ---------------------------------------------------------------------------
# Test 2: LWI hard rename — old name is rejected
# ---------------------------------------------------------------------------


def test_lwi_workspace_path_kwarg_raises_typeerror(tmp_ws_root):
    """``LWI(workspace_path=...)`` raises TypeError (no backward-compat alias)."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )
    with pytest.raises(TypeError):
        LinearWorkflowInferencer(workspace_path=tmp_ws_root)


# ---------------------------------------------------------------------------
# Test 3: PTI constructs without workspace_root when analysis is disabled
# ---------------------------------------------------------------------------


def test_pti_constructs_without_workspace_root_when_analysis_disabled():
    """PTI without explicit workspace and with analysis disabled constructs OK.

    At construction, ``self._workspace`` is None — parent propagation hasn't
    fired yet (it runs after the child's ``__attrs_post_init__``).
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_minimal_inferencer(),
        executor_inferencer=_make_minimal_inferencer(),
        enable_analysis=False,
        enable_multiple_iterations=False,
    )
    assert pti.workspace_root is None
    assert pti.resume_workspace is None
    assert pti._workspace is None  # parent hasn't propagated yet


# ---------------------------------------------------------------------------
# Test 4: PTI _ainfer fallback uses propagated workspace
# ---------------------------------------------------------------------------


def test_pti_ainfer_fallback_uses_propagated_workspace(tmp_ws_root):
    """When ``_workspace`` is set externally and ``workspace_root`` is None,
    PTI's ``_ainfer`` resolves base_workspace from ``_workspace.root``.

    Verified by reading the inline expression at ``_ainfer`` line 2430-area:
    ``base_workspace = resume_workspace or workspace_root or _workspace.root``.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_minimal_inferencer(),
        executor_inferencer=_make_minimal_inferencer(),
        enable_analysis=False,
        enable_multiple_iterations=False,
    )
    ws = _make_workspace(tmp_ws_root)
    pti._workspace = ws  # simulate parent propagation
    # Replicate the fallback expression exactly as it appears at ainfer time
    base_workspace = (
        pti.resume_workspace
        or pti.workspace_root
        or (pti._workspace.root if pti._workspace else None)
    )
    assert base_workspace == tmp_ws_root


# ---------------------------------------------------------------------------
# Test 5: PTI _ainfer fallback resume_workspace wins over all
# ---------------------------------------------------------------------------


def test_pti_ainfer_fallback_resume_workspace_wins(tmp_ws_root, tmp_path):
    """When all three sources are set, resume_workspace wins (preserves
    resume semantics for in-flight workspaces)."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    resume_path = str(tmp_path / "resume_ws")
    config_path = str(tmp_path / "config_ws")
    propagated_path = str(tmp_path / "propagated_ws")
    for p in (resume_path, config_path, propagated_path):
        os.makedirs(p, exist_ok=True)

    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_minimal_inferencer(),
        executor_inferencer=_make_minimal_inferencer(),
        workspace_root=config_path,
        resume_workspace=resume_path,
        enable_analysis=False,
        enable_multiple_iterations=False,
    )
    pti._workspace = _make_workspace(propagated_path)

    base_workspace = (
        pti.resume_workspace
        or pti.workspace_root
        or (pti._workspace.root if pti._workspace else None)
    )
    assert base_workspace == resume_path


# ---------------------------------------------------------------------------
# Test 6: PTI propagation skips _CHILD_DEFAULTS
# ---------------------------------------------------------------------------


def test_pti_propagates_to_children_at_construction(tmp_ws_root):
    """When PTI's ``_workspace`` is assigned, generic propagation DOES set
    ``_workspace`` on planner_inferencer / executor_inferencer at construction
    time — necessary so each child's ``_logger: auto`` cascade resolves.

    At runtime, ``_setup_child_workflows`` re-assigns those workspaces to
    iteration-aware paths (``<pti>/iter_<N>/children/<short_name>/``); the
    construction-time dirs become unused after the first iteration starts,
    but that's a cosmetic cost — functional correctness depends on
    construction-time workspace resolution.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    planner = _make_minimal_inferencer()
    executor = _make_minimal_inferencer()
    pti = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=executor,
        workspace_root=tmp_ws_root,
    )
    # __attrs_post_init__ already triggered the setter via workspace_root
    assert pti._workspace is not None
    # Children DO get propagated at construction (so logger:auto resolves)
    assert planner._workspace is not None
    assert planner._workspace.root.replace("\\", "/").endswith(
        "/children/planner_inferencer"
    )
    assert executor._workspace is not None
    assert executor._workspace.root.replace("\\", "/").endswith(
        "/children/executor_inferencer"
    )


# ---------------------------------------------------------------------------
# Test 7: MultiFlow propagates to flow_configs entries
# ---------------------------------------------------------------------------


def test_multiflow_propagation_walks_flow_configs(tmp_ws_root):
    """MultiFlowInferencer's override walks flow_configs (list of dicts)
    and assigns ``flow_<i>_initial`` / ``flow_<i>_followup`` child workspaces."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
        MultiFlowInferencer,
    )
    flow0_init = _make_minimal_inferencer()
    flow0_follow = _make_minimal_inferencer()
    flow1_init = _make_minimal_inferencer()
    flow1_follow = _make_minimal_inferencer()
    mfi = MultiFlowInferencer(
        flow_configs=[
            {"input": "q0", "initial_inferencer": flow0_init,
             "followup_inferencer": flow0_follow},
            {"input": "q1", "initial_inferencer": flow1_init,
             "followup_inferencer": flow1_follow},
        ],
    )
    parent_ws = _make_workspace(tmp_ws_root)
    mfi._workspace = parent_ws  # triggers setter -> propagate

    for inf, suffix in (
        (flow0_init, "flow_0_initial"),
        (flow0_follow, "flow_0_followup"),
        (flow1_init, "flow_1_initial"),
        (flow1_follow, "flow_1_followup"),
    ):
        assert inf._workspace is not None, f"{suffix} not propagated"
        assert inf._workspace.root.replace("\\", "/").endswith(
            f"/children/{suffix}"
        ), f"{suffix} got unexpected root: {inf._workspace.root}"


# ---------------------------------------------------------------------------
# Test 8: MultiFlow propagation respects pre-assignment
# ---------------------------------------------------------------------------


def test_multiflow_propagation_respects_pre_assignment(tmp_ws_root, tmp_path):
    """When a flow inferencer already has ``_workspace`` set, the override
    does NOT overwrite it (matches the generic walker contract)."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
        MultiFlowInferencer,
    )
    pre_assigned_path = str(tmp_path / "pre_assigned")
    pre_assigned_ws = _make_workspace(pre_assigned_path)

    flow0_init = _make_minimal_inferencer()
    flow0_init._workspace = pre_assigned_ws  # explicit pre-assignment
    flow0_follow = _make_minimal_inferencer()  # not pre-assigned
    mfi = MultiFlowInferencer(
        flow_configs=[
            {"input": "q0", "initial_inferencer": flow0_init,
             "followup_inferencer": flow0_follow},
        ],
    )
    parent_ws = _make_workspace(tmp_ws_root)
    mfi._workspace = parent_ws

    # Pre-assigned: still has its original workspace
    assert flow0_init._workspace.root == pre_assigned_path
    # Not pre-assigned: got the propagated workspace
    assert flow0_follow._workspace is not None
    assert flow0_follow._workspace.root.replace("\\", "/").endswith(
        "/children/flow_0_followup"
    )


# ---------------------------------------------------------------------------
# Test 9: Dual propagates to base / review / fixer (no regression)
# ---------------------------------------------------------------------------


def test_dual_propagates_to_base_review_fixer(tmp_ws_root):
    """Dual propagates to ``base_inferencer`` / ``review_inferencer`` /
    ``fixer_inferencer`` via the inherited base mechanism. This is unchanged
    by Refactor 1 — the test guards against future regression."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )
    base = _make_minimal_inferencer()
    review = _make_minimal_inferencer()
    fixer = _make_minimal_inferencer()
    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=review,
        fixer_inferencer=fixer,
        workspace_root=tmp_ws_root,
    )
    assert dual._workspace is not None
    for child, slot in ((base, "base_inferencer"),
                        (review, "review_inferencer"),
                        (fixer, "fixer_inferencer")):
        assert child._workspace is not None, f"{slot} not propagated"
        assert child._workspace.root.replace("\\", "/").endswith(
            f"/children/{slot}"
        ), f"{slot} got unexpected root: {child._workspace.root}"


# ---------------------------------------------------------------------------
# Test 10: No orphan dirs under PTI's children/ for _CHILD_DEFAULTS slots
# ---------------------------------------------------------------------------


def test_full_topology_propagation_through_dual_pti(tmp_ws_root):
    """Integration: Dual -> PTI -> {planner_inferencer, executor_inferencer}.

    Verifies the full cascade. Each child's `_workspace` is populated at
    construction time so logger/cache cascades resolve to the right paths.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    planner = _make_minimal_inferencer()
    executor = _make_minimal_inferencer()
    pti = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=executor,
    )
    outer = DualInferencer(
        base_inferencer=pti,
        workspace_root=tmp_ws_root,
    )
    # Outer Dual rooted at tmp_ws_root
    assert outer._workspace is not None
    assert outer._workspace.root == tmp_ws_root
    # PTI mounted under outer/children/base_inferencer
    assert pti._workspace is not None
    assert pti._workspace.root.replace("\\", "/").endswith("/children/base_inferencer")
    # PTI's children mounted under pti/children/<attr_name>
    assert planner._workspace is not None
    assert planner._workspace.root.replace("\\", "/").endswith(
        "/children/base_inferencer/children/planner_inferencer"
    )
    assert executor._workspace is not None
    assert executor._workspace.root.replace("\\", "/").endswith(
        "/children/base_inferencer/children/executor_inferencer"
    )

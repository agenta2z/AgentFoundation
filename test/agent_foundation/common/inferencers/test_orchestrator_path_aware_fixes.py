"""Tests for the orchestrator path-aware infrastructure that was kept after
the speculative-injection revert.

Scope:
  * MultiFlowInferencer._resolve_flow_output_path delegates to the shared
    resolve_canonical_output_path helper (path resolution behavior).
  * resolve_canonical_output_path helper integration smoke tests.

NOT in scope (intentionally removed):
  * BTA worker_output_paths injection — reverted (no template consumed it).
  * MFI structured worker_output_paths into aggregator extra_feed — reverted.
  * PTI plan_output_path injection — reverted.
  * Reflective base_response_path injection — reverted.
  * LWI dynamic_step_output_paths state channel — reverted.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional
from unittest.mock import MagicMock

import pytest

from agent_foundation.common.inferencers.inferencer_workspace import (
    InferencerWorkspace,
    resolve_canonical_output_path,
)


# ----------------------------------------------------------------------
# Helper: build a workspace whose outputs/output.md exists
# ----------------------------------------------------------------------
def _make_ws_with_output(tmpdir: str, content: str = "X") -> InferencerWorkspace:
    ws = InferencerWorkspace(root=tmpdir, use_final_deliverables_folder=True)
    ws.ensure_dirs()
    out_path = ws.output_path("output.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    return ws


def _make_ws_with_deliverable(tmpdir: str, content: str = "X") -> InferencerWorkspace:
    ws = InferencerWorkspace(root=tmpdir, use_final_deliverables_folder=True)
    ws.ensure_dirs()
    deliv_path = ws.deliverable_path("output.md")
    os.makedirs(os.path.dirname(deliv_path), exist_ok=True)
    with open(deliv_path, "w") as f:
        f.write(content)
    return ws


# ----------------------------------------------------------------------
# MultiFlow._resolve_flow_output_path delegates to shared helper
# ----------------------------------------------------------------------
class TestMFDualResolveFlowOutputPath:
    """MultiFlowInferencer._resolve_flow_output_path delegates to the
    canonical resolve_canonical_output_path helper for 3-tier resolution."""

    def test_mfdual_resolve_flow_output_path_returns_deliverable(self, tmp_path):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
            MultiFlowInferencer,
        )

        ws = _make_ws_with_deliverable(str(tmp_path / "flow0"), "FLOW0_DELIV")
        inferencer = MagicMock()
        inferencer._workspace = ws

        mfi = MagicMock(spec=MultiFlowInferencer)
        mfi.flow_configs = [{"followup_inferencer": inferencer, "initial_inferencer": None}]
        mfi._resolve_flow_output_path = (
            MultiFlowInferencer._resolve_flow_output_path.__get__(mfi)
        )

        path = mfi._resolve_flow_output_path(0)
        assert path is not None
        assert os.path.isfile(path)
        with open(path) as f:
            assert f.read() == "FLOW0_DELIV"

    def test_mfdual_resolve_flow_output_path_falls_back_to_outputs(self, tmp_path):
        """Tier 2: outputs/output.md when no deliverable exists."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
            MultiFlowInferencer,
        )

        ws = _make_ws_with_output(str(tmp_path / "flow0"), "FLOW0_OUT")
        inferencer = MagicMock()
        inferencer._workspace = ws

        mfi = MagicMock(spec=MultiFlowInferencer)
        mfi.flow_configs = [{"followup_inferencer": inferencer, "initial_inferencer": None}]
        mfi._resolve_flow_output_path = (
            MultiFlowInferencer._resolve_flow_output_path.__get__(mfi)
        )

        path = mfi._resolve_flow_output_path(0)
        assert path is not None
        assert os.path.isfile(path)

    def test_mfdual_resolve_flow_output_path_returns_none_when_unavailable(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
            MultiFlowInferencer,
        )

        mfi = MagicMock(spec=MultiFlowInferencer)
        mfi.flow_configs = []  # no flows
        mfi._resolve_flow_output_path = (
            MultiFlowInferencer._resolve_flow_output_path.__get__(mfi)
        )

        assert mfi._resolve_flow_output_path(0) is None

    def test_mfdual_resolve_flow_output_path_prefers_followup_over_initial(
        self, tmp_path
    ):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
            MultiFlowInferencer,
        )

        # Both initial and followup have outputs; followup should win
        ws_init = _make_ws_with_output(str(tmp_path / "init"), "INITIAL")
        ws_followup = _make_ws_with_output(str(tmp_path / "fu"), "FOLLOWUP")

        init_inf = MagicMock()
        init_inf._workspace = ws_init
        fu_inf = MagicMock()
        fu_inf._workspace = ws_followup

        mfi = MagicMock(spec=MultiFlowInferencer)
        mfi.flow_configs = [
            {"initial_inferencer": init_inf, "followup_inferencer": fu_inf}
        ]
        mfi._resolve_flow_output_path = (
            MultiFlowInferencer._resolve_flow_output_path.__get__(mfi)
        )

        path = mfi._resolve_flow_output_path(0)
        with open(path) as f:
            assert f.read() == "FOLLOWUP"


# ----------------------------------------------------------------------
# Cross-cutting smoke test: helper integration with real workspace
# ----------------------------------------------------------------------
class TestHelperIntegration:
    """Smoke test: helper works with real InferencerWorkspace instances."""

    def test_resolve_returns_abspath(self, tmp_path):
        ws = _make_ws_with_output(str(tmp_path / "ws"), "X")
        path = resolve_canonical_output_path(ws)
        assert path is not None
        assert os.path.isabs(path)
        assert path == os.path.abspath(path)

    def test_resolve_handles_none_workspace(self):
        assert resolve_canonical_output_path(None) is None

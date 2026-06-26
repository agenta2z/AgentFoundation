"""Tests for Part A — Path-aware peer/own-flow visibility in MultiFlowInferencer.

These tests verify that ``_format_followup_input`` injects on-disk paths for
both the flow's own prior output AND each visible peer's output, so the LLM
can read full file contents rather than relying solely on inline summaries.

Implementation reference:
- agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_inferencer.py
- Plan: _docs/_plans/mfdual_hygiene_INTEGRATED_plan.md (Part A)
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Optional
from unittest.mock import MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)


def _make_workspace_stub(
    *,
    has_deliverables: bool = False,
    deliverable_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> MagicMock:
    """Create a workspace mock matching InferencerWorkspace surface."""
    ws = MagicMock()
    ws.has_deliverables = has_deliverables
    ws.deliverable_path.return_value = deliverable_path
    ws.output_path.return_value = output_path
    return ws


def _make_inferencer_with_workspace(workspace) -> MagicMock:
    """Wrap a workspace into an inferencer mock."""
    inf = MagicMock()
    inf._workspace = workspace
    return inf


class _MFITestStub(MultiFlowInferencer):
    """Bypass attrs init so we can construct a minimal instance for unit tests."""

    def __init__(self, flow_configs):
        # Skip attrs.__init__ — we only need flow_configs for the helper
        self.flow_configs = flow_configs
        self._latest_per_flow = {i: None for i in range(len(flow_configs))}


class TestResolveFlowOutputPath(unittest.TestCase):
    """Tests for the _resolve_flow_output_path helper (Phase A1)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tmp_rovodev_mfi_")
        # Create real files for the resolver to find
        self.deliverable_md = os.path.join(self.tmpdir, "deliv.md")
        with open(self.deliverable_md, "w") as f:
            f.write("DELIVERABLE_CONTENT")
        self.output_md = os.path.join(self.tmpdir, "out.md")
        with open(self.output_md, "w") as f:
            f.write("OUTPUT_CONTENT")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_deliverable_path_when_has_deliverables_true(self):
        """Tier 1 (deliverable) takes precedence over Tier 2 (output)."""
        ws = _make_workspace_stub(
            has_deliverables=True,
            deliverable_path=self.deliverable_md,
            output_path=self.output_md,
        )
        inf = _make_inferencer_with_workspace(ws)
        mfi = _MFITestStub([{"followup_inferencer": inf}])
        self.assertEqual(mfi._resolve_flow_output_path(0), self.deliverable_md)

    def test_falls_back_to_output_path_when_no_deliverable(self):
        """Tier 2 fallback when no deliverable exists."""
        ws = _make_workspace_stub(
            has_deliverables=False,
            deliverable_path=None,
            output_path=self.output_md,
        )
        inf = _make_inferencer_with_workspace(ws)
        mfi = _MFITestStub([{"followup_inferencer": inf}])
        self.assertEqual(mfi._resolve_flow_output_path(0), self.output_md)

    def test_returns_none_when_neither_path_exists(self):
        """Graceful None return when nothing on disk."""
        ws = _make_workspace_stub(
            has_deliverables=True,
            deliverable_path="/nonexistent/path.md",
            output_path="/also/nonexistent.md",
        )
        inf = _make_inferencer_with_workspace(ws)
        mfi = _MFITestStub([{"followup_inferencer": inf}])
        self.assertIsNone(mfi._resolve_flow_output_path(0))

    def test_returns_none_when_no_workspace(self):
        """Graceful None when inferencer has no _workspace."""
        inf = MagicMock()
        inf._workspace = None
        mfi = _MFITestStub([{"followup_inferencer": inf}])
        self.assertIsNone(mfi._resolve_flow_output_path(0))

    def test_returns_none_when_flow_idx_out_of_bounds(self):
        """Bounds-check: returns None for invalid index, doesn't raise."""
        mfi = _MFITestStub([])
        self.assertIsNone(mfi._resolve_flow_output_path(0))
        self.assertIsNone(mfi._resolve_flow_output_path(99))

    def test_falls_back_to_initial_inferencer_if_no_followup(self):
        """If no followup_inferencer, helper checks initial_inferencer."""
        ws = _make_workspace_stub(
            has_deliverables=False,
            deliverable_path=None,
            output_path=self.output_md,
        )
        inf = _make_inferencer_with_workspace(ws)
        mfi = _MFITestStub([{"initial_inferencer": inf}])  # no followup key
        self.assertEqual(mfi._resolve_flow_output_path(0), self.output_md)


class TestFormatFollowupInputInjectsPaths(unittest.TestCase):
    """Tests for path injection in _format_followup_input (Phase A2)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tmp_rovodev_mfi_fmt_")
        self.flow_0_path = os.path.join(self.tmpdir, "flow_0.md")
        self.flow_1_path = os.path.join(self.tmpdir, "flow_1.md")
        for p in (self.flow_0_path, self.flow_1_path):
            with open(p, "w") as f:
                f.write("CONTENT_" + os.path.basename(p))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mfi_with_real_paths(self):
        ws_0 = _make_workspace_stub(has_deliverables=False, output_path=self.flow_0_path)
        ws_1 = _make_workspace_stub(has_deliverables=False, output_path=self.flow_1_path)
        cfgs = [
            {"followup_inferencer": _make_inferencer_with_workspace(ws_0)},
            {"followup_inferencer": _make_inferencer_with_workspace(ws_1)},
        ]
        return _MFITestStub(cfgs)

    def test_own_flow_path_appears_in_output(self):
        """The flow's OWN prior-output path should appear in formatted text."""
        mfi = self._make_mfi_with_real_paths()
        out = mfi._format_followup_input(
            your_prev="my prior text",
            flow_idx=0,
            step_idx=1,
            visible_plans={},
        )
        self.assertIn(self.flow_0_path, out)
        self.assertIn("Your previous full artifact is on disk at", out)

    def test_peer_flow_path_appears_in_output(self):
        """A visible peer's path should appear in formatted text."""
        mfi = self._make_mfi_with_real_paths()
        out = mfi._format_followup_input(
            your_prev="my prior text",
            flow_idx=0,
            step_idx=1,
            visible_plans={1: "peer summary text"},
        )
        self.assertIn(self.flow_1_path, out)
        self.assertIn("The full peer artifact is available at", out)

    def test_both_own_and_peer_paths_appear_when_visible_plans_nonempty(self):
        """When visible_plans is non-empty, BOTH own and peer paths appear."""
        mfi = self._make_mfi_with_real_paths()
        out = mfi._format_followup_input(
            your_prev="my prior text",
            flow_idx=0,
            step_idx=1,
            visible_plans={1: "peer summary"},
        )
        self.assertIn(self.flow_0_path, out)  # own
        self.assertIn(self.flow_1_path, out)  # peer

    def test_no_path_text_when_workspace_yields_no_path(self):
        """If a flow has no resolvable path, no path-text is appended for it."""
        ws_no = _make_workspace_stub(has_deliverables=False, output_path=None)
        cfgs = [
            {"followup_inferencer": _make_inferencer_with_workspace(ws_no)},
        ]
        mfi = _MFITestStub(cfgs)
        out = mfi._format_followup_input(
            your_prev="my prior text",
            flow_idx=0,
            step_idx=1,
            visible_plans={},
        )
        # Should not raise; no "on disk at" message for own
        self.assertNotIn("Your previous full artifact is on disk at", out)
        # Should still include the inline text and headers
        self.assertIn("my prior text", out)

    def test_text_summary_still_included_alongside_path(self):
        """Path-aware fix is ADDITIVE — inline text content remains."""
        mfi = self._make_mfi_with_real_paths()
        out = mfi._format_followup_input(
            your_prev="MY_PRIOR_TEXT",
            flow_idx=0,
            step_idx=1,
            visible_plans={1: "PEER_SUMMARY_TEXT"},
        )
        self.assertIn("MY_PRIOR_TEXT", out)
        self.assertIn("PEER_SUMMARY_TEXT", out)


class TestFollowupPathUnderCtxPublishedWorkspace(unittest.TestCase):
    """M7 regression guard for own_path/peer_path resolution.

    The unit tests above set ``inf._workspace`` directly — which is NOT how production
    works. Under the M7 ctx/workspace decoupling the per-run workspace is PUBLISHED into the
    run-context (via ``_rc_child(workspace=...)``) and the flow-config leaf instances'
    ``_workspace`` stays ``None``. Resolving the followup file path off that stale instance
    therefore returned ``None`` for every followup step in production (0/7 in run
    ``wsfix8``), even though those unit tests stayed green.

    This test reproduces the real condition (instance ``_workspace`` unset; workspaces arrive
    only via the run-context) and asserts the followup input STILL references the prior
    step's full output on disk — exercising the per-run ``_latest_per_flow_path`` capture
    instead of the leaf instance. Fails before the fix (no ``on disk at`` block), passes
    after.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tmp_rovodev_mfi_ctxpath_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_followup_input_references_prior_step_output_under_ctx(self):
        import asyncio
        import re

        from attr import attrib, attrs

        from agent_foundation.common.inferencers.inferencer_base import InferencerBase
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        @attrs
        class _StepLeaf(InferencerBase):
            """CLI-like leaf: writes its full artifact to ``outputs/output.md`` at its
            ctx-resolved (step) workspace. Its instance ``_workspace`` is NEVER set, so the
            only way to resolve its output path is the LIVE run-context (the M7 condition)."""

            scripted: str = attrib(default="<Response>step</Response>")

            def _write(self):
                ws = self._workspace  # getter → ctx step workspace under M7
                if ws is not None:
                    out = ws.output_path("output.md")
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out, "w", encoding="utf-8") as f:
                        f.write("FULL ARTIFACT BODY\n" + self.scripted)

            def _infer(self, inference_input, inference_config=None, **kw):
                self._write()
                return self.scripted

            async def _ainfer(self, inference_input, inference_config=None, **kw):
                self._write()
                return self.scripted

        f0_initial = _StepLeaf(scripted="<Response>INITIAL</Response>")
        f0_followup = _StepLeaf(scripted="<Response>ROUND01</Response>")
        # M7 condition: the flow leaves' instance backing workspace is UNSET.
        self.assertIsNone(getattr(f0_initial, "_InferencerBase__workspace", "set"))

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "design the plan",
                    "initial_inferencer": f0_initial,
                    "followup_inferencer": f0_followup,
                    "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                    "max_dynamic_steps": 2,
                },
            ],
            aggregator_inferencer=_StepLeaf(scripted="<Response>AGG</Response>"),
            inject_upstream_artifacts=True,
            workspace=InferencerWorkspace(root=os.path.join(self.tmpdir, "ws")),
            checkpoint_dir=os.path.join(self.tmpdir, "ckpt"),
        )

        # Spy on the followup-input formatter to capture exactly what the leaf would receive.
        captured = []
        _orig = mfi._format_followup_input

        def _spy(**kwargs):
            out = _orig(**kwargs)
            captured.append(out)
            return out

        mfi._format_followup_input = _spy

        asyncio.run(mfi.ainfer("build the plan"))

        self.assertTrue(captured, "no followup input was built (the flow did not iterate)")
        joined = "\n\n".join(captured)
        self.assertIn(
            "Your previous full artifact is on disk at",
            joined,
            "followup input is missing the prior-step on-disk reference — the M7 own_path "
            "regression (path resolved to None from the stale leaf instance _workspace)",
        )
        paths = re.findall(r"`([^`]*initial[/\\]outputs[/\\]output\.md)`", joined)
        self.assertTrue(
            paths, f"no initial-step output path referenced in followup input:\n{joined[:600]}"
        )
        self.assertTrue(
            os.path.isfile(paths[0]),
            f"referenced prior-step output does not exist on disk: {paths[0]!r}",
        )


if __name__ == "__main__":
    unittest.main()

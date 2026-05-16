"""E2E tests for the unified _finalize_output symlink architecture.

Verifies:
1. LWI symlinks to last dynamic step
2. BTA symlinks to aggregator
3. Dual creates per-round workspaces and symlinks to canonical child
4. Full Dual{BTA} topology workspace structure
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unittest

import attr
from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


@attrs
class _Mock(InferencerBase):
    """Mock inferencer returning scripted <Response>-wrapped output."""

    scripted_response: str = attrib(default="<Response>Mock summary</Response>")

    def _infer(self, inference_input, **kwargs):
        return self.scripted_response

    async def _ainfer(self, inference_input, **kwargs):
        return self.scripted_response


class TestLWIFinalizeOutputSymlink(unittest.TestCase):
    """LWI symlinks outputs/output.md to last dynamic step's output."""

    def test_lwi_symlinks_to_last_child(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
            LinearWorkflowInferencer,
        )

        tmpdir = tempfile.mkdtemp(prefix="lwi_sym_")
        try:
            initial = _Mock(scripted_response="<Response>Initial</Response>")
            followup = _Mock(scripted_response="<Response>Followup</Response>")

            lwi = LinearWorkflowInferencer(
                dynamic_mode=True,
                default_initial_inferencer=initial,
                default_followup_inferencer=followup,
                end_condition=lambda s, r: True,
                max_dynamic_steps=2,
                workspace=InferencerWorkspace(root=tmpdir),
                output_path="output.md",
            )

            # Simulate children wrote output
            for child_name in ("initial", "round01"):
                child_ws = lwi._workspace.child(child_name)
                child_ws.ensure_dirs()
                with open(child_ws.output_path("output.md"), "w") as f:
                    f.write(f"# Full artifact from {child_name}")

            # Simulate LWI state after 2 steps
            lwi._state = {"dynamic_step_results": ["r0", "r1"]}
            lwi._finalize_output("<Response>Summary</Response>")

            # Verify: outputs/output.md exists and has round01's content
            own = os.path.join(lwi._workspace.outputs_dir, "output.md")
            self.assertTrue(os.path.exists(own))
            content = open(own).read()
            self.assertIn("round01", content,
                "LWI should symlink to LAST child (round01), not initial")

            # Verify: is a symlink (on platforms that support it)
            if os.name != "nt":
                self.assertTrue(os.path.islink(own),
                    "outputs/output.md should be a symlink")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestBTAFinalizeOutputSymlink(unittest.TestCase):
    """BTA symlinks to aggregator's output."""

    def test_bta_symlinks_to_aggregator(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )

        tmpdir = tempfile.mkdtemp(prefix="bta_sym_")
        try:
            aggregator = _Mock(scripted_response="<Response>Aggregated</Response>")
            breakdown = _Mock(scripted_response="subtask_0")

            bta = BreakdownThenAggregateInferencer(
                breakdown_inferencer=breakdown,
                aggregator_inferencer=aggregator,
                workspace=InferencerWorkspace(
                    root=tmpdir, use_final_deliverables_folder=True),
                output_path="output.md",
            )

            # Simulate runtime: BTA assigns aggregator workspace in _build_subgraph_spec
            agg_ws = bta._workspace.child("aggregator")
            agg_ws.ensure_dirs()
            aggregator._workspace = agg_ws
            with open(agg_ws.output_path("output.md"), "w") as f:
                f.write("# Aggregated plan")
            fd = agg_ws.deliverables_dir
            if fd:
                os.makedirs(fd, exist_ok=True)
                shutil.copy2(agg_ws.output_path("output.md"),
                             os.path.join(fd, "output.md"))

            # Verify aggregator workspace and file before calling finalize
            self.assertIsNotNone(aggregator._workspace)
            agg_out = aggregator._workspace.output_path("output.md")
            self.assertTrue(os.path.isfile(agg_out),
                f"Aggregator output.md should exist at {agg_out}")

            bta._finalize_output("<Response>BTA summary</Response>")

            # BTA's outputs/output.md should exist (symlink to aggregator)
            own = bta._workspace.output_path("output.md")
            self.assertTrue(os.path.exists(own),
                f"BTA output should exist at {own}")
            content = open(own).read()
            self.assertIn("Aggregated", content,
                f"BTA output should have aggregator content, got: {content[:100]}")

            # BTA's final_deliverables/ should have aggregator's deliverables
            bta_fd = bta._workspace.deliverables_dir
            if bta_fd:
                self.assertTrue(os.path.exists(os.path.join(bta_fd, "output.md")))

            # No double nesting
            if bta_fd:
                nested = os.path.join(bta_fd, "final_deliverables")
                self.assertFalse(os.path.isdir(nested),
                    "No double nesting in deliverables")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestDualPerRoundWorkspace(unittest.TestCase):
    """Dual creates per-round workspaces and tracks canonical child."""

    def test_dual_assigns_propose_workspace(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )

        tmpdir = tempfile.mkdtemp(prefix="dual_ws_")
        try:
            base = _Mock()
            review = _Mock()
            fixer = _Mock()

            dual = DualInferencer(
                base_inferencer=base,
                review_inferencer=review,
                fixer_inferencer=fixer,
                workspace=InferencerWorkspace(root=tmpdir),
            )

            # base gets propose/ workspace
            self.assertIsNotNone(base._workspace)
            self.assertTrue(
                base._workspace.root.replace("\\", "/").endswith("/children/propose"))

            # review and fixer are deferred (per-round assignment at runtime)
            self.assertIsNone(review._workspace)
            self.assertIsNone(fixer._workspace)

            # Verify propose/ directory exists on disk
            propose_dir = os.path.join(tmpdir, "children", "propose")
            self.assertTrue(os.path.isdir(propose_dir))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dual_finalize_output_symlinks_to_propose(self):
        """When only propose runs (no fix), canonical = propose."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )

        tmpdir = tempfile.mkdtemp(prefix="dual_prop_")
        try:
            base = _Mock()
            dual = DualInferencer(
                base_inferencer=base,
                workspace=InferencerWorkspace(root=tmpdir),
                output_path="output.md",
            )

            # Simulate propose wrote output
            base_ws = base._workspace
            with open(base_ws.output_path("output.md"), "w") as f:
                f.write("# Proposal content")

            # Set tracker (normally done by _step_propose_impl)
            dual._last_output_child_ws = base_ws
            dual._finalize_output("<Response>Proposal summary</Response>")

            own = dual._workspace.output_path("output.md")
            self.assertTrue(os.path.exists(own))
            self.assertIn("Proposal content", open(own).read())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dual_finalize_output_symlinks_to_fixer(self):
        """When fix runs, canonical = fixer's round workspace."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )

        tmpdir = tempfile.mkdtemp(prefix="dual_fix_")
        try:
            base = _Mock()
            review = _Mock()
            fixer = _Mock()
            dual = DualInferencer(
                base_inferencer=base,
                review_inferencer=review,
                fixer_inferencer=fixer,
                workspace=InferencerWorkspace(root=tmpdir),
                output_path="output.md",
            )

            # Simulate per-round workspace for fixer
            round_ws = dual._workspace.child("round_01")
            fix_ws = round_ws.child("fix")
            fix_ws.ensure_dirs()
            with open(fix_ws.output_path("output.md"), "w") as f:
                f.write("# Fixed artifact")

            # Set tracker (normally done by _step_fix_impl)
            dual._last_output_child_ws = fix_ws
            dual._finalize_output("<Response>Fix summary</Response>")

            own = dual._workspace.output_path("output.md")
            self.assertTrue(os.path.exists(own))
            self.assertIn("Fixed artifact", open(own).read())

            # Verify workspace structure
            self.assertTrue(os.path.isdir(
                os.path.join(tmpdir, "children", "round_01", "children", "fix")))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestFullDualBTAE2E(unittest.TestCase):
    """E2E: Dual wrapping BTA, verify full workspace structure after ainfer()."""

    def test_dual_bta_e2e_workspace_structure(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )

        tmpdir = tempfile.mkdtemp(prefix="e2e_dual_bta_")
        try:
            # Build topology: Dual { base=BTA { workers, aggregator } }
            aggregator = _Mock(
                scripted_response="<Response>Aggregated plan</Response>",
                output_is_deliverable=True,
                output_path="output.md",
            )
            breakdown = _Mock(scripted_response='["subtask_0"]')
            worker = _Mock(
                scripted_response="<Response>Worker output</Response>",
                output_path="output.md",
            )
            bta = BreakdownThenAggregateInferencer(
                breakdown_inferencer=breakdown,
                aggregator_inferencer=aggregator,
                output_path="output.md",
            )
            fixer = _Mock(
                scripted_response="<Response>Fixed plan</Response>",
                output_is_deliverable=True,
                output_path="output.md",
            )

            dual = DualInferencer(
                base_inferencer=bta,
                fixer_inferencer=fixer,
                workspace=InferencerWorkspace(
                    root=tmpdir, use_final_deliverables_folder=True),
                output_path="output.md",
            )

            # Verify construction-time workspace
            self.assertIsNotNone(bta._workspace,
                "BTA (as base) should get propose/ workspace")
            self.assertTrue(
                bta._workspace.root.replace("\\", "/").endswith("/children/propose"))

            # Verify NO stale workspace patterns
            children_dir = os.path.join(tmpdir, "children")
            if os.path.isdir(children_dir):
                child_names = set(os.listdir(children_dir))
                self.assertNotIn("base_inferencer", child_names,
                    "Old base_inferencer/ path should not exist")
                self.assertIn("propose", child_names,
                    "propose/ should exist as BTA's workspace")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestBTAPathRecomputationForAggregator(unittest.TestCase):
    """Verify that BTA recomputes worker output paths AFTER workers finish,
    so the aggregator gets file path references (not None)."""

    def test_build_agg_input_recomputes_paths(self):
        """After workers complete, _build_agg_input finds their output files
        via resolve_canonical_output_path (including LWI symlinks)."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )
        from agent_foundation.common.inferencers.inferencer_workspace import (
            resolve_canonical_output_path,
        )

        tmpdir = tempfile.mkdtemp(prefix="bta_recomp_")
        try:
            aggregator = _Mock(
                scripted_response="<Response>Aggregated</Response>",
                output_is_deliverable=True,
                output_path="output.md",
            )
            breakdown = _Mock(scripted_response='["task_0"]')

            bta = BreakdownThenAggregateInferencer(
                breakdown_inferencer=breakdown,
                aggregator_inferencer=aggregator,
                workspace=InferencerWorkspace(
                    root=tmpdir, use_final_deliverables_folder=True),
                output_path="output.md",
            )

            # Simulate: a worker (LWI flow) completed and wrote output
            worker = _Mock(output_path="output.md")
            worker_ws = bta._workspace.child("flow_0")
            worker_ws.ensure_dirs()
            worker._workspace = worker_ws
            with open(worker_ws.output_path("output.md"), "w") as f:
                f.write("# Full worker artifact (35KB)")

            # At construction time: path is None (file didn't exist yet)
            # After workers complete: resolve finds the file
            resolved = resolve_canonical_output_path(
                worker._workspace, filename="output.md"
            )
            self.assertIsNotNone(resolved,
                "After worker completes, resolve_canonical_output_path should find output.md")
            self.assertTrue(resolved.endswith("output.md"))

            # Verify the file is readable through the resolved path
            content = open(resolved).read()
            self.assertIn("35KB", content)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_lwi_symlink_then_resolve(self):
        """LWI symlinks last child → resolve_canonical_output_path finds it
        → aggregator gets path to full artifact."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
            LinearWorkflowInferencer,
        )
        from agent_foundation.common.inferencers.inferencer_workspace import (
            resolve_canonical_output_path,
        )

        tmpdir = tempfile.mkdtemp(prefix="lwi_resolve_")
        try:
            initial = _Mock()
            followup = _Mock()
            lwi = LinearWorkflowInferencer(
                dynamic_mode=True,
                default_initial_inferencer=initial,
                default_followup_inferencer=followup,
                end_condition=lambda s, r: True,
                max_dynamic_steps=2,
                workspace=InferencerWorkspace(root=tmpdir),
                output_path="output.md",
            )

            # Simulate: round01 child wrote the full artifact
            round01_ws = lwi._workspace.child("round01")
            round01_ws.ensure_dirs()
            with open(round01_ws.output_path("output.md"), "w") as f:
                f.write("# Complete 798-line research artifact")

            # LWI finalize creates symlink
            lwi._state = {"dynamic_step_results": ["r0", "r1"]}
            lwi._finalize_output("<Response>Summary</Response>")

            # resolve_canonical_output_path on the LWI workspace should find it
            resolved = resolve_canonical_output_path(
                lwi._workspace, filename="output.md"
            )
            self.assertIsNotNone(resolved,
                "resolve_canonical_output_path should find LWI's symlinked output")

            # Reading through the resolved path gives the FULL artifact
            content = open(resolved).read()
            self.assertIn("798-line", content,
                "Aggregator should get FULL artifact content through symlink, "
                "not just the <Response> summary")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPTIMultiFlagSymlink(unittest.TestCase):
    """PTI symlinks per-flag outputs from iteration children."""

    def test_pti_output_mode_plan_only(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
            PlanThenImplementInferencer, PTIOutputMode,
        )

        tmpdir = tempfile.mkdtemp(prefix="pti_plan_")
        try:
            planner = _Mock(output_path="plan.md")
            executor = _Mock(output_path="implementation.md")
            pti = PlanThenImplementInferencer(
                planner_inferencer=planner,
                executor_inferencer=executor,
                output_mode=PTIOutputMode.PLAN,
                workspace=InferencerWorkspace(root=tmpdir),
                output_path="plan.md",
            )

            # Simulate: planner wrote plan.md in iteration workspace
            iter_ws_path = pti._get_iteration_workspace(tmpdir, 1)
            from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace as IW
            iter_ws = IW(root=iter_ws_path)
            planner_ws = iter_ws.child("planner")
            planner_ws.ensure_dirs()
            with open(planner_ws.output_path("plan.md"), "w") as f:
                f.write("# The Plan")

            pti._state = {"iteration": 1}
            pti._finalize_output("<Response>Plan summary</Response>")

            own = pti._workspace.output_path("plan.md")
            self.assertTrue(os.path.exists(own), "plan.md should be symlinked")
            self.assertIn("The Plan", open(own).read())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestSymlinkFallbackOnWindows(unittest.TestCase):
    """Verify copy fallback when symlinks fail."""

    def test_symlink_or_copy_fallback(self):
        tmpdir = tempfile.mkdtemp(prefix="sym_fallback_")
        try:
            src_file = os.path.join(tmpdir, "source.md")
            with open(src_file, "w") as f:
                f.write("source content")
            dst_file = os.path.join(tmpdir, "dest.md")

            InferencerBase._symlink_or_copy(src_file, dst_file)

            self.assertTrue(os.path.exists(dst_file))
            self.assertEqual(open(dst_file).read(), "source content")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_symlink_or_copy_skip_existing(self):
        tmpdir = tempfile.mkdtemp(prefix="sym_skip_")
        try:
            src_file = os.path.join(tmpdir, "source.md")
            with open(src_file, "w") as f:
                f.write("new content")
            dst_file = os.path.join(tmpdir, "dest.md")
            with open(dst_file, "w") as f:
                f.write("existing content")

            InferencerBase._symlink_or_copy(src_file, dst_file)

            # Should NOT overwrite
            self.assertEqual(open(dst_file).read(), "existing content")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

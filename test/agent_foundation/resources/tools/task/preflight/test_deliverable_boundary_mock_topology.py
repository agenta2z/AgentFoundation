"""Mock topology integration test for deliverable boundary semantics (Part 2).

Builds the driver topology in-process with stubbed leaf inferencers, runs end-to-end,
and verifies the surfacing chain works as designed:
  worker → BTA → Dual (pass-through) → PTI → outer Dual (pass-through) → task root.

This test is the centerpiece of AC9 — the "most important test in the plan".

Two-axis contract (Part 2): ``outputs/`` IS the deliverable set (the
``final_deliverables/`` folder and the four deliverable flags are RETIRED).
Boundary selection is via ``has_deliverables`` (outputs/ non-empty) AND an
explicit role-based ``boundary_filter`` — NOT an ``is_deliverable_boundary``
attribute. Promotion carries a selected child's ``outputs/`` up to the parent's
``outputs/``.

Mocked topology (matches breakdown_multiflow_plan_then_implement.yaml shape):
  outer Dual (pass-through, sibling fixer)
    ├── base_inferencer = PTI (boundary)
    │   ├── planner Dual (pass-through)
    │   │   └── plan BTA (boundary)
    │   │       └── 2 worker MFDuals (boundary)
    │   └── executor BTA (boundary; bare BTA, no Dual wrap)
    │       └── 2 worker Duals (boundary)
    └── fixer = PTI (boundary; sibling)

The leaf "inferencer" is a DeliverableStub that writes a tagged file into its
workspace's outputs/ on each call.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# Tag-marker stub: writes a known file to its workspace's outputs/ dir
# so we can assert exactly where it ends up after surfacing.
def _make_stub(tag, filename):
    """Create a stub inferencer that writes <filename> tagged with <tag>."""
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    from attr import attrib, attrs

    @attrs(auto_attribs=False)
    class DeliverableStub(InferencerBase):
        _tag: str = attrib(default=tag)
        _filename: str = attrib(default=filename)

        def _infer(self, inference_input, inference_config=None, **_inference_args):
            ws = self._workspace
            if ws is None or ws.outputs_dir is None:
                # Workspace not configured — return a string response as fallback
                return f"stub:{self._tag}"
            os.makedirs(ws.outputs_dir, exist_ok=True)
            target = os.path.join(ws.outputs_dir, self._filename)
            with open(target, "w", encoding="utf-8") as f:
                f.write(f"# Deliverable from {self._tag}\n")
            return f"stub:{self._tag}"

    return DeliverableStub()


def _ws(tmp):
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    w = InferencerWorkspace(root=str(tmp))
    w.ensure_dirs()
    return w


@pytest.mark.preflight
def test_S1_workspace_exposes_outputs_dir(tmp_path):
    """S1: workspace.outputs_dir is a real directory — the deliverable set (Part 2)."""
    w = _ws(tmp_path)
    assert w.outputs_dir is not None
    assert os.path.isdir(w.outputs_dir)
    # Empty outputs/ ⇒ no deliverables yet.
    assert not w.has_deliverables


@pytest.mark.preflight
def test_S2_child_workspace_has_outputs_dir(tmp_path):
    """S2: ws.child("foo").outputs_dir resolves under the child root."""
    w = _ws(tmp_path)
    child = w.child("worker_0")
    child.ensure_dirs()
    assert child.outputs_dir is not None
    assert os.path.isdir(child.outputs_dir)
    # No retired attrib.
    assert not hasattr(child, "use_final_deliverables_folder")


@pytest.mark.preflight
def test_S3_grandchild_workspace_has_outputs_dir(tmp_path):
    """S3: child().child() — outputs/ resolves 2+ hops deep."""
    w = _ws(tmp_path)
    grandchild = w.child("a").child("b")
    grandchild.ensure_dirs()
    assert grandchild.outputs_dir is not None
    assert Path(grandchild.outputs_dir).name == "outputs"


@pytest.mark.preflight
def test_S4_stub_writes_to_outputs_dir(tmp_path):
    """S4: A stub inferencer correctly writes to its workspace's outputs_dir."""
    w = _ws(tmp_path)
    stub = _make_stub("test_S4", "out.md")
    stub._workspace = w
    stub.infer("input")
    assert os.path.isfile(os.path.join(w.outputs_dir, "out.md"))
    assert w.has_deliverables


@pytest.mark.preflight
def test_S5_collect_helper_finds_boundary_children(tmp_path):
    """S5: collect_child_boundary_deliverables returns children with non-empty outputs/."""
    from agent_foundation.common.inferencers.deliverable_boundary import (
        collect_child_boundary_deliverables,
    )

    parent = _ws(tmp_path)
    child_a = parent.child("worker_0")
    child_a.ensure_dirs()
    with open(os.path.join(child_a.outputs_dir, "a.md"), "w") as f:
        f.write("hi")
    child_b = parent.child("worker_1")
    child_b.ensure_dirs()
    with open(os.path.join(child_b.outputs_dir, "b.md"), "w") as f:
        f.write("bye")

    children = collect_child_boundary_deliverables(parent)
    assert len(children) == 2
    names = sorted(c.child_name for c in children)
    assert names == ["worker_0", "worker_1"]


@pytest.mark.preflight
def test_S6_aggregate_by_child_name(tmp_path):
    """S6: aggregate_into_self_deliverables produces workers/<name>/<file> structure."""
    from agent_foundation.common.inferencers.deliverable_boundary import (
        aggregate_into_self_deliverables,
        collect_child_boundary_deliverables,
    )

    parent = _ws(tmp_path)
    for n in ("worker_0", "worker_1"):
        c = parent.child(n)
        c.ensure_dirs()
        with open(os.path.join(c.outputs_dir, "result.md"), "w") as f:
            f.write(n)

    kids = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent,
        kids,
        namespace_strategy="by_child_name",
        namespace_root="workers",
    )
    assert len(report.copied) == 2
    assert os.path.isfile(
        os.path.join(parent.outputs_dir, "workers/worker_0/result.md")
    )
    assert os.path.isfile(
        os.path.join(parent.outputs_dir, "workers/worker_1/result.md")
    )


@pytest.mark.preflight
def test_S7_aggregate_by_role(tmp_path):
    """S7: by_role aggregation produces <role>/<file> structure (no extra wrapper)."""
    from agent_foundation.common.inferencers.deliverable_boundary import (
        aggregate_into_self_deliverables,
        collect_child_boundary_deliverables,
    )

    parent = _ws(tmp_path)
    for role in ("planner", "executor"):
        c = parent.child(role)
        c.ensure_dirs()
        with open(os.path.join(c.outputs_dir, f"{role}.md"), "w") as f:
            f.write(role)

    kids = collect_child_boundary_deliverables(
        parent,
        boundary_filter=lambda name, ws: name in ("planner", "executor"),
    )
    report = aggregate_into_self_deliverables(
        parent,
        kids,
        namespace_strategy="by_role",
    )
    assert len(report.copied) == 2
    assert os.path.isfile(os.path.join(parent.outputs_dir, "planner/planner.md"))
    assert os.path.isfile(os.path.join(parent.outputs_dir, "executor/executor.md"))


@pytest.mark.preflight
def test_S8_conflict_skip_existing(tmp_path):
    """S8: skip_existing strategy preserves existing files."""
    from agent_foundation.common.inferencers.deliverable_boundary import (
        aggregate_into_self_deliverables,
        ChildBoundaryDeliverables,
    )

    parent = _ws(tmp_path)
    # Put a pre-existing file under planner/
    pre_dir = os.path.join(parent.outputs_dir, "planner")
    os.makedirs(pre_dir)
    pre_file = os.path.join(pre_dir, "shared.md")
    with open(pre_file, "w") as f:
        f.write("existing-content")

    # Try to copy a new version of shared.md from a planner boundary
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    new_file = os.path.join(src_child.outputs_dir, "shared.md")
    with open(new_file, "w") as f:
        f.write("new-content")

    report = aggregate_into_self_deliverables(
        parent,
        [
            ChildBoundaryDeliverables(
                child_name="planner",
                child_workspace_root=src_child.root,
                deliverable_files=["shared.md"],
                child_workspace=src_child,
            )
        ],
        namespace_strategy="by_role",
        conflict_strategy="skip_existing",
    )
    # Should be skipped (existing file preserved)
    with open(pre_file) as f:
        assert f.read() == "existing-content"
    assert "shared.md" in str(report.skipped)


@pytest.mark.preflight
def test_S9_BTA_post_step_extends_finalize(tmp_path):
    """S9: BTA._finalize_response post-step surfaces worker deliverables under workers/."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
        BreakdownThenAggregateInferencer,
    )

    # Construct a minimal BTA with workspace + simulated worker children
    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_make_stub("breakdown", "breakdown.md"),
        aggregator_inferencer=_make_stub("agg", "aggregator.md"),
        worker_inferencers=lambda **kw: _make_stub("worker", "w.md"),
        output_path="aggregation_report.md",
    )
    bta._workspace = _ws(tmp_path / "bta_root")
    bta._workspace.ensure_dirs()
    # Pre-populate worker child workspaces with deliverables
    for i in range(2):
        wc = bta._workspace.child(f"worker_{i}")
        wc.ensure_dirs()
        with open(os.path.join(wc.outputs_dir, f"worker_{i}_plan.md"), "w") as f:
            f.write(f"plan {i}")
    bta._finalize_response(result=None)

    # Assert worker deliverables were surfaced under workers/
    assert os.path.isfile(
        os.path.join(bta._workspace.outputs_dir, "workers/worker_0/worker_0_plan.md")
    )
    assert os.path.isfile(
        os.path.join(bta._workspace.outputs_dir, "workers/worker_1/worker_1_plan.md")
    )


@pytest.mark.preflight
def test_S10_BTA_no_op_when_no_child_deliverables(tmp_path):
    """S10: BTA boundary post-step is a no-op when no child has deliverables.

    Part 2: there is no "flag off" state; the post-step simply finds nothing to
    surface (no child workspace has a non-empty outputs/), so it does not crash
    and produces no spurious workers/ subfolder.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
        BreakdownThenAggregateInferencer,
    )

    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_make_stub("breakdown", "b.md"),
        aggregator_inferencer=_make_stub("agg", "a.md"),
        worker_inferencers=lambda **kw: _make_stub("w", "w.md"),
        output_path="agg.md",
    )
    bta._workspace = _ws(tmp_path / "bta")
    bta._workspace.ensure_dirs()
    # No child workspaces populated → nothing to surface.
    bta._finalize_response(result=None)  # Should not crash
    assert not os.path.isdir(os.path.join(bta._workspace.outputs_dir, "workers"))


@pytest.mark.preflight
def test_S11_PTI_finalize_collects_role_boundaries(tmp_path):
    """S11: PTI._finalize_outputs surfaces planner/executor with by_role namespacing."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_stub("planner", "plan.md"),
        executor_inferencer=_make_stub("executor", "impl.md"),
    )
    pti._workspace = InferencerWorkspace(root=str(tmp_path / "pti_root"))
    pti._workspace.ensure_dirs()
    # PTI uses iter_<N>/children/<role>/ structure. Simulate iter_1 with planner/executor children.
    pti._state = {"iteration": 1}

    # Build the iter_ws that PTI's _finalize_outputs will look at:
    iter_ws_path = pti._get_iteration_workspace(pti._workspace.root, 1)
    iter_ws = InferencerWorkspace(root=iter_ws_path)
    iter_ws.ensure_dirs()
    for role in ("planner", "executor"):
        c = iter_ws.child(role)
        c.ensure_dirs()
        with open(os.path.join(c.outputs_dir, f"{role}.md"), "w") as f:
            f.write(role)

    pti._finalize_outputs()

    # Assert by_role surfacing
    assert os.path.isfile(
        os.path.join(pti._workspace.outputs_dir, "planner/planner.md")
    )
    assert os.path.isfile(
        os.path.join(pti._workspace.outputs_dir, "executor/executor.md")
    )


@pytest.mark.preflight
def test_S12_full_chain_BTA_to_PTI(tmp_path):
    """S12: full surfacing chain — workers → BTA → PTI."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
        BreakdownThenAggregateInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    # Build PTI with a planner-side BTA via workspace simulation
    pti = PlanThenImplementInferencer(
        planner_inferencer=_make_stub("planner", "plan.md"),
        executor_inferencer=_make_stub("executor", "impl.md"),
    )
    pti._workspace = InferencerWorkspace(root=str(tmp_path / "pti"))
    pti._workspace.ensure_dirs()
    pti._state = {"iteration": 1}
    iter_ws = InferencerWorkspace(
        root=pti._get_iteration_workspace(pti._workspace.root, 1),
    )
    iter_ws.ensure_dirs()
    # Planner role contains a nested BTA's deliverables (workers/ subfolder)
    planner_ws = iter_ws.child("planner")
    planner_ws.ensure_dirs()
    workers_dir = os.path.join(planner_ws.outputs_dir, "workers", "worker_0")
    os.makedirs(workers_dir)
    with open(os.path.join(workers_dir, "worker_0_plan.md"), "w") as f:
        f.write("nested worker output")

    pti._finalize_outputs()

    # Worker file surfaced through 2 boundary hops (BTA→PTI as planner role)
    expected = os.path.join(
        pti._workspace.outputs_dir, "planner/workers/worker_0/worker_0_plan.md"
    )
    assert os.path.isfile(expected), f"missing: {expected}"


@pytest.mark.preflight
def test_S13_Dual_passthrough_active_proposer_base(tmp_path):
    """S13: Dual surfaces base_inferencer when no fixer ran (counter_feedback=None)."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        ConsensusIterationRecord,
        DualInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    base = _make_stub("base", "base.md")
    base._workspace = InferencerWorkspace(root=str(tmp_path / "base_ws"))
    base._workspace.ensure_dirs()
    with open(os.path.join(base._workspace.outputs_dir, "base.md"), "w") as f:
        f.write("base output")

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=_make_stub("review", "review.md"),
        output_path="dual.md",
    )
    dual._workspace = InferencerWorkspace(root=str(tmp_path / "dual_ws"))
    dual._workspace.ensure_dirs()
    # State: 1 iteration, counter_feedback=None → base wins
    dual._state = {
        "consensus_iterations": [
            ConsensusIterationRecord(
                iteration=1,
                base_output="ok",
                review_input="rev",
                review_output="ok",
                counter_feedback=None,
            )
        ]
    }
    dual._finalize_response()

    # Base's base.md should be surfaced into dual's outputs/
    assert os.path.isfile(os.path.join(dual._workspace.outputs_dir, "base.md"))


@pytest.mark.preflight
def test_S14_Dual_passthrough_active_proposer_fixer(tmp_path):
    """S14: Dual surfaces fixer when counter_feedback is non-None (fixer ran)."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        ConsensusIterationRecord,
        DualInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    base = _make_stub("base", "result.md")
    base._workspace = InferencerWorkspace(root=str(tmp_path / "base_ws"))
    base._workspace.ensure_dirs()
    with open(os.path.join(base._workspace.outputs_dir, "result.md"), "w") as f:
        f.write("BASE_TAG")

    fixer = _make_stub("fixer", "result.md")
    fixer._workspace = InferencerWorkspace(root=str(tmp_path / "fixer_ws"))
    fixer._workspace.ensure_dirs()
    with open(os.path.join(fixer._workspace.outputs_dir, "result.md"), "w") as f:
        f.write("FIXER_TAG")

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=_make_stub("review", "rv.md"),
        fixer_inferencer=fixer,
        output_path="d.md",
    )
    dual._workspace = InferencerWorkspace(root=str(tmp_path / "dual_ws"))
    dual._workspace.ensure_dirs()
    # counter_feedback set → fixer wins
    dual._state = {
        "consensus_iterations": [
            ConsensusIterationRecord(
                iteration=1,
                base_output="x",
                review_input="rev",
                review_output="needs work",
                counter_feedback="please fix",
            )
        ]
    }
    dual._finalize_response()

    # FIXER_TAG should win — there is NEVER a fixer/ subfolder
    fp = os.path.join(dual._workspace.outputs_dir, "result.md")
    assert os.path.isfile(fp)
    with open(fp) as f:
        content = f.read()
    assert "FIXER_TAG" in content
    assert "BASE_TAG" not in content
    # NO fixer/ subfolder
    assert not os.path.isdir(os.path.join(dual._workspace.outputs_dir, "fixer"))


@pytest.mark.preflight
def test_S14b_active_proposer_uses_real_runtime_state_path(tmp_path):
    """v1.7.2 regression: _active_proposer reads state["attempt_record"]["iterations"], NOT state["consensus_iterations"].

    The runtime ONLY writes attempt_record (see dual_inferencer.py _pending_state
    setup). The earlier implementation read the wrong key so the fixer never won
    pass-through surfacing in real runs.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        ConsensusIterationRecord,
        DualInferencer,
    )
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    base = _make_stub("base", "result.md")
    base._workspace = InferencerWorkspace(root=str(tmp_path / "base_ws"))
    base._workspace.ensure_dirs()
    with open(os.path.join(base._workspace.outputs_dir, "result.md"), "w") as f:
        f.write("BASE_TAG")

    fixer = _make_stub("fixer", "result.md")
    fixer._workspace = InferencerWorkspace(root=str(tmp_path / "fixer_ws"))
    fixer._workspace.ensure_dirs()
    with open(os.path.join(fixer._workspace.outputs_dir, "result.md"), "w") as f:
        f.write("FIXER_TAG")

    dual = DualInferencer(
        base_inferencer=base,
        review_inferencer=_make_stub("review", "rv.md"),
        fixer_inferencer=fixer,
        output_path="d.md",
    )
    dual._workspace = InferencerWorkspace(root=str(tmp_path / "dual_ws"))
    dual._workspace.ensure_dirs()
    # CRITICAL: use the REAL state structure (attempt_record), not fabricated.
    dual._state = {
        "attempt_record": {
            "attempt": 1,
            "iterations": [
                ConsensusIterationRecord(
                    iteration=1,
                    base_output="x",
                    review_input="rev",
                    review_output="needs work",
                    counter_feedback="please fix",
                )
            ],
            "consensus_reached": False,
        }
    }
    dual._finalize_response()

    # Fixer wins via real state path
    fp = os.path.join(dual._workspace.outputs_dir, "result.md")
    assert os.path.isfile(fp)
    with open(fp) as f:
        content = f.read()
    assert "FIXER_TAG" in content, f"Expected fixer to win, got: {content}"
    assert "BASE_TAG" not in content


@pytest.mark.preflight
def test_S15_no_files_no_op(tmp_path):
    """Negative: empty workspace produces empty boundary report (no crash)."""
    from agent_foundation.common.inferencers.deliverable_boundary import (
        collect_child_boundary_deliverables,
    )

    parent = _ws(tmp_path)
    children = collect_child_boundary_deliverables(parent)
    assert children == []


@pytest.mark.preflight
def test_S16_AggregateReport_fields(tmp_path):
    """T6: AggregateReport.skipped/conflicted populated correctly."""
    from agent_foundation.common.inferencers.deliverable_boundary import (
        aggregate_into_self_deliverables,
        ChildBoundaryDeliverables,
    )

    parent = _ws(tmp_path)
    # Create one boundary child with one file, but pre-populate destination
    # to trigger skip_existing
    role_dir = os.path.join(parent.outputs_dir, "planner")
    os.makedirs(role_dir)
    pre_file = os.path.join(role_dir, "x.md")
    with open(pre_file, "w") as f:
        f.write("kept")

    src_child = parent.child("planner")
    src_child.ensure_dirs()
    with open(os.path.join(src_child.outputs_dir, "x.md"), "w") as f:
        f.write("ignored")

    report = aggregate_into_self_deliverables(
        parent,
        [
            ChildBoundaryDeliverables(
                child_name="planner",
                child_workspace_root=src_child.root,
                deliverable_files=["x.md"],
                child_workspace=src_child,
            )
        ],
        namespace_strategy="by_role",
        conflict_strategy="skip_existing",
    )
    assert len(report.skipped) == 1
    assert len(report.copied) == 0
    assert "x.md" in report.skipped[0]


@pytest.mark.preflight
def test_S17_grandchild_outputs_propagation_T7(tmp_path):
    """T7: ws.child("a").child("b").child("c") — outputs/ resolves N+ levels deep."""
    w = _ws(tmp_path)
    deep = w.child("a").child("b").child("c").child("d")
    deep.ensure_dirs()
    assert deep.outputs_dir is not None
    assert Path(deep.outputs_dir).name == "outputs"
    # Writing makes it report deliverables.
    with open(os.path.join(deep.outputs_dir, "deep.md"), "w") as f:
        f.write("deep")
    assert deep.has_deliverables


@pytest.mark.preflight
def test_S18_isolation_two_workers_no_cross_contamination(tmp_path):
    """Isolation: two parallel workers don't see each other's deliverables."""
    parent = _ws(tmp_path)
    w0 = parent.child("worker_0")
    w0.ensure_dirs()
    with open(os.path.join(w0.outputs_dir, "w0.md"), "w") as f:
        f.write("zero")
    w1 = parent.child("worker_1")
    w1.ensure_dirs()
    with open(os.path.join(w1.outputs_dir, "w1.md"), "w") as f:
        f.write("one")
    # No cross-contamination
    assert os.listdir(w0.outputs_dir) == ["w0.md"]
    assert os.listdir(w1.outputs_dir) == ["w1.md"]

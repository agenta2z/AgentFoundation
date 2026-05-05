"""Phase 1 unit tests for deliverable_boundary.py helpers.

Covers:
- collect_child_boundary_deliverables (in-process + on-disk)
- aggregate_into_self_deliverables (4 conflict strategies, 3 namespace strategies)
- surface_boundary_deliverables (pass-through helper)
- T2 (error conflict strategy raises)
- T6 (AggregateReport fields populated)
"""

import os

import pytest

from agent_foundation.common.inferencers.deliverable_boundary import (
    AggregateReport,
    ChildBoundaryDeliverables,
    DeliverableConflictError,
    aggregate_into_self_deliverables,
    collect_child_boundary_deliverables,
    surface_boundary_deliverables,
)
from agent_foundation.common.inferencers.inferencer_workspace import (
    InferencerWorkspace,
)


def _make_parent_with_workers(tmp_path, n=3, content_template="content_{i}"):
    """Helper: build a parent workspace with N child workers each having
    a single deliverable file."""
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    children_ws = []
    for i in range(n):
        child = parent.child(f"worker_{i}")
        child.ensure_dirs()
        with open(os.path.join(child.deliverables_dir, f"out_{i}.md"), "w") as f:
            f.write(content_template.format(i=i))
        children_ws.append(child)
    return parent, children_ws


# ---------------------------------------------------------------------------
# collect_child_boundary_deliverables
# ---------------------------------------------------------------------------


def test_collect_on_disk_finds_all_workers(tmp_path):
    parent, _ = _make_parent_with_workers(tmp_path, n=3)
    children = collect_child_boundary_deliverables(parent)
    assert len(children) == 3
    names = [c.child_name for c in children]
    assert names == ["worker_0", "worker_1", "worker_2"]


def test_collect_on_disk_skips_empty_dirs(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    # worker_0 has files
    w0 = parent.child("worker_0")
    w0.ensure_dirs()
    with open(os.path.join(w0.deliverables_dir, "x.md"), "w") as f:
        f.write("x")
    # worker_1 has empty deliverables dir
    w1 = parent.child("worker_1")
    w1.ensure_dirs()
    children = collect_child_boundary_deliverables(parent)
    assert len(children) == 1
    assert children[0].child_name == "worker_0"


def test_collect_with_filter(tmp_path):
    parent, _ = _make_parent_with_workers(tmp_path, n=3)
    only_w1 = collect_child_boundary_deliverables(
        parent,
        boundary_filter=lambda name, ws: name == "worker_1",
    )
    assert len(only_w1) == 1
    assert only_w1[0].child_name == "worker_1"


def test_collect_returns_empty_when_no_children_dir(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    children = collect_child_boundary_deliverables(parent)
    assert children == []


def test_collect_with_in_process_inferencer(tmp_path):
    """When parent_inferencer is provided, use is_deliverable_boundary flag."""
    from attr import attrib, attrs

    from agent_foundation.common.inferencers.inferencer_base import (
        InferencerBase,
    )

    @attrs
    class MockChild(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **kwargs):
            return "ok"

    @attrs
    class MockParent(InferencerBase):
        child_a: object = attrib(default=None)
        child_b: object = attrib(default=None)

        def _infer(self, inference_input, inference_config=None, **kwargs):
            return "ok"

    parent_ws = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent_ws.ensure_dirs()

    # Set up children: child_a is boundary, child_b is not
    child_a_ws = parent_ws.child("child_a")
    child_a_ws.ensure_dirs()
    with open(os.path.join(child_a_ws.deliverables_dir, "a.md"), "w") as f:
        f.write("a")

    child_b_ws = parent_ws.child("child_b")
    child_b_ws.ensure_dirs()
    with open(os.path.join(child_b_ws.deliverables_dir, "b.md"), "w") as f:
        f.write("b")

    child_a = MockChild(workspace=child_a_ws, is_deliverable_boundary=True)
    child_b = MockChild(workspace=child_b_ws, is_deliverable_boundary=False)
    parent = MockParent(
        workspace=parent_ws, child_a=child_a, child_b=child_b
    )

    # In-process detection: only child_a is a boundary
    children = collect_child_boundary_deliverables(parent_ws, parent)
    # NOTE: in-process pass finds child_a; on-disk fallback may find child_b too
    # Since child_b has files on disk but is_deliverable_boundary=False,
    # the on-disk fallback will still pick it up (graceful degradation).
    # The in-process pass guarantees child_a is found via the FLAG.
    names = [c.child_name for c in children]
    assert "child_a" in names


# ---------------------------------------------------------------------------
# aggregate_into_self_deliverables — namespace strategies
# ---------------------------------------------------------------------------


def test_aggregate_by_child_name(tmp_path):
    parent, _ = _make_parent_with_workers(tmp_path, n=3)
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(parent, children)
    assert len(report.copied) == 3
    paths = parent.deliverable_paths()
    assert "worker_0/out_0.md" in paths
    assert "worker_1/out_1.md" in paths
    assert "worker_2/out_2.md" in paths


def test_aggregate_with_namespace_root(tmp_path):
    parent, _ = _make_parent_with_workers(tmp_path, n=2)
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent, children, namespace_root="workers"
    )
    paths = parent.deliverable_paths()
    assert "workers/worker_0/out_0.md" in paths
    assert "workers/worker_1/out_1.md" in paths


def test_aggregate_flat(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    # Two workers with DIFFERENT filenames (no conflict)
    for i, name in enumerate(["alpha.md", "beta.md"]):
        w = parent.child(f"w{i}")
        w.ensure_dirs()
        with open(os.path.join(w.deliverables_dir, name), "w") as f:
            f.write(name)
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent, children, namespace_strategy="flat"
    )
    paths = parent.deliverable_paths()
    assert "alpha.md" in paths
    assert "beta.md" in paths
    assert len(report.copied) == 2


def test_aggregate_by_role(tmp_path):
    """by_role uses child_name directly; caller provides role-typed names."""
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    for role in ["planner", "executor"]:
        w = parent.child(role)
        w.ensure_dirs()
        with open(os.path.join(w.deliverables_dir, f"{role}.md"), "w") as f:
            f.write(role)
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent, children, namespace_strategy="by_role"
    )
    paths = parent.deliverable_paths()
    assert "planner/planner.md" in paths
    assert "executor/executor.md" in paths


# ---------------------------------------------------------------------------
# aggregate_into_self_deliverables — conflict strategies
# ---------------------------------------------------------------------------


def test_conflict_skip_existing(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    # Pre-populate dst
    with open(os.path.join(parent.deliverables_dir, "shared.md"), "w") as f:
        f.write("existing")
    w = parent.child("w")
    w.ensure_dirs()
    with open(os.path.join(w.deliverables_dir, "shared.md"), "w") as f:
        f.write("new")
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent, children,
        namespace_strategy="flat",
        conflict_strategy="skip_existing",
    )
    assert "shared.md" in report.skipped
    with open(os.path.join(parent.deliverables_dir, "shared.md")) as f:
        assert f.read() == "existing"


def test_conflict_error_raises(tmp_path):
    """T2: conflict_strategy='error' raises DeliverableConflictError."""
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    for i in range(2):
        w = parent.child(f"w{i}")
        w.ensure_dirs()
        with open(os.path.join(w.deliverables_dir, "shared.md"), "w") as f:
            f.write(f"content_{i}")
    children = collect_child_boundary_deliverables(parent)
    with pytest.raises(DeliverableConflictError):
        aggregate_into_self_deliverables(
            parent, children,
            namespace_strategy="flat",
            conflict_strategy="error",
        )


def test_conflict_largest_picks_largest(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    # w0 writes a small file; w1 writes a larger one
    for i, content in enumerate(["a", "this is much larger content"]):
        w = parent.child(f"w{i}")
        w.ensure_dirs()
        with open(os.path.join(w.deliverables_dir, "shared.md"), "w") as f:
            f.write(content)
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent, children,
        namespace_strategy="flat",
        conflict_strategy="largest",
    )
    with open(os.path.join(parent.deliverables_dir, "shared.md")) as f:
        assert f.read() == "this is much larger content"


def test_conflict_first_wins(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    for i in range(2):
        w = parent.child(f"w{i}")
        w.ensure_dirs()
        with open(os.path.join(w.deliverables_dir, "shared.md"), "w") as f:
            f.write(f"content_{i}")
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent, children,
        namespace_strategy="flat",
        conflict_strategy="first_wins",
    )
    # First-seen wins (worker_0 / w0)
    with open(os.path.join(parent.deliverables_dir, "shared.md")) as f:
        assert f.read() == "content_0"
    assert "shared.md" in report.conflicted


# ---------------------------------------------------------------------------
# T6: AggregateReport fields populated correctly
# ---------------------------------------------------------------------------


def test_aggregate_report_fields_populated(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    # Pre-populate dst with one file (skip_existing → goes to skipped)
    with open(os.path.join(parent.deliverables_dir, "preexist.md"), "w") as f:
        f.write("orig")
    # w0 will have preexist.md (collision) + new.md (will copy)
    w0 = parent.child("w0")
    w0.ensure_dirs()
    with open(os.path.join(w0.deliverables_dir, "preexist.md"), "w") as f:
        f.write("from-w0")
    with open(os.path.join(w0.deliverables_dir, "new.md"), "w") as f:
        f.write("new")
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent, children,
        namespace_strategy="flat",
        conflict_strategy="skip_existing",
    )
    assert any("new.md" == c[1] for c in report.copied)
    assert "preexist.md" in report.skipped


# ---------------------------------------------------------------------------
# surface_boundary_deliverables (pass-through helper for Dual / LWI)
# ---------------------------------------------------------------------------


def test_surface_boundary_deliverables_basic(tmp_path):
    parent = InferencerWorkspace(
        root=str(tmp_path / "parent"), use_final_deliverables_folder=True
    )
    parent.ensure_dirs()
    child = parent.child("active_proposer")
    child.ensure_dirs()
    with open(os.path.join(child.deliverables_dir, "out.md"), "w") as f:
        f.write("from active proposer")
    copied = surface_boundary_deliverables(parent, child)
    assert "out.md" in copied
    paths = parent.deliverable_paths()
    assert "out.md" in paths


# === v1.7.1 Bug fix tests (post-implementation review) ===

def test_AC7_logging_fires_for_zero_copy(tmp_path, caplog):
    """v1.7 AC7 + Bug 3 fix: aggregate logs INFO even when nothing copied."""
    import logging
    parent = InferencerWorkspace(root=str(tmp_path), use_final_deliverables_folder=True)
    parent.ensure_dirs()
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    role_dir = os.path.join(parent.deliverables_dir, "planner")
    os.makedirs(role_dir)
    with open(os.path.join(role_dir, "x.md"), "w") as f:
        f.write("existing")
    with open(os.path.join(src_child.deliverables_dir, "x.md"), "w") as f:
        f.write("new")

    with caplog.at_level(logging.INFO, logger="agent_foundation.common.inferencers.deliverable_boundary"):
        report = aggregate_into_self_deliverables(
            parent,
            [ChildBoundaryDeliverables("planner", src_child.root, ["x.md"], child_workspace=src_child)],
            namespace_strategy="by_role",
            conflict_strategy="skip_existing",
        )
    assert any("Boundary aggregate" in r.message for r in caplog.records)
    assert len(report.copied) == 0
    assert len(report.skipped) == 1


def test_idempotent_double_run(tmp_path):
    """Running aggregate twice produces identical state (R7 idempotent reruns)."""
    parent = InferencerWorkspace(root=str(tmp_path), use_final_deliverables_folder=True)
    parent.ensure_dirs()
    c = parent.child("worker_0")
    c.ensure_dirs()
    with open(os.path.join(c.deliverables_dir, "out.md"), "w") as f:
        f.write("v1")
    kids = collect_child_boundary_deliverables(parent)
    r1 = aggregate_into_self_deliverables(parent, kids, namespace_strategy="by_child_name", namespace_root="workers")
    assert len(r1.copied) == 1
    r2 = aggregate_into_self_deliverables(parent, kids, namespace_strategy="by_child_name", namespace_root="workers", conflict_strategy="skip_existing")
    assert len(r2.copied) == 0
    assert len(r2.skipped) == 1
    final = os.path.join(parent.deliverables_dir, "workers/worker_0/out.md")
    with open(final) as f:
        assert f.read() == "v1"


def test_largest_pre_existing_smaller_overwritten(tmp_path):
    """Bug 1 fix: largest strategy overwrites pre-existing SMALLER file."""
    parent = InferencerWorkspace(root=str(tmp_path), use_final_deliverables_folder=True)
    parent.ensure_dirs()
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    role_dir = os.path.join(parent.deliverables_dir, "planner")
    os.makedirs(role_dir)
    pre = os.path.join(role_dir, "x.md")
    with open(pre, "w") as f:
        f.write("X")
    with open(os.path.join(src_child.deliverables_dir, "x.md"), "w") as f:
        f.write("XXXXXXXXX")
    report = aggregate_into_self_deliverables(
        parent,
        [ChildBoundaryDeliverables("planner", src_child.root, ["x.md"], child_workspace=src_child)],
        namespace_strategy="by_role", conflict_strategy="largest",
    )
    assert len(report.copied) == 1
    with open(pre) as f:
        assert f.read() == "XXXXXXXXX"


def test_largest_pre_existing_larger_kept(tmp_path):
    """Bug 1 fix: largest KEEPS pre-existing LARGER file."""
    parent = InferencerWorkspace(root=str(tmp_path), use_final_deliverables_folder=True)
    parent.ensure_dirs()
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    role_dir = os.path.join(parent.deliverables_dir, "planner")
    os.makedirs(role_dir)
    pre = os.path.join(role_dir, "x.md")
    with open(pre, "w") as f:
        f.write("BIGFILECONTENT")
    with open(os.path.join(src_child.deliverables_dir, "x.md"), "w") as f:
        f.write("smol")
    report = aggregate_into_self_deliverables(
        parent,
        [ChildBoundaryDeliverables("planner", src_child.root, ["x.md"], child_workspace=src_child)],
        namespace_strategy="by_role", conflict_strategy="largest",
    )
    assert len(report.copied) == 0
    assert len(report.conflicted) == 1
    with open(pre) as f:
        assert f.read() == "BIGFILECONTENT"


def test_first_wins_pre_existing_kept(tmp_path):
    """Bug 1 fix: first_wins keeps pre-existing file."""
    parent = InferencerWorkspace(root=str(tmp_path), use_final_deliverables_folder=True)
    parent.ensure_dirs()
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    role_dir = os.path.join(parent.deliverables_dir, "planner")
    os.makedirs(role_dir)
    pre = os.path.join(role_dir, "x.md")
    with open(pre, "w") as f:
        f.write("FIRST")
    with open(os.path.join(src_child.deliverables_dir, "x.md"), "w") as f:
        f.write("LATER")
    report = aggregate_into_self_deliverables(
        parent,
        [ChildBoundaryDeliverables("planner", src_child.root, ["x.md"], child_workspace=src_child)],
        namespace_strategy="by_role", conflict_strategy="first_wins",
    )
    assert len(report.copied) == 0
    assert len(report.conflicted) == 1
    with open(pre) as f:
        assert f.read() == "FIRST"

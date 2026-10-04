"""Phase 1 unit tests for deliverable_boundary.py helpers.

Part 2 (two-axis model) retired the ``final_deliverables/`` subfolder and the
``is_deliverable_boundary`` inferencer flag. Deliverables live directly in
``outputs/`` and boundary SELECTION is driven by ``has_deliverables``
(outputs/ non-empty) AND the caller-supplied ``boundary_filter`` (PTI passes a
role allow-list). These tests were retargeted: every ``child.deliverables_dir``
now reads/writes ``outputs/`` (via ``output_path``/``outputs_dir``), and the
in-process detection test drives selection via ``boundary_filter`` instead of the
deleted flag.

Covers:
- collect_child_boundary_deliverables (in-process + on-disk; boundary_filter selection)
- aggregate_into_self_deliverables (4 conflict strategies, 3 namespace strategies)
- surface_boundary_deliverables (pass-through helper)
- T2 (error conflict strategy raises)
- T6 (AggregateReport fields populated)
"""

import os

import pytest
from agent_foundation.common.inferencers.deliverable_boundary import (
    aggregate_into_self_deliverables,
    AggregateReport,
    ChildBoundaryDeliverables,
    collect_child_boundary_deliverables,
    DeliverableConflictError,
    surface_boundary_deliverables,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


def _make_parent_with_workers(tmp_path, n=3, content_template="content_{i}"):
    """Helper: build a parent workspace with N child workers each having
    a single deliverable file in outputs/."""
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    children_ws = []
    for i in range(n):
        child = parent.child(f"worker_{i}")
        child.ensure_dirs()
        with open(child.output_path(f"out_{i}.md"), "w") as f:
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
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    # worker_0 has files
    w0 = parent.child("worker_0")
    w0.ensure_dirs()
    with open(w0.output_path("x.md"), "w") as f:
        f.write("x")
    # worker_1 has empty outputs dir (has_deliverables is False)
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
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    children = collect_child_boundary_deliverables(parent)
    assert children == []


def test_collect_in_process_selects_via_boundary_filter(tmp_path):
    """Part 2: with ``is_deliverable_boundary`` retired, a child is a deliverable
    source iff it ``has_deliverables`` (outputs/ non-empty) AND passes the
    caller-supplied ``boundary_filter``. PTI passes a role allow-list; here we
    select only ``child_a`` by name even though both children have deliverables
    on disk.
    """
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    from attr import attrib, attrs

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

    parent_ws = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent_ws.ensure_dirs()

    # Both children produce deliverables on disk (outputs/ non-empty).
    child_a_ws = parent_ws.child("child_a")
    child_a_ws.ensure_dirs()
    with open(child_a_ws.output_path("a.md"), "w") as f:
        f.write("a")

    child_b_ws = parent_ws.child("child_b")
    child_b_ws.ensure_dirs()
    with open(child_b_ws.output_path("b.md"), "w") as f:
        f.write("b")

    child_a = MockChild(workspace=child_a_ws)
    child_b = MockChild(workspace=child_b_ws)
    parent = MockParent(workspace=parent_ws, child_a=child_a, child_b=child_b)

    # Role allow-list filter selects only child_a; child_b is excluded even
    # though it has deliverables on disk (both in-process and on-disk passes
    # honor boundary_filter).
    children = collect_child_boundary_deliverables(
        parent_ws,
        parent,
        boundary_filter=lambda name, ws: name == "child_a",
    )
    names = [c.child_name for c in children]
    assert "child_a" in names
    assert "child_b" not in names


def test_collect_in_process_requires_has_deliverables(tmp_path):
    """A child inferencer whose outputs/ is empty is NOT selected, even when it
    passes the boundary_filter (has_deliverables gates selection)."""
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    from attr import attrib, attrs

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

    parent_ws = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent_ws.ensure_dirs()

    # child_a has a deliverable; child_b's outputs/ is empty.
    child_a_ws = parent_ws.child("child_a")
    child_a_ws.ensure_dirs()
    with open(child_a_ws.output_path("a.md"), "w") as f:
        f.write("a")
    child_b_ws = parent_ws.child("child_b")
    child_b_ws.ensure_dirs()  # empty outputs/

    parent = MockParent(
        workspace=parent_ws,
        child_a=MockChild(workspace=child_a_ws),
        child_b=MockChild(workspace=child_b_ws),
    )

    # Filter accepts BOTH names, but only child_a has deliverables on disk.
    children = collect_child_boundary_deliverables(
        parent_ws,
        parent,
        boundary_filter=lambda name, ws: True,
    )
    names = [c.child_name for c in children]
    assert names == ["child_a"]


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
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    # Two workers with DIFFERENT filenames (no conflict)
    for i, name in enumerate(["alpha.md", "beta.md"]):
        w = parent.child(f"w{i}")
        w.ensure_dirs()
        with open(w.output_path(name), "w") as f:
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
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    for role in ["planner", "executor"]:
        w = parent.child(role)
        w.ensure_dirs()
        with open(w.output_path(f"{role}.md"), "w") as f:
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
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    # Pre-populate dst
    with open(parent.output_path("shared.md"), "w") as f:
        f.write("existing")
    w = parent.child("w")
    w.ensure_dirs()
    with open(w.output_path("shared.md"), "w") as f:
        f.write("new")
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent,
        children,
        namespace_strategy="flat",
        conflict_strategy="skip_existing",
    )
    assert "shared.md" in report.skipped
    with open(parent.output_path("shared.md")) as f:
        assert f.read() == "existing"


def test_conflict_error_raises(tmp_path):
    """T2: conflict_strategy='error' raises DeliverableConflictError."""
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    for i in range(2):
        w = parent.child(f"w{i}")
        w.ensure_dirs()
        with open(w.output_path("shared.md"), "w") as f:
            f.write(f"content_{i}")
    children = collect_child_boundary_deliverables(parent)
    with pytest.raises(DeliverableConflictError):
        aggregate_into_self_deliverables(
            parent,
            children,
            namespace_strategy="flat",
            conflict_strategy="error",
        )


def test_conflict_largest_picks_largest(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    # w0 writes a small file; w1 writes a larger one
    for i, content in enumerate(["a", "this is much larger content"]):
        w = parent.child(f"w{i}")
        w.ensure_dirs()
        with open(w.output_path("shared.md"), "w") as f:
            f.write(content)
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent,
        children,
        namespace_strategy="flat",
        conflict_strategy="largest",
    )
    with open(parent.output_path("shared.md")) as f:
        assert f.read() == "this is much larger content"


def test_conflict_first_wins(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    for i in range(2):
        w = parent.child(f"w{i}")
        w.ensure_dirs()
        with open(w.output_path("shared.md"), "w") as f:
            f.write(f"content_{i}")
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent,
        children,
        namespace_strategy="flat",
        conflict_strategy="first_wins",
    )
    # First-seen wins (worker_0 / w0)
    with open(parent.output_path("shared.md")) as f:
        assert f.read() == "content_0"
    assert "shared.md" in report.conflicted


# ---------------------------------------------------------------------------
# T6: AggregateReport fields populated correctly
# ---------------------------------------------------------------------------


def test_aggregate_report_fields_populated(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    # Pre-populate dst with one file (skip_existing → goes to skipped)
    with open(parent.output_path("preexist.md"), "w") as f:
        f.write("orig")
    # w0 will have preexist.md (collision) + new.md (will copy)
    w0 = parent.child("w0")
    w0.ensure_dirs()
    with open(w0.output_path("preexist.md"), "w") as f:
        f.write("from-w0")
    with open(w0.output_path("new.md"), "w") as f:
        f.write("new")
    children = collect_child_boundary_deliverables(parent)
    report = aggregate_into_self_deliverables(
        parent,
        children,
        namespace_strategy="flat",
        conflict_strategy="skip_existing",
    )
    assert any("new.md" == c[1] for c in report.copied)
    assert "preexist.md" in report.skipped


# ---------------------------------------------------------------------------
# surface_boundary_deliverables (pass-through helper for Dual / LWI)
# ---------------------------------------------------------------------------


def test_surface_boundary_deliverables_basic(tmp_path):
    parent = InferencerWorkspace(root=str(tmp_path / "parent"))
    parent.ensure_dirs()
    child = parent.child("active_proposer")
    child.ensure_dirs()
    with open(child.output_path("out.md"), "w") as f:
        f.write("from active proposer")
    copied = surface_boundary_deliverables(parent, child)
    assert "out.md" in copied
    paths = parent.deliverable_paths()
    assert "out.md" in paths


# === v1.7.1 Bug fix tests (post-implementation review) ===


def test_AC7_logging_fires_for_zero_copy(tmp_path, caplog):
    """v1.7 AC7 + Bug 3 fix: aggregate logs INFO even when nothing copied."""
    import logging

    parent = InferencerWorkspace(root=str(tmp_path))
    parent.ensure_dirs()
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    role_dir = parent.output_path("planner")
    os.makedirs(role_dir)
    with open(os.path.join(role_dir, "x.md"), "w") as f:
        f.write("existing")
    with open(src_child.output_path("x.md"), "w") as f:
        f.write("new")

    with caplog.at_level(
        logging.INFO, logger="agent_foundation.common.inferencers.deliverable_boundary"
    ):
        report = aggregate_into_self_deliverables(
            parent,
            [
                ChildBoundaryDeliverables(
                    "planner", src_child.root, ["x.md"], child_workspace=src_child
                )
            ],
            namespace_strategy="by_role",
            conflict_strategy="skip_existing",
        )
    assert any("Boundary aggregate" in r.message for r in caplog.records)
    assert len(report.copied) == 0
    assert len(report.skipped) == 1


def test_idempotent_double_run(tmp_path):
    """Running aggregate twice produces identical state (R7 idempotent reruns)."""
    parent = InferencerWorkspace(root=str(tmp_path))
    parent.ensure_dirs()
    c = parent.child("worker_0")
    c.ensure_dirs()
    with open(c.output_path("out.md"), "w") as f:
        f.write("v1")
    kids = collect_child_boundary_deliverables(parent)
    r1 = aggregate_into_self_deliverables(
        parent, kids, namespace_strategy="by_child_name", namespace_root="workers"
    )
    assert len(r1.copied) == 1
    r2 = aggregate_into_self_deliverables(
        parent,
        kids,
        namespace_strategy="by_child_name",
        namespace_root="workers",
        conflict_strategy="skip_existing",
    )
    assert len(r2.copied) == 0
    assert len(r2.skipped) == 1
    final = parent.output_path("workers/worker_0/out.md")
    with open(final) as f:
        assert f.read() == "v1"


def test_largest_pre_existing_smaller_overwritten(tmp_path):
    """Bug 1 fix: largest strategy overwrites pre-existing SMALLER file."""
    parent = InferencerWorkspace(root=str(tmp_path))
    parent.ensure_dirs()
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    role_dir = parent.output_path("planner")
    os.makedirs(role_dir)
    pre = os.path.join(role_dir, "x.md")
    with open(pre, "w") as f:
        f.write("X")
    with open(src_child.output_path("x.md"), "w") as f:
        f.write("XXXXXXXXX")
    report = aggregate_into_self_deliverables(
        parent,
        [
            ChildBoundaryDeliverables(
                "planner", src_child.root, ["x.md"], child_workspace=src_child
            )
        ],
        namespace_strategy="by_role",
        conflict_strategy="largest",
    )
    assert len(report.copied) == 1
    with open(pre) as f:
        assert f.read() == "XXXXXXXXX"


def test_largest_pre_existing_larger_kept(tmp_path):
    """Bug 1 fix: largest KEEPS pre-existing LARGER file."""
    parent = InferencerWorkspace(root=str(tmp_path))
    parent.ensure_dirs()
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    role_dir = parent.output_path("planner")
    os.makedirs(role_dir)
    pre = os.path.join(role_dir, "x.md")
    with open(pre, "w") as f:
        f.write("BIGFILECONTENT")
    with open(src_child.output_path("x.md"), "w") as f:
        f.write("smol")
    report = aggregate_into_self_deliverables(
        parent,
        [
            ChildBoundaryDeliverables(
                "planner", src_child.root, ["x.md"], child_workspace=src_child
            )
        ],
        namespace_strategy="by_role",
        conflict_strategy="largest",
    )
    assert len(report.copied) == 0
    assert len(report.conflicted) == 1
    with open(pre) as f:
        assert f.read() == "BIGFILECONTENT"


def test_first_wins_pre_existing_kept(tmp_path):
    """Bug 1 fix: first_wins keeps pre-existing file."""
    parent = InferencerWorkspace(root=str(tmp_path))
    parent.ensure_dirs()
    src_child = parent.child("planner")
    src_child.ensure_dirs()
    role_dir = parent.output_path("planner")
    os.makedirs(role_dir)
    pre = os.path.join(role_dir, "x.md")
    with open(pre, "w") as f:
        f.write("FIRST")
    with open(src_child.output_path("x.md"), "w") as f:
        f.write("LATER")
    report = aggregate_into_self_deliverables(
        parent,
        [
            ChildBoundaryDeliverables(
                "planner", src_child.root, ["x.md"], child_workspace=src_child
            )
        ],
        namespace_strategy="by_role",
        conflict_strategy="first_wins",
    )
    assert len(report.copied) == 0
    assert len(report.conflicted) == 1
    with open(pre) as f:
        assert f.read() == "FIRST"

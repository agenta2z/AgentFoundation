"""RunContext: child() semantics, workspace derivation, handles, node lazy (M1 / §2.0)."""

import pytest

from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext, RuntimeBindings


def _root():
    return RunContext.root(workspace=InferencerWorkspace(root="/tmp/run"))


def test_child_composes_logical_path():
    root = _root()
    assert root.path == "/"
    assert root.child("a").path == "/a"
    assert root.child("a").child("b").path == "/a/b"


def test_child_workspace_uses_real_single_component_api_children_nesting():
    """§2.0 Note A: workspace nests via children/, NOT a naive '/a/b' join."""
    root = _root()
    ws = root.child("review").child("worker_0").workspace
    assert ws.root == "/tmp/run/children/review/children/worker_0"


def test_child_rejects_multi_segment_slot():
    root = _root()
    for bad in ("a/b", "followup_iterations/iteration_2", "..", ".", "", "a\\b"):
        with pytest.raises(ValueError):
            root.child(bad)


def test_child_is_fresh_context_not_a_view():
    root = _root()
    c1 = root.child("x")
    c2 = root.child("x")
    assert c1 is not c2  # fresh object each time
    assert c1.path == c2.path


def test_child_shares_tier2_and_stores_by_reference():
    root = _root()
    c = root.child("x")
    assert c.runtime is root.runtime  # Tier-2 shared by reference
    assert c._store is root._store  # Tier-1 shared (per-turn)
    assert c._handle_store is root._handle_store  # Tier-3 shared (connection-scoped)


def test_node_is_lazy_and_idempotent_per_path():
    root = _root()
    c = root.child("x")
    assert c.node() is c.node()  # same node on re-access
    assert c.node() is root.child("x").node()  # re-derived child -> same store node


def test_handles_are_idempotent_per_path_and_reused_on_rederivation():
    root = _root()
    h1 = root.child("x").handles
    h1.sdk_client = "CLIENT"
    # Re-deriving the same child path returns the SAME handles (get-or-create).
    h2 = root.child("x").handles
    assert h2 is h1
    assert h2.sdk_client == "CLIENT"


def test_sibling_paths_have_isolated_handles():
    root = _root()
    a = root.child("worker_0").handles
    b = root.child("worker_1").handles
    a.session_id = "A"
    b.session_id = "B"
    assert a is not b
    assert a.session_id == "A" and b.session_id == "B"


def test_root_factory_provides_default_tiers():
    root = RunContext.root(workspace=None)
    assert isinstance(root.runtime, RuntimeBindings)
    assert root.node() is not None

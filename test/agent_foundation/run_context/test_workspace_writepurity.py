"""M7 workspace write-purity: under a context the orchestrator publishes the child
workspace to ctx and does NOT mutate the child instance; reads resolve from ctx.
Byte-identical (instance mutation) without a context."""

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext, enter_run, exit_run


@attrs
class _Inf(InferencerBase):
    def _infer(self, x, inference_config=None, **kw):
        return x


def _orchestrate_like_dual(orch, child, slot, ws):
    """Mirror the Dual/BTA guarded-write pattern."""
    child_ctx = orch._rc_child(slot)
    orch._publish_workspace_to_ctx(child_ctx, ws)
    if child_ctx is None:
        child._workspace = ws  # legacy fallback
    return orch._read_child_workspace(child, slot)


def test_under_context_child_instance_not_mutated_but_read_resolves():
    orch, child = _Inf(), _Inf()
    ws = InferencerWorkspace(root="/tmp/run/review")
    root = RunContext.root(workspace=InferencerWorkspace(root="/tmp/run"))
    tok = enter_run(root)
    try:
        eff = _orchestrate_like_dual(orch, child, "review", ws)
        # write-pure: the child instance backing was NOT set...
        assert child.__dict__.get("_InferencerBase__workspace") is None
        # ...yet the orchestrator-side read resolves the published workspace.
        assert eff.root == "/tmp/run/review"
    finally:
        exit_run(tok)


def test_without_context_mutates_instance_byte_identical():
    orch, child = _Inf(), _Inf()
    ws = InferencerWorkspace(root="/tmp/run/review")
    # no active ctx -> legacy: child instance IS set, read resolves it
    eff = _orchestrate_like_dual(orch, child, "review", ws)
    assert child._workspace.root == "/tmp/run/review"
    assert eff.root == "/tmp/run/review"


def test_two_workers_get_isolated_workspaces_under_context():
    orch, w0, w1 = _Inf(), _Inf(), _Inf()
    root = RunContext.root(workspace=InferencerWorkspace(root="/tmp/run"))
    tok = enter_run(root)
    try:
        e0 = _orchestrate_like_dual(orch, w0, "worker_0", InferencerWorkspace(root="/tmp/run/w0"))
        e1 = _orchestrate_like_dual(orch, w1, "worker_1", InferencerWorkspace(root="/tmp/run/w1"))
        assert e0.root == "/tmp/run/w0" and e1.root == "/tmp/run/w1"
        assert w0.__dict__.get("_InferencerBase__workspace") is None  # neither mutated
        assert w1.__dict__.get("_InferencerBase__workspace") is None
    finally:
        exit_run(tok)

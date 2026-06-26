"""The UNIFIED default-context model (D6/§2.2 mint policy), made explicit + asserted.

There is NOT a parallel "legacy vs ctx" code path. Every public ``infer``/``ainfer``
runs under a single normalization (the bridge ``enter_run``), so DOWNSTREAM code never
branches on ``run_context is None`` — it reads ``active_run_context()``, which is:

  * the caller's RunContext, if provided (host-managed run); else
  * the already-active context, if this is a NESTED public call (reuse, no detached
    root); else
  * a FRESH default root minted from the instance's own fields (legacy call).

The only "legacy" part is how that default root is *initialized* from today's instance
fields — not a second behavioral branch. The instance-resident reads that remain are
the deliberate option-(a)/D6 carve-outs (post-call readers) and the out-of-inference
fallback (a direct ``_infer``/setup call with no run to attach a context to) — NOT a
default mechanism. These tests pin that model so it can't silently regress into two paths.
"""

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    active_run_context,
)


@attrs(slots=False)
class _Leaf(InferencerBase):
    def _infer(self, x, inference_config=None, **kw):
        # capture the active context seen DURING inference
        self.__dict__["_seen_ctx"] = active_run_context()
        return x


def test_legacy_call_still_runs_under_a_context():
    """run_context=None is NOT a separate code path — the bridge mints a default
    root, so ``_infer`` always sees a real context."""
    leaf = _Leaf()
    leaf.infer("x")
    assert leaf.__dict__["_seen_ctx"] is not None


def test_explicit_context_is_the_one_seen():
    leaf = _Leaf()
    root = RunContext.root(workspace=None)
    leaf.infer("x", run_context=root)
    assert leaf.__dict__["_seen_ctx"] is root


def test_default_root_is_fresh_per_call_not_a_shared_singleton():
    """Critical: the default context is minted PER true-root call (never shared on
    the instance), so per-run state can't leak across runs."""
    leaf = _Leaf()
    leaf.infer("a")
    store_a = leaf.__dict__["_seen_ctx"]._store
    leaf.infer("b")
    store_b = leaf.__dict__["_seen_ctx"]._store
    assert store_a is not store_b


def test_default_root_is_initialized_from_the_instance_workspace():
    """The 'legacy' part is only how the default root is INITIALIZED from instance
    fields — here, the instance workspace flows into ``ctx.workspace``."""
    leaf = _Leaf()
    object.__setattr__(
        leaf, "_InferencerBase__workspace", InferencerWorkspace(root="/tmp/inst_ws")
    )
    leaf.infer("x")
    ctx = leaf.__dict__["_seen_ctx"]
    assert ctx.workspace is not None and ctx.workspace.root == "/tmp/inst_ws"


def test_nested_public_call_reuses_active_context_no_detached_root():
    """A nested public call with run_context=None REUSES the active parent context
    (the §2.2 mint policy) — it does not mint a detached root."""

    child = _Leaf()

    @attrs(slots=False)
    class _Parent(InferencerBase):
        def _infer(self, x, inference_config=None, **kw):
            self.__dict__["_parent_ctx"] = active_run_context()
            child.infer("inner")  # nested public call, run_context=None
            return x

    parent = _Parent()
    parent.infer("outer")
    # the nested child saw the SAME store as the parent (reused, not a fresh root)
    assert child.__dict__["_seen_ctx"]._store is parent.__dict__["_parent_ctx"]._store


def test_out_of_inference_direct_call_has_no_context():
    """The ONLY no-context case is a direct ``_infer`` outside any public entry —
    there is genuinely no run to attach a context to (instance is the right home)."""
    assert active_run_context() is None

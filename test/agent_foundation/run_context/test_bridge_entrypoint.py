"""Regression tests for the §9.3 E4 ``bridge_entrypoint`` decorator.

The CLI / streaming leaf public overrides (rovodev/codex/claude_code/kiro/devmate
``ainfer``/``infer``) bypass ``InferencerBase.ainfer`` and call ``_ainfer_single``
directly.  Without installing the ``_active_ctx`` bridge they render their prompt and
resolve their session/handles under the **caller's** active context instead of their
own.  Under a parallel orchestrator (e.g. the BTA/MFI aggregator invoked directly on
the propose node) the leaf then claims the parent's node and trips the §2.7 creator
collision guard (``CollisionError``) — the exact failure observed in the first real
multi-flow plan run.

``bridge_entrypoint`` installs the bridge from the keyword-only ``run_context`` so the
leaf claims its OWN child node; a **true bare call** (no run_context, no active ctx)
installs no context, preserving the leaf's pre-decorator (ctx-less, instance-backed)
behavior byte-for-byte.
"""

import pytest

from agent_foundation.common.inferencers.run_context import (
    CollisionError,
    RunContext,
    RunStateStore,
    active_run_context,
    bridge_entrypoint,
    enter_run,
    exit_run,
)


def _parent_propose():
    """A parent 'propose' node already claimed by a MultiFlowInferencer (the orchestrator)."""
    mfi = RunContext.root(store=RunStateStore()).child("propose")
    mfi.node(creator=("MultiFlowInferencer", "/propose"))
    return mfi


class _Leaf:
    """A leaf whose decorated public ``ainfer`` claims its node (mimicking ``_effective_role``)."""

    _workspace = None

    @bridge_entrypoint
    async def ainfer(self, inp, inference_config=None, **kwargs):
        ctx = active_run_context()
        if ctx is not None:
            ctx.node(creator=("RovoDevCliInferencer", ctx.path))
        return ctx.path if ctx is not None else None


class _UndecoratedLeaf:
    """The pre-fix leaf: no bridge → renders under the caller's ctx (reproduces the bug)."""

    _workspace = None

    async def ainfer(self, inp, inference_config=None, **kwargs):
        ctx = active_run_context()
        ctx.node(creator=("RovoDevCliInferencer", ctx.path))
        return ctx.path


@pytest.mark.asyncio
async def test_provided_run_context_isolates_aggregator_no_collision():
    """The aggregator dispatch passes run_context=parent.child('aggregator'); the leaf
    must claim THAT node, not the parent's propose node."""
    mfi = _parent_propose()
    token = enter_run(mfi)  # the orchestrator's active ctx is /propose
    try:
        path = await _Leaf().ainfer("x", run_context=mfi.child("aggregator"))
    finally:
        exit_run(token)
    assert path == "/propose/aggregator"


@pytest.mark.asyncio
async def test_undecorated_leaf_reproduces_the_collision():
    """Without the decorator the leaf renders under the active /propose ctx and trips
    the creator guard — the regression this decorator prevents."""
    mfi = _parent_propose()
    token = enter_run(mfi)
    try:
        with pytest.raises(CollisionError):
            await _UndecoratedLeaf().ainfer("x", run_context=mfi.child("aggregator"))
    finally:
        exit_run(token)


@pytest.mark.asyncio
async def test_true_bare_call_installs_no_context():
    """No provided run_context AND no active ctx → the leaf runs ctx-less (byte-identical
    legacy), so per-call state writers hit the instance backing, not a discarded root."""
    assert active_run_context() is None
    path = await _Leaf().ainfer("x")
    assert path is None
    assert active_run_context() is None  # nothing leaked


@pytest.mark.asyncio
async def test_nested_bare_call_reuses_active_context():
    """A bare call WITH a context already active (a flow leaf under an orchestrator that
    set the ctx but did not re-pass it) reuses that ctx rather than minting a detached root."""
    mfi = _parent_propose()
    token = enter_run(mfi.child("flow_0"))
    try:
        path = await _Leaf().ainfer("x")
    finally:
        exit_run(token)
    assert path == "/propose/flow_0"

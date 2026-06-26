"""Bridge: mint policy (legacy-root vs nested-reuse) + per-task ContextVar isolation."""

import asyncio

from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    active_run_context,
    enter_run,
    exit_run,
)


def _ws():
    return InferencerWorkspace(root="/tmp/run")


def test_no_active_ctx_outside_a_run():
    assert active_run_context() is None


def test_provided_run_context_is_used():
    ctx = RunContext.root(workspace=_ws())
    tok = enter_run(ctx)
    try:
        assert active_run_context() is ctx
    finally:
        exit_run(tok)
    assert active_run_context() is None


def test_true_root_none_legacy_mints():
    ws = _ws()
    tok = enter_run(None, default_workspace=ws)
    try:
        active = active_run_context()
        assert active is not None and active.path == "/"
        assert active.workspace is ws  # legacy-mint from the instance workspace
    finally:
        exit_run(tok)


def test_nested_none_reuses_active_ctx_not_a_detached_root():
    """M3: a nested public call with run_context=None must REUSE the active ctx."""
    parent = RunContext.root(workspace=_ws())
    outer = enter_run(parent)
    try:
        assert active_run_context() is parent
        # nested public entrypoint, run_context=None -> reuse, NOT mint a fresh root
        inner = enter_run(None, default_workspace=InferencerWorkspace(root="/other"))
        try:
            assert active_run_context() is parent  # reused, not detached
        finally:
            exit_run(inner)
        assert active_run_context() is parent  # restored
    finally:
        exit_run(outer)


def test_token_restores_previous_on_exit():
    a = RunContext.root(workspace=_ws())
    b = RunContext.root(workspace=_ws())
    ta = enter_run(a)
    tb = enter_run(b)
    assert active_run_context() is b
    exit_run(tb)
    assert active_run_context() is a
    exit_run(ta)
    assert active_run_context() is None


def test_contextvar_is_per_task_under_gather():
    """Each gathered coroutine sets its own per-input child; no sibling clobber.

    This is *why* _active_ctx must be a ContextVar (not a plain attr): N concurrent
    branches on one logical run must each see their own child path.
    """
    root = RunContext.root(workspace=_ws())

    async def worker(i):
        tok = enter_run(root.child(f"parallel_{i}"))
        try:
            await asyncio.sleep(0)  # yield so tasks interleave
            seen = active_run_context().path
            await asyncio.sleep(0)
            # After interleaving, each task still sees ITS OWN child path.
            return seen, active_run_context().path
        finally:
            exit_run(tok)

    async def main():
        return await asyncio.gather(*(worker(i) for i in range(8)))

    results = asyncio.run(main())
    for i, (seen1, seen2) in enumerate(results):
        assert seen1 == f"/parallel_{i}"
        assert seen2 == f"/parallel_{i}"  # not clobbered by siblings
    # Outside the run, nothing leaked.
    assert active_run_context() is None

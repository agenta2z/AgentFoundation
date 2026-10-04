"""Strict ``(store, path)`` path claims (plan v8 §5.1, P3 c3)."""

import asyncio
import copy
import pickle
import threading

import pytest
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    aopen_invocation,
    ConcurrentInvocationError,
    enter_run,
    exit_run,
    frame_for,
    InvocationCleanupError,
    InvocationContractError,
    open_invocation,
    RunContext,
    RunStateStore,
)
from agent_foundation.common.inferencers.run_context.invocation import (
    _current_invocation,
)


class _Owner:
    pass


class _Leaf:
    pass


class _FailingClose:
    def close(self):
        raise OSError("pipe")


def _root(store=None):
    return RunContext.root(
        workspace=InferencerWorkspace(root="/tmp/run"),
        store=store,
    )


# -- one path, one invocation -----------------------------------------------


def test_sibling_tasks_at_one_path_are_rejected():
    ctx = _root()
    ran = []

    async def call(owner, started, release):
        async with aopen_invocation(owner):
            ran.append(owner)
            started.set()
            await release.wait()

    async def main():
        token = enter_run(ctx)
        try:
            started, release = asyncio.Event(), asyncio.Event()
            first = asyncio.create_task(call(_Owner(), started, release))
            await started.wait()
            with pytest.raises(ConcurrentInvocationError):
                await call(_Owner(), asyncio.Event(), release)
            release.set()
            await first
        finally:
            exit_run(token)

    asyncio.run(main())
    assert len(ran) == 1


def test_two_instances_at_one_path_are_rejected():
    ctx = _root()
    token = enter_run(ctx)
    try:
        with open_invocation(_Owner()):
            with pytest.raises(ConcurrentInvocationError):
                with open_invocation(_Leaf()):
                    pass
    finally:
        exit_run(token)


def test_nested_public_call_at_the_same_path_is_rejected():
    owner = _Owner()
    token = enter_run(_root())
    try:
        with open_invocation(owner):
            with pytest.raises(ConcurrentInvocationError):
                with open_invocation(owner):
                    pass
    finally:
        exit_run(token)


def test_sequential_calls_at_one_path_are_allowed():
    ctx = _root()
    owner = _Owner()
    token = enter_run(ctx)
    try:
        for _ in range(3):
            with open_invocation(owner) as frame:
                assert ctx.store.claims.holder("/") is frame
            assert ctx.store.claims.holder("/") is None
    finally:
        exit_run(token)


def test_child_path_does_not_conflict_with_parent():
    ctx = _root()
    token = enter_run(ctx)
    try:
        with open_invocation(_Owner()) as outer:
            child_token = enter_run(ctx.child("worker_0"))
            try:
                with open_invocation(_Leaf()) as inner:
                    assert ctx.store.claims.holder("/worker_0") is inner
                    assert ctx.store.claims.holder("/") is outer
            finally:
                exit_run(child_token)
    finally:
        exit_run(token)


def test_distinct_stores_at_one_path_do_not_conflict():
    first, second = _root(), _root()
    token = enter_run(first)
    try:
        with open_invocation(_Owner()):
            inner_token = enter_run(second)
            try:
                with open_invocation(_Owner()) as frame:
                    assert second.store.claims.holder("/") is frame
            finally:
                exit_run(inner_token)
    finally:
        exit_run(token)


def test_invocation_without_a_ctx_claims_nothing():
    with open_invocation(_Owner()) as outer:
        with open_invocation(_Owner()) as inner:
            assert outer.ctx is None
            assert inner.ctx is None


# -- a rejected call runs nothing -------------------------------------------


def test_rejected_call_runs_nothing_and_leaves_the_chain_unchanged():
    holder_owner, rejected_owner = _Owner(), _Owner()
    body = []
    token = enter_run(_root())
    try:
        with open_invocation(holder_owner) as holder:
            with pytest.raises(ConcurrentInvocationError):
                with open_invocation(rejected_owner):
                    body.append("ran")
            assert body == []
            assert _current_invocation.get() is holder
            assert frame_for(rejected_owner) is None
    finally:
        exit_run(token)


def test_rejected_async_call_runs_nothing():
    body = []

    async def main():
        token = enter_run(_root())
        try:
            async with aopen_invocation(_Owner()) as holder:
                with pytest.raises(ConcurrentInvocationError):
                    async with aopen_invocation(_Owner()):
                        body.append("ran")
                assert _current_invocation.get() is holder
        finally:
            exit_run(token)

    asyncio.run(main())
    assert body == []


# -- diagnostics ------------------------------------------------------------


def test_rejection_names_holder_entry_age_and_caller():
    token = enter_run(_root())
    try:
        with open_invocation(_Owner(), entry="infer"):
            with pytest.raises(ConcurrentInvocationError) as info:
                with open_invocation(_Leaf(), entry="infer"):
                    pass
    finally:
        exit_run(token)
    message = str(info.value)
    assert "Path '/' is already in an open invocation: _Owner.infer, opened " in (
        message
    )
    assert "s ago; refusing _Leaf.infer at the same path." in message
    assert "ctx.child(<slot>)" in message
    assert "aclosing" not in message


def test_rejection_by_an_open_stream_adds_the_closing_hint():
    token = enter_run(_root())
    try:
        with open_invocation(_Owner(), entry="infer_streaming"):
            with pytest.raises(ConcurrentInvocationError) as info:
                with open_invocation(_Leaf()):
                    pass
    finally:
        exit_run(token)
    assert "The holder is a stream that is still open" in str(info.value)
    assert "contextlib.aclosing / contextlib.closing" in str(info.value)


# -- release ----------------------------------------------------------------


def test_claim_is_released_after_the_body_raises():
    ctx = _root()
    token = enter_run(ctx)
    try:
        with pytest.raises(ValueError):
            with open_invocation(_Owner()):
                raise ValueError("x")
        assert ctx.store.claims.holder("/") is None
    finally:
        exit_run(token)


def test_claim_is_released_after_a_cleanup_failure():
    ctx = _root()
    token = enter_run(ctx)
    try:
        with pytest.raises(InvocationCleanupError):
            with open_invocation(_Owner()) as frame:
                frame.ledger.register(_FailingClose(), "proc")
        assert ctx.store.claims.holder("/") is None
    finally:
        exit_run(token)


def test_claim_is_released_after_cancellation():
    ctx = _root()

    async def main():
        token = enter_run(ctx)
        try:
            started = asyncio.Event()

            async def call():
                async with aopen_invocation(_Owner()):
                    started.set()
                    await asyncio.Event().wait()

            task = asyncio.create_task(call())
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            exit_run(token)

    asyncio.run(main())
    assert ctx.store.claims.holder("/") is None


def test_release_by_a_non_holder_is_a_contract_error():
    with open_invocation(_Leaf()) as stranger:
        pass
    ctx = _root()
    token = enter_run(ctx)
    try:
        with open_invocation(_Owner()) as holder:
            with pytest.raises(InvocationContractError) as info:
                ctx.store.claims.release("/", stranger)
            assert ctx.store.claims.holder("/") is holder
    finally:
        exit_run(token)
    assert str(info.value) == (
        "_Leaf.infer cannot release path '/': the claim is held by _Owner.infer."
    )


def test_release_of_an_unclaimed_path_is_a_contract_error():
    with open_invocation(_Owner()) as frame:
        pass
    with pytest.raises(InvocationContractError, match="held by no invocation"):
        RunStateStore().claims.release("/", frame)


# -- threads ----------------------------------------------------------------


def test_exactly_one_thread_acquires_a_contended_path():
    store = RunStateStore()
    with open_invocation(_Owner()) as template:
        pass
    frames = [copy.copy(template) for _ in range(8)]
    barrier = threading.Barrier(len(frames))
    won, lost = [], []

    def contend(frame):
        barrier.wait()
        try:
            store.claims.acquire("/x", frame)
            won.append(frame)
        except ConcurrentInvocationError:
            lost.append(frame)

    threads = [threading.Thread(target=contend, args=(f,)) for f in frames]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(won) == 1
    assert len(lost) == len(frames) - 1
    assert store.claims.holder("/x") is won[0]


# -- eviction ---------------------------------------------------------------


def test_evict_subtree_refuses_while_a_claim_is_live_below():
    ctx = _root()
    ctx.child("review").child("worker_0").node()
    token = enter_run(ctx.child("review").child("worker_0"))
    try:
        with open_invocation(_Leaf()):
            with pytest.raises(InvocationContractError) as info:
                ctx.store.evict_subtree("/review")
    finally:
        exit_run(token)
    assert str(info.value) == (
        "Cannot evict the subtree at '/review': invocations are still open "
        "below it at ['/review/worker_0']. Children must finish before their "
        "subtree is evicted."
    )
    assert ctx.store.has("/review/worker_0")


def test_evict_subtree_allows_the_callers_own_claim():
    ctx = _root()
    review = ctx.child("review")
    review.node()
    review.child("worker_0").node()
    token = enter_run(review)
    try:
        with open_invocation(_Owner()):
            assert ctx.store.evict_subtree("/review") == 2
    finally:
        exit_run(token)


def test_evict_subtree_ignores_claims_outside_the_prefix():
    ctx = _root()
    ctx.child("review").node()
    token = enter_run(ctx.child("reviewer"))
    try:
        with open_invocation(_Leaf()):
            assert ctx.store.evict_subtree("/review") == 1
    finally:
        exit_run(token)


def test_live_below_is_strictly_below_and_sorted():
    store = RunStateStore()
    with open_invocation(_Owner()) as frame:
        pass
    for path in ("/a/z", "/a", "/ab", "/a/b"):
        store.claims.acquire(path, frame)
    assert store.claims.live_below("/a") == ["/a/b", "/a/z"]
    assert store.claims.live_below("/a/") == ["/a/b", "/a/z"]
    assert store.claims.live_below("/") == ["/a", "/a/b", "/a/z", "/ab"]


# -- claims are live state, never persisted ---------------------------------


def test_to_json_never_carries_claims(tmp_path):
    ctx = _root()
    ctx.node()
    token = enter_run(ctx)
    try:
        with open_invocation(_Owner()):
            data = ctx.store.to_json()
            ctx.store.save(str(tmp_path / "store.json"))
    finally:
        exit_run(token)
    assert set(data) == {"nodes"}
    for loaded in (
        RunStateStore.from_json(data),
        RunStateStore.load(str(tmp_path / "store.json")),
    ):
        assert loaded.has("/")
        assert loaded.claims.holder("/") is None
        with open_invocation(_Owner()) as frame:
            loaded.claims.acquire("/", frame)
            loaded.claims.release("/", frame)


def test_copied_or_pickled_store_keeps_nodes_and_drops_claims():
    ctx = _root()
    ctx.node()
    token = enter_run(ctx)
    try:
        with open_invocation(_Owner()):
            copies = [copy.deepcopy(ctx.store), pickle.loads(pickle.dumps(ctx.store))]
    finally:
        exit_run(token)
    for restored in copies:
        assert restored.has("/")
        assert restored.claims.holder("/") is None
        assert restored.claims is not ctx.store.claims


# -- ctx.store --------------------------------------------------------------


def test_ctx_store_is_the_shared_tier1_store():
    store = RunStateStore()
    ctx = _root(store)
    assert ctx.store is store
    assert ctx.child("a").child("b").store is store

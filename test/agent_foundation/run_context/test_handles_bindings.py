"""Tier-3 LiveHandleStore lifetime + Tier-2 RuntimeBindings (M1 / §2.0 Note B / P-#6)."""

from agent_foundation.common.inferencers.run_context import (
    LiveHandles,
    LiveHandleStore,
    RunContext,
    RunStateStore,
    RuntimeBindings,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


def test_handle_store_get_or_create_reuse():
    store = LiveHandleStore()
    h1 = store.get_or_create("/a")
    h2 = store.get_or_create("/a")
    assert h1 is h2
    assert store.get_or_create("/b") is not h1


def test_handles_missing_attr_reads_none_then_settable():
    h = LiveHandles(path="/a")
    assert h.sdk_client is None  # probe before connect, no AttributeError
    h.sdk_client = object()
    assert h.sdk_client is not None


def test_handle_store_teardown():
    store = LiveHandleStore()
    store.get_or_create("/a").client = 1
    store.get_or_create("/b").client = 2
    store.teardown("/a")
    assert store.peek("/a") is None
    assert store.peek("/b") is not None
    store.teardown()  # all
    assert len(store) == 0


def test_tier3_outlives_per_turn_tier1_store_discard():
    """§2.0 Note B / T-MAJOR2: discarding the per-turn store must NOT drop handles.

    Model two turns sharing one connection-scoped handle store; the Tier-1 store
    is replaced each turn, but the leaf's session_id persists (no relaunch).
    """
    ws = InferencerWorkspace(root="/tmp/run")
    handle_store = LiveHandleStore()  # connection-scoped — outlives turns
    runtime = RuntimeBindings()

    # Turn 1: fresh Tier-1 store, shared connection-scoped handle store.
    turn1 = RunContext.root(workspace=ws, runtime=runtime,
                            store=RunStateStore(), handle_store=handle_store)
    leaf1 = turn1.child("leaf")
    leaf1.handles.live_session_id = "sess-123"

    # Turn 2: Tier-1 store DISCARDED + replaced; same connection-scoped handle store.
    turn2 = RunContext.root(workspace=ws, runtime=runtime,
                            store=RunStateStore(), handle_store=handle_store)
    leaf2 = turn2.child("leaf")
    # Continuity preserved — no relaunch, session_id survives the per-turn discard.
    assert leaf2.handles.live_session_id == "sess-123"
    assert leaf2.handles is leaf1.handles


def test_runtime_bindings_shared_by_reference_for_cancellation():
    """P-#6: a cancellation token set on the shared binding is visible to children."""
    root = RunContext.root(workspace=None)
    token = {"cancelled": False}
    root.runtime.cancellation_token = token
    child = root.child("worker_0")
    # child sees the SAME token object (shared by reference)
    assert child.runtime.cancellation_token is token
    token["cancelled"] = True
    assert child.runtime.cancellation_token["cancelled"] is True


def test_runtime_bindings_marked_never_persist():
    assert getattr(RuntimeBindings, "__never_persist__", False) is True
    assert getattr(LiveHandles, "__never_persist__", False) is True
    assert getattr(LiveHandleStore, "__never_persist__", False) is True

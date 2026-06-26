"""§5 DoD: conversational pause-state -> Tier-1 conversation blob round-trips on
resume (D7/§2.8); Tier-3 multi-turn continuity (V7)."""

import json

from agent_foundation.common.inferencers.run_context import (
    RunContext,
    RunStateStore,
)


def test_conversational_pause_state_round_trips_through_tier1():
    """The pause-state blob (sop_state / messages / dynamic_context) persists in
    NodeRunState.conversation and rehydrates byte-for-byte (form unchanged, D7)."""
    store = RunStateStore()
    # The existing _serialize_pause_state shape (dict form, unchanged).
    pause_blob = {
        "sop_state": {"current_phase": "p2", "phase_status": "IN_PROGRESS"},
        "messages": [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "working"},
        ],
        "dynamic_context": {"chars": 1234},
        "suspended_sops": [],
    }
    store.node("/conversation").conversation = pause_blob

    # Persist + rehydrate (resume).
    restored = RunStateStore.from_json(json.loads(json.dumps(store.to_json())))
    back = restored.node("/conversation").conversation
    assert back == pause_blob
    assert back["sop_state"]["current_phase"] == "p2"
    assert len(back["messages"]) == 2


def test_tier3_multi_turn_continuity_no_relaunch():
    """V7: a connection-scoped handle survives across turns (no per-turn relaunch).

    Two turns share one connection-scoped handle store (Tier-3); the per-turn
    Tier-1 store is replaced each turn, but the live session/client persists.
    """
    from agent_foundation.common.inferencers.run_context import (
        LiveHandleStore,
        RuntimeBindings,
    )

    ws = None
    handle_store = LiveHandleStore()  # connection-scoped — outlives turns
    runtime = RuntimeBindings()

    turn1 = RunContext.root(
        workspace=ws, runtime=runtime, store=RunStateStore(), handle_store=handle_store
    )
    leaf1 = turn1.child("agent")
    leaf1.handles.sdk_client = "CLIENT-1"
    leaf1.handles.live_session_id = "sess-xyz"

    # New turn: Tier-1 store discarded + replaced; same connection handle store.
    turn2 = RunContext.root(
        workspace=ws, runtime=runtime, store=RunStateStore(), handle_store=handle_store
    )
    leaf2 = turn2.child("agent")
    # Continuity preserved across the turn boundary — no relaunch.
    assert leaf2.handles.sdk_client == "CLIENT-1"
    assert leaf2.handles.live_session_id == "sess-xyz"


def test_tier3_concurrency_isolation_across_gather_branches():
    """Tier-3 handles are isolated per concurrent branch (V8) on one store."""
    from agent_foundation.common.inferencers.run_context import LiveHandleStore

    store = LiveHandleStore()
    root = RunContext.root(workspace=None, handle_store=store)
    n = 8
    for i in range(n):
        root.child(f"worker_{i}").handles.session_id = f"S{i}"
    # Each branch kept its own handle — no cross-talk.
    for i in range(n):
        assert root.child(f"worker_{i}").handles.session_id == f"S{i}"
    assert len(store) == n


def test_rehydrate_from_resumed_store_wires_restore_into_the_run():
    """M9/D7 WIRING: the conversational run reads ctx.node().conversation (loaded
    from a resumed RunStateStore) and restores it — closing the serialize->restore
    loop that was previously open (_restore_pause_state had no caller)."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )
    from agent_foundation.common.inferencers.run_context import (
        RunContext,
        enter_run,
        exit_run,
    )

    ci = ConversationalInferencer.__new__(ConversationalInferencer)
    ci.__dict__["_pending_resume_state"] = None
    blob = {
        "messages": [{"role": "user", "content": "resumed"}],
        "prior_context": {},
        "turn_number": 2,
        "iteration": 3,
    }
    root = RunContext.root(workspace=None)
    root.node().conversation = blob
    tok = enter_run(root)
    try:
        ci._rehydrate_from_resumed_store()
    finally:
        exit_run(tok)
    assert ci._messages == [{"role": "user", "content": "resumed"}]
    assert ci._pending_resume_state == {"turn_number": 2, "iteration": 3}


def test_rehydrate_is_noop_without_a_conversation_blob():
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )
    from agent_foundation.common.inferencers.run_context import (
        RunContext,
        enter_run,
        exit_run,
    )

    ci = ConversationalInferencer.__new__(ConversationalInferencer)
    ci.__dict__["_pending_resume_state"] = None
    tok = enter_run(RunContext.root(workspace=None))
    try:
        ci._rehydrate_from_resumed_store()  # empty node -> no restore, no raise
    finally:
        exit_run(tok)
    assert ci._pending_resume_state is None

"""Example: explicit RunContext — host-minted root, three-tier state, provenance.

Demonstrates the separation the ``run_context`` package delivers:

* the **host** mints a root ``RunContext`` (workspace-rooted);
* an orchestrator threads ``ctx.child(slot)`` to children (deterministic §2.7 paths);
* per-call state lives in ``ctx.node.call`` (Tier-1), NOT on the instance;
* Tier-3 live handles are per-branch (connection-scoped);
* the Tier-1 store persists/rehydrates for resume (M9).

Run:
    PYTHONPATH=src:../RichPythonUtils/src python \
        examples/agent_foundation/common/inferencers/example_runcontext_explicit.py
"""

from __future__ import annotations

import json

from agent_foundation.common.inferencers.run_context import (
    MultiFlowState,
    RunContext,
    RunStateStore,
    active_run_context,
    enter_run,
    exit_run,
)


def main() -> None:
    # 1. The host mints the root (the task executor / SOP CLI / ConversationService).
    root = RunContext.root(workspace=None)
    print("root path:", root.path)

    # 2. An orchestrator derives deterministic child contexts (§2.7 slots).
    review = root.child("review")
    worker0 = review.child("worker_0")
    worker1 = review.child("worker_1")
    print("child paths:", review.path, worker0.path, worker1.path)

    # 3. Per-call state lives in the context node (Tier-1), not on any instance.
    worker0.node().call = MultiFlowState(winner_idx=0, ranking=[0, 1])
    worker1.node().call = MultiFlowState(winner_idx=1, ranking=[1, 0])

    # 4. Tier-3 handles are per-branch (connection-scoped) — distinct per worker.
    worker0.handles.live_session_id = "sess-A"
    worker1.handles.live_session_id = "sess-B"
    assert worker0.handles.live_session_id != worker1.handles.live_session_id
    print("isolated session ids:", worker0.handles.live_session_id,
          worker1.handles.live_session_id)

    # 5. The bridge makes the active context readable without threading it as a param.
    token = enter_run(worker0)
    try:
        assert active_run_context().path == "/review/worker_0"
        print("active ctx inside bridge:", active_run_context().path)
    finally:
        exit_run(token)

    # 6. Tier-1 persists + rehydrates for resume (M9) — typed state survives.
    blob = json.dumps(root._store.to_json())
    restored = RunStateStore.from_json(json.loads(blob))
    back = restored.node("/review/worker_0").call
    assert isinstance(back, MultiFlowState) and back.winner_idx == 0
    print("resume round-trip OK; restored winner_idx:", back.winner_idx)

    print("\nexample_runcontext_explicit: all assertions passed.")


if __name__ == "__main__":
    main()

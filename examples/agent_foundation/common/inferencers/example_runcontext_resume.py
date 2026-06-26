"""Example (M9): persist a RunStateStore to disk and rehydrate it (resume).

Demonstrates the resume path the task executor / SOP host wire: Tier-1 state
(typed + dict + conversation blob) is saved as JSON and reconstructed byte-for-
byte. Tier-2 sinks and Tier-3 live handles are NOT persisted (re-supplied on
resume). No credentials. CI-gated.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent_foundation.common.inferencers.run_context import (
    DualState,
    MFDualState,
    RunStateStore,
)


def main() -> None:
    store = RunStateStore()
    # typed (nested) call state
    n = store.node("/review/worker_0", creator=("BTA", "worker_0"))
    n.call = MFDualState()
    n.call.dual = DualState(chosen_role="fixer", review_workspace="/ws/review")
    # dict state + a conversation pause blob (D7 form)
    store.node("/leaf").call = {"attempt": 2}
    store.node("/conv").conversation = {"sop_state": {"phase": "p2"}, "messages": [1, 2]}

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "run_state" / "store.json")
        store.save(path)                       # persist (host does this post-run)
        restored = RunStateStore.load(path)    # rehydrate (host does this on resume)

    rebuilt = restored.node("/review/worker_0").call
    assert isinstance(rebuilt, MFDualState) and rebuilt.dual.chosen_role == "fixer"
    assert restored.node("/leaf").call == {"attempt": 2}
    assert restored.node("/conv").conversation["sop_state"]["phase"] == "p2"
    print("resume example OK: typed + dict + conversation rehydrated")


if __name__ == "__main__":
    main()

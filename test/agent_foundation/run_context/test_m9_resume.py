"""M9: resume — RunStateStore persists/rehydrates from disk (Tier-1 only)."""

import os

import pytest

from agent_foundation.common.inferencers.run_context import (
    CollisionError,
    DualState,
    MFDualState,
    NodeRunState,
    RunStateStore,
)


def _populate():
    s = RunStateStore()
    # typed call state (nested, composed)
    n = s.node("/review/worker_0", creator=("BTA", "worker_0"))
    n.call = MFDualState()
    n.call.dual = DualState(chosen_role="fixer", review_workspace="/ws/review")
    # dict call state
    s.node("/leaf").call = {"effective_input": "hello", "attempt": 2}
    # conversation blob (the V9 pause-state form, unchanged)
    s.node("/conv").conversation = {
        "messages": [{"role": "user", "content": "hi"}],
        "sop_state": {"phase": "p1"},
    }
    # HITL checkpoint
    s.node("/conv").checkpoints["planning"] = {"approved": True, "user_input": "go"}
    return s


def test_resume_round_trip_through_disk(tmp_path):
    s = _populate()
    path = os.path.join(str(tmp_path), "run_state", "store.json")
    s.save(path)
    assert os.path.exists(path)

    s2 = RunStateStore.load(path)
    # typed nested state reconstructed
    rebuilt = s2.node("/review/worker_0").call
    assert isinstance(rebuilt, MFDualState)
    assert isinstance(rebuilt.dual, DualState)
    assert rebuilt.dual.chosen_role == "fixer"
    assert rebuilt.dual.review_workspace == "/ws/review"
    # dict state
    assert s2.node("/leaf").call == {"effective_input": "hello", "attempt": 2}
    # conversation blob (rehydrate source for D7/§2.8)
    assert s2.node("/conv").conversation["sop_state"] == {"phase": "p1"}
    # checkpoint
    assert s2.node("/conv").checkpoints["planning"]["approved"] is True


def test_creator_not_persisted_resume_safe(tmp_path):
    s = _populate()
    path = os.path.join(str(tmp_path), "store.json")
    s.save(path)
    s2 = RunStateStore.load(path)
    # On resume nodes carry NO creator -> first re-entry re-tags, distinct then raises.
    s2.node("/review/worker_0", creator=("BTA", "worker_0"))  # re-tag, no raise
    with pytest.raises(CollisionError):
        s2.node("/review/worker_0", creator=("OTHER", "x"))


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunStateStore.load(os.path.join(str(tmp_path), "nope.json"))

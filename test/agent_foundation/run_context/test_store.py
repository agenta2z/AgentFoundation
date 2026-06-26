"""Tier-1 store: get-or-create, collision guard (resume-safe), reset, god-object guard."""

import json

import pytest

from agent_foundation.common.inferencers.run_context import (
    CollisionError,
    MultiFlowState,
    NodeRunState,
    RunStateStore,
    RuntimeBindings,
)


def test_node_get_or_create_is_idempotent():
    s = RunStateStore()
    n1 = s.node("/a")
    n2 = s.node("/a")
    assert n1 is n2
    assert len(s) == 1


def test_same_creator_reentry_is_noop():
    s = RunStateStore()
    a = s.node("/x", creator=("Dual", "review"))
    b = s.node("/x", creator=("Dual", "review"))  # retry
    assert a is b


def test_distinct_creator_collision_raises():
    s = RunStateStore()
    s.node("/x", creator=("Dual", "review"))
    with pytest.raises(CollisionError):
        s.node("/x", creator=("MFI", "worker_0"))


def test_creator_key_is_stable_signature_not_id():
    """The guard must use a value signature, so equal (class, slot) tuples are 'same creator'."""
    s = RunStateStore()
    s.node("/x", creator=("Dual", "review"))
    # A *different tuple object* with equal value must NOT collide.
    s.node("/x", creator=tuple(["Dual", "review"]))


def test_reset_attempt_clears_only_attempt():
    s = RunStateStore()
    n = s.node("/a")
    n.call = MultiFlowState(winner_idx=1)
    n.attempt = {"try": 2}
    n.reset_attempt()
    assert n.attempt is None
    assert isinstance(n.call, MultiFlowState) and n.call.winner_idx == 1


def test_store_round_trip_preserves_typed_and_dict():
    s = RunStateStore()
    s.node("/a", creator=("X", "a")).call = MultiFlowState(winner_idx=7)
    s.node("/a").attempt = {"k": "v"}
    s.node("/b").conversation = {"messages": [1, 2, 3]}
    text = json.dumps(s.to_json())
    s2 = RunStateStore.from_json(json.loads(text))
    assert s2.node("/a").call.winner_idx == 7
    assert s2.node("/a").attempt == {"k": "v"}
    assert s2.node("/b").conversation == {"messages": [1, 2, 3]}


def test_creator_is_not_persisted_and_resume_retags():
    s = RunStateStore()
    s.node("/a", creator=("Orig", "slot"))
    dumped = s.to_json()
    assert "_creator" not in dumped["nodes"]["/a"]
    assert "creator" not in dumped["nodes"]["/a"]
    s2 = RunStateStore.from_json(dumped)
    # Untagged on resume -> first claim re-tags (no raise) ...
    s2.node("/a", creator=("First", "slot"))
    # ... a subsequent *distinct* creator then collides.
    with pytest.raises(CollisionError):
        s2.node("/a", creator=("Second", "slot"))


def test_god_object_guard_no_tier2_or_tier3_in_serialized_output():
    """to_json must never contain RuntimeBindings / LiveHandles markers."""
    s = RunStateStore()
    n = s.node("/a")
    # Even if someone wrongly stuffs a binding into a dict bucket, the marker is detectable.
    n.conversation = {"safe": True}
    text = json.dumps(s.to_json())
    assert "__never_persist__" not in text
    assert "RuntimeBindings" not in text
    assert "LiveHandles" not in text
    # And RuntimeBindings itself is flagged non-persistable.
    assert getattr(RuntimeBindings, "__never_persist__", False) is True


def test_node_equality_ignores_transient_creator():
    a = NodeRunState(path="/a", creator=("A", "x"))
    b = NodeRunState(path="/a", creator=("B", "y"))
    assert a == b  # _creator has eq=False

"""Typed node outcome channel (plan v8 §5.3): codec, store clear/publish, helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading

import attrs
import pytest
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    BTAState,
    CollisionError,
    decode_state,
    encode_state,
    NodeOutcomeState,
    NodeRunState,
    publish_outcome,
    read_outcome,
    RenderedTaskContractState,
    RoleState,
    RunContext,
    RunStateStore,
)


def _contract(text="do the task", role=None, source_path="/worker_0"):
    return RenderedTaskContractState(
        text=text, sha256="ab" * 32, role=role, source_path=source_path
    )


def _outcome(**overrides):
    fields = {
        "task_contract": _contract(),
        "summary": BTAState(effective_sub_queries=["a", "b"]),
        "final_output": "final",
        "invocation_id": "inv-1",
        "cleanup_errors": ["stage: RuntimeError: boom"],
    }
    fields.update(overrides)
    return NodeOutcomeState(**fields)


def _root(store=None):
    return RunContext.root(workspace=InferencerWorkspace(root="/tmp/run"), store=store)


# -- codec ---------------------------------------------------------------------


def test_outcome_round_trips_through_the_codec_with_nested_typed_fields():
    outcome = _outcome(summary=RoleState(new_role="reviewer", template_version="v2"))

    decoded = decode_state(json.loads(json.dumps(encode_state(outcome))))

    assert decoded == outcome
    assert isinstance(decoded.task_contract, RenderedTaskContractState)
    assert isinstance(decoded.summary, RoleState)
    assert decoded.cleanup_errors == ("stage: RuntimeError: boom",)
    assert isinstance(decoded.cleanup_errors, tuple)


def test_outcome_defaults_are_empty():
    outcome = NodeOutcomeState()

    assert outcome.task_contract is None
    assert outcome.summary is None
    assert outcome.final_output is None
    assert outcome.invocation_id == ""
    assert outcome.cleanup_errors == ()
    assert decode_state(encode_state(outcome)) == outcome


def test_outcome_and_contract_are_frozen():
    outcome = _outcome()

    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        outcome.final_output = "other"
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        outcome.task_contract.text = "other"


def test_evolve_keeps_cleanup_errors_a_tuple():
    outcome = attrs.evolve(NodeOutcomeState(), cleanup_errors=["a", "b"])

    assert outcome.cleanup_errors == ("a", "b")


# -- NodeRunState / RunStateStore persistence ------------------------------------


def test_node_outcome_round_trips_through_to_json():
    node = NodeRunState(path="/w", outcome=_outcome())

    restored = NodeRunState.from_json(json.loads(json.dumps(node.to_json())))

    assert restored.outcome == node.outcome


def test_old_node_without_an_outcome_key_loads_none():
    data = NodeRunState(path="/w").to_json()
    data.pop("outcome")

    assert NodeRunState.from_json(data).outcome is None


def test_store_save_and_load_round_trip_the_outcome(tmp_path):
    store = RunStateStore()
    store.publish_outcome("/w", ("Leaf", "w"), _outcome())
    target = str(tmp_path / "store.json")

    store.save(target)
    loaded = RunStateStore.load(target)

    assert loaded.peek("/w").outcome == _outcome()


def test_claims_and_creator_are_not_persisted(tmp_path):
    store = RunStateStore()
    store.publish_outcome("/w", ("Leaf", "w"), _outcome())
    target = str(tmp_path / "store.json")

    store.save(target)
    with open(target, encoding="utf-8") as f:
        raw = json.load(f)
    loaded = RunStateStore.load(target)

    assert set(raw) == {"nodes"}
    assert "creator" not in json.dumps(raw)
    assert loaded.peek("/w")._creator is None
    assert loaded.claims.holder("/w") is None


def test_a_fresh_process_rehydrates_the_typed_outcome_without_a_degrade_warning(
    tmp_path,
):
    store = RunStateStore()
    store.publish_outcome("/w", ("Leaf", "w"), _outcome())
    target = str(tmp_path / "store.json")
    store.save(target)
    script = textwrap.dedent(
        f"""
        from agent_foundation.common.inferencers.run_context.store import RunStateStore

        outcome = RunStateStore.load({target!r}).peek("/w").outcome
        assert type(outcome).__name__ == "NodeOutcomeState", type(outcome)
        assert type(outcome.task_contract).__name__ == "RenderedTaskContractState"
        assert type(outcome.summary).__name__ == "BTAState"
        assert outcome.cleanup_errors == ("stage: RuntimeError: boom",)
        """
    )

    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


# -- store clear / publish ---------------------------------------------------------


def test_clear_outcome_drops_the_outcome_and_keeps_the_node():
    store = RunStateStore()
    store.publish_outcome("/w", ("Leaf", "w"), _outcome())
    store.node("/w").call = BTAState()

    store.clear_outcome("/w")

    assert store.peek("/w").outcome is None
    assert store.peek("/w").call == BTAState()


def test_reads_and_clears_on_a_missing_path_create_no_node():
    store = RunStateStore()
    ctx = _root(store).child("missing")

    store.clear_outcome("/missing")
    assert store.peek("/missing") is None
    assert read_outcome(ctx) is None
    assert len(store) == 0


def test_publish_outcome_under_a_different_creator_collides():
    store = RunStateStore()
    store.node("/w", ("Leaf", "w"))

    with pytest.raises(CollisionError):
        store.publish_outcome("/w", ("OtherLeaf", "w"), _outcome())
    assert store.peek("/w").outcome is None


def test_publish_outcome_creates_and_tags_the_node():
    store = RunStateStore()

    store.publish_outcome("/w", ("Leaf", "w"), _outcome())

    assert store.peek("/w")._creator == ("Leaf", "w")
    assert store.peek("/w").outcome == _outcome()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda store: store.clear_outcome("/w"),
        lambda store: store.publish_outcome("/w", ("Leaf", "w"), NodeOutcomeState()),
    ],
    ids=["clear", "publish"],
)
def test_clear_and_publish_wait_for_the_store_lock(mutate):
    store = RunStateStore()
    store.publish_outcome("/w", ("Leaf", "w"), _outcome())
    held, release, done = threading.Event(), threading.Event(), threading.Event()

    def hold_lock():
        with store._lock:
            held.set()
            release.wait(10)

    def run_mutation():
        mutate(store)
        done.set()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert held.wait(10)
    mutator = threading.Thread(target=run_mutation)
    mutator.start()
    try:
        assert not done.wait(0.2)
        assert store.peek("/w").outcome == _outcome()
    finally:
        release.set()
        holder.join(10)
        mutator.join(10)
    assert done.is_set()


# -- ctx helpers --------------------------------------------------------------------


def test_publish_and_read_outcome_through_a_ctx_and_its_child():
    root = _root()
    child = root.child("worker_0")

    publish_outcome(child, ("Leaf", "worker_0"), _outcome(invocation_id="child"))
    publish_outcome(root, ("Parent", ""), _outcome(invocation_id="root"))

    assert read_outcome(child).invocation_id == "child"
    assert read_outcome(root).invocation_id == "root"
    assert root.store.peek(child.path).outcome.invocation_id == "child"


def test_read_outcome_is_none_before_any_publish():
    root = _root()
    root.store.node(root.path, ("Parent", ""))

    assert read_outcome(root) is None

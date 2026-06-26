"""§2.11: HITL decision persists into ctx.node.checkpoints (Tier-1); no-op without ctx."""

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    _record_hitl_checkpoint,
)
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    RunStateStore,
    enter_run,
    exit_run,
)


def test_records_decision_into_checkpoints():
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        _record_hitl_checkpoint({"choice": "approve", "user_input": "go"})
        _record_hitl_checkpoint(None)  # a reject/cancel
    finally:
        exit_run(tok)
    cps = root.node().checkpoints
    assert len(cps) == 2
    assert cps["hitl_0"]["approved"] is True
    assert cps["hitl_0"]["user_input"] == {"choice": "approve", "user_input": "go"}
    assert cps["hitl_1"]["approved"] is False


def test_noop_without_context():
    # No active ctx -> must not raise, must not record anywhere.
    _record_hitl_checkpoint("approve")  # simply returns


def test_hitl_checkpoint_survives_resume_round_trip():
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        _record_hitl_checkpoint({"approved_plan": True})
    finally:
        exit_run(tok)
    # persist + rehydrate the store (M9) -> the approval is restored
    import json
    restored = RunStateStore.from_json(json.loads(json.dumps(root._store.to_json())))
    assert restored.node("/").checkpoints["hitl_0"]["user_input"] == {"approved_plan": True}

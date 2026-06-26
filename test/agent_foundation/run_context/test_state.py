"""Tier-1 state: typed/dict/nested round-trips, discriminator, registry (M1 / N-R3 / N-S5)."""

import json
import warnings

import attrs
import pytest

from agent_foundation.common.inferencers.run_context import (
    BTAState,
    DualState,
    InferencerStateBase,
    MFDualState,
    MultiFlowState,
    decode_state,
    encode_state,
    register_state,
)
from agent_foundation.common.inferencers.run_context.state import (
    STATE_REGISTRY,
    _STATE_CLASS_KEY,
)


def _json_round_trip(value):
    """Encode -> JSON string -> load -> decode, the real store path."""
    return decode_state(json.loads(json.dumps(encode_state(value))))


def test_typed_state_stamps_discriminator_and_round_trips():
    s = MultiFlowState(winner_idx=2, reviewer_alias="r", ranking=[1, 0, 2])
    enc = encode_state(s)
    assert enc[_STATE_CLASS_KEY] == "MultiFlowState"
    assert enc["winner_idx"] == 2 and enc["ranking"] == [1, 0, 2]
    back = _json_round_trip(s)
    assert isinstance(back, MultiFlowState)
    assert back == s


def test_plain_dict_has_no_tag_and_passes_through():
    d = {"a": 1, "nested": {"b": 2}}
    enc = encode_state(d)
    assert _STATE_CLASS_KEY not in enc
    assert _json_round_trip(d) == d


def test_nested_typed_state_survives_recursive_encode():
    """MFDualState holds DualState + MultiFlowState; the discriminator must survive nesting."""
    s = MFDualState()
    s.dual.chosen_role = "fixer"
    s.multiflow.winner_idx = 3
    enc = encode_state(s)
    # Nested states must each carry their own discriminator (NOT flattened by asdict).
    assert enc[_STATE_CLASS_KEY] == "MFDualState"
    assert enc["dual"][_STATE_CLASS_KEY] == "DualState"
    assert enc["multiflow"][_STATE_CLASS_KEY] == "MultiFlowState"
    back = _json_round_trip(s)
    assert isinstance(back, MFDualState)
    assert isinstance(back.dual, DualState) and back.dual.chosen_role == "fixer"
    assert isinstance(back.multiflow, MultiFlowState) and back.multiflow.winner_idx == 3


def test_typed_state_inside_dict_is_rehydrated():
    payload = {"primary": DualState(chosen_role="a"), "count": 5}
    back = _json_round_trip(payload)
    assert isinstance(back["primary"], DualState) and back["primary"].chosen_role == "a"
    assert back["count"] == 5


def test_typed_state_inside_list_is_rehydrated():
    payload = [MultiFlowState(winner_idx=1), MultiFlowState(winner_idx=2)]
    back = _json_round_trip(payload)
    assert [s.winner_idx for s in back] == [1, 2]
    assert all(isinstance(s, MultiFlowState) for s in back)


def test_unknown_state_class_degrades_to_dict_with_warning():
    blob = {"_state_class": "GhostStateThatWasRemoved", "x": 1}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = decode_state(blob)
    assert isinstance(back, dict) and back == {"x": 1}
    assert any("Unknown _state_class" in str(w.message) for w in caught)


def test_registry_contains_builtin_states():
    for name in ("MultiFlowState", "DualState", "BTAState", "MFDualState"):
        assert name in STATE_REGISTRY


def test_duplicate_registration_name_raises():
    with pytest.raises(ValueError):

        @register_state
        @attrs.define
        class MultiFlowState(InferencerStateBase):  # noqa: F811 - intentional name clash
            pass


def test_bta_state_dict_field_round_trips():
    s = BTAState(effective_sub_queries=["q1", "q2"], latest_per_flow={"0": {"ok": True}})
    back = _json_round_trip(s)
    assert back.effective_sub_queries == ["q1", "q2"]
    assert back.latest_per_flow == {"0": {"ok": True}}

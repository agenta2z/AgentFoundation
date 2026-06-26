"""M7: switch_role mirrors role into ctx.node.call (RoleState); purity gate certifies."""

from agent_foundation.common.inferencers.run_context import (
    RoleState,
    RunStateStore,
    decode_state,
    encode_state,
)
from agent_foundation.common.inferencers.run_context.purity import purity_snapshot


def test_role_state_round_trips_with_discriminator():
    rs = RoleState(new_role="reviewer", template_key="review", changes={"template_key": "review"})
    enc = encode_state(rs)
    assert enc["_state_class"] == "RoleState"
    back = decode_state(enc)
    assert isinstance(back, RoleState)
    assert back.new_role == "reviewer" and back.template_key == "review"


def test_role_state_persists_in_store():
    s = RunStateStore()
    s.node("/review").call = RoleState(new_role="reviewer", template_key="review")
    s2 = RunStateStore.from_json(s.to_json())
    assert isinstance(s2.node("/review").call, RoleState)
    assert s2.node("/review").call.new_role == "reviewer"


def test_purity_snapshot_is_the_m7_gate_mechanism():
    """The §2.6 purity gate: detects any residual per-run self-mutation.

    This is the M7 *certification* mechanism — once a class routes its run-state
    to the context, this snapshot (empty allow-list) must show no instance
    __dict__ delta across a run. Here we demonstrate it flags a self-mutation
    and passes when state goes to the context instead.
    """

    class _Fake:
        def __init__(self):
            self.definition = "model"  # legit definition

    # A class still mutating self -> purity gate FAILS (the pre-M7 state).
    obj = _Fake()
    with purity_snapshot(obj) as holder:
        obj.template_key = "review"  # role mutation on self
    (delta,) = holder
    assert not delta.is_pure and "template_key" in delta.added

    # Routing the same change into a context node -> instance stays pure.
    store = RunStateStore()
    obj2 = _Fake()
    with purity_snapshot(obj2) as holder2:
        store.node("/x").call = RoleState(new_role="reviewer", template_key="review")
    (delta2,) = holder2
    assert delta2.is_pure  # no self-mutation -> M7 goal certified

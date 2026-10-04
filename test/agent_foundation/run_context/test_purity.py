"""M0/§2.6 purity snapshot — the M7 no-self-mutation gate, self-tested."""

import logging
import threading

import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context.purity import (
    assert_pure,
    diff_vars,
    purity_snapshot,
    snapshot_vars,
)
from attr import attrs


class _Fake:
    def __init__(self):
        self.definition = "model-x"  # legitimate definition field (set in __init__)


def test_pure_when_nothing_mutates():
    obj = _Fake()
    with purity_snapshot(obj) as holder:
        _ = obj.definition  # read only
    (delta,) = holder
    assert delta.is_pure


def test_detects_added_per_run_field():
    obj = _Fake()
    with purity_snapshot(obj) as holder:
        obj._turn_counter = 3  # orphan per-run self-mutation (the M7 hazard)
    (delta,) = holder
    assert not delta.is_pure
    assert "_turn_counter" in delta.added


def test_detects_changed_field():
    obj = _Fake()
    with purity_snapshot(obj) as holder:
        obj.definition = "mutated"
    (delta,) = holder
    assert "definition" in delta.changed
    assert delta.changed["definition"] == ("model-x", "mutated")


def test_allow_list_permits_named_keys():
    obj = _Fake()
    with purity_snapshot(obj, allow=["_session_id"]) as holder:
        obj._session_id = "sess"  # allowed (e.g., during compat window)
    (delta,) = holder
    assert delta.is_pure


def test_snapshot_handles_uncopyable_values():
    obj = _Fake()
    obj.live = lambda: None  # not deepcopy-able
    snap = snapshot_vars(obj)
    assert "live" in snap  # falls back to identity rather than raising


def test_diff_vars_direct():
    before = {"a": 1}
    after = {"a": 2, "b": 9}
    d = diff_vars(before, after, allow=frozenset())
    assert d.changed == {"a": (1, 2)} and d.added == {"b": 9}


# --- I10: the gate can't false-pass -------------------------------------------------


class _Box:  # noqa: B903 -- must keep a __dict__ and no __eq__
    def __init__(self, value):
        self.value = value


class _Slotted:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


@attrs(slots=False)
class _Child(InferencerBase):
    def _infer(self, x, inference_config=None, **kw):
        return x


def _delta(obj, mutate):
    with purity_snapshot(obj) as holder:
        mutate(obj)
    return holder[0]


def test_detects_removed_field():
    obj = _Fake()
    delta = _delta(obj, lambda o: delattr(o, "definition"))
    assert not delta.is_pure
    assert delta.removed == {"definition": "model-x"}


@pytest.mark.parametrize(
    "initial, mutate",
    [
        ([1, [2]], lambda v: v[1].append(3)),
        ({"a": {"b": 1}}, lambda v: v["a"].update(b=2)),
        ({"a": [1]}, lambda v: v["a"].clear()),
        ((1, [2]), lambda v: v[1].append(3)),
        ({"s": {1}}, lambda v: v["s"].add(2)),
        ([_Box(1)], lambda v: setattr(v[0], "value", 2)),
        (_Box(_Box(1)), lambda v: setattr(v.value, "value", 2)),
        ([_Slotted(1)], lambda v: setattr(v[0], "value", 2)),
    ],
    ids=["list", "dict", "dict-list", "tuple-list", "set", "obj", "obj-obj", "slots"],
)
def test_detects_nested_in_place_mutation(initial, mutate):
    obj = _Fake()
    obj.state = initial
    delta = _delta(obj, lambda o: mutate(o.state))
    assert set(delta.changed) == {"state"}


def test_detects_replacement_by_an_equal_looking_object():
    obj = _Fake()
    obj.helper = _Box(1)
    delta = _delta(obj, lambda o: setattr(o, "helper", _Box(1)))
    assert set(delta.changed) == {"helper"}


def test_unchanged_object_without_eq_is_pure():
    obj = _Fake()
    obj.helper = _Box([1, {"k": _Box(2)}])
    assert _delta(obj, lambda o: None).is_pure


def test_equal_builtin_container_rebind_is_pure():
    obj = _Fake()
    obj.items = [1, 2]
    assert _delta(obj, lambda o: setattr(o, "items", [1, 2])).is_pure


def test_nan_rebind_is_pure_and_signed_zero_is_a_change():
    obj = _Fake()
    obj.score = float("nan")
    assert _delta(obj, lambda o: setattr(o, "score", float("nan"))).is_pure
    obj.score = 0.0
    assert set(_delta(obj, lambda o: setattr(o, "score", -0.0)).changed) == {"score"}


def test_cycles_and_self_references_terminate():
    obj = _Fake()
    ring = []
    ring.append(ring)
    obj.ring = ring
    obj.me = obj
    obj.box = _Box(None)
    obj.box.value = obj.box
    assert _delta(obj, lambda o: None).is_pure
    delta = _delta(obj, lambda o: o.ring.append(1))
    assert set(delta.changed) == {"ring"}


def test_loggers_locks_and_handlers_are_normalized():
    obj = _Fake()
    obj.logger = {"x": logging.getLogger("purity.test")}
    obj._lock = threading.Lock()

    def recreate(o):
        o._lock = threading.Lock()
        logging.getLogger("purity.unrelated.growth")
        o.logger["x"].addHandler(logging.NullHandler())

    assert _delta(obj, recreate).is_pure
    delta = _delta(obj, lambda o: o.logger.update(y=logging.getLogger("purity.y")))
    assert set(delta.changed) == {"logger"}


def test_child_inferencer_is_a_boundary():
    obj = _Fake()
    obj.child = _Child()
    obj.children = [obj.child]
    assert _delta(obj, lambda o: setattr(o.child, "_turn", 1)).is_pure
    delta = _delta(obj, lambda o: setattr(o, "child", _Child()))
    assert set(delta.changed) == {"child"}


def test_assert_pure_uses_the_baseline():
    obj = _Fake()
    before = snapshot_vars(obj)
    assert assert_pure(obj, before).is_pure
    obj.definition = "mutated"
    obj._turn = 1
    with pytest.raises(
        AssertionError, match=r"added=\['_turn'\].*changed=\['definition'\]"
    ):
        assert_pure(obj, before)
    assert assert_pure(obj, before, allow=["definition", "_turn"]).is_pure

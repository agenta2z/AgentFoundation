"""M0/§2.6 purity snapshot — the M7 no-self-mutation gate, self-tested."""

from agent_foundation.common.inferencers.run_context.purity import (
    diff_vars,
    purity_snapshot,
    snapshot_vars,
)


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

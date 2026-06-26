"""M0/§2.6 — the purity snapshot: the M7 per-class no-self-mutation gate.

M7 stops a class mutating per-run state on ``self``.  Verifying that **completely**
(vs. trusting a hand-enumeration of fields) is a ``vars(inferencer)`` before/after
snapshot with an **empty allow-list**: any new/changed instance-``__dict__`` key is
a residual self-mutation the enumeration missed (the ~30 orphan per-run fields —
counters, emit-once flags, caches).  Same invariant-over-enumeration discipline as
the child-call lint, applied to mutable state.

Note: ``_active_ctx`` is a **module-level** ContextVar (not in instance ``__dict__``),
so it never trips the snapshot — only genuine instance mutation does.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any, Iterator, NamedTuple


class PurityDelta(NamedTuple):
    added: dict[str, Any]
    changed: dict[str, tuple[Any, Any]]  # name -> (before, after)

    @property
    def is_pure(self) -> bool:
        return not self.added and not self.changed


def snapshot_vars(obj: Any) -> dict[str, Any]:
    """Shallow-copy ``vars(obj)`` (best-effort deep for comparison robustness)."""
    raw = getattr(obj, "__dict__", {})
    out: dict[str, Any] = {}
    for k, v in raw.items():
        try:
            out[k] = copy.deepcopy(v)
        except Exception:
            out[k] = v  # uncopyable (e.g. live handle) — compare by identity
    return out


def diff_vars(before: dict[str, Any], after: dict[str, Any], allow: frozenset[str]) -> PurityDelta:
    added: dict[str, Any] = {}
    changed: dict[str, tuple[Any, Any]] = {}
    for key, val in after.items():
        if key in allow:
            continue
        if key not in before:
            added[key] = val
        elif before[key] != val:
            changed[key] = (before[key], val)
    return PurityDelta(added=added, changed=changed)


@contextlib.contextmanager
def purity_snapshot(obj: Any, *, allow: Iterable[str] = ()) -> Iterator[list[PurityDelta]]:
    """Context manager asserting ``obj``'s instance ``__dict__`` did not change.

    Yields a one-element list that, on exit, holds the :class:`PurityDelta`.
    The caller asserts ``delta.is_pure``.  ``allow`` is the (empty, for the real
    M7 gate) allow-list of permitted keys.
    """
    allow_set = frozenset(allow)
    before = snapshot_vars(obj)
    holder: list[PurityDelta] = []
    try:
        yield holder
    finally:
        after = snapshot_vars(obj)
        holder.append(diff_vars(before, after, allow_set))


def assert_pure(obj: Any, *, allow: Iterable[str] = ()) -> PurityDelta:
    """Convenience: compute the delta now (call before/after a run yourself)."""
    return diff_vars(snapshot_vars(obj), snapshot_vars(obj), frozenset(allow))

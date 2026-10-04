"""M0/§2.6 — the purity snapshot: the M7 per-class no-self-mutation gate.

M7 stops a class mutating per-run state on ``self``.  Verifying that **completely**
(vs. trusting a hand-enumeration of fields) is a ``vars(inferencer)`` before/after
snapshot with an **empty allow-list**: any added, removed or changed instance-``__dict__``
key is a residual self-mutation the enumeration missed (the ~30 orphan per-run fields —
counters, emit-once flags, caches).  Same invariant-over-enumeration discipline as
the child-call lint, applied to mutable state.

Values are compared by **fingerprint**, never by ``==`` on deep copies (a copy of an
object without ``__eq__`` never equals its original, and ``==`` can't see a nested
in-place mutation of a shared object):

* builtin containers compare by content, recursively;
* any other object compares by identity first, then by its ``vars()`` / ``__slots__``
  fields, so an in-place mutation at any depth is visible;
* cycles and aliasing are encoded as back-references;
* loggers, locks and the Tier-3 handle stores are normalized: re-creating them
  is idempotent definition caching, and a logger reaches the process-wide logging
  manager, whose unrelated growth must not read as a change;
* another ``InferencerBase`` is a boundary compared by identity only: children are
  gated on their own.  So is a ``unittest.mock`` test double, whose call log is not
  definition state.

Note: ``_active_ctx`` is a **module-level** ContextVar (not in instance ``__dict__``),
so it never trips the snapshot — only genuine instance mutation does.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import copy
import datetime
import decimal
import enum
import fractions
import functools
import logging
import pathlib
import re
import subprocess
import threading
import types
import uuid
from collections import deque
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Optional
from unittest import mock

from .handles import LiveHandles, LiveHandleStore


class PurityDelta(NamedTuple):
    added: dict[str, Any]
    changed: dict[str, tuple[Any, Any]]  # name -> (before, after)
    removed: dict[str, Any]

    @property
    def is_pure(self) -> bool:
        return not self.added and not self.changed and not self.removed


class VarsSnapshot(dict):
    """``name -> best-effort deep copy`` (for reports) plus the comparison fingerprints.

    ``_keepalive`` pins every identity-compared object, so a replaced object's id
    can't be reused by its replacement while the snapshot is alive.
    """

    def __init__(
        self,
        reports: Mapping[str, Any],
        fingerprints: dict[str, Any],
        keepalive: list[Any],
    ) -> None:
        super().__init__(reports)
        self.fingerprints = fingerprints
        self._keepalive = keepalive


_ATOMS = (type(None), bool, int, float, complex, str, bytes, type(Ellipsis), range)
_VALUE_TYPES = (
    datetime.date,
    datetime.time,
    datetime.timedelta,
    decimal.Decimal,
    fractions.Fraction,
    uuid.UUID,
    pathlib.PurePath,
    re.Pattern,
)
_VALUE_ATOMS = _ATOMS + _VALUE_TYPES
_IDENTITY_TYPES = (
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.ModuleType,
    types.MethodWrapperType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.CodeType,
    property,
    classmethod,
    staticmethod,
    asyncio.AbstractEventLoop,
    asyncio.Future,
    concurrent.futures.Future,
    concurrent.futures.Executor,
    threading.Thread,
    subprocess.Popen,
    mock.NonCallableMock,
)
_NAMED_LOGGING = (
    logging.Logger,
    logging.LoggerAdapter,
    logging.PlaceHolder,
    logging.Manager,
)
_NORMALIZED = (
    logging.Handler,
    logging.Formatter,
    logging.Filter,
    type(threading.Lock()),
    type(threading.RLock()),
    threading.Condition,
    threading.Event,
    threading.Semaphore,
    asyncio.Lock,
    asyncio.Event,
    asyncio.Condition,
    asyncio.Semaphore,
    LiveHandleStore,
    LiveHandles,
)
_MAPPINGS = (dict, types.MappingProxyType)
_SEQUENCES = (list, tuple, deque)
_SETS = (set, frozenset)
_NO_ROOT = object()
_UNSET = object()


@functools.lru_cache(maxsize=None)
def _slot_names(cls: type) -> tuple[str, ...]:
    names = []
    for klass in cls.__mro__:
        slots = vars(klass).get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{klass.__name__.lstrip('_')}{name}"
            names.append(name)
    return tuple(names)


class _Fingerprinter:
    """Fingerprints one attribute value; back-reference ordinals are local to it.

    ``keep`` (identity-compared objects) and ``shared`` (objects a report copy must
    not duplicate) are snapshot-wide ``id -> object`` maps.
    """

    def __init__(
        self,
        root: Any,
        boundary: Optional[type],
        keep: dict[int, Any],
        shared: dict[int, Any],
    ) -> None:
        self.root = root
        self.boundary = boundary
        self.keep = keep
        self.shared = shared
        self.memo: dict[int, int] = {}

    def fp(self, value: Any) -> Any:
        if isinstance(value, enum.Enum):
            return ("enum", type(value), value.name)
        if isinstance(value, (float, complex)):
            return (type(value), repr(value))  # NaN != NaN; repr is exact
        if isinstance(value, _VALUE_ATOMS):
            return (type(value), value)
        if value is self.root:
            return ("root",)
        token = self._shallow_token(value)
        if token is not None:
            return token
        ordinal = self.memo.get(id(value))
        if ordinal is not None:
            return ("ref", ordinal)
        self.memo[id(value)] = len(self.memo)
        return self._deep_token(value)

    def _pin(self, value: Any, *, shared: bool) -> None:
        self.keep[id(value)] = value
        if shared:
            self.shared[id(value)] = value

    def _shallow_token(self, value: Any) -> Any:
        if isinstance(value, _NAMED_LOGGING):
            self._pin(value, shared=True)
            return ("normalized", type(value), getattr(value, "name", None))
        if isinstance(value, _NORMALIZED):
            self._pin(value, shared=True)
            return ("normalized", type(value))
        if isinstance(value, types.MethodType):
            self._pin(value.__self__, shared=True)
            self._pin(value.__func__, shared=True)
            return ("method", id(value.__self__), id(value.__func__))
        is_boundary = self.boundary is not None and isinstance(value, self.boundary)
        if is_boundary or isinstance(value, _IDENTITY_TYPES):
            self._pin(value, shared=True)
            return ("identity", type(value), id(value))
        return None

    def _deep_token(self, value: Any) -> Any:
        if isinstance(value, _MAPPINGS):
            items = tuple((self.fp(k), self.fp(v)) for k, v in value.items())
            return ("map", type(value), items)
        if isinstance(value, _SEQUENCES):
            return ("seq", type(value), tuple(self.fp(v) for v in value))
        if isinstance(value, _SETS):
            members = sorted((self.fp(v) for v in value), key=repr)
            return ("set", type(value), tuple(members))
        self._pin(value, shared=False)
        fields = self._fields(value)
        if fields is None:
            return ("opaque", type(value), id(value))
        return ("obj", type(value), id(value), fields)

    def _fields(self, value: Any) -> Optional[tuple[Any, ...]]:
        attrs = getattr(value, "__dict__", None)
        attrs = attrs if isinstance(attrs, dict) else None
        slots = _slot_names(type(value))
        if attrs is None and not slots:
            return None
        out = [(k, self.fp(v)) for k, v in (attrs or {}).items()]
        for name in slots:
            v = getattr(value, name, _UNSET)
            out.append((name, ("unset",) if v is _UNSET else self.fp(v)))
        return tuple(out)


def _inferencer_base() -> type:
    from ..inferencer_base import InferencerBase

    return InferencerBase


def _report_copies(
    obj: Any, raw: Mapping[str, Any], shared: dict[int, Any]
) -> dict[str, Any]:
    memo: dict[int, Any] = {id(o): o for o in shared.values()}
    memo[id(obj)] = obj
    out: dict[str, Any] = {}
    for k, v in raw.items():
        try:
            out[k] = copy.deepcopy(v, memo)
        except Exception:
            out[k] = v  # uncopyable: the report shows the live value
    return out


def snapshot_vars(obj: Any) -> VarsSnapshot:
    """Fingerprint every ``vars(obj)`` value (plus a best-effort copy for reports)."""
    raw = getattr(obj, "__dict__", {})
    keep: dict[int, Any] = {}
    shared: dict[int, Any] = {}
    boundary = _inferencer_base()
    fingerprints = {
        k: _Fingerprinter(obj, boundary, keep, shared).fp(v) for k, v in raw.items()
    }
    reports = _report_copies(obj, raw, shared)
    return VarsSnapshot(reports, fingerprints, list(keep.values()))


def _fingerprints_of(values: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(values, VarsSnapshot):
        return values.fingerprints
    keep: dict[int, Any] = {}
    return {
        k: _Fingerprinter(_NO_ROOT, None, keep, {}).fp(v) for k, v in values.items()
    }


def diff_vars(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    allow: frozenset[str] = frozenset(),
) -> PurityDelta:
    """Delta between two ``snapshot_vars`` results (or two plain dicts)."""
    fp_before, fp_after = _fingerprints_of(before), _fingerprints_of(after)
    added = {k: after[k] for k in after if k not in before and k not in allow}
    removed = {k: before[k] for k in before if k not in after and k not in allow}
    changed = {
        k: (before[k], after[k])
        for k in after
        if k in before and k not in allow and fp_before[k] != fp_after[k]
    }
    return PurityDelta(added=added, changed=changed, removed=removed)


def describe_delta(delta: PurityDelta) -> str:
    parts = [
        f"{label}={sorted(keys)}"
        for label, keys in (
            ("added", delta.added),
            ("removed", delta.removed),
            ("changed", delta.changed),
        )
        if keys
    ]
    return ", ".join(parts) or "pure"


@contextlib.contextmanager
def purity_snapshot(
    obj: Any, *, allow: Iterable[str] = ()
) -> Iterator[list[PurityDelta]]:
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
        holder.append(diff_vars(before, snapshot_vars(obj), allow_set))


def assert_pure(
    obj: Any, before: VarsSnapshot, *, allow: Iterable[str] = ()
) -> PurityDelta:
    """Raise ``AssertionError`` if ``obj`` changed since ``before = snapshot_vars(obj)``."""
    delta = diff_vars(before, snapshot_vars(obj), frozenset(allow))
    if not delta.is_pure:
        raise AssertionError(
            f"{type(obj).__qualname__} instance state changed: {describe_delta(delta)}"
        )
    return delta

"""Durable record of one native conversation's vendor session.

Holds only opaque session coordinates, fingerprints and delivery cursors —
never rendered instructions, user text, transcripts or credentials. Host
notices are stored as typed references; their text is rendered when they are
delivered.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Protocol

from agent_foundation.common.inferencers.run_context.state import (
    decode_state,
    encode_state,
)

RECORD_SCHEMA = "native/v1"
# Bumped when the meaning of stored coordinates changes incompatibly; a record
# written by another adapter version is not resumed (the session rotates).
ADAPTER_VERSION = "1"


class StaleRecordError(RuntimeError):
    """A save would overwrite a newer record: one of a newer session
    generation (written after a rotation by another inferencer instance), or
    one saved after the copy being saved was loaded."""


@dataclass
class NativeSessionRecord:
    conversation_key: str
    backend: str = ""
    adapter_version: str = ADAPTER_VERSION
    vendor_session_id: str = ""
    coordinates: dict[str, Any] = field(default_factory=dict)  # e.g. Metamate fbid
    generation: int = 0
    cwd: str = ""
    principal: str = ""
    model: str = ""
    permission_fingerprint: str = ""
    nonce: str = ""
    l1_core_hash: str = ""
    catalog_snapshot: list[str] = field(default_factory=list)
    l2_hash: str = ""
    l2_generation: int = 0
    last_turn: int = 0
    last_round: int = 0
    turn_boundaries: dict[str, str] = field(default_factory=dict)
    outbox: list[dict[str, Any]] = field(default_factory=list)
    outbox_cursor: int = 0
    next_notice_id: int = 0
    pending_widget: bool = False
    submission: str = "prepared"
    status: str = "new"
    last_meta: dict[str, Any] = field(default_factory=dict)
    saved_at: float = 0.0
    schema: str = RECORD_SCHEMA

    @property
    def started(self) -> bool:
        return bool(self.vendor_session_id) and self.status != "new"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NativeSessionRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def rotate(self) -> None:
        """Retire the vendor session: the next turn starts a new one under the
        next generation (pending notices are kept)."""
        self.generation += 1
        self.vendor_session_id = ""
        self.status = "new"
        self.l1_core_hash = ""
        self.l2_hash = ""
        self.turn_boundaries = {}
        self.submission = "prepared"

    def add_notice(self, notice_type: str, **ref: Any) -> None:
        """Queue a one-shot host notice by reference (no rendered text)."""
        self.outbox.append({"id": self.next_notice_id, "type": notice_type, **ref})
        self.next_notice_id += 1

    def pending_notices(self) -> list[dict[str, Any]]:
        return self.outbox[self.outbox_cursor :]

    def mark_delivered(self, upto: int) -> None:
        """Drop notices ``[0, upto)`` after a successful vendor turn delivered
        them; notices queued while that turn ran stay pending."""
        upto = max(self.outbox_cursor, min(upto, len(self.outbox)))
        self.outbox = self.outbox[upto:]
        self.outbox_cursor = 0

    def newer_than(self, other: Optional["NativeSessionRecord"]) -> bool:
        """At least as new as ``other`` in the order stores compare and swap
        on, so a record that wins here can also be saved over ``other``."""
        if other is None:
            return True
        return (self.generation, self.saved_at) >= (other.generation, other.saved_at)


def check_not_stale(
    stored: Optional[NativeSessionRecord], record: NativeSessionRecord
) -> None:
    """Refuse a save that would replace a newer record: a higher generation,
    or the same generation saved after ``record`` was (its ``saved_at``)."""
    if stored is None:
        return
    if stored.generation > record.generation:
        raise StaleRecordError(
            f"Record for {record.conversation_key!r} is at generation "
            f"{stored.generation}; refusing to overwrite it with generation "
            f"{record.generation}."
        )
    if stored.generation == record.generation and stored.saved_at > record.saved_at:
        raise StaleRecordError(
            f"Record for {record.conversation_key!r} was saved again after this "
            "copy was loaded; refusing to overwrite the newer save."
        )


def compare_and_save(
    lock: AbstractContextManager[Any],
    load: Callable[[], Optional[NativeSessionRecord]],
    write: Callable[[dict[str, Any]], None],
    record: NativeSessionRecord,
) -> None:
    """Save ``record`` unless the stored record is newer (``check_not_stale``),
    atomically under ``lock``. Every save stamps a ``saved_at`` above the
    stored one, so a copy loaded before any later save always loses."""
    with lock:
        stored = load()
        check_not_stale(stored, record)
        now = time.time()
        floor = stored.saved_at if stored is not None else 0.0
        saved_at = now if now > floor else math.nextafter(floor, math.inf)
        data = record.to_dict()
        data["saved_at"] = saved_at
        write(data)
        record.saved_at = saved_at


def end_vendor_session(
    store: "NativeSessionRecordStore", conversation_key: str, *, recap: bool = True
) -> bool:
    """Retire a conversation's vendor session without an inferencer — e.g. the
    host moved the conversation to another backend, so the vendor session will
    miss the turns that run there. The next native turn starts a fresh vendor
    session, with a recap of the host history when ``recap``. Returns whether
    a started session was retired."""
    record = store.load(conversation_key)
    if record is None or not record.started:
        return False
    record.rotate()
    if recap:
        record.add_notice("recap")
    store.save(record)
    return True


class NativeSessionRecordStore(Protocol):
    """Where a host keeps native session records (keyed by conversation).

    ``save`` is a compare-and-swap on ``(generation, saved_at)``
    (``compare_and_save``): it refuses (``StaleRecordError``) to replace a
    record of a newer generation — the fence that keeps a stale inferencer
    from undoing a rotation made by another instance — or a record of the
    same generation saved after the copy being saved was loaded."""

    def load(self, conversation_key: str) -> Optional[NativeSessionRecord]: ...

    def save(self, record: NativeSessionRecord) -> None: ...


class InMemoryRecordStore:
    """Process-local store (default; also used by tests)."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def load(self, conversation_key: str) -> Optional[NativeSessionRecord]:
        data = self._records.get(conversation_key)
        return NativeSessionRecord.from_dict(data) if data else None

    def save(self, record: NativeSessionRecord) -> None:
        key = record.conversation_key
        compare_and_save(
            self._lock,
            lambda: self.load(key),
            lambda data: self._records.__setitem__(key, data),
            record,
        )


class CallbackRecordStore:
    """Adapter for hosts that persist the record inside their own session
    storage (e.g. OpenStartup's ``session["native_session"]``).

    ``lock`` serializes the compare-and-swap; hosts whose storage several
    stores write pass the lock they all share."""

    def __init__(
        self,
        load_fn: Callable[[str], Optional[dict[str, Any]]],
        save_fn: Callable[[str, dict[str, Any]], None],
        *,
        lock: Optional[AbstractContextManager[Any]] = None,
    ) -> None:
        self._load_fn = load_fn
        self._save_fn = save_fn
        self._lock = lock if lock is not None else threading.Lock()

    def load(self, conversation_key: str) -> Optional[NativeSessionRecord]:
        data = self._load_fn(conversation_key)
        return NativeSessionRecord.from_dict(data) if data else None

    def save(self, record: NativeSessionRecord) -> None:
        key = record.conversation_key
        compare_and_save(
            self._lock,
            lambda: self.load(key),
            lambda data: self._save_fn(key, data),
            record,
        )


class RunContextRecordStore:
    """Keeps records in the checkpoints of the session-stable node of a
    RunContext's run-state store, encoded with ``encode_state``, so they are
    persisted with that store (``RunStateStore.save`` / ``load``).

    That node is the store's root: hosts derive each turn's context as a
    child (``ctx.child("turn_N")``) of one session context whose store they
    persist across turns, and a per-turn node is not carried to the next turn.
    Any context of that store may be passed."""

    KEY = "native_session_v1"
    _SESSION_NODE = "/"

    def __init__(self, run_context: Any) -> None:
        self._ctx = run_context
        self._lock = threading.Lock()

    def _bucket(self) -> dict[str, Any]:
        node = self._ctx.store.node(self._SESSION_NODE)
        return node.checkpoints.setdefault(self.KEY, {})

    def load(self, conversation_key: str) -> Optional[NativeSessionRecord]:
        data = decode_state(self._bucket().get(conversation_key))
        return NativeSessionRecord.from_dict(data) if data else None

    def save(self, record: NativeSessionRecord) -> None:
        key = record.conversation_key
        compare_and_save(
            self._lock,
            lambda: self.load(key),
            lambda data: self._bucket().__setitem__(key, encode_state(data)),
            record,
        )

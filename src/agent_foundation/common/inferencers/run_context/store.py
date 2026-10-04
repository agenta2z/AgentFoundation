"""Tier-1: the serializable, path-keyed run-state store.

Per the plan (§2.1 / §2.7 / N-R4 / N-S2 / P-#7):

* :class:`NodeRunState` holds the per-path state in two lifetime buckets —
  ``call`` (one top-level ainfer call) and ``attempt`` (one retry attempt;
  ``reset_attempt`` clears it) — plus ``conversation`` / ``checkpoints`` /
  ``provenance``.  Each of ``call`` / ``attempt`` is a typed state **or** a dict.
* :class:`RunStateStore` is **per-turn** (the app mints one root per top-level
  turn and discards it; the persisted ``to_json`` is the durable artifact —
  bounded memory, P-#7).  Tier-3 handles are connection-scoped and live
  elsewhere (see ``handles.py``), so discarding this store does not drop a
  connection.
* ``node(path, creator)`` is **get-or-create + a collision guard**.  The creator
  key is a **stable** ``(class_qualname, slot)`` signature (never ``id()``), kept
  **transient** (not serialized) — so resume is safe: rehydrated nodes carry no
  creator tag and are re-tagged as they are first re-entered.  A re-request by a
  *different* creator raises; re-entry by the same creator (retry) is a no-op.
* ``claims`` (:class:`ActivePathClaims`) records which invocation currently holds
  each path (plan v8 §5.1). It is strict: a second overlapping invocation at one
  path raises, whatever its class or nesting. Claims are live runtime state: they
  are never serialized, and a rehydrated or copied store starts with none.
* ``outcome`` holds the typed :class:`NodeOutcomeState` an invocation published at
  its successful close. The seam clears it when the next invocation opens at the
  path; neither the clear nor a read creates or claims the node.
"""

from __future__ import annotations

import threading
import time
from typing import Any, TYPE_CHECKING

import attrs

from .errors import ConcurrentInvocationError, InvocationContractError
from .state import decode_state, encode_state, RoleState

if TYPE_CHECKING:
    from .invocation import InvocationFrame

# A stable, resume-safe creator signature: (class_qualname, slot).  Never id().
CreatorKey = tuple[str, str]


class CollisionError(RuntimeError):
    """Two distinct creators claimed the same context path (silent-corruption guard)."""


@attrs.define
class NodeRunState:
    """Serializable per-path state.  ``_creator`` is transient (never persisted)."""

    path: str
    call: Any = None  # typed state | dict | None — one top-level ainfer call
    attempt: Any = None  # typed state | dict | None — one retry attempt
    conversation: dict[str, Any] = attrs.field(factory=dict)
    checkpoints: dict[str, Any] = attrs.field(factory=dict)
    provenance: list[Any] = attrs.field(factory=list)
    # M7 role override (template/role switch), kept SEPARATE from ``call`` so a typed
    # dispatch/runner state and a role switch can coexist on one node (GT#15).  Serialized
    # like ``call``/``attempt`` (a ``RoleState`` typed value or ``None``).
    role_state: Any = None
    # typed ``NodeOutcomeState`` published at an invocation's successful close; ``None``
    # until then, and again from the moment the next invocation opens at this path.
    outcome: Any = None
    # Transient runtime metadata — NOT serialized (eq=False so it never affects equality).
    _creator: CreatorKey | None = attrs.field(default=None, eq=False)
    # Tier-1 **transient scratch** — non-picklable per-run working state (Part G/G2:
    # ``_current_*`` / ``_steps`` / live expansion flags). eq=False AND excluded from
    # ``to_json`` (re-derived per run, never resumed).
    scratch: dict[str, Any] = attrs.field(factory=dict, eq=False)

    def reset_attempt(self) -> None:
        """Clear attempt-scoped state (a retry begins)."""
        self.attempt = None

    def to_json(self) -> dict[str, Any]:
        """Serialize Tier-1 only — never ``_creator``/``scratch`` (transient) or Tier-2/3 refs."""
        return {
            "path": self.path,
            "call": encode_state(self.call),
            "attempt": encode_state(self.attempt),
            "role_state": encode_state(self.role_state),
            "outcome": encode_state(self.outcome),
            "conversation": dict(self.conversation),
            "checkpoints": dict(self.checkpoints),
            "provenance": list(self.provenance),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "NodeRunState":
        call = decode_state(data.get("call"))
        role_state = decode_state(data.get("role_state"))
        # Older saves recorded ``switch_role`` in ``call``; lift it out (one-way).
        if isinstance(call, RoleState):
            role_state = call if role_state is None else role_state
            call = None
        return cls(
            path=data["path"],
            call=call,
            attempt=decode_state(data.get("attempt")),
            role_state=role_state,
            outcome=decode_state(data.get("outcome")),
            conversation=dict(data.get("conversation") or {}),
            checkpoints=dict(data.get("checkpoints") or {}),
            provenance=list(data.get("provenance") or []),
            # _creator/scratch deliberately omitted -> defaults on resume (transient).
        )


def _describe(frame: InvocationFrame) -> str:
    return f"{type(frame.owner).__name__}.{frame.entry}"


def _concurrent_invocation_message(
    path: str, holder: InvocationFrame, caller: InvocationFrame
) -> str:
    age = time.monotonic() - holder.opened_at
    message = (
        f"Path {path!r} is already in an open invocation: {_describe(holder)}, "
        f"opened {age:.1f}s ago; refusing {_describe(caller)} at the same path. "
        f"One path hosts one invocation at a time: nested or concurrent work runs "
        f"at a child path (ctx.child(<slot>))."
    )
    if "streaming" in holder.entry:
        message += (
            " The holder is a stream that is still open: close it "
            "(contextlib.aclosing / contextlib.closing) before calling again at "
            "this path."
        )
    return message


class ActivePathClaims:
    """The open invocation holding each path of one :class:`RunStateStore`."""

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._holders: dict[str, InvocationFrame] = {}

    def acquire(self, path: str, frame: InvocationFrame) -> None:
        with self._lock:
            holder = self._holders.get(path)
            if holder is not None:
                raise ConcurrentInvocationError(
                    _concurrent_invocation_message(path, holder, frame)
                )
            self._holders[path] = frame

    def release(self, path: str, frame: InvocationFrame) -> None:
        with self._lock:
            holder = self._holders.get(path)
            if holder is not frame:
                raise InvocationContractError(
                    f"{_describe(frame)} cannot release path {path!r}: the claim is "
                    f"held by {_describe(holder) if holder else 'no invocation'}."
                )
            del self._holders[path]

    def holder(self, path: str) -> InvocationFrame | None:
        with self._lock:
            return self._holders.get(path)

    def live_below(self, prefix: str) -> list[str]:
        """The claimed paths strictly below ``prefix``, sorted."""
        below = prefix.rstrip("/") + "/"
        with self._lock:
            return sorted(p for p in self._holders if p.startswith(below))


class RunStateStore:
    """Tier-1, per-turn, path-keyed store.  Shared by reference down the ctx tree."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeRunState] = {}
        self._lock = threading.RLock()
        self.claims = ActivePathClaims(self._lock)

    def __getstate__(self) -> dict[str, Any]:
        return {"_nodes": self._nodes}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._nodes = state["_nodes"]
        self._lock = threading.RLock()
        self.claims = ActivePathClaims(self._lock)

    def node(self, path: str, creator: CreatorKey | None = None) -> NodeRunState:
        """Get-or-create the node at ``path`` with the resume-safe collision guard."""
        existing = self._nodes.get(path)
        if existing is not None:
            if (
                creator is not None
                and existing._creator is not None
                and existing._creator != creator
            ):
                raise CollisionError(
                    f"Path {path!r} already claimed by creator {existing._creator!r}; "
                    f"a different creator {creator!r} requested it. Two distinct nodes "
                    f"must not share one context path (silent state/workspace corruption)."
                )
            # Re-entry by the same creator (retry) — or an untagged node being
            # re-tagged on resume — is a no-op; tag it if not yet tagged.
            if existing._creator is None and creator is not None:
                existing._creator = creator
            return existing
        node = NodeRunState(path=path, creator=creator)
        self._nodes[path] = node
        return node

    def has(self, path: str) -> bool:
        return path in self._nodes

    def peek(self, path: str) -> NodeRunState | None:
        """Read-only lookup: return the node at ``path`` or ``None`` (no create, no guard).

        Unlike :meth:`node` this never creates or claims a node, so it is safe for an
        ancestor-walk that probes paths which may not exist (e.g. resolving the MFI's
        cross-flow attempt node from inside a flow's child ctx — Part B walk-up).
        """
        return self._nodes.get(path)

    def clear_outcome(self, path: str) -> None:
        """Drop the outcome at ``path`` without creating or claiming the node."""
        with self._lock:
            node = self._nodes.get(path)
            if node is not None:
                node.outcome = None

    def publish_outcome(
        self, path: str, creator: CreatorKey | None, outcome: Any
    ) -> None:
        with self._lock:
            self.node(path, creator).outcome = outcome

    def evict_subtree(self, prefix: str) -> int:
        """Drop every node at or under ``prefix`` — release a subtree's path claims (U1).

        Before a retry RE-DERIVES a subtree (e.g. the Dual re-selects its review
        panel), the stale ``(class_qualname, slot)`` creator claims from the prior
        attempt must be released — otherwise a slot rebound to a DIFFERENT leaf
        class trips the :class:`CollisionError` guard in :meth:`node`. Scoped
        STRICTLY to ``prefix`` and its descendants (path ``== prefix`` or starting
        ``prefix + '/'``); it NEVER clears the whole store, so sibling workers that
        share this per-turn store are untouched. An empty/``"/"`` prefix is a
        no-op (refuses to evict the root). Returns the number of nodes removed.

        Raises :class:`InvocationContractError` if an invocation is still open
        strictly below ``prefix``: children must finish before their subtree is
        evicted. The caller's own claim at ``prefix`` is allowed.
        """
        boundary = (prefix or "").strip().rstrip("/")
        if not boundary:
            return 0  # refuse to evict the whole store (empty / root prefix)
        with self._lock:
            live = self.claims.live_below(boundary)
            if live:
                raise InvocationContractError(
                    f"Cannot evict the subtree at {boundary!r}: invocations are "
                    f"still open below it at {live}. Children must finish before "
                    f"their subtree is evicted."
                )
            victims = [
                p for p in self._nodes if p == boundary or p.startswith(boundary + "/")
            ]
            for p in victims:
                del self._nodes[p]
            return len(victims)

    def to_json(self) -> dict[str, Any]:
        return {"nodes": {p: n.to_json() for p, n in self._nodes.items()}}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RunStateStore":
        store = cls()
        for path, node_data in (data.get("nodes") or {}).items():
            store._nodes[path] = NodeRunState.from_json(node_data)
        return store

    # -- M9 resume: file persistence ---------------------------------------
    def save(self, path: str) -> None:
        """Persist the Tier-1 store to a JSON file (M9 resume).

        The host (task executor / SOP CLI / ConversationService — §9.4) calls
        this to durably snapshot run state alongside the existing result
        checkpoints. Tier-2 (sinks) and Tier-3 (live handles) are deliberately
        NOT persisted — they are re-supplied / re-established on resume.

        Uses atomic write (temp file + rename) to prevent truncated store.json
        on serialization errors. If serialization fails, the previous store.json
        (if any) is left intact rather than being truncated mid-write.
        """
        import json
        import os
        import tempfile

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = self.to_json()
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".store_", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "RunStateStore":
        """Rehydrate a Tier-1 store from a JSON file (M9 resume).

        Creator tags are NOT restored (transient, N-R4) — they are rebuilt as
        nodes are re-entered during the resumed run. Conversation/checkpoints/
        typed-or-dict call state round-trip via ``from_json``.
        """
        import json

        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def __len__(self) -> int:
        return len(self._nodes)

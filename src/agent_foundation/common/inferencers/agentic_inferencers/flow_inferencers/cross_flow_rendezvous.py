"""Cross-flow step rendezvous — a phaser-style barrier for MultiFlow lock-step rounds.

When ``MultiFlowInferencer`` runs N flows concurrently (each an independent
``LinearWorkflowInferencer`` loop), peer visibility is a best-effort snapshot of
``_latest_per_flow``. A fast flow can build its ``round01`` input before a slow peer
finishes ``initial`` and read ``(no output yet)``. This rendezvous closes that race:
each flow, after publishing its round ``x`` output and before reading peers for round
``x+1``, arrives here and blocks until **all still-active flows** have published round
``x`` — then all release together.

Design notes
------------
* **Set-based, not a counter.** ``arrive``/``leave`` are keyed by ``flow_idx`` and are
  idempotent. WorkGraph retries a failed worker node from scratch, so a flow's
  ``leave`` may fire twice (once per attempt); membership semantics make the second a
  no-op, and an ``arrive`` from a flow that already departed returns immediately rather
  than corrupting the shared generation (see ``arrive_and_wait``).
* **Single generation.** Because the barrier forces lock-step, all active flows stay
  within one generation; ``_arrived`` only ever holds arrivals for the current barrier.
  This is *why* a departed flow must NOT re-register: re-entry at a low barrier while
  peers wait at a higher one would make ``_arrived == _active`` and fire a spurious
  release. Departure is therefore permanent (a retried flow runs the rest of the
  attempt uncoordinated — acceptable "best effort").
* **Deadlock-free under one invariant:** *every index seeded into ``_active`` eventually
  calls ``leave``.* The caller (MultiFlow) guarantees this by (a) seeding ``_active``
  with only the flows that will actually run, and (b) calling ``leave`` in a ``finally``
  around each flow worker.
* **Transient.** Holds an ``asyncio`` primitive; it must live in ``node.scratch``
  (never serialized) and is re-created fresh on resume.
* **Lazy loop bind.** The ``asyncio.Condition`` is created on first use so the object
  can be constructed outside a running loop (e.g. during synchronous graph setup) and
  still bind to whatever loop drives the BTA fan-out.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Optional, Set


class CrossFlowRendezvous:
    """A drop-out-tolerant, retry-idempotent step barrier across sibling flows."""

    def __init__(self, participants: Iterable[int]) -> None:
        # Flow indices expected to participate this attempt (the flows that will
        # actually run their ``_ainfer`` — completed/cached flows are excluded).
        self._active: Set[int] = set(participants)
        self._arrived: Set[int] = set()
        self._gen: int = 0
        self._cond: Optional[asyncio.Condition] = None  # lazy-bound to the running loop
        # Frozen peer view captured ATOMICALLY when a generation advances (all active
        # flows have published round N-1). Released flows read THIS, not the live buffer,
        # so a fast peer racing ahead into round N can't contaminate a slow peer's read
        # (lock-step consistency: every flow sees the same round N-1 snapshot).
        self._snapshot = None
        self._pending_snapshot_fn = None

    # ------------------------------------------------------------------
    def _ensure_cond(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @property
    def active_count(self) -> int:
        return len(self._active)

    def is_active(self, idx: int) -> bool:
        return idx in self._active

    # ------------------------------------------------------------------
    async def arrive_and_wait(self, idx: int, snapshot_fn=None):
        """Block until every still-active flow has arrived at this barrier.

        ``snapshot_fn`` (optional): a zero-arg callable returning the peer view to freeze.
        It is invoked exactly once per generation, by whichever event advances it (the
        last arrival, or a ``leave`` that completes the reduced quorum), while every active
        flow's round N-1 output is published. All released flows receive that SAME frozen
        value — the return of ``arrive_and_wait`` — so reads are lock-step consistent.

        Returns the frozen snapshot (or ``None`` if no ``snapshot_fn`` was supplied).
        Returns immediately if ``idx`` already departed — see the permanent-departure
        rationale in the module docstring.
        """
        cond = self._ensure_cond()
        async with cond:
            if idx not in self._active:
                # Departed (e.g. post-retry re-run). Do not block and do not touch
                # ``_arrived``/``_gen`` — re-registering would corrupt peers' barrier.
                return self._snapshot
            if snapshot_fn is not None:
                self._pending_snapshot_fn = snapshot_fn
            self._arrived.add(idx)
            if self._arrived == self._active:
                # Last to arrive: freeze the snapshot, release everyone, open next gen.
                self._advance_locked()
            else:
                g = self._gen
                # ``_gen`` is monotonic, so once it advances the predicate stays true
                # (no lost wakeup, no missed generation).
                await cond.wait_for(lambda: self._gen != g)
            return self._snapshot

    async def leave(self, idx: int) -> None:
        """Permanently remove ``idx`` from the barrier (stop / completion / crash).

        Idempotent (double-leave from a WorkGraph retry is a no-op) and
        cancellation-proof: the mutation+notify runs under ``asyncio.shield`` so a
        ``CancelledError`` on the departing worker can't drop the departure and hang
        the surviving peers.
        """
        cond = self._ensure_cond()
        await asyncio.shield(self._leave_locked(idx, cond))

    async def _leave_locked(self, idx: int, cond: asyncio.Condition) -> None:
        async with cond:
            if idx not in self._active:
                return  # idempotent: already departed
            self._active.discard(idx)
            self._arrived.discard(idx)
            if self._active and self._arrived == self._active:
                # The remaining arrived flows now satisfy the (reduced) quorum.
                self._advance_locked()
            elif not self._active:
                # Nobody left; wake any stragglers so they don't wait forever. Safe:
                # a real waiter is by definition still in ``_active``, so this branch
                # only fires when no one is genuinely blocked.
                cond.notify_all()

    def _advance_locked(self) -> None:
        # Freeze the peer view BEFORE releasing anyone, so every flow (the advancer and
        # the woken waiters) reads the identical round N-1 snapshot. ``_pending_snapshot_fn``
        # was registered by the arriving flows this generation; it reads the live buffer
        # (already fully populated — publish precedes arrive), so even a leave-triggered
        # advance captures a consistent view.
        if self._pending_snapshot_fn is not None:
            self._snapshot = self._pending_snapshot_fn()
        self._gen += 1
        self._arrived.clear()
        assert self._cond is not None  # _advance_locked is only called under the lock
        self._cond.notify_all()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"CrossFlowRendezvous(active={sorted(self._active)}, "
            f"arrived={sorted(self._arrived)}, gen={self._gen})"
        )

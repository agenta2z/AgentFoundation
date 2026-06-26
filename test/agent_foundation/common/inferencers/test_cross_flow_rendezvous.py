"""Unit tests for ``CrossFlowRendezvous`` — the lock-step step barrier.

These exercise the barrier algebra in isolation (no inferencer deps), covering the
deadlock-freedom scenarios from the design review:
- N flows release together (and not before the last arrives)
- early departure reduces the release threshold (drop-out)
- a departed-then-arrived flow does not block (retry tolerance)
- double-leave is a no-op (retry idempotency)
- N=1, all-leave, multi-generation lock-step
- mixed arrive/leave orderings release waiters
"""

from __future__ import annotations

import asyncio
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.cross_flow_rendezvous import (
    CrossFlowRendezvous,
)


def _run(coro):
    """Run an async test body on a fresh event loop (no runner-loop pollution)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestCrossFlowRendezvous(unittest.TestCase):
    # ------------------------------------------------------------------
    def test_all_arrive_release_together(self):
        async def body():
            rdv = CrossFlowRendezvous([0, 1, 2])
            passed: list[int] = []

            async def arrive(idx):
                await rdv.arrive_and_wait(idx)
                passed.append(idx)

            t0 = asyncio.ensure_future(arrive(0))
            t1 = asyncio.ensure_future(arrive(1))
            # Let 0 and 1 reach the barrier and block.
            await asyncio.sleep(0.02)
            self.assertFalse(t0.done(), "flow 0 must wait for flow 2")
            self.assertFalse(t1.done(), "flow 1 must wait for flow 2")
            self.assertEqual(passed, [])

            t2 = asyncio.ensure_future(arrive(2))
            await asyncio.wait_for(asyncio.gather(t0, t1, t2), timeout=1.0)
            self.assertEqual(sorted(passed), [0, 1, 2])

        _run(body())

    # ------------------------------------------------------------------
    def test_leave_reduces_threshold(self):
        """If flow 2 departs, flows 0 and 1 release without it."""

        async def body():
            rdv = CrossFlowRendezvous([0, 1, 2])
            passed: list[int] = []

            async def arrive(idx):
                await rdv.arrive_and_wait(idx)
                passed.append(idx)

            t0 = asyncio.ensure_future(arrive(0))
            t1 = asyncio.ensure_future(arrive(1))
            await asyncio.sleep(0.02)
            self.assertFalse(t0.done())
            self.assertFalse(t1.done())

            # Flow 2 stops early — never arrives, departs instead.
            await rdv.leave(2)
            await asyncio.wait_for(asyncio.gather(t0, t1), timeout=1.0)
            self.assertEqual(sorted(passed), [0, 1])
            self.assertEqual(rdv.active_count, 2)

        _run(body())

    # ------------------------------------------------------------------
    def test_leave_before_arrive(self):
        """Departure recorded before peers arrive still releases them."""

        async def body():
            rdv = CrossFlowRendezvous([0, 1, 2])
            await rdv.leave(2)
            passed: list[int] = []

            async def arrive(idx):
                await rdv.arrive_and_wait(idx)
                passed.append(idx)

            await asyncio.wait_for(
                asyncio.gather(arrive(0), arrive(1)), timeout=1.0
            )
            self.assertEqual(sorted(passed), [0, 1])

        _run(body())

    # ------------------------------------------------------------------
    def test_arrive_after_leave_returns_immediately(self):
        """A retried (already-departed) flow does not block at the barrier."""

        async def body():
            rdv = CrossFlowRendezvous([0, 1])
            await rdv.leave(0)
            # Flow 0 'retries' and arrives though it already departed.
            await asyncio.wait_for(rdv.arrive_and_wait(0), timeout=1.0)
            # And it must not have disturbed flow 1's solo barrier.
            await asyncio.wait_for(rdv.arrive_and_wait(1), timeout=1.0)

        _run(body())

    # ------------------------------------------------------------------
    def test_double_leave_is_noop(self):
        async def body():
            rdv = CrossFlowRendezvous([0, 1])
            await rdv.leave(0)
            await rdv.leave(0)  # must not underflow / raise
            self.assertEqual(rdv.active_count, 1)
            # Remaining flow still releases solo.
            await asyncio.wait_for(rdv.arrive_and_wait(1), timeout=1.0)

        _run(body())

    # ------------------------------------------------------------------
    def test_single_flow_never_blocks(self):
        async def body():
            rdv = CrossFlowRendezvous([0])
            await asyncio.wait_for(rdv.arrive_and_wait(0), timeout=1.0)
            await asyncio.wait_for(rdv.arrive_and_wait(0), timeout=1.0)  # next round

        _run(body())

    # ------------------------------------------------------------------
    def test_all_leave_no_hang(self):
        async def body():
            rdv = CrossFlowRendezvous([0, 1])
            await rdv.leave(0)
            await rdv.leave(1)
            self.assertEqual(rdv.active_count, 0)
            # A late arrive from anyone returns immediately (no participants).
            await asyncio.wait_for(rdv.arrive_and_wait(0), timeout=1.0)

        _run(body())

    # ------------------------------------------------------------------
    def test_multi_generation_lockstep(self):
        """Two consecutive barriers both synchronize; generation advances."""

        async def body():
            rdv = CrossFlowRendezvous([0, 1])
            order: list[str] = []

            async def flow(idx, rounds):
                for r in range(rounds):
                    await rdv.arrive_and_wait(idx)
                    order.append(f"{idx}:r{r}")

            await asyncio.wait_for(
                asyncio.gather(flow(0, 3), flow(1, 3)), timeout=2.0
            )
            # Every round saw both flows (6 entries), and no flow ran ahead a full
            # round: at each round boundary both indices appear before the next.
            self.assertEqual(len(order), 6)
            for r in range(3):
                round_entries = {e for e in order if e.endswith(f"r{r}")}
                self.assertEqual(round_entries, {f"0:r{r}", f"1:r{r}"})

        _run(body())

    # ------------------------------------------------------------------
    def test_mixed_max_steps_no_deadlock(self):
        """A short flow (stops after 1 barrier) doesn't hang a longer peer."""

        async def body():
            rdv = CrossFlowRendezvous([0, 1])
            passed: list[str] = []

            async def short_flow():
                await rdv.arrive_and_wait(0)  # barrier 1
                passed.append("0:b1")
                await rdv.leave(0)  # stops here

            async def long_flow():
                await rdv.arrive_and_wait(1)  # barrier 1 (with flow 0)
                passed.append("1:b1")
                await rdv.arrive_and_wait(1)  # barrier 2 (solo, flow 0 departed)
                passed.append("1:b2")
                await rdv.leave(1)

            await asyncio.wait_for(
                asyncio.gather(short_flow(), long_flow()), timeout=2.0
            )
            self.assertIn("0:b1", passed)
            self.assertIn("1:b1", passed)
            self.assertIn("1:b2", passed)

        _run(body())

    # ------------------------------------------------------------------
    def test_snapshot_frozen_and_shared(self):
        """All released flows get the SAME snapshot, frozen at the advance (last arrival)."""

        async def body():
            rdv = CrossFlowRendezvous([0, 1])
            box = {"v": "before_publish"}
            results: dict[int, object] = {}

            async def flow(idx):
                results[idx] = await rdv.arrive_and_wait(idx, snapshot_fn=lambda: box["v"])

            t0 = asyncio.ensure_future(flow(0))
            await asyncio.sleep(0.02)  # flow 0 is now waiting at the barrier
            box["v"] = "all_published"  # state when the LAST flow arrives
            t1 = asyncio.ensure_future(flow(1))
            await asyncio.wait_for(asyncio.gather(t0, t1), timeout=1.0)
            box["v"] = "round_N_contamination"  # post-release mutation must NOT leak in
            self.assertEqual(results[0], "all_published")
            self.assertEqual(results[1], "all_published")

        _run(body())

    def test_leave_triggered_advance_captures_snapshot(self):
        """A leave that completes the reduced quorum still freezes a snapshot for waiters."""

        async def body():
            rdv = CrossFlowRendezvous([0, 1])
            box = {"v": "x"}
            res: dict[int, object] = {}

            async def flow0():
                res[0] = await rdv.arrive_and_wait(0, snapshot_fn=lambda: box["v"])

            t0 = asyncio.ensure_future(flow0())
            await asyncio.sleep(0.02)  # flow 0 waiting
            box["v"] = "frozen_at_leave"
            await rdv.leave(1)  # advance via leave → must capture the snapshot
            await asyncio.wait_for(t0, timeout=1.0)
            self.assertEqual(res[0], "frozen_at_leave")

        _run(body())

    def test_leave_completes_under_cancellation(self):
        """A worker cancelled while departing still releases its peers (D3)."""

        async def body():
            rdv = CrossFlowRendezvous([0, 1])
            passed: list[int] = []

            async def waiter():
                await rdv.arrive_and_wait(1)
                passed.append(1)

            async def departing():
                # Simulate the worker's finally calling leave, then being cancelled.
                try:
                    await asyncio.sleep(3600)  # will be cancelled
                finally:
                    await rdv.leave(0)

            w = asyncio.ensure_future(waiter())
            d = asyncio.ensure_future(departing())
            await asyncio.sleep(0.02)
            self.assertFalse(w.done(), "waiter blocks until flow 0 departs")
            d.cancel()
            # Even though d is cancelled, its finally's leave(0) must release w.
            await asyncio.wait_for(w, timeout=1.0)
            self.assertEqual(passed, [1])
            with self.assertRaises(asyncio.CancelledError):
                await d

        _run(body())


if __name__ == "__main__":
    unittest.main()

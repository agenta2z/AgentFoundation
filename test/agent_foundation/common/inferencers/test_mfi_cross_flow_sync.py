"""MFI-level tests for cross-flow step coordination (cross_flow_sync barrier).

C2 coverage here: the knob gates rendezvous seeding, the rendezvous is seeded with the
right participant set, and resolution (no-ctx backing path) returns it. The full
under-ctx barrier behavior (publish→wait→read) and deregister are covered by the
integration tests added in later commits.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.cross_flow_rendezvous import (
    CrossFlowRendezvous,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)


class _AsyncSlowInferencer(InferencerBase):
    """Async inferencer that sleeps before returning — a deliberately slow flow leg."""

    def __init__(self, response, delay):
        super().__init__()
        self._response = response
        self._delay = delay

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._response  # sync fallback (async path uses _ainfer below)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        await asyncio.sleep(self._delay)
        return self._response


class _CapturingInferencer(InferencerBase):
    """Records every prompt it receives; returns responses in order (async-path safe)."""

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self._idx = 0
        self.received_prompts = []

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.received_prompts.append(str(inference_input))
        r = self._responses[self._idx] if self._idx < len(self._responses) else f"ov_{self._idx}"
        self._idx += 1
        return r


def _run_ainfer(mfi, inp, timeout=10.0):
    """Run mfi.ainfer on a fresh loop with a timeout (so a barrier deadlock fails fast)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(mfi.ainfer(inp), timeout))
    finally:
        loop.close()


def _make_bare_mfi(n_flows: int, *, cross_flow_sync=False, coordinated_stop=False):
    """A bare MFI (attrs init bypassed) with just the cross-flow state wired up.

    Avoids the aggregator-config validation in ``__attrs_post_init__`` — we only exercise
    ``_reset_cross_flow_state`` (no-ctx path) + the coordination helpers.
    """
    obj = MultiFlowInferencer.__new__(MultiFlowInferencer)
    obj.flow_configs = [{"input": f"t{i}"} for i in range(n_flows)]
    obj.cross_flow_sync = cross_flow_sync
    obj.coordinated_stop = coordinated_stop
    obj._latest_per_flow_backing = {}
    obj._latest_per_flow_path_backing = {}
    obj._all_judgments_backing = []
    obj._cross_flow_rendezvous_backing = None
    return obj


class TestCrossFlowSyncSeeding(unittest.TestCase):
    def test_disabled_by_default(self):
        obj = _make_bare_mfi(3)
        self.assertFalse(obj._coordination_enabled)
        obj._reset_cross_flow_state()
        self.assertIsNone(obj._resolve_rendezvous())

    def test_cross_flow_sync_enables_and_seeds(self):
        obj = _make_bare_mfi(3, cross_flow_sync=True)
        self.assertTrue(obj._coordination_enabled)
        obj._reset_cross_flow_state()
        rdv = obj._resolve_rendezvous()
        self.assertIsInstance(rdv, CrossFlowRendezvous)
        self.assertEqual(rdv.active_count, 3)
        for i in range(3):
            self.assertTrue(rdv.is_active(i))

    def test_coordinated_stop_is_alias(self):
        obj = _make_bare_mfi(2, coordinated_stop=True)
        self.assertTrue(obj._coordination_enabled)
        obj._reset_cross_flow_state()
        rdv = obj._resolve_rendezvous()
        self.assertIsInstance(rdv, CrossFlowRendezvous)
        self.assertEqual(rdv.active_count, 2)

    def test_active_flow_indices_default_all(self):
        obj = _make_bare_mfi(4, cross_flow_sync=True)
        self.assertEqual(obj._resolve_active_flow_indices(), {0, 1, 2, 3})

    def test_reset_reseeds_fresh_rendezvous(self):
        """Each attempt rebuilds the rendezvous (it is per-process-run, never reused)."""
        obj = _make_bare_mfi(2, cross_flow_sync=True)
        obj._reset_cross_flow_state()
        first = obj._resolve_rendezvous()
        obj._reset_cross_flow_state()
        second = obj._resolve_rendezvous()
        self.assertIsNot(first, second)


class TestCrossFlowSyncBarrierIntegration(unittest.TestCase):
    """C3: the barrier makes a fast flow's round01 wait for a slow peer's initial.

    flow_0 finishes ``initial`` instantly; flow_1's ``initial`` sleeps. Without the
    barrier, flow_0 builds ``round01`` before flow_1 has published its initial output, so
    it sees no peer output (the documented "(no output yet)" race). With the barrier,
    flow_0 blocks until flow_1 publishes, so its round01 prompt carries ``f1_init``.
    """

    PEER_TMPL = (
        "prev={{ your_prev }} | "
        "{% for idx, plan in visible_plans.items() %}peer{{ idx }}={{ plan }};{% endfor %}"
    )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build(self, cross_flow_sync):
        # flow_0 fast, flow_1 slow on its INITIAL leg.
        f0_init = _AsyncSlowInferencer("f0_init", delay=0.0)
        f0_fu = _CapturingInferencer(["f0_round01"])
        f1_init = _AsyncSlowInferencer("f1_init", delay=0.3)
        f1_fu = _CapturingInferencer(["f1_round01"])
        stop_after_round01 = lambda s, r: s.get("dynamic_step_count", 0) >= 2
        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_0",
                    "initial_inferencer": f0_init,
                    "followup_inferencer": f0_fu,
                    "followup_prompt": self.PEER_TMPL,
                    "end_condition": stop_after_round01,
                    "max_dynamic_steps": 2,
                },
                {
                    "input": "task_1",
                    "initial_inferencer": f1_init,
                    "followup_inferencer": f1_fu,
                    "followup_prompt": self.PEER_TMPL,
                    "end_condition": stop_after_round01,
                    "max_dynamic_steps": 2,
                },
            ],
            visible_flows="all",
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
            cross_flow_sync=cross_flow_sync,
        )
        return mfi, f0_fu, f1_fu

    def test_barrier_on_fast_flow_sees_slow_peer_initial(self):
        mfi, f0_fu, f1_fu = self._build(cross_flow_sync=True)
        _run_ainfer(mfi, "master")
        # flow_0's round01 followup prompt must carry flow_1's initial output.
        self.assertTrue(f0_fu.received_prompts, "flow_0 followup never ran")
        f0_round01 = f0_fu.received_prompts[0]
        self.assertIn("f1_init", f0_round01,
                      f"barrier ON: flow_0 round01 should see slow peer's initial; got {f0_round01!r}")
        self.assertNotIn("(no output yet)", f0_round01)

    def test_barrier_off_fast_flow_races_ahead(self):
        """Contrast: without the barrier the fast flow reliably misses the slow peer."""
        mfi, f0_fu, f1_fu = self._build(cross_flow_sync=False)
        _run_ainfer(mfi, "master")
        self.assertTrue(f0_fu.received_prompts, "flow_0 followup never ran")
        f0_round01 = f0_fu.received_prompts[0]
        # flow_1 (sleeping 0.3s) hasn't published its initial when fast flow_0 builds
        # round01 → peer1 renders empty/None, NOT 'f1_init'.
        self.assertNotIn("f1_init", f0_round01,
                         f"barrier OFF: expected the race (no peer output); got {f0_round01!r}")


class TestCrossFlowSyncEarlyStop(unittest.TestCase):
    """C4: an early-stopping flow departs the barrier so a longer peer never deadlocks."""

    PEER_TMPL = (
        "prev={{ your_prev }} | "
        "{% for idx, plan in visible_plans.items() %}peer{{ idx }}={{ plan }};{% endfor %}"
    )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_early_stop_does_not_deadlock_peer(self):
        # flow_0: SLOW initial, then stops (max_dynamic_steps=1 → only the initial step,
        # so it NEVER reaches a round barrier). It must leave() on completion, else flow_1
        # (waiting at the round01 barrier) hangs forever.
        f0_init = _AsyncSlowInferencer("f0_init", delay=0.2)
        f0_fu = _CapturingInferencer([])  # never used (flow_0 stops after initial)
        # flow_1: fast, runs initial + round01 + round02 (max_dynamic_steps=3).
        f1_init = _AsyncSlowInferencer("f1_init", delay=0.0)
        f1_fu = _CapturingInferencer(["f1_round01", "f1_round02"])
        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_0",
                    "initial_inferencer": f0_init,
                    "followup_inferencer": f0_fu,
                    "followup_prompt": self.PEER_TMPL,
                    "max_dynamic_steps": 1,
                },
                {
                    "input": "task_1",
                    "initial_inferencer": f1_init,
                    "followup_inferencer": f1_fu,
                    "followup_prompt": self.PEER_TMPL,
                    "max_dynamic_steps": 3,
                },
            ],
            visible_flows="all",
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
            cross_flow_sync=True,
        )
        # If the early-stop didn't depart the barrier, this would hang → wait_for raises.
        _run_ainfer(mfi, "master", timeout=10.0)
        # flow_1 ran BOTH followup rounds → it was released at gen1 (by flow_0's leave)
        # and at gen2 (solo).
        self.assertEqual(
            len(f1_fu.received_prompts), 2,
            f"flow_1 should run round01 + round02; got {f1_fu.received_prompts}")
        # flow_0 stopped after initial → its followup never ran.
        self.assertEqual(f0_fu.received_prompts, [])
        # The departed peer's output is still visible (published by its response_builder
        # before it left; captured in the barrier snapshot).
        self.assertIn("f0_init", f1_fu.received_prompts[0])


class TestCrossFlowDepartSafetyNet(unittest.TestCase):
    """C6: the BTA worker-boundary depart safety net (covers cache/cancel skip of _ainfer)."""

    class _Worker:
        pass

    def _depart(self, mfi, worker):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(mfi._cross_flow_depart_if_tagged(worker))
        finally:
            loop.close()

    def test_noop_for_untagged_worker(self):
        mfi = _make_bare_mfi(2, cross_flow_sync=True)
        mfi._reset_cross_flow_state()
        self._depart(mfi, self._Worker())  # untagged → no-op, no error
        self.assertEqual(mfi._resolve_rendezvous().active_count, 2)

    def test_departs_tagged_worker(self):
        mfi = _make_bare_mfi(2, cross_flow_sync=True)
        mfi._reset_cross_flow_state()
        w = self._Worker()
        w._cross_flow_index = 1
        self._depart(mfi, w)
        rdv = mfi._resolve_rendezvous()
        self.assertEqual(rdv.active_count, 1)
        self.assertFalse(rdv.is_active(1))

    def test_depart_idempotent_with_double_call(self):
        """The safety-net depart and the LWI-level depart both fire — must be harmless."""
        mfi = _make_bare_mfi(2, cross_flow_sync=True)
        mfi._reset_cross_flow_state()
        w = self._Worker()
        w._cross_flow_index = 0
        self._depart(mfi, w)
        self._depart(mfi, w)  # second depart (idempotent)
        self.assertEqual(mfi._resolve_rendezvous().active_count, 1)


class TestCrossFlowSyncResumeSafety(unittest.TestCase):
    """C6: a coordinated run in a POPULATED workspace (re-run/resume) never deadlocks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build(self):
        # Fresh inferencer instances each call (simulates a new process resuming).
        return MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_0",
                    "initial_inferencer": _AsyncSlowInferencer("f0_init", 0.0),
                    "followup_inferencer": _CapturingInferencer(["f0_r01"]),
                    "followup_prompt": "p={{ your_prev }}",
                    "max_dynamic_steps": 2,
                },
                {
                    "input": "task_1",
                    "initial_inferencer": _AsyncSlowInferencer("f1_init", 0.0),
                    "followup_inferencer": _CapturingInferencer(["f1_r01"]),
                    "followup_prompt": "p={{ your_prev }}",
                    "max_dynamic_steps": 2,
                },
            ],
            visible_flows="all",
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
            cross_flow_sync=True,
        )

    def test_rerun_in_populated_workspace_no_deadlock(self):
        # First run populates the workspace.
        _run_ainfer(self._build(), "master", timeout=10.0)
        # Second run over the SAME workspace: some workers may backup-resume (skip _ainfer).
        # The worker-boundary depart must still release the barrier → no hang.
        _run_ainfer(self._build(), "master", timeout=10.0)


if __name__ == "__main__":
    unittest.main()

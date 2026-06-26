"""Tests for Part C — cross-flow coordination flag semantics.

Cross-flow lock-step coordination is now IMPLEMENTED (the step barrier). This test pins
the public flag contract:
1. ``coordinated_stop`` and ``cross_flow_sync`` attributes exist with default False.
2. The async path (``_ainfer``) with coordination enabled NO LONGER raises — it runs the
   barrier transparently (proceeds to the BTA fan-out).
3. The sync path (``_infer``) with coordination enabled raises a clear "requires async"
   error (the barrier is an awaitable rendezvous needing one event loop).
4. Default (both False) preserves the existing independent execution path on both paths.

Implementation reference:
- agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_inferencer.py
- agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/cross_flow_rendezvous.py
- Plan: _docs/_plan/inferencer_architecture/INTEGRATED_cross_flow_coordination_plan.md
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
    breakdown_then_aggregate_inferencer as bta_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)


class TestCoordinationAttributes(unittest.TestCase):
    """The coordination flags exist with backward-compatible defaults."""

    def test_attributes_exist(self):
        attr_names = {a.name for a in MultiFlowInferencer.__attrs_attrs__}
        self.assertIn("coordinated_stop", attr_names)
        self.assertIn("cross_flow_sync", attr_names)

    def test_defaults_are_false(self):
        by_name = {a.name: a for a in MultiFlowInferencer.__attrs_attrs__}
        self.assertEqual(by_name["coordinated_stop"].default, False)
        self.assertEqual(by_name["cross_flow_sync"].default, False)


def _make_minimal_mfi(*, coordinated_stop=False, cross_flow_sync=False):
    """Bypass attrs init; set just what the _ainfer/_infer gates touch."""
    obj = MultiFlowInferencer.__new__(MultiFlowInferencer)
    obj.coordinated_stop = coordinated_stop
    obj.cross_flow_sync = cross_flow_sync
    obj.flow_configs = []
    obj._apply_runtime_input_propagation = MagicMock()
    obj._reset_cross_flow_state = MagicMock()
    obj._normalize_aggregator_output = lambda x: x
    obj._extract_dispatch_state = lambda x: None
    obj._maybe_strip_response = lambda x: x
    return obj


class TestSyncPathRequiresAsync(unittest.TestCase):
    """Sync _infer with coordination enabled raises 'requires async' (loud, no silent drop)."""

    def test_infer_raises_for_coordinated_stop(self):
        mfi = _make_minimal_mfi(coordinated_stop=True)
        with self.assertRaises(NotImplementedError) as ctx:
            mfi._infer("input")
        msg = str(ctx.exception)
        self.assertIn("async", msg.lower())
        self.assertIn("ainfer", msg)

    def test_infer_raises_for_cross_flow_sync(self):
        mfi = _make_minimal_mfi(cross_flow_sync=True)
        with self.assertRaises(NotImplementedError):
            mfi._infer("input")

    def test_infer_ok_when_disabled(self):
        mfi = _make_minimal_mfi()  # both False
        mfi._infer = MultiFlowInferencer._infer.__get__(mfi)  # ensure real method
        original = bta_module.BreakdownThenAggregateInferencer._infer
        bta_module.BreakdownThenAggregateInferencer._infer = lambda self, x, **kw: "ok"
        try:
            self.assertEqual(mfi._infer("input"), "ok")
        finally:
            bta_module.BreakdownThenAggregateInferencer._infer = original


class TestAsyncPathRunsBarrier(unittest.TestCase):
    """Async _ainfer with coordination enabled NO LONGER raises — it proceeds to BTA."""

    def _run_ainfer(self, mfi):
        async def _stub_bta_ainfer(self, inference_input, **kwargs):
            return "stubbed_result"

        original = bta_module.BreakdownThenAggregateInferencer._ainfer
        bta_module.BreakdownThenAggregateInferencer._ainfer = _stub_bta_ainfer
        try:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(mfi._ainfer("input"))
            finally:
                loop.close()
        finally:
            bta_module.BreakdownThenAggregateInferencer._ainfer = original

    def test_ainfer_does_not_raise_for_coordinated_stop(self):
        mfi = _make_minimal_mfi(coordinated_stop=True)
        self.assertEqual(self._run_ainfer(mfi), "stubbed_result")

    def test_ainfer_does_not_raise_for_cross_flow_sync(self):
        mfi = _make_minimal_mfi(cross_flow_sync=True)
        self.assertEqual(self._run_ainfer(mfi), "stubbed_result")

    def test_ainfer_default_path_unchanged(self):
        mfi = _make_minimal_mfi()  # both False
        self.assertEqual(self._run_ainfer(mfi), "stubbed_result")


if __name__ == "__main__":
    unittest.main()

"""Tests for Part C — coordinated_stop scaffold (Phase C1).

The full lock-step implementation is deferred to PR #2. This test verifies the
scaffold:
1. The attribute exists with default False
2. Setting True raises NotImplementedError loudly (no silent fallback)
3. Default (False) preserves existing independent execution path

Implementation reference:
- agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_inferencer.py
- Plan: _docs/_plans/mfdual_hygiene_INTEGRATED_plan.md (Part C)
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)


class TestCoordinatedStopAttribute(unittest.TestCase):
    """Verify the coordinated_stop attribute exists with the right default."""

    def test_attribute_exists_with_default_false(self):
        """Attribute must exist on the class for YAML configurations to reference."""
        # Use __new__ to bypass attrs init
        obj = MultiFlowInferencer.__new__(MultiFlowInferencer)
        # Default value
        # The attrs default isn't applied via __new__, but we can check class attrs
        self.assertTrue(
            hasattr(MultiFlowInferencer, "__attrs_attrs__"),
            "MultiFlowInferencer should be an attrs class",
        )
        attr_names = {a.name for a in MultiFlowInferencer.__attrs_attrs__}
        self.assertIn("coordinated_stop", attr_names,
                      "coordinated_stop attribute missing")

    def test_default_value_is_false(self):
        attr = next(
            a for a in MultiFlowInferencer.__attrs_attrs__
            if a.name == "coordinated_stop"
        )
        self.assertEqual(attr.default, False,
                         "Default for coordinated_stop must be False (backward compat)")


class TestCoordinatedStopLoudFailure(unittest.TestCase):
    """Verify NotImplementedError is raised when coordinated_stop=True.

    Uses sync _infer to test the gate; an isolated asyncio.run() also tests
    _ainfer (without polluting the global event loop in subsequent tests).
    """

    def _make_minimal_mfi(self, coordinated_stop):
        """Bypass attrs init, set only attributes we need."""
        obj = MultiFlowInferencer.__new__(MultiFlowInferencer)
        obj.coordinated_stop = coordinated_stop
        # Other attrs accessed in _ainfer before our gate
        obj.flow_configs = []
        return obj

    def test_ainfer_raises_when_coordinated_stop_true(self):
        """Use asyncio.new_event_loop() to avoid polluting test-runner loop."""
        mfi = self._make_minimal_mfi(coordinated_stop=True)

        async def _check():
            with self.assertRaises(NotImplementedError) as ctx:
                await mfi._ainfer("input")
            msg = str(ctx.exception)
            self.assertIn("coordinated_stop", msg)
            self.assertIn("§C", msg)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_check())
        finally:
            loop.close()

    def test_infer_raises_when_coordinated_stop_true(self):
        mfi = self._make_minimal_mfi(coordinated_stop=True)
        with self.assertRaises(NotImplementedError) as ctx:
            mfi._infer("input")
        msg = str(ctx.exception)
        self.assertIn("coordinated_stop", msg)


class TestCoordinatedStopDefaultPath(unittest.TestCase):
    """Verify default (False) preserves the existing execution path."""

    def test_ainfer_does_not_raise_when_coordinated_stop_false(self):
        """When coordinated_stop=False, the gate is bypassed (proceeds to BTA path).

        Uses isolated event loop to avoid polluting test-runner loop.
        """
        obj = MultiFlowInferencer.__new__(MultiFlowInferencer)
        obj.coordinated_stop = False
        obj.flow_configs = []
        obj._apply_runtime_input_propagation = MagicMock()
        obj._reset_cross_flow_state = MagicMock()
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
            breakdown_then_aggregate_inferencer as bta_module,
        )

        async def _stub_bta_ainfer(self, inference_input, **kwargs):
            return "stubbed_result"

        original = bta_module.BreakdownThenAggregateInferencer._ainfer
        bta_module.BreakdownThenAggregateInferencer._ainfer = _stub_bta_ainfer
        try:
            obj._normalize_aggregator_output = lambda x: x
            obj._extract_dispatch_state = lambda x: None
            obj._maybe_strip_response = lambda x: x

            async def _run():
                return await obj._ainfer("input")

            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_run())
            finally:
                loop.close()
            self.assertEqual(result, "stubbed_result")
        finally:
            bta_module.BreakdownThenAggregateInferencer._ainfer = original


if __name__ == "__main__":
    unittest.main()

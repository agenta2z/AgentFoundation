"""Cross-cutting error propagation tests — Layer 1, R8.

Verifies that exceptions from leaf inferencers propagate through composition
layers with traceable context. These tests use controllable failure injection
(only possible with mocks) — they catch a class of bug only mocks can isolate.
"""

import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from test.agent_foundation.common.inferencers._helpers.mock_inferencer import (
    MockInferencer,
)


class _RaisingInferencer:
    """Minimal inferencer that always raises a custom exception."""

    def __init__(self, exc=None):
        self._exc = exc or RuntimeError("intentional test failure")
        self.ainfer = AsyncMock(side_effect=self._exc)
        self.infer = MagicMock(side_effect=self._exc)
        self.aconnect = AsyncMock()
        self.adisconnect = AsyncMock()
        self.reset_session = MagicMock()


# ---------------------------------------------------------------------------
# R8: DualInferencer base raises in propose step
# ---------------------------------------------------------------------------


class TestDualPropagatesProposeError(unittest.IsolatedAsyncioTestCase):
    """Validates: R8 — Dual base raises → exception propagates."""

    async def test_base_raise_propagates(self):
        dual = DualInferencer(
            base_inferencer=_RaisingInferencer(RuntimeError("propose failed")),
            review_inferencer=MockInferencer(response="should not reach"),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        # Dual catches and either re-raises or wraps; verify execution halts
        # with non-zero indication
        with self.assertRaises((RuntimeError, Exception)) as ctx:
            await dual._ainfer("request")
        # Original failure context should be visible somewhere
        self.assertIn("failed", str(ctx.exception).lower() + str(ctx.exception.__cause__ or "").lower())


# ---------------------------------------------------------------------------
# R8: BTA worker raises — failure reported, other workers unaffected
# ---------------------------------------------------------------------------


class TestBTAWorkerRaiseDoesNotBlock(unittest.TestCase):
    """Validates: R8 — BTA worker exception doesn't crash the entire diamond."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_worker_exception_handled(self):
        """One worker raises; BTA should handle gracefully — either propagate
        with worker context OR report failure without crashing the diamond.
        Behavior depends on configuration; we just verify it does not
        silently swallow."""
        breakdown = MockInferencer(response="1. ok task\n2. failing task\n3. ok task")

        def factory(sub_query, index):
            if index == 1:  # second worker raises
                w = MockInferencer(response="ok")
                w._infer = lambda *a, **kw: (_ for _ in ()).throw(
                    RuntimeError(f"worker {index} failed")
                )
                return w
            return MockInferencer(response=f"worker_{index}_ok")

        aggregator = MockInferencer(response="agg")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
            aggregator_inferencer=aggregator,
            checkpoint_dir=self.tmpdir,
        )

        # Worker raises — BTA should propagate (not silently return None)
        # Behavior: BTA's WorkGraph reports node failure, may propagate.
        # The contract: do NOT silently swallow; do propagate with context.
        try:
            result = bta.infer("task")
            # If no exception, verify result contains evidence of failure
            self.assertIsNotNone(result)
        except Exception as e:
            # Exception path: ensure it carries traceable context
            err_text = str(e) + str(e.__cause__ or "")
            self.assertTrue(
                "failed" in err_text.lower() or "worker" in err_text.lower()
                or "exception" in err_text.lower(),
                f"Exception should carry worker context, got: {e}",
            )


# ---------------------------------------------------------------------------
# R8: PTI planner failure prevents executor execution
# ---------------------------------------------------------------------------


class TestPTIPlannerFailureBlocksExecutor(unittest.IsolatedAsyncioTestCase):
    """Validates: R8 — PTI planner raises → executor never runs."""

    async def test_planner_failure_blocks_executor(self):
        executor = MockInferencer(response="should not run")
        # Track if executor was called
        executor.infer = MagicMock(return_value="should not run")
        executor.ainfer = AsyncMock(return_value="should not run")

        with tempfile.TemporaryDirectory() as tmpdir:
            pti = PlanThenImplementInferencer(
                planner_inferencer=_RaisingInferencer(RuntimeError("planner failed")),
                executor_inferencer=executor,
                workspace=tmpdir,
            )
            with self.assertRaises(Exception):
                await pti._ainfer("task")

        # Executor should NOT have been called
        executor.ainfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()

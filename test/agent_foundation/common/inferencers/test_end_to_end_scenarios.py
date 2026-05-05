"""End-to-end multi-inferencer composition scenarios — Layer 1, R10.

These tests verify that **realistic composition narratives** survive the
entire inferencer stack — distinct from per-inferencer unit tests because
they exercise inferencers wired together with realistic state shapes.

The flagship test is the **hierarchical composition** (R10.6):
MultiFlowInferencer where each flow is a PlanThenImplementInferencer with
DualInferencer children at planner and executor — a 4-layer stack.
"""

import json
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    DualInferencerResponse,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock(response=None, side_effect=None):
    inf = MagicMock()
    inf.ainfer = AsyncMock(
        side_effect=side_effect,
        return_value=response if side_effect is None else None,
    )
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()
    inf.reset_session = MagicMock()
    return inf


def _review_json(approved=True, severity="COSMETIC"):
    return f'```json\n{json.dumps({"approved": approved, "severity": severity, "issues": [], "reasoning": "ok"})}\n```'


# ---------------------------------------------------------------------------
# R10.1: Code review e2e (Dual)
# ---------------------------------------------------------------------------


class TestCodeReviewE2E(unittest.IsolatedAsyncioTestCase):
    """Validates: R10.1 — DualInferencer code-review consensus loop."""

    async def test_code_review_converges_after_one_fix(self):
        """proposer→reviewer→fixer→reviewer (approve) — final result reflects fix."""
        proposal = "def add(a, b): return a + b  # initial draft"
        fixed = "def add(a: int, b: int) -> int:\n    return a + b  # with type hints"

        dual = DualInferencer(
            base_inferencer=_make_mock(proposal),
            review_inferencer=_make_mock(side_effect=[
                _review_json(approved=False, severity="MINOR"),
                _review_json(approved=True),
            ]),
            fixer_inferencer=_make_mock(fixed),
            consensus_config=ConsensusConfig(
                max_iterations=3,
                consensus_threshold=Severity.COSMETIC,
            ),
        )

        result = await dual._ainfer("write add(a, b)")
        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        # 2 review iterations
        self.assertEqual(result.total_iterations, 2)


# ---------------------------------------------------------------------------
# R10.2: Research-then-implement e2e (PTI)
# ---------------------------------------------------------------------------


class TestResearchThenImplementE2E(unittest.IsolatedAsyncioTestCase):
    """Validates: R10.2 — PTI plan→executor data flow with realistic narrative."""

    async def test_executor_receives_planner_output(self):
        plan_text = (
            "## Research Plan\n"
            "1. Survey existing libs\n"
            "2. Compare API ergonomics\n"
            "3. Recommend top choice\n"
        )
        impl_text = "Survey complete. Recommended: requests."

        captured_executor_input = []

        async def executor_capture(prompt, *args, **kwargs):
            captured_executor_input.append(str(prompt))
            return impl_text

        executor = MagicMock()
        executor.ainfer = AsyncMock(side_effect=executor_capture)
        executor.aconnect = AsyncMock()
        executor.adisconnect = AsyncMock()
        executor.reset_session = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            pti = PlanThenImplementInferencer(
                planner_inferencer=_make_mock(plan_text),
                executor_inferencer=executor,
                workspace=tmpdir,
                planner_outputs_plan_to_file=False,  # inline plan in executor input
            )

            result = await pti._ainfer("Choose an HTTP lib")

        self.assertIsInstance(result, PlanThenImplementResponse)
        # Executor must have seen the plan content in its input
        self.assertEqual(len(captured_executor_input), 1)
        self.assertIn("Survey existing", captured_executor_input[0])


# ---------------------------------------------------------------------------
# R10.3: Parallel decomposition e2e (BTA)
# ---------------------------------------------------------------------------


class TestParallelDecompositionE2E(unittest.TestCase):
    """Validates: R10.3 — BTA splits → workers run → aggregator merges."""

    def test_three_subtasks_three_workers_aggregated(self):
        """Realistic 3-subtask decomposition; aggregator receives all 3."""
        from test.agent_foundation.common.inferencers._helpers.mock_inferencer import (
            MockInferencer,
        )

        breakdown = MockInferencer(
            response="1. Database schema\n2. API endpoints\n3. Auth middleware"
        )

        agg_inputs = []

        def agg_fn(inp):
            agg_inputs.append(str(inp))
            return "Combined design document"

        aggregator = MockInferencer(response=agg_fn)

        # Each worker returns a deterministic, distinct result
        def worker_factory(sub_query, index):
            return MockInferencer(response=f"WORKER_{index}_RESULT_for_{sub_query}")

        with tempfile.TemporaryDirectory() as tmpdir:
            bta = BreakdownThenAggregateInferencer(
                breakdown_inferencer=breakdown,
                worker_factory=worker_factory,
                aggregator_inferencer=aggregator,
                checkpoint_dir=tmpdir,
            )

            result = bta.infer("design a REST API for e-commerce")

        # Aggregator received the 3 worker results in its prompt
        self.assertEqual(len(agg_inputs), 1)
        agg_prompt = agg_inputs[0]
        self.assertIn("WORKER_0_RESULT", agg_prompt)
        self.assertIn("WORKER_1_RESULT", agg_prompt)
        self.assertIn("WORKER_2_RESULT", agg_prompt)
        self.assertEqual(result, "Combined design document")


# ---------------------------------------------------------------------------
# R10.4: Multi-flow research e2e (MFI)
# ---------------------------------------------------------------------------


class TestMultiFlowResearchE2E(unittest.TestCase):
    """Validates: R10.4 — 3 parallel research flows + synthesis."""

    def test_three_flows_independent_termination(self):
        """3 flows with different end_conditions; aggregator sees all 3."""
        from test.agent_foundation.common.inferencers._helpers.mock_inferencer import (
            MockInferencer,
        )

        # Each flow's initial inferencer returns a unique research finding
        flow0_init = MockInferencer(response="Flow0 finding: auth")
        flow1_init = MockInferencer(response="Flow1 finding: rate-limiting")
        flow2_init = MockInferencer(response="Flow2 finding: privacy")
        # Followup inferencers required even if end_condition stops first
        flow0_followup = MockInferencer(response="should not be called")
        flow1_followup = MockInferencer(response="should not be called")
        flow2_followup = MockInferencer(response="should not be called")

        agg_inputs = []

        def agg_fn(inp):
            agg_inputs.append(str(inp))
            return "Combined research synthesis"

        aggregator = MockInferencer(response=agg_fn)

        # Each flow stops after 1 step via end_condition
        end_after_first = lambda s, r: True

        with tempfile.TemporaryDirectory() as tmpdir:
            mfi = MultiFlowInferencer(
                flow_configs=[
                    {"input": "research auth",
                     "initial_inferencer": flow0_init,
                     "followup_inferencer": flow0_followup,
                     "end_condition": end_after_first, "max_dynamic_steps": 5},
                    {"input": "research rate-limiting",
                     "initial_inferencer": flow1_init,
                     "followup_inferencer": flow1_followup,
                     "end_condition": end_after_first, "max_dynamic_steps": 5},
                    {"input": "research privacy",
                     "initial_inferencer": flow2_init,
                     "followup_inferencer": flow2_followup,
                     "end_condition": end_after_first, "max_dynamic_steps": 5},
                ],
                aggregator_inferencer=aggregator,
                checkpoint_dir=tmpdir,
            )
            result = mfi.infer("security architecture overview")

        self.assertEqual(result, "Combined research synthesis")
        # Aggregator must have seen results from all 3 flows
        self.assertEqual(len(agg_inputs), 1)
        agg_input = agg_inputs[0]
        self.assertIn("auth", agg_input)
        self.assertIn("rate-limiting", agg_input)
        self.assertIn("privacy", agg_input)


# ---------------------------------------------------------------------------
# R10.6 ⭐ HIERARCHICAL COMPOSITION — the unique stress-test
# MFI of PTI of Dual: 4-layer stack
# ---------------------------------------------------------------------------


class TestHierarchicalComposition(unittest.TestCase):
    """⭐ R10.6 — The flagship: MultiFlowInferencer where each flow is a
    PlanThenImplementInferencer with DualInferencer children at planner
    AND executor. 4 layers must compose correctly and propagate results
    upward.

    Stack (deepest → shallowest):
    Dual children → PTI → MFI flows → MFI aggregator
    """

    def test_mfi_of_pti_of_dual_executes_and_propagates(self):
        """Build the entire stack and verify execution + result propagation."""

        # Helper to construct one PTI with Dual children
        def _build_pti_with_dual_children(label):
            # Dual planner: proposer + reviewer (auto-approve immediately)
            dual_planner = DualInferencer(
                base_inferencer=_make_mock(f"PLAN_{label}: do X then Y"),
                review_inferencer=_make_mock(_review_json(approved=True)),
                consensus_config=ConsensusConfig(
                    max_iterations=2,
                    consensus_threshold=Severity.COSMETIC,
                ),
            )
            # Dual executor
            dual_executor = DualInferencer(
                base_inferencer=_make_mock(f"IMPL_{label}: complete"),
                review_inferencer=_make_mock(_review_json(approved=True)),
                consensus_config=ConsensusConfig(
                    max_iterations=2,
                    consensus_threshold=Severity.COSMETIC,
                ),
            )
            return PlanThenImplementInferencer(
                planner_inferencer=dual_planner,
                executor_inferencer=dual_executor,
                planner_outputs_plan_to_file=False,
            )

        # Build 2 PTIs (one per flow); MFI's dynamic-mode LWI also needs
        # followup_inferencer (not used because end_condition stops at step 0)
        from test.agent_foundation.common.inferencers._helpers.mock_inferencer import (
            MockInferencer,
        )
        pti_alpha = _build_pti_with_dual_children("ALPHA")
        pti_beta = _build_pti_with_dual_children("BETA")
        followup_alpha = MockInferencer(response="alpha-followup-not-called")
        followup_beta = MockInferencer(response="beta-followup-not-called")

        # MFI aggregator captures inputs
        agg_inputs = []

        def agg_fn(inp):
            agg_inputs.append(str(inp))
            return "TOP_LEVEL_SYNTHESIS"

        aggregator = MockInferencer(response=agg_fn)

        end_after_first = lambda s, r: True

        with tempfile.TemporaryDirectory() as tmpdir:
            mfi = MultiFlowInferencer(
                flow_configs=[
                    {
                        "input": "build feature alpha",
                        "initial_inferencer": pti_alpha,
                        "followup_inferencer": followup_alpha,
                        "end_condition": end_after_first,
                        "max_dynamic_steps": 5,
                    },
                    {
                        "input": "build feature beta",
                        "initial_inferencer": pti_beta,
                        "followup_inferencer": followup_beta,
                        "end_condition": end_after_first,
                        "max_dynamic_steps": 5,
                    },
                ],
                aggregator_inferencer=aggregator,
                checkpoint_dir=tmpdir,
            )

            result = mfi.infer("build product P")

        # ───────────────────────────────────────────────────────────
        # Top-level: aggregator received synthesized output
        # ───────────────────────────────────────────────────────────
        self.assertEqual(result, "TOP_LEVEL_SYNTHESIS")

        # Aggregator must have been called once with both PTI results
        self.assertEqual(len(agg_inputs), 1)
        agg_prompt = agg_inputs[0]

        # Both PTIs' implementation outputs should have propagated up
        self.assertIn("IMPL_ALPHA", agg_prompt,
                      "ALPHA implementation must propagate up through PTI to MFI")
        self.assertIn("IMPL_BETA", agg_prompt,
                      "BETA implementation must propagate up through PTI to MFI")


if __name__ == "__main__":
    unittest.main()

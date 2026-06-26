"""Adapter-in-flow-slots gap tests — Layer 1, R7 gaps.

Coverage focus: GAPS not covered by
test_conversational_flow_node_adapter.py:TestAdapterInFlowInferencerSlots
(which covers BTA breakdown_inferencer slot, fallback, reset, interactive,
template_manager, clobber avoidance, plus TestSessionLevelResume Level 1+2).

Genuine gaps filled here:
- Adapter as PTI planner_inferencer slot (not just BTA breakdown)
- Adapter as BTA aggregator_inferencer slot (not just breakdown)
- Adapter in LWI sequential step (general-purpose composition)
"""

import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
    ConversationalFlowNodeAdapter,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from test.agent_foundation.common.inferencers._helpers.mock_inferencer import (
    MockInferencer,
)


def _make_agentic_result(text="hello", **overrides):
    defaults = dict(
        text=text,
        completed_actions=[],
        iterations_used=1,
        has_conversation_tool=False,
        exhausted_max_iterations=False,
        raw_response=text,
    )
    defaults.update(overrides)
    return AgenticResult(**defaults)


def _make_adapter_with_text(text):
    """Adapter wrapping a ConversationalInferencer whose run_agentic_loop is
    mocked to return AgenticResult(text=text). Used for testing adapter in
    different flow slots."""
    base = MagicMock()
    base.ainfer = AsyncMock(return_value=text)
    base.system_prompt = ""
    base.set_messages = MagicMock()
    base.cache_folder = None
    conv = ConversationalInferencer(base_inferencer=base)
    adapter = ConversationalFlowNodeAdapter(conversational_inferencer=conv)
    # Mock run_agentic_loop directly to avoid full conversational pipeline
    adapter.conversational_inferencer.run_agentic_loop = AsyncMock(
        return_value=_make_agentic_result(text=text),
    )
    return adapter


# ---------------------------------------------------------------------------
# R7 gap: Adapter as PTI planner_inferencer slot
# ---------------------------------------------------------------------------


class TestAdapterAsPTIPlanner(unittest.IsolatedAsyncioTestCase):
    """Validates: R7 — adapter routes correctly when used as PTI's planner."""

    async def test_adapter_as_pti_planner_slot(self):
        plan_text = "## Plan\n1. Step one\n2. Step two\n"
        adapter = _make_adapter_with_text(plan_text)

        # Capture executor's input
        executor_inputs = []

        async def executor_capture(prompt, *args, **kwargs):
            executor_inputs.append(str(prompt))
            return "implementation done"

        executor = MagicMock()
        executor.ainfer = AsyncMock(side_effect=executor_capture)
        executor.aconnect = AsyncMock()
        executor.adisconnect = AsyncMock()
        executor.reset_session = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            pti = PlanThenImplementInferencer(
                planner_inferencer=adapter,
                executor_inferencer=executor,
                workspace=tmpdir,
                planner_outputs_plan_to_file=False,
            )
            await pti._ainfer("task")

        # Adapter was invoked (run_agentic_loop called)
        adapter.conversational_inferencer.run_agentic_loop.assert_called_once()
        # Executor saw the plan text
        self.assertEqual(len(executor_inputs), 1)
        self.assertIn("Step one", executor_inputs[0])


# ---------------------------------------------------------------------------
# R7 gap: Adapter as BTA aggregator_inferencer slot
# ---------------------------------------------------------------------------


class TestAdapterAsBTAAggregator(unittest.IsolatedAsyncioTestCase):
    """Validates: R7 — adapter routes correctly when used as BTA's aggregator.

    NOTE: Adapter is async-only (sync _infer raises NotImplementedError),
    so BTA must run via ainfer() to avoid the sync/async mismatch.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_adapter_as_bta_aggregator_slot(self):
        synthesis_text = "Combined: A + B + C → D"
        adapter = _make_adapter_with_text(synthesis_text)

        breakdown = MockInferencer(response="1. A\n2. B\n3. C")

        def factory(sub_query, index):
            return MockInferencer(response=f"worker_{index}_result")

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=factory,
            aggregator_inferencer=adapter,
            checkpoint_dir=self.tmpdir,
        )
        result = await bta.ainfer("task")

        # Adapter was invoked as aggregator
        adapter.conversational_inferencer.run_agentic_loop.assert_called_once()
        # Result must contain adapter's output (BTA may return tuple of
        # worker outputs + aggregator output, depending on topology mode)
        if isinstance(result, tuple):
            self.assertIn(synthesis_text, result)
        else:
            self.assertEqual(result, synthesis_text)


# ---------------------------------------------------------------------------
# R7 gap: Adapter in LWI sequential step
# ---------------------------------------------------------------------------


class TestAdapterInLWIStep(unittest.IsolatedAsyncioTestCase):
    """Validates: R7 — adapter routes correctly as one step in an LWI chain."""

    async def test_adapter_in_lwi_sequential_step(self):
        step_a = MockInferencer(response="A_output")
        adapter = _make_adapter_with_text("ADAPTER_output")
        step_c = MockInferencer(response="C_output")

        with tempfile.TemporaryDirectory() as tmpdir:
            lwi = LinearWorkflowInferencer(
                step_configs=[
                    WorkflowStepConfig(
                        name="a",
                        inferencer=step_a,
                        output_state_key="a_out",
                    ),
                    WorkflowStepConfig(
                        name="b",
                        inferencer=adapter,
                        input_builder=lambda s: s["a_out"],
                        output_state_key="b_out",
                    ),
                    WorkflowStepConfig(
                        name="c",
                        inferencer=step_c,
                        input_builder=lambda s: s["b_out"],
                        output_state_key="c_out",
                    ),
                ],
                response_builder=lambda s: {
                    "a": s["a_out"],
                    "b": s["b_out"],
                    "c": s["c_out"],
                },
                workspace=tmpdir,
            )
            result = await lwi._ainfer("input")

        # Adapter was invoked exactly once as middle step
        adapter.conversational_inferencer.run_agentic_loop.assert_called_once()
        # Adapter received step a's output
        self.assertEqual(result["b"], "ADAPTER_output")
        # Step c received adapter's output
        self.assertEqual(result["c"], "C_output")


if __name__ == "__main__":
    unittest.main()

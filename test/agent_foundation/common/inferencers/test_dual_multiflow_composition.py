"""Composition integration tests — DualInferencer with MultiFlowInferencer as base.

Hand-wired form:

    dual = DualInferencer(
        base_inferencer  = multi_flow,    # MultiFlowInferencer instance
        review_inferencer= reviewer_cli,  # independent
        fixer_inferencer = fixer_cli,     # independent
        ...
    )
    result = await dual.ainfer(prompt)

This file deliberately exercises the *hand-wired* composition. The convenience
class ``MultiFlowDualInferencer`` is tested separately in
``test_multi_flow_dual_inferencer.py``.

Tier ordering (simple → sophisticated):
  T1 — Wiring smoke: DualInferencer accepts MultiFlow as base; reviewer/fixer
       are independent inferencers.
  T2 — Happy path N=2: consensus on first iteration; consensus after fix.
  T3 — N=3 scaling: three flows + reviewer + fixer end-to-end.
"""

import json
import re
import shutil
import tempfile
import unittest

from attr import attrib, attrs

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    DualInferencerResponse,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


# ---------------------------------------------------------------------------
# Mock CLI inferencers — emit deterministic structured outputs.
# ---------------------------------------------------------------------------


@attrs
class _ScriptedInferencer(InferencerBase):
    """An InferencerBase that returns canned responses in order; records calls."""

    _script = attrib(factory=list)
    _calls = attrib(factory=list, init=False)
    _idx = attrib(default=0, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._calls.append(str(inference_input))
        if self._idx < len(self._script):
            r = self._script[self._idx]
        else:
            r = f"<overflow_{self._idx}>"
        self._idx += 1
        return r

    @property
    def call_count(self):
        return len(self._calls)


def _flow_output(plan_text: str, decision: str = "stop") -> str:
    """Format a single MultiFlow worker step's output."""
    return f"<Plan>\n{plan_text}\n</Plan>\n<Decision>{decision}</Decision>"


def _aggregator_output(integrated_plan: str) -> str:
    return f"<FinalPlan>\n{integrated_plan}\n</FinalPlan>"


def _review_approve(message: str = "OK") -> str:
    return (
        "```json\n"
        + json.dumps(
            {
                "approved": True,
                "severity": "MINOR",
                "issues": [],
                "reasoning": message,
            },
            indent=2,
        )
        + "\n```"
    )


def _review_reject(issue_desc: str = "needs work") -> str:
    return (
        "```json\n"
        + json.dumps(
            {
                "approved": False,
                "severity": "MAJOR",
                "issues": [
                    {
                        "severity": "MAJOR",
                        "category": "logic",
                        "description": issue_desc,
                        "location": "n/a",
                        "suggestion": "fix it",
                    }
                ],
                "reasoning": "rejected",
            },
            indent=2,
        )
        + "\n```"
    )


def _parse_decision_tag(s: str):
    m = re.search(r"<Decision>([\s\S]*?)</Decision>", s)
    return (m.group(1).strip() if m else None)


def _parse_finalplan_tag(s: str) -> str:
    m = re.search(r"<FinalPlan>([\s\S]*?)</FinalPlan>", s)
    return (m.group(1).strip() if m else s)


# ---------------------------------------------------------------------------
# Helpers — build a MultiFlow "propose engine" for use as DualInferencer.base
# ---------------------------------------------------------------------------


def _build_multi_flow(
    *,
    n_flows: int,
    flow_outputs,    # list of lists: flow_outputs[i] = [step0, step1, ...]
    aggregator_output: str,
    workspace_dir: str,
    visible_flows: str = "all",
):
    """Construct a MultiFlow with N flows, each scripted to emit deterministic
    plans, and an aggregator that produces a single integrated plan."""
    assert len(flow_outputs) == n_flows

    def _end_at_count(count):
        return lambda s, r: s.get("dynamic_step_count", 0) >= count

    flow_configs = []
    for i in range(n_flows):
        steps = flow_outputs[i]
        # initial_inferencer emits step 0; followup_inferencer emits steps 1..N-1
        init = _ScriptedInferencer(script=[steps[0]])
        if len(steps) > 1:
            followup = _ScriptedInferencer(script=steps[1:])
        else:
            followup = _ScriptedInferencer(script=[])
        flow_configs.append(
            {
                "input": f"task_{i}",
                "initial_inferencer": init,
                "followup_inferencer": followup,
                "end_condition": _end_at_count(len(steps)),
                "max_dynamic_steps": max(len(steps), 1),
            }
        )

    aggregator = _ScriptedInferencer(script=[aggregator_output])

    multi_flow = MultiFlowInferencer(
        flow_configs=flow_configs,
        visible_flows=visible_flows,
        aggregator_inferencer=aggregator,
        aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
        response_parser=_parse_finalplan_tag,
        checkpoint_dir=workspace_dir,
    )
    return multi_flow, aggregator


# ===========================================================================
# T1 — Wiring smoke
# ===========================================================================


class TestT1WiringSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dual_accepts_multi_flow_as_base_and_independent_review_fix(self):
        flow_outputs = [
            [_flow_output("plan_0_v0", "stop")],
            [_flow_output("plan_1_v0", "stop")],
        ]
        multi_flow, _ = _build_multi_flow(
            n_flows=2,
            flow_outputs=flow_outputs,
            aggregator_output=_aggregator_output("integrated_plan"),
            workspace_dir=self.tmp,
        )
        reviewer = _ScriptedInferencer(script=[_review_approve()])
        fixer = _ScriptedInferencer(script=[])  # not invoked

        dual = DualInferencer(
            base_inferencer=multi_flow,
            review_inferencer=reviewer,
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        # Smoke checks before running
        self.assertIs(dual.base_inferencer, multi_flow)
        self.assertIs(dual.review_inferencer, reviewer)
        self.assertIs(dual.fixer_inferencer, fixer)
        # Reviewer/fixer are independent — neither equal to MultiFlow's flows or aggregator
        self.assertIsNot(dual.review_inferencer, dual.base_inferencer)
        self.assertIsNot(dual.fixer_inferencer, dual.base_inferencer)


# ===========================================================================
# T2 — Happy path, N=2
# ===========================================================================


class TestT2HappyPathN2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_consensus_on_first_iteration(self):
        flow_outputs = [
            [_flow_output("plan_a", "stop")],
            [_flow_output("plan_b", "stop")],
        ]
        multi_flow, aggregator = _build_multi_flow(
            n_flows=2,
            flow_outputs=flow_outputs,
            aggregator_output=_aggregator_output("integrated_plan"),
            workspace_dir=self.tmp,
        )
        reviewer = _ScriptedInferencer(script=[_review_approve("looks good")])
        fixer = _ScriptedInferencer(script=[])  # never called on first-iter consensus

        dual = DualInferencer(
            base_inferencer=multi_flow,
            review_inferencer=reviewer,
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(max_iterations=2),
        )
        result = dual.infer("master task")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        # base_output_str should have been the integrated plan (whitespace-tolerant)
        self.assertEqual(str(result.base_response).strip(), "integrated_plan")
        # reviewer called exactly once; fixer not invoked
        self.assertEqual(reviewer.call_count, 1)
        self.assertEqual(fixer.call_count, 0)
        # Aggregator called exactly once
        self.assertEqual(aggregator.call_count, 1)

    def test_consensus_after_fix(self):
        flow_outputs = [
            [_flow_output("plan_a", "stop")],
            [_flow_output("plan_b", "stop")],
        ]
        multi_flow, aggregator = _build_multi_flow(
            n_flows=2,
            flow_outputs=flow_outputs,
            aggregator_output=_aggregator_output("integrated_v1"),
            workspace_dir=self.tmp,
        )
        # Reviewer rejects round 1, approves round 2
        reviewer = _ScriptedInferencer(script=[_review_reject(), _review_approve()])
        # Fixer produces an improved plan
        fixer = _ScriptedInferencer(script=["integrated_v2"])

        dual = DualInferencer(
            base_inferencer=multi_flow,
            review_inferencer=reviewer,
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(max_iterations=3),
        )
        result = dual.infer("master")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        self.assertEqual(reviewer.call_count, 2)
        self.assertEqual(fixer.call_count, 1)
        # Aggregator called once (only one MultiFlow propose)
        self.assertEqual(aggregator.call_count, 1)
        # Final plan is what fixer produced (whitespace-tolerant)
        self.assertEqual(str(result.base_response).strip(), "integrated_v2")

    def test_no_consensus_max_iterations(self):
        flow_outputs = [
            [_flow_output("plan_a", "stop")],
            [_flow_output("plan_b", "stop")],
        ]
        multi_flow, _ = _build_multi_flow(
            n_flows=2,
            flow_outputs=flow_outputs,
            aggregator_output=_aggregator_output("integrated"),
            workspace_dir=self.tmp,
        )
        # Reviewer rejects every iteration
        reviewer = _ScriptedInferencer(
            script=[_review_reject(), _review_reject(), _review_reject()]
        )
        fixer = _ScriptedInferencer(
            script=["fix_v1", "fix_v2", "fix_v3"]
        )

        dual = DualInferencer(
            base_inferencer=multi_flow,
            review_inferencer=reviewer,
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(
                max_iterations=2, max_consensus_attempts=1,
            ),
        )
        result = dual.infer("master")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertFalse(result.consensus_achieved)
        # Reviewer called max_iterations times; fixer max_iterations - 1 times
        # (loop_back_to fires until limit; final review still runs)
        self.assertGreaterEqual(reviewer.call_count, 2)
        self.assertGreaterEqual(fixer.call_count, 1)


# ===========================================================================
# T3 — N=3 scaling
# ===========================================================================


class TestT3NFlowScaling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_n3_consensus_first_iteration(self):
        flow_outputs = [
            [_flow_output("plan_0", "stop")],
            [_flow_output("plan_1", "stop")],
            [_flow_output("plan_2", "stop")],
        ]
        multi_flow, aggregator = _build_multi_flow(
            n_flows=3,
            flow_outputs=flow_outputs,
            aggregator_output=_aggregator_output("integrated_3way"),
            workspace_dir=self.tmp,
        )
        reviewer = _ScriptedInferencer(script=[_review_approve()])
        fixer = _ScriptedInferencer(script=[])

        dual = DualInferencer(
            base_inferencer=multi_flow,
            review_inferencer=reviewer,
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        result = dual.infer("master")

        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        self.assertEqual(str(result.base_response).strip(), "integrated_3way")
        # Aggregator received all 3 plans in its prompt
        agg_input = aggregator._calls[0]
        for plan_text in ("plan_0", "plan_1", "plan_2"):
            self.assertIn(plan_text, agg_input)
        for label in ("Flow 0", "Flow 1", "Flow 2"):
            self.assertIn(label, agg_input)


if __name__ == "__main__":
    unittest.main()

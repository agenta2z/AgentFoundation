"""Unit tests for MultiFlowDualInferencer (convenience class).

Tier ordering:
  T1 — Construction & validation (inheritance, base_inferencer auto-construction,
       guards against direct base_inferencer assignment, etc.)
  T2 — Attr forwarding (each multi_flow_* attr lands on the inner MultiFlow)
  T3 — Equivalence to hand-wired composition (same inputs → same response)
  T4 — End-to-end happy paths
  T9 — Edge cases (workspace forwarding, registered YAML target)
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
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
    DEFAULT_MULTIFLOW_FOLLOWUP_TEMPLATE,
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


# ---------------------------------------------------------------------------
# Helpers (deterministic mocks)
# ---------------------------------------------------------------------------


@attrs
class _Scripted(InferencerBase):
    _script = attrib(factory=list)
    _idx = attrib(default=0, init=False)
    _calls = attrib(factory=list, init=False)

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


def _aggregator_text(integrated_plan: str) -> str:
    return f"<FinalPlan>\n{integrated_plan}\n</FinalPlan>"


def _approve(message: str = "OK") -> str:
    return (
        "```json\n"
        + json.dumps(
            {"approved": True, "severity": "MINOR", "issues": [], "reasoning": message}
        )
        + "\n```"
    )


def _reject(desc: str = "x") -> str:
    return (
        "```json\n"
        + json.dumps(
            {
                "approved": False,
                "severity": "MAJOR",
                "issues": [
                    {"severity": "MAJOR", "category": "logic",
                     "description": desc, "location": "n/a", "suggestion": "fix"}
                ],
                "reasoning": "rejected",
            }
        )
        + "\n```"
    )


def _parse_finalplan(s: str) -> str:
    m = re.search(r"<FinalPlan>([\s\S]*?)</FinalPlan>", s)
    return m.group(1).strip() if m else s


def _make_minimal_flow_configs(n: int = 2):
    """Build N flow configs that each emit a single plan and immediately stop."""
    return [
        {
            "input": f"task_{i}",
            "initial_inferencer": _Scripted(script=[f"plan_{i}"]),
            "followup_inferencer": _Scripted(script=[]),
            "end_condition": lambda s, r: True,
            "max_dynamic_steps": 1,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# T1 — Construction & validation
# ---------------------------------------------------------------------------


class TestT1Construction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inheritance_chain(self):
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            multi_flow_aggregator_inferencer=_Scripted(script=[_aggregator_text("p")]),
            review_inferencer=_Scripted(script=[_approve()]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        self.assertIsInstance(mfdi, DualInferencer)
        self.assertIsInstance(mfdi, LinearWorkflowInferencer)

    def test_base_inferencer_auto_constructed_as_multiflow(self):
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            multi_flow_aggregator_inferencer=_Scripted(script=[_aggregator_text("p")]),
            review_inferencer=_Scripted(script=[_approve()]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        self.assertIsInstance(mfdi.base_inferencer, MultiFlowInferencer)
        self.assertIsInstance(mfdi.base_inferencer, BreakdownThenAggregateInferencer)

    def test_setting_base_inferencer_directly_raises(self):
        with self.assertRaises(ValueError) as ctx:
            MultiFlowDualInferencer(
                flow_configs=_make_minimal_flow_configs(2),
                base_inferencer=_Scripted(script=[]),  # forbidden
                review_inferencer=_Scripted(script=[]),
                fixer_inferencer=_Scripted(script=[]),
            )
        self.assertIn("base_inferencer", str(ctx.exception))

    def test_empty_flow_configs_raises(self):
        with self.assertRaises(ValueError):
            MultiFlowDualInferencer(
                flow_configs=[],
                review_inferencer=_Scripted(script=[]),
                fixer_inferencer=_Scripted(script=[]),
            )


# ---------------------------------------------------------------------------
# T2 — Attr forwarding
# ---------------------------------------------------------------------------


class TestT2AttrForwarding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_visible_flows_forwarded(self):
        agg = _Scripted(script=[_aggregator_text("x")])
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            visible_flows="all",
            multi_flow_aggregator_inferencer=agg,
            review_inferencer=_Scripted(script=[_approve()]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        self.assertEqual(mfdi.base_inferencer.visible_flows, "all")

    def test_aggregator_inferencer_forwarded(self):
        agg = _Scripted(script=[_aggregator_text("x")])
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            multi_flow_aggregator_inferencer=agg,
            review_inferencer=_Scripted(script=[_approve()]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        self.assertIs(mfdi.base_inferencer.aggregator_inferencer, agg)

    def test_followup_and_aggregator_prompts_forwarded(self):
        agg = _Scripted(script=[_aggregator_text("x")])
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_followup_prompt="my_followup_template",
            review_inferencer=_Scripted(script=[_approve()]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        self.assertEqual(
            mfdi.base_inferencer.aggregator_prompt, DEFAULT_AGGREGATOR_PROMPT_TEMPLATE
        )
        self.assertEqual(
            mfdi.base_inferencer.multiflow_followup_prompt, "my_followup_template"
        )

    def test_response_parser_forwarded(self):
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            multi_flow_aggregator_inferencer=_Scripted(script=[_aggregator_text("x")]),
            multi_flow_response_parser=_parse_finalplan,
            review_inferencer=_Scripted(script=[_approve()]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        self.assertIs(mfdi.base_inferencer.response_parser, _parse_finalplan)

    def test_workspace_root_propagates_to_multi_flow(self):
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            multi_flow_aggregator_inferencer=_Scripted(script=[_aggregator_text("x")]),
            review_inferencer=_Scripted(script=[_approve()]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            workspace=self.tmp,
        )
        self.assertEqual(mfdi.base_inferencer._workspace.root, self.tmp)

    def test_initial_prompt_propagates_into_flow_configs(self):
        flow_configs = _make_minimal_flow_configs(2)
        # Pre-set initial_prompt on flow 1 only — the class-level default
        # should fill in flow 0 but not override flow 1.
        flow_configs[1]["initial_prompt"] = "custom_for_flow_1"

        mfdi = MultiFlowDualInferencer(
            flow_configs=flow_configs,
            multi_flow_initial_prompt="class_level_initial",
            multi_flow_aggregator_inferencer=_Scripted(script=[_aggregator_text("x")]),
            review_inferencer=_Scripted(script=[_approve()]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        self.assertEqual(
            mfdi.flow_configs[0]["initial_prompt"], "class_level_initial"
        )
        # Per-flow override preserved
        self.assertEqual(mfdi.flow_configs[1]["initial_prompt"], "custom_for_flow_1")


# ---------------------------------------------------------------------------
# T3 — Equivalence to hand-wired composition
# ---------------------------------------------------------------------------


def _build_hand_wired(
    *,
    workspace_dir: str,
    aggregator_text: str = "integrated_plan",
    review_first: bool = True,
):
    """Build an equivalent hand-wired DualInferencer + MultiFlowInferencer."""
    flow_configs = _make_minimal_flow_configs(2)
    agg = _Scripted(script=[_aggregator_text(aggregator_text)])
    multi_flow = MultiFlowInferencer(
        flow_configs=flow_configs,
        visible_flows="all",
        aggregator_inferencer=agg,
        aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
        multiflow_followup_prompt=DEFAULT_MULTIFLOW_FOLLOWUP_TEMPLATE,
        response_parser=_parse_finalplan,
        checkpoint_dir=workspace_dir,
    )
    reviewer = _Scripted(script=[_approve() if review_first else _reject()])
    fixer = _Scripted(script=[] if review_first else ["fixed_plan"])
    dual = DualInferencer(
        base_inferencer=multi_flow,
        review_inferencer=reviewer,
        fixer_inferencer=fixer,
        consensus_config=ConsensusConfig(max_iterations=2),
    )
    return dual, multi_flow, reviewer, fixer, agg


def _build_convenience(
    *,
    workspace_dir: str,
    aggregator_text: str = "integrated_plan",
    review_first: bool = True,
):
    """Build the same scenario via MultiFlowDualInferencer."""
    flow_configs = _make_minimal_flow_configs(2)
    agg = _Scripted(script=[_aggregator_text(aggregator_text)])
    reviewer = _Scripted(script=[_approve() if review_first else _reject()])
    fixer = _Scripted(script=[] if review_first else ["fixed_plan"])
    mfdi = MultiFlowDualInferencer(
        flow_configs=flow_configs,
        visible_flows="all",
        multi_flow_aggregator_inferencer=agg,
        multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
        multi_flow_followup_prompt=DEFAULT_MULTIFLOW_FOLLOWUP_TEMPLATE,
        multi_flow_response_parser=_parse_finalplan,
        review_inferencer=reviewer,
        fixer_inferencer=fixer,
        consensus_config=ConsensusConfig(max_iterations=2),
        checkpoint_dir=workspace_dir,
    )
    return mfdi, reviewer, fixer, agg


class TestT3EquivalenceToHandWired(unittest.TestCase):
    def setUp(self):
        self.hand_dir = tempfile.mkdtemp()
        self.conv_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.hand_dir, ignore_errors=True)
        shutil.rmtree(self.conv_dir, ignore_errors=True)

    def _run_both_and_compare(self, *, review_first: bool):
        hand_dual, _, hand_rev, hand_fix, hand_agg = _build_hand_wired(
            workspace_dir=self.hand_dir, review_first=review_first,
        )
        conv_mfdi, conv_rev, conv_fix, conv_agg = _build_convenience(
            workspace_dir=self.conv_dir, review_first=review_first,
        )
        hand_result = hand_dual.infer("master")
        conv_result = conv_mfdi.infer("master")

        # Same response shape and same key fields
        self.assertIsInstance(hand_result, DualInferencerResponse)
        self.assertIsInstance(conv_result, DualInferencerResponse)
        self.assertEqual(hand_result.consensus_achieved, conv_result.consensus_achieved)
        self.assertEqual(
            str(hand_result.base_response).strip(),
            str(conv_result.base_response).strip(),
        )
        self.assertEqual(hand_result.total_iterations, conv_result.total_iterations)

        # Same call counts
        self.assertEqual(hand_rev.call_count, conv_rev.call_count)
        self.assertEqual(hand_fix.call_count, conv_fix.call_count)
        self.assertEqual(hand_agg.call_count, conv_agg.call_count)

    def test_equivalent_consensus_first_iteration(self):
        self._run_both_and_compare(review_first=True)

    def test_equivalent_consensus_after_fix(self):
        # Reviewer approves on second call — convenience and hand-wired
        # both need 2 review calls + 1 fix. Adjust scripts:
        flow_configs_hand = _make_minimal_flow_configs(2)
        agg_hand = _Scripted(script=[_aggregator_text("integrated_v1")])
        multi_flow_hand = MultiFlowInferencer(
            flow_configs=flow_configs_hand,
            visible_flows="all",
            aggregator_inferencer=agg_hand,
            aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            response_parser=_parse_finalplan,
            checkpoint_dir=self.hand_dir,
        )
        rev_hand = _Scripted(script=[_reject(), _approve()])
        fix_hand = _Scripted(script=["integrated_v2"])
        dual_hand = DualInferencer(
            base_inferencer=multi_flow_hand,
            review_inferencer=rev_hand,
            fixer_inferencer=fix_hand,
            consensus_config=ConsensusConfig(max_iterations=3),
        )

        flow_configs_conv = _make_minimal_flow_configs(2)
        agg_conv = _Scripted(script=[_aggregator_text("integrated_v1")])
        rev_conv = _Scripted(script=[_reject(), _approve()])
        fix_conv = _Scripted(script=["integrated_v2"])
        mfdi = MultiFlowDualInferencer(
            flow_configs=flow_configs_conv,
            visible_flows="all",
            multi_flow_aggregator_inferencer=agg_conv,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_response_parser=_parse_finalplan,
            review_inferencer=rev_conv,
            fixer_inferencer=fix_conv,
            consensus_config=ConsensusConfig(max_iterations=3),
            checkpoint_dir=self.conv_dir,
        )

        hand_result = dual_hand.infer("master")
        conv_result = mfdi.infer("master")

        self.assertEqual(hand_result.consensus_achieved, conv_result.consensus_achieved)
        self.assertEqual(
            str(hand_result.base_response).strip(),
            str(conv_result.base_response).strip(),
        )
        self.assertEqual(rev_hand.call_count, rev_conv.call_count)
        self.assertEqual(fix_hand.call_count, fix_conv.call_count)


# ---------------------------------------------------------------------------
# T4 — End-to-end happy paths
# ---------------------------------------------------------------------------


class TestT4EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_e2e_n2_consensus_first_iteration(self):
        agg = _Scripted(script=[_aggregator_text("integrated")])
        rev = _Scripted(script=[_approve()])
        fix = _Scripted(script=[])
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            visible_flows="all",
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_response_parser=_parse_finalplan,
            review_inferencer=rev,
            fixer_inferencer=fix,
            consensus_config=ConsensusConfig(max_iterations=2),
            checkpoint_dir=self.tmp,
        )
        result = mfdi.infer("master")
        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        self.assertEqual(str(result.base_response).strip(), "integrated")
        self.assertEqual(rev.call_count, 1)
        self.assertEqual(fix.call_count, 0)
        self.assertEqual(agg.call_count, 1)

    def test_e2e_n3_happy_path(self):
        agg = _Scripted(script=[_aggregator_text("integrated_3way")])
        rev = _Scripted(script=[_approve()])
        fix = _Scripted(script=[])
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(3),
            visible_flows="all",
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_response_parser=_parse_finalplan,
            review_inferencer=rev,
            fixer_inferencer=fix,
            inject_upstream_artifacts=False,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        result = mfdi.infer("master")
        self.assertIsInstance(result, DualInferencerResponse)
        self.assertTrue(result.consensus_achieved)
        self.assertEqual(str(result.base_response).strip(), "integrated_3way")
        # Aggregator received all 3 plans
        agg_input = agg._calls[0]
        for label in ("Flow 0", "Flow 1", "Flow 2"):
            self.assertIn(label, agg_input)


# ---------------------------------------------------------------------------
# T9 — Edge cases (YAML target registration smoke + initial_response_override)
# ---------------------------------------------------------------------------


class TestT9EdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_yaml_target_registered(self):
        """The ``MultiFlowDual`` alias should be in the target registry."""
        # Importing registered_targets has the side effect of calling register_alias().
        from agent_foundation.common.configs import registered_targets  # noqa: F401
        from rich_python_utils.config_utils._registry import resolve_target

        cls_path = resolve_target("MultiFlowDual")
        self.assertIn("multi_flow_dual_inferencer", cls_path)
        self.assertIn("MultiFlowDualInferencer", cls_path)


# ===========================================================================
# Round 7 — dispatch (winner identification + reviewer/fixer selection)
# ===========================================================================


def _aggregator_text_with_winner(integrated: str, winner_idx: int) -> str:
    return (
        f"<FinalPlan>\n{integrated}\n</FinalPlan>"
        f"\n<Winner>flow_{winner_idx}</Winner>"
    )


def _aggregator_text_with_alias(
    integrated: str, *, winner_idx: int, reviewer: str = "", fixer: str = ""
) -> str:
    parts = [f"<FinalPlan>\n{integrated}\n</FinalPlan>"]
    parts.append(f"<Winner>flow_{winner_idx}</Winner>")
    if reviewer:
        parts.append(f"<Reviewer>{reviewer}</Reviewer>")
    if fixer:
        parts.append(f"<Fixer>{fixer}</Fixer>")
    return "\n".join(parts)


def _parse_winner_tag(s: str):
    m = re.search(r"<Winner>\s*flow_(\d+)\s*</Winner>", s)
    return int(m.group(1)) if m else None


def _parse_reviewer_tag(s: str):
    m = re.search(r"<Reviewer>\s*([\w-]+)\s*</Reviewer>", s)
    return m.group(1) if m else None


def _parse_fixer_tag(s: str):
    m = re.search(r"<Fixer>\s*([\w-]+)\s*</Fixer>", s)
    return m.group(1) if m else None


class TestT7ResolveId(unittest.TestCase):
    """Round 7 / T7: _resolve_id handles instance, alias, None, unknown."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build_mfdi(self, **kw):
        defaults = dict(
            flow_configs=_make_minimal_flow_configs(2),
            multi_flow_aggregator_inferencer=_Scripted(script=[_aggregator_text("x")]),
            review_inferencer=_Scripted(script=[]),
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
        )
        defaults.update(kw)
        return MultiFlowDualInferencer(**defaults)

    def test_resolve_instance_returned_as_is(self):
        mfdi = self._build_mfdi()
        x = _Scripted(script=[])
        self.assertIs(mfdi._resolve_id(x), x)

    def test_resolve_none_returns_none(self):
        mfdi = self._build_mfdi()
        self.assertIsNone(mfdi._resolve_id(None))

    def test_resolve_alias_via_pool(self):
        x = _Scripted(script=[])
        mfdi = self._build_mfdi(inferencer_pool={"X": x})
        self.assertIs(mfdi._resolve_id("X"), x)

    def test_resolve_unknown_alias_raises_keyerror(self):
        mfdi = self._build_mfdi(inferencer_pool={})
        with self.assertRaises(KeyError):
            mfdi._resolve_id("Unknown")


class TestT8RuleBasedAvoidance(unittest.TestCase):
    """Round 7 / T8: rule-based dispatch — avoid self-review, winner-as-fixer."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_review_default_winner_avoidance_swaps_to_pool(self):
        # Build flows where flow_1's inferencer == review_default → avoidance fires.
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        agg = _Scripted(script=[_aggregator_text_with_winner("integrated", winner_idx=1)])
        reviewer_fallback = _Scripted(script=[_approve()])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_winner_parser=_parse_winner_tag,
            multi_flow_response_parser=_parse_finalplan,
            review_default=flow1_inf,                # SAME as winner → avoidance
            review_priority_pool=[reviewer_fallback],
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        result = mfdi.infer("master")
        self.assertIsInstance(result, DualInferencerResponse)
        # After propose: review_inferencer must be the fallback, NOT flow1_inf
        self.assertIs(mfdi.review_inferencer, reviewer_fallback)
        self.assertEqual(reviewer_fallback.call_count, 1)

    def test_review_default_no_avoidance_when_winner_differs(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        agg = _Scripted(script=[_aggregator_text_with_winner("integrated", winner_idx=0)])
        reviewer_default = _Scripted(script=[_approve()])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_winner_parser=_parse_winner_tag,
            multi_flow_response_parser=_parse_finalplan,
            review_default=reviewer_default,         # NOT same as winner
            review_priority_pool=[_Scripted(script=[])],  # never picked
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        mfdi.infer("master")
        self.assertIs(mfdi.review_inferencer, reviewer_default)

    def test_review_default_winner_avoidance_pool_empty_uses_default(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        agg = _Scripted(script=[_aggregator_text_with_winner("integrated", winner_idx=1)])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_winner_parser=_parse_winner_tag,
            multi_flow_response_parser=_parse_finalplan,
            review_default=flow1_inf,                # SAME as winner
            review_priority_pool=[],                 # empty → fall back to default
            review_inferencer=None,
            # IMPORTANT: provide review_inferencer that mocks an approval response
            # because the default reviewer (flow1_inf) only emits "plan_1" once.
            # Keep simple by overriding script post-construction.
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        # Append an approval to flow1_inf's script so the review step succeeds.
        flow1_inf._script.append(_approve())
        mfdi.infer("master")
        # Falls back to default (flow1_inf) — self-review degradation
        self.assertIs(mfdi.review_inferencer, flow1_inf)

    def test_fixer_match_winner_sets_fixer_to_winner_inferencer(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        agg = _Scripted(script=[_aggregator_text_with_winner("integrated", winner_idx=0)])
        reviewer = _Scripted(script=[_approve()])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_winner_parser=_parse_winner_tag,
            multi_flow_response_parser=_parse_finalplan,
            review_inferencer=reviewer,
            fixer_match_winner=True,                 # winner → fixer
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        mfdi.infer("master")
        # winner_idx=0 → fixer = flow0_inf
        self.assertIs(mfdi.fixer_inferencer, flow0_inf)


class TestT9AliasDispatch(unittest.TestCase):
    """Round 7 / T9: alias-based (LLM-driven) dispatch."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_alias_in_review_default_resolves_via_pool(self):
        # review_default is a STRING alias, not an instance.
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        kiro_inst = _Scripted(script=[_approve()])
        agg = _Scripted(script=[_aggregator_text_with_winner("integrated", winner_idx=0)])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            inferencer_pool={"Kiro": kiro_inst, "Claude": _Scripted(script=[])},
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_winner_parser=_parse_winner_tag,
            multi_flow_response_parser=_parse_finalplan,
            review_default="Kiro",                   # alias → resolves to kiro_inst
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        mfdi.infer("master")
        self.assertIs(mfdi.review_inferencer, kiro_inst)

    def test_alias_dispatch_takes_precedence_over_rule(self):
        """LLM emits <Reviewer>Claude</Reviewer>; review_default=Kiro → reviewer is Claude."""
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        kiro_inst = _Scripted(script=[])
        claude_inst = _Scripted(script=[_approve()])
        agg = _Scripted(script=[
            _aggregator_text_with_alias("integrated", winner_idx=0, reviewer="Claude")
        ])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            inferencer_pool={"Kiro": kiro_inst, "Claude": claude_inst},
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_winner_parser=_parse_winner_tag,
            multi_flow_reviewer_alias_parser=_parse_reviewer_tag,
            multi_flow_response_parser=_parse_finalplan,
            review_default="Kiro",                   # rule-based default
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        mfdi.infer("master")
        # LLM picked Claude → that wins over rule-based default Kiro
        self.assertIs(mfdi.review_inferencer, claude_inst)

    def test_alias_dispatch_unknown_alias_falls_back_to_rule(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        kiro_inst = _Scripted(script=[_approve()])
        agg = _Scripted(script=[
            _aggregator_text_with_alias("integrated", winner_idx=0, reviewer="MysteryAgent")
        ])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            inferencer_pool={"Kiro": kiro_inst},     # no MysteryAgent
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_winner_parser=_parse_winner_tag,
            multi_flow_reviewer_alias_parser=_parse_reviewer_tag,
            multi_flow_response_parser=_parse_finalplan,
            review_default="Kiro",                   # fallback rule
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        mfdi.infer("master")
        # Unknown alias → fall back to rule-based; review_default=Kiro
        self.assertIs(mfdi.review_inferencer, kiro_inst)


class TestT10AconnectWalk(unittest.TestCase):
    """Round 7 / T10: aconnect walks all candidate inferencers."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_iter_child_inferencers_includes_pool_and_priority(self):
        flow0_inf = _Scripted(script=[])
        flow1_inf = _Scripted(script=[])
        pool_a = _Scripted(script=[])
        pool_b = _Scripted(script=[])
        priority = _Scripted(script=[])
        review_def = _Scripted(script=[])
        fixer = _Scripted(script=[])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=_Scripted(script=[_aggregator_text("x")]),
            inferencer_pool={"A": pool_a, "B": pool_b},
            review_default=review_def,
            review_priority_pool=[priority],
            fixer_inferencer=fixer,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        cands = list(mfdi._iter_child_inferencers())
        # Dedup by identity (the iterator dedups internally)
        ids = {id(c) for c in cands}
        for inf in (flow0_inf, flow1_inf, pool_a, pool_b, priority, review_def, fixer):
            self.assertIn(id(inf), ids,
                          f"_iter_child_inferencers missing {inf!r}")


class TestT11DispatchRecomputesPerCall(unittest.TestCase):
    """Round 7 / T11: dispatch logic recomputes per invocation, not stale.

    Unit-tests `_select_reviewer_and_fixer` directly by toggling the inner
    MultiFlow's `_last_winner_idx`. This isolates the dispatch correctness
    from the multi-attempt consensus-loop machinery (which is exercised
    elsewhere)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dispatch_recomputes_when_winner_changes(self):
        flow0_inf = _Scripted(script=[])
        flow1_inf = _Scripted(script=[])
        agg = _Scripted(script=[_aggregator_text("ignored")])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_winner_parser=_parse_winner_tag,
            review_default=flow1_inf,                # default = flow1
            review_priority_pool=[flow0_inf],
            fixer_match_winner=True,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )

        mfi = mfdi.base_inferencer

        # Simulate "attempt 1" — winner = flow_0 (NOT same as review_default)
        mfi._last_winner_idx = 0
        mfdi._select_reviewer_and_fixer()
        self.assertIs(mfdi.review_inferencer, flow1_inf)   # default (no avoidance)
        self.assertIs(mfdi.fixer_inferencer, flow0_inf)    # winner

        # Simulate "attempt 2" — winner = flow_1 (== review_default → avoidance)
        mfi._last_winner_idx = 1
        mfdi._select_reviewer_and_fixer()
        self.assertIs(mfdi.review_inferencer, flow0_inf)   # priority pool fallback
        self.assertIs(mfdi.fixer_inferencer, flow1_inf)    # winner

        # Simulate "attempt 3" — winner reset to flow_0 again
        mfi._last_winner_idx = 0
        mfdi._select_reviewer_and_fixer()
        self.assertIs(mfdi.review_inferencer, flow1_inf)   # back to default
        self.assertIs(mfdi.fixer_inferencer, flow0_inf)


# =============================================================================
# Post-mortem fixes — Fix 1c (DualInferencer iter_child) + Fix 5 (sanity warns)
# =============================================================================


class TestPostMortemFixes(unittest.TestCase):
    """Tests for post-mortem fixes (plan: real-CLI fixes)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dual_inferencer_iter_child_inferencers_yields_triple(self):
        """DualInferencer._iter_child_inferencers yields (base, review, fixer)
        with id-based dedup when fixer is aliased to base."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )
        base = _Scripted(script=[])
        reviewer = _Scripted(script=[])
        # fixer aliased to base — must dedup
        dual = DualInferencer(
            base_inferencer=base,
            review_inferencer=reviewer,
            fixer_inferencer=base,  # aliased to base
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        children = list(dual._iter_child_inferencers())
        ids = [id(c) for c in children]
        # base appears exactly once despite being in two slots
        self.assertEqual(ids.count(id(base)), 1)
        self.assertIn(id(reviewer), ids)
        self.assertEqual(len(children), 2)  # dedup'd

    def test_sanity_warning_when_winner_none_with_review_default(self):
        """Fix 5: warning fires when review_default is set but winner_idx
        is None (e.g., aggregator output has no <Winner> tag)."""
        import logging
        flow0_inf = _Scripted(script=[])
        flow1_inf = _Scripted(script=[])
        review_def = _Scripted(script=[])
        priority = _Scripted(script=[])

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=_Scripted(script=[_aggregator_text("plain")]),
            review_default=review_def,
            review_priority_pool=[priority],
            fixer_match_winner=True,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        mfi = mfdi.base_inferencer
        # Force "no winner detected" state.
        mfi._last_winner_idx = None
        with self.assertLogs(
            "agent_foundation.common.inferencers.agentic_inferencers."
            "flow_inferencers.multi_flow_dual_inferencer",
            level=logging.WARNING,
        ) as cm:
            mfdi._select_reviewer_and_fixer()
        # Both warnings fire because both review_default AND fixer_match_winner are set
        warnings_text = "\n".join(cm.output)
        self.assertIn("review_default", warnings_text)
        self.assertIn("fixer_match_winner", warnings_text)


# ---------------------------------------------------------------------------
# T12 — reviewer_match_second (runner-up as reviewer)
# ---------------------------------------------------------------------------


def _aggregator_text_with_winner_and_ranking(
    integrated: str, winner_idx: int, ranking: list
) -> str:
    winner_json = json.dumps({"winner_index": winner_idx, "reason": "test"})
    ranking_json = json.dumps({"ranking": ranking, "reason": "test"})
    return (
        f"<FinalPlan>\n{integrated}\n</FinalPlan>\n"
        f"```json winner_pick\n{winner_json}\n```\n"
        f"```json ranking\n{ranking_json}\n```"
    )


def _aggregator_text_with_ranking(integrated: str, ranking: list) -> str:
    ranking_json = json.dumps({"ranking": ranking, "reason": "test"})
    return (
        f"<FinalPlan>\n{integrated}\n</FinalPlan>\n"
        f"```json ranking\n{ranking_json}\n```"
    )


class TestT12ReviewerMatchSecond(unittest.TestCase):
    """Round 7 / T12: reviewer_match_second — runner-up as reviewer."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_seeds_reviewer_at_construction(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=_Scripted(script=[]),
            reviewer_match_second=True,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        self.assertIs(mfdi.review_inferencer, flow1_inf)

    def test_auto_enables_winner_pick(self):
        mfdi = MultiFlowDualInferencer(
            flow_configs=_make_minimal_flow_configs(2),
            multi_flow_aggregator_inferencer=_Scripted(script=[]),
            reviewer_match_second=True,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        self.assertTrue(mfdi.winner_pick)

    def test_single_flow_raises(self):
        with self.assertRaises(ValueError) as ctx:
            MultiFlowDualInferencer(
                flow_configs=_make_minimal_flow_configs(1),
                multi_flow_aggregator_inferencer=_Scripted(script=[]),
                reviewer_match_second=True,
                consensus_config=ConsensusConfig(max_iterations=1),
            )
        self.assertIn("reviewer_match_second", str(ctx.exception))

    def test_selects_runner_up_2_flows(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        agg = _Scripted(
            script=[_aggregator_text_with_winner_and_ranking("integrated", 0, [0, 1])]
        )
        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_response_parser=_parse_finalplan,
            reviewer_match_second=True,
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        result = mfdi.infer("master")
        self.assertIsInstance(result, DualInferencerResponse)
        self.assertIs(mfdi.review_inferencer, flow1_inf)

    def test_selects_runner_up_3_flows(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        flow2_inf = _Scripted(script=["plan_2"])
        agg = _Scripted(
            script=[_aggregator_text_with_winner_and_ranking("integrated", 2, [2, 0, 1])]
        )
        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t2", "initial_inferencer": flow2_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_response_parser=_parse_finalplan,
            reviewer_match_second=True,
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        result = mfdi.infer("master")
        self.assertIsInstance(result, DualInferencerResponse)
        self.assertIs(mfdi.review_inferencer, flow0_inf)

    def test_fallback_no_ranking(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        agg = _Scripted(
            script=[_aggregator_text_with_winner("integrated", winner_idx=0)]
        )
        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_winner_parser=_parse_winner_tag,
            multi_flow_response_parser=_parse_finalplan,
            reviewer_match_second=True,
            fixer_inferencer=_Scripted(script=[]),
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        result = mfdi.infer("master")
        self.assertIsInstance(result, DualInferencerResponse)
        self.assertIs(mfdi.review_inferencer, flow1_inf)

    def test_combined_with_fixer_match_winner(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        agg = _Scripted(
            script=[_aggregator_text_with_winner_and_ranking("integrated", 0, [0, 1])]
        )
        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=agg,
            multi_flow_aggregator_prompt=DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
            multi_flow_response_parser=_parse_finalplan,
            reviewer_match_second=True,
            fixer_match_winner=True,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        result = mfdi.infer("master")
        self.assertIsInstance(result, DualInferencerResponse)
        self.assertIs(mfdi.review_inferencer, flow1_inf)
        self.assertIs(mfdi.fixer_inferencer, flow0_inf)

    def test_recomputes_on_winner_change(self):
        flow0_inf = _Scripted(script=["plan_0"])
        flow1_inf = _Scripted(script=["plan_1"])
        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_inf,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=_Scripted(script=[]),
            reviewer_match_second=True,
            fixer_match_winner=True,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        mfi = mfdi.base_inferencer
        mfi._last_winner_idx = 0
        mfi._last_ranking = [0, 1]
        mfdi._select_reviewer_and_fixer()
        self.assertIs(mfdi.review_inferencer, flow1_inf)
        self.assertIs(mfdi.fixer_inferencer, flow0_inf)

        mfi._last_winner_idx = 1
        mfi._last_ranking = [1, 0]
        mfdi._select_reviewer_and_fixer()
        self.assertIs(mfdi.review_inferencer, flow0_inf)
        self.assertIs(mfdi.fixer_inferencer, flow1_inf)

    def test_shared_instance_warns(self):
        import logging
        shared = _Scripted(script=["plan"])
        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": shared,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": shared,
                 "followup_inferencer": _Scripted(script=[]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            multi_flow_aggregator_inferencer=_Scripted(script=[]),
            reviewer_match_second=True,
            consensus_config=ConsensusConfig(max_iterations=1),
            checkpoint_dir=self.tmp,
        )
        mfi = mfdi.base_inferencer
        mfi._last_winner_idx = 0
        with self.assertLogs(
            "agent_foundation.common.inferencers.agentic_inferencers."
            "flow_inferencers.multi_flow_dual_inferencer",
            level=logging.WARNING,
        ) as cm:
            mfdi._select_reviewer_and_fixer()
        self.assertIn("runner-up", "\n".join(cm.output))


# ---------------------------------------------------------------------------
# T12p — parse_ranking_tag parser tests
# ---------------------------------------------------------------------------


class TestT12Parser(unittest.TestCase):
    """Parser tests for parse_ranking_tag."""

    def test_json_block(self):
        from agent_foundation.common.inferencers.flow_parsers import parse_ranking_tag
        text = 'Some text\n```json ranking\n{"ranking": [2, 0, 1], "reason": "ok"}\n```\nmore'
        self.assertEqual(parse_ranking_tag(text), [2, 0, 1])

    def test_none_when_absent(self):
        from agent_foundation.common.inferencers.flow_parsers import parse_ranking_tag
        self.assertIsNone(parse_ranking_tag("no ranking here"))

    def test_non_string(self):
        from agent_foundation.common.inferencers.flow_parsers import parse_ranking_tag
        self.assertIsNone(parse_ranking_tag(42))
        self.assertIsNone(parse_ranking_tag(None))

    def test_bool_indices_excluded(self):
        from agent_foundation.common.inferencers.flow_parsers import parse_ranking_tag
        text = '```json ranking\n{"ranking": [true, 0, 1]}\n```'
        self.assertEqual(parse_ranking_tag(text), [0, 1])


if __name__ == "__main__":
    unittest.main()

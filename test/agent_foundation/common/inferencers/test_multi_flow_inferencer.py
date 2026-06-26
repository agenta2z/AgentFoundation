"""Unit tests for MultiFlowInferencer.

Tests covering:
1. 3 parallel flows with different end conditions → all complete → aggregator synthesizes
2. Heterogeneous flows (different inferencers per flow)
3. disable_aggregator=True returns collected worker outputs
4. max_concurrency limits parallel execution

Requirements: 38.1-38.7, 39.1-39.5, 40.1-40.3
"""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)


# ---------------------------------------------------------------------------
# Helpers — reuse the _make_sequential_inferencer pattern from
# test_lwi_dynamic_properties.py
# ---------------------------------------------------------------------------


def _make_sequential_inferencer(results_list):
    """Create a mock inferencer that returns results from a list in order.

    Each call to ainfer()/infer() returns the next item from results_list.
    """
    call_count = [0]
    mock = MagicMock(spec=[])

    def _infer(inp, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(results_list):
            return results_list[idx]
        return f"overflow_{idx}"

    async def _ainfer(inp, **kwargs):
        return _infer(inp, **kwargs)

    mock.infer = _infer
    mock.ainfer = _ainfer
    mock._call_count = call_count
    return mock


@attrs
class SimpleMockInferencer(InferencerBase):
    """A real InferencerBase subclass that returns a configurable response."""

    _response = attrib(default="mock response")
    _call_count = attrib(default=0, init=False)
    _received_inputs = attrib(factory=list, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._call_count += 1
        self._received_inputs.append(inference_input)
        if callable(self._response):
            return self._response(inference_input)
        return self._response


# ---------------------------------------------------------------------------
# Test 1: 3 parallel flows with different end conditions
# Validates: Requirements 38.1, 38.2, 38.3, 38.4, 38.6
# ---------------------------------------------------------------------------


class TestThreeParallelFlows(unittest.TestCase):
    """3 parallel flows with different end conditions → all complete
    independently → aggregator synthesizes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_three_flows_all_complete_and_aggregate(self):
        """Each flow has different max_dynamic_steps and end_condition.
        All 3 complete independently and the aggregator receives all results."""

        # Flow 0: runs 2 steps (end_condition triggers at step 2)
        flow0_initial = _make_sequential_inferencer(["f0_step0"])
        flow0_followup = _make_sequential_inferencer(["f0_step1"])

        # Flow 1: runs 3 steps (max_dynamic_steps=3, no end_condition)
        flow1_initial = _make_sequential_inferencer(["f1_step0"])
        flow1_followup = _make_sequential_inferencer(["f1_step1", "f1_step2"])

        # Flow 2: runs 1 step (end_condition triggers immediately)
        flow2_initial = _make_sequential_inferencer(["f2_step0"])
        flow2_followup = _make_sequential_inferencer([])

        aggregator_inputs = []

        def agg_fn(inp):
            aggregator_inputs.append(inp)
            return "synthesized_result"

        aggregator = SimpleMockInferencer(response=agg_fn)

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_0",
                    "initial_inferencer": flow0_initial,
                    "followup_inferencer": flow0_followup,
                    "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                    "max_dynamic_steps": 5,
                },
                {
                    "input": "task_1",
                    "initial_inferencer": flow1_initial,
                    "followup_inferencer": flow1_followup,
                    "end_condition": None,
                    "max_dynamic_steps": 3,
                },
                {
                    "input": "task_2",
                    "initial_inferencer": flow2_initial,
                    "followup_inferencer": flow2_followup,
                    "end_condition": lambda s, r: True,  # stop immediately after step 0
                    "max_dynamic_steps": 10,
                },
            ],
            aggregator_inferencer=aggregator,
            checkpoint_dir=self.tmpdir,
        )

        result = mfi.infer("build something")

        # Aggregator should have been called
        self.assertEqual(aggregator._call_count, 1)
        self.assertEqual(result, "synthesized_result")

        # Aggregator input should contain results from all 3 flows
        agg_input = aggregator_inputs[0]
        # The aggregator receives a prompt containing all worker results
        self.assertIsNotNone(agg_input)


# ---------------------------------------------------------------------------
# Test 2: Heterogeneous flows (different inferencers per flow)
# Validates: Requirements 38.6, 39.1
# ---------------------------------------------------------------------------


class TestHeterogeneousFlows(unittest.TestCase):
    """Each flow uses different initial/followup inferencers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_each_flow_uses_its_own_inferencers(self):
        """Create flows with different initial/followup inferencers.
        Verify each flow uses its own inferencers by checking call counts."""

        # Flow A: uses inferencer_a_init and inferencer_a_followup
        inf_a_init = _make_sequential_inferencer(["A_init_out"])
        inf_a_followup = _make_sequential_inferencer(["A_followup_out"])

        # Flow B: uses inferencer_b_init and inferencer_b_followup
        inf_b_init = _make_sequential_inferencer(["B_init_out"])
        inf_b_followup = _make_sequential_inferencer(["B_followup_out"])

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_A",
                    "initial_inferencer": inf_a_init,
                    "followup_inferencer": inf_a_followup,
                    "max_dynamic_steps": 2,
                },
                {
                    "input": "task_B",
                    "initial_inferencer": inf_b_init,
                    "followup_inferencer": inf_b_followup,
                    "max_dynamic_steps": 2,
                },
            ],
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )

        result = mfi.infer("go")

        # Each initial inferencer should have been called exactly once
        self.assertEqual(inf_a_init._call_count[0], 1,
                         "Flow A initial inferencer should be called once")
        self.assertEqual(inf_b_init._call_count[0], 1,
                         "Flow B initial inferencer should be called once")

        # Each followup inferencer should have been called exactly once
        # (2 steps total per flow: 1 initial + 1 followup)
        self.assertEqual(inf_a_followup._call_count[0], 1,
                         "Flow A followup inferencer should be called once")
        self.assertEqual(inf_b_followup._call_count[0], 1,
                         "Flow B followup inferencer should be called once")


# ---------------------------------------------------------------------------
# Test 3: disable_aggregator=True returns collected worker outputs
# Validates: Requirements 38.5, 39.1
# ---------------------------------------------------------------------------


class TestDisableAggregator(unittest.TestCase):
    """disable_aggregator=True returns raw worker outputs (tree topology)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_disable_aggregator_returns_worker_outputs(self):
        """With disable_aggregator=True, raw worker outputs are returned."""

        flow0_init = _make_sequential_inferencer(["flow0_result"])
        flow1_init = _make_sequential_inferencer(["flow1_result"])

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_0",
                    "initial_inferencer": flow0_init,
                    "followup_inferencer": flow0_init,
                    "end_condition": lambda s, r: True,
                    "max_dynamic_steps": 1,
                },
                {
                    "input": "task_1",
                    "initial_inferencer": flow1_init,
                    "followup_inferencer": flow1_init,
                    "end_condition": lambda s, r: True,
                    "max_dynamic_steps": 1,
                },
            ],
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )

        result = mfi.infer("go")

        # Result should be a tuple of worker outputs (no aggregation)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Test 4: max_concurrency limits parallel execution
# Validates: Requirements 39.4
# ---------------------------------------------------------------------------


class TestMaxConcurrency(unittest.TestCase):
    """max_concurrency limits parallel execution — verify all flows complete."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_max_concurrency_1_all_flows_complete(self):
        """With max_concurrency=1, all flows still complete (sequentially)."""

        flows = []
        for i in range(3):
            init = _make_sequential_inferencer([f"flow{i}_out"])
            flows.append({
                "input": f"task_{i}",
                "initial_inferencer": init,
                "followup_inferencer": init,
                "end_condition": lambda s, r: True,
                "max_dynamic_steps": 1,
            })

        mfi = MultiFlowInferencer(
            flow_configs=flows,
            disable_aggregator=True,
            max_concurrency=1,
            checkpoint_dir=self.tmpdir,
        )

        result = mfi.infer("go")

        # All 3 flows should complete
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_max_concurrency_2_all_flows_complete(self):
        """With max_concurrency=2 and 4 flows, all flows still complete."""

        flows = []
        for i in range(4):
            init = _make_sequential_inferencer([f"flow{i}_out"])
            flows.append({
                "input": f"task_{i}",
                "initial_inferencer": init,
                "followup_inferencer": init,
                "end_condition": lambda s, r: True,
                "max_dynamic_steps": 1,
            })

        mfi = MultiFlowInferencer(
            flow_configs=flows,
            disable_aggregator=True,
            max_concurrency=2,
            checkpoint_dir=self.tmpdir,
        )

        result = mfi.infer("go")

        # All 4 flows should complete
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)


# ===========================================================================
# Round-4/5/6 — MultiFlow refactor as a BreakdownThenAggregateInferencer subclass
# ===========================================================================

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (  # noqa: E402
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (  # noqa: E402
    _resolve_visible_indices,
)

# Inline Jinja fixtures for tests that need to exercise the aggregator-prompt
# and followup-prompt code paths. The renderer just needs any non-empty Jinja
# string with the right placeholders to produce a non-empty prompt feed for
# the aggregator / followup mocks.
_TEST_AGGREGATOR_PROMPT = """\
Original task:
{{ input }}

Each team's final output:
{% for idx, plan in worker_plans.items() %}
=== Flow {{ idx }} ===
{{ plan if plan else '(no output)' }}

{% endfor %}\
Produce a final integrated synthesis.
"""

_TEST_FOLLOWUP_PROMPT = """\
Original task:
{{ input }}

Your previous output:
{{ your_prev }}

{% if visible_plans %}
Other teams' latest outputs:
{% for idx, plan in visible_plans.items() %}
--- Flow {{ idx }} ---
{{ plan if plan else '(no output yet)' }}

{% endfor %}
{% endif %}\
Continue iterating; integrate the best ideas from any other teams' outputs.
"""


# ---------------------------------------------------------------------------
# T1 — Inheritance & shape
# ---------------------------------------------------------------------------


class TestInheritanceAndShape(unittest.TestCase):
    """T1: MultiFlow IS a BTA with breakdown disabled and workers as dynamic LWIs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_minimal_mfi(self, **overrides):
        flows = [
            {
                "input": "task_0",
                "initial_inferencer": _make_sequential_inferencer(["A"]),
                "followup_inferencer": _make_sequential_inferencer(["A_fu"]),
                "max_dynamic_steps": 1,
                "end_condition": lambda s, r: True,
            }
        ]
        kw = {
            "flow_configs": flows,
            "disable_aggregator": True,
            "checkpoint_dir": self.tmpdir,
        }
        kw.update(overrides)
        return MultiFlowInferencer(**kw)

    def test_inherits_from_bta(self):
        mfi = self._make_minimal_mfi()
        self.assertIsInstance(mfi, BreakdownThenAggregateInferencer)

    def test_predefined_sub_queries_derived_from_flow_configs(self):
        flows = [
            {"input": "alpha",
             "initial_inferencer": _make_sequential_inferencer(["x"]),
             "max_dynamic_steps": 1, "end_condition": lambda s, r: True},
            {"input": "beta",
             "initial_inferencer": _make_sequential_inferencer(["y"]),
             "max_dynamic_steps": 1, "end_condition": lambda s, r: True},
        ]
        mfi = MultiFlowInferencer(
            flow_configs=flows,
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        self.assertEqual(mfi.predefined_sub_queries, ["alpha", "beta"])
        self.assertIsNone(mfi.breakdown_inferencer)
        self.assertIsNotNone(mfi.worker_inferencers)

    def test_empty_flow_configs_raises(self):
        with self.assertRaises(ValueError):
            MultiFlowInferencer(flow_configs=[])

    def test_flow_config_missing_input_raises(self):
        with self.assertRaises(ValueError):
            MultiFlowInferencer(
                flow_configs=[{"initial_inferencer": _make_sequential_inferencer(["x"])}],
            )

    def test_passing_predefined_sub_queries_directly_raises(self):
        with self.assertRaises(ValueError):
            MultiFlowInferencer(
                flow_configs=[{
                    "input": "q",
                    "initial_inferencer": _make_sequential_inferencer(["x"]),
                    "max_dynamic_steps": 1, "end_condition": lambda s, r: True,
                }],
                predefined_sub_queries=["override"],
            )

    def test_passing_breakdown_inferencer_directly_raises(self):
        dummy = _make_sequential_inferencer(["bd"])
        with self.assertRaises(ValueError):
            MultiFlowInferencer(
                flow_configs=[{
                    "input": "q",
                    "initial_inferencer": _make_sequential_inferencer(["x"]),
                    "max_dynamic_steps": 1, "end_condition": lambda s, r: True,
                }],
                breakdown_inferencer=dummy,
            )

    def test_resolve_visible_indices_helper(self):
        self.assertEqual(_resolve_visible_indices(0, 3, "self"), [0])
        self.assertEqual(_resolve_visible_indices(0, 3, "all"), [0, 1, 2])
        self.assertEqual(_resolve_visible_indices(0, 3, [1, 2]), [1, 2])
        # Out-of-range indices are filtered out
        self.assertEqual(_resolve_visible_indices(0, 3, [1, 99, -1]), [1])
        with self.assertRaises(ValueError):
            _resolve_visible_indices(0, 3, "everyone")


# ---------------------------------------------------------------------------
# T2 — Cross-flow visibility via `visible_flows`
# ---------------------------------------------------------------------------


class _PromptCapturingInferencer(InferencerBase):
    """An InferencerBase that records every prompt it receives."""

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self._idx = 0
        self.received_prompts = []

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.received_prompts.append(str(inference_input))
        if self._idx < len(self._responses):
            r = self._responses[self._idx]
        else:
            r = f"overflow_{self._idx}"
        self._idx += 1
        return r


def _silent_aggregator():
    return SimpleMockInferencer(response=lambda x: "agg")


class TestCrossFlowVisibility(unittest.TestCase):
    """T2/T4: visible_flows propagation and prompt rendering."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_two_flow_mfi(self, *, visible_flows, followup_template):
        flow0_init = _PromptCapturingInferencer(["plan_0_step0"])
        flow0_followup = _PromptCapturingInferencer(["plan_0_step1"])
        flow1_init = _PromptCapturingInferencer(["plan_1_step0"])
        flow1_followup = _PromptCapturingInferencer(["plan_1_step1"])

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_0",
                    "initial_inferencer": flow0_init,
                    "followup_inferencer": flow0_followup,
                    "followup_prompt": followup_template,
                    "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                    "max_dynamic_steps": 5,
                },
                {
                    "input": "task_1",
                    "initial_inferencer": flow1_init,
                    "followup_inferencer": flow1_followup,
                    "followup_prompt": followup_template,
                    "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                    "max_dynamic_steps": 5,
                },
            ],
            visible_flows=visible_flows,
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        return mfi, (flow0_init, flow0_followup, flow1_init, flow1_followup)

    def test_visible_self_no_other_plans_in_followup_prompt(self):
        tmpl = "your_prev={{ your_prev }} | visible_count={{ visible_plans|length }}"
        mfi, (_, f0_fu, _, f1_fu) = self._build_two_flow_mfi(
            visible_flows="self", followup_template=tmpl,
        )
        mfi.infer("master")
        # Followup prompts should report 0 visible peers
        for fu in (f0_fu, f1_fu):
            self.assertTrue(any("visible_count=0" in p for p in fu.received_prompts),
                            f"expected visible_count=0 in {fu.received_prompts}")

    def test_visible_all_includes_other_plans_in_followup_prompt(self):
        tmpl = (
            "your_prev={{ your_prev }} | "
            "visible_keys={{ visible_plans.keys()|list }} | "
            "{% for idx, plan in visible_plans.items() %}"
            "peer{{ idx }}={{ plan }};"
            "{% endfor %}"
        )
        mfi, (_, f0_fu, _, f1_fu) = self._build_two_flow_mfi(
            visible_flows="all", followup_template=tmpl,
        )
        mfi.infer("master")
        # Flow 0's followup prompt should include flow 1's latest output
        f0_prompt = f0_fu.received_prompts[0]
        # depending on scheduling, flow 1 may or may not have produced yet; but
        # by the time the followup *step* runs the OTHER flow's step 0 has run
        # in sync mode (sequential). Allow either "plan_1_step0" or "(no output yet)".
        # Prefer to assert the visible_keys field is non-empty and excludes self.
        self.assertIn("peer1=", f0_prompt)
        self.assertNotIn("peer0=", f0_prompt)  # self excluded
        f1_prompt = f1_fu.received_prompts[0]
        self.assertIn("peer0=", f1_prompt)
        self.assertNotIn("peer1=", f1_prompt)

    def test_visible_explicit_indices(self):
        tmpl = "{% for idx in visible_plans %}peer{{ idx }};{% endfor %}"
        mfi, (_, f0_fu, _, f1_fu) = self._build_two_flow_mfi(
            visible_flows=[1], followup_template=tmpl,
        )
        # Override per-flow: flow 1 sees nothing, flow 0 sees flow 1
        mfi.flow_configs[1]["visible_flows"] = []
        mfi = MultiFlowInferencer(
            flow_configs=mfi.flow_configs,
            visible_flows="self",   # default class-level
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        # Resolve via the helper directly
        self.assertEqual(mfi._resolve_flow_visibility(1), [])

    def test_per_flow_visibility_override(self):
        tmpl = "v={{ visible_plans|length }}"
        flows = [
            {
                "input": "t0",
                "initial_inferencer": _PromptCapturingInferencer(["p0"]),
                "followup_inferencer": _PromptCapturingInferencer(["p0_fu"]),
                "followup_prompt": tmpl,
                "visible_flows": "all",     # override per-flow
                "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                "max_dynamic_steps": 5,
            },
            {
                "input": "t1",
                "initial_inferencer": _PromptCapturingInferencer(["p1"]),
                "followup_inferencer": _PromptCapturingInferencer(["p1_fu"]),
                "followup_prompt": tmpl,
                # no override → uses class-level
                "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                "max_dynamic_steps": 5,
            },
        ]
        mfi = MultiFlowInferencer(
            flow_configs=flows,
            visible_flows="self",
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        self.assertEqual(mfi._resolve_flow_visibility(0), [0, 1])  # override → all
        self.assertEqual(mfi._resolve_flow_visibility(1), [1])     # default → self


# ---------------------------------------------------------------------------
# T5 — Template machinery
# ---------------------------------------------------------------------------


class TestTemplateMachinery(unittest.TestCase):
    """T5: default templates render; custom dynamic_input_builder takes precedence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_followup_template_used_when_class_level_set(self):
        captured = []

        def _capturing_followup_inferencer():
            class _Cap(InferencerBase):
                def _infer(self, x, inference_config=None, **kw):
                    captured.append(str(x))
                    return "step1_result"
            return _Cap()

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_x",
                    "initial_inferencer": _make_sequential_inferencer(["init_result"]),
                    "followup_inferencer": _capturing_followup_inferencer(),
                    "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                    "max_dynamic_steps": 5,
                }
            ],
            visible_flows="self",
            multiflow_followup_prompt=_TEST_FOLLOWUP_PROMPT,
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("ignored_master")

        # The default template should mention "Continue iterating"
        self.assertTrue(captured)
        self.assertIn("Continue iterating", captured[0])
        # And include the per-flow input
        self.assertIn("task_x", captured[0])
        # And include the previous result
        self.assertIn("init_result", captured[0])

    def test_custom_dynamic_input_builder_takes_precedence(self):
        captured = []

        class _Cap(InferencerBase):
            def _infer(self, x, inference_config=None, **kw):
                captured.append(str(x))
                return "fu_result"

        def my_builder(state, prev_result):
            # Should override the template entirely
            return f"CUSTOM_BUILDER_INPUT[{prev_result}]"

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task",
                    "initial_inferencer": _make_sequential_inferencer(["init"]),
                    "followup_inferencer": _Cap(),
                    "dynamic_input_builder": my_builder,   # takes precedence
                    "followup_prompt": _TEST_FOLLOWUP_PROMPT,
                    "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                    "max_dynamic_steps": 5,
                }
            ],
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("ignored")
        self.assertTrue(captured[0].startswith("CUSTOM_BUILDER_INPUT["))

    def test_no_template_no_user_builder_legacy_passthrough(self):
        """When no template and no user builder, prev_result passes through unchanged."""
        captured = []

        class _Cap(InferencerBase):
            def _infer(self, x, inference_config=None, **kw):
                captured.append(str(x))
                return "fu_result"

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task",
                    "initial_inferencer": _make_sequential_inferencer(["init_payload"]),
                    "followup_inferencer": _Cap(),
                    # no followup_prompt, no dynamic_input_builder
                    "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                    "max_dynamic_steps": 5,
                }
            ],
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("ignored")
        self.assertEqual(captured[0], "init_payload")  # prev_result passes through

    def test_initial_prompt_template_renders(self):
        captured = []

        class _Cap(InferencerBase):
            def _infer(self, x, inference_config=None, **kw):
                captured.append(str(x))
                return "init_result"

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "task_q",
                    "initial_inferencer": _Cap(),
                    "followup_inferencer": _make_sequential_inferencer([]),
                    "initial_prompt": "INITIAL[{{ input }}]@flow{{ flow_idx }}",
                    "end_condition": lambda s, r: True,
                    "max_dynamic_steps": 1,
                }
            ],
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("master")
        self.assertEqual(captured[0], "INITIAL[task_q]@flow0")


# ---------------------------------------------------------------------------
# T6 — Aggregator template + judgment_parser
# ---------------------------------------------------------------------------


class TestAggregatorIntegration(unittest.TestCase):
    """T6: aggregator_prompt template renders; judgment_parser feeds summary."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_aggregator_template_renders_with_worker_plans(self):
        agg_inputs = []

        class _AggCap(InferencerBase):
            def _infer(self, x, inference_config=None, **kw):
                agg_inputs.append(str(x))
                return "<FinalPlan>integrated</FinalPlan>"

        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "t0",
                    "initial_inferencer": _make_sequential_inferencer(["plan0_final"]),
                    "followup_inferencer": _make_sequential_inferencer([]),
                    "end_condition": lambda s, r: True,
                    "max_dynamic_steps": 1,
                },
                {
                    "input": "t1",
                    "initial_inferencer": _make_sequential_inferencer(["plan1_final"]),
                    "followup_inferencer": _make_sequential_inferencer([]),
                    "end_condition": lambda s, r: True,
                    "max_dynamic_steps": 1,
                },
            ],
            aggregator_inferencer=_AggCap(),
            aggregator_prompt=_TEST_AGGREGATOR_PROMPT,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("the_master_task")

        self.assertEqual(len(agg_inputs), 1)
        prompt = agg_inputs[0]
        self.assertIn("the_master_task", prompt)
        self.assertIn("plan0_final", prompt)
        self.assertIn("plan1_final", prompt)
        self.assertIn("Flow 0", prompt)
        self.assertIn("Flow 1", prompt)

    def test_response_parser_strips_finalplan_tag(self):
        import re

        class _Agg(InferencerBase):
            def _infer(self, x, inference_config=None, **kw):
                return "<FinalPlan>integrated_text</FinalPlan>"

        def parse_finalplan(s):
            m = re.search(r"<FinalPlan>([\s\S]*?)</FinalPlan>", s)
            return m.group(1).strip() if m else s

        mfi = MultiFlowInferencer(
            flow_configs=[
                {"input": "t0",
                 "initial_inferencer": _make_sequential_inferencer(["p0"]),
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True,
                 "max_dynamic_steps": 1},
                {"input": "t1",
                 "initial_inferencer": _make_sequential_inferencer(["p1"]),
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True,
                 "max_dynamic_steps": 1},
            ],
            aggregator_inferencer=_Agg(),
            response_parser=parse_finalplan,
            checkpoint_dir=self.tmpdir,
        )
        result = mfi.infer("master")
        self.assertEqual(result, "integrated_text")

    def test_judgment_parser_accumulates_judgments(self):
        # Each followup output emits a <Judgment> tag; aggregator template
        # references all_judgments_summary.
        import re

        class _Agg(InferencerBase):
            def __init__(self):
                super().__init__()
                self.last_input = None

            def _infer(self, x, inference_config=None, **kw):
                self.last_input = str(x)
                return "<FinalPlan>final</FinalPlan>"

        def parse_judgment(s):
            m = re.search(r"<Judgment>([\s\S]*?)</Judgment>", s)
            return m.group(1).strip() if m else None

        agg = _Agg()
        flow_outputs_with_judgment = [
            "plan_a_v0 <Judgment>flow_0</Judgment>",
            "plan_a_v1 <Judgment>tie</Judgment>",
        ]
        mfi = MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "t0",
                    "initial_inferencer": _make_sequential_inferencer(
                        [flow_outputs_with_judgment[0]]),
                    "followup_inferencer": _make_sequential_inferencer(
                        [flow_outputs_with_judgment[1]]),
                    "followup_prompt": "v={{ your_prev }}",
                    "end_condition": lambda s, r: s.get("dynamic_step_count", 0) >= 2,
                    "max_dynamic_steps": 5,
                },
            ],
            aggregator_inferencer=agg,
            aggregator_prompt=_TEST_AGGREGATOR_PROMPT,
            judgment_parser=parse_judgment,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("master")
        # Flow 0 produced step 0 with <Judgment>flow_0</Judgment>; the judgment
        # is parsed at the start of step 1 (when prev_result = step 0 output).
        # So we expect at least one entry in _all_judgments.
        self.assertEqual(len(mfi._all_judgments), 1)
        self.assertEqual(mfi._all_judgments[0][0], 0)   # flow index
        self.assertEqual(mfi._all_judgments[0][2], "flow_0")
        # And the aggregator prompt should mention the judgment summary.
        self.assertIn("flow_0", agg.last_input)


# ---------------------------------------------------------------------------
# T7 — N-flow scaling
# ---------------------------------------------------------------------------


class TestNFlowScaling(unittest.TestCase):
    """T7: any N flows works — exercises N=2, 3, and 5."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_n_flows(self, n):
        flows = [
            {
                "input": f"task_{i}",
                "initial_inferencer": _make_sequential_inferencer([f"plan_{i}"]),
                "followup_inferencer": _make_sequential_inferencer([]),
                "end_condition": lambda s, r: True,
                "max_dynamic_steps": 1,
            }
            for i in range(n)
        ]
        mfi = MultiFlowInferencer(
            flow_configs=flows,
            disable_aggregator=True,
            checkpoint_dir=tempfile.mkdtemp(dir=self.tmpdir),
        )
        result = mfi.infer("master")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), n)

    def test_n_equals_2(self):
        self._run_n_flows(2)

    def test_n_equals_3(self):
        self._run_n_flows(3)

    def test_n_equals_5(self):
        self._run_n_flows(5)


# ---------------------------------------------------------------------------
# T9 — Backward compat
# ---------------------------------------------------------------------------


class TestBackwardCompat(unittest.TestCase):
    """T9: MultiFlow constructed with no new attrs behaves like a plain BTA
    with predefined_sub_queries + worker_inferencers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_minimal_config_runs_without_new_attrs(self):
        """Construct with only flow_configs + disable_aggregator. No
        visible_flows, no templates, no parsers."""
        flows = [
            {"input": "a",
             "initial_inferencer": _make_sequential_inferencer(["A"]),
             "followup_inferencer": _make_sequential_inferencer([]),
             "end_condition": lambda s, r: True,
             "max_dynamic_steps": 1},
            {"input": "b",
             "initial_inferencer": _make_sequential_inferencer(["B"]),
             "followup_inferencer": _make_sequential_inferencer([]),
             "end_condition": lambda s, r: True,
             "max_dynamic_steps": 1},
        ]
        mfi = MultiFlowInferencer(
            flow_configs=flows,
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
        )
        # Confirm sane defaults
        self.assertEqual(mfi.visible_flows, "self")
        self.assertIsNone(mfi.prompt_formatter)
        self.assertIsNone(mfi.multiflow_followup_prompt)
        self.assertIsNone(mfi.aggregator_prompt)
        self.assertIsNone(mfi.judgment_parser)
        self.assertIsNone(mfi.response_parser)
        result = mfi.infer("ignored")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ===========================================================================
# Round 7 — opt-in dispatch state extraction
# ===========================================================================


import re  # noqa: E402


class TestRound7DispatchState(unittest.TestCase):
    """Round 7: winner_parser + alias parsers + accessors (opt-in)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_two_flow_mfi(self, *, agg_output, **mfi_kwargs):
        flow0_init = _make_sequential_inferencer(["plan_0"])
        flow1_init = _make_sequential_inferencer(["plan_1"])

        class _AggCap(InferencerBase):
            def __init__(self, output):
                super().__init__()
                self._output = output

            def _infer(self, x, inference_config=None, **kw):
                return self._output

        return MultiFlowInferencer(
            flow_configs=[
                {
                    "input": "t0",
                    "initial_inferencer": flow0_init,
                    "followup_inferencer": _make_sequential_inferencer([]),
                    "end_condition": lambda s, r: True,
                    "max_dynamic_steps": 1,
                },
                {
                    "input": "t1",
                    "initial_inferencer": flow1_init,
                    "followup_inferencer": _make_sequential_inferencer([]),
                    "end_condition": lambda s, r: True,
                    "max_dynamic_steps": 1,
                },
            ],
            aggregator_inferencer=_AggCap(agg_output),
            aggregator_prompt=_TEST_AGGREGATOR_PROMPT,
            checkpoint_dir=self.tmpdir,
            **mfi_kwargs,
        ), flow0_init, flow1_init

    @staticmethod
    def _parse_winner_tag(s):
        m = re.search(r"<Winner>\s*flow_(\d+)\s*</Winner>", s)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_reviewer_tag(s):
        m = re.search(r"<Reviewer>\s*([\w-]+)\s*</Reviewer>", s)
        return m.group(1) if m else None

    @staticmethod
    def _parse_fixer_tag(s):
        m = re.search(r"<Fixer>\s*([\w-]+)\s*</Fixer>", s)
        return m.group(1) if m else None

    def test_no_dispatch_parsers_set_keeps_state_none(self):
        """No parsers configured → all _last_* fields stay None (Round 5/6 backward compat)."""
        mfi, _, _ = self._build_two_flow_mfi(
            agg_output="<FinalPlan>integrated</FinalPlan>",
        )
        mfi.infer("master")
        self.assertIsNone(mfi.get_winner_flow_idx())
        self.assertIsNone(mfi.get_winner_inferencer())
        self.assertIsNone(mfi.get_chosen_reviewer_alias())
        self.assertIsNone(mfi.get_chosen_fixer_alias())

    def test_winner_parser_extracts_flow_idx(self):
        mfi, _, _ = self._build_two_flow_mfi(
            agg_output="<FinalPlan>integrated</FinalPlan>\n<Winner>flow_1</Winner>",
            winner_parser=self._parse_winner_tag,
        )
        mfi.infer("master")
        self.assertEqual(mfi.get_winner_flow_idx(), 1)

    def test_get_winner_inferencer_returns_correct_flow_inferencer(self):
        mfi, flow0_init, flow1_init = self._build_two_flow_mfi(
            agg_output="<FinalPlan>integrated</FinalPlan>\n<Winner>flow_0</Winner>",
            winner_parser=self._parse_winner_tag,
        )
        mfi.infer("master")
        self.assertIs(mfi.get_winner_inferencer(), flow0_init)

    def test_alias_parsers_extract_strings(self):
        mfi, _, _ = self._build_two_flow_mfi(
            agg_output=(
                "<FinalPlan>integrated</FinalPlan>"
                "\n<Reviewer>Kiro</Reviewer>"
                "\n<Fixer>Claude</Fixer>"
            ),
            reviewer_alias_parser=self._parse_reviewer_tag,
            fixer_alias_parser=self._parse_fixer_tag,
        )
        mfi.infer("master")
        self.assertEqual(mfi.get_chosen_reviewer_alias(), "Kiro")
        self.assertEqual(mfi.get_chosen_fixer_alias(), "Claude")

    def test_parser_failure_logs_and_keeps_field_none(self):
        def boom(s):
            raise RuntimeError("intentional")

        mfi, _, _ = self._build_two_flow_mfi(
            agg_output="<FinalPlan>integrated</FinalPlan>\n<Winner>flow_1</Winner>",
            winner_parser=boom,
            reviewer_alias_parser=self._parse_reviewer_tag,
        )
        mfi.infer("master")
        # winner_parser raised → stays None; other parsers are independent
        self.assertIsNone(mfi.get_winner_flow_idx())
        # reviewer_alias_parser ran fine (no Reviewer tag → None)
        self.assertIsNone(mfi.get_chosen_reviewer_alias())

    def test_disable_aggregator_skips_dispatch_extraction(self):
        flow0_init = _make_sequential_inferencer(["plan_0"])
        flow1_init = _make_sequential_inferencer(["plan_1"])
        mfi = MultiFlowInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": flow0_init,
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": flow1_init,
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            disable_aggregator=True,
            checkpoint_dir=self.tmpdir,
            winner_parser=self._parse_winner_tag,  # set, but should be skipped
        )
        mfi.infer("master")
        # No aggregator → no extraction even though winner_parser is set
        self.assertIsNone(mfi.get_winner_flow_idx())


# =============================================================================
# Post-mortem fixes (plan: real-CLI fixes) — Fix 1c MultiFlow children + Fix 2
# =============================================================================


@attrs
class _FixedOutputAgg(InferencerBase):
    """Real InferencerBase subclass returning a configurable string from
    each _infer / _ainfer call. Necessary because aggregator output goes
    through MultiFlow's _normalize_aggregator_output which expects an
    InferencerBase return type, not a MagicMock."""

    output: str = attrib(default="")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.output


@attrs
class _TerminalResponseAgg(InferencerBase):
    """Real InferencerBase subclass returning a TerminalInferencerResponse-like
    object (has `.output` attr) instead of a plain string — exactly mirrors
    what `ClaudeCodeCliInferencer` / `KiroCliInferencer` return in real use."""

    output_text: str = attrib(default="")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_response import (
            TerminalInferencerResponse,
        )
        return TerminalInferencerResponse(output=self.output_text)


class TestPostMortemFixes(unittest.TestCase):
    """Tests for the post-mortem fixes."""

    @staticmethod
    def _parse_winner_tag(s):
        import re
        m = re.search(r"<Winner>\s*flow_(\d+)\s*</Winner>", str(s))
        return int(m.group(1)) if m else None

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---- Fix 1c: MultiFlow yields workers + aggregator ---------------------

    def test_multi_flow_iter_child_inferencers(self):
        """MultiFlow._iter_child_inferencers yields aggregator + workers
        (deduplicated by id when same instance reused)."""
        agg = _make_sequential_inferencer(["agg_out"])
        f0_init = _make_sequential_inferencer(["f0_init"])
        f1_init = _make_sequential_inferencer(["f1_init"])
        # Same instance reused as both flows' followup — verify dedup
        shared_inf = _make_sequential_inferencer(["shared"])
        mfi = MultiFlowInferencer(
            flow_configs=[
                {"input": "t0", "initial_inferencer": f0_init,
                 "followup_inferencer": shared_inf,
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1", "initial_inferencer": f1_init,
                 "followup_inferencer": shared_inf,
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            aggregator_inferencer=agg,
            checkpoint_dir=self.tmpdir,
        )
        children = list(mfi._iter_child_inferencers())
        ids = {id(c) for c in children}
        self.assertIn(id(agg), ids)
        self.assertIn(id(f0_init), ids)
        self.assertIn(id(f1_init), ids)
        self.assertIn(id(shared_inf), ids)
        # shared_inf yielded only once (dedup)
        shared_yields = [c for c in children if c is shared_inf]
        self.assertEqual(len(shared_yields), 1)

    # ---- Fix 2: dispatch state lifetime ------------------------------------

    def test_dispatch_state_set_on_first_call(self):
        """Sanity: aggregator output with valid Winner tag → winner_idx set."""
        mfi = MultiFlowInferencer(
            flow_configs=[
                {"input": "t0",
                 "initial_inferencer": _make_sequential_inferencer(["p0"]),
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1",
                 "initial_inferencer": _make_sequential_inferencer(["p1"]),
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            aggregator_inferencer=_FixedOutputAgg(
                output="<FinalPlan>x</FinalPlan>\n<Winner>flow_0</Winner>"
            ),
            winner_parser=self._parse_winner_tag,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("master")
        self.assertEqual(mfi.get_winner_flow_idx(), 0)

    def test_dispatch_state_set_with_terminal_response_aggregator(self):
        """Real-CLI parity: aggregator returns a `TerminalInferencerResponse`
        (object with `.output` attr), NOT a plain string. This is exactly what
        `ClaudeCodeCliInferencer` / `KiroCliInferencer` return in real use.
        Verifies the post-mortem fix to `_normalize_aggregator_output` that
        prefers the LAST non-None tuple element (was: last string element,
        which dropped the response object in favor of the last worker's text)."""
        mfi = MultiFlowInferencer(
            flow_configs=[
                {"input": "t0",
                 "initial_inferencer": _make_sequential_inferencer(["p0"]),
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1",
                 "initial_inferencer": _make_sequential_inferencer(["p1"]),
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            aggregator_inferencer=_TerminalResponseAgg(
                output_text="<FinalPlan>x</FinalPlan>\n<Winner>flow_0</Winner>"
            ),
            winner_parser=self._parse_winner_tag,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("master")
        self.assertEqual(mfi.get_winner_flow_idx(), 0,
                         f"winner_idx should be 0, got {mfi.get_winner_flow_idx()}")

    def test_dispatch_state_resets_on_new_top_level_call(self):
        """Reset is once per top-level ainfer call. A second call's
        malformed output does NOT carry over winner from the first call."""
        mfi = MultiFlowInferencer(
            flow_configs=[
                {"input": "t0",
                 "initial_inferencer": _make_sequential_inferencer(["p0"]),
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
                {"input": "t1",
                 "initial_inferencer": _make_sequential_inferencer(["p1"]),
                 "followup_inferencer": _make_sequential_inferencer([]),
                 "end_condition": lambda s, r: True, "max_dynamic_steps": 1},
            ],
            aggregator_inferencer=_FixedOutputAgg(
                output="<FinalPlan>x</FinalPlan>\n<Winner>flow_0</Winner>"
            ),
            winner_parser=self._parse_winner_tag,
            checkpoint_dir=self.tmpdir,
        )
        mfi.infer("master")
        self.assertEqual(mfi.get_winner_flow_idx(), 0)

        # SECOND call: swap aggregator to malformed (no <Winner>), refresh
        # workers, fresh checkpoint dir to avoid cache hits.
        mfi.aggregator_inferencer = _FixedOutputAgg(
            output="<FinalPlan>plain plan, no winner tag</FinalPlan>"
        )
        for cfg in mfi.flow_configs:
            cfg["initial_inferencer"] = _make_sequential_inferencer(["p"])
        mfi.checkpoint_dir = tempfile.mkdtemp()
        mfi.infer("master2")
        # Reset on ainfer entry → previous winner cleared. Malformed
        # aggregator output → new winner not set. Result: None.
        self.assertIsNone(mfi.get_winner_flow_idx())


if __name__ == "__main__":
    unittest.main()

"""research_propose wiring of the per-flow ``bta_inferencer`` fan-out.

Loads the research_propose tool's ``breakdown-multiflow-plan`` config the way the
task executor does (tool defaults, env prefix, ``config_overrides``), then checks
the default (no fan-out), the env enable and kill switches, the materialized
fan-out template, and the prompts its breakdown and aggregator render. The task
tool's ``default.yaml`` imports the same config as its planner, so it is checked
too.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import agent_foundation.common.configs  # noqa: F401 — registers inferencer aliases
import agent_foundation.resources.tools.task as task_tool
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer import (
    CodexCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    DEFAULT_TOOL_CALL_BUDGET,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import _FreshCloneFactory
from agent_foundation.common.inferencers.run_context import open_invocation
from agent_foundation.resources.tools.registry import load_tool
from hydra.errors import InstantiationException
from omegaconf import OmegaConf
from rich_python_utils.config_utils import instantiate, load_config

_ENV_PREFIXES = ("RESEARCH_PROPOSE__", "TASK__")
# Pins the CLI inferencers to their default commands, skipping the binary probe
# each constructor otherwise runs as a subprocess.
_CLI_COMMAND_ENV = {"CLAUDE_CODE_COMMAND": "claude", "CODEX_COMMAND": "codex"}
_FLOW_INFERENCERS = "RESEARCH_PROPOSE__FLOW_INFERENCERS"
_FLOW_BTA = "RESEARCH_PROPOSE__FLOW_BTA_INFERENCERS"
_FLOW_FOLLOWUP_BTA = "RESEARCH_PROPOSE__FLOW_FOLLOWUP_BTA_INFERENCERS"
_METAMATE_LAST = '["ClaudeCodeCLI","DevmateCLI","CodexCLI","MetamateSDK"]'
_FAN_OUT_LAST = '[null,null,null,"${_params.metamate_fan_out_roles}"]'
_ENABLE_ENV = {_FLOW_INFERENCERS: _METAMATE_LAST, _FLOW_BTA: _FAN_OUT_LAST}
# The imported plan config's ``_params`` resolve under ``planner_inferencer``.
_TASK_ENABLE_ENV = {
    "TASK__FLOW_INFERENCERS": '["ClaudeCodeCLI","MetamateSDK"]',
    "TASK__FLOW_BTA_INFERENCERS": (
        '[null,"${planner_inferencer._params.metamate_fan_out_roles}"]'
    ),
}
_SLOTS = ("initial_inferencer", "followup_inferencer")
_ROLES = ("own", "fixer_inferencer")
_FENCE = "```json iteration_judgment"


def _patch_env(env: dict) -> patch.dict:
    values = {k: v for k, v in os.environ.items() if not k.startswith(_ENV_PREFIXES)}
    values.update(_CLI_COMMAND_ENV)
    values.update(env)
    return patch.dict(os.environ, values, clear=True)


def _load(workspace_root: str) -> tuple:
    """``(resolved config container, instantiated planner)`` for research_propose,
    loaded like the task executor under the current environment."""
    defaults = load_tool("research_propose").derived_from["defaults"]
    path = Path(task_tool.__file__).parent / "configs" / f"{defaults['config']}.yaml"
    cfg = load_config(
        str(path),
        overrides={
            "_params.workspace_root": workspace_root,
            "_template_master_version": defaults["template_master_version"],
        },
        env_prefix=defaults["env_prefix"],
        config_defaults=dict(defaults["config_overrides"]),
    )
    raw = OmegaConf.to_container(cfg, resolve=True)
    return raw, instantiate(OmegaConf.create(raw))


def _load_task_planner(workspace_root: str):
    """The planner of the task tool's ``default.yaml`` under the current
    environment."""
    path = Path(task_tool.__file__).parent / "configs" / "default.yaml"
    cfg = load_config(str(path), overrides={"_params.workspace_root": workspace_root})
    return instantiate(cfg).planner_inferencer


def _flows(planner) -> list:
    """The flow configs of the planner's MultiFlowDual workers."""
    return planner.base_inferencer.worker_inferencers().base_inferencer.flow_configs


def _raw_flows(raw: dict) -> list:
    return raw["base_inferencer"]["worker_inferencers"]["flow_configs"]


def _bta_specs(flows) -> list:
    return [[flow[slot].bta_inferencer for slot in _SLOTS] for flow in flows]


def _materialize(leaf):
    """The fan-out a call on ``leaf`` builds: materialized and its aggregator
    seeded, before any model call."""
    fanout = leaf._materialize_fanout(leaf._fanout_prototype(), {})
    leaf._seed_aggregator_feed(fanout)
    return fanout


def _render_aggregator(leaf, contract: str) -> str:
    fanout = _materialize(leaf)
    with open_invocation(fanout):
        fanout._open_attempt(contract, use_async=False)
        fanout._inject_aggregator_extra_feed(["WORKER_0_FINDINGS", "WORKER_1_FINDINGS"])
    return str(fanout.aggregator_inferencer.infer(contract, render_only=True))


class ResearchProposeFanoutDefaultTest(unittest.TestCase):
    def test_default_load_fans_out_no_flow(self) -> None:
        root = self.enterContext(tempfile.TemporaryDirectory())
        with _patch_env({}):
            raw, planner = _load(root)
            flows = _flows(planner)

        self.assertEqual(len(flows), 2)
        self.assertEqual(_bta_specs(flows), [[None, None], [None, None]])
        for flow in _raw_flows(raw):
            for slot in _SLOTS:
                self.assertIsNone(flow[slot]["bta_inferencer"])


class ResearchProposeFanoutEnabledTest(unittest.TestCase):
    """Flow 3 (MetamateSDK) fanned out through the enable env var."""

    @classmethod
    def setUpClass(cls) -> None:
        root = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls.enterClassContext(_patch_env(_ENABLE_ENV))
        cls.raw, planner = _load(root)
        cls.flows = _flows(planner)

    def test_enable_maps_only_the_metamate_flow(self) -> None:
        specs = _bta_specs(self.flows)
        self.assertEqual(specs[:3], [[None, None]] * 3)
        for slot, spec in zip(_SLOTS, specs[3]):
            self.assertIsInstance(self.flows[3][slot], MetamateSDKInferencer)
            self.assertEqual(set(spec), set(_ROLES))
            proto = self.flows[3][slot]._fanout_prototype()
            self.assertIsInstance(proto, BreakdownThenAggregateInferencer)
            self.assertEqual(proto.expected_parent_types, ("MetamateSDK",))

    def test_list_valued_template_fields_are_not_distributed(self) -> None:
        for slot in _SLOTS:
            raw_spec = _raw_flows(self.raw)[3][slot]["bta_inferencer"]
            for role in _ROLES:
                template = raw_spec[role]
                self.assertEqual(
                    template["worker_query_fields"], ["description", "scope", "todos"]
                )
                self.assertEqual(template["expected_parent_types"], ["MetamateSDK"])

    def test_each_site_and_role_holds_its_own_template(self) -> None:
        templates = [
            self.flows[3][slot].bta_inferencer[role]
            for slot in _SLOTS
            for role in _ROLES
        ]
        self.assertEqual(len({id(t) for t in templates}), len(templates))
        protos = [template() for template in templates]
        for attr in ("breakdown_inferencer", "aggregator_inferencer"):
            children = {id(getattr(proto, attr)) for proto in protos}
            self.assertEqual(len(children), len(protos), attr)

    def test_each_call_builds_fresh_slots(self) -> None:
        leaf = self.flows[3]["initial_inferencer"]
        proto = leaf._fanout_prototype()
        first = leaf._materialize_fanout(proto, {})
        second = leaf._materialize_fanout(proto, {})
        for attr in ("breakdown_inferencer", "aggregator_inferencer"):
            self.assertIsNot(getattr(first, attr), getattr(second, attr), attr)
            self.assertIsNot(getattr(first, attr), getattr(proto, attr), attr)

    def test_materialized_fan_out_saves_and_resumes_its_results(self) -> None:
        for slot in _SLOTS:
            fanout = _materialize(self.flows[3][slot])
            self.assertIs(fanout.enable_result_save, True, slot)
            self.assertIs(fanout.resume_with_saved_results, True, slot)

    def test_materialized_breakdown_gets_injectables_and_slot_defaults(self) -> None:
        leaf = self.flows[3]["initial_inferencer"]
        breakdown = _materialize(leaf).breakdown_inferencer

        self.assertIsInstance(breakdown, ClaudeCodeCliInferencer)
        # The config walk builds the ``_template_manager`` injectable per site.
        manager = breakdown.template_manager
        self.assertIsInstance(manager, type(leaf.template_manager))
        self.assertEqual(
            manager.default_template_key,
            self.raw["_template_manager"]["default_template_key"],
        )
        self.assertEqual(manager.templates, leaf.template_manager.templates)
        self.assertEqual(breakdown.template_root_space, "task_breakdown")
        self.assertEqual(breakdown.template_master_version, "research_propose")
        self.assertEqual(
            breakdown.template_variables, {"task_instructions": "research_exploration"}
        )
        self.assertEqual(breakdown.model_name, self.raw["_model_name"])
        self.assertEqual(breakdown.debug_mode, self.raw["_debug_mode"])
        self.assertEqual(
            breakdown.template_extra_feed,
            {
                "enable_deep_mode": True,
                "enable_elegant_mode": True,
                "max_breakdown": 3,
                "max_worker_query_chars": 4096,
            },
        )
        self.assertEqual(
            [spec["label"] for spec in breakdown.expected_extraction],
            ["decomposed_subtasks"],
        )
        self.assertIsNone(breakdown.bta_inferencer)

    def test_materialized_aggregator_uses_plain_aggregation_framing(self) -> None:
        leaf = self.flows[3]["initial_inferencer"]
        aggregator = _materialize(leaf).aggregator_inferencer

        self.assertIsInstance(aggregator, ClaudeCodeCliInferencer)
        self.assertEqual(aggregator.template_root_space, "plan")
        self.assertEqual(aggregator.template_version, "aggregation")
        self.assertEqual(aggregator.template_master_version, "aggregation")
        self.assertEqual(aggregator.template_variables, {})
        self.assertFalse(aggregator.modes["deep_mode"])
        self.assertIsInstance(
            aggregator.output_guardrail_inferencer, ClaudeCodeCliInferencer
        )
        self.assertIsNone(aggregator.bta_inferencer)

    def test_workers_are_fresh_copies_of_the_leaf(self) -> None:
        fanout = _materialize(self.flows[3]["initial_inferencer"])

        self.assertIsInstance(fanout.worker_inferencers, _FreshCloneFactory)
        worker = fanout.worker_inferencers.prototype
        self.assertIsInstance(worker, MetamateSDKInferencer)
        self.assertIsNone(worker.bta_inferencer)
        self.assertEqual(
            fanout.worker_inference_args,
            {"tool_call_budget": 6, "max_answer_findings": 8, "prepared_input": True},
        )

    def test_workers_get_the_budget_measured_on_shards(self) -> None:
        """The template pins each worker call's budget and answer-format cap to
        the values measured on shards; the leaf keeps MetaMate's defaults (budget
        on, cap off) for its own unfanned turns."""
        leaf = self.flows[3]["initial_inferencer"]
        self.assertEqual(leaf.tool_call_budget, DEFAULT_TOOL_CALL_BUDGET)
        self.assertIsNone(leaf.max_answer_findings)
        fanout = _materialize(leaf)
        self.assertEqual(fanout.worker_inference_args["tool_call_budget"], 6)
        self.assertEqual(fanout.worker_inference_args["max_answer_findings"], 8)

    def test_followup_aggregator_adds_one_iteration_judgment_fence(self) -> None:
        leaf = self.flows[3]["followup_inferencer"]
        contract = str(leaf.infer("FOLLOWUP_STEP_INPUT", render_only=True))
        prompt = _render_aggregator(leaf, contract)
        own_framing = prompt.replace(contract, "")

        self.assertEqual(prompt.count(contract), 1)
        self.assertEqual(contract.count(_FENCE), 1)
        self.assertEqual(own_framing.count(_FENCE), 1)
        self.assertIn("You are aggregating/integrating", own_framing)
        self.assertIn("WORKER_1_FINDINGS", own_framing)
        self.assertNotIn("proposal_index", own_framing)

    def test_initial_aggregator_adds_no_iteration_judgment_fence(self) -> None:
        leaf = self.flows[3]["initial_inferencer"]
        contract = str(leaf.infer("INITIAL_STEP_INPUT", render_only=True))
        own_framing = _render_aggregator(leaf, contract).replace(contract, "")

        self.assertNotIn(_FENCE, own_framing)
        self.assertIn("You are aggregating/integrating", own_framing)
        self.assertNotIn("proposal_index", own_framing)

    def test_breakdown_renders_the_research_exploration_instructions(self) -> None:
        leaf = self.flows[3]["initial_inferencer"]
        contract = str(leaf.infer("INITIAL_STEP_INPUT", render_only=True))
        breakdown = _materialize(leaf).breakdown_inferencer
        prompt = str(breakdown.infer(contract, render_only=True))
        own_framing = prompt.replace(contract, "")

        self.assertEqual(prompt.count(contract), 1)
        self.assertIn("**research/exploration** task", own_framing)
        self.assertIn("under 4096 characters", own_framing)
        self.assertIn("into 3 focused subtasks", own_framing)
        self.assertNotIn("execution/implementation", own_framing)
        self.assertNotIn("The worker will read the full plan", own_framing)
        lines = [line.strip() for line in own_framing.splitlines()]
        self.assertNotIn("research_exploration", lines)


class ResearchProposeFanoutEnvSwitchTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.root = self.enterContext(tempfile.TemporaryDirectory())

    def test_followup_kill_switch_keeps_initial_fan_out(self) -> None:
        with _patch_env({**_ENABLE_ENV, _FLOW_FOLLOWUP_BTA: "null"}):
            _, planner = _load(self.root)
            specs = _bta_specs(_flows(planner))

        self.assertEqual([followup for _, followup in specs], [None] * 4)
        self.assertEqual([initial for initial, _ in specs[:3]], [None] * 3)
        self.assertEqual(set(specs[3][0]), set(_ROLES))

    def test_list_syntax_on_followup_key_fails_loud(self) -> None:
        with _patch_env({**_ENABLE_ENV, _FLOW_FOLLOWUP_BTA: "[null]"}):
            with self.assertRaises(InstantiationException) as raised:
                _flows(_load(self.root)[1])

        cause = raised.exception.__cause__
        self.assertIsInstance(cause, TypeError)
        self.assertIn("bta_inferencer", str(cause))
        self.assertIn("'[null]'", str(cause))

    async def test_misaligned_index_fails_before_any_model_call(self) -> None:
        misaligned = '["ClaudeCodeCLI","MetamateSDK","CodexCLI","DevmateCLI"]'
        with _patch_env({_FLOW_INFERENCERS: misaligned, _FLOW_BTA: _FAN_OUT_LAST}):
            _, planner = _load(self.root)
            leaf = _flows(planner)[3]["initial_inferencer"]
            self.assertIsInstance(leaf, DevmateCliInferencer)
            with patch.object(DevmateCliInferencer, "_ainfer", new=AsyncMock()) as call:
                with self.assertRaisesRegex(TypeError, "expected_parent_types"):
                    await leaf.ainfer("REQUEST")

        call.assert_not_awaited()

    def test_flow_count_and_inferencer_overrides_apply_with_fan_out(self) -> None:
        with _patch_env({**_ENABLE_ENV, "RESEARCH_PROPOSE__NUM_FLOWS": "5"}):
            _, planner = _load(self.root)
            flows = _flows(planner)

        self.assertEqual(
            [type(flow["initial_inferencer"]) for flow in flows],
            [
                ClaudeCodeCliInferencer,
                DevmateCliInferencer,
                CodexCliInferencer,
                MetamateSDKInferencer,
                ClaudeCodeCliInferencer,
            ],
        )
        specs = _bta_specs(flows)
        self.assertEqual([specs[i] for i in (0, 1, 2, 4)], [[None, None]] * 4)
        self.assertEqual([set(spec) for spec in specs[3]], [set(_ROLES)] * 2)


class TaskToolFanoutConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = self.enterContext(tempfile.TemporaryDirectory())

    def test_default_load_fans_out_no_flow(self) -> None:
        with _patch_env({}):
            flows = _flows(_load_task_planner(self.root))

        self.assertEqual(_bta_specs(flows), [[None, None], [None, None]])

    def test_enable_maps_only_the_metamate_flow(self) -> None:
        with _patch_env(_TASK_ENABLE_ENV):
            flows = _flows(_load_task_planner(self.root))

        specs = _bta_specs(flows)
        self.assertEqual(specs[0], [None, None])
        for slot, spec in zip(_SLOTS, specs[1]):
            self.assertIsInstance(flows[1][slot], MetamateSDKInferencer)
            self.assertEqual(set(spec), set(_ROLES))
            proto = flows[1][slot]._fanout_prototype()
            self.assertIsInstance(proto, BreakdownThenAggregateInferencer)
            self.assertEqual(proto.expected_parent_types, ("MetamateSDK",))

    def test_breakdown_renders_the_research_exploration_instructions(self) -> None:
        with _patch_env(_TASK_ENABLE_ENV):
            leaf = _flows(_load_task_planner(self.root))[1]["initial_inferencer"]
            contract = str(leaf.infer("INITIAL_STEP_INPUT", render_only=True))
            breakdown = _materialize(leaf).breakdown_inferencer
            prompt = str(breakdown.infer(contract, render_only=True))
        own_framing = prompt.replace(contract, "")

        self.assertEqual(prompt.count(contract), 1)
        self.assertIn("**research/exploration** task", own_framing)
        self.assertNotIn("execution/implementation", own_framing)
        lines = [line.strip() for line in own_framing.splitlines()]
        self.assertNotIn("research_exploration", lines)

"""Per-call BTA fan-out through ``InferencerBase.bta_inferencer``.

A fanned-out inferencer ("P") renders its contract once, hands it to a fresh
per-call ``BreakdownThenAggregateInferencer`` whose workers are copies of P,
and runs its own boundary (post-processor, expected extraction, output
promotion) once on the aggregated result. Covers routing, per-call
materialization, role selection (MultiFlowDual, MultiFlowInferencer and a
standalone ``switch_role``), feed isolation, kwargs projection, streaming,
Metamate conversations, lifecycle, capability gating, validation, resume and
the structured records.
"""

import asyncio
import contextlib
import functools
import hashlib
import importlib
import json
import os
import pkgutil
import re
import shutil
import tempfile
import unittest
import uuid
from unittest.mock import patch

import agent_foundation.common.configs  # noqa: F401 — registers inferencer aliases
import agent_foundation.common.inferencers as inferencers_pkg
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer import (
    CodexCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer import (
    KiroCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer import (
    OpenClawInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import (
    _FreshCloneFactory,
    ACTOR_SCOPED_FEED_KEYS,
    BTA_INFERENCER_SLOT,
    InferencerBase,
    MAX_BTA_FANOUT_DEPTH,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    FallbackInferMode,
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.template_feed_scope import (
    TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory

_CONTRACT = "[plan|initial|None] task"
_COMPLETED_CACHE = "STALE\n--- STREAM COMPLETED SUCCESSFULLY ---\n"


class _Recorder:
    """State shared by a scripted leaf and every copy of it: the calls, renders
    and disconnects of all copies, scripted failures, breakdowns and review
    verdicts."""

    def __init__(self):
        self.calls = []
        self.renders = []
        self.disconnects = []
        self.fail = {}
        self.breakdowns = []
        self.reviews = []

    def resume_identity(self):
        # A scripted backend: a resume continues the same script.
        return "scripted recorder"

    def __deepcopy__(self, memo):
        return self

    def prompts(self):
        return [prompt for _, prompt, _ in self.calls]

    def worker_calls(self):
        return [call for call in self.calls if not call[1].startswith("[")]

    def respond(self, prompt):
        """Scripted answer by prompt header; raises while ``fail`` has budget
        left for a marker the prompt contains."""
        for marker, remaining in self.fail.items():
            if remaining and marker in prompt:
                self.fail[marker] = remaining - 1
                raise RuntimeError(f"{marker} boom")
        head = prompt[: prompt.index("]") + 1] if prompt.startswith("[") else ""
        if "|review|" in head:
            return self.reviews.pop(0)
        if head.startswith("[task_breakdown|"):
            if self.breakdowns:
                return self.breakdowns.pop(0)
            return "1. alpha shard\n2. beta shard"
        if "|aggregation" in head:
            return "AGGREGATED"
        return f"W({prompt})"


class _TemplateManager:
    """Renders ``[root|key|master] input`` (a recovery prompt as
    ``RECOVER(partial)``) and records each render's key, root, master version
    and feed."""

    def __init__(self, recorder):
        self.recorder = recorder

    def resume_identity(self):
        # The recorder is runtime state; the rendering rule is the identity.
        return "[root|key|master] input"

    def __call__(
        self, key, *, active_template_root_space=None, master_version=None, **feed
    ):
        self.recorder.renders.append(
            (key, active_template_root_space, master_version, dict(feed))
        )
        if "agent_response" in feed:
            return f"RECOVER({feed['agent_response']})"
        return (
            f"[{active_template_root_space}|{key}|{master_version}] {feed.get('input')}"
        )

    def get_raw_template(self, *args, **kwargs):
        return "x"

    def load_variables(self, *args, **kwargs):
        return {}

    def add_template_root(self, *args, **kwargs):
        pass

    def __deepcopy__(self, memo):
        return self


@attrs(slots=False)
class ScriptedLeaf(TemplatedInferencerBase):
    """Templated leaf answering through its shared ``recorder``."""

    recorder = attrib(default=None, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        prompt = str(inference_input)
        self.recorder.calls.append((self, prompt, kwargs))
        return self.recorder.respond(prompt)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)

    async def adisconnect(self):
        self.recorder.disconnects.append(self)


@attrs
class Plain(InferencerBase):
    """Non-templated leaf returning ``response``; records its inputs in ``calls``."""

    response = attrib(default="plain")
    calls = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.calls.append(inference_input)
        return self.response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)


def _bta(**kwargs):
    return BreakdownThenAggregateInferencer(
        fallback_mode=FallbackMode.NEVER, max_retry=0, **kwargs
    )


@contextlib.contextmanager
def _logged(cls=InferencerBase):
    """Records ``(instance, log_type, log_item)`` for every ``log_info`` on an
    instance of ``cls``, still logging it."""
    records = []
    original = cls.log_info

    def spy(self, log_item, log_type=None, *args, **kwargs):
        records.append((self, log_type, log_item))
        return original(self, log_item, log_type, *args, **kwargs)

    with patch.object(cls, "log_info", spy):
        yield records


def _of_type(records, log_type):
    return [(inst, item) for inst, kind, item in records if kind == log_type]


class _FanoutMixin:
    def setUp(self):
        super().setUp()
        self.tmp = self.make_tmp()
        self.rec = _Recorder()

    def make_tmp(self):
        path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def ws(self):
        return InferencerWorkspace(root=self.tmp)

    def leaf(self, root="plan", key="initial", **kwargs):
        kwargs.setdefault("fallback_mode", FallbackMode.NEVER)
        kwargs.setdefault("max_retry", 0)
        return ScriptedLeaf(
            template_manager=_TemplateManager(self.rec),
            template_root_space=root,
            template_key=key,
            recorder=self.rec,
            **kwargs,
        )

    def fanned(self, template=None, **kwargs):
        """A workspace-bound P fanning out through ``template`` (default: an
        all-blank BTA)."""
        return self.leaf(
            workspace=self.ws(),
            bta_inferencer=_bta() if template is None else template,
            **kwargs,
        )

    def assert_fanned_out(self, p, result):
        self.assertEqual(result, "AGGREGATED")
        prompts = self.rec.prompts()
        self.assertEqual(len(prompts), 4)
        self.assertEqual(prompts[0], f"[task_breakdown||None] {_CONTRACT}")
        self.assertEqual(sorted(prompts[1:3]), ["alpha shard", "beta shard"])
        self.assertTrue(prompts[3].startswith(f"[plan||aggregation] {_CONTRACT}"))
        for inst, _, _ in self.rec.calls:
            self.assertIs(type(inst), ScriptedLeaf)
            self.assertIsNot(inst, p)


# ---------------------------------------------------------------------------
# Routing: the seam after render, before the backend
# ---------------------------------------------------------------------------


class FanoutRoutingTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    async def test_call_runs_through_a_per_call_bta_of_p_copies(self):
        p = self.fanned()
        self.assert_fanned_out(p, await p.ainfer("task"))

    async def test_no_template_calls_the_backend(self):
        p = self.leaf(workspace=self.ws())
        self.assertEqual(await p.ainfer("task"), f"W({_CONTRACT})")
        self.assertEqual(
            [(i is p, s) for i, s, _ in self.rec.calls], [(True, _CONTRACT)]
        )

    async def test_render_only_returns_the_contract(self):
        p = self.fanned()
        self.assertEqual(await p.ainfer("task", render_only=True), _CONTRACT)
        self.assertEqual(self.rec.calls, [])

    async def test_contract_renders_once_and_feeds_the_breakdown(self):
        await self.fanned().ainfer("task")
        self.assertEqual(
            [render[:3] for render in self.rec.renders],
            [
                ("initial", "plan", None),
                ("", "task_breakdown", None),
                ("", "plan", "aggregation"),
            ],
        )
        self.assertEqual(self.rec.renders[1][3]["input"], _CONTRACT)

    async def test_workers_see_only_shards_and_p_boundary_runs_once(self):
        pre, post = [], []
        p = self.fanned(
            input_preprocessor=lambda x: pre.append(x) or x,
            response_post_processor=lambda r: post.append(r) or f"POST({r})",
            expected_extraction=[
                {"label": "iteration_judgment", "source": "response", "kind": "control"}
            ],
        )
        original = ScriptedLeaf._run_expected_extraction
        with patch.object(
            ScriptedLeaf,
            "_run_expected_extraction",
            autospec=True,
            side_effect=original,
        ) as extraction:
            result = await p.ainfer("task SENTINEL")

        self.assertEqual(result, "POST(AGGREGATED)")
        self.assertEqual((pre, post), (["task SENTINEL"], ["AGGREGATED"]))
        owners = [call.args[0] for call in extraction.call_args_list]
        self.assertEqual(len(owners), 2)
        self.assertEqual(sum(owner is p for owner in owners), 1)
        self.assertEqual(len(self.rec.renders), 3)
        workers = [prompt for _, prompt, _ in self.rec.worker_calls()]
        self.assertEqual(sorted(workers), ["alpha shard", "beta shard"])

    async def test_predefined_sub_queries_skip_the_breakdown(self):
        p = self.fanned(_bta(predefined_sub_queries=["alpha shard", "beta shard"]))
        self.assertEqual(await p.ainfer("task"), "AGGREGATED")
        prompts = self.rec.prompts()
        self.assertEqual(sorted(prompts[:2]), ["alpha shard", "beta shard"])
        self.assertTrue(prompts[2].startswith(f"[plan||aggregation] {_CONTRACT}"))
        self.assertEqual(len(prompts), 3)

    async def test_disabled_aggregator_runs_no_aggregation(self):
        p = self.fanned(_bta(disable_aggregator=True), output_path="plan.md")
        result = await p.ainfer("task")
        prompts = self.rec.prompts()
        self.assertEqual(prompts[0], f"[task_breakdown||None] {_CONTRACT}")
        self.assertEqual(sorted(prompts[1:]), ["alpha shard", "beta shard"])
        self.assertEqual(result, "W(beta shard)")
        with open(os.path.join(self.tmp, "outputs", "plan.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), result)


class FanoutSyncRoutingTest(_FanoutMixin, unittest.TestCase):
    def test_sync_call_runs_through_a_per_call_bta_of_p_copies(self):
        p = self.fanned()
        self.assert_fanned_out(p, p.infer("task"))

    def test_sync_no_template_calls_the_backend(self):
        p = self.leaf(workspace=self.ws())
        self.assertEqual(p.infer("task"), f"W({_CONTRACT})")
        self.assertEqual(
            [(i is p, s) for i, s, _ in self.rec.calls], [(True, _CONTRACT)]
        )


# ---------------------------------------------------------------------------
# Materialization: slots, workers, P copies, the template itself
# ---------------------------------------------------------------------------


class FanoutMaterializationTest(_FanoutMixin, unittest.TestCase):
    def materialize(self, p):
        return p._materialize_fanout(p._fanout_prototype(), {})

    def guarded(self, template=None):
        return self.fanned(
            template,
            output_guardrail_inferencer=Plain(response="ok"),
            fallback_inferencer=Plain(),
            max_retry=3,
            input_preprocessor=str.strip,
        )

    def test_blank_slots_become_role_clones(self):
        p = self.guarded()
        fanout = self.materialize(p)
        breakdown, aggregator = (
            fanout.breakdown_inferencer,
            fanout.aggregator_inferencer,
        )

        self.assertEqual(
            (breakdown.template_root_space, breakdown.template_key),
            ("task_breakdown", ""),
        )
        self.assertIsNone(breakdown.output_guardrail_inferencer)
        self.assertIsNone(breakdown.fallback_inferencer)
        self.assertEqual(
            [spec["label"] for spec in breakdown.expected_extraction],
            ["decomposed_subtasks"],
        )
        self.assertEqual(breakdown.template_extra_feed, {})
        self.assertEqual(
            (
                aggregator.template_root_space,
                aggregator.template_key,
                aggregator.template_version,
                aggregator.template_master_version,
            ),
            ("plan", "", "aggregation", "aggregation"),
        )
        self.assertIs(aggregator.modes["deep_mode"], False)
        self.assertIsInstance(aggregator.output_guardrail_inferencer, Plain)
        self.assertIsNot(
            aggregator.output_guardrail_inferencer, p.output_guardrail_inferencer
        )
        self.assertEqual(
            p._fanout_slot_sources(p._fanout_prototype()),
            {"breakdown_source": "p_clone", "aggregator_source": "p_clone"},
        )

    def test_workers_keep_p_settings_and_every_p_copy_is_unbound(self):
        p = self.guarded()
        fanout = self.materialize(p)
        self.assertIsInstance(fanout.worker_inferencers, _FreshCloneFactory)
        worker = fanout.worker_inferencers.prototype

        self.assertEqual(worker.max_retry, 3)
        self.assertIsInstance(worker.output_guardrail_inferencer, Plain)
        self.assertIsNot(
            worker.output_guardrail_inferencer, p.output_guardrail_inferencer
        )
        self.assertIsInstance(worker.fallback_inferencer, Plain)
        self.assertIs(fanout.worker_inference_args["prepared_input"], True)
        self.assertIsNone(fanout._workspace)
        for copy in (worker, fanout.breakdown_inferencer, fanout.aggregator_inferencer):
            self.assertIsNot(copy, p)
            self.assertIsNone(copy.bta_inferencer)
            self.assertIsNone(copy._workspace)
            self.assertIsNone(copy.input_preprocessor)

    def test_explicit_slots_are_rebuilt_as_configured(self):
        breakdown = self.leaf("task_breakdown")
        aggregator = self.leaf(template_master_version="aggregation")
        template = _bta(
            breakdown_inferencer=breakdown, aggregator_inferencer=aggregator
        )
        p = self.guarded(template)
        fanout = self.materialize(p)

        built_breakdown, built_aggregator = (
            fanout.breakdown_inferencer,
            fanout.aggregator_inferencer,
        )
        self.assertIsNot(built_breakdown, breakdown)
        self.assertEqual(
            (built_breakdown.template_root_space, built_breakdown.template_key),
            ("task_breakdown", "initial"),
        )
        self.assertIsNot(built_aggregator, aggregator)
        self.assertEqual(built_aggregator.template_master_version, "aggregation")
        self.assertIsNone(built_aggregator.output_guardrail_inferencer)
        self.assertEqual(
            p._fanout_slot_sources(template),
            {"breakdown_source": "explicit", "aggregator_source": "explicit"},
        )

    def test_factory_aggregator_is_built_before_seeding(self):
        factory = LazyConfigFactory(
            {
                "_target_": f"{ScriptedLeaf.__module__}.ScriptedLeaf",
                "template_master_version": "aggregation",
                "template_extra_feed": {"include_iteration_judgment": False},
            }
        )
        template = _bta(aggregator_inferencer=factory)
        p = self.fanned(
            template,
            template_extra_feed={"include_iteration_judgment": True, "peer_flag": 1},
        )
        fanout = self.materialize(p)
        seeded = p._seed_aggregator_feed(fanout)

        self.assertIsInstance(fanout.aggregator_inferencer, ScriptedLeaf)
        self.assertEqual(
            fanout.aggregator_inferencer.template_extra_feed,
            {"include_iteration_judgment": False, "peer_flag": 1},
        )
        self.assertEqual(seeded, ["peer_flag"])
        self.assertIs(template.aggregator_inferencer, factory)
        self.assertEqual(factory.template_extra_feed, {})

    def test_disabled_aggregator_stays_disabled(self):
        template = _bta(disable_aggregator=True)
        p = self.fanned(template)
        fanout = self.materialize(p)
        self.assertIsNone(fanout.aggregator_inferencer)
        self.assertIsInstance(fanout.breakdown_inferencer, ScriptedLeaf)
        self.assertEqual(
            p._fanout_slot_sources(template),
            {"breakdown_source": "p_clone", "aggregator_source": "disabled"},
        )

    def test_predefined_sub_queries_leave_the_breakdown_blank(self):
        template = _bta(predefined_sub_queries=["alpha shard"])
        p = self.fanned(template)
        fanout = self.materialize(p)
        self.assertIsNone(fanout.breakdown_inferencer)
        self.assertIsInstance(fanout.aggregator_inferencer, ScriptedLeaf)
        self.assertEqual(
            p._fanout_slot_sources(template),
            {"breakdown_source": "predefined", "aggregator_source": "p_clone"},
        )


class FanoutTemplateTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    async def test_template_workers_are_overridden_with_a_warning(self):
        foreign = Plain(response="FOREIGN")
        p = self.fanned(_bta(worker_inferencers=foreign))
        with _logged() as records, self.assertWarns(UserWarning):
            result = await p.ainfer("task")

        self.assertEqual(result, "AGGREGATED")
        self.assertEqual(foreign.calls, [])
        workers = [(type(i), s) for i, s, _ in self.rec.worker_calls()]
        self.assertEqual(
            sorted(workers, key=str),
            [(ScriptedLeaf, "alpha shard"), (ScriptedLeaf, "beta shard")],
        )
        [(_, record)] = _of_type(records, "BtaFanOut")
        self.assertIs(record["worker_template_overridden"], True)

    async def test_template_is_never_bound_run_walked_or_mutated(self):
        breakdown = self.leaf("task_breakdown")
        aggregator = self.leaf(template_master_version="aggregation")
        template = _bta(
            breakdown_inferencer=breakdown, aggregator_inferencer=aggregator
        )
        p = self.fanned(
            template, template_extra_feed={"include_iteration_judgment": True}
        )
        for _ in range(2):
            self.assertEqual(await p.ainfer("task"), "AGGREGATED")

        self.assertIsNone(template._workspace)
        self.assertIsNone(template.worker_inferencers)
        self.assertEqual(template.worker_inference_args, {})
        self.assertIs(template.breakdown_inferencer, breakdown)
        self.assertIs(template.aggregator_inferencer, aggregator)
        self.assertEqual(aggregator.template_extra_feed, {})
        touched = [i for i, _, _ in self.rec.calls] + self.rec.disconnects
        for node in (breakdown, aggregator):
            self.assertIsNone(node._workspace)
            self.assertFalse(any(t is node for t in touched))
        self.assertFalse(any(c is template for c in p._iter_child_inferencers()))

    async def test_explicit_node_keeps_its_own_template(self):
        aggregator = self.leaf(
            template_master_version="aggregation", bta_inferencer=_bta()
        )
        p = self.fanned(_bta(aggregator_inferencer=aggregator))
        with _logged() as records:
            result = await p.ainfer("task")

        self.assertEqual(result, "AGGREGATED")
        depths = [(i is p, item["depth"]) for i, item in _of_type(records, "BtaFanOut")]
        self.assertEqual(depths, [(True, 0), (False, 1)])

    def nested_context(self, segments):
        ctx = RunContext.root(self.ws())
        for _ in range(segments):
            ctx = ctx.child(BTA_INFERENCER_SLOT)
        return ctx

    async def test_depth_limit_raises_before_any_model_call(self):
        p = self.fanned()
        with self.assertRaisesRegex(RuntimeError, "MAX_BTA_FANOUT_DEPTH"):
            await p.ainfer(
                "task", run_context=self.nested_context(MAX_BTA_FANOUT_DEPTH)
            )
        self.assertEqual(self.rec.calls, [])

        ctx = self.nested_context(MAX_BTA_FANOUT_DEPTH - 1)
        self.assertEqual(await p.ainfer("task", run_context=ctx), "AGGREGATED")


# ---------------------------------------------------------------------------
# Output promotion and the leaf finalize
# ---------------------------------------------------------------------------


class FanoutOutputTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    async def test_aggregated_result_replaces_a_stale_canonical_output(self):
        p = self.fanned(output_path="plan.md")
        canonical = os.path.join(self.tmp, "outputs", p.output_path)
        os.makedirs(os.path.dirname(canonical), exist_ok=True)
        with open(canonical, "w", encoding="utf-8") as f:
            f.write("STALE")

        await p.ainfer("task")

        self.assertTrue(os.path.islink(canonical))
        aggregator_output = os.path.join(
            BTA_INFERENCER_SLOT, "children", "aggregator", "outputs", p.output_path
        )
        self.assertTrue(os.path.realpath(canonical).endswith(aggregator_output))
        with open(canonical, encoding="utf-8") as f:
            self.assertEqual(f.read(), "AGGREGATED")

    async def test_orchestrator_host_finalizes_as_a_leaf(self):
        p = _bta(
            breakdown_inferencer=self.leaf("task_breakdown"),
            aggregator_inferencer=self.leaf(template_master_version="aggregation"),
            worker_inferencers=self.leaf(),
            workspace=self.ws(),
            bta_inferencer=_bta(
                breakdown_inferencer=self.leaf("task_breakdown"),
                aggregator_inferencer=self.leaf(template_master_version="aggregation"),
            ),
        )
        self.assertEqual(await p.ainfer("task"), "AGGREGATED")
        self.assertTrue(
            os.path.isfile(os.path.join(self.tmp, "outputs", p.output_path))
        )
        # The host never ran as a BTA: no run summary of its own.
        self.assertIsNone(p.last_call_summary)


# ---------------------------------------------------------------------------
# Role selection follows the render (I8)
# ---------------------------------------------------------------------------


def _verdict(approved):
    issues = [
        {
            "severity": "MAJOR",
            "category": "logic",
            "description": "gap",
            "location": "n/a",
            "suggestion": "fix",
        }
    ]
    body = {
        "approved": approved,
        "severity": "MINOR" if approved else "MAJOR",
        "issues": [] if approved else issues,
        "reasoning": "r",
    }
    return f"```json\n{json.dumps(body)}\n```"


_DUAL_AGGREGATION = (
    "<FinalPlan>\nintegrated\n</FinalPlan>\n"
    '```json winner_pick\n{"winner_index": 0, "reason": "t"}\n```\n'
    '```json ranking\n{"ranking": [0, 1], "reason": "t"}\n```'
)


class FanoutRoleSelectionTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    def switched(self, p, name, role, key):
        """Switch ``p`` to ``role`` at ``root/name`` as MultiFlowDual does;
        returns ``(root, role_context)``."""
        root = RunContext.root(self.ws())
        ctx = root.child(name)
        token = enter_run(ctx)
        try:
            p.switch_role(new_role=role, template_key=key)
        finally:
            exit_run(token)
        return root, ctx

    async def fixer_under_context(self, p):
        _, fix = self.switched(p, "fix", "fixer_inferencer", "followup")
        return await p.ainfer("task", run_context=fix)

    async def reviewer_under_context(self, p):
        _, review = self.switched(p, "review", "review_inferencer", "review")
        return await p.ainfer("task", run_context=review)

    async def propose_after_fix(self, p):
        root, _ = self.switched(p, "fix", "fixer_inferencer", "followup")
        return await p.ainfer("task", run_context=root.child("round_02"))

    async def bare_call_after_fix(self, p):
        self.switched(p, "fix", "fixer_inferencer", "followup")
        return await p.ainfer("task")

    async def standalone_switch(self, p):
        p.switch_role("fixer_inferencer", template_key="followup")
        return await p.ainfer("task")

    async def assert_selection(self, scenario, mapping, *, fans, role):
        with self.subTest(scenario=scenario.__name__, mapping=mapping):
            self.tmp, self.rec = self.make_tmp(), _Recorder()
            self.rec.reviews = ["VERDICT"]
            p = self.fanned({r: _bta() if on else None for r, on in mapping.items()})
            with _logged() as records:
                await scenario(p)
            fanned = [it["role"] for i, it in _of_type(records, "BtaFanOut") if i is p]
            bypassed = [
                it["role"] for i, it in _of_type(records, "BtaFanOutBypassed") if i is p
            ]
            self.assertEqual((fanned, bypassed), ([role], []) if fans else ([], [role]))
            self.assertEqual(any(i is p for i, _, _ in self.rec.calls), not fans)

    async def test_fixer_under_a_context(self):
        both = {"own": True, "fixer_inferencer": True}
        await self.assert_selection(
            self.fixer_under_context, both, fans=True, role="fixer_inferencer"
        )
        await self.assert_selection(
            self.fixer_under_context, {"own": True}, fans=False, role="fixer_inferencer"
        )

    async def test_reviewer_under_a_context(self):
        both = {"own": True, "fixer_inferencer": True}
        disabled = {"own": True, "review_inferencer": False}
        await self.assert_selection(
            self.reviewer_under_context, both, fans=False, role="review_inferencer"
        )
        await self.assert_selection(
            self.reviewer_under_context, disabled, fans=False, role="review_inferencer"
        )

    async def test_later_propose_under_a_context_is_own(self):
        await self.assert_selection(
            self.propose_after_fix, {"fixer_inferencer": True}, fans=False, role="own"
        )
        await self.assert_selection(
            self.propose_after_fix, {"own": True}, fans=True, role="own"
        )

    async def test_bare_call_after_a_context_switch_is_own(self):
        await self.assert_selection(
            self.bare_call_after_fix, {"own": True}, fans=True, role="own"
        )
        await self.assert_selection(
            self.bare_call_after_fix, {"fixer_inferencer": True}, fans=False, role="own"
        )

    async def test_standalone_switch_then_bare_call(self):
        fixer = {"fixer_inferencer": True}
        await self.assert_selection(
            self.standalone_switch, fixer, fans=True, role="fixer_inferencer"
        )
        await self.assert_selection(
            self.standalone_switch, {"own": True}, fans=False, role="fixer_inferencer"
        )

    async def test_multi_flow_initial_and_followup_leaves_are_own(self):
        initial = self.leaf(bta_inferencer={"own": _bta()})
        followup = self.leaf(key="", bta_inferencer={"own": _bta()})
        flow = {
            "input": "Q",
            "initial_inferencer": initial,
            "followup_inferencer": followup,
            "max_dynamic_steps": 2,
            "end_condition": lambda state, result: False,
        }
        mfi = MultiFlowInferencer(
            flow_configs=[flow],
            disable_aggregator=True,
            workspace=self.ws(),
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )
        with _logged() as records:
            await mfi.ainfer("go")

        fanned = _of_type(records, "BtaFanOut")
        self.assertEqual({item["role"] for _, item in fanned}, {"own"})
        self.assertTrue(any(i is initial for i, _ in fanned))
        self.assertTrue(any(i is followup for i, _ in fanned))
        self.assertFalse(any(i in (initial, followup) for i, _, _ in self.rec.calls))

    def dual(self, leaves):
        flows = [
            {
                "input": f"t{index}",
                "initial_inferencer": leaf,
                "followup_inferencer": Plain(),
                "end_condition": lambda state, result: True,
                "max_dynamic_steps": 1,
            }
            for index, leaf in enumerate(leaves)
        ]
        return MultiFlowDualInferencer(
            flow_configs=flows,
            multi_flow_aggregator_inferencer=Plain(response=_DUAL_AGGREGATION),
            reviewer_strategy="runner_up",
            fixer_strategy="winner",
            consensus_config=ConsensusConfig(max_iterations=2),
            workspace=self.ws(),
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )

    async def test_multi_flow_dual_fans_out_only_the_fixer(self):
        self.rec.reviews = [_verdict(False), _verdict(True)]
        roles = {"fixer_inferencer": _bta(), "review_inferencer": None}
        leaves = [
            self.leaf(id=name, bta_inferencer=dict(roles)) for name in ("L0", "L1")
        ]
        with _logged() as records:
            result = await self.dual(leaves).ainfer("go")

        self.assertEqual(result.base_response, "AGGREGATED")
        fanned = [
            (i.id, it["role"], it["depth"]) for i, it in _of_type(records, "BtaFanOut")
        ]
        self.assertEqual(fanned, [("L0", "fixer_inferencer", 0)])
        self.assertEqual(len(_of_type(records, "BtaFanOutComplete")), 1)
        bypassed = sorted(
            (i.id, it["role"]) for i, it in _of_type(records, "BtaFanOutBypassed")
        )
        self.assertEqual(
            bypassed,
            [
                ("L0", "own"),
                ("L1", "own"),
                ("L1", "review_inferencer"),
                ("L1", "review_inferencer"),
            ],
        )
        contracts = [
            r[3]["input"] for r in self.rec.renders if r[1] == "task_breakdown"
        ]
        self.assertEqual(contracts, ["[plan|followup|None] go"])
        fix_ws = os.path.join(
            self.tmp,
            "children",
            "round_01",
            "children",
            "fix",
            "children",
            BTA_INFERENCER_SLOT,
        )
        self.assertTrue(os.path.isdir(fix_ws))


# ---------------------------------------------------------------------------
# Contract and feed isolation
# ---------------------------------------------------------------------------


class FanoutFeedTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    async def test_contract_names_no_executor_location(self):
        await self.fanned(has_local_access=True, output_path="plan.md").ainfer("task")
        p_feed = self.rec.renders[0][3]
        self.assertIs(p_feed["has_local_access"], True)
        for key in ACTOR_SCOPED_FEED_KEYS:
            self.assertNotIn(key, p_feed)

    async def test_unfanned_render_keeps_the_executor_location(self):
        p = self.leaf(workspace=self.ws(), has_local_access=True, output_path="plan.md")
        await p.ainfer("task")
        self.assertLessEqual(set(ACTOR_SCOPED_FEED_KEYS), set(self.rec.renders[0][3]))

    async def test_ancestor_feed_override_reaches_only_the_contract(self):
        root = RunContext.root(self.ws())
        root.handles.set(
            TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE,
            {"upstream_artifacts": "PEER-BUNDLE", "peer_flag": 1},
        )
        await self.fanned().ainfer("task", run_context=root.child("flow"))

        p_feed, breakdown_feed, aggregator_feed = (r[3] for r in self.rec.renders)
        self.assertEqual(
            (p_feed["upstream_artifacts"], p_feed["peer_flag"]), ("PEER-BUNDLE", 1)
        )
        self.assertNotIn("upstream_artifacts", breakdown_feed)
        self.assertNotIn("peer_flag", breakdown_feed)
        self.assertNotIn("peer_flag", aggregator_feed)
        self.assertNotEqual(aggregator_feed.get("upstream_artifacts"), "PEER-BUNDLE")

    async def test_aggregator_inherits_p_instance_flags_but_not_owned_keys(self):
        p = self.fanned(
            template_extra_feed={
                "include_iteration_judgment": True,
                "upstream_artifacts": "X",
            }
        )
        with _logged() as records:
            await p.ainfer("task")

        _, breakdown_feed, aggregator_feed = (r[3] for r in self.rec.renders)
        self.assertIs(aggregator_feed["include_iteration_judgment"], True)
        self.assertNotEqual(aggregator_feed.get("upstream_artifacts"), "X")
        self.assertNotIn("include_iteration_judgment", breakdown_feed)
        [(_, record)] = _of_type(records, "BtaFanOut")
        self.assertEqual(record["seeded_feed_keys"], ["include_iteration_judgment"])

    async def test_aggregator_declared_flag_wins(self):
        aggregator = self.leaf(
            template_master_version="aggregation",
            template_extra_feed={"include_iteration_judgment": False},
        )
        p = self.fanned(
            _bta(aggregator_inferencer=aggregator),
            template_extra_feed={"include_iteration_judgment": True},
        )
        await p.ainfer("task")
        self.assertIs(self.rec.renders[2][3]["include_iteration_judgment"], False)

    async def test_factory_aggregator_is_seeded_like_an_instance(self):
        aggregator = functools.partial(
            ScriptedLeaf,
            template_manager=_TemplateManager(self.rec),
            template_root_space="plan",
            template_master_version="aggregation",
            recorder=self.rec,
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )
        p = self.fanned(
            _bta(aggregator_inferencer=aggregator),
            template_extra_feed={"include_iteration_judgment": True},
        )
        self.assertEqual(await p.ainfer("task"), "AGGREGATED")
        self.assertIs(self.rec.renders[2][3]["include_iteration_judgment"], True)


# ---------------------------------------------------------------------------
# Kwargs projection
# ---------------------------------------------------------------------------


class FanoutKwargsTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    async def test_framework_kwargs_reach_the_fanout_and_the_rest_reach_workers(self):
        seen = []
        original = BreakdownThenAggregateInferencer.ainfer

        async def spy(bta, *args, **kwargs):
            seen.append(dict(kwargs))
            return await original(bta, *args, **kwargs)

        with patch.object(BreakdownThenAggregateInferencer, "ainfer", spy):
            await self.fanned().ainfer("task", total_timeout_seconds=500, custom_flag=1)

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["total_timeout_seconds"], 500)
        self.assertNotIn("custom_flag", seen[0])
        for _, prompt, kwargs in self.rec.calls:
            self.assertEqual(
                "custom_flag" in kwargs, not prompt.startswith("["), prompt
            )
            self.assertNotIn("prepared_input", kwargs)
            self.assertNotIn("total_timeout_seconds", kwargs)


# ---------------------------------------------------------------------------
# Metamate: independent conversations, streaming, stale P-level caches
# ---------------------------------------------------------------------------


def _metamate_backend(recorder):
    async def backend(self, prompt, **kwargs):
        recorder.calls.append((self, prompt, dict(kwargs)))
        self._conversation_uuid = uuid.uuid4().hex
        yield recorder.respond(prompt)

    return backend


class FanoutMetamateTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        backend = patch.object(
            MetamateSDKInferencer, "_ainfer_streaming", _metamate_backend(self.rec)
        )
        backend.start()
        self.addCleanup(backend.stop)

    def metamate(self, **kwargs):
        kwargs.setdefault("bta_inferencer", _bta())
        # These tests compare prompts; MetaMate's default work budget would append
        # to each (it is covered in test_metamate_tool_call_budget).
        kwargs.setdefault("tool_call_budget", None)
        return MetamateSDKInferencer(
            api_key=None,
            code_scope_judge=None,
            enable_inferencer_variable_expansion=False,
            use_default_prompt_templates=False,
            template_manager=_TemplateManager(self.rec),
            template_root_space="plan",
            template_key="initial",
            workspace=self.ws(),
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
            **kwargs,
        )

    async def test_each_fanout_call_gets_its_own_conversation(self):
        p = self.metamate()
        p._conversation_uuid = "P-CONV"
        self.assertEqual(await p.ainfer("task"), "AGGREGATED")

        callers = [inst for inst, _, _ in self.rec.calls]
        self.assertEqual(len({id(inst) for inst in callers}), 4)
        self.assertFalse(any(inst is p for inst in callers))
        self.assertEqual(
            [kw.get("conversation_uuid") for _, _, kw in self.rec.calls], [None] * 4
        )
        self.assertEqual(p._conversation_uuid, "P-CONV")

    async def test_single_conversation_kwargs_are_rejected(self):
        p = self.metamate()
        for kwargs in (
            {"session_id": "abc"},
            {"return_sdk_response": True},
            {"conversation_uuid": "u"},
        ):
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, "independent instances"),
            ):
                await p.ainfer("task", **kwargs)
        self.assertEqual(self.rec.calls, [])

    async def test_new_session_is_dropped_and_timeouts_stay_on_the_fanout(self):
        await self.metamate().ainfer(
            "task", new_session=True, total_timeout_seconds=500
        )
        worker_kwargs = [sorted(kw) for _, _, kw in self.rec.worker_calls()]
        self.assertEqual(
            worker_kwargs, [["conversation_fbid", "conversation_uuid"]] * 2
        )

    async def test_streaming_yields_the_fanout_result_once(self):
        chunks = [chunk async for chunk in self.metamate().ainfer_streaming("task")]
        self.assertEqual(chunks, ["AGGREGATED"])
        prompts = self.rec.prompts()
        self.assertEqual(prompts[0], "[task_breakdown||None] task")
        self.assertEqual(sorted(prompts[1:3]), ["alpha shard", "beta shard"])
        self.assertTrue(prompts[3].startswith("[plan||aggregation] task"))

    async def run_with_stale_cache(self, content, **kwargs):
        """One call of a resuming P whose cache holds an earlier stream of this
        call's contract."""
        p = self.metamate(
            resume_with_saved_results=True,
            fallback_infer_mode=FallbackInferMode.UPDATE,
            **kwargs,
        )
        prompt_hash = hashlib.sha256(_CONTRACT.encode()).hexdigest()[:8]
        folder = os.path.join(p._effective_cache_folder(), type(p).__name__, "earlier")
        os.makedirs(folder)
        with open(
            os.path.join(folder, f"stream_1_{prompt_hash}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(content)
        return await p.ainfer("task")

    async def test_unfanned_call_resumes_a_completed_stream_cache(self):
        self.assertEqual(
            await self.run_with_stale_cache(_COMPLETED_CACHE, bta_inferencer=None),
            "STALE",
        )
        self.assertEqual(self.rec.calls, [])

    async def test_unfanned_call_recovers_a_partial_stream_cache(self):
        await self.run_with_stale_cache("STALE partial", bta_inferencer=None)
        self.assertEqual(self.rec.prompts(), ["RECOVER(STALE partial)"])

    async def test_fanned_call_ignores_a_completed_p_stream_cache(self):
        self.assertEqual(
            await self.run_with_stale_cache(_COMPLETED_CACHE), "AGGREGATED"
        )
        self.assertFalse(any("STALE" in prompt for prompt in self.rec.prompts()))

    async def test_fanned_call_ignores_a_partial_p_stream_cache(self):
        self.assertEqual(await self.run_with_stale_cache("STALE partial"), "AGGREGATED")
        self.assertFalse(
            any("STALE" in s or "RECOVER" in s for s in self.rec.prompts())
        )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class FanoutDisconnectTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    def assert_disconnected(self, p, count):
        self.assertEqual(len({id(inst) for inst in self.rec.disconnects}), count)
        self.assertEqual(len(self.rec.disconnects), count)
        self.assertFalse(any(inst is p for inst in self.rec.disconnects))

    async def test_breakdown_aggregator_and_workers_disconnect_on_success(self):
        p = self.fanned()
        await p.ainfer("task")
        self.assert_disconnected(p, 4)

    async def test_disconnect_after_a_worker_quorum_failure(self):
        self.rec.fail = {"beta shard": 1}
        p = self.fanned(_bta(min_successful_workers=2))
        with self.assertRaisesRegex(RuntimeError, "BTA quorum unmet"):
            await p.ainfer("task")
        self.assert_disconnected(p, 4)

    async def test_disconnect_after_a_breakdown_failure(self):
        self.rec.fail = {"[task_breakdown|": 1}
        p = self.fanned()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            await p.ainfer("task")
        self.assert_disconnected(p, 2)

    def test_the_sync_fan_out_closes_its_bta_when_the_call_ends(self):
        """B25: the sync path never disconnected its per-call BTA."""
        p = self.fanned()
        p.infer("task")
        self.assert_disconnected(p, 4)


# ---------------------------------------------------------------------------
# Capability gate: every entrypoint reaches the seam or opts out
# ---------------------------------------------------------------------------

_SEAM_CLASSES = (InferencerBase, StreamingInferencerBase, TemplatedInferencerBase)
_ENTRYPOINTS = (
    "ainfer",
    "infer",
    "_ainfer_single",
    "_infer_single",
    "ainfer_streaming",
    "infer_streaming",
)
# Entrypoint overrides that route a delegating call into the base seam
# (exercised by CliEntrypointBridgeTest).
_BRIDGED_OVERRIDES = {
    ClaudeCodeCliInferencer: ("ainfer", "infer"),
    CodexCliInferencer: ("ainfer", "infer"),
    KiroCliInferencer: ("ainfer", "infer"),
    DevmateCliInferencer: ("ainfer", "ainfer_streaming", "infer_streaming"),
    RovoDevCliInferencer: ("ainfer", "infer"),
}


def _import_inferencer_modules():
    prefix = inferencers_pkg.__name__ + "."
    for module in pkgutil.walk_packages(inferencers_pkg.__path__, prefix):
        try:
            importlib.import_module(module.name)
        except ImportError:
            continue


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _unbridged_overrides(cls):
    overrides = []
    for name in _ENTRYPOINTS:
        owner = next((k for k in cls.__mro__ if name in k.__dict__), None)
        if (
            owner is not None
            and owner not in _SEAM_CLASSES
            and name not in _BRIDGED_OVERRIDES.get(owner, ())
        ):
            overrides.append(f"{name}@{owner.__qualname__}")
    return overrides


class FanoutCapabilityTest(unittest.TestCase):
    def test_every_entrypoint_override_is_bridged_or_opted_out(self):
        _import_inferencer_modules()
        prefix = inferencers_pkg.__name__ + "."
        classes = {
            c for c in _subclasses(InferencerBase) if c.__module__.startswith(prefix)
        }
        unbridged = {
            cls.__qualname__: _unbridged_overrides(cls)
            for cls in classes
            if cls._SUPPORTS_BTA_FANOUT and _unbridged_overrides(cls)
        }
        self.assertEqual(unbridged, {})
        opted_out = {cls for cls in classes if not cls._SUPPORTS_BTA_FANOUT}
        self.assertLessEqual({ConversationalInferencer, OpenClawInferencer}, opted_out)
        for owner, names in _BRIDGED_OVERRIDES.items():
            for name in names:
                self.assertIn(name, owner.__dict__, f"{owner.__qualname__}.{name}")

    def test_opted_out_classes_reject_a_template_at_construction(self):
        with self.assertRaisesRegex(TypeError, "does not support bta_inferencer"):
            OpenClawInferencer(auth_token="token", bta_inferencer=_bta())
        with self.assertRaisesRegex(TypeError, "does not support bta_inferencer"):
            ConversationalInferencer(base_inferencer=Plain(), bta_inferencer=_bta())

    def test_opted_out_classes_accept_none(self):
        self.assertIsNone(OpenClawInferencer(auth_token="token").bta_inferencer)
        conversational = ConversationalInferencer(
            base_inferencer=Plain(), bta_inferencer=None
        )
        self.assertIsNone(conversational.bta_inferencer)


class _Reached(Exception):
    pass


def _reach_fanout(self, worker_args, call_id, contract):
    raise _Reached(sorted(worker_args), str(contract))


class CliEntrypointBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def call(self, p, entry):
        if entry == "infer":
            return await asyncio.to_thread(p.infer, "hello", flag=1)
        if entry == "ainfer_streaming":
            return [chunk async for chunk in p.ainfer_streaming("hello", flag=1)]
        if entry == "infer_streaming":
            return await asyncio.to_thread(
                lambda: list(p.infer_streaming("hello", flag=1))
            )
        return await p.ainfer("hello", flag=1)

    async def test_bridged_entrypoints_reach_the_fanout_not_the_backend(self):
        for cls in _BRIDGED_OVERRIDES:
            p = cls(bta_inferencer=_bta())
            with (
                patch.object(InferencerBase, "_prepare_fanout", _reach_fanout),
                patch.object(cls, "_ainfer", side_effect=AssertionError("backend")),
                patch.object(cls, "_infer", side_effect=AssertionError("backend")),
                patch.object(
                    cls, "_ainfer_streaming", side_effect=AssertionError("backend")
                ),
                patch.object(
                    cls, "_infer_streaming", side_effect=AssertionError("backend")
                ),
            ):
                for entry in ("ainfer", "infer", "ainfer_streaming", "infer_streaming"):
                    with self.subTest(cls=cls.__name__, entry=entry):
                        with self.assertRaises(_Reached) as reached:
                            await self.call(p, entry)
                        self.assertEqual(reached.exception.args[0], ["flag"])
                with self.subTest(cls=cls.__name__, entry="session_id"):
                    with self.assertRaisesRegex(ValueError, "independent instances"):
                        await p.ainfer("hello", session_id="x")

    async def test_opted_out_entrypoints_reject_a_late_template(self):
        p = OpenClawInferencer(auth_token="token")
        p.bta_inferencer = _bta()
        with self.assertRaisesRegex(TypeError, "does not support bta_inferencer"):
            await p.ainfer("hello")
        with self.assertRaisesRegex(TypeError, "does not support bta_inferencer"):
            await p.ainfer_streaming("hello").__anext__()
        with self.assertRaisesRegex(TypeError, "does not support bta_inferencer"):
            await asyncio.to_thread(p.infer, "hello")
        with self.assertRaisesRegex(TypeError, "does not support bta_inferencer"):
            await asyncio.to_thread(lambda: list(p.infer_streaming("hello")))


# ---------------------------------------------------------------------------
# Validation: every row raises before any model call
# ---------------------------------------------------------------------------


class FanoutValidationTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    def test_role_mapping_needs_a_templated_host(self):
        with self.assertRaisesRegex(TypeError, "role mapping"):
            Plain(bta_inferencer={"own": _bta()})

    def test_string_spec_is_rejected_at_construction(self):
        with self.assertRaisesRegex(TypeError, "must be a BTA template"):
            self.leaf(bta_inferencer="[null]")
        with self.assertRaisesRegex(TypeError, re.escape("bta_inferencer['own']")):
            self.leaf(bta_inferencer={"own": "[null]"})

    async def assert_rejected(self, p, error, pattern, calls=()):
        with self.assertRaisesRegex(error, pattern):
            await p.ainfer("task")
        self.assertEqual(self.rec.prompts(), list(calls))

    async def test_template_must_build_a_bta(self):
        await self.assert_rejected(
            self.fanned(Plain()),
            TypeError,
            "must build a BreakdownThenAggregateInferencer",
        )

    async def test_expected_parent_types_guard_the_host(self):
        template = _bta(expected_parent_types=["MetamateSDK"])
        await self.assert_rejected(
            self.fanned(template), TypeError, "expected_parent_types"
        )

        host = f"{TemplatedInferencerBase.__module__}.{TemplatedInferencerBase.__qualname__}"
        p = self.fanned(_bta(expected_parent_types=[host]))
        self.assertEqual(await p.ainfer("task"), "AGGREGATED")

    async def test_host_needs_a_workspace_and_a_relative_output_path(self):
        unbound = self.leaf(bta_inferencer=_bta())
        absolute = self.fanned(output_path=os.path.join(self.tmp, "abs.md"))
        for p in (unbound, absolute):
            with self.subTest(output_path=p.output_path):
                await self.assert_rejected(
                    p, ValueError, "workspace-relative output_path"
                )

    async def test_aggregator_must_receive_the_contract(self):
        template = _bta(inject_upstream_artifacts_to_aggregator=False)
        await self.assert_rejected(
            self.fanned(template), ValueError, "inject_upstream_artifacts_to_aggregator"
        )

    async def test_explicit_aggregator_must_render(self):
        for aggregator in (Plain(), functools.partial(Plain)):
            with self.subTest(aggregator=aggregator):
                await self.assert_rejected(
                    self.fanned(_bta(aggregator_inferencer=aggregator)),
                    ValueError,
                    "cannot render the worker outputs",
                )

    async def test_blank_slots_need_a_rendering_host(self):
        p = Plain(workspace=self.ws(), bta_inferencer=_bta())
        with self.assertRaisesRegex(ValueError, "cannot render prompts"):
            await p.ainfer("task")
        self.assertEqual(p.calls, [])

    async def test_oversized_shard_fails_after_the_breakdown(self):
        template = _bta(max_worker_query_chars=5)
        await self.assert_rejected(
            self.fanned(template),
            ValueError,
            "max_worker_query_chars=5",
            calls=[f"[task_breakdown||None] {_CONTRACT}"],
        )


# ---------------------------------------------------------------------------
# Resume from the fan-out's per-node checkpoints
# ---------------------------------------------------------------------------


def _subtasks_fence(*descriptions):
    subtasks = [{"description": d} for d in descriptions]
    return f"```json decomposed_subtasks\n{json.dumps({'subtasks': subtasks})}\n```"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class FanoutResumeTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    def bta_path(self, *parts):
        return os.path.join(self.tmp, "children", BTA_INFERENCER_SLOT, *parts)

    def aggregator_artifacts(self):
        feeds = [
            feed for _, _, master, feed in self.rec.renders if master == "aggregation"
        ]
        self.assertEqual(len(feeds), 1)
        return feeds[0]["upstream_artifacts"]

    def resumable(self):
        """P whose first call fails the quorum (beta fails once). A re-run
        breakdown would dispatch the second answer's shards."""
        self.rec.breakdowns = [
            _subtasks_fence("alpha shard", "beta shard"),
            _subtasks_fence("gamma shard", "delta shard"),
        ]
        self.rec.fail = {"beta shard": 1}
        template = _bta(
            breakdown_format="json_subtasks",
            min_successful_workers=2,
            enable_result_save=True,
            resume_with_saved_results=True,
        )
        return self.fanned(template, output_path="plan.md")

    def assert_crashed(self):
        promoted = self.bta_path("checkpoints", "breakdown", "decomposed_subtasks.json")
        self.assertTrue(os.path.isfile(promoted))
        return len(self.rec.calls)

    def assert_resumed(self, first, result):
        self.assertEqual(result, "AGGREGATED")
        resumed = self.rec.prompts()[first:]
        self.assertEqual(len(resumed), 2)
        self.assertIn("beta shard", resumed[0])
        self.assertTrue(resumed[1].startswith(f"[plan||aggregation] {_CONTRACT}"))
        artifacts = self.aggregator_artifacts()
        self.assertIn("W(**Description**: alpha shard)", artifacts)
        self.assertIn("W(**Description**: beta shard)", artifacts)

    async def test_resume_reruns_only_the_failed_worker(self):
        p = self.resumable()
        with self.assertRaisesRegex(RuntimeError, "quorum unmet"):
            await p.ainfer("task")
        first = self.assert_crashed()

        self.assert_resumed(first, await p.ainfer("task"))

    def test_sync_resume_reruns_only_the_failed_worker(self):
        p = self.resumable()
        with self.assertRaisesRegex(RuntimeError, "quorum unmet"):
            p.infer("task")
        first = self.assert_crashed()

        self.assert_resumed(first, p.infer("task"))

    async def test_backup_resume_reads_each_workers_own_output(self):
        p = self.fanned(output_path="plan.md")
        _write(self.bta_path("outputs", "plan.md"), "STALE AGGREGATE")
        _write(
            self.bta_path("children", "worker_00", "outputs", "plan.md"), "ALPHA DONE"
        )

        self.assertEqual(await p.ainfer("task"), "AGGREGATED")

        self.assertEqual([c[1] for c in self.rec.worker_calls()], ["beta shard"])
        artifacts = self.aggregator_artifacts()
        self.assertIn("ALPHA DONE", artifacts)
        self.assertIn("W(beta shard)", artifacts)
        self.assertNotIn("STALE AGGREGATE", artifacts)


class FanoutLinearWorkflowTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    """A fanned followup leaf serves every LWI round; each round's fan-out must
    run in that round's workspace, or a resumable template reloads the previous
    round's shards and worker results."""

    async def run_rounds(self, *, pinned):
        self.rec.breakdowns = [
            _subtasks_fence(f"r{i} alpha", f"r{i} beta") for i in (1, 2)
        ]
        template = _bta(
            breakdown_format="json_subtasks",
            enable_result_save=True,
            resume_with_saved_results=True,
        )
        lwi = LinearWorkflowInferencer(
            workspace=self.ws() if pinned else None,
            dynamic_mode=True,
            default_initial_inferencer=self.leaf(output_path="output.md"),
            default_followup_inferencer=self.leaf(
                bta_inferencer=template, output_path="output.md"
            ),
            end_condition=lambda state, _: len(state["dynamic_step_results"]) >= 3,
            max_dynamic_steps=3,
            output_path="output.md",
        )
        run_context = None if pinned else RunContext.root(self.ws())
        await lwi.ainfer("task", run_context=run_context)

    def assert_each_round_fanned_out_on_its_own(self):
        self.assertEqual(self.rec.breakdowns, [])
        self.assertEqual(
            sorted(call[1] for call in self.rec.worker_calls()),
            [f"**Description**: r{i} {s}" for i in (1, 2) for s in ("alpha", "beta")],
        )
        for round_name in ("round01", "round02"):
            promoted = os.path.join(
                self.tmp,
                "children",
                round_name,
                "children",
                BTA_INFERENCER_SLOT,
                "checkpoints",
                "breakdown",
                "decomposed_subtasks.json",
            )
            self.assertTrue(os.path.isfile(promoted), promoted)

    async def test_constructor_workspace(self):
        await self.run_rounds(pinned=True)
        self.assert_each_round_fanned_out_on_its_own()

    async def test_context_workspace(self):
        await self.run_rounds(pinned=False)
        self.assert_each_round_fanned_out_on_its_own()


# ---------------------------------------------------------------------------
# Structured records
# ---------------------------------------------------------------------------

_FANOUT_RECORD_KEYS = {
    "call_id",
    "depth",
    "role",
    "template_type",
    "bta_ws",
    "worker_template_overridden",
    "breakdown_source",
    "aggregator_source",
    "seeded_feed_keys",
    "worker_arg_keys",
    "contract_chars",
    "contract_sha256",
}


class FanoutRecordsTest(_FanoutMixin, unittest.IsolatedAsyncioTestCase):
    async def logged_calls(self, p, n=1):
        logged = []

        def record(log_item, log_type=None, *args, **kwargs):
            logged.append((log_type, log_item))

        with patch.object(p, "log_info", side_effect=record):
            for _ in range(n):
                await p.ainfer("task")
        return logged

    async def test_fanout_start_and_complete_records(self):
        logged = await self.logged_calls(self.fanned(), n=2)
        starts = [item for kind, item in logged if kind == "BtaFanOut"]
        completes = [item for kind, item in logged if kind == "BtaFanOutComplete"]

        self.assertEqual(len(starts), 2)
        self.assertEqual(set(starts[0]), _FANOUT_RECORD_KEYS)
        digest = hashlib.sha256(_CONTRACT.encode()).hexdigest()
        for start in starts:
            self.assertEqual(
                (start["contract_sha256"], start["contract_chars"]),
                (digest, len(_CONTRACT)),
            )
        self.assertNotEqual(starts[0]["call_id"], starts[1]["call_id"])
        self.assertEqual(
            {
                k: starts[0][k]
                for k in (
                    "depth",
                    "role",
                    "template_type",
                    "worker_template_overridden",
                )
            },
            {
                "depth": 0,
                "role": "own",
                "template_type": "BreakdownThenAggregateInferencer",
                "worker_template_overridden": False,
            },
        )
        self.assertTrue(
            starts[0]["bta_ws"].endswith(os.path.join("children", BTA_INFERENCER_SLOT))
        )
        self.assertEqual(
            (starts[0]["seeded_feed_keys"], starts[0]["worker_arg_keys"]), ([], [])
        )
        for start, complete in zip(starts, completes, strict=True):
            self.assertEqual(
                set(complete),
                {"call_id", "n_subtasks", "aggregated_chars", "elapsed_s"},
            )
            self.assertEqual(
                (
                    complete["call_id"],
                    complete["n_subtasks"],
                    complete["aggregated_chars"],
                ),
                (start["call_id"], 2, len("AGGREGATED")),
            )
        self.assertIn(("InferenceInput", _CONTRACT), logged)

    async def test_bypass_record(self):
        p = self.fanned({"fixer_inferencer": _bta()})
        logged = await self.logged_calls(p)
        bypassed = [item for kind, item in logged if kind == "BtaFanOutBypassed"]
        self.assertEqual(
            bypassed,
            [
                {
                    "reason": "role_not_mapped",
                    "role": "own",
                    "mapped_roles": ["fixer_inferencer"],
                }
            ],
        )
        self.assertEqual(self.rec.prompts(), [_CONTRACT])

    async def test_input_stats_records(self):
        with _logged(BreakdownThenAggregateInferencer) as records:
            await self.fanned().ainfer("task")
        stats = [item for _, kind, item in records if kind == "InferenceInputStats"]
        self.assertEqual([s["stage"] for s in stats], ["dispatch", "aggregate"])
        self.assertEqual(
            (stats[0]["shard_chars"], stats[0]["max_shard_chars"]), ([11, 10], 11)
        )
        self.assertGreater(stats[1]["aggregator_input_chars"], 0)

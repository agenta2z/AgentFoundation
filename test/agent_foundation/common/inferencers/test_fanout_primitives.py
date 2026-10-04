"""Primitives a per-call BTA fan-out builds on; none changes existing behaviour.

Covers the feed-scope barrier, template fields skipped by the child walker,
explicit cascading into a per-call child, atomic canonical-output replacement,
the BTA's worker-call knobs (reserved args, query-size cap, input stats,
``adisconnect``), the role-state reads on ``TemplatedInferencerBase``, and the
public config helpers ``collect_slot_defaults``, ``import_target`` and
``LazyConfigFactory.fresh``.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import agent_foundation.common.configs  # noqa: F401 — registers inferencer aliases
import attr
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.template_defaults import AGGREGATION_DEFAULTS
from agent_foundation.common.inferencers.template_feed_scope import (
    publish_child_template_feed,
    resolve_ctx_feed_override,
    TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE,
    TEMPLATE_EXTRA_FEED_SCOPE_HANDLE,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode
from rich_python_utils.config_utils import collect_slot_defaults, import_target
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory

_BREAKDOWN = "1. q0\n2. q11\n3. q222"


@attrs
class Leaf(InferencerBase):
    """Returns ``response``; records each call's input and kwargs in ``calls``."""

    _response = attrib(default="leaf")
    calls = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.calls.append((inference_input, kwargs))
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)


@attrs
class Pair(Leaf):
    left = attrib(default=None, kw_only=True)


@attrs
class TemplateHolder(Leaf):
    live = attrib(default=None, kw_only=True)
    template = attrib(
        default=None, kw_only=True, metadata={"inferencer_template": True}
    )


@attrs
class DisconnectLeaf(Leaf):
    """Appends ``label`` to the shared ``disconnected`` list, then raises ``error``."""

    label = attrib(default="", kw_only=True)
    disconnected = attrib(factory=list, kw_only=True)
    error = attrib(default=None, kw_only=True)

    async def adisconnect(self):
        self.disconnected.append(self.label)
        if self.error is not None:
            raise self.error


@attrs(slots=False)
class RoleLeaf(TemplatedInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "ok"


class _TmpDirMixin:
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()


def _worker_factory(calls):
    def factory(sub_query, index):
        return Leaf(
            response=f"w{index}",
            calls=calls,
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )

    return factory


def _bta(root, **kwargs):
    kwargs.setdefault("breakdown_inferencer", Leaf(response=_BREAKDOWN))
    kwargs.setdefault("aggregator_inferencer", Leaf(response="AGG"))
    kwargs.setdefault("worker_inferencers", _worker_factory([]))
    return BreakdownThenAggregateInferencer(
        workspace=InferencerWorkspace(root=root),
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Feed-scope barrier (template_feed_scope)
# ---------------------------------------------------------------------------


class FeedScopeBarrierTest(unittest.TestCase):
    def setUp(self):
        self.root = RunContext.root()
        self.root.handles.set(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE, {"outer": 1})
        self.host = self.root.child("host")
        self.scope = self.host.child("bta_inferencer")
        self.scope.handles.set(TEMPLATE_EXTRA_FEED_SCOPE_HANDLE, True)

    def test_override_above_the_barrier_is_hidden_inside_the_scope(self):
        self.assertIsNone(resolve_ctx_feed_override(self.scope))
        self.assertIsNone(resolve_ctx_feed_override(self.scope.child("aggregator")))
        self.assertEqual(resolve_ctx_feed_override(self.host), {"outer": 1})
        self.assertEqual(
            resolve_ctx_feed_override(self.host.child("sibling")), {"outer": 1}
        )

    def test_override_at_or_below_the_barrier_resolves_and_the_deepest_wins(self):
        self.scope.handles.set(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE, {"scope": 2})
        aggregator = self.scope.child("aggregator")
        self.assertEqual(resolve_ctx_feed_override(self.scope), {"scope": 2})
        self.assertEqual(resolve_ctx_feed_override(aggregator), {"scope": 2})

        aggregator.handles.set(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE, {"deep": 3})
        self.assertEqual(resolve_ctx_feed_override(aggregator), {"deep": 3})

    def _publish_under(self, ctx, child_slot):
        token = enter_run(ctx)
        try:
            publish_child_template_feed(None, child_slot, {"own": 4})
        finally:
            exit_run(token)

    def test_publish_inside_the_scope_does_not_compose_outer_keys(self):
        self._publish_under(self.scope, "aggregator")
        self.assertEqual(
            resolve_ctx_feed_override(self.scope.child("aggregator")), {"own": 4}
        )

        self._publish_under(self.scope, None)
        self.assertEqual(resolve_ctx_feed_override(self.scope), {"own": 4})

    def test_publish_outside_a_scope_still_composes_outer_keys(self):
        self._publish_under(self.host, "sibling")
        self.assertEqual(
            resolve_ctx_feed_override(self.host.child("sibling")),
            {"outer": 1, "own": 4},
        )


# ---------------------------------------------------------------------------
# Template fields are not live children
# ---------------------------------------------------------------------------


class TemplateFieldWalkTest(unittest.TestCase):
    def test_walker_skips_template_fields(self):
        live, template = Leaf(), Leaf()
        holder = TemplateHolder(live=live, template=template)
        visited = []
        holder._for_each_child_inferencer(
            lambda child, _field, _key: visited.append(child), lambda *_: None
        )
        self.assertEqual(visited, [live])

    def test_construction_cascade_does_not_reach_a_template(self):
        holder = TemplateHolder(live=Leaf(), template=Leaf(), debug_mode=True)
        self.assertTrue(holder.live.debug_mode)
        self.assertIsNone(holder.template.debug_mode)


# ---------------------------------------------------------------------------
# _cascade_attributes_into: explicit cascade into a per-call child
# ---------------------------------------------------------------------------


class CascadeAttributesIntoTest(unittest.TestCase):
    def test_unset_child_and_its_descendants_inherit(self):
        grandchild = Leaf()
        child = Pair(left=grandchild)
        Leaf(debug_mode=True)._cascade_attributes_into(child)
        self.assertTrue(child.debug_mode)
        self.assertTrue(grandchild.debug_mode)

    def test_explicit_child_value_wins(self):
        child = Leaf(debug_mode=False)
        Leaf(debug_mode=True)._cascade_attributes_into(child)
        self.assertIs(child.debug_mode, False)

    def test_unset_parent_and_non_inferencer_child_are_ignored(self):
        child = Leaf()
        Leaf()._cascade_attributes_into(child)
        self.assertIsNone(child.debug_mode)

        stranger = object()
        Leaf(debug_mode=True)._cascade_attributes_into(stranger)
        self.assertFalse(hasattr(stranger, "debug_mode"))


# ---------------------------------------------------------------------------
# _symlink_child_output(replace_canonical=...)
# ---------------------------------------------------------------------------


class SymlinkChildOutputTest(_TmpDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.parent = Leaf(
            output_path="out.md",
            workspace=InferencerWorkspace(root=os.path.join(self.tmp, "parent")),
        )
        self.child_ws = InferencerWorkspace(root=os.path.join(self.tmp, "child"))
        self.child_ws.ensure_dirs()
        with open(self.child_ws.output_path("result.md"), "w") as f:
            f.write("child result")
        self.own = self.parent._workspace.output_path("out.md")

    def _write_stale_own_output(self):
        os.makedirs(os.path.dirname(self.own), exist_ok=True)
        with open(self.own, "w") as f:
            f.write("stale")

    def _read_own(self):
        with open(self.own) as f:
            return f.read()

    def test_existing_own_output_is_kept_by_default(self):
        self._write_stale_own_output()
        self.parent._symlink_child_output(self.child_ws, "result.md")
        self.assertEqual(self._read_own(), "stale")

    def test_replace_canonical_swaps_in_the_child_output_atomically(self):
        self._write_stale_own_output()
        self.parent._symlink_child_output(
            self.child_ws, "result.md", replace_canonical=True
        )
        self.assertEqual(self._read_own(), "child result")
        self.assertTrue(os.path.islink(self.own))
        leftovers = [
            n for n in os.listdir(os.path.dirname(self.own)) if n.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])

    def test_replace_canonical_creates_a_missing_own_output(self):
        self.parent._symlink_child_output(
            self.child_ws, "result.md", replace_canonical=True
        )
        self.assertEqual(self._read_own(), "child result")


# ---------------------------------------------------------------------------
# BTA worker-call knobs
# ---------------------------------------------------------------------------


class BtaWorkerInferenceArgsTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    def test_reserved_keys_are_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, r"may not set \['run_context'\]"):
            _bta(self.tmp, worker_inference_args={"run_context": None, "x": 1})

    async def test_async_workers_receive_the_extra_args(self):
        calls = []
        bta = _bta(
            self.tmp,
            worker_inferencers=_worker_factory(calls),
            worker_inference_args={"temperature": 0.3},
        )
        self.assertEqual(await bta.ainfer("task"), "AGG")
        self.assertEqual([kw for _, kw in calls], [{"temperature": 0.3}] * 3)

    def test_sync_workers_receive_the_extra_args(self):
        calls = []
        bta = _bta(
            self.tmp,
            worker_inferencers=_worker_factory(calls),
            worker_inference_args={"temperature": 0.3},
        )
        self.assertEqual(bta.infer("task"), "AGG")
        self.assertEqual([kw for _, kw in calls], [{"temperature": 0.3}] * 3)


class BtaWorkerQueryCapTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    _OVERSIZED = r"exceed max_worker_query_chars=3 \(index: chars\) \{2: 4\}"

    async def test_async_oversized_query_raises_before_any_worker_runs(self):
        calls = []
        bta = _bta(
            self.tmp,
            worker_inferencers=_worker_factory(calls),
            max_worker_query_chars=3,
        )
        with self.assertRaisesRegex(ValueError, self._OVERSIZED):
            await bta.ainfer("task")
        self.assertEqual(calls, [])

    def test_sync_oversized_query_raises_before_any_worker_runs(self):
        calls = []
        bta = _bta(
            self.tmp,
            worker_inferencers=_worker_factory(calls),
            max_worker_query_chars=3,
        )
        with self.assertRaisesRegex(ValueError, self._OVERSIZED):
            bta.infer("task")
        self.assertEqual(calls, [])

    async def test_queries_within_the_cap_run(self):
        calls = []
        bta = _bta(
            self.tmp,
            worker_inferencers=_worker_factory(calls),
            max_worker_query_chars=4,
        )
        self.assertEqual(await bta.ainfer("task"), "AGG")
        self.assertEqual(sorted(q for q, _ in calls), ["q0", "q11", "q222"])


class BtaInputStatsTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_and_aggregate_stats_are_logged(self):
        bta = _bta(self.tmp)
        records = []

        def _record(payload, log_type=None, *args, **kwargs):
            if log_type == "InferenceInputStats":
                records.append(payload)

        with patch.object(bta, "log_info", side_effect=_record):
            await bta.ainfer("task")

        shards = {"shard_count": 3, "shard_chars": [2, 3, 4], "max_shard_chars": 4}
        self.assertEqual(len(records), 2)
        self.assertEqual(
            records[0], {"stage": "dispatch", **shards, "aggregator_input_chars": None}
        )
        self.assertEqual(records[1]["stage"], "aggregate")
        self.assertEqual(
            {k: records[1][k] for k in shards}, shards, "aggregate repeats shard sizes"
        )
        self.assertIsInstance(records[1]["aggregator_input_chars"], int)


class BtaDisconnectTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_the_call_closes_its_workers_and_adisconnect_its_stages(self):
        log = []

        def factory(sub_query, index):
            return DisconnectLeaf(
                response=f"w{index}", label=f"w{index}", disconnected=log
            )

        bta = _bta(
            self.tmp,
            breakdown_inferencer=DisconnectLeaf(
                response=_BREAKDOWN, label="breakdown", disconnected=log
            ),
            aggregator_inferencer=DisconnectLeaf(
                response="AGG", label="aggregator", disconnected=log
            ),
            worker_inferencers=factory,
        )
        await bta.ainfer("task")
        # The workers this call's factory built close when its attempt ends.
        self.assertEqual(sorted(log), ["w0", "w1", "w2"])

        await bta.adisconnect()

        self.assertEqual(sorted(log[3:]), ["aggregator", "breakdown"])

    async def test_a_shared_child_is_disconnected_once(self):
        log = []
        shared = DisconnectLeaf(response=_BREAKDOWN, label="shared", disconnected=log)
        bta = _bta(self.tmp, breakdown_inferencer=shared, aggregator_inferencer=shared)
        await bta.adisconnect()
        self.assertEqual(log, ["shared"])

    async def test_every_child_is_attempted_and_the_first_failure_is_raised(self):
        log = []
        bta = _bta(
            self.tmp,
            breakdown_inferencer=DisconnectLeaf(
                label="breakdown",
                disconnected=log,
                error=RuntimeError("breakdown down"),
            ),
            aggregator_inferencer=DisconnectLeaf(
                label="aggregator",
                disconnected=log,
                error=ValueError("aggregator down"),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "breakdown down"):
            await bta.adisconnect()
        self.assertEqual(sorted(log), ["aggregator", "breakdown"])


# ---------------------------------------------------------------------------
# Role-state reads on TemplatedInferencerBase
# ---------------------------------------------------------------------------


class RoleStateTest(unittest.TestCase):
    def test_switch_without_ctx_records_the_applied_role(self):
        leaf = RoleLeaf()
        leaf.switch_role("fixer_inferencer", template_key="followup")
        self.assertEqual(leaf._applied_role, "fixer_inferencer")
        self.assertEqual(leaf.template_key, "followup")
        self.assertIsNone(leaf._active_role_state())

    def test_switch_that_changes_no_template_attr_records_nothing(self):
        leaf = RoleLeaf()
        leaf.switch_role("reviewer")
        self.assertIsNone(getattr(leaf, "_applied_role", None))

    def test_switch_under_ctx_records_role_state_not_instance_state(self):
        leaf = RoleLeaf()
        token = enter_run(RunContext.root().child("fix"))
        try:
            leaf.switch_role("fixer_inferencer", template_key="followup")
            state = leaf._active_role_state()
            effective_key = leaf._effective_role()[0]
        finally:
            exit_run(token)

        self.assertEqual(
            (state.new_role, state.template_key), ("fixer_inferencer", "followup")
        )
        self.assertEqual(effective_key, "followup")
        self.assertIsNone(getattr(leaf, "_applied_role", None))
        self.assertEqual(leaf.template_key, RoleLeaf().template_key)
        self.assertIsNone(leaf._active_role_state())

    def test_role_selector_attrs_are_template_fields(self):
        field_names = {a.name for a in attr.fields(TemplatedInferencerBase)}
        self.assertTrue(TemplatedInferencerBase._ROLE_SELECTOR_ATTRS)
        self.assertLessEqual(
            set(TemplatedInferencerBase._ROLE_SELECTOR_ATTRS), field_names
        )


# ---------------------------------------------------------------------------
# Public config helpers
# ---------------------------------------------------------------------------


class CollectSlotDefaultsTest(unittest.TestCase):
    def test_bta_role_bundles_are_returned_as_a_copy(self):
        defaults = collect_slot_defaults(BreakdownThenAggregateInferencer)
        self.assertEqual(defaults, BreakdownThenAggregateInferencer.SLOT_DEFAULTS)
        self.assertIs(defaults["aggregator_inferencer"], AGGREGATION_DEFAULTS)

        defaults.clear()
        self.assertIn(
            "breakdown_inferencer",
            collect_slot_defaults(BreakdownThenAggregateInferencer),
        )


class ImportTargetTest(unittest.TestCase):
    def test_registered_alias_and_dotted_path_resolve_to_the_class(self):
        self.assertIs(import_target("BTA"), BreakdownThenAggregateInferencer)
        self.assertIs(import_target(f"{Leaf.__module__}.Leaf"), Leaf)

    def test_unknown_alias_raises_key_error(self):
        with self.assertRaisesRegex(KeyError, "Unknown target alias"):
            import_target("NoSuchFanoutAlias")

    def test_unimportable_targets_raise_import_error(self):
        with self.assertRaises(ImportError):
            import_target("no_such_module_for_fanout_primitives.Thing")
        with self.assertRaisesRegex(ImportError, "has no attribute 'NoSuchThing'"):
            import_target(f"{Leaf.__module__}.NoSuchThing")

    def test_alias_to_a_non_dotted_path_raises_import_error(self):
        with patch.dict(
            "rich_python_utils.config_utils._registry._registry",
            {"FanoutBareAlias": "bare"},
        ):
            with self.assertRaisesRegex(ImportError, "not a dotted import path"):
                import_target("FanoutBareAlias")


class LazyConfigFactoryFreshTest(unittest.TestCase):
    def _factory(self):
        return LazyConfigFactory(
            {
                "_target_": f"{Pair.__module__}.Pair",
                "left": {"_target_": f"{Leaf.__module__}.Leaf", "response": "r"},
            },
            {"debug_mode": True},
        )

    def test_fresh_is_a_new_factory_with_an_empty_feed(self):
        factory = self._factory()
        factory.template_extra_feed["published"] = 1

        fresh = factory.fresh()

        self.assertIsNot(fresh, factory)
        self.assertEqual(fresh.template_extra_feed, {})
        self.assertEqual(factory.template_extra_feed, {"published": 1})
        self.assertEqual(fresh.target, factory.target)

    def test_fresh_builds_an_equivalent_independent_tree(self):
        factory = self._factory()
        built, original = factory.fresh()(), factory()

        self.assertIsInstance(built, Pair)
        self.assertEqual(built.left._response, "r")
        self.assertTrue(built.left.debug_mode)
        self.assertIsNot(built.left, original.left)

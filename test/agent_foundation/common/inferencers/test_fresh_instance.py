"""``fresh_instance`` rebuilds an inferencer from its construction recipe.

The recipe (captured in ``InferencerBase.__new__``) records constructor intent
only: a fresh copy carries no workspace/logger binding, post-construction
setattr, identity or parent link, re-derives its post-init state, and keeps
children the source shares shared. Also covers the ``prepared_input`` kwarg,
``_FreshCloneFactory`` as a BTA worker factory, and resume-tail logging.
"""

import copy
import inspect
import pickle
import shutil
import tempfile
import unittest
from collections import OrderedDict
from functools import partial
from unittest.mock import patch

import agent_foundation.common.configs  # noqa: F401 — registers inferencer aliases
import attr
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers import (
    breakdown_then_aggregate_inferencer as bta_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.flow_parsers import parse_decision_stop
from agent_foundation.common.inferencers.inferencer_base import (
    _FreshCloneFactory,
    _LIVE_FIELD,
    _PrototypeCloneFactory,
    InferencerBase,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs
from omegaconf import OmegaConf
from rich_python_utils.common_utils.function_helper import FallbackMode
from rich_python_utils.config_utils import import_target, instantiate, list_registered
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory

# Registered inferencers the class audit must reach (each constructs with defaults).
_CORE_AUDITED = {"BTA", "ClaudeCodeCLI", "DevmateCLI", "Dual", "MetamateSDK", "PTI"}


@attrs
class Leaf(InferencerBase):
    """Returns ``response(input)`` (or ``response``)."""

    _response = attrib(default="leaf")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        if callable(self._response):
            return self._response(inference_input)
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)


@attrs
class Pair(Leaf):
    left = attrib(default=None, kw_only=True)
    right = attrib(default=None, kw_only=True)
    helper = attrib(default=None, kw_only=True)


@attrs
class LazyHolder(Leaf):
    worker_factory = attrib(default=None, kw_only=True)


@attrs
class CachedLeaf(Leaf):
    """Resumes every call from a (simulated) cached result."""

    def _try_resume_from_cache(self, inference_input, inference_config=None, **kwargs):
        return "cached"


class VarArgsLeaf(Leaf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@attrs(slots=False)
class PromptLeaf(TemplatedInferencerBase):
    """Templated leaf recording every prompt it receives."""

    prompts = attrib(factory=list)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.prompts.append(inference_input)
        return "ok"


def _render(key, active_template_root_space=None, master_version=None, **feed):
    return f"RENDERED[{key}]:{feed.get('input')}"


def _make_leaf():
    return Leaf(response="made")


def _stop(_state, _result):
    return True


def _contained(value):
    """Values held by a container, clone factory, partial or bound method."""
    if isinstance(value, (_PrototypeCloneFactory, _FreshCloneFactory)):
        return [value.prototype]
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if isinstance(value, partial):
        return [value.func, *value.args, *value.keywords.values()]
    if inspect.ismethod(value):
        return [value.__self__]
    return []


def _reachable_inferencers(root):
    """``{id: inferencer}`` for every inferencer reachable from ``root``'s config fields."""
    found, seen, stack = {}, set(), [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if not isinstance(value, InferencerBase):
            stack.extend(_contained(value))
            continue
        found[id(value)] = value
        stack.extend(
            getattr(value, a.name, None)
            for a in attr.fields(type(value))
            if a.name not in value.NON_CONFIG_ATTR_NAMES
        )
    return found


class _TmpDirMixin:
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()


# ---------------------------------------------------------------------------
# T2.1: runtime bindings are never copied
# ---------------------------------------------------------------------------


class UnboundCopyTest(_TmpDirMixin, unittest.TestCase):
    def test_fresh_copy_of_bound_leaf_is_unbound(self):
        guard = Leaf(response="ok")
        src = Leaf(
            response="p",
            output_guardrail_inferencer=guard,
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )
        Leaf()._bind_rebuilt_child_ws(
            src, "slot", InferencerWorkspace(root=self.tmp), owned=True
        )
        self.assertIsNotNone(src._workspace)
        self.assertIn("_workspace", src.logger)
        self.assertIsNotNone(guard._workspace)

        fresh = src.fresh_instance()

        self.assertIsNone(fresh._workspace)
        self.assertNotIn("_workspace", fresh.logger)
        self.assertIsNot(fresh.logger, src.logger)
        self.assertIsNot(fresh.output_guardrail_inferencer, guard)
        self.assertIsNone(fresh.output_guardrail_inferencer._workspace)
        self.assertEqual(
            (fresh._response, fresh.max_retry, fresh.fallback_mode),
            ("p", 0, FallbackMode.NEVER),
        )
        self.assertNotEqual(fresh.id, src.id)


# ---------------------------------------------------------------------------
# T2.2: post-init state is re-derived on the copy
# ---------------------------------------------------------------------------


class BtaRederivationTest(unittest.TestCase):
    def test_bta_copy_rederives_closures_feed_and_worker_factory(self):
        breakdown = PromptLeaf(template_extra_feed={"k": "v"})
        src = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown,
            worker_inferencers=Leaf(response="w"),
            aggregator_inferencer=Leaf(response="agg"),
            max_breakdown=3,
        )
        breakdown.template_extra_feed["late"] = True

        fresh = src.fresh_instance(max_breakdown=5)

        self.assertEqual(
            fresh.breakdown_inferencer.template_extra_feed,
            {"k": "v", "max_breakdown": 5},
        )
        self.assertEqual(
            breakdown.template_extra_feed, {"k": "v", "max_breakdown": 3, "late": True}
        )
        registry = bta_module._subgraph_registry(fresh)
        for key in ("bta_diamond", "bta_workers"):
            cells = [c.cell_contents for c in registry[key].__closure__]
            self.assertTrue(any(c is fresh for c in cells), key)
            self.assertFalse(any(c is src for c in cells), key)
        self.assertIs(type(fresh.worker_inferencers), _PrototypeCloneFactory)
        self.assertIsNot(
            fresh.worker_inferencers.prototype, src.worker_inferencers.prototype
        )
        self.assertEqual(fresh.worker_inferencers.prototype._response, "w")
        self.assertEqual(fresh.output_path, src.output_path)


# ---------------------------------------------------------------------------
# T2.3: aliasing is preserved; nothing is shared with the source
# ---------------------------------------------------------------------------


class AliasingTest(unittest.TestCase):
    def test_child_shared_in_source_stays_shared_in_copy(self):
        child = Leaf(response="c")
        fresh = Pair(left=child, right=child).fresh_instance()
        self.assertIs(fresh.left, fresh.right)
        self.assertIsNot(fresh.left, child)

    def test_dual_explicit_fixer_alias_is_preserved(self):
        base = Leaf(response="b")
        src = DualInferencer(
            base_inferencer=base,
            review_inferencer=Leaf(response="r"),
            fixer_inferencer=base,
        )
        fresh = src.fresh_instance()
        self.assertIs(fresh.fixer_inferencer, fresh.base_inferencer)
        self.assertIsNot(fresh.base_inferencer, base)
        self.assertFalse(
            _reachable_inferencers(src).keys() & _reachable_inferencers(fresh).keys()
        )

    def test_dual_implicit_fixer_is_rederived(self):
        src = DualInferencer(
            base_inferencer=Leaf(response="b"), review_inferencer=Leaf(response="r")
        )
        self.assertNotIn("fixer_inferencer", src._init_recipe)
        fresh = src.fresh_instance()
        self.assertIs(fresh.fixer_inferencer, fresh.base_inferencer)
        self.assertIsNot(fresh.base_inferencer, src.base_inferencer)

    def test_only_plain_containers_are_rebuilt(self):
        child = Leaf(response="c")
        src = Pair(left={"plain": [child]}, right=OrderedDict(subclass=child))
        fresh = src.fresh_instance()
        self.assertIsNot(fresh.left["plain"][0], child)
        self.assertIs(fresh.right["subclass"], child)


# ---------------------------------------------------------------------------
# T2.4: unknown override / no recipe / cycle
# ---------------------------------------------------------------------------


class RebuildErrorsTest(unittest.TestCase):
    def test_override_must_name_an_init_parameter(self):
        src = Leaf(response="a")
        with self.assertRaisesRegex(TypeError, r"unknown overrides \['_response'\]"):
            src.fresh_instance(_response="b")
        self.assertEqual(src.fresh_instance(response="b")._response, "b")

    def test_unbindable_init_leaves_no_recipe(self):
        src = VarArgsLeaf(response="a")
        self.assertIsNone(src._init_recipe)
        with self.assertRaisesRegex(TypeError, "not constructed through a bindable"):
            src.fresh_instance()

    def test_inferencer_reachable_from_its_own_recipe_is_rejected(self):
        factory = _FreshCloneFactory(None)
        src = Pair(helper=factory)
        factory.prototype = src
        with self.assertRaisesRegex(ValueError, "cycle"):
            src.fresh_instance()


# ---------------------------------------------------------------------------
# T2.5: constructor intent only
# ---------------------------------------------------------------------------


class PostConstructionSetattrTest(unittest.TestCase):
    def test_setattr_after_construction_is_not_carried(self):
        src = Leaf(response="a", max_retry=2)
        src.max_retry = 7
        src._response = "mutated"
        src.output_path = "late.md"

        fresh = src.fresh_instance()

        self.assertEqual(fresh.max_retry, 2)
        self.assertEqual(fresh._response, "a")
        self.assertEqual(fresh.output_path, Leaf().output_path)


# ---------------------------------------------------------------------------
# T2.6: copies carry the recipe
# ---------------------------------------------------------------------------


class RecipeSurvivesCopiesTest(unittest.TestCase):
    def test_copies_carry_the_recipe_and_can_rebuild(self):
        src = Leaf(response="a", max_retry=2)
        clones = {
            "copy": copy.copy(src),
            "deepcopy": copy.deepcopy(src),
            "pickle": pickle.loads(pickle.dumps(src)),
            "deepcopy_with_fresh_id": src.deepcopy_with_fresh_id(),
        }
        for how, clone in clones.items():
            with self.subTest(how=how):
                self.assertEqual(clone._init_recipe, src._init_recipe)
                fresh = clone.fresh_instance()
                self.assertEqual((fresh._response, fresh.max_retry), ("a", 2))

    def test_deepcopy_recipe_references_the_copied_children(self):
        child = Leaf(response="c")
        clone = copy.deepcopy(Pair(left=child))
        self.assertIs(clone._init_recipe["left"], clone.left)
        self.assertIsNot(clone.left, child)

    def test_live_field_placeholder_survives_copies(self):
        src = LazyHolder(worker_factory=_make_leaf)
        self.assertIs(src._init_recipe["worker_factory"], _LIVE_FIELD)
        for clone in (
            copy.copy(src),
            copy.deepcopy(src),
            pickle.loads(pickle.dumps(src)),
        ):
            self.assertIs(clone._init_recipe["worker_factory"], _LIVE_FIELD)
            self.assertIs(clone.fresh_instance().worker_factory, _make_leaf)


# ---------------------------------------------------------------------------
# T2.7: lazy fields rebuild from the live factory, never the interim partial
# ---------------------------------------------------------------------------


class LazyFieldRebuildTest(unittest.TestCase):
    def test_lazy_worker_field_rebuilds_from_the_live_factory(self):
        cfg = OmegaConf.create(
            {
                "_target_": "BTA",
                "predefined_sub_queries": ["x"],
                "worker_inferencers": {
                    "_target_": f"{Leaf.__module__}.{Leaf.__qualname__}",
                    "response": "w",
                },
            }
        )
        src = instantiate(cfg)
        self.assertIs(src._init_recipe["worker_inferencers"], _LIVE_FIELD)
        self.assertIsInstance(src.worker_inferencers, LazyConfigFactory)

        factory = src.fresh_instance().worker_inferencers

        self.assertIsInstance(factory, LazyConfigFactory)
        self.assertIsNot(factory, src.worker_inferencers)
        first, second = factory(), factory()
        self.assertIsNot(first, second)
        self.assertEqual((first._response, second._response), ("w", "w"))


# ---------------------------------------------------------------------------
# T2.8: MFI re-derivation; Metamate copies
# ---------------------------------------------------------------------------


def _flow(query, initial, followup, **extra):
    return {
        "input": query,
        "initial_inferencer": initial,
        "followup_inferencer": followup,
        "max_dynamic_steps": 1,
        **extra,
    }


class DerivedStateRebuildTest(_TmpDirMixin, unittest.TestCase):
    def test_mfi_rederives_sub_queries_and_keeps_one_shared_leaf(self):
        leaf = Leaf(response="f")
        src = MultiFlowInferencer(
            flow_configs=[
                _flow("a", leaf, leaf, end_condition=_stop),
                _flow("b", leaf, leaf, end_condition=_stop),
            ],
            disable_aggregator=True,
            checkpoint_dir=self.tmp,
        )

        fresh = src.fresh_instance()

        self.assertEqual(fresh.predefined_sub_queries, ["a", "b"])
        self.assertIsNot(fresh.predefined_sub_queries, src.predefined_sub_queries)
        slots = [
            cfg[key]
            for cfg in fresh.flow_configs
            for key in ("initial_inferencer", "followup_inferencer")
        ]
        self.assertTrue(all(slot is slots[0] for slot in slots))
        self.assertIsNot(slots[0], leaf)

    def test_mfi_iteration_judgment_is_reapplied_to_the_fresh_followup(self):
        followup = PromptLeaf()
        src = MultiFlowInferencer(
            flow_configs=[_flow("a", Leaf(), followup, iteration_judgment=True)],
            disable_aggregator=True,
            checkpoint_dir=self.tmp,
        )

        cfg = src.fresh_instance().flow_configs[0]

        self.assertIsNot(cfg["followup_inferencer"], followup)
        self.assertEqual(
            cfg["followup_inferencer"].template_extra_feed,
            {"include_iteration_judgment": True},
        )
        self.assertIsNot(
            cfg["followup_inferencer"].template_extra_feed,
            followup.template_extra_feed,
        )
        self.assertIs(cfg["end_condition"], parse_decision_stop)

    def test_metamate_copies_share_the_stateless_scope_judge(self):
        src = MetamateSDKInferencer()
        self.assertIs(src.fresh_instance().code_scope_judge, src.code_scope_judge)
        self.assertIs(copy.deepcopy(src).code_scope_judge, src.code_scope_judge)


# ---------------------------------------------------------------------------
# T2.9: every registered default-constructible inferencer rebuilds unshared
# ---------------------------------------------------------------------------


def _default_instance(alias):
    """A default-constructed instance of ``alias``, or ``None`` when it is not an
    inferencer, its optional backend is not installed, or it requires arguments."""
    try:
        cls = import_target(alias)
    except ImportError:
        return None
    if not (isinstance(cls, type) and issubclass(cls, InferencerBase)):
        return None
    try:
        return cls()
    except (TypeError, ValueError):
        return None


class RegisteredClassAuditTest(unittest.TestCase):
    def test_every_default_constructible_inferencer_rebuilds_unshared(self):
        audited = set()
        for alias in sorted(list_registered()):
            source = _default_instance(alias)
            if source is None:
                continue
            audited.add(alias)
            with self.subTest(alias=alias):
                self._assert_independent_copy(source)
        self.assertLessEqual(_CORE_AUDITED, audited)

    def _assert_independent_copy(self, source):
        fresh = source.fresh_instance()
        self.assertIs(type(fresh), type(source))
        self.assertIsNot(fresh, source)
        self.assertTrue(hasattr(fresh, "__dict__"))
        fresh_reach = _reachable_inferencers(fresh)
        self.assertFalse(_reachable_inferencers(source).keys() & fresh_reach.keys())
        for inf in fresh_reach.values():
            self.assertIsNone(inf._workspace, type(inf).__name__)


# ---------------------------------------------------------------------------
# T2.10: identity and parent links are dropped, never walked
# ---------------------------------------------------------------------------


class ParentLinksTest(unittest.TestCase):
    def test_identity_and_parent_links_are_not_carried(self):
        # Parents without a recipe: rebuilding would raise if it walked them.
        src = Leaf(id="fixed-id", parent_debuggables=[VarArgsLeaf()])
        src.set_parent_debuggable(VarArgsLeaf())

        fresh = src.fresh_instance()

        self.assertFalse(fresh.parent_debuggables)
        self.assertNotEqual(fresh.id, src.id)


# ---------------------------------------------------------------------------
# T2.11: prepared_input skips preprocessing and rendering
# ---------------------------------------------------------------------------


def _templated_leaf(seen):
    def _preprocess(text):
        seen.append(text)
        return f"PRE:{text}"

    return PromptLeaf(
        template_manager=_render,
        template_key="k",
        template_extra_feed={"task_instructions": "TI"},
        input_preprocessor=_preprocess,
    )


class PreparedInputTest(unittest.IsolatedAsyncioTestCase):
    def test_normal_call_preprocesses_renders_and_snapshots(self):
        seen = []
        leaf = _templated_leaf(seen)
        leaf.infer("q")
        self.assertEqual(seen, ["q"])
        self.assertEqual(leaf.prompts, ["RENDERED[k]:PRE:q"])
        self.assertEqual(leaf._last_rendered_task_instructions, "TI")

    def test_prepared_input_is_sent_verbatim(self):
        seen = []
        leaf = _templated_leaf(seen)
        self.assertEqual(leaf.infer("VERBATIM", prepared_input=True), "ok")
        self.assertEqual(seen, [])
        self.assertEqual(leaf.prompts, ["VERBATIM"])
        self.assertEqual(leaf._last_rendered_task_instructions, "")

    async def test_async_prepared_input_is_sent_verbatim(self):
        seen = []
        leaf = _templated_leaf(seen)
        self.assertEqual(await leaf.ainfer("VERBATIM", prepared_input=True), "ok")
        self.assertEqual(seen, [])
        self.assertEqual(leaf.prompts, ["VERBATIM"])
        self.assertEqual(leaf._last_rendered_task_instructions, "")


# ---------------------------------------------------------------------------
# T2.12: _FreshCloneFactory as the BTA worker factory
# ---------------------------------------------------------------------------


class _Recorder:
    """Shared by reference across ``fresh_instance`` copies (not an exact list)."""

    def __init__(self):
        self.ran = []
        self.closed = []


@attrs
class RecordingLeaf(Leaf):
    recorder = attrib(default=None, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.recorder.ran.append(self)
        return super()._infer(inference_input, inference_config, **kwargs)

    async def adisconnect(self):
        self.recorder.closed.append(self)


class FreshCloneFactoryWorkersTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    def _bta(self, proto):
        return BreakdownThenAggregateInferencer(
            predefined_sub_queries=["a", "b", "c"],
            worker_inferencers=_FreshCloneFactory(proto),
            aggregator_inferencer=Leaf(response="agg"),
            workspace=InferencerWorkspace(root=self.tmp),
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )

    def _proto(self):
        return RecordingLeaf(
            response="w",
            recorder=_Recorder(),
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )

    def _assert_fresh_workers(self, proto):
        workers = proto.recorder.ran
        self.assertEqual(len(workers), 3)
        self.assertEqual(len({id(w) for w in workers}), 3)
        self.assertFalse(any(w is proto for w in workers))
        self.assertTrue(
            all(type(w) is RecordingLeaf and w._response == "w" for w in workers)
        )
        # Each clone is owned by its attempt and closed when it ends.
        self.assertEqual(
            {id(w) for w in proto.recorder.closed}, {id(w) for w in workers}
        )

    async def test_async_bta_builds_one_fresh_worker_per_sub_query(self):
        proto = self._proto()
        await self._bta(proto).ainfer("task")
        self._assert_fresh_workers(proto)

    def test_sync_bta_builds_one_fresh_worker_per_sub_query(self):
        proto = self._proto()
        self._bta(proto).infer("task")
        self._assert_fresh_workers(proto)


# ---------------------------------------------------------------------------
# T2.13: resume tails complete the response without the response logs
# ---------------------------------------------------------------------------


def _log_type_recorder(log_types):
    def _record(*args, **kwargs):
        log_types.append(args[1] if len(args) > 1 else kwargs.get("log_type"))

    return _record


class ResumeTailLoggingTest(unittest.IsolatedAsyncioTestCase):
    _RESPONSE_LOGS = {"InferenceResponse", "PostProcessedResponse"}

    def _run(self, leaf, call):
        log_types = []
        with patch.object(leaf, "log_debug", side_effect=_log_type_recorder(log_types)):
            result = call(leaf)
        return result, set(log_types)

    def test_sync_resume_post_processes_without_response_logs(self):
        leaf = CachedLeaf(response_post_processor=str.upper)
        result, log_types = self._run(leaf, lambda inf: inf.infer("q"))
        self.assertEqual(result, "CACHED")
        self.assertFalse(log_types & self._RESPONSE_LOGS)

    async def test_async_resume_post_processes_without_response_logs(self):
        leaf = CachedLeaf(response_post_processor=str.upper)
        log_types = []
        with patch.object(leaf, "log_debug", side_effect=_log_type_recorder(log_types)):
            result = await leaf.ainfer("q")
        self.assertEqual(result, "CACHED")
        self.assertFalse(set(log_types) & self._RESPONSE_LOGS)

    def test_normal_call_logs_response_and_post_processed_response(self):
        leaf = Leaf(response="done", response_post_processor=str.upper)
        result, log_types = self._run(leaf, lambda inf: inf.infer("q"))
        self.assertEqual(result, "DONE")
        self.assertLessEqual(self._RESPONSE_LOGS, log_types)

"""Locks the breakdown block-registry work:

1. ``aggregation_guidance`` survives a resume. It is captured as a side effect of
   ``_parse_json_subtasks``, which resume short-circuits — so it is recovered from
   the promoted breakdown checkpoint
   (``checkpoints/breakdown/decomposed_subtasks.json``), or the aggregator silently
   loses its guidance section (no log, no error).
2. ``_subtasks_from_fence_dict`` — the TRANSFORM half, split out at the ``data``
   boundary so one implementation serves the text parser and anything holding the
   already-parsed fence.
3. The block is registered ONCE on the class (``SLOT_DEFAULTS`` →
   ``BREAKDOWN_TEMPLATE_DEFAULTS``) rather than in each of ~12 topology YAMLs.
"""

import contextlib
import json
import os
import shutil
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import open_invocation
from agent_foundation.common.inferencers.template_defaults import (
    BREAKDOWN_TEMPLATE_DEFAULTS,
)
from attr import attrib, attrs


@attrs
class _StubAggregator:
    """Stands in for the aggregator slot — only ``template_extra_feed`` matters."""

    template_extra_feed = attrib(factory=dict)


def _bare_bta(**attrs_):
    """A BTA carrying only the fields the parse/checkpoint paths touch."""
    obj = BreakdownThenAggregateInferencer.__new__(BreakdownThenAggregateInferencer)
    obj.worker_query_fields = ("description", "todos")
    obj.expand_todos_to_workers = False
    obj.aggregator_inferencer = None
    obj.inject_upstream_artifacts_to_aggregator = True
    for k, v in attrs_.items():
        setattr(obj, k, v)
    return obj


@contextlib.contextmanager
def _in_attempt(bta):
    """BTA's private hooks run inside an attempt of their owner's invocation;
    yields the attempt."""
    with open_invocation(bta):
        yield bta._open_attempt("q", use_async=False)


class TestAggregationGuidanceSurvivesResume(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _bta(self):
        bta = _bare_bta(
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=True,
        )
        ws = InferencerWorkspace(root=self.tmpdir)
        ws.ensure_dirs()
        # Set the backing field directly: the ``_workspace`` SETTER fires
        # production side effects (``_configure_for_workspace`` /
        # ``_propagate_workspace_to_children``) a ``__new__``-built bare object
        # can't satisfy; the getter reads this mangled backing verbatim.
        object.__setattr__(bta, "_InferencerBase__workspace", ws)
        return bta

    def _write_promoted(self, bta, guidance=None):
        """Write the promoted breakdown fence the resume path reads — the same
        ``checkpoints/breakdown/decomposed_subtasks.json`` the promoter publishes.
        """
        fence = {"subtasks": [{"description": "D"}]}
        if guidance is not None:
            fence["aggregation_guidance"] = guidance
        path = bta._workspace.checkpoint_path(
            os.path.join("breakdown", "decomposed_subtasks.json")
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fence, f)
        return path

    def test_promoted_checkpoint_restores_guidance(self):
        loader = self._bta()
        self._write_promoted(loader, guidance="MERGE BY THEME")

        with _in_attempt(loader):
            sub_queries, guidance = loader._load_promoted_breakdown()
        self.assertTrue(sub_queries)
        self.assertEqual(guidance, "MERGE BY THEME")

    def test_promoted_checkpoint_without_guidance_loads_as_none(self):
        # A breakdown that produced no guidance: sub_queries present, guidance None.
        loader = self._bta()
        self._write_promoted(loader, guidance=None)

        with _in_attempt(loader):
            sub_queries, guidance = loader._load_promoted_breakdown()
        self.assertTrue(sub_queries)
        self.assertIsNone(guidance)

    def _wire_aggregator(self, bta, feed=None):
        """Attach a stub aggregator and stub out the unrelated collaborators
        ``_inject_aggregator_extra_feed`` touches (logging, upstream formatting)."""
        agg = _StubAggregator()
        if feed is not None:
            agg.template_extra_feed = feed
        bta.aggregator_inferencer = agg
        bta.log_info = lambda *a, **k: None
        bta._format_worker_results_text = lambda *a, **k: "upstream"
        return agg

    def test_guidance_reaches_the_aggregator_feed_after_resume(self):
        # The point of the mechanism: after a resume (the attempt's breakdown
        # never parsed, so it holds no guidance) the aggregator prompt still gets
        # its guidance, restored lazily from the promoted breakdown checkpoint.
        loader = self._bta()
        self._write_promoted(loader, guidance="MERGE BY THEME")
        agg = self._wire_aggregator(loader)

        with _in_attempt(loader) as attempt:
            self.assertIsNone(attempt.aggregation_guidance)
            loader._inject_aggregator_extra_feed(["r1"])

        self.assertEqual(
            agg.template_extra_feed.get("aggregation_guidance"), "MERGE BY THEME"
        )

    def test_negative_control_missing_guidance_is_actively_popped(self):
        # Why a lost guidance is silent AND destructive: with neither in-process
        # guidance NOR a promoted checkpoint, the key is not merely absent — it is
        # POPPED from the feed (stale-guidance guard).
        bta = self._bta()  # workspace present, but no promoted file written
        agg = self._wire_aggregator(bta, feed={"aggregation_guidance": "STALE"})
        with _in_attempt(bta) as attempt:
            self.assertIsNone(attempt.aggregation_guidance)
            bta._inject_aggregator_extra_feed(["r1"])
        self.assertNotIn("aggregation_guidance", agg.template_extra_feed)


class TestSubtasksFromFenceDict(unittest.TestCase):
    FENCE_DICT = {
        "subtasks": [{"description": "D1", "todos": ["t1"]}],
        "aggregation_guidance": "GUIDE",
    }

    def test_matches_the_full_text_parser(self):
        direct = _bare_bta()._subtasks_from_fence_dict(self.FENCE_DICT)
        self.assertIsNotNone(direct)
        queries, guidance = direct

        text = (
            "<Response>\n```json decomposed_subtasks\n"
            + json.dumps(self.FENCE_DICT)
            + "\n```\n</Response>"
        )
        self.assertEqual(queries, _bare_bta()._parse_json_subtasks(text))
        self.assertEqual(_bare_bta()._parse_json_breakdown(text), (queries, "GUIDE"))
        self.assertEqual(guidance, "GUIDE")

    def test_is_pure_wrt_instance_state(self):
        # Writes nothing on the instance — the caller records the guidance, so a
        # shared / re-roled inferencer cannot be polluted by a bare transform call.
        bta = _bare_bta()
        before = dict(vars(bta))
        bta._subtasks_from_fence_dict(self.FENCE_DICT)
        self.assertEqual(vars(bta), before)

    def test_no_subtasks_returns_none(self):
        self.assertIsNone(_bare_bta()._subtasks_from_fence_dict({"subtasks": []}))

    def test_missing_guidance_is_none(self):
        _q, guidance = _bare_bta()._subtasks_from_fence_dict(
            {"subtasks": [{"description": "D"}]}
        )
        self.assertIsNone(guidance)

    def test_registered_as_the_block_parser(self):
        self.assertEqual(
            BreakdownThenAggregateInferencer.BLOCK_PARSERS.get("decomposed_subtasks"),
            "_subtasks_from_fence_dict",
        )


class TestBreakdownSlotDefaultRegistersTheBlock(unittest.TestCase):
    def test_slot_default_carries_expected_extraction(self):
        node = {}
        BREAKDOWN_TEMPLATE_DEFAULTS.apply_to(node)
        specs = node.get("expected_extraction")
        self.assertTrue(specs, "breakdown slot should register its block by default")
        spec = specs[0]
        self.assertEqual(spec["label"], "decomposed_subtasks")
        self.assertEqual(spec["source"], "response")
        self.assertEqual(spec["persist_to"], "decomposed_subtasks.json")
        # The declarative promotion marker: on a fresh run the breakdown child's
        # decomposed_subtasks.json is pulled up into the parent's checkpoints/ so
        # resume rebuilds the fan-out + guidance from it (retires
        # breakdown_result.json).
        self.assertEqual(spec["checkpoint_scope"], "parent")
        # A non-JSON breakdown (numbered_list, or "auto" falling back) legitimately
        # has no fence; fallback_to_source keeps the universal default quiet.
        self.assertTrue(spec["fallback_to_source"])

    def test_template_root_space_still_applied(self):
        node = {}
        BREAKDOWN_TEMPLATE_DEFAULTS.apply_to(node)
        self.assertEqual(node.get("template_root_space"), "task_breakdown")

    def test_yaml_can_still_override(self):
        node = {"expected_extraction": []}
        BREAKDOWN_TEMPLATE_DEFAULTS.apply_to(node)
        self.assertEqual(node["expected_extraction"], [], "fill-iff-absent")

    def test_defaults_are_deep_copied_per_node(self):
        # Two nodes must not share the same list object, or a mutation in one
        # topology would leak into every other.
        a, b = {}, {}
        BREAKDOWN_TEMPLATE_DEFAULTS.apply_to(a)
        BREAKDOWN_TEMPLATE_DEFAULTS.apply_to(b)
        self.assertIsNot(a["expected_extraction"], b["expected_extraction"])


if __name__ == "__main__":
    unittest.main()

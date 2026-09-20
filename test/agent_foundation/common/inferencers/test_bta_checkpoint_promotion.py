"""Locks the generic child->parent checkpoint promotion that retires BTA's
hand-rolled ``breakdown_result.json``.

One declarative mechanism replaces the bespoke resume file: an
``expected_extraction`` entry marked ``checkpoint_scope: "parent"`` makes the
extraction register's own output durable at the PARENT's ``checkpoints/<child>/``,
and on resume that one promoted ``decomposed_subtasks.json`` feeds BOTH the
worker fan-out (via the ``subgraph_registry`` factories) and the aggregator's
guidance. These deterministic tests are the authoritative gate:

* (T1) Promoter contract (``InferencerBase._promote_child_checkpoints``): a child
  declaring ``checkpoint_scope="parent"`` is copied up atomically; a
  non-declaring child is not; missing source / missing workspace are no-ops.
* (T2) Resume rebuild: the ``subgraph_registry`` factory rebuilds the worker
  fan-out purely from the promoted file — no ``breakdown_result.json``, no
  ``_cached_sub_queries``.
* (T4) Memoized loader: ``_load_promoted_breakdown`` reads/parses at most once.
* (T5) Seed left ``None``: the emitted ``GraphExpansionResult`` carries no seed
  (the generic WorkGraph reconstruction path drops it — nothing reads it).
"""

import filecmp
import json
import os
import shutil
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from attr import attrib, attrs


@attrs
class _MockInferencer(InferencerBase):
    """Minimal concrete InferencerBase returning a fixed response."""

    _response = attrib(default="mock response")
    _call_count = attrib(default=0, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._call_count += 1
        return self._response


# The breakdown fence the extraction register writes to
# ``outputs/decomposed_subtasks.json`` — 3 subtasks + guidance. The promoter
# publishes it verbatim; ``_subtasks_from_fence_dict`` derives the sub_queries.
_FENCE = {
    "subtasks": [
        {"description": "Alpha"},
        {"description": "Beta"},
        {"description": "Gamma"},
    ],
    "aggregation_guidance": "MERGE BY THEME",
}


def _json_breakdown_response(descriptions):
    """A ```json`` fenced breakdown response the json_subtasks parser accepts."""
    return (
        "```json\n"
        + json.dumps({"subtasks": [{"description": d} for d in descriptions]})
        + "\n```"
    )


class PromoteChildCheckpointsTest(unittest.TestCase):
    """(T1) ``InferencerBase._promote_child_checkpoints`` — the generic PULL promoter."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.parent_ws = InferencerWorkspace(root=self.tmpdir)
        self.parent_ws.ensure_dirs()
        self.parent = _MockInferencer()
        self.parent._workspace = self.parent_ws

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_child(self, expected_extraction, write_output=True):
        child_ws = self.parent_ws.child("breakdown")
        child_ws.ensure_dirs()
        child = _MockInferencer()
        child._workspace = child_ws
        child.expected_extraction = expected_extraction
        if write_output:
            out = child_ws.output_path("decomposed_subtasks.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(_FENCE, f)
        return child, child_ws

    def _promoted_path(self):
        return self.parent_ws.checkpoint_path(
            os.path.join("breakdown", "decomposed_subtasks.json")
        )

    def test_declaring_child_is_promoted_bytewise(self):
        child, child_ws = self._make_child(
            [
                {
                    "label": "decomposed_subtasks",
                    "source": "response",
                    "persist_to": "decomposed_subtasks.json",
                    "checkpoint_scope": "parent",
                }
            ]
        )
        self.parent._promote_child_checkpoints(child)

        promoted = self._promoted_path()
        self.assertTrue(os.path.isfile(promoted), "declaring child must be promoted")
        with open(promoted, encoding="utf-8") as f:
            self.assertEqual(json.load(f), _FENCE)
        self.assertTrue(
            filecmp.cmp(
                child_ws.output_path("decomposed_subtasks.json"),
                promoted,
                shallow=False,
            ),
            "promoted file must be byte-identical to the child's output",
        )

    def test_non_declaring_child_not_promoted(self):
        # Same entry WITHOUT checkpoint_scope — the register still wrote the file,
        # but nothing opts it into parent-scope promotion.
        child, _ = self._make_child(
            [
                {
                    "label": "decomposed_subtasks",
                    "source": "response",
                    "persist_to": "decomposed_subtasks.json",
                }
            ]
        )
        self.parent._promote_child_checkpoints(child)
        self.assertFalse(
            os.path.exists(self._promoted_path()),
            "child without checkpoint_scope='parent' must NOT be promoted",
        )

    def test_promotion_leaves_no_tmp_sibling(self):
        child, _ = self._make_child(
            [{"persist_to": "decomposed_subtasks.json", "checkpoint_scope": "parent"}]
        )
        self.parent._promote_child_checkpoints(child)
        self.assertTrue(os.path.isfile(self._promoted_path()))
        self.assertFalse(
            os.path.exists(self._promoted_path() + ".tmp"),
            "atomic tmp->rename must leave no .tmp file behind",
        )

    def test_missing_source_is_noop(self):
        # Declares promotion but the register never wrote the file (e.g. a crash
        # before breakdown finished): best-effort promoter must not raise and must
        # not create a target.
        child, _ = self._make_child(
            [{"persist_to": "decomposed_subtasks.json", "checkpoint_scope": "parent"}],
            write_output=False,
        )
        self.parent._promote_child_checkpoints(child)
        self.assertFalse(os.path.exists(self._promoted_path()))

    def test_no_parent_workspace_is_noop(self):
        child, _ = self._make_child(
            [{"persist_to": "decomposed_subtasks.json", "checkpoint_scope": "parent"}]
        )
        parent = _MockInferencer()  # no _workspace bound
        parent._promote_child_checkpoints(child)  # must not raise
        self.assertFalse(os.path.exists(self._promoted_path()))

    def test_path_traversal_persist_to_is_rejected(self):
        # A malformed/hostile persist_to must never escape checkpoints/<child>/.
        child, _ = self._make_child(
            [{"persist_to": "../escape.json", "checkpoint_scope": "parent"}]
        )
        self.parent._promote_child_checkpoints(child)
        self.assertFalse(
            os.path.exists(os.path.join(self.tmpdir, "escape.json")),
            "'..' in persist_to must be rejected, not followed",
        )


class ResumeRebuildFromPromotedFileTest(unittest.TestCase):
    """(T2/T4) The ``subgraph_registry`` factory rebuilds the worker fan-out
    purely from the promoted ``decomposed_subtasks.json`` — with no
    ``breakdown_result.json`` and no ``_cached_sub_queries``."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_bta(self):
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=_MockInferencer(response="unused"),
            worker_inferencers=lambda sub_query, index: _MockInferencer(
                response=f"w{index}"
            ),
            aggregator_inferencer=None,
            checkpoint_dir=self.tmpdir,
            resume_with_saved_results=True,
        )
        ws = InferencerWorkspace(root=self.tmpdir)
        ws.ensure_dirs()
        bta._workspace = ws
        bta.name = "test_bta"
        return bta, ws

    def _write_promoted(self, ws):
        path = ws.checkpoint_path(os.path.join("breakdown", "decomposed_subtasks.json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_FENCE, f)
        return path

    def test_factory_rebuilds_fanout_from_promoted_file(self):
        bta, ws = self._make_bta()
        self._write_promoted(ws)
        # No hand-rolled resume file exists, and the loader memo is cold — this is
        # a fresh resume process rebuilding from disk alone.
        self.assertFalse(
            os.path.exists(os.path.join(self.tmpdir, "breakdown_result.json"))
        )
        self.assertIsNone(bta._promoted_breakdown_cache)

        spec = bta.subgraph_registry["bta_workers"]("expansion-id")

        worker_nodes = [n for n in spec.nodes if "worker" in n.name]
        self.assertEqual(len(worker_nodes), len(_FENCE["subtasks"]))

    def test_loader_returns_queries_and_guidance(self):
        bta, ws = self._make_bta()
        self._write_promoted(ws)
        sub_queries, guidance = bta._load_promoted_breakdown()
        self.assertEqual(len(sub_queries), len(_FENCE["subtasks"]))
        self.assertEqual(guidance, "MERGE BY THEME")

    def test_absent_promoted_file_returns_none_pair(self):
        bta, _ = self._make_bta()  # nothing written to checkpoints/
        self.assertEqual(bta._load_promoted_breakdown(), (None, None))

    def test_loader_is_memoized_single_read(self):
        bta, ws = self._make_bta()
        path = self._write_promoted(ws)
        first = bta._load_promoted_breakdown()
        # Remove the file: a genuine second read would now miss. The memo must
        # serve the identical parsed tuple, proving a single read+parse.
        os.remove(path)
        second = bta._load_promoted_breakdown()
        self.assertIs(first, second)
        self.assertEqual(len(second[0]), len(_FENCE["subtasks"]))

    def test_unparseable_promoted_file_degrades_to_none(self):
        bta, ws = self._make_bta()
        path = ws.checkpoint_path(os.path.join("breakdown", "decomposed_subtasks.json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ not valid json !!!")
        # Graceful degradation (the corrupted-checkpoint contract): behave as
        # "no checkpoint" so Step 0 runs a fresh breakdown instead of erroring.
        self.assertEqual(bta._load_promoted_breakdown(), (None, None))


class SeedLeftNoneTest(unittest.TestCase):
    """(T5) BTA emits ``GraphExpansionResult`` with ``seed=None`` — the generic
    WorkGraph Priority-2 reconstruction drops the seed, so BTA stops populating
    it (the durable state lives in the promoted file, not the seed)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_bta(self):
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=_MockInferencer(
                response=_json_breakdown_response(["A", "B"])
            ),
            worker_inferencers=lambda sub_query, index: _MockInferencer(
                response=f"w{index}"
            ),
            aggregator_inferencer=None,
            breakdown_format="json_subtasks",
            checkpoint_dir=self.tmpdir,
        )
        ws = InferencerWorkspace(root=self.tmpdir)
        ws.ensure_dirs()
        bta._workspace = ws
        bta.name = "seed_bta"
        return bta

    def test_emitted_expansion_has_no_seed(self):
        bta = self._make_bta()
        fn = bta._make_breakdown_fn("decompose this", None)
        result = fn()  # sync path (use_async defaults False)
        self.assertIsNone(result.seed)
        self.assertIsNone(result.reconstruct_from_seed)
        # Sanity: the fresh path still produced a real fan-out subgraph.
        self.assertIsNotNone(result.subgraph)


class NoBreakdownResultJsonTest(unittest.TestCase):
    """(T6) The hand-rolled ``breakdown_result.json`` is fully retired — its
    save/load pair is gone and the fresh breakdown path never writes it."""

    def test_save_load_methods_removed(self):
        self.assertFalse(
            hasattr(BreakdownThenAggregateInferencer, "_save_breakdown_checkpoint"),
            "_save_breakdown_checkpoint must be deleted",
        )
        self.assertFalse(
            hasattr(BreakdownThenAggregateInferencer, "_load_breakdown_checkpoint"),
            "_load_breakdown_checkpoint must be deleted",
        )

    def test_breakdown_path_writes_no_breakdown_result_json(self):
        tmpdir = tempfile.mkdtemp()
        try:
            bta = BreakdownThenAggregateInferencer(
                breakdown_inferencer=_MockInferencer(
                    response=_json_breakdown_response(["A", "B"])
                ),
                worker_inferencers=lambda sub_query, index: _MockInferencer(
                    response=f"w{index}"
                ),
                aggregator_inferencer=None,
                breakdown_format="json_subtasks",
                checkpoint_dir=tmpdir,
            )
            ws = InferencerWorkspace(root=tmpdir)
            ws.ensure_dirs()
            bta._workspace = ws
            bta.name = "t6_bta"
            # Run the breakdown emit path — exactly where the retired
            # _save_breakdown_checkpoint used to write breakdown_result.json.
            bta._make_breakdown_fn("decompose this", None)()

            offenders = [
                os.path.join(root, "breakdown_result.json")
                for root, _dirs, files in os.walk(tmpdir)
                if "breakdown_result.json" in files
            ]
            self.assertEqual(
                offenders, [], f"breakdown_result.json must not appear: {offenders}"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

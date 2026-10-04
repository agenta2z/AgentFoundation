"""BTA worker results pair with their sub-queries by declaration index.

Worker ``i``'s result reaches the aggregator (and the no-aggregator return) in
slot ``i`` regardless of completion order, failed workers, ``max_concurrency``
or resume from checkpoints written by an older build. The aggregator's
per-call feed is published ctx-scoped under a RunContext, composed over any
ancestor override.
"""

import asyncio
import json
import os
import pickle
import shutil
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    _FailedWorkerSentinel,
    align_worker_results,
    BreakdownThenAggregateInferencer,
    make_upstream_injecting_aggregator_prompt_builder,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import open_invocation, RunContext
from agent_foundation.common.inferencers.template_feed_scope import (
    TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode

_BREAKDOWN = "1. q0\n2. q1\n3. q2"
_MARKER = "__bta_worker_outcome__"
_BOOM = "ValueError: boom"


@attrs
class DelayedLeaf(InferencerBase):
    """Returns ``response(input)`` (or ``response``) after ``delay`` seconds."""

    _response = attrib(default="leaf")
    _delay = attrib(default=0.0)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        if callable(self._response):
            return self._response(inference_input)
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        await asyncio.sleep(self._delay)
        return self._infer(inference_input, inference_config, **kwargs)


def _render_feed(key, active_template_root_space=None, master_version=None, **feed):
    return (
        f"INPUT={feed.get('input', '')}|"
        f"UPSTREAM={feed.get('upstream_artifacts', '<none>')}|"
        f"ANCESTOR={feed.get('ancestor_only', '<none>')}"
    )


@attrs(slots=False)
class FeedAggregator(TemplatedInferencerBase):
    """Templated aggregator whose prompt shows the feed it rendered with."""

    prompts = attrib(factory=list)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.prompts.append(inference_input)
        return "AGG"


class StreamRecorder:
    """Graph reporter that records every streamed payload."""

    def __init__(self):
        self.streams = []

    async def on_graph_topology(self, event):
        pass

    async def on_node_status(self, node_id, status, error="", output_path=""):
        pass

    async def on_node_stream(self, node_id, content, is_final=True):
        self.streams.append((node_id, content))

    async def on_graph_reconcile(self, node_statuses):
        pass

    def child_reporter(self, parent_node_id):
        return self

    def node_stream_observer(self, node_id, flush_interval_ms=200.0):
        async def _observer(chunk):
            self.streams.append((node_id, chunk))

        return _observer

    def node_interactive(self, node_id):
        return None


def _recording_leaf(log, response="AGG"):
    def _respond(inference_input):
        log.append(inference_input)
        return response

    return DelayedLeaf(response=_respond)


def _failing_leaf():
    def _respond(_inference_input):
        raise RuntimeError("aggregator down")

    return DelayedLeaf(response=_respond, fallback_mode=FallbackMode.NEVER, max_retry=0)


def _worker_factory(delays, fail_index=None, completions=None, prefix="out"):
    def factory(sub_query, index):
        def _respond(_inference_input):
            if index == fail_index:
                raise ValueError("boom")
            if completions is not None:
                completions.append(index)
            return f"{prefix}-{index}"

        return DelayedLeaf(
            response=_respond,
            delay=delays[index],
            output_path="worker.md",
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
        )

    return factory


def _bta(
    root=None, *, delays=(0.0, 0.0, 0.0), fail_index=None, completions=None, **kwargs
):
    kwargs.setdefault("breakdown_inferencer", DelayedLeaf(response=_BREAKDOWN))
    kwargs.setdefault("aggregator_inferencer", None)
    return BreakdownThenAggregateInferencer(
        worker_inferencers=_worker_factory(delays, fail_index, completions),
        workspace=InferencerWorkspace(root=root) if root else None,
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
        **kwargs,
    )


def _sections(values, failures=None):
    """The ``### Upstream Outcome N`` text for index-aligned ``values``."""
    failures = failures or {}
    return "\n\n".join(
        f"### Upstream Outcome {i + 1}\n"
        + (f"(failed: {failures[i]})" if i in failures else values[i])
        for i in range(len(values))
    )


def _synthetic_sections(text):
    """Map each ``## Upstream Outcome N`` of a synthetic aggregation to its body."""
    parts = text.split("\n\n## Upstream Outcome ")[1:]
    return {
        int(part.split("\n", 1)[0]): part.split("\n", 1)[1].strip() for part in parts
    }


class _TmpDirMixin:
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()


# ---------------------------------------------------------------------------
# T1.1: completion order does not reorder the aggregator input
# ---------------------------------------------------------------------------


class OutOfOrderCompletionTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_aggregator_input_is_in_declaration_order(self):
        completions, agg_inputs = [], []
        bta = _bta(
            self.tmp,
            delays=(0.4, 0.2, 0.0),
            completions=completions,
            aggregator_inferencer=_recording_leaf(agg_inputs),
            inject_upstream_artifacts_to_aggregator=False,
        )

        result = await bta.ainfer("orig")

        self.assertEqual(completions, [2, 1, 0])
        self.assertEqual(agg_inputs, [_sections(["out-0", "out-1", "out-2"])])
        self.assertEqual(result, "AGG")


class SyncDeclarationOrderTest(_TmpDirMixin, unittest.TestCase):
    def test_sync_aggregator_input_is_in_declaration_order(self):
        agg_inputs = []
        bta = _bta(
            self.tmp,
            aggregator_inferencer=_recording_leaf(agg_inputs),
            inject_upstream_artifacts_to_aggregator=False,
        )

        result = bta.infer("orig")

        self.assertEqual(agg_inputs, [_sections(["out-0", "out-1", "out-2"])])
        self.assertEqual(result, "AGG")


# ---------------------------------------------------------------------------
# T1.2: a failed middle worker keeps its slot (no shift of later workers)
# ---------------------------------------------------------------------------


class FailedWorkerSlotTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_failed_worker_is_labeled_at_its_own_index(self):
        agg_inputs = []
        bta = _bta(
            self.tmp,
            delays=(0.05, 0.0, 0.02),
            fail_index=1,
            min_successful_workers=2,
            aggregator_inferencer=_recording_leaf(agg_inputs),
            inject_upstream_artifacts_to_aggregator=False,
        )

        await bta.ainfer("orig")

        self.assertEqual(agg_inputs, [_sections(["out-0", None, "out-2"], {1: _BOOM})])

    async def test_synthetic_aggregation_labels_the_failed_shard(self):
        bta = _bta(
            self.tmp,
            delays=(0.05, 0.0, 0.02),
            fail_index=1,
            min_successful_workers=2,
            aggregator_inferencer=_failing_leaf(),
            inject_upstream_artifacts_to_aggregator=False,
        )

        result = await bta.ainfer("orig")

        self.assertTrue(result.startswith("# Synthetic Aggregation"), result)
        sections = _synthetic_sections(result)
        self.assertEqual(sorted(sections), [1, 2, 3])
        self.assertEqual(sections[2], f"(failed: {_BOOM})")
        for idx in (0, 2):
            body = sections[idx + 1]
            self.assertTrue(body.endswith(f"out-{idx}"), body)
            self.assertIn(f"worker_0{idx}", body)


# ---------------------------------------------------------------------------
# T1.3: the sync path never hands failure markers to the aggregator and
# enforces the quorum
# ---------------------------------------------------------------------------


class SyncQuorumTest(_TmpDirMixin, unittest.TestCase):
    def test_failure_reaches_aggregator_as_a_labeled_slot(self):
        agg_inputs = []
        bta = _bta(
            self.tmp,
            fail_index=1,
            min_successful_workers=2,
            aggregator_inferencer=_recording_leaf(agg_inputs),
            inject_upstream_artifacts_to_aggregator=False,
        )

        bta.infer("orig")

        self.assertEqual(agg_inputs, [_sections(["out-0", None, "out-2"], {1: _BOOM})])
        self.assertNotIn(_MARKER, agg_inputs[0])

    def test_custom_builder_gets_survivor_values_and_failures(self):
        calls, agg_inputs = [], []

        def builder(worker_results, original_query=None, worker_failures=None):
            calls.append((list(worker_results), dict(worker_failures or {})))
            return "built"

        bta = _bta(
            self.tmp,
            fail_index=1,
            min_successful_workers=2,
            aggregator_inferencer=_recording_leaf(agg_inputs),
            aggregator_prompt_builder=builder,
        )

        bta.infer("orig")

        self.assertEqual(calls, [(["out-0", "out-2"], {1: _BOOM})])
        self.assertEqual(align_worker_results(*calls[0]), ["out-0", None, "out-2"])
        self.assertEqual(agg_inputs, ["built"])

    def test_unmet_quorum_raises_before_the_aggregator_runs(self):
        agg_inputs = []
        bta = _bta(
            self.tmp,
            fail_index=1,
            min_successful_workers=3,
            aggregator_inferencer=_recording_leaf(agg_inputs),
        )

        with self.assertRaisesRegex(RuntimeError, "quorum unmet"):
            bta.infer("orig")
        self.assertEqual(agg_inputs, [])


# ---------------------------------------------------------------------------
# T1.4: resume from checkpoints written before outcome records
# ---------------------------------------------------------------------------


class LegacyCheckpointResumeTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    """Worker node checkpoints holding bare values (an older build) are paired
    by the node's own index even though loaded workers reach the aggregator
    before the re-run one, and a checkpointed failure re-runs.

    The breakdown / graph-expansion / aggregator checkpoints are removed before
    resuming: a loaded start node short-circuits its whole subtree, so the
    worker node checkpoints are only read when the fan-out is rebuilt (from the
    committed plan since P8). The stages are closures without a resume identity,
    so the resume trusts the checkpoints explicitly (``trust_legacy``).
    """

    def _run(self, delays, fail_index, calls, agg_inputs, breakdown_calls):
        def factory(sub_query, index):
            def _respond(_inference_input):
                calls.append(index)
                if index == fail_index:
                    raise ValueError("boom")
                return f"out-{index}"

            return DelayedLeaf(response=_respond, delay=delays[index], max_retry=0)

        def _breakdown(_inference_input):
            breakdown_calls.append(1)
            return _BREAKDOWN

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=DelayedLeaf(response=_breakdown),
            worker_inferencers=factory,
            aggregator_inferencer=_recording_leaf(agg_inputs),
            workspace=InferencerWorkspace(root=self.tmp),
            enable_result_save=True,
            resume_with_saved_results=True,
            min_successful_workers=2,
            inject_upstream_artifacts_to_aggregator=False,
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
            resume_identity_policy="trust_legacy",
        )
        return asyncio.wait_for(bta.ainfer("orig"), timeout=60)

    def _worker_checkpoint(self, index):
        name = f"worker_0{index}"
        path = os.path.join(
            self.tmp, "children", name, "checkpoints", f"{name}_result", "main.pkl"
        )
        self.assertTrue(os.path.isfile(path), path)
        return path

    def _rewrite_as_older_build(self, worker_zero_value=None):
        for index in (1, 2):
            with open(self._worker_checkpoint(index), "wb") as f:
                pickle.dump(f"legacy-{index}", f)
        if worker_zero_value is not None:
            with open(self._worker_checkpoint(0), "wb") as f:
                pickle.dump(worker_zero_value, f)
        ws = InferencerWorkspace(root=self.tmp)
        promoted = ws.checkpoint_path(
            os.path.join("breakdown", "decomposed_subtasks.json")
        )
        os.makedirs(os.path.dirname(promoted), exist_ok=True)
        with open(promoted, "w", encoding="utf-8") as f:
            json.dump({"subtasks": [{"description": d} for d in ("q0", "q1", "q2")]}, f)
        for rel in (
            "breakdown/breakdown_result",
            "breakdown/__graph_expansion__breakdown_result",
            "aggregator_result",
        ):
            shutil.rmtree(os.path.join(self.tmp, "checkpoints", rel))

    async def _crash_then_resume(self, worker_zero_value=None):
        await self._run((0.0, 0.05, 0.1), 0, [], [], [])
        self._rewrite_as_older_build(worker_zero_value)
        calls, agg_inputs, breakdown_calls = [], [], []
        result = await self._run(
            (0.1, 0.0, 0.0), None, calls, agg_inputs, breakdown_calls
        )
        return result, calls, agg_inputs, breakdown_calls

    async def test_bare_checkpoints_upcast_by_node_index(self):
        result, calls, agg_inputs, breakdown_calls = await self._crash_then_resume()

        self.assertEqual(result, "AGG")
        self.assertEqual(calls, [0])
        self.assertEqual(breakdown_calls, [])
        self.assertEqual(agg_inputs, [_sections(["out-0", "legacy-1", "legacy-2"])])

    async def test_legacy_failure_sentinel_checkpoint_reruns(self):
        sentinel = _FailedWorkerSentinel("worker_00", "ValueError", "boom")

        _, calls, agg_inputs, _ = await self._crash_then_resume(sentinel)

        self.assertEqual(calls, [0])
        self.assertEqual(agg_inputs, [_sections(["out-0", "legacy-1", "legacy-2"])])


# ---------------------------------------------------------------------------
# T1.5: max_concurrency composes with an aggregator
# ---------------------------------------------------------------------------


class ConcurrencyLimitTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_index_order_under_each_concurrency_limit(self):
        expected_completions = {1: [0, 2], 2: [2, 0], 3: [2, 0]}
        for limit, expected in expected_completions.items():
            with self.subTest(max_concurrency=limit):
                completions, agg_inputs = [], []
                bta = _bta(
                    tempfile.mkdtemp(dir=self.tmp),
                    delays=(0.4, 0.0, 0.1),
                    fail_index=1,
                    completions=completions,
                    min_successful_workers=2,
                    max_concurrency=limit,
                    aggregator_inferencer=_recording_leaf(agg_inputs),
                    inject_upstream_artifacts_to_aggregator=False,
                )

                result = await asyncio.wait_for(bta.ainfer("orig"), timeout=30)

                self.assertEqual(result, "AGG")
                self.assertEqual(completions, expected)
                self.assertEqual(
                    agg_inputs, [_sections(["out-0", None, "out-2"], {1: _BOOM})]
                )


# ---------------------------------------------------------------------------
# T1.6: the outcome record never escapes the BTA
# ---------------------------------------------------------------------------


class OutcomeMarkerContainmentTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_stream_payloads_and_result_carry_no_marker(self):
        reporter = StreamRecorder()
        bta = _bta(
            self.tmp,
            delays=(0.05, 0.0, 0.02),
            graph_reporter=reporter,
            aggregator_inferencer=_recording_leaf([]),
        )

        result = await bta.ainfer("orig")

        self.assertEqual(result, "AGG")
        worker_streams = {
            node: content for node, content in reporter.streams if "worker" in node
        }
        self.assertEqual(sorted(worker_streams.values()), ["out-0", "out-1", "out-2"])
        for node, content in reporter.streams:
            self.assertNotIn(_MARKER, str(content), node)

    async def test_no_aggregator_returns_values_in_declaration_order(self):
        result = await _bta(self.tmp, delays=(0.05, 0.02, 0.0)).ainfer("orig")

        self.assertEqual(result, ("out-0", "out-1", "out-2"))

    async def test_no_aggregator_drops_a_failed_worker_under_quorum(self):
        bta = _bta(
            self.tmp, delays=(0.05, 0.0, 0.02), fail_index=1, min_successful_workers=2
        )

        result = await bta.ainfer("orig")

        self.assertEqual(result, ("out-0", "out-2"))
        self.assertNotIn(_MARKER, repr(result))

    async def test_no_aggregator_unmet_quorum_raises(self):
        bta = _bta(self.tmp, fail_index=1, min_successful_workers=3)

        with self.assertRaisesRegex(RuntimeError, "quorum unmet"):
            await bta.ainfer("orig")


# ---------------------------------------------------------------------------
# T1.7 / T1.8: the aggregator feed is ctx-scoped and composes with ancestors
# ---------------------------------------------------------------------------


class CtxScopedAggregatorFeedTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_bta_feed_wins_over_ancestor_override_and_keeps_its_keys(self):
        """A BTA running as an MFI followup leaf sits below MFI's ctx-published
        followup feed; its own ``upstream_artifacts`` must win at the aggregator
        while ancestor keys the BTA does not own still reach it."""
        root = RunContext.root()
        root.handles.set(
            TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE,
            {"upstream_artifacts": "peer bundle", "ancestor_only": "kept"},
        )
        aggregator = FeedAggregator(
            template_manager=_render_feed, template_root_space="aggregation"
        )
        bta = _bta(
            self.tmp,
            aggregator_inferencer=aggregator,
            inject_upstream_artifacts_to_aggregator=True,
        )

        result = await bta.ainfer("orig", run_context=root.child("followup"))

        self.assertEqual(result, "AGG")
        upstream = _sections(["out-0", "out-1", "out-2"])
        self.assertEqual(
            aggregator.prompts, [f"INPUT=orig|UPSTREAM={upstream}|ANCESTOR=kept"]
        )
        self.assertNotIn("upstream_artifacts", aggregator.template_extra_feed)


class InstanceAggregatorFeedTest(unittest.TestCase):
    def test_without_ctx_the_feed_is_written_to_the_aggregator_instance(self):
        aggregator = FeedAggregator(
            template_manager=_render_feed,
            template_root_space="aggregation",
            template_extra_feed={"aggregation_guidance": "stale", "mode": "deep"},
        )
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=DelayedLeaf(response=_BREAKDOWN),
            aggregator_inferencer=aggregator,
        )
        builder = make_upstream_injecting_aggregator_prompt_builder()

        with open_invocation(bta):
            bta._open_attempt("orig", use_async=False)
            agg_input = builder(
                ["out-0", "out-2"],
                original_query="orig",
                bta=bta,
                worker_failures={1: _BOOM},
            )

        self.assertEqual(agg_input, "orig")
        self.assertEqual(
            aggregator.template_extra_feed,
            {
                "mode": "deep",
                "upstream_artifacts": _sections(["out-0", None, "out-2"], {1: _BOOM}),
            },
        )


# ---------------------------------------------------------------------------
# T1.9: worker deliverables and output paths are paired by index
# ---------------------------------------------------------------------------


class WorkerDeliverablePairingTest(_TmpDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_no_aggregator_surfaces_each_worker_under_its_own_name(self):
        bta = _bta(
            self.tmp, delays=(0.05, 0.0, 0.02), fail_index=1, min_successful_workers=2
        )

        await bta.ainfer("orig")

        workers_dir = os.path.join(self.tmp, "outputs", "workers")
        self.assertEqual(sorted(os.listdir(workers_dir)), ["worker_00", "worker_02"])
        for idx in (0, 2):
            surfaced = os.path.join(workers_dir, f"worker_0{idx}")
            files = os.listdir(surfaced)
            self.assertEqual(len(files), 1, files)
            with open(os.path.join(surfaced, files[0]), encoding="utf-8") as f:
                self.assertEqual(f.read(), f"out-{idx}")

    async def test_local_aggregator_references_each_worker_file_at_its_index(self):
        agg_inputs = []
        aggregator = _recording_leaf(agg_inputs)
        aggregator.has_local_access = True
        bta = _bta(
            self.tmp,
            delays=(0.05, 0.0, 0.02),
            fail_index=1,
            min_successful_workers=2,
            aggregator_inferencer=aggregator,
            inject_upstream_artifacts_to_aggregator=False,
        )

        await bta.ainfer("orig")

        sections = agg_inputs[0].split("\n\n")
        self.assertEqual(sections[1], f"### Upstream Outcome 2\n(failed: {_BOOM})")
        for idx in (0, 2):
            heading, ref = sections[idx].split("\n")
            self.assertEqual(heading, f"### Upstream Outcome {idx + 1}")
            path = ref.removeprefix("(See file: `").removesuffix("`)")
            self.assertIn(os.path.join("children", f"worker_0{idx}"), path)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), f"out-{idx}")

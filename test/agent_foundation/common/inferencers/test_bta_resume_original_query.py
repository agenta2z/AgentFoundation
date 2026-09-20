"""BTA resume regression — the aggregator's Original User Request survives resume.

Guards a resume-only fidelity bug: on resume, WorkGraph rebuilds the
worker+aggregator subgraph via the ``subgraph_registry`` factories (inside
``_reconstruct_graph_expansions``) BEFORE the breakdown fn runs. The fresh path
threads the original request into the aggregator inside ``_make_breakdown_fn``
(``_original_query=_inf_input``), but on resume that threading is too late —
the aggregator node has already been rebuilt by the factory. So the factories
themselves must forward the per-call-cached original request; otherwise the
aggregator's ``## Original User Request`` slot (and the synthetic-fallback
``Original task``) render blank on resume.

These are fast unit tests — no real ``claude`` subprocess. They mirror the
inline-MockInferencer idiom from test_breakdown_then_aggregate.py /
test_bta_orchestration.py.
"""

import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrib, attrs


@attrs
class _MockInferencer(InferencerBase):
    """Minimal concrete InferencerBase returning a fixed response.

    Inlined (rather than importing the shared ``_helpers`` copy) so this target
    depends only on ``attrs`` + ``agent_foundation`` — matching the
    self-contained pattern already used by test_breakdown_then_aggregate.py.
    """

    _response: str = attrib(default="mock response")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._response


# Breakdown response that yields exactly 2 workers via the json_subtasks parser
# (the same shape test_bta_orchestration.py exercises).
_BREAKDOWN_JSON = (
    "```json\n"
    + json.dumps(
        {
            "subtasks": [
                {"description": "First subtask"},
                {"description": "Second subtask"},
            ]
        }
    )
    + "\n```"
)

_ORIGINAL_REQUEST = "suggest small improvements to the CLI script"


def _make_bta(checkpoint_dir: str) -> BreakdownThenAggregateInferencer:
    """Construct a workers-only BTA wired with mock inferencers (no real CLI).

    No ``aggregator_inferencer`` — matching the wired test_bta_resume.py idiom.
    Resolving the aggregator's *published* child workspace in ``_finalize_output``
    needs a full graph workspace (``target_path``), which is out of scope for a
    mock unit test. ``_cached_original_query`` is populated aggregator-independently
    at the very top of ``_infer``/``_ainfer`` (before any aggregator logic), so a
    workers-only topology fully exercises the per-call cache assignment these
    behavioral tests assert. The resume-path factory threading itself is covered
    directly (and aggregator-agnostically) by
    ``test_resume_factories_thread_original_query``.
    """
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_MockInferencer(response=_BREAKDOWN_JSON),
        worker_inferencers=lambda sub_query, index: _MockInferencer(
            response=f"worker {index}"
        ),
        breakdown_format="json_subtasks",
        checkpoint_dir=checkpoint_dir,
        max_breakdown=2,
    )


class BtaResumeOriginalQueryTest(unittest.TestCase):
    """Sync-path coverage for the resume ``_original_query`` threading."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_post_init_defaults_original_query_to_none(self):
        """Fresh construction leaves the cache empty (populated per-call by infer)."""
        bta = _make_bta(self.tmpdir)
        self.assertIsNone(bta._cached_original_query)

    def test_resume_factories_thread_original_query(self):
        """Both subgraph_registry factories forward the cached original request to
        ``_build_subgraph_spec`` — the exact contract that was missing.

        The factory sources sub_queries from the promoted breakdown checkpoint
        (``_load_promoted_breakdown()[0]``) and threads ``_original_query`` from
        the per-call cache. Regression: before the fix it passed no
        ``_original_query``, so ``kwargs.get("_original_query", "")`` was empty and
        the rebuilt aggregator rendered a blank ``## Original User Request``.
        """
        bta = _make_bta(self.tmpdir)
        # Simulate the resume state _infer/_ainfer + _load_promoted_breakdown
        # establish before WorkGraph rebuilds the subgraph: the promoted breakdown
        # checkpoint is loaded (memoized) and the original request cached.
        bta._promoted_breakdown_cache = (["q1", "q2"], None)
        bta._cached_original_query = _ORIGINAL_REQUEST

        for factory_key in ("bta_diamond", "bta_workers"):
            with self.subTest(factory=factory_key):
                self.assertIn(factory_key, bta.subgraph_registry)
                with patch.object(
                    BreakdownThenAggregateInferencer, "_build_subgraph_spec"
                ) as mock_spec:
                    mock_spec.return_value = "SENTINEL_SPEC"
                    result = bta.subgraph_registry[factory_key]("expansion-id")
                self.assertEqual(result, "SENTINEL_SPEC")
                mock_spec.assert_called_once_with(
                    ["q1", "q2"], _original_query=_ORIGINAL_REQUEST
                )

    def test_infer_populates_cached_original_query(self):
        """A real (mock-backed) infer caches the request so the resume factories
        have it to thread — proves the per-call assignment in ``_infer`` runs."""
        bta = _make_bta(self.tmpdir)
        bta.infer(_ORIGINAL_REQUEST)
        self.assertEqual(bta._cached_original_query, _ORIGINAL_REQUEST)


class BtaResumeOriginalQueryAsyncTest(unittest.IsolatedAsyncioTestCase):
    """Async path — research_propose resumes via ``ainfer``, so guard it too."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_ainfer_populates_cached_original_query(self):
        bta = _make_bta(self.tmpdir)
        await bta.ainfer(_ORIGINAL_REQUEST)
        self.assertEqual(bta._cached_original_query, _ORIGINAL_REQUEST)


if __name__ == "__main__":
    unittest.main()

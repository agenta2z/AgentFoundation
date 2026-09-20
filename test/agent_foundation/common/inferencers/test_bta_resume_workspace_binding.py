"""BTA resume regression — rebuilt non-shared nodes keep a durable workspace.

Guards a resume-only recovery bug: on resume, WorkGraph rebuilds the
worker+aggregator subgraph via the ``subgraph_registry`` factories (inside
``_reconstruct_graph_expansions``), which re-invoke ``_build_subgraph_spec``.
Under an active run-context the rebuilt aggregator (and workers) previously got
ONLY an ephemeral, ctx-scoped workspace binding (published into the child
context), because the durable instance binding was gated behind ``if
_agg_child is None`` — the legacy/no-ctx branch that never fires under a real
context.

On the initial run that is harmless: the active context is stable across the
node's whole ``ainfer``, so the ctx-scoped binding resolves throughout. On
resume the active context is NOT guaranteed to still be the dispatched child
context by the time the node reaches its recovery gate (the runtime-observed
"temporal flip"). When it isn't, the aggregator's ``_workspace`` getter resolves
``None`` → ``resolve_output_path`` returns a RELATIVE path → the recovery gate's
``os.path.isabs`` check fails → the self-heal UPDATE is silently downgraded to a
full restart, throwing away the partial deliverable.

The fix binds each rebuilt NON-SHARED node (the single aggregator, per-subtask
workers) durably at its ``_build_subgraph_spec`` dispatch site via
``_bind_rebuilt_child_ws`` — publishing into the child context AND setting the
durable instance backing (tier-2), which resolves independently of whichever
context is active. These tests reproduce the "flip" deterministically by
building the spec under a context and then asserting the binding still resolves
an ABSOLUTE output path after the context is exited.

The aggregator is deliberately excluded from construction-time propagation
(``_workspace_propagation_skip``), and factory-built workers are not reached by
it either, so ``_build_subgraph_spec`` is the ONLY place these nodes are bound —
which is exactly why the dispatch-site binding is load-bearing.

These are fast unit tests — no real ``claude`` subprocess and no graph
execution; they only build the subgraph spec (which runs the binding sites) and
inspect the resulting instance bindings.
"""

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
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)
from attr import attrib, attrs


@attrs
class _MockInferencer(InferencerBase):
    """Minimal concrete InferencerBase returning a fixed response.

    Inlined (rather than importing a shared helper) so this target depends only
    on ``attrs`` + ``agent_foundation``, matching the self-contained pattern in
    the sibling test modules. Being a REAL ``InferencerBase`` subclass matters:
    ``_build_subgraph_spec`` treats a bare callable as a factory to invoke and
    only binds children that pass ``isinstance(..., InferencerBase)``.
    """

    _response: str = attrib(default="mock response")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._response


# Breakdown response shape is irrelevant here (we pass sub_queries to
# _build_subgraph_spec directly), but keep a valid one for construction parity
# with the sibling resume tests.
_BREAKDOWN_JSON = (
    "```json\n"
    + json.dumps({"subtasks": [{"description": "a"}, {"description": "b"}]})
    + "\n```"
)


class BtaResumeWorkspaceBindingTest(unittest.TestCase):
    """Rebuilt aggregator/workers keep a resume-robust (tier-2) workspace."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bta_ws_bind_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_bta_with_agg(self):
        """A BTA wired with an aggregator + a worker factory that records the
        exact worker instances it builds (so the tests can inspect their
        bindings without reaching into WorkGraph node internals)."""
        root_ws = InferencerWorkspace(root=self.tmpdir)
        workers: list[_MockInferencer] = []

        def worker_factory(sub_query, index):
            worker = _MockInferencer(response=f"worker {index}")
            workers.append(worker)
            return worker

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=_MockInferencer(response=_BREAKDOWN_JSON),
            worker_inferencers=worker_factory,
            aggregator_inferencer=_MockInferencer(response="<Response>agg</Response>"),
            breakdown_format="json_subtasks",
            workspace=root_ws,
            output_path="output.md",
            max_breakdown=2,
        )
        return bta, workers, root_ws

    def _build_spec_under_ctx(self, bta, root_ws):
        """Run ``_build_subgraph_spec`` (the resume-rebuild entry) under an
        active run-context, then exit it — reproducing the resume "flip" where
        the dispatched child context is no longer active at the recovery gate."""
        root = RunContext.root(workspace=root_ws)
        tok = enter_run(root)
        try:
            bta._build_subgraph_spec(["q1", "q2"], _original_query="req")
        finally:
            exit_run(tok)

    def test_build_subgraph_spec_durably_binds_aggregator_under_ctx(self):
        """After a ctx-scoped rebuild, the aggregator STILL resolves an absolute
        output path with no active context — the exact condition the recovery
        gate's ``os.path.isabs`` check needs to honor UPDATE over restart.

        Pre-fix this fails: under a context the durable binding was skipped, so
        once the context is exited the getter resolves ``None`` and
        ``resolve_output_path`` returns the raw relative ``output.md``.
        """
        bta, _workers, root_ws = self._make_bta_with_agg()
        self._build_spec_under_ctx(bta, root_ws)

        agg = bta.aggregator_inferencer
        resolved = agg.resolve_output_path("output.md")
        self.assertTrue(
            os.path.isabs(resolved),
            f"aggregator path must be absolute after ctx exit; got {resolved!r}",
        )
        self.assertIsNotNone(agg._workspace)
        self.assertEqual(agg._workspace.root, root_ws.child("aggregator").root)

    def test_build_subgraph_spec_durably_binds_workers_under_ctx(self):
        """Every rebuilt per-subtask worker is durably bound too (site (d))."""
        bta, workers, root_ws = self._make_bta_with_agg()
        self._build_spec_under_ctx(bta, root_ws)

        self.assertEqual(len(workers), 2, "two sub_queries must build two workers")
        for idx, worker in enumerate(workers):
            resolved = worker.resolve_output_path("output.md")
            self.assertTrue(
                os.path.isabs(resolved),
                f"worker {idx} path must be absolute after ctx exit; got {resolved!r}",
            )
            self.assertEqual(
                worker._workspace.root,
                root_ws.child(bta._worker_child_name(idx)).root,
            )

    def test_build_subgraph_spec_binds_aggregator_async_config(self):
        """The async-dispatch configuration binds the aggregator identically —
        research_propose resumes via ``ainfer`` (``use_async`` path)."""
        bta, _workers, root_ws = self._make_bta_with_agg()
        bta.use_async = True
        self._build_spec_under_ctx(bta, root_ws)

        resolved = bta.aggregator_inferencer.resolve_output_path("output.md")
        self.assertTrue(os.path.isabs(resolved), f"got {resolved!r}")

    def test_bind_rebuilt_child_ws_survives_ctx_exit(self):
        """Helper contract: a binding made under a context survives the context
        being exited, via the durable tier-2 instance backing."""
        bta, _workers, root_ws = self._make_bta_with_agg()
        agg = _MockInferencer()
        agg_ws = bta._workspace.child("aggregator")

        root = RunContext.root(workspace=root_ws)
        tok = enter_run(root)
        try:
            bta._bind_rebuilt_child_ws(agg, "aggregator", agg_ws)
            self.assertTrue(os.path.isabs(agg.resolve_output_path("output.md")))
        finally:
            exit_run(tok)

        self.assertIs(getattr(agg, "_InferencerBase__workspace", None), agg_ws)
        resolved = agg.resolve_output_path("output.md")
        self.assertTrue(
            os.path.isabs(resolved),
            f"durable binding must resolve absolute with no ctx; got {resolved!r}",
        )

    def test_bind_rebuilt_child_ws_no_ctx_binds(self):
        """Backward-compat: with no active context the helper binds durably,
        identical to the prior legacy (no-ctx) branch."""
        bta, _workers, root_ws = self._make_bta_with_agg()
        agg = _MockInferencer()
        agg_ws = bta._workspace.child("aggregator")

        bta._bind_rebuilt_child_ws(agg, "aggregator", agg_ws)

        self.assertIs(getattr(agg, "_InferencerBase__workspace", None), agg_ws)
        self.assertTrue(os.path.isabs(agg.resolve_output_path("output.md")))

    def test_bind_rebuilt_child_ws_noop_for_non_inferencer(self):
        """A non-InferencerBase child (e.g. a factory / duck-typed callable) is
        NOT durably mutated — preserving the invariant that factories receive
        workspaces at runtime rather than by instance pinning. The M7
        shared-instance write-purity guard and ``_propagate_workspace_to_children``
        are untouched by this change.
        """
        bta, _workers, root_ws = self._make_bta_with_agg()
        agg_ws = bta._workspace.child("aggregator")

        class _Plain:
            pass

        plain = _Plain()
        root = RunContext.root(workspace=root_ws)
        tok = enter_run(root)
        try:
            bta._bind_rebuilt_child_ws(plain, "aggregator", agg_ws)
        finally:
            exit_run(tok)

        self.assertFalse(
            hasattr(plain, "_workspace"),
            "helper must not durably mutate a non-InferencerBase child",
        )


if __name__ == "__main__":
    unittest.main()

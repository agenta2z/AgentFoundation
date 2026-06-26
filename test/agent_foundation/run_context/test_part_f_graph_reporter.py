"""Part F / GT#13 — graph node identity = ctx path (reporter routed through Tier-2).

The graph-visualization reporter is a Tier-2 shared sink (``ctx.runtime.graph_reporter``).
A SINGLE shared inferencer instance reused across N concurrent ``RunContext``s must
emit N **distinct** viz nodes — one per ``ctx.path`` — instead of last-write-wins onto
one node (the GT#13 instance-coupling bug). This module proves:

  * ``RuntimeBindings.child_reporter(ctx.path)`` fans a shared sink out per path so the
    SAME flat node id resolves to DISTINCT qualified node ids under sibling ctxs;
  * a real BTA's ``_resolve_graph_reporter`` (and the per-node wiring it drives) keys on
    ``ctx.path`` — two concurrent ctxs sharing one runtime + one BTA instance produce
    distinct nodes (``parallel_0/worker_0`` vs ``parallel_1/worker_0``), no cross-talk;
  * with NO active context the resolver returns the **instance** reporter unchanged
    (byte-identical legacy behavior).
"""

from __future__ import annotations

from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (  # noqa: E501
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    RuntimeBindings,
    active_run_context,
    enter_run,
    exit_run,
    mint_root,
)


# ---------------------------------------------------------------------------
# A recording reporter implementing the graph_reporter protocol surface BTA uses
# (4 async sinks + 3 factories). It records the *fully-qualified* node id each
# event lands on, so namespacing by ctx.path is observable.
# ---------------------------------------------------------------------------
class RecordingReporter:
    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix
        # Shared event log across all child_reporters derived from this root, so a
        # single test can read every node id emitted anywhere in the path tree.
        self.events: list[tuple[str, str]] = []

    def _q(self, node_id: str) -> str:
        return f"{self._prefix}/{node_id}" if self._prefix else node_id

    # -- 4 async sinks --
    async def on_graph_topology(self, event: Any) -> None:
        for n in getattr(event, "nodes", []) or []:
            self.events.append(("topology", n["id"]))

    async def on_node_status(self, node_id: str, status: str,
                             error: str = "", output_path: str = "") -> None:
        self.events.append(("status", self._q(node_id)))

    async def on_node_stream(self, node_id: str, content: str,
                             is_final: bool = True) -> None:
        self.events.append(("stream", self._q(node_id)))

    async def on_graph_reconcile(self, node_statuses: dict) -> None:
        for nid in node_statuses:
            self.events.append(("reconcile", self._q(nid)))

    # -- 3 factories -- (each scopes node ids under the qualified prefix)
    def child_reporter(self, parent_node_id: str) -> "RecordingReporter":
        child = RecordingReporter(prefix=self._q(parent_node_id))
        child.events = self.events  # share the log
        return child

    def node_stream_observer(self, node_id: str, flush_interval_ms: float = 200.0):
        qid = self._q(node_id)
        log = self.events

        async def _observer(chunk: str) -> None:
            log.append(("observer", qid))

        return _observer

    def node_interactive(self, node_id: str) -> Any:
        self.events.append(("interactive_built", self._q(node_id)))
        return object()


def _make_bta(reporter=None) -> BreakdownThenAggregateInferencer:
    bta = BreakdownThenAggregateInferencer(disable_aggregator=True)
    if reporter is not None:
        bta.graph_reporter = reporter
    return bta


# ---------------------------------------------------------------------------
# 1. RuntimeBindings.child_reporter: the Tier-2 fan-out keyed by ctx.path
# ---------------------------------------------------------------------------
def test_runtime_child_reporter_keys_nodes_by_ctx_path():
    """One shared runtime (one reporter); two sibling ctxs. The SAME flat node id
    resolves to DISTINCT qualified node ids — node identity is the ctx path."""
    import asyncio

    reporter = RecordingReporter()
    runtime = RuntimeBindings(graph_reporter=reporter)
    root = RunContext.root(runtime=runtime)
    c0, c1 = root.child("parallel_0"), root.child("parallel_1")

    r0 = runtime.child_reporter(c0.path, base_path="/")
    r1 = runtime.child_reporter(c1.path, base_path="/")

    async def _emit():
        await r0.on_node_status("worker_0", "completed")
        await r1.on_node_status("worker_0", "completed")

    asyncio.run(_emit())

    landed = [nid for (_kind, nid) in reporter.events if _kind == "status"]
    assert landed == ["parallel_0/worker_0", "parallel_1/worker_0"], landed
    # The two concurrent branches did NOT collapse onto a single node.
    assert len(set(landed)) == 2


def test_runtime_child_reporter_root_is_flat_and_no_reporter_is_none():
    """At the run root (path '/') node ids stay FLAT (byte-identical); with no sink
    the fan-out is a None no-op."""
    reporter = RecordingReporter()
    runtime = RuntimeBindings(graph_reporter=reporter)
    # root path '/' relative to base '/' -> empty relative -> raw reporter (flat ids)
    assert runtime.child_reporter("/", base_path="/") is reporter
    # No reporter wired -> None (no-op).
    empty = RuntimeBindings()
    assert empty.child_reporter("/x/y", base_path="/") is None


# ---------------------------------------------------------------------------
# 2. BTA._resolve_graph_reporter keys on ctx.path (shared instance, shared runtime)
# ---------------------------------------------------------------------------
def test_bta_resolver_distinct_nodes_for_concurrent_ctxs_on_shared_instance():
    """ONE shared BTA instance + ONE shared runtime; two sibling ctxs. The
    orchestrator's resolved reporter is namespaced by ctx.path, so emitting the
    SAME flat node id under each ctx lands on DISTINCT viz nodes."""
    import asyncio

    reporter = RecordingReporter()
    runtime = RuntimeBindings()  # graph_reporter starts unset — seeded from instance
    bta = _make_bta(reporter)  # shared instance carries the reporter as definition

    root = RunContext.root(runtime=runtime)
    c0, c1 = root.child("parallel_0"), root.child("parallel_1")

    async def _emit_under(ctx):
        tok = enter_run(ctx)
        try:
            rep = bta._resolve_graph_reporter()
            # Seed-once lifted the instance reporter into the SHARED Tier-2 sink.
            assert runtime.graph_reporter is reporter
            # Force a real suspension between resolve and emit so the two branches
            # genuinely interleave on the shared instance — proving the resolution
            # is per-ctx (ContextVar-isolated), not per-instance last-write-wins.
            import asyncio as _a
            await _a.sleep(0)
            await rep.on_node_status("worker_0", "completed")
        finally:
            exit_run(tok)

    async def _both():
        # gather() proves no reliance on per-instance mutable reporter state.
        await asyncio.gather(_emit_under(c0), _emit_under(c1))

    asyncio.run(_both())

    landed = sorted(nid for (_kind, nid) in reporter.events if _kind == "status")
    assert landed == ["parallel_0/worker_0", "parallel_1/worker_0"], landed


def test_bta_node_reporter_wiring_keys_observer_by_ctx_path():
    """The per-node stream observer the orchestrator wires for a worker is tagged by
    the worker's ctx path — so a reused instance's worker_0 observer under two ctxs
    flushes to two distinct nodes (the GT#13 last-write-wins fix)."""
    import asyncio

    reporter = RecordingReporter()
    runtime = RuntimeBindings()
    bta = _make_bta(reporter)

    root = RunContext.root(runtime=runtime)
    c0, c1 = root.child("parallel_0"), root.child("parallel_1")

    async def _observer_under(ctx):
        tok = enter_run(ctx)
        try:
            # The orchestrator's path-namespaced sink; tag the worker's flat id.
            rep = bta._resolve_graph_reporter()
            obs = rep.node_stream_observer("worker_0")
            await obs("chunk")
        finally:
            exit_run(tok)

    async def _both():
        await asyncio.gather(_observer_under(c0), _observer_under(c1))

    asyncio.run(_both())
    flushed = sorted(nid for (_kind, nid) in reporter.events if _kind == "observer")
    assert flushed == ["parallel_0/worker_0", "parallel_1/worker_0"], flushed


# ---------------------------------------------------------------------------
# 3. No-ctx legacy path: resolver returns the instance reporter unchanged
# ---------------------------------------------------------------------------
def test_no_ctx_resolver_returns_instance_reporter_byte_identical():
    """With NO active context the resolver returns the instance attrib exactly —
    flat node ids, no Tier-2 indirection (byte-identical legacy behavior)."""
    import asyncio

    reporter = RecordingReporter()
    bta = _make_bta(reporter)
    assert active_run_context() is None  # precondition: no ctx
    assert bta._resolve_graph_reporter() is reporter

    async def _emit():
        rep = bta._resolve_graph_reporter()
        await rep.on_node_status("worker_0", "completed")

    asyncio.run(_emit())
    landed = [nid for (_kind, nid) in reporter.events if _kind == "status"]
    assert landed == ["worker_0"], landed  # flat — no path prefix


def test_no_ctx_resolver_none_when_unset():
    """No reporter + no ctx -> None (no-op), the byte-identical disabled path."""
    bta = _make_bta(reporter=None)
    assert active_run_context() is None
    assert bta._resolve_graph_reporter() is None


# ---------------------------------------------------------------------------
# 4. Legacy-mint (bare ainfer, no caller ctx): a fresh root still flat at root
# ---------------------------------------------------------------------------
def test_legacy_mint_root_is_flat():
    """A bare call legacy-mints a root (path '/'); the resolver seeds Tier-2 and
    returns the raw reporter -> flat node ids (byte-identical to pre-Part-F)."""
    import asyncio

    reporter = RecordingReporter()
    bta = _make_bta(reporter)
    root = mint_root()  # legacy_mint=True, path "/"
    assert root.legacy_mint is True

    async def _emit():
        tok = enter_run(root)
        try:
            rep = bta._resolve_graph_reporter()
            assert rep is reporter  # path '/' -> raw reporter
            await rep.on_node_status("aggregator", "completed")
        finally:
            exit_run(tok)

    asyncio.run(_emit())
    landed = [nid for (_kind, nid) in reporter.events if _kind == "status"]
    assert landed == ["aggregator"], landed

"""Tier-2: shared transient bindings (concurrency-safe sinks).

Per the plan (§2.1, J8): shared **by reference** down the context tree and
**never persisted**.  All members are concurrency-safe sinks **except**
``interactive`` — the one guarded exception (parallel HITL prompts share one
transport and must be serialized; see §2.11).  ``cancellation_token`` is
read-only and shared by reference, so a cancel set anywhere is visible to every
in-flight branch at its next check (§2.0 Note / P-#6).

This object must never appear in ``RunStateStore.to_json()`` output — the
"god-object guard" test asserts that.
"""

from __future__ import annotations

from typing import Any

import attrs


@attrs.define
class RuntimeBindings:
    """Tier-2 shared sinks (NOT serialized).

    Marked with ``__never_persist__`` so the store's god-object guard can detect
    an accidental attempt to serialize it.
    """

    __never_persist__ = True

    graph_reporter: Any = None  # fan-out via child_reporter (concurrency-safe)
    interactive: Any = None  # ⚠️ NOT concurrency-safe under parallel HITL (§2.11/J8)
    checkpoint_store: Any = None  # optional live progress sink (§8.4 — not load-bearing)
    cancellation_token: Any = None  # read-only, shared by reference (P-#6)

    # -- Part F: graph node identity = ctx path -----------------------------
    @staticmethod
    def reporter_node_id(ctx_path: str, base_path: str = "/") -> str:
        """Project a ctx ``path`` onto a graph-viz **node id** (Part F / GT#13).

        Graph node identity is the running unit's ``ctx.path`` — *not* its Python
        object identity — so a single shared inferencer reused across N concurrent
        ``RunContext``s emits N distinct nodes (one per path), never last-write-wins
        onto one node.  The node id is the path **relative to** ``base_path`` (the
        path of the orchestrator whose ``graph_reporter`` is the sink), with the
        leading slash dropped and remaining separators preserved so the reporter's
        own ``child_reporter`` prefix-composition stays nesting-safe.

        Examples (``base_path='/bta'``):
            ``/bta/worker_0``            -> ``worker_0``
            ``/bta/worker_0/sub/leaf``   -> ``worker_0/sub/leaf``
            ``/bta``                     -> ``""`` (the orchestrator's own node)
        """
        p = ctx_path or "/"
        base = (base_path or "/").rstrip("/")
        if base and (p == base or p.startswith(base + "/")):
            rel = p[len(base):]
        else:
            rel = p
        return rel.strip("/")

    def child_reporter(self, ctx_path: str, base_path: str = "/") -> Any:
        """The per-node sub-reporter for ``ctx_path``, derived from the shared sink.

        This is the Tier-2 fan-out the plan reserves (``"fan-out via
        child_reporter (concurrency-safe)"``): node identity is the **ctx path**,
        resolved at render time against the *shared* ``graph_reporter`` — so a
        reused inferencer instance reports to a distinct viz node per concurrent
        run, with no per-instance mutable reporter state.

        Returns ``None`` when no reporter is wired (byte-identical no-op), the
        shared reporter itself for the orchestrator's own node (empty relative
        path), else ``graph_reporter.child_reporter(node_id)`` namespaced by the
        path-derived node id.
        """
        reporter = self.graph_reporter
        if reporter is None:
            return None
        node_id = self.reporter_node_id(ctx_path, base_path)
        if not node_id:
            return reporter
        child = getattr(reporter, "child_reporter", None)
        if callable(child):
            return child(node_id)
        return reporter

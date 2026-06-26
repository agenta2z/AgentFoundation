"""The app-minted ``RunContext`` — the carrier threaded through inference.

Per the plan §2.0 (consolidated Core API), corrected by N-R4 / T-MAJOR1 / T-MAJOR2:

* ``path`` is the **logical** state-store key ("/", "/review/worker_0") — slash-joined
  single-component slots.
* ``workspace`` is the **real** workspace, derived **per single-component slot** via
  the existing ``InferencerWorkspace.child`` (which **rejects** embedded "/" and nests
  under ``children/``).  The logical path and the workspace are *correlated slot-trees*,
  **not string-identical** (§2.0 Note A).
* ``runtime`` (Tier-2) is shared **by reference**; ``_store`` (Tier-1) is the per-turn
  store, shared by reference; ``_handle_store`` (Tier-3) is the **connection-scoped**
  handle store, shared by reference (§2.0 Note B).
* ``child(slot)`` returns a **fresh** ``RunContext`` (not a mutating view) — cheap:
  shares runtime/_store/_handle_store, ``handles`` is a get-or-create lookup, ``node``
  is lazy.  Re-deriving ``child("x")`` twice yields equivalent contexts pointing at the
  **same** store node and **same** handles (idempotent).
"""

from __future__ import annotations

from typing import Any

from .bindings import RuntimeBindings
from .handles import LiveHandles, LiveHandleStore
from .store import CreatorKey, NodeRunState, RunStateStore


class RunContext:
    """Immutable-identity per-run context.  Descendants are produced only via ``child``."""

    __slots__ = ("path", "workspace", "runtime", "_store", "_handle_store", "legacy_mint")

    def __init__(
        self,
        path: str,
        workspace: Any,
        runtime: RuntimeBindings,
        store: RunStateStore,
        handle_store: LiveHandleStore,
        legacy_mint: bool = False,
    ) -> None:
        self.path = path
        self.workspace = workspace
        self.runtime = runtime
        self._store = store
        self._handle_store = handle_store
        # True only for a context that ``bridge.mint_root`` legacy-minted for a bare
        # ``infer()`` (no caller ctx). Its Tier-1 store is discarded on exit, so per-call
        # state-writers ALSO mirror to the instance backing for post-call getters; under a
        # real caller ctx this stays False → backing is never polluted (concurrency-safe).
        self.legacy_mint = legacy_mint

    # -- root factory -------------------------------------------------------
    @classmethod
    def root(
        cls,
        workspace: Any = None,
        *,
        runtime: RuntimeBindings | None = None,
        store: RunStateStore | None = None,
        handle_store: LiveHandleStore | None = None,
        legacy_mint: bool = False,
    ) -> "RunContext":
        """Mint a root context (path ``"/"``).  The app host calls this (§9.4).

        ``legacy_mint`` is set ``True`` only by ``bridge.mint_root`` (a bare ``infer()``
        with no caller ctx); the app-host root keeps the default ``False``.
        """
        return cls(
            path="/",
            workspace=workspace,
            runtime=runtime if runtime is not None else RuntimeBindings(),
            store=store if store is not None else RunStateStore(),
            handle_store=handle_store if handle_store is not None else LiveHandleStore(),
            legacy_mint=legacy_mint,
        )

    # -- child derivation ---------------------------------------------------
    def child(self, slot: str, *, workspace: Any = None) -> "RunContext":
        """Derive a child context for a **single-component** ``slot`` (no "/").

        For a multi-segment label, chain calls: ``ctx.child("a").child("b")`` —
        never ``ctx.child("a/b")`` (the real workspace API rejects separators).

        ``workspace`` (default ``None`` → path-mirror ``self.workspace.child(slot)``):
        when given, the child uses it as its workspace VERBATIM. This decouples a
        child's on-disk workspace from its (possibly namespaced) ctx-node *name* —
        e.g. a BTA worker whose node id is ``plan_bta.worker_0`` but whose intended
        workspace dir is ``worker_0``. Descendants then path-mirror from this
        explicit workspace, so the namespaced node name never leaks on disk.
        """
        if not slot or "/" in slot or "\\" in slot or slot in (".", ".."):
            raise ValueError(
                f"RunContext.child slot must be a single path component, got {slot!r}. "
                f"Chain child() calls for nested paths (the workspace API rejects '/')."
            )
        child_path = self.path.rstrip("/") + "/" + slot
        if workspace is not None:
            child_workspace = workspace
        else:
            child_workspace = (
                self.workspace.child(slot) if self.workspace is not None else None
            )
        return RunContext(
            path=child_path,
            workspace=child_workspace,
            runtime=self.runtime,  # Tier-2 shared by reference
            store=self._store,  # Tier-1 shared (per-turn)
            handle_store=self._handle_store,  # Tier-3 shared (connection-scoped)
            legacy_mint=self.legacy_mint,  # propagate: descendants of a legacy root mirror too
        )

    # -- Tier-1 node access -------------------------------------------------
    def node(self, creator: CreatorKey | None = None) -> NodeRunState:
        """The Tier-1 :class:`NodeRunState` for this path (get-or-create + guard)."""
        return self._store.node(self.path, creator)

    # -- Tier-3 handles -----------------------------------------------------
    @property
    def handles(self) -> LiveHandles:
        """Connection-scoped live handles for this branch (get-or-create, idempotent)."""
        return self._handle_store.get_or_create(self.path)

    def __repr__(self) -> str:
        return f"RunContext(path={self.path!r})"

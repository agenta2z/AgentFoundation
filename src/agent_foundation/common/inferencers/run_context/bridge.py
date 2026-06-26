"""The compat bridge: the module-level ``_active_ctx`` ContextVar + mint policy.

Per the plan D6 / §2.2 / E1 / M3:

* ``_active_ctx`` is a **module-level** ``contextvars.ContextVar`` — per-task, so
  concurrent ``asyncio.gather`` branches on one instance don't clobber each other
  (a plain instance attr would).  Public entrypoints install the bridge; private
  funnels / orchestrators / retry / recovery **read** it.
* **Mint policy** (``run_context is None``): if a context is **already active**
  (a *nested* public call — e.g. a streaming primitive re-entering the public
  ``ainfer_streaming``), **reuse** it (stay attached to the parent tree); only
  **legacy-mint a fresh root** at a *true root* (no active ctx).  This keeps a
  nested public call from minting a detached root that replaces the parent.

This module is the M1 mechanism; wiring it into ``InferencerBase`` public
entrypoints (and the per-class workspace-backing sync) is the M2 step.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
from typing import Any

from .context import RunContext

# The permanent, per-task reader.  Set by the bridge at public entrypoints.
_active_ctx: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "_af_active_run_context", default=None
)


def active_run_context() -> RunContext | None:
    """Read the active context for this task (``None`` at a true root before mint)."""
    return _active_ctx.get()


def mint_root(workspace: Any = None, **kwargs: Any) -> RunContext:
    """Legacy-mint a fresh root context (the ``run_context is None`` true-root case).

    Stamps ``legacy_mint=True`` so per-call state-writers also mirror to the instance
    backing (this root's Tier-1 store is discarded on exit, so post-call getters need the
    backing). A caller-supplied or app-host root keeps ``legacy_mint=False``.
    """
    kwargs.setdefault("legacy_mint", True)
    return RunContext.root(workspace=workspace, **kwargs)


def enter_run(
    run_context: RunContext | None,
    *,
    default_workspace: Any = None,
) -> contextvars.Token:
    """Install the bridge for a public entrypoint; return the reset token.

    Resolution (the mint policy):

    * ``run_context`` provided          -> use it.
    * ``None`` and a ctx already active -> **reuse** the active ctx (nested call).
    * ``None`` and no active ctx        -> **legacy-mint** a root from
      ``default_workspace`` (true root; byte-identical when the workspace matches).
    """
    if run_context is not None:
        resolved = run_context
    else:
        active = _active_ctx.get()
        resolved = active if active is not None else mint_root(default_workspace)
    return _active_ctx.set(resolved)


def exit_run(token: contextvars.Token) -> None:
    """Restore the previous active context (the bridge's ``finally``)."""
    _active_ctx.reset(token)


def bridge_entrypoint(fn):
    """Decorator for a PUBLIC inference entrypoint that bypasses
    ``InferencerBase.ainfer``/``infer`` and calls ``_ainfer_single`` /
    ``_infer_single`` directly — the CLI / streaming leaf overrides (§9.3 E4).

    Such an override receives ``run_context`` but, without this, never installs
    the ``_active_ctx`` bridge — so the leaf renders its prompt and resolves its
    session/handles under the **caller's** active context instead of its own.
    Under a parallel orchestrator (e.g. the BTA/MFI aggregator invoked directly
    on the propose node) that makes the leaf claim the parent's node and trips
    the §2.7 creator collision guard (``CollisionError``).

    When ``run_context`` is provided (or a context is already active for a nested
    call) this installs the bridge — ``enter_run`` / ``exit_run`` — exactly like the
    base entrypoint, and forwards the remaining args unchanged.  ``run_context`` is
    captured here and is NOT passed into the wrapped body (so it never lands in
    ``**kwargs`` / the provider splat).

    **True bare call** (``run_context is None`` AND no context already active): the
    bridge is deliberately NOT installed — the leaf runs ctx-less, exactly as it did
    before this decorator existed.  This is byte-identical: a leaf's per-call state
    writers (e.g. the ``active_session_id`` compat-setter) target the instance
    backing when there is no active context, preserving across-call session
    continuity.  (Minting a legacy root here instead would redirect those writes to
    the discarded root's handle store — the ``legacy_mint`` mirror is not wired into
    those setters — silently dropping the leaf's session id.)  Works for both sync
    and async entrypoints.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def _async_entry(self, *args, run_context=None, **kwargs):
            if run_context is None and _active_ctx.get() is None:
                return await fn(self, *args, **kwargs)
            token = enter_run(
                run_context, default_workspace=getattr(self, "_workspace", None)
            )
            try:
                return await fn(self, *args, **kwargs)
            finally:
                exit_run(token)

        return _async_entry

    @functools.wraps(fn)
    def _sync_entry(self, *args, run_context=None, **kwargs):
        if run_context is None and _active_ctx.get() is None:
            return fn(self, *args, **kwargs)
        token = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            return fn(self, *args, **kwargs)
        finally:
            exit_run(token)

    return _sync_entry

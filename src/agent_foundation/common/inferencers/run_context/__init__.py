"""Explicit per-run state for inferencers — the ``RunContext`` / three-tier model.

This package implements the foundation primitives from the design plan
``swift-launching-backus.md`` (M0 + M1): an immutable, app-minted **RunContext**
threaded through inference, backed by a **three-tier state model**:

* **Tier 1 — serializable, path-keyed** :class:`RunStateStore` / :class:`NodeRunState`
  (per-turn lifetime; the durable artifact for resume).
* **Tier 2 — shared concurrency-safe sinks** :class:`RuntimeBindings`
  (shared by reference down the tree; never persisted).
* **Tier 3 — connection-scoped live handles** :class:`LiveHandleStore` /
  :class:`LiveHandles` (lifetime = the ``aconnect``/``adisconnect`` connection,
  NOT the per-turn run — so multi-turn session continuity is preserved).

Everything here is **additive**: nothing in this package mutates or imports the
existing inferencer hierarchy.  The migration that threads these primitives
through ``ainfer``/``infer`` (M2+) is a separate, incremental step.
"""

from .bindings import RuntimeBindings
from .bridge import (
    active_run_context,
    bridge_entrypoint,
    enter_run,
    exit_run,
    mint_root,
)
from .context import RunContext
from .handles import LiveHandles, LiveHandleStore
from .state import (
    STATE_REGISTRY,
    BTAState,
    DualState,
    InferencerStateBase,
    LinearWorkflowState,
    MFDualState,
    MultiFlowAttemptState,
    MultiFlowState,
    RoleState,
    decode_state,
    encode_state,
    register_state,
)
from .store import CollisionError, NodeRunState, RunStateStore

__all__ = [
    "RunContext",
    "RunStateStore",
    "NodeRunState",
    "CollisionError",
    "RuntimeBindings",
    "LiveHandles",
    "LiveHandleStore",
    "InferencerStateBase",
    "MultiFlowState",
    "MultiFlowAttemptState",
    "DualState",
    "BTAState",
    "MFDualState",
    "LinearWorkflowState",
    "RoleState",
    "STATE_REGISTRY",
    "register_state",
    "encode_state",
    "decode_state",
    "active_run_context",
    "bridge_entrypoint",
    "enter_run",
    "exit_run",
    "mint_root",
]

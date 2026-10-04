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
    resolve_run,
)
from .context import RunContext
from .errors import (
    ConcurrentInvocationError,
    InvocationCleanupError,
    InvocationContractError,
    NoInvocationError,
    StageOwnershipError,
    UncertifiedConcurrentUseError,
)
from .handles import LEGACY_HANDLE_SCOPE, LiveHandles, LiveHandleStore
from .invocation import (
    aopen_invocation,
    ctx_bound_gen,
    declared_compat_fields,
    declared_runtime_keys,
    discard_result,
    frame_for,
    framed_agen,
    framed_gen,
    host_pure_certified,
    invocation_of,
    InvocationFrame,
    open_invocation,
    publish_result,
    read_result,
    ResourceLedger,
    RuntimeKey,
)
from .outcome import publish_outcome, read_outcome
from .state import (
    BtaCallSummary,
    BTAState,
    decode_state,
    DualState,
    encode_state,
    InferencerStateBase,
    LinearWorkflowState,
    MFDualState,
    MultiFlowAttemptState,
    MultiFlowState,
    NodeOutcomeState,
    register_state,
    RenderedTaskContractState,
    ROLE_STATE_ATTRS,
    RoleState,
    STATE_REGISTRY,
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
    "LEGACY_HANDLE_SCOPE",
    "InferencerStateBase",
    "MultiFlowState",
    "MultiFlowAttemptState",
    "DualState",
    "BTAState",
    "MFDualState",
    "LinearWorkflowState",
    "RoleState",
    "ROLE_STATE_ATTRS",
    "RenderedTaskContractState",
    "BtaCallSummary",
    "NodeOutcomeState",
    "publish_outcome",
    "read_outcome",
    "STATE_REGISTRY",
    "register_state",
    "encode_state",
    "decode_state",
    "active_run_context",
    "bridge_entrypoint",
    "enter_run",
    "exit_run",
    "mint_root",
    "resolve_run",
    "RuntimeKey",
    "InvocationFrame",
    "ResourceLedger",
    "frame_for",
    "host_pure_certified",
    "invocation_of",
    "open_invocation",
    "aopen_invocation",
    "framed_agen",
    "ctx_bound_gen",
    "publish_result",
    "discard_result",
    "read_result",
    "declared_runtime_keys",
    "declared_compat_fields",
    "framed_gen",
    "InvocationContractError",
    "ConcurrentInvocationError",
    "UncertifiedConcurrentUseError",
    "StageOwnershipError",
    "InvocationCleanupError",
    "NoInvocationError",
]

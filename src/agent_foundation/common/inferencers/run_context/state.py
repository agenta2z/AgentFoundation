"""Tier-1 state values: typed ``InferencerStateBase`` subclasses + a dict surface.

Per the plan (§2.1 / §2.3 / N-R3 / N-S5):

* A node's ``call`` / ``attempt`` state is **either** a thin typed state (built-in
  orchestrators) **or** a plain ``dict`` (generic / leaf / custom).
* There is **one serialization path** with a **discriminator**: the canonical
  encoder is "if the value has ``to_json()``, call it; else dict-passthrough" —
  **never raw ``attrs.asdict``** (which would flatten nested typed states before
  they could stamp their ``_state_class`` tag).
* ``to_json`` **recurses field-by-field**, calling nested states' ``to_json`` so
  a composed state (``MFDualState`` -> ``DualState`` + ``MultiFlowState``)
  round-trips.  ``from_json`` dispatches on ``_state_class`` via a small registry;
  an **unknown** tag degrades to a plain dict + a warning (forward-compatible,
  never a hard crash on resume).
"""

from __future__ import annotations

import warnings
from typing import Any

import attrs

# Registry: discriminator name -> typed-state class.  Populated by @register_state.
STATE_REGISTRY: dict[str, type] = {}

_STATE_CLASS_KEY = "_state_class"


def register_state(cls: type) -> type:
    """Register a typed-state class under its name for ``from_json`` dispatch.

    Explicit (rather than ``__init_subclass__``) to avoid the attrs-slots
    re-creation timing pitfall — the decorator wraps the *final* attrs class.
    """
    name = cls.__name__
    existing = STATE_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Duplicate state class name {name!r}: {existing!r} vs {cls!r}. "
            f"State-class names must be unique (they are the serialized discriminator)."
        )
    STATE_REGISTRY[name] = cls
    return cls


def encode_state(value: Any) -> Any:
    """Canonical Tier-1 encoder (the store-level path).

    * value with ``to_json()`` (a typed state) -> its ``to_json()`` (stamps
      ``_state_class``);
    * ``dict`` -> recurse per value (may hold nested typed states);
    * ``list``/``tuple`` -> recurse per item;
    * everything else -> passthrough (must be JSON-serializable).

    **Never** calls raw ``attrs.asdict`` on a typed state — that is the N-R3 bug.
    """
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return to_json()
    if isinstance(value, dict):
        return {k: encode_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_state(v) for v in value]
    return value


def decode_state(data: Any) -> Any:
    """Inverse of :func:`encode_state`.

    Dispatches on ``_state_class``: a known tag reconstructs the typed object
    (recursively decoding fields first); an **unknown** tag degrades to a plain
    dict + a warning; an untagged dict is a plain dict (the dict discriminator is
    the *absence* of the tag); lists recurse; primitives pass through.
    """
    if isinstance(data, dict):
        kind = data.get(_STATE_CLASS_KEY)
        if kind is None:
            return {k: decode_state(v) for k, v in data.items()}
        state_cls = STATE_REGISTRY.get(kind)
        decoded_fields = {
            k: decode_state(v) for k, v in data.items() if k != _STATE_CLASS_KEY
        }
        if state_cls is None:
            warnings.warn(
                f"Unknown _state_class {kind!r} on resume; degrading to a plain dict "
                f"(forward-compat). Register the class to rehydrate it as typed.",
                stacklevel=2,
            )
            return decoded_fields
        return state_cls(**decoded_fields)
    if isinstance(data, list):
        return [decode_state(v) for v in data]
    return data


class InferencerStateBase:
    """Base for thin typed per-node states.

    Subclasses are ``@attrs.define`` classes decorated with ``@register_state``;
    they declare fields only — serialization/reset is provided generically here
    (no per-class marshaling).
    """

    def state_key(self) -> str:
        """The ``_state_class`` discriminator written into the serialized form."""
        return type(self).__name__

    def to_json(self) -> dict[str, Any]:
        """Field-by-field encode + ``_state_class`` stamp (recurses into nested states)."""
        result: dict[str, Any] = {_STATE_CLASS_KEY: self.state_key()}
        if attrs.has(type(self)):
            for field in attrs.fields(type(self)):
                result[field.name] = encode_state(getattr(self, field.name))
        else:  # pragma: no cover - defensive; typed states are always attrs
            for name, val in vars(self).items():
                result[name] = encode_state(val)
        return result

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Any:
        """Reconstruct via the shared registry-dispatching decoder."""
        return decode_state(data)


# ---------------------------------------------------------------------------
# Built-in orchestrator typed states (thin — field declarations only).
# ---------------------------------------------------------------------------


@register_state
@attrs.define
class MultiFlowState(InferencerStateBase):
    """MultiFlowInferencer per-call dispatch state (the deliberately-outliving fields)."""

    winner_idx: int | None = None
    reviewer_alias: str | None = None
    fixer_alias: str | None = None
    ranking: list[int] | None = None  # None when unset (byte-identical with the old _last_ranking)
    # Runtime sub-queries published by ``propagate_runtime_input``; read by the
    # inherited BTA accessor ``_get_effective_predefined_sub_queries`` on its typed
    # branch.  Renamed from the unused ``cached_sub_queries`` (G4 — zero readers).
    effective_sub_queries: list[str] | None = None
    flow_inputs: list | None = None


@register_state
@attrs.define
class MultiFlowAttemptState(InferencerStateBase):
    """MFI per-ATTEMPT working state — cross-flow visibility buffer + judgments.

    Lives in ``ctx.node().attempt`` (cleared by ``reset_attempt``); distinct from the
    per-CALL dispatch result in ``MultiFlowState`` (GT#3 lifetime split).
    """

    latest_per_flow: dict[int, Any] = attrs.field(factory=dict)
    # Per-flow on-disk output PATH (canonical resolved path of each flow's most-recent
    # output), captured from the LIVE run-context at the same points latest_per_flow text
    # is captured. Mirrors latest_per_flow so the followup own_path/peer_path resolution
    # reads a live per-run path instead of a stale leaf-instance _workspace (M7).
    latest_per_flow_path: dict[int, Any] = attrs.field(factory=dict)
    judgments: list = attrs.field(factory=list)


@register_state
@attrs.define
class DualState(InferencerStateBase):
    """DualInferencer per-call dispatch state."""

    chosen_role: str | None = None
    runner_up: str | None = None
    review_workspace: str | None = None
    fix_workspace: str | None = None
    # Reviewer panel (``reviewer_match_all_non_winners``) — alias/index refs, never objects.
    panel_aliases: list[str] = attrs.field(factory=list)


@register_state
@attrs.define
class BTAState(InferencerStateBase):
    """BreakdownThenAggregate per-call state (predefined/effective sub-queries)."""

    effective_sub_queries: list[str] | None = None
    latest_per_flow: dict[str, Any] = attrs.field(factory=dict)


@register_state
@attrs.define
class RoleState(InferencerStateBase):
    """M7 per-call role state — what ``switch_role`` records into ``ctx.node.call``
    instead of mutating ``self`` (the role/template definition fields)."""

    new_role: str | None = None
    template_key: str | None = None
    template_root_space: str | None = None
    template_version: str | None = None
    modes: Any = None
    changes: dict[str, Any] = attrs.field(factory=dict)


@register_state
@attrs.define
class LinearWorkflowState(InferencerStateBase):
    """LinearWorkflow/Workflow per-call working-state **carrier** (the picklable runner
    state that ``_pending_state``/``_state`` hold).

    Composed into ``MFDualState.runner`` so the workflow working state and the
    dispatch/role state can both live at ``ctx.node().call`` without colliding (GT#14).
    ``state`` is the seed/working dict ``_init_state``/``_pending_state`` carry; the
    remaining serialized runner fields (loop counts, exec seq, splice args, per-step
    attempt counts) are populated by the full runner-state virtualization (Part G/G2).
    """

    state: dict[str, Any] | None = None
    loop_counts: dict[str, int] = attrs.field(factory=dict)
    exec_seq: int = 0
    splice_orig_args: Any = None
    splice_orig_kwargs: Any = None
    splice_step_index: int | None = None
    step_attempt_counts: dict[str, int] = attrs.field(factory=dict)


@register_state
@attrs.define
class MFDualState(InferencerStateBase):
    """MFDual = Dual + MultiFlow by **composition** (not multiple inheritance).

    Holds nested typed states; ``to_json`` recurses into each so the discriminator
    survives (the N-S5 nested-serialization requirement).
    """

    dual: DualState = attrs.field(factory=DualState)
    multiflow: MultiFlowState = attrs.field(factory=MultiFlowState)
    # LinearWorkflow runner carrier — the per-call workflow working state for the Dual
    # consensus loop (so it does not collide with ``dual``/``multiflow`` — GT#14/G1).
    runner: LinearWorkflowState = attrs.field(factory=LinearWorkflowState)

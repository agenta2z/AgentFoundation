"""Ctx-scoped per-call ``template_extra_feed`` overrides.

An orchestrator passes per-call data (e.g. ``upstream_artifacts``) into a child's
wrapper template by publishing a feed dict into a RunContext Tier-3 handle rather
than mutating the child's instance ``template_extra_feed``: under shared-instance
reuse an instance write clobbers concurrent calls. The child merges the override
at render time (``TemplatedInferencerBase._build_template_feed``).

Resolution walks UP the active ctx path and returns the first non-empty override,
so a publisher does not need to know the leaf's exact descendant path (e.g. LWI's
``step_{i}`` slot). A publish composes the nearest ancestor override under its own
keys, so publishing at a deeper node never hides ancestor keys it does not own.

A node carrying ``TEMPLATE_EXTRA_FEED_SCOPE_HANDLE`` is a scope barrier: overrides
published at or below it still resolve, but the walk never continues past it, so
nothing inside the scope sees an override published above it.

A templated parent's own ``template_extra_feed`` and ``modes`` reach its children
through a second channel (B17): the parent publishes them at its own node for its
descendants only (``publish_propagated``), and a renderer resolves the nearest
publication strictly above its own node (``resolve_propagated``), stopping at the
same scope barrier. An outer publication wins over an inner one, as the legacy
instance push let an outer orchestrator's keys override an inner one's. With no
active ctx the parent pushes into its child instances instead (a setup-time API).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from agent_foundation.common.inferencers.run_context import active_run_context

TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE: str = "__template_extra_feed_override__"
TEMPLATE_EXTRA_FEED_SCOPE_HANDLE: str = "__template_extra_feed_scope__"

# The publisher's own keys at a node, kept apart from the composed override so a
# republish re-reads the ancestors instead of its own stale copies of them
# (handles are connection-scoped and outlive a single call).
_OWN_FEED_HANDLE: str = "__template_extra_feed_override_own__"

# A templated parent's own feed / modes, published for its descendants (B17).
TEMPLATE_PROPAGATED_FEED_HANDLE: str = "__template_propagated_feed__"
TEMPLATE_PROPAGATED_MODES_HANDLE: str = "__template_propagated_modes__"


def _path_segments(path: str) -> list[str]:
    return [s for s in path.split("/") if s]


def _walk_up_override(
    store: Any, segments: list[str], *, include_own: bool = True
) -> dict | None:
    for depth in range(len(segments), -1, -1):
        handles = store.peek("/" + "/".join(segments[:depth]))
        if handles is None:
            continue
        override = handles.get(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE, None)
        own_node = depth == len(segments)
        if (include_own or not own_node) and isinstance(override, dict) and override:
            return override
        if handles.get(TEMPLATE_EXTRA_FEED_SCOPE_HANDLE, None):
            return None
    return None


def resolve_ctx_feed_override(ctx: Any = None) -> dict | None:
    """Return the first non-empty feed override on ``ctx``'s path or an ancestor
    (``ctx`` defaults to the active RunContext), else ``None``.

    Handles are path-keyed and not inherited by child contexts, so a leaf rendering
    at ``/flow_0/step_1`` finds an override published at ``/flow_0`` by walking up.
    """
    if ctx is None:
        ctx = active_run_context()
    if ctx is None:
        return None
    return _walk_up_override(ctx._handle_store, _path_segments(ctx.path))


def _write_instance_feed(
    child_inf: Any, feed: Mapping[str, Any], drop_keys: Iterable[str]
) -> None:
    if child_inf is None or not hasattr(child_inf, "template_extra_feed"):
        return
    if child_inf.template_extra_feed is None:
        child_inf.template_extra_feed = {}
    child_inf.template_extra_feed.update(feed)
    for key in drop_keys:
        child_inf.template_extra_feed.pop(key, None)


def publish_child_template_feed(
    child_inf: Any,
    child_slot: str | None,
    feed: Mapping[str, Any],
    *,
    drop_keys: Iterable[str] = (),
) -> bool:
    """Publish per-call ``feed`` for ``child_inf``'s wrapper template.

    With an active RunContext the override is set at ``ctx.child(child_slot)``
    (the slot the orchestrator threads as ``run_context=`` into the child), or at
    the active node when ``child_slot`` is ``None`` (for a descendant whose exact
    path the publisher cannot know). The stored override is the nearest ancestor
    override, then this node's previously published keys, then ``feed``, minus
    ``drop_keys``; the child instance is left untouched. Returns ``True``.

    With no active RunContext, ``feed`` is merged into the child instance's
    ``template_extra_feed`` and ``drop_keys`` are popped from it (the legacy
    behavior). Returns ``False``.
    """
    drop_keys = tuple(drop_keys)
    ctx = active_run_context()
    if ctx is None:
        _write_instance_feed(child_inf, feed, drop_keys)
        return False

    target = ctx.child(child_slot) if child_slot else ctx
    handles = target.handles
    own = {**(handles.get(_OWN_FEED_HANDLE, None) or {}), **feed}
    inherited = _walk_up_override(
        target._handle_store, _path_segments(target.path), include_own=False
    )
    merged = {**(inherited or {}), **own}
    for key in drop_keys:
        own.pop(key, None)
        merged.pop(key, None)
    handles.set(_OWN_FEED_HANDLE, own)
    handles.set(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE, merged)
    return True


def _walk_up_propagated(store: Any, segments: list[str], handle: str) -> dict | None:
    """The nearest non-empty ``handle`` publication strictly above the node at
    ``segments``; a scope barrier at that node or on the way stops the walk."""
    own = store.peek("/" + "/".join(segments))
    if own is not None and own.get(TEMPLATE_EXTRA_FEED_SCOPE_HANDLE, None):
        return None
    for depth in range(len(segments) - 1, -1, -1):
        handles = store.peek("/" + "/".join(segments[:depth]))
        if handles is None:
            continue
        value = handles.get(handle, None)
        if isinstance(value, dict) and value:
            return value
        if handles.get(TEMPLATE_EXTRA_FEED_SCOPE_HANDLE, None):
            return None
    return None


def resolve_propagated(handle: str, ctx: Any = None) -> dict | None:
    """The feed or modes (``handle``) the nearest templated ancestor of ``ctx``
    (default: the active RunContext) published for its descendants, else ``None``."""
    if ctx is None:
        ctx = active_run_context()
    if ctx is None:
        return None
    return _walk_up_propagated(ctx._handle_store, _path_segments(ctx.path), handle)


def publish_propagated(ctx: Any, handle: str, values: Mapping[str, Any]) -> None:
    """Publish a templated parent's own ``values`` (feed or modes) at its node
    ``ctx`` for its descendants, composed under the nearest outer publication (an
    outer key wins). Rewritten on every call, so it never carries a stale key."""
    inherited = _walk_up_propagated(ctx._handle_store, _path_segments(ctx.path), handle)
    ctx.handles.set(handle, {**dict(values), **(inherited or {})})

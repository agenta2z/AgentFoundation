"""Deliverable Boundary Semantics — surfacing helpers.

This module implements the v1.7 Deliverable Boundary contract documented in
``OpenStartup/_dev/_plan/deliverable-boundary-semantics-plan.md``.

The architectural rule (see §14 of the plan):

    Each orchestration boundary owns one canonical deliverables directory.
    Deliverables propagate exactly one boundary HOP per collect-aggregate
    cycle. Non-boundary inferencers (Dual pass-through, LWI pass-through)
    forward their active child's deliverables transparently — they do NOT
    count as a hop.

Key concepts
------------

- **Boundary** — an inferencer with ``is_deliverable_boundary=True``.
- **Pass-through** — a non-boundary inferencer that surfaces its active
  child's deliverables without aggregation/namespacing.
- **Namespace strategy** — how a parent boundary subfolder collected child
  deliverables (``by_child_name``, ``flat``, ``by_role``).
- **Conflict strategy** — how same-named files are resolved during aggregation
  (``skip_existing``, ``largest``, ``first_wins``, ``error``).

Public API
----------

- ``collect_child_boundary_deliverables(parent_workspace, parent_inferencer=None, ...)``
- ``aggregate_into_self_deliverables(parent_workspace, children, ...)``
- ``surface_boundary_deliverables(parent_workspace, child_workspace, ...)``  (pass-through helper)
- ``ChildBoundaryDeliverables`` (dataclass)
- ``AggregateReport`` (dataclass)
- ``DeliverableConflictError`` (exception)
- Strategy enums: ``NamespaceStrategy``, ``ConflictStrategy``
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy types (string enums via Literal-style constants for YAML friendliness)
# ---------------------------------------------------------------------------

NamespaceStrategy = str  # "by_child_name" | "flat" | "by_role"
ConflictStrategy = str   # "skip_existing" | "largest" | "first_wins" | "error"

NAMESPACE_BY_CHILD_NAME = "by_child_name"
NAMESPACE_FLAT = "flat"
NAMESPACE_BY_ROLE = "by_role"

CONFLICT_SKIP_EXISTING = "skip_existing"
CONFLICT_LARGEST = "largest"
CONFLICT_FIRST_WINS = "first_wins"
CONFLICT_ERROR = "error"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DeliverableConflictError(RuntimeError):
    """Raised when conflict_strategy='error' and a same-named file collision
    occurs during aggregation."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChildBoundaryDeliverables:
    """A discovered child boundary's published deliverables."""

    child_name: str                  # e.g. "worker_0" or "planner"
    child_workspace_root: str        # absolute path to child's workspace root
    deliverable_files: List[str]     # paths relative to child's deliverables_dir
    child_workspace: Any = None      # the InferencerWorkspace object (for surface_outputs_from)


@dataclass
class AggregateReport:
    """Summary of an aggregate_into_self_deliverables call."""

    copied: List[Tuple[str, str]] = field(default_factory=list)        # (src_rel, dst_rel)
    conflicted: List[str] = field(default_factory=list)                # dst paths that had conflicts
    skipped: List[str] = field(default_factory=list)                   # dst paths that were skipped
    errors: List[str] = field(default_factory=list)                    # error messages (if any)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_filter(name: str, ws: Any) -> bool:  # noqa: ARG001
    """Default boundary filter — accepts all child boundaries."""
    return True


def collect_child_boundary_deliverables(
    parent_workspace: Any,
    parent_inferencer: Optional[Any] = None,
    *,
    boundary_filter: Callable[[str, Any], bool] = _default_filter,
) -> List[ChildBoundaryDeliverables]:
    """Collect deliverables from immediate child boundaries.

    Boundary detection cascade:

      1. If ``parent_inferencer`` is provided, walk its direct child inferencer
         attribs and check each for ``is_deliverable_boundary=True``
         (in-process, primary signal).
      2. Otherwise (or as fallback for resume scenarios), walk
         ``parent_workspace.children_dir`` on disk, treating any subdir whose
         ``outputs/final_deliverables/`` exists AND is non-empty as a
         boundary.
      3. Either way, NEVER recurse past a boundary — the helper hard-stops
         at the first boundary in each branch (this is the one-boundary-up
         rule from §2.4 of the design doc).

    Args:
        parent_workspace: The parent inferencer's workspace. Must have a
            valid ``children_dir`` attribute.
        parent_inferencer: Optional. If provided, used as the primary
            boundary-detection signal (in-process). When None, falls back to
            on-disk detection.
        boundary_filter: Optional callable filtering child boundaries by
            (child_name, child_workspace) → bool.

    Returns:
        List of ChildBoundaryDeliverables, one per discovered boundary
        child that passes ``boundary_filter``.
    """
    # Avoid circular import at module load time
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    if parent_workspace is None or parent_workspace.children_dir is None:
        return []

    discovered: List[ChildBoundaryDeliverables] = []
    seen_names: set = set()

    # Pass 1: in-process detection via parent_inferencer's child attrs
    if parent_inferencer is not None:
        children_iter = _iter_inferencer_children(parent_inferencer)
        for child_name_attr, child_inf in children_iter:
            if child_inf is None:
                continue
            if not getattr(child_inf, "is_deliverable_boundary", False):
                continue
            child_ws = getattr(child_inf, "_workspace", None)
            if child_ws is None:
                continue
            # The workspace's basename is the canonical "name" for namespacing
            child_dir_name = os.path.basename(child_ws.root)
            if child_dir_name in seen_names:
                continue
            if not boundary_filter(child_dir_name, child_ws):
                continue
            files = _list_deliverable_files(child_ws)
            if not files:
                continue
            discovered.append(
                ChildBoundaryDeliverables(
                    child_name=child_dir_name,
                    child_workspace_root=child_ws.root,
                    deliverable_files=files,
                    child_workspace=child_ws,
                )
            )
            seen_names.add(child_dir_name)

    # Pass 2: on-disk fallback (for resume scenarios or when parent_inferencer
    # doesn't fully cover the workspace tree).
    if os.path.isdir(parent_workspace.children_dir):
        for child_name in sorted(os.listdir(parent_workspace.children_dir)):
            if child_name in seen_names:
                continue
            child_root = os.path.join(parent_workspace.children_dir, child_name)
            if not os.path.isdir(child_root):
                continue
            # Reconstruct the child workspace so deliverables_dir is consistent.
            # Inherit the parent's use_final_deliverables_folder setting (since
            # propagation already enforced this at construction time).
            child_ws = InferencerWorkspace(
                root=child_root,
                use_final_deliverables_folder=(
                    parent_workspace.use_final_deliverables_folder
                ),
            )
            d = child_ws.deliverables_dir
            if d is None or not os.path.isdir(d):
                continue
            files = _list_deliverable_files(child_ws)
            if not files:
                continue
            if not boundary_filter(child_name, child_ws):
                continue
            discovered.append(
                ChildBoundaryDeliverables(
                    child_name=child_name,
                    child_workspace_root=child_root,
                    deliverable_files=files,
                    child_workspace=child_ws,
                )
            )
            seen_names.add(child_name)

    return discovered


def aggregate_into_self_deliverables(
    parent_workspace: Any,
    children: List[ChildBoundaryDeliverables],
    *,
    namespace_strategy: NamespaceStrategy = NAMESPACE_BY_CHILD_NAME,
    conflict_strategy: ConflictStrategy = CONFLICT_SKIP_EXISTING,
    namespace_root: Optional[str] = None,
) -> AggregateReport:
    """Copy collected child deliverables into parent's deliverables_dir.

    Args:
        parent_workspace: The parent inferencer's workspace. Must have
            ``deliverables_dir`` configured.
        children: List of ChildBoundaryDeliverables (from
            ``collect_child_boundary_deliverables``).
        namespace_strategy: How to subfolder the children:
            - ``"by_child_name"``: ``deliverables/{child_name}/...``
            - ``"flat"``: ``deliverables/...`` (no subfolder; conflicts likely)
            - ``"by_role"``: same as ``by_child_name`` but expects role-typed
              child names (planner, executor, etc.). The caller pre-applies
              the role mapping if needed.
        conflict_strategy: How to resolve same-named file collisions:
            - ``"skip_existing"``: never overwrite existing files
            - ``"largest"``: pick the file with the most bytes
            - ``"first_wins"``: first child seen wins
            - ``"error"``: raise DeliverableConflictError on collision
        namespace_root: Optional extra subfolder under deliverables_dir
            (e.g., "workers" for ``by_child_name`` to produce
            ``deliverables/workers/{child_name}/...``).

    Returns:
        AggregateReport with copied/conflicted/skipped lists.
    """
    if parent_workspace is None or parent_workspace.deliverables_dir is None:
        return AggregateReport()

    report = AggregateReport()
    if not children:
        return report

    # Track conflicts across children for `largest` and `error` strategies
    seen_dst_to_src: dict = {}  # dst_rel -> (src_abs, size, child_name)

    for child in children:
        if child.child_workspace is None:
            continue
        # Compute namespace
        if namespace_strategy in (NAMESPACE_BY_CHILD_NAME, NAMESPACE_BY_ROLE):
            ns_parts = []
            if namespace_root:
                ns_parts.append(namespace_root)
            ns_parts.append(child.child_name)
            namespace = "/".join(ns_parts)
        elif namespace_strategy == NAMESPACE_FLAT:
            namespace = namespace_root or None
        else:
            raise ValueError(
                f"Unknown namespace_strategy: {namespace_strategy!r}"
            )

        # Iterate the child's deliverable files explicitly so we can apply
        # conflict_strategy correctly.
        child_dir = child.child_workspace.deliverables_dir
        for rel in child.deliverable_files:
            src_abs = os.path.join(child_dir, rel)
            if namespace:
                dst_rel = os.path.join(namespace, rel)
            else:
                dst_rel = rel
            dst_abs = os.path.join(parent_workspace.deliverables_dir, dst_rel)

            # Apply conflict strategy
            if dst_rel in seen_dst_to_src:
                # Collision within this aggregate call (between two children
                # in `flat` namespace, typically).
                prev = seen_dst_to_src[dst_rel]
                if conflict_strategy == CONFLICT_ERROR:
                    raise DeliverableConflictError(
                        f"Conflict on {dst_rel!r}: "
                        f"{prev[2]!r} vs {child.child_name!r}"
                    )
                elif conflict_strategy == CONFLICT_FIRST_WINS:
                    report.conflicted.append(dst_rel)
                    continue
                elif conflict_strategy == CONFLICT_LARGEST:
                    new_size = os.path.getsize(src_abs)
                    if new_size <= prev[1]:
                        report.conflicted.append(dst_rel)
                        continue
                    # else: overwrite
                elif conflict_strategy == CONFLICT_SKIP_EXISTING:
                    # Already wrote it; skip
                    report.skipped.append(dst_rel)
                    continue

            if os.path.exists(dst_abs):
                if conflict_strategy == CONFLICT_ERROR:
                    raise DeliverableConflictError(
                        f"Conflict on {dst_rel!r}: pre-existing file"
                    )
                if conflict_strategy == CONFLICT_SKIP_EXISTING:
                    report.skipped.append(dst_rel)
                    continue
                # v1.7 BUG FIX: handle ALL strategies on pre-existing files,
                # not just SKIP_EXISTING / ERROR. Previously LARGEST and
                # FIRST_WINS silently overwrote pre-existing files.
                if conflict_strategy == CONFLICT_FIRST_WINS:
                    # Pre-existing file IS the "first"; skip.
                    report.conflicted.append(dst_rel)
                    continue
                if conflict_strategy == CONFLICT_LARGEST:
                    existing_size = os.path.getsize(dst_abs)
                    new_size = os.path.getsize(src_abs)
                    if new_size <= existing_size:
                        report.conflicted.append(dst_rel)
                        continue
                    # else: fall through and overwrite

            os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            import shutil
            shutil.copy2(src_abs, dst_abs)
            report.copied.append((src_abs, dst_rel))
            seen_dst_to_src[dst_rel] = (
                src_abs, os.path.getsize(src_abs), child.child_name
            )

    # v1.7 AC7: log boundary events even when nothing was copied (full skip,
    # all conflicts) — empty events still need observability for debugging.
    logger.info(
        "Boundary aggregate: %d copied, %d conflicted, %d skipped → %s",
        len(report.copied), len(report.conflicted), len(report.skipped),
        parent_workspace.deliverables_dir,
    )
    return report


def surface_boundary_deliverables(
    parent_workspace: Any,
    child_workspace: Any,
    *,
    namespace: Optional[str] = None,
    skip_existing: bool = True,
) -> List[str]:
    """Pass-through surfacing helper for non-boundary parents (Dual, LWI).

    Copies a single child's deliverables to the parent's deliverables_dir
    without aggregation logic. This is for ``DualInferencer`` and
    ``LinearWorkflowInferencer`` where the parent is NOT a boundary itself
    but needs to forward its active child's deliverables.

    Returns:
        List of relative paths copied (relative to parent's deliverables_dir).
    """
    if parent_workspace is None or child_workspace is None:
        return []
    return parent_workspace.surface_outputs_from(
        child_workspace, namespace=namespace, skip_existing=skip_existing,
    )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _iter_inferencer_children(inferencer: Any) -> List[Tuple[str, Any]]:
    """Discover an inferencer's direct child inferencer attribs.

    Prefer the existing ``_iter_child_inferencers()`` method if defined
    (e.g., on BTA at ``breakdown_then_aggregate_inferencer.py:1066``).
    Falls back to ``attrs.fields()`` introspection for inferencers without
    that method.
    """
    # Avoid circular import
    from agent_foundation.common.inferencers.inferencer_base import (
        InferencerBase,
    )
    if hasattr(inferencer, "_iter_child_inferencers"):
        try:
            # _iter_child_inferencers() yields just InferencerBase objects.
            # We need (name, inferencer) tuples; derive name from workspace
            # basename or use a synthetic name. For consistency with the
            # attr-based fallback path, fall through to attrs.fields() —
            # the existing API gives us only objects, not their attr names,
            # which we need for namespacing.
            result = []
            for child in inferencer._iter_child_inferencers():
                # Try to find the attr name by scanning fields()
                from attr import fields
                attr_name = None
                for f in fields(type(inferencer)):
                    if getattr(inferencer, f.name, None) is child:
                        attr_name = f.name
                        break
                if attr_name is None:
                    # Fallback: use workspace basename or class name
                    ws = getattr(child, "_workspace", None)
                    attr_name = (
                        os.path.basename(ws.root) if ws is not None
                        else type(child).__name__
                    )
                result.append((attr_name, child))
            if result:
                return result
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        from attr import fields
        result = []
        for f in fields(type(inferencer)):
            v = getattr(inferencer, f.name, None)
            if isinstance(v, InferencerBase):
                result.append((f.name, v))
        return result
    except Exception:  # pragma: no cover
        return []


def _list_deliverable_files(workspace: Any) -> List[str]:
    """Return all file paths in workspace.deliverables_dir, recursively."""
    d = workspace.deliverables_dir
    if not (d and os.path.isdir(d)):
        return []
    result = []
    for root_dir, _dirs, files in os.walk(d):
        for f in files:
            abs_path = os.path.join(root_dir, f)
            rel_path = os.path.relpath(abs_path, d)
            result.append(rel_path)
    return sorted(result)

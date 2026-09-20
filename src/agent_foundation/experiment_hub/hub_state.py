# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Experiment Hub state derivation and reconciliation.

This module is the single source of truth for "what should the UI show for an
in-progress hub" — derived from the ``WorkflowContext.task_queue`` plus on-disk
task workspaces. There is intentionally no parallel ``hub_state.json`` file with
hub shell metadata: everything except combo submissions is reconstructable.

  - ``reconcile_task_queue_with_disk`` runs ONCE at session restore time. It
    walks each task_queue entry, finds the matching workspace dir, and
    rewrites the in-memory ``status`` / ``workspace`` fields to match disk
    truth. The result is then persisted so subsequent reads see consistent
    state.
  - Workspace mapping prefers the ``.task_meta.json`` sidecar. Falls back to
    byte-for-byte comparison of ``tasks/<dir>/request.txt`` against the queue
    entry's ``request`` for legacy entries that pre-date the sidecar.

Ported faithfully from RankEvolve's ``server.hub_state``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# Filenames inside a task workspace that signal its terminal status.
_CONSENSUS_FILENAME = "implementation_consensus_summary.json"


def _build_workspace_index(tasks_dir: Path) -> dict[str, Path]:
    """Scan ``tasks/<dir>/`` once and return a ``task_id -> workspace_dir`` map.

    Sources, in priority order:
      1. ``.task_meta.json`` sidecar — exact ``task_id`` field. Authoritative
         for workspaces created after the sidecar lands.
      2. Legacy: no sidecar — caller falls back to request-text matching.
    """
    index: dict[str, Path] = {}
    if not tasks_dir.is_dir():
        return index
    for child in tasks_dir.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / ".task_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                task_id = meta.get("task_id", "")
                if task_id and task_id not in index:
                    index[task_id] = child
            except Exception as e:
                logger.debug("Skipping bad .task_meta.json in %s: %s", child, e)
    return index


def _match_legacy_workspace(
    entry: dict[str, Any],
    tasks_dir: Path,
    claimed: set[str] | None = None,
) -> Path | None:
    """Fallback workspace matcher for legacy queue entries (no sidecar).

    Compares the entry's ``request`` field against ``tasks/<dir>/request.txt``
    after newline normalization. Returns the first dir whose content matches
    AND has not already been claimed by another entry in this reconcile pass.

    Determinism: if the user re-runs an identical hypothesis, multiple queue
    entries can share the same ``request`` text. Without ``claimed`` tracking +
    ordered iteration the same dir would be returned for both entries
    (filesystem iterdir order is undefined). We sort dirs by name DESCENDING
    (workspace dirs are timestamped, so most recent first) and skip any
    already claimed.
    """
    expected = (entry.get("request") or "").replace("\r\n", "\n").strip()
    if not expected or not tasks_dir.is_dir():
        return None
    children = sorted(
        (c for c in tasks_dir.iterdir() if c.is_dir()),
        key=lambda c: c.name,
        reverse=True,
    )
    for child in children:
        if claimed is not None and str(child) in claimed:
            continue
        req_file = child / "request.txt"
        if not req_file.is_file():
            continue
        try:
            actual = req_file.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
            if actual == expected:
                return child
        except Exception:
            continue
    return None


def _derive_status_from_workspace(workspace: Path) -> str | None:
    """Determine actual status from on-disk markers.

    Returns:
      "completed" — ``results/implementation_consensus_summary.json`` is present.
      "error"     — workspace exists but consensus marker is absent and we're
                    confident the task is no longer running (post-restart, no
                    process is running, so any in-flight workspace is interrupted).
      None        — workspace doesn't exist (caller should leave entry as-is).

    NOTE: we deliberately return "error" (not "running") for interrupted
    workspaces. A phantom-running entry would wedge the queue runner forever
    via ``running_count >= max_concurrent`` (``get_next_runnable`` would never
    return). Marking "error" is the safer truth.
    """
    if not workspace.is_dir():
        return None
    consensus = workspace / "results" / _CONSENSUS_FILENAME
    if consensus.is_file():
        return "completed"
    return "error"


def reconcile_task_queue_with_disk(wc: Any, tasks_dir: Path) -> int:
    """Update ``wc.task_queue`` entry statuses + workspace fields from disk.

    For each entry:
      - Resolve its workspace via the sidecar index, or fall back to
        request-text matching.
      - Set ``entry["workspace"]`` to the absolute path (so future loads
        don't have to re-scan).
      - Set ``entry["status"]`` based on on-disk markers.
      - If no workspace match: leave the entry as-is.

    Returns the number of entries whose status was changed (useful for tests
    and for the caller to decide whether a follow-up persist is worthwhile).
    """
    logger.debug(
        "reconcile_task_queue_with_disk invoked for %d entries", len(wc.task_queue)
    )
    if not wc.task_queue:
        return 0

    sidecar_index = _build_workspace_index(tasks_dir)
    # Track dirs already claimed by sidecar lookups OR by legacy matches in
    # this reconcile pass — prevents two queue entries from claiming the same
    # workspace when their request text is identical (re-run scenario).
    claimed: set[str] = {str(p) for p in sidecar_index.values()}
    changed = 0

    for entry in wc.task_queue:
        # Implhyp wrapper + per-batch entries are managed authoritatively
        # by the implement-hypothesis bridge's chip callbacks (the hub
        # controller's _on_batch_status writes directly to the entry's
        # status). The reconciler doesn't understand implementation_summary.json,
        # so it could clobber correct statuses with stale ones inferred from
        # the wrong on-disk markers. Skip them.
        if entry.get("tool_name") in (
            "implement_hypothesis",
            "implement_hypothesis_batch",
        ):
            continue
        task_id = entry.get("task_id", "")
        workspace = sidecar_index.get(task_id)
        if workspace is None:
            workspace = _match_legacy_workspace(entry, tasks_dir, claimed=claimed)
            if workspace is None:
                continue
            claimed.add(str(workspace))

        new_status = _derive_status_from_workspace(workspace)
        if new_status is None:
            continue

        old_status = entry.get("status", "")
        old_workspace = entry.get("workspace", "")
        new_workspace = str(workspace.absolute())
        if old_status != new_status or old_workspace != new_workspace:
            entry["status"] = new_status
            entry["workspace"] = new_workspace
            changed += 1
            logger.info(
                "Reconciled task %s: status %s -> %s, workspace %s",
                task_id,
                old_status,
                new_status,
                new_workspace,
            )

    if changed > 0:
        logger.info(
            "reconcile_task_queue_with_disk processed %d entries, %d changed",
            len(wc.task_queue),
            changed,
        )
    return changed

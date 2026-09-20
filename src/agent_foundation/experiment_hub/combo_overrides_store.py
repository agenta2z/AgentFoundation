# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Atomic-JSON store for the per-hub ``combos/current.json`` sidecar.

Factored out of :mod:`agent_foundation.experiment_hub.combo_overrides_service`
(ported from RankEvolve ``combo_overrides_routes.py``). Holds the storage
primitives — path layout, atomic tempfile+os.replace writes, per-(session,
hub) lock management, and the external snapshot-file archive (with reaper) —
so the service layer carries only business logic.

The sidecar carries the user-applied "active combos" set for one
Implementation Hub. Stored at::

    <session_dir>/hub/<mid>/combos/current.json

Pre-apply snapshots live in a sibling ``_archive/`` dir (forensic trail
+ revert payload, bounded by a per-section reaper)::

    <session_dir>/hub/<mid>/combos/_archive/<ts>_apply_<n>.json

Two storage refinements over the proposal_overrides sidecar:

1. **Per-hub file** (instead of per-session with a ``_by_hub`` dict):
   the file IS the hub-scoped state; per-(session, hub) lock matches
   file boundary perfectly.
2. **External snapshot files** (instead of inline ``prior_snapshot`` in
   the log): bounds ``current.json`` growth + enables forensic audit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# Strict pattern for multi_task_id — same as hub_submissions_routes
# (path-traversal defense via the URL parameter).
_MULTI_TASK_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# Per-hub archive bound. Reaper keeps the last N apply snapshots
# (mtime-sorted, oldest preserved as ground truth) — mirrors the
# `learnings_archive._reap_archive` pattern.
_ARCHIVE_KEEP_LAST_N: int = 20


def validate_multi_task_id(multi_task_id: str) -> None:
    if not _MULTI_TASK_ID_PATTERN.match(multi_task_id):
        raise ValueError("Invalid multi_task_id")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def now_filename_ts() -> str:
    """UTC timestamp safe for use in filenames (no colons)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def combos_dir(session_dir: Path, multi_task_id: str) -> Path:
    """Return ``<session>/hub/<mid>/combos`` (the per-hub combos namespace).
    The directory holds ``current.json`` + ``_archive/<ts>_apply_<n>.json``
    snapshot files.
    """
    return session_dir / "hub" / multi_task_id / "combos"


def overrides_path(session_dir: Path, multi_task_id: str) -> Path:
    return combos_dir(session_dir, multi_task_id) / "current.json"


def archive_dir(session_dir: Path, multi_task_id: str) -> Path:
    return combos_dir(session_dir, multi_task_id) / "_archive"


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + "_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def empty_overrides() -> dict[str, Any]:
    """Per-hub schema. The file IS per-hub, so no ``_by_hub`` map."""
    return {
        "active_combos": [],
        "applied_changes_log": [],
        "generatedAt": None,
    }


# Per-(session, hub) asyncio.Lock for serializing read-modify-write on the
# sidecar. Per-hub (not per-session) so concurrent applies on different
# hubs of the same session don't serialize unnecessarily — mirrors
# hub_submissions_routes.
_overrides_locks: dict[tuple[str, str], asyncio.Lock] = {}


def lock_for(session_id: str, multi_task_id: str) -> asyncio.Lock:
    key = (session_id, multi_task_id)
    lock = _overrides_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _overrides_locks[key] = lock
    return lock


# --- Snapshot file management ---------------------------------------------


_ARCHIVE_FILE_RE: re.Pattern[str] = re.compile(r"^\d{8}T\d{6}Z_apply_(\d+)\.json$")


def next_archive_index(archive_path: Path) -> int:
    """Scan existing snapshot filenames and return the next free index.
    Indices are 1-based so the first apply produces ``_apply_001.json``.
    Tolerates gaps from reaping (always returns max+1, not min-of-missing).
    """
    if not archive_path.is_dir():
        return 1
    max_idx = 0
    try:
        for entry in archive_path.iterdir():
            m = _ARCHIVE_FILE_RE.match(entry.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
    except OSError as e:
        logger.warning("Failed to scan archive dir %s: %s", archive_path, e)
    return max_idx + 1


def write_snapshot_file(archive_path: Path, snapshot: dict[str, Any]) -> str:
    """Write a snapshot to ``<archive_dir>/<ts>_apply_<n>.json``. Returns
    the basename (relative to ``archive_dir``) for storage in the log
    entry's ``snapshotRef`` field.
    """
    archive_path.mkdir(parents=True, exist_ok=True)
    idx = next_archive_index(archive_path)
    fname = f"{now_filename_ts()}_apply_{idx:03d}.json"
    target = archive_path / fname
    atomic_write(target, snapshot)
    return fname


def reap_archive(archive_path: Path, keep_last_n: int = _ARCHIVE_KEEP_LAST_N) -> int:
    """Mtime-sorted reaper: deletes oldest snapshots beyond ``keep_last_n``.
    Always preserves the very oldest as ground truth (so log entries
    pointing to reaped snapshots have a baseline to fall back to in
    forensic context). Returns count of files deleted.
    """
    if not archive_path.is_dir():
        return 0
    try:
        entries = [
            p
            for p in archive_path.iterdir()
            if p.is_file() and _ARCHIVE_FILE_RE.match(p.name)
        ]
    except OSError as e:
        logger.warning("Failed to list archive dir %s: %s", archive_path, e)
        return 0
    if len(entries) <= keep_last_n:
        return 0
    # Sort newest-first by mtime; preserve ground-truth (oldest) + last N.
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    keep_set: set[Path] = set(entries[:keep_last_n])
    if entries:
        keep_set.add(entries[-1])  # always keep the OLDEST as ground truth
    deleted = 0
    for p in entries:
        if p in keep_set:
            continue
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            logger.warning("Failed to reap %s: %s", p, e)
    return deleted


def load_snapshot(archive_path: Path, snapshot_ref: str) -> dict[str, Any] | None:
    """Load a snapshot file by its basename. Returns None if missing
    (possibly reaped) or malformed.
    """
    if not snapshot_ref:
        return None
    # Defense-in-depth: ensure the ref is a bare filename (no traversal).
    if not _ARCHIVE_FILE_RE.match(snapshot_ref):
        logger.warning(
            "Refusing to load snapshot with suspicious ref: %r", snapshot_ref
        )
        return None
    return load(archive_path / snapshot_ref)

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Per-hub implementations store — UI-display sidecar for ``/implement-hypothesis``.

Mirrors the shape of :mod:`submissions_service` (atomic-write under ``_lock_for``)
but for a separate file: ``hub_<mid>_implementations.json``. Each row records one
batch produced by the implement-hypothesis bridge:

    {
      "batch_id": "B1",
      "hypothesis_ids": ["H1", "H17"],
      "diff_hash": "...",         # optional — set when the writer can compute it
      "status": "completed",      # queued | running | completed | error
      "workspace_path": "...",
      "rationale": "...",         # optional
      "createdAt": <ms>,
      "updatedAt": <ms>
    }

Ported faithfully from RankEvolve's ``webui.backend.routes.hub_implementations_store``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any


logger: logging.Logger = logging.getLogger(__name__)


# Reuse the same per-(session, multi_task_id) lock space as submissions —
# the two files live side-by-side so atomic writes serialize safely.
_implementations_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _lock_for(session_id: str, multi_task_id: str) -> asyncio.Lock:
    key = (session_id, multi_task_id)
    lock = _implementations_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _implementations_locks[key] = lock
    return lock


def _path(session_dir: Path, multi_task_id: str) -> Path:
    return session_dir / f"hub_{multi_task_id}_implementations.json"


def load_hub_implementations(
    session_dir: Path, multi_task_id: str
) -> list[dict[str, Any]]:
    """Read the implementations array. Returns ``[]`` on missing/corrupt file."""
    path = _path(session_dir, multi_task_id)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("implementations", [])
        return rows if isinstance(rows, list) else []
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return []


def _atomic_write(
    session_dir: Path, multi_task_id: str, rows: list[dict[str, Any]]
) -> None:
    target = _path(session_dir, multi_task_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"multi_task_id": multi_task_id, "implementations": rows}
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f"hub_{multi_task_id}_impls_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def append_hub_implementation(
    session_dir: Path, session_id: str, multi_task_id: str, row: dict[str, Any]
) -> dict[str, Any]:
    """Append a new row (or update an existing one if ``batch_id`` matches).

    Serialized through ``_lock_for`` so concurrent writers from parallel PTI
    workers don't lose each other's updates. Returns the merged row.
    """
    async with _lock_for(session_id, multi_task_id):
        rows = await asyncio.to_thread(
            load_hub_implementations, session_dir, multi_task_id
        )
        batch_id = row.get("batch_id")
        merged: dict[str, Any] | None = None
        if batch_id:
            for i, existing in enumerate(rows):
                if existing.get("batch_id") == batch_id:
                    merged = {**existing, **row}
                    rows[i] = merged
                    break
        if merged is None:
            merged = dict(row)
            rows.append(merged)
        await asyncio.to_thread(_atomic_write, session_dir, multi_task_id, rows)
        return merged

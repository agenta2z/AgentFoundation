# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

# UNAUTHENTICATED — single-user demo logic. Gate behind Tier-1 auth before deploy.

"""Service logic for the client-side Auto Mode state mirror.

Ported from RankEvolve ``autopilot_state_routes.py``. FastAPI stripped: each
former endpoint is a plain service function taking an explicit
``session_dir: Path`` (plus ``session_id`` for the per-session lock key). The
thin REST router lives in OpenTeam and calls these functions.

The client owns decision-making (``hooks/useAutopilot.js``); these functions
just mirror the state to ``<session_dir>/_hub/autopilot/current.json`` for
cross-tab visibility + telemetry. Server is NOT a state-machine driver; the
full server-side autopilot is deferred until users complain about losing
autopilot on browser refresh.

Former endpoints → service functions:
  POST .../_hub/autopilot/state → upsert_autopilot_state
  GET  .../_hub/autopilot/state → get_autopilot_state
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# Per-session asyncio.Lock for serializing the state-mirror writes.
_state_locks: dict[str, asyncio.Lock] = {}


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _state_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _state_locks[session_id] = lock
    return lock


def _state_path(session_dir: Path) -> Path:
    return session_dir / "_hub" / "autopilot" / "current.json"


def _atomic_write(path: Path, data: Any) -> None:
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def upsert_autopilot_state(
    session_id: str, session_dir: Path, body: dict[str, Any]
) -> dict[str, Any]:
    """Mirror the client-side autopilot state to disk.

    Body shape: ``{multi_task_id: str | None, state: dict}``.
    Writes the full body verbatim to
    ``<session_dir>/_hub/autopilot/current.json``, plus a
    ``mirroredAt`` server-side timestamp.

    No interpretation, no validation beyond shape — the client owns
    the schema. This is a write-through; failures are non-fatal for
    the client (the autopilot continues either way).
    """
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    payload: dict[str, Any] = {
        "multi_task_id": body.get("multi_task_id"),
        "state": body.get("state") or {},
        "mirroredAt": _now_iso(),
    }
    async with _lock_for(session_id):
        await asyncio.to_thread(_atomic_write, _state_path(session_dir), payload)
    return {"ok": True, "mirroredAt": payload["mirroredAt"]}


async def get_autopilot_state(session_dir: Path) -> dict[str, Any]:
    """Read the most-recent autopilot state mirror. Returns
    ``{state: null}`` shape if absent — clients treat null as
    "no autopilot has run in this session yet"."""
    path = _state_path(session_dir)
    if not path.is_file():
        return {"multi_task_id": None, "state": None, "mirroredAt": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {"multi_task_id": None, "state": None, "mirroredAt": None}

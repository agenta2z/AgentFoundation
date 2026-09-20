# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

# UNAUTHENTICATED — single-user demo logic. Gate behind Tier-1 auth before deploy.

"""Service logic for the per-session ``proposal_overrides.json`` sidecar.

Ported from RankEvolve ``proposal_overrides_routes.py``. FastAPI stripped:
each former endpoint is a plain service function taking an explicit
``session_dir: Path`` (plus ``session_id`` for the per-session lock key). The
thin REST router lives in OpenTeam and calls these functions.

The sidecar carries user-applied hypothesis re-rankings + deprioritize lists.
Stored at ``<session_dir>/proposal_overrides.json``. The Selection-tab widget
reads it via the ``useProposalOverrides`` React hook + applies overrides at
render-time via the ``mergeOverrides`` pure helper. Original
``phase_outputs.research_proposals_data`` in ``session_state.json`` is NEVER
mutated — preserves the agent server's single-writer invariant.

Former endpoints → service functions:
  GET  .../proposal_overrides             → get_overrides
  POST .../proposal_overrides             → upsert_overrides
  POST .../proposal_overrides/revert_last → revert_last_apply

Reversibility: deleting the sidecar file restores canonical ranks.
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _overrides_path(session_dir: Path) -> Path:
    return session_dir / "proposal_overrides.json"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


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


def _empty_overrides() -> dict[str, Any]:
    return {
        "rankings": [],
        "deprioritize": [],
        "applied_changes_log": [],
        "generatedAt": None,
    }


# Per-session asyncio.Lock for serializing read-modify-write on the sidecar.
_overrides_locks: dict[str, asyncio.Lock] = {}


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _overrides_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _overrides_locks[session_id] = lock
    return lock


async def get_overrides(session_dir: Path) -> dict[str, Any]:
    """Read the sidecar; returns empty object if absent (NEVER errors —
    clients treat absence as 'no overrides applied yet')."""
    return _load(_overrides_path(session_dir)) or _empty_overrides()


async def upsert_overrides(
    session_id: str, session_dir: Path, body: dict[str, Any]
) -> dict[str, Any]:
    """Merge body's ``rankings`` (and optional ``deprioritize``) into the
    existing sidecar. Snapshots the prior state into ``applied_changes_log``
    so ``revert_last_apply`` can roll back.

    Body shape: ``{"rankings": [{"id","oldRank","newRank","rationale","confidence"}, ...],
                   "deprioritize": [{"id","reason"}, ...] (optional)}``.
    """
    path = _overrides_path(session_dir)

    async with _lock_for(session_id):
        existing = _load(path) or _empty_overrides()
        # Snapshot prior state for revert
        prior = {
            "rankings": list(existing.get("rankings", [])),
            "deprioritize": list(existing.get("deprioritize", [])),
        }
        # Merge by H id (incoming wins)
        by_id = {r["id"]: r for r in existing.get("rankings", []) if r.get("id")}
        for r in body.get("rankings", []) or []:
            if not r.get("id"):
                continue
            by_id[r["id"]] = {**r, "appliedAt": _now_iso()}
        existing["rankings"] = sorted(
            by_id.values(), key=lambda r: r.get("newRank", 999)
        )
        if "deprioritize" in body:
            existing["deprioritize"] = body["deprioritize"]
        existing.setdefault("applied_changes_log", []).append(
            {
                "appliedAt": _now_iso(),
                "action": "apply_rerank",
                "affected_h_ids": [
                    r["id"] for r in (body.get("rankings", []) or []) if r.get("id")
                ],
                "deprioritize_count": len(existing.get("deprioritize", [])),
                "prior_snapshot": prior,
            }
        )
        existing["generatedAt"] = _now_iso()
        await asyncio.to_thread(_atomic_write, path, existing)

    return {"ok": True, "overrides": existing}


async def revert_last_apply(session_id: str, session_dir: Path) -> dict[str, Any]:
    """Undo the most recent ``apply_rerank`` entry. Pops the latest log entry,
    restores the snapshotted ``rankings`` + ``deprioritize`` state, and appends
    a new ``revert`` entry to the log so the operation itself is auditable.
    Itself reversible: revert can be reverted (the next apply will snapshot
    the post-revert state).
    """
    path = _overrides_path(session_dir)

    async with _lock_for(session_id):
        existing = _load(path)
        if not existing:
            return {"ok": False, "reason": "no overrides on disk"}
        log = existing.get("applied_changes_log") or []
        # Find the most recent apply_rerank entry
        last_idx = next(
            (
                i
                for i in range(len(log) - 1, -1, -1)
                if log[i].get("action") == "apply_rerank"
            ),
            None,
        )
        if last_idx is None:
            return {"ok": False, "reason": "no apply history to revert"}
        last = log[last_idx]
        snap = last.get("prior_snapshot") or {"rankings": [], "deprioritize": []}
        existing["rankings"] = list(snap.get("rankings", []))
        existing["deprioritize"] = list(snap.get("deprioritize", []))
        log.append(
            {
                "appliedAt": _now_iso(),
                "action": "revert",
                "reverted_apply_at": last.get("appliedAt"),
                "reverted_h_ids": last.get("affected_h_ids", []),
            }
        )
        existing["applied_changes_log"] = log
        existing["generatedAt"] = _now_iso()
        await asyncio.to_thread(_atomic_write, path, existing)

    return {"ok": True, "overrides": existing}

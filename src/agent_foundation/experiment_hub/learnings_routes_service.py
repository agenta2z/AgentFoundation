# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

# UNAUTHENTICATED — single-user demo logic. Gate behind Tier-1 auth before deploy.

"""Service logic for the per-session Accumulated Learnings doc.

Ported from RankEvolve ``learnings_routes.py``. FastAPI stripped: each former
endpoint is a plain service function taking an explicit ``session_dir: Path``.
The thin REST router lives in OpenTeam and calls these functions.

Backed by the file at ``<session_dir>/_learnings/accumulated_learnings.md``.
The doc is split server-side into ``{markdownBody, actions}`` so the React
client never has to parse a markdown JSON fence.

Single-writer invariant: writes go only to ``<session_dir>/_learnings/`` —
a directory NOT touched by the agent server. ``session_state.json`` is never
mutated.

Former endpoints → service functions:
  GET  .../learnings                       → get_learnings
  POST .../learnings/regenerate            → regenerate_learnings
  GET  .../learnings/archives              → list_learnings_archives
  GET  .../learnings/archives/{archive_id} → get_learnings_archive
  POST .../learnings/restore/{archive_id}  → restore_learnings_archive

``get_learnings_archive`` returns the archived markdown body as a plain ``str``
(the OpenTeam router wraps it in a ``text/markdown`` response).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from agent_foundation.experiment_hub import learnings_archive as la
from agent_foundation.experiment_hub.learnings_generator import (
    regenerate_accumulated_learnings,
)
from agent_foundation.experiment_hub.learnings_parser import split_fence

logger: logging.Logger = logging.getLogger(__name__)


def _restore_enabled() -> bool:
    """Restore endpoint behind a feature flag — set the env var to enable.

    Evaluated lazily (not at module scope) to stay lazy-import-compatible
    and to pick up env changes between calls.
    """
    return os.environ.get("RANKEVOLVE_LEARNINGS_RESTORE_ENABLED", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _learnings_md_path(session_dir: Path) -> Path:
    return session_dir / "_learnings" / "accumulated_learnings.md"


async def get_learnings(session_dir: Path) -> dict[str, Any]:
    """Serve the accumulated_learnings.md split into body + actions.

    Returns ``{markdownBody, actions, exists, lastModified, absPath}``. If the
    doc does not exist on disk, returns ``{exists: False}`` so the client can
    show a "Generate now" prompt.
    """
    md_path = _learnings_md_path(session_dir)
    if not md_path.is_file():
        return {
            "markdownBody": "",
            "actions": None,
            "exists": False,
            "lastModified": None,
            "absPath": str(md_path),
        }
    md = md_path.read_text(encoding="utf-8")
    body, actions = split_fence(md, "learnings_actions")
    return {
        "markdownBody": body,
        "actions": actions,
        "exists": True,
        "lastModified": md_path.stat().st_mtime,
        "absPath": str(md_path),
    }


async def regenerate_learnings(session_id: str, session_dir: Path) -> dict[str, Any]:
    """Recompute the doc + JSON fence for a session.

    Heavy I/O bound — wrapped in ``asyncio.to_thread`` to keep the event loop
    responsive. If a subagent has already enriched the markdown body, that
    body is preserved and only the trailing ``learnings_actions`` JSON fence
    is refreshed (per the merge logic in ``regenerate_accumulated_learnings``).
    """
    try:
        result = await asyncio.to_thread(regenerate_accumulated_learnings, session_dir)
    except Exception:
        logger.exception("Failed to regenerate learnings for %s", session_id)
        raise
    return result


async def list_learnings_archives(session_dir: Path) -> dict[str, Any]:
    """Return the versioned archive index (newest first).

    Each row: ``{archive_id, version, archived_at, reason, source, status,
    supersedes, prior_md_sha256, prior_md_size_bytes}``. Only ``status==
    "committed"`` rows are returned to the UI; in-flight or invalid rows are
    filtered (still readable on disk for debugging).
    """
    rows = await asyncio.to_thread(la._read_archive_index, session_dir)
    committed = [r for r in rows if r.get("status", "committed") == "committed"]
    return {"archives": committed}


async def get_learnings_archive(session_dir: Path, archive_id: str) -> str:
    """Return one archived ``accumulated_learnings.md`` body verbatim as a
    plain ``str`` (the OpenTeam router serves it as ``text/markdown``).
    Raises ``KeyError`` if the archive does not exist.
    """
    archive_md = la._archive_root(session_dir) / archive_id / la.LIVE_MD_NAME
    if not archive_md.is_file():
        raise KeyError(f"Archive {archive_id} not found")
    body = await asyncio.to_thread(archive_md.read_text, "utf-8")
    return body


async def restore_learnings_archive(
    session_id: str, session_dir: Path, archive_id: str
) -> dict[str, Any]:
    """Restore an archive into LIVE. Flag-gated for v1
    (``RANKEVOLVE_LEARNINGS_RESTORE_ENABLED=1``). The current LIVE becomes a
    new archive entry first (operation is reversible).

    Raises ``RuntimeError`` if the restore endpoint is gated (disabled),
    ``FileNotFoundError`` if the archive does not exist.
    """
    if not _restore_enabled():
        raise RuntimeError(
            "Restore endpoint is gated. Set "
            "RANKEVOLVE_LEARNINGS_RESTORE_ENABLED=1 to enable."
        )
    try:
        result = await la.revert_promote(
            session_dir,
            archive_id,
            triggered_by=f"user-restore-via-rest:{session_id}",
            session_id=session_id,
        )
    except FileNotFoundError:
        raise
    except Exception:
        logger.exception(
            "restore_learnings_archive failed for %s/%s", session_id, archive_id
        )
        raise
    return result

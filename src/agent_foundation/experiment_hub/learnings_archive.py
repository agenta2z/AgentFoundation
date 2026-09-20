# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Stage → validate → archive → atomic-swap pipeline for the per-session
``_learnings/accumulated_learnings.md`` doc.

Plan: ``/home/zgchen/.claude/plans/humming-tinkering-wirth.md`` v5 §A.

Filesystem layout (mirrored under each session_dir):

    <session>/_learnings/
      accumulated_learnings.md             # LIVE
      learnings_actions.precomputed.json   # LIVE sidecar
      _staging/<run_id>/                   # transient; per-run subdir for forensics
        accumulated_learnings.md
        learnings_actions.precomputed.json
        _stage_meta.json                   # state machine
      _archive/
        index.json                         # newest-first; cheap UI listing
        <utc-ts>_v<N>/                     # one per committed refresh
          accumulated_learnings.md         # byte-copy of LIVE BEFORE the swap
          learnings_actions.precomputed.json
          archive_info.json
          commit_manifest.json             # post-commit audit copy of _stage_meta.json

Single-writer invariant: every write goes through ``_atomic_write``
(tmp + os.replace) inside a per-session asyncio.Lock.

Crash-safety invariants:
  - Archive happens BEFORE swap (copy, not move) — swap-step crash leaves
    LIVE untouched; the just-written archive entry is a recovery source.
  - JSON sidecar replace happens BEFORE markdown replace — drawer-readers
    see (old .md, new JSON) for a sub-millisecond window rather than the
    inconsistent (new .md, old JSON).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Constants — module-level so callers can introspect/override in tests
# ────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION: int = 1
DEFAULT_KEEP_LAST_N: int = 20
DEFAULT_MAX_TOTAL_BYTES: int = 50 * 1024 * 1024
STAGING_DIR_NAME: str = "_staging"
ARCHIVE_DIR_NAME: str = "_archive"
LIVE_MD_NAME: str = "accumulated_learnings.md"
LIVE_META_NAME: str = "learnings_actions.precomputed.json"
INDEX_FILE_NAME: str = "index.json"
STAGE_META_NAME: str = "_stage_meta.json"
ARCHIVE_INFO_NAME: str = "archive_info.json"
COMMIT_MANIFEST_NAME: str = "commit_manifest.json"

_MIN_BODY_BYTES: int = 500
_MAX_BODY_BYTES: int = 1 * 1024 * 1024

# Stale-staging GC threshold — sweep on every refresh start.
_STAGING_STALE_AGE_SECONDS: int = 24 * 60 * 60


# ────────────────────────────────────────────────────────────────────────────
# Per-session refresh lock (with outer mutex for lazy-allocate race)
# ────────────────────────────────────────────────────────────────────────────

_aggregate_refresh_locks: dict[str, asyncio.Lock] = {}
_aggregate_refresh_locks_mutex: asyncio.Lock = asyncio.Lock()


async def _refresh_lock_for(session_id: str) -> asyncio.Lock:
    """Return the per-session refresh lock, lazy-allocating under an outer
    mutex so two concurrent first-time lookups don't allocate two locks.

    Held for the entirety of validate + archive + swap + index update.
    """
    async with _aggregate_refresh_locks_mutex:
        lock = _aggregate_refresh_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _aggregate_refresh_locks[session_id] = lock
        return lock


# ────────────────────────────────────────────────────────────────────────────
# Atomic write (mirrors learnings_generator._atomic_write)
# ────────────────────────────────────────────────────────────────────────────


def _atomic_write(path: Path, data: Any) -> None:
    """tmp + os.replace. JSON-encoded if data is not a str."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + "_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if isinstance(data, str):
                fh.write(data)
            else:
                json.dump(data, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ────────────────────────────────────────────────────────────────────────────
# Path helpers
# ────────────────────────────────────────────────────────────────────────────


def _learnings_dir(session_dir: Path) -> Path:
    return session_dir / "_learnings"


def _staging_root(session_dir: Path) -> Path:
    return _learnings_dir(session_dir) / STAGING_DIR_NAME


def _archive_root(session_dir: Path) -> Path:
    return _learnings_dir(session_dir) / ARCHIVE_DIR_NAME


def _live_md(session_dir: Path) -> Path:
    return _learnings_dir(session_dir) / LIVE_MD_NAME


def _live_meta(session_dir: Path) -> Path:
    return _learnings_dir(session_dir) / LIVE_META_NAME


def _index_path(session_dir: Path) -> Path:
    return _archive_root(session_dir) / INDEX_FILE_NAME


# ────────────────────────────────────────────────────────────────────────────
# Identifiers
# ────────────────────────────────────────────────────────────────────────────


def _utc_iso_compact(dt: datetime | None = None) -> str:
    """e.g. ``20260426T143000Z`` (alphabetic sort = chronological)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _utc_iso(dt: datetime | None = None) -> str:
    """e.g. ``2026-04-26T14:30:00Z``."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def mint_run_id(now: datetime | None = None) -> str:
    return f"agg_{_utc_iso_compact(now)}_{secrets.token_hex(3)}"


def mint_archive_id(version: int, now: datetime | None = None) -> str:
    return f"{_utc_iso_compact(now)}_v{version}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# Staging lifecycle
# ────────────────────────────────────────────────────────────────────────────


def _clean_staging(session_dir: Path, run_id: str | None = None) -> int:
    """If ``run_id`` given, remove just that staging subdir.
    Otherwise, sweep all ``_staging/<*>`` entries older than 24h.

    Returns count of dirs removed.
    """
    root = _staging_root(session_dir)
    if not root.is_dir():
        return 0
    removed = 0
    if run_id:
        target = root / run_id
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            removed = 1
        return removed
    # Sweep stale: by directory mtime, older than threshold.
    now = datetime.now(timezone.utc).timestamp()
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age > _STAGING_STALE_AGE_SECONDS:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed


def staging_dir_for(session_dir: Path, run_id: str) -> Path:
    """Return ``<session>/_learnings/_staging/<run_id>/`` (creates if absent)."""
    d = _staging_root(session_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_stage_meta(staging_dir: Path, meta: dict[str, Any]) -> None:
    _atomic_write(staging_dir / STAGE_META_NAME, meta)


def update_stage_meta(staging_dir: Path, **fields: Any) -> dict[str, Any]:
    """Read-modify-write the staging manifest, returning the merged dict."""
    path = staging_dir / STAGE_META_NAME
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    data.update(fields)
    _atomic_write(path, data)
    return data


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────

_FENCE_RE: re.Pattern[str] = re.compile(
    r"```json\s+learnings_actions\s*\n(.*?)\n```", re.DOTALL
)
_HEADING_RE: re.Pattern[str] = re.compile(r"^##\s+\S", re.MULTILINE)


def _validate_staged(
    staging_md_path: Path,
    current_md_path: Path | None = None,
    *,
    tier: str = "non_strict",
    min_bytes: int = _MIN_BODY_BYTES,
    max_bytes: int = _MAX_BODY_BYTES,
) -> dict[str, Any]:
    """Validate the staged markdown against the current LIVE doc.

    Returns ``{ok: bool, errors: list[str], no_op: bool}``.
    ``no_op=True`` ⇒ md5 match with current; not a failure (caller short-
    circuits without archive/swap).
    """
    errors: list[str] = []
    if not staging_md_path.is_file():
        return {"ok": False, "errors": ["staged file missing"], "no_op": False}
    try:
        body = staging_md_path.read_text(encoding="utf-8")
    except OSError as e:
        return {"ok": False, "errors": [f"read failed: {e}"], "no_op": False}

    size = len(body.encode("utf-8"))
    if size < min_bytes:
        errors.append(f"body too small ({size} B < {min_bytes} B)")
    if size > max_bytes:
        errors.append(f"body too large ({size} B > {max_bytes} B)")
    if not body.strip():
        errors.append("body is whitespace-only")

    fence_match = _FENCE_RE.search(body)
    if fence_match is None:
        errors.append("trailing learnings_actions JSON fence missing")
    else:
        try:
            json.loads(fence_match.group(1))
        except (json.JSONDecodeError, ValueError) as e:
            errors.append(f"learnings_actions fence not parseable: {e}")

    if not _HEADING_RE.search(body):
        errors.append("body lacks any `## ` heading")

    no_op = False
    if not errors and current_md_path is not None and current_md_path.is_file():
        if _md5_file(staging_md_path) == _md5_file(current_md_path):
            no_op = True

    if tier == "strict":
        # Reserved for future variant-aware checks. Skip for v1.
        pass

    return {"ok": not errors, "errors": errors, "no_op": no_op}


# ────────────────────────────────────────────────────────────────────────────
# Index
# ────────────────────────────────────────────────────────────────────────────


def _read_archive_index(session_dir: Path) -> list[dict[str, Any]]:
    path = _index_path(session_dir)
    if not path.is_file():
        return _rebuild_archive_index(session_dir)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "archive index corrupt at %s (%s); rebuilding from disk.", path, e
        )
        return _rebuild_archive_index(session_dir)


def _write_archive_index(session_dir: Path, entries: list[dict[str, Any]]) -> None:
    _atomic_write(_index_path(session_dir), entries)


def _rebuild_archive_index(session_dir: Path) -> list[dict[str, Any]]:
    """Walk ``_archive/<*>/archive_info.json`` and write a fresh
    ``index.json`` (newest-first by version)."""
    root = _archive_root(session_dir)
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        info_path = entry / ARCHIVE_INFO_NAME
        if not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skipping corrupt %s: %s", info_path, e)
            continue
        entries.append(_index_row_from_info(info))
    entries.sort(key=lambda r: int(r.get("version", 0)), reverse=True)
    if entries:
        _atomic_write(_index_path(session_dir), entries)
    return entries


def _index_row_from_info(info: dict[str, Any]) -> dict[str, Any]:
    """Pick the subset of archive_info.json fields the index/UI needs."""
    return {
        "schema_version": info.get("schema_version", SCHEMA_VERSION),
        "archive_id": info.get("archive_id"),
        "version": info.get("version", 0),
        "archived_at": info.get("archived_at"),
        "reason": info.get("reason"),
        "source": info.get("source"),
        "status": info.get("status", "committed"),
        "supersedes": info.get("supersedes"),
        "prior_md_sha256": info.get("prior_md_sha256"),
        "prior_md_size_bytes": info.get("prior_md_size_bytes"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Pruning (preserve-oldest invariant)
# ────────────────────────────────────────────────────────────────────────────


def _parse_version_from_dirname(p: Path) -> int:
    try:
        return int(p.name.rsplit("_v", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _archive_dirs_sorted(archive_root: Path) -> list[Path]:
    """All ``_archive/<id>/`` dirs sorted by parsed version desc, mtime tiebreak."""
    if not archive_root.is_dir():
        return []
    return sorted(
        (
            p
            for p in archive_root.iterdir()
            if p.is_dir() and (p / ARCHIVE_INFO_NAME).exists()
        ),
        key=lambda p: (_parse_version_from_dirname(p), p.stat().st_mtime),
        reverse=True,
    )


def _dir_size_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _reap_archive(
    archive_root: Path,
    keep_last_n: int = DEFAULT_KEEP_LAST_N,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> int:
    """Remove archive dirs beyond ``keep_last_n`` (newest first) AND beyond
    ``max_total_bytes``. **Always preserves the oldest** entry as ground
    truth, regardless of caps.

    Returns count of dirs removed. Best-effort; non-fatal on errors.
    """
    if not archive_root.is_dir():
        return 0
    entries = _archive_dirs_sorted(archive_root)
    if not entries:
        return 0

    # Always keep oldest. Then keep the newest (keep_last_n - 1).
    oldest = entries[-1]
    keep_set: set[Path] = {oldest}
    keep_count = max(0, keep_last_n - 1)
    if keep_count > 0:
        keep_set.update(entries[:keep_count])

    pruned = 0
    for stale in entries:
        if stale not in keep_set:
            shutil.rmtree(stale, ignore_errors=True)
            pruned += 1

    if max_total_bytes <= 0:
        return pruned

    # Size-cap fallback: prune second-oldest first until under cap; oldest stays.
    remaining = [p for p in _archive_dirs_sorted(archive_root) if p != oldest]
    remaining.sort(  # ascending: oldest-second-onwards first
        key=lambda p: (_parse_version_from_dirname(p), p.stat().st_mtime)
    )
    total = sum(_dir_size_bytes(p) for p in [oldest, *remaining])
    for stale in remaining:
        if total <= max_total_bytes:
            break
        try:
            stale_size = _dir_size_bytes(stale)
        except OSError:
            continue
        shutil.rmtree(stale, ignore_errors=True)
        total -= stale_size
        pruned += 1
    return pruned


def prune_learnings_archives(
    archive_root: Path,
    keep_last_n: int = DEFAULT_KEEP_LAST_N,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> int:
    """Public-facing wrapper of :func:`_reap_archive`."""
    return _reap_archive(archive_root, keep_last_n, max_total_bytes)


# ────────────────────────────────────────────────────────────────────────────
# Archive + atomic 2-file promote
# ────────────────────────────────────────────────────────────────────────────


def _next_version(session_dir: Path) -> int:
    idx = _read_archive_index(session_dir)
    if not idx:
        return 1
    return int(idx[0].get("version", 0)) + 1


def _build_archive_info(
    *,
    version: int,
    archive_id: str,
    kind: str,
    source: str,
    reason: str,
    triggered_by: str,
    source_combo_hashes: list[dict[str, Any]] | None,
    baseline_submission_id: str | None,
    supersedes: str | None,
    previous_archive_ts: str | None,
    replaced_by_run: str,
    prior_md_path: Path | None,
    prior_meta_path: Path | None,
    archived_at: str,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "archive_id": archive_id,
        "kind": kind,
        "source": source,
        "status": "committed",
        "started_at": None,
        "completed_at": archived_at,
        "archived_at": archived_at,
        "reason": reason,
        "triggered_by": triggered_by,
        "source_combo_hashes": source_combo_hashes or [],
        "baseline_submission_id": baseline_submission_id,
        "supersedes": supersedes,
        "previous_archive_ts": previous_archive_ts,
        "replaced_by_run": replaced_by_run,
    }
    if prior_md_path is not None and prior_md_path.is_file():
        info["prior_md_sha256"] = _sha256_file(prior_md_path)
        info["prior_md_size_bytes"] = prior_md_path.stat().st_size
    if prior_meta_path is not None and prior_meta_path.is_file():
        info["prior_meta_sha256"] = _sha256_file(prior_meta_path)
        info["prior_meta_size_bytes"] = prior_meta_path.stat().st_size
    return info


async def archive_current_and_promote_staged(
    session_dir: Path,
    staging_dir: Path,
    *,
    run_id: str,
    kind: str = "aggregate_only_refresh",
    source: str = "refresh-llm",
    reason: str = "refresh from UI",
    triggered_by: str = "system",
    source_combo_hashes: list[dict[str, Any]] | None = None,
    baseline_submission_id: str | None = None,
    keep_last_n: int = DEFAULT_KEEP_LAST_N,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    already_locked: bool = False,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Phase 2 of the lifecycle: archive current LIVE → atomic-swap staged
    → write index → reap → cleanup staging.

    Caller MUST have written the staged ``.md`` and ``.precomputed.json``
    into ``staging_dir`` first. Caller MUST also have validated the staged
    output. ``run_id`` is used in audit fields.

    If ``already_locked`` is False, acquires the per-session lock by
    ``session_id`` (which then must be supplied).

    Returns ``{archive_id, version, archived_at, archives_pruned, prior_md_sha256?}``.
    """

    async def _do() -> dict[str, Any]:
        return _archive_and_swap_sync(
            session_dir=session_dir,
            staging_dir=staging_dir,
            run_id=run_id,
            kind=kind,
            source=source,
            reason=reason,
            triggered_by=triggered_by,
            source_combo_hashes=source_combo_hashes,
            baseline_submission_id=baseline_submission_id,
            keep_last_n=keep_last_n,
            max_total_bytes=max_total_bytes,
        )

    if already_locked:
        return await _do()
    if session_id is None:
        raise ValueError(
            "archive_current_and_promote_staged: session_id required "
            "when already_locked=False"
        )
    lock = await _refresh_lock_for(session_id)
    async with lock:
        return await _do()


def _archive_and_swap_sync(
    *,
    session_dir: Path,
    staging_dir: Path,
    run_id: str,
    kind: str,
    source: str,
    reason: str,
    triggered_by: str,
    source_combo_hashes: list[dict[str, Any]] | None,
    baseline_submission_id: str | None,
    keep_last_n: int,
    max_total_bytes: int,
) -> dict[str, Any]:
    archive_root = _archive_root(session_dir)
    archive_root.mkdir(parents=True, exist_ok=True)

    live_md = _live_md(session_dir)
    live_meta = _live_meta(session_dir)

    staged_md = staging_dir / LIVE_MD_NAME
    staged_meta = staging_dir / LIVE_META_NAME
    if not staged_md.is_file():
        raise FileNotFoundError(f"staged markdown missing at {staged_md}")
    # staged precomputed is OPTIONAL — older callers may have skipped it; in
    # that case we fall back to copying the existing live precomputed.

    now = datetime.now(timezone.utc)
    archived_at = _utc_iso(now)

    # Index invariant: every row in `index.json` MUST point at an archive
    # dir containing a real `accumulated_learnings.md`. If there is no
    # prior live doc to preserve, an archive entry would be a row pointing
    # at a 404 — surfacing in the drawer as "Archive body unavailable" and
    # in the toolbar as a phantom "v1 archived". Skip the entire archive
    # ceremony in that case; the swap still happens, and the NEXT refresh
    # will archive THIS just-promoted content as v1.
    have_prior = live_md.is_file()
    logger.info(
        "archive_and_swap: enter run_id=%s have_prior=%s staging=%s "
        "live_md_exists=%s live_meta_exists=%s",
        run_id,
        have_prior,
        staging_dir.name,
        live_md.is_file(),
        live_meta.is_file(),
    )

    archive_id: str | None = None
    version: int = 0
    archive_dir: Path | None = None
    info: dict[str, Any] | None = None
    idx: list[dict[str, Any]] = []

    if have_prior:
        version = _next_version(session_dir)
        archive_id = mint_archive_id(version, now)
        archive_dir = archive_root / archive_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Step 15: COPY (NOT move) live files into archive. Live stays
        # intact until the swap commits — swap-step crash leaves live
        # recoverable from the archive copy.
        shutil.copy2(live_md, archive_dir / LIVE_MD_NAME)
        if live_meta.is_file():
            shutil.copy2(live_meta, archive_dir / LIVE_META_NAME)

        # Determine chain pointers from current index.
        idx = _read_archive_index(session_dir)
        supersedes = idx[0]["archive_id"] if idx else None
        previous_archive_ts = supersedes.split("_v")[0] if supersedes else None

        # Step 16: write archive_info.json AFTER the copies (so its sha256
        # fields refer to the just-archived files).
        info = _build_archive_info(
            version=version,
            archive_id=archive_id,
            kind=kind,
            source=source,
            reason=reason,
            triggered_by=triggered_by,
            source_combo_hashes=source_combo_hashes,
            baseline_submission_id=baseline_submission_id,
            supersedes=supersedes,
            previous_archive_ts=previous_archive_ts,
            replaced_by_run=run_id,
            prior_md_path=archive_dir / LIVE_MD_NAME,
            prior_meta_path=(
                archive_dir / LIVE_META_NAME
                if (archive_dir / LIVE_META_NAME).is_file()
                else None
            ),
            archived_at=archived_at,
        )
        _atomic_write(archive_dir / ARCHIVE_INFO_NAME, info)

    # Step 17: ATOMIC PROMOTE — JSON sidecar FIRST, then markdown. Runs
    # whether or not we archived: this is the critical commit point.
    if staged_meta.is_file():
        os.replace(str(staged_meta), str(live_meta))
    os.replace(str(staged_md), str(live_md))

    # Step 18: update staging manifest in place, then snapshot to archive
    # as commit_manifest.json (audit trail). Snapshot only when we created
    # an archive dir to write into.
    stage_meta_path = staging_dir / STAGE_META_NAME
    if stage_meta_path.is_file():
        try:
            sm = json.loads(stage_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sm = {}
    else:
        sm = {}
    sm.update(
        {
            "status": "committed",
            "completed_at": archived_at,
            "archive_id": archive_id,
            "version": version,
            "first_ever": not have_prior,
        }
    )
    _atomic_write(stage_meta_path, sm)
    if archive_dir is not None:
        _atomic_write(archive_dir / COMMIT_MANIFEST_NAME, sm)

    # Step 20: prepend new index entry; cap to keep_last_n rows. Skipped
    # when there is no archive entry to add.
    archives_pruned = 0
    if info is not None:
        new_row = _index_row_from_info(info)
        new_index = [new_row] + idx
        if keep_last_n > 0:
            new_index = new_index[: max(keep_last_n, 1)]
        _write_archive_index(session_dir, new_index)
        # Step 21: reap on disk (preserves oldest; respects size cap).
        archives_pruned = _reap_archive(
            archive_root,
            keep_last_n,
            max_total_bytes,
        )

    # Step 22: cleanup the just-committed staging dir.
    shutil.rmtree(staging_dir, ignore_errors=True)

    logger.info(
        "archive_and_swap: exit run_id=%s archive_id=%s version=%s "
        "first_ever=%s pruned=%s",
        run_id,
        archive_id,
        version,
        not have_prior,
        archives_pruned,
    )
    return {
        "archive_id": archive_id,
        "version": version,
        "archived_at": archived_at,
        "archives_pruned": archives_pruned,
        "first_ever": not have_prior,
        "prior_md_sha256": info.get("prior_md_sha256") if info else None,
    }


# ────────────────────────────────────────────────────────────────────────────
# Restore / revert
# ────────────────────────────────────────────────────────────────────────────


async def revert_promote(
    session_dir: Path,
    archive_id: str,
    *,
    triggered_by: str = "system",
    keep_last_n: int = DEFAULT_KEEP_LAST_N,
    session_id: str | None = None,
    already_locked: bool = False,
) -> dict[str, Any]:
    """Restore a previously-archived version into LIVE. Goes through the
    same lock + same archive-then-swap dance (the current LIVE becomes a
    new archive entry first, so the operation is itself reversible).

    Used both for the UI restore button AND for automatic recovery when a
    Phase 2 promote crashes mid-way.
    """

    async def _do() -> dict[str, Any]:
        src = _archive_root(session_dir) / archive_id
        if not (src / LIVE_MD_NAME).is_file():
            raise FileNotFoundError(f"archive {archive_id} missing markdown")
        # Lay out a fresh staging dir from the archive content.
        run_id = mint_run_id()
        staging = staging_dir_for(session_dir, run_id)
        shutil.copy2(src / LIVE_MD_NAME, staging / LIVE_MD_NAME)
        if (src / LIVE_META_NAME).is_file():
            shutil.copy2(src / LIVE_META_NAME, staging / LIVE_META_NAME)
        write_stage_meta(
            staging,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "kind": "restore",
                "source": "restore-from-archive",
                "status": "staged",
                "started_at": _utc_iso(),
                "reason": f"restore from {archive_id}",
                "triggered_by": triggered_by,
                "restored_from": archive_id,
            },
        )
        return _archive_and_swap_sync(
            session_dir=session_dir,
            staging_dir=staging,
            run_id=run_id,
            kind="restore",
            source="restore-from-archive",
            reason=f"restore from {archive_id}",
            triggered_by=triggered_by,
            source_combo_hashes=[],
            baseline_submission_id=None,
            keep_last_n=keep_last_n,
            max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
        )

    if already_locked:
        return await _do()
    if session_id is None:
        raise ValueError(
            "revert_promote: session_id required when already_locked=False"
        )
    lock = await _refresh_lock_for(session_id)
    async with lock:
        return await _do()


# ────────────────────────────────────────────────────────────────────────────
# Public API surface for the bridge
# ────────────────────────────────────────────────────────────────────────────

__all__ = (
    "SCHEMA_VERSION",
    "DEFAULT_KEEP_LAST_N",
    "DEFAULT_MAX_TOTAL_BYTES",
    "STAGING_DIR_NAME",
    "ARCHIVE_DIR_NAME",
    "LIVE_MD_NAME",
    "LIVE_META_NAME",
    "INDEX_FILE_NAME",
    "STAGE_META_NAME",
    "ARCHIVE_INFO_NAME",
    "COMMIT_MANIFEST_NAME",
    "_refresh_lock_for",
    "_clean_staging",
    "staging_dir_for",
    "write_stage_meta",
    "update_stage_meta",
    "_validate_staged",
    "_read_archive_index",
    "_write_archive_index",
    "_rebuild_archive_index",
    "_reap_archive",
    "prune_learnings_archives",
    "archive_current_and_promote_staged",
    "revert_promote",
    "mint_run_id",
    "mint_archive_id",
)

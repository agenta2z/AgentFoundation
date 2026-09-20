# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Background poller (WebUI backend) that updates submission entries
with live FBLearner Flow / MAST job status via the meta CLI.

Lives in the WebUI backend (NOT the agent server) to keep
``hub_*_submissions.json`` writes in a single process — preserving the
in-process ``_lock_for(session_id, mid)`` invariant that
``submissions_service.py`` relies on.

Section 6 of the design doc:
- Poll interval: 30s.
- Concurrency: per-poll iteration fans out to all active flows
  concurrently via ``asyncio.gather`` with ``return_exceptions=True`` so
  one bad flow doesn't sink the whole iteration.
- Lock discipline: snapshot pending flows under lock, fetch metadata
  WITHOUT the lock, re-acquire to merge. Avoids holding the per-hub
  asyncio lock during 1-2s/flow CLI shell-outs that would otherwise
  block every user PATCH on the same hub for 10-20s every 30s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


_FLOW_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_\-]{4,64}$")


def _extract_flow_id(flow_uri: str | None) -> str:
    """Pull the flow ID off the end of an MLHub URL. Returns ``""`` for
    falsy input. Mirrors ``submission_runner._extract_flow_id`` (kept
    inline here to avoid importing across the agent-server <-> WebUI
    process boundary).

    Delegates to ``integrations.fblearner.utils.parse_experiment_identifier``
    so the WebUI-side parser stays in sync with the agent-server side.
    The try/except + falsy guard preserves the prior contract: empty
    string for any unparseable input.
    """
    if not flow_uri:
        return ""
    # TODO(port): confirm fblearner.utils import path under AF. No AF-namespace
    # equivalent exists yet, so the real Meta SDK module is kept as-is per the
    # rename map and pulled in via an explicit @manual BUCK hint.
    from rankevolve.src.integrations.fblearner.utils import (  # @manual
        flow_id_to_experiment_id,
        parse_experiment_identifier,
    )

    try:
        ident = parse_experiment_identifier(str(flow_uri))
    except ValueError:
        return ""
    return flow_id_to_experiment_id(ident.flow_id)


def _is_terminal(state: str | None) -> bool:
    if not state:
        return False
    return str(state).upper() in ("COMPLETE", "DEAD", "CANCELLED", "FAILED")


def _normalize_state(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce CLI output shape into the camelCase fields the React
    reducer expects. CLI shape varies across services so we look up a
    handful of common keys defensively."""
    state = (
        meta.get("state") or meta.get("status") or meta.get("flow_state") or "PENDING"
    )
    metrics = meta.get("metrics") or meta.get("monitoring") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "state": str(state).upper(),
        "metrics": {
            # Surface the keys the JobMonitorView already renders.
            "gpuUtil": metrics.get("gpuUtil") or metrics.get("gpu_util"),
            "hostsHealthy": metrics.get("hostsHealthy") or metrics.get("hosts_healthy"),
            "errors": metrics.get("errors"),
            "costUsd": metrics.get("costUsd") or metrics.get("cost_usd"),
            "finalMetrics": metrics.get("finalMetrics")
            or metrics.get("final_metrics")
            or {},
        },
        "error": meta.get("error") or "",
    }


async def _fetch_flow_metadata(flow_id: str) -> dict[str, Any]:
    """Shell out to ``meta ai.workflow-run status --id=<id> --output=json``.

    Section 8.5 #2: the documented flag is ``--output=json``, NOT
    ``--json``. 10s timeout — the meta CLI can hang on network issues.
    """
    if not _FLOW_ID_PATTERN.match(flow_id or ""):
        raise ValueError(f"Refusing to query unsafe flow_id={flow_id!r}")
    proc = await asyncio.create_subprocess_exec(
        "meta",
        "ai.workflow-run",
        "status",
        f"--id={flow_id}",
        "--output=json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        raise
    if proc.returncode != 0:
        # Don't raise — the CLI may legitimately return non-zero for a
        # not-yet-registered flow. Caller's return_exceptions=True
        # records this as a poll failure (debug-level log, not an
        # exception), keeping the iteration going.
        return {"state": "PENDING"}
    try:
        meta = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"meta CLI returned non-JSON for flow {flow_id}: {e}") from e
    return _normalize_state(meta)


class FBLearnerPoller:
    """Background poller that updates submission entries with live
    FBLearner Flow / MAST job status via the meta CLI.

    Lifecycle: ``start()`` schedules the loop coroutine; ``stop()``
    flips the stop flag and awaits the loop's clean exit. Both are
    invoked from the WebUI's ``lifespan`` context manager.
    """

    POLL_INTERVAL_SEC: float = 30.0

    def __init__(self, sessions_dir: Path, app_state: Any) -> None:
        self.sessions_dir: Path = sessions_dir
        self.app_state: Any = app_state
        self._task: asyncio.Task | None = None
        self._stop: bool = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "FBLearnerPoller started (sessions_dir=%s, interval=%ss)",
            self.sessions_dir,
            self.POLL_INTERVAL_SEC,
        )

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self.POLL_INTERVAL_SEC + 5)
            except asyncio.TimeoutError:
                self._task.cancel()
            except Exception:
                pass
            self._task = None
        logger.info("FBLearnerPoller stopped")

    async def _loop(self) -> None:
        while not self._stop:
            try:
                await self._poll_all()
            except Exception as e:
                logger.warning("FBLearnerPoller iteration failed: %s", e)
            # Sleep in small chunks so stop() unblocks promptly.
            slept = 0.0
            while slept < self.POLL_INTERVAL_SEC and not self._stop:
                await asyncio.sleep(0.5)
                slept += 0.5

    async def _poll_all(self) -> None:
        if not self.sessions_dir.is_dir():
            return
        for session_dir in self.sessions_dir.iterdir():
            if self._stop:
                return
            if not session_dir.is_dir():
                continue
            session_id = self._session_id_for_dir(session_dir)
            if session_id is None:
                continue
            for subs_file in session_dir.glob("hub_*_submissions.json"):
                # Extract mid from filename: hub_<mid>_submissions.json
                stem = subs_file.stem  # hub_<mid>_submissions
                if not (stem.startswith("hub_") and stem.endswith("_submissions")):
                    continue
                mid = stem[len("hub_") : -len("_submissions")]
                try:
                    await self._poll_one(session_id, mid, subs_file)
                except Exception as e:
                    logger.warning(
                        "FBLearnerPoller: poll_one failed for %s/%s: %s",
                        session_id,
                        mid,
                        e,
                    )

    def _session_id_for_dir(self, session_dir: Path) -> str | None:
        """Recover ``session_id`` from the dir name.

        Naming is ``<session_id>_<YYYYMMDD>_<HHMMSS>`` and may pick up a
        collision suffix ``_<n>`` in future. The store's
        ``find_session_dir`` is the authoritative inverse — we delegate
        to it via ``app_state.session_store`` rather than open-coding
        rsplit (which would break either today on the multi-underscore
        timestamp or tomorrow on collision suffixes).
        """
        store = getattr(self.app_state, "session_store", None)
        if store is None:
            return None
        # Strategy: derive a candidate session_id by trimming progressively
        # longer suffixes off the dir name and asking the store to map it
        # back. Bounded to a few iterations so a malformed name doesn't
        # spin forever.
        name = session_dir.name
        candidates: list[str] = []
        # Trim candidate suffixes that look like timestamp+collision.
        # Most names are ``<id>_<YYYYMMDD>_<HHMMSS>``: 2 trailing
        # underscore-separated tokens. Try 1-3 trailing tokens to allow
        # for collision suffix ``<id>_<date>_<time>_<n>``.
        parts = name.split("_")
        for n in (2, 3, 1):
            if n >= len(parts):
                continue
            cand = "_".join(parts[:-n])
            if cand and cand not in candidates:
                candidates.append(cand)
        for cand in candidates:
            try:
                resolved = store.find_session_dir(cand)
            except Exception:
                continue
            if resolved is not None and resolved.resolve() == session_dir.resolve():
                return cand
        return None

    async def _poll_one(self, session_id: str, mid: str, path: Path) -> None:
        """Three-phase pattern (Section 6 lock-discipline rule):
        1. Snapshot non-terminal flow_ids under lock.
        2. Fetch metadata WITHOUT the lock (concurrent shell-outs).
        3. Re-acquire lock to merge & write.
        """
        # Lazy import — keeps the poller importable in demo mode where
        # submissions_service' imports may not be wired.
        from agent_foundation.experiment_hub.submissions_service import (
            _atomic_write_submissions,
            _lock_for,
            load_hub_submissions,
        )

        async with _lock_for(session_id, mid):
            submissions = await asyncio.to_thread(
                load_hub_submissions, path.parent, mid
            )
            pending: list[tuple[str, str]] = []
            for sub in submissions:
                flow_id = _extract_flow_id(sub.get("flowUri"))
                if not flow_id:
                    continue
                if _is_terminal(sub.get("fblearnerState")):
                    continue
                sub_id = sub.get("id") or sub.get("submission_id") or ""
                if not sub_id:
                    continue
                pending.append((sub_id, flow_id))

        if not pending:
            return

        # Lock released — fetch concurrently without blocking PATCHes.
        results = await asyncio.gather(
            *[_fetch_flow_metadata(fid) for _, fid in pending],
            return_exceptions=True,
        )

        async with _lock_for(session_id, mid):
            current = await asyncio.to_thread(load_hub_submissions, path.parent, mid)
            by_id = {
                (entry.get("id") or entry.get("submission_id") or ""): entry
                for entry in current
            }
            changed = False
            for (sub_id, flow_id), meta in zip(pending, results):
                entry = by_id.get(sub_id)
                if entry is None:
                    continue
                if isinstance(meta, Exception):
                    logger.debug(
                        "FBLearnerPoller: fetch failed for %s (%s): %s",
                        sub_id,
                        flow_id,
                        meta,
                    )
                    continue
                entry["fblearnerState"] = meta.get("state")
                entry["fblearnerMetrics"] = meta.get("metrics") or {}
                entry["fblearnerLastPolledAt"] = int(time.time())
                if meta.get("error"):
                    entry["fblearnerError"] = meta["error"]
                changed = True
            if changed:
                await asyncio.to_thread(
                    _atomic_write_submissions, path.parent, mid, current
                )

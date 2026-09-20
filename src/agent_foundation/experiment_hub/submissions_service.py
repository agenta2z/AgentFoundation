# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Plain-Python services for Implementation Hub combo submissions.

Ported from RankEvolve's ``hub_submissions_routes.py`` FastAPI router into an
OpenTeam-agnostic library: every endpoint becomes a plain async/sync function
that takes an explicit ``session_dir: Path`` (instead of deriving it from a
FastAPI ``Request``) and returns plain ``dict`` / ``list`` (instead of being
serialized by FastAPI). HTTP error semantics are preserved as plain
exceptions: ``HTTPException(404)`` -> ``KeyError``, ``HTTPException(400)`` ->
``ValueError``.

Submissions are user actions in the Review & Combo view (picking a combo of
hypotheses and clicking "Submit Experiment"). They have NO backend
representation today — this module is the only persistence path. After a
restart, the WebUI backend reads these files and bundles them into the
session_init resume payload (Layer 3) so the user's prior submissions
reappear.

File layout (per session, per multi-task hub):
  <session_dir>/hub_<multi_task_id>_submissions.json

Schema:
  {
    "multi_task_id": "multi-3c9a7b44",
    "submissions": [
      {
        "submission_id": "sub-abc123",
        "combo_key": "H1,H17",
        "hypothesis_ids": ["H1", "H17"],
        "submitted_at": "2026-04-18T...",
        "status": "submitted|running|completed|error",
        "experiment_id": null,
        "notes": ""
      },
      ...
    ]
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

logger: logging.Logger = logging.getLogger(__name__)

# Strict pattern for multi_task_id to prevent path-traversal via the URL parameter.
# Matches values produced by `f"multi-{uuid.uuid4().hex[:8]}"` (tool_executor.py:289)
# plus a small safety margin for future ID schemes.
_MULTI_TASK_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# Per-(session, hub) asyncio.Lock to serialize read-modify-write on the
# submissions JSON file. Prevents lost-update race when two POSTs/PATCHes
# from different tabs land in flight: tmp+rename only protects against
# torn writes, not against logical races (load A, load B, write A, write B
# -> A's append lost). Lazy-allocated per key on first contention.
_submissions_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _lock_for(session_id: str, multi_task_id: str) -> asyncio.Lock:
    key = (session_id, multi_task_id)
    lock = _submissions_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _submissions_locks[key] = lock
    return lock


def _validate_multi_task_id(multi_task_id: str) -> None:
    if not _MULTI_TASK_ID_PATTERN.match(multi_task_id):
        raise ValueError("Invalid multi_task_id")


def _submissions_path(session_dir: Path, multi_task_id: str) -> Path:
    return session_dir / f"hub_{multi_task_id}_submissions.json"


def load_hub_submissions(session_dir: Path, multi_task_id: str) -> list[dict[str, Any]]:
    """Read the submissions array for a hub. Returns [] if file absent or
    unreadable (corrupt JSON, IO error). Public helper — used by Layer 3's
    session_init resume payload builder and by ``list_submissions`` below."""
    path = _submissions_path(session_dir, multi_task_id)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        subs = data.get("submissions", [])
        return subs if isinstance(subs, list) else []
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return []


def _atomic_write_submissions(
    session_dir: Path, multi_task_id: str, submissions: list[dict[str, Any]]
) -> None:
    """tmp + os.replace, matching persist_session_state's pattern at
    session_manager.py:343-344. Avoids partial-write corruption on crash."""
    target = _submissions_path(session_dir, multi_task_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"multi_task_id": multi_task_id, "submissions": submissions}
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=f"hub_{multi_task_id}_subs_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Lazy reconciliation of stuck submissions on session_init
# ---------------------------------------------------------------------------

# Contract-line patterns the runner emits to ``stream_run.txt`` (see
# submission_runner.py:_FLOW_URI_RE / _MAST_JOB_RE). Same shapes the live
# persistence path consumes; mirroring them here lets the reconciler heal
# stuck submissions even when the live event was lost (server restart
# mid-run, transient WS disconnect, silent _apply_submission_state failure
# pre-Change-1).
_FLOW_URI_PATTERN: re.Pattern[str] = re.compile(r"^FLOW_URI:\s*(\S+)\s*$", re.MULTILINE)
_MAST_JOB_PATTERN: re.Pattern[str] = re.compile(r"^MAST_JOB:\s*(\S+)\s*$", re.MULTILINE)


# Stream-end markers — ported as local constants. The canonical RankEvolve
# source imported these from ``rankevolve.src.common.streaming.markers``;
# the AF equivalent is ``agent_foundation.experiment_hub.hub_markers`` (same
# literal marker strings the SubmissionRunner writes). STREAM_FAIL_MARKER is a
# PREFIX (the runner appends the error reason after it), so usages below use
# substring containment, not equality. Aliased to the local names this module
# already references so the reconciler logic stays identical.
from agent_foundation.experiment_hub.hub_markers import (
    STREAM_DONE_MARKER as _STREAM_DONE_MARKER_TEXT,
    STREAM_FAIL_MARKER as _STREAM_FAIL_MARKER_TEXT,
)

# Submission workspace dirs are named ``submission_<YYYYmmdd_HHMMSS>_task-<hex>``
# by ``tool_executor._exec_submission_run``. The runner writes its stream to
# ``<workspace>/_runtime/inferencer_cache/submission/stream_run.txt`` (see
# submission_runner.py module docstring).
_SUBMISSION_WORKSPACE_GLOB: str = "submission_*_task-*"
_STREAM_RUN_RELPATH: tuple[str, ...] = (
    "_runtime",
    "inferencer_cache",
    "submission",
    "stream_run.txt",
)


def _harvest_workspace_terminal_state(workspace_dir: Path) -> dict[str, Any] | None:
    """Read a submission workspace's stream file and extract the terminal
    state if the run has finished. Returns None for workspaces that lack
    a stream file or are still running (no terminal marker)."""
    stream_path = workspace_dir.joinpath(*_STREAM_RUN_RELPATH)
    if not stream_path.is_file():
        return None
    try:
        text = stream_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(
            "reconcile_stuck_submissions: failed to read %s: %s", stream_path, e
        )
        return None

    has_done = _STREAM_DONE_MARKER_TEXT in text
    has_fail = _STREAM_FAIL_MARKER_TEXT in text
    if not (has_done or has_fail):
        return None  # still running

    flow_match = _FLOW_URI_PATTERN.search(text)
    mast_match = _MAST_JOB_PATTERN.search(text)
    return {
        "status": "completed" if has_done else "error",
        "flowUri": flow_match.group(1) if flow_match else None,
        "mastJob": mast_match.group(1) if mast_match else None,
        "runFinishedAt": int(stream_path.stat().st_mtime * 1000),
        "_workspace_mtime_ms": int(stream_path.stat().st_mtime * 1000),
    }


def _match_submission_to_workspace(
    submission: dict[str, Any], workspaces: list[Path]
) -> Path | None:
    """Pair a submission entry with the workspace dir that produced its
    run. Strategy:
      1. If submission carries a ``runTaskId`` field (the runner records
         this when available), match by exact suffix match on the dir name.
      2. Else fall back to ``runStartedAt`` proximity (within 5 minutes
         of the workspace's stream-file mtime). Cheap heuristic; fine
         for the typical case of one submission spawned per minute."""
    run_task_id = submission.get("runTaskId") or ""
    if run_task_id:
        for ws in workspaces:
            if ws.name.endswith(f"task-{run_task_id}") or ws.name.endswith(run_task_id):
                return ws
    started_at = submission.get("runStartedAt") or 0
    if not started_at:
        return None
    PROXIMITY_MS = 5 * 60 * 1000  # 5 minutes
    best, best_delta = None, PROXIMITY_MS
    for ws in workspaces:
        try:
            ws_mtime = int(ws.stat().st_mtime * 1000)
        except OSError:
            continue
        delta = abs(ws_mtime - started_at)
        if delta < best_delta:
            best, best_delta = ws, delta
    return best


def reconcile_stuck_submissions(
    session_dir: Path,
    multi_task_id: str,
    submissions: list[dict[str, Any]],
) -> int:
    """Heal submissions stuck at ``status: submitted/running`` by reading
    their workspace stream files and patching the JSON in place.

    Idempotent: skips entries already at ``status: completed/error``,
    so calling it on every session_init is cheap on the happy path.

    Same atomic-write + lock semantics the live persistence path uses;
    single-writer invariant preserved.

    Returns the number of entries healed (so the caller can re-load the
    file if anything changed).
    """
    stuck = [s for s in submissions if s.get("status") in ("submitted", "running")]
    if not stuck:
        return 0

    # Per-session layout: submission task workspaces live at
    # ``<session_dir>/tasks/`` (was ``<server>/tasks/`` under the old flat
    # layout). session_dir = <server>/sessions/<sid>_<ts>.
    tasks_dir = session_dir / "tasks"
    if not tasks_dir.is_dir():
        return 0
    workspaces = sorted(tasks_dir.glob(_SUBMISSION_WORKSPACE_GLOB))
    if not workspaces:
        return 0

    healed = 0
    for entry in submissions:
        if entry.get("status") not in ("submitted", "running"):
            continue
        ws = _match_submission_to_workspace(entry, workspaces)
        if ws is None:
            continue
        terminal = _harvest_workspace_terminal_state(ws)
        if terminal is None:
            continue  # still running — leave alone
        terminal.pop("_workspace_mtime_ms", None)
        # Only set non-None values so we never blank an existing field.
        for k, v in terminal.items():
            if v is not None:
                entry[k] = v
        healed += 1
        logger.info(
            "reconcile_stuck_submissions: healed submission %s from workspace %s "
            "(status=%s flowUri=%s mastJob=%s)",
            entry.get("id") or entry.get("submission_id"),
            ws.name,
            entry.get("status"),
            entry.get("flowUri"),
            entry.get("mastJob"),
        )

    if healed:
        _atomic_write_submissions(session_dir, multi_task_id, submissions)
    return healed


# --- Baseline-choice sidecar (canonical user override) ----------------
#
# Per-session JSON at ``<session_dir>/_learnings/baseline_choice.json``
# carries the active-baseline override across reloads / browsers / users.
# Single-writer invariant via ``_atomic_write_baseline_choice`` below
# (tmp + os.replace), serialized through the per-hub ``_lock_for``.


def _baseline_choice_path(session_dir: Path) -> Path:
    return session_dir / "_learnings" / "baseline_choice.json"


def load_baseline_choice(session_dir: Path) -> dict[str, Any]:
    """Read ``_learnings/baseline_choice.json``. Returns ``{}`` if absent or
    unreadable. Public helper — used by the GET overlay and the set-baseline
    function below.
    """
    path = _baseline_choice_path(session_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}


def _atomic_write_baseline_choice(session_dir: Path, choice: dict[str, Any]) -> None:
    """tmp + os.replace, matching the submissions writer pattern."""
    target = _baseline_choice_path(session_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix="baseline_choice_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(choice, fh, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def get_implementations(
    session_dir: Path,
    multi_task_id: str,
) -> dict[str, Any]:
    """Return the per-batch implementations sidecar for a hub. Empty list
    if file absent. Plan §Phase 7 — UI display only; not authoritative.

    Was ``GET /{multi_task_id}/implementations``.
    """
    from agent_foundation.experiment_hub.implementations_store import (
        load_hub_implementations,
    )

    _validate_multi_task_id(multi_task_id)
    rows = await asyncio.to_thread(load_hub_implementations, session_dir, multi_task_id)
    return {"multi_task_id": multi_task_id, "implementations": rows}


async def list_submissions(
    session_dir: Path,
    multi_task_id: str,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    """Return the submissions list for a hub. Empty list if file absent.

    Was ``GET /{multi_task_id}/submissions``.

    When the active baseline can be resolved (per the order: ``baseline_id``
    arg → ``_learnings/baseline_choice.json`` → ``isBaseline=True``
    rows by highest NDCG → first ``isBaseline=True`` → none), every row's
    verdict fields are overlaid via ``verdict_computer.compute_verdict_for_row``
    so chip and matrix labels track the chosen baseline. The persisted values
    on disk are NOT mutated — they remain advisory cache that the overlay
    supersedes on every read.
    """
    # Local imports avoid a heavier module-load path for the demo binary which
    # mounts these routes but never serves them under demo mode.
    from agent_foundation.experiment_hub.verdict_computer import (
        overlay_verdicts,
        resolve_baseline,
    )

    _validate_multi_task_id(multi_task_id)
    submissions = await asyncio.to_thread(
        load_hub_submissions, session_dir, multi_task_id
    )

    choice = await asyncio.to_thread(load_baseline_choice, session_dir)
    choice_id = (
        choice.get("baseline_submission_id") if isinstance(choice, dict) else None
    )
    baseline, source = resolve_baseline(
        submissions,
        explicit_baseline_id=baseline_id,
        choice_baseline_id=choice_id if isinstance(choice_id, str) else None,
    )
    overlaid = overlay_verdicts(submissions, baseline)
    return {
        "multi_task_id": multi_task_id,
        "submissions": overlaid,
        "_baselineSource": source,
        "_baselineSubmissionId": baseline.get("id") if baseline else None,
    }


async def set_baseline(
    session_dir: Path,
    session_id: str,
    multi_task_id: str,
    payload: dict[str, Any],
    broadcast_baseline_changed: (Callable[..., Awaitable[None]] | None) = None,
) -> dict[str, Any]:
    """Persist the active baseline choice for a hub.

    Was ``POST /{multi_task_id}/baseline``.

    Body:
      ``{ "baseline_submission_id": "<sid>" | null, "selected_by"?: "ui" }``

    Sending ``null`` (or omitting the field) clears the override and lets
    the resolver fall back to ``isBaseline=True`` rows.

    Writes ``<session_dir>/_learnings/baseline_choice.json`` atomically
    under ``_lock_for(session_id, multi_task_id)``. After writing, schedules
    a ``baseline_changed`` broadcast via the injected
    ``broadcast_baseline_changed`` callback (best-effort) so other open tabs
    viewing the same session refresh. The transport-specific broadcaster is
    injected by the caller (OpenTeam REST layer) instead of imported here,
    keeping this module transport-agnostic.
    """
    _validate_multi_task_id(multi_task_id)

    raw_id = (
        payload.get("baseline_submission_id") if isinstance(payload, dict) else None
    )
    sid: str | None
    if raw_id is None:
        sid = None
    elif isinstance(raw_id, str) and raw_id.strip():
        sid = raw_id.strip()
    else:
        raise ValueError("baseline_submission_id must be a non-empty string or null")

    if sid is not None:
        # Validate the id exists in the hub. Reject stale/forged ids early
        # so the UI never silently picks an absent baseline.
        submissions = await asyncio.to_thread(
            load_hub_submissions, session_dir, multi_task_id
        )
        if not any(s.get("id") == sid for s in submissions):
            raise KeyError(f"submission {sid} not found in hub {multi_task_id}")

    choice = {
        "multi_task_id": multi_task_id,
        "baseline_submission_id": sid,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selected_by": (payload.get("selected_by") or "ui")
        if isinstance(payload, dict)
        else "ui",
    }

    async with _lock_for(session_id, multi_task_id):
        await asyncio.to_thread(_atomic_write_baseline_choice, session_dir, choice)

    # Best-effort broadcast so other tabs refresh. Failures here must NOT
    # fail the request — the persisted state is the authoritative source;
    # the broadcast is only a convenience.
    if broadcast_baseline_changed is not None:
        try:
            await broadcast_baseline_changed(
                session_id=session_id,
                multi_task_id=multi_task_id,
                baseline_submission_id=sid,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning(
                "baseline_changed broadcast failed for session=%s mid=%s: %s",
                session_id,
                multi_task_id,
                e,
            )

    return choice


async def add_submission(
    session_dir: Path,
    session_id: str,
    multi_task_id: str,
    submission: dict[str, Any],
) -> dict[str, Any]:
    """Append a submission. Generates id and submittedAt if absent.

    Was ``POST /{multi_task_id}/submissions``.

    Accepts either camelCase (id, selectedItems, comboKey, submittedAt) — the
    canonical shape used by the React reducer — or the legacy snake_case
    docstring shape (submission_id, hypothesis_ids, combo_key, submitted_at).
    Stores whatever the caller posted; the canonical shape is enforced by the
    frontend's addSubmission helper before POST."""
    _validate_multi_task_id(multi_task_id)

    async with _lock_for(session_id, multi_task_id):
        submissions = await asyncio.to_thread(
            load_hub_submissions, session_dir, multi_task_id
        )
        new_entry = dict(submission)
        # Defaults — only set if neither shape provided the field. Use both
        # canonical (id, submittedAt) and legacy (submission_id, submitted_at)
        # detection so we don't double-assign when caller used canonical shape.
        if "id" not in new_entry and "submission_id" not in new_entry:
            new_entry["id"] = f"sub-{uuid.uuid4().hex[:8]}"
        if "submittedAt" not in new_entry and "submitted_at" not in new_entry:
            new_entry["submittedAt"] = datetime.now(timezone.utc).isoformat()
        new_entry.setdefault("status", "submitted")
        # Plan v4 R2: always persist enable_flags on every submission record so
        # post-hoc analysis tools can read the resolved scoped flag list
        # directly without back-deriving from selectedItems × hypothesisFlagMap.
        # Default to [] when omitted (baseline-only run); keep whatever the
        # caller posted otherwise (typically a list of scoped names like
        # ["hstu_encoder.enable_h17", "hstu_encoder.enable_h8"]).
        new_entry.setdefault("enable_flags", new_entry.get("enableFlags") or [])
        submissions.append(new_entry)

        await asyncio.to_thread(
            _atomic_write_submissions, session_dir, multi_task_id, submissions
        )
    return new_entry


async def update_submission(
    session_dir: Path,
    session_id: str,
    multi_task_id: str,
    submission_id: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Update a submission's mutable fields (status, experiment_id, notes).

    Was ``PATCH /{multi_task_id}/submissions/{submission_id}``.

    Identifies the target by either `id` (canonical) or `submission_id` (legacy).

    PATCH allow-list extended (Step 4 of the design doc) so the agent server
    can ship live FBLearner state through the same single-writer path:

      - status                  (canonical lifecycle field; legacy enum
                                 {submitted, running, completed, error,
                                 cancelled})
      - experiment_id           (legacy snake_case — kept for back-compat)
      - notes                   (user free-form notes on a run)
      - result                  (final metrics blob)
      - runTaskId               (task subtab id of the streaming subprocess)
      - runStartedAt            (epoch ms)
      - runFinishedAt           (epoch ms)
      - experimentId            (NEW canonical camelCase for FBLearner ID)
      - flowUri                 (full FBLearner flow URL)
      - mastJob                 (mast job name, required for `mast cancel`)
      - setupId                 (which setup launched this run)
      - fblearnerState          (PENDING|RUNNING|COMPLETE|DEAD)
      - fblearnerMetrics        (optional metrics blob)
      - fblearnerLastPolledAt   (epoch sec)
      - fblearnerError          (error text when state=DEAD)

    Casing convention: NEW fields use camelCase per the React reducer's
    expected shape. Legacy ``experiment_id`` stays for back-compat but is no
    longer written by new code.
    """
    _validate_multi_task_id(multi_task_id)

    async with _lock_for(session_id, multi_task_id):
        submissions = await asyncio.to_thread(
            load_hub_submissions, session_dir, multi_task_id
        )
        for entry in submissions:
            if (
                entry.get("id") == submission_id
                or entry.get("submission_id") == submission_id
            ):
                for k, v in patch.items():
                    # Permit only safe mutable fields; ignore others to prevent
                    # accidental clobbering of immutable identity fields. New
                    # fields (camelCase) ride alongside legacy snake_case.
                    if k in _MUTABLE_SUBMISSION_FIELDS:
                        entry[k] = v
                await asyncio.to_thread(
                    _atomic_write_submissions, session_dir, multi_task_id, submissions
                )
                return entry
    raise KeyError(f"Submission not found: {submission_id}")


# --- Mutable-field allow-list ------------------------------------------
#
# Copied verbatim from RankEvolve's hub_submissions_routes.py (lines
# 629-675). Used by ``update_submission`` above to filter PATCH bodies down
# to safe mutable fields, never touching immutable identity fields.

_MUTABLE_SUBMISSION_FIELDS: frozenset[str] = frozenset(
    {
        "status",
        "experiment_id",
        "notes",
        "result",
        "runTaskId",
        "runStartedAt",
        "runFinishedAt",
        "experimentId",
        "flowUri",
        "mastJob",
        "setupId",
        "fblearnerState",
        "fblearnerMetrics",
        "fblearnerLastPolledAt",
        "fblearnerError",
        # Round 7: local-run / verdict / lazy-analysis fields (back-compat additive).
        "runMode",
        "runHost",
        "runLogPath",
        "isBaseline",
        "verdict",
        "verdictLabel",
        "deltaPct",
        "comparisonEpoch",
        "stability",
        "stabilityCov",
        "baselineSubmissionId",
        "epochTrajectory",
        "analysisSummary",
        "analysisFile",
        # Plan v7 C2: backpointer from a submission to its auto-analysis
        # task subtab. Populated by _run_submission_analysis when the
        # post-completion DualInferencer fires; consumed by JobMonitorView's
        # clickable Stepper (B4: Analyzing/Done labels switchTab to this id).
        "analysisTaskId",
        "killReason",
        "finalMetrics",
        "epochsCompleted",
        # Combo-overrides revert annotation. Set when an apply removes a
        # combo whose row already exists; informational only — Run /
        # Status logic ignores it. The Review & Combos UI moves rows
        # carrying a non-null inactive_since under "Historical combos".
        "inactive_since",
    }
)


def _enqueue_agent_message(
    queue_root_path: str, session_id: str, msg_type: str, body: dict[str, Any]
) -> None:
    """Direct-write a message to the per-session input queue.

    Avoids the register/deregister churn of constructing a one-shot
    AgentServiceBridge for every REST POST. The agent server's
    SessionAwareServer reads from the same file queue regardless of
    sender, so this works as long as the session is registered (it is —
    registration happens via the WebSocket handshake before any REST
    POST that references the session).
    """
    # TODO(port): StorageBasedQueueService lives under
    # ``rankevolve.src.utils.service_utils.queue_service`` which is not in the
    # AF rename map and is not stdlib. Wire this to the AF queue service (or an
    # injected enqueue callback) before enabling run/cancel submission flows.
    raise NotImplementedError(
        "submissions_service._enqueue_agent_message: AF queue service not yet "
        "wired (was rankevolve StorageBasedQueueService)"
    )


async def run_submission(
    session_dir: Path,
    session_id: str,
    multi_task_id: str,
    submission_id: str,
    payload: dict[str, Any],
    queue_root_path: str | None,
) -> dict[str, Any]:
    """Launch the user's submit_v<n>.py via the agent server's SubmissionRunner.

    Was ``POST /{multi_task_id}/submissions/{submission_id}/run``.

    Body shape (camelCase):

      {
        scriptPath,        # absolute — must match the canonical store
        launchPath,        # sibling launch.json
        enableFlags: [...],# hypothesis IDs (mapped to enable_<name>)
        experimentName,    # ${EXP_NAME} substitution; FBLearner job naming
        submissionLabel,   # human label for the run task subtab
      }

    Behavior:
      1. Validates the submission exists in hub_<mid>_submissions.json.
      2. Forwards a ``run_submission`` message to the agent server via
         the per-session input queue. The agent server enqueues a
         submission_run task; SubmissionRunner spawns the script and
         streams output back through the existing pipeline.
      3. PATCHes the submission row to status='submitted' if it isn't
         already (so the Monitor view picks up the row immediately).
      4. Returns the patched submission record.
    """
    from agent_foundation.experiment_hub.setup_store import load_hub_setup

    _validate_multi_task_id(multi_task_id)

    script_path = (payload.get("scriptPath") or "").strip()
    launch_path = (payload.get("launchPath") or "").strip()
    enable_flags = payload.get("enableFlags") or []
    if not isinstance(enable_flags, list):
        enable_flags = []
    experiment_name = payload.get("experimentName") or f"combo_{submission_id}"
    submission_label = payload.get("submissionLabel") or f"Submission {submission_id}"
    app_layer_version = (payload.get("appLayerVersion") or "").strip()

    if not script_path or not launch_path:
        raise ValueError("scriptPath and launchPath are required")

    # Resolve the setupId by reading the canonical setup state once. The
    # frontend usually has it cached but we re-read defensively to avoid
    # binding a run to a stale setup.
    setup_state = await asyncio.to_thread(load_hub_setup, session_dir, multi_task_id)
    setup_id = (setup_state or {}).get("setupId", "") or ""
    # Auto-build recipe (the user's referenceCommand from the SetupWizard).
    # When the user leaves App-layer version blank in the modal, the runner
    # uses this command to build a fresh fbpkg and capture
    # ``Built FBPKG: <version>`` from stdout. Empty string means
    # "no auto-build configured" — the runner will require a manual
    # ``app_layer_version`` instead.
    build_command = ((setup_state or {}).get("inputs") or {}).get(
        "referenceCommand", ""
    ) or ""

    async with _lock_for(session_id, multi_task_id):
        submissions = await asyncio.to_thread(
            load_hub_submissions, session_dir, multi_task_id
        )
        target = None
        for entry in submissions:
            if (
                entry.get("id") == submission_id
                or entry.get("submission_id") == submission_id
            ):
                target = entry
                break
        if target is None:
            raise KeyError(f"Submission not found: {submission_id}")
        # Optimistic field updates — the agent server's runner will
        # overwrite some via submission_state events as the run progresses.
        target["status"] = "submitted"
        target["setupId"] = setup_id
        target["runStartedAt"] = int(time.time() * 1000)
        await asyncio.to_thread(
            _atomic_write_submissions, session_dir, multi_task_id, submissions
        )

    if queue_root_path is None:
        raise RuntimeError("Agent service not available (no queue_root_path)")
    try:
        await asyncio.to_thread(
            _enqueue_agent_message,
            str(queue_root_path),
            session_id,
            "run_submission",
            {
                "multi_task_id": multi_task_id,
                "submission_id": submission_id,
                "setup_id": setup_id,
                "script_path": script_path,
                "launch_path": launch_path,
                "enable_flags": enable_flags,
                "experiment_name": experiment_name,
                "submission_label": submission_label,
                "app_layer_version": app_layer_version,
                "build_command": build_command,
            },
        )
    except Exception as e:
        logger.error(
            "Failed to enqueue run_submission for session=%s mid=%s sub=%s: %s",
            session_id,
            multi_task_id,
            submission_id,
            e,
        )
        # Roll back optimistic status so the user can retry.
        async with _lock_for(session_id, multi_task_id):
            current = await asyncio.to_thread(
                load_hub_submissions, session_dir, multi_task_id
            )
            for entry in current:
                if (
                    entry.get("id") == submission_id
                    or entry.get("submission_id") == submission_id
                ):
                    entry["status"] = "error"
                    entry["fblearnerError"] = (f"Failed to enqueue run: {e}")[:200]
                    await asyncio.to_thread(
                        _atomic_write_submissions,
                        session_dir,
                        multi_task_id,
                        current,
                    )
                    break
        raise RuntimeError(str(e))

    return target


async def _cancel_mast_job_best_effort(mast_job: str) -> None:
    """Best-effort: cancel an FBLearner / MAST job from the WebUI side.

    Section 11 #3: try ``meta mast.job cancel <name>`` first, then fall
    back to ``mast kill <name>`` (some flows like surreal/lm2 use kill
    instead of cancel). NEVER raises — the local subprocess cancel via
    ``task_cancel_by_id`` is already done by the time this fires; failing
    here just means the FBLearner-side job continues until its own
    timeout but the user-visible state is already 'cancelled'.

    Detached via ``asyncio.create_task`` from the cancel function so the
    response isn't blocked on the meta CLI shell-out (which can take
    several seconds on a busy region).
    """
    for cmd in (
        ["meta", "mast.job", "cancel", mast_job],
        ["mast", "kill", mast_job],
    ):
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.warning(
                "mast cancel via %s skipped — CLI not on PATH",
                cmd[0],
            )
            continue
        except Exception as e:
            logger.warning(
                "mast cancel via %s spawn failed for %s: %s",
                cmd[0],
                mast_job,
                e,
            )
            continue
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            logger.warning(
                "mast cancel via %s timed out for %s",
                cmd[0],
                mast_job,
            )
            continue
        if proc.returncode == 0:
            logger.info("mast cancel succeeded for %s via %s", mast_job, cmd[0])
            return
        logger.warning(
            "mast cancel via %s rc=%s for %s: %s",
            cmd[0],
            proc.returncode,
            mast_job,
            stderr.decode("utf-8", errors="replace")[:200],
        )
    logger.error("All mast cancel attempts exhausted for %s", mast_job)


async def cancel_submission_run(
    session_dir: Path,
    session_id: str,
    multi_task_id: str,
    submission_id: str,
    queue_root_path: str | None,
) -> dict[str, Any]:
    """Cancel a running submission.

    Was ``POST /{multi_task_id}/submissions/{submission_id}/cancel``.

    Two-phase cancel (Section 11 #3):
      1. Forward ``task_cancel_by_id`` to the agent server. The handler
         cancels the asyncio task → ``SubmissionRunner.cancel`` terminates
         the local subprocess → emits ``submission_state`` with
         ``status='cancelled'`` → WebUI applies it under the per-hub
         lock so the run row flips immediately.
      2. If the run captured a ``mastJob`` (only true after the script
         printed ``MAST_JOB:`` on stdout), shell out to
         ``meta mast.job cancel`` so the remote FBLearner job stops too.
         Detached so the response isn't blocked on CLI latency.

    The common 'subprocess hung in import' case has no ``mastJob`` set
    yet — only step 1 fires.
    """
    _validate_multi_task_id(multi_task_id)

    async with _lock_for(session_id, multi_task_id):
        submissions = await asyncio.to_thread(
            load_hub_submissions, session_dir, multi_task_id
        )
        target = None
        for entry in submissions:
            if (
                entry.get("id") == submission_id
                or entry.get("submission_id") == submission_id
            ):
                target = entry
                break
        if target is None:
            raise KeyError(f"Submission not found: {submission_id}")
        run_task_id = target.get("runTaskId") or ""
        mast_job = (target.get("mastJob") or "").strip()

    if not run_task_id:
        # Was HTTP 409 — caller should treat as a conflict/precondition error.
        raise ValueError("Run task id not yet assigned — wait for the run to start")

    if queue_root_path is None:
        raise RuntimeError("Agent service not available (no queue_root_path)")
    try:
        await asyncio.to_thread(
            _enqueue_agent_message,
            str(queue_root_path),
            session_id,
            "task_cancel_by_id",
            {"task_id": run_task_id},
        )
    except Exception as e:
        logger.error("Failed to enqueue cancel: %s", e)
        raise RuntimeError(str(e))

    # Phase 2: detach the FBLearner cancel so the response returns
    # immediately. Failure here is logged but not surfaced to the user —
    # phase 1 already terminated the local subprocess and the user sees
    # 'cancelled' on the run row.
    if mast_job:
        try:
            asyncio.create_task(_cancel_mast_job_best_effort(mast_job))
        except RuntimeError as e:
            logger.warning("Could not schedule mast cancel: %s", e)

    return {
        "submission_id": submission_id,
        "run_task_id": run_task_id,
        "mast_job": mast_job or None,
    }

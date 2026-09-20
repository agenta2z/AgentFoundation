# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Plain-Python services for per-hub submission-setup state.

Ported from RankEvolve's ``hub_submissions_routes.py`` (the "Per-hub
submission-setup state" section + the setup-related FastAPI endpoints) into an
OpenTeam-agnostic library. Each endpoint becomes a plain async/sync function
taking an explicit ``session_dir: Path`` and returning plain ``dict`` /
``list``. HTTP error semantics preserved as plain exceptions:
``HTTPException(404)`` -> ``KeyError``, ``HTTPException(400)`` -> ``ValueError``,
and the 409/413/422/500/502/503 cases -> ``ValueError`` / ``RuntimeError`` as
noted inline.

Hub setup is a SINGLE OBJECT per hub that captures the user's wizard inputs +
the generated script versions, persisted at
``<session_dir>/hub_<multi_task_id>_setup.json``. Mirrors the per-hub
submissions file pattern so resume/reconcile can read both with the same
shape rules.

Single-writer invariant: ONLY this WebUI process writes hub_<mid>_setup.json.
The agent server emits a ``setup_completed`` event on the response queue; the
WebUI's poll_responses extension handles it and calls
``apply_setup_completed_event`` here under the same in-process lock
(``submissions_service._lock_for``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from agent_foundation.experiment_hub.submissions_service import _lock_for

logger: logging.Logger = logging.getLogger(__name__)

# Phase D: direct-upload escape hatch — bypasses PTI by reading user-provided
# submit.py + launch.json from server-side absolute paths and writing the
# canonical version directly. Validation pre-flight via the shared helper
# (`launch_validation`) ensures uploads pass through the same gate as
# PTI-generated content.
_IMPORT_FILE_SIZE_CAP = 1_000_000  # 1 MB per file


def _validate_multi_task_id(multi_task_id: str) -> None:
    # Reuse the submissions_service validator so both stores enforce the same
    # path-traversal guard.
    from agent_foundation.experiment_hub.submissions_service import (
        _validate_multi_task_id as _validate,
    )

    _validate(multi_task_id)


def _setup_path(session_dir: Path, multi_task_id: str) -> Path:
    return session_dir / f"hub_{multi_task_id}_setup.json"


def _setup_scripts_dir(session_dir: Path, multi_task_id: str) -> Path:
    """Canonical store for generated submission scripts (per-hub).

    Numbered files (``submit_v<n>.py`` and ``launch_v<n>.json``) accumulate
    here as the user re-generates via PTI or saves manual edits in the
    drawer. Both writers run in this WebUI process per Section 5 of the
    design doc, so the in-process lock at ``_lock_for(session_id, mid)``
    actually serializes them.
    """
    return session_dir / "setup_scripts" / multi_task_id


def load_hub_setup(session_dir: Path, multi_task_id: str) -> dict[str, Any]:
    """Read the setup STATE for a hub. Returns the inner state dict
    (the value of the on-disk top-level ``"setup"`` key), or ``{}`` if
    absent or unreadable. Public helper — used by the GET endpoint and
    by Layer 3's session_init resume payload builder (Step 8).

    Symmetric inverse of ``_atomic_write_setup`` which wraps ``state``
    in ``{"multi_task_id": ..., "setup": state}``. ``data.get("setup",
    data)`` tolerates legacy unwrapped files (no envelope) by returning
    ``data`` itself.
    """
    path = _setup_path(session_dir, multi_task_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        result = data.get("setup", data)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}


def _atomic_write_setup(
    session_dir: Path, multi_task_id: str, setup_state: dict[str, Any]
) -> None:
    """tmp + os.replace, matching the submissions writer pattern."""
    target = _setup_path(session_dir, multi_task_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"multi_task_id": multi_task_id, "setup": setup_state}
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f"hub_{multi_task_id}_setup_",
        suffix=".tmp",
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


def _atomic_write_text(target: Path, text: str) -> None:
    """tmp + os.replace for an arbitrary text file. Used by the script
    version writer below."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _next_script_version(setup_state: dict[str, Any]) -> int:
    """Pick the next ``submit_v<n>.py`` version number.

    Reads the persisted ``scriptVersions`` array. Empty / missing → start at
    1. Highest existing version + 1 otherwise. Always picking a new number
    (never overwriting) means accidental drawer-Save-during-PTI doesn't
    silently destroy either side's content.
    """
    versions = setup_state.get("scriptVersions") or []
    if not isinstance(versions, list):
        return 1
    max_v = 0
    for v in versions:
        try:
            n = int((v or {}).get("version", 0))
        except (TypeError, ValueError):
            continue
        if n > max_v:
            max_v = n
    return max_v + 1


def _write_canonical_script_version(
    session_dir: Path,
    multi_task_id: str,
    setup_state: dict[str, Any],
    script_content: str,
    launch_content: str,
    source: str,
) -> tuple[Path, Path, int]:
    """Write the next ``submit_v<n>.py`` + ``launch_v<n>.json`` and return
    ``(script_path, launch_path, version)``.

    Caller must hold ``_lock_for(session_id, multi_task_id)``. ``source``
    is recorded on the version entry as either ``'pti'`` or ``'edit'`` so
    the drawer can render history with provenance.
    """
    n = _next_script_version(setup_state)
    base = _setup_scripts_dir(session_dir, multi_task_id)
    base.mkdir(parents=True, exist_ok=True)
    script_path = base / f"submit_v{n}.py"
    launch_path = base / f"launch_v{n}.json"
    _atomic_write_text(script_path, script_content)
    _atomic_write_text(launch_path, launch_content)
    return script_path, launch_path, n


def apply_setup_completed_event(
    session_dir: Path,
    multi_task_id: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Apply a ``setup_completed`` event from the agent server to the
    canonical hub setup file.

    Caller must hold ``_lock_for(session_id, multi_task_id)``. Returns the
    new setup-state dict (so the caller can broadcast it on the WebSocket
    without re-reading the file).

    On status='ready': writes the canonical ``submit_v<n+1>.py`` + sibling
    ``launch_v<n+1>.json`` from the event's ``script_content`` /
    ``launch_content``, appends to ``scriptVersions``, sets ``scriptPath`` /
    ``launchPath``, marks ``status='ready'``, clears any prior ``error``.

    On status='error': leaves any prior ``scriptVersions`` / ``scriptPath``
    intact (so a previous successful generation isn't wiped by a failed
    re-generate), records the error message, sets ``status='error'``.
    """
    setup_state = load_hub_setup(session_dir, multi_task_id) or {}
    # Preserve existing inputs / scriptVersions; only the resolved fields
    # change. The user's wizard inputs were committed on the original POST
    # and shouldn't be touched here.
    setup_state.setdefault("inputs", {})
    setup_state.setdefault("scriptVersions", [])

    status = event.get("status") or "error"
    setup_state["status"] = status
    setup_state["error"] = event.get("error", "") or ""
    setup_state["taskId"] = event.get("task_id") or setup_state.get("taskId")
    setup_state["setupId"] = event.get("setup_id") or setup_state.get("setupId")
    setup_state["setupName"] = (
        event.get("setup_name") or setup_state.get("setupName") or ""
    )

    if status == "ready":
        script_content = event.get("script_content") or ""
        launch_content = event.get("launch_content") or ""
        # Defensive: require both contents — the agent server's hook
        # already validates them, but a malformed event must not write a
        # partial version pair.
        if script_content and launch_content:
            script_path, launch_path, version = _write_canonical_script_version(
                session_dir,
                multi_task_id,
                setup_state,
                script_content,
                launch_content,
                source="pti",
            )
            setup_state["scriptVersions"] = list(
                setup_state.get("scriptVersions", [])
            ) + [
                {
                    "version": version,
                    "mtime": int(time.time() * 1000),
                    "scriptPath": str(script_path),
                    "launchPath": str(launch_path),
                    "source": "pti",
                }
            ]
            setup_state["scriptPath"] = str(script_path)
            setup_state["launchPath"] = str(launch_path)
            setup_state["generatedAt"] = int(time.time() * 1000)
        else:
            # Demote to error — agent server claimed ready but produced
            # no content (should never happen given the hook's validation,
            # but the WebUI must still self-protect).
            setup_state["status"] = "error"
            setup_state["error"] = (
                "setup_completed event marked status=ready but carried no "
                "script_content / launch_content"
            )
    _atomic_write_setup(session_dir, multi_task_id, setup_state)
    return setup_state


def apply_setup_task_started_event(
    session_dir: Path,
    multi_task_id: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Apply a ``setup_task_started`` event from the agent server to the
    canonical hub setup file.

    Caller must hold ``_lock_for(session_id, multi_task_id)``. Returns the
    new setup-state dict (so the caller can broadcast it on the WebSocket
    without re-reading the file).

    Populates ``setup.taskId`` (and ``setupId`` if previously empty) early
    in the in_progress lifecycle — well before the setup_completed event
    fires. Without this, the SubmissionFooterBar's State B "click to view
    progress" navigation has no taskId to navigate to throughout the
    entire PTI run.

    Idempotent: does not overwrite an already-populated taskId; does not
    touch ``status`` or any other field. Specifically does NOT transition
    away from ``status='in_progress'`` — that's setup_completed's job.
    """
    setup_state = load_hub_setup(session_dir, multi_task_id) or {}
    setup_state.setdefault("inputs", {})
    setup_state.setdefault("scriptVersions", [])

    new_task_id = event.get("task_id") or ""
    if new_task_id and not setup_state.get("taskId"):
        setup_state["taskId"] = new_task_id
    new_setup_id = event.get("setup_id") or ""
    if new_setup_id and not setup_state.get("setupId"):
        setup_state["setupId"] = new_setup_id

    _atomic_write_setup(session_dir, multi_task_id, setup_state)
    return setup_state


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
    # injected enqueue callback) before enabling setup generate flows.
    raise NotImplementedError(
        "setup_store._enqueue_agent_message: AF queue service not yet wired "
        "(was rankevolve StorageBasedQueueService)"
    )


async def save_setup_script_version(
    session_dir: Path,
    session_id: str,
    multi_task_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist a user-edited script version (drawer Save).

    Was ``POST /{multi_task_id}/submission-setup/script-version``.

    Body: ``{scriptContent, launchContent?}``. ``launchContent`` is
    optional — if omitted, the previous launch.json is reused (the
    common case for small drawer edits that don't change the buck
    target). The plan rule (Section 5 ``launch.json sharing``) is:
    most drawer-Save edits work fine against the original launch.json;
    only edits that change imports such that the buck target's BUCK
    file no longer covers them require a Re-generate.

    Returns the new full setup-state dict so the caller can dispatch
    SUBMISSION_SCRIPT_VERSION_SAVED with no follow-up GET.
    """
    _validate_multi_task_id(multi_task_id)

    script_content = payload.get("scriptContent")
    launch_content = payload.get("launchContent")
    if not isinstance(script_content, str) or not script_content:
        raise ValueError("scriptContent is required")

    async with _lock_for(session_id, multi_task_id):
        setup_state = await asyncio.to_thread(
            load_hub_setup, session_dir, multi_task_id
        )
        if not setup_state:
            raise KeyError("No setup exists for this hub — run setup first")
        # Reuse prior launch.json content when caller didn't supply a
        # new one. Read the most recent launch_v<n>.json from disk; if
        # absent, fall back to the launch_path on the setup state.
        if launch_content is None:
            prior_launch = setup_state.get("launchPath") or ""
            if prior_launch:
                try:
                    launch_content = Path(prior_launch).read_text(encoding="utf-8")
                except Exception as e:
                    # Was HTTP 409 — precondition failure (no readable prior
                    # launch.json to reuse).
                    raise ValueError(
                        "Could not read prior launch.json — supply "
                        "launchContent in the request body or "
                        f"Re-generate via PTI: {e}"
                    ) from e
            else:
                # Was HTTP 409 — no prior launch.json to reuse.
                raise ValueError(
                    "No prior launch.json to reuse — supply "
                    "launchContent in the request body"
                )
        if not isinstance(launch_content, str) or not launch_content:
            raise ValueError("launchContent must be a non-empty string when provided")

        script_path, launch_path, version = await asyncio.to_thread(
            _write_canonical_script_version,
            session_dir,
            multi_task_id,
            setup_state,
            script_content,
            launch_content,
            "edit",
        )
        setup_state["scriptVersions"] = list(setup_state.get("scriptVersions", [])) + [
            {
                "version": version,
                "mtime": int(time.time() * 1000),
                "scriptPath": str(script_path),
                "launchPath": str(launch_path),
                "source": "edit",
            }
        ]
        setup_state["scriptPath"] = str(script_path)
        setup_state["launchPath"] = str(launch_path)
        setup_state["generatedAt"] = int(time.time() * 1000)
        # Drawer Save during PTI doesn't change status — the backend
        # status field tracks the LAST PTI run. Section 11 #6 toast
        # handling lives on the frontend.
        await asyncio.to_thread(
            _atomic_write_setup, session_dir, multi_task_id, setup_state
        )

    return {"multi_task_id": multi_task_id, "setup": setup_state}


async def get_setup_script_version(
    session_dir: Path,
    multi_task_id: str,
    version: int,
) -> dict[str, Any]:
    """Read a specific submit_v<n>.py + launch_v<n>.json pair.

    Was ``GET /{multi_task_id}/submission-setup/script-version/{version}``.

    Powers the drawer's Versions/Diff vs PTI views. Raises ``KeyError`` (was
    HTTP 404) if the version isn't recorded in setup state OR the files are
    missing on disk (e.g., user manually deleted them).
    """
    _validate_multi_task_id(multi_task_id)

    setup_state = await asyncio.to_thread(load_hub_setup, session_dir, multi_task_id)
    versions = setup_state.get("scriptVersions") or []
    target = None
    for v in versions:
        try:
            n = int((v or {}).get("version", -1))
        except (TypeError, ValueError):
            continue
        if n == version:
            target = v
            break
    if target is None:
        raise KeyError(f"Version {version} not recorded")
    script_path = Path(target.get("scriptPath", ""))
    launch_path = Path(target.get("launchPath", ""))
    try:
        script_content = await asyncio.to_thread(
            script_path.read_text, encoding="utf-8"
        )
        launch_content = await asyncio.to_thread(
            launch_path.read_text, encoding="utf-8"
        )
    except FileNotFoundError as e:
        raise KeyError(f"Version {version} files missing on disk: {e}") from e
    return {
        "version": version,
        "scriptPath": str(script_path),
        "launchPath": str(launch_path),
        "source": target.get("source", ""),
        "mtime": target.get("mtime", 0),
        "scriptContent": script_content,
        "launchContent": launch_content,
    }


async def get_submission_setup(
    session_dir: Path,
    multi_task_id: str,
) -> dict[str, Any]:
    """Return the per-hub setup state. Empty ``{}`` if the file is absent.

    Was ``GET /{multi_task_id}/submission-setup``.
    """
    _validate_multi_task_id(multi_task_id)
    setup_state = await asyncio.to_thread(load_hub_setup, session_dir, multi_task_id)
    return {"multi_task_id": multi_task_id, "setup": setup_state}


async def post_submission_setup(
    session_dir: Path,
    session_id: str,
    multi_task_id: str,
    payload: dict[str, Any],
    queue_root_path: str | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create or replace the per-hub setup. Two modes:

    Was ``POST /{multi_task_id}/submission-setup``.

    - ``mode="generate"`` (default): proxy ``setup_submission`` to the agent
      server so PTI starts generating ``submit_v1.py``. (Original behavior.)
    - ``mode="import"``: BYPASS PTI; read user-provided ``submit.py`` +
      ``launch.json`` from server-side absolute paths, validate, and write
      the canonical ``submit_v<n>.py`` directly with ``source="upload"``.
      ``dry_run=true`` returns the validation report without writing.

    Body shape (camelCase):

      {
        setupName,                # required, displayed in UI
        mode,                     # "generate" (default) | "import"

        # generate-mode fields (PTI):
        referenceScripts: [...],  # absolute paths the user provided
        libraryTemplate,          # template id from the library, or null
        referenceCommand,         # build/launch command (cd … && app-layer …)
        additionalInstructions,
        selectedHypothesisIds: [...],
        hypothesisFlagMap: {hid → enable_<name>},
        setupId,                  # optional; generated if absent

        # import-mode fields:
        scriptPath,               # absolute path to user's submit.py
        launchPath,               # absolute path to user's launch.json
      }

    Behavior:
      1. Read existing setup file under lock; preserve scriptVersions on
         Re-generate (the M1 design doc rule — never lose prior versions).
      2. If existing status is ``in_progress``, raise (was HTTP 409) to
         prevent double-fire.
      3. Branch on ``mode``:
         - generate: write ``status='in_progress'`` + enqueue PTI task.
         - import: validate paths + content; if ``dry_run`` return report;
           otherwise write canonical version via ``_write_canonical_script_version``,
           set ``status='ready'``, skip queue enqueue.
      4. Return the new setup state.
    """
    _validate_multi_task_id(multi_task_id)

    setup_name = (payload.get("setupName") or "").strip()
    if not setup_name:
        raise ValueError("setupName is required")

    mode = (payload.get("mode") or "generate").strip()
    if mode not in ("generate", "import"):
        raise ValueError(f"mode must be 'generate' or 'import' (got {mode!r})")

    # Phase D: import-mode branches into a separate helper to keep the
    # generate-mode flow visually clean.
    if mode == "import":
        return await _post_submission_setup_import(
            session_id=session_id,
            session_dir=session_dir,
            multi_task_id=multi_task_id,
            setup_name=setup_name,
            payload=payload,
            dry_run=dry_run,
        )

    # generate mode (existing PTI behavior):
    setup_id = payload.get("setupId") or f"sub-{uuid.uuid4().hex[:8]}"
    inputs = {
        "name": setup_name,
        "mode": "generate",
        "referenceScripts": payload.get("referenceScripts") or [],
        "libraryTemplate": payload.get("libraryTemplate") or None,
        "referenceCommand": payload.get("referenceCommand") or "",
        "additionalInstructions": payload.get("additionalInstructions") or "",
        "selectedHypothesisIds": payload.get("selectedHypothesisIds") or [],
        "hypothesisFlagMap": payload.get("hypothesisFlagMap") or {},
    }

    async with _lock_for(session_id, multi_task_id):
        existing = await asyncio.to_thread(load_hub_setup, session_dir, multi_task_id)
        if existing.get("status") == "in_progress":
            # Open Question #6 (concurrent Re-generate guard): block a
            # second concurrent enqueue. The frontend disables the button
            # too, but a stale tab could still race. Was HTTP 409.
            raise ValueError("A setup is already in progress for this hub")
        new_state = {
            "status": "in_progress",
            "setupId": setup_id,
            "setupName": setup_name,
            "taskId": None,
            "inputs": inputs,
            "scriptPath": existing.get("scriptPath"),
            "launchPath": existing.get("launchPath"),
            # Critical (M1 fix): preserve prior scriptVersions on Re-generate
            # so previous PTI runs and drawer edits stay accessible.
            "scriptVersions": existing.get("scriptVersions") or [],
            "generatedAt": existing.get("generatedAt"),
            "error": None,
        }
        await asyncio.to_thread(
            _atomic_write_setup, session_dir, multi_task_id, new_state
        )

    # Forward to the agent server. Direct queue write — avoids the
    # register/deregister churn that creating an AgentServiceBridge for a
    # one-shot send would incur.
    if queue_root_path is None:
        raise RuntimeError("Agent service not available (no queue_root_path)")
    try:
        await asyncio.to_thread(
            _enqueue_agent_message,
            str(queue_root_path),
            session_id,
            "setup_submission",
            {
                "multi_task_id": multi_task_id,
                "setup_id": setup_id,
                "setup_name": setup_name,
                "reference_scripts": inputs["referenceScripts"],
                "library_template": inputs["libraryTemplate"],
                "reference_command": inputs["referenceCommand"],
                "additional_instructions": inputs["additionalInstructions"],
                "selected_hypothesis_ids": inputs["selectedHypothesisIds"],
            },
        )
    except Exception as e:
        logger.error(
            "Failed to enqueue setup_submission for session=%s mid=%s: %s",
            session_id,
            multi_task_id,
            e,
        )
        # Roll back the in_progress marker so the next click can try again.
        async with _lock_for(session_id, multi_task_id):
            err_state = dict(new_state)
            err_state["status"] = "error"
            err_state["error"] = f"Failed to enqueue setup task: {e}"[:200]
            await asyncio.to_thread(
                _atomic_write_setup, session_dir, multi_task_id, err_state
            )
        raise RuntimeError(str(e))

    return {"multi_task_id": multi_task_id, "setup": new_state}


async def _post_submission_setup_import(
    *,
    session_id: str,
    session_dir: Path,
    multi_task_id: str,
    setup_name: str,
    payload: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Handle ``post_submission_setup`` with ``mode="import"``."""
    from agent_foundation.experiment_hub.launch_validation import (
        merge_reports,
        validate_launch_json,
        validate_runner_script,
    )

    script_path_str = (payload.get("scriptPath") or "").strip()
    launch_path_str = (payload.get("launchPath") or "").strip()
    if not script_path_str or not launch_path_str:
        raise ValueError("import mode requires both scriptPath and launchPath")

    script_path = Path(script_path_str)
    launch_path = Path(launch_path_str)
    if not script_path.is_absolute():
        raise ValueError("scriptPath must be an absolute path")
    if not launch_path.is_absolute():
        raise ValueError("launchPath must be an absolute path")
    if not script_path.is_file():
        raise ValueError(
            f"scriptPath does not exist or is not a regular file: {script_path}"
        )
    if not launch_path.is_file():
        raise ValueError(
            f"launchPath does not exist or is not a regular file: {launch_path}"
        )

    # Size cap (1 MB per file) to prevent accidental upload of huge artifacts.
    # Was HTTP 413 (payload too large).
    try:
        if script_path.stat().st_size > _IMPORT_FILE_SIZE_CAP:
            raise ValueError(f"scriptPath exceeds {_IMPORT_FILE_SIZE_CAP} bytes")
        if launch_path.stat().st_size > _IMPORT_FILE_SIZE_CAP:
            raise ValueError(f"launchPath exceeds {_IMPORT_FILE_SIZE_CAP} bytes")
    except OSError as e:
        raise ValueError(f"Failed to stat path: {e}") from e

    try:
        script_content = script_path.read_text(encoding="utf-8")
        launch_content = launch_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"Failed to read path: {e}") from e

    # Validation pre-flight (shared helper; same gate as PTI's _setup_completion_hook).
    report = merge_reports(
        validate_launch_json(launch_content),
        validate_runner_script(script_content),
    )

    if dry_run:
        return {
            "multi_task_id": multi_task_id,
            "validation_report": report,
        }

    # Block on errors; warnings are advisory ("user explicitly bypassed PTI").
    # Was HTTP 422 (unprocessable entity).
    if report.get("severity") == "error":
        raise ValueError({"validation_report": report})

    setup_id = payload.get("setupId") or f"sub-{uuid.uuid4().hex[:8]}"
    inputs = {
        "name": setup_name,
        "mode": "import",
        "scriptPath": script_path_str,
        "launchPath": launch_path_str,
        "selectedHypothesisIds": payload.get("selectedHypothesisIds") or [],
        "hypothesisFlagMap": payload.get("hypothesisFlagMap") or {},
    }

    async with _lock_for(session_id, multi_task_id):
        existing = await asyncio.to_thread(load_hub_setup, session_dir, multi_task_id)
        if existing.get("status") == "in_progress":
            # Was HTTP 409.
            raise ValueError("A setup is already in progress for this hub")

        # Build provisional setup_state (pre-write) so _write_canonical_script_version
        # picks the right next version number from existing.scriptVersions.
        provisional = dict(existing)
        if not isinstance(provisional.get("scriptVersions"), list):
            provisional["scriptVersions"] = []

        try:
            written_script_path, written_launch_path, version = await asyncio.to_thread(
                _write_canonical_script_version,
                session_dir,
                multi_task_id,
                provisional,
                script_content,
                launch_content,
                "upload",
            )
        except Exception as e:
            logger.error(
                "Direct-upload write failed for session=%s mid=%s: %s",
                session_id,
                multi_task_id,
                e,
            )
            # Was HTTP 500.
            raise RuntimeError(str(e)) from e

        now_ms = int(time.time() * 1000)
        version_entry = {
            "version": version,
            "scriptPath": str(written_script_path),
            "launchPath": str(written_launch_path),
            "source": "upload",
            "createdAt": now_ms,
        }
        prior_versions = existing.get("scriptVersions") or []
        new_state = {
            "status": "ready",
            "setupId": setup_id,
            "setupName": setup_name,
            "taskId": None,
            "inputs": inputs,
            "scriptPath": str(written_script_path),
            "launchPath": str(written_launch_path),
            "scriptVersions": [*prior_versions, version_entry],
            "generatedAt": now_ms,
            "error": "",
        }
        await asyncio.to_thread(
            _atomic_write_setup, session_dir, multi_task_id, new_state
        )

    return {
        "multi_task_id": multi_task_id,
        "setup": new_state,
        "validation_report": report,
    }

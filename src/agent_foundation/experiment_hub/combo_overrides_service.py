# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

# UNAUTHENTICATED — single-user demo logic. Gate behind Tier-1 auth before deploy.

"""Service logic for the per-hub ``combos/current.json`` sidecar.

Ported from RankEvolve ``combo_overrides_routes.py``. FastAPI stripped: each
former endpoint is a plain service function taking an explicit
``session_dir: Path`` (plus ``session_id`` for the per-(session, hub) lock key
and the optional ``emit_event`` broadcast callback). The thin REST router lives
in OpenTeam and calls these functions.

The sidecar carries the user-applied "active combos" set for one
Implementation Hub, stored at ``<session_dir>/hub/<mid>/combos/current.json``.
Storage primitives (paths, atomic writes, locks, archive snapshots) live in
:mod:`agent_foundation.experiment_hub.combo_overrides_store`.

Former endpoints → service functions:
  GET  .../combo_overrides/{mid}              → get_combo_overrides
  POST .../combo_overrides/{mid}/apply        → apply_combo_overrides
  POST .../combo_overrides/{mid}/revert_last  → revert_last_combo_apply

Reversibility: deleting the ``hub/<mid>/combos/`` subdir restores the
"no active combos" state. Selection tab unfilters; Review & Combos
shows all submissions chronologically.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_foundation.experiment_hub.combo_overrides_store import (
    archive_dir as _archive_dir,
    atomic_write as _atomic_write,
    empty_overrides as _empty_overrides,
    load as _load,
    load_snapshot as _load_snapshot,
    lock_for as _lock_for,
    now_iso as _now_iso,
    overrides_path as _overrides_path,
    reap_archive as _reap_archive,
    validate_multi_task_id as _validate_multi_task_id,
    write_snapshot_file as _write_snapshot_file,
)

logger: logging.Logger = logging.getLogger(__name__)

# Optional broadcast callback. OpenTeam injects a coroutine that fans the
# ``combo_overrides_changed`` event out to connected WS clients; AF itself is
# transport-agnostic so the default is a no-op. Replaces the former direct
# import of ``agent_websocket_routes.broadcast_combo_overrides_changed``.
EmitEvent = Callable[..., Awaitable[None]]


# --- Submission row side-effect (auto-create rows for ready combos) -------


def _ensure_ready_combo_submissions(
    session_dir: Path,
    multi_task_id: str,
    new_combos: list[dict[str, Any]],
) -> list[str]:
    """For each combo with ``applyState == 'ready'`` and no existing row in
    ``hub_<mid>_submissions.json`` matching ``future_combo_id == comboId``,
    append a ``status: 'submitted'`` row. Preserves today's
    ``handleScheduleCombos`` UX (ready combos auto-create runnable rows).

    Combos with ``applyState != 'ready'`` (i.e. ``pending_implementation``
    or ``pending_config``) are skipped — the row would be unrunnable. They
    persist to ``current.json`` so the UI can show them greyed-out in
    Review & Combos and Selection-tab pre-narrowing covers them; when
    prerequisites land, the user re-applies and the row materializes.

    Caller must hold the per-(session, hub) submissions lock.

    Returns the list of newly-created submission ids (for the broadcast
    payload + caller logging).
    """
    # Local import to avoid circulars (submissions_service imports nothing
    # from this module, but keep the boundary clean).
    from agent_foundation.experiment_hub.submissions_service import (
        _atomic_write_submissions,
        _lock_for as submissions_lock_for,
        load_hub_submissions,
    )

    # Filter to combos whose applyState is 'ready' (both Hks implemented
    # AND configStatus ready). Falls back to the legacy configStatus check
    # for any combo lacking the applyState field (backwards-compat for
    # callers/tests that haven't been updated yet).
    def _is_ready(c: dict[str, Any]) -> bool:
        if not isinstance(c, dict):
            return False
        applied_state = c.get("applyState")
        if isinstance(applied_state, str):
            return applied_state == "ready"
        # Legacy fallback when caller hasn't tagged applyState.
        return (c.get("configStatus") or "") == "ready"

    ready: list[dict[str, Any]] = [c for c in new_combos if _is_ready(c)]
    if not ready:
        return []

    created: list[str] = []
    submissions = load_hub_submissions(session_dir, multi_task_id)
    existing_combo_ids: set[str] = set()
    for entry in submissions:
        cfg = entry.get("config") or {}
        fci = cfg.get("future_combo_id") if isinstance(cfg, dict) else None
        if isinstance(fci, str) and fci:
            existing_combo_ids.add(fci)
        # Also dedupe on the canonical comboId field if present.
        ci = entry.get("comboId")
        if isinstance(ci, str) and ci:
            existing_combo_ids.add(ci)

    new_rows: list[dict[str, Any]] = []
    for combo in ready:
        cid = combo.get("comboId") or combo.get("futureComboId") or ""
        if not cid or cid in existing_combo_ids:
            continue
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"
        new_rows.append(
            {
                "id": sub_id,
                "comboKey": combo.get("comboKey"),
                "comboId": cid,
                "selectedItems": combo.get("selectedItems") or [],
                "submittedAt": _now_iso(),
                "status": "submitted",
                "config": {
                    "future_combo_id": cid,
                    "gin_config": combo.get("configPathProposed") or "",
                    "expected_ndcg10_lift": combo.get("expectedNdcg10Lift"),
                    "estimated_compute_hours": combo.get("estimatedComputeHours"),
                },
                "_appliedFromCombo": True,
            }
        )
        created.append(sub_id)
        existing_combo_ids.add(cid)

    if new_rows:
        submissions.extend(new_rows)
        _atomic_write_submissions(session_dir, multi_task_id, submissions)
    # ``submissions_lock_for`` is referenced for clarity; the caller awaits
    # the lock around this whole helper. We only call sync helpers here.
    _ = submissions_lock_for  # silence unused-import linter
    return created


def _mark_submissions_inactive(
    session_dir: Path,
    multi_task_id: str,
    combo_ids_no_longer_active: set[str],
) -> list[str]:
    """Mark hub submissions whose ``config.future_combo_id`` (or ``comboId``)
    is in ``combo_ids_no_longer_active`` with ``inactive_since: <now>``.

    NEVER deletes rows (preserves run history). NEVER touches rows whose
    combo is still active. NEVER touches rows that already have a non-
    null ``inactive_since``.

    Returns the list of submission ids that were marked (for the snapshot
    payload).
    """
    if not combo_ids_no_longer_active:
        return []
    from agent_foundation.experiment_hub.submissions_service import (
        _atomic_write_submissions,
        load_hub_submissions,
    )

    submissions = load_hub_submissions(session_dir, multi_task_id)
    if not submissions:
        return []
    ts = _now_iso()
    marked: list[str] = []
    for entry in submissions:
        if entry.get("inactive_since"):
            continue
        cfg = entry.get("config") or {}
        fci = cfg.get("future_combo_id") if isinstance(cfg, dict) else None
        ci = entry.get("comboId")
        in_inactive = (isinstance(fci, str) and fci in combo_ids_no_longer_active) or (
            isinstance(ci, str) and ci in combo_ids_no_longer_active
        )
        if in_inactive:
            entry["inactive_since"] = ts
            sid = entry.get("id") or entry.get("submission_id") or ""
            if isinstance(sid, str) and sid:
                marked.append(sid)
    if marked:
        _atomic_write_submissions(session_dir, multi_task_id, submissions)
    return marked


# --- applyState derivation ------------------------------------------------


def _derive_apply_state(
    combo: dict[str, Any],
    impls: dict[str, dict[str, Any]],
    setup_state: dict[str, Any],
    session_dir: Path,
) -> None:
    """Plan v3: model-aware ``applyState`` derivation. Mutates ``combo``
    in place with one of four states:

      * ``ready``                — combo can run NOW (Model A flag composition
                                   OR Model B per-combo gin file)
      * ``pending_implementation`` — at least one member H lacks an
                                     implementation (gates first)
      * ``pending_flag_map``     — hub supports ``${ENABLE_FLAGS}``
                                   composition but THIS combo's flag-map
                                   entries are bare/identity placeholders;
                                   the runner would reject at submit time
      * ``pending_config``       — hub does NOT support flag composition
                                   AND no per-combo gin file is ready

    Plus diagnostic fields the UI uses to render specific chips:

      * ``pendingHypotheses``    — gating Hks (only when pending_implementation)
      * ``readyVia`` ∈ {flag_composition, gin_config} — only when ready
      * ``resolvedBindings``     — list of scoped names (only when
                                   readyVia=flag_composition)
      * ``bareBindings``         — Hks with bare/missing flag-map entries
                                   (only when pending_flag_map)
      * ``configPathProposed``   — display field overridden when
                                   readyVia=flag_composition so legacy
                                   chip code shows the truthful path

    Defense-in-depth context: the runner's ``_write_overlay_gin``
    (``submit_job.py:305-356``) is the LAST line of defense — it raises
    ``RuntimeError`` on bare tokens and post-parse-verifies bindings.
    This gate is the FIRST line so the user sees the specific issue
    BEFORE clicking Submit.
    """
    from agent_foundation.experiment_hub.combo_capability import (
        combo_flag_bindings,
        hub_supports_flag_composition,
    )
    from agent_foundation.experiment_hub.hypothesis_implementations import (
        gating_hypotheses_for_combo,
    )

    gating = gating_hypotheses_for_combo(combo, impls)
    if gating:
        combo["applyState"] = "pending_implementation"
        combo["pendingHypotheses"] = gating
        return

    flag_composes = hub_supports_flag_composition(setup_state, session_dir)
    flag_map = (setup_state.get("inputs") or {}).get("hypothesisFlagMap") or {}

    if flag_composes:
        all_scoped, scoped, bare = combo_flag_bindings(combo, flag_map)
        if all_scoped:
            combo["applyState"] = "ready"
            combo["readyVia"] = "flag_composition"
            combo["resolvedBindings"] = scoped
            combo["pendingHypotheses"] = []
            # Cosmetic override so legacy chip code shows the truthful
            # path instead of the misleading "(NEW)" gin filename.
            combo["configPathProposed"] = "(flag-composed: " + ", ".join(scoped) + ")"
            return
        # Hub IS Model A but THIS combo's flag bindings aren't scoped.
        combo["applyState"] = "pending_flag_map"
        combo["bareBindings"] = bare
        combo["pendingHypotheses"] = []
        return

    # Hub is Model B (legacy per-combo gin file).
    if (combo.get("configStatus") or "") == "ready":
        combo["applyState"] = "ready"
        combo["readyVia"] = "gin_config"
    else:
        combo["applyState"] = "pending_config"
    combo["pendingHypotheses"] = []


def _inflate_apply_state(
    combos: list[dict[str, Any]],
    impls: dict[str, dict[str, Any]],
    setup_state: dict[str, Any],
    session_dir: Path,
) -> None:
    """Stamp ``applyState`` + diagnostic fields on combo entries.

    Mutates ``combos`` in place. Per Plan v3, re-derives EVERY combo
    that lacks the new diagnostic fields (``readyVia`` / ``bareBindings``),
    even if it has an existing legacy ``applyState`` — this catches
    stale combos persisted by pre-fix servers and brings them up to the
    new four-state contract on every read.

    Combos that already carry the new fields are left untouched (a
    matching-version write is the authoritative source).
    """
    for c in combos:
        if not isinstance(c, dict):
            continue
        # Up-to-date entries already have one of: readyVia (any ready),
        # bareBindings (pending_flag_map), or pendingHypotheses set
        # (pending_implementation). Plain pending_config entries lack
        # all three but only pre-Plan-v3 servers would emit those AND
        # the legacy "Pending config" rendering is still valid for
        # Model-B hubs — they pass through.
        has_v3_fields = any(k in c for k in ("readyVia", "bareBindings"))
        if has_v3_fields:
            continue
        _derive_apply_state(c, impls, setup_state, session_dir)


# --- Service functions ----------------------------------------------------


async def get_combo_overrides(session_dir: Path, multi_task_id: str) -> dict[str, Any]:
    """Read this hub's combos slice. Returns ``{active_combos: [],
    applied_changes_log: [], generatedAt: null}`` if absent.

    Plan v3: also inflates ``applyState`` to the four-state contract
    (ready / pending_implementation / pending_flag_map / pending_config)
    using hub setup capability + flag-map quality. Surfaces a hub-wide
    ``_hub_flag_map_legacy_warning`` listing any identity-placeholder
    H IDs so the UI banner can render once per hub.
    """
    from agent_foundation.experiment_hub.combo_capability import (
        hub_flag_map_has_identity,
    )
    from agent_foundation.experiment_hub.hypothesis_implementations import (
        derive_hypothesis_implementations,
    )
    from agent_foundation.experiment_hub.setup_store import load_hub_setup

    _validate_multi_task_id(multi_task_id)
    overrides = _load(_overrides_path(session_dir, multi_task_id)) or _empty_overrides()
    active = overrides.get("active_combos") or []
    active_list = list(active) if isinstance(active, list) else []
    # Always load setup so the hub-wide banner can fire even when the
    # active list is empty (e.g., user just landed on the page).
    setup_state = await asyncio.to_thread(load_hub_setup, session_dir, multi_task_id)
    flag_map = (setup_state.get("inputs") or {}).get("hypothesisFlagMap") or {}
    legacy_warning = hub_flag_map_has_identity(flag_map)
    if active_list:
        impls = await asyncio.to_thread(
            derive_hypothesis_implementations, session_dir, multi_task_id
        )
        _inflate_apply_state(active_list, impls, setup_state, session_dir)
    log = overrides.get("applied_changes_log") or []
    return {
        "multi_task_id": multi_task_id,
        "active_combos": active_list,
        "applied_changes_log": list(log) if isinstance(log, list) else [],
        "generatedAt": overrides.get("generatedAt"),
        # Hub-wide identity-map warning consumed by MultiChoiceComboView's
        # dismissible banner (Plan v3 Change 3c). Empty list = no banner.
        "_hub_flag_map_legacy_warning": legacy_warning,
    }


async def apply_combo_overrides(
    session_id: str,
    session_dir: Path,
    multi_task_id: str,
    body: dict[str, Any],
    emit_event: EmitEvent | None = None,
) -> dict[str, Any]:
    """Replace the active combos for ``multi_task_id`` with ``body.combos``.

    Body shape:
      ``{"combos": [combo_obj, ...], "source_archive_id"?: str, "applied_by"?: str}``

    Per-combo state tagging (Phase C — replaces former HTTP 422 preflight):
    each combo is stamped with an ``applyState`` field:
      • ``"ready"`` — all Hks implemented AND ``configStatus == "ready"``
      • ``"pending_implementation"`` — at least one Hk lacks an implementation
      • ``"pending_config"`` — Hks all implemented but ``configStatus != "ready"``
    All combos are persisted regardless of state; the UI gates RUNNABILITY
    downstream (Review & Combos greys out pending combos with a label).
    The user's intent (combos-of-interest) is captured even before
    prerequisites land — Selection-tab pre-narrowing covers all combos.

    Side effect: for each combo with ``applyState == "ready"`` and no
    existing row in ``hub_<mid>_submissions.json`` matching
    ``future_combo_id == comboId``, append a ``status: "submitted"`` row.
    Pending combos do NOT auto-create rows (the row would be unrunnable);
    when implementations/configs land, the combo's applyState flips on
    next apply (or via the deferred auto-promotion routine).

    Snapshot: full pre-apply state is written to a separate file at
    ``hub/<mid>/combos/_archive/<ts>_apply_<n>.json``; the log entry
    holds only a ``snapshotRef`` pointing at it (bounds current.json
    growth). Reaper trims to last ``_ARCHIVE_KEEP_LAST_N``.

    After write, emits ``combo_overrides_changed`` so other tabs re-fetch.
    """
    from agent_foundation.experiment_hub.hypothesis_implementations import (
        derive_hypothesis_implementations,
    )
    from agent_foundation.experiment_hub.setup_store import load_hub_setup

    _validate_multi_task_id(multi_task_id)
    path = _overrides_path(session_dir, multi_task_id)
    archive_path = _archive_dir(session_dir, multi_task_id)

    incoming = body.get("combos") if isinstance(body, dict) else None
    if not isinstance(incoming, list):
        raise ValueError("body.combos must be a list of combo objects")
    # Defensive shape check: every entry must be a dict with at least a
    # comboId. We don't deep-validate the rest; the frontend supplies the
    # full shape from the learnings doc and downstream consumers tolerate
    # missing optional fields.
    for i, c in enumerate(incoming):
        if not isinstance(c, dict):
            raise ValueError(f"body.combos[{i}] is not an object")
        if not c.get("comboId"):
            raise ValueError(f"body.combos[{i}] missing comboId")

    # Plan v3: tag each combo with applyState via the model-aware
    # _derive_apply_state helper. Always re-derives — the persisted
    # value is overwritten on every apply. Loads hub setup so capability
    # + flag-map quality can be inspected. Replaces the former
    # one-dimensional gate that only checked configStatus.
    impls = await asyncio.to_thread(
        derive_hypothesis_implementations, session_dir, multi_task_id
    )
    setup_state = await asyncio.to_thread(load_hub_setup, session_dir, multi_task_id)
    for c in incoming:
        _derive_apply_state(c, impls, setup_state, session_dir)

    source_archive_id = (
        body.get("source_archive_id") if isinstance(body, dict) else None
    )
    applied_by = body.get("applied_by") if isinstance(body, dict) else None

    async with _lock_for(session_id, multi_task_id):
        existing = _load(path) or _empty_overrides()
        prior_combos = list(existing.get("active_combos") or [])
        new_combos = list(incoming)

        prior_combo_ids: set[str] = {
            c.get("comboId")
            for c in prior_combos
            if isinstance(c, dict) and isinstance(c.get("comboId"), str)
        }
        new_combo_ids: set[str] = {
            c.get("comboId")
            for c in new_combos
            if isinstance(c, dict) and isinstance(c.get("comboId"), str)
        }
        no_longer_active = prior_combo_ids - new_combo_ids

        # Submissions side effects (under per-(session, hub) submissions
        # lock — different lock space from combo_overrides; ordering is
        # always combo_overrides → submissions, so no deadlock).
        from agent_foundation.experiment_hub.submissions_service import (
            _lock_for as submissions_lock_for,
        )

        async with submissions_lock_for(session_id, multi_task_id):
            created_sub_ids = await asyncio.to_thread(
                _ensure_ready_combo_submissions,
                session_dir,
                multi_task_id,
                new_combos,
            )
            marked_inactive_ids = await asyncio.to_thread(
                _mark_submissions_inactive,
                session_dir,
                multi_task_id,
                no_longer_active,
            )

        applied_at = _now_iso()

        # Write snapshot file FIRST (forensic; failure is logged but
        # doesn't fail the apply — log entry will carry snapshotRef=null
        # and revert falls back to the inline priorActiveCombos field).
        snapshot: dict[str, Any] = {
            "appliedAt": applied_at,
            "action": "apply",
            "multi_task_id": multi_task_id,
            "priorActiveCombos": prior_combos,
            "newActiveCombos": new_combos,
            "appendedSubmissionIds": created_sub_ids,
            "markedInactiveSubmissionIds": marked_inactive_ids,
            "source_archive_id": source_archive_id,
            "applied_by": applied_by or "ui",
        }
        snapshot_ref: str | None = None
        try:
            snapshot_ref = await asyncio.to_thread(
                _write_snapshot_file, archive_path, snapshot
            )
        except OSError as e:
            logger.warning(
                "Failed to write combos snapshot for session=%s mid=%s: %s",
                session_id,
                multi_task_id,
                e,
            )

        # Update current.json: bump active_combos, append log entry.
        existing["active_combos"] = new_combos
        log_entry: dict[str, Any] = {
            "appliedAt": applied_at,
            "action": "apply_combos",
            "multi_task_id": multi_task_id,
            "applied_combo_ids": [c.get("comboId") for c in new_combos],
            "snapshotRef": snapshot_ref,
            "source_archive_id": source_archive_id,
            "applied_by": applied_by or "ui",
            # Inline fallback — keeps revert working if the snapshot file
            # write failed above. Small (just the prior combo list).
            "priorActiveCombosInline": prior_combos,
        }
        existing.setdefault("applied_changes_log", []).append(log_entry)
        existing["generatedAt"] = applied_at
        await asyncio.to_thread(_atomic_write, path, existing)

        # Reap old snapshots (best-effort; reaper failure is non-fatal).
        try:
            await asyncio.to_thread(_reap_archive, archive_path)
        except OSError as e:
            logger.warning(
                "Combos archive reaper failed for session=%s mid=%s: %s",
                session_id,
                multi_task_id,
                e,
            )

    # Best-effort broadcast — failures here MUST NOT fail the request.
    if emit_event is not None:
        try:
            await emit_event(
                session_id=session_id,
                multi_task_id=multi_task_id,
                action="apply",
                applied_combo_ids=[c.get("comboId") for c in new_combos],
                created_submission_ids=created_sub_ids,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning(
                "combo_overrides_changed broadcast failed for session=%s mid=%s: %s",
                session_id,
                multi_task_id,
                e,
            )

    return {
        "ok": True,
        "multi_task_id": multi_task_id,
        "active_combos": new_combos,
        "created_submission_ids": created_sub_ids,
        "marked_inactive_submission_ids": marked_inactive_ids,
        "snapshot_ref": snapshot_ref,
        "log_entry": log_entry,
    }


async def revert_last_combo_apply(
    session_id: str,
    session_dir: Path,
    multi_task_id: str,
    emit_event: EmitEvent | None = None,
) -> dict[str, Any]:
    """Undo the most recent ``apply_combos`` entry for this hub.

    Loads the snapshot file referenced by the log entry's ``snapshotRef``
    to restore the prior active_combos faithfully. If the snapshot file
    is missing (reaped or write-failed), falls back to the inline
    ``priorActiveCombosInline`` field on the log entry.

    Submissions are NOT deleted — they may have started running
    (preserving run history is an invariant). Combos that were auto-
    added by the reverted apply are marked ``inactive_since`` so the
    Review & Combos tab moves them to the "Historical combos" section.

    Itself reversible: revert can be reverted (the next apply will
    snapshot the post-revert state).
    """
    _validate_multi_task_id(multi_task_id)
    path = _overrides_path(session_dir, multi_task_id)
    archive_path = _archive_dir(session_dir, multi_task_id)

    async with _lock_for(session_id, multi_task_id):
        existing = _load(path)
        if not existing:
            raise KeyError("no overrides on disk")
        log = existing.get("applied_changes_log") or []
        # Find the most recent apply_combos entry (file is per-hub already,
        # so no need to filter by multi_task_id).
        last_idx: int | None = None
        for i in range(len(log) - 1, -1, -1):
            entry = log[i]
            if isinstance(entry, dict) and entry.get("action") == "apply_combos":
                last_idx = i
                break
        if last_idx is None:
            raise KeyError(f"no apply history to revert for hub {multi_task_id}")
        last = log[last_idx]
        snapshot_ref = last.get("snapshotRef")

        # Prefer the snapshot file; fall back to inline if missing/reaped.
        snap = (
            await asyncio.to_thread(_load_snapshot, archive_path, snapshot_ref)
            if isinstance(snapshot_ref, str)
            else None
        )
        if snap is not None:
            prior_combos = list(snap.get("priorActiveCombos") or [])
            snapshot_source = "snapshot_file"
        else:
            prior_combos = list(last.get("priorActiveCombosInline") or [])
            snapshot_source = "inline_fallback"
            if isinstance(snapshot_ref, str) and snapshot_ref:
                logger.warning(
                    "Revert falling back to inline (snapshot file missing): %s",
                    snapshot_ref,
                )

        current_combos = list(existing.get("active_combos") or [])
        existing["active_combos"] = prior_combos

        log.append(
            {
                "appliedAt": _now_iso(),
                "action": "revert_combos",
                "multi_task_id": multi_task_id,
                "reverted_apply_at": last.get("appliedAt"),
                "reverted_snapshot_ref": snapshot_ref,
                "snapshot_source": snapshot_source,
                "restored_combo_ids": [
                    c.get("comboId") for c in prior_combos if isinstance(c, dict)
                ],
            }
        )
        existing["applied_changes_log"] = log
        existing["generatedAt"] = _now_iso()
        await asyncio.to_thread(_atomic_write, path, existing)

        # Mark submissions for combos that just became inactive.
        from agent_foundation.experiment_hub.submissions_service import (
            _lock_for as submissions_lock_for,
        )

        current_combo_ids: set[str] = {
            c.get("comboId")
            for c in current_combos
            if isinstance(c, dict) and isinstance(c.get("comboId"), str)
        }
        prior_combo_ids: set[str] = {
            c.get("comboId")
            for c in prior_combos
            if isinstance(c, dict) and isinstance(c.get("comboId"), str)
        }
        no_longer_active = current_combo_ids - prior_combo_ids
        async with submissions_lock_for(session_id, multi_task_id):
            await asyncio.to_thread(
                _mark_submissions_inactive,
                session_dir,
                multi_task_id,
                no_longer_active,
            )

    # Best-effort broadcast.
    if emit_event is not None:
        try:
            await emit_event(
                session_id=session_id,
                multi_task_id=multi_task_id,
                action="revert",
                applied_combo_ids=[
                    c.get("comboId") for c in prior_combos if isinstance(c, dict)
                ],
                created_submission_ids=[],
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning(
                "combo_overrides_changed broadcast failed for session=%s mid=%s: %s",
                session_id,
                multi_task_id,
                e,
            )

    return {
        "ok": True,
        "multi_task_id": multi_task_id,
        "active_combos": prior_combos,
        "snapshot_source": snapshot_source,
    }

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Pure-function derivation of "is hypothesis Hk implemented?" from existing
on-disk state. NO new persistence — this module reads the canonical
``hub_<mid>_implementations.json`` (written by the Implement Hypothesis
workflow) plus optionally the ``hub_<mid>_submissions.json`` runs as a
secondary signal.

A hypothesis is considered implemented when ANY of the following holds:

  1. ``hub_<mid>_implementations.json`` carries a row with status='completed'
     whose ``hypothesis_ids`` includes Hk. This is the strong signal —
     the Implement workflow produced a usable artifact for the hypothesis
     (see :mod:`agent_foundation.experiment_hub.implementations_store`).

  2. (Secondary) at least one row in ``hub_<mid>_submissions.json`` carries
     Hk in ``selectedItems`` AND has a terminal status (``completed`` /
     ``analyzed``) AND ``config.gin_config`` is non-empty and not the
     literal placeholder ``"(NEW)"``. This catches manual paths where a
     run was launched without going through the Implement workflow.

The result is consumed by:

  * The Apply Combos preflight gate (combo_overrides routes) — combos
    referencing un-implemented Hks are blocked with an error.
  * The Accumulated Learnings side panel — combos with un-implemented Hks
    are rendered with a disabled checkbox and a "Pending implementation"
    chip naming the gating Hks.

No caching: derivation is O(submissions + implementations) per call, both
of which are small JSON arrays per session. If perf becomes a concern,
add a per-session in-memory cache invalidated by the ``submission_state``
WS event.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

# Submission status values that count as "the run actually executed".
# Mirrors the terminal_statuses set in the agent server's experiment-combos
# bridge (used by the aggregator-input collector). Kept in sync via the test
# suite.
_TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "analyzed"})

# Literal placeholder the LLM emits in ``configPathProposed`` for combos
# whose gin doesn't yet exist (see the learnings generator — emits e.g.
# ``"hstu-h1-h17.gin (NEW)"``). Treat any path containing this token as
# "not a real implementation".
_PROPOSED_GIN_TOKEN: str = "(NEW)"


def _is_real_gin_path(gin: str | None) -> bool:
    """A submission's ``config.gin_config`` is a usable implementation iff
    it's a non-empty string that doesn't carry the LLM's ``(NEW)``
    placeholder. We DO NOT stat the file here — the placeholder check is
    sufficient to distinguish "Implement workflow produced this" from
    "LLM proposed but not yet generated", and avoids costly disk IO on
    paths that may live on slow filers.
    """
    if not gin or not isinstance(gin, str):
        return False
    return _PROPOSED_GIN_TOKEN not in gin


def derive_hypothesis_implementations(
    session_dir: Path,
    multi_task_id: str,
) -> dict[str, dict[str, Any]]:
    """Return ``{Hk: {implemented: bool, basis: str, batch_id?, workspace_path?, gin_config?}}``.

    Iterates the implementations sidecar first (strong signal), then
    augments with submission scans (secondary signal). Each Hk that
    appears under EITHER signal is marked implemented with the basis
    field naming the source. Hks that appear in NEITHER are returned
    with ``{implemented: False, basis: "no_evidence"}`` only when the
    caller explicitly enumerates them — this function does NOT itself
    know the universe of Hks; callers pass that universe via
    ``derive_for_hypothesis_ids`` below if they want exhaustive output.
    """
    # Local imports to avoid pulling heavier service modules into anyone who
    # only wants the derivation logic (e.g., a CLI debug tool).
    from agent_foundation.experiment_hub.implementations_store import (
        load_hub_implementations,
    )
    from agent_foundation.experiment_hub.submissions_service import load_hub_submissions

    out: dict[str, dict[str, Any]] = {}

    # Strong signal — Implement workflow output.
    impls = load_hub_implementations(session_dir, multi_task_id)
    for row in impls:
        if (row.get("status") or "") != "completed":
            continue
        hids = row.get("hypothesis_ids") or []
        if not isinstance(hids, list):
            continue
        for h in hids:
            if not isinstance(h, str) or not h:
                continue
            # First-write-wins: if multiple impl batches cover the same H,
            # keep the earliest (most stable reference). The Implement
            # workflow is append-only so first wins is deterministic.
            if h not in out:
                out[h] = {
                    "implemented": True,
                    "basis": "implementation_batch_completed",
                    "batch_id": row.get("batch_id"),
                    "workspace_path": row.get("workspace_path"),
                }

    # Secondary signal — submission with real gin.
    submissions = load_hub_submissions(session_dir, multi_task_id)
    for entry in submissions:
        status = entry.get("status") or ""
        if status not in _TERMINAL_RUN_STATUSES:
            continue
        cfg = entry.get("config") or {}
        gin = cfg.get("gin_config") if isinstance(cfg, dict) else None
        if not _is_real_gin_path(gin):
            continue
        selected = entry.get("selectedItems") or entry.get("hypothesis_ids") or []
        if not isinstance(selected, list):
            continue
        for h in selected:
            if not isinstance(h, str) or not h:
                continue
            if h not in out:
                out[h] = {
                    "implemented": True,
                    "basis": "submission_terminal_with_gin",
                    "submission_id": entry.get("id") or entry.get("submission_id"),
                    "gin_config": gin,
                }
    return out


def derive_for_hypothesis_ids(
    session_dir: Path,
    multi_task_id: str,
    hypothesis_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Same as :func:`derive_hypothesis_implementations` but ALSO returns an
    explicit ``{implemented: False, basis: "no_evidence"}`` entry for every
    Hk in ``hypothesis_ids`` that has no implementation evidence.

    Use this when the caller wants exhaustive coverage (e.g., the side
    panel rendering all combo Hks with status badges).
    """
    derived = derive_hypothesis_implementations(session_dir, multi_task_id)
    out: dict[str, dict[str, Any]] = {}
    for h in hypothesis_ids:
        if not isinstance(h, str) or not h:
            continue
        out[h] = derived.get(h) or {
            "implemented": False,
            "basis": "no_evidence",
        }
    return out


def gating_hypotheses_for_combo(
    combo: dict[str, Any],
    implementations: dict[str, dict[str, Any]],
) -> list[str]:
    """Return the list of Hks in ``combo.selectedItems`` that are NOT
    implemented per the supplied ``implementations`` map.

    Used by the Apply Combos preflight gate to build the gating-Hks
    response payload per-combo.
    """
    selected = combo.get("selectedItems") or []
    if not isinstance(selected, list):
        return []
    return [
        h
        for h in selected
        if isinstance(h, str)
        and h
        and not (implementations.get(h) or {}).get("implemented")
    ]

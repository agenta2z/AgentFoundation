# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""``JsonSidecarStore``-backed wrapper for a hub's submissions file.

The faithful, full-fidelity submissions logic (reconcile, baseline, run/cancel,
PATCH allow-listing, the WS-event appliers) lives in :mod:`submissions_service`,
ported verbatim from RankEvolve's single-writer ``hub_submissions_routes``
(per-(session, mid) ``asyncio.Lock`` + tempfile+os.replace). This module layers
the shared :class:`JsonSidecarStore` envelope on top so the Experiment-Hub
stores have a uniform store surface — ``read`` / ``write`` / ``update`` with the
``allowed_fields`` mutable-field allow-list — without duplicating the service
logic.

``_MUTABLE_SUBMISSION_FIELDS`` is re-exported here from ``submissions_service``
so callers that only need the allow-list don't import the heavier service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_foundation.experiment_hub.submissions_service import (
    _MUTABLE_SUBMISSION_FIELDS,
    load_hub_submissions,
)
from agent_foundation.server.dashboard.dashboard_store import JsonSidecarStore

# Re-export so ``from ...submissions_store import _MUTABLE_SUBMISSION_FIELDS``
# works for callers that don't want the whole service module.
__all__ = (
    "SubmissionsStore",
    "_MUTABLE_SUBMISSION_FIELDS",
    "load_hub_submissions",
)


class SubmissionsStore(JsonSidecarStore):
    """Atomic JSON sidecar for ``hub_<mid>_submissions.json``.

    Envelope shape: ``{"multi_task_id": <mid>, "submissions": [ ... ]}``.
    Per-row PATCHes go through :meth:`patch_row`, which enforces the
    ``_MUTABLE_SUBMISSION_FIELDS`` allow-list — the same single-writer
    mutable-field invariant the service path enforces.
    """

    def __init__(self, session_dir: str | Path, multi_task_id: str) -> None:
        self._multi_task_id: str = multi_task_id
        path = Path(session_dir) / f"hub_{multi_task_id}_submissions.json"
        super().__init__(
            path,
            default={"multi_task_id": multi_task_id, "submissions": []},
        )

    @property
    def multi_task_id(self) -> str:
        return self._multi_task_id

    def rows(self) -> list[dict[str, Any]]:
        """Return the current submissions array (a copy of the on-disk list)."""
        data = self.read()
        rows = data.get("submissions", [])
        return list(rows) if isinstance(rows, list) else []

    async def patch_row(
        self, submission_id: str, patch: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply ``patch`` to the row identified by ``id``/``submission_id``.

        Only keys in ``_MUTABLE_SUBMISSION_FIELDS`` are applied (others are
        ignored — the immutable-identity guard). Returns the merged row, or
        ``None`` if no row matched.
        """
        matched: dict[str, Any] | None = None

        def _mutate(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal matched
            rows = state.get("submissions")
            if not isinstance(rows, list):
                rows = []
                state["submissions"] = rows
            for entry in rows:
                if (
                    entry.get("id") == submission_id
                    or entry.get("submission_id") == submission_id
                ):
                    for k, v in patch.items():
                        if k in _MUTABLE_SUBMISSION_FIELDS:
                            entry[k] = v
                    matched = entry
                    break
            return state

        await self.update(_mutate)
        return matched

    async def append_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Append a new submission row. Returns the appended row."""

        def _mutate(state: dict[str, Any]) -> dict[str, Any]:
            rows = state.get("submissions")
            if not isinstance(rows, list):
                rows = []
            rows.append(dict(row))
            state["submissions"] = rows
            return state

        await self.update(_mutate)
        return dict(row)

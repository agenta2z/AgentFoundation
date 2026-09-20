# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

# UNAUTHENTICATED — single-user demo logic. Gate behind Tier-1 auth before deploy.

"""Service logic exposing per-hypothesis implementation status for a hub.

Ported from RankEvolve ``hypothesis_implementations_routes.py``. FastAPI
stripped: the former endpoint is a plain service function taking an explicit
``session_dir: Path``. The thin REST router lives in OpenTeam and calls this
function.

Wraps the pure-function derivation in
:mod:`agent_foundation.experiment_hub.hypothesis_implementations`. The Apply
Combos preflight gate uses the same derivation, so the UI's disabled state and
the server's gate always agree.

When ``ids`` is supplied (comma-separated), the response includes explicit
``{implemented: false, basis: "no_evidence"}`` entries for any listed Hk
lacking evidence — useful when the side panel wants to render a status badge
for every Hk in a combo's ``selectedItems`` regardless of whether it appears
under any submission/implementation row.

Former endpoint → service function:
  GET .../{mid}/hypothesis_implementations → get_hypothesis_implementations
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

_MULTI_TASK_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_HYPOTHESIS_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _validate_multi_task_id(multi_task_id: str) -> None:
    if not _MULTI_TASK_ID_PATTERN.match(multi_task_id):
        raise ValueError("Invalid multi_task_id")


def _parse_ids(ids: str | None) -> list[str]:
    if not ids:
        return []
    parsed: list[str] = []
    for raw in ids.split(","):
        h = raw.strip()
        if not h:
            continue
        if not _HYPOTHESIS_ID_PATTERN.match(h):
            raise ValueError(f"Invalid hypothesis id: {h!r}")
        parsed.append(h)
    return parsed


async def get_hypothesis_implementations(
    session_dir: Path,
    multi_task_id: str,
    ids: str | None = None,
) -> dict[str, Any]:
    """Return ``{multi_task_id, implementations: {Hk: {implemented, basis, ...}}}``.

    Implemented as a thin wrapper over the pure derivation. Cheap (one
    JSON load + one JSON load per call); uncached. If perf becomes a
    concern, add a per-session in-memory cache invalidated by the
    ``submission_state`` event.
    """
    from agent_foundation.experiment_hub.hypothesis_implementations import (
        derive_for_hypothesis_ids,
        derive_hypothesis_implementations,
    )

    _validate_multi_task_id(multi_task_id)
    parsed_ids = _parse_ids(ids)

    if parsed_ids:
        result = await asyncio.to_thread(
            derive_for_hypothesis_ids,
            session_dir,
            multi_task_id,
            parsed_ids,
        )
    else:
        result = await asyncio.to_thread(
            derive_hypothesis_implementations,
            session_dir,
            multi_task_id,
        )
    return {
        "multi_task_id": multi_task_id,
        "implementations": result,
    }

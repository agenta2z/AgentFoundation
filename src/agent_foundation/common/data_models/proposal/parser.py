"""Proposal parser — read/write ``ProposalIndex`` from workspace artifacts.

Three strategies in priority order:
    A. Read ``outputs/proposals.json`` sidecar (fast path, AF-native).
    B. Extract ``proposal_index`` JSON fence from markdown (reuses
       ``_extract_json_block`` from ``flow_parsers``).
    C. Regex-parse a Priority Ranking Table from markdown (last resort,
       recovers ``id``, ``rank``, ``title`` only).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import Proposal, ProposalGroup, ProposalIndex

_logger = logging.getLogger(__name__)

_SIDECAR_NAME = "proposals.json"

_FENCE_RE = re.compile(
    r"```json\s+proposal_index\b[^\n]*\n([\s\S]*?)\n\s*```"
)

_TABLE_ROW_RE = re.compile(
    r"^\s*\|\s*(\d+)\s*\|\s*([\w-]+)\s*\|\s*(.+?)\s*\|",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_proposals(workspace: Path) -> ProposalIndex | None:
    """Try Strategy A → B → C. Returns ``None`` only if all three fail."""
    result = _strategy_a(workspace)
    if result is not None:
        return result

    md_path = _find_markdown(workspace)
    if md_path is not None:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        result = _strategy_b(text)
        if result is not None:
            return result
        result = _strategy_c(text)
        if result is not None:
            return result

    return None


def parse_proposal_file(path: Path) -> ProposalIndex | None:
    """Strategy A: load ``proposals.json`` sidecar directly."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return ProposalIndex.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _logger.warning("Failed to parse %s: %s", path, exc)
        return None


def parse_proposal_index_from_text(text: str) -> ProposalIndex | None:
    """Extract ``ProposalIndex`` from text containing a JSON fence."""
    return _strategy_b(text)


def write_proposal_index(path: Path, index: ProposalIndex) -> None:
    """Atomic write: tmp → fsync → rename. Never produces partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = index.to_dict()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _strategy_a(workspace: Path) -> ProposalIndex | None:
    """Read sidecar ``outputs/proposals.json``."""
    for candidate in (
        workspace / "outputs" / _SIDECAR_NAME,
        workspace / _SIDECAR_NAME,
    ):
        result = parse_proposal_file(candidate)
        if result is not None:
            return result
    return None


def _strategy_b(text: str) -> ProposalIndex | None:
    """Extract ``proposal_index`` JSON fence from markdown."""
    m = _FENCE_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if not isinstance(data, dict):
            return None
        return ProposalIndex.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _logger.warning("Malformed proposal_index fence: %s", exc)
        return None


def _strategy_c(text: str) -> ProposalIndex | None:
    """Regex-parse a Priority Ranking Table. Recovers ``id``, ``rank``, ``title``."""
    rows = _TABLE_ROW_RE.findall(text)
    if not rows:
        return None
    proposals: list[Proposal] = []
    for rank_str, pid, title in rows:
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        proposals.append(Proposal(id=pid.strip(), rank=rank, title=title.strip()))
    if not proposals:
        return None
    return ProposalIndex(
        version="1",
        total_count=len(proposals),
        groups=[ProposalGroup(phase=1, label="Recovered", proposals=proposals)],
        warnings=["parsed-from-ranking-table-only"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_markdown(workspace: Path) -> Path | None:
    """Locate the aggregator's markdown output."""
    for name in ("unified_plan.md", "output.md", "final_result.md"):
        for subdir in ("outputs", "outputs/final_deliverables", "."):
            candidate = workspace / subdir / name
            if candidate.is_file():
                return candidate
    return None


def make_empty_index(
    source_workspace: str = "", warnings: list[str] | None = None,
) -> ProposalIndex:
    """Create an empty index with metadata (used when extraction fails)."""
    return ProposalIndex(
        version="1",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_workspace=source_workspace,
        total_count=0,
        warnings=warnings or [],
    )

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Parser for extracting structured proposal data from unified plan output.

Supports two strategies:
  A (primary): Extract from ```json proposal_summary``` code fence
  B (fallback): Parse the markdown Consolidated Proposal List + Implementation
    Roadmap + Priority Ranking Table from the aggregator's unified_plan.md

Both read from the aggregator's unified_plan.md — NOT the raw
final_result.md (which is a 7000+ line concatenation of all worker output).

Ported faithfully from RankEvolve's ``agentic_foundation.common.ui.proposal_parser``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agent_foundation.experiment_hub.proposal_models import (
    Batch,
    ProposalPhase,
    ProposalSelectionData,
    StructuredProposal,
)

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Strategy A: JSON code fence
_JSON_FENCE_RE = re.compile(
    r"```json\s+proposal_summary\s*\n(.*?)\n```",
    re.DOTALL,
)

# Priority Ranking Table rows (used for rank/phase assignment)
_TABLE_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|"  # Rank
    r"\s*(H\d+)\s*\|"  # ID
    r"\s*([^|]+?)\s*\|"  # Name
    r"\s*([^|]*?)\s*\|"  # Source Workers
    r"\s*([^|]*?)\s*\|"  # Impact
    r"\s*([^|]*?)\s*\|"  # Probability
    r"\s*([^|]*?)\s*\|"  # Complexity
    r"\s*(\d+)\s*\|"  # Phase
    r"\s*([^|]*?)\s*\|",  # Notes
    re.MULTILINE,
)

# Consolidated Proposal List: theme headers
_THEME_RE = re.compile(r"^### Theme \d+:\s*(.+)$", re.MULTILINE)

# Consolidated Proposal List: hypothesis headers
_HYPOTHESIS_RE = re.compile(r"^#### (H\d+):\s*(.+)$", re.MULTILINE)

# Consolidated Proposal List: attribute lines (uses [^*]+ to match "Cross-refs" hyphen)
_ATTR_RE = re.compile(r"^- \*\*([^*]+)\*\*:\s*(.+)$", re.MULTILINE)

# Implementation Roadmap: phase headers
_ROADMAP_PHASE_RE = re.compile(
    r"^### Phase (\d+):\s*(.+?)(?:\s*\((.+?)\))?\s*$", re.MULTILINE
)

# Implementation Roadmap: batch headers (two variants)
_BATCH_HEADER_RE = re.compile(
    r"^#### Batch (\w+)\s*\((.+?)(?:\s*(?:—|--)\s*(.+?))?\)\s*:?\s*$",
    re.MULTILINE,
)

# Batch table: extract hypothesis ID from first column
_BATCH_TABLE_ID_RE = re.compile(r"^\|\s*(H\d+)\s*\|", re.MULTILINE)


# ---------------------------------------------------------------------------
# Strategy A: JSON code fence
# ---------------------------------------------------------------------------


def _parse_json_strategy(content: str) -> ProposalSelectionData | None:
    """Extract from ```json proposal_summary``` code fence."""
    match = _JSON_FENCE_RE.search(content)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return ProposalSelectionData.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("proposal_summary JSON parse failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Strategy B: Markdown two-section extraction
# ---------------------------------------------------------------------------

# Map of attribute names → StructuredProposal field names
_ATTR_FIELD_MAP = {
    "Problem": "problem",
    "Approach": "approach",
    "Impact": "impact",
    "Probability": "probability",
    "Complexity": "complexity",
    "Cross-refs": "cross_refs",
    "Source": "_source_raw",  # handled specially
}


def _parse_consolidated_proposals(content: str) -> dict[str, StructuredProposal]:
    """Parse ## Consolidated Proposal List into a dict keyed by H-ID."""
    # Find section boundaries
    section_start = content.find("## Consolidated Proposal List")
    if section_start < 0:
        return {}
    # Find the next ## section
    next_section = content.find("\n## ", section_start + 10)
    section = (
        content[section_start:next_section]
        if next_section > 0
        else content[section_start:]
    )

    proposals: dict[str, StructuredProposal] = {}
    current_theme = ""

    # Split into lines for sequential processing
    lines = section.split("\n")
    current_id = ""
    current_title = ""
    attrs: dict[str, str] = {}

    def _flush():
        """Save the current hypothesis if we have one."""
        nonlocal current_id, current_title, attrs
        if not current_id:
            return
        source_raw = attrs.get("_source_raw", "")
        source_workers = [
            s.strip().split("-")[0] for s in source_raw.split(",") if s.strip()
        ]
        proposals[current_id] = StructuredProposal(
            id=current_id,
            rank=0,  # filled later from ranking table
            title=current_title,
            theme=current_theme,
            source_workers=source_workers,
            impact=attrs.get("impact", ""),
            probability=attrs.get("probability", ""),
            complexity=attrs.get("complexity", ""),
            problem=attrs.get("problem", ""),
            approach=attrs.get("approach", ""),
            cross_refs=attrs.get("cross_refs", ""),
            one_line_summary=current_title,
        )
        current_id = ""
        attrs = {}

    for line in lines:
        # Theme header
        theme_m = _THEME_RE.match(line)
        if theme_m:
            _flush()
            current_theme = theme_m.group(1).strip()
            continue

        # Hypothesis header
        hyp_m = _HYPOTHESIS_RE.match(line)
        if hyp_m:
            _flush()
            current_id = hyp_m.group(1)
            current_title = hyp_m.group(2).strip()
            continue

        # Attribute line
        if current_id:
            attr_m = _ATTR_RE.match(line)
            if attr_m:
                attr_name = attr_m.group(1).strip()
                attr_val = attr_m.group(2).strip()
                field_name = _ATTR_FIELD_MAP.get(attr_name)
                if field_name:
                    attrs[field_name] = attr_val

    _flush()  # last hypothesis
    return proposals


def _parse_implementation_roadmap(content: str) -> list[ProposalPhase]:
    """Parse ## Implementation Roadmap into phases with batches."""
    section_start = content.find("## Implementation Roadmap")
    if section_start < 0:
        return []
    next_section = content.find("\n## ", section_start + 10)
    section = (
        content[section_start:next_section]
        if next_section > 0
        else content[section_start:]
    )

    phases: list[ProposalPhase] = []
    current_phase: ProposalPhase | None = None

    # Find all phase headers
    for phase_m in _ROADMAP_PHASE_RE.finditer(section):
        phase_num = int(phase_m.group(1))
        label = phase_m.group(2).strip()
        timeline = (phase_m.group(3) or "").strip()
        current_phase = ProposalPhase(
            phase=phase_num,
            label=label,
            description=f"{label} ({timeline})" if timeline else label,
        )
        phases.append(current_phase)

    if not phases:
        return []

    # Find all batches and assign to phases
    for batch_m in _BATCH_HEADER_RE.finditer(section):
        batch_id = batch_m.group(1)  # "1A", "2B", "3A"
        timeline = batch_m.group(2).strip()
        label = (batch_m.group(3) or "").strip()

        # Find hypothesis IDs in the batch's table
        batch_start = batch_m.end()
        # Find the next batch or phase or section heading
        next_heading = re.search(r"\n(?:####|###|##) ", section[batch_start:])
        batch_section = (
            section[batch_start : batch_start + next_heading.start()]
            if next_heading
            else section[batch_start:]
        )
        h_ids = _BATCH_TABLE_ID_RE.findall(batch_section)

        batch = Batch(
            id=batch_id,
            label=label,
            timeline=timeline,
            hypothesis_ids=h_ids,
        )

        # Assign to the correct phase based on batch ID prefix
        phase_num = int(batch_id[0])
        for p in phases:
            if p.phase == phase_num:
                p.batches.append(batch)
                break

    return phases


def _merge_roadmap_and_proposals(
    phases: list[ProposalPhase],
    proposals_by_id: dict[str, StructuredProposal],
    rank_map: dict[str, int],
) -> None:
    """Map hypothesis IDs from roadmap batches to detailed proposal data.

    Populates each phase's proposals list by looking up IDs from batches
    in the proposals_by_id dict. Also sets ranks from the ranking table.
    """
    assigned_ids: set[str] = set()

    for phase in phases:
        phase_proposals: list[StructuredProposal] = []
        for batch in phase.batches:
            for h_id in batch.hypothesis_ids:
                if h_id in proposals_by_id and h_id not in assigned_ids:
                    proposal = proposals_by_id[h_id]
                    proposal.rank = rank_map.get(h_id, 0)
                    phase_proposals.append(proposal)
                    assigned_ids.add(h_id)
        # Sort by rank within phase
        phase_proposals.sort(key=lambda p: p.rank if p.rank > 0 else 9999)
        phase.proposals = phase_proposals

    # Any proposals not in the roadmap → append to last phase
    unassigned = [p for h_id, p in proposals_by_id.items() if h_id not in assigned_ids]
    if unassigned and phases:
        for p in unassigned:
            p.rank = rank_map.get(p.id, 0)
        phases[-1].proposals.extend(sorted(unassigned, key=lambda p: p.rank or 9999))


def _parse_ranking_table(content: str) -> dict[str, tuple[int, int, str]]:
    """Parse Priority Ranking Table → {H-ID: (rank, phase, notes)}."""
    result: dict[str, tuple[int, int, str]] = {}
    for row in _TABLE_ROW_RE.findall(content):
        rank_s, h_id, name, sources, impact, prob, complexity, phase_s, notes = row
        result[h_id.strip()] = (int(rank_s), int(phase_s), notes.strip())
    return result


def _parse_markdown_strategy(content: str) -> ProposalSelectionData | None:
    """Strategy B: Parse Consolidated Proposal List + Roadmap + Ranking Table."""

    # Step 1: Parse detailed proposals from Consolidated Proposal List
    proposals_by_id = _parse_consolidated_proposals(content)
    if not proposals_by_id:
        logger.info(
            "No proposals found in Consolidated Proposal List, trying table-only fallback"
        )
        return _parse_table_only_fallback(content)

    # Step 2: Parse ranking table for rank/phase/notes
    ranking = _parse_ranking_table(content)
    rank_map = {h_id: rank for h_id, (rank, _, _) in ranking.items()}

    # Enrich proposals with ranking table data (notes, rank)
    for h_id, (rank, phase, notes) in ranking.items():
        if h_id in proposals_by_id:
            proposals_by_id[h_id].rank = rank
            if notes and not proposals_by_id[h_id].notes:
                proposals_by_id[h_id].notes = notes

    # Step 3: Parse Implementation Roadmap for phase/batch structure
    phases = _parse_implementation_roadmap(content)

    if phases:
        # Merge: roadmap structure + proposal details
        _merge_roadmap_and_proposals(phases, proposals_by_id, rank_map)
    else:
        # Fallback: group by ranking table's Phase column
        phase_map: dict[int, list[StructuredProposal]] = {}
        for h_id, (rank, phase_num, notes) in ranking.items():
            if h_id in proposals_by_id:
                phase_map.setdefault(phase_num, []).append(proposals_by_id[h_id])
        _PHASE_LABELS = {
            1: ("Quick Wins", "Low-risk, high-confidence proposals"),
            2: ("Core Improvements", "Medium-risk, high-impact proposals"),
            3: ("Exploration", "High-risk/high-reward or long-term proposals"),
        }
        phases = []
        for phase_num in sorted(phase_map):
            label, desc = _PHASE_LABELS.get(phase_num, (f"Phase {phase_num}", ""))
            proposals = sorted(phase_map[phase_num], key=lambda p: p.rank)
            phases.append(
                ProposalPhase(
                    phase=phase_num, label=label, description=desc, proposals=proposals
                )
            )

    total = sum(len(p.proposals) for p in phases)
    themes = sorted({p.theme for phase in phases for p in phase.proposals if p.theme})

    return ProposalSelectionData(phases=phases, total_count=total, themes=themes)


def _parse_table_only_fallback(content: str) -> ProposalSelectionData | None:
    """Minimal fallback: parse only the Priority Ranking Table (no details)."""
    rows = _TABLE_ROW_RE.findall(content)
    if not rows:
        return None

    _PHASE_LABELS = {
        1: ("Quick Wins", "Low-risk, high-confidence proposals"),
        2: ("Core Improvements", "Medium-risk, high-impact proposals"),
        3: ("Exploration", "High-risk/high-reward or long-term proposals"),
    }

    phase_map: dict[int, list[StructuredProposal]] = {}
    for rank_s, h_id, name, sources, impact, prob, complexity, phase_s, notes in rows:
        phase_num = int(phase_s)
        proposal = StructuredProposal(
            id=h_id.strip(),
            rank=int(rank_s),
            title=name.strip(),
            source_workers=[s.strip() for s in sources.split(",") if s.strip()],
            impact=impact.strip(),
            probability=prob.strip(),
            complexity=complexity.strip(),
            notes=notes.strip(),
            one_line_summary=name.strip(),
        )
        phase_map.setdefault(phase_num, []).append(proposal)

    phases = []
    for phase_num in sorted(phase_map):
        label, desc = _PHASE_LABELS.get(phase_num, (f"Phase {phase_num}", ""))
        proposals = sorted(phase_map[phase_num], key=lambda p: p.rank)
        phases.append(
            ProposalPhase(
                phase=phase_num, label=label, description=desc, proposals=proposals
            )
        )

    total = sum(len(p.proposals) for p in phases)
    return ProposalSelectionData(phases=phases, total_count=total, themes=[])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _find_unified_plan(workspace_path: str | Path) -> Path | None:
    """Locate the aggregator's unified_plan.md in the workspace."""
    ws = Path(workspace_path)
    plan = ws / "checkpoints" / "bta" / "aggregator" / "outputs" / "unified_plan.md"
    if plan.exists():
        return plan
    plan = ws / "outputs" / "unified_plan.md"
    if plan.exists():
        return plan
    return None


def parse_proposals(workspace_path: str | Path) -> ProposalSelectionData | None:
    """Parse research proposals from the unified plan in the workspace.

    Args:
        workspace_path: Root of the research workspace (e.g., tasks/research_20260403_161316)

    Returns:
        ProposalSelectionData with proposals grouped by phase, or None if parsing fails.
    """
    plan_file = _find_unified_plan(workspace_path)
    if plan_file is None:
        logger.info("No unified_plan.md found in workspace %s", workspace_path)
        return None

    content = plan_file.read_text(encoding="utf-8")

    # Strategy A: JSON code fence (primary)
    result = _parse_json_strategy(content)
    if result is not None:
        logger.info(
            "Parsed proposals from JSON fence: %d phases, %d total",
            len(result.phases),
            result.total_count,
        )
        return result

    # Strategy B: Markdown two-section extraction (fallback)
    result = _parse_markdown_strategy(content)
    if result is not None:
        logger.info(
            "Parsed proposals from markdown (fallback): %d phases, %d total, %d batches",
            len(result.phases),
            result.total_count,
            sum(len(p.batches) for p in result.phases),
        )
        return result

    logger.warning("Could not parse proposals from %s", plan_file)
    return None

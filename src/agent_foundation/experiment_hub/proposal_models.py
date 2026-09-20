# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Structured data models for research proposals.

Used by the experiment-hub proposal-selection path to carry parsed proposal
data from the server to the frontend widget. The richer hub-side shape:
``ProposalSelectionData`` / ``StructuredProposal`` carry slots / includes /
batches / themes / combo_constraints.

Ported faithfully from RankEvolve's ``agentic_foundation.common.ui.proposal_models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StructuredProposal:
    """A single research proposal/hypothesis from the unified plan."""

    id: str  # "H1", "H2", ...
    rank: int  # Priority rank (1 = highest)
    title: str  # Short descriptive title
    theme: str = ""  # "Multi-Task Architecture", etc.
    source_workers: list[str] = field(default_factory=list)  # ["W0", "W4"]
    impact: str = ""  # "Low", "Medium", "High", "Med-High"
    probability: str = ""  # "75%", "High (>70%)"
    complexity: str = ""  # "Low", "Medium", "High"
    one_line_summary: str = ""
    notes: str = ""
    problem: str = ""  # Detailed problem statement from Consolidated Proposal List
    approach: str = ""  # Proposed approach description
    cross_refs: str = ""  # "Synergistic with H2, H4"
    slots: list[str] = field(
        default_factory=list
    )  # Mutually-exclusive slot memberships ("backbone", "embedding_conditioning", ...)
    includes: list[str] = field(
        default_factory=list
    )  # Hypothesis IDs this combo pre-bundles (e.g., H15 includes ["H8","H14"])

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "rank": self.rank,
            "title": self.title,
        }
        if self.theme:
            d["theme"] = self.theme
        if self.source_workers:
            d["source_workers"] = self.source_workers
        if self.impact:
            d["impact"] = self.impact
        if self.probability:
            d["probability"] = self.probability
        if self.complexity:
            d["complexity"] = self.complexity
        if self.one_line_summary:
            d["one_line_summary"] = self.one_line_summary
        if self.notes:
            d["notes"] = self.notes
        if self.problem:
            d["problem"] = self.problem
        if self.approach:
            d["approach"] = self.approach
        if self.cross_refs:
            d["cross_refs"] = self.cross_refs
        if self.slots:
            d["slots"] = list(self.slots)
        if self.includes:
            d["includes"] = list(self.includes)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredProposal:
        return cls(
            id=data.get("id", ""),
            rank=data.get("rank", 0),
            title=data.get("title", ""),
            theme=data.get("theme", ""),
            source_workers=data.get("source_workers", []),
            impact=data.get("impact", ""),
            probability=data.get("probability", ""),
            complexity=data.get("complexity", ""),
            one_line_summary=data.get("one_line_summary", ""),
            notes=data.get("notes", ""),
            problem=data.get("problem", ""),
            approach=data.get("approach", ""),
            cross_refs=data.get("cross_refs", ""),
            slots=list(data.get("slots", []) or []),
            includes=list(data.get("includes", []) or []),
        )


@dataclass
class Batch:
    """A batch within an implementation phase (e.g., Batch 1A: Loss & Training)."""

    id: str  # "1A", "2B", "3A"
    label: str = ""  # "Loss & Training, Independent"
    timeline: str = ""  # "Week 1"
    hypothesis_ids: list[str] = field(default_factory=list)  # ["H17", "H1", ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "timeline": self.timeline,
            "hypothesis_ids": self.hypothesis_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Batch:
        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            timeline=data.get("timeline", ""),
            hypothesis_ids=data.get("hypothesis_ids", []),
        )


@dataclass
class ProposalPhase:
    """A group of proposals in the same implementation phase."""

    phase: int  # 1, 2, 3
    label: str  # "Quick Wins", "Core Improvements", "Exploration"
    description: str = ""
    proposals: list[StructuredProposal] = field(default_factory=list)
    batches: list[Batch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "phase": self.phase,
            "label": self.label,
            "description": self.description,
            "proposals": [p.to_dict() for p in self.proposals],
        }
        if self.batches:
            d["batches"] = [b.to_dict() for b in self.batches]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProposalPhase:
        return cls(
            phase=data.get("phase", 0),
            label=data.get("label", ""),
            description=data.get("description", ""),
            proposals=[
                StructuredProposal.from_dict(p) for p in data.get("proposals", [])
            ],
            batches=[Batch.from_dict(b) for b in data.get("batches", [])],
        )


@dataclass
class ComboConstraint:
    """Asymmetric/typed constraints that the per-hypothesis ``slots`` field
    cannot express on its own (requires, recommends, and a backstop for
    mutually_exclusive groups that span beyond a single slot).

    For pure mutual exclusion within a configuration knob, prefer the per-H
    ``slots`` field on :class:`StructuredProposal`. Slots produce O(1)
    authoring per new hypothesis; ``ComboConstraint`` covers the remainder.
    """

    id: str
    kind: str  # "mutually_exclusive" | "requires" | "recommends"
    hypothesis_ids: list[str] = field(default_factory=list)
    requires_ids: list[str] = field(default_factory=list)
    requires_any_of: bool = False
    label: str = ""
    reason: str = ""
    severity: str = "error"  # "error" | "warning" | "info"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "hypothesis_ids": list(self.hypothesis_ids),
        }
        if self.requires_ids:
            d["requires_ids"] = list(self.requires_ids)
        if self.requires_any_of:
            d["requires_any_of"] = True
        if self.label:
            d["label"] = self.label
        if self.reason:
            d["reason"] = self.reason
        if self.severity and self.severity != "error":
            d["severity"] = self.severity
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComboConstraint:
        return cls(
            id=data.get("id", ""),
            kind=data.get("kind", "mutually_exclusive"),
            hypothesis_ids=list(data.get("hypothesis_ids", []) or []),
            requires_ids=list(data.get("requires_ids", []) or []),
            requires_any_of=bool(data.get("requires_any_of", False)),
            label=data.get("label", ""),
            reason=data.get("reason", ""),
            severity=data.get("severity", "error"),
        )


@dataclass
class ProposalSelectionData:
    """Top-level container for all parsed proposals, grouped by phase."""

    phases: list[ProposalPhase] = field(default_factory=list)
    total_count: int = 0
    themes: list[str] = field(default_factory=list)
    combo_constraints: list[ComboConstraint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "phases": [p.to_dict() for p in self.phases],
            "total_count": self.total_count,
            "themes": self.themes,
        }
        if self.combo_constraints:
            d["combo_constraints"] = [c.to_dict() for c in self.combo_constraints]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProposalSelectionData:
        result = cls(
            phases=[ProposalPhase.from_dict(p) for p in data.get("phases", [])],
            total_count=data.get("total_count", 0),
            themes=data.get("themes", []),
            combo_constraints=[
                ComboConstraint.from_dict(c)
                for c in data.get("combo_constraints", []) or []
            ],
        )
        # Soft validator. Combo-disguised hypotheses (with non-empty
        # `includes`) violate the "atomic, independently flag-gated change"
        # definition the propose templates spell out. We log a WARNING
        # (don't raise) so existing sessions whose unified_plan.md already
        # has these IDs keep loading; the WebUI's mergeOverrides + chip
        # rendering surfaces the issue to the user.
        try:
            import logging as _logging

            _logger = _logging.getLogger(__name__)
            offenders: list[str] = []
            for phase in result.phases:
                for prop in phase.proposals or []:
                    inc = getattr(prop, "includes", None)
                    if isinstance(inc, list) and len([x for x in inc if x]) > 0:
                        offenders.append(getattr(prop, "id", "?"))
            if offenders:
                _logger.warning(
                    "ProposalSelectionData: %d combo-disguised hypothesis "
                    "id(s) detected (non-empty `includes`): %s. These "
                    "should be split into atomic flag-gated hypotheses + a "
                    "matching combos entry.",
                    len(offenders),
                    ", ".join(offenders),
                )
        except Exception:
            # Validator is advisory; never break parsing.
            pass
        return result

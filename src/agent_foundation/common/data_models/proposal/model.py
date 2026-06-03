"""Proposal data models — generic, domain-agnostic structured proposal schema.

Inspired by RankEvolve's ``StructuredProposal``/``ProposalSelectionData`` but
trimmed to framework-level generics. Domain-specific fields (probability, slots,
batches) go in ``Proposal.metadata`` or in subclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Proposal:
    """A single actionable proposal with metadata for ranking and selection."""

    id: str
    rank: int
    title: str
    summary: str = ""
    impact: str = ""
    complexity: str = ""
    approach: str = ""
    problem: str = ""
    dependencies: list[str] = field(default_factory=list)
    cross_refs: str = ""
    proposal_file: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "rank": self.rank,
            "title": self.title,
        }
        if self.summary:
            d["summary"] = self.summary
        if self.impact:
            d["impact"] = self.impact
        if self.complexity:
            d["complexity"] = self.complexity
        if self.approach:
            d["approach"] = self.approach
        if self.problem:
            d["problem"] = self.problem
        if self.dependencies:
            d["dependencies"] = list(self.dependencies)
        if self.cross_refs:
            d["cross_refs"] = self.cross_refs
        if self.proposal_file:
            d["proposal_file"] = self.proposal_file
        if self.tags:
            d["tags"] = list(self.tags)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Proposal:
        return cls(
            id=d["id"],
            rank=int(d.get("rank", 0)),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            impact=d.get("impact", ""),
            complexity=d.get("complexity", ""),
            approach=d.get("approach", ""),
            problem=d.get("problem", ""),
            dependencies=list(d.get("dependencies", [])),
            cross_refs=d.get("cross_refs", ""),
            proposal_file=d.get("proposal_file", ""),
            tags=list(d.get("tags", [])),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class ProposalGroup:
    """Phase-based grouping of proposals (Quick Wins, Core, Exploration)."""

    phase: int
    label: str
    description: str = ""
    proposals: list[Proposal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"phase": self.phase, "label": self.label}
        if self.description:
            d["description"] = self.description
        d["proposals"] = [p.to_dict() for p in self.proposals]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProposalGroup:
        return cls(
            phase=int(d["phase"]),
            label=d.get("label", ""),
            description=d.get("description", ""),
            proposals=[Proposal.from_dict(p) for p in d.get("proposals", [])],
        )


@dataclass
class ProposalConstraint:
    """Inter-proposal constraint (mutually exclusive, requires, recommends)."""

    id: str
    kind: str
    proposal_ids: list[str] = field(default_factory=list)
    requires_ids: list[str] = field(default_factory=list)
    label: str = ""
    reason: str = ""
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "proposal_ids": list(self.proposal_ids),
        }
        if self.requires_ids:
            d["requires_ids"] = list(self.requires_ids)
        if self.label:
            d["label"] = self.label
        if self.reason:
            d["reason"] = self.reason
        if self.severity != "error":
            d["severity"] = self.severity
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProposalConstraint:
        return cls(
            id=d["id"],
            kind=d["kind"],
            proposal_ids=list(d.get("proposal_ids", [])),
            requires_ids=list(d.get("requires_ids", [])),
            label=d.get("label", ""),
            reason=d.get("reason", ""),
            severity=d.get("severity", "error"),
        )


@dataclass
class ProposalIndex:
    """Top-level container for a set of ranked, grouped proposals."""

    version: str = "1"
    created_at: str = ""
    source_workspace: str = ""
    total_count: int = 0
    groups: list[ProposalGroup] = field(default_factory=list)
    constraints: list[ProposalConstraint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def all_proposals(self) -> list[Proposal]:
        """Flat list of all proposals across groups, sorted by rank ascending."""
        proposals = [p for g in self.groups for p in g.proposals]
        proposals.sort(key=lambda p: p.rank)
        return proposals

    def get_proposals_by_ids(self, ids: list[str]) -> list[Proposal]:
        """Return proposals matching *ids*, preserving the requested order.

        Raises ``KeyError`` listing valid IDs if any requested ID is missing.
        """
        by_id = {p.id: p for p in self.all_proposals()}
        missing = [i for i in ids if i not in by_id]
        if missing:
            valid = sorted(by_id.keys())
            raise KeyError(
                f"Unknown proposal IDs: {missing}. Valid IDs: {valid}"
            )
        return [by_id[i] for i in ids]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "source_workspace": self.source_workspace,
            "total_count": self.total_count,
            "groups": [g.to_dict() for g in self.groups],
            "constraints": [c.to_dict() for c in self.constraints],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProposalIndex:
        return cls(
            version=str(d.get("version", "1")),
            created_at=d.get("created_at", ""),
            source_workspace=d.get("source_workspace", ""),
            total_count=int(d.get("total_count", 0)),
            groups=[ProposalGroup.from_dict(g) for g in d.get("groups", [])],
            constraints=[
                ProposalConstraint.from_dict(c)
                for c in d.get("constraints", [])
            ],
            warnings=list(d.get("warnings", [])),
        )

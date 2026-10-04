"""Proposal data models — generic, domain-agnostic structured proposal schema.

Inspired by RankEvolve's ``StructuredProposal``/``ProposalSelectionData`` but
trimmed to framework-level generics. Domain-specific fields (probability, slots,
batches) go in ``Proposal.metadata`` or in subclasses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


def _as_list(x: Any) -> list[str]:
    """Normalise a scalar OR list into a ``list[str]``.

    Real LLM constraint output emits the ``from``/``to`` edge fields as either a
    single id (``"P5"``) or a list of ids (``["P1", "P3"]``); both must
    round-trip to ``list[str]`` so downstream consumers never branch on type.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    return [str(x)]


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
class Batch:
    """A batch of proposals within an implementation phase.

    Container-level grouping the Experiment Hub uses to queue selected
    proposals batch-by-batch (e.g. "Batch 1A: Loss & Training"). IDs are
    canonical AF **proposal ids** (``proposal_ids``); ``from_dict`` also
    accepts the hub-shaped ``hypothesis_ids`` alias so an externally produced
    payload still parses at this boundary (see correction #16).
    """

    id: str
    label: str = ""
    timeline: str = ""
    proposal_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id}
        if self.label:
            d["label"] = self.label
        if self.timeline:
            d["timeline"] = self.timeline
        if self.proposal_ids:
            d["proposal_ids"] = list(self.proposal_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Batch:
        # Tolerate the hub dialect: ``hypothesis_ids`` is the H-id-shaped alias
        # for the canonical ``proposal_ids``. Accept either so a hub-produced
        # batch round-trips into the canonical AF shape.
        return cls(
            id=str(d.get("id", "")),
            label=d.get("label", ""),
            timeline=d.get("timeline", ""),
            proposal_ids=_as_list(d.get("proposal_ids", d.get("hypothesis_ids"))),
        )


@dataclass
class ProposalGroup:
    """Phase-based grouping of proposals (Quick Wins, Core, Exploration)."""

    phase: int
    label: str
    description: str = ""
    proposals: list[Proposal] = field(default_factory=list)
    batches: list[Batch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"phase": self.phase, "label": self.label}
        if self.description:
            d["description"] = self.description
        d["proposals"] = [p.to_dict() for p in self.proposals]
        if self.batches:
            d["batches"] = [b.to_dict() for b in self.batches]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProposalGroup:
        return cls(
            phase=int(d["phase"]),
            label=d.get("label", ""),
            description=d.get("description", ""),
            proposals=[Proposal.from_dict(p) for p in d.get("proposals", [])],
            batches=[Batch.from_dict(b) for b in d.get("batches", []) or []],
        )


@dataclass
class ProposalConstraint:
    """Inter-proposal constraint (mutually exclusive, requires, recommends)."""

    id: str
    kind: str
    proposal_ids: list[str] = field(default_factory=list)
    requires_ids: list[str] = field(default_factory=list)
    requires_any_of: bool = False
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
        # ``requires_any_of`` toggles ``requires_ids`` from ALL-of (default,
        # the conjunctive AND) to ANY-of (disjunctive OR). Only serialised when
        # True so existing proposals.json stays byte-identical (additive).
        if self.requires_any_of:
            d["requires_any_of"] = True
        if self.label:
            d["label"] = self.label
        if self.reason:
            d["reason"] = self.reason
        if self.severity != "error":
            d["severity"] = self.severity
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProposalConstraint:
        # Tolerant of real LLM constraint dialects, which deviate from the
        # canonical schema. Aliases observed in production output:
        #   kind         <- type
        #   proposal_ids <- from   (scalar or list)
        #   requires_ids <- to     (scalar or list)
        #   reason       <- rule | note
        # Missing keys fall back to sensible defaults rather than raising
        # KeyError, so a single odd constraint cannot abort the whole parse.
        return cls(
            id=str(d.get("id", "")),
            kind=str(d.get("kind", d.get("type", "unknown"))),
            proposal_ids=_as_list(d.get("proposal_ids", d.get("from"))),
            requires_ids=_as_list(d.get("requires_ids", d.get("to"))),
            requires_any_of=bool(d.get("requires_any_of", False)),
            label=d.get("label", ""),
            reason=d.get("reason", d.get("rule", d.get("note", ""))),
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
    themes: list[str] = field(default_factory=list)
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
            raise KeyError(f"Unknown proposal IDs: {missing}. Valid IDs: {valid}")
        return [by_id[i] for i in ids]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "created_at": self.created_at,
            "source_workspace": self.source_workspace,
            "total_count": self.total_count,
            "groups": [g.to_dict() for g in self.groups],
            "constraints": [c.to_dict() for c in self.constraints],
        }
        # Container-level theme labels (e.g. "Multi-Task Architecture") the
        # Experiment Hub uses for theme grouping. Emitted only when present so
        # existing proposals.json without it stays byte-identical (additive).
        if self.themes:
            d["themes"] = list(self.themes)
        d["warnings"] = list(self.warnings)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProposalIndex:
        # Per-constraint tolerance: a single malformed constraint must never
        # discard the whole index (which would lose every valid proposal too).
        # Bad constraints are logged and skipped; everything else is kept.
        constraints: list[ProposalConstraint] = []
        for c in d.get("constraints", []):
            try:
                constraints.append(ProposalConstraint.from_dict(c))
            except Exception as exc:  # noqa: BLE001 — defence-in-depth at parse boundary
                _logger.warning("Skipping malformed proposal constraint %r: %s", c, exc)
        return cls(
            version=str(d.get("version", "1")),
            created_at=d.get("created_at", ""),
            source_workspace=d.get("source_workspace", ""),
            total_count=int(d.get("total_count", 0)),
            groups=[ProposalGroup.from_dict(g) for g in d.get("groups", [])],
            constraints=constraints,
            themes=[str(t) for t in d.get("themes", []) or []],
            warnings=list(d.get("warnings", [])),
        )

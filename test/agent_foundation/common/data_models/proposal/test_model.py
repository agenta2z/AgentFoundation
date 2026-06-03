"""Tests for proposal data models — round-trip, methods, edge cases."""
import pytest

from agent_foundation.common.data_models.proposal.model import (
    Proposal,
    ProposalConstraint,
    ProposalGroup,
    ProposalIndex,
)


@pytest.fixture
def sample_index():
    return ProposalIndex(
        version="1",
        created_at="2026-06-02T20:00:00Z",
        source_workspace="/tmp/ws",
        total_count=3,
        groups=[
            ProposalGroup(
                phase=1,
                label="Quick Wins",
                description="Low-hanging fruit",
                proposals=[
                    Proposal(id="P1", rank=1, title="Add caching",
                             summary="Cache hot paths", impact="high",
                             complexity="low", approach="Redis layer",
                             problem="Latency", dependencies=[],
                             cross_refs="synergistic with P2",
                             tags=["perf"], metadata={"est_hours": 4}),
                    Proposal(id="P2", rank=2, title="Batch queries",
                             summary="Batch DB calls", impact="medium",
                             complexity="low"),
                ],
            ),
            ProposalGroup(
                phase=2,
                label="Core",
                proposals=[
                    Proposal(id="P3", rank=3, title="Rewrite auth",
                             dependencies=["P1"], tags=["security"]),
                ],
            ),
        ],
        constraints=[
            ProposalConstraint(
                id="C1", kind="requires",
                proposal_ids=["P3"], requires_ids=["P1"],
                label="auth needs cache", reason="P3 depends on P1",
                severity="error",
            ),
        ],
        warnings=[],
    )


class TestProposalRoundTrip:
    def test_proposal_round_trip(self):
        p = Proposal(id="P1", rank=1, title="Test", summary="s",
                     impact="high", complexity="low", approach="a",
                     problem="p", dependencies=["P2"], cross_refs="see P2",
                     tags=["t"], metadata={"k": "v"})
        d = p.to_dict()
        p2 = Proposal.from_dict(d)
        assert p2.id == p.id
        assert p2.rank == p.rank
        assert p2.title == p.title
        assert p2.summary == p.summary
        assert p2.impact == p.impact
        assert p2.complexity == p.complexity
        assert p2.approach == p.approach
        assert p2.problem == p.problem
        assert p2.dependencies == p.dependencies
        assert p2.cross_refs == p.cross_refs
        assert p2.tags == p.tags
        assert p2.metadata == p.metadata

    def test_proposal_empty_defaults_round_trip(self):
        p = Proposal(id="P1", rank=1, title="Minimal")
        d = p.to_dict()
        p2 = Proposal.from_dict(d)
        assert p2.summary == ""
        assert p2.dependencies == []
        assert p2.metadata == {}

    def test_proposal_index_round_trip(self, sample_index):
        d = sample_index.to_dict()
        idx2 = ProposalIndex.from_dict(d)
        assert idx2.version == sample_index.version
        assert idx2.created_at == sample_index.created_at
        assert idx2.source_workspace == sample_index.source_workspace
        assert idx2.total_count == sample_index.total_count
        assert len(idx2.groups) == 2
        assert len(idx2.groups[0].proposals) == 2
        assert idx2.groups[0].proposals[0].id == "P1"
        assert idx2.groups[0].proposals[0].metadata == {"est_hours": 4}
        assert idx2.groups[0].proposals[0].cross_refs == "synergistic with P2"
        assert len(idx2.constraints) == 1
        assert idx2.constraints[0].requires_ids == ["P1"]

    def test_metadata_preserves_arbitrary_dict(self):
        p = Proposal(id="P1", rank=1, title="T",
                     metadata={"probability": "75%", "slots": ["a", "b"],
                                "nested": {"deep": True}})
        d = p.to_dict()
        p2 = Proposal.from_dict(d)
        assert p2.metadata["probability"] == "75%"
        assert p2.metadata["slots"] == ["a", "b"]
        assert p2.metadata["nested"]["deep"] is True

    def test_constraint_default_severity(self):
        c = ProposalConstraint(id="C1", kind="mutually_exclusive",
                                proposal_ids=["P1", "P2"])
        d = c.to_dict()
        assert "severity" not in d  # default "error" omitted
        c2 = ProposalConstraint.from_dict(d)
        assert c2.severity == "error"

    def test_constraint_non_default_severity_preserved(self):
        c = ProposalConstraint(id="C1", kind="recommends",
                                proposal_ids=["P1"], severity="warning")
        d = c.to_dict()
        assert d["severity"] == "warning"
        c2 = ProposalConstraint.from_dict(d)
        assert c2.severity == "warning"


class TestProposalIndexMethods:
    def test_all_proposals_sorted_by_rank(self, sample_index):
        all_p = sample_index.all_proposals()
        assert [p.id for p in all_p] == ["P1", "P2", "P3"]
        assert [p.rank for p in all_p] == [1, 2, 3]

    def test_get_proposals_by_ids(self, sample_index):
        selected = sample_index.get_proposals_by_ids(["P3", "P1"])
        assert [p.id for p in selected] == ["P3", "P1"]

    def test_get_proposals_by_ids_unknown_raises(self, sample_index):
        with pytest.raises(KeyError, match="P99"):
            sample_index.get_proposals_by_ids(["P1", "P99"])

    def test_get_proposals_by_ids_shows_valid(self, sample_index):
        with pytest.raises(KeyError, match="P1.*P2.*P3"):
            sample_index.get_proposals_by_ids(["P99"])

    def test_all_proposals_empty_index(self):
        idx = ProposalIndex()
        assert idx.all_proposals() == []

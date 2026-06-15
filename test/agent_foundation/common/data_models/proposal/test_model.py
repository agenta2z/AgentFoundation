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


class TestConstraintTolerance:
    """D2: ``from_dict`` tolerates real LLM constraint dialects without crashing.

    Every case below mirrors a shape actually observed in research-propose
    output; canonical-schema dicts must remain byte-for-byte unchanged.
    """

    def test_canonical_schema_round_trip(self):
        d = {
            "id": "C1",
            "kind": "requires",
            "proposal_ids": ["P2"],
            "requires_ids": ["P1"],
        }
        c = ProposalConstraint.from_dict(d)
        assert c.id == "C1"
        assert c.kind == "requires"
        assert c.proposal_ids == ["P2"]
        assert c.requires_ids == ["P1"]
        # Canonical dict round-trips identically (defaults omitted by to_dict).
        assert c.to_dict() == d

    def test_dialect_alpha_free_form_rule(self):
        c = ProposalConstraint.from_dict(
            {"type": "ordering", "rule": "P10 must precede every other proposal."}
        )
        assert c.kind == "ordering"
        assert c.reason == "P10 must precede every other proposal."
        assert c.id == ""
        assert c.proposal_ids == []
        assert c.requires_ids == []

    def test_dialect_beta_scalar_to(self):
        c = ProposalConstraint.from_dict(
            {
                "type": "requires",
                "from": "P5",
                "to": "P1",
                "note": "ORPO benefits most when stacked on Reason-first base.",
            }
        )
        assert c.kind == "requires"
        assert c.proposal_ids == ["P5"]
        assert c.requires_ids == ["P1"]
        assert c.reason == "ORPO benefits most when stacked on Reason-first base."

    def test_dialect_beta_list_to(self):
        c = ProposalConstraint.from_dict(
            {"type": "requires", "from": "P4", "to": ["P1", "P3"]}
        )
        assert c.proposal_ids == ["P4"]
        assert c.requires_ids == ["P1", "P3"]

    def test_dialect_beta_reason_not_note(self):
        c = ProposalConstraint.from_dict(
            {"type": "recommends", "from": "P3", "to": "P1", "reason": "stronger together"}
        )
        assert c.kind == "recommends"
        assert c.reason == "stronger together"

    def test_completely_empty_dict_uses_defaults(self):
        c = ProposalConstraint.from_dict({})
        assert c.id == ""
        assert c.kind == "unknown"
        assert c.proposal_ids == []
        assert c.requires_ids == []
        assert c.label == ""
        assert c.reason == ""
        assert c.severity == "error"

    def test_index_partial_failure_keeps_valid_constraints(self, caplog):
        import logging

        data = {
            "version": "1",
            "total_count": 0,
            "groups": [],
            "constraints": [
                {"type": "requires", "from": "P5", "to": "P1"},
                "this-is-not-a-dict-and-must-be-skipped",  # malformed
                {"id": "C2", "kind": "ordering", "rule": "P1 first"},
            ],
        }
        with caplog.at_level(logging.WARNING):
            idx = ProposalIndex.from_dict(data)
        # The 2 valid constraints survive; the malformed one is dropped.
        assert len(idx.constraints) == 2
        assert idx.constraints[0].proposal_ids == ["P5"]
        assert idx.constraints[1].kind == "ordering"
        assert any("malformed proposal constraint" in r.message for r in caplog.records)

    def test_index_with_constraints_does_not_crash_whole_parse(self):
        # A messy mix still yields a usable index with all proposals intact.
        data = {
            "total_count": 1,
            "groups": [
                {"phase": 1, "label": "G", "proposals": [
                    {"id": "P1", "rank": 1, "title": "Keep me"}]},
            ],
            "constraints": [{"type": "ordering", "rule": "free form"}],
        }
        idx = ProposalIndex.from_dict(data)
        assert [p.id for p in idx.all_proposals()] == ["P1"]
        assert idx.constraints[0].kind == "ordering"

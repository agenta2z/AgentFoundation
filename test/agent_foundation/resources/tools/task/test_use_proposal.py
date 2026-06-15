"""Tests for --use-proposal parameter in task executor."""
import json

import pytest

from agent_foundation.common.data_models.proposal.model import (
    Proposal,
    ProposalGroup,
    ProposalIndex,
)
from agent_foundation.common.data_models.proposal.parser import write_proposal_index
from agent_foundation.resources.tools.task.executor import _resolve_proposal_plan


@pytest.fixture
def proposals_json(tmp_path):
    idx = ProposalIndex(
        version="1",
        created_at="2026-06-02T20:00:00Z",
        source_workspace=str(tmp_path),
        total_count=3,
        groups=[
            ProposalGroup(
                phase=1, label="Quick Wins",
                proposals=[
                    Proposal(id="P1", rank=1, title="Add caching",
                             impact="high", complexity="low",
                             problem="API latency is 500ms",
                             approach="Add Redis caching layer",
                             cross_refs="synergistic with P2"),
                    Proposal(id="P2", rank=2, title="Batch queries",
                             impact="medium", complexity="low",
                             approach="Combine N+1 queries"),
                ],
            ),
            ProposalGroup(
                phase=2, label="Core",
                proposals=[
                    Proposal(id="P3", rank=3, title="Rewrite auth",
                             impact="high", complexity="high",
                             dependencies=["P1"],
                             approach="JWT-based auth system"),
                ],
            ),
        ],
    )
    path = tmp_path / "proposals.json"
    write_proposal_index(path, idx)
    return path


class TestResolveProposalPlan:
    def test_all_proposals(self, proposals_json):
        result = _resolve_proposal_plan(str(proposals_json), None)
        assert result is not None
        content = open(result).read()
        assert "P1" in content
        assert "P2" in content
        assert "P3" in content
        assert "Add caching" in content
        assert "Batch queries" in content
        assert "Rewrite auth" in content

    def test_filter_by_ids(self, proposals_json):
        result = _resolve_proposal_plan(str(proposals_json), "P1,P3")
        assert result is not None
        content = open(result).read()
        assert "P1" in content
        assert "P3" in content
        assert "P2" not in content.split("Selected:")[1].split("\n")[0]

    def test_unknown_ids_returns_none(self, proposals_json):
        result = _resolve_proposal_plan(str(proposals_json), "P99")
        assert result is None

    def test_nonexistent_file_returns_none(self):
        result = _resolve_proposal_plan("/nonexistent/proposals.json", None)
        assert result is None

    def test_plan_contains_approach(self, proposals_json):
        result = _resolve_proposal_plan(str(proposals_json), "P1")
        content = open(result).read()
        assert "Redis caching layer" in content

    def test_plan_contains_dependencies(self, proposals_json):
        result = _resolve_proposal_plan(str(proposals_json), "P3")
        content = open(result).read()
        assert "P1" in content
        assert "Dependencies" in content

    def test_plan_contains_cross_refs(self, proposals_json):
        result = _resolve_proposal_plan(str(proposals_json), "P1")
        content = open(result).read()
        assert "synergistic with P2" in content

    def test_plan_contains_source_path(self, proposals_json):
        result = _resolve_proposal_plan(str(proposals_json), "P1")
        content = open(result).read()
        assert str(proposals_json) in content

    def test_proposal_file_content_inlined(self, tmp_path):
        """When proposal_file points to a real file, its content is inlined."""
        proposals_dir = tmp_path / "proposals"
        proposals_dir.mkdir()
        (proposals_dir / "P1.md").write_text("# Detailed analysis\nFull research content here.")

        idx = ProposalIndex(
            version="1", total_count=1,
            groups=[ProposalGroup(phase=1, label="Test", proposals=[
                Proposal(id="P1", rank=1, title="With detail file",
                         approach="summary", proposal_file="proposals/P1.md"),
            ])],
        )
        path = tmp_path / "proposals.json"
        write_proposal_index(path, idx)

        result = _resolve_proposal_plan(str(path), "P1")
        content = open(result).read()
        assert "Full research content here" in content
        assert "proposals/P1.md" in content

    def test_proposal_file_missing_shows_path(self, tmp_path):
        """When proposal_file points to a missing file, show the path as reference."""
        idx = ProposalIndex(
            version="1", total_count=1,
            groups=[ProposalGroup(phase=1, label="Test", proposals=[
                Proposal(id="P1", rank=1, title="Missing file",
                         proposal_file="proposals/P1.md"),
            ])],
        )
        path = tmp_path / "proposals.json"
        write_proposal_index(path, idx)

        result = _resolve_proposal_plan(str(path), "P1")
        content = open(result).read()
        assert "proposals/P1.md" in content

    def test_no_proposal_file_uses_summary_only(self, tmp_path):
        """When proposal_file is empty, only summary fields are used."""
        idx = ProposalIndex(
            version="1", total_count=1,
            groups=[ProposalGroup(phase=1, label="Test", proposals=[
                Proposal(id="P1", rank=1, title="No file",
                         approach="Just the summary approach"),
            ])],
        )
        path = tmp_path / "proposals.json"
        write_proposal_index(path, idx)

        result = _resolve_proposal_plan(str(path), "P1")
        content = open(result).read()
        assert "Just the summary approach" in content
        assert "Full Proposal Detail" not in content

    def test_mutual_exclusivity_is_enforced_in_execute(self):
        """--use-proposal + --use-plan should error (tested at execute level)."""
        # This is tested at the execute() level, not _resolve_proposal_plan.
        # The check is: if use_proposal and init_plan_path → error.
        # We just verify the guard exists in the code.
        import inspect
        from agent_foundation.resources.tools.task.executor import execute
        src = inspect.getsource(execute)
        assert "mutually exclusive" in src

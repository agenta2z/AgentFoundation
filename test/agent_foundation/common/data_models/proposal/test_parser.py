"""Tests for proposal parser — 3 strategies + atomic write."""
import json
import textwrap

import pytest

from agent_foundation.common.data_models.proposal.model import (
    Proposal,
    ProposalGroup,
    ProposalIndex,
)
from agent_foundation.common.data_models.proposal.parser import (
    make_empty_index,
    parse_proposal_file,
    parse_proposal_index_from_text,
    parse_proposals,
    write_proposal_index,
)


@pytest.fixture
def sample_index():
    return ProposalIndex(
        version="1",
        created_at="2026-06-02T20:00:00Z",
        source_workspace="/tmp/ws",
        total_count=2,
        groups=[
            ProposalGroup(
                phase=1, label="Quick Wins",
                proposals=[
                    Proposal(id="P1", rank=1, title="Add caching",
                             impact="high", complexity="low"),
                    Proposal(id="P2", rank=2, title="Batch queries",
                             impact="medium", complexity="low"),
                ],
            ),
        ],
    )


@pytest.fixture
def markdown_with_fence(sample_index):
    fence = json.dumps(sample_index.to_dict(), indent=2)
    return textwrap.dedent(f"""\
        # Unified Plan

        Some prose explanation of the proposals...

        ## Proposal List

        Here are the proposals:

        ```json proposal_index
        {fence}
        ```

        ## Conclusion

        These proposals are ranked by impact.
    """)


@pytest.fixture
def markdown_with_table():
    return textwrap.dedent("""\
        # Priority Ranking

        | Rank | ID | Title | Impact |
        |---|---|---|---|
        | 1 | P1 | Add caching | high |
        | 2 | P2 | Batch queries | medium |
        | 3 | P3 | Rewrite auth | high |
    """)


class TestStrategyA:
    def test_parse_sidecar_json(self, tmp_path, sample_index):
        out = tmp_path / "outputs"
        out.mkdir()
        write_proposal_index(out / "proposals.json", sample_index)
        result = parse_proposals(tmp_path)
        assert result is not None
        assert result.total_count == 2
        assert result.groups[0].proposals[0].id == "P1"

    def test_parse_proposal_file_direct(self, tmp_path, sample_index):
        path = tmp_path / "proposals.json"
        write_proposal_index(path, sample_index)
        result = parse_proposal_file(path)
        assert result is not None
        assert len(result.all_proposals()) == 2

    def test_parse_nonexistent_returns_none(self, tmp_path):
        result = parse_proposal_file(tmp_path / "nope.json")
        assert result is None

    def test_parse_malformed_json_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        result = parse_proposal_file(path)
        assert result is None


class TestStrategyB:
    def test_extract_fence_from_markdown(self, markdown_with_fence, sample_index):
        result = parse_proposal_index_from_text(markdown_with_fence)
        assert result is not None
        assert result.total_count == 2
        assert result.groups[0].proposals[0].id == "P1"
        assert result.groups[0].proposals[1].title == "Batch queries"

    def test_no_fence_returns_none(self):
        result = parse_proposal_index_from_text("# Just prose\nNo fence here.")
        assert result is None

    def test_malformed_fence_returns_none(self):
        text = "```json proposal_index\n{bad json{{{\n```"
        result = parse_proposal_index_from_text(text)
        assert result is None

    def test_fence_with_extra_text_on_line(self, sample_index):
        fence = json.dumps(sample_index.to_dict())
        text = f"```json proposal_index (structured output)\n{fence}\n```"
        result = parse_proposal_index_from_text(text)
        assert result is not None
        assert result.total_count == 2

    def test_fallback_to_strategy_b(self, tmp_path, markdown_with_fence):
        (tmp_path / "outputs").mkdir()
        (tmp_path / "outputs" / "unified_plan.md").write_text(markdown_with_fence)
        result = parse_proposals(tmp_path)
        assert result is not None
        assert result.total_count == 2


class TestStrategyC:
    def test_parse_ranking_table(self, markdown_with_table):
        from agent_foundation.common.data_models.proposal.parser import _strategy_c
        result = _strategy_c(markdown_with_table)
        assert result is not None
        assert result.total_count == 3
        proposals = result.all_proposals()
        assert proposals[0].id == "P1"
        assert proposals[0].rank == 1
        assert proposals[0].title == "Add caching"
        assert proposals[2].id == "P3"

    def test_no_table_returns_none(self):
        from agent_foundation.common.data_models.proposal.parser import _strategy_c
        result = _strategy_c("# No table here\nJust prose.")
        assert result is None

    def test_fallback_to_strategy_c(self, tmp_path, markdown_with_table):
        (tmp_path / "outputs").mkdir()
        (tmp_path / "outputs" / "unified_plan.md").write_text(markdown_with_table)
        result = parse_proposals(tmp_path)
        assert result is not None
        assert result.total_count == 3
        assert "parsed-from-ranking-table-only" in result.warnings


class TestAtomicWrite:
    def test_write_creates_valid_json(self, tmp_path, sample_index):
        path = tmp_path / "proposals.json"
        write_proposal_index(path, sample_index)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == "1"
        assert data["total_count"] == 2

    def test_write_creates_parent_dirs(self, tmp_path, sample_index):
        path = tmp_path / "deep" / "nested" / "proposals.json"
        write_proposal_index(path, sample_index)
        assert path.exists()

    def test_write_no_tmp_file_left(self, tmp_path, sample_index):
        path = tmp_path / "proposals.json"
        write_proposal_index(path, sample_index)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_write_round_trip(self, tmp_path, sample_index):
        path = tmp_path / "proposals.json"
        write_proposal_index(path, sample_index)
        loaded = parse_proposal_file(path)
        assert loaded is not None
        assert loaded.total_count == sample_index.total_count
        assert loaded.groups[0].proposals[0].id == "P1"
        assert loaded.groups[0].proposals[0].impact == "high"


class TestMakeEmptyIndex:
    def test_empty_index_has_metadata(self):
        idx = make_empty_index(source_workspace="/tmp",
                                warnings=["test-warning"])
        assert idx.total_count == 0
        assert idx.source_workspace == "/tmp"
        assert "test-warning" in idx.warnings
        assert idx.created_at != ""

    def test_empty_index_round_trips(self):
        idx = make_empty_index()
        d = idx.to_dict()
        idx2 = ProposalIndex.from_dict(d)
        assert idx2.total_count == 0


class TestParseProposalsPriority:
    def test_strategy_a_wins_over_b(self, tmp_path, sample_index):
        """When both sidecar and fence exist, sidecar (A) wins."""
        out = tmp_path / "outputs"
        out.mkdir()
        write_proposal_index(out / "proposals.json", sample_index)
        different_index = ProposalIndex(
            version="1", total_count=99,
            groups=[ProposalGroup(phase=1, label="Different",
                                  proposals=[Proposal(id="X1", rank=1, title="Other")])],
        )
        fence = json.dumps(different_index.to_dict(), indent=2)
        (out / "unified_plan.md").write_text(
            f"```json proposal_index\n{fence}\n```"
        )
        result = parse_proposals(tmp_path)
        assert result is not None
        assert result.total_count == 2  # sidecar wins (2), not fence (99)

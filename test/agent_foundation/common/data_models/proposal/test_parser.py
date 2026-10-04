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
                phase=1,
                label="Quick Wins",
                proposals=[
                    Proposal(
                        id="P1",
                        rank=1,
                        title="Add caching",
                        impact="high",
                        complexity="low",
                    ),
                    Proposal(
                        id="P2",
                        rank=2,
                        title="Batch queries",
                        impact="medium",
                        complexity="low",
                    ),
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


class TestLargeRealisticFence:
    """Commit 1 / D1: a large research-propose markdown (~80 KB) with a sizable
    proposal_index fence (incl. real LLM constraint dialects) must parse fully.

    This guards the truncated-response file-fallback path: the file on disk is
    big, and extraction must succeed end-to-end (Strategy B + D2 tolerance).
    """

    def _build_large_index_dict(self, n_proposals: int = 40) -> dict:
        groups = []
        for phase in range(1, 5):
            proposals = []
            for i in range(n_proposals // 4):
                pid = f"P{phase}_{i}"
                proposals.append(
                    {
                        "id": pid,
                        "rank": phase * 100 + i,
                        "title": f"Proposal {pid} — optimize component {i}",
                        "summary": "Lorem ipsum dolor sit amet, " * 8,
                        "impact": "high" if i % 2 else "medium",
                        "complexity": "low" if i % 3 else "high",
                        "approach": "Refactor the module and add a cache. " * 6,
                    }
                )
            groups.append(
                {"phase": phase, "label": f"Phase {phase}", "proposals": proposals}
            )
        # Mix canonical + dialect-alpha + dialect-beta constraints (D2).
        constraints = [
            {
                "id": "C1",
                "kind": "requires",
                "proposal_ids": ["P1_0"],
                "requires_ids": ["P2_0"],
            },
            {"type": "ordering", "rule": "P1_0 must precede every other proposal."},
            {
                "type": "requires",
                "from": "P3_0",
                "to": ["P1_0", "P2_0"],
                "note": "depends on earlier phases",
            },
            {"type": "recommends", "from": "P4_0", "to": "P1_0"},
        ]
        return {
            "version": "1",
            "total_count": n_proposals,
            "groups": groups,
            "constraints": constraints,
        }

    def test_parse_large_markdown_with_fence(self):
        index_dict = self._build_large_index_dict(40)
        fence_json = json.dumps(index_dict, indent=2)
        # Pad with prose front and back so the document is comfortably large.
        prose = ("This section discusses the rationale at length. " * 40 + "\n") * 30
        markdown = (
            "# Unified Research Plan\n\n"
            + prose
            + "\n```json proposal_index\n"
            + fence_json
            + "\n```\n\n"
            + "## Appendix\n\n"
            + prose
        )
        assert len(markdown) > 70_000, "fixture should be a large document"
        assert len(fence_json) > 10_000, "fence should be sizable"

        result = parse_proposal_index_from_text(markdown)
        assert result is not None
        assert result.total_count == 40
        assert len(result.all_proposals()) == 40
        # Constraints: all 4 dialects parsed (none dropped — they're all dicts).
        assert len(result.constraints) == 4
        # Dialect beta with list `to` normalised to a list.
        beta = next(c for c in result.constraints if c.proposal_ids == ["P3_0"])
        assert beta.requires_ids == ["P1_0", "P2_0"]


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
        idx = make_empty_index(source_workspace="/tmp", warnings=["test-warning"])
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
            version="1",
            total_count=99,
            groups=[
                ProposalGroup(
                    phase=1,
                    label="Different",
                    proposals=[Proposal(id="X1", rank=1, title="Other")],
                )
            ],
        )
        fence = json.dumps(different_index.to_dict(), indent=2)
        (out / "unified_plan.md").write_text(f"```json proposal_index\n{fence}\n```")
        result = parse_proposals(tmp_path)
        assert result is not None
        assert result.total_count == 2  # sidecar wins (2), not fence (99)


# ---------------------------------------------------------------------------
# AF → hub canonicalizer (inverse of canonicalize_proposal_index_dict)
# ---------------------------------------------------------------------------


class TestCanonicalizeProposalIndexToHubDict:
    """Test the AF → hub dialect converter used at the OpenStartup hub boundary."""

    @staticmethod
    def _af_input():
        # Mirrors ProposalIndex.to_dict() output shape.
        return {
            "version": "1",
            "total_count": 3,
            "groups": [
                {
                    "phase": 1,
                    "label": "Attention",
                    "description": "attn bets",
                    "proposals": [
                        {
                            "id": "P1",
                            "rank": 1,
                            "title": "Diagnostics",
                            "summary": "sum",
                            "impact": "high",
                            "complexity": "low",
                            "problem": "no baselines",
                            "approach": "instrument",
                            "dependencies": [],
                            "tags": ["diag"],
                        },
                        {"id": "P2", "rank": 2, "title": "Differential SiLU"},
                    ],
                    "batches": [
                        {"id": "b1", "label": "Diag", "proposal_ids": ["P1"]},
                    ],
                },
                {
                    "phase": 2,
                    "label": "Sequence",
                    "proposals": [
                        {"id": "P11", "rank": 1, "title": "Temporal"},
                    ],
                },
            ],
            "constraints": [
                {
                    "id": "c1",
                    "kind": "conflicts",
                    "proposal_ids": ["P1", "P2"],
                    "reason": "both touch hstu.py",
                    "severity": "error",
                },
            ],
            "themes": ["attn"],
            "warnings": [],
        }

    def test_renames_groups_to_phases(self):
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        out = canonicalize_proposal_index_to_hub_dict(self._af_input())
        assert "phases" in out
        assert "groups" not in out
        assert len(out["phases"]) == 2
        assert out["phases"][0]["label"] == "Attention"

    def test_renames_batch_proposal_ids_to_hypothesis_ids(self):
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        out = canonicalize_proposal_index_to_hub_dict(self._af_input())
        batch = out["phases"][0]["batches"][0]
        assert "hypothesis_ids" in batch
        assert "proposal_ids" not in batch
        assert batch["hypothesis_ids"] == ["P1"]

    def test_renames_constraints_to_combo_constraints(self):
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        out = canonicalize_proposal_index_to_hub_dict(self._af_input())
        assert "combo_constraints" in out
        assert "constraints" not in out
        c = out["combo_constraints"][0]
        assert c["id"] == "c1"
        assert c["hypothesis_ids"] == ["P1", "P2"]
        assert "proposal_ids" not in c
        # severity + reason preserved
        assert c["severity"] == "error"
        assert c["reason"] == "both touch hstu.py"

    def test_ids_are_NOT_rewritten(self):
        """Intentionally does NOT map P# → H# (unlike the inverse direction)."""
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        out = canonicalize_proposal_index_to_hub_dict(self._af_input())
        all_ids = [p["id"] for ph in out["phases"] for p in ph["proposals"]]
        assert all_ids == ["P1", "P2", "P11"]
        # Batch and constraint id lists also stay P#
        assert out["phases"][0]["batches"][0]["hypothesis_ids"] == ["P1"]
        assert out["combo_constraints"][0]["hypothesis_ids"] == ["P1", "P2"]

    def test_idempotent_on_already_hub_input(self):
        """Double-canonicalize should be a fixed point for hub-shaped data."""
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        af = self._af_input()
        hub = canonicalize_proposal_index_to_hub_dict(af)
        hub_again = canonicalize_proposal_index_to_hub_dict(hub)
        # Structural equality on the canonical fields.
        assert hub_again["phases"][0]["label"] == hub["phases"][0]["label"]
        assert (
            hub_again["phases"][0]["batches"][0]["hypothesis_ids"]
            == (hub["phases"][0]["batches"][0]["hypothesis_ids"])
        )
        assert (
            hub_again["combo_constraints"][0]["hypothesis_ids"]
            == (hub["combo_constraints"][0]["hypothesis_ids"])
        )
        # No spurious `groups` or `constraints` key reintroduced.
        assert "groups" not in hub_again
        assert "constraints" not in hub_again

    def test_spread_preserves_hub_only_fields(self):
        """Proposals must pass through by shallow copy so hub-only fields survive.

        Simulates a partially-hub payload where a proposal already carries the
        hub-only fields (theme, source_workers, slots, includes, probability,
        one_line_summary, notes, _overrideMeta, deprioritized).
        """
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        af = self._af_input()
        af["groups"][0]["proposals"][0].update(
            {
                "theme": "Attention",
                "source_workers": ["w1", "w2"],
                "slots": ["attention"],
                "includes": ["P2"],
                "probability": 0.9,
                "one_line_summary": "line",
                "notes": "note",
                "_overrideMeta": {"newRank": 5, "oldRank": 1},
                "deprioritized": False,
            }
        )
        out = canonicalize_proposal_index_to_hub_dict(af)
        p1 = out["phases"][0]["proposals"][0]
        for k in [
            "theme",
            "source_workers",
            "slots",
            "includes",
            "probability",
            "one_line_summary",
            "notes",
            "_overrideMeta",
            "deprioritized",
        ]:
            assert k in p1, f"missing preserved field: {k}"
        # Existing fields also preserved.
        assert p1["tags"] == ["diag"]

    def test_missing_input_handled_gracefully(self):
        """Non-dict inputs pass through unchanged; missing keys → empty."""
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        assert canonicalize_proposal_index_to_hub_dict(None) is None
        assert canonicalize_proposal_index_to_hub_dict("string") == "string"
        # Empty dict → empty canonical hub shape.
        empty = canonicalize_proposal_index_to_hub_dict({})
        assert empty["phases"] == []
        assert empty["combo_constraints"] == []

    def test_input_not_mutated(self):
        """Original dict is left untouched; a new normalised dict is returned."""
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        af = self._af_input()
        af_snapshot = json.dumps(af, sort_keys=True)
        canonicalize_proposal_index_to_hub_dict(af)
        assert json.dumps(af, sort_keys=True) == af_snapshot

    def test_preserves_unknown_top_level_keys(self):
        """Forward-compatible: unknown keys pass through."""
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_to_hub_dict,
        )

        af = self._af_input()
        af["some_future_key"] = "hello"
        out = canonicalize_proposal_index_to_hub_dict(af)
        assert out["some_future_key"] == "hello"

    def test_round_trip_hub_to_af_to_hub(self):
        """canonicalize_proposal_index_dict then to_hub_dict → structurally hub-canonical."""
        from agent_foundation.common.data_models.proposal.parser import (
            canonicalize_proposal_index_dict,
            canonicalize_proposal_index_to_hub_dict,
        )

        hub_input = {
            "phases": [
                {
                    "phase": 1,
                    "label": "P1",
                    "proposals": [{"id": "P1", "title": "T"}],
                    "batches": [{"id": "b1", "hypothesis_ids": ["P1"]}],
                },
            ],
            "combo_constraints": [
                {"id": "c1", "kind": "conflicts", "hypothesis_ids": ["P1"]}
            ],
        }
        af = canonicalize_proposal_index_dict(hub_input)
        # AF-canonical: groups + constraints
        assert "groups" in af
        assert "constraints" in af
        back_to_hub = canonicalize_proposal_index_to_hub_dict(af)
        # Round trip yields hub shape again
        assert "phases" in back_to_hub
        assert "combo_constraints" in back_to_hub
        assert back_to_hub["phases"][0]["batches"][0]["hypothesis_ids"] == ["P1"]


class TestAttachProposalFileAbs:
    """Tests for ``attach_proposal_file_abs`` — the per-proposal file-path
    enricher that lets the widget lazy-fetch ``P{N}.md`` via ``/api/view``.
    """

    def _build_workspace(self, tmp_path, proposal_files: dict[str, str]) -> tuple:
        """Create a fake proposals workspace: proposals.json + proposals/<file>."""
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        proposals_dir = outputs / "proposals"
        proposals_dir.mkdir()
        for name, content in proposal_files.items():
            (proposals_dir / name).write_text(content, encoding="utf-8")
        proposals_json = outputs / "proposals.json"
        data = {
            "version": "1",
            "total_count": len(proposal_files),
            "groups": [
                {
                    "phase": 1,
                    "label": "Quick Wins",
                    "proposals": [
                        {
                            "id": f"P{i + 1}",
                            "rank": i + 1,
                            "title": name,
                            "proposal_file": f"proposals/{name}",
                        }
                        for i, name in enumerate(proposal_files)
                    ],
                }
            ],
        }
        proposals_json.write_text(json.dumps(data), encoding="utf-8")
        return proposals_json, data

    def test_attaches_abs_for_every_existing_file(self, tmp_path):
        """Every proposal whose file exists gets `proposal_file_abs` = resolved path."""
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        proposals_json, data = self._build_workspace(
            tmp_path,
            {
                "P1.md": "# P1\nContent.",
                "P2.md": "# P2\nContent.",
                "P3.md": "# P3\nContent.",
            },
        )
        attach_proposal_file_abs(data, proposals_json)
        proposals = data["groups"][0]["proposals"]
        assert len(proposals) == 3
        for i, p in enumerate(proposals, 1):
            expected = str((tmp_path / "outputs" / "proposals" / f"P{i}.md").resolve())
            assert p["proposal_file_abs"] == expected

    def test_omits_when_file_missing(self, tmp_path):
        """A proposal whose `proposal_file` points to a missing file gets NO key."""
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        proposals_json, data = self._build_workspace(
            tmp_path,
            {"P1.md": "exists"},
        )
        # Add a proposal pointing to a nonexistent file
        data["groups"][0]["proposals"].append(
            {
                "id": "P2",
                "rank": 2,
                "title": "Missing",
                "proposal_file": "proposals/P2.md",
            }
        )
        attach_proposal_file_abs(data, proposals_json)
        proposals = data["groups"][0]["proposals"]
        assert "proposal_file_abs" in proposals[0]
        assert "proposal_file_abs" not in proposals[1]

    def test_path_traversal_omitted(self, tmp_path):
        """A `proposal_file` with `../..` escape is refused (key omitted)."""
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        proposals_json, data = self._build_workspace(tmp_path, {})
        data["groups"][0]["proposals"].append(
            {
                "id": "PMAL",
                "rank": 1,
                "title": "Malicious",
                "proposal_file": "../../etc/passwd",
            }
        )
        attach_proposal_file_abs(data, proposals_json)
        proposals = data["groups"][0]["proposals"]
        assert "proposal_file_abs" not in proposals[0]

    def test_value_is_str_not_path(self, tmp_path):
        """`proposal_file_abs` must be a plain `str` (JSON-serializable), not `Path`."""
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        proposals_json, data = self._build_workspace(tmp_path, {"P1.md": "x"})
        attach_proposal_file_abs(data, proposals_json)
        val = data["groups"][0]["proposals"][0]["proposal_file_abs"]
        assert isinstance(val, str)
        # Must round-trip through json without a TypeError
        assert json.dumps(data)  # would raise on Path values

    def test_absent_proposal_file_omitted(self, tmp_path):
        """Proposals without a `proposal_file` field pass through untouched."""
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        proposals_json, data = self._build_workspace(tmp_path, {})
        data["groups"][0]["proposals"].append(
            {
                "id": "P1",
                "rank": 1,
                "title": "No file",
            }
        )
        attach_proposal_file_abs(data, proposals_json)
        p = data["groups"][0]["proposals"][0]
        assert "proposal_file_abs" not in p
        assert "proposal_file" not in p

    def test_non_dict_input_noop(self):
        """Non-dict `proposals` → no-op (no crash)."""
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        attach_proposal_file_abs(None, "/tmp/x.json")  # type: ignore[arg-type]
        attach_proposal_file_abs([], "/tmp/x.json")  # type: ignore[arg-type]
        attach_proposal_file_abs("string", "/tmp/x.json")  # type: ignore[arg-type]
        # If we reach here, no exception was raised.

    def test_malformed_groups_shape_skipped(self, tmp_path):
        """Non-list `groups` or non-dict entries are skipped without raising."""
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        # groups is a dict (bad), not a list
        proposals_json = tmp_path / "outputs" / "proposals.json"
        proposals_json.parent.mkdir()
        proposals_json.write_text("{}")
        bad = {"groups": {"phase": 1}}
        attach_proposal_file_abs(bad, proposals_json)
        assert bad == {"groups": {"phase": 1}}
        # groups is a list but entries are strings (bad)
        bad2: dict = {"groups": ["not-a-dict"]}
        attach_proposal_file_abs(bad2, proposals_json)
        assert bad2 == {"groups": ["not-a-dict"]}

    def test_phases_dialect_handled(self, tmp_path):
        """Helper also walks `phases` (hub dialect) not just `groups`."""
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        proposals_json, _ = self._build_workspace(tmp_path, {"P1.md": "x"})
        # Build a hub-dialect dict (uses `phases`, not `groups`)
        hub_data = {
            "phases": [
                {
                    "phase": 1,
                    "label": "Q",
                    "proposals": [
                        {
                            "id": "P1",
                            "rank": 1,
                            "title": "T",
                            "proposal_file": "proposals/P1.md",
                        },
                    ],
                }
            ]
        }
        attach_proposal_file_abs(hub_data, proposals_json)
        p = hub_data["phases"][0]["proposals"][0]
        assert "proposal_file_abs" in p
        assert p["proposal_file_abs"].endswith("proposals/P1.md")

    def test_idempotent_on_repeat_call(self, tmp_path):
        """Calling the helper twice on the same data yields identical state.

        Load-bearing for the restore path (M2): after session-resume the widget
        marker is re-parsed and re-enriched from disk; a non-idempotent helper
        would corrupt state on the second run. The plan documents enrichment
        as idempotent — this test guards that invariant.
        """
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        proposals_json, data = self._build_workspace(
            tmp_path,
            {"P1.md": "one", "P2.md": "two", "P3.md": "three"},
        )
        attach_proposal_file_abs(data, proposals_json)
        snapshot = json.dumps(data, sort_keys=True)
        # Second call — mirrors what happens on a session-resume re-enrichment
        attach_proposal_file_abs(data, proposals_json)
        assert json.dumps(data, sort_keys=True) == snapshot

    def test_preexisting_abs_preserved_when_file_still_exists(self, tmp_path):
        """If `proposal_file_abs` is already set (e.g., restored from a persisted
        marker) AND the referenced file still exists, the helper re-derives to
        the same value — the restore path stays coherent.
        """
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
        )

        proposals_json, data = self._build_workspace(
            tmp_path,
            {"P1.md": "content"},
        )
        # Simulate a restored payload that already has proposal_file_abs
        pre_computed = str((tmp_path / "outputs" / "proposals" / "P1.md").resolve())
        data["groups"][0]["proposals"][0]["proposal_file_abs"] = pre_computed
        attach_proposal_file_abs(data, proposals_json)
        assert data["groups"][0]["proposals"][0]["proposal_file_abs"] == pre_computed

    def test_canonicalize_preserves_proposal_file_abs(self, tmp_path):
        """The dashboard-parity keystone: after `attach_proposal_file_abs` mutates a
        `groups`-shaped AF-native dict, `canonicalize_proposal_index_to_hub_dict`
        MUST preserve `proposal_file_abs` on every `phases[].proposals[]`.
        Without this invariant, the Experiment Hub Selection tab won't render the
        full-proposal doc (the widget's `metadata.proposals` comes via the
        canonicalize spread).
        """
        from agent_foundation.common.data_models.proposal.parser import (
            attach_proposal_file_abs,
            canonicalize_proposal_index_to_hub_dict,
        )

        proposals_json, data = self._build_workspace(
            tmp_path,
            {"P1.md": "one", "P2.md": "two", "P3.md": "three"},
        )
        attach_proposal_file_abs(data, proposals_json)
        hub = canonicalize_proposal_index_to_hub_dict(data)
        assert "phases" in hub
        phases = hub["phases"]
        assert len(phases) == 1
        assert len(phases[0]["proposals"]) == 3
        for p in phases[0]["proposals"]:
            assert "proposal_file_abs" in p
            assert p["proposal_file_abs"].endswith(
                f"proposals/{p['proposal_file'].split('/')[-1]}"
            )


class TestSummaryNormalizer:
    """Tests for D6 — LLM `summary` field hardening at the parse boundary."""

    def _build_index_with_summary(
        self, summary: str, problem: str = "", title: str = "T"
    ) -> dict:
        return {
            "version": "1",
            "total_count": 1,
            "groups": [
                {
                    "phase": 1,
                    "label": "L",
                    "proposals": [
                        {
                            "id": "P1",
                            "rank": 1,
                            "title": title,
                            "summary": summary,
                            "problem": problem,
                        }
                    ],
                }
            ],
        }

    def test_table_row_summary_replaced_by_problem_sentence(self):
        from agent_foundation.common.data_models.proposal.parser import (
            _normalize_proposal_index_summaries,
        )

        data = self._build_index_with_summary(
            "| A | B | C |",
            problem="This is the first sentence. This is the second.",
        )
        _normalize_proposal_index_summaries(data)
        p = data["groups"][0]["proposals"][0]
        assert p["summary"] == "This is the first sentence."

    def test_bold_only_heading_replaced(self):
        from agent_foundation.common.data_models.proposal.parser import (
            _normalize_proposal_index_summaries,
        )

        data = self._build_index_with_summary(
            "**Metrics**",
            problem="A short problem. Another sentence.",
        )
        _normalize_proposal_index_summaries(data)
        p = data["groups"][0]["proposals"][0]
        assert p["summary"] == "A short problem."

    def test_truncated_summary_replaced_by_problem(self):
        from agent_foundation.common.data_models.proposal.parser import (
            _normalize_proposal_index_summaries,
        )

        data = self._build_index_with_summary(
            "This is truncated mid…",
            problem="Complete problem statement. And more.",
        )
        _normalize_proposal_index_summaries(data)
        p = data["groups"][0]["proposals"][0]
        assert p["summary"] == "Complete problem statement."

    def test_triple_dot_truncated_summary_replaced(self):
        from agent_foundation.common.data_models.proposal.parser import (
            _normalize_proposal_index_summaries,
        )

        data = self._build_index_with_summary(
            "Half a sentence...",
            problem="Full sentence here.",
        )
        _normalize_proposal_index_summaries(data)
        p = data["groups"][0]["proposals"][0]
        assert p["summary"] == "Full sentence here."

    def test_falls_back_to_title_when_problem_empty(self):
        from agent_foundation.common.data_models.proposal.parser import (
            _normalize_proposal_index_summaries,
        )

        data = self._build_index_with_summary(
            "| bad |", problem="", title="Fallback title"
        )
        _normalize_proposal_index_summaries(data)
        p = data["groups"][0]["proposals"][0]
        assert p["summary"] == "Fallback title"

    def test_good_summary_preserved(self):
        """A well-formed summary is not modified."""
        from agent_foundation.common.data_models.proposal.parser import (
            _normalize_proposal_index_summaries,
        )

        good = "A proper prose one-liner about the proposal."
        data = self._build_index_with_summary(good, problem="does not matter")
        _normalize_proposal_index_summaries(data)
        assert data["groups"][0]["proposals"][0]["summary"] == good

    def test_warning_on_truncated_prose(self, caplog):
        """Any prose field ending with `…` emits a warning (surfacing LLM misbehavior)."""
        import logging as _logging

        from agent_foundation.common.data_models.proposal.parser import (
            _normalize_proposal_index_summaries,
        )

        data = self._build_index_with_summary(
            "Good summary.",
            problem="Truncated problem…",
        )
        with caplog.at_level(
            _logging.WARNING,
            logger="agent_foundation.common.data_models.proposal.parser",
        ):
            _normalize_proposal_index_summaries(data)
        # Warning should mention the truncation marker + which field
        assert any("truncation" in r.message.lower() for r in caplog.records)

    def test_parse_proposal_file_applies_normalizer(self, tmp_path):
        """Integration: parse_proposal_file runs the normalizer end-to-end."""
        from agent_foundation.common.data_models.proposal.parser import (
            parse_proposal_file,
        )

        data = {
            "version": "1",
            "total_count": 1,
            "groups": [
                {
                    "phase": 1,
                    "label": "L",
                    "proposals": [
                        {
                            "id": "P1",
                            "rank": 1,
                            "title": "T",
                            "summary": "| junk |",
                            "problem": "Real problem statement here.",
                        }
                    ],
                }
            ],
        }
        p = tmp_path / "proposals.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        index = parse_proposal_file(p)
        assert index is not None
        assert index.groups[0].proposals[0].summary == "Real problem statement here."

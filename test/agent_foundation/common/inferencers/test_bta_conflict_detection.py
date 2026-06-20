"""Tests for BTA conflict detection helpers.

Tests _canonicalize_text, _sha256_of_file_canonical, _detect_conflicts_and_promote,
and make_conflict_aware_prompt_builder.
"""

import os
import shutil

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    _canonicalize_text,
    _sha256_of_file_canonical,
    _detect_conflicts_and_promote,
    make_conflict_aware_prompt_builder,
)


class TestCanonicalizeText:
    def test_strips_trailing_whitespace(self):
        result = _canonicalize_text(b"hello   \nworld  \n")
        assert result == b"hello\nworld\n"

    def test_normalizes_crlf(self):
        result = _canonicalize_text(b"hello\r\nworld\r\n")
        assert result == b"hello\nworld\n"

    def test_normalizes_cr(self):
        result = _canonicalize_text(b"hello\rworld\r")
        assert result == b"hello\nworld\n"

    def test_ensures_trailing_newline(self):
        result = _canonicalize_text(b"hello\nworld")
        assert result == b"hello\nworld\n"

    def test_binary_passthrough(self):
        binary = bytes(range(256))
        result = _canonicalize_text(binary)
        assert result == binary

    def test_empty_content(self):
        result = _canonicalize_text(b"")
        assert result == b"\n"


class TestSha256Canonical:
    def test_identical_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\nworld\n")
        f2.write_text("hello\nworld\n")
        assert _sha256_of_file_canonical(str(f1)) == _sha256_of_file_canonical(str(f2))

    def test_trailing_whitespace_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\nworld\n")
        f2.write_text("hello   \nworld  \n")
        assert _sha256_of_file_canonical(str(f1)) == _sha256_of_file_canonical(str(f2))

    def test_crlf_vs_lf_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"hello\nworld\n")
        f2.write_bytes(b"hello\r\nworld\r\n")
        assert _sha256_of_file_canonical(str(f1)) == _sha256_of_file_canonical(str(f2))

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\n")
        f2.write_text("world\n")
        assert _sha256_of_file_canonical(str(f1)) != _sha256_of_file_canonical(str(f2))


@pytest.fixture
def worker_tree(tmp_path):
    """Create a mock worker output tree for conflict detection tests."""
    children = tmp_path / "children"

    w0 = children / "worker_0" / "outputs" / "final_deliverables"
    w0.mkdir(parents=True)
    (w0 / "skills" / "alpha").mkdir(parents=True)
    (w0 / "skills" / "alpha" / "SKILL.md").write_text("# Alpha Skill v1\nShort version.\n")
    (w0 / "tools" / "beta").mkdir(parents=True)
    (w0 / "tools" / "beta" / "tool.json").write_text('{"name": "beta"}\n')

    w1 = children / "worker_1" / "outputs" / "final_deliverables"
    w1.mkdir(parents=True)
    (w1 / "skills" / "alpha").mkdir(parents=True)
    (w1 / "skills" / "alpha" / "SKILL.md").write_text("# Alpha Skill v1\nShort version.\n")
    (w1 / "tools" / "gamma").mkdir(parents=True)
    (w1 / "tools" / "gamma" / "tool.json").write_text('{"name": "gamma"}\n')

    w2 = children / "worker_2" / "outputs" / "final_deliverables"
    w2.mkdir(parents=True)
    (w2 / "skills" / "alpha").mkdir(parents=True)
    (w2 / "skills" / "alpha" / "SKILL.md").write_text(
        "# Alpha Skill v2\nMuch longer and more complete version with extra details.\n"
    )

    dst = tmp_path / "deliverables"
    dst.mkdir()

    return tmp_path, children, dst


class TestDetectConflictsAndPromote:
    def test_agreed_files_promoted(self, worker_tree):
        _, children, dst = worker_tree
        promoted, conflicts = _detect_conflicts_and_promote(str(dst), str(children))
        promoted_paths = {p["path"] for p in promoted}
        assert "tools/beta/tool.json" in promoted_paths
        assert "tools/gamma/tool.json" in promoted_paths
        assert os.path.exists(os.path.join(str(dst), "tools/beta/tool.json"))
        assert os.path.exists(os.path.join(str(dst), "tools/gamma/tool.json"))

    def test_conflicting_files_not_promoted(self, worker_tree):
        _, children, dst = worker_tree
        promoted, conflicts = _detect_conflicts_and_promote(str(dst), str(children))
        assert "skills/alpha/SKILL.md" in conflicts
        assert not os.path.exists(os.path.join(str(dst), "skills/alpha/SKILL.md"))

    def test_conflict_has_all_worker_versions(self, worker_tree):
        _, children, dst = worker_tree
        _, conflicts = _detect_conflicts_and_promote(str(dst), str(children))
        cands = conflicts["skills/alpha/SKILL.md"]
        workers = {c["worker"] for c in cands}
        assert "worker_0" in workers
        assert "worker_1" in workers
        assert "worker_2" in workers

    def test_agreed_between_two_workers(self, worker_tree):
        _, children, dst = worker_tree
        promoted, _ = _detect_conflicts_and_promote(str(dst), str(children))
        alpha_agreed = [p for p in promoted if p["path"] == "tools/beta/tool.json"]
        assert len(alpha_agreed) == 1
        assert alpha_agreed[0]["source_workers"] == ["worker_0"]

    def test_unique_file_promoted(self, worker_tree):
        _, children, dst = worker_tree
        promoted, _ = _detect_conflicts_and_promote(str(dst), str(children))
        gamma = [p for p in promoted if p["path"] == "tools/gamma/tool.json"]
        assert len(gamma) == 1

    def test_empty_children_dir(self, tmp_path):
        children = tmp_path / "children"
        children.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        promoted, conflicts = _detect_conflicts_and_promote(str(dst), str(children))
        assert promoted == []
        assert conflicts == {}

    def test_single_worker(self, tmp_path):
        children = tmp_path / "children"
        w0 = children / "worker_0" / "outputs"
        w0.mkdir(parents=True)
        (w0 / "result.md").write_text("content\n")
        dst = tmp_path / "dst"
        dst.mkdir()
        promoted, conflicts = _detect_conflicts_and_promote(str(dst), str(children))
        assert len(promoted) == 1
        assert conflicts == {}

    def test_prefers_final_deliverables_over_outputs(self, tmp_path):
        children = tmp_path / "children"
        w0_fd = children / "worker_0" / "outputs" / "final_deliverables"
        w0_fd.mkdir(parents=True)
        (w0_fd / "good.md").write_text("from final_deliverables\n")
        w0_out = children / "worker_0" / "outputs"
        (w0_out / "bad.md").write_text("from outputs root\n")
        dst = tmp_path / "dst"
        dst.mkdir()
        promoted, _ = _detect_conflicts_and_promote(str(dst), str(children))
        promoted_paths = {p["path"] for p in promoted}
        assert "good.md" in promoted_paths
        assert "bad.md" not in promoted_paths


class TestMakeConflictAwarePromptBuilder:
    def test_last_writer_wins_returns_text(self):
        builder = make_conflict_aware_prompt_builder(conflict_resolution_mode="last_writer_wins")
        result = builder(["result1", "result2"])
        assert "### Result 1" in result
        assert "result1" in result

    def test_no_worker_paths_degrades_to_text(self):
        builder = make_conflict_aware_prompt_builder()
        result = builder(["result1"], worker_output_paths=None)
        assert "### Result 1" in result

    def test_delegate_mode_with_real_files(self, worker_tree):
        root, children, dst = worker_tree
        w0_output = os.path.join(
            str(children), "worker_0", "outputs", "final_deliverables", "skills", "alpha", "SKILL.md"
        )

        # Mock BTA with aggregator_inferencer that has template_extra_feed
        class MockAggregator:
            template_extra_feed = {}

        class MockBTA:
            aggregator_inferencer = MockAggregator()

        mock_bta = MockBTA()

        builder = make_conflict_aware_prompt_builder(
            conflict_resolution_mode="delegate_to_aggregator",
        )
        result = builder(
            ["summary from worker 0", "summary from worker 1", "summary from worker 2"],
            worker_output_paths=[w0_output, None, None],
            bta=mock_bta,
        )
        # Option 2: return string is worker summaries only
        assert "### Result 1" in result
        assert "summary from worker 0" in result

        # Structured data injected into template_extra_feed
        feed = mock_bta.aggregator_inferencer.template_extra_feed
        assert "deliverables_promoted" in feed
        assert "deliverables_with_conflicts" in feed
        assert "deliverables_dst" in feed
        assert "worker_summaries" in feed

        # Verify conflict was detected for skills/alpha/SKILL.md
        conflict_paths = [c["path"] for c in feed["deliverables_with_conflicts"]]
        assert "skills/alpha/SKILL.md" in conflict_paths

        # Verify agreed files were promoted
        promoted_paths = [p["path"] for p in feed["deliverables_promoted"]]
        assert "tools/beta/tool.json" in promoted_paths

    def test_local_access_aggregator_uses_paths_not_inline(self, worker_tree):
        """When aggregator has_local_access=True, worker results should be
        referenced by path (not inlined) to avoid ARG_MAX errors."""
        root, children, dst = worker_tree
        w0_path = os.path.join(
            str(children), "worker_0", "outputs", "final_deliverables", "skills", "alpha", "SKILL.md"
        )
        w1_path = os.path.join(
            str(children), "worker_1", "outputs", "final_deliverables", "tools", "gamma", "tool.json"
        )

        class MockLocalAggregator:
            has_local_access = True
            template_extra_feed = {}

        class MockBTA:
            aggregator_inferencer = MockLocalAggregator()

        builder = make_conflict_aware_prompt_builder(
            conflict_resolution_mode="delegate_to_aggregator",
        )
        large_result = "x" * 500_000  # 500KB — would exceed ARG_MAX if inlined
        result = builder(
            [large_result, "short result"],
            worker_output_paths=[w0_path, w1_path],
            bta=MockBTA(),
        )
        # With local access + paths: should use path references, not inline
        assert "See file:" in result
        assert w0_path in result
        assert w1_path in result
        # The 500KB content should NOT appear in the result
        assert large_result not in result
        # Result should be small (paths only)
        assert len(result) < 10_000

    def test_no_local_access_aggregator_inlines_full_text(self):
        """When aggregator has_local_access=False, full text must be inlined."""

        class MockRemoteAggregator:
            has_local_access = False
            template_extra_feed = {}

        class MockBTA:
            aggregator_inferencer = MockRemoteAggregator()

        builder = make_conflict_aware_prompt_builder(
            conflict_resolution_mode="last_writer_wins",
        )
        result = builder(
            ["full content here", "more content"],
            worker_output_paths=["/some/path", "/other/path"],
            bta=MockBTA(),
        )
        assert "full content here" in result
        assert "more content" in result

    def test_no_bta_falls_back_to_inline(self):
        """When bta=None, must inline full text (no local access info)."""
        builder = make_conflict_aware_prompt_builder(
            conflict_resolution_mode="delegate_to_aggregator",
        )
        result = builder(
            ["content A", "content B"],
            worker_output_paths=None,
            bta=None,
        )
        assert "content A" in result
        assert "content B" in result

    def test_mixed_paths_and_none(self, worker_tree):
        """Workers with paths get path refs; workers without get inlined."""
        root, children, dst = worker_tree
        w0_path = os.path.join(
            str(children), "worker_0", "outputs", "final_deliverables", "skills", "alpha", "SKILL.md"
        )

        class MockLocalAgg:
            has_local_access = True
            template_extra_feed = {}

        class MockBTA:
            aggregator_inferencer = MockLocalAgg()

        builder = make_conflict_aware_prompt_builder(
            conflict_resolution_mode="delegate_to_aggregator",
        )
        result = builder(
            ["worker 0 output", "worker 1 output (no path)"],
            worker_output_paths=[w0_path, None],
            bta=MockBTA(),
        )
        # Worker 0: has path → path reference
        assert "See file:" in result
        assert w0_path in result
        # Worker 1: no path → inlined
        assert "worker 1 output (no path)" in result

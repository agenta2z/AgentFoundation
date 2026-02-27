"""Unit tests for BudgetAwareKnowledgeProvider."""

import pytest

from agent_foundation.knowledge.retrieval.knowledge_provider import (
    BudgetAwareKnowledgeProvider,
    CONTEXT_BUDGET,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
)
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece


def _make_scored(content, info_type="context", summary=None, updated_at=None):
    """Helper to create a ScoredPiece with given fields."""
    piece = KnowledgePiece(
        content=content,
        info_type=info_type,
        summary=summary,
        updated_at=updated_at,
    )
    return ScoredPiece(piece=piece, score=1.0, normalized_score=1.0)


class TestBudgetAwareKnowledgeProvider:
    def setup_method(self):
        self.provider = BudgetAwareKnowledgeProvider()

    def test_empty_pieces_returns_empty_string(self):
        result = self.provider.format_knowledge([], 5000)
        assert result == ""

    def test_context_pieces_formatted_as_paragraphs(self):
        pieces = [_make_scored("Some context info.", info_type="context")]
        result = self.provider.format_knowledge(pieces, 5000)
        assert "## Relevant Context" in result
        assert "Some context info." in result

    def test_instructions_formatted_as_bullets(self):
        pieces = [_make_scored("Do this thing.", info_type="instructions")]
        result = self.provider.format_knowledge(pieces, 5000)
        assert "## Instructions" in result
        assert "- Do this thing." in result

    def test_skills_progressive_disclosure_shows_summary(self):
        pieces = [
            _make_scored(
                "Full skill content here that is long.",
                info_type="skills",
                summary="Short summary",
            )
        ]
        result = self.provider.format_knowledge(pieces, 5000)
        assert "## Available Skills" in result
        assert "Short summary" in result

    def test_skills_expands_top_skill_if_budget_allows(self):
        pieces = [
            _make_scored(
                "Full skill content.",
                info_type="skills",
                summary="Summary",
            )
        ]
        result = self.provider.format_knowledge(pieces, 5000)
        assert "(Expanded)" in result
        assert "Full skill content." in result

    def test_episodic_includes_temporal_markers(self):
        pieces = [
            _make_scored(
                "Had a meeting.",
                info_type="episodic",
                updated_at="2024-01-15T10:30:00+00:00",
            )
        ]
        result = self.provider.format_knowledge(pieces, 5000)
        assert "## Recent History" in result
        assert "[2024-01-15]" in result
        assert "Had a meeting." in result

    def test_episodic_unknown_timestamp(self):
        piece = KnowledgePiece(content="Event happened.", info_type="episodic")
        # Override updated_at to None to test fallback
        piece.updated_at = None
        sp = ScoredPiece(piece=piece, score=1.0)
        result = self.provider.format_knowledge([sp], 5000)
        assert "[unknown]" in result

    def test_user_profile_formatted_as_key_value(self):
        pieces = [_make_scored("Prefers dark mode.", info_type="user_profile")]
        result = self.provider.format_knowledge(pieces, 5000)
        assert "## User Preferences" in result
        assert "Prefers dark mode." in result

    def test_overall_budget_enforcement(self):
        # With a very small budget, output should be limited
        pieces = [
            _make_scored("A" * 400, info_type="context"),
            _make_scored("B" * 400, info_type="instructions"),
        ]
        # Budget of 50 tokens = ~200 chars. Only first section should fit.
        result = self.provider.format_knowledge(pieces, 50)
        token_count = len(result) // 4
        assert token_count <= 50

    def test_per_type_budget_enforcement(self):
        # Create many context pieces that exceed the context budget of 3000 tokens
        pieces = [
            _make_scored("X" * 4000, info_type="context")
            for _ in range(10)
        ]
        result = self.provider.format_knowledge(pieces, 100000)
        # The context section should respect its 3000 token budget
        context_tokens = len(result) // 4
        assert context_tokens <= CONTEXT_BUDGET["context"]

    def test_multiple_info_types_in_output(self):
        pieces = [
            _make_scored("Skill content.", info_type="skills", summary="A skill"),
            _make_scored("An instruction.", info_type="instructions"),
            _make_scored("Some context.", info_type="context"),
        ]
        result = self.provider.format_knowledge(pieces, 10000)
        assert "## Available Skills" in result
        assert "## Instructions" in result
        assert "## Relevant Context" in result

    def test_unknown_info_type_is_ignored(self):
        pieces = [_make_scored("Unknown type.", info_type="unknown_type")]
        result = self.provider.format_knowledge(pieces, 5000)
        assert result == ""

    def test_skills_without_summary_uses_content_prefix(self):
        pieces = [
            _make_scored("Full content of the skill.", info_type="skills")
        ]
        result = self.provider.format_knowledge(pieces, 5000)
        assert "Full content of the skill." in result

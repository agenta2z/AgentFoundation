"""
Knowledge provider — formats and injects knowledge into prompts with budget enforcement.

Groups ScoredPieces by info_type and formats each group with a dedicated formatter:
  - skills: progressive disclosure (summaries first, expand top skill if budget allows)
  - instructions: bullet points
  - context: paragraphs
  - episodic: timestamped entries (date prefix from updated_at)
  - user_profile: key-value summaries

Enforces per-info-type token budgets and an overall available_tokens limit.
"""

from collections import defaultdict
from typing import Dict, List

from agent_foundation.knowledge.retrieval.models.results import ScoredPiece
from agent_foundation.knowledge.retrieval.utils import count_tokens


# Per-info-type token budgets with explicit priority order
CONTEXT_BUDGET: Dict[str, int] = {
    "skills": 2000,
    "instructions": 1000,
    "context": 3000,
    "episodic": 1000,
    "user_profile": 500,
}


class BudgetAwareKnowledgeProvider:
    """Formats and injects knowledge into prompts with budget enforcement."""

    def _count_tokens(self, text: str) -> int:
        return count_tokens(text)

    def format_knowledge(
        self,
        pieces: List[ScoredPiece],
        available_tokens: int,
    ) -> str:
        """Format retrieved knowledge with per-type and overall budget enforcement.

        Args:
            pieces: Scored knowledge pieces to format.
            available_tokens: Overall token budget for the output.

        Returns:
            Formatted string with sections for each info_type, respecting budgets.
        """
        by_type: Dict[str, List[ScoredPiece]] = defaultdict(list)
        for piece in pieces:
            info_type = piece.info_type or "context"
            by_type[info_type].append(piece)

        formatted_sections: List[str] = []
        remaining_tokens = available_tokens

        for info_type, budget in CONTEXT_BUDGET.items():
            type_pieces = by_type.get(info_type, [])
            if not type_pieces:
                continue

            type_budget = min(budget, remaining_tokens)
            formatted = self._format_type(type_pieces, info_type, type_budget)
            tokens_used = self._count_tokens(formatted)

            if tokens_used <= remaining_tokens:
                formatted_sections.append(formatted)
                remaining_tokens -= tokens_used

        return "\n\n".join(formatted_sections)

    def _format_type(
        self,
        pieces: List[ScoredPiece],
        info_type: str,
        budget_tokens: int,
    ) -> str:
        """Format pieces of a given type using the appropriate formatter."""
        formatters = {
            "skills": self._format_skills,
            "instructions": self._format_instructions,
            "context": self._format_context,
            "episodic": self._format_episodic,
            "user_profile": self._format_profile,
        }
        formatter = formatters.get(info_type)
        if formatter is None:
            return ""
        return formatter(pieces, budget_tokens)

    def _format_skills(self, pieces: List[ScoredPiece], budget: int) -> str:
        """Skills: Progressive disclosure with budget enforcement.

        Shows summaries first, then expands the top skill if budget allows.
        """
        formatted = ["## Available Skills\n"]

        for piece in pieces:
            summary = piece.piece.summary or piece.piece.content[:100]
            line = f"- **{piece.piece_id}**: {summary}"
            if self._count_tokens("\n".join(formatted + [line])) > budget:
                break
            formatted.append(line)

        tokens_used = self._count_tokens("\n".join(formatted))

        # Expand top skill if budget allows
        if pieces:
            top_skill = pieces[0]
            expansion = (
                f"\n### {top_skill.piece_id} (Expanded)\n{top_skill.piece.content}"
            )
            if tokens_used + self._count_tokens(expansion) <= budget:
                formatted.append(expansion)

        return "\n".join(formatted)

    def _format_instructions(
        self, pieces: List[ScoredPiece], budget: int
    ) -> str:
        """Instructions: Bullet points with budget enforcement."""
        formatted = ["## Instructions\n"]
        for piece in pieces:
            line = f"- {piece.piece.content}"
            if self._count_tokens("\n".join(formatted + [line])) > budget:
                break
            formatted.append(line)
        return "\n".join(formatted)

    def _format_context(self, pieces: List[ScoredPiece], budget: int) -> str:
        """Context: Factual paragraphs with budget enforcement."""
        formatted = ["## Relevant Context\n"]
        for piece in pieces:
            block = piece.piece.content + "\n"
            if self._count_tokens("\n".join(formatted) + block) > budget:
                break
            formatted.append(block)
        return "\n".join(formatted)

    def _format_episodic(self, pieces: List[ScoredPiece], budget: int) -> str:
        """Episodic: With temporal markers (date prefix) and budget enforcement."""
        formatted = ["## Recent History\n"]
        for piece in pieces:
            timestamp = (
                piece.updated_at[:10] if piece.updated_at else "unknown"
            )
            line = f"[{timestamp}] {piece.piece.content}"
            if self._count_tokens("\n".join(formatted + [line])) > budget:
                break
            formatted.append(line)
        return "\n".join(formatted)

    def _format_profile(self, pieces: List[ScoredPiece], budget: int) -> str:
        """User profile: Key-value summaries with budget enforcement."""
        formatted = ["## User Preferences\n"]
        for piece in pieces:
            summary = piece.piece.summary or piece.piece.content[:100]
            line = f"- {summary}"
            if self._count_tokens("\n".join(formatted + [line])) > budget:
                break
            formatted.append(line)
        return "\n".join(formatted)

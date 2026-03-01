"""
Automatic Skill Synthesis from related knowledge pieces.

Detects clusters of similar knowledge pieces and synthesizes them into
coherent procedural skills via LLM analysis. The synthesized skill is
returned as a Procedure-type KnowledgePiece with info_type="skills".

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent_foundation.knowledge.ingestion.prompts.skill_synthesis import (
    SKILL_SYNTHESIS_PROMPT,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore

logger = logging.getLogger(__name__)


@dataclass
class SkillSynthesisConfig:
    """Configuration for skill synthesis."""

    min_pieces_for_skill: int = 3
    min_avg_similarity: float = 0.75
    max_neighbors: int = 10


@dataclass
class SkillSynthesisResult:
    """Result of skill synthesis analysis."""

    is_skill_worthy: bool
    confidence: float
    reasoning: str
    synthesized_skill: Optional[Dict[str, Any]] = None


class SkillSynthesizer:
    """Automatic skill synthesis from knowledge clusters.

    Searches for similar pieces, filters by similarity threshold, checks
    cluster size. Calls LLM to evaluate skill-worthiness and synthesize
    procedure. Creates Procedure-type KnowledgePiece with info_type="skills".
    """

    def __init__(
        self,
        piece_store: KnowledgePieceStore,
        llm_fn: Optional[Callable[[str], str]] = None,
        config: Optional[SkillSynthesisConfig] = None,
    ):
        self.piece_store = piece_store
        self.llm_fn = llm_fn
        self.config = config or SkillSynthesisConfig()

    def check_and_synthesize(
        self,
        new_piece: KnowledgePiece,
    ) -> Optional[KnowledgePiece]:
        """Check if new piece triggers skill synthesis.

        Searches for neighboring pieces similar to the new piece, evaluates
        whether the cluster meets synthesis thresholds, and if so, calls
        the LLM to synthesize a structured procedure.

        Args:
            new_piece: The newly added knowledge piece to check.

        Returns:
            A synthesized skill KnowledgePiece if synthesis succeeds,
            or None if thresholds are not met, LLM determines pieces
            do not form a skill, or the LLM call fails.
        """
        # Search for similar neighbors
        neighbors = self.piece_store.search(
            query=new_piece.embedding_text or new_piece.content,
            entity_id=new_piece.entity_id,
            top_k=self.config.max_neighbors,
        )

        # Filter by similarity threshold
        candidates = [
            (p, s) for p, s in neighbors if s >= self.config.min_avg_similarity
        ]

        # Check cluster size (candidates + the new piece itself)
        if len(candidates) + 1 < self.config.min_pieces_for_skill:
            return None

        # Check average similarity meets threshold
        avg_sim = (
            sum(s for _, s in candidates) / len(candidates) if candidates else 0
        )
        if avg_sim < self.config.min_avg_similarity:
            return None

        # No LLM function means we can't synthesize
        if self.llm_fn is None:
            logger.warning("No LLM function provided for skill synthesis.")
            return None

        # Attempt LLM-based synthesis
        cluster_pieces = [new_piece] + [p for p, _ in candidates]
        result = self._synthesize_skill(cluster_pieces, avg_sim)

        if result.is_skill_worthy and result.synthesized_skill:
            return self._create_skill_piece(result.synthesized_skill, cluster_pieces)

        return None

    def _synthesize_skill(
        self, pieces: List[KnowledgePiece], avg_similarity: float
    ) -> SkillSynthesisResult:
        """Call LLM to determine if pieces form a skill.

        Args:
            pieces: The cluster of knowledge pieces to evaluate.
            avg_similarity: Average similarity score of the cluster.

        Returns:
            SkillSynthesisResult with LLM analysis.
        """
        pieces_formatted = "\n\n".join(
            [
                f"Piece {i + 1} ({p.knowledge_type.value}):\n{p.content[:300]}"
                for i, p in enumerate(pieces)
            ]
        )

        all_tags: set = set()
        all_domains: set = set()
        for p in pieces:
            all_tags.update(p.tags)
            all_domains.add(p.domain)

        common_tags = ", ".join(sorted(all_tags)[:5]) or "none"
        domains = ", ".join(sorted(all_domains))

        prompt = SKILL_SYNTHESIS_PROMPT.format(
            pieces_formatted=pieces_formatted,
            num_pieces=len(pieces),
            avg_similarity=avg_similarity,
            common_tags=common_tags,
            domains=domains,
        )

        try:
            response = self.llm_fn(prompt)
            parsed = json.loads(response)
            return SkillSynthesisResult(
                is_skill_worthy=parsed.get("is_skill_worthy", False),
                confidence=parsed.get("confidence", 0),
                reasoning=parsed.get("reasoning", ""),
                synthesized_skill=parsed.get("synthesized_skill"),
            )
        except Exception as e:
            logger.warning("Skill synthesis failed: %s", e)
            return SkillSynthesisResult(
                is_skill_worthy=False,
                confidence=0,
                reasoning=f"Error: {e}",
            )

    def _create_skill_piece(
        self,
        skill: Dict[str, Any],
        source_pieces: List[KnowledgePiece],
    ) -> KnowledgePiece:
        """Create a KnowledgePiece from synthesized skill.

        Args:
            skill: The synthesized skill dictionary from LLM.
            source_pieces: The cluster pieces used for synthesis.

        Returns:
            A new KnowledgePiece with knowledge_type=Procedure,
            info_type="skills", and aggregated tags from source pieces.
        """
        steps = skill.get("steps", [])
        content = f"# {skill.get('name', 'Synthesized Skill')}\n\n"
        content += f"{skill.get('description', '')}\n\n"
        content += "## Steps\n"
        for step in steps:
            step_num = step.get("step", "")
            step_desc = step.get("description", "")
            content += f"{step_num}. {step_desc}\n"

        # Aggregate tags from all source pieces
        aggregated_tags = list(set(tag for p in source_pieces for tag in p.tags))

        return KnowledgePiece(
            content=content,
            knowledge_type=KnowledgeType.Procedure,
            info_type="skills",
            domain=source_pieces[0].domain if source_pieces else "general",
            tags=aggregated_tags,
            source="skill_synthesis",
            spaces=source_pieces[0].spaces if source_pieces else ["main"],
        )

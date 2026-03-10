"""
Concrete post-processors for the composable retrieval pipeline.

Provides four post-processor implementations:

- ``FlatStringPostProcessor`` (Path A): Delegates to ``KnowledgeFormatter.format()``
  for flat string output.
- ``GroupedDictPostProcessor`` (Path B): Groups ``RetrievalResult`` by info_type
  using routing rules matching ``KnowledgeProvider._group_by_info_type()``.
- ``AggregatingPostProcessor`` (Path C): Merges scored pieces from multiple
  sub-query results using weighted score aggregation.
- ``BudgetAwarePostProcessor`` (Path B variant): Enforces per-info-type token
  budgets and overall ``available_tokens`` limit.

Requirements: 8.1, 8.2, 9.1, 9.2, 10.1, 10.2, 11.1, 11.2, 11.3
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from attr import attrib, attrs

from agent_foundation.knowledge.retrieval.formatter import (
    KnowledgeFormatter,
    RetrievalResult,
)
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece
from agent_foundation.knowledge.retrieval.retrieval_pipeline import (
    AgenticRetrievalResult,
    PostProcessor,
    SubQuery,
)
from agent_foundation.knowledge.retrieval.provider import (
    _default_formatter as _fallback_formatter,
)
from agent_foundation.knowledge.retrieval.knowledge_provider import CONTEXT_BUDGET
from agent_foundation.knowledge.retrieval.utils import count_tokens

logger = logging.getLogger(__name__)


# ── Path A: Flat String ──────────────────────────────────────────────────


@attrs
class FlatStringPostProcessor(PostProcessor):
    """Path A: RetrievalResult → formatted string via KnowledgeFormatter.

    Reproduces the behavior of ``KnowledgeBase.__call__()``.

    Requirements: 8.1, 8.2
    """

    formatter: KnowledgeFormatter = attrib(factory=KnowledgeFormatter)

    def process(
        self,
        results: Union[RetrievalResult, List[RetrievalResult]],
        **kwargs: Any,
    ) -> str:
        if isinstance(results, list):
            results = results[0]
        return self.formatter.format(results)


# ── Path B: Grouped Dict ─────────────────────────────────────────────────


@attrs
class GroupedDictPostProcessor(PostProcessor):
    """Path B: RetrievalResult → Dict[str, str] grouped by info_type.

    Reproduces the behavior of ``KnowledgeProvider.__call__()`` (in ``provider.py``).
    Supports optional LLM consolidation via a ``KnowledgeConsolidator``.

    Grouping rules (matching ``KnowledgeProvider._group_by_info_type``):

    - Metadata → ``metadata_info_type`` (default ``"user_profile"``)
    - Global metadata → same group (fallback for entity metadata; used only
      if entity metadata is ``None``)
    - Pieces → ``piece.info_type`` (or ``"context"`` if ``None``)
    - Graph edges with linked piece → ``linked_piece.info_type``
    - Depth-1 graph edges from user node (no linked piece) → ``"user_profile"``
      (requires ``active_entity_id`` to identify user-originating edges)
    - Other graph edges (no linked piece) → ``"context"``

    Formatter callable signature: ``(metadata, pieces, graph_context) -> str``

    Requirements: 9.1, 9.2
    """

    type_formatters: dict = attrib(factory=dict)
    default_formatter: Optional[Callable] = attrib(default=None)
    metadata_info_type: str = attrib(default="user_profile")
    active_entity_id: Optional[str] = attrib(default=None)
    consolidator: Optional[Any] = attrib(default=None)

    def process(
        self,
        results: Union[RetrievalResult, List[RetrievalResult]],
        query: str = "",
        **kwargs: Any,
    ) -> Dict[str, str]:
        if isinstance(results, list):
            results = results[0]

        # 1. Group by info_type
        groups = self._group_by_info_type(results)

        # 2. Format each group
        output: Dict[str, str] = {}
        for info_type, group_data in groups.items():
            formatter = self.type_formatters.get(info_type, self.default_formatter)
            if formatter is None:
                formatter = _fallback_formatter
            formatted = formatter(
                group_data["metadata"],
                group_data["pieces"],
                group_data["graph_context"],
            )
            output[info_type] = formatted

        # 3. Optional consolidation
        if self.consolidator is not None:
            output = self.consolidator.consolidate(query, output)

        return output

    def _group_by_info_type(self, result: RetrievalResult) -> Dict[str, dict]:
        """Group a RetrievalResult by info_type string.

        Routing rules match ``KnowledgeProvider._group_by_info_type()``.
        """
        groups: Dict[str, dict] = {}

        def _ensure_group(info_type: str) -> dict:
            if info_type not in groups:
                groups[info_type] = {
                    "metadata": None,
                    "pieces": [],
                    "graph_context": [],
                }
            return groups[info_type]

        # Route metadata to metadata_info_type
        if result.metadata and result.metadata.properties:
            group = _ensure_group(self.metadata_info_type)
            group["metadata"] = result.metadata

        # Route global metadata (fallback for entity metadata)
        if result.global_metadata and result.global_metadata.properties:
            group = _ensure_group(self.metadata_info_type)
            if group["metadata"] is None:
                group["metadata"] = result.global_metadata

        # Route pieces by their info_type
        for piece, score in result.pieces:
            info_type = piece.info_type or "context"
            group = _ensure_group(info_type)
            group["pieces"].append((piece, score))

        # Route graph edges
        for edge in result.graph_context:
            linked_piece = edge.get("piece")

            if linked_piece is not None and hasattr(linked_piece, "info_type"):
                # Graph edge with linked piece → route to piece's info_type
                info_type = linked_piece.info_type or "context"
                group = _ensure_group(info_type)
                group["graph_context"].append(edge)
            elif edge.get("depth", 0) == 1 and self.active_entity_id is not None:
                # Depth-1 edge from user node (no linked piece) → "user_profile"
                group = _ensure_group("user_profile")
                group["graph_context"].append(edge)
            else:
                # Other graph edges → "context"
                group = _ensure_group("context")
                group["graph_context"].append(edge)

        return groups


# ── Path C: Aggregating ──────────────────────────────────────────────────


@attrs
class AggregatingPostProcessor(PostProcessor):
    """Path C: List[RetrievalResult] → AgenticRetrievalResult.

    Reproduces the behavior of ``AgenticRetriever.retrieve()``.
    Aggregates scores across sub-query results using configurable strategy.

    Supported strategies (matching ``AgenticRetriever._aggregate_scores``):

    - ``"max"``: Final score = maximum weighted score across sub-queries.
    - ``"sum"`` or ``"weighted_sum"``: Final score = sum of all weighted scores.

    When ``is_fallback=True``, uses simple dedup-merge logic matching
    ``AgenticRetriever._merge_with_fallback()``: the first result is the
    fallback ``RetrievalResult``, remaining are the original per-sub-query
    ``RetrievalResult`` objects.

    Requirements: 10.1, 10.2
    """

    aggregation_strategy: str = attrib(default="max")
    top_k: int = attrib(default=10)
    min_results: int = attrib(default=3)

    def process(
        self,
        results: Union[RetrievalResult, List[RetrievalResult]],
        sub_queries: Optional[List[SubQuery]] = None,
        is_fallback: bool = False,
        **kwargs: Any,
    ) -> AgenticRetrievalResult:
        if not isinstance(results, list):
            results = [results]

        sub_queries = sub_queries or []

        if is_fallback:
            return self._process_fallback(results, sub_queries)
        else:
            return self._process_normal(results, sub_queries)

    def _process_normal(
        self,
        results: List[RetrievalResult],
        sub_queries: List[SubQuery],
    ) -> AgenticRetrievalResult:
        """Normal aggregation: aggregate scores across sub-query results."""
        # Build per-sub-query ScoredPiece lists with weights
        sub_results: Dict[str, List[ScoredPiece]] = {}
        weights: Dict[str, float] = {}

        for i, result in enumerate(results):
            sq = sub_queries[i] if i < len(sub_queries) else SubQuery(query="")
            key = f"subquery_{i}_{sq.query[:30]}"
            weights[key] = sq.weight

            scored = [
                ScoredPiece(piece=piece, score=score)
                for piece, score in result.pieces
            ]
            sub_results[key] = scored

        aggregated = self._aggregate_scores(sub_results, weights)

        # Check needs_fallback BEFORE top_k truncation
        needs_fallback = len(aggregated) < self.min_results

        return AgenticRetrievalResult(
            pieces=aggregated[: self.top_k],
            sub_queries=sub_queries,
            used_fallback=False,
            needs_fallback=needs_fallback,
        )

    def _process_fallback(
        self,
        results: List[RetrievalResult],
        sub_queries: List[SubQuery],
    ) -> AgenticRetrievalResult:
        """Fallback path: first result is fallback, rest are per-sub-query."""
        fallback_result = results[0]
        per_sub_query_results = results[1:]

        # 1. Aggregate the per-sub-query results
        sub_results: Dict[str, List[ScoredPiece]] = {}
        weights: Dict[str, float] = {}

        for i, result in enumerate(per_sub_query_results):
            sq = sub_queries[i] if i < len(sub_queries) else SubQuery(query="")
            key = f"subquery_{i}_{sq.query[:30]}"
            weights[key] = sq.weight

            scored = [
                ScoredPiece(piece=piece, score=score)
                for piece, score in result.pieces
            ]
            sub_results[key] = scored

        aggregated = self._aggregate_scores(sub_results, weights)

        # 2. Convert fallback pieces to ScoredPiece
        fallback_scored = [
            ScoredPiece(piece=piece, score=score)
            for piece, score in fallback_result.pieces
        ]

        # 3. Merge with fallback (dedup by piece_id, keep higher score)
        merged = self._merge_with_fallback(aggregated, fallback_scored)

        return AgenticRetrievalResult(
            pieces=merged[: self.top_k],
            sub_queries=sub_queries,
            used_fallback=True,
            needs_fallback=False,
        )

    def _aggregate_scores(
        self,
        sub_results: Dict[str, List[ScoredPiece]],
        weights: Dict[str, float],
    ) -> List[ScoredPiece]:
        """Aggregate scores from multiple sub-query results.

        Matches ``AgenticRetriever._aggregate_scores()`` logic.
        """
        piece_scores: Dict[str, Dict[str, Any]] = {}

        for key, scored_pieces in sub_results.items():
            weight = weights.get(key, 1.0)
            for sp in scored_pieces:
                pid = sp.piece.piece_id
                if pid not in piece_scores:
                    piece_scores[pid] = {"piece": sp.piece, "scores": []}
                piece_scores[pid]["scores"].append(sp.score * weight)

        result: List[ScoredPiece] = []
        for _pid, data in piece_scores.items():
            scores = data["scores"]
            if self.aggregation_strategy == "max":
                final_score = max(scores)
            elif self.aggregation_strategy in ("sum", "weighted_sum"):
                final_score = sum(scores)
            else:
                final_score = max(scores)

            result.append(ScoredPiece(piece=data["piece"], score=final_score))

        result.sort(key=lambda x: (-x.score, x.piece.piece_id))
        return result

    @staticmethod
    def _merge_with_fallback(
        primary: List[ScoredPiece],
        fallback: List[ScoredPiece],
    ) -> List[ScoredPiece]:
        """Merge primary results with fallback, dedup by piece_id.

        Matches ``AgenticRetriever._merge_with_fallback()`` logic.
        """
        merged: Dict[str, ScoredPiece] = {}

        for sp in primary:
            merged[sp.piece.piece_id] = sp

        for sp in fallback:
            pid = sp.piece.piece_id
            if pid not in merged or sp.score > merged[pid].score:
                merged[pid] = sp

        result = list(merged.values())
        result.sort(key=lambda x: (-x.score, x.piece.piece_id))
        return result


# ── Path B variant: Budget-Aware ─────────────────────────────────────────

@attrs
class BudgetAwarePostProcessor(PostProcessor):
    """Path B variant: RetrievalResult → formatted string with token budgets.

    Reproduces the behavior of ``BudgetAwareKnowledgeProvider.format_knowledge()``.
    Enforces per-info-type token budgets and overall ``available_tokens`` limit.

    Extracts pieces from the ``RetrievalResult`` and discards metadata/graph_context,
    matching the current ``BudgetAwareKnowledgeProvider`` behavior.

    Requirements: 11.1, 11.2, 11.3
    """

    available_tokens: int = attrib(default=8000)
    budget: Dict[str, int] = attrib(factory=lambda: dict(CONTEXT_BUDGET))

    def process(
        self,
        results: Union[RetrievalResult, List[RetrievalResult]],
        **kwargs: Any,
    ) -> str:
        if isinstance(results, list):
            results = results[0]

        # Convert (KnowledgePiece, float) tuples to ScoredPiece
        scored_pieces = [
            ScoredPiece(piece=piece, score=score)
            for piece, score in results.pieces
        ]

        return self._format_with_budget(scored_pieces)

    def _format_with_budget(self, pieces: List[ScoredPiece]) -> str:
        """Format pieces with per-type and overall budget enforcement.

        Matches ``BudgetAwareKnowledgeProvider.format_knowledge()`` logic.
        """
        by_type: Dict[str, List[ScoredPiece]] = defaultdict(list)
        for piece in pieces:
            info_type = piece.info_type or "context"
            by_type[info_type].append(piece)

        formatted_sections: List[str] = []
        remaining_tokens = self.available_tokens

        for info_type, type_budget in self.budget.items():
            type_pieces = by_type.get(info_type, [])
            if not type_pieces:
                continue

            effective_budget = min(type_budget, remaining_tokens)
            formatted = self._format_type(type_pieces, info_type, effective_budget)
            tokens_used = count_tokens(formatted)

            if tokens_used <= remaining_tokens:
                formatted_sections.append(formatted)
                remaining_tokens -= tokens_used

        return "\n\n".join(formatted_sections)

    @staticmethod
    def _format_type(
        pieces: List[ScoredPiece],
        info_type: str,
        budget_tokens: int,
    ) -> str:
        """Format pieces of a given type using the appropriate formatter.

        Matches ``BudgetAwareKnowledgeProvider._format_type()`` logic.
        """
        formatters = {
            "skills": BudgetAwarePostProcessor._format_skills,
            "instructions": BudgetAwarePostProcessor._format_instructions,
            "context": BudgetAwarePostProcessor._format_context,
            "episodic": BudgetAwarePostProcessor._format_episodic,
            "user_profile": BudgetAwarePostProcessor._format_profile,
        }
        formatter = formatters.get(info_type)
        if formatter is None:
            return ""
        return formatter(pieces, budget_tokens)

    @staticmethod
    def _format_skills(pieces: List[ScoredPiece], budget: int) -> str:
        """Skills: Progressive disclosure with budget enforcement."""
        formatted = ["## Available Skills\n"]
        for piece in pieces:
            summary = piece.piece.summary or piece.piece.content[:100]
            line = f"- **{piece.piece_id}**: {summary}"
            if count_tokens("\n".join(formatted + [line])) > budget:
                break
            formatted.append(line)

        tokens_used = count_tokens("\n".join(formatted))

        # Expand top skill if budget allows
        if pieces:
            top_skill = pieces[0]
            expansion = (
                f"\n### {top_skill.piece_id} (Expanded)\n{top_skill.piece.content}"
            )
            if tokens_used + count_tokens(expansion) <= budget:
                formatted.append(expansion)

        return "\n".join(formatted)

    @staticmethod
    def _format_instructions(pieces: List[ScoredPiece], budget: int) -> str:
        """Instructions: Bullet points with budget enforcement."""
        formatted = ["## Instructions\n"]
        for piece in pieces:
            line = f"- {piece.piece.content}"
            if count_tokens("\n".join(formatted + [line])) > budget:
                break
            formatted.append(line)
        return "\n".join(formatted)

    @staticmethod
    def _format_context(pieces: List[ScoredPiece], budget: int) -> str:
        """Context: Factual paragraphs with budget enforcement."""
        formatted = ["## Relevant Context\n"]
        for piece in pieces:
            block = piece.piece.content + "\n"
            if count_tokens("\n".join(formatted) + block) > budget:
                break
            formatted.append(block)
        return "\n".join(formatted)

    @staticmethod
    def _format_episodic(pieces: List[ScoredPiece], budget: int) -> str:
        """Episodic: With temporal markers and budget enforcement."""
        formatted = ["## Recent History\n"]
        for piece in pieces:
            timestamp = (
                piece.updated_at[:10] if piece.updated_at else "unknown"
            )
            line = f"[{timestamp}] {piece.piece.content}"
            if count_tokens("\n".join(formatted + [line])) > budget:
                break
            formatted.append(line)
        return "\n".join(formatted)

    @staticmethod
    def _format_profile(pieces: List[ScoredPiece], budget: int) -> str:
        """User profile: Key-value summaries with budget enforcement."""
        formatted = ["## User Preferences\n"]
        for piece in pieces:
            summary = piece.piece.summary or piece.piece.content[:100]
            line = f"- {summary}"
            if count_tokens("\n".join(formatted + [line])) > budget:
                break
            formatted.append(line)
        return "\n".join(formatted)

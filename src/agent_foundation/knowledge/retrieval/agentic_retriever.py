"""
AgenticRetriever for multi-query knowledge retrieval.

This module provides an agentic retrieval layer that:
1. Decomposes complex queries into sub-queries
2. Executes sub-queries with different domain/tag filters
3. Aggregates and deduplicates results using configurable strategies
4. Supports fallback to unfiltered search when results are insufficient

The AgenticRetriever wraps a KnowledgeBase and is designed for agents
that need to retrieve knowledge across multiple domains or with query
reformulation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from attr import attrib, attrs

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """A decomposed sub-query with optional filters.

    Attributes:
        query: The search query string.
        domain: Optional domain filter.
        tags: Optional tag filter.
        weight: Relative weight for scoring (default 1.0).
    """

    query: str
    domain: Optional[str] = None
    tags: Optional[List[str]] = None
    weight: float = 1.0


@dataclass
class AgenticRetrievalResult:
    """Results from agentic multi-query retrieval.

    Attributes:
        pieces: Aggregated list of ScoredPiece results.
        sub_queries: The sub-queries that were executed.
        used_fallback: Whether fallback to unfiltered search was triggered.
    """

    pieces: List[ScoredPiece] = field(default_factory=list)
    sub_queries: List[SubQuery] = field(default_factory=list)
    used_fallback: bool = False


@attrs
class AgenticRetriever:
    """Multi-query agentic retriever with query decomposition and aggregation.

    Wraps a KnowledgeBase and supports:
    1. Query decomposition via an optional decomposer function
    2. Execution of sub-queries with domain/tag filters
    3. Score aggregation with configurable strategy (max, sum, weighted_sum)
    4. Graceful fallback when filtered results are insufficient

    Attributes:
        kb: The underlying KnowledgeBase for retrieval.
        query_decomposer: Optional function to decompose query into SubQueries.
            Signature: (query: str) -> List[SubQuery]
            If None, uses the original query without decomposition.
        top_k: Maximum number of results to return.
        min_results: Minimum results threshold before fallback.
        aggregation_strategy: How to combine scores ("max", "sum", "weighted_sum").
    """

    kb: KnowledgeBase = attrib()
    query_decomposer: Optional[Callable[[str], List[SubQuery]]] = attrib(default=None)
    top_k: int = attrib(default=10)
    min_results: int = attrib(default=3)
    aggregation_strategy: str = attrib(default="max")

    def retrieve(
        self,
        query: str,
        entity_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgenticRetrievalResult:
        """Execute multi-query retrieval with decomposition and aggregation.

        1. Decompose query into sub-queries (if decomposer provided)
        2. Execute each sub-query against the KnowledgeBase
        3. Aggregate scores across sub-queries
        4. Apply fallback if results are insufficient

        Args:
            query: The original user query.
            entity_id: Optional entity scope for retrieval.
            **kwargs: Additional arguments passed to kb.retrieve().

        Returns:
            AgenticRetrievalResult with aggregated pieces and metadata.
        """
        # Step 1: Decompose query
        if self.query_decomposer is not None:
            sub_queries = self.query_decomposer(query)
        else:
            sub_queries = [SubQuery(query=query)]

        # Step 2: Execute sub-queries and collect results
        sub_results: Dict[str, List[ScoredPiece]] = {}
        for i, sq in enumerate(sub_queries):
            result = self.kb.retrieve(
                query=sq.query,
                entity_id=entity_id,
                domain=sq.domain,
                tags=sq.tags,
                top_k=self.top_k,
                min_results=1,
                **kwargs,
            )
            # Convert (KnowledgePiece, float) tuples to ScoredPiece
            scored = [
                ScoredPiece(piece=piece, score=score)
                for piece, score in result.pieces
            ]
            sub_results[f"subquery_{i}_{sq.query[:30]}"] = scored

        # Step 3: Aggregate scores
        aggregated = self._aggregate_scores(sub_results, sub_queries)

        # Step 4: Check for fallback
        used_fallback = False
        if len(aggregated) < self.min_results:
            fallback_result = self.kb.retrieve(
                query=query,
                entity_id=entity_id,
                top_k=self.top_k,
                min_results=self.min_results,
                **kwargs,
            )
            fallback_scored = [
                ScoredPiece(piece=piece, score=score)
                for piece, score in fallback_result.pieces
            ]
            aggregated = self._merge_with_fallback(aggregated, fallback_scored)
            used_fallback = True

        return AgenticRetrievalResult(
            pieces=aggregated[: self.top_k],
            sub_queries=sub_queries,
            used_fallback=used_fallback,
        )

    def _aggregate_scores(
        self,
        sub_results: Dict[str, List[ScoredPiece]],
        sub_queries: List[SubQuery],
    ) -> List[ScoredPiece]:
        """Aggregate scores from multiple sub-query results.

        Strategies:
        - "max": Take the maximum weighted score for each piece.
        - "sum" or "weighted_sum": Sum all weighted scores for each piece.

        Args:
            sub_results: Per-sub-query results keyed by sub-query label.
            sub_queries: Original sub-queries (for weights).

        Returns:
            Aggregated and sorted list of ScoredPiece objects.
        """
        piece_scores: Dict[str, Dict[str, Any]] = {}

        weights = {
            f"subquery_{i}_{sq.query[:30]}": sq.weight
            for i, sq in enumerate(sub_queries)
        }

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

    def _merge_with_fallback(
        self,
        primary: List[ScoredPiece],
        fallback: List[ScoredPiece],
    ) -> List[ScoredPiece]:
        """Merge primary results with fallback, deduplicating by piece_id.

        Primary results take precedence; fallback pieces are added only
        if they have a higher score or are not already present.

        Args:
            primary: Primary aggregated results.
            fallback: Fallback results from unfiltered search.

        Returns:
            Merged and sorted list of ScoredPiece objects.
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


def create_domain_decomposer(
    domains: List[str],
) -> Callable[[str], List[SubQuery]]:
    """Create a simple domain-based query decomposer.

    Returns a function that creates one SubQuery per domain, all with
    the original query but different domain filters.

    Args:
        domains: List of domains to search.

    Returns:
        A decomposer function: (query: str) -> List[SubQuery].
    """

    def decomposer(query: str) -> List[SubQuery]:
        return [SubQuery(query=query, domain=d, weight=1.0) for d in domains]

    return decomposer


def create_llm_decomposer(
    llm_fn: Callable[[str], str],
    domains: Optional[List[str]] = None,
) -> Callable[[str], List[SubQuery]]:
    """Create an LLM-backed query decomposer.

    The LLM reformulates the query into multiple sub-queries, each
    potentially targeting different domains or aspects. On failure,
    falls back to a single undecomposed sub-query.

    Args:
        llm_fn: LLM inference function (prompt -> response string).
        domains: Optional list of available domains for the LLM to choose from.

    Returns:
        A decomposer function: (query: str) -> List[SubQuery].
    """
    prompt_template = """You are a query decomposition assistant.

Given a user query, decompose it into 1-3 sub-queries that together cover
the full information need. Each sub-query should target a specific aspect.

{domains_section}
User query: {query}

Return a JSON array of objects with keys:
- "query": the sub-query string
- "domain": the target domain (or null for general)
- "weight": importance weight (0.5-1.5)

Example:
[
  {{"query": "flash attention memory optimization", "domain": "model_optimization", "weight": 1.0}},
  {{"query": "flash attention H100 performance", "domain": "training_efficiency", "weight": 0.8}}
]

Return only the JSON array, no explanation.
"""

    def decomposer(query: str) -> List[SubQuery]:
        domains_section = ""
        if domains:
            domains_section = f"Available domains: {', '.join(domains)}\n\n"

        prompt = prompt_template.format(
            domains_section=domains_section,
            query=query,
        )
        try:
            response = llm_fn(prompt)
            data = json.loads(response)
            return [
                SubQuery(
                    query=item["query"],
                    domain=item.get("domain"),
                    tags=item.get("tags"),
                    weight=item.get("weight", 1.0),
                )
                for item in data
            ]
        except Exception:
            logger.warning(
                "LLM decomposer failed, falling back to original query",
                exc_info=True,
            )
            return [SubQuery(query=query)]

    return decomposer

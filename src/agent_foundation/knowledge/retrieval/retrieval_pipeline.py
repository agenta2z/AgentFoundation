"""
Pipeline abstractions for composable knowledge retrieval.

Provides the core building blocks for a configurable retrieval pipeline:
- ``SubQuery`` and ``AgenticRetrievalResult`` dataclasses (canonical location;
  re-exported from ``agentic_retriever.py`` for backward compatibility).
- ``QueryExpander`` ABC for query decomposition / expansion.
- ``PostProcessor`` ABC for transforming retrieval results into any output format.
- ``RetrievalPipeline`` class that composes optional expansion, core retrieval
  via ``KnowledgeBase`` layer methods, and configurable post-processing.

Two execution paths:
1. **Single-query** (no expander): delegates to ``kb.retrieve()`` directly.
2. **Multi-query** (with expander): calls ``retrieve_metadata()`` and
   ``retrieve_identity_graph()`` once, then loops ``retrieve_pieces()`` and
   ``retrieve_search_graph()`` per sub-query.

Requirements: 7.1, 7.2, 7.3, 7.5, 14.1, 14.2, 14.3, 14.4
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from attr import attrib, attrs

if TYPE_CHECKING:
    from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


# ── Data classes (canonical location) ────────────────────────────────────


@dataclass
class SubQuery:
    """A sub-query produced by query expansion.

    Defined here (not in ``agentic_retriever.py``) to avoid circular imports
    between ``retrieval_pipeline.py`` and ``agentic_retriever.py``.

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

    Moved here from ``agentic_retriever.py`` so that ``post_processors.py``
    can import it without depending on the deprecated ``AgenticRetriever``
    module.

    Attributes:
        pieces: Aggregated list of ScoredPiece results.
        sub_queries: The sub-queries that were executed.
        used_fallback: Whether fallback to unfiltered search was triggered.
        needs_fallback: Internal flag set by ``AggregatingPostProcessor``
            BEFORE top_k truncation, indicating whether the pipeline should
            trigger fallback retrieval.  Not part of the public API —
            consumed only by ``RetrievalPipeline._needs_fallback()``.
    """

    pieces: List[Any] = field(default_factory=list)  # List[ScoredPiece]
    sub_queries: List[SubQuery] = field(default_factory=list)
    used_fallback: bool = False
    needs_fallback: bool = False


# ── Abstract base classes ────────────────────────────────────────────────


class QueryExpander(ABC):
    """Abstract base for query expansion / decomposition."""

    @abstractmethod
    def expand(self, query: str) -> List[SubQuery]:
        """Decompose a query into sub-queries.

        Args:
            query: The original query string.

        Returns:
            List of ``SubQuery`` objects.  If no expansion is needed,
            returns a single ``SubQuery`` wrapping the original query.
        """
        ...


class PostProcessor(ABC):
    """Abstract base for post-processing ``RetrievalResult``(s)."""

    @abstractmethod
    def process(
        self,
        results: Any,
        query: str = "",
        sub_queries: Optional[List[SubQuery]] = None,
        **kwargs: Any,
    ) -> Any:
        """Transform retrieval results into the desired output format.

        Args:
            results: Single ``RetrievalResult`` or list (for multi-query).
            query: The original user query (needed by consolidation).
            sub_queries: The sub-queries that produced the results.
            **kwargs: Additional configuration.

        Returns:
            Output in the format specific to the post-processor.
        """
        ...


# ── Pipeline ─────────────────────────────────────────────────────────────


@attrs
class RetrievalPipeline:
    """Composable retrieval pipeline with configurable stages.

    Composes: ``[optional expansion] → KB layer methods → [post-processing]``

    Consolidation is configured on the post-processor (e.g. via a
    ``consolidator`` attribute on ``GroupedDictPostProcessor``), not on the
    pipeline itself.

    Attributes:
        kb: The underlying ``KnowledgeBase``.
        expander: Optional query expander (``None`` = pass-through).
        post_processor: Post-processor for output formatting.
        min_results: Minimum results before fallback (for agentic path).
        top_k: Maximum results to return.
    """

    kb: KnowledgeBase = attrib()  # type annotation for IDE support
    post_processor: PostProcessor = attrib()
    expander: Optional[QueryExpander] = attrib(default=None)
    min_results: int = attrib(default=1)
    top_k: int = attrib(default=10)

    # ── public API ───────────────────────────────────────────────────

    def execute(
        self,
        query: str,
        entity_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the full pipeline.

        Single-query path (no expander):
            Delegates to ``kb.retrieve()`` directly, then passes the result
            to the post-processor.

        Multi-query path (with expander):
            Calls ``retrieve_metadata()`` and ``retrieve_identity_graph()``
            once, then loops ``retrieve_pieces()`` and
            ``retrieve_search_graph()`` per sub-query.  Merges all L3a
            search contexts with L3b identity context once after the loop.

        Args:
            query: The user query.
            entity_id: Optional entity scope.
            **kwargs: Passed through to KB layer methods.

        Returns:
            Output from the post-processor (type depends on configuration).
        """
        if self.expander is None:
            return self._execute_single_query(query, entity_id, **kwargs)
        else:
            return self._execute_multi_query(query, entity_id, **kwargs)

    # ── private helpers ──────────────────────────────────────────────

    def _execute_single_query(
        self,
        query: str,
        entity_id: Optional[str],
        **kwargs: Any,
    ) -> Any:
        """Single-query path: delegate to ``kb.retrieve()`` directly."""
        result = self.kb.retrieve(query=query, entity_id=entity_id, **kwargs)
        output = self.post_processor.process(result, query=query)

        # Fallback logic (same as multi-query path)
        if self._needs_fallback(output):
            fallback_result = self.kb.retrieve(
                query=query,
                entity_id=entity_id,
                top_k=self.top_k,
                min_results=self.min_results,
                **kwargs,
            )
            output = self.post_processor.process(
                [fallback_result, result],
                query=query,
                sub_queries=None,
                is_fallback=True,
                **kwargs,
            )

        return output

    def _execute_multi_query(
        self,
        query: str,
        entity_id: Optional[str],
        **kwargs: Any,
    ) -> Any:
        """Multi-query path: L1/L3b once, L2/L3a per sub-query."""
        # Deferred import to avoid circular dependency
        from agent_foundation.knowledge.retrieval.graph_walk import merge_graph_contexts
        from agent_foundation.knowledge.retrieval.formatter import RetrievalResult

        spaces = kwargs.get("spaces")
        include_global = kwargs.get("include_global", True)

        # L1: metadata — once
        metadata, global_metadata = self.kb.retrieve_metadata(
            entity_id=entity_id,
            include_global=include_global,
            spaces=spaces,
        )

        # L3b: identity graph — once (before sub-queries, so no L2 dedup)
        identity_ctx = self.kb.retrieve_identity_graph(
            entity_id=entity_id,
            spaces=spaces,
        )

        # Expand query into sub-queries
        sub_queries = self.expander.expand(query)

        all_pieces: List[Any] = []  # per-sub-query piece lists
        all_search_ctx: List[Dict[str, Any]] = []

        for sq in sub_queries:
            # L2: pieces per sub-query
            pieces = self.kb.retrieve_pieces(
                query=sq.query,
                entity_id=entity_id,
                domain=sq.domain,
                tags=sq.tags,
                top_k=self.top_k,
                include_global=include_global,
                min_results=1,
                spaces=spaces,
            )
            all_pieces.append(pieces)

            # Build dedup set from this sub-query's L2 pieces
            already_retrieved_piece_ids: Optional[Dict[str, str]] = None
            if pieces:
                already_retrieved_piece_ids = {
                    p.piece_id: p.info_type for p, _ in pieces
                }

            # L3a: search graph per sub-query
            search_ctx = self.kb.retrieve_search_graph(
                query=sq.query,
                top_k=self.top_k,
                spaces=spaces,
                already_retrieved_piece_ids=already_retrieved_piece_ids,
            )
            all_search_ctx.extend(search_ctx)

        # Merge all L3a + L3b ONCE after all sub-queries
        merged_ctx = (
            merge_graph_contexts(all_search_ctx, identity_ctx)
            if (all_search_ctx or identity_ctx)
            else []
        )

        # Assemble per-sub-query RetrievalResults with shared metadata
        # and a COPY of merged graph context to avoid cross-result mutation
        results: List[Any] = []
        for pieces in all_pieces:
            r = RetrievalResult()
            r.metadata = metadata
            r.global_metadata = global_metadata
            r.pieces = pieces
            r.graph_context = list(merged_ctx)  # copy
            results.append(r)

        output = self.post_processor.process(
            results,
            query=query,
            sub_queries=sub_queries,
            **kwargs,
        )

        # Fallback logic for agentic path
        if self._needs_fallback(output):
            fallback_result = self.kb.retrieve(
                query=query,
                entity_id=entity_id,
                top_k=self.top_k,
                min_results=self.min_results,
                **kwargs,
            )
            output = self.post_processor.process(
                [fallback_result] + (results if isinstance(results, list) else [results]),
                query=query,
                sub_queries=sub_queries,
                is_fallback=True,
                **kwargs,
            )

        return output

    def _needs_fallback(self, output: Any) -> bool:
        """Check if output needs fallback retrieval.

        For Path C (``AgenticRetrievalResult``), checks the ``needs_fallback``
        flag set by ``AggregatingPostProcessor`` BEFORE top_k truncation.

        For Path A (str) and Path B (Dict), always returns ``False``.
        """
        if hasattr(output, "needs_fallback"):
            return output.needs_fallback
        return False


# ── Legacy adapter ───────────────────────────────────────────────────────


class _LegacyQueryExpander(QueryExpander):
    """Adapter wrapping an existing ``query_decomposer`` callable as a ``QueryExpander``.

    The current ``AgenticRetriever`` uses a ``query_decomposer: Callable[[str], List[SubQuery]]``
    (created by ``create_domain_decomposer`` or ``create_llm_decomposer``).  This adapter
    lets the pipeline consume it without changing the decomposer interface.
    """

    def __init__(self, decomposer: Callable[[str], List[SubQuery]]):
        self._decomposer = decomposer

    def expand(self, query: str) -> List[SubQuery]:
        return self._decomposer(query)


# ── Factory functions ────────────────────────────────────────────────────


def create_domain_decomposer(
    domains: List[str],
) -> Callable[[str], List[SubQuery]]:
    """Create a simple domain-based query decomposer.

    Returns a function that creates one ``SubQuery`` per domain, all with
    the original query but different domain filters.

    Args:
        domains: List of domains to search.

    Returns:
        A decomposer function: ``(query: str) -> List[SubQuery]``.
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
    potentially targeting different domains or aspects.  On failure,
    falls back to a single undecomposed sub-query.

    Args:
        llm_fn: LLM inference function (prompt -> response string).
        domains: Optional list of available domains for the LLM to choose from.

    Returns:
        A decomposer function: ``(query: str) -> List[SubQuery]``.
    """

    def decomposer(query: str) -> List[SubQuery]:
        from agent_foundation.knowledge.prompt_templates import render_prompt

        domains_section = ""
        if domains:
            domains_section = f"Available domains: {', '.join(domains)}\n\n"

        prompt = render_prompt(
            "retrieval/QueryDecomposition",
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

# pyre-strict
"""
Knowledge Discovery Adapter for Metamate Integration.

This adapter provides parallel search across internal knowledge sources:
- Wiki (InternWiki, Static Docs)
- Workplace (Posts, Q&A)
- Diffs (Phabricator)
- Tasks (Phabricator)
- SEVs (SEV database)

IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php
Verify at https://www.internalfb.com/metamate/agent/tools before implementation.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.interfaces import (
    MetamateClientInterface,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.exceptions import KnowledgeSearchError
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import (
    AggregatedKnowledgeResults,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeSource,
    MetamateConfig,
)


logger: logging.Logger = logging.getLogger(__name__)


class KnowledgeDiscoveryAdapter:
    """
    Adapter for parallel search across internal knowledge sources.

    This adapter provides:
    - Parallel search across multiple knowledge sources
    - Result aggregation and ranking
    - Graceful handling of individual source failures
    - Configurable timeouts per source

    IMPORTANT: Uses canonical tool names from MetamateAgentEngineTypes.php:
    - knowledge_search: MetamateAgentKnowledgeSearchToolV3 (for WIKI and WORKPLACE)
    - metamate_diff_search: MetamateAgentDiffSearchTool (for DIFF)
    - task.search: MetamateAgentTaskSearchTool (for TASK)
    - sev.search: MetamateAgentSevSearchTool (for SEV)

    Example:
        ```python
        adapter = KnowledgeDiscoveryAdapter(client=metamate_client)

        query = KnowledgeQuery(
            query_text="how to configure model training",
            sources={KnowledgeSource.WIKI, KnowledgeSource.DIFF},
            max_results_per_source=5,
        )

        results = await adapter.search(query)
        for result in results.results:
            print(f"{result.source.value}: {result.title}")
        ```
    """

    TOOL_MAPPING: dict[KnowledgeSource, str] = {
        KnowledgeSource.WIKI: "knowledge_search",
        KnowledgeSource.WORKPLACE: "knowledge_search",
        KnowledgeSource.DIFF: "metamate_diff_search",
        KnowledgeSource.TASK: "task.search",
        KnowledgeSource.SEV: "sev.search",
    }

    DOC_TYPE_MAPPING: dict[KnowledgeSource, list[str]] = {
        KnowledgeSource.WIKI: ["intern_wiki_page", "static_docs"],
        KnowledgeSource.WORKPLACE: ["post", "qa"],
    }

    def __init__(
        self,
        client: MetamateClientInterface,
        config: MetamateConfig | None = None,
    ) -> None:
        """
        Initialize the Knowledge Discovery adapter.

        Args:
            client: Metamate client for API communication.
            config: Optional configuration (uses client config if not provided).
        """
        self._client = client
        self._config = config or client.config

    async def search(
        self,
        query: KnowledgeQuery,
    ) -> AggregatedKnowledgeResults:
        """
        Search sources in parallel and aggregate results.

        Args:
            query: Knowledge query with search parameters.

        Returns:
            AggregatedKnowledgeResults with ranked results.
        """
        start_time = time.time()

        sources = query.sources or set(KnowledgeSource)

        tasks: list[asyncio.Task[list[KnowledgeResult]]] = []
        source_list = list(sources)

        for source in source_list:
            task = asyncio.create_task(
                self._search_source(query, source),
                name=f"search_{source.value}",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[KnowledgeResult] = []
        errors: list[str] = []
        sources_searched: list[KnowledgeSource] = []

        for source, result in zip(source_list, results):
            if isinstance(result, Exception):
                errors.append(f"{source.value}: {str(result)}")
                logger.warning("Search failed for %s: %s", source.value, result)
            else:
                all_results.extend(result)
                sources_searched.append(source)

        ranked_results = self._rank_results(all_results, query)

        execution_time_ms = int((time.time() - start_time) * 1000)

        return AggregatedKnowledgeResults(
            results=ranked_results,
            sources_searched=sources_searched,
            total_found=len(all_results),
            query=query,
            execution_time_ms=execution_time_ms,
            errors=errors if errors else None,
        )

    async def search_single_source(
        self,
        query: KnowledgeQuery,
        source: KnowledgeSource,
    ) -> list[KnowledgeResult]:
        """
        Search a single knowledge source.

        Args:
            query: Knowledge query with search parameters.
            source: The source to search.

        Returns:
            List of knowledge results from the source.

        Raises:
            KnowledgeSearchError: If search fails.
        """
        return await self._search_source(query, source)

    async def _search_source(
        self,
        query: KnowledgeQuery,
        source: KnowledgeSource,
    ) -> list[KnowledgeResult]:
        """Search a single knowledge source."""
        tool_name = self.TOOL_MAPPING.get(source)
        if tool_name is None:
            raise KnowledgeSearchError(
                message=f"Unknown source: {source}",
                source=source.value,
                query=query.query_text,
            )

        try:
            parameters = self._build_parameters(query, source)
            response = await self._client.invoke_tool(
                tool_name=tool_name,
                parameters=parameters,
            )
            return self._parse_results(response.data, source, query)

        except Exception as e:
            raise KnowledgeSearchError(
                message=f"Search failed for {source.value}: {e}",
                source=source.value,
                query=query.query_text,
                cause=e,
            ) from e

    def _build_parameters(
        self,
        query: KnowledgeQuery,
        source: KnowledgeSource,
    ) -> dict[str, Any]:
        """Build parameters for the search tool."""
        base_params: dict[str, Any] = {
            "query": query.query_text,
            "limit": query.max_results_per_source,
        }

        if source in {KnowledgeSource.WIKI, KnowledgeSource.WORKPLACE}:
            doc_types = self.DOC_TYPE_MAPPING.get(source, [])
            base_params["doc_types"] = doc_types

        if source == KnowledgeSource.DIFF:
            if "file_path" in query.filters:
                base_params["file_path"] = query.filters["file_path"]

        if source == KnowledgeSource.TASK:
            if "status" in query.filters:
                base_params["status"] = query.filters["status"]

        if source == KnowledgeSource.SEV:
            if "severity" in query.filters:
                base_params["severity"] = query.filters["severity"]
            if "time_range" in query.filters:
                base_params["time_range"] = query.filters["time_range"]

        base_params.update(query.filters.get(source.value, {}))

        return base_params

    def _parse_results(
        self,
        data: Any,
        source: KnowledgeSource,
        query: KnowledgeQuery,
    ) -> list[KnowledgeResult]:
        """Parse results from tool response."""
        results: list[KnowledgeResult] = []

        if not isinstance(data, dict):
            return results

        raw_results = data.get("results", [])
        for item in raw_results:
            result = self._parse_single_result(item, source)
            if result is not None:
                results.append(result)

        return results

    def _parse_single_result(
        self,
        item: dict[str, Any],
        source: KnowledgeSource,
    ) -> KnowledgeResult | None:
        """Parse a single result item."""
        try:
            result_id = item.get("id") or item.get("diff_id") or item.get("task_id")
            if result_id is None:
                result_id = str(uuid.uuid4())

            return KnowledgeResult(
                result_id=str(result_id),
                source=source,
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                snippet=item.get("snippet", item.get("summary", "")),
                relevance_score=item.get("relevance_score", 0.0),
                author=item.get("author"),
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
                metadata=self._extract_metadata(item, source),
            )
        except Exception as e:
            logger.warning("Failed to parse result: %s", e)
            return None

    def _extract_metadata(
        self,
        item: dict[str, Any],
        source: KnowledgeSource,
    ) -> dict[str, Any]:
        """Extract source-specific metadata."""
        metadata: dict[str, Any] = {"source_type": source.value}

        if source == KnowledgeSource.DIFF:
            metadata["status"] = item.get("status")
            metadata["files_changed"] = item.get("files_changed", [])

        elif source == KnowledgeSource.TASK:
            metadata["status"] = item.get("status")
            metadata["priority"] = item.get("priority")
            metadata["assignee"] = item.get("assignee")

        elif source == KnowledgeSource.SEV:
            metadata["severity"] = item.get("severity")
            metadata["status"] = item.get("status")
            metadata["resolution"] = item.get("resolution")

        return metadata

    def _rank_results(
        self,
        results: list[KnowledgeResult],
        query: KnowledgeQuery,
    ) -> list[KnowledgeResult]:
        """Rank results by relevance."""
        return sorted(
            results,
            key=lambda r: (r.relevance_score, r.created_at or ""),
            reverse=True,
        )

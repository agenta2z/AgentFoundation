# pyre-strict
"""
Deep Research Adapter for Metamate Integration.

This adapter provides full deep research capability with:
- Streaming execution with real-time progress
- Parallel information gathering
- Synthesis engine integration
- Fallback to Stan scripts when Metamate unavailable

IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php
Verify at https://www.internalfb.com/metamate/agent/tools before implementation.
"""

import logging
import time
import uuid
from typing import Any, AsyncIterator

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.interfaces import (
    FallbackClientInterface,
    MetamateClientInterface,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.exceptions import (
    MetamateUnavailableError,
    ResearchError,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import (
    Citation,
    Finding,
    MetamateConfig,
    ResearchEvent,
    ResearchEventType,
    ResearchPlan,
    ResearchQuery,
    ResearchReport,
    ResearchResults,
)


logger: logging.Logger = logging.getLogger(__name__)


class DeepResearchAdapter:
    """
    Adapter for executing deep research with Metamate integration.

    This adapter provides:
    - Research plan creation from queries
    - Streaming execution with real-time progress events
    - Parallel information gathering from multiple sources
    - Synthesis of findings with citations
    - Automatic fallback to Stan scripts when Metamate unavailable

    Example:
        ```python
        adapter = DeepResearchAdapter(client=metamate_client)

        queries = [
            ResearchQuery(query_text="How to optimize ranking models?"),
            ResearchQuery(query_text="Best practices for model compression"),
        ]

        plan = await adapter.create_research_plan(queries)

        async for event in adapter.execute_research(plan):
            if event.type == ResearchEventType.SOURCE_FOUND:
                print(f"Found source: {event.source.title}")
            elif event.type == ResearchEventType.REPORT_READY:
                print(f"Report ready: {event.report.title}")
        ```
    """

    def __init__(
        self,
        client: MetamateClientInterface,
        fallback_client: FallbackClientInterface | None = None,
        config: MetamateConfig | None = None,
    ) -> None:
        """
        Initialize the Deep Research adapter.

        Args:
            client: Metamate client for API communication.
            fallback_client: Optional fallback client for when Metamate is unavailable.
            config: Optional configuration (uses client config if not provided).
        """
        self._client = client
        self._fallback_client = fallback_client
        self._config = config or client.config

    async def create_research_plan(
        self,
        queries: list[ResearchQuery],
    ) -> ResearchPlan:
        """
        Create a research plan from a list of queries.

        Args:
            queries: List of research queries to plan.

        Returns:
            ResearchPlan with organized queries and estimated time.
        """
        plan_id = str(uuid.uuid4())

        sorted_queries = sorted(queries, key=lambda q: q.priority, reverse=True)

        estimated_time = len(queries) * 30

        return ResearchPlan(
            plan_id=plan_id,
            queries=sorted_queries,
            total_estimated_time_seconds=estimated_time,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={"total_queries": len(queries)},
        )

    async def execute_research(
        self,
        plan: ResearchPlan,
    ) -> AsyncIterator[ResearchEvent]:
        """
        Execute research with real-time progress streaming.

        Args:
            plan: Research plan to execute.

        Yields:
            ResearchEvent objects as research progresses.

        Raises:
            ResearchError: If research fails completely.
        """
        try:
            if not await self._client.health_check():
                raise MetamateUnavailableError()

            yield ResearchEvent(
                type=ResearchEventType.PLAN_CREATED,
                timestamp=time.time(),
                plan=plan,
                message=f"Research plan created with {len(plan.queries)} queries",
            )

            yield ResearchEvent(
                type=ResearchEventType.GATHERING_STARTED,
                timestamp=time.time(),
                message="Starting information gathering",
                progress=0.0,
            )

            all_sources: list[Citation] = []
            total_queries = len(plan.queries)

            for idx, query in enumerate(plan.queries):
                progress = (idx + 1) / total_queries * 0.5

                sources = await self._gather_sources_for_query(query)

                for source in sources:
                    all_sources.append(source)
                    yield ResearchEvent(
                        type=ResearchEventType.SOURCE_FOUND,
                        timestamp=time.time(),
                        source=source,
                        progress=progress,
                        message=f"Found: {source.title}",
                    )

            yield ResearchEvent(
                type=ResearchEventType.GATHERING_COMPLETE,
                timestamp=time.time(),
                message=f"Gathering complete: found {len(all_sources)} sources",
                progress=0.5,
            )

            yield ResearchEvent(
                type=ResearchEventType.SYNTHESIS_STARTED,
                timestamp=time.time(),
                message="Starting synthesis",
                progress=0.5,
            )

            findings: list[Finding] = []
            async for finding in self._synthesize_findings(all_sources, plan):
                findings.append(finding)
                yield ResearchEvent(
                    type=ResearchEventType.FINDING_GENERATED,
                    timestamp=time.time(),
                    finding=finding,
                    progress=0.5 + (len(findings) / max(1, len(all_sources)) * 0.3),
                    message=f"Generated finding: {finding.content[:50]}...",
                )

            yield ResearchEvent(
                type=ResearchEventType.SYNTHESIS_COMPLETE,
                timestamp=time.time(),
                message=f"Synthesis complete: generated {len(findings)} findings",
                progress=0.8,
            )

            yield ResearchEvent(
                type=ResearchEventType.REPORT_STARTED,
                timestamp=time.time(),
                message="Generating report",
                progress=0.8,
            )

            report = await self._generate_report(plan, findings, all_sources)

            yield ResearchEvent(
                type=ResearchEventType.REPORT_READY,
                timestamp=time.time(),
                report=report,
                progress=1.0,
                message="Research complete",
            )

        except MetamateUnavailableError:
            yield ResearchEvent(
                type=ResearchEventType.FALLBACK_ACTIVATED,
                timestamp=time.time(),
                message="Metamate unavailable, using fallback",
            )
            async for event in self._execute_fallback(plan):
                yield event

        except Exception as e:
            logger.error("Research failed: %s", e)
            yield ResearchEvent(
                type=ResearchEventType.ERROR,
                timestamp=time.time(),
                error=str(e),
                message=f"Research failed: {e}",
            )
            raise ResearchError(
                message=f"Research execution failed: {e}",
                research_plan_id=plan.plan_id,
                cause=e,
            ) from e

    async def execute_research_sync(
        self,
        plan: ResearchPlan,
    ) -> ResearchResults:
        """
        Execute research synchronously and return complete results.

        Args:
            plan: Research plan to execute.

        Returns:
            ResearchResults with complete report and stats.
        """
        events: list[ResearchEvent] = []
        report: ResearchReport | None = None
        fallback_used = False
        errors: list[str] = []

        async for event in self.execute_research(plan):
            events.append(event)

            if event.type == ResearchEventType.REPORT_READY:
                report = event.report
            elif event.type == ResearchEventType.FALLBACK_ACTIVATED:
                fallback_used = True
            elif event.type == ResearchEventType.ERROR:
                errors.append(event.error or "Unknown error")

        if report is None:
            raise ResearchError(
                message="Research did not produce a report",
                research_plan_id=plan.plan_id,
            )

        source_count = sum(
            1 for e in events if e.type == ResearchEventType.SOURCE_FOUND
        )
        finding_count = sum(
            1 for e in events if e.type == ResearchEventType.FINDING_GENERATED
        )

        return ResearchResults(
            report=report,
            queries_executed=len(plan.queries),
            sources_found=source_count,
            findings_generated=finding_count,
            fallback_used=fallback_used,
            errors=errors,
        )

    async def _gather_sources_for_query(
        self,
        query: ResearchQuery,
    ) -> list[Citation]:
        """Gather sources for a single research query."""
        response = await self._client.invoke_tool(
            tool_name="knowledge_search",
            parameters={
                "query": query.query_text,
                "limit": query.max_sources,
                "doc_types": self._get_doc_types_for_query(query),
            },
        )

        return self._parse_sources(response.data, query)

    def _get_doc_types_for_query(self, query: ResearchQuery) -> list[str]:
        """Get document types to search based on query type."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import QueryType

        if query.query_type == QueryType.INTERNAL:
            return ["wiki", "workplace", "diff", "task"]
        elif query.query_type == QueryType.EXTERNAL:
            return ["web", "academic"]
        else:
            return ["wiki", "workplace", "diff", "task", "web", "academic"]

    def _parse_sources(
        self,
        data: Any,
        query: ResearchQuery,
    ) -> list[Citation]:
        """Parse sources from tool response."""
        sources: list[Citation] = []

        if not isinstance(data, dict):
            return sources

        results = data.get("results", [])
        for result in results:
            sources.append(
                Citation(
                    source_id=result.get("id", str(uuid.uuid4())),
                    source_type=result.get("source_type", "unknown"),
                    title=result.get("title", "Untitled"),
                    url=result.get("url", ""),
                    snippet=result.get("snippet", ""),
                    relevance_score=result.get("relevance_score", 0.0),
                    author=result.get("author"),
                    metadata={"query": query.query_text},
                )
            )

        return sources

    async def _synthesize_findings(
        self,
        sources: list[Citation],
        plan: ResearchPlan,
    ) -> AsyncIterator[Finding]:
        """Synthesize findings from gathered sources."""
        grouped_sources: dict[str, list[Citation]] = {}
        for source in sources:
            source_type = source.source_type
            if source_type not in grouped_sources:
                grouped_sources[source_type] = []
            grouped_sources[source_type].append(source)

        finding_idx = 0
        for source_type, type_sources in grouped_sources.items():
            sorted_sources = sorted(
                type_sources,
                key=lambda s: s.relevance_score,
                reverse=True,
            )[:5]

            if sorted_sources:
                finding_idx += 1
                avg_confidence = sum(s.relevance_score for s in sorted_sources) / len(
                    sorted_sources
                )

                yield Finding(
                    finding_id=f"finding-{finding_idx}",
                    content=f"Based on {len(sorted_sources)} {source_type} sources: "
                    f"{sorted_sources[0].snippet[:200]}...",
                    confidence_score=avg_confidence,
                    citations=sorted_sources,
                    category=source_type,
                )

    async def _generate_report(
        self,
        plan: ResearchPlan,
        findings: list[Finding],
        all_sources: list[Citation],
    ) -> ResearchReport:
        """Generate the final research report."""
        query_summary = ", ".join(q.query_text[:30] for q in plan.queries[:3])

        summary_parts = [f"Research on: {query_summary}"]
        if findings:
            summary_parts.append(f"Found {len(findings)} key findings.")
            top_finding = max(findings, key=lambda f: f.confidence_score)
            summary_parts.append(f"Key insight: {top_finding.content[:100]}...")

        return ResearchReport(
            report_id=str(uuid.uuid4()),
            title=f"Research Report: {query_summary}",
            summary=" ".join(summary_parts),
            findings=findings,
            all_citations=all_sources,
            total_sources_searched=len(all_sources),
            metadata={"plan_id": plan.plan_id},
        )

    async def _execute_fallback(
        self,
        plan: ResearchPlan,
    ) -> AsyncIterator[ResearchEvent]:
        """Execute research using fallback mechanism."""
        if self._fallback_client is None:
            yield ResearchEvent(
                type=ResearchEventType.ERROR,
                timestamp=time.time(),
                error="No fallback client configured",
                message="Fallback unavailable",
            )
            return

        yield ResearchEvent(
            type=ResearchEventType.GATHERING_STARTED,
            timestamp=time.time(),
            message="Starting fallback research",
            progress=0.0,
        )

        all_sources: list[Citation] = []
        for query in plan.queries:
            fallback_result = await self._fallback_client.execute_research_fallback(
                query=query.query_text,
                query_type=query.query_type.value,
            )

            for result in fallback_result.get("results", []):
                source = Citation(
                    source_id=str(uuid.uuid4()),
                    source_type="fallback",
                    title=result.get("title", "Fallback Result"),
                    url=result.get("url", ""),
                    snippet=result.get("snippet", ""),
                    relevance_score=result.get("relevance_score", 0.5),
                )
                all_sources.append(source)
                yield ResearchEvent(
                    type=ResearchEventType.SOURCE_FOUND,
                    timestamp=time.time(),
                    source=source,
                    message=f"Fallback found: {source.title}",
                )

        yield ResearchEvent(
            type=ResearchEventType.GATHERING_COMPLETE,
            timestamp=time.time(),
            message=f"Fallback gathering complete: {len(all_sources)} sources",
            progress=0.5,
        )

        findings = [
            Finding(
                finding_id="fallback-finding-1",
                content=f"Fallback research found {len(all_sources)} sources.",
                confidence_score=0.6,
                citations=all_sources,
                category="fallback",
            )
        ]

        for finding in findings:
            yield ResearchEvent(
                type=ResearchEventType.FINDING_GENERATED,
                timestamp=time.time(),
                finding=finding,
                progress=0.8,
            )

        report = ResearchReport(
            report_id=str(uuid.uuid4()),
            title="Fallback Research Report",
            summary=f"Research completed via fallback with {len(all_sources)} sources.",
            findings=findings,
            all_citations=all_sources,
            total_sources_searched=len(all_sources),
            metadata={"plan_id": plan.plan_id, "fallback": True},
        )

        yield ResearchEvent(
            type=ResearchEventType.REPORT_READY,
            timestamp=time.time(),
            report=report,
            progress=1.0,
            message="Fallback research complete",
        )

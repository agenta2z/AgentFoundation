# pyre-strict
"""
Debug Assistant Adapter for Metamate Integration.

This adapter provides debugging assistance using Metamate + Devmate:
- Similar issue search across tasks and SEVs
- Related diff search for error locations
- Code context retrieval
- Debug hint synthesis

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
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.exceptions import MetamateError
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import (
    Citation,
    DebugAssistantResults,
    DebugContext,
    DebugHint,
    KnowledgeResult,
    KnowledgeSource,
    MetamateConfig,
    RelatedDiff,
)


logger: logging.Logger = logging.getLogger(__name__)


class DebugAssistantAdapter:
    """
    Adapter for debugging assistance with Metamate integration.

    This adapter provides:
    - Search for similar issues in tasks and SEVs
    - Find related diffs that may have caused the issue
    - Retrieve code context for error locations
    - Synthesize actionable debug hints

    IMPORTANT: Uses canonical tool names from MetamateAgentEngineTypes.php:
    - task.search: MetamateAgentTaskSearchTool
    - sev.search: MetamateAgentSevSearchTool
    - metamate_diff_search: MetamateAgentDiffSearchTool
    - knowledge_search: MetamateAgentKnowledgeSearchToolV3

    Example:
        ```python
        adapter = DebugAssistantAdapter(client=metamate_client)

        context = DebugContext(
            error_message="IndexError: list index out of range",
            file_path="fbcode/ranking/model.py",
            line_number=42,
            stack_trace="...",
        )

        results = await adapter.investigate(context)
        for hint in results.hints:
            print(f"Hint ({hint.confidence:.0%}): {hint.content}")
        ```
    """

    def __init__(
        self,
        client: MetamateClientInterface,
        config: MetamateConfig | None = None,
    ) -> None:
        """
        Initialize the Debug Assistant adapter.

        Args:
            client: Metamate client for API communication.
            config: Optional configuration (uses client config if not provided).
        """
        self._client = client
        self._config = config or client.config

    async def investigate(
        self,
        context: DebugContext,
    ) -> DebugAssistantResults:
        """
        Investigate a debug context and return assistance results.

        Args:
            context: Debug context with error information.

        Returns:
            DebugAssistantResults with hints, related diffs, and similar issues.
        """
        start_time = time.time()

        context = self._enrich_context(context)

        similar_issues_task = asyncio.create_task(
            self._find_similar_issues(context),
            name="similar_issues",
        )
        related_diffs_task = asyncio.create_task(
            self._find_related_diffs(context),
            name="related_diffs",
        )
        sev_task = asyncio.create_task(
            self._find_related_sevs(context),
            name="related_sevs",
        )

        results = await asyncio.gather(
            similar_issues_task,
            related_diffs_task,
            sev_task,
            return_exceptions=True,
        )

        similar_issues: list[KnowledgeResult] = (
            results[0] if not isinstance(results[0], Exception) else []
        )
        related_diffs: list[RelatedDiff] = (
            results[1] if not isinstance(results[1], Exception) else []
        )
        sev_results: list[KnowledgeResult] = (
            results[2] if not isinstance(results[2], Exception) else []
        )

        all_similar_issues = similar_issues + sev_results

        hints = self._synthesize_hints(context, all_similar_issues, related_diffs)

        execution_time_ms = int((time.time() - start_time) * 1000)

        return DebugAssistantResults(
            context=context,
            hints=hints,
            related_diffs=related_diffs,
            similar_issues=all_similar_issues,
            execution_time_ms=execution_time_ms,
        )

    def _enrich_context(self, context: DebugContext) -> DebugContext:
        """Enrich debug context with extracted patterns."""
        if not context.error_pattern and context.error_message:
            context.error_pattern = self._extract_error_pattern(context.error_message)

        return context

    def _extract_error_pattern(self, error_message: str) -> str:
        """Extract a searchable pattern from error message."""
        pattern = error_message.split(":")
        if pattern:
            return pattern[0].strip()
        return error_message[:100]

    async def _find_similar_issues(
        self,
        context: DebugContext,
    ) -> list[KnowledgeResult]:
        """Find similar issues in tasks."""
        try:
            response = await self._client.invoke_tool(
                tool_name="task.search",
                parameters={
                    "query": context.error_pattern or context.error_message,
                    "limit": 5,
                    "status": "any",
                },
            )
            return self._parse_task_results(response.data)
        except Exception as e:
            logger.warning("Failed to search tasks: %s", e)
            return []

    async def _find_related_diffs(
        self,
        context: DebugContext,
    ) -> list[RelatedDiff]:
        """Find diffs that touched the error location."""
        try:
            response = await self._client.invoke_tool(
                tool_name="metamate_diff_search",
                parameters={
                    "query": context.error_pattern or context.error_message,
                    "file_path": context.file_path,
                    "limit": 5,
                },
            )
            return self._parse_diff_results(response.data)
        except Exception as e:
            logger.warning("Failed to search diffs: %s", e)
            return []

    async def _find_related_sevs(
        self,
        context: DebugContext,
    ) -> list[KnowledgeResult]:
        """Find related SEVs."""
        try:
            response = await self._client.invoke_tool(
                tool_name="sev.search",
                parameters={
                    "keywords": context.error_pattern or context.error_message,
                    "limit": 3,
                },
            )
            return self._parse_sev_results(response.data)
        except Exception as e:
            logger.warning("Failed to search SEVs: %s", e)
            return []

    def _parse_task_results(self, data: Any) -> list[KnowledgeResult]:
        """Parse task search results."""
        results: list[KnowledgeResult] = []

        if not isinstance(data, dict):
            return results

        for item in data.get("results", []):
            results.append(
                KnowledgeResult(
                    result_id=str(item.get("task_id", uuid.uuid4())),
                    source=KnowledgeSource.TASK,
                    title=item.get("title", "Untitled Task"),
                    url=item.get("url", ""),
                    snippet=item.get("description", "")[:200],
                    relevance_score=item.get("relevance_score", 0.0),
                    author=item.get("author"),
                    metadata={
                        "status": item.get("status"),
                        "priority": item.get("priority"),
                    },
                )
            )

        return results

    def _parse_diff_results(self, data: Any) -> list[RelatedDiff]:
        """Parse diff search results."""
        results: list[RelatedDiff] = []

        if not isinstance(data, dict):
            return results

        for item in data.get("results", []):
            results.append(
                RelatedDiff(
                    diff_id=str(item.get("diff_id", "")),
                    title=item.get("title", "Untitled Diff"),
                    url=item.get("url", ""),
                    relevance_score=item.get("relevance_score", 0.0),
                    author=item.get("author"),
                    files_changed=item.get("files_changed", []),
                )
            )

        return results

    def _parse_sev_results(self, data: Any) -> list[KnowledgeResult]:
        """Parse SEV search results."""
        results: list[KnowledgeResult] = []

        if not isinstance(data, dict):
            return results

        for item in data.get("results", []):
            results.append(
                KnowledgeResult(
                    result_id=str(item.get("sev_id", uuid.uuid4())),
                    source=KnowledgeSource.SEV,
                    title=item.get("title", "Untitled SEV"),
                    url=item.get("url", ""),
                    snippet=item.get("summary", "")[:200],
                    relevance_score=item.get("relevance_score", 0.0),
                    metadata={
                        "severity": item.get("severity"),
                        "status": item.get("status"),
                        "resolution": item.get("resolution"),
                    },
                )
            )

        return results

    def _synthesize_hints(
        self,
        context: DebugContext,
        similar_issues: list[KnowledgeResult],
        related_diffs: list[RelatedDiff],
    ) -> list[DebugHint]:
        """Synthesize debug hints from gathered information."""
        hints: list[DebugHint] = []
        hint_idx = 0

        if similar_issues:
            top_issue = max(similar_issues, key=lambda i: i.relevance_score)
            hint_idx += 1
            hints.append(
                DebugHint(
                    hint_id=f"hint-{hint_idx}",
                    hint_type="investigation",
                    content=f"Similar issue found: {top_issue.title}. "
                    f"Check {top_issue.url} for potential solutions.",
                    confidence=min(top_issue.relevance_score + 0.1, 1.0),
                    source="task_search",
                    citations=[
                        Citation(
                            source_id=top_issue.result_id,
                            source_type=top_issue.source.value,
                            title=top_issue.title,
                            url=top_issue.url,
                            snippet=top_issue.snippet,
                            relevance_score=top_issue.relevance_score,
                        )
                    ],
                )
            )

        if related_diffs:
            top_diff = max(related_diffs, key=lambda d: d.relevance_score)
            hint_idx += 1
            hints.append(
                DebugHint(
                    hint_id=f"hint-{hint_idx}",
                    hint_type="investigation",
                    content=f"Recent change found: {top_diff.title}. "
                    f"This diff ({top_diff.diff_id}) may be related to the issue.",
                    confidence=min(top_diff.relevance_score + 0.1, 1.0),
                    source="diff_search",
                    citations=[
                        Citation(
                            source_id=top_diff.diff_id,
                            source_type="diff",
                            title=top_diff.title,
                            url=top_diff.url,
                            snippet=f"Changed files: {', '.join(top_diff.files_changed[:3])}",
                            relevance_score=top_diff.relevance_score,
                        )
                    ],
                )
            )

        if context.error_pattern:
            hint_idx += 1
            hints.append(
                DebugHint(
                    hint_id=f"hint-{hint_idx}",
                    hint_type="fix",
                    content=f"Error pattern '{context.error_pattern}' suggests "
                    f"checking input validation and boundary conditions.",
                    confidence=0.5,
                    source="pattern_analysis",
                    citations=[],
                )
            )

        if not hints:
            hints.append(
                DebugHint(
                    hint_id="hint-fallback",
                    hint_type="investigation",
                    content="No specific hints found. Consider searching for the "
                    "error message in InternWiki or asking in relevant Workplace groups.",
                    confidence=0.3,
                    source="fallback",
                    citations=[],
                )
            )

        return sorted(hints, key=lambda h: h.confidence, reverse=True)

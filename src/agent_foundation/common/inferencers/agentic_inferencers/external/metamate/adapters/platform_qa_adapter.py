# pyre-strict
"""
Platform Q&A Adapter for Metamate Integration.

This adapter provides Q&A capabilities with:
- Metamate Assistant integration
- Citation-backed answers
- Automatic fallback to Meta AI API

IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php
Verify at https://www.internalfb.com/metamate/agent/tools before implementation.
"""

import logging
import time
import uuid
from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.interfaces import (
    FallbackClientInterface,
    MetamateClientInterface,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.exceptions import MetamateUnavailableError
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import (
    Citation,
    MetamateConfig,
    QAQuery,
    QAResponse,
)


logger: logging.Logger = logging.getLogger(__name__)


class PlatformQAAdapter:
    """
    Adapter for Platform Q&A with Metamate Assistant.

    This adapter provides:
    - Question answering with Metamate Assistant
    - Citation-backed responses
    - Automatic fallback to Meta AI API when Metamate unavailable
    - Context-aware responses

    Example:
        ```python
        adapter = PlatformQAAdapter(
            client=metamate_client,
            fallback_client=fallback_client,
        )

        query = QAQuery(
            question="How do I configure FBLearner experiments?",
            context="I'm setting up a new model training workflow",
        )

        response = await adapter.ask(query)
        print(f"Answer: {response.answer}")
        for citation in response.citations:
            print(f"  - {citation.title}: {citation.url}")
        ```
    """

    def __init__(
        self,
        client: MetamateClientInterface,
        fallback_client: FallbackClientInterface | None = None,
        config: MetamateConfig | None = None,
    ) -> None:
        """
        Initialize the Platform Q&A adapter.

        Args:
            client: Metamate client for API communication.
            fallback_client: Optional fallback client for when Metamate is unavailable.
            config: Optional configuration (uses client config if not provided).
        """
        self._client = client
        self._fallback_client = fallback_client
        self._config = config or client.config

    async def ask(
        self,
        query: QAQuery,
    ) -> QAResponse:
        """
        Ask a question and get an answer with citations.

        Args:
            query: Q&A query with question and optional context.

        Returns:
            QAResponse with answer and supporting citations.
        """
        start_time = time.time()

        try:
            if not await self._client.health_check():
                raise MetamateUnavailableError()

            response = await self._ask_metamate(query)
            response.execution_time_ms = int((time.time() - start_time) * 1000)
            return response

        except MetamateUnavailableError:
            if self._fallback_client is None or not self._config.fallback_enabled:
                raise

            logger.info("Metamate unavailable, using fallback for Q&A")
            response = await self._ask_fallback(query)
            response.execution_time_ms = int((time.time() - start_time) * 1000)
            response.fallback_used = True
            return response

    async def _ask_metamate(
        self,
        query: QAQuery,
    ) -> QAResponse:
        """Ask question using Metamate Assistant."""
        response = await self._client.invoke_tool(
            tool_name="knowledge_search",
            parameters={
                "query": query.question,
                "limit": query.max_citations,
                "doc_types": ["intern_wiki_page", "static_docs", "post"],
                "context": query.context,
            },
        )

        return self._parse_metamate_response(response.data, query)

    def _parse_metamate_response(
        self,
        data: Any,
        query: QAQuery,
    ) -> QAResponse:
        """Parse Metamate response into QAResponse."""
        citations: list[Citation] = []

        if isinstance(data, dict):
            results = data.get("results", [])
            for result in results[: query.max_citations]:
                citations.append(
                    Citation(
                        source_id=result.get("id", str(uuid.uuid4())),
                        source_type=result.get("source_type", "unknown"),
                        title=result.get("title", "Untitled"),
                        url=result.get("url", ""),
                        snippet=result.get("snippet", ""),
                        relevance_score=result.get("relevance_score", 0.0),
                    )
                )

        if citations:
            answer = self._synthesize_answer(query.question, citations)
            confidence = sum(c.relevance_score for c in citations) / len(citations)
        else:
            answer = f"I couldn't find specific information about: {query.question}"
            confidence = 0.3

        return QAResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
        )

    def _synthesize_answer(
        self,
        question: str,
        citations: list[Citation],
    ) -> str:
        """Synthesize an answer from citations."""
        if not citations:
            return f"No information found for: {question}"

        top_citation = max(citations, key=lambda c: c.relevance_score)

        answer_parts = [
            f"Based on the available documentation:\n\n{top_citation.snippet}",
        ]

        if len(citations) > 1:
            answer_parts.append(
                f"\n\nAdditional relevant sources: {len(citations) - 1}"
            )

        return "".join(answer_parts)

    async def _ask_fallback(
        self,
        query: QAQuery,
    ) -> QAResponse:
        """Ask question using fallback (Meta AI API)."""
        if self._fallback_client is None:
            return QAResponse(
                answer="Fallback not available.",
                citations=[],
                confidence=0.0,
                fallback_used=True,
            )

        fallback_response = await self._fallback_client.execute_qa_fallback(
            question=query.question,
            context=query.context,
        )

        citations: list[Citation] = []
        for citation_data in fallback_response.get("citations", []):
            citations.append(
                Citation(
                    source_id=str(uuid.uuid4()),
                    source_type="fallback",
                    title=citation_data.get("title", "Fallback Source"),
                    url=citation_data.get("url", ""),
                    snippet=citation_data.get("snippet", ""),
                    relevance_score=0.5,
                )
            )

        return QAResponse(
            answer=fallback_response.get("answer", "No answer available."),
            citations=citations,
            confidence=fallback_response.get("confidence", 0.5),
            fallback_used=True,
        )

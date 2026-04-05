# pyre-strict
"""
Mock Metamate Client for Testing.

This module provides a mock implementation of the Metamate client
for testing purposes, enabling tests to run without a live Metamate
service connection.

IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php
Verify at https://www.internalfb.com/metamate/agent/tools before implementation.
"""

import time
import uuid
from typing import Any, AsyncIterator

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.interfaces import (
    MetamateClientInterface,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.exceptions import (
    MetamateToolError,
    MetamateUnavailableError,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import (
    Citation,
    KnowledgeResult,
    KnowledgeSource,
    MetamateConfig,
    ToolResponse,
)


class MockMetamateClient(MetamateClientInterface):
    """
    Mock client for testing Metamate integration.

    This client provides configurable mock responses for testing:
    - Configurable tool responses
    - Failure injection
    - Response latency simulation
    - Health check control

    IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php:
    - knowledge_search: MetamateAgentKnowledgeSearchToolV3
    - metamate_diff_search: MetamateAgentDiffSearchTool
    - task.search: MetamateAgentTaskSearchTool
    - sev.search: MetamateAgentSevSearchTool

    Example:
        ```python
        mock_client = MockMetamateClient()
        mock_client.set_tool_response(
            "knowledge_search",
            {"results": [{"title": "Test Result"}]}
        )
        await mock_client.connect()

        response = await mock_client.invoke_tool(
            "knowledge_search",
            {"query": "test"}
        )
        ```
    """

    def __init__(
        self,
        config: MetamateConfig | None = None,
    ) -> None:
        """
        Initialize the mock Metamate client.

        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self._config = config or MetamateConfig()
        self._connected = False
        self._is_healthy = True
        self._tool_responses: dict[str, Any] = {}
        self._tool_errors: dict[str, Exception] = {}
        self._latency_ms: dict[str, int] = {}
        self._invocation_history: list[dict[str, Any]] = []
        self._sessions: dict[str, dict[str, Any]] = {}

        self._setup_default_responses()

    def _setup_default_responses(self) -> None:
        """Set up default mock responses for common tools."""
        self._tool_responses["knowledge_search"] = {
            "results": [
                {
                    "id": "wiki-1",
                    "title": "Mock Wiki Article",
                    "url": "https://internalfb.com/wiki/mock",
                    "snippet": "This is a mock wiki article for testing.",
                    "relevance_score": 0.95,
                    "source_type": "wiki",
                }
            ],
            "total_count": 1,
        }

        self._tool_responses["metamate_diff_search"] = {
            "results": [
                {
                    "diff_id": "D12345678",
                    "title": "Mock Diff",
                    "url": "https://phabricator.intern.facebook.com/D12345678",
                    "author": "testuser",
                    "relevance_score": 0.85,
                }
            ],
            "total_count": 1,
        }

        self._tool_responses["task.search"] = {
            "results": [
                {
                    "task_id": "T12345678",
                    "title": "Mock Task",
                    "url": "https://internalfb.com/tasks/12345678",
                    "status": "Open",
                    "relevance_score": 0.80,
                }
            ],
            "total_count": 1,
        }

        self._tool_responses["sev.search"] = {
            "results": [
                {
                    "sev_id": "SEV-1234",
                    "title": "Mock SEV",
                    "severity": "S2",
                    "status": "Resolved",
                    "relevance_score": 0.75,
                }
            ],
            "total_count": 1,
        }

    async def connect(self) -> None:
        """Connect to mock Metamate service."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from mock Metamate service."""
        self._connected = False
        self._invocation_history.clear()
        self._sessions.clear()

    async def health_check(self) -> bool:
        """Check mock service health."""
        return self._connected and self._is_healthy

    async def invoke_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolResponse:
        """
        Invoke a mock Metamate tool.

        Args:
            tool_name: Canonical name of the tool.
            parameters: Tool-specific parameters.

        Returns:
            ToolResponse with mock result.

        Raises:
            MetamateToolError: If configured to fail.
            MetamateUnavailableError: If service is unavailable.
        """
        if not self._connected:
            raise MetamateUnavailableError()

        self._invocation_history.append(
            {
                "tool_name": tool_name,
                "parameters": parameters,
                "timestamp": time.time(),
            }
        )

        if tool_name in self._tool_errors:
            raise self._tool_errors[tool_name]

        latency = self._latency_ms.get(tool_name, 10)
        start_time = time.time()
        await self._simulate_latency(latency)

        response_data = self._tool_responses.get(
            tool_name,
            {
                "message": f"Mock response for {tool_name}",
                "parameters": parameters,
            },
        )

        return ToolResponse(
            success=True,
            tool_name=tool_name,
            data=response_data,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )

    async def invoke_tool_streaming(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """
        Invoke a mock Metamate tool with streaming response.

        Args:
            tool_name: Canonical name of the tool.
            parameters: Tool-specific parameters.

        Yields:
            Mock streaming chunks.
        """
        if not self._connected:
            raise MetamateUnavailableError()

        self._invocation_history.append(
            {
                "tool_name": tool_name,
                "parameters": parameters,
                "timestamp": time.time(),
                "streaming": True,
            }
        )

        if tool_name in self._tool_errors:
            raise self._tool_errors[tool_name]

        response_data = self._tool_responses.get(tool_name, {})
        results = response_data.get("results", [{"chunk": 0}])

        for i, result in enumerate(results):
            await self._simulate_latency(5)
            yield {"chunk": i, "data": result}

    async def create_research_session(
        self,
        config: dict[str, Any],
    ) -> str:
        """Create a mock research session."""
        if not self._connected:
            raise MetamateUnavailableError()

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "config": config,
            "created_at": time.time(),
            "status": "active",
        }
        return session_id

    @property
    def config(self) -> MetamateConfig:
        """Get the client configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    def set_healthy(self, is_healthy: bool) -> None:
        """Set the health status of the mock service."""
        self._is_healthy = is_healthy

    def set_tool_response(self, tool_name: str, response: Any) -> None:
        """
        Set the mock response for a specific tool.

        Args:
            tool_name: Canonical name of the tool.
            response: Mock response to return.
        """
        self._tool_responses[tool_name] = response

    def set_tool_error(self, tool_name: str, error: Exception) -> None:
        """
        Configure a tool to raise an error.

        Args:
            tool_name: Canonical name of the tool.
            error: Exception to raise.
        """
        self._tool_errors[tool_name] = error

    def clear_tool_error(self, tool_name: str) -> None:
        """Clear the configured error for a tool."""
        self._tool_errors.pop(tool_name, None)

    def set_latency(self, tool_name: str, latency_ms: int) -> None:
        """
        Set simulated latency for a tool.

        Args:
            tool_name: Canonical name of the tool.
            latency_ms: Latency in milliseconds.
        """
        self._latency_ms[tool_name] = latency_ms

    def get_invocation_history(self) -> list[dict[str, Any]]:
        """Get the history of tool invocations."""
        return self._invocation_history.copy()

    def clear_invocation_history(self) -> None:
        """Clear the invocation history."""
        self._invocation_history.clear()

    def get_mock_knowledge_results(
        self,
        source: KnowledgeSource,
        count: int = 3,
    ) -> list[KnowledgeResult]:
        """Generate mock knowledge results for testing."""
        results = []
        for i in range(count):
            results.append(
                KnowledgeResult(
                    result_id=f"{source.value}-{i}",
                    source=source,
                    title=f"Mock {source.value.title()} Result {i}",
                    url=f"https://mock.fb.com/{source.value}/{i}",
                    snippet=f"This is mock content for {source.value} result {i}.",
                    relevance_score=0.9 - (i * 0.1),
                    author="mockuser",
                    created_at="2026-01-01T00:00:00Z",
                )
            )
        return results

    def get_mock_citations(self, count: int = 3) -> list[Citation]:
        """Generate mock citations for testing."""
        citations = []
        for i in range(count):
            citations.append(
                Citation(
                    source_id=f"source-{i}",
                    source_type="wiki",
                    title=f"Mock Citation {i}",
                    url=f"https://mock.fb.com/wiki/{i}",
                    snippet=f"Mock citation snippet {i}",
                    relevance_score=0.9 - (i * 0.1),
                )
            )
        return citations

    async def _simulate_latency(self, latency_ms: int) -> None:
        """Simulate network latency."""
        import asyncio

        await asyncio.sleep(latency_ms / 1000.0)

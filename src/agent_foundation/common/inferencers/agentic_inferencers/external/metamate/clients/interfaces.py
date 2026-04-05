# pyre-strict
"""
Abstract Interfaces for Metamate Clients.

This module defines the abstract base classes for Metamate client
implementations, enabling dependency injection and testing.
"""

import abc
from typing import Any, AsyncIterator

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import MetamateConfig, ToolResponse


class MetamateClientInterface(abc.ABC):
    """
    Abstract interface for Metamate client implementations.

    This interface defines the contract for communicating with
    the Metamate platform, including tool invocation, health checks,
    and session management.

    IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php
    Verify at https://www.internalfb.com/metamate/agent/tools before implementation.
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to Metamate service.

        Raises:
            MetamateConnectionError: If connection fails.
        """
        ...

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from Metamate service."""
        ...

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Check if Metamate service is available and healthy.

        Returns:
            True if service is available, False otherwise.
        """
        ...

    @abc.abstractmethod
    async def invoke_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolResponse:
        """
        Invoke a Metamate tool synchronously.

        Args:
            tool_name: Canonical name of the tool (e.g., "knowledge_search",
                "metamate_diff_search", "task.search", "sev.search").
            parameters: Tool-specific parameters.

        Returns:
            ToolResponse with the result.

        Raises:
            MetamateToolError: If tool execution fails.
            MetamateTimeoutError: If tool execution times out.
        """
        ...

    @abc.abstractmethod
    async def invoke_tool_streaming(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """
        Invoke a Metamate tool with streaming response.

        Args:
            tool_name: Canonical name of the tool.
            parameters: Tool-specific parameters.

        Yields:
            Streaming chunks from the tool.

        Raises:
            MetamateToolError: If tool execution fails.
        """
        ...

    @abc.abstractmethod
    async def create_research_session(
        self,
        config: dict[str, Any],
    ) -> str:
        """
        Create a new deep research session.

        Args:
            config: Research session configuration.

        Returns:
            Session ID for the created session.
        """
        ...

    @property
    @abc.abstractmethod
    def config(self) -> MetamateConfig:
        """Get the client configuration."""
        ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if client is connected."""
        ...


class FallbackClientInterface(abc.ABC):
    """
    Abstract interface for fallback client implementations.

    This interface defines the contract for fallback mechanisms
    when the primary Metamate service is unavailable.
    """

    @abc.abstractmethod
    async def execute_research_fallback(
        self,
        query: str,
        query_type: str,
    ) -> dict[str, Any]:
        """
        Execute research using fallback mechanism (e.g., Stan scripts).

        Args:
            query: Research query.
            query_type: Type of research (external, internal, hybrid).

        Returns:
            Research results from fallback service.
        """
        ...

    @abc.abstractmethod
    async def execute_qa_fallback(
        self,
        question: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute Q&A using fallback mechanism (e.g., Meta AI API).

        Args:
            question: Question to answer.
            context: Optional context for the question.

        Returns:
            Q&A response from fallback service.
        """
        ...

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Check if fallback service is available.

        Returns:
            True if fallback is available, False otherwise.
        """
        ...

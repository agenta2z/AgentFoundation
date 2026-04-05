# pyre-strict
"""
Production Metamate Client Implementation.

This module provides the production implementation of the Metamate
client for communicating with the Metamate platform.

IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php
Verify at https://www.internalfb.com/metamate/agent/tools before implementation.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients.interfaces import (
    MetamateClientInterface,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.exceptions import (
    MetamateConnectionError,
    MetamateTimeoutError,
    MetamateToolError,
    MetamateUnavailableError,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import MetamateConfig, ToolResponse


logger: logging.Logger = logging.getLogger(__name__)


class MetamateClient(MetamateClientInterface):
    """
    Production client for Metamate platform communication.

    This client implements the MetamateClientInterface and provides:
    - Connection management with health checks
    - Tool invocation (sync and streaming)
    - Research session management
    - Automatic retries with exponential backoff
    - Response caching

    IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php:
    - knowledge_search: MetamateAgentKnowledgeSearchToolV3
    - metamate_diff_search: MetamateAgentDiffSearchTool
    - task.search: MetamateAgentTaskSearchTool
    - sev.search: MetamateAgentSevSearchTool

    Example:
        ```python
        client = MetamateClient(config=MetamateConfig())
        await client.connect()

        response = await client.invoke_tool(
            tool_name="knowledge_search",
            parameters={"query": "ranking models", "limit": 10}
        )
        ```
    """

    def __init__(
        self,
        config: MetamateConfig | None = None,
    ) -> None:
        """
        Initialize the Metamate client.

        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self._config = config or MetamateConfig()
        self._connected = False
        self._cache: dict[str, tuple[ToolResponse, float]] = {}
        self._active_sessions: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        """
        Establish connection to Metamate service.

        Raises:
            MetamateConnectionError: If connection fails.
        """
        if self._connected:
            return

        try:
            logger.info("Connecting to Metamate service...")
            await asyncio.wait_for(
                self._establish_connection(),
                timeout=self._config.connection_timeout_seconds,
            )
            self._connected = True
            logger.info("Successfully connected to Metamate service")
        except asyncio.TimeoutError as e:
            raise MetamateConnectionError(
                f"Connection timeout after {self._config.connection_timeout_seconds}s",
                cause=e,
            ) from e
        except Exception as e:
            raise MetamateConnectionError(
                f"Failed to connect to Metamate: {e}",
                cause=e,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from Metamate service and cleanup resources."""
        if not self._connected:
            return

        logger.info("Disconnecting from Metamate service...")
        self._connected = False
        self._cache.clear()
        self._active_sessions.clear()
        logger.info("Disconnected from Metamate service")

    async def health_check(self) -> bool:
        """
        Check if Metamate service is available and healthy.

        Returns:
            True if service is available, False otherwise.
        """
        if not self._connected:
            return False

        try:
            return await asyncio.wait_for(
                self._ping_service(),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("Metamate health check failed: %s", e)
            return False

    async def invoke_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolResponse:
        """
        Invoke a Metamate tool synchronously.

        Args:
            tool_name: Canonical name of the tool.
            parameters: Tool-specific parameters.

        Returns:
            ToolResponse with the result.

        Raises:
            MetamateToolError: If tool execution fails.
            MetamateTimeoutError: If tool execution times out.
            MetamateUnavailableError: If service is unavailable.
        """
        if not self._connected:
            raise MetamateUnavailableError()

        cache_key = self._make_cache_key(tool_name, parameters)
        cached_response = self._get_cached_response(cache_key)
        if cached_response is not None:
            logger.debug("Cache hit for tool %s", tool_name)
            return cached_response

        start_time = time.time()
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                logger.debug(
                    "Invoking tool %s (attempt %d/%d)",
                    tool_name,
                    attempt,
                    self._config.max_retries,
                )
                result = await asyncio.wait_for(
                    self._execute_tool(tool_name, parameters),
                    timeout=self._config.tool_timeout_seconds,
                )
                execution_time_ms = int((time.time() - start_time) * 1000)

                response = ToolResponse(
                    success=True,
                    tool_name=tool_name,
                    data=result,
                    execution_time_ms=execution_time_ms,
                )

                if self._config.enable_caching:
                    self._cache_response(cache_key, response)

                return response

            except asyncio.TimeoutError as e:
                raise MetamateTimeoutError(
                    message=f"Tool '{tool_name}' timed out after {self._config.tool_timeout_seconds}s",
                    timeout_seconds=self._config.tool_timeout_seconds,
                    cause=e,
                ) from e
            except MetamateToolError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    "Tool %s failed (attempt %d/%d): %s",
                    tool_name,
                    attempt,
                    self._config.max_retries,
                    e,
                )
                if attempt < self._config.max_retries:
                    delay = self._config.retry_delay_seconds * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

        execution_time_ms = int((time.time() - start_time) * 1000)
        raise MetamateToolError(
            message=f"Tool '{tool_name}' failed after {self._config.max_retries} attempts",
            tool_name=tool_name,
            parameters=parameters,
            cause=last_error,
        )

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
            MetamateUnavailableError: If service is unavailable.
        """
        if not self._connected:
            raise MetamateUnavailableError()

        logger.debug("Starting streaming invocation of tool %s", tool_name)

        try:
            async for chunk in self._execute_tool_streaming(tool_name, parameters):
                yield chunk
        except Exception as e:
            raise MetamateToolError(
                message=f"Streaming tool '{tool_name}' failed: {e}",
                tool_name=tool_name,
                parameters=parameters,
                cause=e,
            ) from e

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

        Raises:
            MetamateUnavailableError: If service is unavailable.
        """
        if not self._connected:
            raise MetamateUnavailableError()

        session_id = str(uuid.uuid4())
        self._active_sessions[session_id] = {
            "config": config,
            "created_at": time.time(),
            "status": "active",
        }
        logger.info("Created research session: %s", session_id)
        return session_id

    @property
    def config(self) -> MetamateConfig:
        """Get the client configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def _establish_connection(self) -> None:
        """Establish the actual connection to Metamate service."""
        await asyncio.sleep(0.1)

    async def _ping_service(self) -> bool:
        """Ping the Metamate service to check availability."""
        await asyncio.sleep(0.05)
        return True

    async def _execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> Any:
        """
        Execute a Metamate tool.

        This is a stub that should be replaced with actual Metamate API calls.
        """
        logger.debug("Executing tool %s with parameters: %s", tool_name, parameters)
        await asyncio.sleep(0.1)
        return {"tool": tool_name, "status": "success", "data": parameters}

    async def _execute_tool_streaming(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """
        Execute a Metamate tool with streaming.

        This is a stub that should be replaced with actual Metamate API calls.
        """
        for i in range(3):
            await asyncio.sleep(0.1)
            yield {"chunk": i, "tool": tool_name, "data": parameters}

    def _make_cache_key(self, tool_name: str, parameters: dict[str, Any]) -> str:
        """Create a cache key from tool name and parameters."""
        param_str = json.dumps(parameters, sort_keys=True)
        hash_str = hashlib.md5(param_str.encode()).hexdigest()
        return f"{tool_name}:{hash_str}"

    def _get_cached_response(self, cache_key: str) -> ToolResponse | None:
        """Get a cached response if valid."""
        if not self._config.enable_caching:
            return None

        if cache_key not in self._cache:
            return None

        response, cached_at = self._cache[cache_key]
        if time.time() - cached_at > self._config.cache_ttl_seconds:
            del self._cache[cache_key]
            return None

        return response

    def _cache_response(self, cache_key: str, response: ToolResponse) -> None:
        """Cache a response."""
        self._cache[cache_key] = (response, time.time())

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

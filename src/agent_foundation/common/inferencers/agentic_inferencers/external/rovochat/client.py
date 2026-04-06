# pyre-strict
from __future__ import annotations

"""RovoChat Inferencer — async HTTP client.

Provides ``RovoChatClient`` for interacting with the RovoChat REST API.
Uses ``httpx`` (soft dependency) for async HTTP requests with NDJSON
streaming support.

Usage::

    from rovochat.auth import RovoChatAuth
    from rovochat.client import RovoChatClient
    from rovochat.types import RovoChatConfig

    config = RovoChatConfig(cloud_id="...")
    auth = RovoChatAuth(uct_token="...")
    client = RovoChatClient(config=config, auth=auth)

    # Create a conversation
    conv_id = await client.create_conversation()

    # Stream a message
    async for event in client.send_message_stream(conv_id, "What is Rovo?"):
        print(event)

    # Or get the full response
    response = await client.send_message(conv_id, "What is Rovo?")
    print(response.content)
"""

import logging
import uuid
from typing import Any, AsyncIterator, Optional

from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.auth import (
    RovoChatAuth,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.common import (
    CONVERSATION_PATH,
    GATEWAY_CONVERSATION_PATH,
    GATEWAY_MESSAGE_STREAM_PATH,
    MESSAGE_STREAM_PATH,
    build_adf_message,
    extract_text_from_event,
    is_terminal_event,
    parse_ndjson_line,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.exceptions import (
    RovoChatAPIError,
    RovoChatAuthError,
    RovoChatConnectionError,
    RovoChatTimeoutError,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.types import (
    ConversationInfo,
    RovoChatConfig,
    RovoChatResponse,
    StreamEvent,
)

logger: logging.Logger = logging.getLogger(__name__)


class RovoChatClient:
    """Async HTTP client for the RovoChat REST API.

    Manages conversation creation, message sending with NDJSON streaming,
    and response accumulation. Uses ``httpx.AsyncClient`` for non-blocking
    HTTP operations.

    Attributes:
        config: RovoChat API configuration.
        auth: Authentication manager.
    """

    def __init__(self, config: RovoChatConfig, auth: RovoChatAuth) -> None:
        self.config = config
        self.auth = auth

    def _build_headers(self) -> dict[str, str]:
        """Construct HTTP headers for RovoChat API calls.

        Combines authentication headers with required product/experience
        headers and the Cloud ID. Uses lowercase header names for gateway
        mode (as expected by the Atlassian gateway).

        Returns:
            Dictionary of HTTP headers.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        }

        if self.config.use_gateway:
            # Gateway mode: lowercase headers as expected by Atlassian gateway
            headers["x-product"] = self.config.product
            headers["x-experience-id"] = self.config.experience_id
            if self.config.cloud_id:
                headers["x-cloudid"] = self.config.cloud_id
        else:
            # Direct mode: standard capitalized headers
            headers["X-Product"] = self.config.product
            headers["X-Experience-Id"] = self.config.experience_id
            if self.config.cloud_id:
                headers["Atl-Cloudid"] = self.config.cloud_id

        # Add Lanyard config if provided (direct mode only)
        if self.config.lanyard_config and not self.config.use_gateway:
            headers["Lanyard-Config"] = self.config.lanyard_config

        # Add authentication headers
        auth_headers = self.auth.get_auth_headers()
        headers.update(auth_headers)

        return headers

    def _conversation_url(self) -> str:
        """Return the conversation creation URL based on gateway mode."""
        path = GATEWAY_CONVERSATION_PATH if self.config.use_gateway else CONVERSATION_PATH
        return f"{self.config.base_url}{path}"

    def _message_stream_url(self, conversation_id: str) -> str:
        """Return the message stream URL based on gateway mode."""
        template = GATEWAY_MESSAGE_STREAM_PATH if self.config.use_gateway else MESSAGE_STREAM_PATH
        path = template.format(conversation_id=conversation_id)
        return f"{self.config.base_url}{path}"

    def _build_message_body(
        self,
        text: str,
        agent_named_id: str = "",
        agent_id: str = "",
        message_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Construct the JSON request body for sending a message.

        Wraps the plain text prompt into ADF format and builds the full
        ``RovoChatMessageStreamRequest`` body.

        Args:
            text: Plain text message to send.
            agent_named_id: Named identifier for the recipient agent.
            agent_id: Agent ID for routing.
            message_id: Optional client-generated message ID.
            **kwargs: Additional body fields to include.

        Returns:
            Request body dictionary.
        """
        body: dict[str, Any] = {
            "content": build_adf_message(text),
            "mimeType": "text/adf",
            "store_message": self.config.store_message,
            "citations_enabled": self.config.citations_enabled,
            "context": kwargs.get("context", {}),
        }

        if agent_named_id:
            body["recipient_agent_named_id"] = agent_named_id
        if agent_id:
            body["agentId"] = agent_id
        if message_id:
            body["messageId"] = message_id

        return body

    async def create_conversation(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> ConversationInfo:
        """Create a new RovoChat conversation.

        Args:
            agent_id: Optional agent ID to assign to the conversation.
            name: Optional conversation name.

        Returns:
            ``ConversationInfo`` with the new conversation's details.

        Raises:
            RovoChatConnectionError: If the API is unreachable.
            RovoChatAuthError: If authentication fails (401/403).
            RovoChatAPIError: If the API returns an error response.
        """
        try:
            import httpx
        except ImportError as e:
            raise RovoChatConnectionError(
                f"httpx package required for RovoChat client: {e}. "
                "Install it with: pip install httpx"
            ) from e

        url = self._conversation_url()
        headers = self._build_headers()
        # Override Accept for non-streaming endpoint
        headers["Accept"] = "application/json"

        body: dict[str, Any] = {}
        if agent_id:
            body["agentId"] = agent_id
        if name:
            body["name"] = name

        logger.debug("Creating conversation at %s", url)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds)
            ) as client:
                response = await client.post(url, headers=headers, json=body)
        except httpx.ConnectError as e:
            raise RovoChatConnectionError(
                f"Cannot reach RovoChat API at {url}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise RovoChatTimeoutError(
                f"Timeout creating conversation at {url}: {e}"
            ) from e
        except Exception as e:
            raise RovoChatConnectionError(
                f"HTTP error creating conversation: {e}"
            ) from e

        _check_response_status(response)

        try:
            data = response.json()
        except Exception:
            data = {}

        conversation_id = data.get("id", "")
        if not conversation_id:
            raise RovoChatAPIError(
                "Conversation creation returned no ID",
                status_code=response.status_code,
                details={"response": data},
            )

        agent_data = data.get("agent") or {}
        info = ConversationInfo(
            conversation_id=str(conversation_id),
            owner=str(data.get("owner", "")),
            tenant_id=str(data.get("tenantId") or data.get("tenant_id") or ""),
            agent_name=str(agent_data.get("name", "")),
            agent_id=str(agent_data.get("id", "")),
            metadata=data,
        )

        logger.info(
            "Created conversation: id=%s, agent=%s",
            info.conversation_id,
            info.agent_name or "(default)",
        )
        return info

    async def send_message_stream(
        self,
        conversation_id: str,
        text: str,
        agent_named_id: str = "",
        agent_id: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Send a message and yield NDJSON stream events.

        Opens a streaming HTTP connection to the RovoChat message endpoint
        and yields ``StreamEvent`` objects as NDJSON lines arrive.

        Args:
            conversation_id: Target conversation ID.
            text: Plain text message to send.
            agent_named_id: Named identifier for the recipient agent.
            agent_id: Agent ID for routing.
            **kwargs: Additional request body fields.

        Yields:
            ``StreamEvent`` objects parsed from NDJSON lines.

        Raises:
            RovoChatConnectionError: If the API is unreachable.
            RovoChatAuthError: If authentication fails.
            RovoChatAPIError: If the API returns an error.
            RovoChatTimeoutError: If the stream times out.
        """
        try:
            import httpx
        except ImportError as e:
            raise RovoChatConnectionError(
                f"httpx package required for RovoChat client: {e}. "
                "Install it with: pip install httpx"
            ) from e

        url = self._message_stream_url(conversation_id)
        headers = self._build_headers()
        body = self._build_message_body(
            text=text,
            agent_named_id=agent_named_id,
            agent_id=agent_id,
            message_id=str(uuid.uuid4()),
            **kwargs,
        )

        logger.debug(
            "Sending message to conversation=%s (text=%s...)",
            conversation_id,
            text[:50],
        )

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.timeout_seconds,
                    read=self.config.stream_timeout_seconds,
                    write=self.config.timeout_seconds,
                    pool=self.config.timeout_seconds,
                )
            ) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=body
                ) as response:
                    # Check status before streaming
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        _check_response_status_raw(
                            response.status_code, error_body.decode("utf-8", errors="replace")
                        )

                    # Stream NDJSON lines
                    async for line in response.aiter_lines():
                        event = parse_ndjson_line(line)
                        if event is not None:
                            yield event

        except RovoChatAuthError:
            raise
        except RovoChatAPIError:
            raise
        except httpx.ConnectError as e:
            raise RovoChatConnectionError(
                f"Cannot reach RovoChat API at {url}: {e}"
            ) from e
        except httpx.ReadTimeout as e:
            raise RovoChatTimeoutError(
                f"Stream read timeout for conversation={conversation_id}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise RovoChatTimeoutError(
                f"Timeout sending message to conversation={conversation_id}: {e}"
            ) from e
        except (RovoChatConnectionError, RovoChatTimeoutError):
            raise
        except Exception as e:
            raise RovoChatConnectionError(
                f"HTTP error sending message: {e}"
            ) from e

    async def send_message(
        self,
        conversation_id: str,
        text: str,
        agent_named_id: str = "",
        agent_id: str = "",
        **kwargs: Any,
    ) -> RovoChatResponse:
        """Send a message and return the complete response.

        Convenience method that accumulates all stream events and extracts
        the final response text.

        Args:
            conversation_id: Target conversation ID.
            text: Plain text message to send.
            agent_named_id: Named identifier for the recipient agent.
            agent_id: Agent ID for routing.
            **kwargs: Additional request body fields.

        Returns:
            ``RovoChatResponse`` with accumulated content and metadata.
        """
        events: list[StreamEvent] = []
        text_parts: list[str] = []
        final_response_text: Optional[str] = None

        async for event in self.send_message_stream(
            conversation_id, text, agent_named_id, agent_id, **kwargs
        ):
            events.append(event)
            extracted = extract_text_from_event(event)
            if extracted:
                text_parts.append(extracted)
                # If this is a terminal event, its text is the final response
                if is_terminal_event(event):
                    final_response_text = extracted

        # Determine content: prefer final response text (complete),
        # fall back to accumulation of all text parts (incremental)
        if final_response_text:
            content = final_response_text
        else:
            content = "".join(text_parts)

        # Extract message ID and citations from terminal event
        message_id: Optional[str] = None
        citations: list[dict[str, Any]] = []
        for event in reversed(events):
            if is_terminal_event(event):
                msg = event.data.get("message", {})
                if isinstance(msg, dict):
                    message_id = msg.get("id") or msg.get("messageId")
                    citations = msg.get("citations", [])
                break

        return RovoChatResponse(
            content=content,
            conversation_id=conversation_id,
            message_id=str(message_id) if message_id else None,
            citations=citations if isinstance(citations, list) else [],
            events=events,
        )


def _check_response_status(response: Any) -> None:
    """Check HTTP response status and raise appropriate exceptions.

    Args:
        response: ``httpx.Response`` object.

    Raises:
        RovoChatAuthError: For 401/403 responses.
        RovoChatAPIError: For other 4xx/5xx responses.
    """
    status = response.status_code
    if status < 400:
        return

    try:
        body = response.text
    except Exception:
        body = ""

    _check_response_status_raw(status, body)


def _check_response_status_raw(status_code: int, body: str) -> None:
    """Check HTTP status code and raise appropriate exceptions.

    Args:
        status_code: HTTP status code.
        body: Response body text.

    Raises:
        RovoChatAuthError: For 401/403 responses.
        RovoChatAPIError: For other 4xx/5xx responses.
    """
    if status_code < 400:
        return

    details = {"response_body": body[:1000]} if body else {}

    if status_code in (401, 403):
        raise RovoChatAuthError(
            f"Authentication failed (HTTP {status_code}). "
            "Check your UCT token or ASAP credentials.",
            details=details,
        )

    raise RovoChatAPIError(
        f"RovoChat API error",
        status_code=status_code,
        details=details,
    )

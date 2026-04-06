# pyre-strict
"""RovoChat Inferencer — type definitions.

This module contains all type definitions used across the RovoChat
integration, including configuration, stream events, and response types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RovoChatConfig:
    """Configuration for the RovoChat HTTP client.

    Attributes:
        base_url: Base URL of the RovoChat API
            (e.g. ``"https://convo-ai.stg.atlassian.com"``).
        cloud_id: Atlassian Cloud ID for the target tenant.
        lanyard_config: Lanyard configuration ID for authorization.
            Empty string means no Lanyard-Config header is sent.
        product: Product identifier sent via ``X-Product`` header.
        experience_id: Experience identifier sent via ``X-Experience-Id`` header.
        store_message: Whether to persist messages server-side.
        citations_enabled: Whether to request citations in responses.
        timeout_seconds: HTTP request timeout in seconds.
        stream_timeout_seconds: Timeout for streaming response reads.
        max_retries: Maximum retry attempts for transient failures.
    """

    base_url: str = ""
    cloud_id: str = ""
    lanyard_config: str = ""
    product: str = "rovo"
    experience_id: str = "ai-mate"
    store_message: bool = True
    citations_enabled: bool = True
    timeout_seconds: float = 60.0
    stream_timeout_seconds: float = 600.0
    max_retries: int = 3
    use_gateway: bool = False


# =============================================================================
# Stream Events
# =============================================================================


@dataclass
class StreamEvent:
    """A single event from the RovoChat NDJSON response stream.

    Attributes:
        event_type: The event type string (e.g. ``"FINAL_RESPONSE"``,
            ``"CONTENT_DELTA"``, ``"STATUS_UPDATE"``, ``"ERROR"``).
        data: Parsed JSON data from the event line.
        raw_line: The original NDJSON line string.
    """

    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""


# =============================================================================
# Response Types
# =============================================================================


@dataclass
class RovoChatMessage:
    """A single message in a RovoChat conversation.

    Attributes:
        role: Message author role (``"user"`` or ``"assistant"``).
        content: The message text content.
        message_id: Server-assigned message identifier.
        metadata: Additional message metadata.
    """

    role: str
    content: str
    message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RovoChatResponse:
    """Complete response from a RovoChat message exchange.

    Attributes:
        content: The assistant's response text.
        conversation_id: Conversation ID for multi-turn follow-ups.
        message_id: Server-assigned message ID for the response.
        citations: List of citation objects returned by the service.
        events: All stream events received during the exchange.
        metadata: Additional response metadata (e.g. agent info, timing).
    """

    content: str = ""
    conversation_id: str = ""
    message_id: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    events: list[StreamEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationInfo:
    """Information about a created or resumed conversation.

    Attributes:
        conversation_id: The unique conversation identifier.
        owner: Account ID of the conversation owner.
        tenant_id: Atlassian Cloud tenant ID.
        agent_name: Name of the agent assigned to the conversation.
        agent_id: ID of the agent assigned to the conversation.
        metadata: Additional conversation metadata from the API.
    """

    conversation_id: str
    owner: str = ""
    tenant_id: str = ""
    agent_name: str = ""
    agent_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

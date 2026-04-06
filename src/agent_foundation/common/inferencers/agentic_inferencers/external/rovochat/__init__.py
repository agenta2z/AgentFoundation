# pyre-strict
"""RovoChat Integration — standalone inferencer for Atlassian's RovoChat service.

This package provides a streaming inferencer that queries the RovoChat
conversational AI platform via its REST API. It follows the same pattern
as the MetaMate inferencer but uses HTTP/NDJSON streaming instead of
GraphQL polling.

Components:
    - ``RovoChatInferencer``: Async streaming inferencer (StreamingInferencerBase)
    - ``RovoChatClient``: Low-level async HTTP client for the RovoChat API
    - ``RovoChatAuth``: Authentication manager (UCT token / ASAP JWT)
    - ``RovoChatConfig``: API configuration dataclass
    - ``RovoChatResponse``: Complete response with content, citations, events
    - ``StreamEvent``: Individual NDJSON stream event
    - Exception hierarchy: ``RovoChatError`` and subclasses

Quick Start::

    from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat import (
        RovoChatInferencer,
    )

    # With a pre-generated UCT token
    inferencer = RovoChatInferencer(
        base_url="https://team.atlassian.com",
        cloud_id="your-cloud-id",
        uct_token="your-uct-token",
    )

    # Single query
    result = inferencer("What is Atlassian Rovo?")
    print(result)

    # Streaming
    async for chunk in inferencer.ainfer_streaming("Explain Rovo agents"):
        print(chunk, end="", flush=True)

    # Multi-turn
    r1 = inferencer.new_session("I'm building a Jira integration")
    r2 = inferencer("What APIs should I use?")
"""

# Inferencer
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.rovochat_inferencer import (  # noqa: F401
    RovoChatInferencer,
)

# Client
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.client import (  # noqa: F401
    RovoChatClient,
)

# Auth
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.auth import (  # noqa: F401
    RovoChatAuth,
)

# Types
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.types import (  # noqa: F401
    ConversationInfo,
    RovoChatConfig,
    RovoChatMessage,
    RovoChatResponse,
    StreamEvent,
)

# Exceptions
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.exceptions import (  # noqa: F401
    RovoChatAPIError,
    RovoChatAuthError,
    RovoChatConnectionError,
    RovoChatError,
    RovoChatTimeoutError,
)

__all__ = [
    # Inferencer
    "RovoChatInferencer",
    # Client
    "RovoChatClient",
    # Auth
    "RovoChatAuth",
    # Types
    "RovoChatConfig",
    "RovoChatResponse",
    "RovoChatMessage",
    "ConversationInfo",
    "StreamEvent",
    # Exceptions
    "RovoChatError",
    "RovoChatAuthError",
    "RovoChatConnectionError",
    "RovoChatTimeoutError",
    "RovoChatAPIError",
]

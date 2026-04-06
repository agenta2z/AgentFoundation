# pyre-strict
from __future__ import annotations

"""RovoChat Inferencer — shared constants, ADF helpers, and parsing utilities.

Provides:
- API defaults (base URL, headers, timeouts)
- ``build_adf_message()`` — wrap plain text into Atlassian Document Format
- ``parse_ndjson_line()`` — parse a single NDJSON line
- ``extract_text_from_event()`` — extract text content from a stream event
- ``is_terminal_event()`` — detect end-of-stream events
- ``needs_continuation()`` — detect clarification questions requiring follow-up
"""

import json
import logging
from typing import Any, List

from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.types import (
    StreamEvent,
)

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL: str = "https://convo-ai.us-east-1.staging.atl-paas.net"
DEFAULT_PRODUCT: str = "rovo"
DEFAULT_EXPERIENCE_ID: str = "ai-mate"
DEFAULT_TIMEOUT: float = 60.0
DEFAULT_STREAM_TIMEOUT: float = 600.0
DEFAULT_TOTAL_TIMEOUT: int = 1800
DEFAULT_IDLE_TIMEOUT: int = 600
MAX_CONTINUATIONS: int = 5

# API path templates — two variants:
#
# **Direct** (staging/internal): ``/api/rovo/v1/chat/...``
#   Used when calling the convo-ai service directly (e.g. via staging URL).
#
# **Gateway** (production): ``/gateway/api/assist/rovo/v1/chat/...``
#   Used when calling via the Atlassian site gateway
#   (e.g. ``https://hello.atlassian.net/gateway/...``).
#   This is the standard path for Basic Auth with an API token.
#
API_V1_PREFIX: str = "/api/rovo/v1/chat"
GATEWAY_API_V1_PREFIX: str = "/gateway/api/assist/rovo/v1/chat"

CONVERSATION_PATH: str = f"{API_V1_PREFIX}/conversation"
MESSAGE_STREAM_PATH: str = (
    f"{API_V1_PREFIX}/conversation/{{conversation_id}}/message/stream"
)

GATEWAY_CONVERSATION_PATH: str = f"{GATEWAY_API_V1_PREFIX}/conversation"
GATEWAY_MESSAGE_STREAM_PATH: str = (
    f"{GATEWAY_API_V1_PREFIX}/conversation/{{conversation_id}}/message/stream"
)

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------
# Primary ROVOCHAT_* env vars
ENV_BASE_URL: str = "ROVOCHAT_BASE_URL"
ENV_CLOUD_ID: str = "ROVOCHAT_CLOUD_ID"
ENV_EMAIL: str = "ROVOCHAT_EMAIL"
ENV_API_TOKEN: str = "ROVOCHAT_API_TOKEN"
ENV_UCT_TOKEN: str = "ROVOCHAT_UCT_TOKEN"
ENV_ASAP_ISSUER: str = "ROVOCHAT_ASAP_ISSUER"
ENV_ASAP_PRIVATE_KEY: str = "ROVOCHAT_ASAP_PRIVATE_KEY"
ENV_ASAP_KEY_ID: str = "ROVOCHAT_ASAP_KEY_ID"
ENV_ASAP_AUDIENCE: str = "ROVOCHAT_ASAP_AUDIENCE"

# Fallback env vars — common Atlassian credential names
ENV_FALLBACK_EMAIL: tuple = ("JIRA_EMAIL", "ATLASSIAN_EMAIL")
ENV_FALLBACK_API_TOKEN: tuple = ("JIRA_API_TOKEN", "ATLASSIAN_API_TOKEN")
ENV_FALLBACK_BASE_URL: tuple = ("JIRA_URL",)

# Auto-continue reply when the agent asks clarification questions
AUTO_CONTINUE_REPLY: str = (
    "Please proceed with your best judgment. Provide a comprehensive, "
    "detailed answer with all relevant information."
)

# ---------------------------------------------------------------------------
# Terminal event types — indicate the stream is complete
# ---------------------------------------------------------------------------

_TERMINAL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "FINAL_RESPONSE",
        "ERROR",
        "COMPLETED",
        "DONE",
    }
)

# Non-content event types — these events are informational and don't
# contain response text. Used for filtering during text extraction.
_NON_CONTENT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "HEART_BEAT",
        "TRACE",
        "CONVERSATION_CHANNEL_DATA",
    }
)

# ---------------------------------------------------------------------------
# Continuation detection — phrases that indicate a clarification question
# ---------------------------------------------------------------------------

_CONTINUATION_PHRASES: List[str] = [
    "should i proceed",
    "shall i proceed",
    "would you like me to",
    "do you want me to",
    "please proceed",
    "let me know if",
    "want me to look into",
    "want me to research",
    "narrow this",
    "tell me your",
    "tell me whether",
    "tell me which",
    "which one are you",
    "are you most interested in",
    "i can tailor",
    "more specific",
    "could you clarify",
    "can you specify",
]


# ---------------------------------------------------------------------------
# ADF (Atlassian Document Format) helpers
# ---------------------------------------------------------------------------


def build_adf_message(text: str) -> dict[str, Any]:
    """Wrap plain text into a minimal ADF (Atlassian Document Format) document.

    Produces the structure expected by ``RovoChatMessageStreamRequest.content``:

    .. code-block:: json

        {
          "version": 1,
          "type": "doc",
          "content": [{
            "type": "paragraph",
            "content": [{"type": "text", "text": "..."}]
          }]
        }

    Multi-line text is split into separate paragraphs.

    Args:
        text: Plain text string to wrap.

    Returns:
        ADF document as a dictionary.
    """
    paragraphs: list[dict[str, Any]] = []
    for line in text.split("\n"):
        if line.strip():
            paragraphs.append(
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": line}],
                }
            )
        else:
            # Empty line → empty paragraph (preserves spacing)
            paragraphs.append({"type": "paragraph", "content": []})

    # Ensure at least one paragraph
    if not paragraphs:
        paragraphs.append(
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text}],
            }
        )

    return {
        "version": 1,
        "type": "doc",
        "content": paragraphs,
    }


# ---------------------------------------------------------------------------
# NDJSON parsing
# ---------------------------------------------------------------------------


def parse_ndjson_line(line: str) -> StreamEvent | None:
    """Parse a single NDJSON (newline-delimited JSON) line into a StreamEvent.

    Handles:
    - Empty / whitespace-only lines (returns None)
    - Malformed JSON (logs warning, returns None)
    - Valid JSON objects with or without a ``type`` field

    Args:
        line: A single line from the NDJSON response stream.

    Returns:
        A ``StreamEvent`` instance, or ``None`` if the line is empty/invalid.
    """
    stripped = line.strip()
    if not stripped:
        return None

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("Malformed NDJSON line (ignoring): %s", stripped[:200])
        return None

    if not isinstance(data, dict):
        logger.warning("NDJSON line is not a JSON object: %s", stripped[:200])
        return None

    # Extract event type from various possible fields
    event_type = (
        data.get("type")
        or data.get("eventType")
        or data.get("event_type")
        or data.get("event")
        or "UNKNOWN"
    )

    return StreamEvent(
        event_type=str(event_type),
        data=data,
        raw_line=stripped,
    )


# ---------------------------------------------------------------------------
# Response text extraction
# ---------------------------------------------------------------------------


def extract_text_from_event(event: StreamEvent) -> str | None:
    """Extract text content from a stream event.

    Only extracts text from content-bearing event types (``ANSWER_PART``,
    ``FINAL_RESPONSE``, etc.). Non-content events such as ``TRACE``,
    ``HEART_BEAT``, and ``CONVERSATION_CHANNEL_DATA`` are skipped even
    if they contain a ``message.content`` field (which holds search
    queries, document titles, or other metadata — not response text).

    Supports multiple response formats:

    1. ``event.data["message"]["content"]`` — response text (ANSWER_PART / FINAL_RESPONSE)
    2. ``event.data["content"]`` — content delta
    3. ``event.data["text"]`` — plain text field
    4. ``event.data["delta"]["content"]`` — delta-style content
    5. ``event.data["message"]["body"]["content"]`` — nested body with ADF content

    For ADF content, extracts text from paragraph nodes.

    Args:
        event: A parsed ``StreamEvent``.

    Returns:
        Extracted text string, or ``None`` if no text found.
    """
    # Skip non-content event types — these contain metadata, not response text
    if event.event_type.upper() in _NON_CONTENT_EVENT_TYPES:
        return None

    data = event.data

    # 1. message.content (string)
    message = data.get("message")
    if isinstance(message, dict):
        # Direct string content
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

        # ADF content in message body
        body = message.get("body")
        if isinstance(body, dict):
            body_content = body.get("content")
            if isinstance(body_content, str) and body_content.strip():
                return body_content
            # ADF doc in body
            if isinstance(body_content, list):
                return _extract_text_from_adf_nodes(body_content)
            adf_content = body.get("content")
            if isinstance(adf_content, dict) and adf_content.get("type") == "doc":
                return _extract_text_from_adf_nodes(adf_content.get("content", []))

    # 2. Direct content field (string)
    content = data.get("content")
    if isinstance(content, str) and content.strip():
        return content

    # 3. Direct text field
    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text

    # 4. Delta-style content
    delta = data.get("delta")
    if isinstance(delta, dict):
        delta_content = delta.get("content")
        if isinstance(delta_content, str) and delta_content.strip():
            return delta_content

    return None


def _extract_text_from_adf_nodes(nodes: list[dict[str, Any]]) -> str | None:
    """Recursively extract text from ADF content nodes.

    Args:
        nodes: List of ADF node dictionaries.

    Returns:
        Concatenated text, or ``None`` if no text found.
    """
    parts: list[str] = []

    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_type = node.get("type", "")

        if node_type == "text":
            text = node.get("text", "")
            if text:
                parts.append(text)
        elif node_type == "codeBlock":
            # Extract code content
            code_content = node.get("content", [])
            for code_node in code_content:
                if isinstance(code_node, dict) and code_node.get("type") == "text":
                    parts.append(code_node.get("text", ""))
            continue

        # Recurse into child content
        children = node.get("content")
        if isinstance(children, list):
            child_text = _extract_text_from_adf_nodes(children)
            if child_text:
                parts.append(child_text)

    return "\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Terminal event detection
# ---------------------------------------------------------------------------


def is_terminal_event(event: StreamEvent) -> bool:
    """Check if a stream event indicates the end of the response stream.

    Args:
        event: A parsed ``StreamEvent``.

    Returns:
        ``True`` if this event is terminal (stream should stop).
    """
    if event.event_type.upper() in _TERMINAL_EVENT_TYPES:
        return True

    # Also check for status fields within the data
    status = event.data.get("status", "")
    if isinstance(status, str) and status.upper() in ("COMPLETED", "ERROR", "DONE", "FAILED"):
        return True

    return False


# ---------------------------------------------------------------------------
# Continuation detection
# ---------------------------------------------------------------------------


def needs_continuation(text: str) -> bool:
    """Return True when the assistant's response is a clarification question.

    Heuristics (matching MetaMate's approach):
    1. Short text (< 2000 chars) containing a known continuation phrase.
    2. Short text (< 800 chars) ending with a question mark.

    Args:
        text: The assistant response text.

    Returns:
        Whether a continuation reply should be sent.
    """
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    for phrase in _CONTINUATION_PHRASES:
        if phrase in lower and len(stripped) < 2000:
            return True
    if len(stripped) < 800 and stripped.endswith("?"):
        return True
    return False

"""SSE and Bedrock event-stream parsing utilities for AI Gateway streaming.

Provides parsers for two streaming formats:
- SSE (Server-Sent Events): Used by the proximity proxy (Anthropic-native format)
- Bedrock event-stream: Binary length-prefixed format used by direct mode

Both formats ultimately carry Anthropic-style events (content_block_delta, etc.).
"""

import base64
import json
import logging
import struct
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


def extract_text_delta(event: dict) -> Optional[str]:
    """Extract text from an Anthropic streaming event.

    Handles content_block_delta events with text_delta payloads.

    Args:
        event: Parsed JSON event dict from the streaming response.

    Returns:
        The text delta string, or None if the event doesn't contain text.
    """
    if event.get("type") != "content_block_delta":
        return None
    delta = event.get("delta", {})
    if delta.get("type") != "text_delta":
        return None
    return delta.get("text")


async def parse_sse_stream(response) -> AsyncIterator[str]:
    """Parse an SSE stream from an httpx streaming response, yielding text deltas.

    Handles the Anthropic-native SSE format used by the proximity proxy:
        event: content_block_delta
        data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}

    Args:
        response: An httpx streaming response (from client.stream()).

    Yields:
        Text delta strings as they arrive.
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue

        # SSE data lines start with "data: "
        if line.startswith("data: "):
            data_str = line[6:]  # Strip "data: " prefix
            if data_str == "[DONE]":
                break
            try:
                event_data = json.loads(data_str)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON SSE data: %s", data_str[:100])
                continue

            text = extract_text_delta(event_data)
            if text is not None:
                yield text


async def parse_bedrock_event_stream(response) -> AsyncIterator[str]:
    """Parse a Bedrock binary event-stream response, yielding text deltas.

    Bedrock's /invoke-with-response-stream returns a binary event-stream where
    each frame consists of:
    - 4 bytes: total length (big-endian uint32)
    - 4 bytes: headers length (big-endian uint32)
    - N bytes: headers (key-value pairs, each with type/name/value encoding)
    - M bytes: payload (JSON containing an Anthropic-style event)
    - 4 bytes: CRC checksum

    The payload contains a "bytes" field with base64-encoded JSON that
    represents the actual Anthropic event.

    Args:
        response: An httpx streaming response (from client.stream()).

    Yields:
        Text delta strings as they arrive.
    """
    buffer = b""

    async for chunk in response.aiter_bytes():
        buffer += chunk

        while len(buffer) >= 8:
            # Read prelude: total_length and headers_length
            total_length = struct.unpack("!I", buffer[:4])[0]
            # Prelude CRC is bytes 8-12

            if len(buffer) < total_length:
                break  # Need more data

            # The frame structure:
            # [4B total_len][4B headers_len][4B prelude_crc][headers][payload][4B message_crc]
            headers_length = struct.unpack("!I", buffer[4:8])[0]
            prelude_size = 12  # 4 + 4 + 4 (prelude CRC)
            headers_end = prelude_size + headers_length
            payload_end = total_length - 4  # Subtract message CRC

            payload_bytes = buffer[headers_end:payload_end]

            # Advance buffer past this frame
            buffer = buffer[total_length:]

            if not payload_bytes:
                continue

            try:
                payload = json.loads(payload_bytes)
            except json.JSONDecodeError:
                logger.debug(
                    "Skipping non-JSON Bedrock payload: %s",
                    payload_bytes[:100],
                )
                continue

            # Bedrock wraps the Anthropic event in a "bytes" field (base64-encoded)
            if "bytes" in payload:
                try:
                    inner_bytes = base64.b64decode(payload["bytes"])
                    event_data = json.loads(inner_bytes)
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug("Failed to decode Bedrock inner payload: %s", e)
                    continue
            else:
                # Some events may be directly in the payload
                event_data = payload

            text = extract_text_delta(event_data)
            if text is not None:
                yield text

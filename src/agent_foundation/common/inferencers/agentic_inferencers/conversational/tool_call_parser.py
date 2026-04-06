
# pyre-strict

"""Parse LLM responses for <tool_call> blocks interspersed with text.

The LLM is instructed to emit tool invocations as:
    <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

This module extracts those blocks and returns a ParsedResponse containing
both the text segments and the parsed tool calls.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# Match <ActionTools>```json ... ```</ActionTools> format used by the conversation template
_ACTION_TOOLS_RE = re.compile(
    r"<ActionTools>\s*```json\s*\n?(.*?)\n?\s*```\s*</ActionTools>",
    re.DOTALL,
)

# Legacy fallback: <Tools>```json ... ```</Tools> for backward compat during transition
_TOOLS_BLOCK_RE_LEGACY = re.compile(
    r"<Tools>\s*```json\s*\n?(.*?)\n?\s*```\s*</Tools>",
    re.DOTALL,
)


@dataclass
class ParsedToolCall:
    """A single parsed tool invocation from the LLM response."""

    name: str
    arguments: dict[str, Any]
    raw: str


@dataclass
class ParsedResponse:
    """Result of parsing an LLM response — text segments + tool calls."""

    text_segments: list[str] = field(default_factory=list)
    tool_calls: list[ParsedToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def text(self) -> str:
        """Join all text segments into a single string."""
        return "\n".join(seg.strip() for seg in self.text_segments if seg.strip())


def parse_llm_response(
    response: str,
    valid_tool_names: set[str] | None = None,
) -> ParsedResponse:
    """Parse an LLM response for <tool_call> blocks.

    Args:
        response: The raw LLM response text.
        valid_tool_names: Optional set of valid tool names for validation.
            If provided, tool calls with unrecognized names are logged
            as warnings and still included in the result.

    Returns:
        ParsedResponse with text segments and tool calls.
    """
    result = ParsedResponse()

    # Collect all matches from both patterns, sorted by position
    all_matches: list[tuple[int, int, str]] = []  # (start, end, raw_json)

    for match in _TOOL_CALL_RE.finditer(response):
        all_matches.append((match.start(), match.end(), match.group(1)))

    for match in _ACTION_TOOLS_RE.finditer(response):
        all_matches.append((match.start(), match.end(), match.group(1)))

    # Legacy fallback: also match old <Tools> tags
    for match in _TOOLS_BLOCK_RE_LEGACY.finditer(response):
        all_matches.append((match.start(), match.end(), match.group(1)))

    all_matches.sort(key=lambda x: x[0])

    last_end = 0
    for start, end, raw_json in all_matches:
        # Skip overlapping matches
        if start < last_end:
            continue

        # Capture text before this tool call
        text_before = response[last_end:start]
        if text_before.strip():
            result.text_segments.append(text_before)
        last_end = end

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in tool_call block: %s", raw_json[:200])
            result.text_segments.append(response[start:end])
            continue

        name = data.get("name", "")
        arguments = data.get("arguments", {})

        if not name:
            logger.warning("tool_call missing 'name' field: %s", raw_json[:200])
            result.text_segments.append(response[start:end])
            continue

        if valid_tool_names is not None and name not in valid_tool_names:
            logger.warning("Unrecognized tool name in tool_call: %s", name)

        result.tool_calls.append(
            ParsedToolCall(name=name, arguments=arguments, raw=raw_json)
        )

    # Capture any remaining text after the last tool call
    trailing = response[last_end:]
    if trailing.strip():
        result.text_segments.append(trailing)

    # If no tool calls were found, the entire response is text
    if not result.tool_calls and not result.text_segments:
        result.text_segments.append(response)

    return result

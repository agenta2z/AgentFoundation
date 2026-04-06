

"""Conversation response parser — extracts conversation tools from LLM output.

Supports two formats:
  1. Legacy: <ConversationTools>```json ... ```</ConversationTools>
  2. New: ```json ToolsToInvoke ... ``` (with "type": "conversation" entries)

Parser hierarchy:
  1. conversation_response_parser.py  — extracts conversation tools
  2. tool_call_parser.py              — extracts <tool_call>/<ActionTools> agentic tools
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
)

logger = logging.getLogger(__name__)

# Legacy format: <ConversationTools>```json ... ```</ConversationTools>
_CONV_TOOLS_RE = re.compile(
    r"<ConversationTools>\s*```json\s*\n?(.*?)\n?\s*```\s*</ConversationTools>",
    re.DOTALL,
)

# New format: ```json ToolsToInvoke ... ``` (one JSON object per line)
_TOOLS_TO_INVOKE_RE = re.compile(
    r"```json\s+ToolsToInvoke\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


@dataclass
class ConversationResponse:
    """Result of parsing an LLM response for conversation tools.

    Attributes:
        text: The text portion of the response (outside tool blocks).
        conversation_tool: First parsed conversation tool (for backward compat).
        conversation_tools: All parsed conversation tools (new: supports multiple).
        action_tools: Raw action tool dicts from ToolsToInvoke (for downstream).
        raw_response: The complete raw LLM response.
    """

    text: str = ""
    conversation_tool: ConversationTool | None = None
    conversation_tools: list[ConversationTool] = field(default_factory=list)
    action_tools: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""

    @property
    def has_conversation_tool(self) -> bool:
        return self.conversation_tool is not None


def _tool_invocation_to_conversation_tool(data: dict[str, Any]) -> ConversationTool:
    """Convert a ToolsToInvoke JSON object to a ConversationTool.

    The new format uses:
        {"type": "conversation", "name": "single_choice",
         "arguments": {"prompt": "...", "choices": [...]},
         "output": ["var1"]}

    Maps to ConversationTool:
        tool_type = data["name"]
        prompt = data["arguments"]["prompt"]
        choices = data["arguments"]["choices"]
        output_vars = data["output"]
    """
    args = data.get("arguments", {})
    choices_raw = args.get("choices", [])
    choices = [
        ChoiceItem.from_dict(c) if isinstance(c, dict)
        else ChoiceItem(label=str(c), value=str(c))
        for c in choices_raw
    ]

    # Extract optional metadata fields (e.g., view path for artifacts)
    metadata: dict[str, Any] = {}
    view_path = args.get("view")
    if view_path:
        metadata["view"] = view_path

    return ConversationTool(
        tool_type=data.get("name", "clarification"),
        prompt=args.get("prompt", ""),
        choices=choices,
        allow_custom=args.get("allow_custom", True),
        expected_input_type=args.get("expected_input_type", "free_text"),
        prefix=args.get("prefix", ""),
        output_vars=data.get("output", []),
        metadata=metadata,
    )


def parse_conversation_response(response: str) -> ConversationResponse:
    """Parse an LLM response for conversation tool invocations.

    Supports two formats:
      1. Legacy <ConversationTools> tags
      2. New ```json ToolsToInvoke blocks with "type": "conversation" entries

    Returns ConversationResponse with extracted tools and text.
    """
    result = ConversationResponse(raw_response=response)

    # --- Path 1: Legacy <ConversationTools> tags ---
    match = _CONV_TOOLS_RE.search(response)
    if match is not None:
        text_before = response[: match.start()].strip()
        text_after = response[match.end() :].strip()
        text_parts = [p for p in (text_before, text_after) if p]
        result.text = "\n\n".join(text_parts)

        raw_json = match.group(1)
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in <ConversationTools>: %s", raw_json[:200])
            result.text = response
            return result

        tool_type = data.get("tool_type", "")
        if tool_type:
            tool = ConversationTool.from_dict(data)
            result.conversation_tool = tool
            result.conversation_tools = [tool]
        else:
            result.text = response
        return result

    # --- Path 2: New ```json ToolsToInvoke block ---
    match = _TOOLS_TO_INVOKE_RE.search(response)
    if match is not None:
        # Extract text outside the code fence
        text_before = response[: match.start()].strip()
        text_after = response[match.end() :].strip()
        text_parts = [p for p in (text_before, text_after) if p]
        result.text = "\n\n".join(text_parts)

        # Parse each line as a JSON object
        raw_block = match.group(1)
        for line in raw_block.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in ToolsToInvoke line: %s", line[:200])
                continue

            tool_type = data.get("type", "")
            if tool_type == "conversation":
                tool = _tool_invocation_to_conversation_tool(data)
                result.conversation_tools.append(tool)
            elif tool_type == "action":
                result.action_tools.append(data)
            else:
                logger.warning("Unknown tool type in ToolsToInvoke: %s", tool_type)

        # Set backward-compat single tool field
        if result.conversation_tools:
            result.conversation_tool = result.conversation_tools[0]

        return result

    # --- No tool tags found ---
    result.text = response
    return result

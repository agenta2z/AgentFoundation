

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

    # Extract optional metadata fields.
    metadata: dict[str, Any] = {}
    for _key in (
        "view",
        "view_label",
        "yes_label",
        "no_label",
        "group_id",
        "on_group_resolve",
        "on_yes_action",
        "hide_no_button",
        # proposal_selection args: the SOP author passes proposals_path (often
        # via Jinja substitution); the rest tune widget behaviour.
        "proposals",
        "proposals_path",
        "preselected_ids",
        "allow_zero",
    ):
        _val = args.get(_key)
        if _val is not None and _val != "":
            metadata[_key] = _val

    # Also accept top-level `metadata` block for forward-compat
    top_level_meta = data.get("metadata")
    if isinstance(top_level_meta, dict):
        for _k, _v in top_level_meta.items():
            if _v is not None and _v != "":
                metadata[_k] = _v

    # Normalize plural/singular tool type variants.
    # LLM/SOP uses "multiple_choices" but ConversationToolType uses "multiple_choice".
    _name = data.get("name", "clarification")
    _TOOL_TYPE_ALIASES = {
        "multiple_choices": "multiple_choice",
        "single_choices": "single_choice",
    }
    _tool_type = _TOOL_TYPE_ALIASES.get(_name, _name)

    # Auto-upgrade: clarification with "view" is semantically a confirmation-with-review.
    # The LLM sometimes uses "clarification" when it means "confirmation with view".
    if _tool_type == "clarification" and metadata.get("view"):
        _tool_type = "confirmation"
        if "widget_type" not in metadata:
            metadata["widget_type"] = "confirmation"
        logger.info("[parser] auto-upgraded clarification → confirmation (has view=%s)", metadata["view"])

    if _tool_type != _name:
        logger.info("[parser] tool type normalized: %s → %s", _name, _tool_type)

    return ConversationTool(
        tool_type=_tool_type,
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
    logger.info(
        "[parse_conversation_response] input: %d chars, starts_with=%.120s",
        len(response), response[:120].replace('\n', '\\n'),
    )

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

    # --- Path 2b: ```json ... ``` without ToolsToInvoke label (multi-line OK) ---
    _PLAIN_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n\s*```", re.DOTALL)
    for fence_match in _PLAIN_JSON_FENCE_RE.finditer(response):
        block = fence_match.group(1).strip()
        if '"type"' not in block or '"conversation"' not in block:
            continue  # Not a tool block — skip normal code examples

        text_before = response[:fence_match.start()].strip()
        text_after = response[fence_match.end():].strip()
        text_parts = [p for p in (text_before, text_after) if p]
        result.text = "\n\n".join(text_parts)

        _parse_tool_block(block, result)

        if result.conversation_tools:
            result.conversation_tool = result.conversation_tools[0]
        return result

    # --- Path 3: Raw JSON lines (no code fences, single-line) ---
    _RAW_CONV_RE = re.compile(r'^\s*\{"type"\s*:\s*"(?:conversation|action)".*\}\s*$', re.MULTILINE)
    raw_matches = list(_RAW_CONV_RE.finditer(response))
    if raw_matches:
        text_parts = []
        last_end = 0
        for m in raw_matches:
            before = response[last_end:m.start()].strip()
            if before:
                text_parts.append(before)
            last_end = m.end()
        trailing = response[last_end:].strip()
        if trailing:
            text_parts.append(trailing)
        result.text = "\n\n".join(text_parts)

        for m in raw_matches:
            line = m.group(0).strip()
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "conversation":
                tool = _tool_invocation_to_conversation_tool(data)
                result.conversation_tools.append(tool)
            elif data.get("type") == "action":
                result.action_tools.append(data)

        if result.conversation_tools:
            result.conversation_tool = result.conversation_tools[0]
        return result

    # --- No tool tags found ---
    result.text = response
    return result


def _parse_tool_block(block: str, result: ConversationResponse) -> None:
    """Parse a block of JSON tool objects that may be multi-line (wrapped by terminal).

    Uses brace-depth tracking to find top-level JSON objects, handling nested
    braces and strings containing braces correctly.
    """
    normalized = " ".join(block.split())
    json_strings = _extract_json_objects(normalized)

    for js in json_strings:
        try:
            data = json.loads(js)
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool JSON: %s", js[:200])
            continue
        if data.get("type") == "conversation":
            tool = _tool_invocation_to_conversation_tool(data)
            result.conversation_tools.append(tool)
        elif data.get("type") == "action":
            result.action_tools.append(data)


def _extract_json_objects(text: str) -> list[str]:
    """Extract top-level JSON objects from a string by tracking brace depth.

    Handles nested braces, strings containing braces, and escaped characters.
    Returns a list of JSON object strings.
    """
    objects: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start:i + 1])
                start = -1

    return objects

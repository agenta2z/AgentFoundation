

"""Context data classes for the ConversationalInferencer agentic loop.

CompletedAction — a single tool execution record.
AgenticDynamicContext — accumulated actions with incremental compression support.
ContextBudget — per-section character limits for prompt assembly.
AgenticResult — structured return from run_agentic_loop().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
)


@dataclass
class CompletedAction:
    """A single completed tool execution record."""

    tool: str
    summary: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgenticDynamicContext:
    """Accumulated actions across turns. Supports incremental compression:
    compressed history + recent uncompressed actions are combined in to_text().

    No workflow_state — that lives on session.workflow_context (server layer)
    and is communicated via prior_context and tool result text.
    """

    completed_actions: list[CompletedAction] = field(
        default_factory=list
    )  # full history (for logging)
    _compressed_history: str = ""  # compressed older actions
    _uncompressed_actions: list[CompletedAction] = field(
        default_factory=list
    )  # actions since last compression

    def add_action(self, tool: str, summary: str) -> None:
        action = CompletedAction(tool, summary)
        self.completed_actions.append(action)
        self._uncompressed_actions.append(action)

    def to_text(self) -> str:
        """Returns compressed history + recent uncompressed actions."""
        parts = []
        if self._compressed_history:
            parts.append(self._compressed_history)
        if self._uncompressed_actions:
            parts.append(
                "\n".join(
                    f"- {a.tool}: {a.summary}" for a in self._uncompressed_actions
                )
            )
        return "\n\n".join(parts) if parts else ""

    def compress(self, compressed_text: str) -> None:
        """Store compression result and clear uncompressed buffer."""
        self._compressed_history = compressed_text
        self._uncompressed_actions.clear()

    def total_chars(self) -> int:
        return len(self.to_text())

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_actions": [
                {"tool": a.tool, "summary": a.summary, "timestamp": a.timestamp}
                for a in self.completed_actions
            ],
            "compressed_history": self._compressed_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgenticDynamicContext:
        ctx = cls()
        for a in data.get("completed_actions", []):
            action = CompletedAction(
                tool=a["tool"],
                summary=a["summary"],
                timestamp=a.get("timestamp", 0.0),
            )
            ctx.completed_actions.append(action)
        ctx._compressed_history = data.get("compressed_history", "")
        return ctx


@dataclass
class ContextBudget:
    """Per-section char limits. Initial implementation applies dynamic_context_max
    in _render_prompt. Other fields are extension points for future truncation."""

    prior_context_max: int = 2000  # chars for static context (future)
    dynamic_context_max: int = 4000  # chars for accumulated actions/state (APPLIED)
    conversation_history_max: int = 8000  # chars for message history (future)
    tools_max: int = 3000  # chars for tool definitions (future)
    current_turn_max: int = 2000  # chars for current user message (future)


@dataclass
class AgenticResult:
    """Structured return from run_agentic_loop()."""

    text: str  # final LLM text response
    completed_actions: list[CompletedAction]  # actions from this loop
    iterations_used: int
    has_conversation_tool: bool = False
    conversation_tool: Optional[ConversationTool] = None
    exhausted_max_iterations: bool = False  # True when loop hit max_iterations cap
    raw_response: str = ""  # full LLM response for logging
    last_rendered_prompt: str = ""  # for logging
    last_template_source: str = ""  # for View Prompt
    last_template_feed: dict[str, Any] = field(default_factory=dict)
    last_template_config: dict[str, Any] = field(default_factory=dict)  # for UI rendering

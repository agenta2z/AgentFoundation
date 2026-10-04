"""Vendor-neutral events a native backend emits while running one turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass(frozen=True)
class SessionStarted:
    """The vendor confirmed the session id (pinned or vendor-assigned).

    ``replaced`` is set when the vendor was asked to resume one session but
    silently started a different one (its memory of earlier turns is gone)."""

    session_id: str
    replaced: bool = False


@dataclass(frozen=True)
class MessageStart:
    """A main-thread assistant message began."""

    message_id: str


@dataclass(frozen=True)
class TextDelta:
    """Incremental assistant text of the main thread."""

    message_id: str
    text: str


@dataclass(frozen=True)
class MessageEnd:
    """A main-thread assistant message is complete."""

    message_id: str
    text: str
    tool_use_ids: tuple[str, ...] = ()
    af_tool_use_ids: tuple[str, ...] = ()
    message_uuid: Optional[str] = None


@dataclass(frozen=True)
class ToolActivity:
    """The agent invoked a tool (any tool, main thread or subagent)."""

    name: str
    tool_use_id: str
    subagent: bool = False


@dataclass(frozen=True)
class Compaction:
    """The vendor compacted the conversation history."""


@dataclass(frozen=True)
class TurnEnd:
    """The vendor finished the turn."""

    session_id: str
    stop_reason: Optional[str] = None
    is_error: bool = False
    num_turns: int = 0
    total_cost_usd: Optional[float] = None
    usage: dict[str, Any] = field(default_factory=dict)
    result_text: str = ""
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class VendorError:
    """A transport or vendor failure that ended the turn abnormally.

    ``session_missing`` is set only when the vendor reported that the session
    it was asked to resume does not exist; the turn was not accepted, so the
    host may start a fresh session and submit it there."""

    message: str
    submitted: bool = True
    session_missing: bool = False


VendorEvent = Union[
    SessionStarted,
    MessageStart,
    TextDelta,
    MessageEnd,
    ToolActivity,
    Compaction,
    TurnEnd,
    VendorError,
]

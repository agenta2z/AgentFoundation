"""Inbox item types for the CI event loop.

Three typed variants:
  - UserMessage: real user input
  - ToolCompletion: async tool finished (wake-up signal, no content)
  - SyntheticContinue: internal driver event (SOP phase advance, test inject)

All are frozen attrs classes — immutable value types.
"""

from __future__ import annotations

from typing import Literal, Union

import attrs

_SYNTHETIC_CONTINUE = "Continue per SOP guidance — advance to the next phase."


@attrs.frozen
class UserMessage:
    """A real user input event."""

    kind: Literal["user_message"] = "user_message"
    content: str = ""
    source: str = "user"


@attrs.frozen
class ToolCompletion:
    """Async tool finished. Result already in _messages. Pure wake-up signal."""

    kind: Literal["tool_completion"] = "tool_completion"
    tool_name: str = ""


@attrs.frozen
class SyntheticContinue:
    """Internal driver event (SOP phase advance, test inject)."""

    kind: Literal["synthetic_continue"] = "synthetic_continue"
    reason: str = ""


InboxItem = Union[UserMessage, ToolCompletion, SyntheticContinue]

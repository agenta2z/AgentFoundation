"""ScriptedInteractive — pre-canned user responses for testing conversational widgets.

Implements the InteractiveBase contract with deterministic responses, so
Layer 2 conversational tests can verify widget→user→tool flows without a
human in the loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Union

from attr import attrib, attrs

from agent_foundation.ui.interactive_base import InteractiveBase, InteractionFlags


@attrs(slots=False, kw_only=True)
class ScriptedInteractive(InteractiveBase):
    """Pre-canned responses for conversational widget testing.

    Pass `responses` as a list (dicts for structured, strings for free text,
    None to simulate disconnect/timeout). Each get_input call consumes the
    next response. Sent widgets are recorded in self.sent_widgets.
    """

    responses: List[Any] = attrib(factory=list)
    response_idx: int = attrib(default=0, init=False)
    sent_widgets: List = attrib(factory=list, init=False)
    sent_responses: List = attrib(factory=list, init=False)

    def _get_input(self):
        if self.response_idx >= len(self.responses):
            return None  # simulate timeout/disconnect
        resp = self.responses[self.response_idx]
        self.response_idx += 1
        return resp

    def reset_input(self, flag: InteractionFlags) -> None:
        # No-op for scripted interactive
        return None

    def _send_response(self, response: Any, flag: InteractionFlags = InteractionFlags.TurnCompleted) -> None:
        self.sent_responses.append((response, flag))

    # Async surface (called from run_agentic_loop / _handle_conversation_tools)
    async def aget_input(self):
        return self._get_input()

    async def asend_response(
        self,
        response: Any,
        flag: InteractionFlags = InteractionFlags.TurnCompleted,
        input_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.sent_widgets.append((response, flag, input_mode, kwargs))
        self._send_response(response, flag=flag)

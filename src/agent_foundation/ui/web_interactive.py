"""WebUIInteractive — native async transport for WebSocket-based UIs.

An alternative to QueueInteractive for in-process WebSocket handlers.
Uses native asyncio.Queue instead of file-based QueueServiceBase, avoiding
the sync-to-async thread-hop penalty of QueueInteractive.aget_input().

The WebSocket handler creates a WebUIInteractive per session:
  - Receive loop calls push_input(data) when messages arrive
  - Send loop calls pull_response() and forwards via WebSocket

This does NOT replace QueueInteractive — both extend RichInteractiveBase.
The session manager chooses which to create based on configuration.

Ported from rankevolve/src/agentic_foundation/common/ui/web_interactive.py
with import paths adapted for the agent_foundation package.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from attr import attrs, attrib

from agent_foundation.ui.interactive_base import (
    InteractionFlags,
    LOG_TYPE_SYSTEM_RESPONSE,
    LOG_TYPE_USER_INPUT,
)
from agent_foundation.ui.rich_interactive_base import (
    RichInteractiveBase,
)
from agent_foundation.ui.widget_protocol import (
    WidgetMessage,
    WidgetResponse,
)

logger = logging.getLogger(__name__)


@attrs
class WebUIInteractive(RichInteractiveBase):
    """Native async interactive transport for WebSocket-based UIs.

    Uses asyncio.Queue for zero thread-hop async I/O. Supports widgets
    for rich interactive input collection.
    """

    _input_queue: asyncio.Queue = attrib(factory=asyncio.Queue, init=False)
    _response_queue: asyncio.Queue = attrib(factory=asyncio.Queue, init=False)

    _pending_widget_id: Optional[str] = attrib(default=None, init=False)
    widget_response_timeout_seconds: float = attrib(default=300.0, kw_only=True)

    @property
    def supports_widgets(self) -> bool:
        return True

    @property
    def pending_widget_id(self) -> Optional[str]:
        return self._pending_widget_id

    # -- External API (called by WebSocket handler) --

    async def push_input(self, data: Any) -> None:
        await self._input_queue.put(data)

    async def pull_response(self) -> Any:
        return await self._response_queue.get()

    def has_responses(self) -> bool:
        return not self._response_queue.empty()

    # -- InteractiveBase implementation --

    def _get_input(self) -> Any:
        try:
            return self._input_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def aget_input(self) -> Any:
        """Native async input — no thread-hop needed."""
        self.log_debug(
            "Waiting for async input",
            log_type=LOG_TYPE_USER_INPUT,
        )
        data = await self._input_queue.get()
        if data is not None:
            if self.log_input_content:
                self.log_debug(
                    f"Received input: {data}",
                    log_type=LOG_TYPE_USER_INPUT,
                )
            else:
                self.log_debug("Received input", log_type=LOG_TYPE_USER_INPUT)

        if self._pending_input_mode is not None and data is not None:
            processed = self._postprocess_input(data, self._pending_input_mode)
            self._pending_input_mode = None
            return processed

        self._pending_input_mode = None
        return data

    def reset_input(self, flag: InteractionFlags) -> None:
        pass

    def _send_response(
        self,
        response: Any,
        flag: InteractionFlags = InteractionFlags.TurnCompleted,
    ) -> None:
        self.log_debug(
            f"Sending response, flag={flag.value}",
            log_type=LOG_TYPE_SYSTEM_RESPONSE,
        )

        if isinstance(response, dict):
            response_message = {**response, "flag": flag}
        else:
            response_message = {"response": response, "flag": flag}

        input_mode = self._current_input_mode
        if input_mode is not None:
            if hasattr(input_mode, "to_dict"):
                response_message["input_mode"] = input_mode.to_dict()
            elif isinstance(input_mode, dict):
                response_message["input_mode"] = input_mode

        self._response_queue.put_nowait(response_message)

    # -- Widget support --

    async def send_widget(
        self,
        widget_message: WidgetMessage,
        flag: InteractionFlags = InteractionFlags.PendingInput,
    ) -> WidgetResponse:
        """Send an interactive widget and wait for the user's response."""
        if self._pending_widget_id is not None:
            logger.warning(
                "Widget %s already pending, replacing with %s",
                self._pending_widget_id,
                widget_message.widget_id,
            )

        self._pending_widget_id = widget_message.widget_id

        response_msg: Dict[str, Any] = {
            "type": "pending_input",
            "widget": widget_message.to_dict(),
            "flag": flag,
        }
        self._response_queue.put_nowait(response_msg)

        try:
            raw = await asyncio.wait_for(
                self._wait_for_widget_response(widget_message.widget_id),
                timeout=self.widget_response_timeout_seconds,
            )
            return raw
        except asyncio.TimeoutError:
            logger.warning(
                "Widget response timeout for %s after %.0fs",
                widget_message.widget_id,
                self.widget_response_timeout_seconds,
            )
            self._pending_widget_id = None
            return WidgetResponse(
                widget_id=widget_message.widget_id,
                action="timeout",
            )

    async def _wait_for_widget_response(
        self, widget_id: str
    ) -> WidgetResponse:
        deferred: list[Any] = []
        try:
            while True:
                data = await self._input_queue.get()

                if isinstance(data, dict) and data.get("widget_id") == widget_id:
                    self._pending_widget_id = None
                    return WidgetResponse.from_dict(data)

                if isinstance(data, dict) and data.get("widget_id"):
                    logger.warning(
                        "Received response for widget %s but expected %s",
                        data.get("widget_id"),
                        widget_id,
                    )
                    continue

                deferred.append(data)
        finally:
            for item in deferred:
                await self._input_queue.put(item)

    async def send_display_widget(
        self, widget_message: WidgetMessage
    ) -> None:
        """Send a display-only widget (no input expected)."""
        response_msg: Dict[str, Any] = {
            "type": "widget_update",
            "widget": widget_message.to_dict(),
            "flag": InteractionFlags.MessageOnly,
        }
        self._response_queue.put_nowait(response_msg)

"""InteractionSerializer — serializes concurrent interaction requests through a single interactive.

Use when N concurrent async callers (e.g., N SOPInferencer instances, N BTA
workers, background tasks) share one user-facing interactive and you need
exactly one widget visible to the user at a time.

Caller identity flows implicitly via _CURRENT_INTERACTION_CALLER ContextVar
(defined in interactive_base.py).

Activation: interactive.enable_serialization(InteractionSerializer())
Deactivation: interactive.disable_serialization() + serializer.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """A queued interaction request from a concurrent caller."""

    caller_id: str
    interactive: Any  # InteractiveBase
    response: Any
    flag: Any
    kwargs: dict
    response_future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class InteractionSerializer:
    """Serializes concurrent interaction requests through a single interactive.

    Single-head semantics: only one widget visible at a time. Other requests
    wait in a FIFO queue. When the active widget gets a user response, the
    next queued request is presented automatically.
    """

    def __init__(self) -> None:
        self._pending: deque[PendingRequest] = deque()
        self._active: Optional[PendingRequest] = None
        self._caller_to_request: dict[str, PendingRequest] = {}
        self._lock = asyncio.Lock()

    async def enqueue_send(
        self,
        interactive: Any,
        response: Any,
        flag: Any,
        kwargs: dict,
        caller_id: str = "",
    ) -> None:
        """Enqueue a widget-send request. Returns immediately."""
        if not caller_id:
            from agent_foundation.ui.interactive_base import _CURRENT_INTERACTION_CALLER
            caller_id = _CURRENT_INTERACTION_CALLER.get()
        if not caller_id:
            raise RuntimeError(
                "InteractionSerializer requires caller identity. "
                "Set _CURRENT_INTERACTION_CALLER ContextVar before invoking."
            )
        if caller_id in self._caller_to_request:
            raise RuntimeError(
                f"Caller {caller_id} already has a pending request."
            )

        loop = asyncio.get_running_loop()
        req = PendingRequest(
            caller_id=caller_id,
            interactive=interactive,
            response=response,
            flag=flag,
            kwargs=kwargs,
            response_future=loop.create_future(),
        )
        self._caller_to_request[caller_id] = req
        async with self._lock:
            self._pending.append(req)
        await self._try_present_next()

    async def await_response_for_caller(self, caller_id: str = "") -> Any:
        """Block until the user responds to THIS caller's request."""
        if not caller_id:
            from agent_foundation.ui.interactive_base import _CURRENT_INTERACTION_CALLER
            caller_id = _CURRENT_INTERACTION_CALLER.get()
        if caller_id not in self._caller_to_request:
            raise RuntimeError(
                f"Caller {caller_id} has no outstanding request."
            )
        req = self._caller_to_request[caller_id]
        try:
            return await req.response_future
        finally:
            self._caller_to_request.pop(caller_id, None)

    async def deliver_response(self, response: Any) -> None:
        """Called when the user responds to the active widget."""
        async with self._lock:
            if self._active is None:
                logger.warning("deliver_response called with no active request")
                return
            self._active.response_future.set_result(response)
            self._active = None
        await self._try_present_next()

    async def _try_present_next(self) -> None:
        """Present the next queued request if none is active."""
        async with self._lock:
            if self._active is not None or not self._pending:
                return
            self._active = self._pending.popleft()

        # Present to user via the interactive's direct-send path
        interactive = self._active.interactive
        if hasattr(interactive, "_direct_send"):
            await interactive._direct_send(
                self._active.response, self._active.flag, **self._active.kwargs
            )
        else:
            await asyncio.to_thread(
                interactive.send_response, self._active.response, self._active.flag,
            )

    def has_pending_for(self, caller_id: str) -> bool:
        """Check if a caller has a pending request."""
        return caller_id in self._caller_to_request

    def has_any_active(self) -> bool:
        """Check if any request is currently active."""
        return self._active is not None

    def shutdown(self) -> None:
        """Cancel all outstanding futures on session teardown."""
        for req in self._caller_to_request.values():
            if not req.response_future.done():
                req.response_future.cancel()
        self._pending.clear()
        self._caller_to_request.clear()
        self._active = None

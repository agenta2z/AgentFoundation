# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""InboxDriver — the inbox-mode event loop shared by conversational orchestrators.

Owns the inbox queue, the cooperative-shutdown flag, the turn counter and the
long-lived ``run()`` loop that turns each dequeued item into one host turn.

The driver never references its host: the host injects ``run_turn`` (an async
callable that runs one turn) and delegates its ``SupportsInbox`` surface
(``enable_inbox``, ``inbox_put``, ``inbox_put_user``, ``request_shutdown``,
``shutdown_requested``, ``run``) here.

``run_turn`` is called as::

    await run_turn(
        content,
        origin=...,             # "user" | "tool_completion" | "host_event"
        interactive=...,
        turn_number=...,
        on_new_turn=...,
        on_prompt_rendered=...,
        on_turn_complete=...,
        run_context=...,        # per-turn ``run_context.child("turn_N")`` or None
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, Protocol

from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox import (
    _SYNTHETIC_CONTINUE,
    SyntheticContinue,
    ToolCompletion,
    UserMessage,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocol_text import (
    CONTINUE_AFTER_TOOLS,
)

logger: logging.Logger = logging.getLogger(__name__)

# Turn origins; the values match the native orchestrator's ``TurnOrigin``.
ORIGIN_USER = "user"
ORIGIN_TOOL_COMPLETION = "tool_completion"
ORIGIN_HOST_EVENT = "host_event"


class RunTurn(Protocol):
    def __call__(
        self,
        content: str,
        *,
        origin: str,
        interactive: Any,
        turn_number: int,
        on_new_turn: Any,
        on_prompt_rendered: Any,
        on_turn_complete: Any,
        run_context: Any,
    ) -> Awaitable[Any]: ...


def inbox_item_content(item: object) -> Optional[str]:
    """Turn content for an inbox item; ``None`` means the item is skipped."""
    if isinstance(item, UserMessage):
        return item.content
    if isinstance(item, ToolCompletion):
        return CONTINUE_AFTER_TOOLS
    if isinstance(item, SyntheticContinue):
        return _SYNTHETIC_CONTINUE
    return None


def inbox_item_origin(item: object) -> str:
    """Why the turn for ``item`` runs."""
    if isinstance(item, ToolCompletion):
        return ORIGIN_TOOL_COMPLETION
    if isinstance(item, SyntheticContinue):
        return ORIGIN_HOST_EVENT
    return ORIGIN_USER


class InboxDriver:
    """Inbox queue + shutdown flag + ``run()`` loop for one host.

    ``content_for_item`` maps an inbox item to turn content (``None`` skips the
    item without consuming a turn number). ``log`` receives the per-item failure
    record; hosts pass their own module logger to keep its logger name.
    """

    def __init__(
        self,
        run_turn: RunTurn,
        *,
        content_for_item: Callable[[object], Optional[str]] = inbox_item_content,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.run_turn = run_turn
        self.content_for_item = content_for_item
        self.log: logging.Logger = log if log is not None else logger
        self.queue: Optional[asyncio.Queue[Any]] = None
        self.shutdown_requested = False
        self.running = False
        self.turn_counter = 0
        self.interactive: Any = None
        self.on_new_turn: Any = None
        self.on_prompt_rendered: Any = None
        self.on_turn_complete: Any = None

    def enable(
        self,
        *,
        interactive: Any = None,
        maxsize: int = 0,
        on_new_turn: Any = None,
        on_prompt_rendered: Any = None,
        on_turn_complete: Any = None,
    ) -> None:
        """Create the queue and record the per-turn defaults. Must precede run()."""
        if self.queue is not None:
            raise RuntimeError("Inbox already enabled")
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.interactive = interactive
        self.on_new_turn = on_new_turn
        self.on_prompt_rendered = on_prompt_rendered
        self.on_turn_complete = on_turn_complete

    def put(self, item: object) -> None:
        """Non-blocking enqueue. Raises RuntimeError if the inbox is not enabled."""
        if self.queue is None:
            raise RuntimeError("Inbox not enabled; call enable_inbox() first")
        self.queue.put_nowait(item)

    def put_user(self, content: str, source: str = "user") -> None:
        """Convenience: enqueue a UserMessage."""
        self.put(UserMessage(content=content, source=source))

    def request_shutdown(self) -> None:
        """Cooperative termination: the current turn finishes, then run() returns."""
        self.shutdown_requested = True

    def next_turn_number(self) -> int:
        self.turn_counter += 1
        return self.turn_counter

    async def run(self, *, run_context: Any = None) -> Any:
        """Drain the inbox, one ``run_turn`` per item, until shutdown is requested.

        The shutdown flag is checked between items, so a request made while the
        loop waits on an empty queue takes effect after the next item. A failing
        item is logged and the loop continues. Returns the last turn's result.
        """
        if self.queue is None:
            raise RuntimeError("Inbox not enabled; call enable_inbox() first")
        if self.running:
            raise RuntimeError("run() is already executing")
        self.running = True
        try:
            last_result: Any = None
            while not self.shutdown_requested:
                item = await self.queue.get()
                try:
                    ran, result = await self._run_item(item, run_context)
                    if ran:
                        last_result = result
                except Exception as e:
                    self.log.exception("Inbox item %r failed: %s", item, e)
                finally:
                    self.queue.task_done()
            return last_result
        finally:
            self.running = False

    async def _run_item(self, item: object, run_context: Any) -> tuple[bool, Any]:
        content = self.content_for_item(item)
        if content is None:
            return False, None
        turn = self.next_turn_number()
        turn_ctx = (
            run_context.child(f"turn_{turn}") if run_context is not None else None
        )
        result = await self.run_turn(
            content,
            origin=inbox_item_origin(item),
            interactive=self.interactive,
            turn_number=turn,
            on_new_turn=self.on_new_turn,
            on_prompt_rendered=self.on_prompt_rendered,
            on_turn_complete=self.on_turn_complete,
            run_context=turn_ctx,
        )
        return True, result

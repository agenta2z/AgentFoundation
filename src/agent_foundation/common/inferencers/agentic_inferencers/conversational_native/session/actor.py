"""SessionActor — one asyncio task that owns a backend session.

Some vendor clients (the Claude Agent SDK's ``ClaudeSDKClient``) must be
connected and disconnected from the same task. The actor's task is created
with a fresh ``contextvars.Context`` so nothing from the first caller's run
context leaks into later turns; turn-specific context reaches tool handlers
explicitly through ``TurnScope``.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    TurnEnd,
    VendorError,
    VendorEvent,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    NativeSessionBackend,
    SessionOpenRequest,
    TurnRequest,
)

logger: logging.Logger = logging.getLogger(__name__)

_END = object()


@dataclass
class _Command:
    kind: str
    payload: Any = None
    events: Optional[asyncio.Queue] = None
    done: Optional[asyncio.Future] = None


@dataclass
class TurnOutcome:
    """How the last turn ended from the actor's point of view."""

    interrupted: bool = False
    acknowledged: bool = True
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


class SessionActor:
    def __init__(
        self,
        backend: NativeSessionBackend,
        open_request: SessionOpenRequest,
        *,
        binding: Any = None,
        drain_timeout_s: float = 30.0,
    ) -> None:
        self.backend = backend
        # The SessionBinding the backend's tools and hooks forward through;
        # rebound to whichever inferencer drives the next turn.
        self.binding = binding
        # Fingerprint of the AF tool set the session's tool server was built
        # with; a different current set means the session must be reopened.
        self.manifest_fingerprint: Optional[str] = None
        # Core hash of the session instructions the session was opened with.
        self.l1_core_hash: Optional[str] = None
        self._open_request = open_request
        self._drain_timeout_s = drain_timeout_s
        self._task: Optional[asyncio.Task] = None
        self._commands: Optional[asyncio.Queue] = None
        self._interrupt_requested: Optional[asyncio.Event] = None
        self._in_turn = False
        # Set when a turn could not be drained: the session takes no further
        # commands and its backend is closed.
        self._retired = False
        self._backend_closed = False
        self.last_outcome = TurnOutcome()

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done() and not self._retired

    @property
    def in_turn(self) -> bool:
        return self._in_turn

    @property
    def resumed(self) -> bool:
        """True when this actor continued an existing vendor session (as
        opposed to starting or forking one)."""
        return bool(self._open_request.resume)

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future = loop.create_future()
        self._commands = asyncio.Queue()
        self._interrupt_requested = asyncio.Event()
        self._task = loop.create_task(self._main(ready), context=contextvars.Context())
        await ready

    async def _main(self, ready: asyncio.Future) -> None:
        try:
            await self.backend.open(self._open_request)
        except asyncio.CancelledError:
            ready.cancel()
            await self._close_backend()
            raise
        except Exception as exc:  # surfaced to start()
            ready.set_exception(exc)
            await self._close_backend()
            return
        ready.set_result(None)
        try:
            await self._command_loop()
        finally:
            await self._close_backend()

    async def _command_loop(self) -> None:
        assert self._commands is not None
        while True:
            cmd: _Command = await self._commands.get()
            if cmd.kind == "close":
                if cmd.done is not None and not cmd.done.done():
                    cmd.done.set_result(None)
                return
            try:
                if cmd.kind == "turn":
                    await self._run_turn(cmd)
                elif cmd.kind == "set_model":
                    await self.backend.set_model(cmd.payload)
                    cmd.done.set_result(None)
            except Exception as exc:
                logger.exception("SessionActor command %s failed", cmd.kind)
                if cmd.done is not None and not cmd.done.done():
                    cmd.done.set_exception(exc)
            if self._retired:
                return

    async def _run_turn(self, cmd: _Command) -> None:
        assert self._interrupt_requested is not None and cmd.events is not None
        self._interrupt_requested.clear()
        self._in_turn = True
        outcome = TurnOutcome()
        events = cmd.events

        async def pump() -> None:
            async for event in self.backend.run_turn(cmd.payload):
                await events.put(event)

        pump_task = asyncio.ensure_future(pump())
        interrupt_task = asyncio.ensure_future(self._interrupt_requested.wait())
        try:
            done, _ = await asyncio.wait(
                {pump_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if interrupt_task in done and not pump_task.done():
                outcome.interrupted = True
                try:
                    await asyncio.wait_for(
                        self.backend.interrupt(), timeout=self._drain_timeout_s
                    )
                except Exception as exc:  # timeout or transport error
                    logger.warning("Vendor interrupt failed: %s", exc)
                    outcome.acknowledged = False
                try:
                    await asyncio.wait_for(
                        asyncio.shield(pump_task), timeout=self._drain_timeout_s
                    )
                except asyncio.TimeoutError:
                    outcome.acknowledged = False
                    await self._abandon(pump_task)
            exc = (
                pump_task.exception()
                if pump_task.done() and not pump_task.cancelled()
                else None
            )
            if exc is not None:
                outcome.error = str(exc)
                await events.put(VendorError(message=str(exc)))
        finally:
            interrupt_task.cancel()
            if not pump_task.done():
                pump_task.cancel()
            self._in_turn = False
            self.last_outcome = outcome
            await events.put(_END)
            if cmd.done is not None and not cmd.done.done():
                cmd.done.set_result(outcome)

    async def _abandon(self, pump_task: asyncio.Task) -> None:
        """Plan §6.4 "on timeout disconnect/kill": a turn that did not drain
        may still be producing events, and a client that outlives the turn
        (the Claude SDK's) would hand them to the next turn as its own. The
        turn's stream is cancelled and the session closed before the turn
        ends; the next turn opens a new one, resuming the vendor session."""
        logger.warning(
            "Vendor turn did not drain within %ss; closing the session",
            self._drain_timeout_s,
        )
        self._retired = True
        pump_task.cancel()
        await asyncio.wait({pump_task}, timeout=self._drain_timeout_s)
        await self._close_backend()

    async def run_turn(self, request: TurnRequest) -> AsyncIterator[VendorEvent]:
        """Submit one turn and yield its events. Cancelling the consumer
        interrupts the vendor turn and drains it before returning; a turn
        that does not drain closes the session (``_abandon``)."""
        if not self.alive or self._commands is None:
            raise RuntimeError("SessionActor is not running")
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        done: asyncio.Future = loop.create_future()
        await self._commands.put(_Command("turn", request, events, done))
        finished = False
        try:
            while True:
                event = await events.get()
                if event is _END:
                    finished = True
                    return
                yield event
        finally:
            if not finished:
                await asyncio.shield(self._interrupt_and_wait(done))

    async def _interrupt_and_wait(self, done: asyncio.Future) -> None:
        if self._interrupt_requested is not None:
            self._interrupt_requested.set()
        try:
            await asyncio.wait_for(
                asyncio.shield(done), timeout=self._drain_timeout_s * 2
            )
        except asyncio.TimeoutError:
            logger.warning("Vendor turn did not drain after interrupt; closing actor")
            await self.close()

    async def interrupt(self) -> None:
        if self._interrupt_requested is not None and self._in_turn:
            self._interrupt_requested.set()

    async def set_model(self, model: str) -> None:
        if not self.alive or self._commands is None:
            return
        done: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._commands.put(_Command("set_model", model, done=done))
        await done

    async def close(self) -> None:
        if self._task is None or self._task.done() or self._commands is None:
            return
        if self._in_turn and self._interrupt_requested is not None:
            self._interrupt_requested.set()
        done: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._commands.put(_Command("close", done=done))
        try:
            await asyncio.wait_for(
                asyncio.shield(self._task), timeout=self._drain_timeout_s * 2
            )
        except asyncio.TimeoutError:
            self._task.cancel()

    async def _close_backend(self) -> None:
        if self._backend_closed:
            return
        self._backend_closed = True
        try:
            await self.backend.close()
        except Exception as exc:
            logger.warning("Backend close failed: %s", exc)


def turn_end_from(events: list[VendorEvent]) -> Optional[TurnEnd]:
    for event in reversed(events):
        if isinstance(event, TurnEnd):
            return event
    return None

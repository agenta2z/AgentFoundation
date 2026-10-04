"""NativeRuntimeManager — owns live vendor sessions across host turns.

Injected by the host (OpenStartup's conversation service owns one); a native
inferencer created without one gets a private manager. Hosts commonly evict
and rebuild a conversation's inferencer (backend switch, resume); leases let
the rebuilt inferencer adopt the still-live session instead of reconnecting.
Idle sessions are closed and transparently resumed by id on the next turn.
The local MCP servers for per-turn CLIs are created lazily, one per event loop
(their tasks, sockets and locks belong to the loop that started them).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.actor import (
    SessionActor,
)

logger: logging.Logger = logging.getLogger(__name__)

SessionKey = tuple[str, str, int]  # (conversation_key, backend kind, generation)


@dataclass
class _Entry:
    actor: SessionActor
    leases: int = 0
    idle_timer: Optional[asyncio.TimerHandle] = None
    last_used: float = field(default_factory=time.monotonic)


class NativeRuntimeManager:
    def __init__(
        self, *, idle_close_seconds: float = 1800.0, max_live_sessions: int = 32
    ) -> None:
        self.idle_close_seconds = idle_close_seconds
        self.max_live_sessions = max_live_sessions
        self._entries: dict[SessionKey, _Entry] = {}
        # Leases taken before the key's actor exists; its entry starts with them.
        self._early_leases: dict[SessionKey, int] = {}
        self._lock = asyncio.Lock()
        self._closing: set[asyncio.Task] = set()
        # Per event loop: LocalMcpHttpServer (claude -p, codex exec) and
        # LocalMcpSocketServer (dm), created on first use.
        self._http_servers: dict[asyncio.AbstractEventLoop, Any] = {}
        self._socket_servers: dict[asyncio.AbstractEventLoop, Any] = {}

    @property
    def http_server(self) -> Any:
        """The running loop's HTTP MCP server, if one was started."""
        return _for_running_loop(self._http_servers)

    @property
    def socket_server(self) -> Any:
        """The running loop's unix-socket MCP server, if one was started."""
        return _for_running_loop(self._socket_servers)

    async def ensure_http_server(self) -> Any:
        """The running loop's localhost HTTP MCP server, created on first use
        (per-turn CLI backends register their tools on it)."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.mcp_http import (
            LocalMcpHttpServer,
        )

        return _ensure_for_running_loop(self._http_servers, LocalMcpHttpServer)

    async def ensure_socket_server(self) -> Any:
        """The running loop's unix-socket MCP server, created on first use (the
        Devmate ``dm`` backend registers its tools on it; dm has no HTTP MCP
        transport)."""
        from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.mcp_socket import (
            LocalMcpSocketServer,
        )

        return _ensure_for_running_loop(self._socket_servers, LocalMcpSocketServer)

    async def acquire(self, key: SessionKey) -> None:
        """Register interest in ``key``'s session (one lease per inferencer),
        counted also when its actor does not exist yet."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._early_leases[key] = self._early_leases.get(key, 0) + 1
                return
            entry.leases += 1
            self._cancel_idle(entry)

    async def get_actor(
        self, key: SessionKey, factory: Callable[[], Awaitable[SessionActor]]
    ) -> SessionActor:
        """Return the live actor for ``key``, starting one via ``factory``."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.actor.alive:
                self._cancel_idle(entry)
                entry.last_used = time.monotonic()
                return entry.actor
            await self._evict_stale_generations(key)
            await self._enforce_capacity()
            actor = await factory()
            leases = entry.leases if entry is not None else 0
            leases += self._early_leases.pop(key, 0)
            self._entries[key] = _Entry(actor=actor, leases=max(leases, 1))
            return actor

    def live_actor(self, key: SessionKey) -> Optional[SessionActor]:
        entry = self._entries.get(key)
        return entry.actor if entry is not None and entry.actor.alive else None

    def touch(self, key: SessionKey) -> None:
        """Mark the session used and restart its idle timer (after every turn:
        an idle session is closed even while leased, and the next turn resumes
        it by id)."""
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.last_used = time.monotonic()
        self._cancel_idle(entry)
        if self.idle_close_seconds and self.idle_close_seconds > 0:
            loop = asyncio.get_running_loop()
            entry.idle_timer = loop.call_later(
                self.idle_close_seconds, self._schedule_close, key
            )

    async def release(self, key: SessionKey) -> None:
        """Drop one lease; the session stays alive until idle or evicted."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                early = self._early_leases.pop(key, 0) - 1
                if early > 0:
                    self._early_leases[key] = early
                return
            entry.leases = max(entry.leases - 1, 0)
        if entry.leases == 0:
            self.touch(key)

    async def evict(self, key: SessionKey) -> None:
        async with self._lock:
            entry = self._entries.pop(key, None)
        if entry is not None:
            self._cancel_idle(entry)
            await entry.actor.close()

    async def evict_conversation(self, conversation_key: str) -> None:
        for key in [k for k in self._entries if k[0] == conversation_key]:
            await self.evict(key)

    async def aclose_all(self) -> None:
        for key in list(self._entries):
            await self.evict(key)
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)
        for servers in (self._http_servers, self._socket_servers):
            await _stop_all(servers)

    def _schedule_close(self, key: SessionKey) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.actor.in_turn:
            return
        task = asyncio.get_running_loop().create_task(self._idle_close(key))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _idle_close(self, key: SessionKey) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.actor.in_turn:
            return
        logger.info("Closing idle native session %s (resumable)", key[:2])
        await entry.actor.close()

    async def _evict_stale_generations(self, key: SessionKey) -> None:
        stale = [k for k in self._entries if k[:2] == key[:2] and k[2] != key[2]]
        for k in stale:
            entry = self._entries.pop(k)
            self._cancel_idle(entry)
            await entry.actor.close()

    async def _enforce_capacity(self) -> None:
        """Close least-recently-used live sessions that are not mid-turn so a
        new one fits. Leased sessions qualify: the entry (and its lease) stays,
        and the next turn transparently resumes the vendor session by id."""
        live = [(k, e) for k, e in self._entries.items() if e.actor.alive]
        excess = len(live) - self.max_live_sessions + 1
        if excess <= 0:
            return
        idle = sorted(
            ((k, e) for k, e in live if not e.actor.in_turn),
            key=lambda item: item[1].last_used,
        )
        for _key, entry in idle[:excess]:
            self._cancel_idle(entry)
            await entry.actor.close()
        if len(idle) < excess:
            logger.warning(
                "Native session cap %d exceeded: %d sessions are mid-turn",
                self.max_live_sessions,
                len(live) - len(idle),
            )

    @staticmethod
    def _cancel_idle(entry: _Entry) -> None:
        if entry.idle_timer is not None:
            entry.idle_timer.cancel()
            entry.idle_timer = None


def _for_running_loop(servers: dict[asyncio.AbstractEventLoop, Any]) -> Any:
    try:
        return servers.get(asyncio.get_running_loop())
    except RuntimeError:  # no running loop
        return None


def _ensure_for_running_loop(
    servers: dict[asyncio.AbstractEventLoop, Any], factory: Callable[[], Any]
) -> Any:
    for loop in [loop for loop in servers if loop.is_closed()]:
        servers.pop(loop).abandon()
    loop = asyncio.get_running_loop()
    if loop not in servers:
        servers[loop] = factory()
    return servers[loop]


async def _stop_all(servers: dict[asyncio.AbstractEventLoop, Any]) -> None:
    """Stop every loop's server from the current loop: directly on this loop,
    through ``run_coroutine_threadsafe`` on another running loop, and by
    releasing the sockets of a loop that is gone."""
    current = asyncio.get_running_loop()
    entries = list(servers.items())
    servers.clear()
    for loop, server in entries:
        if loop is current:
            await server.stop()
        elif loop.is_running():
            future = asyncio.run_coroutine_threadsafe(server.stop(), loop)
            await asyncio.wrap_future(future)
        else:
            server.abandon()

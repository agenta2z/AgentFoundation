"""Unix-domain-socket MCP server for the Devmate ``dm`` backend.

``dm``'s ``--mcp-servers`` only supports ``type:"socket"`` (a unix socket
speaking newline-delimited JSON-RPC) and ``type:"stdio"`` — there is no HTTP
transport, so the localhost HTTP server used by the other per-turn CLIs does
not apply. dm-core connects as an MCP client to a socket path we pre-create and
listen on; the framing is "a single UTF-8 JSON object terminated by '\\n'"
(verified against the dm-core bundle and end-to-end by
``scripts/native_spikes/s14_dm_socket_mcp.py``).

One server object (owned by ``NativeRuntimeManager``) serves every live
conversation: each registers its tool set and gets its own 0600 socket in a
private 0700 directory (filesystem permissions are the credential — the NDJSON
MCP framing carries no auth header). The tool-dispatch mapping is shared with
the HTTP server so there is a single source of truth. Unlike the Claude CLI's
HTTP tools, these declare no result size: dm-core reads no tool ``_meta`` from
an MCP server (``anthropic/maxResultSizeChars`` appears in the dm 2026.10.03-0249
bundle only on the tools dm itself serves to Claude), so the bridge's own sizing
(``native_tool_result_max_chars``) bounds every result.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import shutil
import stat
import tempfile
from dataclasses import dataclass
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class _SocketSession:
    tools: dict[str, Any]  # tool name -> BridgeToolSpec
    server: Any  # asyncio.Server
    directory: str


class LocalMcpSocketServer:
    """Per-conversation unix-socket MCP endpoints (newline-delimited JSON-RPC)."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SocketSession] = {}
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return bool(self._sessions)

    async def register(self, tools: list) -> tuple[str, str]:
        """Register a conversation's tools; returns (socket_path, token)."""
        async with self._lock:
            token = secrets.token_urlsafe(12)
            by_name = {spec.name: spec for spec in tools}
            directory = tempfile.mkdtemp(prefix="af_dm_mcp_")
            os.chmod(directory, stat.S_IRWXU)  # 0700
            # Keep the path short: unix sun_path is capped near 108 bytes.
            socket_path = os.path.join(directory, "af.sock")
            server = await asyncio.start_unix_server(
                self._make_handler(token, by_name), path=socket_path
            )
            os.chmod(socket_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
            self._sessions[token] = _SocketSession(
                tools=by_name, server=server, directory=directory
            )
        return socket_path, token

    async def unregister(self, token: str) -> None:
        async with self._lock:
            session = self._sessions.pop(token, None)
        if session is None:
            return
        try:
            session.server.close()
            await session.server.wait_closed()
        except Exception as exc:
            logger.warning("MCP socket teardown failed: %s", exc)
        shutil.rmtree(session.directory, ignore_errors=True)

    async def stop(self) -> None:
        for token in list(self._sessions):
            await self.unregister(token)

    def abandon(self) -> None:
        """Release the endpoints of a server whose event loop is gone (its
        tasks can no longer run, so ``stop()`` is impossible)."""
        for session in self._sessions.values():
            try:
                session.server.close()
            except RuntimeError as exc:  # the loop's waiters cannot be woken
                logger.debug("MCP socket release on a closed loop: %s", exc)
            shutil.rmtree(session.directory, ignore_errors=True)
        self._sessions.clear()

    def _make_handler(self, token: str, by_name: dict[str, Any]) -> Any:
        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await self._serve_connection(token, by_name, reader, writer)

        return handle

    @staticmethod
    async def _serve_connection(
        token: str,
        by_name: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        import anyio
        import mcp.types as mt
        from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.mcp_http import (
            LocalMcpHttpServer,
        )
        from mcp.server.lowlevel import Server
        from mcp.shared.message import SessionMessage

        # No size declaration (module docstring).
        server = LocalMcpHttpServer._build_server(
            Server, token, by_name, result_max_chars=0
        )
        read_w, read_r = anyio.create_memory_object_stream(0)
        write_w, write_r = anyio.create_memory_object_stream(0)
        logger.debug("mcp socket connection accepted")
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    _pump_socket_to_stream, reader, read_w, mt, SessionMessage
                )
                tg.start_soon(_pump_stream_to_socket, write_r, writer)
                await server.run(
                    read_r, write_w, server.create_initialization_options()
                )
                tg.cancel_scope.cancel()
        except Exception as exc:
            logger.debug("MCP socket connection ended: %s", exc)
        finally:
            with contextlib.suppress(Exception):
                writer.close()


async def _pump_socket_to_stream(
    reader: asyncio.StreamReader, read_w: Any, mt: Any, session_message_cls: Any
) -> None:
    """Forward newline-delimited JSON-RPC from the socket into the SDK read stream."""
    async with read_w:
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("utf-8").strip()
            if not text:
                continue
            try:
                message = mt.JSONRPCMessage.model_validate_json(text)
            except Exception as exc:  # malformed -> surfaced to the SDK
                await read_w.send(exc)
                continue
            await read_w.send(session_message_cls(message))


async def _pump_stream_to_socket(write_r: Any, writer: asyncio.StreamWriter) -> None:
    """Forward SDK responses out to the socket as newline-delimited JSON-RPC."""
    try:
        async with write_r:
            async for session_message in write_r:
                data = session_message.message.model_dump_json(
                    by_alias=True, exclude_none=True
                )
                writer.write((data + "\n").encode("utf-8"))
                await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass

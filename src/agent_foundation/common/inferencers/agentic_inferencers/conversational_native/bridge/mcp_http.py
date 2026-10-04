"""Localhost streamable-HTTP MCP server for per-turn CLI backends.

The Claude Agent SDK hosts AF tools in-process; per-turn CLI backends
(``claude -p``, ``codex exec``) connect over MCP instead. One server per event
loop (owned by ``NativeRuntimeManager``) serves every live conversation on that
loop: each registers its tool set under a per-conversation bearer token.
Requests are routed by that token, read from the ``Authorization`` header, so
no credential appears in a URL (CLIs echo URLs on their command lines). The
same token authenticates ``/hook``, the endpoint the Claude CLI's command hooks
relay to (``bridge/af_hook.py``). A conversation that registered a
``turn_active`` predicate is refused outside its vendor turns: its CLI process
only runs during one. Arguments are not validated here; the bridge validates
them against the session's own schema. Verified reachable under the Meta
launcher sandbox by ``scripts/native_spikes/s11_http_mcp.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import hmac
import json
import logging
import secrets
import socket
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger: logging.Logger = logging.getLogger(__name__)

HookHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
TurnActive = Callable[[], bool]
_MAX_HOOK_BODY = 4 * 1024 * 1024
_START_TIMEOUT_S = 30.0
_MAX_RESULT_META = "anthropic/maxResultSizeChars"


@dataclass
class _Session:
    token: str
    tools: dict[str, Any]  # tool name -> BridgeToolSpec
    manager: Any  # StreamableHTTPSessionManager
    runner: asyncio.Task  # owns the manager's run() context
    stop: asyncio.Event
    hook_handler: Optional[HookHandler] = None
    turn_active: Optional[TurnActive] = None


async def _run_manager(
    manager: Any, started: asyncio.Future, stop: asyncio.Event
) -> None:
    # The manager's task group must be entered and exited by the same task.
    try:
        async with manager.run():
            started.set_result(None)
            await stop.wait()
    except BaseException as exc:
        if not started.done():
            started.set_exception(exc)
        raise


def _token_key(token: str) -> str:
    # Lookup by digest; the token itself is then compared in constant time.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _bearer(scope: Any) -> str:
    headers = dict(scope.get("headers") or [])
    auth = headers.get(b"authorization", b"").decode("latin-1")
    return auth[len("Bearer ") :] if auth.startswith("Bearer ") else ""


class LocalMcpHttpServer:
    """Lazily-started uvicorn server exposing per-conversation MCP endpoints."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self._host = host
        self._port: Optional[int] = None
        self._sessions: dict[str, _Session] = {}  # token digest -> session
        self._server: Any = None
        self._sock: Optional[socket.socket] = None
        self._serve_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._serve_task is not None and not self._serve_task.done()

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def _ensure_started(self) -> None:
        if self.running:
            return
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Route

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, 0))
        # Listening now queues early clients until uvicorn accepts them.
        sock.listen(128)
        self._port = sock.getsockname()[1]
        sock.setblocking(False)
        self._sock = sock

        app = Starlette(
            routes=[
                Route("/mcp", endpoint=_ASGIEndpoint(self.handle_mcp)),
                Route("/hook", endpoint=_ASGIEndpoint(self.handle_hook)),
            ]
        )
        config = uvicorn.Config(
            app,
            lifespan="off",
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=2,
            http="h11",
        )
        server = _embedded_server_cls(uvicorn)(config)
        self._server = server
        self._serve_task = asyncio.ensure_future(server.serve(sockets=[sock]))
        deadline = asyncio.get_running_loop().time() + _START_TIMEOUT_S
        while not getattr(server, "started", False):
            if self._serve_task.done():
                self._serve_task.result()  # surfaces the startup error
                raise RuntimeError("AF MCP HTTP server stopped during startup")
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError(
                    f"AF MCP HTTP server did not start within {_START_TIMEOUT_S}s"
                )
            await asyncio.sleep(0.02)

    async def register(
        self,
        tools: list,
        *,
        hook_handler: Optional[HookHandler] = None,
        turn_active: Optional[TurnActive] = None,
        result_max_chars: int = 0,
    ) -> tuple[str, str, dict[str, str]]:
        """Register a conversation's tools (and optionally its hook handler and
        the predicate that says whether its vendor turn is running); returns
        ``(mcp_url, token, headers)``. The URL carries no secret.

        A ``result_max_chars`` (the bridge's result size budget) is declared
        on every tool as ``_meta["anthropic/maxResultSizeChars"]``: Claude
        Code spills a tool result above the declared size (at most 500,000)
        to a file and shows the model only its head; for a tool that declares
        none, one above 50,000 characters or its MCP output token limit
        (claude 2.1.288)."""
        from mcp.server.lowlevel import Server
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from mcp.server.transport_security import TransportSecuritySettings

        async with self._lock:
            await self._ensure_started()
            token = secrets.token_urlsafe(32)
            by_name = {spec.name: spec for spec in tools}
            server = self._build_server(Server, token, by_name, result_max_chars)
            manager = StreamableHTTPSessionManager(
                app=server,
                json_response=True,
                stateless=True,
                security_settings=TransportSecuritySettings(
                    enable_dns_rebinding_protection=True,
                    allowed_hosts=[f"{self._host}:*", "localhost:*"],
                    allowed_origins=[],
                ),
            )
            loop = asyncio.get_running_loop()
            started: asyncio.Future = loop.create_future()
            stop = asyncio.Event()
            runner = loop.create_task(
                _run_manager(manager, started, stop), context=contextvars.Context()
            )
            await started
            self._sessions[_token_key(token)] = _Session(
                token=token,
                tools=by_name,
                manager=manager,
                runner=runner,
                stop=stop,
                hook_handler=hook_handler,
                turn_active=turn_active,
            )
        headers = {"Authorization": f"Bearer {token}"}
        return f"{self.base_url}/mcp", token, headers

    async def unregister(self, token: str) -> None:
        async with self._lock:
            session = self._sessions.pop(_token_key(token), None)
        if session is not None:
            session.stop.set()
            try:
                await asyncio.wait_for(asyncio.shield(session.runner), timeout=5)
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("MCP session teardown failed: %s", exc)
                session.runner.cancel()

    async def stop(self) -> None:
        for session in list(self._sessions.values()):
            await self.unregister(session.token)
        if self._server is not None:
            self._server.should_exit = True
        if self._serve_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._serve_task), timeout=5)
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("MCP http server stop: %s", exc)
            self._serve_task = None
        self._server = None
        self._sock = None

    def abandon(self) -> None:
        """Release the listening socket of a server whose event loop is gone
        (its tasks can no longer run, so ``stop()`` is impossible)."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._sessions.clear()
        self._serve_task = None
        self._server = None

    @staticmethod
    def _build_server(
        server_cls: Any, token: str, by_name: dict[str, Any], result_max_chars: int
    ) -> Any:
        import mcp.types as mt

        server = server_cls(f"af-{_token_key(token)[:6]}")
        meta = {_MAX_RESULT_META: result_max_chars} if result_max_chars else None

        @server.list_tools()
        async def _list() -> list:
            return [
                mt.Tool(
                    name=s.name,
                    description=s.description,
                    inputSchema=s.input_schema,
                    _meta=meta,
                )
                for s in by_name.values()
            ]

        # The bridge validates arguments against the session's own schema.
        @server.call_tool(validate_input=False)
        async def _call(name: str, arguments: dict) -> mt.CallToolResult:
            spec = by_name.get(name)
            if spec is None:
                return mt.CallToolResult(
                    content=[mt.TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True,
                )
            # Errors are surfaced to the model as results, not transport errors.
            result = await spec.handler(arguments or {})
            return mt.CallToolResult(
                content=[
                    mt.TextContent(
                        type="text", text=getattr(result, "text", str(result))
                    )
                ],
                isError=bool(getattr(result, "is_error", False)),
            )

        return server

    def _session_for(self, scope: Any) -> Optional[_Session]:
        token = _bearer(scope)
        if not token:
            return None
        session = self._sessions.get(_token_key(token))
        if session is None or not hmac.compare_digest(session.token, token):
            return None
        return session

    async def handle_mcp(self, scope: Any, receive: Any, send: Any) -> None:
        session = self._session_for(scope)
        if session is None:
            await _respond(send, 401, b"unauthorized")
            return
        if not _in_turn(session):
            await _respond(send, 403, b"no active turn")
            return
        await session.manager.handle_request(scope, receive, send)

    async def handle_hook(self, scope: Any, receive: Any, send: Any) -> None:
        session = self._session_for(scope)
        if session is None or session.hook_handler is None:
            await _respond(send, 401, b"unauthorized")
            return
        if not _in_turn(session):
            await _respond(send, 403, b"no active turn")
            return
        body = await _read_body(receive)
        if body is None:
            await _respond(send, 413, b"payload too large")
            return
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            await _respond(send, 400, b"invalid json")
            return
        try:
            output = await session.hook_handler(payload)
        except Exception as exc:
            # No decision: the relay then fails closed for the hook's event.
            logger.warning("AF hook handler failed: %s", exc)
            await _respond(send, 500, b"hook handler failed")
            return
        await _respond(
            send, 200, json.dumps(output).encode("utf-8"), b"application/json"
        )


def _in_turn(session: _Session) -> bool:
    return session.turn_active is None or bool(session.turn_active())


async def _read_body(receive: Any) -> Optional[bytes]:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > _MAX_HOOK_BODY:
            return None
        chunks.append(chunk)
        if not message.get("more_body"):
            return b"".join(chunks)


async def _respond(
    send: Any, status: int, body: bytes, content_type: bytes = b"text/plain"
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", content_type)],
        }
    )
    await send({"type": "http.response.body", "body": body})


class _ASGIEndpoint:
    """Raw-ASGI endpoint (callable instance, like the S11 spike) so Starlette
    hands the request straight to the handler for every HTTP method."""

    def __init__(self, handler: Callable[[Any, Any, Any], Awaitable[None]]) -> None:
        self._handler = handler

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._handler(scope, receive, send)


def _embedded_server_cls(uvicorn: Any) -> type:
    """A uvicorn server that leaves signals to the host process: it stops
    only through ``LocalMcpHttpServer.stop()`` (uvicorn's ``serve()`` would
    otherwise take over SIGINT/SIGTERM from the host while it runs)."""

    class _EmbeddedServer(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            yield

    return _EmbeddedServer

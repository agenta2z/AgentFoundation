"""S11 spike: can `claude -p --mcp-config <localhost http>` reach a localhost
streamable-HTTP MCP server under the Meta launcher sandbox, and call a tool?

Decides the per-turn-CLI backend transport (claude_cli / devmate_dm / codex_cli).
If this fails, those backends need a stdio relay instead of localhost HTTP.

Run:
    source /tmp/af_env.sh
    PYTHONPATH="$AFL" python3 scripts/native_spikes/s11_http_mcp.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import socket
import sys
import tempfile
import uuid

import mcp.types as mt
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route

TOKEN = secrets.token_urlsafe(24)
CALLED: list[dict] = []


def build_app() -> Starlette:
    server: Server = Server("af")

    @server.list_tools()
    async def _list() -> list[mt.Tool]:
        return [
            mt.Tool(
                name="echo",
                description="Echo the given text back verbatim.",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ]

    @server.call_tool()
    async def _call(name: str, arguments: dict) -> list[mt.ContentBlock]:
        CALLED.append({"name": name, "arguments": arguments})
        return [mt.TextContent(type="text", text=f"ECHO::{arguments.get('text', '')}")]

    mgr = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)

    class _Endpoint:
        async def __call__(self, scope, receive, send) -> None:
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode()
            if auth != f"Bearer {TOKEN}":
                await send(
                    {"type": "http.response.start", "status": 401, "headers": []}
                )
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
            await mgr.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with mgr.run():
            yield

    return Starlette(routes=[Route("/mcp", endpoint=_Endpoint())], lifespan=lifespan)


async def main() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.setblocking(False)

    config = uvicorn.Config(
        build_app(),
        lifespan="on",
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=2,
        http="h11",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[assignment]
    serve_task = asyncio.ensure_future(server.serve(sockets=[sock]))
    await asyncio.sleep(1.0)  # let it start
    url = f"http://127.0.0.1:{port}/mcp"
    print(f"[s11] port={port}")
    failures = []
    try:
        # The bearer token reaches the server: the tool runs, its result
        # comes back into the reply.
        nonce = f"PINGPONG-{uuid.uuid4().hex[:6]}"
        rc, out_s, err_s = await _claude_echo(url, TOKEN, nonce)
        calls = [c for c in CALLED if c["arguments"].get("text") == nonce]
        print(f"[s11] claude rc={rc} stdout:\n{out_s.strip()[-600:]}")
        print(f"[s11] server received calls: {calls}")
        if not (rc == 0 and len(calls) == 1 and f"ECHO::{nonce}" in out_s):
            failures.append("bearer token: tool invoked, result in the reply")
            print(f"[s11] stderr tail:\n{err_s.strip()[-600:]}")
        # Control: with a wrong token the server refuses (401) and no call
        # reaches it — the check above depends on the header.
        wrong = f"PINGPONG-{uuid.uuid4().hex[:6]}"
        rc, out_s, _err = await _claude_echo(url, "wrong-token", wrong)
        refused = [c for c in CALLED if c["arguments"].get("text") == wrong]
        print(f"[s11] wrong token: rc={rc} calls={refused} {out_s.strip()[-200:]!r}")
        if refused:
            failures.append("wrong token: no call reaches the server")
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(serve_task, timeout=5)
    if failures:
        print(f"[s11 FAIL] {failures}")
        return 1
    print(
        "[s11 PASS] localhost HTTP MCP reachable from claude -p with the bearer "
        "header; tool invoked, result returned; a wrong token reaches no tool."
    )
    return 0


async def _claude_echo(url: str, token: str, nonce: str) -> tuple[int, str, str]:
    """One ``claude -p`` asked to call ``echo(nonce)`` on the server at ``url``
    with ``token``; returns ``(rc, stdout, stderr)`` (rc -1 on a timeout)."""
    cfg = {
        "mcpServers": {
            "af": {
                "type": "http",
                "url": url,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    }
    fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="af_mcp_")
    os.write(fd, json.dumps(cfg).encode())
    os.close(fd)
    os.chmod(cfg_path, 0o600)
    prompt = (
        f"Call the echo tool with text set to {nonce}. "
        "Then reply with exactly the tool's returned text and nothing else."
    )
    cmd = [
        "claude",
        "-p",
        "--model",
        "sonnet",
        "--mcp-config",
        cfg_path,
        "--strict-mcp-config",
        "--allowedTools",
        "mcp__af__echo",
        "--permission-mode",
        "bypassPermissions",
        "--",
        prompt,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=180)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", "timed out"
    finally:
        os.unlink(cfg_path)
    return (
        proc.returncode or 0,
        out.decode(errors="replace"),
        err.decode(errors="replace"),
    )


if __name__ == "__main__":
    os.chdir(tempfile.mkdtemp())
    sys.exit(asyncio.run(main()))

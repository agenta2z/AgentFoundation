"""Shared helpers for the native spikes (scripts only, not a library).

Every spike drives the real ``claude`` / ``codex`` binaries and asserts on
transport evidence: stream-json events, hook inputs, MCP calls received by a
server the spike owns, and the Claude Code transcript (JSONL under
``$CLAUDE_CONFIG_DIR/projects/<cwd slug>/<session>.jsonl``). The pieces here
are self-contained (no AgentFoundation imports) so a spike keeps measuring the
vendor even while the native code changes:

* ``Checks``        — ``[PASS]/[FAIL] <check> <detail>`` lines and the exit code.
* transcripts       — locate and read a session's JSONL; prompt snapshots,
                      hook-context attachments, user prompts, usage.
* ``run_claude``    — one ``claude -p --output-format stream-json`` process.
* ``CommandHooks``  — Claude Code command hooks backed by files (stdlib relay).
* ``HttpMcpServer`` — a bearer-protected streamable-HTTP MCP server.
* ``sdk_mcp_server``— the same tools as an in-process Agent SDK server.
* ``write_stdio_canary_server`` — a stdlib stdio MCP server that records
                      that a vendor started it.

Logs are printed; session ids are shortened (never the full id).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

DISALLOWED = ("AskUserQuestion", "EnterPlanMode", "ExitPlanMode")
SYSTEM_PYTHON = shutil.which("python3", path="/usr/local/bin:/usr/bin") or "python3"


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------


class Checks:
    """Collects ``[PASS]/[FAIL]`` results; ``exit_code()`` is non-zero on any failure."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.failures: list[str] = []
        self.passed = 0

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        text = detail if isinstance(detail, str) else repr(detail)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {text[:400]!r}", flush=True)
        if ok:
            self.passed += 1
        else:
            self.failures.append(name)
        return ok

    def info(self, name: str, detail: Any = "") -> None:
        text = detail if isinstance(detail, str) else repr(detail)
        print(f"[INFO] {name} {text[:600]}", flush=True)

    def exit_code(self) -> int:
        if self.failures:
            print(f"{self.label} FAIL ({len(self.failures)}): {self.failures}")
            return 1
        print(f"{self.label} PASS ({self.passed} checks)")
        return 0


def short(session_id: Optional[str]) -> str:
    return (session_id or "")[:8]


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


# ----------------------------------------------------------------------------
# Claude Code transcripts
# ----------------------------------------------------------------------------


def claude_config_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude"))


def project_dir(cwd: str) -> Path:
    return claude_config_dir() / "projects" / re.sub(r"[^a-zA-Z0-9]", "-", cwd)


def transcripts(cwd: str) -> list[Path]:
    """Every session transcript of the project at ``cwd``, oldest first."""
    return sorted(project_dir(cwd).glob("*.jsonl"), key=os.path.getmtime)


def find_transcript(session_id: str) -> Optional[Path]:
    hits = list((claude_config_dir() / "projects").glob(f"*/{session_id}.jsonl"))
    return hits[0] if hits else None


def read_entries(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def session_entries(session_id: str) -> list[dict[str, Any]]:
    return read_entries(find_transcript(session_id))


def attachments(entries: Iterable[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [
        e
        for e in entries
        if e.get("type") == "attachment"
        and isinstance(e.get("attachment"), dict)
        and e["attachment"].get("type") == kind
    ]


def snapshot_texts(entries: Iterable[dict[str, Any]]) -> list[str]:
    """The recorded system prompts (``prompt_snapshot`` attachments), joined."""
    texts = []
    for e in attachments(entries, "prompt_snapshot"):
        prompt = e["attachment"].get("systemPrompt")
        texts.append("\n".join(prompt) if isinstance(prompt, list) else str(prompt))
    return texts


def snapshots_after_compaction(entries: list[dict[str, Any]]) -> list[str]:
    """Recorded system prompts after the last ``compact_boundary`` (none if the
    session was never compacted)."""
    cut = max(
        (
            i
            for i, e in enumerate(entries)
            if e.get("type") == "system" and e.get("subtype") == "compact_boundary"
        ),
        default=None,
    )
    return [] if cut is None else snapshot_texts(entries[cut + 1 :])


def hook_contexts(
    entries: Iterable[dict[str, Any]], event: str = "UserPromptSubmit"
) -> list[dict[str, Any]]:
    """``hook_additional_context`` attachments recorded for ``event``."""
    return [
        e
        for e in attachments(entries, "hook_additional_context")
        if e["attachment"].get("hookEvent") == event
    ]


def entry_text(entry: dict[str, Any]) -> str:
    """Text a transcript entry carries (message text parts or attachment content)."""
    if entry.get("type") == "attachment":
        return json.dumps(entry.get("attachment"))
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


def user_prompts(entries: Iterable[dict[str, Any]]) -> list[str]:
    """Text of every user message that is not only tool results (main thread)."""
    texts = []
    for e in entries:
        if e.get("type") != "user" or e.get("isSidechain"):
            continue
        content = (e.get("message") or {}).get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            if parts:
                texts.append("".join(parts))
    return texts


def tool_results(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for e in entries:
        if e.get("type") != "user":
            continue
        for part in (e.get("message") or {}).get("content") or []:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                out.append(part)
    return out


def tool_uses(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for e in entries:
        if e.get("type") != "assistant":
            continue
        for part in (e.get("message") or {}).get("content") or []:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                out.append(part)
    return out


def tool_result_text(part: dict[str, Any]) -> str:
    content = part.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        c.get("text", "") for c in content or [] if isinstance(c, dict)
    ) or json.dumps(content)


def api_usage(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per API response (assistant entries share a request id per
    content block): ``timestamp``, ``sidechain`` and the usage counters."""
    seen: dict[str, dict[str, Any]] = {}
    for e in entries:
        if e.get("type") != "assistant":
            continue
        message = e.get("message") or {}
        usage = message.get("usage") or {}
        key = e.get("requestId") or message.get("id") or e.get("uuid")
        if not usage or key in seen:
            continue
        seen[key] = {
            "timestamp": e.get("timestamp", ""),
            "sidechain": bool(e.get("isSidechain")),
            "input": int(usage.get("input_tokens") or 0),
            "cache_write": int(usage.get("cache_creation_input_tokens") or 0),
            "cache_read": int(usage.get("cache_read_input_tokens") or 0),
            "output": int(usage.get("output_tokens") or 0),
        }
    return list(seen.values())


# ----------------------------------------------------------------------------
# claude -p
# ----------------------------------------------------------------------------


@dataclass
class CliRun:
    events: list[dict[str, Any]]
    rc: Optional[int]
    stderr: str
    elapsed_s: float
    timed_out: bool = False

    @property
    def init(self) -> dict[str, Any]:
        return next(
            (
                e
                for e in self.events
                if e.get("type") == "system" and e.get("subtype") == "init"
            ),
            {},
        )

    @property
    def result(self) -> dict[str, Any]:
        return next((e for e in reversed(self.events) if e.get("type") == "result"), {})

    @property
    def text(self) -> str:
        return str(self.result.get("result") or "")

    @property
    def session_id(self) -> str:
        return str(self.result.get("session_id") or self.init.get("session_id") or "")

    def assistant(self, *, main_only: bool = True) -> list[dict[str, Any]]:
        return [
            e
            for e in self.events
            if e.get("type") == "assistant"
            and not (main_only and e.get("parent_tool_use_id"))
        ]

    def system(self, subtype: str) -> list[dict[str, Any]]:
        return [
            e
            for e in self.events
            if e.get("type") == "system" and e.get("subtype") == subtype
        ]


def claude_argv(
    prompt: str,
    *,
    model: str,
    session_id: str = "",
    resume: str = "",
    append_file: str = "",
    mcp_config: str = "",
    settings: str = "",
    hermetic: bool = False,
    partial: bool = False,
    allowed_tools: Iterable[str] = (),
    extra: Iterable[str] = (),
) -> list[str]:
    """``claude -p`` argv shaped like the native ``claude_cli`` backend's."""
    argv = [
        shutil.which("claude") or "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--permission-mode",
        "bypassPermissions",
        "--disallowedTools",
        ",".join(DISALLOWED),
    ]
    if partial:
        argv.append("--include-partial-messages")
    if append_file:
        argv += ["--append-system-prompt-file", append_file]
    if settings:
        argv += ["--settings", settings]
    if mcp_config:
        argv += ["--mcp-config", mcp_config]
    if allowed_tools:
        argv += ["--allowedTools", ",".join(allowed_tools)]
    if hermetic:
        argv += ["--setting-sources", "", "--strict-mcp-config"]
    argv += list(extra)
    if resume:
        argv += ["--resume", resume]
    elif session_id:
        argv += ["--session-id", session_id]
    # ``--disallowedTools`` / ``--allowedTools`` / ``--mcp-config`` are variadic:
    # without the terminator a trailing one swallows the prompt.
    return argv + ["--", prompt]


async def run_claude(
    argv: list[str],
    *,
    cwd: str,
    env: Optional[dict[str, str]] = None,
    timeout: float = 300.0,
) -> CliRun:
    """Run one ``claude -p`` turn; every stream-json event gets ``_recv`` (epoch s)."""
    started = time.time()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=dict(os.environ, **(env or {})),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    events: list[dict[str, Any]] = []
    stderr_chunks: list[bytes] = []

    async def _read_stdout() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.strip()
            if not line.startswith(b"{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj["_recv"] = time.time()
            events.append(obj)

    async def _read_stderr() -> None:
        assert proc.stderr is not None
        stderr_chunks.append(await proc.stderr.read())

    timed_out = False
    try:
        await asyncio.wait_for(
            asyncio.gather(_read_stdout(), _read_stderr(), proc.wait()), timeout
        )
    except asyncio.TimeoutError:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
    return CliRun(
        events=events,
        rc=proc.returncode,
        stderr=b"".join(stderr_chunks).decode(errors="replace"),
        elapsed_s=time.time() - started,
        timed_out=timed_out,
    )


# ----------------------------------------------------------------------------
# Command hooks (CLI)
# ----------------------------------------------------------------------------

_HOOK_SCRIPT = r"""
import json, os, sys, time
root = os.path.dirname(os.path.abspath(__file__))
raw = sys.stdin.read()
try:
    payload = json.loads(raw or "{}")
except ValueError:
    payload = {"_raw": raw}
payload["_ts"] = time.time()
event = str(payload.get("hook_event_name", "unknown"))
with open(os.path.join(root, "log.jsonl"), "a") as fh:
    fh.write(json.dumps(payload) + "\n")
path = os.path.join(root, "responses", event + ".json")
if os.path.exists(path):
    with open(path) as fh:
        sys.stdout.write(fh.read())
"""


class CommandHooks:
    """Claude Code command hooks backed by files: each invocation appends its
    stdin (plus ``_ts``) to ``log.jsonl`` and prints ``responses/<event>.json``
    when present. Stdlib only, so it runs under any ``python3``."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        (self.root / "responses").mkdir(parents=True, exist_ok=True)
        self.script = self.root / "hook.py"
        self.script.write_text(_HOOK_SCRIPT, encoding="utf-8")
        self.command = f"{SYSTEM_PYTHON} {self.script}"

    def settings(self, events: dict[str, Optional[str]]) -> str:
        """``--settings`` JSON: ``{event: matcher-or-None}``."""
        hooks: dict[str, list[dict[str, Any]]] = {}
        for event, matcher in events.items():
            entry: dict[str, Any] = {
                "hooks": [{"type": "command", "command": self.command}]
            }
            if matcher:
                entry["matcher"] = matcher
            hooks[event] = [entry]
        return json.dumps({"hooks": hooks})

    def respond(self, event: str, payload: Optional[dict[str, Any]]) -> None:
        path = self.root / "responses" / f"{event}.json"
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")

    def log(self) -> list[dict[str, Any]]:
        return read_entries(self.root / "log.jsonl")

    def mark(self) -> int:
        return len(self.log())


# ----------------------------------------------------------------------------
# MCP tools: HTTP server (CLI) and in-process server (Agent SDK)
# ----------------------------------------------------------------------------

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class SpikeTool:
    name: str
    description: str
    properties: dict[str, Any]
    handler: ToolHandler
    required: tuple[str, ...] = ()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": self.properties,
            "required": list(self.required),
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    started: float
    ended: Optional[float] = None
    cancelled: bool = False


@dataclass
class CallLog:
    calls: list[ToolCall] = field(default_factory=list)

    def wrap(self, tool: SpikeTool) -> ToolHandler:
        async def _run(arguments: dict[str, Any]) -> str:
            call = ToolCall(tool.name, dict(arguments), time.time())
            self.calls.append(call)
            try:
                return await tool.handler(arguments)
            except asyncio.CancelledError:
                call.cancelled = True
                raise
            finally:
                call.ended = time.time()

        return _run

    def named(self, name: str) -> list[ToolCall]:
        return [c for c in self.calls if c.name == name]


class HttpMcpServer:
    """Streamable-HTTP MCP server on 127.0.0.1 (stateless, JSON responses),
    bearer-protected — the transport the ``claude_cli`` / ``codex_cli``
    backends use for AF tools."""

    def __init__(self, tools: list[SpikeTool], log: Optional[CallLog] = None) -> None:
        self.tools = {t.name: t for t in tools}
        self.log = log or CallLog()
        self.token = secrets.token_urlsafe(24)
        self.url = ""
        self._server: Any = None
        self._task: Optional[asyncio.Task] = None

    def _app(self) -> Any:
        import mcp.types as mt
        from mcp.server.lowlevel import Server
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Route

        server: Any = Server("af")
        handlers = {name: self.log.wrap(t) for name, t in self.tools.items()}

        @server.list_tools()
        async def _list() -> list[Any]:
            return [
                mt.Tool(name=t.name, description=t.description, inputSchema=t.schema)
                for t in self.tools.values()
            ]

        @server.call_tool()
        async def _call(name: str, arguments: dict[str, Any]) -> list[Any]:
            text = await handlers[name](arguments or {})
            return [mt.TextContent(type="text", text=text)]

        manager = StreamableHTTPSessionManager(
            app=server, json_response=True, stateless=True
        )
        token = self.token

        class _Endpoint:
            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                headers = dict(scope.get("headers") or [])
                if headers.get(b"authorization", b"").decode() != f"Bearer {token}":
                    await send(
                        {"type": "http.response.start", "status": 401, "headers": []}
                    )
                    await send({"type": "http.response.body", "body": b"unauthorized"})
                    return
                await manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(_app: Any) -> Any:
            async with manager.run():
                yield

        return Starlette(
            routes=[Route("/mcp", endpoint=_Endpoint())], lifespan=lifespan
        )

    async def start(self) -> "HttpMcpServer":
        import uvicorn

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.setblocking(False)
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
        config = uvicorn.Config(
            self._app(),
            lifespan="on",
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=2,
            http="h11",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None
        self._task = asyncio.ensure_future(self._server.serve(sockets=[sock]))
        for _ in range(50):
            if self._server.started:
                break
            await asyncio.sleep(0.1)
        return self

    def write_config(self, directory: str, name: str = "af") -> str:
        """A 0600 ``--mcp-config`` file (the token never goes on argv)."""
        path = os.path.join(directory, f"mcp_{name}.json")
        config = {
            "mcpServers": {
                name: {
                    "type": "http",
                    "url": self.url,
                    "headers": {"Authorization": f"Bearer {self.token}"},
                }
            }
        }
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(config, fh)
        return path

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=5)


def sdk_mcp_server(
    tools: list[SpikeTool], log: CallLog, name: str = "af"
) -> dict[str, Any]:
    """The tools as an in-process Agent SDK MCP server (the ``claude_sdk`` path)."""
    from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool

    def _wrap(t: SpikeTool) -> Any:
        run = log.wrap(t)

        @sdk_tool(t.name, t.description, t.schema)
        async def _handler(args: dict[str, Any]) -> dict[str, Any]:
            text = await run(args)
            return {"content": [{"type": "text", "text": text}]}

        return _handler

    return create_sdk_mcp_server(
        name=name, version="1.0.0", tools=[_wrap(t) for t in tools]
    )


def echo_tool(prefix: str = "ECHO::") -> SpikeTool:
    async def _echo(args: dict[str, Any]) -> str:
        return f"{prefix}{args.get('text', '')}"

    return SpikeTool(
        name="echo",
        description="Echo the given text back verbatim.",
        properties={"text": {"type": "string"}},
        required=("text",),
        handler=_echo,
    )


# ----------------------------------------------------------------------------
# Agent SDK turn
# ----------------------------------------------------------------------------


@dataclass
class SdkRun:
    messages: list[Any]
    elapsed_s: float
    error: str = ""
    stderr: list[str] = field(default_factory=list)

    @property
    def result(self) -> Any:
        from claude_agent_sdk import ResultMessage

        return next(
            (m for m in reversed(self.messages) if isinstance(m, ResultMessage)), None
        )

    @property
    def text(self) -> str:
        result = self.result
        return str(getattr(result, "result", "") or "")

    @property
    def session_id(self) -> str:
        result = self.result
        return str(getattr(result, "session_id", "") or "")

    def assistant(self, *, main_only: bool = True) -> list[Any]:
        from claude_agent_sdk import AssistantMessage

        return [
            m
            for m in self.messages
            if isinstance(m, AssistantMessage)
            and not (main_only and m.parent_tool_use_id)
        ]

    def system(self, subtype: str) -> list[Any]:
        from claude_agent_sdk import SystemMessage

        return [
            m
            for m in self.messages
            if isinstance(m, SystemMessage) and m.subtype == subtype
        ]


def sdk_options(
    *,
    cwd: str,
    model: str,
    session_id: str = "",
    resume: str = "",
    append_file: str = "",
    hooks: Optional[dict[str, list[Any]]] = None,
    mcp_servers: Optional[dict[str, Any]] = None,
    env: Optional[dict[str, str]] = None,
    hermetic: bool = False,
    partial: bool = False,
    stderr: Optional[Callable[[str], None]] = None,
) -> Any:
    """``ClaudeAgentOptions`` shaped like the native ``claude_sdk`` backend's
    (preset prompt + append file, AF tools as ``mcp__af``, hooks)."""
    from claude_agent_sdk import ClaudeAgentOptions

    extra_args: dict[str, Optional[str]] = {}
    if append_file:
        extra_args["append-system-prompt-file"] = append_file
    if hermetic:
        extra_args["setting-sources"] = ""
        extra_args["strict-mcp-config"] = None
    kwargs: dict[str, Any] = {
        "system_prompt": {"type": "preset", "preset": "claude_code"},
        "cwd": cwd,
        "model": model,
        "permission_mode": "bypassPermissions",
        "disallowed_tools": list(DISALLOWED),
        "include_partial_messages": partial,
        "hooks": hooks or {},
        "env": env or {},
        "extra_args": extra_args,
        "cli_path": shutil.which("claude"),
    }
    if stderr is not None:
        kwargs["stderr"] = stderr
    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers
        kwargs["allowed_tools"] = [f"mcp__{name}" for name in mcp_servers]
    if resume:
        kwargs["resume"] = resume
    elif session_id:
        kwargs["session_id"] = session_id
    return ClaudeAgentOptions(**kwargs)


async def run_sdk(
    options: Any,
    prompt: str,
    *,
    timeout: float = 300.0,
    on_message: Optional[Callable[[Any], None]] = None,
    linger_s: float = 0.0,
) -> SdkRun:
    """One SDK turn on a fresh ``ClaudeSDKClient`` (connect, query, drain the
    response, disconnect) from the calling task. ``linger_s`` keeps the client
    (and its in-process MCP servers) alive that long after the result."""
    from claude_agent_sdk import ClaudeSDKClient

    started = time.time()
    messages: list[Any] = []
    stderr_lines: list[str] = []
    if getattr(options, "stderr", None) is None:
        options.stderr = stderr_lines.append

    async def _turn() -> None:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                message._recv = time.time()
                messages.append(message)
                if on_message is not None:
                    on_message(message)
            if linger_s:
                await asyncio.sleep(linger_s)

    error = ""
    try:
        await asyncio.wait_for(_turn(), timeout + linger_s)
    except asyncio.TimeoutError:
        error = f"timed out after {timeout:.0f}s"
    except Exception as exc:  # reported by the caller's checks
        error = f"{type(exc).__name__}: {exc}"
    return SdkRun(
        messages=messages,
        elapsed_s=time.time() - started,
        error=error,
        stderr=stderr_lines,
    )


def hook_matcher(callback: Callable[..., Awaitable[dict]], matcher: str = "") -> Any:
    from claude_agent_sdk import HookMatcher

    return HookMatcher(matcher=matcher or None, hooks=[callback])


# ----------------------------------------------------------------------------
# A stdlib stdio MCP server that records that a vendor started it
# ----------------------------------------------------------------------------

_STDIO_CANARY = r"""
import json, os, sys, time
log = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "canary"
def note(event):
    with open(log, "a") as fh:
        fh.write(json.dumps({"event": event, "ts": time.time(), "pid": os.getpid()}) + "\n")
def reply(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
note("started")
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    method, msg_id = msg.get("method"), msg.get("id")
    if method == "initialize":
        note("initialize")
        version = (msg.get("params") or {}).get("protocolVersion", "2025-06-18")
        reply(msg_id, {"protocolVersion": version, "capabilities": {"tools": {}},
                       "serverInfo": {"name": name, "version": "1.0.0"}})
    elif method == "tools/list":
        note("tools/list")
        reply(msg_id, {"tools": [{"name": name + "_ping",
                                   "description": "Canary tool. Returns PONG.",
                                   "inputSchema": {"type": "object", "properties": {}}}]})
    elif method == "tools/call":
        note("tools/call")
        reply(msg_id, {"content": [{"type": "text", "text": "PONG"}]})
    elif msg_id is not None:
        if method == "ping":
            reply(msg_id, {})
        else:
            reply(msg_id, error={"code": -32601, "message": "method not found"})
"""


def write_stdio_canary_server(directory: str) -> tuple[str, str]:
    """Write the canary server; returns ``(script_path, log_path)``."""
    script = os.path.join(directory, "canary_mcp.py")
    log = os.path.join(directory, "canary_mcp.log")
    Path(script).write_text(_STDIO_CANARY, encoding="utf-8")
    return script, log


def canary_events(log_path: str) -> list[str]:
    return [e.get("event", "") for e in read_entries(Path(log_path))]


def python_path_env() -> dict[str, str]:
    """``PYTHONPATH`` for a child Python that needs this process's imports (the
    spike interpreter drops ``PYTHONPATH`` from the environment it passes on)."""
    return {"PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}

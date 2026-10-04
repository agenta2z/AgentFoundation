"""S14 spike: Devmate ``dm -p`` as a per-turn native backend — session pinning
and resume continuity, the ``devmate-sdk-events`` schema the backend maps, and
the AF-tool MCP round trip over a unix socket.

dm-core's ``--mcp-servers`` supports only ``type:"socket"`` (unix domain socket,
newline-delimited JSON-RPC) and ``type:"stdio"`` — no HTTP. The ``devmate_dm``
backend serves AF tools from an in-process unix-socket server (0600) and has dm
reach it through its stdio transport via the stdlib relay
(``bridge/mcp_stdio_relay.py``): dm's socket transport stalls real-model turns
(see "Real model" below).

Everything is asserted against dm's scripted model (``--scripted-model``,
``DM_ALLOW_SCRIPTED_MODEL=1``): model responses come from a fixture (no auth,
no cost) while tool calls really execute. In ``responseSelection:
"tool-result-count"`` mode the fixture entry served is indexed by the number of
tool results in the request dm assembled for the model, so a fixture can
answer differently depending on whether that request carries an earlier
turn's tool result — which makes history continuity observable without a real
model:

* A  direct socket transport: the scripted ``mcp__af__echo`` call reaches the
     socket server (dm is stopped once it lands; this transport is not used).
* B1 turn 1, ``--session-id <uuid>``, production path (``--agent-harness
     native``, stdio relay -> socket): dm exits by itself (rc 0); every session
     event carries the pinned id; the events the backend maps are present with
     the fields it reads; the relay delivered the echo call.
* B2 turn 2, ``--resume <uuid>``: every session event carries the same id (not
     a replacement session); the turn-2 fixture answers with a second echo call
     only when the request holds turn 1's tool result -> received, final text
     ``HISTORY-OK``.
* B3 control, the turn-2 fixture in a fresh ``--session-id``: no history ->
     ``NO-HISTORY``, no echo call (so B2 depends on the carried history).

Real model (``S14_REAL_MODEL=<model>``, needs auth; not asserted by default):
with ``S14_STDIO_RELAY=1`` a real-model turn over the stdio relay calls the
tool; without it, dm's direct socket transport stalls a real-model turn (the
dm-core bug that motivates the relay).

Run (from the AgentFoundation root):
    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s14_dm_socket_mcp.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import anyio
import mcp.types as mt
from _spike_common import Checks, section, short
from mcp.server.lowlevel import Server
from mcp.shared.message import SessionMessage

CALLED: list[dict] = []

_RELAY = (
    Path(__file__).resolve().parents[2]
    / "src/agent_foundation/common/inferencers/agentic_inferencers"
    / "conversational_native/bridge/mcp_stdio_relay.py"
)
_SYSTEM_PYTHON = shutil.which("python3", path="/usr/local/bin:/usr/bin") or "python3"
_DM_BASE = ["dm", "-p", "--output-format", "devmate-sdk-events"]
_APPEND = ["--append-system-prompt", "You are a spike harness."]
_OK_EXITS = ("COMPLETE", "COMPLETED")
_RUN_TIMEOUT_S = 150.0


class SocketMcpServer:
    """In-process unix-socket MCP server speaking newline-delimited JSON-RPC,
    the framing of dm-core's socket client and of the stdio relay."""

    def __init__(self) -> None:
        self._dir = tempfile.mkdtemp(prefix="af_dm_mcp_")
        os.chmod(self._dir, stat.S_IRWXU)  # 0700
        self.socket_path = os.path.join(self._dir, "af.sock")
        self._server: Any = None

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(
            self._handle, path=self.socket_path
        )
        os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        shutil.rmtree(self._dir, ignore_errors=True)

    @staticmethod
    def _build() -> Server:
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
            return [
                mt.TextContent(type="text", text=f"ECHO::{arguments.get('text', '')}")
            ]

        return server

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        server = self._build()
        read_w, read_r = anyio.create_memory_object_stream(0)
        write_w, write_r = anyio.create_memory_object_stream(0)

        async def pump_in() -> None:
            async with read_w:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    text = line.decode("utf-8").strip()
                    if not text:
                        continue
                    try:
                        msg = mt.JSONRPCMessage.model_validate_json(text)
                    except Exception as exc:  # malformed line -> surfaced to SDK
                        await read_w.send(exc)
                        continue
                    await read_w.send(SessionMessage(msg))

        async def pump_out() -> None:
            try:
                async with write_r:
                    async for sm in write_r:
                        data = sm.message.model_dump_json(
                            by_alias=True, exclude_none=True
                        )
                        writer.write((data + "\n").encode("utf-8"))
                        await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                pass

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(pump_in)
                tg.start_soon(pump_out)
                await server.run(
                    read_r, write_w, server.create_initialization_options()
                )
                tg.cancel_scope.cancel()
        finally:
            writer.close()


# ----------------------------------------------------------------------------
# Fixtures (scripted model)
# ----------------------------------------------------------------------------


def _echo_call(call_id: str, text: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": "mcp__af__echo",
        "arguments": json.dumps({"text": text}),
    }


def _write_fixture(directory: str, name: str, responses: list[dict]) -> str:
    path = os.path.join(directory, name)
    fixture = {
        "config": {"responseSelection": "tool-result-count"},
        "responses": {"dvsc_agent_loop": responses},
    }
    Path(path).write_text(json.dumps(fixture), encoding="utf-8")
    return path


def _turn1_fixture(directory: str, nonce: str) -> str:
    # [n] = the response when the request holds n tool results.
    done = {"text": "TURN1-DONE"}
    return _write_fixture(
        directory,
        f"turn1_{nonce}.json",
        [{"text": "Calling echo.", "toolCalls": [_echo_call("call_t1", nonce)]}]
        + [done] * 3,
    )


def _turn2_fixture(directory: str, nonce: str) -> str:
    # 0 tool results: the request has no turn 1 -> NO-HISTORY, no tool call.
    # 1 (turn 1's): call echo(nonce); 2: HISTORY-OK.
    ok = {"text": "HISTORY-OK"}
    return _write_fixture(
        directory,
        f"turn2_{nonce}.json",
        [
            {"text": "NO-HISTORY"},
            {
                "text": "History has turn 1.",
                "toolCalls": [_echo_call("call_t2", nonce)],
            },
        ]
        + [ok] * 3,
    )


# ----------------------------------------------------------------------------
# One dm run
# ----------------------------------------------------------------------------


@dataclass
class DmRun:
    events: list[dict] = field(default_factory=list)  # non-ephemeral `event`s
    rc: Optional[int] = None
    stderr: str = ""
    exited: bool = False  # by itself, before the deadline
    leftover: list[int] = field(default_factory=list)  # killed after the run

    def of(self, kind: str) -> list[dict]:
        return [e[kind] or {} for e in self.events if kind in e]

    def session_ids(self) -> list[str]:
        return [
            str((body.get("session") or {}).get("id", ""))
            for kind in ("session_start", "session_update", "session_end")
            for body in self.of(kind)
        ]

    def actions(self, variant: str) -> list[dict]:
        out = []
        for body in self.of("action_end"):
            action = body.get("action") or {}
            if variant in (action.get("variant") or {}):
                out.append(action)
        return out

    def llm_texts(self) -> list[str]:
        return [
            str((a.get("output") or {}).get("info") or "")
            for a in self.actions("llm_action")
        ]

    def kinds(self) -> list[str]:
        return [next(iter(e)) for e in self.events if e]


def _pids_naming(token: str) -> list[int]:
    """Live processes (but this one) whose command line contains ``token``."""
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if token.encode() in (proc / "cmdline").read_bytes():
                pids.append(int(proc.name))
        except OSError:
            continue
    return pids


async def _run_dm(
    argv: list[str],
    cwd: str,
    *,
    token: str,
    stop_after_call: Optional[str] = None,
) -> DmRun:
    """Run one ``dm -p``; ``stop_after_call``: stop it once the socket server
    received ``echo(stop_after_call)``. ``token`` (a unique argv string, e.g.
    the fixture path) finds what dm left running afterwards: its dm-core
    ``node`` runs in a process group of its own and outlives the launcher, so
    it is killed by name. Output goes to files: a pipe held open by such a
    survivor would keep the process from being reaped."""
    run = DmRun()
    out_path = os.path.join(cwd, f"dm_{uuid.uuid4().hex[:8]}.out")
    with open(out_path, "wb") as out, open(out_path + ".err", "wb") as err:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(os.environ, DM_ALLOW_SCRIPTED_MODEL="1"),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _RUN_TIMEOUT_S
    while proc.returncode is None and loop.time() < deadline:
        if stop_after_call and _calls(stop_after_call):
            await asyncio.sleep(0.5)  # let the result flush back
            break
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), 0.25)
    run.exited = proc.returncode is not None
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()
    await asyncio.sleep(1.0)
    run.leftover = _pids_naming(token)
    for pid in run.leftover:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    run.rc = proc.returncode
    run.stderr = Path(out_path + ".err").read_text(errors="replace")
    for line in Path(out_path).read_text(errors="replace").splitlines():
        try:
            event = json.loads(line).get("event")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(event, dict) and event and "ephemeral" not in event:
            run.events.append(event)
    return run


def _relay_servers(socket_path: str) -> str:
    return json.dumps(
        [
            {
                "type": "stdio",
                "name": "af",
                "command": _SYSTEM_PYTHON,
                "args": [str(_RELAY), socket_path],
            }
        ]
    )


def _socket_servers(socket_path: str) -> str:
    return json.dumps([{"type": "socket", "name": "af", "socketPath": socket_path}])


def _scripted_argv(
    fixture: str, servers: str, session: list[str], prompt: str
) -> list[str]:
    return (
        _DM_BASE
        + ["--agent-harness", "native", "--scripted-model", fixture]
        + ["--mcp-servers", servers]
        + _APPEND
        + session
        + [prompt]
    )


def _calls(text: str) -> list[dict]:
    return [c for c in CALLED if c["arguments"].get("text") == text]


# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------


async def _direct_socket(c: Checks, mcp_server: SocketMcpServer, work: str) -> None:
    section("A: dm's direct socket transport (scripted)")
    nonce = f"DMSOCK-{uuid.uuid4().hex[:6]}"
    fixture = _turn1_fixture(work, nonce)
    argv = _scripted_argv(
        fixture, _socket_servers(mcp_server.socket_path), [], "Echo the nonce."
    )
    run = await _run_dm(argv, work, token=fixture, stop_after_call=nonce)
    c.check(
        "A socket transport: the scripted mcp__af__echo call reached the socket server",
        len(_calls(nonce)) == 1,
        f"calls={_calls(nonce)} exited_by_itself={run.exited} rc={run.rc}",
    )
    c.info("A processes left after dm was stopped (killed)", run.leftover)


def _schema_checks(c: Checks, run: DmRun, nonce: str) -> None:
    """The events and fields ``session/devmate_dm.py`` maps."""
    tool_uses = run.actions("tool_use_action")
    use = (
        (tool_uses[0].get("variant") or {}).get("tool_use_action", {})
        if tool_uses
        else {}
    )
    ends = run.of("session_end")
    exit_code = (
        str((ends[-1].get("session") or {}).get("exit_code", "")) if ends else ""
    )
    c.info("B1 event kinds", run.kinds())
    c.check(
        "B1 schema: session_start/step_start/step_end/session_end present",
        all(
            k in run.kinds()
            for k in ("session_start", "step_start", "step_end", "session_end")
        ),
        run.kinds(),
    )
    c.check(
        "B1 schema: action_end llm_action carries the text in output.info",
        "Calling echo." in run.llm_texts() and "TURN1-DONE" in run.llm_texts(),
        run.llm_texts(),
    )
    c.check(
        "B1 schema: action_end tool_use_action carries tool_use_id and tool_name",
        bool(use.get("tool_use_id")) and use.get("tool_name") == "mcp__af__echo",
        use,
    )
    c.check(
        "B1 schema: session_end carries a successful exit_code",
        exit_code in _OK_EXITS,
        exit_code,
    )
    c.check(
        "B1 the relay delivered the echo call to the socket server",
        len(_calls(nonce)) == 1,
        _calls(nonce),
    )


async def _resume_continuity(c: Checks, mcp_server: SocketMcpServer, work: str) -> None:
    servers = _relay_servers(mcp_server.socket_path)
    pinned = str(uuid.uuid4())
    t1, t2 = f"DMT1-{uuid.uuid4().hex[:6]}", f"DMRESUMED-{uuid.uuid4().hex[:6]}"

    section("B1: turn 1, pinned --session-id, stdio relay (production path)")
    fixture = _turn1_fixture(work, t1)
    run = await _run_dm(
        _scripted_argv(fixture, servers, ["--session-id", pinned], "Echo T1."),
        work,
        token=fixture,
    )
    c.info("B1 processes left after dm exited (killed)", run.leftover)
    ids = run.session_ids()
    c.check(
        "B1 dm finished the turn by itself (rc 0)",
        run.exited and run.rc == 0,
        f"exited={run.exited} rc={run.rc} {run.stderr.strip()[-200:]!r}",
    )
    c.check(
        "B1 every session event carries the pinned id",
        bool(ids) and set(ids) == {pinned},
        f"pinned={short(pinned)} seen={sorted({short(i) for i in ids})}",
    )
    _schema_checks(c, run, t1)

    section("B2: turn 2, --resume of the pinned id")
    fixture = _turn2_fixture(work, t2)
    run = await _run_dm(
        _scripted_argv(fixture, servers, ["--resume", pinned], "Echo T2."),
        work,
        token=fixture,
    )
    ids = run.session_ids()
    c.check(
        "B2 dm finished the resumed turn by itself (rc 0)",
        run.exited and run.rc == 0,
        f"exited={run.exited} rc={run.rc}",
    )
    c.check(
        "B2 every session event carries the same id (no replacement session)",
        bool(ids) and set(ids) == {pinned},
        f"pinned={short(pinned)} seen={sorted({short(i) for i in ids})}",
    )
    c.check(
        "B2 the resumed request held turn 1's tool result (history carried)",
        len(_calls(t2)) == 1
        and "HISTORY-OK" in run.llm_texts()
        and "NO-HISTORY" not in run.llm_texts(),
        f"echo({t2}) calls={len(_calls(t2))} texts={run.llm_texts()}",
    )

    section("B3: control — the turn-2 fixture in a fresh session")
    control = f"DMCTRL-{uuid.uuid4().hex[:6]}"
    fixture = _turn2_fixture(work, control)
    run = await _run_dm(
        _scripted_argv(
            fixture, servers, ["--session-id", str(uuid.uuid4())], "Echo T2."
        ),
        work,
        token=fixture,
    )
    c.check(
        "B3 a fresh session has no history: NO-HISTORY, no echo call",
        "NO-HISTORY" in run.llm_texts() and not _calls(control),
        f"texts={run.llm_texts()} calls={_calls(control)} rc={run.rc}",
    )


async def _real_model(
    c: Checks, mcp_server: SocketMcpServer, work: str, model: str
) -> None:
    relay = os.environ.get("S14_STDIO_RELAY") == "1"
    section(f"Real model {model} ({'stdio relay' if relay else 'direct socket'})")
    nonce = f"DMREAL-{uuid.uuid4().hex[:6]}"
    if relay:
        servers = _relay_servers(mcp_server.socket_path)
        prompt = f"Call the echo tool (server af) with text {nonce}, then reply with its result."
    else:
        servers = _socket_servers(mcp_server.socket_path)
        prompt = f"Reply with exactly this token and nothing else: {nonce}"
    argv = _DM_BASE + ["--model", model, "--mcp-servers", servers] + _APPEND + [prompt]
    run = await _run_dm(argv, work, token=nonce)
    texts = " ".join(run.llm_texts())
    if relay:
        c.check(
            "real model over the stdio relay calls the tool",
            len(_calls(nonce)) == 1 and run.exited,
            f"calls={_calls(nonce)} exited={run.exited} rc={run.rc}",
        )
    else:
        c.info(
            "real model over the direct socket",
            f"finished={run.exited} rc={run.rc} reply_has_token={nonce in texts}",
        )


async def main() -> int:
    c = Checks("s14")
    work = tempfile.mkdtemp(prefix="s14_dm_")  # outside fbsource: dm's master build
    version = await asyncio.create_subprocess_exec(
        "dm",
        "--version",
        cwd=work,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await version.communicate()
    c.info("dm version", out.decode(errors="replace").strip())
    mcp_server = SocketMcpServer()
    await mcp_server.start()
    try:
        real = os.environ.get("S14_REAL_MODEL")
        if real:
            await _real_model(c, mcp_server, work, real)
        else:
            await _direct_socket(c, mcp_server, work)
            await _resume_continuity(c, mcp_server, work)
    finally:
        relays = _pids_naming(mcp_server.socket_path)
        for pid in relays:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        c.info("relay processes left at the end (killed)", relays)
        await mcp_server.stop()
        shutil.rmtree(work, ignore_errors=True)
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

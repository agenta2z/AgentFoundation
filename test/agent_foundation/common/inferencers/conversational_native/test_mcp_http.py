"""LocalMcpHttpServer: per-conversation bearer routing, the hook endpoint,
revocation, DNS-rebinding protection, host-owned signals, refusal outside a
vendor turn, bridge-side argument validation, and one server per event loop.

The per-turn CLI backends on the real local MCP servers (HTTP; unix socket for
dm): the session arguments each vendor process receives per turn (first,
resumed, after a rotation, after a restart), the permissions of every file a
session writes, and the MCP token's routes (never argv). A stand-in vendor CLI
records its argv and environment."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
import signal
import stat
import sys
import tempfile
import threading
import uuid
from unittest import mock

import httpx
import uvicorn
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge import (
    af_hook,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.mcp_http import (
    LocalMcpHttpServer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    BridgeResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BridgeToolSpec,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.runtime import (
    NativeRuntimeManager,
)
from agent_foundation.resources.tools.registry import load_all_tools
from later.unittest import TestCase

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _tool(name: str, calls: list) -> BridgeToolSpec:
    async def handler(args: dict) -> BridgeResult:
        calls.append((name, args))
        return BridgeResult(text=f"{name} ran with {json.dumps(args, sort_keys=True)}")

    return BridgeToolSpec(
        name=name,
        description=f"The {name} tool.",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        handler=handler,
    )


def _rpc(method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


def _failing_tool(name: str) -> BridgeToolSpec:
    async def handler(args: dict) -> BridgeResult:
        return BridgeResult(text=f"{name} failed", is_error=True)

    return BridgeToolSpec(
        name=name,
        description="Always fails.",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


class LocalMcpHttpServerTest(TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.server = LocalMcpHttpServer()
        self.calls: list = []
        self.hooks: list = []

        async def hook(payload: dict) -> dict:
            self.hooks.append(payload)
            return {"continue": False}

        self.url, self.token, self.headers = await self.server.register(
            [_tool("alpha", self.calls)], hook_handler=hook
        )
        # Loopback only: ignore any proxy configured in the environment.
        self.client = httpx.AsyncClient(timeout=10, trust_env=False)

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        await self.server.stop()
        await super().asyncTearDown()

    async def _mcp(self, body: dict, headers: dict | None = None) -> httpx.Response:
        return await self.client.post(
            self.url, json=body, headers={**_MCP_HEADERS, **(headers or {})}
        )

    async def test_url_carries_no_secret(self) -> None:
        self.assertNotIn(self.token, self.url)
        self.assertEqual(self.headers, {"Authorization": f"Bearer {self.token}"})

    async def test_tools_are_listed_and_called_with_the_conversation_token(
        self,
    ) -> None:
        listed = await self._mcp(_rpc("tools/list"), self.headers)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(
            [t["name"] for t in listed.json()["result"]["tools"]], ["alpha"]
        )
        called = await self._mcp(
            _rpc("tools/call", {"name": "alpha", "arguments": {"x": 3}}), self.headers
        )
        self.assertEqual(called.status_code, 200)
        content = called.json()["result"]["content"]
        self.assertEqual(content[0]["text"], 'alpha ran with {"x": 3}')
        self.assertEqual(self.calls, [("alpha", {"x": 3})])
        # The request's server task ends with the conversation.
        await self.server.unregister(self.token)

    async def test_missing_or_wrong_token_is_refused(self) -> None:
        for headers in (
            {},
            {"Authorization": "Bearer nope"},
            {"Authorization": self.token},
        ):
            with self.subTest(headers=headers):
                response = await self._mcp(_rpc("tools/list"), headers)
                self.assertEqual(response.status_code, 401)
        self.assertEqual(self.calls, [])

    async def test_conversations_are_isolated_by_token(self) -> None:
        other_calls: list = []
        _url, other_token, other_headers = await self.server.register(
            [_tool("beta", other_calls)]
        )
        listed = await self._mcp(_rpc("tools/list"), other_headers)
        self.assertEqual(
            [t["name"] for t in listed.json()["result"]["tools"]], ["beta"]
        )
        await self.server.unregister(other_token)

    async def test_tools_declare_the_result_size_budget_to_claude_code(self) -> None:
        _url, token, headers = await self.server.register(
            [_tool("beta", []), _tool("gamma", [])], result_max_chars=120_000
        )
        listed = await self._mcp(_rpc("tools/list"), headers)
        self.assertEqual(
            [t["_meta"] for t in listed.json()["result"]["tools"]],
            [{"anthropic/maxResultSizeChars": 120_000}] * 2,
        )
        await self.server.unregister(token)
        # Without a budget no limit is declared.
        listed = await self._mcp(_rpc("tools/list"), self.headers)
        self.assertNotIn("_meta", listed.json()["result"]["tools"][0])
        # The request's server task ends with the conversation.
        await self.server.unregister(self.token)

    async def test_hook_endpoint_routes_to_the_conversation(self) -> None:
        hook_url = self.url.rsplit("/", 1)[0] + "/hook"
        response = await self.client.post(
            hook_url, json={"hook_event_name": "PostToolUse"}, headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"continue": False})
        self.assertEqual(self.hooks, [{"hook_event_name": "PostToolUse"}])
        refused = await self.client.post(
            hook_url, json={}, headers={"Authorization": "Bearer x"}
        )
        self.assertEqual(refused.status_code, 401)

    async def _relay(self, payload: dict, token: str) -> dict:
        """``bridge/af_hook.py`` against this server, as Claude Code runs it."""
        env = {"AF_HOOK_URL": f"{self.server.base_url}/hook", "AF_HOOK_TOKEN": token}
        stdin = io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()))
        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(sys, "stdin", stdin),
            mock.patch.object(sys, "stdout", stdout),
            mock.patch.object(sys, "stderr", io.StringIO()),
        ):
            # Blocking client: off the loop that serves the request.
            await asyncio.to_thread(af_hook.main)
        return json.loads(stdout.getvalue()) if stdout.getvalue() else {}

    async def test_relay_prints_the_hosts_decision(self) -> None:
        out = await self._relay(
            {"hook_event_name": "PostToolUse", "tool_name": "mcp__af__x"}, self.token
        )
        self.assertEqual(out, {"continue": False})

    async def test_failing_hook_handler_makes_the_relay_fail_closed(self) -> None:
        async def broken(payload: dict) -> dict:
            raise RuntimeError("host bug")

        _url, token, headers = await self.server.register([], hook_handler=broken)
        response = await self.client.post(
            f"{self.server.base_url}/hook",
            json={"hook_event_name": "PreToolUse"},
            headers=headers,
        )
        self.assertEqual(response.status_code, 500)
        out = await self._relay(
            {"hook_event_name": "PreToolUse", "tool_name": "mcp__af__task"}, token
        )
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")
        await self.server.unregister(token)

    async def test_revoked_token_makes_the_relay_fail_closed(self) -> None:
        await self.server.unregister(self.token)
        out = await self._relay(
            {"hook_event_name": "PostToolUse", "tool_name": "mcp__af__x"}, self.token
        )
        self.assertIs(out["continue"], False)
        self.assertIn("did not answer", out["stopReason"])

    async def test_unregister_revokes_the_token(self) -> None:
        await self.server.unregister(self.token)
        response = await self._mcp(_rpc("tools/list"), self.headers)
        self.assertEqual(response.status_code, 401)

    async def test_a_session_registered_by_one_task_is_closed_by_another(self) -> None:
        # Backends register on their session's task; shutdown closes from another.
        _url, token, _headers = await asyncio.ensure_future(
            self.server.register([_tool("gamma", [])])
        )
        with self.assertNoLogs(
            "agent_foundation.common.inferencers.agentic_inferencers."
            "conversational_native.bridge.mcp_http",
            level="WARNING",
        ):
            await self.server.unregister(token)

    async def test_foreign_host_header_is_rejected(self) -> None:
        response = await self._mcp(
            _rpc("tools/list"), {**self.headers, "Host": "attacker.example:80"}
        )
        self.assertNotEqual(response.status_code, 200)
        self.assertEqual(self.calls, [])
        # The rejected request's server task ends with the conversation.
        await self.server.unregister(self.token)

    async def test_arguments_reach_the_bridge_unvalidated(self) -> None:
        # The bridge validates against the session's schema; the transport
        # must not reject (or rewrite) arguments first.
        called = await self._mcp(
            _rpc("tools/call", {"name": "alpha", "arguments": {"x": "seven"}}),
            self.headers,
        )
        result = called.json()["result"]
        self.assertFalse(result["isError"])
        self.assertEqual(self.calls, [("alpha", {"x": "seven"})])
        # The request's server task ends with the conversation.
        await self.server.unregister(self.token)

    async def test_tool_errors_are_flagged_to_the_vendor(self) -> None:
        _url, token, headers = await self.server.register([_failing_tool("beta")])
        failed = await self._mcp(
            _rpc("tools/call", {"name": "beta", "arguments": {}}), headers
        )
        self.assertTrue(failed.json()["result"]["isError"])
        unknown = await self._mcp(
            _rpc("tools/call", {"name": "nope", "arguments": {}}), headers
        )
        self.assertTrue(unknown.json()["result"]["isError"])
        await self.server.unregister(token)

    async def test_a_valid_token_is_refused_outside_the_vendor_turn(self) -> None:
        in_turn = False
        calls: list = []

        async def hook(payload: dict) -> dict:
            return {}

        _url, token, headers = await self.server.register(
            [_tool("gamma", calls)], hook_handler=hook, turn_active=lambda: in_turn
        )
        hook_url = f"{self.server.base_url}/hook"
        call = _rpc("tools/call", {"name": "gamma", "arguments": {"x": 1}})
        refused = await self._mcp(call, headers)
        self.assertEqual(refused.status_code, 403)
        hook_refused = await self.client.post(hook_url, json={}, headers=headers)
        self.assertEqual(hook_refused.status_code, 403)
        self.assertEqual(calls, [])
        in_turn = True
        accepted = await self._mcp(call, headers)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(calls, [("gamma", {"x": 1})])
        hook_ok = await self.client.post(hook_url, json={}, headers=headers)
        self.assertEqual(hook_ok.status_code, 200)
        await self.server.unregister(token)

    async def test_running_server_leaves_signal_handling_to_the_host(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            owner = getattr(signal.getsignal(sig), "__self__", None)
            self.assertNotIsInstance(owner, uvicorn.Server)


class _LoopThread:
    """An event loop running in its own thread (a second host loop)."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    async def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return await asyncio.wrap_future(future)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        self.loop.close()


class PerLoopServerTest(TestCase):
    async def test_each_event_loop_gets_its_own_server(self) -> None:
        manager = NativeRuntimeManager()
        here = await manager.ensure_http_server()
        self.assertIs(await manager.ensure_http_server(), here)
        self.assertIs(manager.http_server, here)
        other = _LoopThread()
        try:

            async def start_there():
                server = await manager.ensure_http_server()
                await server.register([_tool("delta", [])])
                return server

            there = await other.run(start_there())
            self.assertIsNot(there, here)
            self.assertTrue(there.running)
            # Shutdown from this loop also stops the other loop's server.
            await manager.aclose_all()
            self.assertFalse(there.running)
            self.assertIsNone(manager.http_server)
        finally:
            other.close()

    async def test_a_server_whose_loop_closed_is_released(self) -> None:
        manager = NativeRuntimeManager()

        async def start() -> LocalMcpHttpServer:
            server = await manager.ensure_http_server()
            await server.register([_tool("epsilon", [])])
            return server

        holder: list = []
        thread = threading.Thread(target=lambda: holder.append(asyncio.run(start())))
        thread.start()
        thread.join(timeout=30)
        (dead,) = holder
        self.assertIsNotNone(dead._sock)
        here = await manager.ensure_http_server()
        self.assertIsNot(here, dead)
        self.assertIsNone(dead._sock)  # the closed loop's socket was released
        await manager.aclose_all()


_RECORDING_CLI = """#!/bin/sh
d={directory}
n=$(( $(cat "$d/count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$d/count"
printf '%s\\0' "$0" "$@" > "$d/argv$n"
cat /proc/$$/environ > "$d/env$n"
sid=minted-$n
prev=
for arg; do
  case "$prev" in --session-id|--resume|resume) sid=$arg ;; esac
  prev=$arg
done
sed "s/@SID@/$sid/g" "$d/out"
"""

# One successful vendor turn per backend; @SID@ is the session the process
# was given (``--session-id`` / ``--resume`` / ``exec resume``), else a new one.
_TURN_OUTPUT = {
    "claude_cli": [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "@SID@",
            "mcp_servers": [{"name": "af", "status": "connected"}],
            "tools": ["Read", "mcp__af__clarification"],
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": 1,
            "session_id": "@SID@",
            "result": "ok",
        },
    ],
    "codex_cli": [
        {"type": "thread.started", "thread_id": "@SID@"},
        {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "ok"},
        },
        {"type": "turn.completed", "usage": {}},
    ],
    "devmate_dm": [
        {"event": {"session_start": {"session": {"id": "@SID@"}}}},
        {
            "event": {
                "session_end": {"session": {"id": "@SID@", "exit_code": "COMPLETE"}}
            }
        },
    ],
}


class _RecordingCli:
    """A stand-in vendor CLI: invocation ``n`` records its argv and its
    environment, then prints one turn's output for the session it was given."""

    def __init__(self, kind: str) -> None:
        self.directory = tempfile.mkdtemp(prefix="recording_cli_")
        with open(os.path.join(self.directory, "out"), "w") as f:
            f.write("".join(json.dumps(line) + "\n" for line in _TURN_OUTPUT[kind]))
        self.path = os.path.join(self.directory, "cli")
        with open(self.path, "w") as f:
            f.write(_RECORDING_CLI.format(directory=shlex.quote(self.directory)))
        os.chmod(self.path, 0o700)

    def _fields(self, name: str) -> list[str]:
        with open(os.path.join(self.directory, name), "rb") as f:
            return f.read().decode().split("\0")[:-1]

    def argv(self, n: int) -> list[str]:
        return self._fields(f"argv{n}")

    def env(self, n: int) -> dict[str, str]:
        return dict(item.split("=", 1) for item in self._fields(f"env{n}"))


def _native(
    kind: str, cli: _RecordingCli, cwd: str, session_dir: str, store: object
) -> NativeConversationalInferencer:
    tools = {k: v for k, v in load_all_tools().items() if v.tool_type == "Conversation"}
    return NativeConversationalInferencer(
        backend={
            "kind": kind,
            "cwd": cwd,
            "cli_path": cli.path,
            "l2_envelope_allowed": True,
        },
        tool_registry=tools,
        prior_context={"native_session_dir": session_dir},
        record_store=store,
        conversation_key="conv-cli",
    )


def _session_arg(argv: list[str]) -> tuple[str, str]:
    """The session a vendor process was told to start or continue."""
    for flag in ("--session-id", "--resume"):
        if flag in argv:
            return flag, argv[argv.index(flag) + 1]
    if argv[1:3] == ["exec", "resume"]:
        return "exec resume", argv[3]
    return "new", ""


class PerTurnCliArgvTest(TestCase):
    async def _conversation(self, kind: str) -> tuple[_RecordingCli, list[str]]:
        """Two turns, ``/new``, two turns, then a host restart (a new
        inferencer and runtime on the same record) and one more turn; returns
        the CLI and the recorded vendor session id after each turn."""
        cli = _RecordingCli(kind)
        cwd, session_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
        store = InMemoryRecordStore()
        recorded = []
        native = _native(kind, cli, cwd, session_dir, store)
        try:
            for turn, text in enumerate(["one", "two", "/new", "three", "four"], 1):
                await native.run_agentic_loop(text, turn_number=turn)
                if not text.startswith("/"):
                    recorded.append(store.load("conv-cli").vendor_session_id)
        finally:
            await native.aclose()
        restarted = _native(kind, cli, cwd, session_dir, store)
        try:
            await restarted.run_agentic_loop("five", turn_number=6)
            recorded.append(store.load("conv-cli").vendor_session_id)
        finally:
            await restarted.aclose()
        return cli, recorded

    async def test_pinned_session_first_resume_rotate_and_restart(self) -> None:
        for kind in ("claude_cli", "devmate_dm"):
            with self.subTest(kind):
                cli, recorded = await self._conversation(kind)
                args = [_session_arg(cli.argv(n)) for n in range(1, 6)]
                first, rotated = args[0][1], args[2][1]
                self.assertNotEqual(uuid.UUID(first), uuid.UUID(rotated))
                self.assertEqual(
                    args,
                    [
                        ("--session-id", first),
                        ("--resume", first),
                        ("--session-id", rotated),
                        ("--resume", rotated),
                        ("--resume", rotated),
                    ],
                )
                for n in range(1, 6):
                    argv = cli.argv(n)
                    self.assertFalse({"--session-id", "--resume"} <= set(argv))
                self.assertEqual(recorded, [first, first, rotated, rotated, rotated])

    async def test_recorded_thread_first_resume_rotate_and_restart(self) -> None:
        cli, recorded = await self._conversation("codex_cli")
        self.assertEqual(
            [_session_arg(cli.argv(n)) for n in range(1, 6)],
            [
                ("new", ""),
                ("exec resume", "minted-1"),
                ("new", ""),
                ("exec resume", "minted-3"),
                ("exec resume", "minted-3"),
            ],
        )
        self.assertEqual(
            recorded, ["minted-1", "minted-1", "minted-3", "minted-3", "minted-3"]
        )

    async def test_a_rotated_claude_session_reads_its_own_instructions_file(
        self,
    ) -> None:
        cli, _ = await self._conversation("claude_cli")
        l1 = [
            os.path.basename(argv[argv.index("--append-system-prompt-file") + 1])
            for argv in (cli.argv(n) for n in range(1, 6))
        ]
        self.assertEqual(l1, ["l1_0.md", "l1_0.md", "l1_1.md", "l1_1.md", "l1_1.md"])


def _modes(root: str) -> dict[str, int]:
    """Permission bits of ``root`` and everything under it, by relative path."""
    modes = {".": stat.S_IMODE(os.stat(root).st_mode)}
    for base, dirs, files in os.walk(root):
        for name in dirs + files:
            path = os.path.join(base, name)
            modes[os.path.relpath(path, root)] = stat.S_IMODE(os.stat(path).st_mode)
    return modes


class PrivateFilesTest(TestCase):
    """Every file a session writes for its vendor is private (0600 in 0700
    directories), and the conversation's MCP token reaches the vendor only
    through such a file or the vendor's environment — never argv, which any
    local user can read."""

    async def _one_turn(self, kind: str) -> dict:
        cli = _RecordingCli(kind)
        session_dir = os.path.join(tempfile.mkdtemp(), "session")
        native = _native(
            kind, cli, tempfile.mkdtemp(), session_dir, InMemoryRecordStore()
        )
        try:
            await native.run_agentic_loop("hello", turn_number=1)
            actor = native.runtime_manager.live_actor(("conv-cli", kind, 0))
            argv = cli.argv(1)
            seen = {
                "argv": argv,
                "env": cli.env(1),
                "token": actor.backend._mcp_token,
                "files": _modes(session_dir),
                "session_dir": session_dir,
            }
            if "--mcp-servers" in argv:
                af = json.loads(argv[argv.index("--mcp-servers") + 1])[0]
                socket_path = af["args"][1]
                seen["socket"] = stat.S_IMODE(os.stat(socket_path).st_mode)
                seen["socket_dir"] = _modes(os.path.dirname(socket_path))["."]
        finally:
            await native.aclose()
        return seen

    async def test_session_files_are_private(self) -> None:
        expected = {
            "claude_cli": {"l1_0.md", "settings.json", "mcp_config.json"},
            "codex_cli": {"l1_0.md"},
            "devmate_dm": {"l1_0.md"},
        }
        for kind, files in expected.items():
            with self.subTest(kind):
                seen = await self._one_turn(kind)
                modes = seen["files"]
                self.assertEqual(modes.pop("."), 0o700)
                self.assertEqual(set(modes), files)
                self.assertEqual(set(modes.values()), {0o600})

    async def test_the_mcp_token_never_reaches_argv(self) -> None:
        for kind in _TURN_OUTPUT:
            with self.subTest(kind):
                seen = await self._one_turn(kind)
                token = seen["token"]
                self.assertTrue(token)
                self.assertNotIn(token, "\0".join(seen["argv"]))
                carried_by = {k for k, v in seen["env"].items() if token in v}
                if kind == "claude_cli":
                    self.assertEqual(carried_by, {"AF_HOOK_TOKEN"})
                    path = os.path.join(seen["session_dir"], "mcp_config.json")
                    with open(path) as f:
                        af = json.load(f)["mcpServers"]["af"]
                    self.assertEqual(af["headers"]["Authorization"], f"Bearer {token}")
                    self.assertNotIn(token, af["url"])
                elif kind == "codex_cli":
                    self.assertEqual(carried_by, {"AF_MCP_TOKEN"})
                    self.assertIn(
                        'mcp_servers.af.bearer_token_env_var="AF_MCP_TOKEN"',
                        seen["argv"],
                    )
                else:
                    # The unix socket's permissions are dm's credential.
                    self.assertEqual(carried_by, set())
                    self.assertEqual(seen["socket"], 0o600)
                    self.assertEqual(seen["socket_dir"], 0o700)

    async def test_claude_reads_its_instructions_from_the_private_file(self) -> None:
        seen = await self._one_turn("claude_cli")
        argv = seen["argv"]
        path = argv[argv.index("--append-system-prompt-file") + 1]
        self.assertEqual(path, os.path.join(seen["session_dir"], "l1_0.md"))
        with open(path) as f:
            body = f.read()
        self.assertGreater(len(body), 200)
        self.assertNotIn(body[:200], "\0".join(argv))

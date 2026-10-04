"""ClaudeCliBackend argv, command-hook relay, stream-json mapping and turn
classification (no ``claude`` subprocess: a scripted stand-in is used).

Real end-to-end coverage (HTTP MCP tool calls, deferred widgets, SOP) is in
scripts/native_spikes/e2e_research_sop.py --backend claude_cli; hook semantics
(S2/S4/S12, hermetic) in scripts/native_spikes/s2_s4_s12_claude_cli_hooks.py.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import stat
import sys
import tempfile
import unittest
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge import (
    af_hook,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    Compaction,
    MessageEnd,
    SessionStarted,
    TextDelta,
    TurnEnd,
    VendorError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session import (
    claude_cli as claude_cli_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    L2Channel,
    NativeBackendSpec,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.claude_cli import (
    ClaudeCliBackend,
)
from fakes import scripted_cli
from later.unittest import TestCase as AsyncTestCase
from test_cli_runner import (
    agent_printing,
    cancel_through_actor,
    launcher_with_agent,
    pid_alive,
    read_pid,
)


def _backend(**spec) -> ClaudeCliBackend:
    b = ClaudeCliBackend(NativeBackendSpec(cwd="/work", model="sonnet", **spec))
    b._l1_path = "/tmp/l1.md"
    return b


class _Hooks:
    def __init__(self, l2="", deny=None, stop=False) -> None:
        self.l2, self.deny, self.stop = l2, deny, stop
        self.compacted = False
        self.before_calls: list = []

    def l2_for_turn(self):
        return self.l2

    async def before_af_tool(self, name, tool_use_id, agent_id):
        self.before_calls.append((name, tool_use_id, agent_id))
        return self.deny if agent_id else None

    async def after_af_tool(self, name, tool_use_id):
        return self.stop

    def on_compaction(self):
        self.compacted = True


class _AfterHooks(_Hooks):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.after_calls: list = []

    async def after_af_tool(self, name, tool_use_id):
        self.after_calls.append((name, tool_use_id))
        return self.stop


class CapabilitiesTest(unittest.TestCase):
    def test_hook_first_l2_and_hook_stop(self) -> None:
        caps = ClaudeCliBackend.capabilities
        self.assertEqual(
            caps.preferred_l2_channel(envelope_allowed=False), L2Channel.HOOK
        )
        self.assertTrue(caps.turn_stop_hook)
        self.assertTrue(caps.subagent_attribution)
        self.assertTrue(caps.exact_fork)  # session files fork like the SDK's (S13)


class ArgvTest(unittest.TestCase):
    def test_fresh_turn_pins_session_and_disallows_ask_and_plan(self) -> None:
        b = _backend()
        b._session_id = "pin-1"
        b._mcp_config_path = "/tmp/mcp.json"
        b._settings_path = "/tmp/settings.json"
        argv = b._argv(TurnRequest(text="hello", channel=L2Channel.HOOK))
        self.assertIn("-p", argv)
        self.assertEqual(argv[argv.index("--output-format") + 1], "stream-json")
        self.assertIn("--include-partial-messages", argv)
        self.assertEqual(
            argv[argv.index("--append-system-prompt-file") + 1], "/tmp/l1.md"
        )
        self.assertEqual(argv[argv.index("--session-id") + 1], "pin-1")
        self.assertEqual(argv[argv.index("--settings") + 1], "/tmp/settings.json")
        self.assertEqual(argv[argv.index("--allowedTools") + 1], "mcp__af")
        self.assertNotIn("--resume", argv)
        disallowed = argv[argv.index("--disallowedTools") + 1].split(",")
        self.assertEqual(
            set(disallowed), {"AskUserQuestion", "EnterPlanMode", "ExitPlanMode"}
        )
        self.assertEqual(argv[-1], "hello")

    def test_inherit_keeps_the_users_mcp_servers(self) -> None:
        b = _backend()
        b._session_id = "x"
        b._mcp_config_path = "/tmp/mcp.json"
        argv = b._argv(TurnRequest(text="h"))
        self.assertNotIn("--strict-mcp-config", argv)
        self.assertNotIn("--setting-sources", argv)

    def test_hermetic_limits_settings_and_servers(self) -> None:
        b = _backend(environment="hermetic")
        b._session_id = "x"
        argv = b._argv(TurnRequest(text="h"))
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
        self.assertIn("--strict-mcp-config", argv)

    def test_resume_after_first_turn(self) -> None:
        b = _backend()
        b._session_id = "s-1"
        b._started_emitted = True
        argv = b._argv(TurnRequest(text="again"))
        self.assertEqual(argv[argv.index("--resume") + 1], "s-1")
        self.assertNotIn("--session-id", argv)

    def test_the_claude_binary_is_the_configured_one_else_found_on_path(
        self,
    ) -> None:
        request = TurnRequest(text="h")
        with mock.patch("shutil.which", return_value="/path/bin/claude") as which:
            self.assertEqual(_backend()._argv(request)[0], "/path/bin/claude")
            which.assert_called_with("claude")
            pinned = _backend(cli_path="/opt/claude")
            self.assertEqual(pinned._argv(request)[0], "/opt/claude")
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(_backend()._argv(request)[0], "claude")

    def test_effort_and_model_tag(self) -> None:
        b = ClaudeCliBackend(
            NativeBackendSpec(cwd="/w", model="claude-opus-4.8", effort="xhigh")
        )
        b._l1_path = "/tmp/l1.md"
        argv = b._argv(TurnRequest(text="h"))
        self.assertEqual(argv[argv.index("--effort") + 1], "xhigh")
        self.assertEqual(argv[argv.index("--model") + 1], "claude-opus-4-8")


class _FakeServer:
    base_url = "http://127.0.0.1:1"

    def __init__(self) -> None:
        self.registered: list = []
        self.unregistered: list = []

    async def register(
        self, tools, *, hook_handler=None, turn_active=None, result_max_chars=0
    ):
        self.registered.append((tools, hook_handler))
        self.turn_active = turn_active
        self.result_max_chars = result_max_chars
        return f"{self.base_url}/mcp", "tok", {"Authorization": "Bearer tok"}

    async def unregister(self, token):
        self.unregistered.append(token)


class _Runtime:
    def __init__(self) -> None:
        self.http_server = _FakeServer()

    async def ensure_http_server(self):
        return self.http_server


def _open_request(tmp: str, *, resume=False, tools=()):
    return SessionOpenRequest(
        session_id="sess-9",
        resume=resume,
        l1_text="L1",
        l1_path=os.path.join(tmp, "l1_0.md"),
        tools=list(tools),
        hooks=_Hooks(),
        cwd="/w",
        model="sonnet",
    )


class OpenTest(AsyncTestCase):
    async def test_resume_continues_the_session_after_a_restart(self) -> None:
        tmp = tempfile.mkdtemp()
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd="/w"), runtime_manager=_Runtime()
        )
        await backend.open(_open_request(tmp, resume=True, tools=["t"]))
        argv = backend._argv(TurnRequest(text="hi"))
        self.assertEqual(argv[argv.index("--resume") + 1], "sess-9")
        self.assertNotIn("--session-id", argv)

    async def test_a_model_change_reaches_the_next_process_of_the_session(
        self,
    ) -> None:
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd="/w", model="sonnet"), runtime_manager=_Runtime()
        )
        await backend.open(_open_request(tempfile.mkdtemp(), resume=True, tools=["t"]))
        await backend.set_model("claude-opus-4.8")
        argv = backend._argv(TurnRequest(text="hi"))
        self.assertEqual(argv[argv.index("--model") + 1], "claude-opus-4-8")
        self.assertEqual(argv[argv.index("--resume") + 1], "sess-9")

    async def test_the_tool_server_accepts_the_session_only_during_a_turn(
        self,
    ) -> None:
        runtime = _Runtime()
        cli = _scripted_cli(
            [{"type": "assistant", "message": {"id": "m", "content": []}}], "", 0
        )
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=cli),
            runtime_manager=runtime,
        )
        await backend.open(_open_request(tempfile.mkdtemp(), tools=["t"]))
        turn_active = runtime.http_server.turn_active
        self.assertFalse(turn_active())
        during = [turn_active() async for _ in backend.run_turn(TurnRequest(text="hi"))]
        self.assertTrue(during and all(during))
        self.assertFalse(turn_active())
        await backend.close()
        self.assertEqual(runtime.http_server.unregistered, ["tok"])

    async def test_the_tools_declare_the_bridges_result_size_budget(self) -> None:
        # Claude Code spills an undeclared MCP tool's result above 50,000
        # characters (or its MCP output token limit), hiding a trailing L3.
        runtime = _Runtime()
        backend = ClaudeCliBackend(NativeBackendSpec(cwd="/w"), runtime_manager=runtime)
        request = _open_request(tempfile.mkdtemp(), tools=["t"])
        request.result_max_chars = 120_000
        await backend.open(request)
        self.assertEqual(runtime.http_server.result_max_chars, 120_000)

    async def test_af_health_is_checked_only_when_af_is_attached(self) -> None:
        for tools, expected in ((["t"], True), ([], False)):
            with self.subTest(tools=tools):
                backend = ClaudeCliBackend(
                    NativeBackendSpec(cwd="/w"), runtime_manager=_Runtime()
                )
                await backend.open(_open_request(tempfile.mkdtemp(), tools=tools))
                self.assertIs(backend._expects_af, expected)

    async def test_a_fork_resumes_the_forked_transcript(self) -> None:
        tmp = tempfile.mkdtemp()
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd="/w"), runtime_manager=_Runtime()
        )
        request = _open_request(tmp, tools=["t"])
        request.fork_from = ("sess-9", "m-uuid")
        forks = []

        async def fork(source, boundary):
            forks.append((source, boundary))
            return "forked-id"

        with mock.patch.object(claude_cli_module, "fork_claude_session", fork):
            await backend.open(request)
        self.assertEqual(forks, [("sess-9", "m-uuid")])
        argv = backend._argv(TurnRequest(text="hi"))
        self.assertEqual(argv[argv.index("--resume") + 1], "forked-id")
        self.assertNotIn("--session-id", argv)
        await backend.close()

    async def test_private_files_live_in_the_conversation_dir_with_0600(self) -> None:
        tmp = tempfile.mkdtemp()
        runtime = _Runtime()
        backend = ClaudeCliBackend(NativeBackendSpec(cwd="/w"), runtime_manager=runtime)
        await backend.open(_open_request(tmp, tools=["t"]))
        for path in (backend._mcp_config_path, backend._settings_path):
            self.assertEqual(os.path.dirname(path), tmp)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        config = json.loads(open(backend._mcp_config_path).read())
        self.assertNotIn(
            "tok", config["mcpServers"]["af"]["url"]
        )  # token only in headers
        settings = json.loads(open(backend._settings_path).read())["hooks"]
        self.assertEqual(
            set(settings),
            {
                "UserPromptSubmit",
                "PreToolUse",
                "PostToolUse",
                "PostToolUseFailure",
                "PreCompact",
            },
        )
        self.assertIn("af_hook.py", settings["PostToolUse"][0]["hooks"][0]["command"])
        self.assertEqual(settings["PreToolUse"][0]["matcher"], "mcp__af__.*")
        self.assertEqual(settings["PostToolUseFailure"][0]["matcher"], "mcp__af__.*")
        await backend.close()
        self.assertEqual(runtime.http_server.unregistered, ["tok"])

    async def test_private_files_left_by_an_earlier_session_become_0600(
        self,
    ) -> None:
        tmp = tempfile.mkdtemp()
        for name in ("mcp_config.json", "settings.json"):
            path = os.path.join(tmp, name)
            with open(path, "w") as fh:
                fh.write("stale " * 100)
            os.chmod(path, 0o644)
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd="/w"), runtime_manager=_Runtime()
        )
        await backend.open(_open_request(tmp, tools=["t"]))
        for path in (backend._mcp_config_path, backend._settings_path):
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            with open(path) as fh:
                json.load(fh)  # rewritten whole, no stale tail
        await backend.close()


class HookRelayTest(AsyncTestCase):
    def _backend(self, hooks) -> ClaudeCliBackend:
        b = _backend()
        b._hooks = hooks
        return b

    async def test_user_prompt_submit_carries_l2(self) -> None:
        out = await self._backend(_Hooks(l2="<af_context>x</af_context>"))._on_hook(
            {"hook_event_name": "UserPromptSubmit", "prompt": "hi"}
        )
        self.assertEqual(
            out["hookSpecificOutput"]["additionalContext"], "<af_context>x</af_context>"
        )

    async def test_no_l2_sends_nothing(self) -> None:
        self.assertEqual(
            await self._backend(_Hooks())._on_hook(
                {"hook_event_name": "UserPromptSubmit"}
            ),
            {},
        )

    async def test_subagent_af_call_is_denied(self) -> None:
        hooks = _Hooks(deny="main agent only")
        out = await self._backend(hooks)._on_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "mcp__af__write_brief",
                "tool_use_id": "t1",
                "agent_id": "a1",
            }
        )
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(hooks.before_calls, [("mcp__af__write_brief", "t1", "a1")])

    async def test_main_thread_af_call_is_allowed(self) -> None:
        out = await self._backend(_Hooks(deny="main agent only"))._on_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "mcp__af__write_brief",
                "tool_use_id": "t1",
            }
        )
        self.assertEqual(out, {})

    async def test_post_tool_use_ends_the_turn(self) -> None:
        out = await self._backend(_Hooks(stop=True))._on_hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "mcp__af__clarification",
                "tool_use_id": "t1",
            }
        )
        self.assertIs(out["continue"], False)

    async def test_a_failed_af_call_finishes_without_a_stop_decision(self) -> None:
        # Claude Code runs PostToolUseFailure, not PostToolUse, for an AF call
        # that returned an error, and ignores ``continue: false`` from it.
        hooks = _AfterHooks(stop=True)
        out = await self._backend(hooks)._on_hook(
            {
                "hook_event_name": "PostToolUseFailure",
                "tool_name": "mcp__af__write_brief",
                "tool_use_id": "t1",
                "error": "refused",
            }
        )
        self.assertEqual(out, {})
        self.assertEqual(hooks.after_calls, [("mcp__af__write_brief", "t1")])

    async def test_non_af_tools_are_ignored(self) -> None:
        out = await self._backend(_Hooks(stop=True))._on_hook(
            {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_use_id": "t1"}
        )
        self.assertEqual(out, {})

    async def test_pre_compact_is_reported(self) -> None:
        hooks = _Hooks()
        await self._backend(hooks)._on_hook({"hook_event_name": "PreCompact"})
        self.assertTrue(hooks.compacted)

    def test_the_relay_answers_before_claude_code_gives_up_on_the_hook(self) -> None:
        settings = json.loads(_backend()._settings())["hooks"]
        for event, matchers in settings.items():
            with self.subTest(event=event):
                timeout = matchers[0]["hooks"][0]["timeout"]
                self.assertGreater(timeout, af_hook.TIMEOUT_S)


_DEAD_HOST = {"AF_HOOK_URL": "http://127.0.0.1:9/hook", "AF_HOOK_TOKEN": "t"}


def _run_relay(payload: dict, env: dict) -> tuple[dict, str]:
    """Run ``bridge/af_hook.py`` as Claude Code does: hook input on stdin, the
    decision (if any) as JSON on stdout."""
    stdin = io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()))
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(sys, "stdin", stdin),
        mock.patch.object(sys, "stdout", stdout),
        mock.patch.object(sys, "stderr", stderr),
    ):
        for name in {"AF_HOOK_URL", "AF_HOOK_TOKEN"} - set(env):
            os.environ.pop(name, None)
        rc = af_hook.main()
    assert rc == 0
    out = stdout.getvalue()
    return (json.loads(out) if out else {}), stderr.getvalue()


class HookRelayFailsClosedTest(unittest.TestCase):
    """The relay keeps AF in control when the AF host does not answer."""

    def test_af_tool_call_is_denied(self) -> None:
        out, err = _run_relay(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "mcp__af__write_brief",
                "agent_id": "a1",
            },
            _DEAD_HOST,
        )
        decision = out["hookSpecificOutput"]
        self.assertEqual(decision["permissionDecision"], "deny")
        self.assertIn("did not answer", decision["permissionDecisionReason"])
        self.assertIn("af_hook:", err)

    def test_turn_stops_after_an_af_tool(self) -> None:
        out, _ = _run_relay(
            {"hook_event_name": "PostToolUse", "tool_name": "mcp__af__clarification"},
            _DEAD_HOST,
        )
        self.assertIs(out["continue"], False)
        self.assertIn("stopped", out["stopReason"])

    def test_prompt_is_blocked_instead_of_sent_without_its_context(self) -> None:
        out, _ = _run_relay(
            {"hook_event_name": "UserPromptSubmit", "prompt": "hi"}, _DEAD_HOST
        )
        self.assertEqual(out["decision"], "block")
        self.assertIn("not sent", out["reason"])

    def test_other_tools_and_events_are_left_alone(self) -> None:
        for payload in (
            {"hook_event_name": "PreToolUse", "tool_name": "Bash"},
            {"hook_event_name": "PostToolUse", "tool_name": "Read"},
            {"hook_event_name": "PreCompact"},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(_run_relay(payload, _DEAD_HOST)[0], {})

    def test_missing_relay_environment_fails_closed(self) -> None:
        out, err = _run_relay(
            {"hook_event_name": "PreToolUse", "tool_name": "mcp__af__task"}, {}
        )
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertIn("AF_HOOK_URL", err)


class MapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.b = _backend()

    def test_system_init_emits_session_started(self) -> None:
        events = self.b._map({"type": "system", "subtype": "init", "session_id": "s7"})
        self.assertEqual([type(e) for e in events], [SessionStarted])

    def test_failed_af_server_at_init_is_an_actionable_error(self) -> None:
        self.b._expects_af = True
        events = self.b._map(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s7",
                "mcp_servers": [{"name": "af", "status": "failed"}],
            }
        )
        error = events[-1]
        self.assertIsInstance(error, VendorError)
        self.assertIn("allowManagedMcpServersOnly", error.message)

    def test_init_must_show_af_connected_with_its_tools(self) -> None:
        self.b._expects_af = True
        self.b._started_emitted = True
        connected = {"name": "af", "status": "connected", "source": "dynamic"}
        cases = {
            "unlisted": ({"mcp_servers": [], "tools": ["mcp__af__x"]}, True),
            "pending": (
                {"mcp_servers": [{"name": "af", "status": "pending"}], "tools": []},
                True,
            ),
            "no tools": ({"mcp_servers": [connected], "tools": ["Bash"]}, True),
            "healthy": (
                {"mcp_servers": [connected], "tools": ["Bash", "mcp__af__task"]},
                False,
            ),
        }
        for name, (fields, fails) in cases.items():
            with self.subTest(name):
                events = self.b._map({"type": "system", "subtype": "init", **fields})
                self.assertEqual(any(isinstance(e, VendorError) for e in events), fails)

    def test_init_is_not_checked_without_af_tools(self) -> None:
        events = self.b._map(
            {"type": "system", "subtype": "init", "session_id": "s7", "tools": []}
        )
        self.assertEqual([type(e) for e in events], [SessionStarted])

    def test_stream_events_become_deltas_and_message_end_carries_text(self) -> None:
        self.b._started_emitted = True
        self.assertEqual(
            self.b._map(
                {
                    "type": "stream_event",
                    "event": {"type": "message_start", "message": {"id": "msg_1"}},
                }
            ),
            [],
        )
        delta = self.b._map(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hi"},
                },
            }
        )
        self.assertEqual(delta, [TextDelta(message_id="msg_1", text="Hi")])
        obj = {
            "type": "assistant",
            "uuid": "u1",
            "message": {
                "id": "msg_1",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "mcp__af__write_brief",
                        "input": {},
                    },
                    {"type": "tool_use", "id": "t2", "name": "Read", "input": {}},
                ],
            },
        }
        events = self.b._map(obj)
        self.assertFalse(any(isinstance(e, TextDelta) for e in events))
        end = events[-1]
        self.assertIsInstance(end, MessageEnd)
        self.assertEqual(end.text, "Hi")
        self.assertEqual(end.af_tool_use_ids, ("t1",))
        self.assertEqual(set(end.tool_use_ids), {"t1", "t2"})

    def test_subagent_lines_are_ignored(self) -> None:
        obj = {
            "type": "assistant",
            "parent_tool_use_id": "p",
            "message": {"id": "m", "content": []},
        }
        self.assertEqual(self.b._map(obj), [])

    def test_compact_system_message(self) -> None:
        self.b._started_emitted = True
        self.assertEqual(
            [
                type(e)
                for e in self.b._map({"type": "system", "subtype": "compact_boundary"})
            ],
            [Compaction],
        )

    def test_result_becomes_turn_end(self) -> None:
        self.b._started_emitted = True
        events = self.b._map(
            {
                "type": "result",
                "session_id": "s9",
                "stop_reason": "end_turn",
                "is_error": False,
                "num_turns": 3,
                "total_cost_usd": 0.02,
                "usage": {"input_tokens": 1},
                "result": "done",
            }
        )
        end = events[-1]
        self.assertIsInstance(end, TurnEnd)
        self.assertEqual(end.result_text, "done")

    def test_a_prompt_blocked_by_a_hook_fails_the_turn_unsubmitted(self) -> None:
        # Shapes recorded from claude 2.1.288 with the relay's fail-closed block.
        notice = (
            "UserPromptSubmit operation blocked by hook:\nThe AgentFoundation host "
            "did not answer the UserPromptSubmit hook (refused).\n\n"
            "Original prompt: my private question"
        )
        self.b._started_emitted = True
        self.assertEqual(
            self.b._map(
                {
                    "type": "system",
                    "subtype": "informational",
                    "content": notice,
                    "level": "warning",
                    "prevent_continuation": True,
                }
            ),
            [],
        )
        events = self.b._map(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 0,
                "stop_reason": None,
                "result": notice,
                "session_id": "s9",
            }
        )
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], VendorError)
        self.assertFalse(events[0].submitted)
        self.assertIn("did not answer the UserPromptSubmit hook", events[0].message)
        self.assertNotIn("my private question", events[0].message)

    def test_a_turn_the_model_ran_still_ends_normally(self) -> None:
        self.b._started_emitted = True
        self.b._map(
            {
                "type": "system",
                "subtype": "informational",
                "content": "Operation stopped by hook: x",
                "prevent_continuation": True,
            }
        )
        events = self.b._map(
            {"type": "result", "num_turns": 2, "session_id": "s9", "result": "ok"}
        )
        self.assertIsInstance(events[-1], TurnEnd)


def _scripted_cli(lines: list[dict], stderr: str, rc: int) -> str:
    """A stand-in ``claude`` that prints ``lines`` as stream-json."""
    return scripted_cli("".join(json.dumps(line) + "\n" for line in lines), stderr, rc)


class InterruptStopsTheVendorTest(AsyncTestCase):
    """The installed ``claude`` is a launcher that runs the agent as a child
    process (claude 2.1.288: /usr/local/bin/claude_code/claude starts
    ~/.cache/claude_code_native_versions/<v>/claude); that child inherits the
    turn's stdout."""

    _INIT = {"type": "system", "subtype": "init", "session_id": "s1"}

    async def test_interrupt_ends_the_whole_vendor_process_tree(self) -> None:
        launcher, pid_file = launcher_with_agent(agent_printing(self._INIT))
        directory = tempfile.mkdtemp()
        backend = ClaudeCliBackend(NativeBackendSpec(cwd=directory, cli_path=launcher))
        backend._l1_path = "/tmp/l1.md"
        backend._session_id = "s1"
        events: list = []

        async def consume() -> None:
            async for event in backend.run_turn(TurnRequest(text="hi")):
                events.append(event)

        turn = asyncio.ensure_future(consume())
        agent_pid = 0
        try:
            while not events:
                await asyncio.sleep(0.02)
            agent_pid = await read_pid(pid_file)
            self.assertTrue(pid_alive(agent_pid))
            await asyncio.wait_for(backend.interrupt(), 5)
            self.assertFalse(pid_alive(agent_pid))
        finally:
            if agent_pid and pid_alive(agent_pid):
                os.kill(agent_pid, signal.SIGKILL)
            await asyncio.wait_for(turn, 10)

    async def test_a_cancelled_turn_is_acknowledged_not_uncertain(self) -> None:
        launcher, pid_file = launcher_with_agent(agent_printing(self._INIT))
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=launcher)
        )
        outcome, first, agent, took = await cancel_through_actor(backend, pid_file)
        self.assertIsInstance(first, SessionStarted)
        self.assertTrue(outcome.interrupted)
        self.assertTrue(outcome.acknowledged)
        self.assertLess(took, 5)
        self.assertFalse(pid_alive(agent))


class TurnClassificationTest(AsyncTestCase):
    async def _turn(self, lines, stderr, rc, *, resumed=True) -> list:
        cli = _scripted_cli(lines, stderr, rc)
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=cli)
        )
        backend._l1_path = "/tmp/l1.md"
        backend._session_id = "gone"
        backend._started_emitted = resumed
        return [e async for e in backend.run_turn(TurnRequest(text="hi"))]

    async def test_resume_of_a_missing_session_is_classified(self) -> None:
        events = await self._turn(
            [
                {
                    "type": "result",
                    "subtype": "error_during_execution",
                    "is_error": True,
                    "num_turns": 0,
                    "session_id": "gone",
                }
            ],
            "No conversation found with session ID: gone\n",
            1,
        )
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].session_missing)
        self.assertFalse(events[0].submitted)

    async def test_other_errors_stay_a_failed_turn(self) -> None:
        events = await self._turn(
            [
                {
                    "type": "result",
                    "subtype": "error_during_execution",
                    "is_error": True,
                    "num_turns": 0,
                    "session_id": "gone",
                }
            ],
            "rate limited\n",
            1,
        )
        self.assertIsInstance(events[-1], TurnEnd)
        self.assertTrue(events[-1].is_error)

    async def test_no_result_is_a_vendor_error(self) -> None:
        events = await self._turn([], "boom\n", 3)
        self.assertIsInstance(events[-1], VendorError)
        self.assertFalse(events[-1].session_missing)

    async def test_turn_without_af_at_init_stops_before_the_model_acts(self) -> None:
        cli = _scripted_cli(
            [
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "s1",
                    "mcp_servers": [{"name": "af", "status": "failed"}],
                    "tools": [],
                },
                {
                    "type": "assistant",
                    "message": {"id": "m", "content": [{"type": "text", "text": "x"}]},
                },
                {"type": "result", "num_turns": 1, "session_id": "s1"},
            ],
            "",
            0,
        )
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=cli)
        )
        backend._l1_path = "/tmp/l1.md"
        backend._session_id = "s1"
        backend._expects_af = True
        events = [e async for e in backend.run_turn(TurnRequest(text="hi"))]
        self.assertIsInstance(events[-1], VendorError)
        self.assertIn("'failed'", events[-1].message)
        self.assertFalse(any(isinstance(e, (MessageEnd, TurnEnd)) for e in events))


_L2 = '<af_context nonce="n">state</af_context>'
_INIT = {"type": "system", "subtype": "init", "session_id": "s1"}
_MODEL_START = {
    "type": "stream_event",
    "event": {"type": "message_start", "message": {"id": "msg_1"}},
}


class HookLivenessTest(AsyncTestCase):
    """A managed policy that drops AF's command hooks (allowManagedHooksOnly)
    must not silently cost the turn its context, turn stop and subagent guard."""

    def _watching(self, request=None) -> ClaudeCliBackend:
        backend = _backend()
        backend._hooks = _Hooks(l2=_L2)
        backend._prompt_hook.start(
            request or TurnRequest(text="hi", l2_text=_L2), hooks_registered=True
        )
        return backend

    async def test_the_model_starting_without_the_prompt_hook_stops_the_turn(
        self,
    ) -> None:
        backend = self._watching()
        self.assertEqual([type(e) for e in backend._map(_INIT)], [SessionStarted])
        events = backend._map(_MODEL_START)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], VendorError)
        self.assertIn("allowManagedHooksOnly", events[0].message)
        self.assertTrue(events[0].submitted)

    async def test_a_turn_whose_hook_ran_proceeds(self) -> None:
        backend = self._watching()
        out = await backend._on_hook({"hook_event_name": "UserPromptSubmit"})
        self.assertEqual(out["hookSpecificOutput"]["additionalContext"], _L2)
        backend._map(_INIT)
        self.assertEqual(backend._map(_MODEL_START), [])

    async def test_a_relay_that_blocked_the_prompt_is_reported_as_blocked(
        self,
    ) -> None:
        # The relay's fail-closed block reaches AF's process never, so the
        # hook counts as not run; Claude Code reports the block after init and
        # the model never starts.
        backend = self._watching()
        backend._map(_INIT)
        backend._map(
            {
                "type": "system",
                "subtype": "informational",
                "content": "UserPromptSubmit operation blocked by hook:\nrefused",
                "prevent_continuation": True,
            }
        )
        events = backend._map(
            {"type": "result", "num_turns": 0, "session_id": "s1", "result": "x"}
        )
        self.assertIsInstance(events[-1], VendorError)
        self.assertFalse(events[-1].submitted)
        self.assertNotIn("allowManagedHooksOnly", events[-1].message)

    async def test_a_turn_without_turn_context_still_needs_the_prompt_hook(
        self,
    ) -> None:
        # An L2 is sent only when due, but AF's hook runs on every turn of the
        # hook channel (it returns no context when none is due).
        backend = self._watching(TurnRequest(text="hi"))
        backend._hooks = _Hooks()
        backend._map(_INIT)
        events = backend._map(_MODEL_START)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], VendorError)
        self.assertIn("allowManagedHooksOnly", events[0].message)
        backend = self._watching(TurnRequest(text="hi"))
        backend._hooks = _Hooks()
        self.assertEqual(
            await backend._on_hook({"hook_event_name": "UserPromptSubmit"}), {}
        )
        backend._map(_INIT)
        self.assertEqual(backend._map(_MODEL_START), [])

    async def test_the_hook_is_not_required_off_its_channel_or_for_a_local_command(
        self,
    ) -> None:
        cases = {
            "envelope": TurnRequest(
                text=f"{_L2}\nhi", l2_text=_L2, channel=L2Channel.ENVELOPE
            ),
            "local command": TurnRequest(text="/compact", l2_text=_L2),
            "local command with arguments": TurnRequest(text="/context all"),
        }
        for name, request in cases.items():
            with self.subTest(name):
                backend = self._watching(request)
                self.assertEqual(backend._map(_MODEL_START), [])
        backend = _backend()
        backend._prompt_hook.start(
            TurnRequest(text="hi", l2_text=_L2), hooks_registered=False
        )
        self.assertEqual(backend._map(_MODEL_START), [])

    async def test_the_process_is_killed_before_the_model_output_is_used(
        self,
    ) -> None:
        cli = _scripted_cli(
            [
                _INIT,
                _MODEL_START,
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg_1",
                        "content": [{"type": "text", "text": "answer"}],
                    },
                },
                {"type": "result", "num_turns": 1, "session_id": "s1"},
            ],
            "",
            0,
        )
        backend = ClaudeCliBackend(
            NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=cli)
        )
        backend._l1_path = "/tmp/l1.md"
        backend._session_id = "s1"
        backend._settings_path = os.path.join(tempfile.mkdtemp(), "settings.json")
        events = [
            e async for e in backend.run_turn(TurnRequest(text="hi", l2_text=_L2))
        ]
        self.assertIsInstance(events[-1], VendorError)
        self.assertIn("UserPromptSubmit", events[-1].message)
        self.assertFalse(any(isinstance(e, (MessageEnd, TurnEnd)) for e in events))

    async def test_other_slash_text_needs_the_prompt_hook_once_the_model_starts(
        self,
    ) -> None:
        # Unknown names, paths, user commands and skills reach the model
        # through UserPromptSubmit (claude 2.1.289).
        for text in ("/foo", "/tmp/notes.txt is gone", "/probecmd hello"):
            with self.subTest(text=text):
                backend = self._watching(TurnRequest(text=text, l2_text=_L2))
                backend._map(_INIT)
                events = backend._map(_MODEL_START)
                self.assertEqual(len(events), 1)
                self.assertIsInstance(events[0], VendorError)
                self.assertIn("allowManagedHooksOnly", events[0].message)
                backend = self._watching(TurnRequest(text=text, l2_text=_L2))
                await backend._on_hook({"hook_event_name": "UserPromptSubmit"})
                backend._map(_INIT)
                self.assertEqual(backend._map(_MODEL_START), [])

    async def test_an_unlisted_local_command_runs_without_the_prompt_hook(
        self,
    ) -> None:
        # `/release-notes` runs locally like `/context`: no hook, no model,
        # its output a `<synthetic>` assistant message.
        backend = self._watching(TurnRequest(text="/release-notes", l2_text=_L2))
        backend._map(_INIT)
        events = backend._map(
            {
                "type": "assistant",
                "message": {
                    "id": "local-1",
                    "model": "<synthetic>",
                    "content": [{"type": "text", "text": "not available"}],
                },
            }
        )
        self.assertFalse(any(isinstance(e, VendorError) for e in events))
        self.assertIsInstance(events[-1], MessageEnd)
        # A real model message without its stream start is still checked.
        backend = self._watching(TurnRequest(text="/foo", l2_text=_L2))
        events = backend._map(
            {
                "type": "assistant",
                "message": {"id": "m", "model": "claude-haiku-4-5", "content": []},
            }
        )
        self.assertIsInstance(events[0], VendorError)

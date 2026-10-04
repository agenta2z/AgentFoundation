"""CodexCliBackend argv construction, codex --json mapping and AF call
attribution.

Event shapes mirror real ``codex exec --json`` output captured in
scripts/native_spikes/s15_codex_http_mcp.py, and the MCP ``_meta`` of a
``tools/call`` the one captured in s18_codex_compound_widget.py (codex-cli
0.159.3). Real end-to-end coverage (HTTP MCP tool call + developer_instructions
+ resume) is the s15 spike and a two-turn memory resume; real-model compound
widgets are s18; a real-model SOP run is e2e_research_sop.py --backend
codex_cli.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import attr
import later.unittest
import mcp.types as mt
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
    NativeRuntimeManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    BridgeResult,
    END_TURN_MARKER,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    SessionStarted,
    TextDelta,
    TurnEnd,
    VendorError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BridgeToolSpec,
    CallerTools,
    Evidence,
    L2Channel,
    NativeBackendSpec,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.codex_cli import (
    CodexCliBackend,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex import (
    common as codex_common,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer import (
    CodexCliInferencer,
)
from fakes import GatedCli, RecordingInteractive, scripted_cli
from helpers import FIXTURE_SOPS, RecordingExecutor, tool_registry
from mcp.server.lowlevel.server import request_ctx
from test_cli_runner import (
    agent_printing,
    cancel_through_actor,
    launcher_with_agent,
    pid_alive,
)


def _backend(**spec) -> CodexCliBackend:
    b = CodexCliBackend(NativeBackendSpec(cwd="/work", model="gpt-5-codex", **spec))
    b._l1_text = 'Persona "Ada"\nwith `code` and a newline'
    return b


def _feed(backend: CodexCliBackend, *objs: dict) -> list:
    out = []
    for obj in objs:
        out.extend(backend._map(obj))
    return out


class CapabilitiesTest(unittest.TestCase):
    def test_http_caller_tools_recorded_id_all_verified(self) -> None:
        caps = CodexCliBackend.capabilities
        self.assertEqual(caps.caller_tools, CallerTools.HTTP)
        self.assertEqual(caps.l2_channels, (L2Channel.ENVELOPE,))
        self.assertFalse(caps.pinned_session_id)  # recorded from thread.started
        self.assertFalse(caps.exact_fork)
        self.assertFalse(caps.turn_stop_hook)
        self.assertEqual(caps.evidence["caller_tools"], Evidence.VERIFIED)
        self.assertEqual(caps.evidence["resume"], Evidence.VERIFIED)


class ArgvTest(unittest.TestCase):
    def test_fresh_turn_exec_json_mcp_and_developer_instructions(self) -> None:
        b = _backend()
        b._mcp_url = "http://127.0.0.1:9/mcp/tok"
        b._mcp_token = "tok"
        argv = b._argv(TurnRequest(text="hello", channel=L2Channel.ENVELOPE))
        self.assertEqual(argv[1], "exec")
        self.assertNotIn("resume", argv)
        self.assertIn("--json", argv)
        self.assertIn("--skip-git-repo-check", argv)
        self.assertTrue(
            any(a == 'mcp_servers.af.url="http://127.0.0.1:9/mcp/tok"' for a in argv)
        )
        self.assertTrue(any('bearer_token_env_var="AF_MCP_TOKEN"' in a for a in argv))
        di = next(a for a in argv if a.startswith("developer_instructions="))
        self.assertEqual(
            json.loads(di.split("=", 1)[1]), b._l1_text
        )  # TOML-valid, round-trips
        self.assertEqual(argv[-1], "hello")

    def test_resume_uses_exec_resume_subcommand(self) -> None:
        b = _backend()
        b._session_id = "th-1"
        b._started_emitted = True
        argv = b._argv(TurnRequest(text="again"))
        self.assertEqual(argv[1], "exec")
        self.assertEqual(argv[2], "resume")
        self.assertEqual(argv[3], "th-1")
        self.assertIn("--json", argv)

    def test_bypass_only_in_bypass_permission_mode(self) -> None:
        b = _backend(permission_mode="read-only")
        argv = b._argv(TurnRequest(text="h"))
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)
        b2 = _backend(permission_mode="bypassPermissions")
        self.assertIn(
            "--dangerously-bypass-approvals-and-sandbox",
            b2._argv(TurnRequest(text="h")),
        )

    def test_no_mcp_flags_when_unset(self) -> None:
        b = _backend()
        argv = b._argv(TurnRequest(text="h"))
        self.assertFalse(any(a.startswith("mcp_servers.af") for a in argv))

    def test_the_codex_binary_is_the_configured_one_else_found_on_path(
        self,
    ) -> None:
        request = TurnRequest(text="h")
        with mock.patch("shutil.which", return_value="/path/bin/codex") as which:
            self.assertEqual(_backend()._argv(request)[0], "/path/bin/codex")
            which.assert_called_with("codex")
            pinned = _backend(cli_path="/opt/codex")
            self.assertEqual(pinned._argv(request)[0], "/opt/codex")
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(_backend()._argv(request)[0], "codex")


class ModelTagTest(unittest.TestCase):
    """Codex gets the model the classic Codex CLI inferencer would use: a
    Claude-family tag (cascaded from a mixed config, or a ``/model`` meant
    for another backend) becomes its default, anything else passes."""

    def _model(self, tag: str) -> str:
        b = _backend()
        b._spec.model = tag
        argv = b._argv(TurnRequest(text="h"))
        return argv[argv.index("-m") + 1]

    def test_claude_tags_become_the_codex_default(self) -> None:
        for tag in ("opus[1m]", "claude-opus-4-7", "Sonnet", "fable"):
            with self.subTest(tag=tag):
                self.assertEqual(self._model(tag), "gpt-5.5")

    def test_codex_models_pass_through(self) -> None:
        for tag in ("gpt-5-codex", "gpt-5.4-mini", "o3"):
            with self.subTest(tag=tag):
                self.assertEqual(self._model(tag), tag)

    def test_matches_the_classic_codex_inferencer(self) -> None:
        self.assertEqual(
            codex_common.NON_CODEX_MODEL_PREFIXES,
            CodexCliInferencer._NON_CODEX_MODEL_PREFIXES,
        )
        self.assertEqual(
            codex_common.CODEX_DEFAULT_MODEL,
            attr.fields(CodexCliInferencer).model_name.default,
        )


class MapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.b = _backend()

    def test_thread_started_emits_session_started_once(self) -> None:
        events = self.b._map({"type": "thread.started", "thread_id": "th-9"})
        self.assertEqual([type(e) for e in events], [SessionStarted])
        self.assertEqual(events[0].session_id, "th-9")

    def test_agent_message_item_makes_text_and_message(self) -> None:
        self.b._started_emitted = True
        events = self.b._map(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_0",
                    "type": "agent_message",
                    "text": "Hello there",
                },
            }
        )
        self.assertIsInstance(events[0], TextDelta)
        self.assertEqual(events[0].text, "Hello there")
        self.assertIsInstance(events[1], MessageEnd)
        self.assertEqual(events[1].text, "Hello there")
        self.assertEqual(events[1].message_uuid, "item_0")

    def test_af_mcp_tool_call_item_is_not_reported_again(self) -> None:
        # The call was announced from its MCP request; this item arrives after
        # the call finished and names neither the call nor its output item.
        self.b._started_emitted = True
        events = self.b._map(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "mcp_tool_call",
                    "server": "af",
                    "tool": "clarification",
                    "status": "completed",
                    "result": {"content": []},
                },
            }
        )
        self.assertEqual(events, [])

    def test_non_af_mcp_tool_call_ignored(self) -> None:
        self.b._started_emitted = True
        events = self.b._map(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "mcp_tool_call",
                    "server": "other",
                    "tool": "x",
                },
            }
        )
        self.assertEqual(events, [])

    def test_reasoning_item_ignored(self) -> None:
        self.b._started_emitted = True
        events = self.b._map(
            {
                "type": "item.completed",
                "item": {"id": "r", "type": "reasoning", "text": "thinking"},
            }
        )
        self.assertEqual(events, [])

    def test_turn_completed_becomes_turn_end_with_usage(self) -> None:
        self.b._started_emitted = True
        self.b._session_id = "th-7"
        events = self.b._map(
            {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}
        )
        end = events[-1]
        self.assertIsInstance(end, TurnEnd)
        self.assertEqual(end.session_id, "th-7")
        self.assertFalse(end.is_error)
        self.assertEqual(end.usage["input_tokens"], 5)

    def test_error_event_becomes_vendor_error(self) -> None:
        events = self.b._map({"type": "error", "message": "gateway 500"})
        self.assertEqual([type(e) for e in events], [VendorError])
        self.assertIn("gateway 500", events[0].message)

    def test_full_turn_sequence(self) -> None:
        objs = [
            {"type": "thread.started", "thread_id": "full"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {"id": "i0", "type": "agent_message", "text": "I'll call it"},
            },
            {
                "type": "item.started",
                "item": {
                    "id": "i1",
                    "type": "mcp_tool_call",
                    "server": "af",
                    "tool": "echo",
                    "status": "in_progress",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "i1",
                    "type": "mcp_tool_call",
                    "server": "af",
                    "tool": "echo",
                    "status": "completed",
                    "result": {"content": []},
                },
            },
            {
                "type": "item.completed",
                "item": {"id": "i2", "type": "agent_message", "text": "Done"},
            },
            {"type": "turn.completed", "usage": {}},
        ]
        kinds = [type(e).__name__ for e in _feed(self.b, *objs)]
        self.assertEqual(
            kinds,
            [
                "SessionStarted",
                "TextDelta",
                "MessageEnd",
                "TextDelta",
                "MessageEnd",
                "TurnEnd",
            ],
        )


class HardeningTest(unittest.TestCase):
    def test_token_is_never_on_argv(self) -> None:
        b = _backend()
        b._mcp_url = "http://127.0.0.1:9/mcp"
        b._mcp_token = "SECRET-TOKEN"
        argv = b._argv(TurnRequest(text="hi"))
        self.assertFalse(any("SECRET-TOKEN" in a for a in argv))
        self.assertIn('mcp_servers.af.url="http://127.0.0.1:9/mcp"', argv)

    def test_hermetic_ignores_user_config_and_rules(self) -> None:
        argv = _backend(environment="hermetic")._argv(TurnRequest(text="h"))
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        self.assertNotIn(
            "--ignore-user-config", _backend()._argv(TurnRequest(text="h"))
        )

    def test_effort_and_extra_servers(self) -> None:
        b = _backend(
            effort="high",
            extra_mcp_servers={"docs": {"command": "docs-mcp", "args": ["--ro"]}},
        )
        argv = b._argv(TurnRequest(text="h"))
        self.assertIn('model_reasoning_effort="high"', argv)
        self.assertIn('mcp_servers.docs.command="docs-mcp"', argv)
        self.assertIn('mcp_servers.docs.args=["--ro"]', argv)


class MissingSessionTest(later.unittest.TestCase):
    async def _turn(self, stderr: str, rc: int) -> list:
        cli = scripted_cli("", stderr, rc)
        b = CodexCliBackend(NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=cli))
        b._session_id = "th-gone"
        b._started_emitted = True
        return [e async for e in b.run_turn(TurnRequest(text="hi"))]

    async def test_missing_rollout_is_classified(self) -> None:
        events = await self._turn(
            "Error: thread/resume failed: no rollout found for thread id th-gone (code -32600)\n",
            1,
        )
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].session_missing)
        self.assertFalse(events[0].submitted)

    async def test_other_failures_are_not(self) -> None:
        events = await self._turn("gateway 500\n", 1)
        self.assertIsInstance(events[-1], VendorError)
        self.assertFalse(events[-1].session_missing)
        self.assertTrue(events[-1].submitted)

    async def test_unreachable_af_server_fails_the_turn_unsubmitted(self) -> None:
        # stderr recorded from codex-cli 0.159.3 with `required` and a dead URL.
        events = await self._turn(
            "ERROR codex_core::session: Failed to create session: required MCP "
            "servers failed to initialize: af: handshaking with MCP server failed\n"
            "Error: thread/start: thread/start failed: error creating thread\n",
            1,
        )
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0].submitted)
        self.assertFalse(events[0].session_missing)
        self.assertIn("'af'", events[0].message)
        self.assertIn("af: handshaking with MCP server failed", events[0].message)


class InterruptStopsTheVendorTest(later.unittest.TestCase):
    """The installed ``codex`` is a launcher that runs the agent as a child
    process (~/.codex/packages/.../bin/codex) holding the turn's stdout: a
    cancelled turn must end it, or it keeps running next to the next turn
    (`thread-store conflict: thread … already has an active writer`)."""

    async def test_a_cancelled_turn_ends_the_vendor_tree_and_is_acknowledged(
        self,
    ) -> None:
        launcher, pid_file = launcher_with_agent(
            agent_printing({"type": "thread.started", "thread_id": "th-1"})
        )
        backend = CodexCliBackend(
            NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=launcher)
        )
        outcome, first, agent, took = await cancel_through_actor(backend, pid_file)
        self.assertEqual(first, SessionStarted(session_id="th-1"))
        self.assertTrue(outcome.interrupted)
        self.assertTrue(outcome.acknowledged)
        self.assertLess(took, 5)
        self.assertFalse(pid_alive(agent))


class ToolTimeoutTest(unittest.TestCase):
    def test_af_tool_calls_get_the_spec_timeout_in_whole_seconds(self) -> None:
        for ms, expected in ((7_200_000, 7200), (1_500, 2)):
            with self.subTest(ms=ms):
                b = _backend(
                    mcp_tool_timeout_ms=ms,
                    extra_mcp_servers={"docs": {"command": "docs-mcp"}},
                )
                b._mcp_url = "http://127.0.0.1:9/mcp"
                argv = b._argv(TurnRequest(text="h"))
                self.assertIn(f"mcp_servers.af.tool_timeout_sec={expected}", argv)
                self.assertFalse(any("docs.tool_timeout_sec" in a for a in argv))


class _FakeHttpServer:
    def __init__(self) -> None:
        self.turn_active = None
        self.tools: list = []
        self.unregistered: list = []

    async def register(self, tools, *, hook_handler=None, turn_active=None):
        self.turn_active = turn_active
        self.tools = list(tools)
        return "http://127.0.0.1:9/mcp", "tok", {}

    async def unregister(self, token):
        self.unregistered.append(token)


class _Runtime:
    def __init__(self) -> None:
        self.server = _FakeHttpServer()

    async def ensure_http_server(self):
        return self.server


def _tool(name: str, log: list) -> BridgeToolSpec:
    async def handler(args: dict) -> BridgeResult:
        log.append(("ran", name))
        return BridgeResult(f"{name} done")

    return BridgeToolSpec(
        name=name, description=name, input_schema={"type": "object"}, handler=handler
    )


def _open_request(tools: list, hooks) -> SessionOpenRequest:
    return SessionOpenRequest(
        session_id="",
        resume=False,
        l1_text="L1",
        l1_path="",
        tools=tools,
        hooks=hooks,
        cwd="/w",
        model="",
    )


class _Hooks:
    def __init__(self, log: list, deny=None) -> None:
        self.log = log
        self.deny = deny

    async def before_af_tool(self, tool_name, tool_use_id, agent_id):
        self.log.append(("before", tool_name, tool_use_id, agent_id))
        return self.deny

    async def after_af_tool(self, tool_name, tool_use_id):
        self.log.append(("after", tool_name, tool_use_id))
        return False


def _codex_meta(call_id: str, item_id: str) -> dict:
    """The ``_meta`` codex-cli 0.159.3 sends with a ``tools/call`` made by a
    code-mode ``exec`` script (values shortened)."""
    return {
        "callId": call_id,
        "x-codex-turn-metadata": {"turn_id": "01a10411-turn", "model": "gpt-5.6-sol"},
        "threadId": "01a10411-thread",
        "sessionId": "01a10411-thread",
        "windowId": "01a10411-thread:0",
        "itemId": item_id,
        "traceparent": "00-28694e9019ce835a4061af74e4592636-eeadef964944fd09-01",
        "codex_bridge_mcp_call_id": f"bridge-{call_id}",
        "progressToken": 1,
    }


@contextlib.contextmanager
def _mcp_request(meta=None):
    """Run the code inside an MCP request whose ``_meta`` is ``meta``."""
    params_meta = mt.RequestParams.Meta(**meta) if meta is not None else None
    token = request_ctx.set(SimpleNamespace(meta=params_meta))
    try:
        yield
    finally:
        request_ctx.reset(token)


async def _call(handler, args: dict, meta=None):
    with _mcp_request(meta):
        return await handler(args)


class CallAttributionTest(later.unittest.TestCase):
    """Each AF call is announced from its MCP request: to the hooks as the
    tool use it runs next and to the turn as an AF message of the model
    output item that made it."""

    async def _open(self, log: list, hooks) -> tuple[CodexCliBackend, dict]:
        runtime = _Runtime()
        b = CodexCliBackend(NativeBackendSpec(cwd="/w"), runtime_manager=runtime)
        await b.open(
            _open_request([_tool("clarification", log), _tool("echo", log)], hooks)
        )
        b._events = asyncio.Queue()
        return b, {t.name: t.handler for t in runtime.server.tools}

    @staticmethod
    def _drain(b: CodexCliBackend) -> list:
        out = []
        while not b._events.empty():
            out.append(b._events.get_nowait())
        return out

    async def test_a_call_is_announced_with_its_output_item_before_it_runs(
        self,
    ) -> None:
        log: list = []
        b, _handlers = await self._open(log, _Hooks(log))
        queued_when_run: list = []

        async def probe(args: dict) -> BridgeResult:
            queued_when_run.append(b._events.qsize())
            return BridgeResult("ran")

        probed = b._announced(
            BridgeToolSpec(name="probe", description="", input_schema={}, handler=probe)
        )
        result = await _call(probed.handler, {}, _codex_meta("exec-1", "ctc_A"))
        self.assertEqual(result.text, "ran")
        self.assertEqual(queued_when_run, [1])
        self.assertEqual(
            self._drain(b),
            [
                MessageEnd(
                    message_id="ctc_A",
                    text="",
                    tool_use_ids=("exec-1",),
                    af_tool_use_ids=("exec-1",),
                )
            ],
        )
        self.assertEqual(
            log, [("before", "probe", "exec-1", None), ("after", "probe", "exec-1")]
        )

    async def test_calls_share_a_message_only_when_they_share_an_output_item(
        self,
    ) -> None:
        log: list = []
        b, handlers = await self._open(log, _Hooks(log))
        for call_id, item_id in (
            ("exec-1", "ctc_A"),
            ("exec-2", "ctc_A"),
            ("exec-3", "ctc_B"),
        ):
            result = await _call(
                handlers["clarification"], {}, _codex_meta(call_id, item_id)
            )
            self.assertEqual(result.text, "clarification done")
        events = self._drain(b)
        self.assertEqual(
            [(e.message_id, e.af_tool_use_ids) for e in events],
            [("ctc_A", ("exec-1",)), ("ctc_A", ("exec-2",)), ("ctc_B", ("exec-3",))],
        )

    async def test_a_call_without_codex_meta_is_a_message_of_its_own(self) -> None:
        log: list = []
        b, handlers = await self._open(log, _Hooks(log))
        await _call(handlers["echo"], {})
        await _call(handlers["echo"], {}, {"progressToken": 7})
        events = self._drain(b)
        ids = [e.af_tool_use_ids[0] for e in events]
        self.assertEqual([e.message_id for e in events], ids)
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all(i.startswith("af-") for i in ids))
        self.assertEqual([entry[2] for entry in log if entry[0] == "before"], ids)

    async def test_a_refused_call_neither_runs_nor_is_announced(self) -> None:
        log: list = []
        b, handlers = await self._open(
            log, _Hooks(log, deny="No active AgentFoundation turn.")
        )
        result = await _call(handlers["echo"], {}, _codex_meta("exec-9", "ctc_Z"))
        self.assertTrue(result.is_error)
        self.assertEqual(result.text, "No active AgentFoundation turn.")
        self.assertEqual(log, [("before", "echo", "exec-9", None)])
        self.assertEqual(self._drain(b), [])

    async def test_announcements_join_the_turn_events_where_they_happen(
        self,
    ) -> None:
        log: list = []
        cli = GatedCli(
            [
                [
                    json.dumps({"type": "thread.started", "thread_id": "th-1"}) + "\n",
                    json.dumps({"type": "turn.completed", "usage": {}}) + "\n",
                ]
            ]
        )
        runtime = _Runtime()
        b = CodexCliBackend(
            NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=cli.path),
            runtime_manager=runtime,
        )
        await b.open(_open_request([_tool("echo", log)], _Hooks(log)))
        handler = runtime.server.tools[0].handler
        events = []
        async for event in b.run_turn(TurnRequest(text="hi")):
            events.append(event)
            if isinstance(event, SessionStarted):
                await _call(handler, {}, _codex_meta("exec-1", "ctc_A"))
                cli.release(1, 1)
        self.assertEqual(
            [type(e).__name__ for e in events],
            ["SessionStarted", "MessageEnd", "TurnEnd"],
        )
        self.assertEqual(events[1].message_id, "ctc_A")
        self.assertIsNone(b._events)


_TOPIC = {"prompt": "Topic?", "output": ["topic"]}
_DEPTH = {"prompt": "Depth?", "choices": ["quick", "deep"], "output": ["depth"]}


def _lines(*objs: dict) -> str:
    return "".join(json.dumps(o) + "\n" for o in objs)


def _agent_message(item: str, text: str) -> dict:
    return {
        "type": "item.completed",
        "item": {"id": item, "type": "agent_message", "text": text},
    }


def _mcp_item(item: str, tool: str, args: dict) -> list[dict]:
    started = {
        "id": item,
        "type": "mcp_tool_call",
        "server": "af",
        "tool": tool,
        "arguments": args,
        "result": None,
        "error": None,
        "status": "in_progress",
    }
    done = dict(started, status="completed", result={"content": []})
    return [
        {"type": "item.started", "item": started},
        {"type": "item.completed", "item": done},
    ]


class _CapturingRuntime(NativeRuntimeManager):
    def __init__(self) -> None:
        super().__init__()
        self.server = _FakeHttpServer()

    async def ensure_http_server(self):
        return self.server


class CodexCompoundWidgetTest(later.unittest.TestCase):
    """Plan §5.4 / S18 on Codex: the questions one ``exec`` item asks form one
    compound widget; a question from a later output item is refused."""

    async def test_questions_of_one_output_item_form_one_compound_widget(
        self,
    ) -> None:
        thread = {"type": "thread.started", "thread_id": "01a10419-thread"}
        calls = (
            ("exec-1", "ctc_A", "clarification", _TOPIC),
            ("exec-2", "ctc_A", "single_choice", _DEPTH),
            ("exec-3", "ctc_B", "single_choice", {**_DEPTH, "output": ["depth_b"]}),
        )
        # Codex reports each MCP call as an `item.*` pair once it finished.
        items = [
            _lines(*_mcp_item(f"item_{index + 1}", tool, args))
            for index, (_c, _i, tool, args) in enumerate(calls)
        ]
        cli = GatedCli(
            [
                [
                    _lines(
                        thread,
                        {"type": "turn.started"},
                        _agent_message("item_0", "I'll ask both together."),
                    ),
                    items[0],
                    items[1],
                    items[2]
                    + _lines(
                        _agent_message("item_4", ""),
                        {"type": "turn.completed", "usage": {"input_tokens": 9}},
                    ),
                ],
                [
                    _lines(
                        thread,
                        {"type": "turn.started"},
                        _agent_message("item_0", "Thanks."),
                        {"type": "turn.completed", "usage": {}},
                    )
                ],
            ]
        )
        session_dir = tempfile.mkdtemp(prefix="af_native_codex_")
        runtime = _CapturingRuntime()
        interactive = RecordingInteractive(
            [{"values": {"topic": "lidar", "depth": "deep"}}]
        )
        native = NativeConversationalInferencer(
            backend={
                "kind": "codex_cli",
                "cwd": session_dir,
                "cli_path": cli.path,
                "l2_envelope_allowed": True,
            },
            tool_registry=tool_registry(),
            tool_executor=RecordingExecutor(),
            interactive=interactive,
            prior_context={"native_session_dir": session_dir},
            extra_sop_dirs=[FIXTURE_SOPS],
            allowed_sops=["mini_research"],
            record_store=InMemoryRecordStore(),
            runtime_manager=runtime,
            conversation_key="codex-compound",
        )
        results = []
        try:
            async with native:
                loop = asyncio.ensure_future(native.run_agentic_loop("ask me"))
                await _wait_for(
                    lambda: native.current_turn is not None
                    and native.current_turn.accepted
                )
                handlers = {t.name: t.handler for t in runtime.server.tools}
                for chunk, (call_id, item_id, tool, args) in enumerate(calls, 1):
                    results.append(
                        await _call(
                            handlers[tool], dict(args), _codex_meta(call_id, item_id)
                        )
                    )
                    if chunk < len(calls):
                        cli.release(1, chunk)
                        await _wait_for(lambda k=chunk: cli.printed(1, k))
                running = set(native.current_turn.open_af_tool_uses)
                cli.release(1, len(calls))
                await loop
        finally:
            await runtime.aclose_all()
        self.assertEqual([r.is_error for r in results], [False, False, False])
        self.assertTrue(all(r.text.startswith(END_TURN_MARKER) for r in results))
        self.assertEqual(
            ["question queued" in r.text for r in results], [True, True, False]
        )
        self.assertIn("earlier message is already pending", results[2].text)
        # Finished calls hold no AF call open (the stall watchdog runs again).
        self.assertEqual(running, set())
        self.assertEqual(len(interactive.widgets), 1)
        metadata = interactive.widgets[0]["input_mode"].metadata
        self.assertTrue(metadata.get("compound"))
        self.assertEqual(
            [t["tool_type"] for t in metadata["tools"]],
            ["clarification", "single_choice"],
        )
        answer = cli.prompt(2)
        self.assertIn("topic: lidar", answer)
        self.assertIn("depth: deep", answer)
        self.assertNotIn("depth_b", answer)


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


class TurnWindowTest(later.unittest.TestCase):
    async def test_the_tool_server_accepts_the_session_only_during_a_turn(
        self,
    ) -> None:
        runtime = _Runtime()
        line = {"type": "thread.started", "thread_id": "th-1"}
        cli = scripted_cli(json.dumps(line) + "\n", "", 0)
        b = CodexCliBackend(
            NativeBackendSpec(cwd=tempfile.mkdtemp(), cli_path=cli),
            runtime_manager=runtime,
        )
        await b.open(_open_request([_tool("t", [])], hooks=None))
        turn_active = runtime.server.turn_active
        self.assertFalse(turn_active())
        during = [turn_active() async for _ in b.run_turn(TurnRequest(text="hi"))]
        self.assertTrue(during and all(during))
        self.assertFalse(turn_active())
        await b.close()
        self.assertEqual(runtime.server.unregistered, ["tok"])


class AfHealthTest(unittest.TestCase):
    def test_af_server_is_required(self) -> None:
        b = _backend(extra_mcp_servers={"docs": {"command": "docs-mcp"}})
        b._mcp_url = "http://127.0.0.1:9/mcp"
        argv = b._argv(TurnRequest(text="h"))
        self.assertIn("mcp_servers.af.required=true", argv)
        self.assertFalse(any(a.startswith("mcp_servers.docs.required") for a in argv))

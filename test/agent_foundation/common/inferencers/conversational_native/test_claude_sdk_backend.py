"""ClaudeSdkBackend option construction, event mapping and hook shapes.

No ``claude`` subprocess is spawned: options are built against the real
``ClaudeAgentOptions``, events are mapped from real SDK message dataclasses,
and hooks are invoked directly with fake input dicts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import tempfile
import unittest
from typing import Iterator
from unittest import mock

import claude_agent_sdk
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    NativeConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
    VendorSessionMissing,
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
    claude_sdk as claude_sdk_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.actor import (
    SessionActor,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BridgeToolSpec,
    InterruptNotAcknowledged,
    L2Channel,
    NativeBackendSpec,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.claude_sdk import (
    ClaudeSdkBackend,
)
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    tool as sdk_tool,
    ToolUseBlock,
)
from claude_agent_sdk._internal.transport import subprocess_cli
from helpers import tool_registry
from later.unittest import TestCase as AsyncTestCase


def _request(
    tools=None,
    resume=False,
    session_id="sess-1",
    l1_path="/tmp/l1.md",
    result_max_chars=0,
):
    return SessionOpenRequest(
        session_id=session_id,
        resume=resume,
        l1_text="L1 BODY",
        l1_path=l1_path,
        tools=tools or [],
        hooks=object(),
        cwd="/work/proj",
        model="opus",
        result_max_chars=result_max_chars,
    )


def _tool_spec(name="write_brief"):
    async def handler(args):
        class _R:
            text = f"ran {name}"
            is_error = False

        return _R()

    return BridgeToolSpec(
        name=name,
        description="d",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


class BuildOptionsTest(unittest.TestCase):
    def _options(self, **kw) -> ClaudeAgentOptions:
        backend = ClaudeSdkBackend(
            NativeBackendSpec(cwd="/work/proj", model="opus", **kw.pop("spec", {}))
        )
        backend._hooks = object()
        request = kw.pop("request", None) or _request(**kw)
        resume_id = request.session_id
        return backend._build_options(
            ClaudeAgentOptions, request, resume_id, resume=request.resume
        )

    def test_keeps_claude_prompt_and_appends_via_file(self) -> None:
        opts = self._options()
        self.assertEqual(
            opts.system_prompt, {"type": "preset", "preset": "claude_code"}
        )
        self.assertEqual(opts.extra_args.get("append-system-prompt-file"), "/tmp/l1.md")
        self.assertNotIn("setting-sources", opts.extra_args)  # inherit by default

    def test_disallows_vendor_ask_and_plan_tools(self) -> None:
        opts = self._options()
        self.assertEqual(
            set(opts.disallowed_tools),
            {"AskUserQuestion", "EnterPlanMode", "ExitPlanMode"},
        )

    def test_fresh_pins_session_id_resume_continues(self) -> None:
        fresh = self._options(request=_request(resume=False, session_id="new-id"))
        self.assertEqual(fresh.session_id, "new-id")
        self.assertIsNone(fresh.resume)
        resumed = self._options(request=_request(resume=True, session_id="old-id"))
        self.assertEqual(resumed.resume, "old-id")

    def test_tools_register_the_af_mcp_server(self) -> None:
        opts = self._options(request=_request(tools=[_tool_spec()]))
        self.assertIn("af", opts.mcp_servers)

    def test_hermetic_sets_setting_sources_empty(self) -> None:
        opts = self._options(spec={"environment": "hermetic"})
        self.assertEqual(opts.extra_args.get("setting-sources"), "")
        self.assertIn("strict-mcp-config", opts.extra_args)

    def test_an_environment_the_backend_cannot_honour_is_refused(self) -> None:
        with self.assertRaises(NativeCapabilityError):
            ClaudeSdkBackend(NativeBackendSpec(environment="sandboxed"))
        ClaudeSdkBackend(NativeBackendSpec(environment="hermetic"))

    def test_mcp_tool_timeout_is_in_env(self) -> None:
        opts = self._options(spec={"mcp_tool_timeout_ms": 999})
        self.assertEqual(opts.env.get("MCP_TOOL_TIMEOUT"), "999")

    def test_model_tag_is_normalized_for_claude_code(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec(cwd="/w", model="claude-opus-4.8"))
        backend._hooks = object()
        request = _request()
        request.model = ""
        opts = backend._build_options(
            ClaudeAgentOptions, request, request.session_id, resume=request.resume
        )
        self.assertEqual(opts.model, "claude-opus-4-8")

    def test_the_claude_binary_is_the_configured_one_else_found_on_path(
        self,
    ) -> None:
        with mock.patch("shutil.which", return_value="/path/bin/claude") as which:
            self.assertEqual(self._options().cli_path, "/path/bin/claude")
            which.assert_called_with("claude")
            pinned = self._options(spec={"cli_path": "/opt/claude"})
            self.assertEqual(pinned.cli_path, "/opt/claude")
        with mock.patch("shutil.which", return_value=None):
            # The Agent SDK then looks for its own.
            self.assertIsNone(self._options().cli_path)

    def test_wide_permission_and_effort_values_route_through_extra_args(self) -> None:
        opts = self._options(spec={"permission_mode": "auto", "effort": "xhigh"})
        self.assertEqual(opts.extra_args.get("permission-mode"), "auto")
        self.assertEqual(opts.extra_args.get("effort"), "xhigh")

    def test_native_permission_and_effort_values_use_typed_fields(self) -> None:
        opts = self._options(
            spec={"permission_mode": "bypassPermissions", "effort": "high"}
        )
        self.assertEqual(opts.permission_mode, "bypassPermissions")
        self.assertEqual(opts.effort, "high")
        self.assertNotIn("permission-mode", opts.extra_args)

    def test_af_tools_are_allowed_and_extra_servers_kept(self) -> None:
        opts = self._options(
            spec={"extra_mcp_servers": {"other": {"type": "stdio", "command": "x"}}},
            request=_request(tools=[_tool_spec()]),
        )
        self.assertEqual(opts.allowed_tools, ["mcp__af"])
        self.assertEqual(set(opts.mcp_servers), {"af", "other"})

    def test_token_streaming_is_enabled(self) -> None:
        self.assertTrue(self._options().include_partial_messages)

    def test_result_size_is_declared_for_claude_code_spill(self) -> None:
        from mcp.types import ToolAnnotations

        wrapped = ClaudeSdkBackend._wrap_tool(
            sdk_tool, _tool_spec(), ToolAnnotations(maxResultSizeChars=500)
        )
        self.assertEqual(wrapped.annotations.maxResultSizeChars, 500)
        self.assertTrue(ClaudeSdkBackend.capabilities.owns_result_spill)


class _SpawnCaptured(Exception):
    pass


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@contextlib.contextmanager
def _captured_spawn() -> Iterator[dict]:
    """Capture, instead of run, the Claude Code process the SDK's transport
    spawns (its argv and environment); the connect then fails."""
    spawned: dict = {}

    async def open_process(cmd, **kwargs):
        spawned.update(argv=list(cmd), env=dict(kwargs["env"]))
        raise _SpawnCaptured()

    with (
        mock.patch.object(subprocess_cli.anyio, "open_process", open_process),
        mock.patch.dict(os.environ, {"CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1"}),
    ):
        yield spawned


class SdkCommandLineTest(AsyncTestCase):
    """The Claude Code command line and environment the Agent SDK's own
    transport builds from the backend's options: ``backend.open()`` runs
    through ``ClaudeSDKClient.connect()`` to the spawn."""

    async def _spawned(self, request=None, **spec) -> tuple[list[str], dict]:
        backend = ClaudeSdkBackend(
            NativeBackendSpec(
                cwd="/work/proj", model="opus", cli_path="/opt/claude/claude", **spec
            )
        )
        with _captured_spawn() as spawned, self.assertRaises(CLIConnectionError):
            await backend.open(request or _request(tools=[_tool_spec()]))
        await backend.close()
        return spawned["argv"], spawned["env"]

    async def test_claude_codes_prompt_is_kept_and_l1_appended_from_its_file(
        self,
    ) -> None:
        argv, _ = await self._spawned()
        self.assertEqual(argv[0], "/opt/claude/claude")
        for flag in (
            "--system-prompt",
            "--system-prompt-file",
            "--append-system-prompt",
        ):
            self.assertNotIn(flag, argv)
        self.assertEqual(_flag_value(argv, "--append-system-prompt-file"), "/tmp/l1.md")
        self.assertNotIn("L1 BODY", " ".join(argv))

    async def test_af_is_an_sdk_mcp_server_and_vendor_ask_and_plan_are_disallowed(
        self,
    ) -> None:
        argv, _ = await self._spawned(
            extra_mcp_servers={"docs": {"type": "stdio", "command": "docs-mcp"}}
        )
        config = json.loads(_flag_value(argv, "--mcp-config"))
        self.assertEqual(
            config["mcpServers"],
            {
                "af": {"type": "sdk", "name": "af"},
                "docs": {"type": "stdio", "command": "docs-mcp"},
            },
        )
        self.assertEqual(_flag_value(argv, "--allowedTools"), "mcp__af")
        self.assertEqual(
            set(_flag_value(argv, "--disallowedTools").split(",")),
            {"AskUserQuestion", "EnterPlanMode", "ExitPlanMode"},
        )
        self.assertIn("--include-partial-messages", argv)

    async def test_no_af_server_without_af_tools(self) -> None:
        argv, _ = await self._spawned(request=_request(tools=[]))
        self.assertNotIn("--mcp-config", argv)
        self.assertNotIn("--allowedTools", argv)

    async def test_a_new_session_pins_its_id_and_a_resume_continues_it(self) -> None:
        fresh, _ = await self._spawned(
            request=_request(tools=[_tool_spec()], session_id="new-id")
        )
        self.assertEqual(_flag_value(fresh, "--session-id"), "new-id")
        self.assertNotIn("--resume", fresh)
        resumed, _ = await self._spawned(
            request=_request(tools=[_tool_spec()], resume=True, session_id="old-id")
        )
        self.assertEqual(_flag_value(resumed, "--resume"), "old-id")
        self.assertNotIn("--session-id", resumed)

    async def test_hermetic_flags_only_in_hermetic_mode(self) -> None:
        inherit, _ = await self._spawned()
        self.assertNotIn("--setting-sources", inherit)
        self.assertNotIn("--strict-mcp-config", inherit)
        hermetic, _ = await self._spawned(environment="hermetic")
        self.assertEqual(hermetic.count("--setting-sources"), 1)
        self.assertEqual(_flag_value(hermetic, "--setting-sources"), "")
        strict = hermetic.index("--strict-mcp-config")
        self.assertTrue(hermetic[strict + 1].startswith("--"))  # a bare flag

    async def test_mcp_tool_timeout_reaches_the_claude_process_environment(
        self,
    ) -> None:
        argv, env = await self._spawned(mcp_tool_timeout_ms=123_000)
        self.assertEqual(env["MCP_TOOL_TIMEOUT"], "123000")
        self.assertNotIn("MCP_TOOL_TIMEOUT", " ".join(argv))

    async def test_the_session_instructions_file_is_private(self) -> None:
        session_dir = os.path.join(tempfile.mkdtemp(), "session")
        native = NativeConversationalInferencer(
            backend={
                "kind": "claude_sdk",
                "cwd": tempfile.mkdtemp(),
                "cli_path": "/opt/claude/claude",
            },
            tool_registry=tool_registry(),
            prior_context={"native_session_dir": session_dir},
        )
        try:
            with _captured_spawn() as spawned, self.assertRaises(CLIConnectionError):
                await native.run_agentic_loop("hi", turn_number=1)
        finally:
            await native.aclose()
        self.assertEqual(stat.S_IMODE(os.stat(session_dir).st_mode), 0o700)
        self.assertEqual(os.listdir(session_dir), ["l1_0.md"])
        path = os.path.join(session_dir, "l1_0.md")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        argv = spawned["argv"]
        self.assertEqual(_flag_value(argv, "--append-system-prompt-file"), path)
        with open(path) as f:
            body = f.read()
        self.assertGreater(len(body), 200)
        self.assertNotIn(body[:200], "\0".join(argv))


class EventMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ClaudeSdkBackend(NativeBackendSpec())

    def test_main_thread_text_and_af_tool_ids(self) -> None:
        msg = AssistantMessage(
            content=[
                TextBlock(text="Hello "),
                TextBlock(text="world"),
                ToolUseBlock(id="t1", name="mcp__af__write_brief", input={}),
                ToolUseBlock(id="t2", name="Read", input={}),
            ],
            model="opus",
            message_id="m1",
            uuid="u1",
            session_id="s1",
        )
        events = self.backend._map(msg)
        self.assertIsInstance(events[0], SessionStarted)
        self.assertEqual(events[0].session_id, "s1")
        # Text arrives as stream deltas; the message event carries the whole
        # text for the driver to use only when nothing was streamed.
        self.assertFalse(any(isinstance(e, TextDelta) for e in events))
        end = next(e for e in events if isinstance(e, MessageEnd))
        self.assertEqual(end.text, "Hello world")
        self.assertEqual(end.af_tool_use_ids, ("t1",))
        self.assertEqual(set(end.tool_use_ids), {"t1", "t2"})
        self.assertEqual(end.message_uuid, "u1")

    def test_subagent_message_is_ignored(self) -> None:
        msg = AssistantMessage(
            content=[TextBlock(text="sub work")],
            model="opus",
            parent_tool_use_id="parent-1",
            message_id="m2",
            uuid="u2",
        )
        self.assertEqual(self.backend._map(msg), [])

    def test_failed_af_at_init_is_an_actionable_error(self) -> None:
        self.backend._expects_af = True
        msg = SystemMessage(
            subtype="init", data={"mcp_servers": [{"name": "af", "status": "failed"}]}
        )
        events = self.backend._map(msg)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], VendorError)
        self.assertIn("allowManagedMcpServersOnly", events[0].message)
        self.assertTrue(self.backend._stop_requested)

    def test_init_must_show_af_connected_with_its_tools(self) -> None:
        self.backend._expects_af = True
        connected = {"name": "af", "status": "connected", "source": "sdk"}
        cases = {
            "unlisted": ({"mcp_servers": [], "tools": []}, True),
            "no tools": ({"mcp_servers": [connected], "tools": ["Read"]}, True),
            "healthy": (
                {"mcp_servers": [connected], "tools": ["Read", "mcp__af__write_brief"]},
                False,
            ),
        }
        for name, (data, fails) in cases.items():
            with self.subTest(name):
                events = self.backend._map(SystemMessage(subtype="init", data=data))
                self.assertEqual(bool(events), fails)

    def test_init_is_not_checked_without_af_tools(self) -> None:
        msg = SystemMessage(subtype="init", data={"mcp_servers": [], "tools": []})
        self.assertEqual(self.backend._map(msg), [])

    def test_compaction_system_message(self) -> None:
        msg = SystemMessage(subtype="compact_boundary", data={})
        self.assertEqual([type(e) for e in self.backend._map(msg)], [Compaction])

    def test_result_message_becomes_turn_end(self) -> None:
        msg = ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=2,
            session_id="s9",
            stop_reason="end_turn",
            total_cost_usd=0.01,
            usage={"input_tokens": 5},
            result="final text",
        )
        events = self.backend._map(msg)
        self.assertIsInstance(events[0], SessionStarted)
        end = events[-1]
        self.assertIsInstance(end, TurnEnd)
        self.assertEqual(end.session_id, "s9")
        self.assertEqual(end.result_text, "final text")
        self.assertEqual(end.num_turns, 2)
        self.assertFalse(end.is_error)

    def test_session_started_emitted_once(self) -> None:
        self.backend._session_started("s1")
        self.assertEqual(self.backend._session_started("s1"), [])

    def test_stream_events_become_text_deltas_of_the_api_message(self) -> None:
        start = StreamEvent(
            uuid="e1",
            session_id="s1",
            event={"type": "message_start", "message": {"id": "msg_1"}},
        )
        delta = StreamEvent(
            uuid="e2",
            session_id="s1",
            event={
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hi"},
            },
        )
        self.assertEqual(self.backend._map(start), [])
        self.assertEqual(
            self.backend._map(delta), [TextDelta(message_id="msg_1", text="Hi")]
        )

    def test_subagent_stream_events_are_ignored(self) -> None:
        delta = StreamEvent(
            uuid="e3",
            session_id="s1",
            parent_tool_use_id="p1",
            event={
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "sub"},
            },
        )
        self.assertEqual(self.backend._map(delta), [])

    def test_missing_session_result_is_classified(self) -> None:
        self.backend._stderr.append("No conversation found with session ID: abc")
        msg = ResultMessage(
            subtype="error_during_execution",
            duration_ms=0,
            duration_api_ms=0,
            is_error=True,
            num_turns=0,
            session_id="abc",
        )
        events = self.backend._map(msg)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], VendorError)
        self.assertTrue(events[0].session_missing)
        self.assertFalse(events[0].submitted)

    def test_a_prompt_blocked_by_a_hook_fails_the_turn_unsubmitted(self) -> None:
        notice = (
            "UserPromptSubmit operation blocked by hook:\nAF did not answer.\n\n"
            "Original prompt: my private question"
        )
        informational = SystemMessage(
            subtype="informational",
            data={
                "type": "system",
                "subtype": "informational",
                "content": notice,
                "prevent_continuation": True,
            },
        )
        self.assertEqual(self.backend._map(informational), [])
        result = ResultMessage(
            subtype="success",
            duration_ms=0,
            duration_api_ms=0,
            is_error=False,
            num_turns=0,
            session_id="s1",
            result=notice,
        )
        events = self.backend._map(result)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], VendorError)
        self.assertFalse(events[0].submitted)
        self.assertNotIn("my private question", events[0].message)

    def test_other_error_results_stay_turn_end(self) -> None:
        msg = ResultMessage(
            subtype="error_during_execution",
            duration_ms=0,
            duration_api_ms=0,
            is_error=True,
            num_turns=0,
            session_id="abc",
        )
        end = self.backend._map(msg)[-1]
        self.assertIsInstance(end, TurnEnd)
        self.assertTrue(end.is_error)


class _StubHooks:
    def __init__(self, l2="", deny=None, stop=False) -> None:
        self._l2, self._deny, self._stop = l2, deny, stop
        self.compacted = False

    def l2_for_turn(self):
        return self._l2

    async def before_af_tool(self, name, tool_use_id, agent_id):
        return self._deny

    async def after_af_tool(self, name, tool_use_id):
        return self._stop

    def on_compaction(self):
        self.compacted = True


class HookShapeTest(AsyncTestCase):
    def _backend(self, hooks) -> ClaudeSdkBackend:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._hooks = hooks
        return backend

    async def test_user_prompt_injects_l2_as_additional_context(self) -> None:
        backend = self._backend(_StubHooks(l2="<af_context>ctx</af_context>"))
        out = await backend._on_user_prompt({"prompt": "hi"}, None, None)
        self.assertEqual(
            out["hookSpecificOutput"]["additionalContext"],
            "<af_context>ctx</af_context>",
        )

    async def test_user_prompt_sends_nothing_when_l2_empty(self) -> None:
        backend = self._backend(_StubHooks(l2=""))
        self.assertEqual(await backend._on_user_prompt({"prompt": "x"}, None, None), {})

    async def test_pre_tool_deny_for_subagent(self) -> None:
        backend = self._backend(_StubHooks(deny="main agent only"))
        out = await backend._on_pre_tool(
            {
                "tool_name": "mcp__af__write_brief",
                "tool_use_id": "t1",
                "agent_id": "a1",
            },
            "t1",
            None,
        )
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")

    async def test_pre_tool_ignores_non_af_tools(self) -> None:
        backend = self._backend(_StubHooks(deny="should not matter"))
        self.assertEqual(
            await backend._on_pre_tool({"tool_name": "Read"}, None, None), {}
        )

    async def test_post_tool_stops_turn(self) -> None:
        backend = self._backend(_StubHooks(stop=True))
        out = await backend._on_post_tool(
            {"tool_name": "mcp__af__clarification", "tool_use_id": "t1"}, "t1", None
        )
        self.assertFalse(out["continue_"])

    async def test_a_failed_af_call_finishes_without_a_stop_decision(self) -> None:
        # Claude Code runs PostToolUseFailure, not PostToolUse, for an AF call
        # that returned an error, and ignores ``continue: false`` from it.
        hooks = _StubHooks(stop=True)
        finished = []
        original = hooks.after_af_tool

        async def after(name, tool_use_id):
            finished.append((name, tool_use_id))
            return await original(name, tool_use_id)

        hooks.after_af_tool = after
        backend = self._backend(hooks)
        matcher = backend._build_hooks()["PostToolUseFailure"][0]
        self.assertEqual(matcher.matcher, "mcp__af__.*")
        out = await matcher.hooks[0](
            {
                "hook_event_name": "PostToolUseFailure",
                "tool_name": "mcp__af__write_brief",
                "tool_use_id": "t1",
                "error": "refused",
            },
            "t1",
            None,
        )
        self.assertEqual(out, {})
        self.assertEqual(finished, [("mcp__af__write_brief", "t1")])

    async def test_pre_compact_notifies(self) -> None:
        hooks = _StubHooks()
        backend = self._backend(hooks)
        await backend._on_pre_compact({}, None, None)
        self.assertTrue(hooks.compacted)


class _FailingHooks(_StubHooks):
    def l2_for_turn(self):
        raise RuntimeError("no host")

    async def before_af_tool(self, name, tool_use_id, agent_id):
        raise RuntimeError("no host")

    async def after_af_tool(self, name, tool_use_id):
        raise RuntimeError("no host")


class HookFailsClosedTest(AsyncTestCase):
    """Claude Code runs the tool when an SDK hook callback raises, so the
    callbacks answer like the CLI relay does when AF cannot decide."""

    async def _call(self, event: str, input_data: dict) -> dict:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._hooks = _FailingHooks()
        hook = backend._build_hooks()[event][0].hooks[0]
        return await hook({"hook_event_name": event, **input_data}, "t1", None)

    async def test_af_tool_call_is_denied(self) -> None:
        out = await self._call(
            "PreToolUse", {"tool_name": "mcp__af__write_brief", "tool_use_id": "t1"}
        )
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"], "deny")

    async def test_turn_stops_after_an_af_tool(self) -> None:
        out = await self._call(
            "PostToolUse", {"tool_name": "mcp__af__clarification", "tool_use_id": "t1"}
        )
        self.assertIs(out["continue_"], False)

    async def test_prompt_is_blocked_instead_of_sent_without_its_context(
        self,
    ) -> None:
        out = await self._call("UserPromptSubmit", {"prompt": "hi"})
        self.assertEqual(out["decision"], "block")


_AF_TOOLS = [{"name": "write_brief"}]


class _StatusClient:
    """``get_mcp_status`` answers in sequence (the last one repeats); ``None``
    is a response that does not list ``af`` yet."""

    def __init__(self, statuses, raises=False) -> None:
        self._statuses = list(statuses)
        self._raises = raises
        self.polls = 0

    async def get_mcp_status(self):
        self.polls += 1
        if self._raises:
            raise RuntimeError("unsupported")
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        other = {"name": "plugin:other", "status": "connected"}
        return {
            "mcpServers": [other]
            + ([] if status is None else [{"name": "af", **status}])
        }


class HealthCheckTest(AsyncTestCase):
    async def test_pending_then_connected_passes(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._client = _StatusClient(
            [{"status": "pending"}, {"status": "connected", "tools": _AF_TOOLS}]
        )
        await backend._check_af_connected()
        self.assertIsNotNone(backend._client)

    async def test_not_yet_listed_server_is_polled_until_connected(self) -> None:
        # SDK 0.1.58 + claude 2.1.288 list the in-process server only a few
        # hundred ms after connect().
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._client = _StatusClient(
            [None, None, {"status": "connected", "tools": _AF_TOOLS}]
        )
        await backend._check_af_connected()
        self.assertEqual(backend._client.polls, 3)

    async def test_failed_server_fails_the_open_with_a_policy_hint(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._client = _StatusClient(
            [{"status": "failed", "error": "blocked by policy"}]
        )
        with self.assertRaises(NativeCapabilityError) as ctx:
            await backend._check_af_connected()
        self.assertIn("allowManagedMcpServersOnly", str(ctx.exception))
        self.assertIn("blocked by policy", str(ctx.exception))
        self.assertIsNone(backend._client)

    async def test_server_that_never_connects_fails_the_open(self) -> None:
        for status in (None, {"status": "pending"}, {"status": "connected"}):
            with self.subTest(status=status):
                backend = ClaudeSdkBackend(NativeBackendSpec())
                backend._client = _StatusClient([status])
                with (
                    mock.patch.object(claude_sdk_module, "_HEALTH_TIMEOUT_S", 0.3),
                    self.assertRaises(NativeCapabilityError) as ctx,
                ):
                    await backend._check_af_connected()
                self.assertIn("'af'", str(ctx.exception))
                self.assertIsNone(backend._client)

    async def test_unavailable_status_api_fails_the_open(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._client = _StatusClient([{}], raises=True)
        with self.assertRaises(NativeCapabilityError) as ctx:
            await backend._check_af_connected()
        self.assertIn("get_mcp_status failed", str(ctx.exception))


class _InitLostClient:
    """A connected client whose turn starts with an init that lacks ``af``."""

    def __init__(self) -> None:
        self.interrupts = 0

    async def query(self, text, session_id="default"):
        pass

    async def receive_response(self):
        yield SystemMessage(subtype="init", data={"mcp_servers": [], "tools": []})
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=0,
            duration_api_ms=0,
            is_error=True,
            num_turns=0,
            session_id="s1",
        )

    async def interrupt(self):
        self.interrupts += 1


class TurnHealthTest(AsyncTestCase):
    async def test_turn_without_af_at_init_is_interrupted_and_fails(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._client = _InitLostClient()
        backend._expects_af = True
        events = [e async for e in backend.run_turn(TurnRequest(text="hi"))]
        self.assertIsInstance(events[0], VendorError)
        self.assertIn("'not listed'", events[0].message)
        self.assertEqual(backend._client.interrupts, 1)


class _StoppableClient:
    """A turn that streams until ``interrupt()``; the stop takes effect but
    the interrupt request itself can be made to fail."""

    def __init__(self, interrupt_fails: bool) -> None:
        self.streaming = asyncio.Event()
        self._stop = asyncio.Event()
        self._interrupt_fails = interrupt_fails

    async def query(self, text, session_id="default"):
        pass

    async def receive_response(self):
        self.streaming.set()
        await self._stop.wait()
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=0,
            duration_api_ms=0,
            is_error=True,
            num_turns=1,
            session_id="s1",
        )

    async def interrupt(self):
        self._stop.set()
        if self._interrupt_fails:
            raise RuntimeError("control channel closed")


class _ModelClient:
    def __init__(self, fails: bool) -> None:
        self._fails = fails
        self.models: list = []

    async def set_model(self, model):
        if self._fails:
            raise RuntimeError("unknown model")
        self.models.append(model)


class SetModelTest(AsyncTestCase):
    async def test_the_live_client_switches_model(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._client = _ModelClient(fails=False)
        await backend.set_model("sonnet")
        self.assertEqual(len(backend._client.models), 1)

    async def test_a_refused_switch_raises_instead_of_running_on(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._client = _ModelClient(fails=True)
        with self.assertRaises(RuntimeError):
            await backend.set_model("sonnet")


class InterruptTest(AsyncTestCase):
    async def test_failed_interrupt_request_is_not_acknowledged(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._client = _StoppableClient(interrupt_fails=True)
        with self.assertRaises(InterruptNotAcknowledged) as ctx:
            await backend.interrupt()
        self.assertIn("control channel closed", str(ctx.exception))

    async def test_the_turn_records_whether_the_vendor_acknowledged_the_stop(
        self,
    ) -> None:
        for fails in (False, True):
            with self.subTest(interrupt_fails=fails):
                backend = ClaudeSdkBackend(NativeBackendSpec())
                client = _StoppableClient(interrupt_fails=fails)

                async def open_(request, client=client, backend=backend):
                    backend._client = client

                with mock.patch.object(backend, "open", open_):
                    actor = SessionActor(backend, _request(), drain_timeout_s=5)
                    await actor.start()
                turn = asyncio.ensure_future(
                    _drain(actor.run_turn(TurnRequest(text="hi")))
                )
                await asyncio.wait_for(client.streaming.wait(), 5)
                await actor.interrupt()
                await asyncio.wait_for(turn, 10)
                self.assertTrue(actor.last_outcome.interrupted)
                self.assertIs(actor.last_outcome.acknowledged, not fails)
                await actor.close()


async def _drain(events) -> list:
    return [e async for e in events]


class _MissingSessionClient:
    def __init__(self, options) -> None:
        self._options = options

    async def connect(self) -> None:
        self._options.stderr("No conversation found with session ID: gone")
        raise RuntimeError("Command failed with exit code 1")

    async def disconnect(self) -> None:
        pass


class _RecordingClient:
    opened: list = []

    def __init__(self, options) -> None:
        self.options = options
        _RecordingClient.opened.append(options)

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


class OpenTest(AsyncTestCase):
    async def test_a_fork_resumes_the_forked_transcript(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec(cwd="/w"))
        _RecordingClient.opened = []

        async def fork(source, boundary):
            self.assertEqual((source, boundary), ("src-id", "m-uuid"))
            return "forked-id"

        request = _request(resume=False, session_id="")
        request.fork_from = ("src-id", "m-uuid")
        with (
            mock.patch.object(claude_agent_sdk, "ClaudeSDKClient", _RecordingClient),
            mock.patch.object(backend, "_fork", fork),
        ):
            await backend.open(request)
        await backend.close()
        options = _RecordingClient.opened[-1]
        self.assertEqual(options.resume, "forked-id")
        self.assertIsNone(options.session_id)
        self.assertEqual(backend.session_id, "forked-id")

    async def test_resume_of_a_missing_session_raises_session_missing(self) -> None:
        backend = ClaudeSdkBackend(NativeBackendSpec(cwd="/w"))
        with mock.patch.object(
            claude_agent_sdk, "ClaudeSDKClient", _MissingSessionClient
        ):
            with self.assertRaises(VendorSessionMissing):
                await backend.open(_request(resume=True, session_id="gone"))
        await backend.close()


_L2 = '<af_context nonce="n">state</af_context>'


class _HookTurnClient:
    """A turn shaped like claude 2.1.288's: the UserPromptSubmit callback runs
    (when ``runs_hook``) before ``init``, then the model answers — unless it
    was interrupted at ``init``. A ``local`` command (claude 2.1.289) answers
    with a ``<synthetic>`` message instead, without the model."""

    def __init__(
        self, backend, *, runs_hook: bool, init: bool = True, local: bool = False
    ) -> None:
        self._backend = backend
        self._runs_hook = runs_hook
        self._init = init
        self._local = local
        self.interrupts = 0

    async def query(self, text, session_id="default"):
        if self._runs_hook:
            hook = self._backend._build_hooks()["UserPromptSubmit"][0].hooks[0]
            await hook(
                {"hook_event_name": "UserPromptSubmit", "prompt": text}, None, None
            )

    async def receive_response(self):
        if self._init:
            yield SystemMessage(subtype="init", data={"mcp_servers": [], "tools": []})
        if self._local:
            yield AssistantMessage(
                content=[TextBlock(text="output")],
                model="<synthetic>",
                uuid="u0",
                session_id="s1",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=0,
                duration_api_ms=0,
                is_error=False,
                num_turns=0,
                session_id="s1",
            )
            return
        if not self.interrupts:
            yield StreamEvent(
                uuid="e1",
                session_id="s1",
                event={"type": "message_start", "message": {"id": "msg_1"}},
            )
            yield AssistantMessage(
                content=[TextBlock(text="answer")],
                model="opus",
                message_id="msg_1",
                uuid="u1",
                session_id="s1",
            )
        yield ResultMessage(
            subtype="error_during_execution" if self.interrupts else "success",
            duration_ms=0,
            duration_api_ms=0,
            is_error=bool(self.interrupts),
            num_turns=1,
            session_id="s1",
        )

    async def interrupt(self):
        self.interrupts += 1


class HookLivenessTest(AsyncTestCase):
    """A managed policy that drops AF's hooks (allowManagedHooksOnly) must not
    silently cost the turn its context, its turn stop and its subagent guard."""

    async def _turn(self, request, **client_kw) -> tuple[list, _HookTurnClient]:
        backend = ClaudeSdkBackend(NativeBackendSpec())
        backend._hooks = _StubHooks(l2=request.l2_text)
        client = _HookTurnClient(backend, **client_kw)
        backend._client = client
        return [e async for e in backend.run_turn(request)], client

    async def test_init_without_the_prompt_hook_stops_the_turn(self) -> None:
        events, client = await self._turn(
            TurnRequest(text="hi", l2_text=_L2), runs_hook=False
        )
        self.assertIsInstance(events[0], VendorError)
        self.assertIn("allowManagedHooksOnly", events[0].message)
        self.assertTrue(events[0].submitted)
        self.assertEqual(client.interrupts, 1)
        self.assertFalse(any(isinstance(e, MessageEnd) for e in events))

    async def test_a_turn_whose_hook_ran_proceeds(self) -> None:
        events, client = await self._turn(
            TurnRequest(text="hi", l2_text=_L2), runs_hook=True
        )
        self.assertFalse(any(isinstance(e, VendorError) for e in events))
        self.assertIsInstance(events[-1], TurnEnd)
        self.assertEqual(client.interrupts, 0)

    async def test_the_first_model_output_is_checked_when_no_init_came(
        self,
    ) -> None:
        events, client = await self._turn(
            TurnRequest(text="hi", l2_text=_L2), runs_hook=False, init=False
        )
        self.assertIsInstance(events[0], VendorError)
        self.assertIn("UserPromptSubmit", events[0].message)
        self.assertEqual(client.interrupts, 1)

    async def test_a_turn_without_turn_context_still_needs_the_prompt_hook(
        self,
    ) -> None:
        # An L2 is sent only when due, but AF's hook runs on every turn of the
        # hook channel (it returns no context when none is due).
        events, client = await self._turn(TurnRequest(text="hi"), runs_hook=False)
        self.assertIsInstance(events[0], VendorError)
        self.assertIn("allowManagedHooksOnly", events[0].message)
        self.assertTrue(events[0].submitted)
        self.assertEqual(client.interrupts, 1)
        events, client = await self._turn(TurnRequest(text="hi"), runs_hook=True)
        self.assertFalse(any(isinstance(e, VendorError) for e in events))
        self.assertIsInstance(events[-1], TurnEnd)

    async def test_the_hook_is_not_required_off_its_channel_or_for_a_local_command(
        self,
    ) -> None:
        cases = {
            "envelope": TurnRequest(
                text=f"{_L2}\nhi", l2_text=_L2, channel=L2Channel.ENVELOPE
            ),
            # Claude Code runs its local commands without UserPromptSubmit.
            "local command": TurnRequest(text="/compact", l2_text=_L2),
            "local command with arguments": TurnRequest(text="/context all"),
        }
        for name, request in cases.items():
            with self.subTest(name):
                events, client = await self._turn(request, runs_hook=False)
                self.assertFalse(any(isinstance(e, VendorError) for e in events))
                self.assertEqual(client.interrupts, 0)

    async def test_other_slash_text_needs_the_prompt_hook_once_the_model_starts(
        self,
    ) -> None:
        # Unknown names, paths, user commands and skills reach the model
        # through UserPromptSubmit (claude 2.1.289): checked at the first
        # model output, not at init.
        for text in ("/foo", "/tmp/notes.txt is gone", "/probecmd hello"):
            with self.subTest(text=text):
                events, client = await self._turn(
                    TurnRequest(text=text, l2_text=_L2), runs_hook=False
                )
                self.assertIsInstance(events[0], VendorError)
                self.assertIn("allowManagedHooksOnly", events[0].message)
                self.assertEqual(client.interrupts, 1)
                events, client = await self._turn(
                    TurnRequest(text=text, l2_text=_L2), runs_hook=True
                )
                self.assertFalse(any(isinstance(e, VendorError) for e in events))
                self.assertEqual(client.interrupts, 0)

    async def test_an_unlisted_local_command_runs_without_the_prompt_hook(
        self,
    ) -> None:
        # `/release-notes` runs locally like `/context`, but is not passed
        # through: no hook, no model, so nothing to stop.
        events, client = await self._turn(
            TurnRequest(text="/release-notes", l2_text=_L2),
            runs_hook=False,
            local=True,
        )
        self.assertFalse(any(isinstance(e, VendorError) for e in events))
        self.assertIsInstance(events[-1], TurnEnd)
        self.assertEqual(client.interrupts, 0)

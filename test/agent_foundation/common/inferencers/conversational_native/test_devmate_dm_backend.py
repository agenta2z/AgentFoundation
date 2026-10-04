"""DevmateDmBackend argv construction and devmate-sdk-events mapping (no subprocess).

Event shapes mirror real ``dm -p --output-format devmate-sdk-events`` output
captured in scripts/native_spikes/s14_dm_socket_mcp.py. Real end-to-end coverage
(stdio-relay MCP tool call with a real model) is the s14 spike; a real-model SOP
run is scripts/native_spikes/e2e_research_sop.py --backend devmate_dm.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

import later.unittest
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    BridgeResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    SessionStarted,
    TextDelta,
    TurnEnd,
    VendorError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session import (
    devmate_dm,
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
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.devmate_dm import (
    DevmateDmBackend,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.runtime import (
    NativeRuntimeManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.turn import (
    QueuedWidget,
    TurnOrigin,
    TurnScope,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
    common as devmate_common,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from test_cli_runner import (
    agent_printing,
    cancel_through_actor,
    launcher_with_agent,
    pid_alive,
)


class _FakeSocketServer:
    async def register(self, tools: list) -> tuple[str, str]:
        return "/tmp/af_fake.sock", "tok-123"


class _FakeRuntime:
    def __init__(self) -> None:
        self.socket_server = None

    async def ensure_socket_server(self) -> _FakeSocketServer:
        self.socket_server = _FakeSocketServer()
        return self.socket_server


async def _noop_handler(_args: dict) -> dict:
    return {}


def _backend(**spec) -> DevmateDmBackend:
    b = DevmateDmBackend(NativeBackendSpec(cwd="/work", model="sonnet", **spec))
    b._l1_text = "L1 persona and SOP catalog"
    return b


def _step(backend: DevmateDmBackend, *objs: dict) -> list:
    out = []
    for obj in objs:
        out.extend(backend._map(obj))
    return out


class CapabilitiesTest(unittest.TestCase):
    def test_socket_caller_tools_and_envelope_only(self) -> None:
        caps = DevmateDmBackend.capabilities
        self.assertEqual(caps.caller_tools, CallerTools.SOCKET)
        self.assertEqual(caps.l2_channels, (L2Channel.ENVELOPE,))
        self.assertTrue(caps.pinned_session_id)
        self.assertFalse(caps.exact_fork)
        self.assertFalse(caps.turn_stop_hook)
        # Envelope is the only channel; it is used only when explicitly allowed.
        self.assertIsNone(caps.preferred_l2_channel(envelope_allowed=False))
        self.assertEqual(
            caps.preferred_l2_channel(envelope_allowed=True), L2Channel.ENVELOPE
        )

    def test_subagents_never_see_af_tools_and_hooks_are_unavailable(self) -> None:
        evidence = DevmateDmBackend.capabilities.evidence
        self.assertEqual(evidence["subagent_deny"], Evidence.VERIFIED)
        self.assertEqual(evidence["hooks"], Evidence.UNSUPPORTED)
        self.assertEqual(evidence["hermetic"], Evidence.UNSUPPORTED)

    def test_resume_continuity_is_verified(self) -> None:
        # Every turn after the first runs with --resume; dm's scripted model
        # showed a resumed turn's request carrying the earlier turns.
        self.assertTrue(DevmateDmBackend.capabilities.pinned_session_id)
        self.assertEqual(
            DevmateDmBackend.capabilities.evidence["resume"], Evidence.VERIFIED
        )

    def test_l1_is_relied_on_and_verified(self) -> None:
        # L1 rides --append-system-prompt; dm's logged model request showed it
        # in the system message on fresh, resumed and restarted turns.
        caps = DevmateDmBackend.capabilities
        self.assertIn("l1", caps.relies_on)
        self.assertEqual(caps.evidence["l1"], Evidence.VERIFIED)
        self.assertIn("--append-system-prompt", caps.l1_route)

    def test_hermetic_is_refused_at_construction(self) -> None:
        with self.assertRaises(NativeCapabilityError) as ctx:
            DevmateDmBackend(
                NativeBackendSpec(kind="devmate_dm", environment="hermetic")
            )
        self.assertIn("environment: inherit", str(ctx.exception))
        DevmateDmBackend(NativeBackendSpec(kind="devmate_dm", environment="inherit"))


class ArgvTest(unittest.TestCase):
    def test_fresh_turn_pins_session_and_sets_format(self) -> None:
        b = _backend()
        b._session_id = "pin-1"
        b._mcp_servers_json = '[{"type":"stdio","name":"af","command":"python3","args":["relay.py","/t/af.sock"]}]'
        argv = b._argv(TurnRequest(text="hello", channel=L2Channel.ENVELOPE))
        self.assertIn("-p", argv)
        self.assertEqual(argv[argv.index("--output-format") + 1], "devmate-sdk-events")
        self.assertEqual(argv[argv.index("--agent-harness") + 1], "native")
        self.assertEqual(
            argv[argv.index("--append-system-prompt") + 1], "L1 persona and SOP catalog"
        )
        self.assertEqual(argv[argv.index("--mcp-servers") + 1], b._mcp_servers_json)
        self.assertEqual(argv[argv.index("--session-id") + 1], "pin-1")
        self.assertNotIn("--resume", argv)
        self.assertEqual(argv[-1], "hello")

    def test_resume_after_first_turn(self) -> None:
        b = _backend()
        b._session_id = "s-1"
        b._started_emitted = True
        argv = b._argv(TurnRequest(text="again"))
        self.assertEqual(argv[argv.index("--resume") + 1], "s-1")
        self.assertNotIn("--session-id", argv)

    def test_custom_agent_harness_from_extra(self) -> None:
        b = _backend(extra={"agent_harness": "claude"})
        b._session_id = "x"
        argv = b._argv(TurnRequest(text="h"))
        self.assertEqual(argv[argv.index("--agent-harness") + 1], "claude")

    def test_no_mcp_servers_when_unset(self) -> None:
        b = _backend()
        b._session_id = "x"
        argv = b._argv(TurnRequest(text="h"))
        self.assertNotIn("--mcp-servers", argv)

    def test_cats_file_adds_workflow_flag(self) -> None:
        b = _backend()
        b._session_id = "x"
        b._cats_file = "/tmp/cats.json"  # set from spec.extra/env in open()
        argv = b._argv(TurnRequest(text="h"))
        self.assertEqual(argv[argv.index("--cats-file") + 1], "/tmp/cats.json")

    def test_no_cats_file_flag_when_unset(self) -> None:
        b = _backend()
        b._session_id = "x"
        self.assertNotIn("--cats-file", b._argv(TurnRequest(text="h")))

    def test_the_dm_binary_is_the_configured_one_else_found_on_path(self) -> None:
        request = TurnRequest(text="h")
        with mock.patch("shutil.which", return_value="/path/bin/dm") as which:
            self.assertEqual(_backend()._argv(request)[0], "/path/bin/dm")
            which.assert_called_with("dm")
            self.assertEqual(_backend(cli_path="/opt/dm")._argv(request)[0], "/opt/dm")
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(_backend()._argv(request)[0], "dm")


class ModelTagTest(unittest.TestCase):
    """dm gets the model id the classic dm path would send for the same tag:
    the Devmate tag resolution, then dm's own names (``-long`` for 1M)."""

    def _model(self, tag: str) -> str:
        b = _backend()
        b._spec.model = tag
        argv = b._argv(TurnRequest(text="h"))
        return argv[argv.index("--model") + 1]

    def test_claude_aliases_become_dm_model_ids(self) -> None:
        for tag, expected in (
            ("opus[1m]", "claude-opus-4.7-long"),
            ("claude-opus-4-7[1m]", "claude-opus-4.7-long"),
            ("opus", "opus"),
            ("claude-sonnet-4-6", "claude-sonnet-4.6-long"),
            ("claude-opus-4-6-20260204", "claude-opus-4.6"),
            ("claude-haiku-4.5", "claude-haiku-4.5"),
            ("gpt-5-5", "gpt-5-5"),
        ):
            with self.subTest(tag=tag):
                self.assertEqual(self._model(tag), expected)

    def test_a_model_switch_is_resolved_on_the_next_turn(self) -> None:
        b = _backend()
        asyncio.run(b.set_model("claude-opus-4-6[1m]"))
        argv = b._argv(TurnRequest(text="h"))
        self.assertEqual(argv[argv.index("--model") + 1], "claude-opus-4.6-long")

    def test_matches_the_classic_dm_path(self) -> None:
        classic = DevmateCliInferencer
        self.assertEqual(devmate_common.DM_MODEL_MAP, classic._DM_MODEL_MAP)
        self.assertEqual(devmate_common.DM_KNOWN_MODELS, classic._DM_KNOWN_MODELS)
        self.assertEqual(devmate_common.DM_DEFAULT_MODEL, classic._DM_DEFAULT_MODEL)
        for tag in ("opus[1m]", "claude-opus-4-7", "sonnet", "claude-unknown-9"):
            with self.subTest(tag=tag):
                # Classic resolves a model_id with resolve_model_tag, then maps
                # it with _resolve_dm_model (class tables only).
                expected = classic._resolve_dm_model(
                    classic, devmate_common.resolve_model_tag(tag)
                )
                self.assertEqual(devmate_common.resolve_dm_model(tag), expected)


class CatsFileResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for name in ("DM_CATS_FILE", "PREMINTED_CATS_FILE"):
            os.environ.pop(name, None)
        self.premint = mock.patch.object(
            devmate_dm, "_PREMINTED_CATS", "/nonexistent/cats"
        )
        self.premint.start()

    def tearDown(self) -> None:
        self.premint.stop()
        self.env.stop()

    def test_backend_option_wins(self) -> None:
        os.environ["DM_CATS_FILE"] = __file__
        self.assertEqual(
            devmate_dm.resolve_cats_file({"cats_file": "/x.json"}), "/x.json"
        )

    def test_environment_files_are_used_only_when_present(self) -> None:
        os.environ["DM_CATS_FILE"] = "/nonexistent/dm.json"
        self.assertIsNone(devmate_dm.resolve_cats_file({}))
        with tempfile.NamedTemporaryFile() as cats:
            os.environ["PREMINTED_CATS_FILE"] = cats.name
            self.assertEqual(devmate_dm.resolve_cats_file({}), cats.name)

    def test_devinfra_preminted_file(self) -> None:
        with tempfile.NamedTemporaryFile() as cats:
            with mock.patch.object(devmate_dm, "_PREMINTED_CATS", cats.name):
                self.assertEqual(devmate_dm.resolve_cats_file({}), cats.name)


class MapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.b = _backend()

    def test_session_start_emits_session_started_once(self) -> None:
        events = self.b._map(
            {"event": {"session_start": {"session": {"id": "sess-9"}}}}
        )
        self.assertEqual([type(e) for e in events], [SessionStarted])
        self.assertEqual(events[0].session_id, "sess-9")
        # A later session_update with the same id does not re-emit.
        again = self.b._map(
            {"event": {"session_update": {"session": {"id": "sess-9"}}}}
        )
        self.assertEqual(again, [])

    def test_llm_action_then_step_end_make_text_and_message(self) -> None:
        self.b._started_emitted = True
        events = _step(
            self.b,
            {"event": {"step_start": {"step": {"id": "s1", "number": 1}}}},
            {
                "event": {
                    "action_end": {
                        "action": {
                            "id": "a0",
                            "step_id": "s1",
                            "variant": {"llm_action": {"model": "m"}},
                            "output": {"info": "Hello there"},
                        }
                    }
                }
            },
            {"event": {"step_end": {"step": {"id": "s1"}}}},
        )
        self.assertIsInstance(events[0], TextDelta)
        self.assertEqual(events[0].text, "Hello there")
        end = events[-1]
        self.assertIsInstance(end, MessageEnd)
        self.assertEqual(end.text, "Hello there")
        self.assertEqual(end.tool_use_ids, ())

    def test_af_tool_use_action_tracked_in_message_end(self) -> None:
        self.b._started_emitted = True
        events = _step(
            self.b,
            {"event": {"step_start": {"step": {"id": "s2"}}}},
            {
                "event": {
                    "action_end": {
                        "action": {
                            "id": "a1",
                            "step_id": "s2",
                            "variant": {"llm_action": {"model": "m"}},
                            "output": {"info": "Let me ask"},
                        }
                    }
                }
            },
            {
                "event": {
                    "action_end": {
                        "action": {
                            "id": "a2",
                            "step_id": "s2",
                            "variant": {
                                "tool_use_action": {
                                    "tool_name": "mcp__af__clarification",
                                    "tool_use_id": "t1",
                                }
                            },
                            "output": {"info": "queued"},
                        }
                    }
                }
            },
            {
                "event": {
                    "action_end": {
                        "action": {
                            "id": "a3",
                            "step_id": "s2",
                            "variant": {
                                "tool_use_action": {
                                    "tool_name": "Read",
                                    "tool_use_id": "t2",
                                }
                            },
                            "output": {},
                        }
                    }
                }
            },
            {"event": {"step_end": {"step": {"id": "s2"}}}},
        )
        end = events[-1]
        self.assertIsInstance(end, MessageEnd)
        self.assertEqual(end.af_tool_use_ids, ("t1",))
        self.assertEqual(set(end.tool_use_ids), {"t1", "t2"})

    def test_agent_finished_action_supplies_text_when_no_llm_text(self) -> None:
        self.b._started_emitted = True
        events = _step(
            self.b,
            {"event": {"step_start": {"step": {"id": "s3"}}}},
            {
                "event": {
                    "action_end": {
                        "action": {
                            "id": "a4",
                            "step_id": "s3",
                            "variant": {"agent_finished_action": {}},
                            "output": {"data": {"message": "Final answer"}},
                        }
                    }
                }
            },
            {"event": {"step_end": {"step": {"id": "s3"}}}},
        )
        end = events[-1]
        self.assertIsInstance(end, MessageEnd)
        self.assertEqual(end.text, "Final answer")

    def test_rule_application_step_emits_nothing(self) -> None:
        self.b._started_emitted = True
        events = _step(
            self.b,
            {"event": {"step_start": {"step": {"id": "s0"}}}},
            {
                "event": {
                    "action_end": {
                        "action": {
                            "id": "r",
                            "step_id": "s0",
                            "variant": {
                                "rule_application_action": {
                                    "command": "rule_application_action"
                                }
                            },
                            "output": {"info": "Applied: SKILL.md"},
                        }
                    }
                }
            },
            {"event": {"step_end": {"step": {"id": "s0"}}}},
        )
        self.assertEqual(events, [])

    def test_session_end_becomes_turn_end(self) -> None:
        self.b._started_emitted = True
        events = self.b._map(
            {
                "event": {
                    "session_end": {
                        "session": {
                            "id": "sess-7",
                            "exit_code": "COMPLETE",
                            "exit_message": "done",
                            "duration_ms": 123,
                        }
                    }
                }
            }
        )
        end = events[-1]
        self.assertIsInstance(end, TurnEnd)
        self.assertEqual(end.session_id, "sess-7")
        self.assertEqual(end.result_text, "done")
        self.assertFalse(end.is_error)

    def test_errored_exit_code_is_error(self) -> None:
        self.b._started_emitted = True
        events = self.b._map(
            {
                "event": {
                    "session_end": {
                        "session": {
                            "id": "s",
                            "exit_code": "ERRORED",
                            "exit_message": "boom",
                        }
                    }
                }
            }
        )
        self.assertTrue(events[-1].is_error)

    def test_session_error_becomes_vendor_error(self) -> None:
        events = self.b._map(
            {"event": {"session_error": {"session": {"exit_message": "auth failed"}}}}
        )
        self.assertEqual([type(e) for e in events], [VendorError])
        self.assertIn("auth failed", events[0].message)

    def test_inner_session_subagent_ignored(self) -> None:
        self.b._started_emitted = True
        self.assertEqual(
            self.b._map({"event": {"inner_session": {"session": {"id": "sub"}}}}), []
        )

    def test_ephemeral_telemetry_ignored(self) -> None:
        self.assertEqual(
            self.b._map({"event": {"ephemeral": {"code_compose_lsp_event": {}}}}), []
        )

    def test_full_turn_sequence(self) -> None:
        objs = [
            {"event": {"session_start": {"session": {"id": "full-1"}}}},
            {"event": {"session_update": {"session": {"id": "full-1"}}}},
            {"event": {"step_start": {"step": {"id": "st1"}}}},
            {
                "event": {
                    "action_end": {
                        "action": {
                            "id": "a",
                            "step_id": "st1",
                            "variant": {"llm_action": {}},
                            "output": {"info": "Answer"},
                        }
                    }
                }
            },
            {
                "event": {
                    "action_end": {
                        "action": {
                            "id": "f",
                            "step_id": "st1",
                            "variant": {"agent_finished_action": {}},
                            "output": {"data": {"message": "Answer"}},
                        }
                    }
                }
            },
            {"event": {"step_end": {"step": {"id": "st1"}}}},
            {
                "event": {
                    "session_end": {
                        "session": {
                            "id": "full-1",
                            "exit_code": "COMPLETE",
                            "exit_message": "Answer",
                        }
                    }
                }
            },
        ]
        kinds = [type(e).__name__ for e in _step(self.b, *objs)]
        self.assertEqual(
            kinds, ["SessionStarted", "TextDelta", "MessageEnd", "TurnEnd"]
        )


class OpenTest(later.unittest.TestCase):
    async def test_open_builds_stdio_relay_mcp_config(self) -> None:
        backend = DevmateDmBackend(
            NativeBackendSpec(kind="devmate_dm"), runtime_manager=_FakeRuntime()
        )
        request = SessionOpenRequest(
            session_id="s1",
            resume=False,
            l1_text="L1",
            l1_path="",
            tools=[BridgeToolSpec("echo", "", {}, _noop_handler)],
            hooks=None,
            cwd="/w",
            model="m",
        )
        await backend.open(request)
        cfg = json.loads(backend._mcp_servers_json)
        self.assertEqual(
            cfg[0]["type"], "stdio"
        )  # NOT socket (dm socket MCP stalls real turns)
        self.assertEqual(cfg[0]["name"], "af")
        self.assertTrue(cfg[0]["args"][0].endswith("mcp_stdio_relay.py"))
        self.assertEqual(cfg[0]["args"][1], "/tmp/af_fake.sock")
        self.assertEqual(backend._mcp_token, "tok-123")

    async def test_open_resolves_cats_file_for_workflow_mode(self) -> None:
        backend = DevmateDmBackend(
            NativeBackendSpec(kind="devmate_dm", extra={"cats_file": "/tmp/c.json"}),
            runtime_manager=_FakeRuntime(),
        )
        request = SessionOpenRequest(
            session_id="s1",
            resume=False,
            l1_text="L1",
            l1_path="",
            tools=[BridgeToolSpec("echo", "", {}, _noop_handler)],
            hooks=None,
            cwd="/w",
            model="m",
        )
        await backend.open(request)
        self.assertEqual(backend._cats_file, "/tmp/c.json")
        self.assertIn("--cats-file", backend._argv(TurnRequest(text="hi")))

    async def test_open_resume_sets_started_for_resume_argv(self) -> None:
        backend = DevmateDmBackend(NativeBackendSpec(kind="devmate_dm"))
        request = SessionOpenRequest(
            session_id="s9",
            resume=True,
            l1_text="L1",
            l1_path="",
            tools=[],
            hooks=None,
            cwd="/w",
            model="m",
        )
        await backend.open(request)
        argv = backend._argv(TurnRequest(text="hi"))
        self.assertEqual(argv[argv.index("--resume") + 1], "s9")


class SessionReplacementTest(unittest.TestCase):
    def test_resume_that_lands_in_another_session_is_reported(self) -> None:
        b = _backend()
        b._session_id = "requested"
        b._started_emitted = True  # resuming
        events = b._map({"event": {"session_start": {"session": {"id": "dm_cli_new"}}}})
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].replaced)
        self.assertEqual(events[0].session_id, "dm_cli_new")
        self.assertEqual(b.session_id, "dm_cli_new")
        # Later events of the same (new) session are not reported again.
        self.assertEqual(
            b._map({"event": {"session_update": {"session": {"id": "dm_cli_new"}}}}), []
        )

    def test_resume_of_the_same_session_is_silent(self) -> None:
        b = _backend()
        b._session_id = "same"
        b._started_emitted = True
        self.assertEqual(
            b._map({"event": {"session_start": {"session": {"id": "same"}}}}), []
        )


class ExtraServersTest(later.unittest.TestCase):
    async def test_extra_servers_join_the_mcp_config(self) -> None:
        backend = DevmateDmBackend(
            NativeBackendSpec(
                kind="devmate_dm",
                extra_mcp_servers={"docs": {"type": "stdio", "command": "docs"}},
            ),
            runtime_manager=_FakeRuntime(),
        )
        request = SessionOpenRequest(
            session_id="s1",
            resume=False,
            l1_text="L1",
            l1_path="",
            tools=[BridgeToolSpec("echo", "", {}, _noop_handler)],
            hooks=None,
            cwd="/w",
            model="m",
        )
        await backend.open(request)
        names = [s["name"] for s in json.loads(backend._mcp_servers_json)]
        self.assertEqual(names, ["af", "docs"])


class ToolTimeoutTest(later.unittest.TestCase):
    async def test_af_tool_calls_get_the_spec_timeout(self) -> None:
        backend = DevmateDmBackend(
            NativeBackendSpec(
                kind="devmate_dm",
                mcp_tool_timeout_ms=1_234_000,
                extra_mcp_servers={"docs": {"type": "stdio", "command": "docs"}},
            ),
            runtime_manager=_FakeRuntime(),
        )
        request = SessionOpenRequest(
            session_id="s1",
            resume=False,
            l1_text="L1",
            l1_path="",
            tools=[BridgeToolSpec("echo", "", {}, _noop_handler)],
            hooks=None,
            cwd="/w",
            model="m",
        )
        await backend.open(request)
        af, docs = json.loads(backend._mcp_servers_json)
        self.assertEqual(af["toolCallTimeoutMs"], 1_234_000)
        self.assertNotIn("toolCallTimeoutMs", docs)


_ECHO_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}


def _rpc(method: str, params: dict, request_id: int | None = None) -> bytes:
    message: dict = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        message["id"] = request_id
    return (json.dumps(message) + "\n").encode()


async def _relay_exchange(server: dict, messages: list[bytes], replies: int) -> list:
    """Speak MCP to AF as dm does: spawn the configured stdio server
    (``command`` + ``args``) and exchange newline-delimited JSON-RPC on its
    stdin/stdout."""
    proc = await asyncio.create_subprocess_exec(
        server["command"],
        *server["args"],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    received: list = []
    try:
        for message in messages:
            proc.stdin.write(message)
        await proc.stdin.drain()
        while len(received) < replies:
            line = await asyncio.wait_for(proc.stdout.readline(), 10)
            if not line:
                break
            received.append(json.loads(line))
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    return received


class SocketTransportTest(later.unittest.TestCase):
    """dm reaches AF tools through the stdio relay its ``--mcp-servers``
    names; the relay connects to AF's unix socket, whose filesystem
    permissions are the only credential (the NDJSON framing carries none)."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.calls: list = []

        async def echo(args: dict) -> BridgeResult:
            self.calls.append(args)
            return BridgeResult(text=f"echo: {args.get('text', '')}")

        self.runtime = NativeRuntimeManager()
        self.backend = DevmateDmBackend(
            NativeBackendSpec(kind="devmate_dm", cwd=tempfile.mkdtemp()),
            runtime_manager=self.runtime,
        )
        await self.backend.open(
            SessionOpenRequest(
                session_id="s1",
                resume=False,
                l1_text="L1",
                l1_path="",
                tools=[BridgeToolSpec("echo", "Echo the text.", _ECHO_SCHEMA, echo)],
                hooks=None,
                cwd="/w",
                model="m",
                result_max_chars=16_000,
            )
        )
        self.argv = self.backend._argv(TurnRequest(text="hi"))
        servers = json.loads(self.argv[self.argv.index("--mcp-servers") + 1])
        self.af = servers[0]

    async def asyncTearDown(self) -> None:
        await self.backend.close()
        await self.runtime.aclose_all()
        await super().asyncTearDown()

    async def test_the_socket_is_private_and_only_the_relay_config_names_it(
        self,
    ) -> None:
        self.assertEqual((self.af["type"], self.af["name"]), ("stdio", "af"))
        relay, socket_path = self.af["args"]
        self.assertEqual(os.path.basename(relay), "mcp_stdio_relay.py")
        self.assertTrue(os.path.isfile(relay))
        mode = os.stat(socket_path).st_mode
        self.assertTrue(stat.S_ISSOCK(mode))
        self.assertEqual(stat.S_IMODE(mode), 0o600)
        directory = os.stat(os.path.dirname(socket_path)).st_mode
        self.assertEqual(stat.S_IMODE(directory), 0o700)
        self.assertNotIn(self.backend._mcp_token, "\0".join(self.argv))

    async def test_closing_the_session_removes_its_socket(self) -> None:
        directory = os.path.dirname(self.af["args"][1])
        await self.backend.close()
        self.assertFalse(os.path.exists(directory))

    async def test_the_relay_dm_spawns_serves_the_af_tools(self) -> None:
        replies = await _relay_exchange(
            self.af,
            [
                _rpc(
                    "initialize",
                    {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "dm", "version": "0"},
                    },
                    request_id=1,
                ),
                _rpc("notifications/initialized", {}),
                _rpc("tools/list", {}, request_id=2),
                _rpc(
                    "tools/call",
                    {"name": "echo", "arguments": {"text": "ping"}},
                    request_id=3,
                ),
            ],
            replies=3,
        )
        self.assertEqual([r.get("id") for r in replies], [1, 2, 3])
        self.assertIn("tools", replies[0]["result"]["capabilities"])
        (tool,) = replies[1]["result"]["tools"]
        self.assertEqual(
            (tool["name"], tool["description"], tool["inputSchema"]),
            ("echo", "Echo the text.", _ECHO_SCHEMA),
        )
        # dm reads no tool _meta: the bridge sizes results, nothing is declared.
        self.assertNotIn("_meta", tool)
        result = replies[2]["result"]
        self.assertEqual(result["content"][0]["text"], "echo: ping")
        self.assertFalse(result.get("isError"))
        self.assertEqual(self.calls, [{"text": "ping"}])

    async def test_every_connection_is_served(self) -> None:
        # dm spawns one relay per turn; each is a new socket connection.
        for turn in range(2):
            replies = await _relay_exchange(
                self.af,
                [
                    _rpc(
                        "initialize",
                        {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "dm", "version": "0"},
                        },
                        request_id=1,
                    ),
                    _rpc("notifications/initialized", {}),
                    _rpc(
                        "tools/call",
                        {"name": "echo", "arguments": {"text": f"turn {turn}"}},
                        request_id=2,
                    ),
                ],
                replies=2,
            )
            self.assertEqual(
                replies[1]["result"]["content"][0]["text"], f"echo: turn {turn}"
            )
        self.assertEqual(self.calls, [{"text": "turn 0"}, {"text": "turn 1"}])


class InterruptStopsTheVendorTest(later.unittest.TestCase):
    """``dm`` is a launcher too (srconveyor runs dm-core under node); a
    cancelled turn ends its whole process tree."""

    async def test_a_cancelled_turn_ends_the_vendor_tree_and_is_acknowledged(
        self,
    ) -> None:
        launcher, pid_file = launcher_with_agent(
            agent_printing({"event": {"session_start": {"session": {"id": "s1"}}}})
        )
        backend = DevmateDmBackend(
            NativeBackendSpec(
                kind="devmate_dm", cwd=tempfile.mkdtemp(), cli_path=launcher
            )
        )
        outcome, first, agent, took = await cancel_through_actor(backend, pid_file)
        self.assertEqual(first, SessionStarted(session_id="s1"))
        self.assertTrue(outcome.interrupted)
        self.assertTrue(outcome.acknowledged)
        self.assertLess(took, 5)
        self.assertFalse(pid_alive(agent))


def _tool_action(kind: str, step: str, action: str, tool_use: str) -> dict:
    return {
        "event": {
            kind: {
                "action": {
                    "id": action,
                    "step_id": step,
                    "variant": {
                        "tool_use_action": {
                            "tool_name": "mcp__af__single_choice",
                            "tool_use_id": tool_use,
                        }
                    },
                    "output": {},
                }
            }
        }
    }


def _llm_action(step: str, text: str) -> dict:
    return {
        "event": {
            "action_end": {
                "action": {
                    "id": f"{step}-llm",
                    "step_id": step,
                    "variant": {"llm_action": {"model": "m"}},
                    "output": {"info": text},
                }
            }
        }
    }


class StepAttributionTest(unittest.TestCase):
    """The compound-widget fallback on dm relies on dm reporting a step (one
    model response) only after all of its tool calls ran. Order captured with
    dm 2026.10.03-0249 and a scripted model: a step's two MCP calls run between
    their ``action_start``s and ``action_end``s, then ``step_end``; the next
    step's call starts after that."""

    def test_questions_of_one_step_join_and_a_later_steps_is_refused(self) -> None:
        b = _backend()
        b._started_emitted = True
        turn = TurnScope(
            run_ctx=None, interactive=None, turn_number=1, origin=TurnOrigin.USER
        )

        def feed(*objs: dict) -> list:
            events = _step(b, *objs)
            for event in events:  # as the turn loop does
                if isinstance(event, MessageEnd) and event.af_tool_use_ids:
                    turn.note_af_message(event.af_tool_use_ids, event.message_id)
            return events

        started = feed(
            {"event": {"step_start": {"step": {"id": "s1"}}}},
            _llm_action("s1", "Asking two questions."),
            _tool_action("action_start", "s1", "a1", "call_a"),
            _tool_action("action_start", "s1", "a2", "call_b"),
        )
        self.assertEqual([type(e) for e in started], [TextDelta])
        # call_a and call_b run now; neither can be attributed.
        turn.queue_widget(QueuedWidget(tool=None), None)
        self.assertTrue(turn.accepts_widget(None))
        turn.queue_widget(QueuedWidget(tool=None), None)
        ended = feed(
            _tool_action("action_end", "s1", "a1", "call_a"),
            _tool_action("action_end", "s1", "a2", "call_b"),
            {"event": {"step_end": {"step": {"id": "s1"}}}},
        )
        self.assertEqual([type(e) for e in ended], [MessageEnd])
        self.assertEqual(ended[0].af_tool_use_ids, ("call_a", "call_b"))
        feed(
            {"event": {"step_start": {"step": {"id": "s2"}}}},
            _llm_action("s2", "One more."),
            _tool_action("action_start", "s2", "a3", "call_c"),
        )
        self.assertFalse(turn.accepts_widget(None))  # call_c
        self.assertEqual(len(turn.pending_widgets), 2)

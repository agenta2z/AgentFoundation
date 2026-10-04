"""The host protocol (plan §9.1, §12.1): what a host may use of either
conversational orchestrator, checked against the text-protocol
``ConversationalInferencer`` and the native orchestrator alike — the surface,
one host-driven script (callbacks, result, state round trip), the flow-node
adapter — plus the native semantics §9.1 states: inbox mode,
``on_prompt_rendered`` per vendor turn, ``_ainfer``/``_infer`` and resets.

Plan §12.1 row coverage (assertion: tests; Class.method or a whole Class):
- one host-driven script against CI and native:
    HostScriptTest, HostProtocolParityTest
- ConversationalHost surface and supports_* flags:
    HostSurfaceTest
- callbacks:
    HostScriptTest.test_round_and_turn_callbacks_fire_in_the_same_order_with_the_same_arguments
    HostScriptTest.test_on_prompt_rendered_reports_what_the_model_answered
    NativeHostSemanticsTest.test_on_prompt_rendered_fires_after_each_vendor_turn
    RoundContextTest
- AgenticResult shape + native_meta:
    HostScriptTest.test_the_result_has_the_same_shape
- flow adapter rejects native:
    HostScriptTest.test_the_flow_node_adapter_accepts_classic_and_rejects_native
    FlowNodeRejectionTest
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from typing import Any, Optional
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticDynamicContext,
    AgenticResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
    ConversationalFlowNodeAdapter,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.host_protocol import (
    ConversationalHost,
    SupportsFlowNode,
    SupportsInbox,
    SupportsPromptManifest,
    SupportsRewind,
    SupportsRoundResume,
    SupportsWidgetRecovery,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrib, attrs
from fakes import text, tools
from helpers import FIXTURE_SOPS, make_native, RecordingExecutor, tool_registry
from later.unittest import TestCase

_CAPABILITIES = {
    "supports_widget_recovery": SupportsWidgetRecovery,
    "supports_round_resume": SupportsRoundResume,
    "supports_inbox": SupportsInbox,
    "supports_prompt_manifest": SupportsPromptManifest,
    "supports_flow_node": SupportsFlowNode,
    "supports_rewind": SupportsRewind,
}
_EXPECTED_FLAGS = {
    "classic": {
        "supports_widget_recovery": True,
        "supports_round_resume": True,
        "supports_inbox": True,
        "supports_prompt_manifest": True,
        "supports_flow_node": True,
        "supports_rewind": False,
    },
    "native": {
        "supports_widget_recovery": True,
        "supports_round_resume": False,
        "supports_inbox": True,
        "supports_prompt_manifest": True,
        "supports_flow_node": False,
        "supports_rewind": True,
    },
}
_RUN_KEYWORDS = {
    "run_context",
    "interactive",
    "session_id",
    "turn_number",
    "origin",
    "on_new_turn",
    "on_prompt_rendered",
    "on_turn_complete",
    "on_round_start",
    "on_round_complete",
}
_PROMPT_KEYS = {
    "rendered_prompt",
    "template_source",
    "template_feed",
    "template_config",
}


@attrs(slots=False)
class _ScriptedBase(InferencerBase):
    """Backend of the text-protocol orchestrator: answers from a script."""

    replies: list = attrib(factory=list)
    cache_folder: Optional[str] = attrib(default=None)

    def _infer(self, inp, cfg=None, **kw):
        return self.replies.pop(0) if self.replies else "ok"

    async def _ainfer(self, inp, cfg=None, **kw):
        return self._infer(inp, cfg, **kw)


def make_classic(replies=(), **kwargs) -> ConversationalInferencer:
    return ConversationalInferencer(
        base_inferencer=_ScriptedBase(replies=list(replies)),
        tool_registry=tool_registry(),
        tool_executor=RecordingExecutor(),
        extra_sop_dirs=[FIXTURE_SOPS],
        allowed_sops=["mini_research"],
        max_iterations=3,
        **kwargs,
    )


class _Dispatcher:
    """A host's tool dispatcher (opens dashboards for the orchestrator)."""


class HostSurfaceTest(TestCase):
    async def asyncSetUp(self) -> None:
        native, _, _, _ = make_native([])
        self.addAsyncCleanup(native.aclose)
        self.hosts = {"classic": make_classic(), "native": native}

    def test_both_orchestrators_implement_the_core_protocol(self) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                self.assertIsInstance(host, ConversationalHost)

    def test_each_capability_is_advertised_by_its_flag(self) -> None:
        for name, host in self.hosts.items():
            for flag, protocol in _CAPABILITIES.items():
                with self.subTest(name, flag=flag):
                    expected = _EXPECTED_FLAGS[name][flag]
                    self.assertIs(getattr(host, flag), expected)
                    if expected:
                        self.assertIsInstance(host, protocol)

    def test_run_agentic_loop_declares_every_keyword(self) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                params = inspect.signature(host.run_agentic_loop).parameters
                self.assertFalse(
                    [p for p in params.values() if p.kind is p.VAR_KEYWORD]
                )
                keyword_only = {
                    n for n, p in params.items() if p.kind is p.KEYWORD_ONLY
                }
                self.assertEqual(keyword_only, _RUN_KEYWORDS)
                self.assertEqual(params["origin"].default, "user")

    def test_the_tool_dispatcher_setter_reaches_the_dashboard_coordinator(
        self,
    ) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                dispatcher = _Dispatcher()
                host.tool_dispatcher = dispatcher
                self.assertIs(host.tool_dispatcher, dispatcher)
                self.assertIs(host.dashboard_coordinator.tool_dispatcher, dispatcher)
                host.tool_dispatcher = None
                self.assertIs(host.tool_dispatcher, host.tool_executor)
                self.assertIs(
                    host.dashboard_coordinator.tool_dispatcher, host.tool_executor
                )

    def test_the_private_tool_dispatcher_name_still_sets_the_dispatcher(
        self,
    ) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                dispatcher = _Dispatcher()
                host._tool_dispatcher = dispatcher
                self.assertIs(host.tool_dispatcher, dispatcher)

    def test_suspended_sops_can_be_set(self) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                host.sop_controller.enter("mini_research")
                state = host.sop_state
                host.sop_state = None
                states = [state]
                host.suspended_sops = states
                self.assertEqual(host.suspended_sops, [state])
                states.clear()
                self.assertEqual(host.suspended_sops, [state])
                self.assertEqual(host.sop_controller._suspended_sops, [state])
                host._suspended_sops = []
                self.assertEqual(host.suspended_sops, [])

    def test_check_phase_completion_advances_the_active_sop(self) -> None:
        for name, host in self.hosts.items():
            for method in ("check_phase_completion", "_check_phase_completion"):
                with self.subTest(name, method=method):
                    host.sop_state = None
                    host.sop_controller.enter("mini_research", fresh=True)
                    self.assertEqual(host.sop_state.current_phase, "0")
                    host.sop_state.user_input_gate_passed = True
                    getattr(host, method)()
                    self.assertEqual(host.sop_state.current_phase, "1")

    def test_the_dynamic_context_can_be_replaced(self) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                context = AgenticDynamicContext()
                context.add_action("write_brief", "done")
                host.dynamic_context = context
                self.assertIs(host.dynamic_context, context)
                self.assertIs(host._dynamic_context, context)

    def test_the_sop_catalog_filters_have_one_owner(self) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                self.assertEqual(host.sop_controller.allowed_sops, ["mini_research"])
                host.allowed_sops = ["no_such_sop"]
                self.assertEqual(host.sop_controller.allowed_sops, ["no_such_sop"])
                self.assertEqual(host._filtered_sops(), {})
                host.sop_controller.allowed_sops = ["mini_research"]
                self.assertEqual(host.allowed_sops, ["mini_research"])
                self.assertEqual(list(host._filtered_sops()), ["mini_research"])
                host.disallowed_sops = ["mini_research"]
                self.assertEqual(host.sop_controller.disallowed_sops, ["mini_research"])
                self.assertEqual(host._filtered_sops(), {})
                host.disallowed_sops.clear()
                self.assertEqual(host.sop_controller.disallowed_sops, [])
                self.assertEqual(list(host._filtered_sops()), ["mini_research"])

    def test_shutdown_and_the_prompt_manifest_are_public(self) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                self.assertEqual(set(host.last_prompt_data()), _PROMPT_KEYS)
                self.assertFalse(host.shutdown_requested)
                host.request_shutdown()
                self.assertTrue(host.shutdown_requested)


_REPLY = "Hello there."
_NATIVE_META_KEYS = {
    "backend",
    "session_ref",
    "generation",
    "submission",
    "stop_reason",
    "num_turns",
    "total_cost_usd",
    "usage",
}


class _Recorder:
    """Host callbacks that record what fired, in order, with its arguments
    (the host object itself as ``"host"``)."""

    def __init__(self, renumber_to: Optional[int] = None) -> None:
        self.events: list[tuple] = []
        self.renumber_to = renumber_to
        self.hosts: list[Any] = []

    async def on_new_turn(self, turn_number: int, user_input: str) -> Optional[int]:
        self.events.append(("new_turn", turn_number, user_input))
        return self.renumber_to

    async def on_round_start(self, iteration: int, turn_number: int) -> dict:
        self.events.append(("round_start", iteration, turn_number))
        return {"round": iteration}

    async def on_prompt_rendered(self, host: Any, response_text: str) -> None:
        self.hosts.append(host)
        self.events.append(("prompt_rendered", "host", response_text))

    async def on_round_complete(
        self, host, iteration, turn_number, raw, clean, display, response
    ) -> None:
        self.hosts.append(host)
        self.events.append(
            (
                "round_complete",
                "host",
                iteration,
                turn_number,
                raw,
                clean,
                display,
                response.text,
                response.has_conversation_tool,
            )
        )

    async def on_turn_complete(self, iterations: int) -> None:
        self.events.append(("turn_complete", iterations))

    def callbacks(self) -> dict[str, Any]:
        return {
            "on_new_turn": self.on_new_turn,
            "on_round_start": self.on_round_start,
            "on_prompt_rendered": self.on_prompt_rendered,
            "on_round_complete": self.on_round_complete,
            "on_turn_complete": self.on_turn_complete,
        }

    def names(self) -> list[str]:
        return [event[0] for event in self.events]

    def without(self, name: str) -> list[tuple]:
        return [event for event in self.events if event[0] != name]


async def drive(host: ConversationalHost, recorder: _Recorder) -> AgenticResult:
    """The host-driven script: what a host does with either orchestrator."""
    host.set_prior_context({"session_root_path": "/work"})
    host.update_prior_context(user_name="Ada")
    host.sop_controller.enter("mini_research")
    return await host.run_agentic_loop(
        "hi", session_id="session-1", turn_number=3, **recorder.callbacks()
    )


class HostScriptTest(TestCase):
    """One host-driven script against both orchestrators."""

    async def asyncSetUp(self) -> None:
        native, self.factory, _, _ = make_native([[text(_REPLY)]])
        native_fresh, _, _, _ = make_native([], conversation_key="conv-fresh")
        for host in (native, native_fresh):
            self.addAsyncCleanup(host.aclose)
        self.hosts = {"classic": make_classic([_REPLY]), "native": native}
        self.fresh = {"classic": make_classic(), "native": native_fresh}
        self.recorders = {name: _Recorder(renumber_to=4) for name in self.hosts}
        self.results = {
            name: await drive(host, self.recorders[name])
            for name, host in self.hosts.items()
        }

    def test_round_and_turn_callbacks_fire_in_the_same_order_with_the_same_arguments(
        self,
    ) -> None:
        expected = [
            ("new_turn", 3, "hi"),
            ("round_start", 0, 4),
            ("round_complete", "host", 0, 4, _REPLY, _REPLY, _REPLY, _REPLY, False),
            ("turn_complete", 1),
        ]
        for name, recorder in self.recorders.items():
            with self.subTest(name):
                self.assertEqual(recorder.without("prompt_rendered"), expected)
                self.assertTrue(all(h is self.hosts[name] for h in recorder.hosts))

    def test_on_prompt_rendered_reports_what_the_model_answered(self) -> None:
        # Same arguments, once, before turn_complete. Its place differs by
        # design (plan §9.1): the text protocol renders per model call, inside
        # the round; a native host after its vendor turn's last round.
        classic, native = self.recorders["classic"], self.recorders["native"]
        for recorder in (classic, native):
            self.assertEqual(
                [e for e in recorder.events if e[0] == "prompt_rendered"],
                [("prompt_rendered", "host", _REPLY)],
            )
        self.assertEqual(
            classic.names(),
            [
                "new_turn",
                "round_start",
                "prompt_rendered",
                "round_complete",
                "turn_complete",
            ],
        )
        self.assertEqual(
            native.names(),
            [
                "new_turn",
                "round_start",
                "round_complete",
                "prompt_rendered",
                "turn_complete",
            ],
        )

    def test_the_result_has_the_same_shape(self) -> None:
        for name, result in self.results.items():
            with self.subTest(name):
                self.assertIs(type(result), AgenticResult)
                self.assertEqual(result.text, _REPLY)
                self.assertEqual(result.completed_actions, [])
                self.assertEqual(result.iterations_used, 1)
                self.assertFalse(result.has_conversation_tool)
                self.assertIsNone(result.conversation_tool)
                self.assertFalse(result.exhausted_max_iterations)
                manifest = self.hosts[name].last_prompt_data()
                self.assertTrue(result.last_rendered_prompt)
                self.assertEqual(
                    result.last_rendered_prompt, manifest["rendered_prompt"]
                )
                self.assertEqual(
                    result.last_template_source, manifest["template_source"]
                )
                self.assertEqual(
                    sorted(f.name for f in dataclasses.fields(result)),
                    sorted(f.name for f in dataclasses.fields(AgenticResult)),
                )
        self.assertEqual(self.results["classic"].native_meta, {})
        meta = self.results["native"].native_meta
        self.assertEqual(set(meta), _NATIVE_META_KEYS)
        self.assertEqual(meta["backend"], "claude_sdk")
        self.assertEqual(meta["submission"], "committed")
        self.assertEqual(meta["stop_reason"], "end_turn")
        # A hash of the vendor session id, never the id itself.
        self.assertRegex(meta["session_ref"], r"^[0-9a-f]{12}$")
        self.assertNotIn(self.factory.last.session_id, json.dumps(meta))

    def test_exported_state_restores_into_a_fresh_host(self) -> None:
        for name, host in self.hosts.items():
            with self.subTest(name):
                blob = host.export_state(turn_number=4, iteration=1)
                blob = json.loads(json.dumps(blob))
                fresh = self.fresh[name]
                fresh.restore_state(blob)
                self.assertEqual(fresh.get_messages(), host.get_messages())
                self.assertEqual(
                    fresh.get_messages()[-1], {"role": "assistant", "content": _REPLY}
                )
                self.assertEqual(fresh.prior_context, host.prior_context)
                self.assertEqual(fresh.prior_context["user_name"], "Ada")
                self.assertEqual(fresh.sop_state.sop_name, "mini_research")
                self.assertEqual(
                    fresh.sop_state.current_phase, host.sop_state.current_phase
                )
                again = fresh.export_state(turn_number=4, iteration=1)
                for key in ("messages", "prior_context", "sop_state", "suspended_sops"):
                    self.assertEqual(again[key], blob[key])
        native_blob = self.hosts["native"].export_state(turn_number=4, iteration=1)
        self.assertEqual(
            self.fresh["native"].export_state(turn_number=4, iteration=1)["native"],
            json.loads(json.dumps(native_blob["native"])),
        )

    def test_the_flow_node_adapter_accepts_classic_and_rejects_native(self) -> None:
        adapter = ConversationalFlowNodeAdapter(
            conversational_inferencer=self.fresh["classic"]
        )
        self.assertIs(adapter.conversational_inferencer, self.fresh["classic"])
        with self.assertRaises(TypeError) as ctx:
            ConversationalFlowNodeAdapter(
                conversational_inferencer=self.hosts["native"]
            )
        self.assertIn("cannot be used as a flow node", str(ctx.exception))


class RoundContextTest(TestCase):
    """The round context a host's ``on_round_start`` returns (OpenStartup's
    carries the round directory as ``cache_folder``), on both orchestrators."""

    async def _folders_seen(
        self, host: ConversationalHost, context: dict
    ) -> list[Optional[str]]:
        """The host's ``cache_folder`` when each round completes."""
        seen = []

        async def on_round_start(iteration: int, turn_number: int) -> dict:
            return context

        async def on_round_complete(inf, *_args) -> None:
            seen.append(inf.cache_folder)

        await host.run_agentic_loop(
            "hi", on_round_start=on_round_start, on_round_complete=on_round_complete
        )
        return seen

    async def test_its_cache_folder_becomes_the_hosts_cache_folder(self) -> None:
        folder = "/session/turn_001/round_001"
        context = {"cache_folder": folder, "message_id": "m1"}
        native, _, interactive, _ = make_native([[text(_REPLY)]])
        hosts = {"classic": make_classic([_REPLY]), "native": native}
        async with native:
            for name, host in hosts.items():
                with self.subTest(name):
                    self.assertEqual(await self._folders_seen(host, context), [folder])
                    self.assertEqual(host.cache_folder, folder)
        self.assertEqual(interactive.round_contexts, [context])

    async def test_one_without_a_cache_folder_keeps_the_current_one(self) -> None:
        context = {"cache_folder": "", "message_id": "m1"}
        native, _, _, _ = make_native([[text(_REPLY)]])
        hosts = {"classic": make_classic([_REPLY]), "native": native}
        async with native:
            for name, host in hosts.items():
                with self.subTest(name):
                    host.cache_folder = "/session"
                    self.assertEqual(
                        await self._folders_seen(host, context), ["/session"]
                    )


class NativeHostSemanticsTest(TestCase):
    """The native semantics plan §9.1 states for the host protocol."""

    async def test_inbox_mode_continues_after_a_background_tool(self) -> None:
        scripts = [
            [tools(("write_brief", {"topic": "lidar"})), text("never streamed")],
            [text("The brief is ready.")],
        ]
        native, factory, interactive, executor = make_native(scripts, async_brief=True)
        recorder = _Recorder()
        completed = []

        async def on_turn_complete(iterations: int) -> None:
            completed.append(iterations)
            if len(completed) == 2:
                native.request_shutdown()

        async with native:
            native.enable_inbox(
                interactive,
                on_new_turn=recorder.on_new_turn,
                on_prompt_rendered=recorder.on_prompt_rendered,
                on_turn_complete=on_turn_complete,
            )
            native.inbox_put_user("write the brief")
            result = await native.run()

        continuation = (
            "The background tool write_brief finished; continue with its result."
        )
        self.assertEqual(
            [e for e in recorder.events if e[0] == "new_turn"],
            [("new_turn", 1, "write the brief"), ("new_turn", 2, continuation)],
        )
        self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])
        backend = factory.last
        self.assertEqual(
            [r.text for r in backend.turn_requests], ["write the brief", continuation]
        )
        self.assertIn('origin="user"', backend.l2_seen[0])
        self.assertIn('origin="tool_completion"', backend.l2_seen[1])
        self.assertIn("write_brief finished:\nwrite_brief done", backend.l2_seen[1])
        self.assertEqual(result.text, "The brief is ready.")
        self.assertTrue(native.shutdown_requested)

    async def test_on_prompt_rendered_fires_after_each_vendor_turn(self) -> None:
        scripts = [
            [
                text("Let me ask."),
                tools(("clarification", {"prompt": "Topic?", "output": ["topic"]})),
            ],
            [text("Thanks, lidar it is.")],
        ]
        native, factory, _, _ = make_native(scripts, answers=["lidar"])
        recorder = _Recorder()
        async with native:
            result = await native.run_agentic_loop("ask me", **recorder.callbacks())

        self.assertEqual(len(factory.last.turn_requests), 2)
        self.assertEqual(
            [e[:3] for e in recorder.events if e[0] != "new_turn"],
            [
                ("round_start", 0, 0),
                ("round_complete", "host", 0),
                ("prompt_rendered", "host", "Let me ask."),
                ("round_start", 1, 0),
                ("round_complete", "host", 1),
                ("prompt_rendered", "host", "Thanks, lidar it is."),
                ("turn_complete", 2),
            ],
        )
        self.assertEqual(result.text, "Thanks, lidar it is.")

    async def test_ainfer_is_one_host_turn_without_callbacks(self) -> None:
        native, factory, _, _ = make_native([[text(_REPLY)]])
        async with native:
            with mock.patch.object(
                native, "run_agentic_loop", wraps=native.run_agentic_loop
            ) as loop:
                answer = await native.ainfer("hi")
        self.assertEqual(answer, _REPLY)
        loop.assert_called_once()
        self.assertEqual(loop.call_args.args, ("hi",))
        self.assertEqual(set(loop.call_args.kwargs), {"run_context"})
        self.assertEqual([r.text for r in factory.last.turn_requests], ["hi"])

    async def test_infer_is_refused(self) -> None:
        native, factory, _, _ = make_native([[text(_REPLY)]])
        async with native:
            with self.assertRaisesRegex(NotImplementedError, "async-only"):
                native._infer("hi")
        self.assertEqual(factory.instances, [])

    async def test_reset_for_flow_invocation_starts_a_new_vendor_session(
        self,
    ) -> None:
        native, factory, _, _ = make_native([[text("one")], [text("two")]])
        async with native:
            first = await native.run_agentic_loop("first")
            first_session = factory.last.session_id
            native.reset_for_flow_invocation()
            self.assertEqual(native.get_messages(), [])
            second = await native.run_agentic_loop("second")

        self.assertTrue(first_session)
        self.assertNotEqual(factory.last.session_id, first_session)
        self.assertFalse(factory.last.open_request.resume)
        self.assertNotEqual(
            second.native_meta["session_ref"], first.native_meta["session_ref"]
        )
        self.assertEqual(
            second.native_meta["generation"], first.native_meta["generation"] + 1
        )
        self.assertEqual(
            native.get_messages(),
            [
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "two"},
            ],
        )


@attrs(slots=False)
class _GreetingBase(InferencerBase):
    """Backend for the classic orchestrator: always answers with one text."""

    def _infer(self, inp, cfg=None, **kw):
        return "Hello there."

    async def _ainfer(self, inp, cfg=None, **kw):
        return "Hello there."


class HostProtocolParityTest(TestCase):
    _PROMPT_KEYS = {
        "rendered_prompt",
        "template_source",
        "template_feed",
        "template_config",
    }

    async def _exercise(self, host, fresh_host) -> AgenticResult:
        self.assertIsInstance(host, ConversationalHost)
        for flag in (
            "supports_round_resume",
            "supports_widget_recovery",
            "supports_inbox",
        ):
            self.assertIsInstance(getattr(host, flag), bool)
        result = await host.run_agentic_loop("hi")
        self.assertIsInstance(result, AgenticResult)
        self.assertEqual(result.text, "Hello there.")
        self.assertEqual(set(host.last_prompt_data()), self._PROMPT_KEYS)
        self.assertTrue(host.last_prompt_data()["rendered_prompt"])
        blob = host.export_state(turn_number=1, iteration=1)
        fresh_host.restore_state(blob)
        self.assertEqual(fresh_host.get_messages(), host.get_messages())
        self.assertEqual(
            host.get_messages()[-1], {"role": "assistant", "content": "Hello there."}
        )
        return result

    async def test_classic_and_native_share_the_host_surface(self) -> None:
        classic = ConversationalInferencer(
            base_inferencer=_GreetingBase(), tool_registry={}, max_iterations=3
        )
        classic_fresh = ConversationalInferencer(
            base_inferencer=_GreetingBase(), tool_registry={}, max_iterations=3
        )
        classic_result = await self._exercise(classic, classic_fresh)
        self.assertEqual(classic_result.native_meta, {})
        self.assertTrue(classic.supports_round_resume)

        native, _, _, _ = make_native([[text("Hello there.")]])
        native_fresh, _, _, _ = make_native([], conversation_key="conv-fresh")
        async with native, native_fresh:
            native_result = await self._exercise(native, native_fresh)
            self.assertEqual(native_result.native_meta["backend"], "claude_sdk")
            self.assertEqual(native_result.native_meta["submission"], "committed")
            self.assertFalse(native.supports_round_resume)


class FlowNodeRejectionTest(TestCase):
    async def test_native_is_rejected_as_a_flow_node_with_the_reason(self) -> None:
        native, _, _, _ = make_native([])
        async with native:
            with self.assertRaises(TypeError) as ctx:
                ConversationalFlowNodeAdapter(conversational_inferencer=native)
            self.assertIn("cannot be used as a flow node", str(ctx.exception))

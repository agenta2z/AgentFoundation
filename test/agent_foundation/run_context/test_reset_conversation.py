"""``areset_conversation``: the reset branch's next call starts a new vendor
conversation, and every other branch keeps its own (plan §15, S17).

Each family is a real leaf over the fake transports of ``_leaf_fixtures``. The
branches are two siblings of one host root, the shape of the per-round agent
branches ``ConversationalInferencer`` resets.
"""

from __future__ import annotations

import itertools
import tempfile
import unittest

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_sdk_inferencer import (
    CodexSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
    DevmateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
    openclaw_inferencer as openclaw_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer import (
    OpenClawInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.rovochat_inferencer import (
    RovoChatInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
    rovodev_serve_inferencer as serve_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_serve_inferencer import (
    RovoDevServeInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrs
from later.unittest import TestCase

from ._leaf_fixtures import (
    FakeGateway,
    fixed_code_scope,
    install_fake_claude_sdk,
    install_fake_codex_sdk,
    install_fake_devmate_sdk,
    install_fake_metamate_sdk,
    install_fake_rovochat,
    install_fake_rovodev_serve,
    LEAF_FIXTURES,
)

CLI_LEAVES = ("claude_cli", "codex_cli", "devmate_cli", "kiro_cli", "rovodev_cli")


def _under(ctx, fn, *args):
    token = enter_run(ctx)
    try:
        return fn(*args)
    finally:
        exit_run(token)


def _read(inf, ctx, name):
    return _under(ctx, getattr, inf, name)


def _set_under(ctx, inf, **values):
    token = enter_run(ctx)
    try:
        for name, value in values.items():
            setattr(inf, name, value)
    finally:
        exit_run(token)


class _LeafTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.mp = pytest.MonkeyPatch()
        self.addCleanup(self.mp.undo)
        root = RunContext.root(workspace=InferencerWorkspace(root=self.tmp))
        self.a, self.b = root.child("a"), root.child("b")


class SessionResetTest(_LeafTestCase):
    async def test_every_family_resets_only_the_given_branch_session(self) -> None:
        for name, build in sorted(LEAF_FIXTURES.items()):
            with self.subTest(leaf=name), tempfile.TemporaryDirectory() as tmp:
                mp = pytest.MonkeyPatch()
                try:
                    inf = build(tmp, mp)
                    _set_under(self.a, inf, active_session_id="sid-a")
                    _set_under(self.b, inf, active_session_id="sid-b")

                    await inf.areset_conversation(run_context=self.a)

                    reset = _read(inf, self.a, "active_session_id")
                    if isinstance(inf, OpenClawInferencer):
                        self.assertTrue(reset.startswith(f"{inf.session_id}-"))
                    else:
                        self.assertIsNone(reset)
                    self.assertEqual(_read(inf, self.b, "active_session_id"), "sid-b")
                finally:
                    mp.undo()

    async def test_cli_families_start_the_reset_branch_without_resuming(self) -> None:
        for name in CLI_LEAVES:
            with self.subTest(leaf=name), tempfile.TemporaryDirectory() as tmp:
                mp = pytest.MonkeyPatch()
                try:
                    inf = LEAF_FIXTURES[name](tmp, mp)
                    _set_under(self.a, inf, active_session_id="sid-a")
                    _set_under(self.b, inf, active_session_id="sid-b")

                    await inf.areset_conversation(run_context=self.a)

                    self.assertEqual(
                        _under(self.a, inf._prepare_call, {}),
                        {"session_id": None, "resume": False},
                    )
                    self.assertEqual(
                        _under(self.b, inf._prepare_call, {}),
                        {"session_id": "sid-b", "resume": True},
                    )
                finally:
                    mp.undo()

    async def test_a_bare_reset_clears_the_session_bare_calls_share(self) -> None:
        inf = LEAF_FIXTURES["claude_cli"](self.tmp, self.mp)
        inf.active_session_id = "sid-bare"

        await inf.areset_conversation()

        self.assertIsNone(vars(inf).get("_session_id"))
        self.assertEqual(inf._prepare_call({}), {"session_id": None, "resume": False})

    async def test_the_active_context_is_the_default_branch(self) -> None:
        inf = LEAF_FIXTURES["claude_cli"](self.tmp, self.mp)
        _set_under(self.a, inf, active_session_id="sid-a")
        _set_under(self.b, inf, active_session_id="sid-b")
        token = enter_run(self.a)
        try:
            await inf.areset_conversation()
            self.assertIs(active_run_context(), self.a)
        finally:
            exit_run(token)
        self.assertIsNone(_read(inf, self.a, "active_session_id"))
        self.assertEqual(_read(inf, self.b, "active_session_id"), "sid-b")

    async def test_a_conversation_free_inferencer_is_left_untouched(self) -> None:
        @attrs(slots=False)
        class _Stateless(InferencerBase):
            def _infer(self, inference_input, inference_config=None, **kwargs):
                return "x"

        inf = _Stateless()
        before = dict(vars(inf))
        await inf.areset_conversation(run_context=self.a)
        self.assertEqual(vars(inf), before)


class ConnectedClientResetTest(_LeafTestCase):
    """The SDK leaves hold one live client per branch: it is closed only on the
    reset branch, and a client connected outside any context stays shared."""

    def _claude(self, log):
        install_fake_claude_sdk(self.mp, log)
        return ClaudeCodeSdkInferencer(target_path=self.tmp)

    async def test_claude_sdk_reconnects_only_the_reset_branch(self) -> None:
        log = []
        inf = self._claude(log)
        try:
            await inf.ainfer("first a", run_context=self.a)
            await inf.ainfer("first b", run_context=self.b)
            client_a = _read(inf, self.a, "_client")
            client_b = _read(inf, self.b, "_client")

            await inf.areset_conversation(run_context=self.a)

            self.assertEqual(log, [("connect", None)] * 2 + [("disconnect", None)])
            self.assertIsNone(_read(inf, self.a, "active_session_id"))
            self.assertIs(_read(inf, self.b, "_client"), client_b)
            self.assertEqual(_read(inf, self.b, "active_session_id"), "sid-b")

            await inf.ainfer("second a", run_context=self.a)
            await inf.ainfer("second b", run_context=self.b)

            self.assertEqual(log[3:], [("connect", None)])
            self.assertIsNot(_read(inf, self.a, "_client"), client_a)
            self.assertIs(_read(inf, self.b, "_client"), client_b)
        finally:
            await inf.adisconnect()

    async def test_claude_sdk_keeps_the_client_connected_outside_any_context(
        self,
    ) -> None:
        log = []
        inf = self._claude(log)
        try:
            await inf.aconnect()
            shared = vars(inf)["_client_backing"]
            await inf.ainfer("first a", run_context=self.a)

            await inf.areset_conversation(run_context=self.a)
            await inf.ainfer("second a", run_context=self.a)
            await inf.ainfer("first b", run_context=self.b)

            self.assertEqual(log, [("connect", None)] * 2)
            self.assertIs(vars(inf)["_client_backing"], shared)
            self.assertIs(_read(inf, self.b, "_client"), shared)
            self.assertIsNot(_read(inf, self.a, "_client"), shared)
        finally:
            await inf.adisconnect()
        self.assertEqual(log.count(("disconnect", None)), 2)

    def _codex(self, log):
        install_fake_codex_sdk(self.mp, log)
        return CodexSdkInferencer(target_path=self.tmp)

    async def test_codex_sdk_starts_a_new_thread_only_on_the_reset_branch(
        self,
    ) -> None:
        log = []
        inf = self._codex(log)
        try:
            await inf.ainfer("first a", run_context=self.a)
            await inf.ainfer("first b", run_context=self.b)
            client_b = _read(inf, self.b, "_client")

            await inf.areset_conversation(run_context=self.a)
            await inf.ainfer("second a", run_context=self.a)
            await inf.ainfer("second b", run_context=self.b)

            self.assertEqual(
                log,
                [
                    ("thread_start", "thread-1"),
                    ("thread_start", "thread-2"),
                    ("close", None),
                    ("thread_start", "thread-3"),
                ],
            )
            self.assertEqual(_read(inf, self.a, "active_session_id"), "thread-3")
            self.assertIs(_read(inf, self.b, "_client"), client_b)
            self.assertEqual(_read(inf, self.b, "active_session_id"), "thread-2")
        finally:
            await inf.adisconnect()

    async def test_codex_sdk_keeps_the_client_opened_outside_any_context(
        self,
    ) -> None:
        log = []
        inf = self._codex(log)
        try:
            await inf.aconnect()
            shared = vars(inf)["_client_backing"]
            await inf.ainfer("first a", run_context=self.a)

            await inf.areset_conversation(run_context=self.a)
            await inf.ainfer("second a", run_context=self.a)

            self.assertEqual(
                log, [("thread_start", "thread-1"), ("thread_start", "thread-2")]
            )
            self.assertIs(vars(inf)["_client_backing"], shared)
            self.assertIs(_read(inf, self.b, "_client"), shared)
        finally:
            await inf.adisconnect()


class ServerSideConversationResetTest(_LeafTestCase):
    """Leaves whose conversation lives at the vendor, keyed by an id the leaf
    resumes: the reset branch's next call starts a new one."""

    async def test_devmate_sdk_starts_a_new_session_on_the_reset_branch(self) -> None:
        log = []
        install_fake_devmate_sdk(self.mp, log)
        inf = DevmateSDKInferencer(target_path=self.tmp)
        for prompt, ctx in (("one", self.a), ("one", self.b)):
            await inf.ainfer(prompt, run_context=ctx)

        await inf.areset_conversation(run_context=self.a)
        for prompt, ctx in (("two", self.a), ("two", self.b)):
            await inf.ainfer(prompt, run_context=ctx)

        self.assertEqual(
            log,
            [("start_session", None)] * 2
            + [("start_session", None), ("start_session", "dm-2")],
        )

    async def test_metamate_sdk_starts_a_new_conversation_on_the_reset_branch(
        self,
    ) -> None:
        log = []
        install_fake_metamate_sdk(self.mp, log)
        inf = MetamateSDKInferencer(
            poll_interval_seconds=0.01, code_scope_judge=fixed_code_scope
        )
        for ctx in (self.a, self.b):
            await inf.ainfer("one", run_context=ctx)

        await inf.areset_conversation(run_context=self.a)
        for ctx in (self.a, self.b):
            await inf.ainfer("two", run_context=ctx)

        self.assertEqual(
            log,
            [("engine_start_v2", None)] * 2
            + [("engine_start_v2", None), ("engine_start_v2", "conv-2")],
        )

    async def test_rovochat_creates_a_new_conversation_on_the_reset_branch(
        self,
    ) -> None:
        log = []
        install_fake_rovochat(self.mp, log)
        inf = RovoChatInferencer(
            base_url="https://rovo.example.test", cloud_id="cloud-1", uct_token="uct"
        )
        for ctx in (self.a, self.b):
            await inf.ainfer("one", run_context=ctx)

        await inf.areset_conversation(run_context=self.a)
        for ctx in (self.a, self.b):
            await inf.ainfer("two", run_context=ctx)

        self.assertEqual(
            log,
            [
                ("create_conversation", "conv-1"),
                ("send_message", "conv-1"),
                ("stream_closed", self.a.path),
                ("create_conversation", "conv-2"),
                ("send_message", "conv-2"),
                ("stream_closed", self.b.path),
                ("create_conversation", "conv-3"),
                ("send_message", "conv-3"),
                ("stream_closed", self.a.path),
                ("send_message", "conv-2"),
                ("stream_closed", self.b.path),
            ],
        )
        self.assertEqual(_read(inf, self.a, "active_session_id"), "conv-3")
        self.assertEqual(_read(inf, self.b, "active_session_id"), "conv-2")

    async def test_rovodev_serve_resets_only_the_reset_branch_server(self) -> None:
        log = []
        install_fake_rovodev_serve(self.mp, log)
        ports = itertools.count(8001)
        self.mp.setattr(serve_module, "find_available_port", lambda: next(ports))
        inf = RovoDevServeInferencer(acli_path="acli", target_path=self.tmp)
        try:
            for ctx in (self.a, self.b):
                token = enter_run(ctx)
                try:
                    await inf.aconnect()
                    inf.active_session_id = "active"
                finally:
                    exit_run(token)

            await inf.areset_conversation(run_context=self.a)

            self.assertEqual(sum(entry[0] == "spawn" for entry in log), 2)
            self.assertEqual([e for e in log if e[0] == "reset"], [("reset", 8001)])
            self.assertIsNone(_read(inf, self.a, "active_session_id"))
            self.assertEqual(_read(inf, self.b, "active_session_id"), "active")
        finally:
            await inf.adisconnect()

    async def test_rovodev_serve_without_a_server_has_nothing_to_reset(self) -> None:
        log = []
        install_fake_rovodev_serve(self.mp, log)
        inf = RovoDevServeInferencer(acli_path="acli", target_path=self.tmp)

        await inf.areset_conversation(run_context=self.a)

        self.assertEqual(log, [])

    def _openclaw(self, **kwargs):
        gateway = FakeGateway()
        self.mp.setattr(openclaw_module, "run_subprocess", gateway.run_subprocess)

        async def connect(inf):
            return gateway.connect()

        self.mp.setattr(OpenClawInferencer, "_ws_connect", connect)
        return OpenClawInferencer(auth_token="tok", **kwargs), gateway

    @staticmethod
    def _session_of(gateway, prompt):
        (entry,) = [c for c in gateway.log if c.get("prompt") == prompt]
        return entry["session_id"]

    async def test_openclaw_moves_the_reset_branch_to_a_new_gateway_session(
        self,
    ) -> None:
        inf, gateway = self._openclaw()
        for name, ctx in (("a", self.a), ("b", self.b)):
            await inf.ainfer(f"first {name}", run_context=ctx)

        await inf.areset_conversation(run_context=self.a)
        for name, ctx in (("a", self.a), ("b", self.b)):
            await inf.ainfer(f"second {name}", run_context=ctx)

        default = inf.session_id
        self.assertEqual(self._session_of(gateway, "first a"), default)
        self.assertEqual(self._session_of(gateway, "first b"), default)
        fresh = self._session_of(gateway, "second a")
        self.assertTrue(fresh.startswith(f"{default}-"))
        self.assertEqual(self._session_of(gateway, "second b"), default)
        probes = [
            c["session_id"] for c in gateway.log if c["method"] == "transcript_probe"
        ]
        self.assertIn(fresh, probes)

    async def test_openclaw_cannot_reset_a_pinned_session(self) -> None:
        inf, gateway = self._openclaw(auto_resume=False)
        with self.assertLogs(openclaw_module.logger, "WARNING"):
            await inf.areset_conversation(run_context=self.a)
        await inf.ainfer("question", run_context=self.a)
        self.assertEqual(self._session_of(gateway, "question"), inf.session_id)


@attrs
class _HandleLeaf(StreamingInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "x"

    async def _ainfer_streaming(self, prompt, **kwargs):
        yield "x"


class DetachFromBackingTest(unittest.TestCase):
    """A detached branch reads only its own Tier-3 handles; the others still fall
    back to the connection opened outside any context."""

    def setUp(self) -> None:
        root = RunContext.root(workspace=None)
        self.a, self.b = root.child("a"), root.child("b")
        self.leaf = _HandleLeaf()
        self.leaf._tier3_set("client", "shared")

    def _client_under(self, ctx):
        return _under(ctx, self.leaf._tier3_get, "client")

    def test_a_detached_branch_stops_reading_the_backing(self) -> None:
        _under(self.a, self.leaf._tier3_detach_from_backing)
        self.assertIsNone(self._client_under(self.a))
        self.assertEqual(self._client_under(self.b), "shared")
        self.assertEqual(self.leaf._tier3_get("client"), "shared")

    def test_a_detached_branch_reads_its_own_handles(self) -> None:
        _under(self.a, self.leaf._tier3_detach_from_backing)
        _under(self.a, self.leaf._tier3_set, "client", "own")
        self.assertEqual(self._client_under(self.a), "own")
        self.assertEqual(vars(self.leaf)["_client_backing"], "shared")

    def test_without_a_context_detaching_is_a_no_op(self) -> None:
        self.leaf._tier3_detach_from_backing()
        self.assertEqual(self.leaf._tier3_get("client"), "shared")
        self.assertEqual(self._client_under(self.a), "shared")

    def test_own_handles_are_the_branch_entry_or_the_backing(self) -> None:
        self.assertIsNone(_under(self.a, self.leaf._tier3_own_handles).get("client"))
        self.assertEqual(self.leaf._tier3_own_handles().get("client"), "shared")

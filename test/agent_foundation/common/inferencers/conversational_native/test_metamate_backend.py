"""Metamate backend (remote, tool-less) against a fake Metamate SDK: request
construction, event mapping, resume, interrupt, the idle timeout of a stalled
request, and a native conversation running on it (tool-less session
instructions, envelope context)."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

import agent_foundation
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
    VendorTurnFailed,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    SessionStarted,
    TurnEnd,
    VendorError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session import (
    metamate as metamate_package,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    CallerTools,
    Evidence,
    InterruptNotAcknowledged,
    L2Channel,
    NativeBackendSpec,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.factory import (
    backend_class,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.metamate import (
    backend as metamate,
)
from agent_foundation.resources.tools._ci_host import build_native_from_config
from helpers import FIXTURE_SOPS
from later.unittest import TestCase

NATIVE_CONFIG = (
    Path(agent_foundation.__file__).parent
    / "resources"
    / "configs"
    / "conversational_native"
    / "default.yaml"
)


def _record_type(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


FAKE_TYPES = SimpleNamespace(
    MetamateSDKInput=_record_type,
    MetamateSDKClientConfig=_record_type,
    MetamateSessionConfig=_record_type,
    MetamateOrchestration=_record_type,
    MetamateLLMVMConfig=_record_type,
    MetamateLLMParams=_record_type,
    MetamateAgentConfig=_record_type,
    MetamateInlineAgentConfig=_record_type,
    MetamateOrchestrationType=SimpleNamespace(LLMVM="LLMVM"),
    MetamateAgentProviderType=SimpleNamespace(INLINE="INLINE"),
)


def _event(
    conversation_id: Optional[str] = "conv-1", text: str = "", complete: bool = False
):
    return SimpleNamespace(
        conversation_id=conversation_id, text=text, is_complete=complete
    )


# Script steps: a request the server stops answering after reporting its
# conversation, or before any output (a new conversation's id arrives with
# its first output block).
STALL = "stall"
STALL_BEFORE_OUTPUT = "stall_before_output"
_SDK_IDLE_TIMEOUT_DEFAULT_S = 300.0


class FakeClient:
    """Records each request; answers with ``reply <n>`` unless scripted. A
    stalled request behaves like the SDK's: no new output for the request's
    ``idle_timeout_s`` (the SDK's default when not given), then the SDK's
    idle failure."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script: list = []

    async def execute_and_stream(
        self, input, config, orchestration, session_config=None, **options
    ):
        self.calls.append(
            {
                "input": input,
                "config": config,
                "orchestration": orchestration,
                "session": session_config,
                "options": options,
            }
        )
        step = self.script.pop(0) if self.script else None
        if callable(step):
            async for event in step():
                yield event
            return
        conversation_id = session_config.conversation_id if session_config else "conv-1"
        if step in (STALL, STALL_BEFORE_OUTPUT):
            if step == STALL:
                yield _event(conversation_id)
            idle = options.get("idle_timeout_s", _SDK_IDLE_TIMEOUT_DEFAULT_S)
            await asyncio.sleep(idle)
            raise RuntimeError(
                f"No new output for {idle}s for conversation {conversation_id} "
                "(4 blocks completed before stalling)"
            )
        yield _event(conversation_id)
        yield _event(conversation_id, text=f"reply {len(self.calls)}", complete=True)


def _open_request(**kw: Any) -> SessionOpenRequest:
    fields = {
        "session_id": "",
        "resume": False,
        "l1_text": "SESSION INSTRUCTIONS",
        "l1_path": "/tmp/l1.md",
        "tools": [],
        "hooks": object(),
        "cwd": "/w",
        "model": "",
    }
    fields.update(kw)
    return SessionOpenRequest(**fields)


async def _collect(backend: metamate.MetamateBackend, text: str = "hi") -> list:
    return [
        e
        async for e in backend.run_turn(
            TurnRequest(text=text, channel=L2Channel.ENVELOPE)
        )
    ]


class MetamateBackendTest(TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = FakeClient()
        patcher = mock.patch.object(
            metamate, "_sdk", return_value=(self.client, FAKE_TYPES)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_is_tool_less_with_an_envelope_for_turn_context(self) -> None:
        caps = metamate.MetamateBackend.capabilities
        self.assertIs(caps.caller_tools, CallerTools.NONE)
        self.assertIsNone(caps.preferred_l2_channel(envelope_allowed=False))
        self.assertIs(
            caps.preferred_l2_channel(envelope_allowed=True), L2Channel.ENVELOPE
        )

    async def test_instructions_append_to_the_system_prompt(self) -> None:
        backend = metamate.MetamateBackend(
            NativeBackendSpec(kind="metamate", model="m-1")
        )
        await backend.open(_open_request())
        await _collect(backend)
        call = self.client.calls[0]
        llmvm = call["orchestration"].llmvm_config
        inline = llmvm.agent_config.inline_config
        self.assertEqual(inline.prompt, "SESSION INSTRUCTIONS")
        self.assertEqual(inline.system_prompt_mode, "append")
        self.assertEqual(llmvm.entry, metamate.DEFAULT_ENTRY)
        self.assertEqual(llmvm.llm_params.model, "m-1")
        self.assertIsNone(call["session"])
        self.assertEqual(call["input"].text, "hi")

    async def test_events_and_resume_by_conversation_id(self) -> None:
        backend = metamate.MetamateBackend(NativeBackendSpec(kind="metamate"))
        await backend.open(_open_request())
        events = await _collect(backend)
        self.assertEqual(events[0], SessionStarted(session_id="conv-1", replaced=False))
        self.assertIsInstance(events[1], MessageEnd)
        self.assertEqual(events[1].text, "reply 1")
        self.assertEqual(events[2].session_id, "conv-1")
        self.assertIsInstance(events[2], TurnEnd)
        await _collect(backend, "again")
        self.assertEqual(self.client.calls[1]["session"].conversation_id, "conv-1")

    async def test_reopen_resumes_the_recorded_conversation(self) -> None:
        backend = metamate.MetamateBackend(NativeBackendSpec(kind="metamate"))
        await backend.open(_open_request(session_id="conv-9", resume=True))
        await _collect(backend)
        self.assertEqual(self.client.calls[0]["session"].conversation_id, "conv-9")

    async def test_a_missing_conversation_is_classified(self) -> None:
        async def gone():
            raise RuntimeError("Conversation conv-9 not found after 5 retries")
            yield  # pragma: no cover

        self.client.script = [gone]
        backend = metamate.MetamateBackend(NativeBackendSpec(kind="metamate"))
        await backend.open(_open_request(session_id="conv-9", resume=True))
        (error,) = await _collect(backend)
        self.assertIsInstance(error, VendorError)
        self.assertTrue(error.session_missing)
        self.assertFalse(error.submitted)

    async def test_interrupt_stops_reading_a_running_request(self) -> None:
        started = asyncio.Event()

        async def slow():
            yield _event("conv-1")
            started.set()
            await asyncio.Event().wait()
            yield _event("conv-1", text="never", complete=True)  # pragma: no cover

        self.client.script = [slow]
        backend = metamate.MetamateBackend(NativeBackendSpec(kind="metamate"))
        await backend.open(_open_request())
        turn = asyncio.ensure_future(_collect(backend))
        await asyncio.wait_for(started.wait(), 5)
        # The request keeps running server-side: never an acknowledged stop.
        with self.assertRaises(InterruptNotAcknowledged):
            await backend.interrupt()
        events = await asyncio.wait_for(turn, 5)
        self.assertIsInstance(events[-1], VendorError)
        self.assertIn("interrupted", events[-1].message)
        self.assertTrue(events[-1].submitted)


class IdleTimeoutTest(TestCase):
    """A request Metamate stops answering ends after the backend's idle
    timeout, shorter than the SDK's own, as a submitted (uncertain) turn."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = FakeClient()
        patcher = mock.patch.object(
            metamate, "_sdk", return_value=(self.client, FAKE_TYPES)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_every_request_carries_the_configured_idle_timeout(self) -> None:
        self.assertLess(metamate.DEFAULT_IDLE_TIMEOUT_S, _SDK_IDLE_TIMEOUT_DEFAULT_S)
        for extra, expected in (
            ({}, metamate.DEFAULT_IDLE_TIMEOUT_S),
            ({"idle_timeout_s": None}, metamate.DEFAULT_IDLE_TIMEOUT_S),
            ({"idle_timeout_s": 45}, 45.0),
        ):
            with self.subTest(extra=extra):
                backend = metamate.MetamateBackend(
                    NativeBackendSpec(kind="metamate", extra=extra)
                )
                await backend.open(_open_request())
                await _collect(backend)
                await _collect(backend, "again")
                for call in self.client.calls[-2:]:
                    self.assertEqual(call["options"]["idle_timeout_s"], expected)

    async def _stalled_turn(self, step: str, request: SessionOpenRequest) -> list:
        self.client.script = [step]
        backend = metamate.MetamateBackend(
            NativeBackendSpec(kind="metamate", extra={"idle_timeout_s": 0.2})
        )
        await backend.open(request)
        started = time.monotonic()
        # Without the backend's timeout the fake stalls for the SDK's 300 s.
        events = await asyncio.wait_for(_collect(backend), 5)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(len(self.client.calls), 1)
        return events

    def _assert_submitted_stall(self, error: Any) -> None:
        self.assertIsInstance(error, VendorError)
        self.assertTrue(error.submitted)
        self.assertFalse(error.session_missing)
        self.assertIn("no new output for 0.2s (idle_timeout_s)", error.message)
        self.assertIn("not re-sent", error.message)

    async def test_a_resumed_request_that_stalls_fails_as_submitted(self) -> None:
        events = await self._stalled_turn(
            STALL, _open_request(session_id="conv-9", resume=True)
        )
        self.assertEqual(events[0], SessionStarted(session_id="conv-9", replaced=False))
        self._assert_submitted_stall(events[-1])

    async def test_a_new_request_that_stalls_before_any_output_is_submitted(
        self,
    ) -> None:
        """The SDK reports a new conversation's id with its first output, so
        nothing announced the turn; the request still reached Metamate."""
        (error,) = await self._stalled_turn(STALL_BEFORE_OUTPUT, _open_request())
        self._assert_submitted_stall(error)


class IdleTimeoutConfigTest(unittest.TestCase):
    def test_an_idle_timeout_that_is_not_a_positive_number_is_refused(self) -> None:
        for value in (0, -5, "120", True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "idle_timeout_s"):
                    metamate.MetamateBackend(
                        NativeBackendSpec(
                            kind="metamate", extra={"idle_timeout_s": value}
                        )
                    )


def _sdk_available() -> bool:
    try:
        metamate._sdk()
    except NativeCapabilityError:
        return False
    return True


@unittest.skipUnless(_sdk_available(), "the Metamate SDK is Buck-only")
class RealSdkIdleTimeoutTest(TestCase):
    """Against the real SDK (the tested revision), with its server calls
    replaced: it honours ``idle_timeout_s`` and its idle failure is
    classified as a submitted stall."""

    async def test_the_sdk_idle_failure_is_a_submitted_stall(self) -> None:
        from msl.metamate.sdk import confucius_client  # @manual

        started = SimpleNamespace(
            conversation_id="conv-x",
            sandcastle_job_id=None,
            needs_cold_start_wait=False,
        )
        graphql = mock.MagicMock()
        graphql.return_value.get_conversation_for_stream.return_value = []
        with (
            mock.patch.object(
                confucius_client, "_resolve_auth", mock.AsyncMock(return_value=("", 0))
            ),
            mock.patch.object(
                confucius_client.MetamateSDKConfuciusClient,
                "start",
                mock.AsyncMock(return_value=started),
            ),
            mock.patch.object(confucius_client, "MetamateGraphQLClient", graphql),
        ):
            backend = metamate.MetamateBackend(
                NativeBackendSpec(kind="metamate", extra={"idle_timeout_s": 0.05})
            )
            await backend.open(_open_request())
            (error,) = await asyncio.wait_for(_collect(backend), 60)
        self.assertIsInstance(error, VendorError)
        self.assertTrue(error.submitted)
        self.assertIn("no new output for 0.05s", error.message)


class PackageTest(TestCase):
    def test_the_factory_finds_the_backend_in_its_own_package(self) -> None:
        self.assertIs(backend_class("metamate"), metamate.MetamateBackend)
        self.assertIs(metamate_package.MetamateBackend, metamate.MetamateBackend)

    def test_capabilities_name_the_tested_sdk_and_why_no_fbid_is_kept(self) -> None:
        caps = metamate.MetamateBackend.capabilities
        self.assertTrue(caps.tested_versions)
        self.assertEqual(caps.evidence["fbid_coordinates"], Evidence.UNSUPPORTED)


class EnvironmentTest(TestCase):
    def test_hermetic_is_refused_at_construction(self) -> None:
        with self.assertRaises(NativeCapabilityError):
            metamate.MetamateBackend(
                NativeBackendSpec(kind="metamate", environment="hermetic")
            )


class MissingSdkTest(TestCase):
    async def test_open_without_the_sdk_is_a_capability_error(self) -> None:
        with mock.patch.dict(
            "sys.modules", {"msl": None, "msl.metamate": None, "msl.metamate.sdk": None}
        ):
            backend = metamate.MetamateBackend(NativeBackendSpec(kind="metamate"))
            with self.assertRaises(NativeCapabilityError):
                await backend.open(_open_request())


class MetamateConversationTest(TestCase):
    async def test_native_conversation_runs_tool_less(self) -> None:
        client = FakeClient()
        work = tempfile.mkdtemp(prefix="af_native_mm_")
        with mock.patch.object(metamate, "_sdk", return_value=(client, FAKE_TYPES)):
            native = NativeConversationalInferencer(
                backend={"kind": "metamate", "cwd": work, "l2_envelope_allowed": True},
                record_store=InMemoryRecordStore(),
                conversation_key="mm",
                native_session_dir=work,
                extra_sop_dirs=[FIXTURE_SOPS],
                allowed_sops=["mini_research"],
            )
            async with native:
                first = await native.run_agentic_loop("hello", turn_number=1)
                await native.run_agentic_loop("/sop mini_research", turn_number=2)
                await native.run_agentic_loop("what now?", turn_number=3)
        self.assertEqual(first.text, "reply 1")
        l1 = client.calls[0][
            "orchestration"
        ].llmvm_config.agent_config.inline_config.prompt
        self.assertNotIn("mcp__af__", l1)
        self.assertIn("/sop", l1)
        # The active SOP reaches the model as the labelled envelope.
        self.assertIn("<af_context", client.calls[-1]["input"].text)
        self.assertTrue(client.calls[-1]["input"].text.rstrip().endswith("what now?"))
        self.assertEqual(client.calls[-1]["session"].conversation_id, "conv-1")
        self.assertEqual(native._load_record().vendor_session_id, "conv-1")

    async def test_view_prompt_labels_each_lane_with_the_metamate_route(self) -> None:
        client = FakeClient()
        work = tempfile.mkdtemp(prefix="af_native_mm_")
        with mock.patch.object(metamate, "_sdk", return_value=(client, FAKE_TYPES)):
            native = NativeConversationalInferencer(
                backend={"kind": "metamate", "cwd": work, "l2_envelope_allowed": True},
                record_store=InMemoryRecordStore(),
                conversation_key="mm",
                native_session_dir=work,
            )
            async with native:
                await native.run_agentic_loop("hello", turn_number=1)
                data = native.last_prompt_data()
        rendered = data["rendered_prompt"]
        sent_l1 = client.calls[0][
            "orchestration"
        ].llmvm_config.agent_config.inline_config.prompt
        routes = backend_class("metamate").capabilities.prompt_routes(
            L2Channel.ENVELOPE
        )
        self.assertIn("inline agent config", routes.l1)
        self.assertIn('system_prompt_mode="append"', routes.l1)
        self.assertIn(f"## Session instructions — {routes.l1}\n{sent_l1}\n", rendered)
        self.assertIn(f"## Turn context — {routes.l2}\n<af_context", rendered)
        self.assertIn(f"## User message — {routes.user}\nhello\n", rendered)
        self.assertIn(
            "## State updates — none: this backend has no AF tools\n(none)\n", rendered
        )
        self.assertEqual(data["template_feed"]["l1_route"], routes.l1)
        self.assertEqual(data["template_feed"]["l2_channel"], "envelope")

    async def test_a_stalled_request_ends_the_turn_uncertain_and_is_never_resent(
        self,
    ) -> None:
        client = FakeClient()
        client.script = [None, STALL]
        work = tempfile.mkdtemp(prefix="af_native_mm_")
        with mock.patch.object(metamate, "_sdk", return_value=(client, FAKE_TYPES)):
            native = NativeConversationalInferencer(
                backend={
                    "kind": "metamate",
                    "cwd": work,
                    "l2_envelope_allowed": True,
                    "extra": {"idle_timeout_s": 0.2},
                },
                record_store=InMemoryRecordStore(),
                conversation_key="mm",
                native_session_dir=work,
            )
            async with native:
                await native.run_agentic_loop("hello", turn_number=1)
                with self.assertRaises(VendorTurnFailed) as failed:
                    await asyncio.wait_for(
                        native.run_agentic_loop("still there?", turn_number=2), 5
                    )
                record = native._load_record()
                self.assertEqual(record.submission, "uncertain")
                self.assertEqual(len(client.calls), 2)
                stalled_prompt = native.last_prompt_data()
                await native.run_agentic_loop("next", turn_number=3)
        self.assertIn("no new output for 0.2s", str(failed.exception))
        # "View Prompt" shows the stalled turn and how it ended.
        rendered = stalled_prompt["rendered_prompt"]
        self.assertIn("## Turn outcome — uncertain\n", rendered)
        self.assertIn("no new output for 0.2s (idle_timeout_s)", rendered)
        self.assertIn("\nstill there?\n## State updates", rendered)
        self.assertEqual(stalled_prompt["template_feed"]["turn_outcome"], "uncertain")
        # The stalled text is never sent again; the next turn says it failed.
        self.assertEqual(len(client.calls), 3)
        sent = client.calls[2]["input"].text
        self.assertIn('type="turn_failed"', sent)
        self.assertNotIn("still there?", sent)
        self.assertTrue(sent.rstrip().endswith("next"))
        self.assertEqual(client.calls[2]["session"].conversation_id, "conv-1")

    async def test_the_backend_yaml_sets_the_idle_timeout(self) -> None:
        client = FakeClient()
        with mock.patch.object(metamate, "_sdk", return_value=(client, FAKE_TYPES)):
            native = build_native_from_config(
                NATIVE_CONFIG,
                backend="metamate",
                backend_overrides={"cwd": tempfile.mkdtemp(prefix="af_native_mm_")},
                record_store=InMemoryRecordStore(),
                conversation_key="mm-yaml",
            )
            async with native:
                await native.run_agentic_loop("hello", turn_number=1)
        self.assertEqual(native.backend.extra["idle_timeout_s"], 120)
        self.assertEqual(
            client.calls[0]["options"]["idle_timeout_s"],
            metamate.DEFAULT_IDLE_TIMEOUT_S,
        )

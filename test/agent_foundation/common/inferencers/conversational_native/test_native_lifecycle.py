"""Session lifecycle and persistence of the native orchestrator (plan §7.1,
§7.3, D8-D10): pinned and resumed sessions, resume validation, drift,
session loss, rewinds, session commands and the durable record.

Plan §12.1 row coverage (assertion: tests; Class.method or a whole Class):
- pinned id reused:
    NativeTurnTest.test_plain_turn_streams_one_round_and_pins_session
    NativeTurnTest.test_second_turn_resumes_same_live_session
    NativeSessionLifecycleTest.test_rebuilt_inferencer_resumes_by_persisted_session_id
- resume validation: principal / permission mismatch fails closed, cwd change
  rotates (current policy: a backend, adapter or cwd change is a session loss
  under on_session_loss):
    NativeSessionLifecycleTest.test_principal_change_is_rejected_before_contacting_the_vendor
    NativeSessionLifecycleTest.test_cwd_change_rotates_and_queues_a_recap
    FailureHandlingTest, IdentityRefreshTest, ResumePolicyTest
- rewind only with the host opt-in -> exact fork or RewindUnsupported:
    NativeSessionLifecycleTest.test_rewind_forks_the_vendor_session_at_the_turn_boundary
    RepeatedTurnTest.test_without_the_host_opt_in_a_repeated_turn_continues
    RewindTest, EagerRewindTest
- rewind to turn 1 -> fresh session (plan §7.3):
    RewindTest.test_rewind_to_the_first_turn_starts_a_fresh_session
    EagerRewindTest.test_rewind_to_the_first_turn_opens_a_fresh_session_now
- repeated turn_number=0 -> no fork:
    RepeatedTurnTest.test_a_repeated_turn_zero_never_forks
- D8 notice / rotate / fail:
    NativeSessionLifecycleTest.test_instruction_drift_keeps_session_and_sends_one_notice
    LiveSessionDriftTest, DriftPolicyTest, ToolManifestDriftTest
- D9 recap (bounded, labelled, excludes the current message) / fresh / fail:
    SessionLossPolicyTest, HistorySeedingTest, SessionMissingRecoveryTest,
    ResumePolicyTest
- /new, /clear, reset_for_flow_invocation -> new id:
    NativeSessionLifecycleTest.test_new_command_rotates_to_a_fresh_session
    ClearCommandTest
    test_host_protocol.NativeHostSemanticsTest.test_reset_for_flow_invocation_starts_a_new_vendor_session
- idle close -> transparent resume:
    IdleResumeTest
- record saved after bridge calls and turns:
    RecordPersistenceTest
    NativeSessionLifecycleTest.test_a_crash_after_a_widget_was_queued_leaves_a_usable_record
- export_state / restore_state round trip, newest generation wins:
    NativeSessionLifecycleTest.test_export_and_restore_round_trip
    RestoreConflictTest
Also here: /root (RootCommandTest), /model (ModelChangeTest, SpecIsolationTest),
local slash commands (NativeTurnTest.test_slash_sop_*, test_terminal_*), 0600
session files (PrivateFileTest).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import os
import re
import stat
import tempfile
from pathlib import Path

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeBackendSpec,
    NativeRuntimeManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    END_TURN_MARKER,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    RewindUnsupported,
    SessionResumeRejected,
    StablePolicyChanged,
    VendorTurnFailed,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    TurnEnd,
    VendorError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    L2Channel,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.record import (
    ADAPTER_VERSION,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.turn import (
    SubmissionState,
)
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition
from fakes import (
    FAKE_CAPABILITIES,
    FakeBackendFactory,
    ModelRefusingFactory,
    text,
    tools,
    turn_end,
)
from helpers import make_native, shared_dir, wait_for
from later.unittest import TestCase


class NativeTurnTest(TestCase):
    async def test_plain_turn_streams_one_round_and_pins_session(self) -> None:
        native, factory, interactive, _ = make_native(
            [[text("Hello "), text("there.")]]
        )
        async with native:
            rounds = []

            async def on_round_complete(_inf, it, turn, raw, clean, display, conv):
                rounds.append((it, display))

            result = await native.run_agentic_loop(
                "hi", on_round_complete=on_round_complete
            )
            self.assertEqual(result.text, "Hello there.")
            self.assertEqual(rounds, [(0, "Hello there.")])
            self.assertEqual(interactive.streamed, ["Hello there."])
            backend = factory.last
            self.assertFalse(backend.open_request.resume)
            self.assertTrue(backend.open_request.session_id)
            self.assertIn("Standard Operating Procedures", backend.open_request.l1_text)
            self.assertEqual(backend.turn_requests[0].text, "hi")
            self.assertEqual(backend.turn_requests[0].channel, L2Channel.HOOK)
            self.assertIn("<af_context", backend.l2_seen[0])
            # The mirror records the turn's raw text and the reply (§6.1 5.2).
            self.assertEqual(
                [m["content"] for m in native.get_messages()], ["hi", "Hello there."]
            )

    async def test_second_turn_resumes_same_live_session(self) -> None:
        native, factory, _, _ = make_native([[text("one")], [text("two")]])
        async with native:
            await native.run_agentic_loop("first")
            await native.run_agentic_loop("second")
            self.assertEqual(len(factory.instances), 1)
            self.assertEqual(len(factory.last.turn_requests), 2)
            self.assertEqual(native._load_record().status, "active")

    async def test_slash_sop_is_local_and_its_request_reaches_the_agent(self) -> None:
        native, factory, _, _ = make_native([[text("On it.")]])
        async with native:
            result = await native.run_agentic_loop("/sop mini_research quantum sensors")
            self.assertEqual(native.sop_state.sop_name, "mini_research")
            self.assertEqual(factory.last.turn_requests[0].text, "quantum sensors")
            self.assertEqual(result.text, "On it.")

    async def test_terminal_slash_command_never_reaches_the_agent(self) -> None:
        native, factory, _, _ = make_native([])
        async with native:
            result = await native.run_agentic_loop("/status")
            self.assertEqual(factory.instances, [])
            self.assertTrue(result.text)


class NativeSessionLifecycleTest(TestCase):
    async def test_rebuilt_inferencer_resumes_by_persisted_session_id(self) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store = InMemoryRecordStore()
        native, factory, _, _ = make_native(
            [[text("one")]], record_store=store, session_dir=shared
        )
        async with native:
            await native.run_agentic_loop("first")
            session_id = factory.last.open_request.session_id
        rebuilt, factory2, _, _ = make_native(
            [[text("two")]], record_store=store, session_dir=shared
        )
        async with rebuilt:
            await rebuilt.run_agentic_loop("second")
            request = factory2.last.open_request
            self.assertTrue(request.resume)
            self.assertEqual(request.session_id, session_id)

    async def test_a_crash_after_a_widget_was_queued_leaves_a_usable_record(
        self,
    ) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store = InMemoryRecordStore()
        crash = {}

        async def vendor(backend, request):
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            yield MessageEnd(message_id="m0", text="", af_tool_use_ids=("tu1",))
            turn = backend.open_request.hooks.host.current_turn
            while not turn.accepted:
                await asyncio.sleep(0.005)
            crash["result"] = await handlers["clarification"](
                {"prompt": "Topic?", "output": ["t"]}
            )
            # The host dies here; the store's copy is all that survives.
            crash["record"] = copy.deepcopy(store._records["conv-test"])
            yield TurnEnd(session_id=backend.session_id, num_turns=1)

        native, factory, _, _ = make_native(
            [vendor, [text("ok")]],
            answers=["lidar"],
            record_store=store,
            session_dir=shared,
        )
        async with native:
            await native.run_agentic_loop("ask me")
        session_id = factory.last.open_request.session_id
        self.assertTrue(crash["result"].text.startswith(END_TURN_MARKER))
        record = crash["record"]
        self.assertTrue(record["pending_widget"])
        self.assertEqual(record["submission"], SubmissionState.SUBMITTED.value)
        self.assertEqual(record["status"], "active")
        self.assertEqual(record["vendor_session_id"], session_id)

        restarted = InMemoryRecordStore()
        restarted._records["conv-test"] = record
        recovered, factory2, _, _ = make_native(
            [[text("Where were we?")]], record_store=restarted, session_dir=shared
        )
        async with recovered:
            await recovered.run_agentic_loop("still there?")
            request = factory2.last.open_request
            self.assertTrue(request.resume)
            self.assertEqual(request.session_id, session_id)

    async def test_new_command_rotates_to_a_fresh_session(self) -> None:
        native, factory, _, _ = make_native([[text("one")], [text("two")]])
        async with native:
            await native.run_agentic_loop("first")
            first_id = factory.last.open_request.session_id
            await native.run_agentic_loop("/new")
            await native.run_agentic_loop("second")
            self.assertEqual(len(factory.instances), 2)
            self.assertNotEqual(factory.last.open_request.session_id, first_id)
            self.assertFalse(factory.last.open_request.resume)
            self.assertTrue(factory.instances[0].closed)

    async def test_export_and_restore_round_trip(self) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store_a, store_b = InMemoryRecordStore(), InMemoryRecordStore()
        native, factory, _, _ = make_native(
            [[tools(("enter_sop", {"name": "mini_research"})), text("ok")]],
            record_store=store_a,
            session_dir=shared,
        )
        async with native:
            await native.run_agentic_loop("go")
            blob = native.export_state(turn_number=1, iteration=1)
            session_id = factory.last.open_request.session_id
        rebuilt, factory2, _, _ = make_native(
            [[text("resumed")]], record_store=store_b, session_dir=shared
        )
        async with rebuilt:
            rebuilt.restore_state(blob)
            self.assertEqual(rebuilt.sop_state.sop_name, "mini_research")
            await rebuilt.run_agentic_loop("continue")
            self.assertEqual(factory2.last.open_request.session_id, session_id)
            self.assertTrue(factory2.last.open_request.resume)

    async def test_principal_change_is_rejected_before_contacting_the_vendor(
        self,
    ) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store = InMemoryRecordStore()
        native, _, _, _ = make_native(
            [[text("one")]], record_store=store, session_dir=shared
        )
        async with native:
            await native.run_agentic_loop("first")
        other, factory2, _, _ = make_native(
            [], record_store=store, principal="someone_else", session_dir=shared
        )
        async with other:
            with self.assertRaises(SessionResumeRejected):
                await other.run_agentic_loop("second")
            self.assertEqual(factory2.instances, [])

    async def test_cwd_change_rotates_and_queues_a_recap(self) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store = InMemoryRecordStore()
        native, _, _, _ = make_native(
            [[text("one")]], record_store=store, session_dir=shared
        )
        async with native:
            await native.run_agentic_loop("remember the number 42")
        moved, factory2, _, _ = make_native(
            [[text("two")]], record_store=store, session_dir=shared
        )
        async with moved:
            moved.backend.cwd = moved.backend.cwd + "_moved"
            moved.set_messages(
                [
                    {"role": "user", "content": "remember the number 42"},
                    {"role": "assistant", "content": "one"},
                ]
            )
            await moved.run_agentic_loop("what number?")
            self.assertFalse(factory2.last.open_request.resume)
            self.assertIn("remember the number 42", factory2.last.l2_seen[0])
            self.assertIn('type="recap"', factory2.last.l2_seen[0])

    async def test_instruction_drift_keeps_session_and_sends_one_notice(self) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store = InMemoryRecordStore()
        native, _, _, _ = make_native(
            [[text("one")]], record_store=store, session_dir=shared
        )
        async with native:
            await native.run_agentic_loop("first")
        drifted, factory2, _, _ = make_native(
            [[text("two")], [text("three")]],
            record_store=store,
            soft_max_iterations=7,
            session_dir=shared,
        )
        async with drifted:
            await drifted.run_agentic_loop("second")
            await drifted.run_agentic_loop("third")
            backend = factory2.last
            self.assertTrue(backend.open_request.resume)
            self.assertIn('type="instructions_updated"', backend.l2_seen[0])
            self.assertNotIn('type="instructions_updated"', backend.l2_seen[1])

    async def test_rewind_forks_the_vendor_session_at_the_turn_boundary(self) -> None:
        store = InMemoryRecordStore()
        native, factory, _, _ = make_native(
            [[text("t1")], [text("t2")], [text("t3")]],
            record_store=store,
            rewind_on_repeat_turn=True,
        )
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            source = factory.last.open_request.session_id
            await native.run_agentic_loop("two again", turn_number=2)
            request = factory.last.open_request
            self.assertEqual(request.fork_from, (source, "uuid-m0"))
            self.assertEqual(native._load_record().last_turn, 2)


class FailureHandlingTest(TestCase):
    async def test_permission_mismatch_fails_closed_before_contacting_the_vendor(
        self,
    ) -> None:
        shared, store = (
            tempfile.mkdtemp(prefix="af_native_test_"),
            InMemoryRecordStore(),
        )
        native, _, _, _ = make_native(
            [[text("one")]], record_store=store, session_dir=shared
        )
        async with native:
            await native.run_agentic_loop("first")
        other, factory, _, _ = make_native(
            [],
            record_store=store,
            session_dir=shared,
            backend={"kind": "claude_sdk", "cwd": shared, "permission_mode": "default"},
        )
        async with other:
            with self.assertRaises(SessionResumeRejected):
                await other.run_agentic_loop("second")
            self.assertEqual(factory.instances, [])


class IdentityRefreshTest(TestCase):
    async def test_permission_change_fails_closed_until_new_then_resumes_normally(
        self,
    ) -> None:
        native, factory, _, _ = make_native(
            [[text("one")], [text("two")], [text("three")]]
        )
        async with native:
            await native.run_agentic_loop("first")
            native.backend.permission_mode = "default"
            with self.assertRaises(SessionResumeRejected) as ctx:
                await native.run_agentic_loop("second")
            self.assertIn("/new", str(ctx.exception))
            self.assertEqual(len(factory.instances), 1)
            await native.run_agentic_loop("/new")
            await native.run_agentic_loop("second")
            await native.run_agentic_loop("third")
            record = native._load_record()
            self.assertEqual(
                record.permission_fingerprint, native._permission_fingerprint()
            )
            self.assertEqual(len(factory.instances), 2)
            self.assertFalse(factory.last.open_request.resume)
            self.assertEqual(len(factory.last.turn_requests), 2)
            self.assertTrue(factory.instances[0].closed)

    async def test_stale_adapter_version_rotates_with_a_recap(self) -> None:
        shared, store = shared_dir(), InMemoryRecordStore()
        a, _, _, _ = make_native(
            [[text("noted")]], record_store=store, session_dir=shared
        )
        async with a:
            await a.run_agentic_loop("remember canary-20")
        store._records["conv-test"]["adapter_version"] = "0"
        b, factory, _, _ = make_native(
            [[text("two")]], record_store=store, session_dir=shared
        )
        async with b:
            b.set_messages(
                [
                    {"role": "user", "content": "remember canary-20"},
                    {"role": "assistant", "content": "noted"},
                ]
            )
            await b.run_agentic_loop("what was it?")
            self.assertFalse(factory.last.open_request.resume)
            record = b._load_record()
            self.assertEqual(record.generation, 1)
            self.assertEqual(record.adapter_version, ADAPTER_VERSION)
            self.assertIn('type="recap"', factory.last.l2_seen[0])
            self.assertIn("canary-20", factory.last.l2_seen[0])


_HISTORY = [
    {"role": "user", "content": "remember canary-31"},
    {"role": "assistant", "content": "noted"},
]


class ResumePolicyTest(TestCase):
    async def asyncSetUp(self) -> None:
        self.shared = tempfile.mkdtemp(prefix="af_native_test_")
        self.store = InMemoryRecordStore()
        first, factory, _, _ = make_native(
            [[text("noted")]], record_store=self.store, session_dir=self.shared
        )
        async with first:
            await first.run_agentic_loop("remember canary-31")
        self.session_id = factory.last.open_request.session_id

    async def _next_turn(self, *, backend=None, **kw):
        """The next host turn on a rebuilt inferencer; returns its factory."""
        native, factory, _, _ = make_native(
            [[text("two")]],
            record_store=self.store,
            session_dir=self.shared,
            backend={"kind": "claude_sdk", "cwd": self.shared, **(backend or {})},
            **kw,
        )
        async with native:
            native.set_messages(list(_HISTORY))
            await native.run_agentic_loop("what was it?")
        return factory

    async def _assert_rejected(self, *, backend=None, **kw) -> FakeBackendFactory:
        native, factory, _, _ = make_native(
            [],
            record_store=self.store,
            session_dir=self.shared,
            backend={"kind": "claude_sdk", "cwd": self.shared, **(backend or {})},
            **kw,
        )
        async with native:
            with self.assertRaises(SessionResumeRejected):
                await native.run_agentic_loop("what was it?")
        self.assertEqual(factory.instances, [])
        record = self.store.load("conv-test")
        self.assertEqual(record.generation, 0)
        self.assertEqual(record.vendor_session_id, self.session_id)
        return factory

    def _assert_new_session(self, factory, *, recap: bool) -> None:
        request = factory.last.open_request
        self.assertFalse(request.resume)
        self.assertNotEqual(request.session_id, self.session_id)
        self.assertEqual(self.store.load("conv-test").generation, 1)
        l2 = factory.last.l2_seen[0]
        self.assertEqual('type="recap"' in l2, recap)
        self.assertEqual("canary-31" in l2, recap)

    # --- D9: a session that cannot or must not be continued is lost ----------

    async def test_a_backend_switch_starts_a_new_session_with_a_recap(self) -> None:
        factory = await self._next_turn(backend={"kind": "claude_cli"})
        self._assert_new_session(factory, recap=True)
        self.assertEqual(self.store.load("conv-test").backend, "claude_cli")

    async def test_a_backend_switch_under_fresh_has_no_recap(self) -> None:
        factory = await self._next_turn(
            backend={"kind": "claude_cli"}, on_session_loss="fresh"
        )
        self._assert_new_session(factory, recap=False)

    async def test_a_backend_switch_under_fail_fails_closed(self) -> None:
        await self._assert_rejected(
            backend={"kind": "claude_cli"}, on_session_loss="fail"
        )

    async def test_an_adapter_version_mismatch_is_a_session_loss(self) -> None:
        self.store._records["conv-test"]["adapter_version"] = "0"
        factory = await self._next_turn(on_session_loss="fresh")
        self._assert_new_session(factory, recap=False)

    async def test_an_adapter_version_mismatch_under_fail_fails_closed(self) -> None:
        self.store._records["conv-test"]["adapter_version"] = "0"
        await self._assert_rejected(on_session_loss="fail")

    async def test_a_cwd_change_under_fail_fails_closed(self) -> None:
        await self._assert_rejected(
            backend={"cwd": self.shared + "_moved"}, on_session_loss="fail"
        )

    # --- fingerprints of a session that would be continued ------------------

    async def test_a_principal_mismatch_fails_even_where_the_session_is_lost(
        self,
    ) -> None:
        # A recap would hand the other principal's history to the new session.
        await self._assert_rejected(
            backend={"kind": "claude_cli"}, principal="someone_else"
        )

    async def test_a_permission_change_with_a_backend_switch_is_not_a_resume(
        self,
    ) -> None:
        factory = await self._next_turn(
            backend={"kind": "claude_cli", "permission_mode": "default"}
        )
        self._assert_new_session(factory, recap=True)
        record = self.store.load("conv-test")
        self.assertNotEqual(record.permission_fingerprint, "")

    async def test_a_permission_change_on_the_same_backend_fails_closed(
        self,
    ) -> None:
        await self._assert_rejected(backend={"permission_mode": "default"})


class ModelChangeTest(TestCase):
    """A model change continues the session on the new model (plan §7.3
    ``/model``), never a stale model unnoticed."""

    async def _change_model(self, factory_cls, *, live: bool):
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store, manager = InMemoryRecordStore(), NativeRuntimeManager()
        factory = factory_cls()
        try:
            first, _, _, _ = make_native(
                [[text("one")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                factory=factory,
            )
            await first.run_agentic_loop("first")
            await first.aclose()
            if not live:
                await manager.evict_conversation("conv-test")
            second, _, _, _ = make_native(
                [[text("two")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                factory=factory,
                backend={"kind": "claude_sdk", "cwd": shared, "model": "model-b"},
            )
            await second.run_agentic_loop("second")
            await second.aclose()
        finally:
            await manager.aclose_all()
        return factory, store.load("conv-test")

    async def test_a_refused_live_switch_reopens_the_session_on_the_new_model(
        self,
    ) -> None:
        factory, record = await self._change_model(ModelRefusingFactory, live=True)
        launched, reopened = factory.instances
        self.assertTrue(launched.closed)
        self.assertTrue(reopened.open_request.resume)
        self.assertEqual(
            reopened.open_request.session_id, launched.open_request.session_id
        )
        self.assertEqual(reopened.open_request.model, "model-b")
        self.assertEqual(len(reopened.turn_requests), 1)
        self.assertEqual(record.model, "model-b")
        self.assertEqual(record.generation, 0)

    async def test_a_session_reopened_after_idle_starts_on_the_new_model(
        self,
    ) -> None:
        factory, record = await self._change_model(FakeBackendFactory, live=False)
        reopened = factory.last
        self.assertEqual(len(factory.instances), 2)
        self.assertTrue(reopened.open_request.resume)
        self.assertEqual(reopened.open_request.model, "model-b")
        self.assertEqual(record.model, "model-b")

    async def _model_command(self, factory: FakeBackendFactory):
        """A turn, ``/model model-b``, then the next turn of the same session;
        returns the record."""
        native, _, _, _ = make_native([[text("one")], [text("two")]], factory=factory)
        async with native:
            await native.run_agentic_loop("first")
            reply = await native.run_agentic_loop("/model model-b")
            self.assertIn("model-b", reply.text)
            await native.run_agentic_loop("second")
            return native._load_record()

    async def test_the_model_command_switches_the_live_session_at_the_next_turn(
        self,
    ) -> None:
        factory = FakeBackendFactory()
        record = await self._model_command(factory)
        (backend,) = factory.instances
        self.assertEqual(backend.models_set, ["model-b"])
        self.assertEqual(len(backend.turn_requests), 2)
        self.assertEqual(record.model, "model-b")

    async def test_a_model_command_the_vendor_refuses_live_reopens_the_session(
        self,
    ) -> None:
        factory = ModelRefusingFactory()
        record = await self._model_command(factory)
        launched, reopened = factory.instances
        self.assertTrue(launched.closed)
        self.assertTrue(reopened.open_request.resume)
        self.assertEqual(
            reopened.open_request.session_id, launched.open_request.session_id
        )
        self.assertEqual(reopened.open_request.model, "model-b")
        self.assertEqual(
            [len(launched.turn_requests), len(reopened.turn_requests)], [1, 1]
        )
        self.assertEqual(record.model, "model-b")
        self.assertEqual(record.generation, 0)


class SpecIsolationTest(TestCase):
    async def test_model_command_does_not_leak_into_a_shared_spec(self) -> None:
        spec = NativeBackendSpec(kind="claude_sdk", cwd=shared_dir())
        first, _, _, _ = make_native([], backend=spec, conversation_key="c1")
        second, _, _, _ = make_native([], backend=spec, conversation_key="c2")
        async with first, second:
            await first.run_agentic_loop("/model sonnet-x")
            self.assertEqual(first.backend.model, "sonnet-x")
            self.assertEqual(second.backend.model, "")
            self.assertEqual(spec.model, "")
            self.assertIsNot(first.backend, spec)


class SessionMissingRecoveryTest(TestCase):
    async def test_missing_vendor_session_is_replaced_and_the_turn_resubmitted(
        self,
    ) -> None:
        shared, store = shared_dir(), InMemoryRecordStore()
        a, _, _, _ = make_native([[text("ok")]], record_store=store, session_dir=shared)
        async with a:
            await a.run_agentic_loop("remember canary-77")

        async def missing(backend, request):
            yield VendorError(
                message="No conversation found with session ID",
                submitted=False,
                session_missing=True,
            )

        b, factory, _, _ = make_native(
            [missing, [text("fresh answer")]], record_store=store, session_dir=shared
        )
        async with b:
            # Like OpenStartup, the host's mirror already holds the message
            # being sent; the native must neither duplicate nor recap it.
            b.set_messages(
                [
                    {"role": "user", "content": "remember canary-77"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "what was it?"},
                ]
            )
            result = await b.run_agentic_loop("what was it?")
            self.assertEqual(result.text, "fresh answer")
            self.assertEqual(len(factory.instances), 2)
            first, fresh = factory.instances
            self.assertTrue(first.open_request.resume)
            self.assertFalse(fresh.open_request.resume)
            self.assertNotEqual(
                fresh.open_request.session_id, first.open_request.session_id
            )
            self.assertEqual(b._load_record().generation, 1)
            l2 = fresh.l2_seen[0]
            self.assertIn('type="recap"', l2)
            self.assertIn("canary-77", l2)
            self.assertNotIn("user: what was it?", l2)
            self.assertEqual(fresh.turn_requests[0].text, "what was it?")
            self.assertEqual(
                [m["content"] for m in b.get_messages()].count("what was it?"), 1
            )

    async def test_other_vendor_errors_do_not_rotate(self) -> None:
        async def boom(backend, request):
            yield VendorError(message="gateway 500")

        native, factory, _, _ = make_native([[text("one")], boom])
        async with native:
            await native.run_agentic_loop("first")
            with self.assertRaises(VendorTurnFailed):
                await native.run_agentic_loop("second")
            self.assertEqual(native._load_record().generation, 0)
            self.assertEqual(len(factory.instances), 1)


class HistorySeedingTest(TestCase):
    async def test_first_native_turn_with_host_history_gets_a_recap(self) -> None:
        native, factory, _, _ = make_native([[text("ok")]])
        async with native:
            native.set_messages(
                [
                    {"role": "user", "content": "earlier question CANARY-13"},
                    {"role": "assistant", "content": "earlier answer"},
                ]
            )
            await native.run_agentic_loop("now please")
            l2 = factory.last.l2_seen[0]
            self.assertIn('type="recap"', l2)
            self.assertIn("CANARY-13", l2)
            self.assertNotIn("user: now please", l2)

    async def test_fresh_policy_seeds_no_recap(self) -> None:
        native, factory, _, _ = make_native([[text("ok")]], on_session_loss="fresh")
        async with native:
            native.set_messages([{"role": "user", "content": "earlier CANARY-13"}])
            await native.run_agentic_loop("now")
            self.assertNotIn('type="recap"', factory.last.l2_seen[0])

    async def test_empty_history_seeds_no_recap(self) -> None:
        native, factory, _, _ = make_native([[text("ok")]])
        async with native:
            await native.run_agentic_loop("hello")
            self.assertNotIn('type="recap"', factory.last.l2_seen[0])


class LiveSessionDriftTest(TestCase):
    """A running Claude SDK process read the L1 file at launch and re-records
    that text when it compacts (S1), so rewriting the file does not reach it:
    the drifted session is reopened by resume, keeping its id and history."""

    async def _with_a_live_session(self, scripts, check, **kw) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store, manager = InMemoryRecordStore(), NativeRuntimeManager()
        try:
            first, factory, _, _ = make_native(
                [[text("one")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
            )
            await first.run_agentic_loop("first")
            await first.aclose()
            rebuilt, _, _, _ = make_native(
                scripts,
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                factory=factory,
                **kw,
            )
            await check(rebuilt, factory)
            await rebuilt.aclose()
        finally:
            await manager.aclose_all()

    async def test_drift_reopens_the_live_session_on_the_new_instructions(
        self,
    ) -> None:
        async def check(drifted, factory) -> None:
            await drifted.run_agentic_loop("second")
            await drifted.run_agentic_loop("third")
            launched, reopened = factory.instances
            self.assertTrue(launched.closed)
            request = reopened.open_request
            self.assertTrue(request.resume)
            self.assertEqual(request.session_id, launched.open_request.session_id)
            self.assertNotIn("7 tool rounds", launched.open_request.l1_text)
            self.assertIn("7 tool rounds", request.l1_text)
            self.assertEqual(Path(request.l1_path).read_text(), request.l1_text)
            # Until the vendor compacts, the resumed session still answers from
            # its recorded instructions: the notice supersedes them, once.
            self.assertIn('type="instructions_updated"', reopened.l2_seen[0])
            self.assertNotIn('type="instructions_updated"', reopened.l2_seen[1])
            self.assertEqual(len(reopened.turn_requests), 2)

        await self._with_a_live_session(
            [[text("two")], [text("three")]], check, soft_max_iterations=7
        )

    async def test_a_catalog_change_is_not_drift_and_keeps_the_live_session(
        self,
    ) -> None:
        async def check(rebuilt, factory) -> None:
            rebuilt.disallowed_sops.append("mini_research")
            await rebuilt.run_agentic_loop("second")
            self.assertEqual(len(factory.instances), 1)
            l2 = factory.last.l2_seen[-1]
            self.assertIn("No longer available: mini_research", l2)
            self.assertNotIn('type="instructions_updated"', l2)

        await self._with_a_live_session([[text("two")]], check)


def _extra_tool(*, depth_param: bool = False) -> ToolDefinition:
    parameters = [ParameterDef(name="query", type="string", required=True)]
    if depth_param:
        parameters.append(ParameterDef(name="--depth", type="int"))
    return ToolDefinition(
        name="lookup",
        description="Look something up.",
        tool_type="Action",
        parameters=parameters,
    )


class ToolManifestDriftTest(TestCase):
    """Plan §3.1 / D8: drift is a change of the session instructions or of
    the AF tool set (names and schemas); the catalog and nonce are not."""

    @contextlib.asynccontextmanager
    async def _second_session(self, change, **kwargs):
        """A session started with the ``lookup`` tool, continued by a rebuilt
        inferencer whose tool registry ``change`` edits; yields that
        inferencer, the backend factory and the first L1 core hash."""
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store, manager = InMemoryRecordStore(), NativeRuntimeManager()
        try:
            first, factory, _, _ = make_native(
                [[text("one")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
            )
            first.tool_registry["lookup"] = _extra_tool()
            await first.run_agentic_loop("first")
            recorded = first._load_record().l1_core_hash
            await first.aclose()
            second, _, _, _ = make_native(
                [[text("two")], [text("three")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                factory=factory,
                **kwargs,
            )
            second.tool_registry["lookup"] = _extra_tool()
            change(second)
            async with second:
                yield second, factory, recorded
        finally:
            await manager.aclose_all()

    async def test_a_new_tool_is_drift_announced_once_and_the_session_kept(
        self,
    ) -> None:
        def add_tool(native) -> None:
            native.tool_registry["lookup2"] = dataclasses.replace(
                _extra_tool(), name="lookup2"
            )

        async with self._second_session(add_tool) as (native, factory, recorded):
            await native.run_agentic_loop("second")
            await native.run_agentic_loop("third")
        launched, reopened = factory.instances
        self.assertTrue(reopened.open_request.resume)
        self.assertEqual(
            reopened.open_request.session_id, launched.open_request.session_id
        )
        self.assertIn('type="tools_updated"', reopened.l2_seen[0])
        self.assertNotIn('type="instructions_updated"', reopened.l2_seen[0])
        self.assertNotIn("tools_updated", reopened.l2_seen[1])
        record = native._load_record()
        self.assertNotEqual(record.l1_core_hash, recorded)
        self.assertEqual(
            record.l1_core_hash.partition(".")[0], recorded.partition(".")[0]
        )

    async def test_a_changed_tool_schema_is_drift(self) -> None:
        def widen(native) -> None:
            native.tool_registry["lookup"] = _extra_tool(depth_param=True)

        async with self._second_session(widen) as (native, factory, recorded):
            await native.run_agentic_loop("second")
        self.assertIn('type="tools_updated"', factory.last.l2_seen[0])
        self.assertNotEqual(native._load_record().l1_core_hash, recorded)

    async def test_tool_drift_follows_the_rotate_policy(self) -> None:
        def drop_tool(native) -> None:
            del native.tool_registry["lookup"]

        async with self._second_session(drop_tool, on_l1_drift="rotate") as (
            native,
            factory,
            _,
        ):
            await native.run_agentic_loop("second")
        launched, fresh = factory.instances
        self.assertFalse(fresh.open_request.resume)
        self.assertNotEqual(
            fresh.open_request.session_id, launched.open_request.session_id
        )
        self.assertEqual(native._load_record().generation, 1)

    async def test_an_unchanged_tool_set_is_no_drift(self) -> None:
        async with self._second_session(lambda _n: None) as (
            native,
            factory,
            recorded,
        ):
            await native.run_agentic_loop("second")
        self.assertEqual(len(factory.instances), 1)
        self.assertEqual(native._load_record().l1_core_hash, recorded)
        self.assertNotIn("_updated", factory.last.l2_seen[0])


class _ForkFailingFactory(FakeBackendFactory):
    """Backends whose ``open`` fails for a fork (the original session is fine)."""

    def __call__(self, spec, **runtime):
        backend = super().__call__(spec, **runtime)
        original = backend.open

        async def open_(request):
            if request.fork_from is not None:
                raise RuntimeError("fork transport down")
            await original(request)

        backend.open = open_
        return backend


class RewindTest(TestCase):
    async def test_rewind_without_exact_fork_fails_by_default(self) -> None:
        factory = FakeBackendFactory(
            capabilities=dataclasses.replace(FAKE_CAPABILITIES, exact_fork=False)
        )
        native, factory, _, _ = make_native(
            [[text("t1")], [text("t2")]], factory=factory, rewind_on_repeat_turn=True
        )
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            with self.assertRaises(RewindUnsupported):
                await native.run_agentic_loop("two again", turn_number=2)
            self.assertEqual(native._load_record().generation, 0)
            self.assertEqual(len(factory.instances), 1)

    async def test_rewind_without_exact_fork_can_continue_with_a_recap(self) -> None:
        factory = FakeBackendFactory(
            capabilities=dataclasses.replace(FAKE_CAPABILITIES, exact_fork=False)
        )
        native, factory, _, _ = make_native(
            [[text("t1")], [text("t2")], [text("t3")], [text("t2b")], [text("t3b")]],
            factory=factory,
            rewind_on_repeat_turn=True,
            on_rewind_unsupported="recap",
        )
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            await native.run_agentic_loop("three", turn_number=3)
            await native.run_agentic_loop("two again", turn_number=2)
            self.assertEqual(len(factory.instances), 2)
            self.assertFalse(factory.last.open_request.resume)
            self.assertEqual(native._load_record().generation, 1)
            self.assertIn('type="recap"', factory.last.l2_seen[0])
            # The recapped session holds turn 1 and the re-run turn 2: turn 3
            # continues it (it is not a repeat to rewind again).
            await native.run_agentic_loop("three again", turn_number=3)
            self.assertEqual(len(factory.instances), 2)
            self.assertEqual(native._load_record().generation, 1)

    async def test_rewind_to_the_first_turn_starts_a_fresh_session(self) -> None:
        """Plan §7.3: a repeated turn 1 keeps no earlier turn, so it runs in a
        fresh session — no fork, no recap, not a session loss — whether or not
        the backend can fork."""
        for exact_fork, policy in ((True, "recap"), (False, "fail")):
            with self.subTest(exact_fork=exact_fork, on_session_loss=policy):
                caps = dataclasses.replace(FAKE_CAPABILITIES, exact_fork=exact_fork)
                native, factory, _, _ = make_native(
                    [[text("t1")], [text("t2")], [text("t1b")], [text("t2b")]],
                    factory=FakeBackendFactory(capabilities=caps),
                    rewind_on_repeat_turn=True,
                    on_session_loss=policy,
                )
                async with native:
                    await native.run_agentic_loop("one", turn_number=1)
                    await native.run_agentic_loop("two", turn_number=2)
                    await native.run_agentic_loop("one again", turn_number=1)
                    record = native._load_record()
                    launched, fresh = factory.instances
                    self.assertTrue(launched.closed)
                    self.assertFalse(fresh.open_request.resume)
                    self.assertIsNone(fresh.open_request.fork_from)
                    self.assertNotEqual(
                        fresh.open_request.session_id,
                        launched.open_request.session_id,
                    )
                    self.assertEqual(
                        [r.text for r in fresh.turn_requests], ["one again"]
                    )
                    self.assertNotIn('type="recap"', fresh.l2_seen[0])
                    self.assertEqual((record.generation, record.last_turn), (1, 1))
                    self.assertEqual(record.vendor_session_id, fresh.session_id)
                    self.assertEqual(set(record.turn_boundaries), {"1"})
                    # Turn 2 continues the fresh session: it is no repeat.
                    await native.run_agentic_loop("two again", turn_number=2)
                    self.assertEqual(len(factory.instances), 2)
                    self.assertEqual(len(fresh.turn_requests), 2)

    async def test_failed_fork_restores_the_original_session(self) -> None:
        factory = _ForkFailingFactory()
        native, factory, _, _ = make_native(
            [[text("t1")], [text("t2")], [text("t3")]],
            factory=factory,
            rewind_on_repeat_turn=True,
        )
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            before = native._load_record()
            session_id, generation = before.vendor_session_id, before.generation
            boundaries = dict(before.turn_boundaries)
            with self.assertRaises(RewindUnsupported):
                await native.run_agentic_loop("two again", turn_number=2)
            after = native._load_record()
            self.assertEqual(
                (after.vendor_session_id, after.generation), (session_id, generation)
            )
            self.assertEqual(after.turn_boundaries, boundaries)
            await native.run_agentic_loop("three", turn_number=3)
            reopened = factory.last.open_request
            self.assertTrue(reopened.resume)
            self.assertIsNone(reopened.fork_from)
            self.assertEqual(reopened.session_id, session_id)

    async def test_recovered_widget_answer_for_the_same_turn_does_not_fork(
        self,
    ) -> None:
        native, factory, _, _ = make_native(
            [[tools(("clarification", {"prompt": "Topic?", "output": ["t"]}))]],
            answers=[None],
            rewind_on_repeat_turn=True,
        )
        async with native:
            first = await native.run_agentic_loop("ask", turn_number=1)
            self.assertTrue(first.has_conversation_tool)
            factory.scripts.append([text("Continuing with lidar.")])
            native.set_pending_widget_answer(
                {
                    "tools": [first.conversation_tool],
                    "action_tools": [],
                    "raw_value": "lidar",
                }
            )
            second = await native.run_agentic_loop("__continue__", turn_number=1)
            self.assertEqual(second.text, "Continuing with lidar.")
            self.assertEqual(len(factory.instances), 1)
            self.assertIsNone(factory.last.open_request.fork_from)
            self.assertEqual(native._load_record().generation, 0)


class EagerRewindTest(TestCase):
    """``rewind_to``: the host rewinds the vendor session BEFORE truncating its
    own history (OpenStartup resume-from-turn / checkpoint restore)."""

    async def test_rewind_to_forks_now_and_the_rerun_does_not_fork_again(self) -> None:
        native, factory, _, _ = make_native(
            [[text("t1")], [text("t2")], [text("t3")], [text("t2 again")]],
            rewind_on_repeat_turn=True,
        )
        async with native:
            for n, msg in enumerate(("one", "two", "three"), start=1):
                await native.run_agentic_loop(msg, turn_number=n)
            source = native._load_record().vendor_session_id
            await native.rewind_to(2)
            record = native._load_record()
            self.assertEqual(len(factory.instances), 2)
            self.assertEqual(factory.last.open_request.fork_from[0], source)
            self.assertEqual(record.generation, 1)
            self.assertEqual(record.last_turn, 1)
            # Kept boundaries are translated to the forked session's message ids.
            self.assertEqual(set(record.turn_boundaries), {"1"})
            self.assertTrue(
                record.turn_boundaries["1"].startswith(record.vendor_session_id + ":")
            )
            await native.run_agentic_loop("two again", turn_number=2)
            self.assertEqual(len(factory.instances), 2)  # no second fork

    async def test_rewind_to_after_a_restore_forks_at_the_restored_end(self) -> None:
        native, factory, _, _ = make_native([[text("t1")], [text("t2")]])
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            # A restored record says only turn 1 happened; its vendor session
            # still holds turn 2, so the host forces a fork at turn 1's end.
            record = native._load_record()
            record.last_turn = 1
            native._save_record()
            await native.rewind_to(2)
            self.assertIsNotNone(factory.last.open_request.fork_from)
            self.assertEqual(native._load_record().last_turn, 1)

    async def test_failed_rewind_leaves_the_original_session(self) -> None:
        native, factory, _, _ = make_native(
            [[text("t1")], [text("t2")]], factory=_ForkFailingFactory()
        )
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            before = native._load_record()
            sid, gen = before.vendor_session_id, before.generation
            with self.assertRaises(RewindUnsupported):
                await native.rewind_to(2)
            after = native._load_record()
            self.assertEqual((after.vendor_session_id, after.generation), (sid, gen))

    async def test_failed_fork_under_recap_policy_continues_with_a_recap(self) -> None:
        native, factory, _, _ = make_native(
            [
                [text("t1")],
                [text("t2")],
                [text("t3")],
                [text("t2 again")],
                [text("t3 again")],
            ],
            factory=_ForkFailingFactory(),
            on_rewind_unsupported="recap",
        )
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            await native.run_agentic_loop("three", turn_number=3)
            before = native._load_record()
            sid, gen = before.vendor_session_id, before.generation
            await native.rewind_to(2)
            after = native._load_record()
            self.assertGreater(after.generation, gen)
            self.assertNotEqual(after.vendor_session_id, sid)
            self.assertIsNone(factory.last.open_request.fork_from)
            self.assertFalse(factory.last.open_request.resume)
            self.assertIn("recap", [n["type"] for n in after.pending_notices()])
            # The fresh session recaps turn 1 only: re-running turn 2 and then
            # turn 3 continues it without another rewind.
            self.assertEqual(after.last_turn, 1)
            sessions = len(factory.instances)
            await native.run_agentic_loop("two again", turn_number=2)
            await native.run_agentic_loop("three again", turn_number=3)
            self.assertEqual(len(factory.instances), sessions)
            self.assertEqual(native._load_record().last_turn, 3)

    async def test_rewind_to_the_first_turn_opens_a_fresh_session_now(self) -> None:
        native, factory, _, _ = make_native([[text("t1")], [text("t2")], [text("t1b")]])
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            await native.rewind_to(1)
            launched, fresh = factory.instances
            record = native._load_record()
            self.assertTrue(launched.closed)
            self.assertFalse(fresh.open_request.resume)
            self.assertIsNone(fresh.open_request.fork_from)
            self.assertEqual(
                (record.generation, record.last_turn, record.turn_boundaries),
                (1, 0, {}),
            )
            self.assertEqual(record.pending_notices(), [])
            await native.run_agentic_loop("one again", turn_number=1)
            self.assertEqual(len(factory.instances), 2)  # no second rotation
            self.assertEqual([r.text for r in fresh.turn_requests], ["one again"])
            self.assertEqual(native._load_record().generation, 1)

    async def test_rewind_to_without_a_session_is_a_no_op(self) -> None:
        native, factory, _, _ = make_native([])
        async with native:
            await native.rewind_to(3)
            self.assertEqual(factory.instances, [])


class RootCommandTest(TestCase):
    """``/root`` is a user-requested new agent session in the new directory
    (plan §7.3, like ``/new``), never a session loss."""

    async def test_root_starts_a_new_session_there_under_every_policy(self) -> None:
        for policy in ("recap", "fresh", "fail"):
            with self.subTest(on_session_loss=policy):
                root = tempfile.mkdtemp(prefix="af_native_root_")
                native, factory, _, _ = make_native(
                    [[text("one")], [text("two")]], on_session_loss=policy
                )
                async with native:
                    await native.run_agentic_loop("remember canary-7")
                    reply = await native.run_agentic_loop(f"/root {root}")
                    await native.run_agentic_loop("what was it?")
                    record = native._load_record()
                self.assertIn("new agent session", reply.text)
                launched, started = factory.instances
                self.assertTrue(launched.closed)
                self.assertFalse(started.open_request.resume)
                self.assertEqual(started.open_request.cwd, root)
                self.assertNotEqual(
                    started.open_request.session_id, launched.open_request.session_id
                )
                self.assertEqual((record.cwd, record.generation), (root, 1))
                recap = policy == "recap"
                self.assertEqual('type="recap"' in started.l2_seen[0], recap)
                self.assertEqual("canary-7" in started.l2_seen[0], recap)

    async def test_root_at_the_current_directory_keeps_the_session(self) -> None:
        native, factory, _, _ = make_native([[text("one")], [text("two")]])
        async with native:
            await native.run_agentic_loop("first")
            reply = await native.run_agentic_loop(f"/root {native.backend.cwd}")
            await native.run_agentic_loop("second")
        self.assertIn("already", reply.text)
        self.assertNotIn("new agent session", reply.text)
        (backend,) = factory.instances
        self.assertEqual(len(backend.turn_requests), 2)

    async def test_root_leaves_another_principals_session_to_the_identity_check(
        self,
    ) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        store = InMemoryRecordStore()
        native, _, _, _ = make_native(
            [[text("one")]], record_store=store, session_dir=shared, principal="alice"
        )
        async with native:
            await native.run_agentic_loop("first")
        other, factory, _, _ = make_native(
            [], record_store=store, session_dir=shared, principal="bob"
        )
        async with other:
            other.set_messages([{"role": "user", "content": "first"}])
            await other.run_agentic_loop(f"/root {tempfile.mkdtemp()}")
            with self.assertRaises(SessionResumeRejected):
                await other.run_agentic_loop("second")
        self.assertEqual(factory.instances, [])
        self.assertEqual(store.load("conv-test").generation, 0)


class RepeatedTurnTest(TestCase):
    """Plan §7.3 / D10: a repeated host turn number forks only with the host's
    ``rewind_on_repeat_turn`` opt-in, and never at turn 0 (a host that does
    not number its turns)."""

    async def test_without_the_host_opt_in_a_repeated_turn_continues(self) -> None:
        native, factory, _, _ = make_native([[text("t1")], [text("t2")], [text("t2b")]])
        async with native:
            await native.run_agentic_loop("one", turn_number=1)
            await native.run_agentic_loop("two", turn_number=2)
            await native.run_agentic_loop("two again", turn_number=2)
            record = native._load_record()
        (backend,) = factory.instances
        self.assertIsNone(backend.open_request.fork_from)
        self.assertEqual(len(backend.turn_requests), 3)
        self.assertEqual((record.generation, record.last_turn), (0, 2))

    async def test_a_repeated_turn_zero_never_forks(self) -> None:
        native, factory, _, _ = make_native(
            [[text("a")], [text("b")], [text("c")]], rewind_on_repeat_turn=True
        )
        async with native:
            for message in ("a", "b", "c"):
                await native.run_agentic_loop(message, turn_number=0)
            record = native._load_record()
        (backend,) = factory.instances
        self.assertIsNone(backend.open_request.fork_from)
        self.assertEqual(len(backend.turn_requests), 3)
        self.assertEqual(record.generation, 0)


class DriftPolicyTest(TestCase):
    """D8 ``rotate`` and ``fail`` (``notice`` above): the session instructions
    of an existing session changed."""

    async def _drifted(self, policy: str) -> tuple:
        shared, store = shared_dir(), InMemoryRecordStore()
        first, _, _, _ = make_native(
            [[text("noted")]], record_store=store, session_dir=shared
        )
        async with first:
            await first.run_agentic_loop("remember canary-41")
        recorded = store.load("conv-test")
        drifted, factory, _, _ = make_native(
            [[text("two")]],
            record_store=store,
            session_dir=shared,
            soft_max_iterations=7,
            on_l1_drift=policy,
        )
        drifted.set_messages(
            [
                {"role": "user", "content": "remember canary-41"},
                {"role": "assistant", "content": "noted"},
            ]
        )
        return drifted, factory, store, recorded

    async def test_rotate_starts_a_new_session_on_the_new_instructions(self) -> None:
        native, factory, store, recorded = await self._drifted("rotate")
        async with native:
            await native.run_agentic_loop("what was it?")
        (backend,) = factory.instances
        self.assertFalse(backend.open_request.resume)
        self.assertNotEqual(backend.open_request.session_id, recorded.vendor_session_id)
        self.assertIn("7 tool rounds", backend.open_request.l1_text)
        record = store.load("conv-test")
        self.assertEqual(record.generation, 1)
        self.assertNotEqual(record.l1_core_hash, recorded.l1_core_hash)
        l2 = backend.l2_seen[0]
        self.assertIn('type="recap"', l2)
        self.assertIn("canary-41", l2)
        self.assertNotIn("instructions_updated", l2)

    async def test_fail_refuses_the_turn_before_contacting_the_vendor(self) -> None:
        native, factory, store, recorded = await self._drifted("fail")
        async with native:
            with self.assertRaises(StablePolicyChanged) as ctx:
                await native.run_agentic_loop("what was it?")
        self.assertIn("Session instructions changed", str(ctx.exception))
        self.assertEqual(factory.instances, [])
        self.assertEqual(store.load("conv-test").to_dict(), recorded.to_dict())


class SessionLossPolicyTest(TestCase):
    """D9: the recap is bounded and labelled and leaves out the message being
    sent; under ``fail`` a lost session fails closed, never re-submitted."""

    async def test_a_recap_is_bounded_and_keeps_the_newest_turns(self) -> None:
        native, factory, _, _ = make_native([[text("ok")]], recap_max_chars=120)
        async with native:
            native.set_messages(
                [
                    {"role": "user", "content": "OLDEST " + "x" * 200},
                    {"role": "assistant", "content": "middle"},
                    {"role": "user", "content": "NEWEST question"},
                ]
            )
            await native.run_agentic_loop("now please")
        recap = re.search(
            r'<notice type="recap">(.*?)</notice>', factory.last.l2_seen[0], re.S
        ).group(1)
        label, body = recap.split("\n", 1)
        self.assertEqual(
            label,
            "Earlier conversation in this host session (recap; you did not see "
            "these turns in this agent session):",
        )
        self.assertEqual(len(body), len("… ") + 120)
        self.assertTrue(body.startswith("… "))
        self.assertTrue(body.endswith("assistant: middle\nuser: NEWEST question"))
        self.assertNotIn("OLDEST", body)
        self.assertNotIn("now please", body)

    async def test_history_without_a_session_fails_closed_under_fail(self) -> None:
        native, factory, _, _ = make_native([], on_session_loss="fail")
        async with native:
            native.set_messages([{"role": "user", "content": "earlier"}])
            with self.assertRaises(SessionResumeRejected) as ctx:
                await native.run_agentic_loop("now")
        self.assertIn("has history but no agent session", str(ctx.exception))
        self.assertEqual(factory.instances, [])

    async def test_a_missing_vendor_session_fails_closed_under_fail(self) -> None:
        shared, store = shared_dir(), InMemoryRecordStore()
        first, _, _, _ = make_native(
            [[text("ok")]], record_store=store, session_dir=shared
        )
        async with first:
            await first.run_agentic_loop("remember canary-77")

        async def missing(backend, request):
            yield VendorError(
                message="No conversation found with session ID",
                submitted=False,
                session_missing=True,
            )

        native, factory, _, _ = make_native(
            [missing], record_store=store, session_dir=shared, on_session_loss="fail"
        )
        async with native:
            with self.assertRaises(SessionResumeRejected):
                await native.run_agentic_loop("what was it?")
        (backend,) = factory.instances
        self.assertEqual([r.text for r in backend.turn_requests], ["what was it?"])
        record = store.load("conv-test")
        self.assertEqual((record.generation, record.submission), (0, "prepared"))


class ClearCommandTest(TestCase):
    async def test_clear_starts_a_new_session_and_clears_the_mirror(self) -> None:
        native, factory, _, _ = make_native([[text("noted")], [text("two")]])
        async with native:
            await native.run_agentic_loop("remember canary-5")
            reply = await native.run_agentic_loop("/clear")
            self.assertEqual(
                native.get_messages(),
                [
                    {"role": "user", "content": "/clear"},
                    {"role": "assistant", "content": reply.text},
                ],
            )
            await native.run_agentic_loop("what was it?")
            self.assertEqual(native._load_record().generation, 1)
        launched, fresh = factory.instances
        self.assertTrue(launched.closed)
        self.assertFalse(fresh.open_request.resume)
        self.assertNotEqual(
            fresh.open_request.session_id, launched.open_request.session_id
        )
        self.assertNotIn("canary-5", fresh.l2_seen[0])
        self.assertNotIn('type="recap"', fresh.l2_seen[0])


class IdleResumeTest(TestCase):
    async def test_an_idle_closed_session_resumes_transparently(self) -> None:
        manager = NativeRuntimeManager(idle_close_seconds=0.2)
        try:
            native, factory, _, _ = make_native(
                [[text("one")], [text("two")]], runtime_manager=manager
            )
            await native.run_agentic_loop("first")
            await wait_for(lambda: factory.instances[0].closed, timeout=3.0)
            await native.run_agentic_loop("second")
            await native.aclose()
        finally:
            await manager.aclose_all()
        launched, resumed = factory.instances
        self.assertTrue(resumed.open_request.resume)
        self.assertEqual(
            resumed.open_request.session_id, launched.open_request.session_id
        )
        # The vendor session holds the state it was told: nothing is re-sent.
        self.assertEqual(resumed.l2_seen, [""])
        record = native._load_record()
        self.assertEqual((record.generation, record.submission), (0, "committed"))
        self.assertEqual(record.pending_notices(), [])
        self.assertEqual(
            [m["content"] for m in native.get_messages()],
            ["first", "one", "second", "two"],
        )


class RecordPersistenceTest(TestCase):
    """Plan §7.1: the record is saved after every state-changing bridge call,
    mid-turn, and after every vendor turn."""

    async def test_the_record_is_saved_after_a_bridge_call_and_after_the_turn(
        self,
    ) -> None:
        store = InMemoryRecordStore()
        mid_turn = {}

        async def vendor(backend, request):
            handlers = {t.name: t.handler for t in backend.open_request.tools}
            yield MessageEnd(
                message_id="m0", text="", af_tool_use_ids=("tu1",), message_uuid="u0"
            )
            turn = backend.open_request.hooks.host.current_turn
            await wait_for(lambda: turn.accepted)
            before = store.load("conv-test").saved_at
            await handlers["enter_sop"]({"name": "mini_research"})
            mid_turn["saved"] = store.load("conv-test").saved_at > before
            yield turn_end(backend)

        native, factory, _, _ = make_native([vendor], record_store=store)
        async with native:
            await native.run_agentic_loop("start it", turn_number=4)
        self.assertTrue(mid_turn["saved"])
        record = store.load("conv-test")
        self.assertEqual(record.submission, "committed")
        self.assertEqual(record.last_turn, 4)
        self.assertEqual(record.turn_boundaries, {"4": "u0"})
        self.assertEqual(record.vendor_session_id, factory.last.session_id)


class PrivateFileTest(TestCase):
    """Invariant 9: the conversation's files (session instructions, spilled
    turn context) are 0600, also where a file of that name already existed
    with wider permissions."""

    async def test_files_that_existed_world_readable_are_rewritten_private(
        self,
    ) -> None:
        shared = shared_dir()
        l1 = Path(shared) / "l1_0.md"
        spill = Path(shared) / "turn_context" / "notice_1.txt"
        spill.parent.mkdir(mode=0o700)
        for path in (l1, spill):
            path.write_text("stale", encoding="utf-8")
            os.chmod(path, 0o644)
        native, factory, _, _ = make_native([[text("one")]], session_dir=shared)
        async with native:
            await native.run_agentic_loop("first")
            spilled = native._spill("notice_1.txt", "the full notice")
        self.assertEqual(spilled, str(spill))
        for path in (l1, spill):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(
            l1.read_text(encoding="utf-8"), factory.last.open_request.l1_text
        )
        self.assertEqual(spill.read_text(encoding="utf-8"), "the full notice")
        temporary = [
            entry.name
            for directory in (Path(shared), spill.parent)
            for entry in directory.iterdir()
            if entry.name.startswith(".")
        ]
        self.assertEqual(temporary, [])


class RestoreConflictTest(TestCase):
    """Plan §7.1: on ``restore_state`` the newest record generation wins,
    whether the exported state or the host's store holds it."""

    async def _two_generations(self) -> tuple:
        shared, store = shared_dir(), InMemoryRecordStore()
        native, factory, _, _ = make_native(
            [[text("one")], [text("two")]], record_store=store, session_dir=shared
        )
        async with native:
            await native.run_agentic_loop("first")
            old_blob = native.export_state(turn_number=1)
            old_record = dict(store._records["conv-test"])
            await native.run_agentic_loop("/new")
            await native.run_agentic_loop("second")
            new_blob = native.export_state(turn_number=2)
        newest = factory.last.open_request.session_id
        return shared, store, old_blob, old_record, new_blob, newest

    async def test_a_stale_exported_state_does_not_replace_a_newer_record(
        self,
    ) -> None:
        shared, store, old_blob, _, _, newest = await self._two_generations()
        rebuilt, factory, _, _ = make_native(
            [[text("three")]], record_store=store, session_dir=shared
        )
        async with rebuilt:
            rebuilt.restore_state(old_blob)
            self.assertEqual(rebuilt._load_record().generation, 1)
            await rebuilt.run_agentic_loop("third")
        self.assertTrue(factory.last.open_request.resume)
        self.assertEqual(factory.last.open_request.session_id, newest)

    async def test_a_newer_exported_state_replaces_an_older_record(self) -> None:
        shared, _, _, old_record, new_blob, newest = await self._two_generations()
        older = InMemoryRecordStore()
        older._records["conv-test"] = old_record
        rebuilt, factory, _, _ = make_native(
            [[text("three")]], record_store=older, session_dir=shared
        )
        async with rebuilt:
            rebuilt.restore_state(new_blob)
            await rebuilt.run_agentic_loop("third")
        self.assertTrue(factory.last.open_request.resume)
        self.assertEqual(factory.last.open_request.session_id, newest)
        self.assertEqual(older.load("conv-test").generation, 1)

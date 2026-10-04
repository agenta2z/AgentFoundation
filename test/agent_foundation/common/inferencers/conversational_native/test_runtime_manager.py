"""NativeRuntimeManager and SessionActor (plan §7.2): leases, idle and LRU
close, context isolation and the generation fence.

Plan §12.1 row coverage (assertion: tests; Class.method or a whole Class):
- leases (evict-then-rebuild reuses the actor):
    LeaseTest, RebuildAdoptionTest
    NativeSessionLifecycleTest.test_shared_runtime_manager_reuses_the_live_actor_across_rebuilds
- idle / LRU / backend-switch close:
    RuntimeManagerTest, BackendSwitchTest
- aclose_all idempotent:
    ACloseAllTest
- the actor task has fresh contextvars:
    ContextIsolationTest
- a stale generation cannot write:
    FailureHandlingTest.test_stale_generation_cannot_overwrite_a_rotated_record
    (the stores' fence itself: test_record_store)
"""

from __future__ import annotations

import asyncio
import contextvars
import tempfile

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeRuntimeManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.record import (
    StaleRecordError,
)
from agent_foundation.resources.tools.models import ToolDefinition
from fakes import text, tools, turn_end
from helpers import make_native, shared_dir, wait_for
from later.unittest import TestCase


_TEST_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "af_native_matrix_var", default="unset"
)


class NativeSessionLifecycleTest(TestCase):
    async def test_shared_runtime_manager_reuses_the_live_actor_across_rebuilds(
        self,
    ) -> None:
        shared = tempfile.mkdtemp(prefix="af_native_test_")
        manager = NativeRuntimeManager()
        store = InMemoryRecordStore()
        try:
            native, factory, _, _ = make_native(
                [[text("one")], [text("two")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
            )
            await native.run_agentic_loop("first")
            await native.aclose()
            rebuilt, _, _, _ = make_native(
                [], record_store=store, runtime_manager=manager, session_dir=shared
            )
            rebuilt.backend_factory = factory
            await rebuilt.run_agentic_loop("second")
            await rebuilt.aclose()
            self.assertEqual(len(factory.instances), 1)
            self.assertEqual(len(factory.last.turn_requests), 2)
        finally:
            await manager.aclose_all()


class RebuildAdoptionTest(TestCase):
    async def test_rebuilt_inferencer_drives_af_tools_of_the_adopted_actor(
        self,
    ) -> None:
        shared, store, manager = (
            shared_dir(),
            InMemoryRecordStore(),
            NativeRuntimeManager(),
        )
        try:
            a, factory, _, exec_a = make_native(
                [[text("one")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
            )
            await a.run_agentic_loop("first")
            await a.aclose()
            b, _, _, exec_b = make_native(
                [[tools(("write_brief", {"topic": "lidar"})), text("done")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                factory=factory,
            )
            result = await b.run_agentic_loop("write it")
            await b.aclose()
            backend = factory.last
            self.assertEqual(len(factory.instances), 1)
            self.assertEqual(exec_b.calls, [("write_brief", {"topic": "lidar"})])
            self.assertEqual(exec_a.calls, [])
            self.assertEqual(
                backend.tool_results[0], ("write_brief", "write_brief done", False)
            )
            # The adopted session already holds this state, so no L2 is due.
            self.assertEqual(backend.l2_seen, [backend.l2_seen[0], ""])
            self.assertEqual(
                [x.tool for x in result.completed_actions], ["write_brief"]
            )
        finally:
            await manager.aclose_all()

    async def test_changed_tool_set_reopens_the_live_session(self) -> None:
        shared, store, manager = (
            shared_dir(),
            InMemoryRecordStore(),
            NativeRuntimeManager(),
        )
        try:
            a, factory, _, _ = make_native(
                [[text("one")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
            )
            await a.run_agentic_loop("first")
            await a.aclose()
            b, _, _, _ = make_native(
                [[text("two")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                factory=factory,
            )
            b.tool_registry["probe_tool"] = ToolDefinition(
                name="probe_tool", description="Probe.", tool_type="Action"
            )
            await b.run_agentic_loop("second")
            await b.aclose()
            self.assertEqual(len(factory.instances), 2)
            self.assertTrue(factory.instances[0].closed)
            reopened = factory.instances[1].open_request
            self.assertTrue(reopened.resume)
            self.assertEqual(
                reopened.session_id, factory.instances[0].open_request.session_id
            )
            self.assertIn("probe_tool", [t.name for t in reopened.tools])
            self.assertEqual(len(factory.instances[1].turn_requests), 1)
        finally:
            await manager.aclose_all()

    async def test_host_model_change_applies_to_the_adopted_session(self) -> None:
        shared, store, manager = (
            shared_dir(),
            InMemoryRecordStore(),
            NativeRuntimeManager(),
        )
        try:
            a, factory, _, _ = make_native(
                [[text("one")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
            )
            await a.run_agentic_loop("first")
            await a.aclose()
            b, _, _, _ = make_native(
                [[text("two")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                factory=factory,
                backend={"kind": "claude_sdk", "cwd": shared, "model": "model-b"},
            )
            await b.run_agentic_loop("second")
            await b.aclose()
            self.assertEqual(len(factory.instances), 1)
            self.assertEqual(factory.last.models_set, ["model-b"])
            self.assertEqual(b._load_record().model, "model-b")
        finally:
            await manager.aclose_all()


class RuntimeManagerTest(TestCase):
    async def test_least_recently_used_idle_session_is_closed_even_if_leased(
        self,
    ) -> None:
        manager = NativeRuntimeManager(max_live_sessions=2)
        natives = []
        try:
            for key in ("c1", "c2", "c3"):
                native, factory, _, _ = make_native(
                    [[text(key)], [text(f"{key} again")]],
                    runtime_manager=manager,
                    conversation_key=key,
                )
                natives.append((native, factory))
            for native, _ in natives:
                await native.run_agentic_loop("hi")
            (n1, f1), (_n2, f2), (_n3, f3) = natives
            self.assertTrue(f1.instances[0].closed)
            self.assertFalse(f2.instances[0].closed)
            self.assertFalse(f3.instances[0].closed)
            await n1.run_agentic_loop("again")
            self.assertEqual(len(f1.instances), 2)
            self.assertTrue(f1.instances[1].open_request.resume)
            self.assertTrue(f2.instances[0].closed)
        finally:
            for native, _ in natives:
                await native.aclose()
            await manager.aclose_all()

    async def test_idle_session_closes_and_the_next_turn_resumes(self) -> None:
        manager = NativeRuntimeManager(idle_close_seconds=0.2)
        try:
            native, factory, _, _ = make_native(
                [[text("one")], [text("two")]], runtime_manager=manager
            )
            await native.run_agentic_loop("first")
            await wait_for(lambda: factory.instances[0].closed, timeout=3.0)
            await native.run_agentic_loop("second")
            self.assertEqual(len(factory.instances), 2)
            self.assertTrue(factory.instances[1].open_request.resume)
            await native.aclose()
        finally:
            await manager.aclose_all()


class ContextIsolationTest(TestCase):
    async def test_actor_task_does_not_inherit_the_callers_context(self) -> None:
        seen = {}

        async def peek(backend, request):
            seen["value"] = _TEST_VAR.get()
            yield turn_end(backend)

        native, _, _, _ = make_native([peek])
        async with native:
            token = _TEST_VAR.set("leaked")
            try:
                await native.run_agentic_loop("go")
            finally:
                _TEST_VAR.reset(token)
            self.assertEqual(seen["value"], "unset")


class FailureHandlingTest(TestCase):
    async def test_stale_generation_cannot_overwrite_a_rotated_record(self) -> None:
        shared, store = (
            tempfile.mkdtemp(prefix="af_native_test_"),
            InMemoryRecordStore(),
        )
        stale, _, _, _ = make_native([], record_store=store, session_dir=shared)
        fresh, _, _, _ = make_native([], record_store=store, session_dir=shared)
        async with stale, fresh:
            stale._load_record()
            stale._save_record()
            await fresh.run_agentic_loop("/new")
            self.assertEqual(store.load("conv-test").generation, 1)
            with self.assertRaises(StaleRecordError):
                stale._save_record()
            self.assertEqual(stale._load_record().generation, 1)


class LeaseTest(TestCase):
    async def test_every_inferencer_holds_one_lease_on_the_shared_actor(
        self,
    ) -> None:
        """Plan §7.2: one actor per conversation, one lease per inferencer
        driving it; a released lease leaves the session live (resumable until
        idle or evicted)."""
        shared, store, manager = (
            shared_dir(),
            InMemoryRecordStore(),
            NativeRuntimeManager(),
        )
        try:
            first, factory, _, _ = make_native(
                [[text("one")], [text("two")], [text("three")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
            )
            second, _, _, _ = make_native(
                [],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                factory=factory,
            )
            await first.run_agentic_loop("first")
            await second.run_agentic_loop("second")
            key = first._lease_key
            self.assertEqual(second._lease_key, key)
            self.assertEqual(manager._entries[key].leases, 2)
            await second.aclose()
            self.assertEqual(manager._entries[key].leases, 1)
            await first.run_agentic_loop("third")
            await first.aclose()
            self.assertEqual(manager._entries[key].leases, 0)
            self.assertIsNotNone(manager.live_actor(key))
            (backend,) = factory.instances
            self.assertEqual(len(backend.turn_requests), 3)
            self.assertFalse(backend.closed)
        finally:
            await manager.aclose_all()

    async def test_leases_taken_before_the_actor_exists_all_count(self) -> None:
        """Inferencers that acquire a session before anyone started its actor
        (e.g. both waited on the manager's lock) each hold a lease: releasing
        one does not hand the session to idle close while another holds it."""
        manager = NativeRuntimeManager(idle_close_seconds=0.05)
        key = ("conv-test", "claude_sdk", 0)
        started: list[_StubActor] = []

        async def factory() -> _StubActor:
            started.append(_StubActor())
            return started[-1]

        try:
            await asyncio.gather(manager.acquire(key), manager.acquire(key))
            first, second = await asyncio.gather(
                manager.get_actor(key, factory), manager.get_actor(key, factory)
            )
            self.assertIs(first, second)
            self.assertEqual(len(started), 1)
            self.assertEqual(manager._entries[key].leases, 2)
            await manager.release(key)
            self.assertEqual(manager._entries[key].leases, 1)
            self.assertIsNone(manager._entries[key].idle_timer)
            await asyncio.sleep(0.15)
            self.assertTrue(first.alive)
            await manager.release(key)
            await wait_for(lambda: not first.alive)
            # A lease given up before the actor exists is not counted later.
            other = ("conv-other", "claude_sdk", 0)
            await manager.acquire(other)
            await manager.acquire(other)
            await manager.release(other)
            await manager.get_actor(other, factory)
            self.assertEqual(manager._entries[other].leases, 1)
        finally:
            await manager.aclose_all()


class _StubActor:
    """The part of ``SessionActor`` the manager uses."""

    def __init__(self) -> None:
        self.alive = True
        self.in_turn = False

    async def close(self) -> None:
        self.alive = False


class BackendSwitchTest(TestCase):
    async def test_a_backend_switch_closes_the_old_session_and_recaps_on_the_new(
        self,
    ) -> None:
        """Plan §7.3: on a backend switch the host evicts the conversation
        (OpenStartup awaits it): the old backend's session closes and the new
        backend starts a session of its own with a recap (D9)."""
        shared, store, manager = (
            shared_dir(),
            InMemoryRecordStore(),
            NativeRuntimeManager(),
        )
        try:
            old, old_factory, _, _ = make_native(
                [[text("noted")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
            )
            await old.run_agentic_loop("remember canary-12")
            await old.aclose()
            await manager.evict_conversation("conv-test")
            (launched,) = old_factory.instances
            self.assertTrue(launched.closed)
            self.assertEqual(manager._entries, {})
            new, new_factory, _, _ = make_native(
                [[text("12")]],
                record_store=store,
                runtime_manager=manager,
                session_dir=shared,
                backend={"kind": "claude_cli", "cwd": shared},
            )
            new.set_messages(
                [
                    {"role": "user", "content": "remember canary-12"},
                    {"role": "assistant", "content": "noted"},
                ]
            )
            await new.run_agentic_loop("what was it?")
            await new.aclose()
            request = new_factory.last.open_request
            self.assertFalse(request.resume)
            self.assertNotEqual(request.session_id, launched.open_request.session_id)
            self.assertIn('type="recap"', new_factory.last.l2_seen[0])
            self.assertIn("canary-12", new_factory.last.l2_seen[0])
            record = store.load("conv-test")
            self.assertEqual((record.backend, record.generation), ("claude_cli", 1))
        finally:
            await manager.aclose_all()


class ACloseAllTest(TestCase):
    async def test_aclose_all_closes_each_session_once_and_may_be_repeated(
        self,
    ) -> None:
        manager = NativeRuntimeManager()
        try:
            closes, natives = [], []
            for key in ("c1", "c2"):
                native, factory, _, _ = make_native(
                    [[text("hi")], [text("again")]],
                    runtime_manager=manager,
                    conversation_key=key,
                )
                await native.run_agentic_loop("hi")
                backend = factory.last
                close = backend.close

                async def counted(key=key, close=close) -> None:
                    closes.append(key)
                    await close()

                backend.close = counted
                natives.append((native, factory))
            await manager.aclose_all()
            await manager.aclose_all()
            self.assertEqual(sorted(closes), ["c1", "c2"])
            self.assertEqual(manager._entries, {})
            # The manager stays usable: the next turn resumes the vendor session.
            native, factory = natives[0]
            await native.run_agentic_loop("again")
            self.assertEqual(len(factory.instances), 2)
            self.assertTrue(factory.last.open_request.resume)
            self.assertEqual(
                factory.last.open_request.session_id,
                factory.instances[0].open_request.session_id,
            )
        finally:
            await manager.aclose_all()

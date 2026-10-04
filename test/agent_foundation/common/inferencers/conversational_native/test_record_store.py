"""Native session record stores: compare-and-swap on (generation, saved_at)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session import (
    record as record_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.record import (
    CallbackRecordStore,
    end_vendor_session,
    NativeSessionRecord,
    RunContextRecordStore,
    StaleRecordError,
)
from agent_foundation.common.inferencers.run_context import RunContext, RunStateStore
from fakes import text
from helpers import make_native
from later.unittest import TestCase


def _stores() -> dict:
    data: dict = {}
    return {
        "in-memory": InMemoryRecordStore(),
        "callback": CallbackRecordStore(data.get, data.__setitem__),
        "run-context": RunContextRecordStore(RunContext.root()),
    }


def _started(generation: int = 0) -> NativeSessionRecord:
    return NativeSessionRecord(
        conversation_key="k", generation=generation, vendor_session_id="s"
    )


class CompareAndSwapTest(unittest.TestCase):
    def test_a_copy_loaded_before_a_later_save_loses(self) -> None:
        for name, store in _stores().items():
            with self.subTest(store=name):
                store.save(_started())
                mine, theirs = store.load("k"), store.load("k")
                theirs.last_turn = 2
                store.save(theirs)
                mine.last_turn = 5
                with self.assertRaises(StaleRecordError):
                    store.save(mine)
                self.assertEqual(store.load("k").last_turn, 2)
                self.assertTrue(store.load("k").newer_than(mine))

    def test_a_new_generation_replaces_a_later_save_of_the_old_one(self) -> None:
        for name, store in _stores().items():
            with self.subTest(store=name):
                store.save(_started())
                rotated, other = store.load("k"), store.load("k")
                store.save(other)
                rotated.rotate()
                store.save(rotated)
                self.assertEqual(store.load("k").generation, 1)

    def test_saves_are_stamped_in_order_even_within_one_clock_tick(self) -> None:
        for name, store in _stores().items():
            with (
                self.subTest(store=name),
                mock.patch.object(record_module.time, "time", return_value=100.0),
            ):
                record = _started()
                store.save(record)
                first = record.saved_at
                stale = store.load("k")
                store.save(record)
                self.assertGreater(record.saved_at, first)
                self.assertEqual(store.load("k").saved_at, record.saved_at)
                with self.assertRaises(StaleRecordError):
                    store.save(stale)

    def test_a_failed_write_leaves_the_copy_unchanged(self) -> None:
        data: dict = {}

        def broken(key: str, value: dict) -> None:
            raise OSError("disk full")

        store = CallbackRecordStore(data.get, broken)
        record = _started()
        with self.assertRaises(OSError):
            store.save(record)
        self.assertEqual(record.saved_at, 0.0)

    def test_concurrent_writers_of_one_copy_have_exactly_one_winner(self) -> None:
        data: dict = {}
        store = CallbackRecordStore(data.get, data.__setitem__)
        store.save(_started())

        def slow_load(key: str):
            loaded = data.get(key)
            time.sleep(0.01)  # widen the load -> write window
            return loaded

        racing = CallbackRecordStore(slow_load, data.__setitem__, lock=store._lock)
        copies = [racing.load("k") for _ in range(8)]
        outcomes: list = []

        def save(copy: NativeSessionRecord) -> None:
            try:
                racing.save(copy)
                outcomes.append("saved")
            except StaleRecordError:
                outcomes.append("stale")

        threads = [threading.Thread(target=save, args=(c,)) for c in copies]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(outcomes), ["saved"] + ["stale"] * 7)


class InferencerCompareAndSwapTest(TestCase):
    async def test_a_stale_inferencer_cannot_undo_a_newer_save(self) -> None:
        shared, store = (
            tempfile.mkdtemp(prefix="af_native_test_"),
            InMemoryRecordStore(),
        )
        stale, _, _, _ = make_native([], record_store=store, session_dir=shared)
        fresh, _, _, _ = make_native([], record_store=store, session_dir=shared)
        async with stale, fresh:
            stale._load_record()
            stale._save_record()
            fresh._load_record().last_turn = 4
            fresh._save_record()
            stale._record.last_turn = 9
            with self.assertRaises(StaleRecordError):
                stale._save_record()
            self.assertEqual(store.load("conv-test").last_turn, 4)
            # The stale inferencer re-reads the newer record and saves again.
            self.assertEqual(stale._load_record().last_turn, 4)
            stale._save_record()


def _reloaded(root: RunContext) -> RunContext:
    """The session context a restarted host rebuilds from its saved store."""
    path = os.path.join(tempfile.mkdtemp(prefix="af_run_state_"), "store.json")
    root.store.save(path)
    return RunContext.root(store=RunStateStore.load(path))


class RunContextRecordStoreTest(unittest.TestCase):
    def test_a_record_survives_a_run_state_store_save_and_load(self) -> None:
        root = RunContext.root()
        record = _started(2)
        record.add_notice("recap")
        record.turn_boundaries = {"1": "uuid-1"}
        RunContextRecordStore(root).save(record)
        store = RunContextRecordStore(_reloaded(root))
        self.assertEqual(store.load("k").to_dict(), record.to_dict())
        stale = store.load("k")
        store.save(store.load("k"))
        with self.assertRaises(StaleRecordError):
            store.save(stale)

    def test_records_live_on_the_session_node_not_a_turn_node(self) -> None:
        root = RunContext.root()
        RunContextRecordStore(root.child("turn_1")).save(_started())
        self.assertIsNone(root.store.peek("/turn_1"))
        self.assertIn("k", root.node().checkpoints[RunContextRecordStore.KEY])
        later_turn = _reloaded(root).child("turn_2")
        self.assertEqual(RunContextRecordStore(later_turn).load("k").generation, 0)


class RunContextRecordStoreInferencerTest(TestCase):
    async def test_a_restarted_host_resumes_the_vendor_session(self) -> None:
        shared, root = tempfile.mkdtemp(prefix="af_native_test_"), RunContext.root()
        first, factory, _, _ = make_native(
            [[text("one")]],
            record_store=RunContextRecordStore(root),
            session_dir=shared,
        )
        async with first:
            await first.run_agentic_loop("first", run_context=root.child("turn_1"))
        session_id = factory.last.open_request.session_id
        restarted = _reloaded(root)
        second, factory2, _, _ = make_native(
            [[text("two")]],
            record_store=RunContextRecordStore(restarted),
            session_dir=shared,
        )
        async with second:
            await second.run_agentic_loop(
                "second", run_context=restarted.child("turn_2")
            )
        self.assertTrue(factory2.last.open_request.resume)
        self.assertEqual(factory2.last.open_request.session_id, session_id)


def _record(generation: int) -> NativeSessionRecord:
    return NativeSessionRecord(
        conversation_key="k", generation=generation, vendor_session_id="s"
    )


class RecordStoreFenceTest(unittest.TestCase):
    def _stores(self):
        data: dict = {}
        return [
            InMemoryRecordStore(),
            CallbackRecordStore(data.get, data.__setitem__),
            RunContextRecordStore(RunContext.root()),
        ]

    def test_lower_generation_cannot_overwrite_a_newer_record(self) -> None:
        for store in self._stores():
            with self.subTest(store=type(store).__name__):
                store.save(_record(2))
                with self.assertRaises(StaleRecordError):
                    store.save(_record(1))
                # Same generation: a copy older than the stored save loses
                # (compare-and-swap on saved_at); the loaded record saves.
                with self.assertRaises(StaleRecordError):
                    store.save(_record(2))
                store.save(store.load("k"))
                self.assertEqual(store.load("k").generation, 2)

    def test_end_vendor_session_rotates_a_started_record_with_a_recap(self) -> None:
        store = InMemoryRecordStore()
        record = _record(2)
        record.status = "active"
        record.turn_boundaries = {"1": "m1"}
        store.save(record)
        self.assertTrue(end_vendor_session(store, "k"))
        ended = store.load("k")
        self.assertEqual(ended.generation, 3)
        self.assertFalse(ended.started)
        self.assertEqual(ended.turn_boundaries, {})
        self.assertEqual([n["type"] for n in ended.pending_notices()], ["recap"])

    def test_end_vendor_session_ignores_missing_and_unstarted_records(self) -> None:
        store = InMemoryRecordStore()
        self.assertFalse(end_vendor_session(store, "k"))
        store.save(NativeSessionRecord(conversation_key="k"))
        self.assertFalse(end_vendor_session(store, "k"))
        self.assertEqual(store.load("k").generation, 0)

    def test_run_context_store_round_trips_on_a_session_node(self) -> None:
        ctx = RunContext.root()
        store = RunContextRecordStore(ctx)
        record = _record(3)
        record.add_notice("recap")
        store.save(record)
        self.assertIn("k", ctx.node().checkpoints[RunContextRecordStore.KEY])
        self.assertEqual(store.load("k").to_dict(), record.to_dict())


class PrivacyTest(TestCase):
    async def test_record_holds_notice_refs_while_l2_carries_rendered_text(
        self,
    ) -> None:
        native, factory, _, _ = make_native([[text("noted")], [text("ok")]])
        async with native:
            native.set_messages(
                [{"role": "user", "content": "remember secret-canary-123"}]
            )
            await native.run_agentic_loop("remember secret-canary-123")
            native._queue_notice("instructions_updated")
            native._queue_notice("recap")
            native._queue_notice(
                "tool_completion", body="RESULT-CANARY-9", tool="bg_job"
            )
            for dumped in (
                json.dumps(native._load_record().to_dict()),
                json.dumps(native.export_state(turn_number=1)["native"]),
            ):
                self.assertNotIn("secret-canary-123", dumped)
                self.assertNotIn("RESULT-CANARY-9", dumped)
                self.assertNotIn("Standard Operating Procedures", dumped)
            for entry in native._load_record().outbox:
                self.assertLessEqual(set(entry), {"id", "type", "tool"})
            await native.run_agentic_loop("continue")
            l2 = factory.last.l2_seen[-1]
            self.assertIn('type="instructions_updated"', l2)
            self.assertIn("Standard Operating Procedures", l2)
            self.assertIn('type="recap"', l2)
            self.assertIn("secret-canary-123", l2)
            self.assertIn("RESULT-CANARY-9", l2)

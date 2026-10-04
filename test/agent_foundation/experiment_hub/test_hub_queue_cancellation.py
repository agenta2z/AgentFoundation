"""The Experiment Hub's job queue stops when its host stops it. Cancelling a
running job — what a host does to the tasks ``track_task`` handed it when the
session is deleted or the server shuts down — ends the job, kills every process
it spawned and starts none of the jobs queued behind it; ``aclose()`` and a
cancelled ``join()`` do the same. A job that ends on its own still starts the
next one.

Submission runs are real ``SubmissionRunner`` runs (``local_mock`` launcher) of
a script that never finishes and keeps a grandchild writing into the session.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.experiment_hub.hub_controller import HubController
from agent_foundation.server.workflow_context import WorkflowContext
from later.unittest import TestCase

_SCRIPT = """\
import os, subprocess, sys, time
writes, pids = sys.argv[1], sys.argv[2]
loop = 'while true; do echo grandchild >> "$0"; sleep 0.05; done'
child = subprocess.Popen(["bash", "-c", loop, writes])
with open(pids, "a") as f:
    f.write(f"{os.getpid()} {child.pid}\\n")
while True:
    with open(writes, "a") as f:
        f.write("child\\n")
    print("tick", flush=True)
    time.sleep(0.05)
"""


def _running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


class HubQueueCancellationTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.session_dir = self.root / "session"
        self.writes = self.session_dir / "writes.log"
        self.pids_file = self.root / "pids.txt"
        self.addCleanup(self._kill_leftovers)
        self.wc = WorkflowContext()
        self.events: list[dict[str, Any]] = []
        self.tracked: list[tuple[str, asyncio.Task[None]]] = []
        self.started: list[str] = []
        self.release = asyncio.Event()

    def _kill_leftovers(self) -> None:
        for pid in self._pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _pids(self) -> list[int]:
        if not self.pids_file.exists():
            return []
        return [int(p) for p in self.pids_file.read_text().split()]

    async def _emit(self, _session_id: str, payload: dict[str, Any]) -> None:
        self.events.append(payload)

    async def _exec_task(
        self, _args: dict[str, Any], task_id: str
    ) -> ToolExecutionResult:
        self.started.append(task_id)
        await self.release.wait()
        return ToolExecutionResult(result=f"done {task_id}")

    def _hub(self) -> HubController:
        return HubController(
            session_id="s1",
            session_dir=self.session_dir,
            session_tasks_dir=self.session_dir / "tasks",
            workflow_context=self.wc,
            emit_event=self._emit,
            exec_task=self._exec_task,
            track_task=lambda key, task: self.tracked.append((key, task)),
        )

    async def _queue_task(self, hub: HubController, title: str) -> str:
        return await hub.add_to_experiment_hub(
            multi_task_id="hub-1", request=f"Implement {title}", title=title
        )

    async def _start_submission_run(self, hub: HubController) -> str:
        setup = self.root / "setup"
        setup.mkdir()
        script = setup / "submit_v1.py"
        script.write_text(_SCRIPT)
        launch = setup / "launch.json"
        launch.write_text(
            json.dumps(
                {
                    "launcher": "local_mock",
                    "script_args": [str(self.writes), str(self.pids_file)],
                }
            )
        )
        task_id = await hub.run_submission_script(
            multi_task_id="hub-1",
            submission_id="sub-1",
            setup_id="setup-1",
            script_path=str(script),
            launch_path=str(launch),
            enable_flags=[],
            experiment_name="exp",
            submission_label="Sub 1",
        )
        deadline = time.monotonic() + 30
        while len(self._pids()) < 2 or not self.writes.exists():
            if time.monotonic() > deadline:
                raise AssertionError(f"the run never started: {self.events}")
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)
        return task_id

    async def _until_started(self, count: int) -> None:
        deadline = time.monotonic() + 10
        while len(self.started) < count:
            if time.monotonic() > deadline:
                raise AssertionError(f"started: {self.started}")
            await asyncio.sleep(0.01)

    def _status(self, task_id: str) -> str:
        entry = self.wc.get_entry(task_id)
        assert entry is not None
        return entry["status"]

    async def _assert_writes_stopped(self) -> None:
        size = self.writes.stat().st_size
        await asyncio.sleep(0.5)
        self.assertEqual(self.writes.stat().st_size, size, "something still writes")

    async def test_cancelling_its_tracked_tasks_stops_a_run_and_its_queue(
        self,
    ) -> None:
        hub = self._hub()
        run_id = await self._start_submission_run(hub)
        queued_id = await self._queue_task(hub, "H1")

        # A host stopping the session: cancel and await what it was handed.
        live = [task for _key, task in self.tracked if not task.done()]
        self.assertTrue(live)
        for task in live:
            task.cancel()
        await asyncio.gather(*live, return_exceptions=True)

        self.assertEqual(len(self._pids()), 2)
        self.assertEqual([p for p in self._pids() if _running(p)], [])
        await self._assert_writes_stopped()
        self.assertEqual(self._status(run_id), "error")
        self.assertEqual(self._status(queued_id), "queued")
        self.assertEqual(self.started, [])
        self.assertTrue(all(task.done() for _key, task in self.tracked))

    async def test_aclose_cancels_the_running_job_and_starts_no_queued_one(
        self,
    ) -> None:
        hub = self._hub()
        first = await self._queue_task(hub, "H1")
        second = await self._queue_task(hub, "H2")
        await self._until_started(1)

        await hub.aclose()

        self.assertEqual(self.started, [first])
        self.assertEqual(self._status(first), "error")
        self.assertEqual(self._status(second), "queued")
        later = await self._queue_task(hub, "H3")
        await asyncio.sleep(0.2)
        self.assertEqual(self.started, [first])
        self.assertEqual(self._status(later), "queued")
        self.assertTrue(all(task.done() for _key, task in self.tracked))

    async def test_cancelling_join_stops_the_run_and_its_processes(self) -> None:
        hub = self._hub()
        run_id = await self._start_submission_run(hub)
        queued_id = await self._queue_task(hub, "H1")
        waiter = asyncio.create_task(hub.join())
        await asyncio.sleep(0.1)

        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter

        self.assertEqual([p for p in self._pids() if _running(p)], [])
        await self._assert_writes_stopped()
        self.assertEqual(self._status(run_id), "error")
        self.assertEqual(self._status(queued_id), "queued")
        self.assertEqual(self.started, [])

    async def test_a_job_ending_on_its_own_starts_the_next_one(self) -> None:
        hub = self._hub()
        first = await self._queue_task(hub, "H1")
        second = await self._queue_task(hub, "H2")
        await self._until_started(1)
        self.assertEqual(self._status(second), "queued")

        self.release.set()
        await asyncio.wait_for(hub.join(), 10)

        self.assertEqual(self.started, [first, second])
        self.assertEqual(self._status(first), "completed")
        self.assertEqual(self._status(second), "completed")
        # Every runner the queue started was handed to the host.
        self.assertGreaterEqual(len(self.tracked), 3)
        self.assertTrue(all(task.done() for _key, task in self.tracked))

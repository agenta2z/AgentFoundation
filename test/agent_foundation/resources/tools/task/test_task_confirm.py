"""``task --confirm``: the plan review goes to the caller's interactive transport.

``--confirm`` turns on PTI's plan-review checkpoint (Approve / Modify / Reject)
and hands PTI the interactive transport of the session context — only one whose
questions get answered (``router_interactive_safe``, i.e. a registered receive
queue). Without one, PTI approves the plan unreviewed instead of waiting on a
queue nobody reads.

The production ``default.yaml`` topology is built for real; only PTI's run is
replaced, recording what it was handed, so no CLI starts.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.resources.tools.task import executor
from later.unittest import TestCase


class _Interactive:
    """Stands in for the session's interactive transport."""


class TaskConfirmTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        workspace = root / "tasks" / "task_confirm"
        workspace.mkdir(parents=True)
        self.context = {"working_dir": str(workspace), "session_root": str(root)}
        env = mock.patch.dict(os.environ, {"CLAUDE_CODE_COMMAND": "true"})
        env.start()
        self.addCleanup(env.stop)

    async def _confirm(self, **context: Any) -> tuple[Any, list[tuple[Any, bool]]]:
        runs: list[tuple[Any, bool]] = []

        async def run(pti: PlanThenImplementInferencer, request: str, **_kw: Any):
            runs.append((pti.interactive, pti.enable_checkpoint_plan_review))
            return "plan approved, implemented"

        with mock.patch.object(PlanThenImplementInferencer, "ainfer", run):
            result = await executor.execute(
                {"request": "Refactor the loader.", "confirm": True},
                {**self.context, **context},
            )
        return result, runs

    async def test_the_plan_review_goes_to_the_callers_interactive(self) -> None:
        interactive = _Interactive()

        result, runs = await self._confirm(
            interactive=interactive, router_interactive_safe=True
        )

        self.assertEqual(runs, [(interactive, True)])
        self.assertEqual(result.result, "plan approved, implemented")
        self.assertTrue(result.context_updates["success"])

    async def test_without_an_answerable_interactive_the_plan_is_approved(
        self,
    ) -> None:
        with self.assertLogs(executor._logger, level="WARNING") as logs:
            result, runs = await self._confirm(interactive=_Interactive())

        self.assertEqual(runs, [(None, True)])
        self.assertTrue(result.context_updates["success"])
        self.assertIn("approved without one", "\n".join(logs.output))

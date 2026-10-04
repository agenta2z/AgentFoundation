"""The ``task`` tool's ``model`` argument and phase-skipping flags.

- ``model`` replaces the configured model on every leaf: the presets set it
  only through ``_model_name`` cascades (``default.yaml`` declares one at its
  root and the imported planner its own), which the old walk never touched.
- ``--no-planning`` / ``--no-implementation`` (what ``understand_codebase``'s
  ``--docs-only`` / ``--investigation-only`` map to) skip their phase, as
  ``--execute`` / ``--plan`` do.

The end-to-end cases run the production topology with a fake ``claude`` that
records its command line and then streams until the run is cancelled.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import attr
import rich_python_utils.config_utils as config_utils
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.resources.tools.registry import derived_tool_execute
from agent_foundation.resources.tools.task import executor
from agent_foundation.resources.tools.understand_codebase import cli as uc_cli
from later.unittest import TestCase

_FAKE_CLAUDE = """#!/bin/bash
[ "$1" = "--version" ] && { echo "fake 0.0"; exit 0; }
echo "$*" >> "$FAKE_CLI_STATE/calls.txt"
cat > /dev/null &
while true; do echo '{"type":"stream_event","event":{"type":"ping"}}'; sleep 0.05; done
"""
_UNDERSTAND_CODEBASE = (
    Path(executor.__file__).resolve().parent.parent
    / "understand_codebase"
    / "tool.json"
)


class _Instantiated(Exception):
    """Stops a run right after its topology is built."""


def _cli_leaves(root: Any) -> list[ClaudeCodeCliInferencer]:
    seen: set[int] = set()
    leaves: list[ClaudeCodeCliInferencer] = []
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, ClaudeCodeCliInferencer):
            leaves.append(obj)
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
        elif attr.has(type(obj)):
            stack.extend(getattr(obj, f.name, None) for f in attr.fields(type(obj)))
    return leaves


class TaskModelAndPhaseFlagsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.workspace = self.root / "tasks" / "task_flags"
        self.workspace.mkdir(parents=True)
        self.target = self.root / "target"
        self.target.mkdir()
        (self.target / "module.py").write_text("VALUE = 1\n")
        self.state = self.root / "cli_state"
        self.state.mkdir()
        fake = self.state / "claude"
        fake.write_text(_FAKE_CLAUDE)
        fake.chmod(0o755)
        env = mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_COMMAND": str(fake), "FAKE_CLI_STATE": str(self.state)},
        )
        env.start()
        self.addCleanup(env.stop)

    def _context(self) -> dict[str, str]:
        return {"working_dir": str(self.workspace), "session_root": str(self.root)}

    async def _understand_codebase(self, **arguments: Any) -> str:
        """Run ``understand_codebase`` until its first CLI call; return that
        call's command line (the run is then cancelled)."""
        derived_from = json.loads(_UNDERSTAND_CODEBASE.read_text())["derived_from"]
        run = asyncio.create_task(
            derived_tool_execute(
                {"target": str(self.target), **arguments},
                self._context(),
                derived_from=derived_from,
                tool_name="understand_codebase",
            )
        )
        calls = self.state / "calls.txt"
        deadline = time.monotonic() + 60
        try:
            while not calls.exists():
                self.assertFalse(run.done(), f"the run ended first: {run}")
                self.assertLess(time.monotonic(), deadline, "the CLI never started")
                await asyncio.sleep(0.05)
        finally:
            run.cancel()
            await asyncio.wait({run}, timeout=10)
        return calls.read_text().splitlines()[0]

    def _cli_session_logs(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.workspace))
            for p in self.workspace.rglob("ClaudeCodeCliInferencer-*.jsonl")
        )

    async def test_model_argument_sets_every_leaf_model(self) -> None:
        built: list[Any] = []
        instantiate = config_utils.instantiate

        def capture(cfg: Any, *args: Any, **kwargs: Any) -> Any:
            built.append(instantiate(cfg, *args, **kwargs))
            raise _Instantiated()

        with mock.patch.object(config_utils, "instantiate", capture):
            result = await executor.execute(
                {"request": "Document the module.", "model": "haiku"}, self._context()
            )

        self.assertIn("Instantiation failed", result.result)
        leaves = _cli_leaves(built[0])
        tiered = [leaf for leaf in leaves if leaf.model_tier is not None]
        untiered = [leaf for leaf in leaves if leaf.model_tier is None]
        self.assertGreaterEqual(len(untiered), 8)
        self.assertEqual({leaf.model_name for leaf in untiered}, {"haiku"})
        # Output guardrails keep their configured judge tier.
        self.assertEqual(
            {(leaf.model_tier, leaf.model_name) for leaf in tiered},
            {("default", "sonnet")},
        )

    def test_model_walk_replaces_cascades_and_leaves(self) -> None:
        cfg = {
            "_model_name": "opus[1m]",
            "planner": {"_model_name": "opus[1m]", "leaf": {"_target_": "X"}},
            "executor": {"model_name": "opus", "model_tier": "default"},
        }

        self.assertEqual(executor._walk_replace_model(cfg, "haiku"), 3)
        self.assertEqual(
            cfg,
            {
                "_model_name": "haiku",
                "planner": {"_model_name": "haiku", "leaf": {"_target_": "X"}},
                "executor": {"model_name": "haiku", "model_tier": "default"},
            },
        )

    async def test_understand_codebase_model_reaches_the_cli(self) -> None:
        command = await self._understand_codebase(model="haiku")

        self.assertIn("--model haiku ", command)

    async def test_understand_codebase_docs_only_skips_the_planner(self) -> None:
        await self._understand_codebase(docs_only=True)

        logs = self._cli_session_logs()
        self.assertTrue(logs)
        self.assertEqual(
            [
                log
                for log in logs
                if not log.startswith("children/executor_inferencer/")
            ],
            [],
        )

    async def test_understand_codebase_plans_first_by_default(self) -> None:
        await self._understand_codebase()

        logs = self._cli_session_logs()
        self.assertTrue(logs)
        self.assertEqual(
            [log for log in logs if not log.startswith("children/planner_inferencer/")],
            [],
        )

    async def test_phase_flags_select_the_execute_and_plan_runs(self) -> None:
        cases = [
            ({"no_planning": True}, "execute"),
            ({"no_planning": True, "full": True}, "execute"),
            ({"no_planning": True, "execute": True}, "execute"),
            ({"no_implementation": True}, "plan"),
            ({"no_implementation": True, "plan": True}, "plan"),
        ]
        for flags, mode in cases:
            with mock.patch.object(executor, "_run_topology") as run_topology:
                await executor.execute({"request": "r", **flags}, self._context())
            self.assertEqual(run_topology.call_args.kwargs["mode"], mode, flags)

    async def test_conflicting_phase_flags_are_refused(self) -> None:
        cases = [
            ({"no_planning": True, "no_implementation": True}, "nothing to run"),
            (
                {"no_planning": True, "plan": True},
                "--no-planning conflicts with --plan",
            ),
            (
                {"no_implementation": True, "confirm": True},
                "--no-implementation conflicts with --confirm",
            ),
        ]
        for flags, message in cases:
            with mock.patch.object(executor, "_run_topology") as run_topology:
                result = await executor.execute({"request": "r", **flags}, {})
            run_topology.assert_not_called()
            self.assertIn(message, result.result)
            self.assertEqual(result.context_updates, {"success": False})


class UnderstandCodebaseCliTest(unittest.TestCase):
    def test_cli_runs_the_derived_tool_with_its_flags(self) -> None:
        run = mock.AsyncMock(return_value=SimpleNamespace(result="done"))
        with (
            mock.patch(
                "agent_foundation.resources.tools.registry.derived_tool_execute", run
            ),
            mock.patch("builtins.print"),
        ):
            code = uc_cli.main(["/src/pkg", "--docs-only", "--model", "haiku"])

        self.assertEqual(code, 0)
        run.assert_awaited_once_with(
            {"target": "/src/pkg", "docs_only": True, "model": "haiku"},
            {},
            derived_from=json.loads(_UNDERSTAND_CODEBASE.read_text())["derived_from"],
            tool_name="understand_codebase",
        )

"""The SOP CLI drives its orchestrator through the host protocol only
(``host_protocol.ConversationalHost`` plus ``SupportsInbox`` and
``SupportsPromptManifest``): a host with nothing but those members runs a
whole session, interactive or yolo, with the per-turn artifacts written."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.resources.tools.sop import cli
from later.unittest import TestCase
from rich_python_utils.common_objects.workflow.common.phase_status import PhaseStatus

_PROMPT = {
    "rendered_prompt": "the rendered prompt",
    "template_source": "conversation/main/initial",
    "template_feed": {"sop_active": True},
    "template_config": {"rendering": {"structural_xml_tags": ["SOPStatus"]}},
}


class _ProtocolOnlyHost:
    """The host-protocol members the SOP CLI may use, and no private state of
    a real orchestrator. ``run`` drains the inbox the way an orchestrator
    does: one turn per message, then (interactive) until shutdown."""

    supports_inbox = True
    supports_prompt_manifest = True

    def __init__(self) -> None:
        self.extra_sop_dirs: list = []
        self.cache_folder: Optional[str] = None
        self.sop_state: Any = None
        self.prior_context: dict[str, Any] = {}
        self.inbox: Optional[asyncio.Queue] = None
        self.callbacks: dict[str, Any] = {}
        self.cache_folders: list[Optional[str]] = []
        self.messages: list[dict[str, str]] = []
        self.shutdown = False
        # Yolo: the SOP completes with the first turn.
        self.complete_sop_after_turn = False

    def update_prior_context(self, **updates: Any) -> None:
        self.sop_state = updates.pop("sop_state", self.sop_state)
        self.prior_context.update(updates)

    def enable_inbox(
        self,
        interactive: Any = None,
        *,
        auto_shutdown_on_sop_complete: bool = False,
        maxsize: int = 0,
        on_new_turn: Any = None,
        on_prompt_rendered: Any = None,
        on_turn_complete: Any = None,
    ) -> None:
        self.inbox = asyncio.Queue()
        self.callbacks = {
            "new_turn": on_new_turn,
            "prompt_rendered": on_prompt_rendered,
            "turn_complete": on_turn_complete,
        }

    def inbox_put(self, item: object) -> None:
        self.inbox.put_nowait(item)

    def inbox_put_user(self, content: str, source: str = "user") -> None:
        self.inbox_put(content)

    def request_shutdown(self) -> None:
        self.shutdown = True
        self.inbox.put_nowait(None)

    @property
    def shutdown_requested(self) -> bool:
        return self.shutdown

    def last_prompt_data(self) -> dict[str, Any]:
        return dict(_PROMPT)

    def get_messages(self) -> list[dict[str, str]]:
        return list(self.messages)

    async def run(self, *, run_context: Any = None) -> Optional[AgenticResult]:
        result = None
        turn = 0
        while not self.shutdown:
            content = await self.inbox.get()
            if content is None:
                break
            turn += 1
            await self.callbacks["new_turn"](turn, content)
            self.cache_folders.append(self.cache_folder)
            self.messages += [
                {"role": "user", "content": content},
                {"role": "assistant", "content": f"answer {turn}"},
            ]
            await self.callbacks["prompt_rendered"](self, f"answer {turn}")
            await self.callbacks["turn_complete"](turn)
            result = AgenticResult(
                text=f"answer {turn}", completed_actions=[], iterations_used=1
            )
            if self.complete_sop_after_turn:
                self.sop_state.phase_status = PhaseStatus.COMPLETED
                self.request_shutdown()
        return result


class SopCliHostProtocolTest(TestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = tempfile.mkdtemp(prefix="sop_cli_test_")
        previous = os.getcwd()
        os.chdir(self.cwd)
        self.addCleanup(os.chdir, previous)
        self.host = _ProtocolOnlyHost()
        state = SimpleNamespace(phase_status=PhaseStatus.RUNNING, current_phase="0")
        entered = ToolExecutionResult(
            result="Entered SOP", context_updates={"sop_state": state}
        )
        for target, replacement in (
            (
                "agent_foundation.resources.tools.sop.cli._build_ci_from_config",
                mock.Mock(return_value=self.host),
            ),
            (
                "agent_foundation.resources.tools.sop.executor.execute",
                mock.AsyncMock(return_value=entered),
            ),
        ):
            patcher = mock.patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _turn_dir(self) -> Path:
        (session,) = (Path(self.cwd) / "_runtime" / "sop").iterdir()
        return session / "turns" / "turn_001"

    def _assert_turn_artifacts(self) -> None:
        turn_dir = self._turn_dir()
        round_dir = turn_dir / "round_001"
        self.assertEqual(
            (round_dir / "rendered_prompt.txt").read_text(), _PROMPT["rendered_prompt"]
        )
        self.assertEqual(
            (round_dir / "template_source.txt").read_text(), _PROMPT["template_source"]
        )
        self.assertEqual(
            json.loads((round_dir / "template_feed.json").read_text()),
            _PROMPT["template_feed"],
        )
        self.assertEqual(
            json.loads((round_dir / "template_config.json").read_text()),
            _PROMPT["template_config"],
        )
        self.assertEqual((round_dir / "response.md").read_text(), "answer 1")
        self.assertEqual(
            json.loads((turn_dir / "messages.json").read_text()),
            self.host.get_messages()[:2],
        )
        self.assertEqual(self.host.cache_folders[0], str(turn_dir))

    async def test_a_yolo_session_runs_on_the_host_protocol(self) -> None:
        self.host.complete_sop_after_turn = True
        code = await cli.run_sop(
            "mini_research", "write a brief", yolo=True, extra_sop_dirs=[]
        )
        self.assertEqual(code, 0)
        self._assert_turn_artifacts()
        self.assertTrue(
            (self._turn_dir().parents[1] / "run_state" / "store.json").exists()
        )

    async def test_an_interactive_session_ends_on_the_public_shutdown_flag(
        self,
    ) -> None:
        with mock.patch("builtins.input", side_effect=EOFError):
            code = await cli.run_sop("mini_research", "write a brief", yolo=False)
        self.assertEqual(code, 0)
        self.assertTrue(self.host.shutdown_requested)
        self._assert_turn_artifacts()

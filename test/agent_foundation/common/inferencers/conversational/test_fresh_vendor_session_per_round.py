"""Opt-in: every agentic-loop round reaches a fresh vendor conversation (plan
§15, S17).

Each round's rendered prompt carries the whole conversation, so a vendor session
continued from the previous round holds that history twice. The base here is a
session-keeping vendor: it continues the agent branch's ``active_session_id``,
as a resuming CLI does on the non-streaming path and a connected SDK client does
within a turn. ``fresh_vendor_session_per_round`` on resets the base's
conversation before every round; off — the default, in the attribute and in
``default.yaml``, pending the user's decision on that behavior change — keeps
the continuation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.run_context import RunContext
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.resources.tools import _ci_host
from agent_foundation.resources.tools.models import ToolDefinition
from attr import attrib, attrs
from later.unittest import TestCase

_TOOL_ROUND = (
    "Looking it up.\n"
    "```json ToolsToInvoke\n"
    '{"type": "action", "name": "lookup_record", "arguments": {"key": "A-7"}}\n'
    "```"
)
_FIRST_ANSWER = "The record's status is green."
_SECOND_ANSWER = "The codeword is HERON-913."
_TURNS = (
    "Remember the codeword HERON-913, then look up record A-7.",
    "What was the codeword?",
)
_DEFAULT_CONFIG = (
    Path(_ci_host.__file__).resolve().parents[1] / "configs/conversational/default.yaml"
)


class _Vendor:
    """The vendor's side: its sessions and the prompts each one received."""

    def __init__(self) -> None:
        self.script = [_TOOL_ROUND, _FIRST_ANSWER, _SECOND_ANSWER]
        self.sessions: dict[str, list[str]] = {}

    def open(self) -> str:
        session = f"s{len(self.sessions) + 1}"
        self.sessions[session] = []
        return session

    def answer(self, session: str, prompt: str) -> str:
        self.sessions[session].append(prompt)
        return self.script.pop(0)


@attrs(slots=False)
class _SessionKeepingBase(StreamingInferencerBase):
    vendor: _Vendor = attrib(factory=_Vendor)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        raise NotImplementedError

    async def _ainfer_streaming(self, prompt, **kwargs):
        session = self.active_session_id or self.vendor.open()
        self.active_session_id = session
        yield self.vendor.answer(session, prompt)


class _Collector:
    async def stream_token_batches(
        self, tokens, session_id, send_stream_end=False, turn_number=0
    ) -> str:
        return "".join([chunk async for chunk, _meta in tokens])


class _ToolResult:
    def __init__(self, result: str) -> None:
        self.result = result
        self.context_updates = None


async def _lookup(name, arguments):
    return _ToolResult(f"record {arguments['key']}: status=green")


def _ci(base, **kwargs) -> ConversationalInferencer:
    return ConversationalInferencer(
        base_inferencer=base,
        tool_registry={"lookup_record": ToolDefinition(name="lookup_record")},
        tool_executor=_lookup,
        max_iterations=5,
        **kwargs,
    )


async def _two_turns(ci, *, stream: bool, per_turn_host: bool) -> list[int]:
    root = RunContext.root(workspace=None) if per_turn_host else None
    rounds = []
    for n, text in enumerate(_TURNS, start=1):
        ci.add_message("user", text)
        result = await ci.run_agentic_loop(
            text,
            run_context=root.child(f"turn_{n}") if root else None,
            interactive=_Collector() if stream else None,
            turn_number=n,
        )
        rounds.append(result.iterations_used)
    return rounds


class FreshVendorSessionPerRoundTest(TestCase):
    def _assert_one_session_per_round(self, base, rounds) -> None:
        self.assertEqual(rounds, [2, 1])
        sessions = base.vendor.sessions
        self.assertEqual(len(sessions), 3)
        self.assertTrue(all(len(prompts) == 1 for prompts in sessions.values()))
        (tool_round, continuation, second_turn) = (p[0] for p in sessions.values())
        self.assertIn(_TURNS[0], tool_round)
        self.assertIn("record A-7: status=green", continuation)
        self.assertIn(_TURNS[0], second_turn)
        self.assertIn(_TURNS[1], second_turn)

    async def test_on_non_streaming_rounds_of_a_bare_host(self) -> None:
        base = _SessionKeepingBase()
        ci = _ci(base, fresh_vendor_session_per_round=True)
        rounds = await _two_turns(ci, stream=False, per_turn_host=False)
        self._assert_one_session_per_round(base, rounds)

    async def test_on_non_streaming_rounds_of_a_per_turn_host(self) -> None:
        base = _SessionKeepingBase()
        ci = _ci(base, fresh_vendor_session_per_round=True)
        rounds = await _two_turns(ci, stream=False, per_turn_host=True)
        self._assert_one_session_per_round(base, rounds)

    async def test_on_streaming_rounds_of_a_per_turn_host(self) -> None:
        base = _SessionKeepingBase()
        ci = _ci(base, fresh_vendor_session_per_round=True)
        rounds = await _two_turns(ci, stream=True, per_turn_host=True)
        self._assert_one_session_per_round(base, rounds)

    async def test_by_default_a_bare_host_continues_one_session(self) -> None:
        base = _SessionKeepingBase()
        rounds = await _two_turns(_ci(base), stream=False, per_turn_host=False)
        self.assertEqual(rounds, [2, 1])
        self.assertEqual([len(p) for p in base.vendor.sessions.values()], [3])

    async def test_by_default_a_per_turn_host_continues_within_the_turn(
        self,
    ) -> None:
        base = _SessionKeepingBase()
        rounds = await _two_turns(_ci(base), stream=True, per_turn_host=True)
        self.assertEqual(rounds, [2, 1])
        self.assertEqual([len(p) for p in base.vendor.sessions.values()], [2, 1])

    async def test_a_base_outside_the_inferencer_hierarchy_is_called_as_is(
        self,
    ) -> None:
        vendor = _Vendor()
        session = vendor.open()

        class _DuckBase:
            async def ainfer(self, prompt, *, run_context=None):
                return vendor.answer(session, prompt)

        ci = _ci(_DuckBase(), fresh_vendor_session_per_round=True)
        rounds = await _two_turns(ci, stream=False, per_turn_host=False)
        self.assertEqual(rounds, [2, 1])
        self.assertEqual([len(p) for p in vendor.sessions.values()], [3])


class FreshVendorSessionConfigTest(TestCase):
    def test_the_default_config_leaves_it_off(self) -> None:
        ci = _ci_host.build_ci_from_config(
            _DEFAULT_CONFIG, base_inferencer=_SessionKeepingBase()
        )
        self.assertIs(ci.fresh_vendor_session_per_round, False)

    def test_a_config_can_turn_it_on(self) -> None:
        config = yaml.safe_load(_DEFAULT_CONFIG.read_text())
        config["fresh_vendor_session_per_round"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ci.yaml"
            path.write_text(yaml.safe_dump(config))
            ci = _ci_host.build_ci_from_config(
                path, base_inferencer=_SessionKeepingBase()
            )
        self.assertIs(ci.fresh_vendor_session_per_round, True)

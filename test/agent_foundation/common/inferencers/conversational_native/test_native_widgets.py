"""Deferred widgets of the native orchestrator (plan §5.4, §6.1 step 5.6):
compound widgets after the vendor turn, yolo, bundled actions, dashboard
handoff, recovery and cancellation.

Plan §12.1 row coverage (assertion: tests; Class.method or a whole Class):
- two widget calls in one message -> one compound widget after the turn ->
  next vendor turn with CI's prefix:
    CompoundWidgetTest
    NativeTurnTest.test_compound_widgets_from_one_message_are_presented_together
    NativeTurnTest.test_widget_is_deferred_then_answer_starts_next_vendor_turn
- a widget without text gets its own round + a durable marker:
    test_native_rounds.RoundGroupingTest.test_widget_only_message_gets_its_own_round
    NativeSopProgressionTest.test_widget_marker_is_persisted_with_export_state_blob
- validation_retry:
    NativeWidgetTest.test_an_invalid_widget_batch_is_retried_with_the_error
- yolo inline:
    NativeTurnTest.test_yolo_answers_widgets_inline
    NativeTurnTest.test_yolo_answers_a_path_with_its_prefilled_default
    NativeWidgetTest.test_yolo_then_run_gets_the_synthesized_answer
- yolo + dashboard handoff stops:
    TurnEndingTest
- then_run with __var__ + overrides:
    WidgetAnswerParityTest.test_bundled_action_gets_the_same_arguments_and_answer_text
- mailboxes cleared:
    NativeWidgetTest.test_mailboxes_are_cleared_after_every_answer
- interactive=None -> has_conversation_tool:
    NativeTurnTest.test_without_interactive_a_pending_widget_is_returned
- recovery via set_pending_widget_answer (same turn, no fork):
    NativeSopProgressionTest.test_recovered_* (4 tests)
    test_native_lifecycle.RewindTest.test_recovered_widget_answer_for_the_same_turn_does_not_fork
- cancel while waiting -> notice next turn:
    NativeWidgetTest.test_cancel_while_a_widget_waits_is_announced_next_turn
    NativeWidgetTest.test_a_rearmed_widget_withdraws_the_cancel_notice
- vendor-turn cap (plan §6.1 step 5; default 50, <= 0 -> safety ceiling;
  what the last capped turn leaves unsent reaches the next turn):
    VendorTurnCapTest
Also here: the non-yolo dashboard handoff (DashboardHandoffTest); an answer
whose bundled action starts in the background ends the turn as in CI (D6;
BackgroundActionAnswerTest).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from agent_foundation.common.data_models.proposal.model import (
    Proposal,
    ProposalGroup,
    ProposalIndex,
)
from agent_foundation.common.data_models.proposal.parser import write_proposal_index
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    WidgetMailboxes,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocol_text import (
    TOOL_RESULTS_PREFIX,
    WIDGET_RESPONSE_PREFIX,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    turn_loop,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    END_TURN_MARKER,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    TextDelta,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.resources.tools.models import ToolDefinition
from agent_foundation.resources.tools.registry import load_all_tools
from attr import attrs
from fakes import RecordingInteractive, text, tools, turn_end
from helpers import (
    make_native,
    RecordingExecutor,
    tool_registry as helpers_tool_registry,
    wait_for,
)
from later.unittest import TestCase


class NativeTurnTest(TestCase):
    async def test_widget_is_deferred_then_answer_starts_next_vendor_turn(self) -> None:
        scripts = [
            [
                text("Let me ask."),
                tools(
                    ("enter_sop", {"name": "mini_research"}),
                    (
                        "clarification",
                        {"prompt": "Which topic?", "output": ["research_topic"]},
                    ),
                ),
            ],
            [text("Got it, researching.")],
        ]
        native, factory, interactive, _ = make_native(
            scripts, answers=["quantum sensors"]
        )
        async with native:
            result = await native.run_agentic_loop("start the mini research sop")
            backend = factory.last
            self.assertIn("AF_END_TURN", backend.tool_results[1][1])
            self.assertEqual(len(interactive.widgets), 1)
            self.assertEqual(len(backend.turn_requests), 2)
            second = backend.turn_requests[1].text
            self.assertTrue(second.startswith(WIDGET_RESPONSE_PREFIX))
            self.assertIn("research_topic: quantum sensors", second)
            self.assertIn('origin="widget_answer"', backend.l2_seen[1])
            self.assertEqual(
                native.prior_context.get("research_topic"), "quantum sensors"
            )
            self.assertEqual(result.text, "Got it, researching.")

    async def test_compound_widgets_from_one_message_are_presented_together(
        self,
    ) -> None:
        scripts = [
            [
                tools(
                    ("clarification", {"prompt": "Topic?", "output": ["topic"]}),
                    (
                        "single_choice",
                        {
                            "prompt": "Depth?",
                            "choices": ["quick", "deep"],
                            "output": ["depth"],
                        },
                    ),
                )
            ],
            [text("Thanks.")],
        ]
        answer = {"values": {"topic": "lidar", "depth": "deep"}}
        native, factory, interactive, _ = make_native(scripts, answers=[answer])
        async with native:
            await native.run_agentic_loop("ask me two things")
            self.assertEqual(len(interactive.widgets), 1)
            self.assertTrue(
                interactive.widgets[0]["input_mode"].metadata.get("compound")
            )
            self.assertIn("topic: lidar", factory.last.turn_requests[1].text)
            self.assertIn("depth: deep", factory.last.turn_requests[1].text)

    async def test_yolo_answers_widgets_inline(self) -> None:
        scripts = [
            [
                tools(("clarification", {"prompt": "Topic?", "output": ["topic"]})),
                text("Continuing."),
            ]
        ]
        native, factory, interactive, _ = make_native(scripts, session_yolo=True)
        async with native:
            result = await native.run_agentic_loop("go autonomously")
            self.assertEqual(interactive.widgets, [])
            self.assertIn("Answered autonomously", factory.last.tool_results[0][1])
            self.assertEqual(result.text, "Continuing.")

    async def test_yolo_answers_a_path_with_its_prefilled_default(self) -> None:
        root = "/srv/session"
        target = f"{root}/toy_model"
        widget = {
            "prompt": "Target path?",
            "expected_input_type": "path",
            "prefix": root,
            "default": target,
            "output": ["workflow_target_path"],
        }
        scripts = [[tools(("clarification", widget)), text("Set.")]]
        native, factory, interactive, _ = make_native(scripts, session_yolo=True)
        async with native:
            native.prior_context["session_root_path"] = root
            await native.run_agentic_loop("optimize the model at toy_model")
            self.assertEqual(interactive.widgets, [])
            self.assertEqual(native.prior_context["workflow_target_path"], target)
            self.assertIn(
                f"workflow_target_path: {target}", factory.last.tool_results[0][1]
            )

    async def test_without_interactive_a_pending_widget_is_returned(self) -> None:
        scripts = [
            [tools(("clarification", {"prompt": "Topic?", "output": ["topic"]}))]
        ]
        native, _, _, _ = make_native(scripts)
        async with native:
            native.interactive = None
            result = await native.run_agentic_loop("go", interactive=None)
            self.assertTrue(result.has_conversation_tool)
            self.assertEqual(result.conversation_tool.prompt, "Topic?")


class _WaitingInteractive(RecordingInteractive):
    """Shows widgets but never answers (the user walks away)."""

    def __init__(self) -> None:
        super().__init__()
        self.shown = asyncio.Event()

    async def aget_input(self):
        self.shown.set()
        await asyncio.Event().wait()


class NativeWidgetTest(TestCase):
    """The widget paths of plan §12.1 (test_native_widgets row)."""

    async def test_an_invalid_widget_batch_is_retried_with_the_error(self) -> None:
        duplicate = {"prompt": "Topic?", "output": ["topic"], "parallel_group": 1}
        scripts = [
            [tools(("clarification", duplicate), ("clarification", duplicate))],
            [text("Asking one at a time.")],
        ]
        native, factory, interactive, _ = make_native(scripts)
        async with native:
            result = await native.run_agentic_loop("ask me")
            backend = factory.last
            self.assertEqual(interactive.widgets, [])
            retry = backend.turn_requests[1].text
            self.assertTrue(retry.startswith(TOOL_RESULTS_PREFIX))
            self.assertIn("[parallel_group validation]", retry)
            self.assertIn("duplicate primary output key 'topic'", retry)
            self.assertIn('origin="validation_retry"', backend.l2_seen[1])
            self.assertEqual(native.get_messages()[-2]["content"], retry)
            self.assertEqual(result.text, "Asking one at a time.")

    async def test_a_multiple_choice_answer_reaches_the_next_turn(self) -> None:
        picks = {
            "prompt": "Which?",
            "choices": [{"label": c.upper(), "value": c} for c in "abc"],
            "output": ["picks"],
        }
        native, factory, _, _ = make_native(
            [[tools(("multiple_choice", picks))], [text("Got it.")]],
            answers=["a,c"],
        )
        async with native:
            result = await native.run_agentic_loop("ask me")
            self.assertIn("picks: a,c", factory.last.turn_requests[1].text)
            self.assertEqual(native.prior_context["picks"], "a,c")
            self.assertEqual(native.prior_context["multiple_choice__picks"], "a,c")
            self.assertEqual(result.text, "Got it.")

    async def test_a_multiple_choice_widget_answer_is_decoded(self) -> None:
        picks = {
            "prompt": "Which?",
            "choices": [{"label": c.upper(), "value": c} for c in "abc"],
            "output": ["picks"],
        }
        native, factory, _, _ = make_native(
            [[tools(("multiple_choice", picks))], [text("Got it.")]],
            answers=[{"selections": [{"choice_index": 0}, {"choice_index": 2}]}],
        )
        async with native:
            result = await native.run_agentic_loop("ask me")
            self.assertIn("picks: a,c", factory.last.turn_requests[1].text)
            self.assertEqual(native.prior_context["picks"], "a,c")
            self.assertEqual(result.text, "Got it.")

    async def test_yolo_then_run_gets_the_synthesized_answer(self) -> None:
        widget = {
            "prompt": "Topic?",
            "output": ["topic"],
            "then_run": {"name": "write_brief", "arguments": {"topic": "__topic__"}},
        }
        native, factory, _, executor = make_native(
            [[tools(("clarification", widget)), text("Done.")]], session_yolo=True
        )
        async with native:
            result = await native.run_agentic_loop("go")
            self.assertEqual(
                executor.calls,
                [("write_brief", {"topic": "Follow your best judgment."})],
            )
            tool_text = factory.last.tool_results[0][1]
            self.assertIn("topic: Follow your best judgment.", tool_text)
            self.assertIn("write_brief done", tool_text)
            self.assertEqual(
                [a.tool for a in result.completed_actions], ["write_brief"]
            )

    async def test_mailboxes_are_cleared_after_every_answer(self) -> None:
        confirm = {"prompt": "Proceed?", "output": ["ok"]}
        scripts = [
            [tools(("confirmation", confirm))],
            [tools(("confirmation", confirm))],
            [text("Both confirmed.")],
        ]
        answer = {
            "choice": "yes",
            "param_overrides": {"depth": "deep"},
            "variables": {"style": "short"},
        }
        native, factory, interactive, executor = make_native(
            scripts, answers=[answer, dict(answer)]
        )
        async with native:
            result = await native.run_agentic_loop("go")
            # Without bundled actions the second answer's effects must not
            # collide with leftovers of the first (HandlerResultMergeConflict).
            self.assertEqual(len(interactive.widgets), 2)
            self.assertEqual(len(factory.last.turn_requests), 3)
            self.assertEqual(native.mailboxes, WidgetMailboxes())
            # Turn variables belong to the answer's bundled actions only.
            self.assertNotIn("style", native.prior_context)
            self.assertEqual(executor.calls, [])
            self.assertEqual(result.text, "Both confirmed.")

    async def _cancel_while_waiting(self, scripts: list) -> tuple:
        native, factory, _, _ = make_native(scripts)
        waiting = _WaitingInteractive()
        await native.__aenter__()
        task = asyncio.ensure_future(
            native.run_agentic_loop("ask me", interactive=waiting, turn_number=1)
        )
        await asyncio.wait_for(waiting.shown.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        return native, factory, waiting

    async def test_cancel_while_a_widget_waits_is_announced_next_turn(self) -> None:
        native, factory, _ = await self._cancel_while_waiting(
            [
                [tools(("clarification", {"prompt": "Topic?", "output": ["t"]}))],
                [text("Fine, lidar it is.")],
            ]
        )
        try:
            await native.run_agentic_loop("never mind, use lidar", turn_number=2)
            l2 = factory.last.l2_seen[1]
            self.assertIn('type="widget_cancelled"', l2)
            self.assertIn("dismissed the question", l2)
            self.assertEqual(native._load_record().pending_notices(), [])
        finally:
            await native.aclose()

    async def test_a_rearmed_widget_withdraws_the_cancel_notice(self) -> None:
        native, factory, waiting = await self._cancel_while_waiting(
            [
                [tools(("clarification", {"prompt": "Topic?", "output": ["t"]}))],
                [text("Continuing with lidar.")],
            ]
        )
        try:
            native.set_pending_widget_answer(
                {
                    "tools": waiting.persisted[0]["tools"],
                    "action_tools": [],
                    "raw_value": "lidar",
                }
            )
            result = await native.run_agentic_loop("__continue__", turn_number=1)
            self.assertNotIn("widget_cancelled", factory.last.l2_seen[1])
            self.assertIn("t: lidar", factory.last.turn_requests[1].text)
            self.assertEqual(result.text, "Continuing with lidar.")
        finally:
            await native.aclose()


class TurnEndingTest(TestCase):
    async def test_yolo_dashboard_handoff_ends_the_turn(self) -> None:
        scripts = [
            [
                tools(
                    (
                        "clarification",
                        {
                            "prompt": "Topic?",
                            "output": ["t"],
                            "host_dashboard": "my_dash",
                        },
                    )
                ),
                text("never streamed"),
            ]
        ]
        native, factory, interactive, _ = make_native(scripts, session_yolo=True)
        native.tool_registry["my_dash"] = ToolDefinition(
            name="my_dash", tool_type="Dashboard", dashboard_config={"label": "My Dash"}
        )
        async with native:
            result = await native.run_agentic_loop("go")
            tool_text = factory.last.tool_results[0][1]
            self.assertTrue(tool_text.startswith(END_TURN_MARKER))
            self.assertIn("Answered autonomously", tool_text)
            self.assertNotIn("never streamed", result.text)
            self.assertEqual(interactive.widgets, [])
            self.assertEqual(len(factory.last.turn_requests), 1)


@attrs(slots=False)
class _ListBase(InferencerBase):
    """Backend for the classic orchestrator: answers from a script."""

    script: list = []

    def _infer(self, inp, cfg=None, **kw):
        return self.script.pop(0)

    async def _ainfer(self, inp, cfg=None, **kw):
        return self.script.pop(0)


class _HubExecutor(RecordingExecutor):
    """A tool executor that can also open the experiment hub."""

    def __init__(self) -> None:
        super().__init__()
        self.hubs: list[dict] = []

    async def create_experiment_hub(
        self,
        selected_details,
        proposals_data,
        custom_queries=None,
        group_by="batch",
        auto_implement=True,
    ) -> str:
        self.hubs.append(
            {"selected": selected_details, "auto_implement": auto_implement}
        )
        return "hub"

    async def open_experiment_hub(self, proposals_data, **kw) -> str:
        return "hub"


def _tools_block(*calls: dict) -> str:
    return "```json ToolsToInvoke\n" + "\n".join(json.dumps(c) for c in calls) + "\n```"


def _with_hub(registry: dict) -> dict:
    registry["experiment_hub"] = load_all_tools()["experiment_hub"]
    return registry


def _proposals_file() -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="af_widget_parity_"), "proposals.json")
    write_proposal_index(
        Path(path),
        ProposalIndex(
            version="1",
            total_count=2,
            groups=[
                ProposalGroup(
                    phase=1,
                    label="Quick Wins",
                    proposals=[
                        Proposal(id="P1", rank=1, title="Add caching"),
                        Proposal(id="P2", rank=2, title="Rewrite auth"),
                    ],
                )
            ],
        ),
    )
    return path


class WidgetAnswerParityTest(TestCase):
    """One widget answer through the classic and the native orchestrator: the
    same bindings, bundled-action arguments (``__var__`` + the user's
    parameter overrides), turn variables, mailbox clearing, dashboard opening
    and answer text — both run widget_core's after-answer steps."""

    async def _classic(self, script: list, answer, executor, registry) -> tuple:
        base = _ListBase()
        base.script = list(script)
        interactive = RecordingInteractive([answer])
        new_turns: list[str] = []

        async def on_new_turn(turn: int, user_input: str) -> int:
            new_turns.append(user_input)
            return turn

        classic = ConversationalInferencer(
            base_inferencer=base,
            tool_registry=registry,
            tool_executor=executor,
            max_iterations=5,
        )
        result = await classic.run_agentic_loop(
            "go", interactive=interactive, on_new_turn=on_new_turn
        )
        return classic, result, interactive, new_turns, base

    async def _native(self, script: list, answer, executor) -> tuple:
        native, factory, interactive, _ = make_native(
            script, answers=[answer], executor=executor
        )
        _with_hub(native.tool_registry)
        new_turns: list[str] = []

        async def on_new_turn(turn: int, user_input: str) -> int:
            new_turns.append(user_input)
            return turn

        async with native:
            result = await native.run_agentic_loop("go", on_new_turn=on_new_turn)
        return native, result, interactive, new_turns, factory

    async def test_bundled_action_gets_the_same_arguments_and_answer_text(
        self,
    ) -> None:
        answer = {
            "choice": "yes",
            "param_overrides": {"depth": "deep"},
            "variables": {"style": "short"},
        }
        confirm = {"prompt": "Write it?"}
        action = {"name": "write_brief", "arguments": {"topic": "__go__"}}
        classic_exec, native_exec = RecordingExecutor(), RecordingExecutor()
        classic, c_result, c_ui, c_turns, _ = await self._classic(
            [
                "Let me confirm.\n"
                + _tools_block(
                    {
                        "type": "conversation",
                        "name": "confirmation",
                        "arguments": confirm,
                        "output": ["go"],
                    },
                    {"type": "action", **action},
                ),
                "Done.",
            ],
            answer,
            classic_exec,
            helpers_tool_registry(),
        )
        native, n_result, n_ui, n_turns, factory = await self._native(
            [
                [
                    text("Let me confirm."),
                    tools(
                        (
                            "confirmation",
                            {**confirm, "output": ["go"], "then_run": action},
                        )
                    ),
                ],
                [text("Done.")],
            ],
            answer,
            native_exec,
        )
        expected_call = ("write_brief", {"topic": "yes", "depth": "deep"})
        self.assertEqual(classic_exec.calls, [expected_call])
        self.assertEqual(native_exec.calls, [expected_call])
        # The confirmation's parameter panel comes from the bundled action.
        panel = c_ui.widgets[0]["input_mode"].metadata["tool_params"]
        self.assertEqual([p["name"] for p in panel], ["--depth"])
        self.assertEqual(n_ui.widgets[0]["input_mode"].metadata["tool_params"], panel)
        for host in (classic, native):
            self.assertEqual(host.mailboxes, WidgetMailboxes())
            self.assertEqual(host.prior_context["style"], "short")
            self.assertTrue(host.prior_context["_confirmation_gate_passed"])
        response, variables, results = [
            m["content"] for m in classic.get_messages() if m["role"] == "user"
        ][-3:]
        self.assertEqual(response, f"{WIDGET_RESPONSE_PREFIX}\ngo: yes")
        self.assertEqual(variables, "[style]: short")
        self.assertEqual(
            factory.last.turn_requests[1].text,
            f"{response}\n{variables}\n\n{results}",
        )
        self.assertEqual(c_turns, ["go", response])
        self.assertEqual(n_turns, ["go", response])
        for result in (c_result, n_result):
            self.assertEqual(
                [a.tool for a in result.completed_actions], ["write_brief"]
            )
            self.assertEqual(result.text, "Done.")

    async def test_dashboard_directive_opens_the_hub_and_ends_the_turn(self) -> None:
        path = _proposals_file()
        answer = {"selected_proposals": ["P2"], "auto_implement": True}
        selection = {"prompt": "Pick", "proposals_path": path, "experiment_hub": True}
        classic_exec, native_exec = _HubExecutor(), _HubExecutor()
        classic, c_result, c_ui, _, base = await self._classic(
            [
                "Pick some.\n"
                + _tools_block(
                    {
                        "type": "conversation",
                        "name": "proposal_selection",
                        "arguments": selection,
                    }
                ),
                "never reached",
            ],
            answer,
            classic_exec,
            _with_hub(helpers_tool_registry()),
        )
        native, n_result, n_ui, _, factory = await self._native(
            [
                [text("Pick some."), tools(("proposal_selection", selection))],
                [text("never reached")],
            ],
            answer,
            native_exec,
        )
        hub = [{"selected": [{"id": "P2"}], "auto_implement": True}]
        self.assertEqual(classic_exec.hubs, hub)
        self.assertEqual(native_exec.hubs, hub)
        for host, ui in ((classic, c_ui), (native, n_ui)):
            self.assertEqual(host.mailboxes, WidgetMailboxes())
            self.assertEqual(host.prior_context["selected_proposal_ids"], "P2")
            self.assertEqual(
                ui.widgets[0]["input_mode"].metadata["submit_label"],
                "📊 Go To Experiment Hub",
            )
        self.assertEqual(base.script, ["never reached"])
        self.assertEqual(len(factory.last.turn_requests), 1)
        self.assertEqual(c_result.text, "Pick some.")
        self.assertEqual(n_result.text, "Pick some.")


class DashboardHandoffTest(TestCase):
    """A non-yolo dashboard handoff ends the turn quietly; no vendor turn
    reports the answer, so the mirror keeps it and the next L2 delivers it."""

    async def test_the_answer_reaches_the_next_turn_once(self) -> None:
        path = _proposals_file()
        selection = {"prompt": "Pick", "proposals_path": path, "experiment_hub": True}
        native, factory, interactive, executor = make_native(
            [
                [text("Pick some."), tools(("proposal_selection", selection))],
                [text("The hub has P2.")],
                [text("Anything else?")],
            ],
            answers=[{"selected_proposals": ["P2"]}],
        )
        _with_hub(native.tool_registry)
        answer = f"{WIDGET_RESPONSE_PREFIX}\nselected_proposal_ids: P2"
        async with native:
            result = await native.run_agentic_loop("pick proposals")
            backend = factory.last
            self.assertEqual(len(backend.turn_requests), 1)
            self.assertEqual(result.text, "Pick some.")
            self.assertEqual(
                native.get_messages()[-1], {"role": "user", "content": answer}
            )
            self.assertEqual(executor.calls, [])
            await native.run_agentic_loop("what now?")
            self.assertIn('type="dashboard_handoff"', backend.l2_seen[1])
            self.assertIn(answer, backend.l2_seen[1])
            await native.run_agentic_loop("thanks")
            self.assertNotIn("dashboard_handoff", backend.l2_seen[2])

    async def test_after_a_restart_the_notice_falls_back_to_the_mirror(self) -> None:
        path = _proposals_file()
        selection = {"prompt": "Pick", "proposals_path": path, "experiment_hub": True}
        native, factory, _, _ = make_native(
            [
                [tools(("proposal_selection", selection))],
                [text("The hub has P1.")],
            ],
            answers=[{"selected_proposals": ["P1"]}],
        )
        _with_hub(native.tool_registry)
        async with native:
            await native.run_agentic_loop("pick proposals")
            native._notice_bodies.clear()  # in-memory only, like a restart
            await native.run_agentic_loop("what now?")
            self.assertIn(
                f"{WIDGET_RESPONSE_PREFIX}\nselected_proposal_ids: P1",
                factory.last.l2_seen[1],
            )


class BackgroundActionAnswerTest(TestCase):
    """D6 / F6: an answer whose bundled action starts in the background ends
    the host turn quietly — ``on_turn_complete``, no further model call — as
    in the text-protocol orchestrator. The mirror records the answer and the
    next turn's L2 delivers it once, ahead of the action's completion."""

    _QUESTION = {"prompt": "Topic?", "output": ["topic"]}
    _ACTION = {"name": "write_brief", "arguments": {"topic": "__topic__"}}
    _RESPONSE = f"{WIDGET_RESPONSE_PREFIX}\ntopic: lidar"

    @staticmethod
    def _recorder(events: list):
        async def on_new_turn(turn: int, user_input: str) -> int:
            events.append(("new_turn", user_input))
            return turn

        async def on_turn_complete(iteration: int) -> None:
            events.append(("turn_complete",))

        return {"on_new_turn": on_new_turn, "on_turn_complete": on_turn_complete}

    async def _classic(self, events: list) -> tuple:
        base = _ListBase()
        base.script = [
            "Let me ask.\n"
            + _tools_block(
                {
                    "type": "conversation",
                    "name": "clarification",
                    "arguments": {"prompt": "Topic?"},
                    "output": ["topic"],
                },
                {"type": "action", **self._ACTION},
            ),
            "never reached",
        ]
        executor = RecordingExecutor()
        classic = ConversationalInferencer(
            base_inferencer=base,
            tool_registry=helpers_tool_registry(async_brief=True),
            tool_executor=executor,
            max_iterations=5,
        )
        result = await classic.run_agentic_loop(
            "go", interactive=RecordingInteractive(["lidar"]), **self._recorder(events)
        )
        await wait_for(lambda: executor.calls)
        return classic, result, base, executor

    @staticmethod
    def _notice_types(native) -> list[str]:
        return [n["type"] for n in native._load_record().pending_notices()]

    async def test_the_turn_ends_as_in_classic_and_the_answer_reaches_the_next_turn(
        self,
    ) -> None:
        c_events, n_events = [], []
        classic, c_result, base, c_exec = await self._classic(c_events)
        native, factory, _, n_exec = make_native(
            [
                [
                    text("Let me ask."),
                    tools(
                        ("clarification", {**self._QUESTION, "then_run": self._ACTION})
                    ),
                ],
                [text("Writing it in the background.")],
                [text("You're welcome.")],
            ],
            answers=["lidar"],
            async_brief=True,
        )
        async with native:
            n_result = await native.run_agentic_loop("go", **self._recorder(n_events))
            backend = factory.last
            mirrored = [
                m["content"] for m in native.get_messages() if m["role"] == "user"
            ]
            await wait_for(lambda: "tool_completion" in self._notice_types(native))
            self.assertEqual(base.script, ["never reached"])
            self.assertEqual(len(backend.turn_requests), 1)
            self.assertEqual(
                c_events,
                [("new_turn", "go"), ("new_turn", self._RESPONSE), ("turn_complete",)],
            )
            self.assertEqual(n_events, c_events)
            call = ("write_brief", {"topic": "lidar"})
            self.assertEqual((c_exec.calls, n_exec.calls), ([call], [call]))
            for result in (c_result, n_result):
                self.assertEqual(result.text, "Let me ask.")
                self.assertFalse(result.has_conversation_tool)
                self.assertEqual(
                    [a.tool for a in result.completed_actions], ["write_brief"]
                )
            # The mirror holds what the classic transcript holds, as one message.
            c_user = [
                m["content"] for m in classic.get_messages() if m["role"] == "user"
            ]
            at = c_user.index(self._RESPONSE)
            answer = f"{self._RESPONSE}\n\n{c_user[at + 1]}"
            self.assertTrue(c_user[at + 1].startswith(TOOL_RESULTS_PREFIX))
            self.assertIn("launched asynchronously", answer)
            self.assertEqual(mirrored[-1], answer)
            self.assertEqual(
                self._notice_types(native), ["background_answer", "tool_completion"]
            )

            await native.run_agentic_loop("what now?")
            self.assertEqual(backend.turn_requests[1].text, "what now?")
            l2 = backend.l2_seen[1]
            self.assertIn('type="background_answer"', l2)
            self.assertIn("started in the background", l2)
            self.assertIn(answer, l2)
            self.assertLess(
                l2.index('type="background_answer"'), l2.index('type="tool_completion"')
            )
            await native.run_agentic_loop("thanks")
            self.assertNotIn("background_answer", backend.l2_seen[2])

    async def test_a_recovered_answer_that_starts_one_ends_the_turn_too(self) -> None:
        native, factory, _, executor = make_native(
            [[tools(("clarification", self._QUESTION))]],
            answers=[None],
            async_brief=True,
        )
        events = []
        async with native:
            first = await native.run_agentic_loop("ask")
            factory.scripts.append([text("Noted.")])
            native.set_pending_widget_answer(
                {
                    "tools": [first.conversation_tool],
                    "action_tools": [self._ACTION],
                    "raw_value": "lidar",
                }
            )
            second = await native.run_agentic_loop(
                "__continue__", **self._recorder(events)
            )
            backend = factory.last
            await wait_for(lambda: "tool_completion" in self._notice_types(native))
            self.assertEqual(len(backend.turn_requests), 1)
            self.assertEqual(events, [("new_turn", self._RESPONSE), ("turn_complete",)])
            self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])
            self.assertEqual(
                [a.tool for a in second.completed_actions], ["write_brief"]
            )
            self.assertFalse(native._load_record().pending_widget)
            await native.run_agentic_loop("what now?")
            self.assertEqual(backend.turn_requests[1].text, "what now?")
            self.assertIn('type="background_answer"', backend.l2_seen[1])
            self.assertIn(self._RESPONSE, backend.l2_seen[1])


class NativeSopProgressionTest(TestCase):
    async def test_sop_advances_through_widget_tool_and_confirmation(self) -> None:
        scripts = [
            [
                tools(
                    ("enter_sop", {"name": "mini_research"}),
                    (
                        "clarification",
                        {"prompt": "Topic?", "output": ["research_topic"]},
                    ),
                )
            ],
            [
                tools(("write_brief", {"topic": "lidar"})),
                tools(("confirmation", {"prompt": "Brief OK?"})),
            ],
            [text("Brief accepted; all phases complete.")],
        ]
        native, factory, interactive, executor = make_native(
            scripts, answers=["lidar", "yes"]
        )
        async with native:
            result = await native.run_agentic_loop("run the mini research sop")
            backend = factory.last
            self.assertEqual(len(backend.turn_requests), 3)
            self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])
            state = native.sop_state
            self.assertIn("0", state.completed_phase_ids())
            self.assertIn("1", state.completed_phase_ids())
            write_result = backend.tool_results[0][1]
            self.assertIn("<af_state_update", write_result)
            self.assertIn("Brief accepted", result.text)
            self.assertEqual(len(interactive.widgets), 2)

    async def test_widget_marker_is_persisted_with_export_state_blob(self) -> None:
        scripts = [
            [tools(("clarification", {"prompt": "Topic?", "output": ["t"]}))],
            [text("ok")],
        ]
        native, _, interactive, _ = make_native(scripts, answers=["lidar"])
        async with native:
            await native.run_agentic_loop("ask", turn_number=1)
            persisted = interactive.persisted[0]
            self.assertEqual(persisted["tools"][0].prompt, "Topic?")
            self.assertEqual(persisted["blob"]["schema"], "native/v1")
            self.assertIn("native", persisted["blob"])

    async def test_recovered_widget_answer_continues_without_reinference(self) -> None:
        scripts = [[tools(("clarification", {"prompt": "Topic?", "output": ["t"]}))]]
        native, factory, _, _ = make_native(scripts, answers=[None])
        async with native:
            first = await native.run_agentic_loop("ask")
            self.assertTrue(first.has_conversation_tool)
            tool = first.conversation_tool
            factory.scripts.append([text("Continuing with lidar.")])
            native.set_pending_widget_answer(
                {"tools": [tool], "action_tools": [], "raw_value": "lidar"}
            )
            second = await native.run_agentic_loop("__continue__")
            self.assertIn("t: lidar", factory.last.turn_requests[-1].text)
            self.assertEqual(second.text, "Continuing with lidar.")

    async def test_recovered_widget_continues_after_the_widget_round(self) -> None:
        """A host restoring the persisted widget blob (OpenStartup's recovery)
        numbers the continuation's rounds after the widget's round, so they do
        not overwrite that round's artifacts."""
        scripts = [[tools(("clarification", {"prompt": "Topic?", "output": ["t"]}))]]
        native, factory, interactive, _ = make_native(scripts, answers=[None])
        rounds: list[int] = []

        async def on_round_start(iteration: int, turn_number: int) -> dict:
            rounds.append(iteration)
            return {"round_index": iteration + 1}

        async with native:
            first = await native.run_agentic_loop(
                "ask", turn_number=1, on_round_start=on_round_start
            )
            blob = interactive.persisted[0]["blob"]
            factory.scripts.append([text("Continuing with lidar.")])
            native.restore_state(blob, reattach_sop=False)
            native.set_pending_widget_answer(
                {
                    "tools": [first.conversation_tool],
                    "action_tools": [],
                    "raw_value": "lidar",
                }
            )
            second = await native.run_agentic_loop(
                "__continue__", turn_number=1, on_round_start=on_round_start
            )
        self.assertEqual(second.text, "Continuing with lidar.")
        self.assertEqual(rounds, [0, 1])  # the continuation is the next round

    async def _recover(self, on_new_turn) -> tuple:
        """Leave a widget pending in host turn 3, then answer it through
        ``set_pending_widget_answer`` with the host's turn number 3."""
        scripts = [[tools(("clarification", {"prompt": "Topic?", "output": ["t"]}))]]
        native, factory, interactive, _ = make_native(
            scripts, answers=[None], rewind_on_repeat_turn=True
        )
        new_turns: list[tuple[int, str]] = []
        round_turns: list[int] = []

        async def recording_new_turn(turn: int, user_input: str) -> int:
            new_turns.append((turn, user_input))
            return await on_new_turn(turn, user_input)

        async def on_round_start(iteration: int, turn_number: int) -> None:
            round_turns.append(turn_number)

        async with native:
            first = await native.run_agentic_loop("ask", turn_number=3)
            factory.scripts.append([text("Continuing with lidar.")])
            native.set_pending_widget_answer(
                {
                    "tools": [first.conversation_tool],
                    "action_tools": [],
                    "raw_value": "lidar",
                }
            )
            second = await native.run_agentic_loop(
                "__continue__",
                turn_number=3,
                on_new_turn=recording_new_turn,
                on_round_start=on_round_start,
            )
        self.assertEqual(second.text, "Continuing with lidar.")
        self.assertEqual(new_turns, [(3, f"{WIDGET_RESPONSE_PREFIX}\nt: lidar")])
        self.assertEqual(len(factory.instances), 1)
        self.assertIsNone(factory.last.open_request.fork_from)
        self.assertEqual(native._load_record().generation, 0)
        return native, interactive, round_turns

    async def test_recovered_widget_answer_keeps_the_host_turn_number(self) -> None:
        async def same_turn(turn: int, user_input: str) -> int:
            return turn

        native, interactive, round_turns = await self._recover(same_turn)
        self.assertEqual(round_turns, [3])
        self.assertEqual(native._load_record().last_turn, 3)
        self.assertEqual(interactive.turn_boundaries, [])

    async def test_a_host_turn_change_after_recovery_is_announced(self) -> None:
        async def next_turn(turn: int, user_input: str) -> int:
            return turn + 1

        native, interactive, round_turns = await self._recover(next_turn)
        self.assertEqual(round_turns, [4])
        self.assertEqual(native._load_record().last_turn, 4)
        self.assertEqual(interactive.turn_boundaries, [4])


class CompoundWidgetTest(TestCase):
    async def test_a_compound_widget_is_shown_after_the_turn_and_answered_as_in_ci(
        self,
    ) -> None:
        order = []

        class _OrderedInteractive(RecordingInteractive):
            async def asend_response(self, text, **kwargs):
                order.append("widget shown")
                await super().asend_response(text, **kwargs)

        async def two_questions(backend, request):
            yield TextDelta(message_id="m0", text="Two questions.")
            yield MessageEnd(
                message_id="m0",
                text="Two questions.",
                tool_use_ids=("a1", "a2"),
                af_tool_use_ids=("a1", "a2"),
                message_uuid="u0",
            )
            await backend.call_tool(
                "clarification",
                {"prompt": "Topic?", "output": ["topic"]},
                tool_use_id="a1",
            )
            await backend.call_tool(
                "single_choice",
                {"prompt": "Depth?", "choices": ["quick", "deep"], "output": ["depth"]},
                tool_use_id="a2",
            )
            order.append("vendor turn ended")
            yield turn_end(backend)

        native, factory, _, _ = make_native([two_questions, [text("Thanks.")]])
        ui = _OrderedInteractive([{"values": {"topic": "lidar", "depth": "deep"}}])
        async with native:
            result = await native.run_agentic_loop("ask me", interactive=ui)
        self.assertEqual(order, ["vendor turn ended", "widget shown"])
        (widget,) = ui.widgets
        self.assertTrue(widget["input_mode"].metadata.get("compound"))
        queued = factory.last.tool_results
        self.assertEqual([r[2] for r in queued], [False, False])
        self.assertTrue(all(r[1].startswith(END_TURN_MARKER) for r in queued))
        self.assertEqual(len(factory.last.turn_requests), 2)
        self.assertEqual(
            factory.last.turn_requests[1].text,
            f"{WIDGET_RESPONSE_PREFIX}\ntopic: lidar\ndepth: deep",
        )
        self.assertEqual(result.text, "Thanks.")


_ASK_AGAIN = [tools(("clarification", {"prompt": "Again?", "output": ["again"]}))]


class VendorTurnCapTest(TestCase):
    """Plan §6.1 step 5: one host call runs at most
    ``max_vendor_turns_per_call`` vendor turns (default 50; ``<= 0`` means
    the safety ceiling); each answered widget batch is another vendor turn."""

    async def _run(self, cap: int) -> tuple:
        native, factory, interactive, _ = make_native(
            [_ASK_AGAIN] * 6,
            answers=[f"y{i}" for i in range(1, 7)],
            max_vendor_turns_per_call=cap,
        )
        async with native:
            await native.run_agentic_loop("keep asking")
        return len(factory.last.turn_requests), len(interactive.widgets)

    async def test_a_call_ends_after_the_capped_number_of_vendor_turns(self) -> None:
        native, *_ = make_native([])
        self.assertEqual(native.max_vendor_turns_per_call, 50)
        self.assertEqual(await self._run(2), (2, 2))
        self.assertEqual(await self._run(3), (3, 3))

    async def test_a_cap_of_zero_or_less_means_the_safety_ceiling(self) -> None:
        with mock.patch.object(turn_loop, "_SAFETY_CEILING", 4):
            for cap in (0, -1):
                with self.subTest(cap=cap):
                    self.assertEqual((await self._run(cap))[0], 4)

    async def test_the_result_says_whether_the_cap_cut_the_call_short(self) -> None:
        """``exhausted_max_iterations`` as the text protocol sets it at
        ``max_iterations``: when the cap ended a call that had more to do,
        not when the call's last allowed vendor turn finished it."""
        ask = "Asking.\n" + _tools_block(
            {
                "type": "conversation",
                "name": "clarification",
                "arguments": {"prompt": "Again?"},
                "output": ["again"],
            }
        )
        cases = {
            "cut short": ([_ASK_AGAIN, _ASK_AGAIN], [ask, ask], True),
            "finished in the last allowed turn": (
                [_ASK_AGAIN, [text("ok")]],
                [ask, "ok"],
                False,
            ),
            "finished early": ([[text("ok")]], ["ok"], False),
        }
        for name, (scripts, classic_script, exhausted) in cases.items():
            with self.subTest(name):
                native, _, _, _ = make_native(
                    scripts, answers=["y1", "y2"], max_vendor_turns_per_call=2
                )
                async with native:
                    result = await native.run_agentic_loop("go")
                self.assertIs(result.exhausted_max_iterations, exhausted)
                self.assertFalse(result.has_conversation_tool)
                base = _ListBase()
                base.script = list(classic_script)
                classic = ConversationalInferencer(
                    base_inferencer=base,
                    tool_registry=helpers_tool_registry(),
                    max_iterations=2,
                )
                classic_result = await classic.run_agentic_loop(
                    "go", interactive=RecordingInteractive(["y1", "y2"])
                )
                self.assertIs(classic_result.exhausted_max_iterations, exhausted)

    async def test_an_answer_collected_in_the_last_capped_turn_reaches_the_agent(
        self,
    ) -> None:
        """As in the text protocol, the last vendor turn the cap allows still
        shows its question and applies the answer; no vendor turn of the call
        is left to send it, so the mirror records it and the next turn's L2
        delivers it once."""
        native, factory, _, _ = make_native(
            [_ASK_AGAIN, _ASK_AGAIN, [text("ok")], [text("ok")]],
            answers=["y1", "y2"],
            max_vendor_turns_per_call=2,
        )
        new_turns = []

        async def on_new_turn(turn, user_input):
            new_turns.append(user_input)

        async with native:
            await native.run_agentic_loop("keep asking", on_new_turn=on_new_turn)
            self.assertEqual(native.prior_context["again"], "y2")
            answer = f"{WIDGET_RESPONSE_PREFIX}\nagain: y2"
            self.assertEqual(native.get_messages()[-1]["content"], answer)
            self.assertEqual(
                new_turns, ["keep asking", f"{WIDGET_RESPONSE_PREFIX}\nagain: y1"]
            )
            await native.run_agentic_loop("and now?")
            await native.run_agentic_loop("anything else?")
        backend = factory.last
        self.assertEqual(len(backend.turn_requests), 4)
        self.assertEqual(backend.turn_requests[2].text, "and now?")
        self.assertIn('type="unsent_turn"', backend.l2_seen[2])
        self.assertIn("It has been applied", backend.l2_seen[2])
        self.assertIn("again: y2", backend.l2_seen[2])
        self.assertNotIn("again: y2", backend.l2_seen[3])

    async def test_an_invalid_batch_in_the_last_capped_turn_is_reported_next_turn(
        self,
    ) -> None:
        duplicate = {"prompt": "Topic?", "output": ["topic"], "parallel_group": 1}
        invalid = [tools(("clarification", duplicate), ("clarification", duplicate))]
        native, factory, interactive, _ = make_native(
            [invalid, [text("ok")]], max_vendor_turns_per_call=1
        )
        async with native:
            await native.run_agentic_loop("ask me")
            self.assertEqual(interactive.widgets, [])
            await native.run_agentic_loop("well?")
        backend = factory.last
        self.assertEqual([r.text for r in backend.turn_requests], ["ask me", "well?"])
        self.assertIn('type="unsent_turn"', backend.l2_seen[1])
        self.assertIn("were not shown to the user", backend.l2_seen[1])
        self.assertIn("duplicate primary output key 'topic'", backend.l2_seen[1])

    async def test_after_a_restart_the_unsent_answer_comes_from_the_mirror(
        self,
    ) -> None:
        native, factory, _, _ = make_native(
            [_ASK_AGAIN, [text("ok")]], answers=["y1"], max_vendor_turns_per_call=1
        )
        async with native:
            await native.run_agentic_loop("ask once")
            native._notice_bodies.clear()  # what a restarted host process holds
            await native.run_agentic_loop("and now?")
        self.assertIn("again: y1", factory.last.l2_seen[1])

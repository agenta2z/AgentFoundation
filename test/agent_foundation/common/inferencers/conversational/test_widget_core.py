"""widget_core: the conversation-widget functional core (prepare, present and
collect, decode, yolo, after-answer, recovery) on a real ConversationalInferencer
as the WidgetHost."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational import (
    widget_core,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_response_parser import (
    tool_invocation_to_conversation_tool,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    WidgetMailboxes,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocol_text import (
    TOOL_RESULT_HEADER,
    TOOL_RESULTS_PREFIX,
    WIDGET_RESPONSE_PREFIX,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition
from agent_foundation.resources.tools.registry import load_all_tools
from attr import attrs
from later.unittest import TestCase

_PROPOSALS = {
    "groups": [
        {
            "proposals": [
                {"id": "P1", "title": "Add caching", "impact": "high"},
                {"id": "P2", "title": "Rewrite auth"},
            ]
        }
    ]
}


@attrs(slots=False)
class _NoOpBase(InferencerBase):
    def _infer(self, inp, cfg=None, **kw):
        return ""

    async def _ainfer(self, inp, cfg=None, **kw):
        return ""


class _HubExecutor:
    """Tool executor that can also open the experiment hub (records both)."""

    def __init__(self, context_updates: Optional[dict[str, Any]] = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.hubs: list[dict[str, Any]] = []
        self.context_updates = context_updates or {}

    async def __call__(self, name: str, arguments: dict) -> ToolExecutionResult:
        self.calls.append((name, dict(arguments)))
        return ToolExecutionResult(
            result=f"{name} done", context_updates=dict(self.context_updates)
        )

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


class _Interactive:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.sent: list[dict[str, Any]] = []
        self.persisted: list[dict[str, Any]] = []

    async def asend_response(self, text, flag=None, input_mode=None, **kw):
        self.sent.append(
            {"text": text, "input_mode": input_mode, "prompt_data": kw["prompt_data"]}
        )

    async def aget_input(self):
        return self.answer

    def persist_pending_widget(self, tools, action_tools, blob):
        self.persisted.append(
            {"tools": tools, "action_tools": action_tools, "blob": blob}
        )


def _registry() -> dict[str, ToolDefinition]:
    tools = {
        k: v
        for k, v in load_all_tools().items()
        if v.tool_type in ("Conversation", "Dashboard")
    }
    tools["write_brief"] = ToolDefinition(
        name="write_brief",
        description="Write a research brief.",
        tool_type="Action",
        parameters=[
            ParameterDef(name="topic", type="string", required=True, positional=True),
            ParameterDef(name="--depth", type="string"),
        ],
    )
    return tools


def _ci(executor: Any = None) -> ConversationalInferencer:
    return ConversationalInferencer(
        base_inferencer=_NoOpBase(),
        tool_registry=_registry(),
        tool_executor=executor if executor is not None else _HubExecutor(),
        prior_context={"session_root_path": "/repo"},
    )


def _tool(name: str, output: Optional[list[str]] = None, **arguments: Any):
    return tool_invocation_to_conversation_tool(
        {"name": name, "arguments": arguments, "output": output or []}
    )


class PrepareTest(TestCase):
    async def test_proposals_become_choices_and_the_hub_flag_a_handoff(self) -> None:
        tool = _tool(
            "proposal_selection", prompt="Pick", proposals=_PROPOSALS, experiment_hub=1
        )
        widget_core.prepare(_ci(), [tool])
        self.assertEqual([c.value for c in tool.choices], ["P1", "P2"])
        self.assertEqual(tool.output_vars, ["selected_proposal_ids"])
        self.assertEqual(tool.metadata["open_dashboard"], "experiment_hub")
        self.assertTrue(widget_core.is_dashboard_handoff([tool]))
        self.assertFalse(widget_core.is_dashboard_handoff([_tool("clarification")]))


class PresentAndDecodeTest(TestCase):
    async def test_one_tool_is_shown_persisted_and_decoded(self) -> None:
        ci = _ci()
        tool = _tool("clarification", ["topic"], prompt="Topic?")
        interactive = _Interactive("lidar")
        collected = await widget_core.present_and_collect(
            ci,
            [tool],
            "Let me ask.",
            interactive=interactive,
            turn_number=2,
            iteration=1,
        )
        self.assertEqual(collected, {"topic": "lidar"})
        self.assertEqual(ci.prior_context["topic"], "lidar")
        sent = interactive.sent[0]
        self.assertEqual(sent["text"], "Let me ask.")
        self.assertEqual(sent["input_mode"].prompt, "Topic?")
        self.assertEqual(set(sent["prompt_data"]), set(ci.last_prompt_data()))
        persisted = interactive.persisted[0]
        self.assertEqual(persisted["tools"], [tool])
        self.assertEqual(persisted["blob"]["turn_number"], 2)
        self.assertEqual(persisted["blob"]["iteration"], 1)

    async def test_several_tools_form_one_compound_widget(self) -> None:
        ci = _ci()
        tools = [
            _tool("clarification", ["topic"], prompt="Topic?"),
            _tool(
                "single_choice", ["depth"], prompt="Depth?", choices=["quick", "deep"]
            ),
        ]
        interactive = _Interactive({"values": {"topic": "lidar", "depth": "deep"}})
        collected = await widget_core.present_and_collect(
            ci, tools, "Two questions.", interactive=interactive
        )
        self.assertEqual(collected, {"topic": "lidar", "depth": "deep"})
        mode = interactive.sent[0]["input_mode"]
        self.assertTrue(mode.metadata["compound"])
        self.assertEqual(
            [c["output_var"] for c in mode.metadata["tools"]], ["topic", "depth"]
        )
        # Without (turn, iteration) there is no emit-point blob to persist.
        self.assertEqual(interactive.persisted, [])
        self.assertEqual(ci.prior_context["single_choice__depth"], "deep")

    async def test_composite_bindings_are_returned_not_left_on_the_host(self) -> None:
        ci = _ci()
        tool = _tool(
            "single_choice",
            ["mode"],
            prompt="How?",
            choices=[
                {"label": "Auto", "value": "auto"},
                {
                    "label": "Manual",
                    "value": "manual",
                    "input": {"name": "paths", "expected_input_type": "free_text"},
                },
            ],
        )
        collected = await widget_core.decode(
            ci, [tool], {"choice_index": 1, "inputs": {"paths": "src/"}}
        )
        self.assertEqual(collected, {"mode": "manual", "paths": "src/"})
        self.assertIsNone(getattr(ci, "_last_handler_bindings", None))

    async def test_multiple_choice_and_form_answers_reach_the_next_prompt(
        self,
    ) -> None:
        ci = _ci()
        picks = _tool(
            "multiple_choice", ["picks"], prompt="Which?", choices=["a", "b", "c"]
        )
        self.assertEqual(await widget_core.decode(ci, [picks], "a,c"), {"picks": "a,c"})
        form = _tool("tool_argument_form", ["brief"], prompt="Arguments?")
        await widget_core.decode(
            ci, [form], {"fields": {"topic": "lidar", "depth": "deep"}}
        )
        await widget_core.decode(ci, [form], "short")
        ci._render_prompt("next")
        feed = ci.last_prompt_data()["template_feed"]
        self.assertEqual(feed["picks"], "a,c")
        self.assertEqual(feed["multiple_choice__picks"], "a,c")
        self.assertEqual(feed["topic"], "lidar")
        self.assertEqual(feed["tool_argument_form__depth"], "deep")
        self.assertEqual(feed["brief"], "short")

    async def test_a_multiple_choice_widget_answer_is_decoded(self) -> None:
        ci = _ci()
        picks = _tool(
            "multiple_choice", ["picks"], prompt="Which?", choices=["a", "b", "c"]
        )
        interactive = _Interactive(
            {
                "selections": [
                    {"choice_index": 2},
                    {"custom_text": "d"},
                    {"choice_index": 0},
                ]
            }
        )
        collected = await widget_core.present_and_collect(
            ci, [picks], "Pick some.", interactive=interactive
        )
        self.assertEqual(collected, {"picks": "c,d,a"})
        self.assertEqual(ci.prior_context["multiple_choice__picks"], "c,d,a")

    async def test_a_missing_answer_decodes_to_none(self) -> None:
        ci = _ci()
        self.assertIsNone(await widget_core.decode(ci, [_tool("clarification")], None))
        self.assertIsNone(await widget_core.decode(ci, [], "x"))
        self.assertIsNone(
            await widget_core.present_and_collect(
                ci, [_tool("clarification")], "", interactive=None
            )
        )

    async def test_recover_decodes_the_persisted_widget(self) -> None:
        ci = _ci()
        tool = _tool("clarification", ["topic"], prompt="Topic?")
        then_run = [{"name": "write_brief", "arguments": {"topic": "__topic__"}}]
        recovered = await widget_core.recover(
            ci, {"tools": [tool], "action_tools": then_run, "raw_value": "lidar"}
        )
        self.assertEqual(recovered.tools, [tool])
        self.assertEqual(recovered.then_run, then_run)
        self.assertEqual(recovered.bindings, {"topic": "lidar"})
        unusable = await widget_core.recover(ci, {"tools": [tool], "raw_value": None})
        self.assertIsNone(unusable.bindings)


class YoloTest(TestCase):
    async def test_synthesized_answers_go_through_the_handlers(self) -> None:
        ci = _ci()
        tools = [
            _tool(
                "clarification", ["where"], prompt="Where?", expected_input_type="path"
            ),
            _tool(
                "proposal_selection",
                prompt="Pick",
                proposals=_PROPOSALS,
                experiment_hub=1,
            ),
        ]
        widget_core.prepare(ci, tools)
        collected = await widget_core.synthesize_yolo(ci, tools)
        # A path never gets prose under yolo: the session root.
        self.assertEqual(collected["where"], "/repo")
        self.assertEqual(collected["selected_proposal_ids"], "P1,P2")
        self.assertEqual(ci.prior_context["selected_proposal_ids"], "P1,P2")
        self.assertIsNone(await widget_core.synthesize_yolo(ci, []))

    async def test_a_prefilled_default_is_the_answer(self) -> None:
        ci = _ci()
        tools = [
            _tool(
                "clarification",
                ["target"],
                prompt="Target?",
                expected_input_type="path",
                prefix="/repo",
                default="/repo/toy_model",
            ),
            _tool(
                "clarification",
                ["subdir"],
                prompt="Subdir?",
                expected_input_type="path",
                prefix="/repo",
                default="toy_model/data",
            ),
            _tool("clarification", ["topic"], prompt="Topic?", default="lidar"),
            _tool(
                "clarification",
                ["blank"],
                prompt="Path?",
                expected_input_type="path",
                default="  ",
            ),
            _tool("confirmation", ["go"], prompt="Proceed?", default="no"),
        ]
        collected = await widget_core.synthesize_yolo(ci, tools)
        self.assertEqual(collected["target"], "/repo/toy_model")
        self.assertEqual(collected["subdir"], "/repo/toy_model/data")
        self.assertEqual(collected["topic"], "lidar")
        self.assertEqual(collected["blank"], "/repo")
        self.assertEqual(collected["go"], "yes")
        self.assertEqual(ci.prior_context["target"], "/repo/toy_model")

    async def test_the_classic_loop_answers_a_path_with_its_default(self) -> None:
        base = _ScriptedBase()
        base.script = [
            "Setting up.\n"
            + _tools_block(
                '{"type": "conversation", "name": "clarification", "arguments": '
                '{"prompt": "Target path?", "expected_input_type": "path", '
                '"prefix": "/repo", "default": "/repo/toy_model"}, '
                '"output": ["workflow_target_path"]}'
            ),
            "Done.",
        ]
        ci = _ci()
        ci.base_inferencer = base
        ci.yolo_mode = True
        interactive = _Interactive("never asked")
        result = await ci.run_agentic_loop(
            "optimize the model at /repo/toy_model", interactive=interactive
        )
        self.assertEqual(interactive.sent, [])
        self.assertEqual(ci.prior_context["workflow_target_path"], "/repo/toy_model")
        self.assertIn(
            "'workflow_target_path': '/repo/toy_model'",
            [m["content"] for m in ci.get_messages() if m["role"] == "user"][0],
        )
        self.assertEqual(result.text, "Done.")


class AfterAnswerTest(TestCase):
    async def test_then_run_gets_bindings_overrides_and_turn_variables(self) -> None:
        executor = _HubExecutor()
        ci = _ci(executor)
        tool = _tool("confirmation", ["go"], prompt="Write it?")
        collected = await widget_core.decode(
            ci,
            [tool],
            {
                "choice": "yes",
                "param_overrides": {"depth": "deep"},
                "variables": {"style": "short"},
            },
        )
        applied = []
        answer = await widget_core.after_answer(
            ci,
            [tool],
            [
                {
                    "name": "write_brief",
                    "arguments": {"topic": "__go__", "depth": "quick", "x": "__nope__"},
                }
            ],
            collected,
            on_applied=lambda name, outcome: applied.append((name, outcome.text)),
        )
        self.assertEqual(
            executor.calls,
            [("write_brief", {"topic": "yes", "depth": "deep", "x": "__nope__"})],
        )
        self.assertEqual(ci.mailboxes, WidgetMailboxes())
        self.assertEqual(ci.prior_context["style"], "short")
        self.assertEqual(answer.turn_variables, {"style": "short"})
        self.assertEqual(applied, [("write_brief", "write_brief done")])
        self.assertFalse(answer.dashboard_handoff)
        self.assertFalse(answer.async_dispatched)
        self.assertEqual(answer.response_text, f"{WIDGET_RESPONSE_PREFIX}\ngo: yes")
        self.assertEqual(
            answer.message(),
            f"{WIDGET_RESPONSE_PREFIX}\ngo: yes\n[style]: short\n\n"
            f"{TOOL_RESULTS_PREFIX}\n{TOOL_RESULT_HEADER.format('write_brief')}\n"
            "write_brief done",
        )

    async def test_without_bundled_actions_turn_variables_are_dropped(self) -> None:
        ci = _ci()
        ci.mailboxes.action_overrides = {"depth": "deep"}
        ci.mailboxes.turn_variables = {"style": "short"}
        answer = await widget_core.after_answer(
            ci, [_tool("confirmation")], [], {"input": "yes"}
        )
        self.assertNotIn("style", ci.prior_context)
        self.assertEqual(answer.turn_variables, {})
        self.assertEqual(answer.message(), answer.response_text)
        self.assertEqual(ci.mailboxes, WidgetMailboxes())

    async def test_dashboard_handoff_opens_the_hub_and_runs_nothing(self) -> None:
        executor = _HubExecutor()
        ci = _ci(executor)
        tool = _tool(
            "proposal_selection", prompt="Pick", proposals=_PROPOSALS, experiment_hub=1
        )
        widget_core.prepare(ci, [tool])
        collected = await widget_core.decode(
            ci, [tool], {"selected_proposals": ["P2"], "auto_implement": True}
        )
        self.assertEqual(ci.mailboxes.dashboard_directives, {"auto_implement": True})
        answer = await widget_core.after_answer(
            ci,
            [tool],
            [{"name": "write_brief", "arguments": {"topic": "x"}}],
            collected,
        )
        self.assertTrue(answer.dashboard_handoff)
        self.assertEqual(executor.calls, [])
        self.assertEqual(
            executor.hubs, [{"selected": [{"id": "P2"}], "auto_implement": True}]
        )
        self.assertEqual(ci.mailboxes, WidgetMailboxes())

    async def test_a_result_returning_after_its_turn_is_not_applied(self) -> None:
        executor = _HubExecutor(context_updates={"brief_path": "/b.md"})
        ci = _ci(executor)
        live = [True]

        async def ending_executor(name, arguments):
            live[0] = False
            return await executor(name, arguments)

        ci.tool_executor = ending_executor
        applied = []
        answer = await widget_core.after_answer(
            ci,
            [_tool("clarification", ["topic"])],
            [
                {"name": "write_brief", "arguments": {"topic": "__topic__"}},
                {"name": "write_brief", "arguments": {"topic": "second"}},
            ],
            {"topic": "lidar"},
            on_applied=lambda name, outcome: applied.append(name),
            is_live=lambda: live[0],
        )
        self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])
        self.assertNotIn("brief_path", ci.prior_context)
        self.assertEqual((applied, answer.then_run_results), ([], ()))

    async def test_substitute_vars(self) -> None:
        self.assertEqual(
            widget_core.substitute_vars(
                {"a": "__x__", "b": "__y__", "c": 3, "d": "x"}, {"x": "1"}
            ),
            {"a": "1", "b": "__y__", "c": 3, "d": "x"},
        )
        self.assertEqual(
            widget_core.substitute_vars({"a": "__x__"}, "x"), {"a": "__x__"}
        )
        self.assertEqual(widget_core.substitute_vars(None, {}), {})


@attrs(slots=False)
class _ScriptedBase(InferencerBase):
    script: list = []

    def _infer(self, inp, cfg=None, **kw):
        return self.script.pop(0)

    async def _ainfer(self, inp, cfg=None, **kw):
        return self.script.pop(0)


def _tools_block(*calls: str) -> str:
    return "```json ToolsToInvoke\n" + "\n".join(calls) + "\n```"


_CONFIRM_AND_WRITE = _tools_block(
    '{"type": "conversation", "name": "confirmation", '
    '"arguments": {"prompt": "Write it?"}, "output": ["go"]}',
    '{"type": "action", "name": "write_brief", '
    '"arguments": {"topic": "__go__", "depth": "quick"}}',
)


class ClassicLoopTest(TestCase):
    """The text-protocol loop around the after-answer steps: transcript order,
    the new turn before the bundled actions, and an async bundled action
    ending the turn."""

    async def _run(self, executor: Any, answer: Any) -> tuple:
        events: list[str] = []
        base = _ScriptedBase()
        base.script = ["Let me confirm.\n" + _CONFIRM_AND_WRITE, "Done."]
        ci = _ci(executor)
        ci.base_inferencer = base

        async def on_new_turn(turn: int, user_input: str) -> int:
            events.append(f"new_turn:{user_input}")
            return turn + 1

        original = executor.__call__

        async def recording(name, arguments):
            events.append(f"run:{name}")
            return await original(name, arguments)

        executor.__call__ = recording
        ci.tool_executor = recording
        result = await ci.run_agentic_loop(
            "write a brief",
            interactive=_Interactive(answer),
            on_new_turn=on_new_turn,
        )
        return ci, result, events

    async def test_answer_messages_then_new_turn_then_bundled_actions(self) -> None:
        executor = _HubExecutor()
        ci, result, events = await self._run(
            executor,
            {
                "choice": "yes",
                "param_overrides": {"depth": "deep"},
                "variables": {"style": "short"},
            },
        )
        self.assertEqual(
            executor.calls, [("write_brief", {"topic": "yes", "depth": "deep"})]
        )
        response = f"{WIDGET_RESPONSE_PREFIX}\ngo: yes"
        self.assertEqual(
            events,
            ["new_turn:write a brief", f"new_turn:{response}", "run:write_brief"],
        )
        users = [m["content"] for m in ci.get_messages() if m["role"] == "user"]
        self.assertEqual(
            users,
            [
                response,
                "[style]: short",
                f"{TOOL_RESULTS_PREFIX}\n{TOOL_RESULT_HEADER.format('write_brief')}\n"
                "write_brief done",
            ],
        )
        self.assertEqual(ci.prior_context["style"], "short")
        self.assertEqual(ci.mailboxes, WidgetMailboxes())
        self.assertEqual([a.tool for a in result.completed_actions], ["write_brief"])
        self.assertEqual(result.text, "Done.")

    async def test_async_bundled_action_ends_the_turn(self) -> None:
        executor = _HubExecutor()
        ci = _ci(executor)
        ci.tool_registry["write_brief"].asynchronous = True
        base = _ScriptedBase()
        base.script = ["Let me confirm.\n" + _CONFIRM_AND_WRITE, "never reached"]
        ci.base_inferencer = base
        result = await ci.run_agentic_loop(
            "write a brief", interactive=_Interactive({"choice": "yes"})
        )
        await asyncio.gather(*ci.async_tool_tasks.pending)
        self.assertEqual(base.script, ["never reached"])
        self.assertEqual(result.text, "Let me confirm.")
        contents = [m["content"] for m in ci.get_messages()]
        self.assertIn("launched asynchronously", contents[-2])
        self.assertEqual(
            contents[-1], f"{TOOL_RESULTS_PREFIX}\nwrite_brief: write_brief done"
        )

# pyre-strict

"""``tool_call_budget`` appends a per-turn work budget to the task MetaMate gets,
and ``max_answer_findings`` an answer-format cap after it.

A MetaMate turn runs inside one web request the server cuts at 120 s, and stops at
its per-request memory budget; a count of tool calls keeps a turn inside both. The
budget is on by default (``DEFAULT_TOOL_CALL_BUDGET``); the field sets it for every
call, a call's ``tool_call_budget=`` keyword overrides it, and ``None`` sends the
task unchanged. Writing the answer spends the same 120 s, so the format cap
(off by default) keeps it short; its field and keyword work the same way.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
    metamate_sdk_inferencer,
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    answer_format_directive,
    DEFAULT_TOOL_CALL_BUDGET,
    tool_call_budget_directive,
)

_UNSET = object()


class _FakeClient:
    prompts: list[str] = []

    def __init__(self, cat: Any = None) -> None:
        pass

    def engine_start_v2(self, *, prompt: str, **kwargs: Any) -> Any:
        _FakeClient.prompts.append(prompt)
        return SimpleNamespace(conversation=SimpleNamespace(uuid="conv", fbid="fb"))

    def get_conversation_for_stream(self, conversation_uuid: str) -> Any:
        answer = SimpleNamespace(
            block=SimpleNamespace(
                uuid="a", content=SimpleNamespace(markdown=SimpleNamespace(value="ok"))
            )
        )
        message = SimpleNamespace(
            message=SimpleNamespace(
                role="ASSISTANT", block_uuids=["a"], status="COMPLETED"
            )
        )
        return [message, answer]


def _sent_prompt(
    budget_field: Any = None, findings_field: Any = _UNSET, **call_kwargs: Any
) -> str:
    _FakeClient.prompts = []
    fields: dict[str, Any] = {}
    if budget_field is not _UNSET:
        fields["tool_call_budget"] = budget_field
    if findings_field is not _UNSET:
        fields["max_answer_findings"] = findings_field
    inf = MetamateSDKInferencer(
        api_key=None, poll_interval_seconds=0, code_scope_judge=None, **fields
    )
    with mock.patch.object(
        metamate_sdk_inferencer,
        "resolve_metamate_client_cls",
        return_value=_FakeClient,
    ):
        result = asyncio.run(inf.ainfer("the task", prepared_input=True, **call_kwargs))
    assert result == "ok", result
    (prompt,) = _FakeClient.prompts
    return prompt


class ToolCallBudgetTest(unittest.TestCase):
    def test_the_budget_is_on_by_default(self) -> None:
        self.assertEqual(
            _sent_prompt(budget_field=_UNSET),
            "the task" + tool_call_budget_directive(DEFAULT_TOOL_CALL_BUDGET),
        )
        self.assertEqual(DEFAULT_TOOL_CALL_BUDGET, 6)

    def test_no_budget_leaves_the_task_unchanged(self) -> None:
        self.assertEqual(_sent_prompt(budget_field=None), "the task")

    def test_the_field_appends_the_budget(self) -> None:
        self.assertEqual(
            _sent_prompt(budget_field=6), "the task" + tool_call_budget_directive(6)
        )

    def test_a_call_keyword_budgets_one_call(self) -> None:
        self.assertEqual(
            _sent_prompt(tool_call_budget=4), "the task" + tool_call_budget_directive(4)
        )

    def test_a_call_keyword_overrides_the_field(self) -> None:
        self.assertEqual(
            _sent_prompt(budget_field=6, tool_call_budget=3),
            "the task" + tool_call_budget_directive(3),
        )
        self.assertEqual(
            _sent_prompt(budget_field=6, tool_call_budget=None), "the task"
        )

    def test_the_directive_names_the_budget_and_the_fallback(self) -> None:
        text = tool_call_budget_directive(6)
        self.assertIn("at most 6 tool calls", text)
        self.assertIn('"Not verified"', text)
        self.assertIn("two minutes", text)


class AnswerFormatCapTest(unittest.TestCase):
    def test_the_cap_is_off_by_default(self) -> None:
        self.assertEqual(
            _sent_prompt(budget_field=6), "the task" + tool_call_budget_directive(6)
        )

    def test_the_field_appends_the_cap_after_the_budget(self) -> None:
        self.assertEqual(
            _sent_prompt(budget_field=6, findings_field=8),
            "the task" + tool_call_budget_directive(6) + answer_format_directive(8),
        )

    def test_a_call_keyword_overrides_the_field(self) -> None:
        self.assertEqual(
            _sent_prompt(budget_field=None, max_answer_findings=5),
            "the task" + answer_format_directive(5),
        )
        self.assertEqual(
            _sent_prompt(budget_field=None, findings_field=8, max_answer_findings=None),
            "the task",
        )

    def test_the_directive_names_the_cap_and_the_reason(self) -> None:
        text = answer_format_directive(8)
        self.assertIn("at most 8 findings", text)
        self.assertIn("at most 4 short bullets", text)
        self.assertIn("same two minutes", text)

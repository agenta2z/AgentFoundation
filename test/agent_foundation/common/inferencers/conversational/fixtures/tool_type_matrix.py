# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Phase 0.4 — Tool-type fixture matrix for parity tests.

Enumerates one ConversationTool + one representative response payload per
tool_type. Parity tests (Phase 0.1-0.3) iterate this matrix; adding a new
tool_type = adding one fixture here + one parity assertion.

Fixtures return (tool, response_dict, expected_text) tuples so a parity
test can assert both dispatch shape AND semantic invariants.
"""

from __future__ import annotations

from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
    InputFieldSpec,
)


def clarification_fixture() -> tuple[ConversationTool, dict[str, Any], str]:
    tool = ConversationTool(
        tool_type=ConversationToolType.CLARIFICATION,
        prompt="What's your name?",
        output_vars=["user_name"],
    )
    return tool, {"content": "Alice"}, "Alice"


def single_choice_fixture() -> tuple[ConversationTool, dict[str, Any], str]:
    tool = ConversationTool(
        tool_type=ConversationToolType.SINGLE_CHOICE,
        choices=[
            ChoiceItem(label="Yes", value="yes"),
            ChoiceItem(label="No", value="no"),
        ],
        output_vars=["confirmed"],
    )
    return tool, {"choice_index": 0}, "yes"


def single_choice_composite_fixture() -> tuple[ConversationTool, dict[str, Any], str]:
    """Composite input — choice carries an embedded InputFieldSpec."""
    tool = ConversationTool(
        tool_type=ConversationToolType.SINGLE_CHOICE,
        choices=[
            ChoiceItem(
                label="Auto",
                value="auto",
            ),
            ChoiceItem(
                label="Path",
                value="path",
                input=InputFieldSpec(name="target_path", expected_input_type="path"),
            ),
        ],
        output_vars=["mode"],
    )
    return (
        tool,
        {"choice_index": 1, "inputs": {"target_path": "/tmp/data"}},
        "path",
    )


def multiple_choice_fixture() -> tuple[ConversationTool, dict[str, Any], str]:
    tool = ConversationTool(
        tool_type=ConversationToolType.MULTIPLE_CHOICE,
        choices=[
            ChoiceItem(label="A", value="a"),
            ChoiceItem(label="B", value="b"),
        ],
        output_vars=["selected"],
    )
    return tool, {"content": "a,b"}, "a,b"


def confirmation_fixture() -> tuple[ConversationTool, dict[str, Any], str]:
    tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
    return tool, {"choice": "yes"}, "yes"


def confirmation_with_param_overrides_fixture() -> tuple[
    ConversationTool, dict[str, Any], str
]:
    tool = ConversationTool(tool_type=ConversationToolType.CONFIRMATION)
    return (
        tool,
        {"choice": "yes", "param_overrides": {"x": 1}, "variables": {"v": "hi"}},
        "yes",
    )


def proposal_selection_fixture() -> tuple[ConversationTool, dict[str, Any], str]:
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["selected_proposal_ids"],
    )
    return tool, {"selected_proposals": ["P1", "P2"]}, "P1,P2"


def proposal_selection_with_auto_implement_fixture() -> tuple[
    ConversationTool, dict[str, Any], str
]:
    tool = ConversationTool(
        tool_type=ConversationToolType.PROPOSAL_SELECTION,
        output_vars=["selected_proposal_ids"],
    )
    return (
        tool,
        {"selected_proposals": ["P1"], "auto_implement": True},
        "P1",
    )


def tool_argument_form_fixture() -> tuple[ConversationTool, dict[str, Any], str]:
    tool = ConversationTool(
        tool_type=ConversationToolType.TOOL_ARGUMENT_FORM,
        tool_name="my_tool",
        fields=[{"name": "arg1", "label": "Arg 1"}],
    )
    return tool, {"fields": {"arg1": "value1"}}, "arg1=value1"


ALL_FIXTURES = {
    "clarification": clarification_fixture,
    "single_choice": single_choice_fixture,
    "single_choice_composite": single_choice_composite_fixture,
    "multiple_choice": multiple_choice_fixture,
    "confirmation": confirmation_fixture,
    "confirmation_with_param_overrides": confirmation_with_param_overrides_fixture,
    "proposal_selection": proposal_selection_fixture,
    "proposal_selection_with_auto_implement": proposal_selection_with_auto_implement_fixture,
    "tool_argument_form": tool_argument_form_fixture,
}

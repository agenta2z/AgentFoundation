"""Commit 4: AF-native proposal_selection dispatch.

Covers the inline integration (the live AF path — the handler registry is not
wired in AF): parser carry-through, proposal resolution from ``proposals_path``,
enrichment into choices, ``_build_input_mode`` branch, and interactive
multi-select capture into the output variable.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_foundation.common.data_models.proposal import (
    register_proposal_parser,
)
from agent_foundation.common.data_models.proposal.model import (
    Proposal,
    ProposalGroup,
    ProposalIndex,
)
from agent_foundation.common.data_models.proposal.parser import write_proposal_index
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_response_parser import (
    parse_conversation_response,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
    _build_input_mode,
)
from agent_foundation.ui.input_modes import InputMode


def _sample_index() -> ProposalIndex:
    return ProposalIndex(
        version="1",
        total_count=2,
        groups=[
            ProposalGroup(
                phase=1,
                label="Quick Wins",
                proposals=[
                    Proposal(id="P1", rank=1, title="Add caching",
                             impact="high", complexity="low", summary="cache hot paths"),
                    Proposal(id="P3", rank=3, title="Rewrite auth",
                             impact="medium", complexity="high"),
                ],
            ),
        ],
    )


# --- lightweight carriers binding the real methods under test ----------------


class _Resolver:
    _resolve_proposals_source = ConversationalInferencer._resolve_proposals_source
    _enrich_proposal_selection = ConversationalInferencer._enrich_proposal_selection

    def __init__(self, prior_context=None):
        self.prior_context = prior_context or {}


class _DecodeCI:
    _handle_conversation_tool = ConversationalInferencer._handle_conversation_tool
    set_session_variables = ConversationalInferencer.set_session_variables

    def __init__(self):
        self.prior_context = {}
        self.prompt_renderer = None
        self.interactive = None


class _FakeInteractive:
    def __init__(self, response):
        self._response = response
        self.sent_input_mode = None

    async def asend_response(self, text, flag=None, input_mode=None, prompt_data=None):
        self.sent_input_mode = input_mode

    async def aget_input(self):
        return self._response


# --- parser carry-through ----------------------------------------------------


def test_parser_carries_proposals_path_and_output():
    block = (
        "```json ToolsToInvoke\n"
        + json.dumps(
            {
                "type": "conversation",
                "name": "proposal_selection",
                "arguments": {
                    "prompt": "Pick proposals",
                    "proposals_path": "/ws/outputs/proposals.json",
                    "preselected_ids": "P1",
                },
                "output": ["selected_proposal_ids"],
            }
        )
        + "\n```"
    )
    resp = parse_conversation_response(block)
    assert resp.conversation_tool is not None
    tool = resp.conversation_tool
    assert tool.tool_type == "proposal_selection"
    assert tool.metadata["proposals_path"] == "/ws/outputs/proposals.json"
    assert tool.metadata["preselected_ids"] == "P1"
    assert tool.output_vars == ["selected_proposal_ids"]


# --- resolution + enrichment -------------------------------------------------


def test_resolve_from_proposals_path(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    write_proposal_index(out / "proposals.json", _sample_index())

    tool = ConversationTool(
        tool_type="proposal_selection",
        prompt="Pick",
        metadata={"proposals_path": str(out / "proposals.json")},
    )
    data = _Resolver()._resolve_proposals_source(tool)
    assert data is not None
    assert data["total_count"] == 2
    assert data["groups"][0]["proposals"][0]["id"] == "P1"


def test_resolve_prefers_existing_proposals_dict():
    tool = ConversationTool(
        tool_type="proposal_selection",
        metadata={"proposals": {"total_count": 7, "groups": []}},
    )
    data = _Resolver()._resolve_proposals_source(tool)
    assert data == {"total_count": 7, "groups": []}


def test_resolve_via_registered_parser_fallback():
    class _Parser:
        def parse(self, workspace):
            assert workspace == "/discovered/ws"
            return _sample_index()

    register_proposal_parser(_Parser())
    try:
        tool = ConversationTool(tool_type="proposal_selection", metadata={})
        resolver = _Resolver(prior_context={"workspace_path__research_propose": "/discovered/ws"})
        data = resolver._resolve_proposals_source(tool)
        assert data is not None and data["total_count"] == 2
    finally:
        register_proposal_parser(None)


def test_resolve_returns_none_when_nothing_available():
    tool = ConversationTool(tool_type="proposal_selection", metadata={})
    assert _Resolver()._resolve_proposals_source(tool) is None


def test_enrich_populates_choices_and_output_var(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    write_proposal_index(out / "proposals.json", _sample_index())

    tool = ConversationTool(
        tool_type="proposal_selection",
        prompt="Pick",
        metadata={"proposals_path": str(out / "proposals.json")},
    )
    _Resolver()._enrich_proposal_selection(tool)

    assert "proposals" in tool.metadata
    assert tool.metadata["proposals_count"] == 2
    assert [c.value for c in tool.choices] == ["P1", "P3"]
    assert tool.choices[0].label.startswith("P1: Add caching")
    assert tool.output_vars == ["selected_proposal_ids"]


def test_enrich_no_op_when_unresolvable():
    tool = ConversationTool(tool_type="proposal_selection", metadata={})
    _Resolver()._enrich_proposal_selection(tool)
    assert tool.choices == []
    assert "proposals" not in tool.metadata


# --- build_input_mode --------------------------------------------------------


def test_build_input_mode_proposal_selection():
    tool = ConversationTool(
        tool_type="proposal_selection",
        prompt="Pick proposals",
        choices=[ChoiceItem(label="P1: x", value="P1"),
                 ChoiceItem(label="P3: y", value="P3")],
        metadata={"proposals": {"total_count": 2}, "widget_type": "proposal_selection"},
    )
    mode = _build_input_mode(tool)
    assert mode.mode == InputMode.MULTIPLE_CHOICE
    assert [o.value for o in mode.options] == ["P1", "P3"]
    assert mode.metadata["widget_type"] == "proposal_selection"
    assert mode.metadata["proposals"] == {"total_count": 2}


# --- interactive decode → output variable ------------------------------------


def test_interactive_selection_persists_to_output_var():
    tool = ConversationTool(
        tool_type="proposal_selection",
        prompt="Pick",
        choices=[ChoiceItem(label="P1", value="P1"), ChoiceItem(label="P3", value="P3")],
        output_vars=["selected_proposal_ids"],
    )
    ci = _DecodeCI()
    fake = _FakeInteractive({"user_input": {"selected_proposals": ["P1", "P3"]}})

    result = asyncio.run(ci._handle_conversation_tool(tool, "text", interactive_override=fake))

    assert result == "P1,P3"
    assert ci.prior_context["selected_proposal_ids"] == "P1,P3"


def test_interactive_selection_via_choice_indices():
    tool = ConversationTool(
        tool_type="proposal_selection",
        prompt="Pick",
        choices=[ChoiceItem(label="P1", value="P1"), ChoiceItem(label="P3", value="P3")],
        output_vars=["selected_proposal_ids"],
    )
    ci = _DecodeCI()
    fake = _FakeInteractive({"user_input": {"choice_indices": [1]}})

    result = asyncio.run(ci._handle_conversation_tool(tool, "text", interactive_override=fake))

    assert result == "P3"
    assert ci.prior_context["selected_proposal_ids"] == "P3"

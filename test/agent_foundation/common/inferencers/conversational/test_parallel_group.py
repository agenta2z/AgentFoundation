"""Unit tests for round-lifecycle grouping + display + parallel_group parse.

Covers three pure-function surfaces of the round-lifecycle work (no inferencer
loop here — see test_round_lifecycle.py for the integration coverage):

1. ``group_and_validate`` — partitioning of consecutive runs and the
   GroupValidationError invariants (a/b/c/d).
2. ``display_text`` — stripping tool blocks / thinking / <Response> unwrap.
3. ``parallel_group`` lenient parse round-trips through ConversationTool and the
   ToolsToInvoke parser path.
"""

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    GroupValidationError,
    group_and_validate,
    primary_output_key,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_response_parser import (
    display_text,
    parse_conversation_response,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
    coerce_parallel_group,
)


def _tool(output_var="v", *, tool_type="clarification", parallel_group=None):
    """Build a ConversationTool with a single output var + optional group."""
    return ConversationTool(
        tool_type=tool_type,
        output_vars=[output_var] if output_var else [],
        parallel_group=parallel_group,
    )


# ---------------------------------------------------------------------------
# 1. group_and_validate — partitioning
# ---------------------------------------------------------------------------


class TestGroupPartitioning:
    def test_ungrouped_none_coalesce_into_one_group(self):
        tools = [_tool("a"), _tool("b"), _tool("c")]
        groups = group_and_validate(tools)
        assert len(groups) == 1
        assert groups[0] == tools

    def test_consecutive_same_explicit_group_one_group(self):
        tools = [_tool("a", parallel_group=1), _tool("b", parallel_group=1)]
        groups = group_and_validate(tools)
        assert len(groups) == 1
        assert [t.output_vars[0] for t in groups[0]] == ["a", "b"]

    def test_none_none_one_one_partition(self):
        # [None, None, 1, 1] → two groups: {None,None} then {1,1}.
        tools = [
            _tool("a"),
            _tool("b"),
            _tool("c", parallel_group=1),
            _tool("d", parallel_group=1),
        ]
        groups = group_and_validate(tools)
        assert len(groups) == 2
        assert [t.output_vars[0] for t in groups[0]] == ["a", "b"]
        assert [t.output_vars[0] for t in groups[1]] == ["c", "d"]

    def test_single_tool_one_group(self):
        tools = [_tool("a")]
        groups = group_and_validate(tools)
        assert groups == [tools]

    def test_empty_tools_no_groups(self):
        assert group_and_validate([]) == []


# ---------------------------------------------------------------------------
# 1. group_and_validate — validation invariants
# ---------------------------------------------------------------------------


class TestGroupValidation:
    @pytest.mark.parametrize(
        "side_effecting",
        [ConversationToolType.CONFIRMATION, ConversationToolType.PROPOSAL_SELECTION],
    )
    def test_side_effecting_in_multi_member_group_raises(self, side_effecting):
        tools = [
            _tool("a", tool_type=side_effecting, parallel_group=1),
            _tool("b", parallel_group=1),
        ]
        with pytest.raises(GroupValidationError, match="side-effecting"):
            group_and_validate(tools)

    @pytest.mark.parametrize(
        "side_effecting",
        [ConversationToolType.CONFIRMATION, ConversationToolType.PROPOSAL_SELECTION],
    )
    def test_side_effecting_as_single_member_group_is_allowed(self, side_effecting):
        # A side-effecting tool alone in its group must NOT raise. Use a distinct
        # explicit group so it never coalesces with the neighbour.
        tools = [
            _tool("a", tool_type=side_effecting, parallel_group=1),
        ]
        # Single-member group: no raise.
        groups = group_and_validate(tools)
        assert len(groups) == 1

    def test_duplicate_primary_output_key_in_group_raises(self):
        tools = [_tool("dup", parallel_group=1), _tool("dup", parallel_group=1)]
        with pytest.raises(GroupValidationError, match="duplicate primary output key"):
            group_and_validate(tools)

    def test_duplicate_key_across_separate_groups_is_ok(self):
        # Same key in DIFFERENT groups is fine (b is within-group only). Two
        # ungrouped runs cannot be formed without a grouped tool between them,
        # so use distinct partitions: [None] then [1].
        tools = [_tool("dup"), _tool("dup", parallel_group=1)]
        groups = group_and_validate(tools)
        assert len(groups) == 2

    @pytest.mark.parametrize("reserved", ["values", "user_input"])
    def test_reserved_output_var_in_multi_member_group_raises(self, reserved):
        tools = [
            _tool(reserved, parallel_group=1),
            _tool("other", parallel_group=1),
        ]
        with pytest.raises(GroupValidationError, match="reserved output var"):
            group_and_validate(tools)

    @pytest.mark.parametrize("reserved", ["values", "user_input"])
    def test_reserved_output_var_single_member_group_allowed(self, reserved):
        tools = [_tool(reserved, parallel_group=1)]
        groups = group_and_validate(tools)  # single-member → exempt
        assert len(groups) == 1

    @pytest.mark.parametrize("reserved", ["values", "user_input"])
    def test_reserved_output_var_lone_ungrouped_allowed(self, reserved):
        # A single ungrouped tool forms a single-member group → exempt from (c).
        tools = [_tool(reserved)]
        groups = group_and_validate(tools)
        assert len(groups) == 1

    @pytest.mark.parametrize("reserved", ["values", "user_input"])
    def test_reserved_output_var_in_coalesced_ungrouped_run_raises(self, reserved):
        # NOTE on semantics: two ungrouped (parallel_group=None) tools coalesce
        # into ONE multi-member group, so rule (c) DOES fire here — the
        # direct-map compound payload would key by the reserved var. "Ungrouped"
        # is exempt only when the run is single-member; a coalesced run of >1 is
        # a multi-member group and is validated like any other.
        tools = [_tool(reserved), _tool("other")]
        with pytest.raises(GroupValidationError, match="reserved output var"):
            group_and_validate(tools)

    def test_more_than_one_distinct_nonnone_group_raises(self):
        # Two DISTINCT non-None group ids anywhere in the response → invalid (d).
        tools = [_tool("a", parallel_group=1), _tool("b", parallel_group=2)]
        with pytest.raises(GroupValidationError, match="more than one distinct"):
            group_and_validate(tools)

    def test_one_distinct_group_with_intervening_ungrouped_ok(self):
        # Only ONE distinct non-None id (1) is present → (d) is satisfied even
        # though partitioning yields 3 runs.
        tools = [
            _tool("a", parallel_group=1),
            _tool("b"),
            _tool("c", parallel_group=1),
        ]
        groups = group_and_validate(tools)
        assert [[t.output_vars[0] for t in g] for g in groups] == [["a"], ["b"], ["c"]]


class TestPrimaryOutputKey:
    def test_first_output_var(self):
        assert primary_output_key(_tool("x")) == "x"

    def test_falls_back_to_tool_type_when_no_output_vars(self):
        t = ConversationTool(tool_type="clarification", output_vars=[])
        assert primary_output_key(t) == "clarification"


# ---------------------------------------------------------------------------
# 2. display_text
# ---------------------------------------------------------------------------


class TestDisplayText:
    def test_strips_tools_to_invoke_block_keeps_prose(self):
        raw = (
            "Here is some prose before.\n"
            "```json ToolsToInvoke\n"
            '{"type": "conversation", "name": "clarification", '
            '"arguments": {"prompt": "pick?"}}\n'
            "```\n"
            "And prose after."
        )
        out = display_text(raw)
        assert "Here is some prose before." in out
        assert "And prose after." in out
        assert "ToolsToInvoke" not in out
        assert "clarification" not in out

    def test_unwraps_response_tags(self):
        raw = "<Response>Hello user</Response>"
        assert display_text(raw) == "Hello user"

    def test_strips_thinking_block(self):
        raw = "<thinking>secret reasoning</thinking>Visible answer."
        out = display_text(raw)
        assert out == "Visible answer."
        assert "secret" not in out

    def test_pure_tool_call_returns_empty(self):
        raw = (
            "```json ToolsToInvoke\n"
            '{"type": "conversation", "name": "clarification", '
            '"arguments": {"prompt": "pick?"}}\n'
            "```"
        )
        assert display_text(raw) == ""

    def test_pure_thinking_returns_empty(self):
        raw = "<thinking>only reasoning, no answer</thinking>"
        assert display_text(raw) == ""

    def test_plain_text_passthrough(self):
        raw = "Just plain prose with no markup."
        assert display_text(raw) == raw

    def test_empty_input_returns_empty(self):
        assert display_text("") == ""


# ---------------------------------------------------------------------------
# 3. parallel_group parse (lenient)
# ---------------------------------------------------------------------------


class TestParallelGroupCoerce:
    def test_int_accepted(self):
        assert coerce_parallel_group(3) == 3

    @pytest.mark.parametrize("bad", [True, False, "1", 1.0, None, [1]])
    def test_non_int_collapses_to_none(self, bad):
        assert coerce_parallel_group(bad) is None


class TestConversationToolParallelGroupRoundTrip:
    def test_from_dict_round_trips_int(self):
        t = ConversationTool.from_dict(
            {"tool_type": "clarification", "output": ["v"], "parallel_group": 7}
        )
        assert t.parallel_group == 7
        # to_dict emits it back out.
        assert t.to_dict()["parallel_group"] == 7

    @pytest.mark.parametrize("bad", [True, "x", 2.5])
    def test_from_dict_bad_value_records_metadata_and_none(self, bad):
        t = ConversationTool.from_dict(
            {"tool_type": "clarification", "output": ["v"], "parallel_group": bad}
        )
        assert t.parallel_group is None
        assert t.metadata.get("parallel_group_invalid") == bad

    def test_to_dict_omits_when_none(self):
        t = ConversationTool(tool_type="clarification", parallel_group=None)
        assert "parallel_group" not in t.to_dict()


class TestToolsToInvokeParallelGroupParse:
    def test_top_level_int_parsed(self):
        raw = (
            "```json ToolsToInvoke\n"
            '{"type": "conversation", "name": "clarification", '
            '"arguments": {"prompt": "q1"}, "output": ["a"], "parallel_group": 2}\n'
            "```"
        )
        resp = parse_conversation_response(raw)
        assert len(resp.conversation_tools) == 1
        assert resp.conversation_tools[0].parallel_group == 2

    def test_inside_arguments_parsed(self):
        raw = (
            "```json ToolsToInvoke\n"
            '{"type": "conversation", "name": "clarification", '
            '"arguments": {"prompt": "q1", "parallel_group": 5}, "output": ["a"]}\n'
            "```"
        )
        resp = parse_conversation_response(raw)
        assert resp.conversation_tools[0].parallel_group == 5

    def test_bad_value_collapses_to_none_with_metadata(self):
        raw = (
            "```json ToolsToInvoke\n"
            '{"type": "conversation", "name": "clarification", '
            '"arguments": {"prompt": "q1"}, "output": ["a"], "parallel_group": "oops"}\n'
            "```"
        )
        resp = parse_conversation_response(raw)
        tool = resp.conversation_tools[0]
        assert tool.parallel_group is None
        assert tool.metadata.get("parallel_group_invalid") == "oops"

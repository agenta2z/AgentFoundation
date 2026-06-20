"""Unit tests for typed/composite conversation-tool inputs.

Locks the novel logic added for path/composite inputs:
  * canonicalisation (hyphen keys, tool-name dialect, string `output`)
  * InputFieldSpec / ChoiceItem.input schema round-trip
  * finalize_input_value serialisation + path re-join + containment
  * decode_tool_bindings distinct bindings (composite, proposal, multi-choice)
"""

import json

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ChoiceItem,
    ConversationTool,
    InputFieldSpec,
    canonicalize_tool_data,
    normalize_tool_type,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    decode_tool_bindings,
    finalize_input_value,
    is_contained,
    render_templated_fields,
)


# --- canonicalisation -------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("single-choice", "single_choice"),
    ("single choice", "single_choice"),
    ("single_choices", "single_choice"),
    ("multiple-choice", "multiple_choice"),
    ("clarification", "clarification"),
])
def test_normalize_tool_type(name, expected):
    assert normalize_tool_type(name) == expected


def test_canonicalize_hyphen_keys_and_output_coercion():
    data = canonicalize_tool_data({
        "name": "single-choice",
        "arguments": {
            "expected-input-type": "path",
            "allow-multiple-input": True,
            "choices": [{"label": "x", "value": "x", "input": {"expected-input-type": "path"}}],
        },
        "output": "var_x",
    })
    args = data["arguments"]
    assert args["expected_input_type"] == "path"
    assert args["allow_multiple_input"] is True
    assert args["choices"][0]["input"]["expected_input_type"] == "path"
    assert data["output"] == ["var_x"]  # string coerced to list


def test_underscore_wins_over_hyphen():
    d = canonicalize_tool_data({"arguments": {"expected-input-type": "path", "expected_input_type": "url"}})
    assert d["arguments"]["expected_input_type"] == "url"


# --- schema round-trip ------------------------------------------------------


def test_choiceitem_label_only_backward_compat():
    c = ChoiceItem.from_dict({"label": "A", "value": "a", "description": "d"})
    assert not c.has_input
    assert c.to_dict() == {"label": "A", "value": "a", "description": "d"}


def test_choiceitem_with_input_round_trip():
    c = ChoiceItem("Manual", "manual", input=InputFieldSpec(
        name="p", expected_input_type="path", allow_multiple_input=True, serialization="json"))
    assert c.has_input
    assert ChoiceItem.from_dict(c.to_dict()).to_dict() == c.to_dict()


def test_conversation_tool_round_trip_with_typed_fields():
    t = ConversationTool(
        tool_type="single_choice", prompt="p", allow_custom=False,
        choices=[ChoiceItem("Auto", "auto"), ChoiceItem("Manual", "manual",
                 input=InputFieldSpec(name="p", expected_input_type="path", allow_multiple_input=True))],
        output_vars=["mode"])
    assert ConversationTool.from_dict(t.to_dict()).to_dict() == t.to_dict()


# --- finalize_input_value ---------------------------------------------------


def test_finalize_single_path_rejoin():
    assert finalize_input_value("data/x", expected_input_type="path", prefix="/root") == "/root/data/x"


def test_finalize_absolute_and_tilde_passthrough():
    assert finalize_input_value("/abs", expected_input_type="path", prefix="/root") == "/abs"
    assert finalize_input_value("~/h", expected_input_type="path", prefix="/root") == "~/h"


def test_finalize_multi_default_is_json_array_string():
    out = finalize_input_value(["a", "b"], expected_input_type="path", prefix="/root", allow_multiple_input=True)
    assert json.loads(out) == ["/root/a", "/root/b"]
    assert out != "['/root/a', '/root/b']"  # never str(list)


def test_finalize_multi_comma_serialization():
    assert finalize_input_value(["a", "b"], allow_multiple_input=True, serialization="comma") == "a,b"


def test_finalize_unwraps_content_envelope():
    assert finalize_input_value({"content": "hi"}) == "hi"


def test_finalize_traversal_rejected_and_in_root_ok():
    with pytest.raises(ValueError):
        finalize_input_value("../../out", expected_input_type="path", prefix="/root/sub",
                             session_root="/root", validate=True)
    assert finalize_input_value("data", expected_input_type="path", prefix="/root",
                                session_root="/root", validate=True) == "/root/data"


def test_is_contained_blocks_sibling_prefix():
    assert is_contained("/tmp/root/x", "/tmp/root") is True
    assert is_contained("/tmp/root2/x", "/tmp/root") is False


# --- decode_tool_bindings ---------------------------------------------------


def _composite_tool():
    return ConversationTool(tool_type="single_choice", output_vars=["mode"], choices=[
        ChoiceItem("Auto", "auto_discover"),
        ChoiceItem("Manual", "manual_paths", input=InputFieldSpec(
            name="paths", expected_input_type="path", allow_multiple_input=True, prefix="/root", serialization="json")),
    ])


def test_decode_composite_binds_two_vars():
    b = decode_tool_bindings(_composite_tool(), {"choice_index": 1, "inputs": {"paths": ["a", "b"]}})
    assert b["mode"] == "manual_paths"
    assert json.loads(b["paths"]) == ["/root/a", "/root/b"]


def test_decode_auto_choice_binds_mode_only():
    assert decode_tool_bindings(_composite_tool(), {"choice_index": 0}) == {"mode": "auto_discover"}


def test_decode_proposal_selection_comma_join():
    t = ConversationTool(tool_type="proposal_selection", output_vars=["ids"])
    assert decode_tool_bindings(t, {"selected_proposals": ["P1", "P3"]}) == {"ids": "P1,P3"}


def test_decode_multiple_choice_selections_comma_join():
    t = ConversationTool(tool_type="multiple_choice", output_vars=["picks"],
                         choices=[ChoiceItem("A", "a"), ChoiceItem("B", "b"), ChoiceItem("C", "c")])
    b = decode_tool_bindings(t, {"selections": [{"choice_index": 0}, {"choice_index": 2}]})
    assert b == {"picks": "a,c"}


def test_decode_legacy_aliasing_preserved_for_multi_output_vars():
    # A non-composite clarification with two output vars binds the SAME value to both.
    t = ConversationTool(tool_type="clarification", output_vars=["v1", "v2"])
    assert decode_tool_bindings(t, {"content": "hello"}) == {"v1": "hello", "v2": "hello"}


def test_decode_compound_skips_untouched_tool_no_clobber():
    """An untouched tool (its key absent from values) must NOT bind "" — it
    would clobber prior/default state via set_session_variables."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
        decode_compound_bindings,
    )
    t1 = ConversationTool(tool_type="clarification", output_vars=["a"])
    t2 = ConversationTool(tool_type="clarification", output_vars=["b"])
    # Only t1's key present; t2 untouched.
    out = decode_compound_bindings([t1, t2], {"a": {"content": "x"}})
    assert out == {"a": "x"}
    assert "b" not in out  # not clobbered with ""
    # present-but-empty still binds (user explicitly cleared).
    out2 = decode_compound_bindings([t1, t2], {"a": {"content": "x"}, "b": {"content": ""}})
    assert out2 == {"a": "x", "b": ""}


# --- render_templated_fields ------------------------------------------------


def test_render_resolves_templated_prefix():
    t = ConversationTool(tool_type="clarification", expected_input_type="path",
                         prefix="{{ session_root_path }}",
                         choices=[ChoiceItem("M", "m", input=InputFieldSpec(name="p", prefix="{{ session_root_path }}"))])
    render_templated_fields(t, lambda s: s.replace("{{ session_root_path }}", "/resolved"))
    assert t.prefix == "/resolved"
    assert t.choices[0].input.prefix == "/resolved"


def test_render_none_is_noop():
    t = ConversationTool(tool_type="clarification", prefix="{{ x }}")
    render_templated_fields(t, None)
    assert t.prefix == "{{ x }}"  # left untouched when no renderer

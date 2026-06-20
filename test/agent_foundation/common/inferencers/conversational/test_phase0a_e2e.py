"""End-to-end lock for model_optimization Phase 0a typed/composite inputs.

Exercises the LIVE inline path (parser -> _build_input_mode -> runtime decode),
NOT the dormant handler registry, so a regression in any of the four layers
(parse / schema / build / decode) fails here.

Covers the verbatim Phase 0a wire shapes the LLM emits:
  * clarification with hyphenated `expected-input-type` + `prefix` (path autocomplete)
  * single-choice (hyphenated name) with a composite "Auto discover" + embedded
    multi-path input, and string `output` (coerced to a list)
and the two-variable binding + path finalisation on submit.
"""

import json

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_response_parser import (
    _tool_invocation_to_conversation_tool,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational import (
    conversational_inferencer as ci_mod,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    decode_tool_bindings,
    decode_compound_bindings,
)


SESSION_ROOT = "/session_root_abc"


# Verbatim Phase 0a emissions (hyphenated author dialect, string output).
CLARIFICATION = {
    "type": "conversation",
    "name": "clarification",
    "arguments": {
        "prompt": "Choose the workflow target path.",
        "expected-input-type": "path",
        "prefix": SESSION_ROOT,
    },
    "output": "workflow_target_path",
}

SINGLE_CHOICE = {
    "type": "conversation",
    "name": "single-choice",
    "arguments": {
        "prompt": "Where are the modeling artifacts?",
        "choices": [
            {"label": "Auto discover", "value": "auto_discover", "description": "infer"},
            {
                "label": "Specify paths",
                "value": "manual_paths",
                "input": {
                    "name": "workflow_modeling_artifacts_path",
                    "expected-input-type": "path",
                    "allow-multiple-input": True,
                    "prefix": SESSION_ROOT,
                    "required": True,
                },
            },
        ],
        "allow_custom": False,
    },
    "output": "workflow_modeling_artifacts_mode",
}


def test_phase0a_clarification_parses():
    tool = _tool_invocation_to_conversation_tool(CLARIFICATION)
    assert tool.tool_type == "clarification"
    assert tool.expected_input_type == "path"            # hyphen canonicalised
    assert tool.prefix == SESSION_ROOT
    assert tool.output_vars == ["workflow_target_path"]  # string -> list


def test_phase0a_single_choice_composite_parses():
    tool = _tool_invocation_to_conversation_tool(SINGLE_CHOICE)
    assert tool.tool_type == "single_choice"             # "single-choice" normalised
    assert tool.output_vars == ["workflow_modeling_artifacts_mode"]
    c0, c1 = tool.choices
    assert not c0.has_input and c1.has_input
    assert c1.input.name == "workflow_modeling_artifacts_path"
    assert c1.input.expected_input_type == "path"        # hyphen canonicalised
    assert c1.input.allow_multiple_input is True          # hyphen canonicalised
    assert c1.input.prefix == SESSION_ROOT


def test_phase0a_clarification_build_input_mode_is_path_widget():
    tool = _tool_invocation_to_conversation_tool(CLARIFICATION)
    cfg = ci_mod._build_input_mode(tool).to_dict()
    assert cfg.get("expected_input_type") == "path"
    assert cfg.get("prefix") == SESSION_ROOT
    assert cfg["metadata"]["widget_type"] == "path_input"


def test_phase0a_single_choice_build_input_mode_carries_option_input():
    tool = _tool_invocation_to_conversation_tool(SINGLE_CHOICE)
    cfg = ci_mod._build_input_mode(tool).to_dict()
    opts = cfg["options"]
    assert opts[0] == {"label": "Auto discover", "value": "auto_discover", "description": "infer"}
    assert opts[1]["input"]["name"] == "workflow_modeling_artifacts_path"
    assert opts[1]["input"]["expected_input_type"] == "path"
    assert opts[1]["input"]["allow_multiple_input"] is True


def test_phase0a_manual_response_binds_both_vars():
    tool = _tool_invocation_to_conversation_tool(SINGLE_CHOICE)
    response = {"choice_index": 1, "inputs": {
        "workflow_modeling_artifacts_path": ["data/features", "experiments/run_42"]
    }}
    bindings = decode_tool_bindings(tool, response, session_root=SESSION_ROOT)
    assert bindings["workflow_modeling_artifacts_mode"] == "manual_paths"
    # multi path -> JSON array string (reversible), prefix re-joined, never str(list)
    paths = json.loads(bindings["workflow_modeling_artifacts_path"])
    assert paths == [f"{SESSION_ROOT}/data/features", f"{SESSION_ROOT}/experiments/run_42"]


def test_phase0a_auto_response_binds_mode_only():
    tool = _tool_invocation_to_conversation_tool(SINGLE_CHOICE)
    bindings = decode_tool_bindings(tool, {"choice_index": 0}, session_root=SESSION_ROOT)
    assert bindings == {"workflow_modeling_artifacts_mode": "auto_discover"}
    assert "workflow_modeling_artifacts_path" not in bindings  # no stale path


def test_phase0a_compound_two_tools_one_turn():
    """Phase 0a presents clarification + single_choice as ONE compound turn."""
    clar = _tool_invocation_to_conversation_tool(CLARIFICATION)
    sc = _tool_invocation_to_conversation_tool(SINGLE_CHOICE)
    # Compound widget keys each child payload under the tool's primary output var.
    values = {
        "workflow_target_path": {"content": "models/ranking"},
        "workflow_modeling_artifacts_mode": {
            "choice_index": 1,
            "inputs": {"workflow_modeling_artifacts_path": ["data/x"]},
        },
    }
    bindings = decode_compound_bindings([clar, sc], values, session_root=SESSION_ROOT)
    assert bindings["workflow_target_path"] == f"{SESSION_ROOT}/models/ranking"
    assert bindings["workflow_modeling_artifacts_mode"] == "manual_paths"
    assert json.loads(bindings["workflow_modeling_artifacts_path"]) == [f"{SESSION_ROOT}/data/x"]
    # The mode-var child payload is NOT stringified into the mode variable.
    assert "{'choice_index'" not in bindings["workflow_modeling_artifacts_mode"]

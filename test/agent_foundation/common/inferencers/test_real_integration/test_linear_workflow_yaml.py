"""LinearWorkflowInferencer YAML composition tests — declarative config loading and instantiation.

Validates Requirement 33 from the integration test spec.
Tests load YAML templates, replace placeholders with temp paths, instantiate
objects via ``rich_python_utils.config_utils``, and verify the resulting object tree.
"""

import os
import tempfile
from pathlib import Path

import pytest

from rich_python_utils.config_utils import instantiate, load_config

# Importing configs triggers alias registration (ClaudeCodeCLI, Dual, ConsensusConfig, etc.)
import agent_foundation.common.configs  # noqa: F401

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)

from .conftest import DEFAULT_TIMEOUT, skip_claude

# ---------------------------------------------------------------------------
# Path to YAML templates
# ---------------------------------------------------------------------------
_YAML_DIR = Path(__file__).resolve().parent / "yaml_configs"

CHEAP_PROMPT = "What is 2+2?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_yaml_with_placeholders(yaml_name: str, replacements: dict):
    """Read a YAML template, replace ``{placeholder}`` tokens, write to a temp
    file, and load via ``load_config``.

    Path values are normalized to forward slashes so that Windows backslash
    paths don't break YAML double-quoted string parsing (``\\U`` etc. are
    interpreted as Unicode escapes by the YAML scanner).

    Returns the loaded OmegaConf config.
    """
    src = _YAML_DIR / yaml_name
    raw = src.read_text()
    for key, value in replacements.items():
        safe_value = value.replace("\\", "/")
        raw = raw.replace(f"{{{key}}}", safe_value)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        return load_config(tmp_path)
    finally:
        os.unlink(tmp_path)


# ===========================================================================
# Test 1: YAML instantiation (Req 33.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
def test_linear_workflow_yaml_instantiation(tmp_workspace):
    """Load linear_puzzle_3step.yaml → instantiate → verify LinearWorkflowInferencer
    with 3 WorkflowStepConfig entries, each containing a ClaudeCodeCliInferencer.

    **Validates: Requirement 33.1**
    """
    cfg = _load_yaml_with_placeholders(
        "linear_puzzle_3step.yaml",
        {"workspace_path": str(tmp_workspace["workspace"])},
    )
    obj = instantiate(cfg)

    assert isinstance(obj, LinearWorkflowInferencer), (
        f"Expected LinearWorkflowInferencer, got {type(obj).__name__}"
    )
    assert len(obj.step_configs) == 3, (
        f"Expected 3 step configs, got {len(obj.step_configs)}"
    )

    for i, sc in enumerate(obj.step_configs):
        assert isinstance(sc, WorkflowStepConfig), (
            f"Step {i} should be WorkflowStepConfig, got {type(sc).__name__}"
        )
        assert isinstance(sc.inferencer, ClaudeCodeCliInferencer), (
            f"Step {i} inferencer should be ClaudeCodeCliInferencer, "
            f"got {type(sc.inferencer).__name__}"
        )


# ===========================================================================
# Test 2: YAML attribute verification (Req 33.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
def test_linear_workflow_yaml_attribute_verification(tmp_workspace):
    """Verify step names, output_state_keys, inferencer types and model_names
    on each step loaded from linear_puzzle_3step.yaml.

    **Validates: Requirement 33.1**
    """
    ws = str(tmp_workspace["workspace"])
    cfg = _load_yaml_with_placeholders(
        "linear_puzzle_3step.yaml",
        {"workspace_path": ws},
    )
    obj = instantiate(cfg)

    expected_steps = [
        {"name": "plan", "output_state_key": "plan_output"},
        {"name": "implement", "output_state_key": "implement_output"},
        {"name": "review", "output_state_key": "review_output"},
    ]

    for i, expected in enumerate(expected_steps):
        sc = obj.step_configs[i]
        assert sc.name == expected["name"], (
            f"Step {i} name: expected {expected['name']!r}, got {sc.name!r}"
        )
        assert sc.output_state_key == expected["output_state_key"], (
            f"Step {i} output_state_key: expected {expected['output_state_key']!r}, "
            f"got {sc.output_state_key!r}"
        )
        assert isinstance(sc.inferencer, ClaudeCodeCliInferencer), (
            f"Step {i} inferencer should be ClaudeCodeCliInferencer"
        )
        assert sc.inferencer.model_name == "sonnet", (
            f"Step {i} model_name: expected 'sonnet', got {sc.inferencer.model_name!r}"
        )
        assert Path(sc.inferencer.target_path) == Path(ws), (
            f"Step {i} target_path mismatch"
        )
        assert sc.inferencer.permission_mode == "bypassPermissions", (
            f"Step {i} permission_mode: expected 'bypassPermissions', "
            f"got {sc.inferencer.permission_mode!r}"
        )

    # Verify workspace_path on the LinearWorkflowInferencer itself
    assert Path(obj.workspace_path) == Path(ws), (
        f"workspace_path mismatch: expected {ws}, got {obj.workspace_path}"
    )


# ===========================================================================
# Test 3: YAML real inference call (Req 33.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_linear_workflow_yaml_real_inference_call(tmp_workspace):
    """Load linear_puzzle_3step.yaml → instantiate → set callable fields
    programmatically → run a real inference call → verify result contains
    all 3 step outputs.

    Note: LinearWorkflowInferencer's step_configs contain callables
    (input_builder, output_extractor, etc.) that can't be expressed in YAML.
    The callables are set programmatically after instantiation.

    **Validates: Requirement 33.2**
    """
    ws = str(tmp_workspace["workspace"])
    cfg = _load_yaml_with_placeholders(
        "linear_puzzle_3step.yaml",
        {"workspace_path": ws},
    )
    obj = instantiate(cfg)

    # Set input_builder callables programmatically (not expressible in YAML)
    # Step 0 (plan): uses default (original_input)
    # Step 1 (implement): uses plan_output
    obj.step_configs[1].input_builder = lambda state: (
        f"Implement based on this plan:\n{state.get('plan_output', '')}\n\n"
        f"Original task:\n{state['original_input']}"
    )
    # Step 2 (review): uses implement_output
    obj.step_configs[2].input_builder = lambda state: (
        f"Review this implementation:\n{state.get('implement_output', '')}"
    )

    # Set response_builder programmatically
    obj.response_builder = lambda state: state

    result = await obj.ainfer(CHEAP_PROMPT)

    # Verify result is a state dict with all 3 outputs
    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    assert "plan_output" in result, "plan_output should be in state"
    assert "implement_output" in result, "implement_output should be in state"
    assert "review_output" in result, "review_output should be in state"

    # All outputs should be non-empty strings from real CLI calls
    for key in ("plan_output", "implement_output", "review_output"):
        val = result[key]
        assert val is not None and str(val).strip() != "", (
            f"{key} should contain real CLI output"
        )

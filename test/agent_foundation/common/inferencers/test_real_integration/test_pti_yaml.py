"""PTI YAML composition tests — declarative config loading and instantiation.

Validates Requirement 17 from the integration test spec.
Tests load the ``pti_nested_dual.yaml`` template, replace placeholders with temp
paths, instantiate objects via ``rich_python_utils.config_utils``, and verify the
resulting three-level object tree (PTI → DualInferencer → ClaudeCodeCliInferencer).
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
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
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
        # Normalize Windows backslashes → forward slashes for YAML safety
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
# Test 1: PTI YAML instantiation (Req 17.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
def test_pti_yaml_instantiation(tmp_workspace):
    """Load pti_nested_dual.yaml → instantiate → verify PlanThenImplementInferencer
    with DualInferencer children containing ClaudeCodeCliInferencer grandchildren.

    **Validates: Requirement 17.1**
    """
    cfg = _load_yaml_with_placeholders(
        "pti_nested_dual.yaml",
        {"workspace_path": str(tmp_workspace["workspace"])},
    )
    obj = instantiate(cfg)

    # Top level: PlanThenImplementInferencer
    assert isinstance(obj, PlanThenImplementInferencer), (
        f"Expected PlanThenImplementInferencer, got {type(obj).__name__}"
    )

    # Second level: DualInferencer children
    assert isinstance(obj.planner_inferencer, DualInferencer), (
        f"planner_inferencer should be DualInferencer, got {type(obj.planner_inferencer).__name__}"
    )
    assert isinstance(obj.executor_inferencer, DualInferencer), (
        f"executor_inferencer should be DualInferencer, got {type(obj.executor_inferencer).__name__}"
    )

    # Third level: ClaudeCodeCliInferencer grandchildren (planner)
    assert isinstance(obj.planner_inferencer.base_inferencer, ClaudeCodeCliInferencer), (
        f"planner base_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.planner_inferencer.base_inferencer).__name__}"
    )
    assert isinstance(obj.planner_inferencer.review_inferencer, ClaudeCodeCliInferencer), (
        f"planner review_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.planner_inferencer.review_inferencer).__name__}"
    )

    # Third level: ClaudeCodeCliInferencer grandchildren (executor)
    assert isinstance(obj.executor_inferencer.base_inferencer, ClaudeCodeCliInferencer), (
        f"executor base_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.executor_inferencer.base_inferencer).__name__}"
    )
    assert isinstance(obj.executor_inferencer.review_inferencer, ClaudeCodeCliInferencer), (
        f"executor review_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.executor_inferencer.review_inferencer).__name__}"
    )


# ===========================================================================
# Test 2: PTI YAML attribute verification (Req 17.1, 17.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
def test_pti_yaml_attribute_verification(tmp_workspace):
    """Verify model_name, target_path on nested children, consensus_config on
    DualInferencers, and workspace_path on PTI.

    **Validates: Requirements 17.1, 17.2**
    """
    ws = str(tmp_workspace["workspace"])
    cfg = _load_yaml_with_placeholders(
        "pti_nested_dual.yaml",
        {"workspace_path": ws},
    )
    obj = instantiate(cfg)

    # PTI workspace_path
    assert Path(obj.workspace_path) == Path(ws), (
        f"PTI workspace_path mismatch: {obj.workspace_path} != {ws}"
    )

    # Planner DualInferencer — consensus_config
    assert obj.planner_inferencer.consensus_config.max_iterations == 2, (
        "planner consensus_config.max_iterations should be 2"
    )

    # Executor DualInferencer — consensus_config
    assert obj.executor_inferencer.consensus_config.max_iterations == 2, (
        "executor consensus_config.max_iterations should be 2"
    )

    # Planner base_inferencer attributes
    assert obj.planner_inferencer.base_inferencer.model_name == "sonnet"
    assert Path(obj.planner_inferencer.base_inferencer.target_path) == Path(ws)
    assert obj.planner_inferencer.base_inferencer.permission_mode == "bypassPermissions"

    # Planner review_inferencer attributes
    assert obj.planner_inferencer.review_inferencer.model_name == "sonnet"
    assert Path(obj.planner_inferencer.review_inferencer.target_path) == Path(ws)
    assert obj.planner_inferencer.review_inferencer.permission_mode == "bypassPermissions"

    # Executor base_inferencer attributes
    assert obj.executor_inferencer.base_inferencer.model_name == "sonnet"
    assert Path(obj.executor_inferencer.base_inferencer.target_path) == Path(ws)
    assert obj.executor_inferencer.base_inferencer.permission_mode == "bypassPermissions"

    # Executor review_inferencer attributes
    assert obj.executor_inferencer.review_inferencer.model_name == "sonnet"
    assert Path(obj.executor_inferencer.review_inferencer.target_path) == Path(ws)
    assert obj.executor_inferencer.review_inferencer.permission_mode == "bypassPermissions"


# ===========================================================================
# Test 3: PTI YAML real inference call (Req 17.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_pti_yaml_real_inference_call(tmp_workspace):
    """Load pti_nested_dual.yaml → instantiate → run a real inference call with
    a cheap prompt → verify PlanThenImplementResponse.

    **Validates: Requirement 17.1**
    """
    ws = str(tmp_workspace["workspace"])
    cfg = _load_yaml_with_placeholders(
        "pti_nested_dual.yaml",
        {"workspace_path": ws},
    )
    obj = instantiate(cfg)

    result = await obj.ainfer(CHEAP_PROMPT)

    # Verify response type
    assert isinstance(result, PlanThenImplementResponse), (
        f"Expected PlanThenImplementResponse, got {type(result).__name__}"
    )

    # Plan output should be populated
    assert result.plan_output is not None and result.plan_output != "", (
        "plan_output should be populated after planning phase"
    )

    # Executor output should be populated
    assert result.executor_output is not None, (
        "executor_output should be populated after implementation phase"
    )

    # base_response should contain meaningful content
    assert result.base_response is not None and result.base_response != "", (
        "base_response should contain the final output"
    )

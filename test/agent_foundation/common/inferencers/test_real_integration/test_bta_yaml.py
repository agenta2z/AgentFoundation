"""BTA YAML composition tests — declarative config loading and instantiation.

Validates Requirement 23 from the integration test spec.
Tests load BTA YAML templates, replace placeholders with temp paths, instantiate
objects via ``rich_python_utils.config_utils``, and verify the resulting object tree.

Note: ``worker_factory`` is a callable and cannot be expressed in YAML. The YAML
configs define ``breakdown_inferencer`` and ``aggregator_inferencer`` only; the
``worker_factory`` must be set programmatically after instantiation.
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
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)

from .conftest import DEFAULT_TIMEOUT, skip_claude

# ---------------------------------------------------------------------------
# Path to YAML templates
# ---------------------------------------------------------------------------
_YAML_DIR = Path(__file__).resolve().parent / "yaml_configs"


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
# Test 1: BTA YAML instantiation (Req 23.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
def test_bta_yaml_instantiation(tmp_workspace):
    """Load bta_streaming_workers.yaml → instantiate → verify BTA with correct
    breakdown and aggregator types.

    Note: ``worker_factory`` is a callable and must be set programmatically
    after YAML instantiation.

    **Validates: Requirement 23.1**
    """
    cfg = _load_yaml_with_placeholders(
        "bta_streaming_workers.yaml",
        {"workspace_path": str(tmp_workspace["workspace"])},
    )
    obj = instantiate(cfg)

    # Top level: BreakdownThenAggregateInferencer
    assert isinstance(obj, BreakdownThenAggregateInferencer), (
        f"Expected BreakdownThenAggregateInferencer, got {type(obj).__name__}"
    )

    # Breakdown inferencer is ClaudeCodeCliInferencer
    assert isinstance(obj.breakdown_inferencer, ClaudeCodeCliInferencer), (
        f"breakdown_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.breakdown_inferencer).__name__}"
    )

    # Aggregator inferencer is ClaudeCodeCliInferencer
    assert isinstance(obj.aggregator_inferencer, ClaudeCodeCliInferencer), (
        f"aggregator_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.aggregator_inferencer).__name__}"
    )

    # worker_factory is not set via YAML (it's a callable)
    assert obj.worker_factory is None, (
        "worker_factory should be None from YAML — must be set programmatically"
    )

    # max_breakdown from YAML
    assert obj.max_breakdown == 2, (
        f"max_breakdown should be 2, got {obj.max_breakdown}"
    )


# ===========================================================================
# Test 2: BTA YAML attribute verification (Req 23.2, 23.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
def test_bta_yaml_attribute_verification(tmp_workspace):
    """Verify model_name, target_path on breakdown and aggregator children.

    **Validates: Requirements 23.2, 23.3**
    """
    ws = str(tmp_workspace["workspace"])
    cfg = _load_yaml_with_placeholders(
        "bta_streaming_workers.yaml",
        {"workspace_path": ws},
    )
    obj = instantiate(cfg)

    # Breakdown inferencer attributes
    assert obj.breakdown_inferencer.model_name == "sonnet", (
        f"breakdown model_name should be 'sonnet', got '{obj.breakdown_inferencer.model_name}'"
    )
    assert Path(obj.breakdown_inferencer.target_path) == Path(ws), (
        f"breakdown target_path mismatch: {obj.breakdown_inferencer.target_path} != {ws}"
    )
    assert obj.breakdown_inferencer.permission_mode == "bypassPermissions", (
        f"breakdown permission_mode should be 'bypassPermissions', "
        f"got '{obj.breakdown_inferencer.permission_mode}'"
    )

    # Aggregator inferencer attributes
    assert obj.aggregator_inferencer.model_name == "sonnet", (
        f"aggregator model_name should be 'sonnet', got '{obj.aggregator_inferencer.model_name}'"
    )
    assert Path(obj.aggregator_inferencer.target_path) == Path(ws), (
        f"aggregator target_path mismatch: {obj.aggregator_inferencer.target_path} != {ws}"
    )
    assert obj.aggregator_inferencer.permission_mode == "bypassPermissions", (
        f"aggregator permission_mode should be 'bypassPermissions', "
        f"got '{obj.aggregator_inferencer.permission_mode}'"
    )

    # max_breakdown
    assert obj.max_breakdown == 2, (
        f"max_breakdown should be 2, got {obj.max_breakdown}"
    )


# ===========================================================================
# Test 3: PTI with BTA executor YAML instantiation (Req 23.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
def test_pti_with_bta_executor_yaml_instantiation(tmp_workspace):
    """Load pti_with_bta_executor.yaml → instantiate → verify PTI with
    DualInferencer planner and BTA executor.

    **Validates: Requirement 23.1**
    """
    cfg = _load_yaml_with_placeholders(
        "pti_with_bta_executor.yaml",
        {"workspace_path": str(tmp_workspace["workspace"])},
    )
    obj = instantiate(cfg)

    # Top level: PlanThenImplementInferencer
    assert isinstance(obj, PlanThenImplementInferencer), (
        f"Expected PlanThenImplementInferencer, got {type(obj).__name__}"
    )

    # Planner is DualInferencer
    assert isinstance(obj.planner_inferencer, DualInferencer), (
        f"planner_inferencer should be DualInferencer, "
        f"got {type(obj.planner_inferencer).__name__}"
    )

    # Planner children are ClaudeCodeCliInferencer
    assert isinstance(obj.planner_inferencer.base_inferencer, ClaudeCodeCliInferencer), (
        f"planner base_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.planner_inferencer.base_inferencer).__name__}"
    )
    assert isinstance(obj.planner_inferencer.review_inferencer, ClaudeCodeCliInferencer), (
        f"planner review_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.planner_inferencer.review_inferencer).__name__}"
    )

    # Planner consensus config
    assert obj.planner_inferencer.consensus_config.max_iterations == 2, (
        "planner consensus_config.max_iterations should be 2"
    )

    # Executor is BreakdownThenAggregateInferencer
    assert isinstance(obj.executor_inferencer, BreakdownThenAggregateInferencer), (
        f"executor_inferencer should be BreakdownThenAggregateInferencer, "
        f"got {type(obj.executor_inferencer).__name__}"
    )

    # Executor's breakdown inferencer is ClaudeCodeCliInferencer
    assert isinstance(obj.executor_inferencer.breakdown_inferencer, ClaudeCodeCliInferencer), (
        f"executor breakdown_inferencer should be ClaudeCodeCliInferencer, "
        f"got {type(obj.executor_inferencer.breakdown_inferencer).__name__}"
    )

    # Executor's max_breakdown
    assert obj.executor_inferencer.max_breakdown == 2, (
        f"executor max_breakdown should be 2, got {obj.executor_inferencer.max_breakdown}"
    )

    # PTI workspace_path
    assert Path(obj.workspace_path) == Path(str(tmp_workspace["workspace"])), (
        f"PTI workspace_path mismatch: {obj.workspace_path}"
    )

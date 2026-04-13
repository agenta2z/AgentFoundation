"""DualInferencer YAML composition tests — declarative config loading and instantiation.

Validates Requirements 6 and 7 from the integration test spec.
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
from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer import (
    KiroCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)

from .conftest import DEFAULT_TIMEOUT, skip_both, skip_claude

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
# Test 1: YAML instantiation (Req 6.1)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_both
def test_yaml_instantiation(tmp_workspace):
    """Load dual_claude_kiro.yaml → instantiate → verify DualInferencer with correct children."""
    cfg = _load_yaml_with_placeholders(
        "dual_claude_kiro.yaml",
        {"workspace_path": str(tmp_workspace["workspace"])},
    )
    obj = instantiate(cfg)

    assert isinstance(obj, DualInferencer)
    assert isinstance(obj.base_inferencer, ClaudeCodeCliInferencer)
    assert isinstance(obj.review_inferencer, KiroCliInferencer)


# ===========================================================================
# Test 2: YAML attribute verification (Req 6.2, 6.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_both
def test_yaml_attribute_verification(tmp_workspace):
    """Verify model_name, target_path, permission_mode, trust_mode on children."""
    ws = str(tmp_workspace["workspace"])
    cfg = _load_yaml_with_placeholders(
        "dual_claude_kiro.yaml",
        {"workspace_path": ws},
    )
    obj = instantiate(cfg)

    # Base inferencer (ClaudeCodeCLI)
    assert obj.base_inferencer.model_name == "sonnet"
    assert Path(obj.base_inferencer.target_path) == Path(ws)
    assert obj.base_inferencer.permission_mode == "bypassPermissions"

    # Review inferencer (KiroCliInferencer)
    assert obj.review_inferencer.model_name == "auto"
    assert Path(obj.review_inferencer.target_path) == Path(ws)
    assert obj.review_inferencer.trust_mode == "all"

    # ConsensusConfig
    assert obj.consensus_config.max_iterations == 2
    # consensus_threshold may be a Severity enum or a plain string from YAML
    assert str(obj.consensus_config.consensus_threshold) == "COSMETIC"


# ===========================================================================
# Test 3: Nested YAML composition (Req 6.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_both
def test_yaml_nested_composition(tmp_workspace):
    """Verify nested DualInferencer — a DualInferencer as base_inferencer of another."""
    # Normalize to forward slashes for YAML safety on Windows
    ws = str(tmp_workspace["workspace"]).replace("\\", "/")

    # Build nested YAML inline
    nested_yaml = f"""\
_target_: Dual
base_inferencer:
  _target_: Dual
  base_inferencer:
    _target_: ClaudeCodeCLI
    model_name: sonnet
    target_path: "{ws}"
    permission_mode: bypassPermissions
  review_inferencer:
    _target_: agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer.KiroCliInferencer
    model_name: auto
    target_path: "{ws}"
    trust_mode: all
review_inferencer:
  _target_: ClaudeCodeCLI
  model_name: sonnet
  target_path: "{ws}"
  permission_mode: bypassPermissions
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        tmp.write(nested_yaml)
        tmp_path = tmp.name

    try:
        cfg = load_config(tmp_path)
    finally:
        os.unlink(tmp_path)

    obj = instantiate(cfg)

    # Outer DualInferencer
    assert isinstance(obj, DualInferencer)
    # Inner DualInferencer as base_inferencer
    assert isinstance(obj.base_inferencer, DualInferencer)
    assert isinstance(obj.base_inferencer.base_inferencer, ClaudeCodeCliInferencer)
    assert isinstance(obj.base_inferencer.review_inferencer, KiroCliInferencer)
    # Outer review_inferencer
    assert isinstance(obj.review_inferencer, ClaudeCodeCliInferencer)


# ===========================================================================
# Test 4: YAML checkpoint config (Req 7.1, 7.2)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
def test_yaml_checkpoint_config(tmp_workspace):
    """Load dual_with_checkpoint.yaml → verify checkpoint and cache settings."""
    ws = str(tmp_workspace["workspace"])
    cache = str(tmp_workspace["cache"])
    ckpt = str(tmp_workspace["checkpoint"])

    cfg = _load_yaml_with_placeholders(
        "dual_with_checkpoint.yaml",
        {
            "workspace_path": ws,
            "cache_dir": cache,
            "checkpoint_dir": ckpt,
        },
    )
    obj = instantiate(cfg)

    assert isinstance(obj, DualInferencer)
    assert obj.enable_checkpoint is True
    assert Path(obj.checkpoint_dir) == Path(ckpt)

    # Verify cache settings on children
    assert Path(obj.base_inferencer.cache_folder) == Path(cache)
    assert obj.base_inferencer.resume_with_saved_results is True
    assert Path(obj.review_inferencer.cache_folder) == Path(cache)
    assert obj.review_inferencer.resume_with_saved_results is True


# ===========================================================================
# Test 5: YAML real inference call (Req 6.1, 7.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT)
@skip_claude
async def test_yaml_real_inference_call(tmp_workspace):
    """Load dual_with_checkpoint.yaml → instantiate → run a real inference call."""
    ws = str(tmp_workspace["workspace"])
    cache = str(tmp_workspace["cache"])
    ckpt = str(tmp_workspace["checkpoint"])

    cfg = _load_yaml_with_placeholders(
        "dual_with_checkpoint.yaml",
        {
            "workspace_path": ws,
            "cache_dir": cache,
            "checkpoint_dir": ckpt,
        },
    )
    obj = instantiate(cfg)

    result = await obj.ainfer(CHEAP_PROMPT)
    assert result is not None

    # Verify checkpoint directory has content after a real run
    checkpoint_files = []
    for root, _dirs, files in os.walk(ckpt):
        checkpoint_files.extend(files)
    # Checkpoint files should exist since enable_checkpoint=True
    assert len(checkpoint_files) >= 1, (
        "Checkpoint files should be created after a real inference call "
        "with enable_checkpoint=True"
    )

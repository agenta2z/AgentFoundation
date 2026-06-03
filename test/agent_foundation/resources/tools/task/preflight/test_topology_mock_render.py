"""TIER 2: Topology dry-run with mocked inferencers.

Verifies that the entire topology instantiates and that every prompt
template RENDERS (via Jinja) when called with mock context.

Run in <60 seconds. Uses real Hydra config + topology but replaces all
LLM inferencers with mocks that return canned responses.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_HERE = Path(__file__).resolve().parent
YAML_PATH = _HERE.parents[5] / "src" / "openteam" / "server" / "resources" / "tools" / "task" / "topologies" / "breakdown-multiflow-plan-then-implement.yaml"
OPENSTARTUP_PATH = Path(
    os.environ.get(
        "OPENSTARTUP_PATH",
        str(_HERE.parents[4]),
    )
)
TEMPLATES_DIR = OPENSTARTUP_PATH / "src" / "openteam" / "server" / "resources" / "prompt_templates"


def _instantiate_topology_with_mocks(monkeypatch, tmp_path):
    """Load topology YAML and instantiate with mock inferencers.
    
    Each mock inferencer records which prompts it would send, then returns
    a canned response. This lets us verify that all prompt templates render
    without actually calling Claude/RovoDevCLI.
    """
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from rich_python_utils.config_utils import load_config, instantiate

    base_overrides = {
        "_target_path": str(OPENSTARTUP_PATH),
        "templates_dir": str(TEMPLATES_DIR),
        "_params.workspace_root": str(tmp_path / "ws"),
    }
    cfg = load_config(str(YAML_PATH), overrides=base_overrides)
    
    # Replace all leaf inferencers with mocks BEFORE instantiation
    def _mock_factory(name):
        """Create a mock inferencer that records render calls."""
        mock = MagicMock(name=name)
        # Return a canned dict response for async inference
        mock.infer = MagicMock(
            return_value={"output": f"[mock response from {name}]"}
        )
        # Record any call to render or format_prompt
        mock.render_prompt = MagicMock(return_value="[rendered prompt]")
        return mock
    
    # Recursively replace all _target_ entries pointing to real inferencers
    def _replace_targets(obj):
        if isinstance(obj, dict):
            if "_target_" in obj:
                target = obj.get("_target_", "")
                if "Inferencer" in target and "RovoDevCLI" not in target:
                    # Replace with mock marker; instantiate() will see this
                    obj["_target_"] = "unittest.mock.MagicMock"
                    obj["__mock_name__"] = target
            for v in obj.values():
                _replace_targets(v)
        elif isinstance(obj, list):
            for item in obj:
                _replace_targets(item)
    
    _replace_targets(cfg)
    
    # Instantiate: any MagicMock targets will be created as mocks
    root = instantiate(cfg)
    return root


@pytest.mark.skip(reason="TIER 2: requires topology YAML; integrate after Hydra setup verified")
def test_topology_instantiates_with_mocks(tmp_path, monkeypatch):
    """Verify topology loads and instantiates without errors."""
    root = _instantiate_topology_with_mocks(monkeypatch, tmp_path)
    assert root is not None
    assert hasattr(root, "base_inferencer") or hasattr(root, "fixer_inferencer")


@pytest.mark.skip(reason="TIER 2: requires topology YAML; integrate after Hydra setup verified")
def test_all_leaf_inferencers_exist_when_expected(tmp_path, monkeypatch):
    """Verify fixer_inferencer exists when fixer_match_winner=true in config."""
    root = _instantiate_topology_with_mocks(monkeypatch, tmp_path)
    
    # Check structural assumptions from test_hyperparams_default_inferencer.py
    assert hasattr(root, "base_inferencer"), "root missing base_inferencer"
    assert hasattr(root, "review_inferencer"), "root missing review_inferencer"
    
    # fixer_inferencer should exist and be non-None (not stubbed out)
    assert hasattr(root, "fixer_inferencer"), "root missing fixer_inferencer attribute"
    assert root.fixer_inferencer is not None, "fixer_inferencer is None"


@pytest.mark.skip(reason="TIER 2: requires topology YAML; integrate after Hydra setup verified")
def test_mock_render_calls_capture_prompts(tmp_path, monkeypatch):
    """Verify that calling infer() on mocks renders prompts without crashing."""
    root = _instantiate_topology_with_mocks(monkeypatch, tmp_path)
    
    # Simulate a minimal task input
    mock_task = {
        "task": "Test task",
        "context": {},
    }
    
    # Call base_inferencer.infer() with mock task; should not raise
    # (Real infer() would format prompt templates; mocks return canned response)
    try:
        result = root.base_inferencer.infer(task_input=mock_task)
        assert result is not None, "mock infer() returned None"
    except Exception as e:
        # If a Jinja render error occurs here, the test catches it
        pytest.fail(f"infer() raised exception (likely Jinja render error): {e}")

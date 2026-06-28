"""Env overrides for the breakdown-multiflow-plan task config (TASK__ prefix).

The YAML carries plain ``_params`` defaults + ``env_prefix: TASK``; the loader
(`config_utils._instantiate`) overlays ``TASK__<KEY>`` env vars, coerced to each
key's type. Lists take a single var (comma or JSON). ``num_flows`` is derived
from ``flow_inferencers`` length, so overriding the list also sets the count.
CLI ``--override`` still wins.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rich_python_utils.config_utils._instantiate import load_config

CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "breakdown-multiflow-plan.yaml"
)
if not CONFIG.exists():  # pragma: no cover - layout fallback
    import agent_foundation.resources.tools.task as _taskpkg

    CONFIG = Path(_taskpkg.__file__).parent / "configs" / "breakdown-multiflow-plan.yaml"

_TASK_VARS = [
    "TASK__DEFAULT_INFERENCER", "TASK__MAIN_INFERENCER", "TASK__FLOW_INFERENCERS",
    "TASK__NUM_FLOWS", "TASK__PLAN_MAX_BREAKDOWN", "TASK__FLOW_MAX_DYNAMIC_STEPS",
    "TASK__CONSENSUS_MAX_ITERATIONS", "TASK__REVIEWER_STRATEGY", "TASK__FIXER_STRATEGY",
    "TASK__ENABLE_DEEP_MODE", "TASK__ENABLE_ELEGANT_MODE", "TASK__GUARDRAIL_MODEL_TIER",
]


@pytest.fixture
def clean_env(monkeypatch):
    for v in _TASK_VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _load(overrides=None):
    ov = {"_params.workspace_root": "/tmp/ws"}
    if overrides:
        ov.update(overrides)
    return OmegaConf.to_container(load_config(str(CONFIG), overrides=ov), resolve=True)


def _flow_targets(c):
    fc = c["base_inferencer"]["worker_inferencers"]["flow_configs"]
    return [f["initial_inferencer"]["_target_"] for f in fc]


def test_defaults_and_env_prefix_stripped(clean_env):
    c = _load()
    p = c["_params"]
    assert "env_prefix" not in p
    assert p["main_inferencer"] == "ClaudeCodeCLI"
    assert p["flow_inferencers"] == ["ClaudeCodeCLI", "ClaudeCodeCLI"]
    assert p["num_flows"] == 2 and isinstance(p["num_flows"], int)
    assert p["plan_max_breakdown"] == 3 and p["enable_deep_mode"] is True
    assert _flow_targets(c) == ["ClaudeCodeCLI", "ClaudeCodeCLI"]


def test_main_inferencer_env_propagates_to_flows(clean_env):
    clean_env.setenv("TASK__MAIN_INFERENCER", "DevmateCLI")
    c = _load()
    assert c["_params"]["main_inferencer"] == "DevmateCLI"
    assert _flow_targets(c) == ["DevmateCLI", "DevmateCLI"]


def test_flow_inferencers_single_var_comma_drives_count(clean_env):
    clean_env.setenv("TASK__FLOW_INFERENCERS", "MetamateSDK,RovoDevCLI,DevmateCLI")
    c = _load()
    assert c["_params"]["flow_inferencers"] == ["MetamateSDK", "RovoDevCLI", "DevmateCLI"]
    assert c["_params"]["num_flows"] == 3
    assert _flow_targets(c) == ["MetamateSDK", "RovoDevCLI", "DevmateCLI"]


def test_flow_inferencers_json(clean_env):
    clean_env.setenv("TASK__FLOW_INFERENCERS", '["X","Y"]')
    c = _load()
    assert _flow_targets(c) == ["X", "Y"] and c["_params"]["num_flows"] == 2


def test_typed_scalars(clean_env):
    clean_env.setenv("TASK__PLAN_MAX_BREAKDOWN", "7")
    clean_env.setenv("TASK__ENABLE_DEEP_MODE", "false")
    p = _load()["_params"]
    assert p["plan_max_breakdown"] == 7 and isinstance(p["plan_max_breakdown"], int)
    assert p["enable_deep_mode"] is False


def test_string_scalars_and_guardrail_propagation(clean_env):
    clean_env.setenv("TASK__REVIEWER_STRATEGY", "winner")
    clean_env.setenv("TASK__GUARDRAIL_MODEL_TIER", "max")
    c = _load()
    assert c["_params"]["reviewer_strategy"] == "winner"
    agg = c["base_inferencer"]["aggregator_inferencer"]
    assert agg["output_guardrail_inferencer"]["model_tier"] == "max"


def test_cli_override_beats_env(clean_env):
    clean_env.setenv("TASK__PLAN_MAX_BREAKDOWN", "7")
    c = _load(overrides={"_params.plan_max_breakdown": 9})
    assert c["_params"]["plan_max_breakdown"] == 9


# --- TASK__NUM_FLOWS set independently → graceful distribution (no crash) ---

def test_num_flows_independent_truncates_longer_list(clean_env):
    clean_env.setenv("TASK__FLOW_INFERENCERS", "Aa,Bb,Cc")
    clean_env.setenv("TASK__NUM_FLOWS", "2")
    c = _load()
    assert c["_params"]["num_flows"] == 2
    assert _flow_targets(c) == ["Aa", "Bb"]  # list truncated to count


def test_num_flows_independent_pads_shorter_list(clean_env):
    clean_env.setenv("TASK__FLOW_INFERENCERS", "Aa,Bb")
    clean_env.setenv("TASK__NUM_FLOWS", "4")
    c = _load()
    assert c["_params"]["num_flows"] == 4
    assert _flow_targets(c) == ["Aa", "Bb", "Aa", "Aa"]  # padded with first


def test_num_flows_broadcasts_single_inferencer(clean_env):
    clean_env.setenv("TASK__FLOW_INFERENCERS", "ClaudeCodeCLI")
    clean_env.setenv("TASK__NUM_FLOWS", "3")
    c = _load()
    assert _flow_targets(c) == ["ClaudeCodeCLI", "ClaudeCodeCLI", "ClaudeCodeCLI"]

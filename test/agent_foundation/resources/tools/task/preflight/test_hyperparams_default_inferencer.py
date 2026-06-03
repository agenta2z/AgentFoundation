"""Preflight test for `_params.default_inferencer`.

Validates that the single `_params.default_inferencer` variable controls
all 13 leaf inferencer `_target_` entries in the topology YAML, and that
overriding it via `load_config(overrides=...)` switches every leaf to the
requested backend (e.g., RovoDevCLI) while preserving structural orchestrator
types (PTI, Dual, BTA, MultiFlowDual).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

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


def _collect_leaf_inferencers(root):
    """Return {name: inferencer} for every leaf in the base + fixer subtrees."""
    leaves = {}

    def _from_pti(pti, prefix):
        plan_dual = pti.planner_inferencer
        plan_bta = plan_dual.base_inferencer

        leaves[f"{prefix}.plan_breakdown"] = plan_bta.breakdown_inferencer
        leaves[f"{prefix}.plan_aggregator"] = plan_bta.aggregator_inferencer
        leaves[f"{prefix}.plan_review"] = plan_dual.review_inferencer

        plan_factory = plan_bta.worker_factory["__default__"]
        sample_mfdual = plan_factory()
        leaves[f"{prefix}.mfdual_agg"] = sample_mfdual.multi_flow_aggregator_inferencer
        for i, fc in enumerate(sample_mfdual.flow_configs):
            leaves[f"{prefix}.flow{i}_initial"] = fc["initial_inferencer"]
            leaves[f"{prefix}.flow{i}_followup"] = fc["followup_inferencer"]

        exec_bta = pti.executor_inferencer
        leaves[f"{prefix}.exec_breakdown"] = exec_bta.breakdown_inferencer
        leaves[f"{prefix}.exec_aggregator"] = exec_bta.aggregator_inferencer

        exec_factory = exec_bta.worker_factory["__default__"]
        sample_dual = exec_factory()
        leaves[f"{prefix}.exec_worker_base"] = sample_dual.base_inferencer
        leaves[f"{prefix}.exec_worker_review"] = sample_dual.review_inferencer

    _from_pti(root.base_inferencer, "base")
    _from_pti(root.fixer_inferencer, "fixer")
    leaves["outer_review"] = root.review_inferencer
    return leaves


def _instantiate_yaml(monkeypatch, tmp_path, overrides=None):
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from rich_python_utils.config_utils import load_config, instantiate

    base_overrides = {
        "_target_path": str(OPENSTARTUP_PATH),
        "templates_dir": str(TEMPLATES_DIR),
        "_params.workspace_root": str(tmp_path / "ws"),
    }
    if overrides:
        base_overrides.update(overrides)
    cfg = load_config(str(YAML_PATH), overrides=base_overrides)
    return instantiate(cfg)


# ---- Default: all leaves are ClaudeCodeCliInferencer ----

def test_default_inferencer_resolves_to_claude_code_cli(tmp_path, monkeypatch):
    """With no override, every leaf inferencer must be ClaudeCodeCliInferencer."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
        ClaudeCodeCliInferencer,
    )

    root = _instantiate_yaml(monkeypatch, tmp_path)
    leaves = _collect_leaf_inferencers(root)

    assert len(leaves) >= 25, (
        f"expected >=25 leaf inferencers (13 base + 12 fixer + 1 outer); got {len(leaves)}"
    )
    for name, leaf in leaves.items():
        assert isinstance(leaf, ClaudeCodeCliInferencer), (
            f"{name}: expected ClaudeCodeCliInferencer, got {type(leaf).__name__}"
        )


# ---- Override: all leaves switch to RovoDevCliInferencer ----

def test_default_inferencer_override_to_rovodev(tmp_path, monkeypatch):
    """Overriding _params.default_inferencer=RovoDevCLI must switch every leaf."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
        RovoDevCliInferencer,
    )

    root = _instantiate_yaml(
        monkeypatch, tmp_path,
        overrides={"_params.default_inferencer": "RovoDevCLI"},
    )
    leaves = _collect_leaf_inferencers(root)

    for name, leaf in leaves.items():
        assert isinstance(leaf, RovoDevCliInferencer), (
            f"{name}: expected RovoDevCliInferencer after override, "
            f"got {type(leaf).__name__}"
        )


# ---- Structural types preserved under override ----

def test_structural_types_preserved_under_override(tmp_path, monkeypatch):
    """Overriding the leaf inferencer must NOT change orchestrator types."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
        BreakdownThenAggregateInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
        MultiFlowDualInferencer,
    )

    root = _instantiate_yaml(
        monkeypatch, tmp_path,
        overrides={"_params.default_inferencer": "RovoDevCLI"},
    )

    assert isinstance(root, DualInferencer), (
        f"outer must remain Dual; got {type(root).__name__}"
    )
    assert isinstance(root.base_inferencer, PlanThenImplementInferencer), (
        f"base must remain PTI; got {type(root.base_inferencer).__name__}"
    )
    assert isinstance(root.fixer_inferencer, PlanThenImplementInferencer), (
        f"fixer must remain PTI; got {type(root.fixer_inferencer).__name__}"
    )

    plan_dual = root.base_inferencer.planner_inferencer
    assert isinstance(plan_dual, DualInferencer)
    plan_bta = plan_dual.base_inferencer
    assert isinstance(plan_bta, BreakdownThenAggregateInferencer)

    plan_factory = plan_bta.worker_factory["__default__"]
    assert isinstance(plan_factory, functools.partial)
    sample_mfdual = plan_factory()
    assert isinstance(sample_mfdual, MultiFlowDualInferencer)

    exec_bta = root.base_inferencer.executor_inferencer
    assert isinstance(exec_bta, BreakdownThenAggregateInferencer)

    exec_factory = exec_bta.worker_factory["__default__"]
    sample_exec_dual = exec_factory()
    assert isinstance(sample_exec_dual, DualInferencer)

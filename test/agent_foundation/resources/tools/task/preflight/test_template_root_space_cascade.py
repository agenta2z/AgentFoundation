"""Preflight: verify _template_root_space cascades correctly in the full PTI topology.

The outer Dual sets `_template_root_space: implementation` which cascades to
its children (leaves get template_root_space=implementation). The planner
imports breakdown-multiflow-plan.yaml which declares its own
`_template_root_space: plan`. The imported YAML's declaration must WIN for
the planner subtree's leaves.

BTA breakdown inferencers get `task_breakdown` from SLOT_DEFAULTS
(BREAKDOWN_TEMPLATE_DEFAULTS), overriding any cascade.

Runtime: ~5s, no LLM cost.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent
YAML_PATH = _HERE.parents[5] / "src" / "openteam" / "server" / "resources" / "tools" / "task" / "topologies" / "breakdown-multiflow-plan-then-implement.yaml"
OPENSTARTUP_PATH = Path(os.environ.get("OPENSTARTUP_PATH", str(_HERE.parents[4])))
TEMPLATES_DIR = OPENSTARTUP_PATH / "src" / "openteam" / "server" / "resources" / "prompt_templates"


def _instantiate(tmp_path):
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from rich_python_utils.config_utils import load_config, instantiate

    cfg = load_config(str(YAML_PATH), overrides={
        "_target_path": str(OPENSTARTUP_PATH),
        "templates_dir": str(TEMPLATES_DIR),
        "_params.workspace_root": str(tmp_path / "ws"),
    })
    return instantiate(cfg)


def test_outer_review_has_implementation_space(tmp_path):
    """Outer review inferencer inherits 'implementation' from outer Dual cascade."""
    root = _instantiate(tmp_path)
    assert getattr(root.review_inferencer, "template_root_space", None) == "implementation"


def test_outer_fixer_has_implementation_space(tmp_path):
    """Outer fixer inferencer inherits 'implementation' from outer Dual cascade."""
    root = _instantiate(tmp_path)
    assert getattr(root.fixer_inferencer, "template_root_space", None) == "implementation"


def test_planner_review_leaf_has_plan_space(tmp_path):
    """Planner's review leaf inherits 'plan' from imported YAML's cascade,
    NOT 'implementation' from the outer Dual."""
    root = _instantiate(tmp_path)
    plan_dual = root.base_inferencer.planner_inferencer
    review = plan_dual.review_inferencer
    assert getattr(review, "template_root_space", None) == "plan", (
        "Planner review leaf should have 'plan' from imported YAML, "
        "not 'implementation' from outer cascade."
    )


def test_exec_breakdown_has_task_breakdown_from_slot_defaults(tmp_path):
    """Exec BTA breakdown gets 'task_breakdown' from BREAKDOWN_TEMPLATE_DEFAULTS,
    overriding the 'implementation' cascade."""
    root = _instantiate(tmp_path)
    exec_bta = root.base_inferencer.executor_inferencer
    assert getattr(exec_bta.breakdown_inferencer, "template_root_space", None) == "task_breakdown"


def test_plan_breakdown_has_task_breakdown_from_slot_defaults(tmp_path):
    """Plan BTA breakdown gets 'task_breakdown' from BREAKDOWN_TEMPLATE_DEFAULTS,
    overriding the 'plan' cascade."""
    root = _instantiate(tmp_path)
    plan_bta = root.base_inferencer.planner_inferencer.base_inferencer
    assert getattr(plan_bta.breakdown_inferencer, "template_root_space", None) == "task_breakdown"


def test_exec_worker_review_has_implementation_space(tmp_path):
    """Exec BTA worker's review leaf inherits 'implementation'."""
    root = _instantiate(tmp_path)
    exec_bta = root.base_inferencer.executor_inferencer
    sample_worker = exec_bta.worker_factory["__default__"]()
    assert getattr(sample_worker.review_inferencer, "template_root_space", None) == "implementation"


def test_outer_dual_no_prompt_formatter(tmp_path):
    """Outer Dual should NOT have prompt_formatter (removed in Phase 2 cleanup)."""
    root = _instantiate(tmp_path)
    assert not hasattr(root, "prompt_formatter"), (
        "prompt_formatter attribute should be removed from DualInferencer."
    )

"""Preflight tests for YAML config — deliverable flags on aggregator inferencers.

Verifies that the configured topology
(`breakdown-multiflow-plan-then-implement.yaml`) has the deliverable-promotion
flags set on the right inferencers, so that the run produces the expected
final_deliverables/ surfacing chain.

This is a config-only test (no instantiation needed) — guards against
someone accidentally removing the flags from the YAML.

Required configuration (per cached-hennessy plan, Step 6):
  • plan-stage `aggregator_inferencer.output_is_deliverable: true`
  • `multi_flow_aggregator_inferencer.output_is_deliverable: true`
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


_HERE = Path(__file__).resolve().parent
YAML_PATH = _HERE.parents[5] / "src" / "openteam" / "server" / "resources" / "tools" / "task" / "topologies" / "breakdown-multiflow-plan-then-implement.yaml"
OPENSTARTUP_PATH = _HERE.parents[5]
TEMPLATES_DIR = OPENSTARTUP_PATH / "src" / "openteam" / "server" / "resources" / "prompt_templates"


def _load_raw_yaml():
    """Load the YAML with _import_ resolution (but without full instantiation).

    Uses load_config() so _import_ directives are resolved — the planner
    subtree (extracted to breakdown_multiflow_plan.yaml) is merged in.
    """
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from rich_python_utils.config_utils import load_config
    from omegaconf import OmegaConf
    cfg = load_config(
        str(YAML_PATH),
        overrides={
            "_target_path": str(OPENSTARTUP_PATH),
            "templates_dir": str(TEMPLATES_DIR),
            "_params.workspace_root": "/tmp/_test_deliverable_flags",
        },
    )
    return OmegaConf.to_container(cfg, resolve=True)


def _walk_for_key(node, key):
    """Yield every (path, value) where key appears in the nested config."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield (k, v)
            yield from _walk_for_key(v, key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_for_key(item, key)


# -------------------------------------------------------------------------
# Sanity: file exists and parses
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_YD1_yaml_file_exists():
    """YD1: The configured topology YAML exists at the expected path."""
    assert YAML_PATH.is_file(), (
        f"Topology YAML missing: {YAML_PATH}. "
        "This is the contract anchor — moving it requires updating tests "
        "and any callers."
    )


@pytest.mark.preflight
def test_YD2_yaml_parses_as_valid_yaml():
    """YD2: The topology YAML is well-formed YAML."""
    cfg = _load_raw_yaml()
    assert isinstance(cfg, dict), "Top-level YAML must be a mapping"


# -------------------------------------------------------------------------
# Deliverable flag presence
# -------------------------------------------------------------------------

@pytest.mark.preflight
def test_YD3_some_inferencer_marked_output_is_deliverable():
    """YD3: At least one inferencer in the YAML sets `output_is_deliverable: true`.

    Without this, no leaf will self-promote and the final_deliverables/ chain
    will be empty at the outermost layer.
    """
    cfg = _load_raw_yaml()
    matches = list(_walk_for_key(cfg, "output_is_deliverable"))
    truthy = [(k, v) for k, v in matches if v is True]
    assert truthy, (
        "Expected at least one `output_is_deliverable: true` in the topology "
        "YAML. Without it, leaf inferencers do not surface their outputs as "
        "deliverables (cached-hennessy plan, Step 6). "
        f"All occurrences: {matches}"
    )


@pytest.mark.preflight
def test_YD4_multi_flow_aggregator_marked_deliverable():
    """YD4: `multi_flow_aggregator_inferencer` is marked `output_is_deliverable: true`."""
    cfg = _load_raw_yaml()
    found = []
    for path_value in _walk_for_key(cfg, "multi_flow_aggregator_inferencer"):
        block = path_value[1]
        if isinstance(block, dict):
            found.append(block.get("output_is_deliverable"))
    assert found, "No `multi_flow_aggregator_inferencer` block found in YAML"
    assert any(v is True for v in found), (
        "`multi_flow_aggregator_inferencer.output_is_deliverable` must be "
        f"True so MFDual aggregator outputs surface as deliverables. Got: {found}"
    )


@pytest.mark.preflight
def test_YD5_plan_aggregator_marked_deliverable():
    """YD5: The plan-stage `aggregator_inferencer` is marked `output_is_deliverable: true`.

    This is the BTA aggregator inside the planner Dual. Marking it ensures
    the integrated plan surfaces from plan_bta → planner_dual → outer chain.
    """
    cfg = _load_raw_yaml()
    aggregator_blocks = []
    for _, block in _walk_for_key(cfg, "aggregator_inferencer"):
        if isinstance(block, dict):
            aggregator_blocks.append(block)

    assert aggregator_blocks, "No `aggregator_inferencer` block found in YAML"
    truthy = [b for b in aggregator_blocks
              if b.get("output_is_deliverable") is True]
    assert truthy, (
        "At least one `aggregator_inferencer` in the YAML must set "
        "`output_is_deliverable: true` (the plan-stage aggregator). "
        f"All aggregator blocks found: {len(aggregator_blocks)}; "
        f"with the flag: {len(truthy)}"
    )


@pytest.mark.preflight
def test_YD6_no_unintended_manifest_index_overrides():
    """YD6: `output_manifest_index` should not be explicitly set to False where
    `output_is_deliverable: true` is also set — that would silently disable
    manifest emission.

    (The auto-enable rule at inferencer_base.py:764 still emits the manifest
    when output_is_deliverable=True, so explicit `output_manifest_index: false`
    is redundant; but explicit override is defensible if intentional. This
    test just guards against accidental disabling.)
    """
    cfg = _load_raw_yaml()

    def _check(node, parent_key=""):
        problems = []
        if isinstance(node, dict):
            if (node.get("output_is_deliverable") is True
                    and node.get("output_manifest_index") is False):
                problems.append(
                    f"Block `{parent_key}` sets output_is_deliverable=True "
                    "AND explicitly disables output_manifest_index. This "
                    "likely-unintentionally suppresses manifest emission."
                )
            for k, v in node.items():
                problems.extend(_check(v, parent_key=k))
        elif isinstance(node, list):
            for item in node:
                problems.extend(_check(item, parent_key))
        return problems

    problems = _check(cfg)
    assert not problems, "\n".join(problems)

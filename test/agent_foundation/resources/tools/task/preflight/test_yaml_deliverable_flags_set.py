"""Preflight tests for YAML config — retired deliverable flags are ABSENT.

Part 2 (two-axis model) RETIRED the deliverable flags. ``outputs/`` IS the
deliverable set; promotion is role-based (``promote_child`` /
``_symlink_child_output``); there is no ``final_deliverables/`` subfolder.

Consequently the task topology YAML (``default.yaml``) must NO LONGER carry the
retired config keys ``output_is_deliverable`` or ``use_final_deliverables_folder``.
This is a config-only test (no instantiation of inferencers needed) — it guards
against someone re-adding the retired keys after the migration.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


_HERE = Path(__file__).resolve().parent
YAML_PATH = (
    _HERE.parents[5]
    / "src"
    / "agent_foundation"
    / "resources"
    / "tools"
    / "task"
    / "configs"
    / "default.yaml"
)
OPENSTARTUP_PATH = _HERE.parents[5]
TEMPLATES_DIR = (
    OPENSTARTUP_PATH / "src" / "agent_foundation" / "resources" / "prompt_templates"
)


def _load_raw_yaml():
    """Load the YAML with _import_ resolution (but without full instantiation).

    Uses load_config() so _import_ directives are resolved — the planner
    subtree (extracted to breakdown_multiflow_plan.yaml) is merged in.
    """
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from omegaconf import OmegaConf
    from rich_python_utils.config_utils import load_config

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
# Retired flags MUST be absent (Part 2 two-axis migration)
# -------------------------------------------------------------------------


@pytest.mark.preflight
def test_YD3_output_is_deliverable_key_absent():
    """YD3: The retired ``output_is_deliverable`` key appears NOWHERE in the YAML.

    Part 2 retired the flag — ``outputs/`` IS the deliverable set and promotion
    is role-based. Re-adding the key would be dead config the framework ignores.
    """
    cfg = _load_raw_yaml()
    matches = list(_walk_for_key(cfg, "output_is_deliverable"))
    assert not matches, (
        "Retired key `output_is_deliverable` must NOT appear in the topology "
        "YAML after the Part 2 two-axis migration (deliverables live directly "
        f"in outputs/). Found occurrences: {matches}"
    )


@pytest.mark.preflight
def test_YD4_use_final_deliverables_folder_key_absent():
    """YD4: The retired ``use_final_deliverables_folder`` key appears NOWHERE.

    Part 2 removed the ``final_deliverables/`` subfolder entirely; workspaces
    are constructed as ``InferencerWorkspace(root=...)`` with no such flag.
    """
    cfg = _load_raw_yaml()
    matches = list(_walk_for_key(cfg, "use_final_deliverables_folder"))
    assert not matches, (
        "Retired key `use_final_deliverables_folder` must NOT appear in the "
        "topology YAML after the Part 2 migration (final_deliverables/ retired). "
        f"Found occurrences: {matches}"
    )


@pytest.mark.preflight
def test_YD5_output_manifest_index_key_is_still_permitted():
    """YD5: ``output_manifest_index`` is INDEPENDENT of promotion and may still
    appear. When present, every value must be a real bool (guards against a
    typo'd string sneaking in).

    This is the surviving Axis-A knob (manifest emission → artifacts/); it is
    NOT retired, so its presence is fine and its absence is also fine.
    """
    cfg = _load_raw_yaml()
    for _, value in _walk_for_key(cfg, "output_manifest_index"):
        assert isinstance(value, bool), (
            "`output_manifest_index`, when set, must be a bool; got "
            f"{value!r} ({type(value).__name__})."
        )

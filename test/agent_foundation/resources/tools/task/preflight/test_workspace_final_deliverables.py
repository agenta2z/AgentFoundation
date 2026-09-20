"""Preflight test: the outer Dual's workspace uses the two-axis ``outputs/``
contract (Part 2) — NOT the retired ``final_deliverables/`` folder.

Why this matters
----------------
Part 2 retired ``use_final_deliverables_folder`` and the ``final_deliverables/``
subfolder. ``outputs/`` IS the deliverable set now, so the outer Dual's
workspace must:

  * be a real ``InferencerWorkspace`` (not a surprise subclass), and
  * expose an ``outputs_dir`` under the workspace root, and
  * NOT carry the retired ``use_final_deliverables_folder`` attrib or the
    ``deliverables_dir`` property.

This preflight catches a silent regression where someone re-introduces the
retired flag/folder machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Resolve the YAML config — same convention as the legacy
# `test_yaml_smoke_instantiate`. preflight/ lives one level below task/.
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
# OpenStartup root: preflight/<this> → task/ → tools/ → resources/ → openteam/
# → test/ → OpenStartup/.
OPENSTARTUP_PATH = _HERE.parents[5]
# Templates dir colocated with the test resources (same as legacy smoke test).
TEMPLATES_DIR = _HERE.parents[1] / "task" / "configs" / "prompt_templates"
if not TEMPLATES_DIR.exists():
    # Legacy smoke test points templates_dir at OpenStartup's prompt_templates.
    TEMPLATES_DIR = (
        OPENSTARTUP_PATH / "src" / "agent_foundation" / "resources" / "prompt_templates"
    )


def _load_topology(monkeypatch, tmp_path):
    """Bind required env vars + load + instantiate the YAML config.

    Mirrors the bootstrap done in `test_yaml_smoke_instantiate` so this
    preflight is self-contained.
    """
    monkeypatch.setenv("PROMPT_TEMPLATES_DIR", "prompt_templates")

    # Side-effect import to register Hydra targets (ClaudeCodeCLI, Dual, etc.)
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from rich_python_utils.config_utils import instantiate, load_config

    cfg = load_config(
        str(YAML_PATH),
        overrides={
            "_target_path": str(OPENSTARTUP_PATH),
            "templates_dir": str(TEMPLATES_DIR),
            "_params.workspace_root": str(tmp_path / "ws"),
        },
    )
    return instantiate(cfg)


def test_yaml_loads(tmp_path, monkeypatch):
    """Sanity: the YAML still parses + instantiates."""
    topology = _load_topology(monkeypatch, tmp_path)
    assert topology is not None
    # Outer Dual is the root.
    assert type(topology).__name__ == "DualInferencer"


def test_outer_workspace_uses_outputs_dir(tmp_path, monkeypatch):
    """The outer Dual's workspace exposes an ``outputs_dir`` under its root —
    the two-axis deliverable set (Part 2). No ``final_deliverables/`` folder."""
    topology = _load_topology(monkeypatch, tmp_path)

    ws = topology._workspace
    assert ws is not None, (
        "Outer Dual has no _workspace; expected an InferencerWorkspace "
        "constructed from the `workspace:` block in the YAML."
    )
    outputs_dir = ws.outputs_dir
    assert outputs_dir is not None, "outputs_dir must resolve"
    # Path convention: <root>/outputs — the deliverable set lives directly here.
    assert Path(outputs_dir).name == "outputs", (
        f"outputs_dir={outputs_dir!r} does not end with the expected 'outputs' segment."
    )


def test_outer_workspace_has_no_retired_flag_or_property(tmp_path, monkeypatch):
    """Regression guard: the retired ``use_final_deliverables_folder`` attrib and
    ``deliverables_dir`` property must NOT be present on the outer workspace
    after the Part 2 migration."""
    topology = _load_topology(monkeypatch, tmp_path)

    ws = topology._workspace
    assert not hasattr(ws, "use_final_deliverables_folder"), (
        "Retired attrib `use_final_deliverables_folder` reappeared on the "
        "workspace; Part 2 removed it (outputs/ IS the deliverable set)."
    )
    assert not hasattr(ws, "deliverables_dir"), (
        "Retired property `deliverables_dir` reappeared on the workspace; "
        "callers must use `outputs_dir` / `output_path(...)` now."
    )


def test_inferencer_workspace_class_is_used(tmp_path, monkeypatch):
    """Defensive: ensure the topology really uses InferencerWorkspace (not a
    surprise subclass), so the outputs/ semantics described above hold."""
    topology = _load_topology(monkeypatch, tmp_path)
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )

    assert isinstance(topology._workspace, InferencerWorkspace)

"""Preflight test: verify the YAML's explicit workspace block enables
`use_final_deliverables_folder=True`.

Why this matters
----------------
Without this flag, the BTA at line ~977 in `breakdown_then_aggregate_inferencer.py`
falls back to writing copied worker deliverables directly into `outputs/`,
intermixed with intermediate streaming logs and the final consolidated
output.md. With the flag enabled, the BTA writes them to
`outputs/final_deliverables/` (a distinct subdirectory) which gives a clear
"this is the user-facing deliverable" boundary.

The YAML used to pass `workspace_root: "/tmp/topo_b_ws"` (a string shorthand
that the Dual class auto-converts to `InferencerWorkspace(root=...)` with
default flags). We replaced that with an explicit
`workspace: {_target_: InferencerWorkspace, root: ..., use_final_deliverables_folder: true}`
block so the flag is plumbed through.

This preflight catches a silent regression where someone reverts to the
shorthand form (which would still load OK but lose the deliverables/
separation).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Resolve the YAML config — same convention as the legacy
# `test_yaml_smoke_instantiate`. preflight/ lives one level below task/.
_HERE = Path(__file__).resolve().parent
YAML_PATH = _HERE.parents[5] / "src" / "openteam" / "server" / "resources" / "tools" / "task" / "topologies" / "breakdown-multiflow-plan-then-implement.yaml"
# OpenStartup root: preflight/<this> → task/ → tools/ → resources/ → openteam/
# → test/ → OpenStartup/.
OPENSTARTUP_PATH = _HERE.parents[5]
# Templates dir colocated with the test resources (same as legacy smoke test).
TEMPLATES_DIR = _HERE.parents[1] / "task" / "configs" / "prompt_templates"
if not TEMPLATES_DIR.exists():
    # Legacy smoke test points templates_dir at OpenStartup's prompt_templates.
    TEMPLATES_DIR = OPENSTARTUP_PATH / "src" / "openteam" / "server" / "resources" / "prompt_templates"


def _load_topology(monkeypatch, tmp_path):
    """Bind required env vars + load + instantiate the YAML config.

    Mirrors the bootstrap done in `test_yaml_smoke_instantiate` so this
    preflight is self-contained.
    """
    monkeypatch.setenv("PROMPT_TEMPLATES_DIR", "prompt_templates")

    # Side-effect import to register Hydra targets (ClaudeCodeCLI, Dual, etc.)
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from rich_python_utils.config_utils import load_config, instantiate

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
    """Sanity: the YAML still parses + instantiates with the new workspace block."""
    topology = _load_topology(monkeypatch, tmp_path)
    assert topology is not None
    # Outer Dual is the root.
    assert type(topology).__name__ == "DualInferencer"


def test_outer_workspace_has_final_deliverables_flag_enabled(tmp_path, monkeypatch):
    """The outer Dual's workspace MUST have use_final_deliverables_folder=True
    after the YAML change. Catches accidental revert to the shorthand form."""
    topology = _load_topology(monkeypatch, tmp_path)

    ws = topology._workspace
    assert ws is not None, (
        "Outer Dual has no _workspace; expected an InferencerWorkspace "
        "constructed from the explicit `workspace:` block in the YAML."
    )
    assert getattr(ws, "use_final_deliverables_folder", False), (
        "Outer Dual workspace.use_final_deliverables_folder is falsey; "
        "the explicit `workspace:` block in the YAML may have regressed to "
        "the `workspace_root: <str>` shorthand which doesn't carry the flag."
    )


def test_outer_workspace_deliverables_dir_resolves(tmp_path, monkeypatch):
    """The flag must produce a non-None `deliverables_dir` path under outputs/."""
    topology = _load_topology(monkeypatch, tmp_path)

    ws = topology._workspace
    deliverables_dir = ws.deliverables_dir
    assert deliverables_dir is not None, (
        "deliverables_dir is None despite use_final_deliverables_folder=True; "
        "this indicates the workspace block was not actually instantiated as "
        "an InferencerWorkspace."
    )
    # Path convention: <root>/outputs/final_deliverables
    assert deliverables_dir.endswith(
        "outputs/final_deliverables"
    ) or deliverables_dir.endswith("outputs\\final_deliverables"), (
        f"deliverables_dir={deliverables_dir!r} does not end with the expected "
        f"'outputs/final_deliverables' suffix."
    )


def test_inferencer_workspace_class_is_used(tmp_path, monkeypatch):
    """Defensive: ensure the topology really uses InferencerWorkspace (not a
    surprise subclass), so the flag semantics described in the comments hold."""
    topology = _load_topology(monkeypatch, tmp_path)
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )
    assert isinstance(topology._workspace, InferencerWorkspace)

"""Unit + property tests for task/executor.py helpers + parser.

Run:
    python -m pytest OpenStartup/test/openteam/resources/tools/task/test_task_helpers.py -v
or:
    python OpenStartup/test/openteam/resources/tools/task/test_task_helpers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap sys.path so module imports work without conftest
_HERE = Path(__file__).resolve()
_OPENSTARTUP = _HERE.parents[5]
_REPO_ROOT = _OPENSTARTUP.parent
for _dep in [_OPENSTARTUP / "src",
             _REPO_ROOT / "AgentFoundation" / "src",
             _REPO_ROOT / "RichPythonUtils" / "src"]:
    p = str(_dep)
    if p not in sys.path:
        sys.path.insert(0, p)

import agent_foundation.common.configs.registered_targets  # noqa: F401
from rich_python_utils.config_utils import load_config, instantiate

from agent_foundation.resources.tools.task import executor as ex
from agent_foundation.resources.tools.task.slash_args import (
    parse_slash_args as _parse_slash_args,
    TASK_BOOL_FLAGS as _TASK_BOOL_FLAGS,
    TASK_MODE_ALIASES as _TASK_MODE_ALIASES,
)


# ──────────────────────────────────────────────────────────────────────
# Unit: camelCase → kebab-case (R2.4)
# ──────────────────────────────────────────────────────────────────────

def test_camel_to_kebab_basic():
    assert ex._camel_to_kebab("MultiFlowDual") == "multi-flow-dual"
    assert ex._camel_to_kebab("BTA") == "bta"
    assert ex._camel_to_kebab("PTI") == "pti"
    assert ex._camel_to_kebab("Dual") == "dual"

def test_camel_to_kebab_acronyms():
    # Acronym-aware regex correctly handles upper-then-upper-then-lower
    assert ex._camel_to_kebab("BTADual") == "bta-dual"
    assert ex._camel_to_kebab("ClaudeCodeCLI") == "claude-code-cli"

def test_camel_to_kebab_idempotent():
    s = "multi-flow-dual"
    assert ex._camel_to_kebab(s) == s.lower()


# ──────────────────────────────────────────────────────────────────────
# Unit: _resolve_agent_config (R2)
# ──────────────────────────────────────────────────────────────────────

def test_resolve_default_pti():
    kind, payload = ex._resolve_agent_config("pti")
    assert kind == "file"
    assert payload.name == "default.yaml"

def test_resolve_pti_alias_routes_to_preset():
    # R2.7 — PTI has no _target_ alias; bare "PTI" must hit the preset (not synthesize)
    kind, payload = ex._resolve_agent_config("PTI")
    assert kind == "file"
    assert payload.name == "default.yaml"

def test_resolve_kebab_preset():
    kind, payload = ex._resolve_agent_config("multi-flow-dual")
    assert kind == "file"
    assert payload.name == "multi-flow-dual.yaml"

def test_resolve_camel_alias_via_normalization():
    # R2.4 — MultiFlowDual normalizes to multi-flow-dual
    kind, payload = ex._resolve_agent_config("MultiFlowDual")
    assert kind == "file"
    assert payload.name == "multi-flow-dual.yaml"

def test_resolve_inline_json():
    spec = '{"_target_": "ClaudeCodeCLI"}'
    kind, payload = ex._resolve_agent_config(spec)
    assert kind == "inline"
    assert payload == {"_target_": "ClaudeCodeCLI"}

def test_resolve_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="not a known preset"):
        ex._resolve_agent_config("NotARealName")


# ──────────────────────────────────────────────────────────────────────
# Unit: _walk_replace_model (R3.2)
# ──────────────────────────────────────────────────────────────────────

def test_walk_replace_model_nested():
    cfg = {
        "_target_": "BTA",
        "breakdown_inferencer": {"model_name": "opus"},
        "worker_inferencers": {"__default__": {"model_name": "opus"}},
        "aggregator_inferencer": {"model_name": "opus"},
    }
    n = ex._walk_replace_model(cfg, "sonnet")
    assert n == 3
    assert cfg["breakdown_inferencer"]["model_name"] == "sonnet"
    assert cfg["worker_inferencers"]["__default__"]["model_name"] == "sonnet"
    assert cfg["aggregator_inferencer"]["model_name"] == "sonnet"

def test_walk_replace_model_idempotent():
    """P2 — applying twice with same value yields same cfg as applying once."""
    cfg1 = {"a": {"model_name": "opus"}, "b": [{"model_name": "opus"}]}
    cfg2 = {"a": {"model_name": "opus"}, "b": [{"model_name": "opus"}]}
    ex._walk_replace_model(cfg1, "sonnet")
    ex._walk_replace_model(cfg2, "sonnet")
    ex._walk_replace_model(cfg2, "sonnet")
    assert cfg1 == cfg2


# ──────────────────────────────────────────────────────────────────────
# Unit: _collapse_dual (R3.3)
# ──────────────────────────────────────────────────────────────────────

def test_collapse_dual_basic():
    cfg = {
        "_target_": "BTA",
        "workers": {
            "_target_": "Dual",
            "base_inferencer": {"_target_": "ClaudeCodeCLI", "model_name": "sonnet"},
            "review_inferencer": {"_target_": "ClaudeCodeCLI"},
        },
    }
    n = ex._collapse_dual(cfg)
    assert n == 1
    assert cfg["workers"]["_target_"] == "ClaudeCodeCLI"
    assert "review_inferencer" not in cfg["workers"]

def test_collapse_dual_full_fqn():
    cfg = {"x": {
        "_target_": "agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer.DualInferencer",
        "base_inferencer": {"_target_": "ClaudeCodeCLI"},
    }}
    n = ex._collapse_dual(cfg)
    assert n == 1
    assert cfg["x"]["_target_"] == "ClaudeCodeCLI"

def test_collapse_dual_no_duals_in_cfg_after_walk():
    """P3 — after _collapse_dual, no node has _target_ matching Dual variants."""
    cfg = {
        "_target_": "BTA",
        "breakdown_inferencer": {
            "_target_": "Dual",
            "base_inferencer": {"_target_": "ClaudeCodeCLI"},
        },
        "aggregator_inferencer": {
            "_target_": "Dual",
            "base_inferencer": {
                "_target_": "Dual",  # nested Dual
                "base_inferencer": {"_target_": "ClaudeCodeCLI"},
            },
        },
    }
    ex._collapse_dual(cfg)

    def has_dual(node):
        if isinstance(node, dict):
            if node.get("_target_") in ex._DUAL_TARGETS:
                return True
            return any(has_dual(v) for v in node.values())
        if isinstance(node, list):
            return any(has_dual(v) for v in node)
        return False

    assert not has_dual(cfg)


# ──────────────────────────────────────────────────────────────────────
# Unit: _topology_is_pti (R3.5)
# ──────────────────────────────────────────────────────────────────────

def test_topology_is_pti_true_for_pti_presets():
    assert ex._topology_is_pti(ex._resolve_agent_config("pti"))
    assert ex._topology_is_pti(ex._resolve_agent_config("pti-simple"))

def test_topology_is_pti_false_for_others():
    for spec in ("bta", "bta-dual", "dual", "single", "multi-flow", "multi-flow-dual"):
        assert not ex._topology_is_pti(ex._resolve_agent_config(spec)), f"false for {spec}"


# ──────────────────────────────────────────────────────────────────────
# Property: P4 — every preset is instantiable
# ──────────────────────────────────────────────────────────────────────

def test_property_p4_all_presets_instantiable():
    """P4 — every *.yaml in configs/ instantiates without raising."""
    configs_dir = ex._CONFIGS_DIR
    presets = sorted(configs_dir.glob("*.yaml"))
    assert len(presets) >= 6, "need at least 6 presets"
    for p in presets:
        cfg = load_config(str(p), overrides={
            "workspace_root": "/tmp/p4_test",
            "workspace_path": "/tmp/p4_test",
        })
        inst = instantiate(cfg)
        assert inst is not None, f"instantiate returned None for {p.name}"


# ──────────────────────────────────────────────────────────────────────
# Parser tests (R10 + Patch 3.2)
# ──────────────────────────────────────────────────────────────────────

def test_parser_kv_pair():
    got = _parse_slash_args("--agent-config bta what is 2+2", _TASK_BOOL_FLAGS)
    assert got == {"agent-config": "bta", "request": "what is 2+2"}

def test_parser_bool_flag_then_request():
    got = _parse_slash_args("--plan write a haiku", _TASK_BOOL_FLAGS)
    assert got == {"plan": True, "request": "write a haiku"}

def test_parser_quoted_request():
    got = _parse_slash_args('--agent-config pti --model sonnet "list 3 things"', _TASK_BOOL_FLAGS)
    assert got == {"agent-config": "pti", "model": "sonnet", "request": "list 3 things"}

def test_parser_repeatable_override():
    got = _parse_slash_args(
        "--override planner.model=opus --override exec.model=sonnet ping", _TASK_BOOL_FLAGS
    )
    assert got == {
        "override": ["planner.model=opus", "exec.model=sonnet"],
        "request": "ping",
    }

def test_parser_multiple_bool_flags():
    got = _parse_slash_args("--no-dual --analysis --multi-iter ping", _TASK_BOOL_FLAGS)
    assert got["no-dual"] is True
    assert got["analysis"] is True
    assert got["multi-iter"] is True
    assert got["request"] == "ping"

def test_parser_quoted_with_confirm():
    got = _parse_slash_args('--confirm "plan and approve me"', _TASK_BOOL_FLAGS)
    assert got == {"confirm": True, "request": "plan and approve me"}


# ──────────────────────────────────────────────────────────────────────
# tool.json schema (R1)
# ──────────────────────────────────────────────────────────────────────

def test_tool_json_schema():
    import json
    tool_path = ex._CONFIGS_DIR.parent / "tool.json"
    data = json.loads(tool_path.read_text(encoding="utf-8"))
    assert data["name"] == "task"
    assert data["agent_enabled"] is True
    assert data["slash_enabled"] is True
    assert data["dev_mode_only"] is False
    assert data["asynchronous"] is True
    assert data["is_bridge"] is True
    assert data["executor"] == "agent_foundation.resources.tools.task.executor:execute"
    param_names = {p["name"] for p in data["parameters"]}
    expected = {"request", "--plan", "--execute", "--full", "--confirm",
                "--agent-config", "--override", "--model", "--no-dual",
                "--analysis", "--multi-iter", "--max-iterations",
                "--resume", "--in-place", "--copy-workspace", "--use-plan"}
    assert expected.issubset(param_names), f"missing: {expected - param_names}"


# ──────────────────────────────────────────────────────────────────────
# Dispatcher backward-compat (R1.4 — slash_enabled formula)
# ──────────────────────────────────────────────────────────────────────

def test_slash_enabled_formula_mock_task():
    """mock_task has agent_enabled=False → slash_enabled defaults to True (unchanged)."""
    tool_data = {"agent_enabled": False}
    slash_enabled = tool_data.get("slash_enabled", not tool_data.get("agent_enabled", True))
    assert slash_enabled is True

def test_slash_enabled_formula_create_role():
    """create_role has no agent_enabled field → defaults to True → slash_enabled defaults to False (unchanged)."""
    tool_data = {}
    slash_enabled = tool_data.get("slash_enabled", not tool_data.get("agent_enabled", True))
    assert slash_enabled is False

def test_slash_enabled_formula_task():
    """/task explicit slash_enabled=True with agent_enabled=True → True."""
    tool_data = {"agent_enabled": True, "slash_enabled": True}
    slash_enabled = tool_data.get("slash_enabled", not tool_data.get("agent_enabled", True))
    assert slash_enabled is True


# ──────────────────────────────────────────────────────────────────────
# Phase 1 additions: _run_topology + workspace hint + template_feed via --override
# ──────────────────────────────────────────────────────────────────────

def test_resolve_workspace_respects_safe_dispatcher_hint(tmp_path=None):
    """R1.3 — when session_context provides a working_dir under /tasks/ or /_runtime/,
    _resolve_workspace SHOULD use it without re-allocating."""
    import tempfile
    safe_root = Path(tempfile.mkdtemp()) / "tasks" / "test"
    safe_root.mkdir(parents=True)
    sc = {"working_dir": str(safe_root)}
    got = ex._resolve_workspace(sc, "test-id")
    assert Path(got).resolve() == safe_root.resolve(), \
        f"expected {safe_root}, got {got}"

def test_resolve_workspace_unsafe_hint_falls_through(monkeypatch, tmp_path):
    """R1.3 — when session_context's working_dir does NOT look like a per-task subdir
    (e.g. server source dir), _resolve_workspace should allocate a fresh one."""
    monkeypatch.setenv("OPENTEAM_RUNTIME_DIR", str(tmp_path))
    sc = {"working_dir": "/some/server/dir"}
    got = ex._resolve_workspace(sc, "test-id")
    posix = Path(got).as_posix()
    # Falls through to _allocate_workspace which creates under _runtime/tasks/task/
    assert "/_runtime/" in posix or "/tasks/" in posix, f"expected fallback, got {posix}"

def test_resolve_workspace_no_hint_allocates(monkeypatch, tmp_path):
    """R1.3 — no working_dir hint → allocate fresh via shared helper."""
    monkeypatch.setenv("OPENTEAM_RUNTIME_DIR", str(tmp_path))
    got = ex._resolve_workspace({}, "test-id-no-hint")
    posix = Path(got).as_posix()
    assert "/tasks/" in posix, f"expected tasks/ in path, got {posix}"

def test_template_feed_via_override_reaches_inferencer():
    """A1#6 / R3.4 — proves --template-feed flag is unnecessary: BTA's
    template_extra_feed is reachable via dotted-key --override."""
    import asyncio
    bta_yaml = ex._CONFIGS_DIR / "bta.yaml"
    # Note: pre-2026-05-05 this passed `workspace_root: "/tmp/..."` to satisfy
    # the (now-removed) BTA convenience attrib. Workspace is not needed for
    # this template_extra_feed assertion (only for save_results), so it's
    # omitted entirely — keeps the test focused on what it actually verifies.
    cfg = load_config(str(bta_yaml), overrides={
        "template_extra_feed.role_name": "TestRole",
        "template_extra_feed.role_doc_path": "/some/path",
    })
    inst = instantiate(cfg)
    assert inst.template_extra_feed.get("role_name") == "TestRole", \
        f"expected TestRole, got {inst.template_extra_feed}"
    assert inst.template_extra_feed.get("role_doc_path") == "/some/path"


if __name__ == "__main__":
    # Allow running directly without pytest
    import traceback
    failed = []
    g = dict(globals())
    for name, fn in g.items():
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK   {name}")
            except Exception as e:
                print(f"  FAIL {name}: {type(e).__name__}: {e}")
                failed.append(name)
    print(f"\n{len(failed)} failed of {sum(1 for n in g if n.startswith('test_') and callable(g[n]))}")
    sys.exit(1 if failed else 0)

"""Preflight tests for the planning-complexity `--config` presets.

Covers the new presets added by the complexity-presets plan:
  - `breakdown.yaml`  — Dual{BTA{adaptive workers}} (coverage)
  - `multiflow-plan.yaml`   — root MultiFlowDual (diversity)
  - `full-plan`/`pti`/`multiflow` aliases via `_CONFIG_ALIASES`
  - `--plan` is a clean no-op for plan-only presets (no enable_implementation toggle)

Resolution + structural checks are pure (no LLM, no heavy imports). The
instantiation checks call `load_config`/`instantiate` (object construction only,
no inference) and skip gracefully if backend deps can't be imported.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_foundation.resources.tools.task.executor import (
    _CONFIG_ALIASES,
    _CONFIGS_DIR,
    _config_supports_implementation,
    _disable_aggregation,
    _extract_result_text,
    _resolve_agent_config,
)

pytestmark = pytest.mark.preflight


# ---------------------------------------------------------------------------
# Files exist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["breakdown", "multiflow-plan"])
def test_new_preset_file_exists(name):
    assert (_CONFIGS_DIR / f"{name}.yaml").is_file(), (
        f"{name}.yaml missing from {_CONFIGS_DIR}"
    )


# ---------------------------------------------------------------------------
# Resolution (pure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec,expected_filename",
    [
        ("breakdown", "breakdown.yaml"),
        ("multiflow-plan", "multiflow-plan.yaml"),
        ("multiflow", "multiflow-plan.yaml"),              # alias
        ("full-plan", "breakdown-multiflow-plan.yaml"),  # alias
        ("breakdown-multiflow-plan", "breakdown-multiflow-plan.yaml"),  # back-compat
        ("default", "default.yaml"),
        ("pti", "default.yaml"),                      # alias
    ],
)
def test_resolution(spec, expected_filename):
    kind, path = _resolve_agent_config(spec)
    assert kind == "file", f"{spec} should resolve to a file, got {kind}"
    assert Path(path).name == expected_filename, (
        f"{spec} resolved to {Path(path).name}, expected {expected_filename}"
    )


def test_conversation_alias_registered():
    # The alias is wired now; it resolves once disabled.yaml lands (Phase 2).
    assert _CONFIG_ALIASES.get("conversation") == "disabled"
    if (_CONFIGS_DIR / "disabled.yaml").is_file():
        kind, path = _resolve_agent_config("conversation")
        assert Path(path).name == "disabled.yaml"


def test_unknown_config_raises():
    with pytest.raises(ValueError):
        _resolve_agent_config("nonexistent-preset-xyz")


# ---------------------------------------------------------------------------
# `--plan` no-op detection
# ---------------------------------------------------------------------------

def test_plan_only_presets_do_not_support_implementation():
    for name in ("breakdown", "multiflow-plan", "breakdown-multiflow-plan"):
        src = ("file", _CONFIGS_DIR / f"{name}.yaml")
        assert _config_supports_implementation(src) is False, (
            f"{name}.yaml is plan-only; --plan must be a no-op (no enable_implementation toggle)"
        )


def test_default_supports_implementation():
    src = ("file", _CONFIGS_DIR / "default.yaml")
    assert _config_supports_implementation(src) is True, (
        "default.yaml is the full PTI plan+implement config; --plan should swap/toggle"
    )


# ---------------------------------------------------------------------------
# Structural YAML shape (pure parse — no instantiation)
# ---------------------------------------------------------------------------

def _load_yaml(name):
    return yaml.safe_load((_CONFIGS_DIR / f"{name}.yaml").read_text())


def test_breakdown_yaml_shape():
    cfg = _load_yaml("breakdown")
    assert cfg["_target_"] == "Dual"
    base = cfg["base_inferencer"]
    assert base["_target_"] == "BTA"
    assert base["task_type_arg_name"] == "worker_type"
    wf = base["worker_inferencers"]
    assert isinstance(wf, dict)
    assert set(wf.keys()) == {"leaf", "breakdown", "multiflow", "_default"}
    assert wf["_default"] == "leaf"  # string alias preserved (config walker _DATA_KEYS)
    # The breakdown_inferencer gates the classification block.
    assert (
        base["breakdown_inferencer"]["template_extra_feed"][
            "include_worker_type_classification"
        ]
        is True
    )


def test_multiple_yaml_shape():
    cfg = _load_yaml("multiflow-plan")
    # Root is MFDual directly — NOT wrapped in an outer Dual (MFDual IS a Dual).
    assert cfg["_target_"] == "MultiFlowDual"
    assert cfg.get("winner_pick") is True
    assert "flow_configs" in cfg


def _assert_num_flows_matches(cfg, name):
    """The _repeat_ distribution requires len(flow_inferencers) == num_flows.

    ``num_flows`` may be a literal int, or DERIVED via
    ``${len:${_params.flow_inferencers}}`` — the derived form guarantees the
    match by construction, so both satisfy the invariant.
    """
    num_flows = cfg["_params"]["num_flows"]
    flow_inf = cfg["_params"]["flow_inferencers"]
    if isinstance(num_flows, int):
        assert len(flow_inf) == num_flows, (
            f"{name}: len(flow_inferencers)={len(flow_inf)} != num_flows={num_flows}; "
            "the _repeat_ distribution would raise ValueError at load."
        )
    else:
        assert (
            isinstance(num_flows, str)
            and "len:" in num_flows
            and "flow_inferencers" in num_flows
        ), (
            f"{name}: num_flows={num_flows!r} is neither an int matching "
            "len(flow_inferencers) nor the ${len:${_params.flow_inferencers}} derivation."
        )


def test_multiple_num_flows_matches_flow_inferencers():
    _assert_num_flows_matches(_load_yaml("multiflow-plan"), "multiflow-plan.yaml")


def test_breakdown_num_flows_matches_flow_inferencers():
    _assert_num_flows_matches(_load_yaml("breakdown"), "breakdown.yaml")


# ---------------------------------------------------------------------------
# Instantiation (object construction only — no inference). Skips if backend
# deps are unavailable in the test environment.
# ---------------------------------------------------------------------------

def _instantiate(name, tmp_path):
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    import agent_foundation.resources as _af_res
    from rich_python_utils.config_utils import instantiate, load_config

    templates_dir = Path(_af_res.__file__).parent / "prompt_templates"
    cfg = load_config(
        str(_CONFIGS_DIR / f"{name}.yaml"),
        overrides={
            "_target_path": str(tmp_path),
            "templates_dir": str(templates_dir),
            "_params.workspace_root": str(tmp_path / "ws"),
        },
    )
    return instantiate(cfg)


def test_breakdown_instantiates(tmp_path):
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )
        root = _instantiate("breakdown", tmp_path)
    except ImportError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"backend deps unavailable: {exc}")

    assert isinstance(root, DualInferencer)
    assert isinstance(root.base_inferencer, BreakdownThenAggregateInferencer)
    wf = root.base_inferencer.worker_inferencers
    assert isinstance(wf, dict)
    # Keys preserved; _default stays a string alias.
    assert {"leaf", "breakdown", "multiflow", "_default"} <= set(wf.keys())
    assert wf["_default"] == "leaf"


def test_multiple_instantiates_as_root_mfdual(tmp_path):
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
            MultiFlowDualInferencer,
        )
        root = _instantiate("multiflow-plan", tmp_path)
    except ImportError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"backend deps unavailable: {exc}")

    assert isinstance(root, MultiFlowDualInferencer), (
        f"multiple.yaml must instantiate as root MFDual, got {type(root).__name__}"
    )
    assert len(root.flow_configs) == 3


# ---------------------------------------------------------------------------
# No-aggregate helpers (pure)
# ---------------------------------------------------------------------------

def test_extract_result_text_keeps_all_multi_outputs():
    """Multi-element tuples must be serialized in full, NOT dropped to [0]."""
    out = _extract_result_text(("alpha", "beta", "gamma"))
    assert "alpha" in out and "beta" in out and "gamma" in out
    assert "### Worker 1" in out and "### Worker 3" in out


def test_extract_result_text_single_tuple_unwraps():
    assert _extract_result_text(("solo",)) == "solo"


def test_extract_result_text_empty_tuple():
    assert _extract_result_text(()) == ""


def test_disable_aggregation_sets_flags_on_bta_and_mfdual():
    cfg = {
        "_target_": "Dual",
        "base_inferencer": {
            "_target_": "BTA",
            "worker_inferencers": {"multiflow": {"_target_": "MultiFlowDual"}},
        },
    }
    n = _disable_aggregation(cfg)
    assert n == 2
    assert cfg["base_inferencer"]["disable_aggregator"] is True
    assert (
        cfg["base_inferencer"]["worker_inferencers"]["multiflow"][
            "multi_flow_disable_aggregator"
        ]
        is True
    )

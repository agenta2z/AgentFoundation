"""C8 — migrated ``breakdown-multiflow-plan.yaml`` uses ``worker_inferencers`` and the real
BTA class wraps it as a lazy (fresh-per-subtask) ``LazyConfigFactory``.

Proven without a full-topology / LLM load: the worker recipe is *captured* lazily (the inner
MultiFlowDual is only instantiated when the factory is *called* per subtask), so asserting the
field is a ``LazyConfigFactory`` pointing at ``MultiFlowDual`` is the migration's load-bearing
guarantee.
"""

from pathlib import Path

import yaml
from omegaconf import OmegaConf

import agent_foundation.common.configs  # noqa: F401 — registers BTA/MultiFlowDual aliases
from rich_python_utils.config_utils import instantiate
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory

_PLAN_YAML = (
    Path(__file__).resolve().parents[4]
    / "src/agent_foundation/resources/tools/task/configs/breakdown-multiflow-plan.yaml"
)


def test_migrated_yaml_uses_worker_inferencers():
    """The shipped plan config declares the MFDual worker via ``worker_inferencers:
    {_target_: MultiFlowDual}`` — not the old ``worker_factory: {_factory_: ...}``."""
    doc = yaml.safe_load(_PLAN_YAML.read_text())
    bta = doc["base_inferencer"]
    assert "worker_factory" not in bta, "old worker_factory key should be gone"
    wi = bta["worker_inferencers"]
    assert wi["_target_"] == "MultiFlowDual"
    # The dispatch flags + the _repeat_ flow recipe survived the rename verbatim.
    assert wi["winner_pick"] is True
    assert wi["reviewer_strategy"] == "runner_up"
    assert wi["fixer_strategy"] == "winner"
    assert wi["flow_configs"][0]["_repeat_"] == "${_params.num_flows}"


def test_real_bta_wraps_worker_inferencers_lazily():
    """Instantiating the real BTA with ``worker_inferencers: {_target_: MultiFlowDual}``
    yields a LazyConfigFactory (fresh per subtask), not an eager shared instance."""
    cfg = OmegaConf.create(
        {
            "_target_": "BTA",
            "disable_aggregator": True,
            "worker_inferencers": {
                "_target_": "MultiFlowDual",
                "propagate_runtime_input": True,
                "winner_pick": True,
                "flow_configs": [{"input": "t", "max_dynamic_steps": 1}],
            },
        }
    )
    bta = instantiate(cfg)
    assert isinstance(bta.worker_inferencers, LazyConfigFactory)
    assert "MultiFlowDual" in bta.worker_inferencers.target


def test_worker_inferencers_factory_yields_fresh_correctly_sized_mfdual():
    """C1 regression: the tests above only assert the field *is* a LazyConfigFactory; this proves
    that CALLING it (exactly as BTA does per subtask) yields a FRESH, correctly-sized MFDual with
    distinct flow instances and no inner sharing across calls — the factory contract the whole
    factory-model decision rests on.

    Uses the production loader (``load_config`` resolves ``_params`` and expands the ``_repeat_``
    flow recipe) with the model overridden to a cheap alias so no CLI/LLM setup is needed.
    """
    from rich_python_utils.config_utils._instantiate import load_config

    cfg = load_config(
        str(_PLAN_YAML),
        overrides={
            "_params": {
                "workspace_root": "/tmp/c1_wsroot",
                "default_inferencer": "MockWorker",
            }
        },
    )
    dual = instantiate(cfg)
    bta = dual.base_inferencer
    assert isinstance(bta.worker_inferencers, LazyConfigFactory)

    w1 = bta.worker_inferencers()
    w2 = bta.worker_inferencers()

    # Correctly sized: num_flows=2 (from _params, expanded at load), re-instantiated per call.
    assert type(w1).__name__ == "MultiFlowDualInferencer"
    assert len(w1.flow_configs) == 2
    # Fresh MFDual per call, with NO shared inner instances across calls.
    assert w1 is not w2
    assert (
        w1.flow_configs[0]["initial_inferencer"]
        is not w2.flow_configs[0]["initial_inferencer"]
    )
    # Distinct flow instances within one worker (object-identity = flow ID holds).
    assert (
        w1.flow_configs[0]["initial_inferencer"]
        is not w1.flow_configs[1]["initial_inferencer"]
    )

"""BTA ``worker_inferencers`` assigned a *bare inferencer instance* in Python is auto-wrapped
in a fresh-copy factory (``_PrototypeCloneFactory``), so each subtask gets its own worker —
matching the YAML/``LazyConfigFactory`` path. This is an INDEPENDENT code path: it fires only
for a bare ``InferencerBase`` in a field carrying ``metadata={"lazy_config_factory": True}``
(today: ``worker_inferencers``); config/callable/list/dict worker sources are untouched.
"""

import agent_foundation.common.configs  # noqa: F401 — registers aliases
from omegaconf import OmegaConf

from rich_python_utils.config_utils._instantiate import instantiate
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (  # noqa: E501
    BreakdownThenAggregateInferencer as BTA,
)
from agent_foundation.common.inferencers.inferencer_base import _PrototypeCloneFactory


def _dual():
    """A real (Identifiable) inferencer with a nested Identifiable sub-inferencer."""
    return instantiate(
        OmegaConf.create(
            {
                "_target_": "Dual",
                "base_inferencer": {"_target_": "MockWorker"},
                "review_inferencer": {"_target_": "MockWorker"},
            }
        )
    )


def test_bare_instance_is_wrapped_as_factory():
    bta = BTA(worker_inferencers=_dual())
    assert isinstance(bta.worker_inferencers, _PrototypeCloneFactory)


def test_factory_yields_fresh_distinct_workers():
    bta = BTA(worker_inferencers=_dual())
    # Both call conventions the resolver might use:
    w1 = bta.worker_inferencers(sub_query="q", index=0)
    w2 = bta.worker_inferencers()
    assert type(w1).__name__ == "DualInferencer"
    assert w1 is not w2                 # distinct objects
    assert w1.id != w2.id               # distinct debuggable ids (fresh per call)
    assert w1.base_inferencer is not w2.base_inferencer  # deep-independent


def test_callable_path_unaffected():
    cb = lambda sub_query, index: object()  # noqa: E731
    assert BTA(worker_inferencers=cb).worker_inferencers is cb


def test_list_path_unaffected():
    lst = [object(), object()]
    assert BTA(worker_inferencers=lst).worker_inferencers is lst


def test_dict_of_factories_path_unaffected():
    dct = {"_default": (lambda sub_query, index: object())}
    assert BTA(worker_inferencers=dct, task_type_arg_name="kind").worker_inferencers is dct


def test_yaml_config_path_stays_lazy_config_factory():
    ybta = instantiate(
        OmegaConf.create(
            {
                "_target_": "BTA",
                "disable_aggregator": True,
                "worker_inferencers": {"_target_": "MockWorker"},
            }
        )
    )
    assert isinstance(ybta.worker_inferencers, LazyConfigFactory)
    assert not isinstance(ybta.worker_inferencers, _PrototypeCloneFactory)

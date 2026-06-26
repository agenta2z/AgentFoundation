"""BTA ``worker_inferencers`` — the single worker-source field accepts every shape.

After the 3→1 consolidation, ``worker_inferencers`` is the ONE worker source. It accepts:
a callable ``(sub_query, index) -> Inferencer``; a dict-of-factories keyed by task_type
(with ``_default``); a config recipe (wrapped as a ``LazyConfigFactory`` by the RPU walker
via the ``lazy_config_factory`` metadata); or a pre-built list (round-robin). The legacy
``worker_factory`` and ``workers`` fields are gone. The end-to-end fresh-per-subtask fan-out
(call the factory → fresh, correctly-sized worker) is exercised by
``test_bta_worker_inferencers_yaml.py`` (C1).
"""

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (  # noqa: E501
    BreakdownThenAggregateInferencer as BTA,
)


def _factory(sub_query, index):
    return object()


def test_worker_inferencers_accepts_callable():
    bta = BTA(worker_inferencers=_factory)
    assert bta.worker_inferencers is _factory


def test_worker_inferencers_accepts_dict_of_factories():
    bta = BTA(worker_inferencers={"_default": _factory}, task_type_arg_name="kind")
    assert isinstance(bta.worker_inferencers, dict)
    assert "_default" in bta.worker_inferencers


def test_worker_inferencers_accepts_static_list():
    workers = [object(), object()]
    bta = BTA(worker_inferencers=workers)
    assert bta.worker_inferencers == workers


def test_worker_inferencers_defaults_to_none():
    # No worker source set is fine (e.g. MFI sets it internally post-init).
    assert BTA().worker_inferencers is None


def test_legacy_worker_fields_removed():
    # The merged-away fields are gone — one source of truth, no shadowing footgun.
    bta = BTA(worker_inferencers=_factory)
    assert not hasattr(bta, "worker_factory")
    assert not hasattr(bta, "workers")

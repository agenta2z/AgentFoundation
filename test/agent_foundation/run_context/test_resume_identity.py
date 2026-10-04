"""Canonical resume identities (plan v8 §5.12, P8 commit 3)."""

from __future__ import annotations

import enum
import functools
import itertools

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
    make_conflict_aware_prompt_builder,
)
from agent_foundation.common.inferencers.inferencer_base import (
    _FreshCloneFactory,
    InferencerBase,
)
from agent_foundation.common.inferencers.run_context.resume_identity import (
    identity_bytes,
    identity_digest,
    identity_tree,
    ResumeIdentityUnavailableError,
)
from attr import attrib, attrs
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory


@attrs(slots=False)
class _Leaf(InferencerBase):
    response: str = attrib(default="ok", kw_only=True)
    api_key: str = attrib(default="", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self.response


def _worker(sub_query, index):
    return _Leaf()


def _bta(**kwargs):
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Leaf(response="1. q0"),
        worker_inferencers=_worker,
        breakdown_format="numbered_list",
        **kwargs,
    )


class _Color(enum.Enum):
    RED = "red"


class _Supplied:
    """A value whose ``repr`` changes on every call but which supplies its identity."""

    _counter = itertools.count()

    def __repr__(self):
        return f"<supplied {next(self._counter)}>"

    def resume_identity(self):
        return {"kind": "supplied", "version": 1}


class _Opaque:
    def __repr__(self):
        return "<always the same>"


def test_strings_and_bytes_are_their_own_bytes():
    assert identity_bytes("héllo") == "héllo".encode("utf-8")
    assert identity_bytes(b"\x00\x01") == b"\x00\x01"


def test_json_values_are_canonical():
    assert identity_bytes({"b": [1, 2.5], "a": None}) == b'{"a":null,"b":[1,2.5]}'
    assert identity_digest({"a": 1, "b": 2}) == identity_digest({"b": 2, "a": 1})
    assert identity_tree({3, 1, 2}) == identity_tree({2, 3, 1})
    assert identity_tree((1, "x")) == identity_tree([1, "x"])


def test_enums_types_functions_and_partials_have_names():
    assert identity_tree(_Color.RED)["value"] == "red"
    assert identity_tree(_Leaf) == {"type": f"{__name__}:_Leaf"}
    assert identity_tree(_worker) == {"callable": f"{__name__}:_worker"}
    partial = functools.partial(_worker, index=3)
    assert identity_tree(partial)["keywords"] == {"index": 3}


def test_lambdas_closures_and_opaque_objects_have_no_identity():
    def local():
        pass

    for value, path in (
        (lambda: None, "value"),
        (local, "value"),
        (_Opaque(), "value"),
        ([1, _Opaque()], "value[1]"),
    ):
        with pytest.raises(
            ResumeIdentityUnavailableError, match=path.replace("[", r"\[")
        ):
            identity_tree(value)


def test_repr_is_never_used():
    assert identity_digest(_Supplied()) == identity_digest(_Supplied())
    with pytest.raises(ResumeIdentityUnavailableError, match="_Opaque"):
        identity_digest(_Opaque())


def test_an_inferencer_is_its_class_and_semantic_configuration():
    first, second = _bta(max_breakdown=3), _bta(max_breakdown=3)
    assert first.id != second.id
    assert identity_digest(first) == identity_digest(second)
    assert identity_digest(_bta(max_breakdown=4)) != identity_digest(first)


def test_scheduling_and_placement_do_not_change_identity(tmp_path):
    base = identity_digest(_bta())
    for kwargs in (
        {"max_concurrency": 9},
        {"group_max_concurrency": {"g": 1}},
        {"max_retry": 5},
        {"checkpoint_dir": str(tmp_path)},
        {"enable_result_save": True, "resume_with_saved_results": True},
    ):
        assert identity_digest(_bta(**kwargs)) == base, kwargs


def test_secrets_are_never_hashed():
    plain, keyed = _Leaf(), _Leaf(api_key="sk-very-secret")
    assert identity_digest(plain) == identity_digest(keyed)
    assert b"sk-very-secret" not in identity_bytes(keyed)


def test_factories_supply_their_identity():
    lazy = LazyConfigFactory({"_target_": "Leaf", "response": "x"})
    assert identity_tree(lazy)["value"] == {"_target_": "Leaf", "response": "x"}
    clone = _FreshCloneFactory(_Leaf(response="p"))
    assert identity_digest(clone) == identity_digest(
        _FreshCloneFactory(_Leaf(response="p"))
    )
    assert identity_digest(clone) != identity_digest(_FreshCloneFactory(_Leaf()))
    builder = make_conflict_aware_prompt_builder()
    assert identity_tree(builder)["value"]["factory"] == "conflict_aware_aggregator"


def test_an_unidentifiable_stage_names_its_path():
    bta = _bta(aggregator_prompt_builder=lambda *a, **k: "")
    with pytest.raises(
        ResumeIdentityUnavailableError, match=r"value\.aggregator_prompt_builder"
    ):
        identity_digest(bta)

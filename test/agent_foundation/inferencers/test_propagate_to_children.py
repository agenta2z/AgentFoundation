"""Tests for TemplatedInferencerBase._propagate_to_children() runtime context propagation.

Covers:
- Direct InferencerBase children (attrs fields)
- functools.partial children (worker_factory pattern)
- Dict containing mixed InferencerBase + partial + duck-typed callables
- List/tuple containing InferencerBase + duck-typed callables
- Duck-typed callable fallback (any callable with template_extra_feed dict attr)
- Parent-wins semantics (runtime context overrides yaml defaults)
- Multi-level nesting (3+ levels)
- Edge cases: empty feed, None children, non-matching attrs, frozen partial keywords

Note: After the TemplatedInferencerBase refactor, the template_extra_feed
field and _propagate_to_children logic live on TemplatedInferencerBase rather
than InferencerBase. The StubInferencer here inherits from
TemplatedInferencerBase so it has those fields/methods available for testing.
"""

import functools
from unittest.mock import MagicMock

import pytest
from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)


# ---------------------------------------------------------------------------
# Minimal concrete TemplatedInferencerBase for testing (must implement _infer).
# Inherits TemplatedInferencerBase so template_extra_feed + _propagate_to_children
# are available — these moved off InferencerBase in the templated-base refactor.
# ---------------------------------------------------------------------------

@attrs
class StubInferencer(TemplatedInferencerBase):
    child: "InferencerBase" = attrib(default=None)
    children_list: list = attrib(factory=list)
    children_dict: dict = attrib(factory=dict)
    worker_factory: object = attrib(default=None)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return inference_input


class DuckTypedCallable:
    """A callable wrapper that exposes template_extra_feed as a dict attr."""
    def __init__(self, **kwargs):
        self.template_extra_feed = kwargs.get("template_extra_feed", {})

    def __call__(self, *args, **kwargs):
        return "duck result"


class NonMatchingCallable:
    """A callable WITHOUT template_extra_feed — should be skipped."""
    def __call__(self):
        return "no feed"


class StringFeedCallable:
    """A callable with template_extra_feed as a string — should be skipped."""
    template_extra_feed = "not a dict"

    def __call__(self):
        return "string feed"


# ---------------------------------------------------------------------------
# Basic propagation
# ---------------------------------------------------------------------------

class TestBasicPropagation:
    def test_propagates_to_direct_child(self):
        child = StubInferencer()
        parent = StubInferencer(child=child)
        parent.template_extra_feed = {"role_name": "PGM", "key": "val"}

        parent._propagate_to_children()

        assert child.template_extra_feed["role_name"] == "PGM"
        assert child.template_extra_feed["key"] == "val"

    def test_no_propagation_when_feed_empty(self):
        child = StubInferencer()
        child.template_extra_feed = {"existing": "value"}
        parent = StubInferencer(child=child)
        parent.template_extra_feed = {}

        parent._propagate_to_children()

        assert child.template_extra_feed == {"existing": "value"}

    def test_no_propagation_to_none_child(self):
        parent = StubInferencer(child=None)
        parent.template_extra_feed = {"key": "val"}
        parent._propagate_to_children()  # should not raise

    def test_propagates_to_child_in_list(self):
        c1 = StubInferencer()
        c2 = StubInferencer()
        parent = StubInferencer(children_list=[c1, c2])
        parent.template_extra_feed = {"ctx": "data"}

        parent._propagate_to_children()

        assert c1.template_extra_feed["ctx"] == "data"
        assert c2.template_extra_feed["ctx"] == "data"

    def test_propagates_to_child_in_dict(self):
        c1 = StubInferencer()
        parent = StubInferencer(children_dict={"worker_0": c1})
        parent.template_extra_feed = {"ctx": "data"}

        parent._propagate_to_children()

        assert c1.template_extra_feed["ctx"] == "data"


# ---------------------------------------------------------------------------
# Parent-wins semantics
# ---------------------------------------------------------------------------

class TestParentWinsSemantics:
    def test_parent_overrides_child_default(self):
        child = StubInferencer()
        child.template_extra_feed = {"role_name": "default_from_yaml"}
        parent = StubInferencer(child=child)
        parent.template_extra_feed = {"role_name": "runtime_PGM"}

        parent._propagate_to_children()

        assert child.template_extra_feed["role_name"] == "runtime_PGM"

    def test_parent_adds_new_keys_preserving_child_only_keys(self):
        child = StubInferencer()
        child.template_extra_feed = {"child_only": "keep_me"}
        parent = StubInferencer(child=child)
        parent.template_extra_feed = {"parent_key": "new_val"}

        parent._propagate_to_children()

        assert child.template_extra_feed["child_only"] == "keep_me"
        assert child.template_extra_feed["parent_key"] == "new_val"

    def test_parent_wins_in_dict_child(self):
        child = StubInferencer()
        child.template_extra_feed = {"role_name": "old"}
        parent = StubInferencer(children_dict={"w": child})
        parent.template_extra_feed = {"role_name": "new"}

        parent._propagate_to_children()

        assert child.template_extra_feed["role_name"] == "new"

    def test_parent_wins_in_list_child(self):
        child = StubInferencer()
        child.template_extra_feed = {"role_name": "old"}
        parent = StubInferencer(children_list=[child])
        parent.template_extra_feed = {"role_name": "new"}

        parent._propagate_to_children()

        assert child.template_extra_feed["role_name"] == "new"


# ---------------------------------------------------------------------------
# functools.partial handling
# ---------------------------------------------------------------------------

class TestPartialPropagation:
    def test_top_level_partial(self):
        factory = functools.partial(StubInferencer, template_extra_feed={"yaml_key": "yaml_val"})
        parent = StubInferencer(worker_factory=factory)
        parent.template_extra_feed = {"runtime": "ctx"}

        parent._propagate_to_children()

        new_factory = parent.worker_factory
        assert isinstance(new_factory, functools.partial)
        merged = new_factory.keywords["template_extra_feed"]
        assert merged["runtime"] == "ctx"
        assert merged["yaml_key"] == "yaml_val"

    def test_partial_parent_wins(self):
        factory = functools.partial(StubInferencer, template_extra_feed={"key": "child_val"})
        parent = StubInferencer(worker_factory=factory)
        parent.template_extra_feed = {"key": "parent_val"}

        parent._propagate_to_children()

        merged = parent.worker_factory.keywords["template_extra_feed"]
        assert merged["key"] == "parent_val"

    def test_partial_in_dict(self):
        f1 = functools.partial(StubInferencer, template_extra_feed={})
        f2 = functools.partial(StubInferencer, template_extra_feed={"existing": "keep"})
        parent = StubInferencer(children_dict={"research": f1, "investigation": f2, "__default__": "research"})
        parent.template_extra_feed = {"role_name": "PGM"}

        parent._propagate_to_children()

        d = parent.children_dict
        assert d["research"].keywords["template_extra_feed"]["role_name"] == "PGM"
        assert d["investigation"].keywords["template_extra_feed"]["role_name"] == "PGM"
        assert d["investigation"].keywords["template_extra_feed"]["existing"] == "keep"
        assert d["__default__"] == "research"  # string preserved

    def test_partial_without_existing_feed(self):
        factory = functools.partial(StubInferencer)
        parent = StubInferencer(worker_factory=factory)
        parent.template_extra_feed = {"new_key": "new_val"}

        parent._propagate_to_children()

        merged = parent.worker_factory.keywords["template_extra_feed"]
        assert merged["new_key"] == "new_val"

    def test_partial_creates_fresh_instance_with_feed(self):
        factory = functools.partial(StubInferencer)
        parent = StubInferencer(worker_factory=factory)
        parent.template_extra_feed = {"ctx": "val"}

        parent._propagate_to_children()

        instance = parent.worker_factory()
        assert instance.template_extra_feed["ctx"] == "val"


# ---------------------------------------------------------------------------
# Duck-typed callable fallback
# ---------------------------------------------------------------------------

class TestDuckTypedCallable:
    def test_top_level_duck_typed(self):
        duck = DuckTypedCallable(template_extra_feed={"old": "val"})
        parent = StubInferencer(worker_factory=duck)
        parent.template_extra_feed = {"new": "ctx"}

        parent._propagate_to_children()

        assert duck.template_extra_feed["new"] == "ctx"
        assert duck.template_extra_feed["old"] == "val"

    def test_duck_typed_parent_wins(self):
        duck = DuckTypedCallable(template_extra_feed={"key": "child"})
        parent = StubInferencer(worker_factory=duck)
        parent.template_extra_feed = {"key": "parent"}

        parent._propagate_to_children()

        assert duck.template_extra_feed["key"] == "parent"

    def test_duck_typed_in_dict(self):
        duck = DuckTypedCallable()
        parent = StubInferencer(children_dict={"custom": duck})
        parent.template_extra_feed = {"ctx": "val"}

        parent._propagate_to_children()

        assert duck.template_extra_feed["ctx"] == "val"

    def test_duck_typed_in_list(self):
        duck = DuckTypedCallable()
        parent = StubInferencer(children_list=[duck])
        parent.template_extra_feed = {"ctx": "val"}

        parent._propagate_to_children()

        assert duck.template_extra_feed["ctx"] == "val"

    def test_non_matching_callable_skipped(self):
        obj = NonMatchingCallable()
        parent = StubInferencer(worker_factory=obj)
        parent.template_extra_feed = {"key": "val"}

        parent._propagate_to_children()  # should not raise
        assert not hasattr(obj, "template_extra_feed") or obj.template_extra_feed != {"key": "val"}

    def test_string_feed_callable_skipped(self):
        obj = StringFeedCallable()
        parent = StubInferencer(worker_factory=obj)
        parent.template_extra_feed = {"key": "val"}

        parent._propagate_to_children()

        assert obj.template_extra_feed == "not a dict"


# ---------------------------------------------------------------------------
# Multi-level nesting (3+ levels)
# ---------------------------------------------------------------------------

class TestMultiLevelNesting:
    def test_three_levels_direct(self):
        grandchild = StubInferencer()
        child = StubInferencer(child=grandchild)
        parent = StubInferencer(child=child)
        parent.template_extra_feed = {"role_name": "PGM"}

        # Level 1: parent → child
        parent._propagate_to_children()
        assert child.template_extra_feed["role_name"] == "PGM"

        # Level 2: child → grandchild (simulates child's _ainfer_single calling propagate)
        child._propagate_to_children()
        assert grandchild.template_extra_feed["role_name"] == "PGM"

    def test_three_levels_with_partial(self):
        inner_factory = functools.partial(StubInferencer)
        inner_bta = StubInferencer(worker_factory=inner_factory)
        outer_factory = functools.partial(StubInferencer, worker_factory=inner_factory)
        parent = StubInferencer(children_dict={"creation": outer_factory})
        parent.template_extra_feed = {"tools": "/path/to/tools"}

        # Level 1: parent propagates to partial in dict
        parent._propagate_to_children()
        outer_merged = parent.children_dict["creation"].keywords["template_extra_feed"]
        assert outer_merged["tools"] == "/path/to/tools"

        # Level 2: create instance from partial, propagate again
        outer_instance = parent.children_dict["creation"]()
        assert outer_instance.template_extra_feed["tools"] == "/path/to/tools"
        outer_instance._propagate_to_children()

        # Level 3: inner_factory was in outer's keywords, now in outer_instance.worker_factory
        inner_instance = outer_instance.worker_factory()
        assert inner_instance.template_extra_feed["tools"] == "/path/to/tools"


# ---------------------------------------------------------------------------
# Mixed dict with heterogeneous values
# ---------------------------------------------------------------------------

class TestMixedDict:
    def test_dict_with_inferencer_partial_duck_and_string(self):
        inf = StubInferencer()
        partial_f = functools.partial(StubInferencer)
        duck = DuckTypedCallable()
        parent = StubInferencer(children_dict={
            "direct": inf,
            "factory": partial_f,
            "wrapper": duck,
            "__default__": "direct",  # string — should be preserved
        })
        parent.template_extra_feed = {"ctx": "val"}

        parent._propagate_to_children()

        assert inf.template_extra_feed["ctx"] == "val"
        assert parent.children_dict["factory"].keywords["template_extra_feed"]["ctx"] == "val"
        assert duck.template_extra_feed["ctx"] == "val"
        assert parent.children_dict["__default__"] == "direct"

    def test_dict_preserves_non_changed_entries(self):
        inf = StubInferencer()
        parent = StubInferencer(children_dict={
            "worker": inf,
            "config": {"some": "data"},  # plain dict value, not an inferencer
            "count": 42,
        })
        parent.template_extra_feed = {"ctx": "val"}

        parent._propagate_to_children()

        assert inf.template_extra_feed["ctx"] == "val"
        assert parent.children_dict["config"] == {"some": "data"}
        assert parent.children_dict["count"] == 42


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_propagation_idempotent(self):
        child = StubInferencer()
        parent = StubInferencer(child=child)
        parent.template_extra_feed = {"key": "val"}

        parent._propagate_to_children()
        parent._propagate_to_children()  # second call

        assert child.template_extra_feed == {"key": "val"}

    def test_empty_dict_child(self):
        parent = StubInferencer(children_dict={})
        parent.template_extra_feed = {"key": "val"}
        parent._propagate_to_children()  # should not raise

    def test_empty_list_child(self):
        parent = StubInferencer(children_list=[])
        parent.template_extra_feed = {"key": "val"}
        parent._propagate_to_children()  # should not raise

    def test_none_in_list_skipped(self):
        child = StubInferencer()
        parent = StubInferencer(children_list=[None, child, None])
        parent.template_extra_feed = {"key": "val"}

        parent._propagate_to_children()

        assert child.template_extra_feed["key"] == "val"

    def test_non_inferencer_attrs_skipped(self):
        parent = StubInferencer()
        parent.template_extra_feed = {"key": "val"}
        parent.model_id = "test-model"  # string attr, not inferencer

        parent._propagate_to_children()  # should not modify model_id
        assert parent.model_id == "test-model"

    def test_large_feed_propagated(self):
        child = StubInferencer()
        parent = StubInferencer(child=child)
        parent.template_extra_feed = {f"key_{i}": f"val_{i}" for i in range(100)}

        parent._propagate_to_children()

        assert len(child.template_extra_feed) == 100
        assert child.template_extra_feed["key_99"] == "val_99"

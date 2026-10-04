"""A templated parent's feed and modes reach its children through the ctx (plan
v8 §5.9, B17, P5 c2).

Under a ctx, ``_propagate_to_children`` publishes the parent's own feed and
modes at its node for its descendants, which apply them at render time over
their own; no child instance or factory is rewritten, so a shared child never
accumulates another parent's feed. The precedence, highest first: an explicit
per-call ``extra_feed``, the nearest per-call ctx override, the nearest
ancestor's published feed (an outer ancestor over an inner one), the role
overlay, the child's definition. A scope barrier stops the walk. With no ctx the
parent pushes into its child instances, as before.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

import pytest
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.template_feed_scope import (
    publish_child_template_feed,
    TEMPLATE_EXTRA_FEED_SCOPE_HANDLE,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs

KINDS = ("sync", "async")


class _Templates:
    def __call__(
        self, key, *, active_template_root_space=None, master_version=None, **feed
    ):
        return {
            k: v for k, v in feed.items() if k not in ("input", "__template_space__")
        }


@attrs(slots=False)
class _Child(TemplatedInferencerBase):
    """Returns the feed it rendered with (the stub templates render the feed)."""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return inference_input

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return inference_input


@attrs(slots=False)
class _Parent(TemplatedInferencerBase):
    """A templated inferencer with a child: calls it at the ``child`` slot."""

    child: Any = attrib(default=None, kw_only=True)
    child_feed: Any = attrib(default=None, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.child.infer("q", run_context=self._rc_child("child"), **self._kw())

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return await self.child.ainfer(
            "q", run_context=self._rc_child("child"), **self._kw()
        )

    def _kw(self):
        return {} if self.child_feed is None else {"extra_feed": self.child_feed}


def _templated(cls, feed=None, modes=None, **kwargs):
    inf = cls(**kwargs)
    inf.template_manager = _Templates()
    inf.template_root_space = "plan"
    inf.template_key = "initial"
    inf.template_extra_feed = dict(feed or {})
    inf.modes = dict(modes or {})
    return inf


def _call(inf, kind, **kwargs):
    if kind == "sync":
        return inf.infer("go", **kwargs)
    return asyncio.run(inf.ainfer("go", **kwargs))


def _host():
    return RunContext.root()


@pytest.mark.parametrize("kind", KINDS)
def test_the_child_renders_with_the_parent_feed_and_modes_over_its_own(kind):
    child = _templated(_Child, feed={"k": "child", "c": 1}, modes={"deep_mode": False})
    parent = _templated(
        _Parent, feed={"k": "parent", "p": 1}, modes={"deep_mode": True}, child=child
    )
    rendered = _call(parent, kind, run_context=_host())
    assert (rendered["k"], rendered["c"], rendered["p"]) == ("parent", 1, 1)
    assert rendered["enable_deep_mode"] is True
    assert child.template_extra_feed == {"k": "child", "c": 1}
    assert child.modes == {"deep_mode": False}


def test_a_shared_child_never_accumulates_another_parents_feed():
    child = _templated(_Child)
    a = _templated(_Parent, feed={"from_a": 1}, child=child)
    b = _templated(_Parent, feed={"from_b": 1}, child=child)
    first = a.infer("go", run_context=_host())
    second = b.infer("go", run_context=_host())
    assert "from_a" in first and "from_b" not in first
    assert "from_b" in second and "from_a" not in second
    assert child.template_extra_feed == {}


def test_a_bare_call_no_longer_writes_the_parent_feed_into_the_child():
    child = _templated(_Child)
    parent = _templated(_Parent, feed={"p": 1}, child=child)
    assert parent.infer("go")["p"] == 1
    assert child.template_extra_feed == {}


def test_with_no_ctx_the_explicit_push_still_writes_the_child_instances():
    child = _templated(_Child, feed={"k": "child"})
    parent = _templated(_Parent, feed={"k": "parent"}, modes={"m": True}, child=child)
    parent._propagate_to_children()
    assert child.template_extra_feed == {"k": "parent"}
    assert child.modes == {"m": True}


def test_a_factory_child_is_never_rewritten_under_a_ctx():
    factory = functools.partial(_Child)
    parent = _templated(_Parent, feed={"p": 1})
    parent.child = factory
    token = enter_run(_host())
    try:
        parent._propagate_to_children()
    finally:
        exit_run(token)
    assert parent.child is factory


# -- precedence ---------------------------------------------------------------


def test_precedence_explicit_then_override_then_parent_then_role_then_definition():
    child = _templated(_Child, feed={"d": "definition", "r": "definition"})
    parent = _templated(
        _Parent,
        feed={"d": "parent", "r": "parent", "o": "parent", "e": "parent"},
        child=child,
        child_feed={"e": "explicit"},
    )
    host = _host()
    token = enter_run(host.child("child"))
    try:
        child.switch_role("reviewer", template_extra_feed={"r": "role", "x": "role"})
    finally:
        exit_run(token)
    token = enter_run(host)
    try:
        publish_child_template_feed(child, "child", {"o": "override", "e": "override"})
    finally:
        exit_run(token)
    rendered = parent.infer("go", run_context=host)
    assert {k: rendered[k] for k in "drxoe"} == {
        "d": "parent",
        "r": "parent",
        "x": "role",
        "o": "override",
        "e": "explicit",
    }


def test_an_outer_ancestor_wins_over_an_inner_one():
    child = _templated(_Child)
    inner = _templated(_Parent, feed={"k": "inner", "i": 1}, child=child)
    outer = _templated(_Parent, feed={"k": "outer", "o": 1}, child=inner)
    rendered = outer.infer("go", run_context=_host())
    assert (rendered["k"], rendered["i"], rendered["o"]) == ("outer", 1, 1)


def test_a_scope_barrier_stops_the_parent_feed():
    child = _templated(_Child)
    parent = _templated(_Parent, feed={"p": 1}, child=child)
    host = _host()
    host.child("child").handles.set(TEMPLATE_EXTRA_FEED_SCOPE_HANDLE, True)
    assert "p" not in parent.infer("go", run_context=host)

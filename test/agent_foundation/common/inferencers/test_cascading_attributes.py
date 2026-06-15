"""Tests for InferencerBase._CASCADING_ATTRIBUTES / _propagate_cascading_attributes.

Covers the debug_mode cascade (parent -> unset children, recursively), the
"explicit child value wins" rule, the no-cascade-from-unset-parent rule, the
runtime enable_debug_mode() trigger, list/dict children, the partial-child
boundary (factory children are NOT mutated; they inherit via YAML at
instantiation), and the (name, condition) tuple form.

The cascade is built on _for_each_child_inferencer (the attrs-field walker), so
children declared as attrs fields are discovered without each subclass
overriding _iter_child_inferencers. The construction-time trigger is
InferencerBase.__attrs_post_init__; StubInferencer does not override it, so the
cascade fires at construction (as it does for orchestrators that call super()).
"""

import functools

import pytest
from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)


@attrs
class StubInferencer(TemplatedInferencerBase):
    child: "InferencerBase" = attrib(default=None)
    children_list: list = attrib(factory=list)
    children_dict: dict = attrib(factory=dict)
    worker_factory: object = attrib(default=None)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return inference_input


@attrs
class CustomCascadeInferencer(StubInferencer):
    """Adds a (name, condition) tuple entry to exercise the conditional form."""
    model_id: str = attrib(default=None)
    _CASCADING_ATTRIBUTES = InferencerBase._CASCADING_ATTRIBUTES + [
        ("model_id", lambda parent_val, child_val: not child_val),
    ]


# ---------------------------------------------------------------------------
# Construction-time cascade (post_init)
# ---------------------------------------------------------------------------

class TestConstructionTimeCascade:
    def test_cascades_to_unset_direct_child(self):
        child = StubInferencer()
        assert child.debug_mode is None
        parent = StubInferencer(child=child, debug_mode=True)
        assert parent.debug_mode is True
        assert child.debug_mode is True  # inherited at construction

    def test_explicit_false_child_wins(self):
        child = StubInferencer(debug_mode=False)
        parent = StubInferencer(child=child, debug_mode=True)
        assert parent.debug_mode is True
        assert child.debug_mode is False  # explicit value preserved

    def test_explicit_true_child_unchanged_when_parent_false(self):
        child = StubInferencer(debug_mode=True)
        parent = StubInferencer(child=child, debug_mode=False)
        # parent False is non-None, so it cascades to None children only;
        # the child is explicitly True -> untouched.
        assert child.debug_mode is True

    def test_no_cascade_from_unset_parent(self):
        child = StubInferencer()
        parent = StubInferencer(child=child)  # both unset
        assert parent.debug_mode is None
        assert child.debug_mode is None

    def test_recurses_to_grandchild(self):
        grandchild = StubInferencer()
        child = StubInferencer(child=grandchild)
        parent = StubInferencer(child=child, debug_mode=True)
        assert child.debug_mode is True
        assert grandchild.debug_mode is True  # reached via recursion

    def test_cascades_to_list_children(self):
        c1, c2 = StubInferencer(), StubInferencer(debug_mode=False)
        parent = StubInferencer(children_list=[c1, c2], debug_mode=True)
        assert c1.debug_mode is True
        assert c2.debug_mode is False  # explicit wins

    def test_cascades_to_dict_children(self):
        c1 = StubInferencer()
        parent = StubInferencer(children_dict={"w0": c1}, debug_mode=True)
        assert c1.debug_mode is True


# ---------------------------------------------------------------------------
# Runtime toggle (enable_debug_mode / disable_debug_mode)
# ---------------------------------------------------------------------------

class TestRuntimeToggle:
    def test_enable_debug_mode_cascades(self):
        child = StubInferencer()
        parent = StubInferencer(child=child)  # both unset at construction
        assert child.debug_mode is None
        parent.enable_debug_mode()
        assert parent.debug_mode is True
        assert child.debug_mode is True

    def test_enable_debug_mode_respects_explicit_child(self):
        child = StubInferencer(debug_mode=False)
        parent = StubInferencer(child=child)
        parent.enable_debug_mode()
        assert parent.debug_mode is True
        assert child.debug_mode is False  # explicit wins

    def test_enable_debug_mode_reaches_grandchild(self):
        grandchild = StubInferencer()
        child = StubInferencer(child=grandchild)
        parent = StubInferencer(child=child)
        parent.enable_debug_mode()
        assert child.debug_mode is True
        assert grandchild.debug_mode is True


# ---------------------------------------------------------------------------
# Partial-child boundary (factory children NOT mutated here)
# ---------------------------------------------------------------------------

class TestPartialBoundary:
    def test_partial_child_not_mutated(self):
        factory = functools.partial(StubInferencer)
        parent = StubInferencer(worker_factory=factory, debug_mode=True)
        # on_partial returns None -> the factory is left as-is (mirrors the
        # workspace-propagation precedent). Instantiating it does NOT inherit
        # debug_mode from the parent at runtime (YAML cascade covers that path).
        assert parent.worker_factory is factory
        produced = parent.worker_factory()
        assert produced.debug_mode is None


# ---------------------------------------------------------------------------
# Conditional (name, condition) tuple form
# ---------------------------------------------------------------------------

class TestConditionalForm:
    def test_tuple_condition_cascades_when_falsy(self):
        child = CustomCascadeInferencer()  # model_id None (falsy) -> inherit
        parent = CustomCascadeInferencer(child=child, model_id="opus")
        assert child.model_id == "opus"

    def test_tuple_condition_respects_truthy_child(self):
        child = CustomCascadeInferencer(model_id="haiku")  # truthy -> keep
        parent = CustomCascadeInferencer(child=child, model_id="opus")
        assert child.model_id == "haiku"


# ---------------------------------------------------------------------------
# Real ConversationalInferencer: the user's target scenario
# ---------------------------------------------------------------------------

class TestConversationalInferencerCascade:
    """The CI's __attrs_post_init__ does NOT chain super(), so the
    construction-time cascade does not reach it — enable_debug_mode() is the
    reliable trigger (it cascades to base_inferencer via the override)."""

    def _make_ci(self, base):
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
            ConversationalInferencer,
        )
        return ConversationalInferencer(base_inferencer=base)

    def test_constructor_debug_mode_does_not_cascade_for_ci(self):
        base = StubInferencer()
        ci = self._make_ci(base)
        ci.debug_mode = True  # set after construction; CI has no post_init cascade
        assert base.debug_mode is None  # NOT cascaded by a bare attribute set

    def test_enable_debug_mode_cascades_to_backend(self):
        base = StubInferencer()
        ci = self._make_ci(base)
        assert base.debug_mode is None
        ci.enable_debug_mode()
        assert ci.debug_mode is True
        assert base.debug_mode is True  # reliable cascade via the override

"""
Property-based test for target conversion uniqueness.

Feature: meta-agent-workflow, Property 7: Target strategy conversion uniqueness

*For any* generated selector from the TargetStrategyConverter, the selector
SHALL match exactly one element in the source HTML snapshot. Selectors
matching zero or multiple elements SHALL be discarded.

**Validates: Requirements 3.3, 3.4**
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.models import TraceStep
from science_modeling_tools.automation.meta_agent.target_converter import (
    TargetStrategyConverter,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating HTML pages
# ---------------------------------------------------------------------------

# Attribute value characters (safe for HTML attributes)
_ATTR_CHARS = st.characters(
    whitelist_categories=("L", "N"),
    whitelist_characters=("-", "_"),
)

_ATTR_VALUE = st.text(alphabet=_ATTR_CHARS, min_size=1, max_size=20)

# Tag names to use for generated elements
_TAG_NAMES = st.sampled_from(["div", "span", "button", "input", "a", "p", "li"])


@st.composite
def html_element_attrs(draw):
    """Generate a dict of optional HTML attributes for an element."""
    attrs = {}
    if draw(st.booleans()):
        attrs["data-testid"] = draw(_ATTR_VALUE)
    if draw(st.booleans()):
        attrs["data-qa"] = draw(_ATTR_VALUE)
    if draw(st.booleans()):
        attrs["id"] = draw(_ATTR_VALUE)
    if draw(st.booleans()):
        attrs["class"] = draw(_ATTR_VALUE)
    if draw(st.booleans()):
        attrs["aria-label"] = draw(_ATTR_VALUE)
    if draw(st.booleans()):
        attrs["role"] = draw(st.sampled_from(["button", "link", "textbox", "dialog"]))
    return attrs


def _build_element(tag: str, attrs: dict, framework_id: str | None, text: str) -> str:
    """Build an HTML element string."""
    parts = [f"<{tag}"]
    if framework_id:
        parts.append(f' __id__="{framework_id}"')
    for k, v in attrs.items():
        parts.append(f' {k}="{v}"')
    if tag == "input":
        parts.append(" />")
        return "".join(parts)
    parts.append(f">{text}</{tag}>")
    return "".join(parts)


@st.composite
def html_page_with_target(draw):
    """
    Generate an HTML page containing a target element (with __id__) and
    0-5 sibling elements. Returns (html_string, framework_id).

    The target element always has a unique __id__. Sibling elements may
    share some attributes with the target to create realistic uniqueness
    challenges.
    """
    target_tag = draw(_TAG_NAMES)
    target_attrs = draw(html_element_attrs())
    target_text = draw(st.text(alphabet=_ATTR_CHARS, min_size=1, max_size=30))
    framework_id = "target_" + draw(
        st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=10)
    )

    target_el = _build_element(target_tag, target_attrs, framework_id, target_text)

    # Generate sibling elements (some may share attributes with target)
    num_siblings = draw(st.integers(min_value=0, max_value=5))
    siblings = []
    for i in range(num_siblings):
        sib_tag = draw(_TAG_NAMES)
        sib_attrs = draw(html_element_attrs())
        sib_text = draw(st.text(alphabet=_ATTR_CHARS, min_size=1, max_size=30))
        siblings.append(_build_element(sib_tag, sib_attrs, None, sib_text))

    body_content = "\n  ".join([target_el] + siblings)
    html = f"<html><body>\n  {body_content}\n</body></html>"
    return html, framework_id


# ---------------------------------------------------------------------------
# Property 7: Target strategy conversion uniqueness
# ---------------------------------------------------------------------------


class TestTargetUniquenessProperty:
    """
    Property 7: Target strategy conversion uniqueness

    *For any* generated selector from the TargetStrategyConverter, the
    selector SHALL match exactly one element in the source HTML snapshot.
    Selectors matching zero or multiple elements SHALL be discarded.

    **Validates: Requirements 3.3, 3.4**
    """

    @given(data=html_page_with_target())
    @settings(max_examples=200)
    def test_every_non_agent_selector_matches_exactly_one_element(self, data):
        """
        For any HTML page with a target element, every non-agent strategy
        produced by the converter matches exactly one element in the HTML.
        """
        html, framework_id = data
        converter = TargetStrategyConverter()

        step = TraceStep(
            action_type="click",
            target=f"__id__={framework_id}",
            html_before=html,
        )
        result = converter.convert(step)

        for spec in result.strategies:
            if spec.strategy == "agent":
                # Agent fallback is not a DOM selector — skip
                continue
            is_unique = converter._validate_uniqueness(spec, html)
            assert is_unique, (
                f"Strategy {spec.strategy!r} with value {spec.value!r} "
                f"does not match exactly one element in the HTML.\n"
                f"Framework ID: {framework_id}\n"
                f"HTML: {html[:500]}"
            )

    @given(data=html_page_with_target())
    @settings(max_examples=200)
    def test_result_always_has_at_least_one_strategy(self, data):
        """
        For any valid __id__ target with HTML, the converter always
        produces at least one strategy (either a stable selector or
        the agent fallback).
        """
        html, framework_id = data
        converter = TargetStrategyConverter()

        step = TraceStep(
            action_type="click",
            target=f"__id__={framework_id}",
            html_before=html,
        )
        result = converter.convert(step)

        assert len(result.strategies) >= 1, (
            f"Converter produced zero strategies for __id__={framework_id}"
        )

    @given(data=html_page_with_target())
    @settings(max_examples=200)
    def test_no_duplicate_strategy_types(self, data):
        """
        The converter should not produce duplicate strategy types in
        the output TargetSpecWithFallback.
        """
        html, framework_id = data
        converter = TargetStrategyConverter()

        step = TraceStep(
            action_type="click",
            target=f"__id__={framework_id}",
            html_before=html,
        )
        result = converter.convert(step)

        strategy_names = [s.strategy for s in result.strategies]
        assert len(strategy_names) == len(set(strategy_names)), (
            f"Duplicate strategy types found: {strategy_names}"
        )

"""
Property-based test for action type normalization.

Feature: meta-agent-workflow, Property 4: Action type normalization

*For any* raw log entry with an action type string (including agent-internal
names like "ElementInteraction.Click"), the TraceNormalizer SHALL produce a
TraceStep whose action_type is either a registered type in
ActionMetadataRegistry or is flagged as unrecognized.

**Validates: Requirements 2.1, 2.2, 2.3**
"""

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.models import TraceStep
from science_modeling_tools.automation.meta_agent.normalizer import (
    KNOWN_CANONICAL_TYPES,
    TraceNormalizer,
)
from webaxon.automation.meta_agent.web_normalizer_config import WEB_ACTION_TYPE_MAP


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Agent-internal action type names that should be mapped (web-specific)
_AGENT_INTERNAL_TYPES = list(WEB_ACTION_TYPE_MAP.keys())

# Canonical types that should pass through unchanged
_CANONICAL_TYPES = sorted(KNOWN_CANONICAL_TYPES)

# Strategy: pick from known agent-internal names
agent_internal_type_st = st.sampled_from(_AGENT_INTERNAL_TYPES)

# Strategy: pick from known canonical types
canonical_type_st = st.sampled_from(_CANONICAL_TYPES)

# Strategy: arbitrary text strings (may be unrecognized)
arbitrary_type_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=80,
)

# Strategy: mix of all three categories
any_action_type_st = st.one_of(
    agent_internal_type_st,
    canonical_type_st,
    arbitrary_type_st,
)


def _make_step(action_type: str) -> TraceStep:
    """Create a minimal TraceStep with the given action_type."""
    return TraceStep(action_type=action_type)


# ---------------------------------------------------------------------------
# Property 4: Action type normalization
# ---------------------------------------------------------------------------


class TestActionTypeNormalizationProperty:
    """
    Property 4: Action type normalization

    *For any* raw log entry with an action type string (including
    agent-internal names like "ElementInteraction.Click"), the
    TraceNormalizer SHALL produce a TraceStep whose action_type is either
    a registered type in ActionMetadataRegistry or is flagged as
    unrecognized.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @given(raw_type=any_action_type_st)
    @settings(max_examples=200)
    def test_normalized_type_is_canonical_or_flagged(self, raw_type: str):
        """
        For any raw action type, the normalized step's action_type is
        either a known canonical type or the step is flagged as
        unrecognized in its metadata.
        """
        normalizer = TraceNormalizer()
        step = _make_step(raw_type)
        result = normalizer.normalize_step(step)

        is_canonical = result.action_type in normalizer._canonical_types
        is_flagged = result.metadata.get("unrecognized_action_type", False)

        assert is_canonical or is_flagged, (
            f"Action type {raw_type!r} normalized to {result.action_type!r} "
            f"which is neither canonical nor flagged as unrecognized"
        )

    @given(raw_type=agent_internal_type_st)
    @settings(max_examples=100)
    def test_agent_internal_types_map_to_canonical(self, raw_type: str):
        """
        Agent-internal names (e.g. "ElementInteraction.Click") always
        map to a known canonical type and are never flagged when
        WEB_ACTION_TYPE_MAP is provided.
        """
        normalizer = TraceNormalizer(custom_type_map=WEB_ACTION_TYPE_MAP)
        step = _make_step(raw_type)
        result = normalizer.normalize_step(step)

        expected = WEB_ACTION_TYPE_MAP[raw_type]
        assert result.action_type == expected
        assert result.action_type in normalizer._canonical_types
        assert not result.metadata.get("unrecognized_action_type", False)

    @given(raw_type=canonical_type_st)
    @settings(max_examples=100)
    def test_canonical_types_pass_through_unflagged(self, raw_type: str):
        """
        Already-canonical types pass through unchanged and are not
        flagged as unrecognized.
        """
        normalizer = TraceNormalizer()
        step = _make_step(raw_type)
        result = normalizer.normalize_step(step)

        assert result.action_type == raw_type
        assert result.action_type in normalizer._canonical_types
        assert not result.metadata.get("unrecognized_action_type", False)

    @given(raw_type=any_action_type_st)
    @settings(max_examples=200)
    def test_normalize_action_type_idempotent(self, raw_type: str):
        """
        Normalizing the action type twice produces the same result as
        normalizing once (idempotency).
        """
        normalizer = TraceNormalizer()
        once = normalizer.normalize_action_type(raw_type)
        twice = normalizer.normalize_action_type(once)
        assert once == twice

    @given(
        raw_type=any_action_type_st,
        custom_map=st.dictionaries(
            keys=st.text(min_size=1, max_size=30),
            values=st.sampled_from(_CANONICAL_TYPES),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=200)
    def test_custom_type_map_respected(self, raw_type: str, custom_map: dict):
        """
        When a custom_type_map is provided, its mappings take effect and
        the property still holds: normalized type is canonical or flagged.
        """
        normalizer = TraceNormalizer(custom_type_map=custom_map)
        step = _make_step(raw_type)
        result = normalizer.normalize_step(step)

        is_canonical = result.action_type in normalizer._canonical_types
        is_flagged = result.metadata.get("unrecognized_action_type", False)

        assert is_canonical or is_flagged, (
            f"With custom map, {raw_type!r} → {result.action_type!r} "
            f"is neither canonical nor flagged"
        )

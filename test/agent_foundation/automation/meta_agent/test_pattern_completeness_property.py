"""
Property-based test for pattern extraction completeness.

Feature: meta-agent-workflow, Property 11: Pattern extraction completeness

For any AlignedTraceSet, the ExtractedPatterns SHALL account for every
position index — each index appears in exactly one of: deterministic_steps,
parameterizable_steps, variable_steps, optional_steps, branch_patterns
(as part of a branch), or loop_patterns (as part of a loop body).

**Validates: Requirements 5.1**
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.models import (
    AlignedPosition,
    AlignedTraceSet,
    AlignmentType,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.pattern_extractor import (
    PatternExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url"]
TARGETS = ["btn-ok", "search-box", "nav-link", "field-email"]


def _make_step(
    action_type: str,
    target: str = "default-target",
    args: Optional[Dict] = None,
) -> TraceStep:
    return TraceStep(action_type=action_type, target=target, args=args)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def aligned_trace_set_st(draw) -> AlignedTraceSet:
    """Generate a random AlignedTraceSet with various AlignmentType values.

    Generates positions with DETERMINISTIC, PARAMETERIZABLE, VARIABLE, and
    OPTIONAL types. BRANCH_POINT is not generated because branch detection
    requires specific conditions in the extractor.
    """
    num_traces = draw(st.integers(min_value=2, max_value=4))
    trace_ids = [f"trace_{i}" for i in range(num_traces)]
    num_positions = draw(st.integers(min_value=1, max_value=12))

    positions: List[AlignedPosition] = []

    for idx in range(num_positions):
        atype = draw(
            st.sampled_from([
                AlignmentType.DETERMINISTIC,
                AlignmentType.PARAMETERIZABLE,
                AlignmentType.VARIABLE,
                AlignmentType.OPTIONAL,
            ])
        )

        steps: Dict[str, Optional[TraceStep]] = {}

        if atype == AlignmentType.DETERMINISTIC:
            # All traces have the same action type, target, and args.
            action_type = draw(st.sampled_from(ACTION_TYPES))
            target = draw(st.sampled_from(TARGETS))
            args = draw(
                st.one_of(
                    st.none(),
                    st.fixed_dictionaries({"text": st.just("same_value")}),
                )
            )
            for tid in trace_ids:
                steps[tid] = _make_step(action_type, target, args)

        elif atype == AlignmentType.PARAMETERIZABLE:
            # Same action type and target, different args across traces.
            action_type = draw(st.sampled_from(ACTION_TYPES))
            target = draw(st.sampled_from(TARGETS))
            for i, tid in enumerate(trace_ids):
                steps[tid] = _make_step(
                    action_type, target, {"text": f"value_{i}"}
                )

        elif atype == AlignmentType.VARIABLE:
            # Different action types across traces.
            for tid in trace_ids:
                action_type = draw(st.sampled_from(ACTION_TYPES))
                target = draw(st.sampled_from(TARGETS))
                steps[tid] = _make_step(action_type, target)

        elif atype == AlignmentType.OPTIONAL:
            # Some traces have the step, others have None.
            action_type = draw(st.sampled_from(ACTION_TYPES))
            target = draw(st.sampled_from(TARGETS))
            presence = draw(
                st.lists(
                    st.booleans(),
                    min_size=num_traces,
                    max_size=num_traces,
                ).filter(lambda bools: any(bools) and not all(bools))
            )
            for i, tid in enumerate(trace_ids):
                steps[tid] = _make_step(action_type, target) if presence[i] else None

        positions.append(
            AlignedPosition(
                index=idx,
                alignment_type=atype,
                steps=steps,
                confidence=1.0,
            )
        )

    return AlignedTraceSet(
        positions=positions,
        trace_ids=trace_ids,
        alignment_score=1.0,
    )


# ---------------------------------------------------------------------------
# Helpers for collecting indices from ExtractedPatterns
# ---------------------------------------------------------------------------


def _collect_all_categorised_indices(patterns) -> List[int]:
    """Collect every position index from all categories, with duplicates."""
    indices: List[int] = []

    for pos in patterns.deterministic_steps:
        indices.append(pos.index)

    for pos, _ in patterns.parameterizable_steps:
        indices.append(pos.index)

    for pos in patterns.variable_steps:
        indices.append(pos.index)

    for pos in patterns.optional_steps:
        indices.append(pos.index)

    for bp in patterns.branch_patterns:
        indices.append(bp.branch_point_index)

    for lp in patterns.loop_patterns:
        indices.extend(range(lp.body_start, lp.body_end + 1))

    return indices


# ---------------------------------------------------------------------------
# Property 11: Pattern extraction completeness
# ---------------------------------------------------------------------------


class TestPatternCompletenessProperty:
    """
    Property 11: Pattern extraction completeness

    Every position index appears in exactly one category of
    ExtractedPatterns.

    **Validates: Requirements 5.1**
    """

    @given(aligned_set=aligned_trace_set_st())
    @settings(max_examples=200)
    def test_every_position_in_exactly_one_category(
        self, aligned_set: AlignedTraceSet
    ):
        """
        The union of all category indices equals the set of all position
        indices, and no index appears in more than one category.

        **Validates: Requirements 5.1**
        """
        extractor = PatternExtractor()
        patterns = extractor.extract(aligned_set)

        # All position indices from the input.
        all_indices: Set[int] = {pos.index for pos in aligned_set.positions}

        # Collect categorised indices (may contain duplicates if buggy).
        categorised = _collect_all_categorised_indices(patterns)
        categorised_set: Set[int] = set(categorised)

        # Every position index must appear.
        assert categorised_set == all_indices, (
            f"Mismatch: categorised={sorted(categorised_set)}, "
            f"expected={sorted(all_indices)}, "
            f"missing={sorted(all_indices - categorised_set)}, "
            f"extra={sorted(categorised_set - all_indices)}"
        )

        # No index should appear more than once.
        assert len(categorised) == len(categorised_set), (
            f"Duplicate indices found: "
            f"{sorted(i for i in categorised if categorised.count(i) > 1)}"
        )

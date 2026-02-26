"""
Property-based test for variable step variant frequency.

Feature: meta-agent-workflow, Property 12: Variable step variant frequency

For any AlignedTraceSet with VARIABLE positions where steps have different
action_types, after extraction the recorded variants SHALL include all
distinct action types observed at that position, and the frequency (count)
for each action type SHALL equal the number of traces that had that action
type at that position.

**Validates: Requirements 5.4**
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

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

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url", "wait"]
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
def variable_trace_set_st(draw) -> AlignedTraceSet:
    """Generate an AlignedTraceSet containing VARIABLE positions.

    Each position is VARIABLE: traces have potentially different action types
    at that position. We also include a few DETERMINISTIC positions so the
    extractor has a realistic mix.
    """
    num_traces = draw(st.integers(min_value=2, max_value=5))
    trace_ids = [f"trace_{i}" for i in range(num_traces)]

    # At least one VARIABLE position, up to 8 total positions.
    num_variable = draw(st.integers(min_value=1, max_value=6))
    num_deterministic = draw(st.integers(min_value=0, max_value=4))
    total = num_variable + num_deterministic

    positions: List[AlignedPosition] = []
    idx = 0

    # Generate VARIABLE positions.
    for _ in range(num_variable):
        steps: Dict[str, Optional[TraceStep]] = {}
        for tid in trace_ids:
            action_type = draw(st.sampled_from(ACTION_TYPES))
            target = draw(st.sampled_from(TARGETS))
            steps[tid] = _make_step(action_type, target)
        positions.append(
            AlignedPosition(
                index=idx,
                alignment_type=AlignmentType.VARIABLE,
                steps=steps,
                confidence=1.0,
            )
        )
        idx += 1

    # Generate DETERMINISTIC positions (filler).
    for _ in range(num_deterministic):
        action_type = draw(st.sampled_from(ACTION_TYPES))
        target = draw(st.sampled_from(TARGETS))
        steps = {tid: _make_step(action_type, target) for tid in trace_ids}
        positions.append(
            AlignedPosition(
                index=idx,
                alignment_type=AlignmentType.DETERMINISTIC,
                steps=steps,
                confidence=1.0,
            )
        )
        idx += 1

    return AlignedTraceSet(
        positions=positions,
        trace_ids=trace_ids,
        alignment_score=1.0,
    )


# ---------------------------------------------------------------------------
# Property 12: Variable step variant frequency
# ---------------------------------------------------------------------------


class TestVariantFrequencyProperty:
    """
    Property 12: Variable step variant frequency

    For any VARIABLE position, after pattern extraction the recorded
    variants include all distinct action types at that position, and
    the frequency for each action type equals the number of traces
    that had that action type at that position.

    **Validates: Requirements 5.4**
    """

    @given(aligned_set=variable_trace_set_st())
    @settings(max_examples=200)
    def test_variant_frequencies_match_trace_counts(
        self, aligned_set: AlignedTraceSet,
    ):
        """
        After extraction, each variable step's metadata["variants"] dict
        contains every distinct action type observed at that position,
        with counts equal to the number of traces using that action type.

        **Validates: Requirements 5.4**
        """
        extractor = PatternExtractor()
        patterns = extractor.extract(aligned_set)

        for pos in patterns.variable_steps:
            # Compute expected variant counts from the raw position steps.
            expected_counts: Counter[str] = Counter()
            for tid, step in pos.steps.items():
                if step is not None:
                    expected_counts[step.action_type] += 1

            # Find the first non-None step that should hold the metadata.
            recorded_variants: Optional[Dict[str, int]] = None
            for step in pos.steps.values():
                if step is not None and "variants" in step.metadata:
                    recorded_variants = step.metadata["variants"]
                    break

            assert recorded_variants is not None, (
                f"Position {pos.index}: no 'variants' metadata recorded "
                f"for VARIABLE step"
            )

            # All distinct action types must be present.
            assert set(recorded_variants.keys()) == set(expected_counts.keys()), (
                f"Position {pos.index}: variant keys mismatch. "
                f"Recorded={set(recorded_variants.keys())}, "
                f"Expected={set(expected_counts.keys())}"
            )

            # Frequencies must match exactly.
            for action_type, expected_count in expected_counts.items():
                actual_count = recorded_variants[action_type]
                assert actual_count == expected_count, (
                    f"Position {pos.index}: frequency mismatch for "
                    f"'{action_type}'. Expected={expected_count}, "
                    f"Actual={actual_count}"
                )

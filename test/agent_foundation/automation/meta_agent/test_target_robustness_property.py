"""
Property-based test for deterministic step target robustness.

Feature: meta-agent-workflow, Property 14: Deterministic step target robustness

For any Deterministic_Step with multiple target observations across traces,
the selected target SHALL be a TargetSpecWithFallback containing all unique
(strategy, value) pairs observed, ordered by STRATEGY_PRIORITY:
data-qa > data-testid > id > aria > xpath-text > xpath-class > css > agent.

**Validates: Requirements 5.5**
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.models import (
    AlignedPosition,
    AlignedTraceSet,
    AlignmentType,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.pattern_extractor import (
    PatternExtractor,
    STRATEGY_PRIORITY,
)
from science_modeling_tools.automation.meta_agent.target_converter import (
    TargetSpec,
    TargetSpecWithFallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url"]

# Strategy/value pairs that can appear in targets.
STRATEGY_NAMES = list(STRATEGY_PRIORITY)  # data-qa, data-testid, id, ...

_STRATEGY_RANK = {s: i for i, s in enumerate(STRATEGY_PRIORITY)}


def _make_step(
    action_type: str,
    target: Optional[TargetSpecWithFallback] = None,
) -> TraceStep:
    return TraceStep(action_type=action_type, target=target)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate a single TargetSpec with a strategy from STRATEGY_PRIORITY.
target_spec_st = st.builds(
    TargetSpec,
    strategy=st.sampled_from(STRATEGY_NAMES),
    value=st.text(min_size=1, max_size=15, alphabet="abcdefghijklmnop0123456789-_"),
)


@st.composite
def target_with_fallback_st(draw) -> TargetSpecWithFallback:
    """Generate a TargetSpecWithFallback with 1-4 unique strategies."""
    specs = draw(
        st.lists(target_spec_st, min_size=1, max_size=4, unique_by=lambda s: (s.strategy, s.value))
    )
    return TargetSpecWithFallback(strategies=specs)


@st.composite
def deterministic_trace_set_st(draw) -> AlignedTraceSet:
    """Generate an AlignedTraceSet with DETERMINISTIC positions.

    Each DETERMINISTIC position has the same action_type across all traces
    but each trace may have a different TargetSpecWithFallback containing
    different strategy/value pairs. This exercises the target consolidation
    logic in PatternExtractor._select_best_target.
    """
    num_traces = draw(st.integers(min_value=2, max_value=5))
    trace_ids = [f"trace_{i}" for i in range(num_traces)]

    num_deterministic = draw(st.integers(min_value=1, max_value=5))

    positions: List[AlignedPosition] = []

    for idx in range(num_deterministic):
        action_type = draw(st.sampled_from(ACTION_TYPES))

        # Generate a target for each trace.
        steps: Dict[str, Optional[TraceStep]] = {}
        for tid in trace_ids:
            target = draw(target_with_fallback_st())
            steps[tid] = _make_step(action_type, target)

        positions.append(
            AlignedPosition(
                index=idx,
                alignment_type=AlignmentType.DETERMINISTIC,
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
# Property 14: Deterministic step target robustness
# ---------------------------------------------------------------------------


class TestTargetRobustnessProperty:
    """
    Property 14: Deterministic step target robustness

    For any Deterministic_Step with multiple target observations, the
    selected target is a TargetSpecWithFallback containing all unique
    strategies observed, ordered by STRATEGY_PRIORITY.

    **Validates: Requirements 5.5**
    """

    @given(aligned_set=deterministic_trace_set_st())
    @settings(max_examples=200)
    def test_consolidated_target_contains_all_unique_strategies(
        self, aligned_set: AlignedTraceSet,
    ):
        """
        After extraction, each deterministic step's target is a
        TargetSpecWithFallback containing all unique (strategy, value)
        pairs from all traces at that position.

        **Validates: Requirements 5.5**
        """
        # Collect expected unique (strategy, value) pairs per position
        # BEFORE extraction mutates the steps.
        expected_per_pos: Dict[int, Set[Tuple[str, str]]] = {}
        for pos in aligned_set.positions:
            if pos.alignment_type != AlignmentType.DETERMINISTIC:
                continue
            pairs: Set[Tuple[str, str]] = set()
            for step in pos.steps.values():
                if step is not None and isinstance(step.target, TargetSpecWithFallback):
                    for spec in step.target.strategies:
                        pairs.add((spec.strategy, spec.value))
            expected_per_pos[pos.index] = pairs

        extractor = PatternExtractor()
        patterns = extractor.extract(aligned_set)

        for pos in patterns.deterministic_steps:
            expected_pairs = expected_per_pos[pos.index]

            # After extraction, the step targets should be consolidated.
            # Pick any non-None step — _apply_best_target sets the same
            # target on all steps.
            sample_step = next(
                (s for s in pos.steps.values() if s is not None), None,
            )
            assert sample_step is not None, (
                f"Position {pos.index}: no non-None steps"
            )

            consolidated = sample_step.target
            assert isinstance(consolidated, TargetSpecWithFallback), (
                f"Position {pos.index}: target should be TargetSpecWithFallback, "
                f"got {type(consolidated)}"
            )

            actual_pairs = {
                (spec.strategy, spec.value) for spec in consolidated.strategies
            }

            assert actual_pairs == expected_pairs, (
                f"Position {pos.index}: consolidated target missing strategies. "
                f"expected={expected_pairs}, actual={actual_pairs}"
            )

    @given(aligned_set=deterministic_trace_set_st())
    @settings(max_examples=200)
    def test_consolidated_target_ordered_by_strategy_priority(
        self, aligned_set: AlignedTraceSet,
    ):
        """
        After extraction, each deterministic step's consolidated target
        has strategies ordered by STRATEGY_PRIORITY (data-qa first,
        agent last). Within the same strategy, higher-frequency pairs
        come first.

        **Validates: Requirements 5.5**
        """
        extractor = PatternExtractor()
        patterns = extractor.extract(aligned_set)

        for pos in patterns.deterministic_steps:
            sample_step = next(
                (s for s in pos.steps.values() if s is not None), None,
            )
            assert sample_step is not None

            consolidated = sample_step.target
            if not isinstance(consolidated, TargetSpecWithFallback):
                continue
            if len(consolidated.strategies) <= 1:
                continue

            # Verify ordering: each strategy's rank should be <=
            # the next strategy's rank.
            for i in range(len(consolidated.strategies) - 1):
                rank_a = _STRATEGY_RANK.get(
                    consolidated.strategies[i].strategy,
                    len(STRATEGY_PRIORITY),
                )
                rank_b = _STRATEGY_RANK.get(
                    consolidated.strategies[i + 1].strategy,
                    len(STRATEGY_PRIORITY),
                )
                assert rank_a <= rank_b, (
                    f"Position {pos.index}: strategies not ordered by priority. "
                    f"'{consolidated.strategies[i].strategy}' (rank {rank_a}) "
                    f"before '{consolidated.strategies[i + 1].strategy}' "
                    f"(rank {rank_b}). Full list: "
                    f"{[(s.strategy, s.value) for s in consolidated.strategies]}"
                )

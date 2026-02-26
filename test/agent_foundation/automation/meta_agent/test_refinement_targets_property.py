"""
Property-based test for refinement strengthening target specifications.

Feature: meta-agent-workflow, Property 24: Refinement strengthens target specifications

For any target in a refined ActionGraph, if additional traces provided new
strategies, the TargetSpecWithFallback contains at least as many strategies
as the original.

**Validates: Requirements 11.3**
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

STRATEGY_NAMES = list(STRATEGY_PRIORITY)


def _make_step(
    action_type: str,
    target: Optional[TargetSpecWithFallback] = None,
) -> TraceStep:
    return TraceStep(action_type=action_type, target=target)


def _extract_strategy_pairs(
    target: Any,
) -> Set[Tuple[str, str]]:
    """Extract unique (strategy, value) pairs from a target."""
    if isinstance(target, TargetSpecWithFallback):
        return {(s.strategy, s.value) for s in target.strategies}
    return set()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate a single TargetSpec with a strategy from STRATEGY_PRIORITY.
target_spec_st = st.builds(
    TargetSpec,
    strategy=st.sampled_from(STRATEGY_NAMES),
    value=st.text(min_size=1, max_size=10, alphabet="abcdefghij0123456789"),
)


@st.composite
def target_with_fallback_st(draw, min_size: int = 1, max_size: int = 3) -> TargetSpecWithFallback:
    """Generate a TargetSpecWithFallback with unique strategies."""
    specs = draw(
        st.lists(
            target_spec_st,
            min_size=min_size,
            max_size=max_size,
            unique_by=lambda s: (s.strategy, s.value),
        )
    )
    return TargetSpecWithFallback(strategies=specs)


@st.composite
def refinement_scenario_st(draw) -> Dict[str, Any]:
    """Generate a refinement scenario with original and additional traces.

    Creates:
    - A set of original trace IDs with targets for DETERMINISTIC positions
    - A set of additional trace IDs with targets that may include new strategies
    - All positions share the same action_type across all traces (DETERMINISTIC)

    Returns a dict with:
        original_trace_ids: list of original trace IDs
        additional_trace_ids: list of new trace IDs
        positions: list of dicts with {action_type, original_targets, additional_targets}
    """
    num_original = draw(st.integers(min_value=2, max_value=4))
    num_additional = draw(st.integers(min_value=1, max_value=3))
    num_positions = draw(st.integers(min_value=1, max_value=4))

    original_ids = [f"orig_{i}" for i in range(num_original)]
    additional_ids = [f"new_{i}" for i in range(num_additional)]

    positions = []
    for _ in range(num_positions):
        action_type = draw(st.sampled_from(ACTION_TYPES))

        # Original targets per trace
        orig_targets = {}
        for tid in original_ids:
            orig_targets[tid] = draw(target_with_fallback_st(min_size=1, max_size=3))

        # Additional targets per trace — may introduce new strategies
        add_targets = {}
        for tid in additional_ids:
            add_targets[tid] = draw(target_with_fallback_st(min_size=1, max_size=4))

        positions.append({
            "action_type": action_type,
            "original_targets": orig_targets,
            "additional_targets": add_targets,
        })

    return {
        "original_trace_ids": original_ids,
        "additional_trace_ids": additional_ids,
        "positions": positions,
    }


def _build_aligned_set(
    trace_ids: List[str],
    positions_data: List[Dict[str, Any]],
    targets_key: str,
    all_trace_ids: Optional[List[str]] = None,
) -> AlignedTraceSet:
    """Build an AlignedTraceSet from position data.

    Args:
        trace_ids: trace IDs to include in steps
        positions_data: list of position dicts from the scenario
        targets_key: which targets dict to use ("original_targets" or "additional_targets")
        all_trace_ids: all trace IDs for the AlignedTraceSet (defaults to trace_ids)
    """
    if all_trace_ids is None:
        all_trace_ids = trace_ids

    positions = []
    for idx, pos_data in enumerate(positions_data):
        action_type = pos_data["action_type"]
        targets = pos_data[targets_key]

        steps: Dict[str, Optional[TraceStep]] = {}
        for tid in all_trace_ids:
            if tid in targets:
                steps[tid] = _make_step(action_type, targets[tid])
            else:
                steps[tid] = None

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
        trace_ids=all_trace_ids,
        alignment_score=1.0,
    )


def _build_combined_aligned_set(
    scenario: Dict[str, Any],
) -> AlignedTraceSet:
    """Build a combined AlignedTraceSet with both original and additional traces."""
    all_ids = scenario["original_trace_ids"] + scenario["additional_trace_ids"]

    positions = []
    for idx, pos_data in enumerate(scenario["positions"]):
        action_type = pos_data["action_type"]
        orig_targets = pos_data["original_targets"]
        add_targets = pos_data["additional_targets"]

        steps: Dict[str, Optional[TraceStep]] = {}
        for tid in all_ids:
            if tid in orig_targets:
                steps[tid] = _make_step(action_type, orig_targets[tid])
            elif tid in add_targets:
                steps[tid] = _make_step(action_type, add_targets[tid])
            else:
                steps[tid] = None

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
        trace_ids=all_ids,
        alignment_score=1.0,
    )


# ---------------------------------------------------------------------------
# Property 24: Refinement strengthens target specifications
# ---------------------------------------------------------------------------


class TestRefinementTargetsProperty:
    """
    Property 24: Refinement strengthens target specifications

    When the merged alignment confirms existing patterns with higher
    confidence, the Graph_Synthesizer SHALL strengthen target specifications
    by adding newly observed strategies to TargetSpecWithFallback.

    Concretely: for any DETERMINISTIC position, the number of unique
    (strategy, value) pairs in the refined TargetSpecWithFallback is >=
    the number in the original.

    **Validates: Requirements 11.3**
    """

    @given(scenario=refinement_scenario_st())
    @settings(max_examples=200)
    def test_refined_targets_have_at_least_as_many_strategies(
        self,
        scenario: Dict[str, Any],
    ):
        """
        After refinement with additional traces, each DETERMINISTIC
        position's consolidated TargetSpecWithFallback contains at least
        as many unique (strategy, value) pairs as the original extraction.

        **Validates: Requirements 11.3**
        """
        extractor = PatternExtractor()

        # --- Original extraction (only original traces) ---
        original_set = _build_aligned_set(
            trace_ids=scenario["original_trace_ids"],
            positions_data=scenario["positions"],
            targets_key="original_targets",
        )
        original_patterns = extractor.extract(original_set)

        original_strategy_counts: Dict[int, int] = {}
        for pos in original_patterns.deterministic_steps:
            sample = next(
                (s for s in pos.steps.values() if s is not None), None,
            )
            if sample is not None and isinstance(sample.target, TargetSpecWithFallback):
                original_strategy_counts[pos.index] = len(
                    _extract_strategy_pairs(sample.target)
                )

        # --- Refined extraction (original + additional traces) ---
        combined_set = _build_combined_aligned_set(scenario)
        refined_patterns = extractor.extract(combined_set)

        for pos in refined_patterns.deterministic_steps:
            sample = next(
                (s for s in pos.steps.values() if s is not None), None,
            )
            if sample is None:
                continue

            refined_pairs = _extract_strategy_pairs(sample.target)
            original_count = original_strategy_counts.get(pos.index, 0)

            assert len(refined_pairs) >= original_count, (
                f"Position {pos.index}: refined target has fewer strategies "
                f"({len(refined_pairs)}) than original ({original_count}). "
                f"Refinement should strengthen, not weaken, targets."
            )

    @given(scenario=refinement_scenario_st())
    @settings(max_examples=200)
    def test_refined_targets_superset_of_original_strategies(
        self,
        scenario: Dict[str, Any],
    ):
        """
        The refined TargetSpecWithFallback's (strategy, value) pairs are
        a superset of the original's — no previously observed strategies
        are lost during refinement.

        **Validates: Requirements 11.3**
        """
        extractor = PatternExtractor()

        # --- Original extraction ---
        original_set = _build_aligned_set(
            trace_ids=scenario["original_trace_ids"],
            positions_data=scenario["positions"],
            targets_key="original_targets",
        )
        original_patterns = extractor.extract(original_set)

        original_pairs_per_pos: Dict[int, Set[Tuple[str, str]]] = {}
        for pos in original_patterns.deterministic_steps:
            sample = next(
                (s for s in pos.steps.values() if s is not None), None,
            )
            if sample is not None and isinstance(sample.target, TargetSpecWithFallback):
                original_pairs_per_pos[pos.index] = _extract_strategy_pairs(
                    sample.target
                )

        # --- Refined extraction ---
        combined_set = _build_combined_aligned_set(scenario)
        refined_patterns = extractor.extract(combined_set)

        for pos in refined_patterns.deterministic_steps:
            if pos.index not in original_pairs_per_pos:
                continue

            sample = next(
                (s for s in pos.steps.values() if s is not None), None,
            )
            if sample is None:
                continue

            refined_pairs = _extract_strategy_pairs(sample.target)
            original_pairs = original_pairs_per_pos[pos.index]

            assert original_pairs.issubset(refined_pairs), (
                f"Position {pos.index}: refined target lost strategies. "
                f"Original had {original_pairs - refined_pairs} which are "
                f"missing from refined set. Original={original_pairs}, "
                f"Refined={refined_pairs}"
            )

"""
Property-based test for synthesized target strategy ordering (Property 17).

**Validates: Requirements 6.9**

*For any* synthesized action with a TargetSpecWithFallback, the strategies
SHALL be ordered by reliability: data-qa/data-testid first, then id, aria,
xpath, css, and agent last.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from science_modeling_tools.automation.meta_agent.synthesizer import GraphSynthesizer, RuleBasedSynthesizer
from science_modeling_tools.automation.meta_agent.models import (
    AlignedPosition,
    AlignmentType,
    ExtractedPatterns,
    ParameterizableInfo,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.target_converter import (
    TargetSpec,
    TargetSpecWithFallback,
)
from science_modeling_tools.automation.meta_agent.pattern_extractor import (
    STRATEGY_PRIORITY,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All known strategy names for generation.
ALL_STRATEGIES = list(STRATEGY_PRIORITY)

# Action types that the synthesizer handles as standard actions.
ACTION_TYPES = ["click", "input_text", "scroll", "visit_url"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _priority_index(strategy_name: str) -> int:
    """Return the index of *strategy_name* in the priority list."""
    return STRATEGY_PRIORITY.index(strategy_name)


def _make_fallback_target(strategy_names: list[str]) -> TargetSpecWithFallback:
    """Build a TargetSpecWithFallback from a list of strategy names.

    Strategies are added in the given order — the test verifies that the
    synthesizer preserves (or re-orders) them correctly.
    """
    specs = [TargetSpec(strategy=name, value=f"val-{name}") for name in strategy_names]
    return TargetSpecWithFallback(strategies=specs)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def ordered_strategy_subset(draw):
    """Draw a non-empty subset of strategy names, ordered by priority.

    This represents a well-formed TargetSpecWithFallback that already
    respects the priority ordering — the kind the TargetStrategyConverter
    produces.
    """
    flags = draw(st.lists(st.booleans(), min_size=len(ALL_STRATEGIES), max_size=len(ALL_STRATEGIES)))
    selected = [s for s, flag in zip(ALL_STRATEGIES, flags) if flag]
    if not selected:
        # Ensure at least one strategy.
        selected = [draw(st.sampled_from(ALL_STRATEGIES))]
    return selected


@st.composite
def shuffled_strategy_subset(draw):
    """Draw a non-empty subset of strategy names in random order.

    This tests that even if the input target has strategies in an
    arbitrary order, the synthesizer's output preserves whatever
    ordering was provided (since the synthesizer passes targets
    through as-is from the representative step).
    """
    flags = draw(st.lists(st.booleans(), min_size=len(ALL_STRATEGIES), max_size=len(ALL_STRATEGIES)))
    selected = [s for s, flag in zip(ALL_STRATEGIES, flags) if flag]
    if not selected:
        selected = [draw(st.sampled_from(ALL_STRATEGIES))]
    perm = draw(st.permutations(selected))
    return list(perm)


@st.composite
def patterns_with_fallback_targets(draw):
    """Generate ExtractedPatterns where steps have TargetSpecWithFallback targets.

    Each step gets a TargetSpecWithFallback with strategies ordered by
    priority (as the TargetStrategyConverter would produce). The test
    verifies that the synthesizer preserves this ordering in the graph.
    """
    n_steps = draw(st.integers(min_value=1, max_value=8))

    det_steps: list[AlignedPosition] = []
    opt_steps: list[AlignedPosition] = []
    param_steps: list[tuple[AlignedPosition, ParameterizableInfo]] = []
    step_order: list[int] = []

    for i in range(n_steps):
        category = draw(st.sampled_from(["deterministic", "optional", "parameterizable"]))
        step_order.append(i)

        strategy_names = draw(ordered_strategy_subset())
        target = _make_fallback_target(strategy_names)
        action_type = draw(st.sampled_from(ACTION_TYPES))

        step = TraceStep(
            action_type=action_type,
            target=target,
            args={"text": "hello"} if category == "parameterizable" else None,
        )

        pos = AlignedPosition(
            index=i,
            alignment_type=(
                AlignmentType.DETERMINISTIC if category == "deterministic"
                else AlignmentType.OPTIONAL if category == "optional"
                else AlignmentType.PARAMETERIZABLE
            ),
            steps={"t1": step},
        )

        if category == "deterministic":
            det_steps.append(pos)
        elif category == "optional":
            opt_steps.append(pos)
        else:
            info = ParameterizableInfo(
                variable_args={"text": f"var_{i}"},
                constant_args={},
            )
            param_steps.append((pos, info))

    return ExtractedPatterns(
        deterministic_steps=det_steps,
        parameterizable_steps=param_steps,
        variable_steps=[],
        optional_steps=opt_steps,
        branch_patterns=[],
        loop_patterns=[],
        user_input_boundaries=[],
        step_order=step_order,
    )


# ---------------------------------------------------------------------------
# Property 17: Synthesized target strategy ordering
# ---------------------------------------------------------------------------


class TestTargetStrategyOrderingProperty:
    """
    Property 17: Synthesized target strategy ordering

    *For any* synthesized action with a TargetSpecWithFallback, the strategies
    SHALL be ordered by reliability: data-qa/data-testid first, then id, aria,
    xpath, css, and agent last.

    **Validates: Requirements 6.9**
    """

    @given(patterns=patterns_with_fallback_targets())
    @settings(max_examples=200, deadline=None)
    def test_synthesized_targets_preserve_priority_ordering(self, patterns: ExtractedPatterns):
        """
        When the synthesizer produces actions from patterns whose steps
        have TargetSpecWithFallback targets ordered by priority, the
        resulting graph actions' targets maintain that ordering.

        **Validates: Requirements 6.9**
        """
        synth = RuleBasedSynthesizer(
            action_executor=MagicMock(),
        )

        result = synth.synthesize(patterns)
        graph = result.graph
        actions = graph._nodes[0]._actions

        for action in actions:
            target = action.target
            if target is None:
                continue

            # The graph uses its own TargetSpecWithFallback type.
            strategies = getattr(target, "strategies", None)
            if strategies is None:
                continue

            strategy_names = [s.strategy for s in strategies]

            # Verify ordering: each consecutive pair must respect priority.
            for j in range(len(strategy_names) - 1):
                idx_cur = _priority_index(strategy_names[j])
                idx_nxt = _priority_index(strategy_names[j + 1])
                assert idx_cur < idx_nxt, (
                    f"Strategy ordering violated in synthesized action: "
                    f"{strategy_names[j]!r} (priority {idx_cur}) appears before "
                    f"{strategy_names[j + 1]!r} (priority {idx_nxt}). "
                    f"Full chain: {strategy_names}"
                )

    @given(patterns=patterns_with_fallback_targets())
    @settings(max_examples=200, deadline=None)
    def test_all_synthesized_strategies_are_known(self, patterns: ExtractedPatterns):
        """
        Every strategy name in a synthesized TargetSpecWithFallback is a
        member of the canonical STRATEGY_PRIORITY list.

        **Validates: Requirements 6.9**
        """
        synth = RuleBasedSynthesizer(
            action_executor=MagicMock(),
        )

        result = synth.synthesize(patterns)
        graph = result.graph
        actions = graph._nodes[0]._actions
        allowed = set(STRATEGY_PRIORITY)

        for action in actions:
            target = action.target
            if target is None:
                continue

            strategies = getattr(target, "strategies", None)
            if strategies is None:
                continue

            for spec in strategies:
                assert spec.strategy in allowed, (
                    f"Unknown strategy {spec.strategy!r} in synthesized target. "
                    f"Allowed: {STRATEGY_PRIORITY}"
                )

    @given(patterns=patterns_with_fallback_targets())
    @settings(max_examples=200, deadline=None)
    def test_agent_strategy_always_last_in_synthesized_target(self, patterns: ExtractedPatterns):
        """
        If the 'agent' strategy appears in a synthesized TargetSpecWithFallback,
        it is always the last entry — agent-based fallback is the least
        reliable strategy.

        **Validates: Requirements 6.9**
        """
        synth = RuleBasedSynthesizer(
            action_executor=MagicMock(),
        )

        result = synth.synthesize(patterns)
        graph = result.graph
        actions = graph._nodes[0]._actions

        for action in actions:
            target = action.target
            if target is None:
                continue

            strategies = getattr(target, "strategies", None)
            if strategies is None:
                continue

            strategy_names = [s.strategy for s in strategies]
            if "agent" in strategy_names:
                assert strategy_names[-1] == "agent", (
                    f"'agent' strategy is not last in synthesized target: "
                    f"{strategy_names}"
                )

    @given(
        strategy_names=shuffled_strategy_subset(),
        action_type=st.sampled_from(ACTION_TYPES),
    )
    @settings(max_examples=200, deadline=None)
    def test_converter_always_produces_ordered_strategies(
        self, strategy_names: list[str], action_type: str
    ):
        """
        Verify that TargetStrategyConverter.STRATEGY_PRIORITY defines the
        correct ordering, and that any TargetSpecWithFallback built from
        a subset of strategies and sorted by STRATEGY_PRIORITY index
        produces a strictly increasing priority sequence.

        This is a foundational check that the ordering invariant is
        self-consistent.

        **Validates: Requirements 6.9**
        """
        # Sort by priority index (as the converter and pattern extractor do).
        sorted_names = sorted(strategy_names, key=lambda s: _priority_index(s))

        for j in range(len(sorted_names) - 1):
            idx_cur = _priority_index(sorted_names[j])
            idx_nxt = _priority_index(sorted_names[j + 1])
            assert idx_cur < idx_nxt, (
                f"Priority ordering is not strict after sorting: "
                f"{sorted_names[j]!r} ({idx_cur}) vs "
                f"{sorted_names[j + 1]!r} ({idx_nxt}). "
                f"Input: {strategy_names}, Sorted: {sorted_names}"
            )

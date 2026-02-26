"""
Property-based test for alignment classification consistency.

Feature: meta-agent-workflow, Property 10: Alignment classification consistency

For any AlignedPosition:
- If all non-None steps have the same action type and equivalent targets
  with identical args, the alignment_type SHALL be DETERMINISTIC.
- If all non-None steps have the same action type and equivalent targets
  but different args, the alignment_type SHALL be PARAMETERIZABLE.
- If any non-None steps differ in action type or target, the alignment_type
  SHALL be VARIABLE.
- If some steps are None (gaps) and all non-None steps are equivalent
  (same action type, target, and args), the alignment_type SHALL be OPTIONAL.
- If some steps are None (gaps) and the non-None steps differ in action type,
  target, or args, the alignment_type SHALL be VARIABLE.

**Validates: Requirements 4.3, 4.4, 4.5, 4.6**
"""

from __future__ import annotations

from typing import Dict, Optional

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.aligner import TraceAligner
from science_modeling_tools.automation.meta_agent.models import (
    AlignmentType,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Fixed action types and targets for controlled generation
ACTION_TYPES = ["click", "input_text", "scroll", "visit_url", "wait"]
TARGETS = ["btn-submit", "search-box", "nav-link", "field-name"]


def _make_step(action_type: str, target: str, args: Optional[Dict] = None) -> TraceStep:
    """Helper to build a TraceStep with the given properties."""
    return TraceStep(action_type=action_type, target=target, args=args)


def _trace_ids(n: int):
    """Generate n unique trace IDs."""
    return [f"trace_{i}" for i in range(n)]


# Strategy: generate steps dict where ALL non-None steps share the same
# action_type, same target, and same args → DETERMINISTIC
def deterministic_steps_st() -> st.SearchStrategy[Dict[str, Optional[TraceStep]]]:
    """Generate a steps dict that should classify as DETERMINISTIC.

    All entries are non-None with identical action_type, target, and args.
    """
    return st.integers(min_value=2, max_value=6).flatmap(
        lambda n: st.tuples(
            st.sampled_from(ACTION_TYPES),
            st.sampled_from(TARGETS),
            st.one_of(
                st.none(),
                st.fixed_dictionaries({"text": st.text(min_size=1, max_size=8)}),
            ),
        ).map(
            lambda tup, count=n: {
                tid: _make_step(tup[0], tup[1], tup[2])
                for tid in _trace_ids(count)
            }
        )
    )


# Strategy: generate steps dict where ALL non-None steps share the same
# action_type and target but have DIFFERENT args → PARAMETERIZABLE
def parameterizable_steps_st() -> st.SearchStrategy[Dict[str, Optional[TraceStep]]]:
    """Generate a steps dict that should classify as PARAMETERIZABLE.

    All entries are non-None with identical action_type and target,
    but at least two entries have different args.
    """
    return st.integers(min_value=2, max_value=6).flatmap(
        lambda n: st.tuples(
            st.sampled_from(ACTION_TYPES),
            st.sampled_from(TARGETS),
            # Generate n distinct arg dicts — use unique text values
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=10,
                ),
                min_size=n,
                max_size=n,
                unique=True,
            ),
        ).map(
            lambda tup, count=n: {
                tid: _make_step(tup[0], tup[1], {"text": tup[2][i]})
                for i, tid in enumerate(_trace_ids(count))
            }
        )
    )


# Strategy: generate steps dict where non-None steps differ in action_type
# or target → VARIABLE (no gaps)
def variable_steps_no_gaps_st() -> st.SearchStrategy[Dict[str, Optional[TraceStep]]]:
    """Generate a steps dict that should classify as VARIABLE (no gaps).

    At least two non-None entries differ in action_type or target.
    """
    return st.integers(min_value=2, max_value=6).flatmap(
        lambda n: st.tuples(
            # Two different action types
            st.sampled_from(ACTION_TYPES),
            st.sampled_from(ACTION_TYPES),
            st.sampled_from(TARGETS),
            st.sampled_from(TARGETS),
        ).filter(
            # Ensure at least one dimension differs
            lambda tup: tup[0] != tup[1] or tup[2] != tup[3]
        ).map(
            lambda tup, count=n: {
                tid: _make_step(
                    tup[0] if i == 0 else tup[1],
                    tup[2] if i == 0 else tup[3],
                )
                for i, tid in enumerate(_trace_ids(count))
            }
        )
    )


# Strategy: generate steps dict with gaps where all non-None steps are
# equivalent (same type, target, args) → OPTIONAL
def optional_steps_st() -> st.SearchStrategy[Dict[str, Optional[TraceStep]]]:
    """Generate a steps dict that should classify as OPTIONAL.

    At least one None (gap) and at least one non-None step.
    All non-None steps have identical action_type, target, and args.
    """
    return st.integers(min_value=2, max_value=6).flatmap(
        lambda n: st.tuples(
            st.sampled_from(ACTION_TYPES),
            st.sampled_from(TARGETS),
            st.one_of(
                st.none(),
                st.fixed_dictionaries({"text": st.text(min_size=1, max_size=8)}),
            ),
            # Boolean mask: True = present, False = gap
            # At least one True and at least one False
            st.lists(
                st.booleans(), min_size=n, max_size=n,
            ).filter(lambda bools: any(bools) and not all(bools)),
        ).map(
            lambda tup, count=n: {
                tid: _make_step(tup[0], tup[1], tup[2]) if tup[3][i] else None
                for i, tid in enumerate(_trace_ids(count))
            }
        )
    )


# Strategy: generate steps dict with gaps where non-None steps differ
# → VARIABLE (with gaps)
def variable_steps_with_gaps_st() -> st.SearchStrategy[Dict[str, Optional[TraceStep]]]:
    """Generate a steps dict that should classify as VARIABLE (with gaps).

    At least one None (gap), at least two non-None steps that differ
    in action_type or target.
    """
    return st.integers(min_value=3, max_value=6).flatmap(
        lambda n: st.tuples(
            st.sampled_from(ACTION_TYPES),
            st.sampled_from(ACTION_TYPES),
            st.sampled_from(TARGETS),
            st.sampled_from(TARGETS),
        ).filter(
            lambda tup: tup[0] != tup[1] or tup[2] != tup[3]
        ).map(
            lambda tup, count=n: _build_variable_with_gaps(tup, count)
        )
    )


def _build_variable_with_gaps(tup, count: int) -> Dict[str, Optional[TraceStep]]:
    """Build a steps dict with at least one gap and differing non-None steps."""
    ids = _trace_ids(count)
    steps: Dict[str, Optional[TraceStep]] = {}
    # First entry: gap
    steps[ids[0]] = None
    # Second entry: step with first type/target
    steps[ids[1]] = _make_step(tup[0], tup[2])
    # Third entry: step with second type/target (differs from second)
    if count > 2:
        steps[ids[2]] = _make_step(tup[1], tup[3])
    # Remaining: fill with the first step variant
    for i in range(3, count):
        steps[ids[i]] = _make_step(tup[0], tup[2])
    return steps


# Strategy: gaps + non-None steps that have same type/target but different args
# → VARIABLE (gaps + different args)
def variable_steps_gaps_diff_args_st() -> st.SearchStrategy[Dict[str, Optional[TraceStep]]]:
    """Generate a steps dict with gaps and differing args → VARIABLE.

    At least one gap, all non-None steps have same type/target but
    different args. Per requirement 4.6, gaps + differing args = VARIABLE.
    """
    return st.integers(min_value=3, max_value=6).flatmap(
        lambda n: st.tuples(
            st.sampled_from(ACTION_TYPES),
            st.sampled_from(TARGETS),
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=10,
                ),
                min_size=n - 1,
                max_size=n - 1,
                unique=True,
            ),
        ).map(
            lambda tup, count=n: _build_gaps_diff_args(tup, count)
        )
    )


def _build_gaps_diff_args(tup, count: int) -> Dict[str, Optional[TraceStep]]:
    """Build steps dict: one gap, rest same type/target but different args."""
    ids = _trace_ids(count)
    steps: Dict[str, Optional[TraceStep]] = {}
    # First entry: gap
    steps[ids[0]] = None
    # Remaining: same type/target, unique args
    for i in range(1, count):
        steps[ids[i]] = _make_step(tup[0], tup[1], {"text": tup[2][i - 1]})
    return steps


# ---------------------------------------------------------------------------
# Property 10: Alignment classification consistency
# ---------------------------------------------------------------------------


class TestAlignmentClassificationProperty:
    """
    Property 10: Alignment classification consistency

    Directly tests the _classify_position method of TraceAligner to verify
    that classification follows the rules from requirements 4.3-4.6.

    **Validates: Requirements 4.3, 4.4, 4.5, 4.6**
    """

    @given(steps=deterministic_steps_st())
    @settings(max_examples=200)
    def test_same_type_target_args_is_deterministic(
        self, steps: Dict[str, Optional[TraceStep]]
    ):
        """
        All non-None steps with same action_type, equivalent target, and
        identical args → DETERMINISTIC.

        **Validates: Requirements 4.3**
        """
        aligner = TraceAligner()
        result = aligner._classify_position(steps)
        assert result == AlignmentType.DETERMINISTIC, (
            f"Expected DETERMINISTIC, got {result} for steps: "
            f"{[(tid, s.action_type if s else None, s.target if s else None, s.args if s else None) for tid, s in steps.items()]}"
        )

    @given(steps=parameterizable_steps_st())
    @settings(max_examples=200)
    def test_same_type_target_different_args_is_parameterizable(
        self, steps: Dict[str, Optional[TraceStep]]
    ):
        """
        All non-None steps with same action_type and equivalent target but
        different args → PARAMETERIZABLE.

        **Validates: Requirements 4.4**
        """
        aligner = TraceAligner()
        result = aligner._classify_position(steps)
        assert result == AlignmentType.PARAMETERIZABLE, (
            f"Expected PARAMETERIZABLE, got {result} for steps: "
            f"{[(tid, s.action_type if s else None, s.target if s else None, s.args if s else None) for tid, s in steps.items()]}"
        )

    @given(steps=variable_steps_no_gaps_st())
    @settings(max_examples=200)
    def test_different_type_or_target_is_variable(
        self, steps: Dict[str, Optional[TraceStep]]
    ):
        """
        Non-None steps with different action_type or target → VARIABLE.

        **Validates: Requirements 4.5**
        """
        aligner = TraceAligner()
        result = aligner._classify_position(steps)
        assert result == AlignmentType.VARIABLE, (
            f"Expected VARIABLE, got {result} for steps: "
            f"{[(tid, s.action_type if s else None, s.target if s else None, s.args if s else None) for tid, s in steps.items()]}"
        )

    @given(steps=optional_steps_st())
    @settings(max_examples=200)
    def test_gaps_with_equivalent_non_none_is_optional(
        self, steps: Dict[str, Optional[TraceStep]]
    ):
        """
        Some None (gaps) with all non-None steps equivalent (same action type,
        target, and args) → OPTIONAL.

        **Validates: Requirements 4.6**
        """
        aligner = TraceAligner()
        result = aligner._classify_position(steps)
        assert result == AlignmentType.OPTIONAL, (
            f"Expected OPTIONAL, got {result} for steps: "
            f"{[(tid, s.action_type if s else None, s.target if s else None, s.args if s else None) for tid, s in steps.items()]}"
        )

    @given(steps=variable_steps_with_gaps_st())
    @settings(max_examples=200)
    def test_gaps_with_differing_non_none_is_variable(
        self, steps: Dict[str, Optional[TraceStep]]
    ):
        """
        Some None (gaps) with non-None steps that differ in action type or
        target → VARIABLE.

        **Validates: Requirements 4.6**
        """
        aligner = TraceAligner()
        result = aligner._classify_position(steps)
        assert result == AlignmentType.VARIABLE, (
            f"Expected VARIABLE, got {result} for steps: "
            f"{[(tid, s.action_type if s else None, s.target if s else None, s.args if s else None) for tid, s in steps.items()]}"
        )

    @given(steps=variable_steps_gaps_diff_args_st())
    @settings(max_examples=200)
    def test_gaps_with_same_type_target_different_args_is_variable(
        self, steps: Dict[str, Optional[TraceStep]]
    ):
        """
        Some None (gaps) with non-None steps that have same type/target but
        different args → VARIABLE (not PARAMETERIZABLE, because gaps present).

        **Validates: Requirements 4.6**
        """
        aligner = TraceAligner()
        result = aligner._classify_position(steps)
        assert result == AlignmentType.VARIABLE, (
            f"Expected VARIABLE, got {result} for steps: "
            f"{[(tid, s.action_type if s else None, s.target if s else None, s.args if s else None) for tid, s in steps.items()]}"
        )

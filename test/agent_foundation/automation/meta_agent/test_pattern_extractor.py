"""Unit tests for PatternExtractor.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import pytest

from science_modeling_tools.automation.meta_agent.models import (
    AlignedPosition,
    AlignedTraceSet,
    AlignmentType,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.pattern_extractor import (
    PatternExtractor,
)
from science_modeling_tools.automation.meta_agent.target_converter import (
    TargetSpec,
    TargetSpecWithFallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(action_type: str, target=None, args=None, **kwargs) -> TraceStep:
    return TraceStep(action_type=action_type, target=target, args=args, **kwargs)


def _pos(
    index: int,
    atype: AlignmentType,
    steps: dict[str, TraceStep | None],
    confidence: float = 1.0,
) -> AlignedPosition:
    return AlignedPosition(
        index=index,
        alignment_type=atype,
        steps=steps,
        confidence=confidence,
    )


def _aligned_set(positions: list[AlignedPosition], trace_ids: list[str]) -> AlignedTraceSet:
    return AlignedTraceSet(
        positions=positions,
        trace_ids=trace_ids,
        alignment_score=1.0,
    )


# ---------------------------------------------------------------------------
# Loop detection  (Requirement 5.2)
# ---------------------------------------------------------------------------

class TestLoopDetection:
    """Loop detection finds repeated subsequences."""

    def test_simple_abab_loop(self):
        """[A, B, A, B] should be detected as a loop of body [A, B] with 2 iterations."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click"), "t2": _step("click")}),
            _pos(1, AlignmentType.DETERMINISTIC, {"t1": _step("input_text"), "t2": _step("input_text")}),
            _pos(2, AlignmentType.DETERMINISTIC, {"t1": _step("click"), "t2": _step("click")}),
            _pos(3, AlignmentType.DETERMINISTIC, {"t1": _step("input_text"), "t2": _step("input_text")}),
        ]
        aligned = _aligned_set(positions, ["t1", "t2"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.loop_patterns) == 1
        lp = patterns.loop_patterns[0]
        assert lp.body_start == 0
        assert lp.body_end == 1
        assert lp.min_iterations == 2
        assert lp.max_iterations == 2
        assert len(lp.body_steps) == 2

    def test_no_loop_when_no_repetition(self):
        """[A, B, C] has no repeated subsequence."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click")}),
            _pos(1, AlignmentType.DETERMINISTIC, {"t1": _step("input_text")}),
            _pos(2, AlignmentType.DETERMINISTIC, {"t1": _step("scroll")}),
        ]
        aligned = _aligned_set(positions, ["t1"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.loop_patterns) == 0

    def test_single_element_loop(self):
        """[A, A, A] should detect a loop of body [A] with 3 iterations."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click")}),
            _pos(1, AlignmentType.DETERMINISTIC, {"t1": _step("click")}),
            _pos(2, AlignmentType.DETERMINISTIC, {"t1": _step("click")}),
        ]
        aligned = _aligned_set(positions, ["t1"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.loop_patterns) >= 1
        lp = patterns.loop_patterns[0]
        assert lp.min_iterations >= 2


# ---------------------------------------------------------------------------
# Branch detection  (Requirement 5.3)
# ---------------------------------------------------------------------------

class TestBranchDetection:
    """Branch detection finds BRANCH_POINT positions with multiple action types."""

    def test_branch_point_with_two_action_types(self):
        """A BRANCH_POINT position with different action types produces a BranchPattern."""
        positions = [
            _pos(
                0,
                AlignmentType.BRANCH_POINT,
                {"t1": _step("click"), "t2": _step("input_text")},
            ),
        ]
        aligned = _aligned_set(positions, ["t1", "t2"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.branch_patterns) == 1
        bp = patterns.branch_patterns[0]
        assert bp.branch_point_index == 0
        assert "click" in bp.branches
        assert "input_text" in bp.branches
        assert bp.condition_description == "Unknown: runs diverge at this point"

    def test_branch_point_same_action_type_no_branch(self):
        """A BRANCH_POINT where all steps have the same action type → no branch (only 1 group)."""
        positions = [
            _pos(
                0,
                AlignmentType.BRANCH_POINT,
                {"t1": _step("click"), "t2": _step("click")},
            ),
        ]
        aligned = _aligned_set(positions, ["t1", "t2"])
        patterns = PatternExtractor().extract(aligned)

        # Only one action type group → no branch pattern detected.
        assert len(patterns.branch_patterns) == 0


# ---------------------------------------------------------------------------
# Parameterizable step  (Requirement 5.6)
# ---------------------------------------------------------------------------

class TestParameterizableStep:
    """Parameterizable steps split variable vs constant args correctly."""

    def test_variable_and_constant_args_split(self):
        """Args that differ across traces are variable; same values are constant."""
        positions = [
            _pos(
                0,
                AlignmentType.PARAMETERIZABLE,
                {
                    "t1": _step("input_text", args={"text": "hello", "delay": 100}),
                    "t2": _step("input_text", args={"text": "world", "delay": 100}),
                },
            ),
        ]
        aligned = _aligned_set(positions, ["t1", "t2"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.parameterizable_steps) == 1
        pos, info = patterns.parameterizable_steps[0]
        assert "text" in info.variable_args
        assert "delay" in info.constant_args
        assert info.constant_args["delay"] == 100

    def test_all_args_constant(self):
        """When all args are the same, all go to constant_args."""
        positions = [
            _pos(
                0,
                AlignmentType.PARAMETERIZABLE,
                {
                    "t1": _step("click", args={"x": 10, "y": 20}),
                    "t2": _step("click", args={"x": 10, "y": 20}),
                },
            ),
        ]
        aligned = _aligned_set(positions, ["t1", "t2"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.parameterizable_steps) == 1
        _, info = patterns.parameterizable_steps[0]
        assert len(info.variable_args) == 0
        assert info.constant_args == {"x": 10, "y": 20}

    def test_all_args_variable(self):
        """When all args differ, all go to variable_args."""
        positions = [
            _pos(
                0,
                AlignmentType.PARAMETERIZABLE,
                {
                    "t1": _step("input_text", args={"text": "a", "delay": 50}),
                    "t2": _step("input_text", args={"text": "b", "delay": 200}),
                },
            ),
        ]
        aligned = _aligned_set(positions, ["t1", "t2"])
        patterns = PatternExtractor().extract(aligned)

        _, info = patterns.parameterizable_steps[0]
        assert set(info.variable_args.keys()) == {"text", "delay"}
        assert len(info.constant_args) == 0



# ---------------------------------------------------------------------------
# All positions accounted for  (Requirement 5.1)
# ---------------------------------------------------------------------------

class TestAllPositionsAccountedFor:
    """Every position index appears in exactly one output category."""

    def test_mixed_types_all_covered(self):
        """A mix of DETERMINISTIC, PARAMETERIZABLE, VARIABLE, OPTIONAL positions."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click"), "t2": _step("click")}),
            _pos(1, AlignmentType.PARAMETERIZABLE, {
                "t1": _step("input_text", args={"text": "a"}),
                "t2": _step("input_text", args={"text": "b"}),
            }),
            _pos(2, AlignmentType.VARIABLE, {"t1": _step("click"), "t2": _step("scroll")}),
            _pos(3, AlignmentType.OPTIONAL, {"t1": _step("click"), "t2": None}),
        ]
        aligned = _aligned_set(positions, ["t1", "t2"])
        patterns = PatternExtractor().extract(aligned)

        # Collect all indices from each category.
        det_idx = {p.index for p in patterns.deterministic_steps}
        param_idx = {p.index for (p, _) in patterns.parameterizable_steps}
        var_idx = {p.index for p in patterns.variable_steps}
        opt_idx = {p.index for p in patterns.optional_steps}
        loop_idx = set()
        for lp in patterns.loop_patterns:
            loop_idx.update(range(lp.body_start, lp.body_end + 1))
        branch_idx = {bp.branch_point_index for bp in patterns.branch_patterns}

        all_idx = det_idx | param_idx | var_idx | opt_idx | loop_idx | branch_idx
        expected = {0, 1, 2, 3}
        assert all_idx == expected

    def test_step_order_contains_all_indices(self):
        """step_order should list every position index."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click")}),
            _pos(1, AlignmentType.VARIABLE, {"t1": _step("scroll")}),
        ]
        aligned = _aligned_set(positions, ["t1"])
        patterns = PatternExtractor().extract(aligned)

        assert patterns.step_order == [0, 1]


# ---------------------------------------------------------------------------
# User input boundary detection  (Requirement 5.1 — wait(True))
# ---------------------------------------------------------------------------

class TestUserInputBoundary:
    """User input boundaries are detected from wait actions with target=True."""

    def test_wait_true_target_detected(self):
        """A wait step with target=True is a user input boundary."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("wait", target=True)}),
        ]
        aligned = _aligned_set(positions, ["t1"])
        patterns = PatternExtractor().extract(aligned)

        assert 0 in patterns.user_input_boundaries

    def test_wait_true_in_args_detected(self):
        """A wait step with args={'wait': True} is a user input boundary."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {
                "t1": _step("wait", args={"wait": True}),
            }),
        ]
        aligned = _aligned_set(positions, ["t1"])
        patterns = PatternExtractor().extract(aligned)

        assert 0 in patterns.user_input_boundaries

    def test_non_wait_action_not_boundary(self):
        """A click action is not a user input boundary."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click")}),
        ]
        aligned = _aligned_set(positions, ["t1"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.user_input_boundaries) == 0


# ---------------------------------------------------------------------------
# Target consolidation  (Requirement 5.5)
# ---------------------------------------------------------------------------

class TestTargetConsolidation:
    """Deterministic steps get the best target from multiple observations."""

    def test_consolidates_targets_across_runs(self):
        """Multiple TargetSpec observations are merged into TargetSpecWithFallback."""
        t1_target = TargetSpecWithFallback(strategies=[
            TargetSpec(strategy="id", value="btn-submit"),
        ])
        t2_target = TargetSpecWithFallback(strategies=[
            TargetSpec(strategy="css", value="button.submit"),
        ])
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {
                "t1": _step("click", target=t1_target),
                "t2": _step("click", target=t2_target),
            }),
        ]
        aligned = _aligned_set(positions, ["t1", "t2"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.deterministic_steps) == 1
        det = patterns.deterministic_steps[0]
        # The target on the step should now be a TargetSpecWithFallback
        # containing both strategies.
        for step in det.steps.values():
            if step is not None:
                assert isinstance(step.target, TargetSpecWithFallback)
                strats = [s.strategy for s in step.target.strategies]
                assert "id" in strats
                assert "css" in strats


# ---------------------------------------------------------------------------
# Variable step variant recording  (Requirement 5.4)
# ---------------------------------------------------------------------------

class TestVariableStepVariants:
    """Variable steps record observed variants and frequencies."""

    def test_variants_recorded_in_metadata(self):
        """Variant counts are stored in the first non-None step's metadata."""
        positions = [
            _pos(0, AlignmentType.VARIABLE, {
                "t1": _step("click"),
                "t2": _step("scroll"),
                "t3": _step("click"),
            }),
        ]
        aligned = _aligned_set(positions, ["t1", "t2", "t3"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.variable_steps) == 1
        var_pos = patterns.variable_steps[0]
        # Find the first non-None step to check metadata.
        first_step = next(s for s in var_pos.steps.values() if s is not None)
        assert "variants" in first_step.metadata
        assert first_step.metadata["variants"]["click"] == 2
        assert first_step.metadata["variants"]["scroll"] == 1


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for the pattern extractor."""

    def test_empty_aligned_set(self):
        """An empty aligned set produces empty patterns."""
        aligned = _aligned_set([], ["t1"])
        patterns = PatternExtractor().extract(aligned)

        assert patterns.deterministic_steps == []
        assert patterns.parameterizable_steps == []
        assert patterns.variable_steps == []
        assert patterns.optional_steps == []
        assert patterns.branch_patterns == []
        assert patterns.loop_patterns == []
        assert patterns.user_input_boundaries == []
        assert patterns.step_order == []

    def test_single_position(self):
        """A single deterministic position is correctly extracted."""
        positions = [
            _pos(0, AlignmentType.DETERMINISTIC, {"t1": _step("click")}),
        ]
        aligned = _aligned_set(positions, ["t1"])
        patterns = PatternExtractor().extract(aligned)

        assert len(patterns.deterministic_steps) == 1
        assert patterns.step_order == [0]

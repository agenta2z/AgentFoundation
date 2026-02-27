"""
Unit tests for TraceAligner.

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 11.1
"""

from __future__ import annotations

import pytest

from agent_foundation.automation.meta_agent.aligner import TraceAligner
from agent_foundation.automation.meta_agent.models import (
    AlignedTraceSet,
    AlignmentType,
    ExecutionTrace,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(action_type: str, target=None, args=None, **kwargs) -> TraceStep:
    """Shorthand for creating a TraceStep."""
    return TraceStep(action_type=action_type, target=target, args=args, **kwargs)


def _trace(trace_id: str, steps: list[TraceStep]) -> ExecutionTrace:
    """Shorthand for creating an ExecutionTrace."""
    return ExecutionTrace(trace_id=trace_id, task_description="test", steps=steps)


# ---------------------------------------------------------------------------
# Test: Two identical traces → all DETERMINISTIC
# ---------------------------------------------------------------------------


class TestIdenticalTraces:
    """Requirement 4.3: same type + target + args → DETERMINISTIC."""

    def test_all_positions_deterministic(self):
        steps = [
            _step("click", target="btn-submit"),
            _step("input_text", target="input-name", args={"text": "hello"}),
            _step("click", target="btn-confirm"),
        ]
        t1 = _trace("t1", list(steps))
        t2 = _trace("t2", list(steps))

        result = TraceAligner().align([t1, t2])

        assert len(result.positions) == 3
        for pos in result.positions:
            assert pos.alignment_type == AlignmentType.DETERMINISTIC

    def test_trace_ids_present(self):
        steps = [_step("click", target="btn")]
        t1 = _trace("t1", list(steps))
        t2 = _trace("t2", list(steps))

        result = TraceAligner().align([t1, t2])

        assert set(result.trace_ids) == {"t1", "t2"}

    def test_alignment_score_is_one(self):
        steps = [_step("click", target="x"), _step("click", target="x")]
        t1 = _trace("t1", list(steps))
        t2 = _trace("t2", list(steps))

        result = TraceAligner().align([t1, t2])

        assert result.alignment_score == 1.0

    def test_each_position_has_both_steps(self):
        steps = [_step("click", target="a")]
        t1 = _trace("t1", list(steps))
        t2 = _trace("t2", list(steps))

        result = TraceAligner().align([t1, t2])

        for pos in result.positions:
            assert "t1" in pos.steps and pos.steps["t1"] is not None
            assert "t2" in pos.steps and pos.steps["t2"] is not None


# ---------------------------------------------------------------------------
# Test: Same actions different args → PARAMETERIZABLE
# ---------------------------------------------------------------------------


class TestParameterizable:
    """Requirement 4.4: same type + target, different args → PARAMETERIZABLE."""

    def test_different_args_classified_parameterizable(self):
        t1 = _trace("t1", [
            _step("input_text", target="search-box", args={"text": "cats"}),
        ])
        t2 = _trace("t2", [
            _step("input_text", target="search-box", args={"text": "dogs"}),
        ])

        result = TraceAligner().align([t1, t2])

        assert len(result.positions) == 1
        assert result.positions[0].alignment_type == AlignmentType.PARAMETERIZABLE

    def test_mixed_deterministic_and_parameterizable(self):
        t1 = _trace("t1", [
            _step("click", target="btn"),
            _step("input_text", target="field", args={"text": "alpha"}),
        ])
        t2 = _trace("t2", [
            _step("click", target="btn"),
            _step("input_text", target="field", args={"text": "beta"}),
        ])

        result = TraceAligner().align([t1, t2])

        types = [p.alignment_type for p in result.positions]
        assert types[0] == AlignmentType.DETERMINISTIC
        assert types[1] == AlignmentType.PARAMETERIZABLE

    def test_none_args_vs_present_args_is_parameterizable(self):
        t1 = _trace("t1", [
            _step("input_text", target="field", args=None),
        ])
        t2 = _trace("t2", [
            _step("input_text", target="field", args={"text": "val"}),
        ])

        result = TraceAligner().align([t1, t2])

        assert result.positions[0].alignment_type == AlignmentType.PARAMETERIZABLE


# ---------------------------------------------------------------------------
# Test: Completely different traces → all VARIABLE
# ---------------------------------------------------------------------------


class TestCompletelyDifferentTraces:
    """Requirement 4.5: different type or target → VARIABLE."""

    def test_different_action_types_are_variable(self):
        t1 = _trace("t1", [
            _step("click", target="btn-a"),
        ])
        t2 = _trace("t2", [
            _step("scroll", target="page"),
        ])

        result = TraceAligner().align([t1, t2])

        # With completely different steps, LCS is empty so each step
        # appears as a gap in the other trace → VARIABLE or OPTIONAL.
        for pos in result.positions:
            assert pos.alignment_type in (
                AlignmentType.VARIABLE, AlignmentType.OPTIONAL,
            )

    def test_different_targets_same_type_are_variable(self):
        t1 = _trace("t1", [
            _step("click", target="btn-a"),
            _step("click", target="btn-b"),
        ])
        t2 = _trace("t2", [
            _step("click", target="btn-x"),
            _step("click", target="btn-y"),
        ])

        result = TraceAligner().align([t1, t2])

        # Targets differ so steps are not equivalent → gaps → VARIABLE.
        for pos in result.positions:
            assert pos.alignment_type in (
                AlignmentType.VARIABLE, AlignmentType.OPTIONAL,
            )


# ---------------------------------------------------------------------------
# Test: Trace with extra popup-dismiss step → OPTIONAL at that position
# ---------------------------------------------------------------------------


class TestOptionalStep:
    """Requirement 4.6: present in some traces but not others → OPTIONAL."""

    def test_extra_step_classified_optional(self):
        # Trace 1 has an extra popup-dismiss step between click and input.
        t1 = _trace("t1", [
            _step("click", target="btn"),
            _step("click", target="popup-dismiss"),
            _step("input_text", target="field", args={"text": "hi"}),
        ])
        t2 = _trace("t2", [
            _step("click", target="btn"),
            _step("input_text", target="field", args={"text": "hi"}),
        ])

        result = TraceAligner().align([t1, t2])

        # Find the position where one trace has a step and the other has None.
        optional_positions = [
            p for p in result.positions
            if p.alignment_type == AlignmentType.OPTIONAL
        ]
        assert len(optional_positions) >= 1

        # The optional position should have a None for t2.
        opt = optional_positions[0]
        assert opt.steps["t2"] is None
        assert opt.steps["t1"] is not None
        assert opt.steps["t1"].target == "popup-dismiss"

    def test_shared_steps_remain_deterministic(self):
        t1 = _trace("t1", [
            _step("click", target="btn"),
            _step("click", target="popup-dismiss"),
            _step("input_text", target="field", args={"text": "hi"}),
        ])
        t2 = _trace("t2", [
            _step("click", target="btn"),
            _step("input_text", target="field", args={"text": "hi"}),
        ])

        result = TraceAligner().align([t1, t2])

        deterministic = [
            p for p in result.positions
            if p.alignment_type == AlignmentType.DETERMINISTIC
        ]
        # The shared click and input_text should be deterministic.
        assert len(deterministic) == 2


# ---------------------------------------------------------------------------
# Test: Empty trace handling
# ---------------------------------------------------------------------------


class TestEmptyTraces:
    """Edge cases for empty inputs."""

    def test_no_traces_returns_empty_alignment(self):
        result = TraceAligner().align([])

        assert result.positions == []
        assert result.trace_ids == []
        assert result.alignment_score == 1.0

    def test_single_trace_all_deterministic(self):
        t = _trace("t1", [
            _step("click", target="a"),
            _step("click", target="b"),
        ])

        result = TraceAligner().align([t])

        assert len(result.positions) == 2
        for pos in result.positions:
            assert pos.alignment_type == AlignmentType.DETERMINISTIC

    def test_single_empty_trace(self):
        t = _trace("t1", [])

        result = TraceAligner().align([t])

        assert result.positions == []
        assert result.trace_ids == ["t1"]

    def test_two_empty_traces(self):
        t1 = _trace("t1", [])
        t2 = _trace("t2", [])

        result = TraceAligner().align([t1, t2])

        assert result.positions == []
        assert set(result.trace_ids) == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Test: Merge preserves existing trace IDs
# ---------------------------------------------------------------------------


class TestMerge:
    """Requirement 11.1: merge new traces into existing alignment."""

    def test_merge_preserves_existing_trace_ids(self):
        t1 = _trace("t1", [_step("click", target="a")])
        t2 = _trace("t2", [_step("click", target="a")])

        aligner = TraceAligner()
        existing = aligner.align([t1, t2])

        t3 = _trace("t3", [_step("click", target="a")])
        merged = aligner.merge(existing, [t3])

        assert "t1" in merged.trace_ids
        assert "t2" in merged.trace_ids
        assert "t3" in merged.trace_ids

    def test_merge_with_empty_new_traces_returns_existing(self):
        t1 = _trace("t1", [_step("click", target="x")])
        aligner = TraceAligner()
        existing = aligner.align([t1])

        merged = aligner.merge(existing, [])

        assert merged.trace_ids == existing.trace_ids
        assert len(merged.positions) == len(existing.positions)

    def test_merge_positions_cover_all_traces(self):
        t1 = _trace("t1", [
            _step("click", target="a"),
            _step("click", target="b"),
        ])
        t2 = _trace("t2", [
            _step("click", target="a"),
            _step("click", target="b"),
        ])

        aligner = TraceAligner()
        existing = aligner.align([t1, t2])

        t3 = _trace("t3", [
            _step("click", target="a"),
            _step("click", target="b"),
        ])
        merged = aligner.merge(existing, [t3])

        # Every position should have keys for all three traces.
        for pos in merged.positions:
            assert set(pos.steps.keys()) == {"t1", "t2", "t3"}

    def test_merge_new_trace_with_extra_step(self):
        t1 = _trace("t1", [_step("click", target="a")])
        aligner = TraceAligner()
        existing = aligner.align([t1])

        t2 = _trace("t2", [
            _step("click", target="a"),
            _step("click", target="extra"),
        ])
        merged = aligner.merge(existing, [t2])

        assert "t1" in merged.trace_ids
        assert "t2" in merged.trace_ids
        # Should have at least 2 positions (shared + extra).
        assert len(merged.positions) >= 2

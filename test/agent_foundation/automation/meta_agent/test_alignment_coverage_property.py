"""
Property-based test for alignment coverage.

Feature: meta-agent-workflow, Property 9: Alignment covers all traces

*For any* set of N execution traces provided to the TraceAligner, the
resulting AlignedTraceSet SHALL have trace_ids containing all N trace IDs,
and each AlignedPosition's steps dict SHALL have keys for all N trace IDs.

**Validates: Requirements 4.1**
"""

from __future__ import annotations

from typing import List

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.aligner import TraceAligner
from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Pool of action types to draw from — keeps generated traces realistic
ACTION_TYPES = ["click", "input_text", "scroll", "visit_url", "wait", "no_op"]

# Pool of simple string targets
TARGETS = ["btn-submit", "search-box", "nav-link", "popup-dismiss", "field-name", None]


def trace_step_st() -> st.SearchStrategy[TraceStep]:
    """Generate a random TraceStep with action type and optional target."""
    return st.builds(
        TraceStep,
        action_type=st.sampled_from(ACTION_TYPES),
        target=st.sampled_from(TARGETS),
        args=st.one_of(
            st.none(),
            st.fixed_dictionaries({"text": st.text(min_size=1, max_size=10)}),
        ),
    )


def execution_trace_st(trace_id: str) -> st.SearchStrategy[ExecutionTrace]:
    """Generate an ExecutionTrace with a fixed trace_id and random steps."""
    return st.builds(
        ExecutionTrace,
        trace_id=st.just(trace_id),
        task_description=st.just("test task"),
        steps=st.lists(trace_step_st(), min_size=0, max_size=8),
    )


def traces_st() -> st.SearchStrategy[List[ExecutionTrace]]:
    """Generate a list of 1-6 traces with unique trace IDs."""
    return st.integers(min_value=1, max_value=6).flatmap(
        lambda n: st.tuples(
            *[execution_trace_st(f"trace_{i}") for i in range(n)]
        ).map(list)
    )


# ---------------------------------------------------------------------------
# Property 9: Alignment covers all traces
# ---------------------------------------------------------------------------


class TestAlignmentCoverageProperty:
    """
    Property 9: Alignment covers all traces

    For any set of N execution traces, the AlignedTraceSet.trace_ids
    contains all N trace IDs, and each AlignedPosition's steps dict
    has keys for all N trace IDs.

    **Validates: Requirements 4.1**
    """

    @given(traces=traces_st())
    @settings(max_examples=200)
    def test_trace_ids_contains_all_input_trace_ids(self, traces: List[ExecutionTrace]):
        """
        AlignedTraceSet.trace_ids must contain every input trace's ID.
        """
        aligner = TraceAligner()
        result = aligner.align(traces)

        input_ids = {t.trace_id for t in traces}
        result_ids = set(result.trace_ids)

        assert input_ids == result_ids, (
            f"Expected trace_ids {input_ids}, got {result_ids}"
        )

    @given(traces=traces_st())
    @settings(max_examples=200)
    def test_each_position_has_keys_for_all_traces(self, traces: List[ExecutionTrace]):
        """
        Every AlignedPosition.steps dict must have a key for each trace ID,
        even if the value is None (indicating a gap).
        """
        aligner = TraceAligner()
        result = aligner.align(traces)

        input_ids = {t.trace_id for t in traces}

        for pos in result.positions:
            pos_keys = set(pos.steps.keys())
            assert pos_keys == input_ids, (
                f"Position {pos.index}: expected keys {input_ids}, "
                f"got {pos_keys}"
            )

    @given(traces=traces_st())
    @settings(max_examples=200)
    def test_all_steps_accounted_for_in_alignment(self, traces: List[ExecutionTrace]):
        """
        For each input trace, the number of non-None entries across all
        positions must equal the number of steps in that trace.
        """
        aligner = TraceAligner()
        result = aligner.align(traces)

        for trace in traces:
            non_none_count = sum(
                1 for pos in result.positions
                if pos.steps.get(trace.trace_id) is not None
            )
            assert non_none_count == len(trace.steps), (
                f"Trace '{trace.trace_id}': expected {len(trace.steps)} "
                f"non-None entries, got {non_none_count}"
            )

    @given(traces=traces_st())
    @settings(max_examples=200)
    def test_trace_ids_count_matches_input_count(self, traces: List[ExecutionTrace]):
        """
        The length of trace_ids must equal the number of input traces.
        """
        aligner = TraceAligner()
        result = aligner.align(traces)

        assert len(result.trace_ids) == len(traces), (
            f"Expected {len(traces)} trace IDs, got {len(result.trace_ids)}"
        )

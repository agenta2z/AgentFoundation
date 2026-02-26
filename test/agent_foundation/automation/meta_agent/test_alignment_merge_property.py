"""
Property-based test for alignment merge preserves existing traces.

Feature: meta-agent-workflow, Property 23: Alignment merge preserves existing traces

For any existing AlignedTraceSet and new traces, the merged AlignedTraceSet's
trace_ids SHALL be the union of the existing trace_ids and the new trace IDs.

**Validates: Requirements 11.1**
"""

from __future__ import annotations

from typing import List

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.aligner import TraceAligner
from science_modeling_tools.automation.meta_agent.models import (
    AlignedTraceSet,
    ExecutionTrace,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url", "wait", "no_op"]
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
        steps=st.lists(trace_step_st(), min_size=1, max_size=6),
    )


def existing_traces_st() -> st.SearchStrategy[List[ExecutionTrace]]:
    """Generate 1-4 traces to form the 'existing' aligned set."""
    return st.integers(min_value=1, max_value=4).flatmap(
        lambda n: st.tuples(
            *[execution_trace_st(f"existing_{i}") for i in range(n)]
        ).map(list)
    )


def new_traces_st(offset: int = 0) -> st.SearchStrategy[List[ExecutionTrace]]:
    """Generate 1-3 new traces with IDs distinct from existing ones."""
    return st.integers(min_value=1, max_value=3).flatmap(
        lambda n: st.tuples(
            *[execution_trace_st(f"new_{offset}_{i}") for i in range(n)]
        ).map(list)
    )


# ---------------------------------------------------------------------------
# Property 23: Alignment merge preserves existing traces
# ---------------------------------------------------------------------------


class TestAlignmentMergeProperty:
    """
    Property 23: Alignment merge preserves existing traces

    For any existing AlignedTraceSet and new traces, the merged
    AlignedTraceSet's trace_ids SHALL be the union of the existing
    trace_ids and the new trace IDs.

    **Validates: Requirements 11.1**
    """

    @given(
        existing_traces=existing_traces_st(),
        new_traces=new_traces_st(),
    )
    @settings(max_examples=200)
    def test_merged_trace_ids_is_union(
        self,
        existing_traces: List[ExecutionTrace],
        new_traces: List[ExecutionTrace],
    ):
        """
        Merged trace_ids = union of existing trace_ids and new trace IDs.

        **Validates: Requirements 11.1**
        """
        aligner = TraceAligner()

        # Build the existing AlignedTraceSet from the initial traces.
        existing_aligned = aligner.align(existing_traces)

        # Merge new traces into the existing alignment.
        merged = aligner.merge(existing_aligned, new_traces)

        existing_ids = set(existing_aligned.trace_ids)
        new_ids = {t.trace_id for t in new_traces}
        expected_ids = existing_ids | new_ids

        assert set(merged.trace_ids) == expected_ids, (
            f"Expected trace_ids {expected_ids}, got {set(merged.trace_ids)}"
        )

    @given(
        existing_traces=existing_traces_st(),
        new_traces=new_traces_st(),
    )
    @settings(max_examples=200)
    def test_merged_positions_have_keys_for_all_traces(
        self,
        existing_traces: List[ExecutionTrace],
        new_traces: List[ExecutionTrace],
    ):
        """
        Every position in the merged alignment has step keys for all
        trace IDs (existing + new).

        **Validates: Requirements 11.1**
        """
        aligner = TraceAligner()

        existing_aligned = aligner.align(existing_traces)
        merged = aligner.merge(existing_aligned, new_traces)

        expected_ids = set(merged.trace_ids)

        for pos in merged.positions:
            pos_keys = set(pos.steps.keys())
            assert pos_keys == expected_ids, (
                f"Position {pos.index}: expected keys {expected_ids}, "
                f"got {pos_keys}"
            )

    @given(existing_traces=existing_traces_st())
    @settings(max_examples=200)
    def test_merge_with_empty_new_traces_preserves_existing(
        self,
        existing_traces: List[ExecutionTrace],
    ):
        """
        Merging with an empty list of new traces returns the existing
        alignment unchanged.

        **Validates: Requirements 11.1**
        """
        aligner = TraceAligner()

        existing_aligned = aligner.align(existing_traces)
        merged = aligner.merge(existing_aligned, [])

        assert set(merged.trace_ids) == set(existing_aligned.trace_ids), (
            f"Expected trace_ids {set(existing_aligned.trace_ids)}, "
            f"got {set(merged.trace_ids)}"
        )

    @given(
        existing_traces=existing_traces_st(),
        new_traces=new_traces_st(),
    )
    @settings(max_examples=200)
    def test_existing_trace_steps_preserved_in_merge(
        self,
        existing_traces: List[ExecutionTrace],
        new_traces: List[ExecutionTrace],
    ):
        """
        For each existing trace, the number of non-None entries across
        merged positions equals the original step count.

        **Validates: Requirements 11.1**
        """
        aligner = TraceAligner()

        existing_aligned = aligner.align(existing_traces)
        merged = aligner.merge(existing_aligned, new_traces)

        for trace in existing_traces:
            non_none_count = sum(
                1 for pos in merged.positions
                if pos.steps.get(trace.trace_id) is not None
            )
            assert non_none_count == len(trace.steps), (
                f"Trace '{trace.trace_id}': expected {len(trace.steps)} "
                f"non-None entries after merge, got {non_none_count}"
            )

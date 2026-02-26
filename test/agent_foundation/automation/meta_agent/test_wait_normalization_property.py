"""
Property-based test for wait duration normalization.

Feature: meta-agent-workflow, Property 6: Wait duration normalization to median

*For any* list of wait action durations observed across traces, the
normalized duration SHALL equal the median of the observed values.

**Validates: Requirements 2.4**
"""

import statistics

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)
from science_modeling_tools.automation.meta_agent.normalizer import TraceNormalizer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Positive floats for wait durations (reasonable range to avoid float issues)
positive_duration_st = st.floats(min_value=0.01, max_value=3600.0, allow_nan=False, allow_infinity=False)

# List of at least one positive duration
duration_list_st = st.lists(positive_duration_st, min_size=1, max_size=50)


def _make_wait_step(seconds: float) -> TraceStep:
    """Create a wait TraceStep with the given duration."""
    return TraceStep(action_type="wait", args={"seconds": seconds})


def _make_non_wait_step(action_type: str = "click") -> TraceStep:
    """Create a non-wait TraceStep."""
    return TraceStep(action_type=action_type)


def _make_trace(trace_id: str, steps: list) -> ExecutionTrace:
    """Create an ExecutionTrace with the given steps."""
    return ExecutionTrace(
        trace_id=trace_id,
        task_description="test task",
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Property 6: Wait duration normalization to median
# ---------------------------------------------------------------------------


class TestWaitNormalizationProperty:
    """
    Property 6: Wait duration normalization to median

    *For any* list of wait action durations observed across traces, the
    normalized duration SHALL equal the median of the observed values.

    **Validates: Requirements 2.4**
    """

    @given(durations=duration_list_st)
    @settings(max_examples=200)
    def test_all_wait_steps_normalized_to_median(self, durations: list):
        """
        For any list of wait durations spread across traces, after
        normalization every wait step's seconds equals the median of
        the original durations.
        """
        # Create one trace per duration, each with a single wait step
        traces = [
            _make_trace(f"trace-{i}", [_make_wait_step(d)])
            for i, d in enumerate(durations)
        ]

        expected_median = statistics.median(durations)

        normalizer = TraceNormalizer()
        normalized = normalizer.normalize(traces)

        for trace in normalized:
            for step in trace.steps:
                if step.action_type == "wait" and step.args and "seconds" in step.args:
                    assert step.args["seconds"] == expected_median, (
                        f"Expected median {expected_median}, "
                        f"got {step.args['seconds']} for durations {durations}"
                    )

    @given(durations=duration_list_st)
    @settings(max_examples=200)
    def test_wait_steps_in_single_trace_normalized_to_median(self, durations: list):
        """
        When multiple wait steps exist within a single trace, all are
        normalized to the median of all observed wait durations.
        """
        steps = [_make_wait_step(d) for d in durations]
        traces = [_make_trace("trace-0", steps)]

        expected_median = statistics.median(durations)

        normalizer = TraceNormalizer()
        normalized = normalizer.normalize(traces)

        for step in normalized[0].steps:
            if step.action_type == "wait" and step.args and "seconds" in step.args:
                assert step.args["seconds"] == expected_median

    @given(durations=duration_list_st)
    @settings(max_examples=200)
    def test_non_wait_steps_unaffected(self, durations: list):
        """
        Non-wait steps are not modified by wait duration normalization.
        """
        # Mix wait and non-wait steps in a single trace
        steps = []
        for d in durations:
            steps.append(_make_non_wait_step("click"))
            steps.append(_make_wait_step(d))

        traces = [_make_trace("trace-0", steps)]

        normalizer = TraceNormalizer()
        normalized = normalizer.normalize(traces)

        for step in normalized[0].steps:
            if step.action_type == "click":
                # Non-wait steps should not have seconds injected
                assert step.args is None or "seconds" not in (step.args or {})

    @given(duration=positive_duration_st)
    @settings(max_examples=100)
    def test_single_duration_is_its_own_median(self, duration: float):
        """
        A single wait duration normalizes to itself (median of one
        element is that element).
        """
        traces = [_make_trace("trace-0", [_make_wait_step(duration)])]

        normalizer = TraceNormalizer()
        normalized = normalizer.normalize(traces)

        step = normalized[0].steps[0]
        assert step.args["seconds"] == duration

    @given(
        durations=duration_list_st,
        extra_args=st.dictionaries(
            keys=st.text(min_size=1, max_size=10).filter(lambda k: k != "seconds"),
            values=st.text(min_size=1, max_size=10),
            min_size=0,
            max_size=3,
        ),
    )
    @settings(max_examples=200)
    def test_other_wait_args_preserved(self, durations: list, extra_args: dict):
        """
        Wait step args other than 'seconds' are preserved after
        normalization. The 'wait' key added by UserInputsRequired
        conversion is also preserved if present.
        """
        steps = []
        for d in durations:
            args = {"seconds": d, **extra_args}
            steps.append(TraceStep(action_type="wait", args=args))

        traces = [_make_trace("trace-0", steps)]

        normalizer = TraceNormalizer()
        normalized = normalizer.normalize(traces)

        expected_median = statistics.median(durations)

        for step in normalized[0].steps:
            assert step.args["seconds"] == expected_median
            for key, value in extra_args.items():
                assert step.args[key] == value, (
                    f"Extra arg {key!r} was lost or changed after normalization"
                )

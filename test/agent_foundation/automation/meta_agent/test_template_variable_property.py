"""
Property-based test for parameterizable step template variable detection.

Feature: meta-agent-workflow, Property 13: Parameterizable step template
variable detection

For any AlignedTraceSet with PARAMETERIZABLE positions where steps have the
same action_type and target but different args, after extraction the
ParameterizableInfo SHALL correctly split variable_args vs constant_args
such that variable_args.keys() | constant_args.keys() == full set of
argument keys across all traces at that position.

**Validates: Requirements 5.6**
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from hypothesis import given, settings, strategies as st

from agent_foundation.automation.meta_agent.models import (
    AlignedPosition,
    AlignedTraceSet,
    AlignmentType,
    TraceStep,
)
from agent_foundation.automation.meta_agent.pattern_extractor import (
    PatternExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTION_TYPES = ["click", "input_text", "scroll", "visit_url"]
TARGETS = ["btn-ok", "search-box", "nav-link", "field-email"]

# Arg keys that can appear in steps.
ARG_KEYS = ["text", "value", "delay", "direction", "url", "count"]

# Possible arg values — kept simple so repr-based equality works reliably.
ARG_VALUES = st.one_of(
    st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnop"),
    st.integers(min_value=0, max_value=100),
    st.booleans(),
)


def _make_step(
    action_type: str,
    target: str,
    args: Optional[Dict[str, Any]] = None,
) -> TraceStep:
    return TraceStep(action_type=action_type, target=target, args=args)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def parameterizable_trace_set_st(draw) -> AlignedTraceSet:
    """Generate an AlignedTraceSet with PARAMETERIZABLE positions.

    Each PARAMETERIZABLE position has the same action_type and target across
    all traces, but at least one arg key differs in value across traces.
    """
    num_traces = draw(st.integers(min_value=2, max_value=5))
    trace_ids = [f"trace_{i}" for i in range(num_traces)]

    num_param = draw(st.integers(min_value=1, max_value=5))
    num_deterministic = draw(st.integers(min_value=0, max_value=3))

    positions: List[AlignedPosition] = []
    idx = 0

    # Generate PARAMETERIZABLE positions.
    for _ in range(num_param):
        action_type = draw(st.sampled_from(ACTION_TYPES))
        target = draw(st.sampled_from(TARGETS))

        # Choose arg keys for this position (at least 1).
        num_keys = draw(st.integers(min_value=1, max_value=4))
        chosen_keys = draw(
            st.lists(
                st.sampled_from(ARG_KEYS),
                min_size=num_keys,
                max_size=num_keys,
                unique=True,
            )
        )

        # Pick at least one key to be variable (different across traces).
        num_variable = draw(st.integers(min_value=1, max_value=len(chosen_keys)))
        variable_keys = set(chosen_keys[:num_variable])
        constant_keys = set(chosen_keys[num_variable:])

        # Generate constant values.
        constant_vals: Dict[str, Any] = {}
        for k in constant_keys:
            constant_vals[k] = draw(ARG_VALUES)

        # Generate per-trace args ensuring variable keys differ.
        all_trace_args: List[Dict[str, Any]] = []
        for _ in range(num_traces):
            args: Dict[str, Any] = {}
            for k in constant_keys:
                args[k] = constant_vals[k]
            for k in variable_keys:
                args[k] = draw(ARG_VALUES)
            all_trace_args.append(args)

        # Ensure variable keys actually differ across at least two traces.
        # If by chance all values are the same, force the last trace to differ.
        for k in variable_keys:
            reprs = {repr(a[k]) for a in all_trace_args}
            if len(reprs) == 1:
                # Force a different value for the last trace.
                forced = draw(ARG_VALUES)
                # Keep drawing until we get something different.
                attempts = 0
                while repr(forced) == repr(all_trace_args[0][k]) and attempts < 20:
                    forced = draw(ARG_VALUES)
                    attempts += 1
                if repr(forced) != repr(all_trace_args[0][k]):
                    all_trace_args[-1][k] = forced

        steps: Dict[str, Optional[TraceStep]] = {}
        for i, tid in enumerate(trace_ids):
            steps[tid] = _make_step(action_type, target, all_trace_args[i])

        positions.append(
            AlignedPosition(
                index=idx,
                alignment_type=AlignmentType.PARAMETERIZABLE,
                steps=steps,
                confidence=1.0,
            )
        )
        idx += 1

    # Generate DETERMINISTIC filler positions.
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
# Property 13: Parameterizable step template variable detection
# ---------------------------------------------------------------------------


class TestTemplateVariableProperty:
    """
    Property 13: Parameterizable step template variable detection

    For each parameterizable step, variable_args.keys() | constant_args.keys()
    equals the full set of argument keys across all traces at that position.

    **Validates: Requirements 5.6**
    """

    @given(aligned_set=parameterizable_trace_set_st())
    @settings(max_examples=200)
    def test_variable_constant_keys_cover_all_arg_keys(
        self, aligned_set: AlignedTraceSet,
    ):
        """
        After extraction, for each parameterizable step the union of
        variable_args keys and constant_args keys equals the full set
        of argument keys observed across all traces at that position.

        **Validates: Requirements 5.6**
        """
        extractor = PatternExtractor()
        patterns = extractor.extract(aligned_set)

        for pos, param_info in patterns.parameterizable_steps:
            # Compute the full set of arg keys from the raw position steps.
            expected_keys: Set[str] = set()
            for step in pos.steps.values():
                if step is not None and step.args:
                    expected_keys.update(step.args.keys())

            actual_keys = set(param_info.variable_args.keys()) | set(
                param_info.constant_args.keys()
            )

            assert actual_keys == expected_keys, (
                f"Position {pos.index}: key coverage mismatch. "
                f"variable_args keys={set(param_info.variable_args.keys())}, "
                f"constant_args keys={set(param_info.constant_args.keys())}, "
                f"union={actual_keys}, expected={expected_keys}"
            )

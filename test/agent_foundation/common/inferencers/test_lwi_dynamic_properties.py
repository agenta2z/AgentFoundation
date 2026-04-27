"""Property-based tests for LinearWorkflowInferencer dynamic mode.

Uses Hypothesis to verify correctness properties across randomized inputs.
Each test is tagged with the feature and property number it validates.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sequential_inferencer(results_list):
    """Create a mock inferencer that returns results from a list in order.

    Each call to ainfer() returns the next item from results_list.
    The mock tracks how many times it has been called.
    """
    call_count = [0]
    mock = MagicMock(spec=[])

    async def _ainfer(inp, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(results_list):
            return results_list[idx]
        return f"overflow_{idx}"

    mock.ainfer = _ainfer
    mock._call_count = call_count
    return mock


# ---------------------------------------------------------------------------
# Property 1: Dynamic mode state tracks all step results
# Feature: dynamic-linear-workflow, Property 1: Dynamic mode state tracks all step results
# Validates: Requirements 7.2, 7.3
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    step_count=st.integers(min_value=1, max_value=15),
    result_strings=st.lists(
        st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        min_size=15,
        max_size=15,
    ),
)
def test_dynamic_mode_state_tracks_all_step_results(step_count, result_strings):
    """For any LWI in dynamic mode with N steps executed,
    state["dynamic_step_results"] contains exactly N entries in execution order,
    and state["dynamic_step_count"] equals N.

    **Validates: Requirements 7.2, 7.3**
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )

    # Use the first step_count result strings
    expected_results = result_strings[:step_count]

    # Step 0 uses initial_inferencer, steps 1..N-1 use followup_inferencer.
    # We give the initial inferencer the first result, and the followup
    # inferencer the remaining results.
    initial_inf = _make_sequential_inferencer([expected_results[0]])
    followup_inf = _make_sequential_inferencer(expected_results[1:])

    lwi = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=initial_inf,
        default_followup_inferencer=followup_inf,
        max_dynamic_steps=step_count,
        end_condition=None,
    )

    # infer() bridges to async _ainfer() via _run_async
    state = lwi.infer("test input")

    # Assert state["dynamic_step_results"] has exactly N entries
    assert len(state["dynamic_step_results"]) == step_count, (
        f"Expected {step_count} step results, got {len(state['dynamic_step_results'])}"
    )

    # Assert state["dynamic_step_count"] == N
    assert state["dynamic_step_count"] == step_count, (
        f"Expected dynamic_step_count={step_count}, got {state['dynamic_step_count']}"
    )

    # Assert results are in execution order
    assert state["dynamic_step_results"] == expected_results, (
        f"Results not in execution order: expected {expected_results}, "
        f"got {state['dynamic_step_results']}"
    )


# ---------------------------------------------------------------------------
# Property 2: Dynamic mode inferencer selection from result tuples
# Feature: dynamic-linear-workflow, Property 2: Dynamic mode inferencer selection from result tuples
# Validates: Requirements 3.1, 3.2
# ---------------------------------------------------------------------------


def _build_inferencer_selection_scenario(tuple_flags):
    """Build a complete set of inferencers for testing inferencer selection.

    Pre-creates all inferencers (initial, followup, and per-step overrides)
    so that every inferencer — regardless of which one is called — uses the
    shared ``tuple_flags`` list to decide whether to return a tuple or plain
    result.

    Returns (initial_inf, followup_inf, call_log) where call_log records
    the name of the inferencer called at each global step.
    """
    num_steps = len(tuple_flags)
    call_log = []

    # Pre-create override inferencers for every step that could be reached
    # via a tuple return.  override_inferencers[i] is the inferencer that
    # will be specified as "next" by step i-1 when tuple_flags[i-1] is True.
    override_inferencers = {}

    def _make_inf(name):
        """Create a mock inferencer that logs *name* and consults tuple_flags."""
        mock = MagicMock(spec=[])
        mock._name = name

        async def _ainfer(inp, **kwargs):
            global_idx = len(call_log)
            call_log.append(name)
            result = f"result_{global_idx}"
            # Decide whether to return a tuple based on the global step index
            if global_idx < num_steps and tuple_flags[global_idx]:
                next_step = global_idx + 1
                # Return (result, next_inferencer) — the next step should
                # use the override inferencer for that step index.
                next_inf = override_inferencers.get(next_step)
                if next_inf is not None:
                    return (result, next_inf)
                # Fallback: shouldn't happen if we pre-created all overrides
                return (result, None)
            return result

        mock.ainfer = _ainfer
        return mock

    # Pre-create all override inferencers (one per possible step index 1..N-1)
    for i in range(1, num_steps):
        override_inferencers[i] = _make_inf(f"override_{i}")

    initial_inf = _make_inf("initial")
    followup_inf = _make_inf("followup")

    return initial_inf, followup_inf, call_log


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    tuple_flags=st.lists(
        st.booleans(),
        min_size=2,
        max_size=10,
    ),
)
def test_dynamic_mode_inferencer_selection_from_result_tuples(tuple_flags):
    """For any dynamic step returning (result, next_inferencer), the next step
    uses the specified inferencer; for non-tuple results, uses
    default_followup_inferencer; first step always uses default_initial_inferencer.

    **Validates: Requirements 3.1, 3.2**
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )

    num_steps = len(tuple_flags)
    initial_inf, followup_inf, call_log = _build_inferencer_selection_scenario(
        tuple_flags
    )

    lwi = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=initial_inf,
        default_followup_inferencer=followup_inf,
        max_dynamic_steps=num_steps,
        end_condition=None,
    )

    state = lwi.infer("test input")

    # Verify correct number of steps executed
    assert len(call_log) == num_steps, (
        f"Expected {num_steps} calls, got {len(call_log)}"
    )

    # Verify first step always uses initial inferencer
    assert call_log[0] == "initial", (
        f"First step should use 'initial', got '{call_log[0]}'"
    )

    # Verify subsequent steps use the correct inferencer
    for i in range(1, num_steps):
        prev_flag = tuple_flags[i - 1]
        if prev_flag:
            # Previous step returned a tuple → this step should use the override
            expected_name = f"override_{i}"
        else:
            # Previous step returned plain result → this step should use followup
            expected_name = "followup"
        assert call_log[i] == expected_name, (
            f"Step {i}: expected inferencer '{expected_name}', got '{call_log[i]}'. "
            f"tuple_flags={tuple_flags}, call_log={call_log}"
        )


# ---------------------------------------------------------------------------
# Property 3: Dynamic mode chain termination
# Feature: dynamic-linear-workflow, Property 3: Dynamic mode chain termination
# Validates: Requirements 5.2, 6.2
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    max_dynamic_steps=st.integers(min_value=1, max_value=20),
    end_at_step=st.integers(min_value=1, max_value=25),
)
def test_dynamic_mode_chain_termination(max_dynamic_steps, end_at_step):
    """For any LWI with end_condition and max_dynamic_steps, executed steps
    equals min(first_step_where_end_condition_returns_True, max_dynamic_steps);
    when end_condition is None, exactly max_dynamic_steps steps execute.

    **Validates: Requirements 5.2, 6.2**
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )

    # We need enough results for the maximum possible steps
    max_possible = max(max_dynamic_steps, end_at_step)
    results = [f"result_{i}" for i in range(max_possible)]

    # --- Case 1: with end_condition ---
    def end_condition(state, result):
        return state["dynamic_step_count"] >= end_at_step

    initial_inf = _make_sequential_inferencer([results[0]])
    followup_inf = _make_sequential_inferencer(results[1:])

    lwi = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=initial_inf,
        default_followup_inferencer=followup_inf,
        max_dynamic_steps=max_dynamic_steps,
        end_condition=end_condition,
    )

    state = lwi.infer("test input")

    expected_steps = min(end_at_step, max_dynamic_steps)
    assert state["dynamic_step_count"] == expected_steps, (
        f"With end_condition: expected {expected_steps} steps "
        f"(min({end_at_step}, {max_dynamic_steps})), "
        f"got {state['dynamic_step_count']}"
    )

    # --- Case 2: end_condition is None → exactly max_dynamic_steps ---
    initial_inf2 = _make_sequential_inferencer([results[0]])
    followup_inf2 = _make_sequential_inferencer(results[1:])

    lwi2 = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=initial_inf2,
        default_followup_inferencer=followup_inf2,
        max_dynamic_steps=max_dynamic_steps,
        end_condition=None,
    )

    state2 = lwi2.infer("test input")

    assert state2["dynamic_step_count"] == max_dynamic_steps, (
        f"With end_condition=None: expected {max_dynamic_steps} steps, "
        f"got {state2['dynamic_step_count']}"
    )


# ---------------------------------------------------------------------------
# Helpers for Property 4
# ---------------------------------------------------------------------------


def _make_recording_inferencer(name, transform_fn=None):
    """Create a mock inferencer that records what input it received.

    Each call to ainfer() appends the received input to ``received_inputs``.
    The return value is ``transform_fn(input, call_index)`` if provided,
    otherwise ``f"{name}_out_{call_index}"``.

    Returns (mock, received_inputs) where received_inputs is a list that
    grows with each call.
    """
    received_inputs = []
    mock = MagicMock(spec=[])
    mock._name = name

    async def _ainfer(inp, **kwargs):
        idx = len(received_inputs)
        received_inputs.append(inp)
        if transform_fn is not None:
            return transform_fn(inp, idx)
        return f"{name}_out_{idx}"

    mock.ainfer = _ainfer
    return mock, received_inputs


# ---------------------------------------------------------------------------
# Property 4: Dynamic mode pipeline input passing
# Feature: dynamic-linear-workflow, Property 4: Dynamic mode pipeline input passing
# Validates: Requirements 8.1, 8.2, 8.3
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    step_count=st.integers(min_value=2, max_value=10),
    initial_input=st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(whitelist_categories=("L", "N")),
    ),
)
def test_dynamic_mode_pipeline_input_passing(step_count, initial_input):
    """Step 1 receives inference_input, step K (K>1) receives step K-1's result;
    when dynamic_input_builder is provided, it is called with (state, previous_result)
    instead.

    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
        LinearWorkflowInferencer,
    )

    # --- Case 1: Without dynamic_input_builder (default pipeline) ---
    # Each inferencer appends its step index to the input so we can trace
    # the pipeline chain.
    def transform(inp, idx):
        return f"{inp}->step{idx}"

    initial_inf, initial_received = _make_recording_inferencer(
        "initial", transform_fn=lambda inp, idx: f"{inp}->step0"
    )
    followup_inf, followup_received = _make_recording_inferencer(
        "followup", transform_fn=lambda inp, idx: f"{inp}->step{idx + 1}"
    )

    lwi = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=initial_inf,
        default_followup_inferencer=followup_inf,
        max_dynamic_steps=step_count,
        end_condition=None,
    )

    state = lwi.infer(initial_input)

    # Step 0 should have received the original inference_input
    assert len(initial_received) == 1, (
        f"Initial inferencer should be called once, got {len(initial_received)}"
    )
    assert initial_received[0] == initial_input, (
        f"Step 0 should receive original input '{initial_input}', "
        f"got '{initial_received[0]}'"
    )

    # Steps 1..N-1 should each receive the previous step's result
    assert len(followup_received) == step_count - 1, (
        f"Followup inferencer should be called {step_count - 1} times, "
        f"got {len(followup_received)}"
    )

    # Build expected chain: step 0 output is "{initial_input}->step0",
    # step 1 receives that, outputs "{initial_input}->step0->step1", etc.
    expected_chain_output = f"{initial_input}->step0"
    for k in range(1, step_count):
        # Step K should receive step K-1's output
        assert followup_received[k - 1] == expected_chain_output, (
            f"Step {k} should receive '{expected_chain_output}', "
            f"got '{followup_received[k - 1]}'"
        )
        expected_chain_output = f"{expected_chain_output}->step{k}"

    # --- Case 2: With dynamic_input_builder ---
    # The builder transforms (state, previous_result) into a custom input.
    builder_calls = []

    def dynamic_input_builder(state, previous_result):
        builder_calls.append((dict(state), previous_result))
        return f"built({previous_result})"

    initial_inf2, initial_received2 = _make_recording_inferencer(
        "initial2", transform_fn=lambda inp, idx: f"{inp}->step0"
    )
    followup_inf2, followup_received2 = _make_recording_inferencer(
        "followup2", transform_fn=lambda inp, idx: f"{inp}->step{idx + 1}"
    )

    lwi2 = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=initial_inf2,
        default_followup_inferencer=followup_inf2,
        max_dynamic_steps=step_count,
        end_condition=None,
        dynamic_input_builder=dynamic_input_builder,
    )

    state2 = lwi2.infer(initial_input)

    # Step 0 still receives the original inference_input (builder not used for step 0)
    assert initial_received2[0] == initial_input, (
        f"With builder: Step 0 should still receive original input '{initial_input}', "
        f"got '{initial_received2[0]}'"
    )

    # The builder should have been called for steps 1..N-1
    assert len(builder_calls) == step_count - 1, (
        f"dynamic_input_builder should be called {step_count - 1} times, "
        f"got {len(builder_calls)}"
    )

    # Each followup step should receive the builder's output
    for k in range(1, step_count):
        expected_builder_input = f"built({state2['dynamic_step_results'][k - 1]})"
        assert followup_received2[k - 1] == expected_builder_input, (
            f"With builder: Step {k} should receive builder output "
            f"'{expected_builder_input}', got '{followup_received2[k - 1]}'"
        )

    # Verify builder received the correct previous_result each time
    for k in range(step_count - 1):
        _, prev_result = builder_calls[k]
        assert prev_result == state2["dynamic_step_results"][k], (
            f"Builder call {k} should receive previous result "
            f"'{state2['dynamic_step_results'][k]}', got '{prev_result}'"
        )

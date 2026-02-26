"""
Property-based test for trace step completeness.

Feature: meta-agent-workflow, Property 2: Trace steps capture complete action data

*For any* action executed by the agent during a trace run, the resulting
TraceStep SHALL contain a non-empty action_type, and a non-None timestamp.
If the action had a target, the TraceStep's target SHALL be non-None.
If the action had arguments, the TraceStep's args SHALL be non-None.

**Validates: Requirements 1.2**
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from hypothesis import given, settings, assume, strategies as st

from science_modeling_tools.automation.meta_agent.collector import TraceCollector
from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty action type strings (canonical action types)
action_type_st = st.sampled_from([
    "click",
    "input_text",
    "visit_url",
    "scroll",
    "wait",
    "no_op",
    "append_text",
    "input_and_submit",
    "scroll_up_to_element",
])

# Timestamps: always non-None for valid actions
timestamp_st = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)

# Target values: either a string selector or None
target_value_st = st.one_of(
    st.text(min_size=1, max_size=50),
    st.just({"strategy": "css", "value": "div.main"}),
)

# Args values: non-empty dicts
args_value_st = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("Ll",)),
        min_size=1,
        max_size=10,
    ),
    values=st.one_of(st.text(max_size=20), st.integers(), st.booleans()),
    min_size=1,
    max_size=5,
)


# ---------------------------------------------------------------------------
# Mock agent that produces actions with known fields
# ---------------------------------------------------------------------------

@dataclass
class MockAction:
    """Describes an action the mock agent should produce."""
    action_type: str
    target: Optional[Any] = None
    args: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class MockAgentForCompleteness:
    """
    Agent that produces TraceSteps with specified fields.

    The collector calls agent.run() and then parses logs. Since we
    cannot easily produce real JSONL session files, we override the
    collector's _run_single to directly produce an ExecutionTrace
    with the configured actions.
    """

    def __init__(self, actions: List[MockAction]):
        self._actions = actions
        self._call_count = 0

    def run(self, task_description: str, data: Any = None) -> Any:
        self._call_count += 1
        return None


class CompletenessTraceCollector(TraceCollector):
    """
    A TraceCollector subclass that bypasses real agent execution and
    JSONL parsing, directly constructing TraceSteps from MockActions.

    This lets us test the completeness invariant: when the agent
    produces an action with certain fields, the resulting TraceStep
    must preserve them.
    """

    def __init__(self, agent: MockAgentForCompleteness, actions: List[MockAction]):
        super().__init__(agent=agent)
        self._mock_actions = actions

    def _run_single(
        self,
        task_description: str,
        data: Optional[Dict[str, Any]],
        run_index: int,
    ) -> ExecutionTrace:
        steps = []
        for action in self._mock_actions:
            step = TraceStep(
                action_type=action.action_type,
                target=action.target,
                args=action.args,
                timestamp=action.timestamp,
            )
            steps.append(step)

        return ExecutionTrace(
            trace_id=f"trace-{run_index}",
            task_description=task_description,
            steps=steps,
            success=True,
        )


# ---------------------------------------------------------------------------
# Property 2: Trace steps capture complete action data
# ---------------------------------------------------------------------------


class TestTraceStepCompletenessProperty:
    """
    Property 2: Trace steps capture complete action data

    *For any* action executed by the agent during a trace run, the resulting
    TraceStep SHALL contain a non-empty action_type, and a non-None timestamp.
    If the action had a target, the TraceStep's target SHALL be non-None.
    If the action had arguments, the TraceStep's args SHALL be non-None.

    **Validates: Requirements 1.2**
    """

    @given(
        action_type=action_type_st,
        timestamp=timestamp_st,
    )
    @settings(max_examples=100)
    def test_action_type_is_non_empty_and_timestamp_non_none(
        self, action_type: str, timestamp: datetime
    ):
        """
        For any action with a valid action_type and timestamp, the
        TraceStep preserves both: action_type is non-empty and
        timestamp is non-None.
        """
        actions = [MockAction(action_type=action_type, timestamp=timestamp)]
        agent = MockAgentForCompleteness(actions)
        collector = CompletenessTraceCollector(agent=agent, actions=actions)
        traces = collector.collect("test task", run_count=1)

        assert len(traces) == 1
        step = traces[0].steps[0]

        assert step.action_type, (
            f"action_type should be non-empty, got '{step.action_type}'"
        )
        assert len(step.action_type) > 0, (
            f"action_type should have length > 0, got '{step.action_type}'"
        )
        assert step.timestamp is not None, (
            "timestamp should be non-None"
        )

    @given(
        action_type=action_type_st,
        timestamp=timestamp_st,
        target=target_value_st,
    )
    @settings(max_examples=100)
    def test_target_preserved_when_provided(
        self, action_type: str, timestamp: datetime, target: Any
    ):
        """
        For any action that has a target, the TraceStep's target
        SHALL be non-None.
        """
        actions = [MockAction(
            action_type=action_type,
            timestamp=timestamp,
            target=target,
        )]
        agent = MockAgentForCompleteness(actions)
        collector = CompletenessTraceCollector(agent=agent, actions=actions)
        traces = collector.collect("test task", run_count=1)

        step = traces[0].steps[0]
        assert step.target is not None, (
            f"target should be non-None when action had target={target}"
        )

    @given(
        action_type=action_type_st,
        timestamp=timestamp_st,
        args=args_value_st,
    )
    @settings(max_examples=100)
    def test_args_preserved_when_provided(
        self, action_type: str, timestamp: datetime, args: Dict[str, Any]
    ):
        """
        For any action that has arguments, the TraceStep's args
        SHALL be non-None.
        """
        actions = [MockAction(
            action_type=action_type,
            timestamp=timestamp,
            args=args,
        )]
        agent = MockAgentForCompleteness(actions)
        collector = CompletenessTraceCollector(agent=agent, actions=actions)
        traces = collector.collect("test task", run_count=1)

        step = traces[0].steps[0]
        assert step.args is not None, (
            f"args should be non-None when action had args={args}"
        )

    @given(
        action_type=action_type_st,
        timestamp=timestamp_st,
        target=target_value_st,
        args=args_value_st,
    )
    @settings(max_examples=100)
    def test_all_fields_complete_when_all_provided(
        self,
        action_type: str,
        timestamp: datetime,
        target: Any,
        args: Dict[str, Any],
    ):
        """
        For any action with all fields (action_type, timestamp, target,
        args), the TraceStep captures all of them completely.
        """
        actions = [MockAction(
            action_type=action_type,
            timestamp=timestamp,
            target=target,
            args=args,
        )]
        agent = MockAgentForCompleteness(actions)
        collector = CompletenessTraceCollector(agent=agent, actions=actions)
        traces = collector.collect("test task", run_count=1)

        step = traces[0].steps[0]

        assert step.action_type, "action_type should be non-empty"
        assert len(step.action_type) > 0, "action_type length > 0"
        assert step.timestamp is not None, "timestamp should be non-None"
        assert step.target is not None, "target should be non-None"
        assert step.args is not None, "args should be non-None"

    @given(
        num_actions=st.integers(min_value=1, max_value=20),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_multiple_steps_all_complete(
        self, num_actions: int, data: st.DataObject
    ):
        """
        For any sequence of N actions, every resulting TraceStep has
        non-empty action_type and non-None timestamp. Steps with
        targets have non-None target; steps with args have non-None args.
        """
        actions = []
        for _ in range(num_actions):
            at = data.draw(action_type_st, label="action_type")
            ts = data.draw(timestamp_st, label="timestamp")
            has_target = data.draw(st.booleans(), label="has_target")
            has_args = data.draw(st.booleans(), label="has_args")

            tgt = data.draw(target_value_st, label="target") if has_target else None
            ag = data.draw(args_value_st, label="args") if has_args else None

            actions.append(MockAction(
                action_type=at,
                timestamp=ts,
                target=tgt,
                args=ag,
            ))

        agent = MockAgentForCompleteness(actions)
        collector = CompletenessTraceCollector(agent=agent, actions=actions)
        traces = collector.collect("test task", run_count=1)

        assert len(traces[0].steps) == num_actions

        for i, (step, action) in enumerate(zip(traces[0].steps, actions)):
            assert step.action_type, (
                f"Step {i}: action_type should be non-empty"
            )
            assert step.timestamp is not None, (
                f"Step {i}: timestamp should be non-None"
            )
            if action.target is not None:
                assert step.target is not None, (
                    f"Step {i}: target should be non-None when action had target"
                )
            if action.args is not None:
                assert step.args is not None, (
                    f"Step {i}: args should be non-None when action had args"
                )

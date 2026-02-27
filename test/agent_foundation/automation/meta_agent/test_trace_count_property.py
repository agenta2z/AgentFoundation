"""
Property-based test for trace collection count.

Feature: meta-agent-workflow, Property 1: Trace collection produces correct count

*For any* run count N >= 1 and any agent instance, the TraceCollector SHALL
return exactly N ExecutionTrace objects, including traces from failed runs
(with success=False).

**Validates: Requirements 1.1, 1.4**
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from hypothesis import given, settings, strategies as st

from agent_foundation.automation.meta_agent.collector import TraceCollector
from agent_foundation.automation.meta_agent.models import ExecutionTrace


# ---------------------------------------------------------------------------
# Mock agent
# ---------------------------------------------------------------------------

@dataclass
class MockAgentResult:
    """Simulates an agent run result."""
    session_dir: Optional[str] = None


class MockAgent:
    """Agent that can be configured to fail on specific run indices."""

    def __init__(self, fail_on_indices: Optional[Set[int]] = None):
        self._fail_on_indices = fail_on_indices or set()
        self._call_count = 0

    def run(self, task_description: str, data: Any = None) -> MockAgentResult:
        idx = self._call_count
        self._call_count += 1
        if idx in self._fail_on_indices:
            raise RuntimeError(f"Agent failed on run {idx}")
        return MockAgentResult(session_dir=None)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Run count: N >= 1, capped to keep tests fast
run_count_st = st.integers(min_value=1, max_value=50)

# Failure indices: a subset of [0, N) to simulate partial/full failures
def fail_indices_st(n: int) -> st.SearchStrategy[Set[int]]:
    """Generate a random subset of run indices that should fail."""
    if n == 0:
        return st.just(set())
    return st.frozensets(st.integers(min_value=0, max_value=n - 1)).map(set)


# ---------------------------------------------------------------------------
# Property 1: Trace collection produces correct count
# ---------------------------------------------------------------------------


class TestTraceCountProperty:
    """
    Property 1: Trace collection produces correct count

    *For any* run count N >= 1 and any agent instance, the TraceCollector
    SHALL return exactly N ExecutionTrace objects, including traces from
    failed runs (with success=False).

    **Validates: Requirements 1.1, 1.4**
    """

    @given(n=run_count_st)
    @settings(max_examples=100)
    def test_collector_returns_exactly_n_traces(self, n: int):
        """
        For any N >= 1 with a fully-succeeding agent, the collector
        returns exactly N traces.
        """
        agent = MockAgent()
        collector = TraceCollector(agent=agent)
        traces = collector.collect("test task", run_count=n)

        assert len(traces) == n, (
            f"Expected {n} traces, got {len(traces)}"
        )
        for trace in traces:
            assert isinstance(trace, ExecutionTrace)

    @given(data=st.data())
    @settings(max_examples=100)
    def test_collector_returns_n_traces_with_failures(self, data):
        """
        For any N >= 1 and any subset of failing runs, the collector
        still returns exactly N traces. Failed runs produce traces
        with success=False.
        """
        n = data.draw(run_count_st, label="run_count")
        fail_on = data.draw(fail_indices_st(n), label="fail_indices")

        agent = MockAgent(fail_on_indices=fail_on)
        collector = TraceCollector(agent=agent)
        traces = collector.collect("test task", run_count=n)

        assert len(traces) == n, (
            f"Expected {n} traces, got {len(traces)} "
            f"(fail_on={fail_on})"
        )

        # Failed runs must have success=False
        for i, trace in enumerate(traces):
            assert isinstance(trace, ExecutionTrace)
            if i in fail_on:
                assert trace.success is False, (
                    f"Trace at index {i} should have success=False "
                    f"(was in fail_on={fail_on})"
                )

    @given(n=run_count_st)
    @settings(max_examples=100)
    def test_all_failing_runs_still_return_n_traces(self, n: int):
        """
        Even when every single run fails, the collector returns
        exactly N traces, all with success=False.
        """
        all_fail = set(range(n))
        agent = MockAgent(fail_on_indices=all_fail)
        collector = TraceCollector(agent=agent)
        traces = collector.collect("test task", run_count=n)

        assert len(traces) == n, (
            f"Expected {n} traces when all fail, got {len(traces)}"
        )
        for trace in traces:
            assert trace.success is False

    @given(n=run_count_st)
    @settings(max_examples=100)
    def test_each_trace_has_unique_id(self, n: int):
        """
        All N returned traces have distinct trace_id values.
        """
        agent = MockAgent()
        collector = TraceCollector(agent=agent)
        traces = collector.collect("test task", run_count=n)

        ids = [t.trace_id for t in traces]
        assert len(set(ids)) == n, (
            f"Expected {n} unique trace IDs, got {len(set(ids))}"
        )

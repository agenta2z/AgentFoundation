"""
Property-based tests for ActionGraph loop() execution semantics.

**Feature: action-graph-loop-constructs**

Tests verify that loop() correctly executes with proper iteration counts
and respects max_loop safety limits.
"""

import pytest
from hypothesis import given, strategies as st, settings

import sys
from pathlib import Path

# Add resolve_path for imports - just importing it sets up paths
sys.path.insert(0, str(Path(__file__).parent))
import resolve_path  # noqa: F401

from agent_foundation.automation.schema.action_graph import ActionGraph
from agent_foundation.automation.schema.action_metadata import ActionMetadataRegistry
from agent_foundation.automation.schema.common import ExecutionResult, ExecutionRuntime


# region Test Fixtures

def create_test_graph():
    """Create a minimal ActionGraph for testing."""
    def mock_executor(action_type, target, args=None, **kwargs):
        return {"action": action_type, "target": target}
    
    registry = ActionMetadataRegistry()
    return ActionGraph(
        action_executor=mock_executor,
        action_metadata=registry,
    )


class Counter:
    """Simple counter for tracking iterations."""
    def __init__(self, initial=0):
        self.value = initial
    
    def increment(self, amount=1):
        self.value += amount
        return self.value


# endregion


# region Property 2: Loop execution semantics

class TestLoopExecutionSemantics:
    """
    **Feature: action-graph-loop-constructs, Property 2: Loop execution semantics**
    
    For any loop node with a condition that becomes False after N iterations
    (where N < max_loop), the loop SHALL execute the advance callback exactly
    N times (if provided) and then continue to the next node.
    """
    
    @settings(max_examples=50)
    @given(target_iterations=st.integers(min_value=1, max_value=20))
    def test_loop_executes_correct_iterations(self, target_iterations):
        """Loop executes exactly N times when condition becomes False after N iterations."""
        graph = create_test_graph()
        counter = Counter()
        
        def condition(result, **kwargs):
            return counter.value < target_iterations
        
        def advance(result, **kwargs):
            counter.increment()
            return result
        
        graph.loop(
            condition=condition,
            max_loop=target_iterations + 100,  # Well above target
            advance=advance,
        )
        
        graph.execute()
        
        assert counter.value == target_iterations
    
    def test_loop_exits_immediately_when_condition_false(self):
        """Loop exits without executing advance when condition is initially False."""
        graph = create_test_graph()
        counter = Counter()
        
        def always_false(result, **kwargs):
            return False
        
        def advance(result, **kwargs):
            counter.increment()
            return result
        
        graph.loop(
            condition=always_false,
            advance=advance,
        )
        
        graph.execute()
        
        # Advance should never be called since condition is False from start
        assert counter.value == 0
    
    def test_loop_without_advance_still_checks_condition(self):
        """Loop without advance still checks condition each iteration."""
        graph = create_test_graph()
        condition_checks = Counter()
        
        def counting_condition(result, **kwargs):
            condition_checks.increment()
            return condition_checks.value < 5
        
        graph.loop(
            condition=counting_condition,
            max_loop=100,
            # No advance - just checking condition
        )
        
        graph.execute()
        
        # Condition should be checked 5 times (4 True + 1 False)
        assert condition_checks.value == 5


# endregion


# region Property 3: max_loop safety limit enforcement

class TestMaxLoopSafetyLimit:
    """
    **Feature: action-graph-loop-constructs, Property 3: Loop safety limit enforcement**
    
    For any loop node with an always-true condition, the loop SHALL stop
    after exactly max_loop executions.
    """
    
    @settings(max_examples=20)
    @given(max_loop=st.integers(min_value=1, max_value=50))
    def test_max_loop_limits_iterations(self, max_loop):
        """Loop stops at max_loop even with always-true condition."""
        graph = create_test_graph()
        counter = Counter()
        
        def always_true(result, **kwargs):
            return True
        
        def advance(result, **kwargs):
            counter.increment()
            return result
        
        graph.loop(
            condition=always_true,
            max_loop=max_loop,
            advance=advance,
        )
        
        graph.execute()
        
        # Should stop at max_loop
        assert counter.value == max_loop
    
    def test_max_loop_default_is_1000(self):
        """Default max_loop is 1000."""
        graph = create_test_graph()
        
        graph.loop(condition=lambda r, **kw: False)
        
        loop_node = graph._nodes[-1]
        assert loop_node.max_repeat == 1000


# endregion


# region Test: Loop preserves execution result

class TestLoopResultPreservation:
    """Test that loop correctly passes results through."""
    
    def test_loop_returns_fallback_when_condition_false(self):
        """Loop returns fallback result when condition is False."""
        graph = create_test_graph()
        
        graph.loop(
            condition=lambda r, **kw: False,
        )
        
        result = graph.execute()
        
        # Should succeed (loop just exits)
        assert result.success is True
    
    def test_loop_with_advance_modifying_state(self):
        """Advance can modify external state that condition checks."""
        graph = create_test_graph()
        state = {"value": 0}
        
        def condition(result, **kwargs):
            return state["value"] < 10
        
        def advance(result, **kwargs):
            state["value"] += 2
            return result
        
        graph.loop(
            condition=condition,
            advance=advance,
        )
        
        graph.execute()
        
        # Should have incremented 5 times (0, 2, 4, 6, 8, then 10 fails condition)
        assert state["value"] == 10


# endregion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

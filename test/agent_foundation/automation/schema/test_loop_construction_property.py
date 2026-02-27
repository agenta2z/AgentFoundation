"""
Property-based tests for ActionGraph loop() construction.

**Feature: action-graph-loop-constructs**

Tests verify that loop() correctly creates ActionSequenceNode with
repeat_condition configuration for while-loop behavior.
"""

import pytest
from hypothesis import given, strategies as st, settings

import sys
from pathlib import Path

# Add resolve_path for imports - just importing it sets up paths
sys.path.insert(0, str(Path(__file__).parent))
import resolve_path  # noqa: F401

from agent_foundation.automation.schema.action_graph import ActionGraph, ActionSequenceNode
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


def always_false_condition(result, **kwargs):
    """Condition that always returns False (loop exits immediately)."""
    return False


def always_true_condition(result, **kwargs):
    """Condition that always returns True (infinite loop without max_loop)."""
    return True


# endregion


# region Property 1: loop() adds exactly one node to graph

class TestLoopAddsOneNode:
    """
    **Feature: action-graph-loop-constructs, Property 1: Loop graph construction preserves structure**
    
    For any ActionGraph, calling loop() SHALL add exactly one ActionSequenceNode
    to the graph's _nodes list.
    """
    
    @settings(max_examples=100)
    @given(max_loop=st.integers(min_value=1, max_value=10000))
    def test_loop_adds_exactly_one_node(self, max_loop):
        """loop() adds exactly one node regardless of max_loop value."""
        graph = create_test_graph()
        initial_node_count = len(graph._nodes)
        
        graph.loop(
            condition=always_false_condition,
            max_loop=max_loop,
        )
        
        assert len(graph._nodes) == initial_node_count + 1
    
    def test_loop_node_is_action_sequence_node(self):
        """The added node is an ActionSequenceNode."""
        graph = create_test_graph()
        
        graph.loop(condition=always_false_condition)
        
        loop_node = graph._nodes[-1]
        assert isinstance(loop_node, ActionSequenceNode)
    
    def test_loop_node_has_repeat_condition(self):
        """The loop node has repeat_condition that wraps the provided condition."""
        graph = create_test_graph()
        
        graph.loop(condition=always_false_condition)
        
        loop_node = graph._nodes[-1]
        # repeat_condition is a wrapped version that enforces max_loop limit
        # It should behave like the original condition when under the limit
        assert loop_node.repeat_condition is not None
        mock_result = ExecutionResult(success=True, context=ExecutionRuntime())
        # The wrapped condition should return False (like always_false_condition)
        assert loop_node.repeat_condition(mock_result) is False
    
    def test_loop_node_has_max_repeat(self):
        """The loop node has max_repeat set to ensure at least max_loop iterations possible."""
        graph = create_test_graph()
        
        graph.loop(condition=always_false_condition, max_loop=42)
        
        loop_node = graph._nodes[-1]
        # max_repeat is set to max(2, max_loop) to ensure we enter the while loop
        # The actual max_loop limit is enforced by the wrapped condition
        assert loop_node.max_repeat >= 42
    
    def test_loop_node_has_output_validator_false(self):
        """The loop node has output_validator that always returns False."""
        graph = create_test_graph()
        
        graph.loop(condition=always_false_condition)
        
        loop_node = graph._nodes[-1]
        # output_validator should return False to force retry
        assert loop_node.output_validator is not None
        assert loop_node.output_validator(None) is False


# endregion


# region Test: Loop works without advance callback

class TestLoopWithoutAdvance:
    """Test that loop works correctly without an advance callback."""
    
    def test_loop_without_advance_has_noop_value(self):
        """Loop without advance has a no-op value function."""
        graph = create_test_graph()
        
        graph.loop(condition=always_false_condition)
        
        loop_node = graph._nodes[-1]
        # The value should be a callable that returns its input
        mock_result = ExecutionResult(
            success=True,
            context=ExecutionRuntime(),
        )
        result = loop_node.value(mock_result)
        assert result is mock_result


# endregion


# region Test: Loop works with advance callback

class TestLoopWithAdvance:
    """Test that loop works correctly with an advance callback."""
    
    def test_loop_with_advance_sets_value(self):
        """Loop with advance sets a wrapped advance as the node's value."""
        graph = create_test_graph()
        
        call_count = [0]
        
        def my_advance(result, **kwargs):
            call_count[0] += 1
            return result
        
        graph.loop(
            condition=always_false_condition,
            advance=my_advance,
        )
        
        loop_node = graph._nodes[-1]
        # The value is a wrapped version that tracks iterations
        # Verify it calls the original advance
        mock_result = ExecutionResult(success=True, context=ExecutionRuntime())
        loop_node.value(mock_result)
        assert call_count[0] == 1  # Original advance was called
    
    def test_advance_is_called_with_result(self):
        """Advance callback receives the previous result."""
        graph = create_test_graph()
        received_results = []
        
        def tracking_advance(result, **kwargs):
            received_results.append(result)
            return result
        
        graph.loop(
            condition=always_false_condition,
            advance=tracking_advance,
        )
        
        loop_node = graph._nodes[-1]
        mock_result = ExecutionResult(
            success=True,
            context=ExecutionRuntime(),
        )
        loop_node.value(mock_result)
        
        assert len(received_results) == 1
        assert received_results[0] is mock_result


# endregion


# region Test: Method chaining returns self

class TestMethodChaining:
    """Test that loop() returns self for method chaining."""
    
    def test_loop_returns_self(self):
        """loop() returns the ActionGraph instance for chaining."""
        graph = create_test_graph()
        
        result = graph.loop(condition=always_false_condition)
        
        assert result is graph
    
    def test_loop_can_be_chained(self):
        """Multiple loop() calls can be chained."""
        graph = create_test_graph()
        
        result = (
            graph
            .loop(condition=always_false_condition, max_loop=10)
            .loop(condition=always_false_condition, max_loop=20)
        )
        
        assert result is graph
        # Should have root + 2 loop nodes
        assert len(graph._nodes) == 3


# endregion


# region Test: Loop node is connected to current node

class TestLoopNodeConnection:
    """Test that loop node is properly connected in the graph."""
    
    def test_loop_node_connected_to_previous(self):
        """Loop node is added as next of the current node."""
        graph = create_test_graph()
        root_node = graph._current_node
        
        graph.loop(condition=always_false_condition)
        
        loop_node = graph._nodes[-1]
        assert loop_node in root_node.next
    
    def test_current_node_updated_to_loop_node(self):
        """After loop(), _current_node is the loop node."""
        graph = create_test_graph()
        
        graph.loop(condition=always_false_condition)
        
        loop_node = graph._nodes[-1]
        assert graph._current_node is loop_node


# endregion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

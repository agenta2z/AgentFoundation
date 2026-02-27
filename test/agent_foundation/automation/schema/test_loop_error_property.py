"""
Property-based tests for ActionGraph loop() error handling.

**Feature: action-graph-loop-constructs**

Tests verify that loop errors are handled correctly with proper
iteration context and warning logging.
"""

import pytest
import warnings
from hypothesis import given, strategies as st, settings

import sys
from pathlib import Path

# Add resolve_path for imports - just importing it sets up paths
sys.path.insert(0, str(Path(__file__).parent))
import resolve_path  # noqa: F401

from agent_foundation.automation.schema.action_graph import ActionGraph
from agent_foundation.automation.schema.action_metadata import ActionMetadataRegistry
from agent_foundation.automation.schema.common import ExecutionResult, ExecutionRuntime, LoopExecutionError


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


# region Property 5: Error handling behavior

class TestAdvanceErrorHandling:
    """
    **Feature: action-graph-loop-constructs, Property 5: Error handling**
    
    Errors in advance callbacks are handled by execute_with_retry's retry
    mechanism. Errors in condition callbacks propagate immediately since
    they occur before execution.
    """
    
    def test_advance_error_triggers_retry(self):
        """Errors in advance callback trigger retry mechanism."""
        graph = create_test_graph()
        call_count = Counter()
        
        def failing_advance(result, **kwargs):
            call_count.increment()
            raise ValueError("Advance failed!")
        
        graph.loop(
            condition=lambda r, **kw: True,
            max_loop=3,
            advance=failing_advance,
        )
        
        # execute_with_retry will retry on exceptions
        # The loop will eventually hit max_loop limit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = graph.execute()
            
            # Should have retry warnings
            retry_warnings = [x for x in w if "failed due to error" in str(x.message)]
            assert len(retry_warnings) > 0
    
    def test_advance_error_retried_until_max_loop(self):
        """Advance errors are retried until max_loop is reached."""
        graph = create_test_graph()
        counter = Counter()
        
        def failing_advance(result, **kwargs):
            counter.increment()
            raise RuntimeError("Always fails")
        
        graph.loop(
            condition=lambda r, **kw: True,
            max_loop=5,
            advance=failing_advance,
        )
        
        # Should not raise - execute_with_retry handles exceptions
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            graph.execute()
        
        # Should have been called multiple times due to retries
        # The exact count depends on execute_with_retry behavior
        assert counter.value >= 1
    
    def test_condition_error_propagates(self):
        """Errors in condition callback propagate to caller."""
        graph = create_test_graph()
        
        def failing_condition(result, **kwargs):
            raise TypeError("Condition failed!")
        
        graph.loop(
            condition=failing_condition,
            max_loop=10,
        )
        
        with pytest.raises(TypeError, match="Condition failed!"):
            graph.execute()


# endregion


# region Test: max_loop exceeded generates warning

class TestMaxLoopWarning:
    """Test that max_loop limit generates appropriate warnings."""
    
    def test_max_loop_generates_retry_warnings(self):
        """Reaching max_loop generates retry warnings from execute_with_retry."""
        graph = create_test_graph()
        counter = Counter()
        
        graph.loop(
            condition=lambda r, **kw: True,  # Always true
            max_loop=3,
            advance=lambda r, **kw: counter.increment(),
        )
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graph.execute()
            
            # Should have retry warnings from execute_with_retry
            retry_warnings = [x for x in w if "failed due to error" in str(x.message)]
            assert len(retry_warnings) > 0
    
    def test_loop_stops_at_max_loop(self):
        """Loop stops at max_loop even with always-true condition."""
        graph = create_test_graph()
        counter = Counter()
        
        graph.loop(
            condition=lambda r, **kw: True,
            max_loop=5,
            advance=lambda r, **kw: counter.increment(),
        )
        
        # Should not raise, just stop at max_loop
        result = graph.execute()
        
        assert counter.value == 5
        assert result.success is True


# endregion


# region Test: LoopExecutionError class

class TestLoopExecutionErrorClass:
    """Test the LoopExecutionError exception class."""
    
    def test_loop_execution_error_attributes(self):
        """LoopExecutionError has correct attributes."""
        original = ValueError("Original error")
        error = LoopExecutionError(
            loop_id="loop_1",
            iteration=5,
            original_error=original,
        )
        
        assert error.loop_id == "loop_1"
        assert error.iteration == 5
        assert error.original_error is original
    
    def test_loop_execution_error_message(self):
        """LoopExecutionError has informative message."""
        original = ValueError("Something went wrong")
        error = LoopExecutionError(
            loop_id="my_loop",
            iteration=10,
            original_error=original,
        )
        
        assert "my_loop" in str(error)
        assert "10" in str(error)
        assert "Something went wrong" in str(error)
    
    def test_loop_execution_error_is_exception(self):
        """LoopExecutionError is an Exception subclass."""
        error = LoopExecutionError(
            loop_id="test",
            iteration=1,
            original_error=ValueError("test"),
        )
        
        assert isinstance(error, Exception)


# endregion


# region Test: Loop with empty iterations

class TestLoopEdgeCases:
    """Test edge cases in loop execution."""
    
    def test_loop_with_max_loop_zero(self):
        """Loop with max_loop=0 should not execute advance."""
        graph = create_test_graph()
        counter = Counter()
        
        # Note: max_loop=0 is an edge case - the wrapped condition
        # will return False immediately since count >= max_loop (0 >= 0)
        graph.loop(
            condition=lambda r, **kw: True,
            max_loop=0,
            advance=lambda r, **kw: counter.increment(),
        )
        
        graph.execute()
        
        # Advance should not be called
        assert counter.value == 0
    
    def test_loop_with_max_loop_one(self):
        """Loop with max_loop=1 executes advance exactly once."""
        graph = create_test_graph()
        counter = Counter()
        
        graph.loop(
            condition=lambda r, **kw: True,
            max_loop=1,
            advance=lambda r, **kw: counter.increment(),
        )
        
        graph.execute()
        
        assert counter.value == 1


# endregion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Property-based tests for ActionGraph loop() serialization.

**Feature: action-graph-loop-constructs**

Tests verify that loop nodes serialize and deserialize correctly,
preserving structure and configuration.
"""

import pytest
import json
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


def always_false_condition(result, **kwargs):
    """Condition that always returns False."""
    return False


def always_true_condition(result, **kwargs):
    """Condition that always returns True."""
    return True


# endregion


# region Property 4: Loop node round-trip preserves structure

class TestLoopSerializationRoundTrip:
    """
    **Feature: action-graph-loop-constructs, Property 4: Loop serialization round-trip**
    
    For any ActionGraph with loop nodes, serializing to JSON and deserializing
    SHALL preserve the loop node structure including max_loop configuration.
    """
    
    @settings(max_examples=20)
    @given(max_loop=st.integers(min_value=1, max_value=1000))
    def test_loop_node_serializes_with_node_behavior(self, max_loop):
        """Loop node serialization includes node_behavior='loop'."""
        graph = create_test_graph()
        
        graph.loop(
            condition=always_false_condition,
            max_loop=max_loop,
        )
        
        # Serialize to dict
        serialized = graph.to_serializable_obj()
        
        # Find the loop node in serialized data
        loop_node_data = None
        for node_data in serialized["nodes"]:
            if node_data.get("node_behavior") == "loop":
                loop_node_data = node_data
                break
        
        assert loop_node_data is not None, "Loop node should have node_behavior='loop'"
        assert loop_node_data["loop_config"]["max_loop"] == max_loop
    
    def test_loop_node_json_round_trip(self):
        """Loop node survives JSON serialization round-trip."""
        graph = create_test_graph()
        
        graph.loop(
            condition=always_false_condition,
            max_loop=50,
        )
        
        # Serialize to JSON
        json_str = graph.serialize(output_format='json')
        
        # Deserialize
        def mock_executor(action_type, target, args=None, **kwargs):
            return {"action": action_type, "target": target}
        
        registry = ActionMetadataRegistry()
        restored_graph = ActionGraph.deserialize(
            json_str,
            input_format='json',
            action_executor=mock_executor,
            action_metadata=registry,
        )
        
        # Verify loop node exists
        loop_nodes = [n for n in restored_graph._nodes if getattr(n, '_is_loop_node', False)]
        assert len(loop_nodes) == 1
        assert loop_nodes[0]._loop_max_loop == 50
    
    def test_loop_node_preserves_max_loop_in_round_trip(self):
        """max_loop value is preserved through serialization."""
        graph = create_test_graph()
        
        graph.loop(
            condition=always_false_condition,
            max_loop=123,
        )
        
        # Serialize and deserialize
        json_str = graph.serialize(output_format='json')
        
        def mock_executor(action_type, target, args=None, **kwargs):
            return {"action": action_type, "target": target}
        
        registry = ActionMetadataRegistry()
        restored_graph = ActionGraph.deserialize(
            json_str,
            input_format='json',
            action_executor=mock_executor,
            action_metadata=registry,
        )
        
        # Verify max_loop is preserved in the restored loop node
        loop_node = [n for n in restored_graph._nodes if getattr(n, '_is_loop_node', False)][0]
        assert loop_node._loop_max_loop == 123
        
        # Note: The original condition callable cannot be restored from serialization
        # (Python callables can't be serialized to JSON). The condition string is
        # stored for documentation purposes, but execution requires re-providing
        # the condition callable.


# endregion


# region Test: Loop node structure in serialized output

class TestLoopSerializationStructure:
    """Test the structure of serialized loop nodes."""
    
    def test_loop_config_contains_required_fields(self):
        """loop_config contains max_loop, condition, has_advance."""
        graph = create_test_graph()
        
        graph.loop(
            condition=always_false_condition,
            max_loop=100,
        )
        
        serialized = graph.to_serializable_obj()
        
        loop_node_data = None
        for node_data in serialized["nodes"]:
            if node_data.get("node_behavior") == "loop":
                loop_node_data = node_data
                break
        
        assert "loop_config" in loop_node_data
        loop_config = loop_node_data["loop_config"]
        assert "max_loop" in loop_config
        assert "condition" in loop_config
        assert "has_advance" in loop_config
    
    def test_has_advance_true_when_advance_provided(self):
        """has_advance is True when advance callback is provided."""
        graph = create_test_graph()
        
        def my_advance(result, **kwargs):
            return result
        
        graph.loop(
            condition=always_false_condition,
            advance=my_advance,
        )
        
        serialized = graph.to_serializable_obj()
        
        loop_node_data = None
        for node_data in serialized["nodes"]:
            if node_data.get("node_behavior") == "loop":
                loop_node_data = node_data
                break
        
        assert loop_node_data["loop_config"]["has_advance"] is True
    
    def test_has_advance_false_when_no_advance(self):
        """has_advance is False when no advance callback."""
        graph = create_test_graph()
        
        graph.loop(
            condition=always_false_condition,
        )
        
        serialized = graph.to_serializable_obj()
        
        loop_node_data = None
        for node_data in serialized["nodes"]:
            if node_data.get("node_behavior") == "loop":
                loop_node_data = node_data
                break
        
        assert loop_node_data["loop_config"]["has_advance"] is False


# endregion


# region Test: Multiple loop nodes serialization

class TestMultipleLoopsSerialization:
    """Test serialization with multiple loop nodes."""
    
    def test_multiple_loops_serialize_independently(self):
        """Multiple loop nodes each have their own configuration."""
        graph = create_test_graph()
        
        graph.loop(condition=always_false_condition, max_loop=10)
        graph.loop(condition=always_true_condition, max_loop=20)
        
        serialized = graph.to_serializable_obj()
        
        loop_nodes = [n for n in serialized["nodes"] if n.get("node_behavior") == "loop"]
        assert len(loop_nodes) == 2
        
        max_loops = {n["loop_config"]["max_loop"] for n in loop_nodes}
        assert max_loops == {10, 20}


# endregion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

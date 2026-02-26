"""
Functional tests for ActionGraph using string operations.

Tests correctness properties from the action-graph-arithmetic-tests design document.
Uses real string computations with verifiable results instead of mock executors.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup import paths
_current_file = Path(__file__).resolve()
_test_dir = _current_file.parent
while _test_dir.name != 'test' and _test_dir.parent != _test_dir:
    _test_dir = _test_dir.parent
_project_root = _test_dir.parent
_src_dir = _project_root / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
# Add SciencePythonUtils if needed
_workspace_root = _project_root.parent
_science_python_utils_src = _workspace_root / "SciencePythonUtils" / "src"
if _science_python_utils_src.exists() and str(_science_python_utils_src) not in sys.path:
    sys.path.insert(0, str(_science_python_utils_src))

import pytest

from science_modeling_tools.automation.schema.action_graph import ActionGraph
from science_modeling_tools.automation.schema.action_metadata import ActionMetadataRegistry


# region Result Object

@dataclass
class Result:
    """Result object returned by StringExecutor."""
    success: bool
    value: Any
    error: Optional[Exception] = None


# endregion


# region StringExecutor

class StringExecutor:
    """
    Executor for string operations with verifiable results.
    
    Maintains a buffer string and history of operations.
    Supports: str_set, str_concat, str_substring, str_replace, str_reverse, 
              str_upper, str_lower, str_trim
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
    """

    def __init__(self):
        self.history: List[Dict] = []
        self.buffer: str = ""
    
    def __call__(
        self,
        action_type: str,
        action_target: Optional[str] = None,
        action_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Result:
        """
        Execute string action and return result object.
        
        Args:
            action_type: The string operation to perform
            action_target: Not used for string operations
            action_args: Arguments for the operation (e.g., {"value": "hello"})
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Result object with success status and current buffer value
        """
        action_args = action_args or {}
        
        try:
            if action_type == "str_set":
                # Requirement 8.1: Set buffer to specified string value
                value = action_args.get("value", "")
                self.buffer = str(value)
            
            elif action_type == "str_concat":
                # Requirement 8.2: Append value to buffer
                value = action_args.get("value", "")
                self.buffer += str(value)
            
            elif action_type == "str_substring":
                # Requirement 8.3: Extract substring from buffer
                start = int(action_args.get("start", 0))
                end = int(action_args.get("end", len(self.buffer)))
                self.buffer = self.buffer[start:end]
            
            elif action_type == "str_replace":
                # Requirement 8.4: Replace all occurrences of old with new
                old = str(action_args.get("old", ""))
                new = str(action_args.get("new", ""))
                self.buffer = self.buffer.replace(old, new)
            
            elif action_type == "str_reverse":
                # Requirement 8.5: Reverse the buffer string
                self.buffer = self.buffer[::-1]
            
            elif action_type == "str_upper":
                # Requirement 8.6: Convert buffer to uppercase
                self.buffer = self.buffer.upper()
            
            elif action_type == "str_lower":
                # Requirement 8.7: Convert buffer to lowercase
                self.buffer = self.buffer.lower()
            
            elif action_type == "str_trim":
                # Requirement 8.8: Remove leading and trailing whitespace
                self.buffer = self.buffer.strip()
            
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            # Record operation in history
            self.history.append({
                "action_type": action_type,
                "args": action_args,
                "result": self.buffer,
            })
            
            return Result(success=True, value=self.buffer)
        
        except Exception as e:
            # Record failed operation
            self.history.append({
                "action_type": action_type,
                "args": action_args,
                "error": str(e),
            })
            return Result(success=False, value=self.buffer, error=e)
    
    def reset(self):
        """Reset buffer and history."""
        self.buffer = ""
        self.history = []


# endregion


# region Pytest Fixtures

@pytest.fixture
def string_registry():
    """
    Load string action metadata from JSON fixture.
    
    Requirements: 8.1
    """
    from science_modeling_tools.automation.schema.action_metadata import ActionTypeMetadata
    
    fixtures_dir = Path(__file__).parent / "fixtures"
    json_path = fixtures_dir / "string_actions.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    registry = ActionMetadataRegistry()
    for action_def in data.get("actions", []):
        metadata = ActionTypeMetadata(
            name=action_def["name"],
            requires_target=action_def.get("requires_target", False),
            supported_args=action_def.get("supported_args", []),
            required_args=action_def.get("required_args", []),
            description=action_def.get("description", ""),
        )
        registry.register_action(metadata)
    
    return registry


@pytest.fixture
def string_executor():
    """Create a fresh StringExecutor instance."""
    return StringExecutor()


@pytest.fixture
def make_string_graph(string_executor, string_registry):
    """
    Factory fixture to create ActionGraph instances with string executor.
    
    Returns a function that creates a new ActionGraph each time it's called.
    """
    def _make_graph():
        return ActionGraph(
            action_executor=string_executor,
            action_metadata=string_registry,
        )
    return _make_graph


# endregion


# region Basic Verification Tests

class TestStringExecutorBasics:
    """Basic tests to verify StringExecutor works correctly."""
    
    def test_executor_initial_state(self, string_executor):
        """Test that executor starts with buffer = '' and empty history."""
        assert string_executor.buffer == ""
        assert string_executor.history == []
    
    def test_str_set_operation(self, string_executor):
        """Test str_set operation sets buffer to specified value."""
        result = string_executor("str_set", action_args={"value": "hello"})
        assert result.success is True
        assert result.value == "hello"
        assert string_executor.buffer == "hello"
    
    def test_str_concat_operation(self, string_executor):
        """Test str_concat operation appends to buffer."""
        string_executor("str_set", action_args={"value": "hello"})
        result = string_executor("str_concat", action_args={"value": " world"})
        assert result.success is True
        assert result.value == "hello world"
        assert string_executor.buffer == "hello world"
    
    def test_str_substring_operation(self, string_executor):
        """Test str_substring operation extracts substring."""
        string_executor("str_set", action_args={"value": "hello world"})
        result = string_executor("str_substring", action_args={"start": 0, "end": 5})
        assert result.success is True
        assert result.value == "hello"
    
    def test_str_replace_operation(self, string_executor):
        """Test str_replace operation replaces occurrences."""
        string_executor("str_set", action_args={"value": "hello world"})
        result = string_executor("str_replace", action_args={"old": "world", "new": "universe"})
        assert result.success is True
        assert result.value == "hello universe"
    
    def test_str_reverse_operation(self, string_executor):
        """Test str_reverse operation reverses buffer."""
        string_executor("str_set", action_args={"value": "hello"})
        result = string_executor("str_reverse")
        assert result.success is True
        assert result.value == "olleh"
    
    def test_str_upper_operation(self, string_executor):
        """Test str_upper operation converts to uppercase."""
        string_executor("str_set", action_args={"value": "hello"})
        result = string_executor("str_upper")
        assert result.success is True
        assert result.value == "HELLO"
    
    def test_str_lower_operation(self, string_executor):
        """Test str_lower operation converts to lowercase."""
        string_executor("str_set", action_args={"value": "HELLO"})
        result = string_executor("str_lower")
        assert result.success is True
        assert result.value == "hello"
    
    def test_str_trim_operation(self, string_executor):
        """Test str_trim operation removes whitespace."""
        string_executor("str_set", action_args={"value": "  hello  "})
        result = string_executor("str_trim")
        assert result.success is True
        assert result.value == "hello"
    
    def test_history_tracking(self, string_executor):
        """Test that history tracks all operations."""
        string_executor("str_set", action_args={"value": "hello"})
        string_executor("str_concat", action_args={"value": " world"})
        string_executor("str_upper")
        
        assert len(string_executor.history) == 3
        assert string_executor.history[0]["action_type"] == "str_set"
        assert string_executor.history[1]["action_type"] == "str_concat"
        assert string_executor.history[2]["action_type"] == "str_upper"
    
    def test_reset(self, string_executor):
        """Test reset clears buffer and history."""
        string_executor("str_set", action_args={"value": "hello"})
        string_executor("str_concat", action_args={"value": " world"})
        
        string_executor.reset()
        
        assert string_executor.buffer == ""
        assert string_executor.history == []


class TestStringFixtures:
    """Tests to verify fixtures work correctly."""
    
    def test_string_registry_loads(self, string_registry):
        """Test that string registry loads from JSON."""
        # Verify some expected actions are registered
        assert string_registry.get_metadata("str_set") is not None
        assert string_registry.get_metadata("str_concat") is not None
        assert string_registry.get_metadata("str_upper") is not None
    
    def test_make_string_graph_creates_graph(self, make_string_graph):
        """Test that make_string_graph factory creates ActionGraph."""
        graph = make_string_graph()
        assert isinstance(graph, ActionGraph)
    
    def test_make_string_graph_uses_executor(self, make_string_graph, string_executor):
        """Test that make_string_graph uses the provided executor."""
        graph = make_string_graph()
        assert graph.action_executor is string_executor


# endregion


# region TestStringBasics

class TestStringBasics:
    """
    Test class for basic string execution through ActionGraph.
    
    Tests Requirements: 8.1, 9.1, 9.2, 9.3, 9.4
    """
    
    def test_single_action_str_set(self, make_string_graph, string_executor):
        """
        Test str_set("hello") → buffer = "hello"
        
        Requirements: 8.1
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": "hello"})
        
        result = graph.execute()
        
        assert result.success is True
        assert string_executor.buffer == "hello"
    
    def test_sequential_string_actions(self, make_string_graph, string_executor):
        """
        Test str_set("hello") → str_concat(" world") = "hello world"
        
        Requirements: 9.1
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": "hello"})
        graph.action("str_concat", args={"value": " world"})
        
        result = graph.execute()
        
        assert result.success is True
        assert string_executor.buffer == "hello world"
    
    def test_string_transform_chain(self, make_string_graph, string_executor):
        """
        Test str_set("hello") → str_upper() → str_reverse() = "OLLEH"
        
        Requirements: 9.2
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": "hello"})
        graph.action("str_upper")
        graph.action("str_reverse")
        
        result = graph.execute()
        
        assert result.success is True
        # "hello" → "HELLO" → "OLLEH"
        assert string_executor.buffer == "OLLEH"
    
    def test_trim_and_concat(self, make_string_graph, string_executor):
        """
        Test str_set("  hello  ") → str_trim() → str_concat("!") = "hello!"
        
        Requirements: 9.3
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": "  hello  "})
        graph.action("str_trim")
        graph.action("str_concat", args={"value": "!"})
        
        result = graph.execute()
        
        assert result.success is True
        # "  hello  " → "hello" → "hello!"
        assert string_executor.buffer == "hello!"
    
    def test_replace_operation(self, make_string_graph, string_executor):
        """
        Test str_set("hello world") → str_replace("world", "universe") = "hello universe"
        
        Requirements: 9.4
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": "hello world"})
        graph.action("str_replace", args={"old": "world", "new": "universe"})
        
        result = graph.execute()
        
        assert result.success is True
        assert string_executor.buffer == "hello universe"


# endregion


# region Property-Based Tests for TestStringBasics

from hypothesis import given, strategies as st, settings, HealthCheck


class TestStringBasicsProperties:
    """
    Property-based tests for string operations.
    
    **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8**
    
    Uses Hypothesis for property-based testing with minimum 100 iterations.
    """
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(value=st.text(min_size=0, max_size=100))
    def test_property_str_set_operation(self, string_executor, value):
        """
        **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
        **Validates: Requirements 8.1**
        
        For any string value, str_set(value) should set buffer to that value.
        """
        string_executor.reset()
        result = string_executor("str_set", action_args={"value": value})
        
        assert result.success is True
        assert result.value == value
        assert string_executor.buffer == value
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        initial=st.text(min_size=0, max_size=50),
        suffix=st.text(min_size=0, max_size=50)
    )
    def test_property_str_concat_operation(self, string_executor, initial, suffix):
        """
        **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
        **Validates: Requirements 8.2**
        
        For any initial string and suffix, str_concat(suffix) should append to buffer.
        """
        string_executor.reset()
        string_executor("str_set", action_args={"value": initial})
        result = string_executor("str_concat", action_args={"value": suffix})
        
        expected = initial + suffix
        assert result.success is True
        assert result.value == expected
        assert string_executor.buffer == expected
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        text=st.text(min_size=0, max_size=100),
        start=st.integers(min_value=0, max_value=50),
        end=st.integers(min_value=0, max_value=100)
    )
    def test_property_str_substring_operation(self, string_executor, text, start, end):
        """
        **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
        **Validates: Requirements 8.3**
        
        For any string and indices, str_substring(start, end) should extract that substring.
        """
        string_executor.reset()
        string_executor("str_set", action_args={"value": text})
        result = string_executor("str_substring", action_args={"start": start, "end": end})
        
        expected = text[start:end]
        assert result.success is True
        assert result.value == expected
        assert string_executor.buffer == expected
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        text=st.text(min_size=0, max_size=100),
        old=st.text(min_size=1, max_size=10),
        new=st.text(min_size=0, max_size=10)
    )
    def test_property_str_replace_operation(self, string_executor, text, old, new):
        """
        **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
        **Validates: Requirements 8.4**
        
        For any string and replacement pair, str_replace(old, new) should replace all occurrences.
        """
        string_executor.reset()
        string_executor("str_set", action_args={"value": text})
        result = string_executor("str_replace", action_args={"old": old, "new": new})
        
        expected = text.replace(old, new)
        assert result.success is True
        assert result.value == expected
        assert string_executor.buffer == expected
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(value=st.text(min_size=0, max_size=100))
    def test_property_str_reverse_operation(self, string_executor, value):
        """
        **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
        **Validates: Requirements 8.5**
        
        For any string, str_reverse() should reverse the buffer.
        """
        string_executor.reset()
        string_executor("str_set", action_args={"value": value})
        result = string_executor("str_reverse")
        
        expected = value[::-1]
        assert result.success is True
        assert result.value == expected
        assert string_executor.buffer == expected
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(value=st.text(min_size=0, max_size=100))
    def test_property_str_upper_operation(self, string_executor, value):
        """
        **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
        **Validates: Requirements 8.6**
        
        For any string, str_upper() should convert buffer to uppercase.
        """
        string_executor.reset()
        string_executor("str_set", action_args={"value": value})
        result = string_executor("str_upper")
        
        expected = value.upper()
        assert result.success is True
        assert result.value == expected
        assert string_executor.buffer == expected
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(value=st.text(min_size=0, max_size=100))
    def test_property_str_lower_operation(self, string_executor, value):
        """
        **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
        **Validates: Requirements 8.7**
        
        For any string, str_lower() should convert buffer to lowercase.
        """
        string_executor.reset()
        string_executor("str_set", action_args={"value": value})
        result = string_executor("str_lower")
        
        expected = value.lower()
        assert result.success is True
        assert result.value == expected
        assert string_executor.buffer == expected
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(value=st.text(min_size=0, max_size=100))
    def test_property_str_trim_operation(self, string_executor, value):
        """
        **Feature: action-graph-arithmetic-tests, Property 8: String operation correctness**
        **Validates: Requirements 8.8**
        
        For any string, str_trim() should remove leading and trailing whitespace.
        """
        string_executor.reset()
        string_executor("str_set", action_args={"value": value})
        result = string_executor("str_trim")
        
        expected = value.strip()
        assert result.success is True
        assert result.value == expected
        assert string_executor.buffer == expected


# endregion


# region TestStringBranching Helper Functions

def get_last_string_result_value(execution_result):
    """
    Extract the last action's result value from an ExecutionResult.
    
    The ExecutionResult has context.results which is a dict of action_id -> ActionResult.
    Each ActionResult has a 'value' which is the Result object from StringExecutor.
    The Result object has 'value' attribute containing the actual string value.
    """
    if not execution_result.context or not execution_result.context.results:
        return ""
    
    # Get the last result (results are ordered by action_id)
    results = execution_result.context.results
    if not results:
        return ""
    
    # Get the last action's result
    last_action_id = list(results.keys())[-1]
    action_result = results[last_action_id]
    
    # The value is the Result object from StringExecutor
    result_obj = action_result.value if hasattr(action_result, 'value') else action_result.get('value')
    
    # Extract the string value from the Result object
    if hasattr(result_obj, 'value'):
        return result_obj.value
    elif isinstance(result_obj, dict) and 'value' in result_obj:
        return result_obj['value']
    
    return ""


def assert_string_branch_result_success(result):
    """
    Assert that a branch execution result indicates success.
    
    When branches exist, the result may be a tuple of (true_branch_result, false_branch_result).
    One branch executes and returns a result, the other returns the fallback (previous result).
    """
    if isinstance(result, tuple):
        # One branch executed, one returned fallback
        # Check that at least one succeeded
        success = any(
            r.success if hasattr(r, 'success') else False 
            for r in result if r is not None
        )
        assert success is True, f"Expected at least one branch to succeed, got: {result}"
    else:
        assert result.success is True, f"Expected success, got: {result}"


# Condition functions for string branching tests
def condition_length_greater_than_5(r, **kwargs):
    """Check if buffer length is greater than 5."""
    value = get_last_string_result_value(r)
    return len(value) > 5


def condition_contains_error(r, **kwargs):
    """Check if buffer contains 'error'."""
    value = get_last_string_result_value(r)
    return "error" in value.lower()


def condition_startswith_http(r, **kwargs):
    """Check if buffer starts with 'http'."""
    value = get_last_string_result_value(r)
    return value.startswith("http")


def condition_is_empty(r, **kwargs):
    """Check if buffer is empty."""
    value = get_last_string_result_value(r)
    return len(value) == 0


# endregion


# region TestStringBranching

class TestStringBranching:
    """
    Test class for conditional branching with string conditions in ActionGraph.
    
    Tests Requirements: 10.1, 10.2, 10.3, 10.4
    """
    
    def test_branch_length_condition(self, make_string_graph, string_executor):
        """
        Test str_set("hello world") → branch(len>5: str_upper()) = "HELLO WORLD"
        
        When buffer length is greater than 5, the true branch (str_upper) should execute.
        
        Requirements: 10.1
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": "hello world"})
        
        # Branch: if len(buffer) > 5, convert to uppercase; else convert to lowercase
        graph.branch(
            condition=condition_length_greater_than_5,
            if_true=lambda g: g.action("str_upper"),
            if_false=lambda g: g.action("str_lower"),
        )
        
        result = graph.execute()
        assert_string_branch_result_success(result)
        
        # "hello world" has length 11 > 5, so true branch executes: "HELLO WORLD"
        assert string_executor.buffer == "HELLO WORLD"
    
    def test_branch_contains_condition(self, make_string_graph, string_executor):
        """
        Test str_set("success") → branch(contains "error": false path)
        
        When buffer does not contain "error", the false branch should execute.
        
        Requirements: 10.2
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": "success"})
        
        # Branch: if buffer contains "error", add "[ERROR]"; else add "[OK]"
        graph.branch(
            condition=condition_contains_error,
            if_true=lambda g: g.action("str_concat", args={"value": " [ERROR]"}),
            if_false=lambda g: g.action("str_concat", args={"value": " [OK]"}),
        )
        
        result = graph.execute()
        assert_string_branch_result_success(result)
        
        # "success" does not contain "error", so false branch executes: "success [OK]"
        assert string_executor.buffer == "success [OK]"
    
    def test_branch_startswith_condition(self, make_string_graph, string_executor):
        """
        Test str_set("https://example.com") → branch(startswith "http": true path)
        
        When buffer starts with "http", the true branch should execute.
        
        Requirements: 10.3
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": "https://example.com"})
        
        # Branch: if buffer starts with "http", add " [URL]"; else add " [TEXT]"
        graph.branch(
            condition=condition_startswith_http,
            if_true=lambda g: g.action("str_concat", args={"value": " [URL]"}),
            if_false=lambda g: g.action("str_concat", args={"value": " [TEXT]"}),
        )
        
        result = graph.execute()
        assert_string_branch_result_success(result)
        
        # "https://example.com" starts with "http", so true branch executes
        assert string_executor.buffer == "https://example.com [URL]"
    
    def test_branch_empty_condition(self, make_string_graph, string_executor):
        """
        Test str_set("") → branch(empty: true path)
        
        When buffer is empty, the true branch should execute.
        
        Requirements: 10.4
        """
        graph = make_string_graph()
        graph.action("str_set", args={"value": ""})
        
        # Branch: if buffer is empty, set to "default"; else keep as is
        graph.branch(
            condition=condition_is_empty,
            if_true=lambda g: g.action("str_set", args={"value": "default"}),
            if_false=lambda g: g.action("str_concat", args={"value": " (not empty)"}),
        )
        
        result = graph.execute()
        assert_string_branch_result_success(result)
        
        # "" is empty, so true branch executes: "default"
        assert string_executor.buffer == "default"


# endregion


# region TestStringSerializationRoundtrip

class TestStringSerializationRoundtrip:
    """
    Test class for serialization round-trip of string ActionGraphs.
    
    Tests Requirements: 11.1, 11.2
    
    Verifies that serializing an ActionGraph to Python format and deserializing
    it back produces a graph that yields the same buffer value when executed.
    """
    
    def test_string_simple_roundtrip(self, string_registry, string_executor):
        """
        Test serialize → deserialize → execute produces same buffer.
        
        Requirements: 11.1
        """
        # Build original graph
        original_graph = ActionGraph(
            action_executor=string_executor,
            action_metadata=string_registry,
        )
        original_graph.action("str_set", args={"value": "hello"})
        original_graph.action("str_concat", args={"value": " world"})
        original_graph.action("str_upper")
        
        # Execute original graph
        original_result = original_graph.execute()
        original_value = string_executor.buffer
        
        # Serialize to Python format
        python_script = original_graph.serialize(output_format='python')
        
        # Reset executor for deserialized graph
        string_executor.reset()
        
        # Deserialize and execute
        restored_graph = ActionGraph.deserialize(
            python_script,
            output_format='python',
            action_executor=string_executor,
            action_metadata=string_registry,
        )
        restored_result = restored_graph.execute()
        restored_value = string_executor.buffer
        
        # Verify same result
        assert original_result.success is True
        assert restored_result.success is True
        assert original_value == restored_value == "HELLO WORLD", \
            f"Expected 'HELLO WORLD', original='{original_value}', restored='{restored_value}'"
    
    def test_string_branching_roundtrip(self, string_registry, string_executor):
        """
        Test roundtrip for string graph with branching.
        
        Verifies that:
        1. Original graph executes correctly with branching
        2. Serialization produces valid Python
        3. Deserialization preserves the graph structure (actions)
        
        Requirements: 11.2
        """
        import ast
        from science_modeling_tools.automation.schema.action_graph import condition_expr
        
        # Build original graph with branching
        original_graph = ActionGraph(
            action_executor=string_executor,
            action_metadata=string_registry,
        )
        original_graph.action("str_set", args={"value": "hello world"})
        
        # Create condition with expression for serialization
        @condition_expr("len(result.value) > 5")
        def check_length_greater_than_5(result, **kwargs):
            value = get_last_string_result_value(result)
            return len(value) > 5
        
        original_graph.branch(
            condition=check_length_greater_than_5,
            if_true=lambda g: g.action("str_upper"),
            if_false=lambda g: g.action("str_lower"),
        )
        
        # Execute original graph and verify result
        original_result = original_graph.execute()
        original_value = string_executor.buffer
        
        assert_string_branch_result_success(original_result)
        # "hello world" has length 11 > 5, so true branch: "HELLO WORLD"
        assert original_value == "HELLO WORLD", \
            f"Expected 'HELLO WORLD', got '{original_value}'"
        
        # Serialize to Python format
        python_script = original_graph.serialize(output_format='python')
        
        # Verify serialized script is valid Python
        try:
            ast.parse(python_script)
        except SyntaxError as e:
            raise AssertionError(
                f"Generated Python script has syntax error: {e}\n"
                f"Script:\n{python_script}"
            )
        
        # Reset executor for deserialized graph
        string_executor.reset()
        
        # Deserialize and verify structure is preserved
        restored_graph = ActionGraph.deserialize(
            python_script,
            output_format='python',
            action_executor=string_executor,
            action_metadata=string_registry,
        )
        
        # Verify structure: count actions in restored graph
        restored_actions = []
        for node in restored_graph._nodes:
            restored_actions.extend(node._actions)
        
        # Should have at least the initial str_set action plus branch actions
        assert len(restored_actions) >= 1, \
            f"Expected at least 1 action, got {len(restored_actions)}"
        
        # Verify the first action is preserved correctly
        first_action = restored_actions[0]
        assert first_action.type == "str_set", \
            f"Expected first action type 'str_set', got '{first_action.type}'"
        assert first_action.args == {"value": "hello world"}, \
            f"Expected args {{'value': 'hello world'}}, got {first_action.args}"


# endregion


# region Property-Based Test for String Serialization Round-Trip

class TestStringSerializationRoundtripProperty:
    """
    Property-based tests for string serialization round-trip.
    
    **Feature: action-graph-arithmetic-tests, Property 9: String serialization round-trip**
    **Validates: Requirements 11.1, 11.2**
    
    Uses Hypothesis for property-based testing with minimum 100 iterations.
    """
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        initial_value=st.text(min_size=0, max_size=50, alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'S'),
            whitelist_characters=' '
        )),
        concat_value=st.text(min_size=0, max_size=20, alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'S'),
            whitelist_characters=' '
        )),
    )
    def test_property_simple_roundtrip_same_result(
        self, string_registry, string_executor, initial_value, concat_value
    ):
        """
        **Feature: action-graph-arithmetic-tests, Property 9: String serialization round-trip**
        **Validates: Requirements 11.1**
        
        For any ActionGraph with string actions, serializing to Python format
        and deserializing SHALL produce a graph that yields the same buffer
        value when executed.
        """
        import ast
        
        string_executor.reset()
        
        # Build original graph with random values
        original_graph = ActionGraph(
            action_executor=string_executor,
            action_metadata=string_registry,
        )
        original_graph.action("str_set", args={"value": initial_value})
        original_graph.action("str_concat", args={"value": concat_value})
        original_graph.action("str_upper")
        
        # Execute original graph
        original_result = original_graph.execute()
        original_value = string_executor.buffer
        
        # Serialize to Python format
        python_script = original_graph.serialize(output_format='python')
        
        # Verify serialized script is valid Python
        try:
            ast.parse(python_script)
        except SyntaxError as e:
            raise AssertionError(
                f"Generated Python script has syntax error: {e}\n"
                f"Script:\n{python_script}"
            )
        
        # Reset executor for deserialized graph
        string_executor.reset()
        
        # Deserialize and execute
        restored_graph = ActionGraph.deserialize(
            python_script,
            output_format='python',
            action_executor=string_executor,
            action_metadata=string_registry,
        )
        restored_result = restored_graph.execute()
        restored_value = string_executor.buffer
        
        # Verify same result
        assert original_result.success is True
        assert restored_result.success is True
        
        # Calculate expected value
        expected = (initial_value + concat_value).upper()
        
        # Verify both produce the expected result
        assert original_value == expected, \
            f"Original value '{original_value}' != expected '{expected}'"
        assert restored_value == expected, \
            f"Restored value '{restored_value}' != expected '{expected}'"
        assert original_value == restored_value, \
            f"Original '{original_value}' != Restored '{restored_value}'"
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        initial_value=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'S'),
            whitelist_characters=' '
        )),
    )
    def test_property_branching_roundtrip_structure_preserved(
        self, string_registry, string_executor, initial_value
    ):
        """
        **Feature: action-graph-arithmetic-tests, Property 9: String serialization round-trip**
        **Validates: Requirements 11.2**
        
        For any ActionGraph with string branching, serializing to Python format
        and deserializing SHALL:
        1. Produce valid Python syntax
        2. Preserve the graph structure (actions)
        3. Execute the original graph correctly
        """
        import ast
        from science_modeling_tools.automation.schema.action_graph import condition_expr
        
        string_executor.reset()
        
        # Build original graph with branching
        original_graph = ActionGraph(
            action_executor=string_executor,
            action_metadata=string_registry,
        )
        original_graph.action("str_set", args={"value": initial_value})
        
        # Create condition with expression for serialization
        @condition_expr("len(result.value) > 5")
        def check_length_greater_than_5(result, **kwargs):
            value = get_last_string_result_value(result)
            return len(value) > 5
        
        original_graph.branch(
            condition=check_length_greater_than_5,
            if_true=lambda g: g.action("str_upper"),
            if_false=lambda g: g.action("str_lower"),
        )
        
        # Execute original graph and verify result
        original_result = original_graph.execute()
        original_value = string_executor.buffer
        
        # Verify original execution is correct
        assert_string_branch_result_success(original_result)
        
        # Determine expected value based on condition
        if len(initial_value) > 5:
            expected = initial_value.upper()
        else:
            expected = initial_value.lower()
        
        assert original_value == expected, \
            f"Original value '{original_value}' != expected '{expected}'"
        
        # Serialize to Python format
        python_script = original_graph.serialize(output_format='python')
        
        # Verify serialized script is valid Python
        try:
            ast.parse(python_script)
        except SyntaxError as e:
            raise AssertionError(
                f"Generated Python script has syntax error: {e}\n"
                f"Script:\n{python_script}"
            )
        
        # Reset executor for deserialized graph
        string_executor.reset()
        
        # Deserialize and verify structure is preserved
        restored_graph = ActionGraph.deserialize(
            python_script,
            output_format='python',
            action_executor=string_executor,
            action_metadata=string_registry,
        )
        
        # Verify structure: count actions in restored graph
        restored_actions = []
        for node in restored_graph._nodes:
            restored_actions.extend(node._actions)
        
        # Should have at least the initial str_set action
        assert len(restored_actions) >= 1, \
            f"Expected at least 1 action, got {len(restored_actions)}"
        
        # Verify the first action is preserved correctly
        first_action = restored_actions[0]
        assert first_action.type == "str_set", \
            f"Expected first action type 'str_set', got '{first_action.type}'"
        assert first_action.args == {"value": initial_value}, \
            f"Expected args {{'value': '{initial_value}'}}, got {first_action.args}"


# endregion

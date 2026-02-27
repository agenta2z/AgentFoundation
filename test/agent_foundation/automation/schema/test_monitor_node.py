"""
Unit Tests for MonitorNode (Generic Layer)

Tests the MonitorNode class which provides generic monitoring capability:
- Executes any callable returning MonitorResult
- Loop mechanism controlled by max_repeat/repeat_condition
- Fallback result when max iterations reached
- Works with non-WebDriver callables (API polling, file watching, etc.)

These tests verify the generic nature of MonitorNode without WebDriver dependencies.

**Feature: monitor-action**
**Requirements: 6.1, 6.2, 6.3, 6.4**
"""

# Path resolution - must be first
import sys
from pathlib import Path

# Configuration
PIVOT_FOLDER_NAME = 'test'  # The folder name we're inside of

# Get absolute path to this file
current_file = Path(__file__).resolve()

# Navigate up to find the pivot folder (test directory)
current_path = current_file.parent
while current_path.name != PIVOT_FOLDER_NAME and current_path.parent != current_path:
    current_path = current_path.parent

if current_path.name != PIVOT_FOLDER_NAME:
    raise RuntimeError(f"Could not find '{PIVOT_FOLDER_NAME}' folder in path hierarchy")

# ScienceModelingTools root is parent of test/ directory
smt_root = current_path.parent

# Add src directory to path for imports
src_dir = smt_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add SciencePythonUtils if it exists
projects_root = smt_root.parent
rich_python_utils_src = projects_root / "SciencePythonUtils" / "src"

if rich_python_utils_src.exists() and str(rich_python_utils_src) not in sys.path:
    sys.path.insert(0, str(rich_python_utils_src))

import pytest
from unittest.mock import MagicMock, call
from agent_foundation.automation.schema.monitor import (
    MonitorNode,
    MonitorResult,
    MonitorStatus,
)
from rich_python_utils.common_objects.workflow.common.worknode_base import NextNodesSelector


# =============================================================================
# Test Fixtures
# =============================================================================

def create_success_result(content="matched"):
    """Create a successful NextNodesSelector wrapping MonitorResult."""
    result = MonitorResult(
        success=True,
        status=MonitorStatus.CONDITION_MET,
        matched_content=content,
        check_count=1
    )
    return NextNodesSelector(
        include_self=False,
        include_others=True,
        result=result
    )


def create_failure_result():
    """Create a failure NextNodesSelector wrapping MonitorResult."""
    result = MonitorResult(
        success=False,
        status=MonitorStatus.MAX_ITERATIONS,
        check_count=1
    )
    return NextNodesSelector(
        include_self=True,
        include_others=False,
        result=result
    )


def create_error_result(message="Error occurred"):
    """Create an error NextNodesSelector wrapping MonitorResult."""
    result = MonitorResult(
        success=False,
        status=MonitorStatus.ERROR,
        error_message=message,
        check_count=1
    )
    return NextNodesSelector(
        include_self=False,
        include_others=False,
        result=result
    )


# =============================================================================
# Task 9.1: Test MonitorNode executes any callable returning MonitorResult
# =============================================================================

class TestMonitorNodeExecutesCallable:
    """Tests for MonitorNode executing arbitrary callables."""
    
    def test_executes_simple_callable(self):
        """MonitorNode should execute a simple callable returning NextNodesSelector."""
        def simple_iteration(prev_result=None):
            return create_success_result("simple content")

        node = MonitorNode(
            name="test_monitor",
            iteration=simple_iteration,
            poll_interval=0  # No delay for tests
        )

        result = node._execute_iteration()

        assert result.result.success is True
        assert result.result.matched_content == "simple content"
    
    def test_executes_lambda_callable(self):
        """MonitorNode should execute a lambda callable."""
        node = MonitorNode(
            name="lambda_monitor",
            iteration=lambda prev: create_success_result("lambda result"),
            poll_interval=0  # No delay for tests
        )

        result = node._execute_iteration()

        assert result.result.success is True
        assert result.result.matched_content == "lambda result"
    
    def test_executes_class_method_callable(self):
        """MonitorNode should execute a class method as callable."""
        class Checker:
            def __init__(self):
                self.call_count = 0

            def check(self, prev_result=None):
                self.call_count += 1
                result = MonitorResult(
                    success=True,
                    status=MonitorStatus.CONDITION_MET,
                    check_count=self.call_count
                )
                return NextNodesSelector(include_self=False, include_others=True, result=result)

        checker = Checker()
        node = MonitorNode(
            name="method_monitor",
            iteration=checker.check,
            poll_interval=0  # No delay for tests
        )

        result = node._execute_iteration()

        assert result.result.success is True
        assert checker.call_count == 1
    
    def test_passes_prev_result_to_callable(self):
        """MonitorNode should pass prev_result to the iteration callable."""
        received_prev = None

        def iteration_with_prev(prev_result=None):
            nonlocal received_prev
            received_prev = prev_result
            return create_success_result()

        node = MonitorNode(
            name="prev_monitor",
            iteration=iteration_with_prev,
            poll_interval=0  # No delay for tests
        )

        prev = create_failure_result()
        node._execute_iteration(prev)

        assert received_prev is prev
    
    def test_returns_error_when_no_iteration_configured(self):
        """MonitorNode should return error result when no iteration is set."""
        node = MonitorNode(
            name="empty_monitor",
            iteration=None
        )
        
        result = node._execute_iteration()
        
        assert result.success is False
        assert result.status == MonitorStatus.ERROR
        assert "No iteration configured" in result.error_message


# =============================================================================
# Task 9.2: Test max_repeat limits iterations
# =============================================================================

class TestMaxRepeatLimitsIterations:
    """Tests for max_repeat limiting iteration count."""
    
    def test_node_has_max_repeat_attribute(self):
        """MonitorNode should have max_repeat attribute from WorkGraphNode."""
        node = MonitorNode(
            name="repeat_monitor",
            iteration=lambda p: create_failure_result(),
            max_repeat=10
        )
        
        assert node.max_repeat == 10
    
    def test_max_repeat_defaults_to_one(self):
        """MonitorNode max_repeat should default to 1."""
        node = MonitorNode(
            name="default_monitor",
            iteration=lambda p: create_failure_result()
        )
        
        # WorkGraphNode default is 1
        assert node.max_repeat == 1
    
    def test_iteration_tracks_check_count(self):
        """Iteration callable can track check count in result."""
        call_count = 0

        def counting_iteration(prev_result=None):
            nonlocal call_count
            call_count += 1
            result = MonitorResult(
                success=False,
                status=MonitorStatus.MAX_ITERATIONS,
                check_count=call_count
            )
            return NextNodesSelector(include_self=True, include_others=False, result=result)

        node = MonitorNode(
            name="counting_monitor",
            iteration=counting_iteration,
            max_repeat=5,
            poll_interval=0  # No delay for tests
        )

        # Execute multiple times
        for _ in range(5):
            result = node._execute_iteration()

        assert call_count == 5
        assert result.result.check_count == 5


# =============================================================================
# Task 9.3: Test repeat_condition stops loop when condition met
# =============================================================================

class TestRepeatConditionStopsLoop:
    """Tests for repeat_condition controlling loop termination."""
    
    def test_node_has_repeat_condition_attribute(self):
        """MonitorNode should have repeat_condition attribute from WorkGraphNode."""
        def stop_condition(result):
            return result.success
        
        node = MonitorNode(
            name="condition_monitor",
            iteration=lambda p: create_success_result(),
            repeat_condition=stop_condition
        )
        
        assert node.repeat_condition is stop_condition
    
    def test_repeat_condition_receives_result(self):
        """repeat_condition should receive the iteration result."""
        received_results = []

        def tracking_condition(result):
            received_results.append(result)
            # result is NextNodesSelector, access inner MonitorResult
            return result.result.success

        node = MonitorNode(
            name="tracking_monitor",
            iteration=lambda p: create_success_result("tracked"),
            repeat_condition=tracking_condition,
            poll_interval=0  # No delay for tests
        )

        result = node._execute_iteration()

        # Verify the result can be passed to repeat_condition
        should_stop = node.repeat_condition(result)
        assert should_stop is True
        assert len(received_results) == 1
        assert received_results[0].result.matched_content == "tracked"
    
    def test_repeat_condition_can_stop_on_success(self):
        """repeat_condition returning True should indicate loop should stop."""
        def stop_on_success(result):
            # result is NextNodesSelector
            return result.result.success

        node = MonitorNode(
            name="stop_monitor",
            iteration=lambda p: create_success_result(),
            repeat_condition=stop_on_success,
            poll_interval=0  # No delay for tests
        )

        result = node._execute_iteration()
        should_stop = node.repeat_condition(result)

        assert should_stop is True

    def test_repeat_condition_can_continue_on_failure(self):
        """repeat_condition returning False should indicate loop should continue."""
        def stop_on_success(result):
            # result is NextNodesSelector
            return result.result.success

        node = MonitorNode(
            name="continue_monitor",
            iteration=lambda p: create_failure_result(),
            repeat_condition=stop_on_success,
            poll_interval=0  # No delay for tests
        )

        result = node._execute_iteration()
        should_stop = node.repeat_condition(result)

        assert should_stop is False


# =============================================================================
# Task 9.4: Test fallback_result returned when max iterations reached
# =============================================================================

class TestFallbackResultOnMaxIterations:
    """Tests for fallback_result when max iterations reached."""
    
    def test_node_has_fallback_result_attribute(self):
        """MonitorNode should have fallback_result attribute from WorkGraphNode."""
        fallback = create_failure_result()
        
        node = MonitorNode(
            name="fallback_monitor",
            iteration=lambda p: create_failure_result(),
            fallback_result=fallback
        )
        
        assert node.fallback_result is fallback
    
    def test_fallback_result_can_be_monitor_result(self):
        """fallback_result can be a MonitorResult for consistent typing."""
        fallback = MonitorResult(
            success=False,
            status=MonitorStatus.MAX_ITERATIONS,
            error_message="Max iterations reached without condition met",
            check_count=100
        )
        
        node = MonitorNode(
            name="typed_fallback_monitor",
            iteration=lambda p: create_failure_result(),
            fallback_result=fallback
        )
        
        assert node.fallback_result.status == MonitorStatus.MAX_ITERATIONS
        assert "Max iterations" in node.fallback_result.error_message
    
    def test_fallback_result_defaults_to_none(self):
        """fallback_result should default to None."""
        node = MonitorNode(
            name="no_fallback_monitor",
            iteration=lambda p: create_failure_result()
        )
        
        assert node.fallback_result is None


# =============================================================================
# Task 9.5: Test MonitorNode works with non-WebDriver callables (API polling mock)
# =============================================================================

class TestNonWebDriverCallables:
    """Tests verifying MonitorNode works with non-WebDriver callables."""
    
    def test_api_polling_mock(self):
        """MonitorNode should work with API polling callable."""
        api_responses = [
            {"status": "pending"},
            {"status": "pending"},
            {"status": "ready", "data": "result"}
        ]
        call_index = 0

        def api_poll_iteration(prev_result=None):
            nonlocal call_index
            response = api_responses[min(call_index, len(api_responses) - 1)]
            call_index += 1

            if response.get("status") == "ready":
                result = MonitorResult(
                    success=True,
                    status=MonitorStatus.CONDITION_MET,
                    matched_content=response.get("data"),
                    check_count=call_index
                )
                return NextNodesSelector(include_self=False, include_others=True, result=result)
            result = MonitorResult(
                success=False,
                status=MonitorStatus.MAX_ITERATIONS,
                check_count=call_index
            )
            return NextNodesSelector(include_self=True, include_others=False, result=result)

        node = MonitorNode(
            name="api_monitor",
            iteration=api_poll_iteration,
            max_repeat=10,
            poll_interval=0  # No delay for tests
        )

        # Simulate polling until ready
        result = None
        for _ in range(10):
            result = node._execute_iteration()
            if result.result.success:
                break

        assert result.result.success is True
        assert result.result.matched_content == "result"
        assert result.result.check_count == 3
    
    def test_file_watcher_mock(self):
        """MonitorNode should work with file watching callable."""
        file_exists = False

        def file_watch_iteration(prev_result=None):
            if file_exists:
                result = MonitorResult(
                    success=True,
                    status=MonitorStatus.CONDITION_MET,
                    matched_content="/path/to/file.txt",
                    check_count=1
                )
                return NextNodesSelector(include_self=False, include_others=True, result=result)
            result = MonitorResult(
                success=False,
                status=MonitorStatus.MAX_ITERATIONS,
                check_count=1
            )
            return NextNodesSelector(include_self=True, include_others=False, result=result)

        node = MonitorNode(
            name="file_monitor",
            iteration=file_watch_iteration,
            max_repeat=100,
            poll_interval=0  # No delay for tests
        )

        # File doesn't exist yet
        result = node._execute_iteration()
        assert result.result.success is False

        # File appears
        file_exists = True
        result = node._execute_iteration()
        assert result.result.success is True
        assert result.result.matched_content == "/path/to/file.txt"
    
    def test_queue_monitor_mock(self):
        """MonitorNode should work with queue monitoring callable."""
        queue_items = []

        def queue_monitor_iteration(prev_result=None):
            if len(queue_items) > 0:
                item = queue_items[0]
                result = MonitorResult(
                    success=True,
                    status=MonitorStatus.CONDITION_MET,
                    matched_content=item,
                    metadata={"queue_length": len(queue_items)}
                )
                return NextNodesSelector(include_self=False, include_others=True, result=result)
            result = MonitorResult(
                success=False,
                status=MonitorStatus.MAX_ITERATIONS,
                metadata={"queue_length": 0}
            )
            return NextNodesSelector(include_self=True, include_others=False, result=result)

        node = MonitorNode(
            name="queue_monitor",
            iteration=queue_monitor_iteration,
            max_repeat=50,
            poll_interval=0  # No delay for tests
        )

        # Queue is empty
        result = node._execute_iteration()
        assert result.result.success is False
        assert result.result.metadata["queue_length"] == 0

        # Item arrives in queue
        queue_items.append({"task_id": 123, "data": "payload"})
        result = node._execute_iteration()
        assert result.result.success is True
        assert result.result.matched_content["task_id"] == 123
    
    def test_database_poll_mock(self):
        """MonitorNode should work with database polling callable."""
        db_record = None

        def db_poll_iteration(prev_result=None):
            if db_record is not None:
                result = MonitorResult(
                    success=True,
                    status=MonitorStatus.CONDITION_MET,
                    matched_content=db_record,
                    check_count=1
                )
                return NextNodesSelector(include_self=False, include_others=True, result=result)
            result = MonitorResult(
                success=False,
                status=MonitorStatus.MAX_ITERATIONS,
                check_count=1
            )
            return NextNodesSelector(include_self=True, include_others=False, result=result)

        node = MonitorNode(
            name="db_monitor",
            iteration=db_poll_iteration,
            poll_interval=0  # No delay for tests
        )

        # Record doesn't exist
        result = node._execute_iteration()
        assert result.result.success is False

        # Record is created
        db_record = {"id": 1, "status": "completed"}
        result = node._execute_iteration()
        assert result.result.success is True
        assert result.result.matched_content["status"] == "completed"
    
    def test_stateful_closure_callable(self):
        """MonitorNode should work with stateful closure callables."""
        def create_stateful_monitor(threshold):
            state = {"count": 0}

            def iteration(prev_result=None):
                state["count"] += 1
                if state["count"] >= threshold:
                    result = MonitorResult(
                        success=True,
                        status=MonitorStatus.CONDITION_MET,
                        matched_content=f"Reached {threshold}",
                        check_count=state["count"]
                    )
                    return NextNodesSelector(include_self=False, include_others=True, result=result)
                result = MonitorResult(
                    success=False,
                    status=MonitorStatus.MAX_ITERATIONS,
                    check_count=state["count"]
                )
                return NextNodesSelector(include_self=True, include_others=False, result=result)

            return iteration

        node = MonitorNode(
            name="stateful_monitor",
            iteration=create_stateful_monitor(threshold=3),
            max_repeat=10,
            poll_interval=0  # No delay for tests
        )

        # First two iterations fail
        result = node._execute_iteration()
        assert result.result.success is False
        assert result.result.check_count == 1

        result = node._execute_iteration()
        assert result.result.success is False
        assert result.result.check_count == 2

        # Third iteration succeeds
        result = node._execute_iteration()
        assert result.result.success is True
        assert result.result.check_count == 3
        assert result.result.matched_content == "Reached 3"


"""
Property-based tests for ActionFlow refactoring.

Tests correctness properties from the action-flow-workflow-refactor design document.
Focuses on Phase 2: ActionFlow using Workflow internally.
"""

import sys
import inspect
from pathlib import Path

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
_rich_python_utils_src = _workspace_root / "SciencePythonUtils" / "src"
if _rich_python_utils_src.exists() and str(_rich_python_utils_src) not in sys.path:
    sys.path.insert(0, str(_rich_python_utils_src))

from typing import Any, Dict, List, Optional
from hypothesis import given, strategies as st, settings

from agent_foundation.automation.schema.action_flow import ActionFlow
from agent_foundation.automation.schema.common import (
    Action,
    ActionSequence,
    ActionResult,
    ExecutionRuntime,
    ExecutionResult,
    TargetSpec,
)
from agent_foundation.automation.schema.action_metadata import ActionMetadataRegistry


# region Test Fixtures and Strategies

def mock_executor(
    action_type: str,
    action_target: Optional[str],
    action_args: Optional[Dict[str, Any]] = None,
    action_target_strategy: Optional[str] = None,
) -> str:
    """Mock action executor that returns a success string."""
    return f"executed_{action_type}_{action_target}"


def tracking_executor_factory():
    """Factory that creates a tracking executor with execution history."""
    execution_history = []
    
    def tracking_executor(
        action_type: str,
        action_target: Optional[str],
        action_args: Optional[Dict[str, Any]] = None,
        action_target_strategy: Optional[str] = None,
    ) -> str:
        """Mock executor that tracks all executions."""
        execution_history.append({
            'action_type': action_type,
            'action_target': action_target,
            'action_args': action_args,
            'stack_depth': len(inspect.stack()),
        })
        return f"executed_{action_type}_{action_target}"
    
    return tracking_executor, execution_history


# Hypothesis strategies for generating test data
action_type_strategy = st.sampled_from(['click', 'type', 'navigate', 'wait', 'scroll'])
action_id_strategy = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N')))
# Exclude curly braces from target values to avoid invalid template syntax like '{A:'
# which causes "unmatched '{' in format spec" errors during template processing
target_value_strategy = st.text(
    min_size=1, max_size=50,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P'), blacklist_characters='{}')
)

simple_action_strategy = st.builds(
    Action,
    id=action_id_strategy,
    type=action_type_strategy,
    target=st.one_of(target_value_strategy, st.none()),
    args=st.none(),
)

# endregion


# region Helper Functions

def create_sequence(actions: List[Action], seq_id: str = "test_sequence") -> ActionSequence:
    """Helper to create ActionSequence with required id field."""
    return ActionSequence(id=seq_id, actions=actions)

# endregion


# region Property Tests

@settings(max_examples=100, deadline=None)  # deadline=None to avoid flaky timing issues
@given(st.lists(simple_action_strategy, min_size=1, max_size=200, unique_by=lambda a: a.id))
def test_iterative_execution_stack_depth(actions: List[Action]):
    """
    Property 1: Iterative execution preserves stack depth.
    
    **Feature: action-flow-workflow-refactor, Property 1: Iterative execution preserves stack depth**
    **Validates: Requirements 1.1, 1.2**
    
    For any ActionSequence of length N (where N can be arbitrarily large),
    executing via ActionFlow SHALL maintain call stack depth O(1) regardless of N,
    using Workflow's iterative for-loop execution pattern.
    """
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    
    tracking_executor, execution_history = tracking_executor_factory()
    flow = ActionFlow(action_executor=tracking_executor, action_metadata=registry)
    
    # Record stack depth before execution
    initial_stack_depth = len(inspect.stack())
    
    result = flow.execute(sequence)
    
    assert result.success, f"Execution should succeed, got error: {result.error}"
    assert len(execution_history) == len(actions), \
        f"All {len(actions)} actions should be executed, got {len(execution_history)}"
    
    # Verify stack depth remained constant (O(1)) during execution
    # Allow small constant overhead (< 20 frames) but should not grow with N
    max_stack_depth = max(e['stack_depth'] for e in execution_history)
    stack_growth = max_stack_depth - initial_stack_depth
    
    # Stack growth should be bounded by a constant, not proportional to N
    assert stack_growth < 30, \
        f"Stack depth grew by {stack_growth} frames, should be O(1). " \
        f"Initial: {initial_stack_depth}, Max: {max_stack_depth}, N={len(actions)}"


@settings(max_examples=100)
@given(st.lists(simple_action_strategy, min_size=2, max_size=10, unique_by=lambda a: a.id))
def test_context_flows_through_actions(actions: List[Action]):
    """
    Property 9: Context flows through actions.
    
    **Feature: action-flow-workflow-refactor, Property 9: Context flows through actions**
    **Validates: Requirements 4.1, 4.2, 4.3**
    
    For any ActionSequence with actions [A1, A2, ..., An], when Ak executes
    it SHALL have access to results of [A1, ..., A(k-1)] via ExecutionRuntime,
    and after Ak completes, the ExecutionRuntime SHALL contain Ak's result.
    """
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    flow = ActionFlow(action_executor=mock_executor, action_metadata=registry)
    
    result = flow.execute(sequence)
    
    assert result.success, f"Execution should succeed, got error: {result.error}"
    
    # Verify all action results are stored in context
    for action in actions:
        action_result = result.context.get_result(action.id)
        assert action_result is not None, \
            f"Result for action '{action.id}' should be in context"
        assert isinstance(action_result, ActionResult), \
            f"Result should be ActionResult, got {type(action_result)}"
        assert action_result.success is True, \
            f"Action '{action.id}' result should be successful"


@settings(max_examples=100)
@given(
    st.lists(simple_action_strategy, min_size=1, max_size=5, unique_by=lambda a: a.id),
    st.dictionaries(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L',))),
        st.text(min_size=1, max_size=20),
        min_size=0,
        max_size=3,
    )
)
def test_api_backward_compatibility(actions: List[Action], initial_vars: Dict[str, str]):
    """
    Property 10: API backward compatibility.
    
    **Feature: action-flow-workflow-refactor, Property 10: API backward compatibility**
    **Validates: Requirements 5.1, 5.2**
    
    For any valid call to the current ActionFlow.execute(),
    the refactored version SHALL accept the same parameters
    and return an ExecutionResult with the same structure.
    """
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    flow = ActionFlow(action_executor=mock_executor, action_metadata=registry)
    
    # Test with initial_variables parameter
    result = flow.execute(sequence, initial_variables=initial_vars)
    
    # Verify return type
    assert isinstance(result, ExecutionResult), \
        f"execute() should return ExecutionResult, got {type(result)}"
    
    # Verify ExecutionResult structure
    assert hasattr(result, 'success'), "ExecutionResult should have 'success' attribute"
    assert hasattr(result, 'context'), "ExecutionResult should have 'context' attribute"
    assert hasattr(result, 'error'), "ExecutionResult should have 'error' attribute"
    assert hasattr(result, 'failed_action_id'), "ExecutionResult should have 'failed_action_id' attribute"
    
    # Verify context contains initial variables
    assert isinstance(result.context, ExecutionRuntime), \
        f"context should be ExecutionRuntime, got {type(result.context)}"
    
    for key, value in initial_vars.items():
        assert result.context.variables.get(key) == value, \
            f"Initial variable '{key}' should be preserved in context"


@settings(max_examples=50)
@given(st.integers(min_value=1, max_value=5))
def test_execution_stops_on_failure(fail_at_index: int):
    """
    Test that execution stops when an action fails.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Error handling behavior**
    
    When an action fails, subsequent actions should not be executed.
    """
    num_actions = fail_at_index + 3  # Ensure there are actions after the failing one
    actions = [
        Action(id=f'action_{i}', type='click', target=f'target_{i}')
        for i in range(num_actions)
    ]
    
    execution_count = [0]
    
    def failing_executor(
        action_type: str,
        action_target: Optional[str],
        action_args: Optional[Dict[str, Any]] = None,
        action_target_strategy: Optional[str] = None,
    ) -> str:
        execution_count[0] += 1
        if execution_count[0] == fail_at_index + 1:  # 1-indexed
            raise ValueError(f"Simulated failure at action {fail_at_index}")
        return f"executed_{action_type}"
    
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    flow = ActionFlow(action_executor=failing_executor, action_metadata=registry)
    
    result = flow.execute(sequence)
    
    assert result.success is False, "Execution should fail"
    assert result.failed_action_id == f'action_{fail_at_index}', \
        f"Failed action ID should be 'action_{fail_at_index}', got '{result.failed_action_id}'"
    assert execution_count[0] == fail_at_index + 1, \
        f"Should have executed {fail_at_index + 1} actions, got {execution_count[0]}"


def test_single_action_sequence_execution():
    """
    Test that single action sequence executes successfully.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Edge case handling (minimum valid sequence)
    """
    actions = [Action(id='single_action', type='click', target='#button')]
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    flow = ActionFlow(action_executor=mock_executor, action_metadata=registry)
    
    result = flow.execute(sequence)
    
    assert result.success is True, "Single action sequence should succeed"
    assert isinstance(result.context, ExecutionRuntime), "Context should be ExecutionRuntime"
    assert result.context.get_result('single_action') is not None, "Result should be stored"


def test_action_flow_is_workflow():
    """
    Test that ActionFlow is a Workflow subclass.

    **Feature: action-flow-workflow-refactor**
    **Validates: ActionFlow directly inherits from Workflow for O(1) stack depth**
    """
    from rich_python_utils.common_objects.workflow.workflow import Workflow

    assert issubclass(ActionFlow, Workflow), \
        "ActionFlow should be a subclass of Workflow"


@settings(max_examples=50)
@given(st.lists(simple_action_strategy, min_size=1, max_size=5, unique_by=lambda a: a.id))
def test_result_persistence_on_save_enabled(actions: List[Action]):
    """
    Property 11: Result persistence on save enabled.
    
    **Feature: action-flow-workflow-refactor, Property 11: Result persistence on save enabled**
    **Validates: Requirements 6.1**
    
    When enable_result_save is True, after executing an action,
    the ActionResult SHALL be persisted to disk at the configured path.
    """
    import tempfile
    import os
    
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        flow = ActionFlow(
            action_executor=mock_executor,
            action_metadata=registry,
            enable_result_save=True,
            result_save_dir=temp_dir,
        )
        
        result = flow.execute(sequence)
        
        assert result.success, f"Execution should succeed, got error: {result.error}"
        
        # Verify result files were created for each action
        for action in actions:
            result_path = os.path.join(temp_dir, f"action_result_{action.id}.pkl")
            assert os.path.exists(result_path), \
                f"Result file should exist for action '{action.id}' at {result_path}"


@settings(max_examples=50)
@given(st.lists(simple_action_strategy, min_size=2, max_size=5, unique_by=lambda a: a.id))
def test_resume_skips_saved_results(actions: List[Action]):
    """
    Property 12: Resume skips saved results.
    
    **Feature: action-flow-workflow-refactor, Property 12: Resume skips saved results**
    **Validates: Requirements 6.2**
    
    When resume_with_saved_results is True and saved results exist,
    the corresponding actions SHALL be skipped and their saved results
    SHALL be loaded into ExecutionRuntime.
    """
    import tempfile
    
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # First execution: save results
        flow1 = ActionFlow(
            action_executor=mock_executor,
            action_metadata=registry,
            enable_result_save=True,
            result_save_dir=temp_dir,
        )
        result1 = flow1.execute(sequence)
        assert result1.success, "First execution should succeed"
        
        # Track which actions are executed in second run
        execution_count = [0]
        
        def counting_executor(
            action_type: str,
            action_target: Optional[str],
            action_args: Optional[Dict[str, Any]] = None,
            action_target_strategy: Optional[str] = None,
        ) -> str:
            execution_count[0] += 1
            return f"executed_{action_type}_{action_target}"
        
        # Second execution: resume with saved results
        flow2 = ActionFlow(
            action_executor=counting_executor,
            action_metadata=registry,
            enable_result_save=True,
            resume_with_saved_results=True,
            result_save_dir=temp_dir,
        )
        result2 = flow2.execute(sequence)
        
        assert result2.success, "Second execution should succeed"
        
        # No actions should be executed since all have saved results
        assert execution_count[0] == 0, \
            f"No actions should be executed on resume, got {execution_count[0]}"
        
        # All results should still be in context (loaded from saved)
        for action in actions:
            action_result = result2.context.get_result(action.id)
            assert action_result is not None, \
                f"Result for action '{action.id}' should be loaded from saved"
            assert action_result.success is True, \
                f"Loaded result for '{action.id}' should be successful"


def test_partial_resume_executes_remaining():
    """
    Test that partial resume executes only remaining actions.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Requirements 6.2 (partial resume scenario)**
    
    When some actions have saved results and others don't,
    only the actions without saved results should be executed.
    """
    import tempfile
    import os
    
    actions = [
        Action(id='action_1', type='click', target='#btn1'),
        Action(id='action_2', type='click', target='#btn2'),
        Action(id='action_3', type='click', target='#btn3'),
    ]
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # First execution: save results for first 2 actions only
        flow1 = ActionFlow(
            action_executor=mock_executor,
            action_metadata=registry,
            enable_result_save=True,
            result_save_dir=temp_dir,
        )
        
        # Execute only first 2 actions
        partial_sequence = create_sequence(actions[:2])
        result1 = flow1.execute(partial_sequence)
        assert result1.success, "Partial execution should succeed"
        
        # Verify only 2 result files exist
        assert os.path.exists(os.path.join(temp_dir, "action_result_action_1.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "action_result_action_2.pkl"))
        assert not os.path.exists(os.path.join(temp_dir, "action_result_action_3.pkl"))
        
        # Track which actions are executed in second run
        executed_actions = []
        
        def tracking_executor(
            action_type: str,
            action_target: Optional[str],
            action_args: Optional[Dict[str, Any]] = None,
            action_target_strategy: Optional[str] = None,
        ) -> str:
            executed_actions.append(action_target)
            return f"executed_{action_type}_{action_target}"
        
        # Second execution: resume with full sequence
        flow2 = ActionFlow(
            action_executor=tracking_executor,
            action_metadata=registry,
            enable_result_save=True,
            resume_with_saved_results=True,
            result_save_dir=temp_dir,
        )
        result2 = flow2.execute(sequence)
        
        assert result2.success, "Resume execution should succeed"
        
        # Only action_3 should be executed
        assert len(executed_actions) == 1, \
            f"Only 1 action should be executed, got {len(executed_actions)}"
        assert executed_actions[0] == '#btn3', \
            f"Only action_3 should be executed, got {executed_actions}"
        
        # All results should be in context
        for action in actions:
            assert result2.context.get_result(action.id) is not None, \
                f"Result for '{action.id}' should be in context"


# endregion


# region Serialization Property Tests

# **Feature: serializable-mixin, Property 6: ActionFlow Sequence Preservation**
# **Validates: Requirements 3.1, 3.2**
@settings(max_examples=100)
@given(st.lists(simple_action_strategy, min_size=1, max_size=5, unique_by=lambda a: a.id))
def test_action_flow_serialization_preserves_sequence(actions: List[Action]):
    """
    Property 6: ActionFlow Sequence Preservation.
    
    **Feature: serializable-mixin, Property 6: ActionFlow Sequence Preservation**
    **Validates: Requirements 3.1, 3.2**
    
    For any ActionFlow with an ActionSequence, serializing then deserializing
    SHALL preserve the action sequence structure.
    """
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    
    original = ActionFlow(
        action_executor=mock_executor,
        action_metadata=registry,
        sequence=sequence,
        enable_result_save=True,
        resume_with_saved_results=False,
        result_save_dir="/tmp/test",
    )
    
    # Serialize
    serialized = original.to_serializable_obj()
    
    # Verify serialized structure
    assert '_type' in serialized, "Missing _type field"
    assert '_module' in serialized, "Missing _module field"
    assert 'version' in serialized, "Missing version field"
    assert 'sequence' in serialized, "Missing sequence field"
    assert 'config' in serialized, "Missing config field"
    
    # Deserialize
    restored = ActionFlow.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    # Verify sequence is preserved
    assert restored.sequence is not None, "Sequence should be restored"
    assert len(restored.sequence.actions) == len(original.sequence.actions), \
        f"Action count mismatch: {len(restored.sequence.actions)} != {len(original.sequence.actions)}"
    
    # Verify each action is preserved
    for orig_action, rest_action in zip(original.sequence.actions, restored.sequence.actions):
        assert rest_action.id == orig_action.id, \
            f"Action ID mismatch: {rest_action.id} != {orig_action.id}"
        assert rest_action.type == orig_action.type, \
            f"Action type mismatch: {rest_action.type} != {orig_action.type}"
        assert rest_action.target == orig_action.target, \
            f"Action target mismatch: {rest_action.target} != {orig_action.target}"


# **Feature: serializable-mixin, Property 6: ActionFlow Sequence Preservation**
# **Validates: Requirements 3.1, 3.2**
@settings(max_examples=100)
@given(st.lists(simple_action_strategy, min_size=1, max_size=5, unique_by=lambda a: a.id))
def test_action_flow_serialization_preserves_config(actions: List[Action]):
    """
    Property 6: ActionFlow config preservation.
    
    **Feature: serializable-mixin, Property 6: ActionFlow Sequence Preservation**
    **Validates: Requirements 3.1, 3.2**
    
    For any ActionFlow, serializing then deserializing SHALL preserve
    the configuration settings.
    """
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    
    original = ActionFlow(
        action_executor=mock_executor,
        action_metadata=registry,
        sequence=sequence,
        enable_result_save=True,
        resume_with_saved_results=True,
        result_save_dir="/tmp/custom_dir",
    )
    
    # Serialize and deserialize
    serialized = original.to_serializable_obj()
    restored = ActionFlow.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    # Verify config is preserved
    assert restored.enable_result_save == original.enable_result_save, \
        f"enable_result_save mismatch: {restored.enable_result_save} != {original.enable_result_save}"
    assert restored.resume_with_saved_results == original.resume_with_saved_results, \
        f"resume_with_saved_results mismatch: {restored.resume_with_saved_results} != {original.resume_with_saved_results}"
    assert restored.result_save_dir == original.result_save_dir, \
        f"result_save_dir mismatch: {restored.result_save_dir} != {original.result_save_dir}"


def test_action_flow_serialization_requires_action_executor():
    """
    Test that from_serializable_obj raises ValueError without action_executor.
    
    **Feature: serializable-mixin**
    **Validates: Context injection requirement**
    """
    actions = [Action(id='test_action', type='click', target='#btn')]
    sequence = create_sequence(actions)
    registry = ActionMetadataRegistry()
    
    original = ActionFlow(
        action_executor=mock_executor,
        action_metadata=registry,
        sequence=sequence,
    )
    
    serialized = original.to_serializable_obj()
    
    # Should raise ValueError without action_executor
    try:
        ActionFlow.from_serializable_obj(serialized)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "action_executor" in str(e), \
            f"Error message should mention action_executor: {e}"


def test_action_flow_serialization_with_none_sequence():
    """
    Test serialization with None sequence.
    
    **Feature: serializable-mixin**
    **Validates: Edge case handling**
    """
    registry = ActionMetadataRegistry()
    
    original = ActionFlow(
        action_executor=mock_executor,
        action_metadata=registry,
        sequence=None,
    )
    
    # Serialize
    serialized = original.to_serializable_obj()
    assert serialized['sequence'] is None, "Sequence should be None in serialized form"
    
    # Deserialize
    restored = ActionFlow.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    assert restored.sequence is None, "Restored sequence should be None"


# endregion


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running property tests for ActionFlow refactoring...")
    
    test_action_flow_workflow_is_workflow()
    print("✓ test_action_flow_workflow_is_workflow passed")
    
    test_single_action_sequence_execution()
    print("✓ test_single_action_sequence_execution passed")
    
    test_iterative_execution_stack_depth()
    print("✓ test_iterative_execution_stack_depth passed")
    
    test_context_flows_through_actions()
    print("✓ test_context_flows_through_actions passed")
    
    test_api_backward_compatibility()
    print("✓ test_api_backward_compatibility passed")
    
    test_execution_stops_on_failure()
    print("✓ test_execution_stops_on_failure passed")
    
    test_result_persistence_on_save_enabled()
    print("✓ test_result_persistence_on_save_enabled passed")
    
    test_resume_skips_saved_results()
    print("✓ test_resume_skips_saved_results passed")
    
    test_partial_resume_executes_remaining()
    print("✓ test_partial_resume_executes_remaining passed")
    
    # Serialization property tests
    print("\nRunning serialization property tests...")
    
    test_action_flow_serialization_preserves_sequence()
    print("✓ test_action_flow_serialization_preserves_sequence passed")
    
    test_action_flow_serialization_preserves_config()
    print("✓ test_action_flow_serialization_preserves_config passed")
    
    test_action_flow_serialization_requires_action_executor()
    print("✓ test_action_flow_serialization_requires_action_executor passed")
    
    test_action_flow_serialization_with_none_sequence()
    print("✓ test_action_flow_serialization_with_none_sequence passed")
    
    print("\nAll property tests passed!")

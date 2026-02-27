"""
Property-based tests for ActionNode.

Tests correctness properties from the action-flow-workflow-refactor design document.
"""

import sys
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

from typing import Any, Dict, Optional
from hypothesis import given, strategies as st, settings
from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode

from agent_foundation.automation.schema.action_node import ActionNode
from agent_foundation.automation.schema.common import (
    Action,
    ActionResult,
    ExecutionRuntime,
    TargetSpec,
    TargetSpecWithFallback,
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


def failing_executor(
    action_type: str,
    action_target: Optional[str],
    action_args: Optional[Dict[str, Any]] = None,
    action_target_strategy: Optional[str] = None,
) -> str:
    """Mock executor that always fails."""
    raise ValueError(f"Failed to execute {action_type}")


# Hypothesis strategies for generating test data
action_type_strategy = st.sampled_from(['click', 'type', 'navigate', 'wait', 'scroll'])
action_id_strategy = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N')))
target_value_strategy = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P')))

target_spec_strategy = st.builds(
    TargetSpec,
    strategy=st.sampled_from(['id', 'xpath', 'css', None]),
    value=target_value_strategy,
)

target_with_fallback_strategy = st.builds(
    TargetSpecWithFallback,
    strategies=st.lists(target_spec_strategy, min_size=1, max_size=5),
)

simple_action_strategy = st.builds(
    Action,
    id=action_id_strategy,
    type=action_type_strategy,
    target=st.one_of(target_value_strategy, target_spec_strategy, st.none()),
    args=st.none(),
)

action_with_fallback_strategy = st.builds(
    Action,
    id=action_id_strategy,
    type=action_type_strategy,
    target=target_with_fallback_strategy,
    args=st.none(),
)

# endregion


# region Property Tests

@settings(max_examples=100)
@given(simple_action_strategy)
def test_action_node_is_workgraph_node(action: Action):
    """
    Property 2: ActionNode is WorkGraphNode.
    
    **Feature: action-flow-workflow-refactor, Property 2: ActionNode is WorkGraphNode**
    **Validates: Requirements 2.1**
    
    For any Action converted to ActionNode, the ActionNode SHALL be an instance of WorkGraphNode.
    """
    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    assert isinstance(node, WorkGraphNode), \
        f"ActionNode should be instance of WorkGraphNode, got {type(node)}"


@settings(max_examples=100)
@given(simple_action_strategy)
def test_action_node_attribute_preservation(action: Action):
    """
    Property 3: ActionNode attribute preservation.
    
    **Feature: action-flow-workflow-refactor, Property 3: ActionNode attribute preservation**
    **Validates: Requirements 2.2**
    
    For any ActionNode created with action A, executor E, and metadata M,
    the node's action, action_executor, and action_metadata attributes
    SHALL equal (A, E, M) respectively.
    """
    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    assert node.action is action, "ActionNode.action should be the same Action instance"
    assert node.action_executor is mock_executor, "ActionNode.action_executor should be the same executor"
    assert node.action_metadata is registry, "ActionNode.action_metadata should be the same registry"


@settings(max_examples=100)
@given(simple_action_strategy)
def test_action_node_execution_returns_action_result(action: Action):
    """
    Property 4: ActionNode execution returns ActionResult.
    
    **Feature: action-flow-workflow-refactor, Property 4: ActionNode execution returns ActionResult**
    **Validates: Requirements 2.3, 2.4**
    
    For any ActionNode that completes execution successfully,
    the return value SHALL be an ActionResult with success=True and the appropriate value.
    """
    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    context = ExecutionRuntime()
    result = node.run(context)
    
    assert isinstance(result, ActionResult), \
        f"ActionNode.run() should return ActionResult, got {type(result)}"
    assert result.success is True, "ActionResult.success should be True for successful execution"


@settings(max_examples=100)
@given(st.lists(target_spec_strategy, min_size=1, max_size=5))
def test_fallback_retry_count_matches_strategies(strategies):
    """
    Property 5: Fallback retry count matches strategies.
    
    **Feature: action-flow-workflow-refactor, Property 5: Fallback retry count matches strategies**
    **Validates: Requirements 3.1**
    
    For any Action with TargetSpecWithFallback containing N strategies,
    the ActionNode's max_repeat SHALL equal N.
    """
    target = TargetSpecWithFallback(strategies=strategies)
    action = Action(id='test_action', type='click', target=target)
    registry = ActionMetadataRegistry()
    
    node = ActionNode(
        action=action,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    assert node.max_repeat == len(strategies), \
        f"max_repeat should equal number of strategies ({len(strategies)}), got {node.max_repeat}"


@settings(max_examples=50)
@given(st.integers(min_value=1, max_value=5))
def test_fallback_stops_on_success(success_index: int):
    """
    Property 8: Fallback stops on success.
    
    **Feature: action-flow-workflow-refactor, Property 8: Fallback stops on success**
    **Validates: Requirements 3.4**
    
    For any Action with fallback strategies where strategy K succeeds,
    strategies K+1 through N SHALL NOT be executed.
    """
    # Create strategies where only the success_index-th one succeeds
    num_strategies = success_index + 2  # Ensure there are strategies after the successful one
    strategies = [
        TargetSpec(strategy='id', value=f'target_{i}')
        for i in range(num_strategies)
    ]
    
    target = TargetSpecWithFallback(strategies=strategies)
    action = Action(id='test_action', type='click', target=target)
    registry = ActionMetadataRegistry()
    
    # Track which strategies were attempted
    attempted_strategies = []
    
    def tracking_executor(
        action_type: str,
        action_target: Optional[str],
        action_args: Optional[Dict[str, Any]] = None,
        action_target_strategy: Optional[str] = None,
    ) -> str:
        attempted_strategies.append(action_target)
        # Fail until we reach the success_index
        if len(attempted_strategies) <= success_index:
            raise ValueError(f"Strategy {action_target} failed")
        return f"success_{action_target}"
    
    node = ActionNode(
        action=action,
        action_executor=tracking_executor,
        action_metadata=registry,
    )
    
    context = ExecutionRuntime()
    result = node.run(context)
    
    # Should have attempted exactly success_index + 1 strategies (0-indexed)
    assert len(attempted_strategies) == success_index + 1, \
        f"Should have attempted {success_index + 1} strategies, but attempted {len(attempted_strategies)}"
    
    # Result should be successful
    assert isinstance(result, ActionResult), "Result should be ActionResult"
    assert result.success is True, "Result should be successful"


# endregion


# region Serialization Property Tests

# **Feature: serializable-mixin, Property 7: ActionNode Configuration Preservation**
# **Validates: Requirements 4.1, 4.2, 4.3**
@settings(max_examples=100)
@given(simple_action_strategy)
def test_action_node_serialization_preserves_action(action: Action):
    """
    Property 7: ActionNode Configuration Preservation.
    
    **Feature: serializable-mixin, Property 7: ActionNode Configuration Preservation**
    **Validates: Requirements 4.1, 4.2, 4.3**
    
    For any ActionNode with action and retry configuration, serializing then
    deserializing SHALL preserve the action definition and retry settings.
    """
    registry = ActionMetadataRegistry()
    
    original = ActionNode(
        action=action,
        action_executor=mock_executor,
        action_metadata=registry,
        enable_result_save=True,
        result_save_dir="/tmp/test",
    )
    
    # Serialize
    serialized = original.to_serializable_obj()
    
    # Verify serialized structure
    assert '_type' in serialized, "Missing _type field"
    assert '_module' in serialized, "Missing _module field"
    assert 'action' in serialized, "Missing action field"
    assert 'config' in serialized, "Missing config field"
    
    # Deserialize
    restored = ActionNode.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    # Verify action is preserved
    assert restored.action.id == original.action.id, \
        f"Action ID mismatch: {restored.action.id} != {original.action.id}"
    assert restored.action.type == original.action.type, \
        f"Action type mismatch: {restored.action.type} != {original.action.type}"
    assert restored.action.target == original.action.target, \
        f"Action target mismatch: {restored.action.target} != {original.action.target}"


# **Feature: serializable-mixin, Property 7: ActionNode Configuration Preservation**
# **Validates: Requirements 4.1, 4.2, 4.3**
@settings(max_examples=100)
@given(simple_action_strategy)
def test_action_node_serialization_preserves_config(action: Action):
    """
    Property 7: ActionNode config preservation.
    
    **Feature: serializable-mixin, Property 7: ActionNode Configuration Preservation**
    **Validates: Requirements 4.1, 4.2, 4.3**
    
    For any ActionNode, serializing then deserializing SHALL preserve
    the configuration settings including enable_result_save and result_save_dir.
    """
    registry = ActionMetadataRegistry()
    
    original = ActionNode(
        action=action,
        action_executor=mock_executor,
        action_metadata=registry,
        enable_result_save=True,
        result_save_dir="/tmp/custom_dir",
    )
    
    # Serialize and deserialize
    serialized = original.to_serializable_obj()
    restored = ActionNode.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    # Verify config is preserved
    assert restored.enable_result_save == original.enable_result_save, \
        f"enable_result_save mismatch: {restored.enable_result_save} != {original.enable_result_save}"
    assert restored.result_save_dir == original.result_save_dir, \
        f"result_save_dir mismatch: {restored.result_save_dir} != {original.result_save_dir}"


# **Feature: serializable-mixin, Property 7: ActionNode Configuration Preservation**
# **Validates: Requirements 4.1, 4.2, 4.3**
@settings(max_examples=50)
@given(action_with_fallback_strategy)
def test_action_node_serialization_preserves_fallback_strategies(action: Action):
    """
    Property 7: ActionNode fallback strategy preservation.
    
    **Feature: serializable-mixin, Property 7: ActionNode Configuration Preservation**
    **Validates: Requirements 4.1, 4.2, 4.3**
    
    For any ActionNode with fallback strategies, serializing then deserializing
    SHALL preserve the fallback strategy configuration.
    """
    registry = ActionMetadataRegistry()
    
    original = ActionNode(
        action=action,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    # Serialize and deserialize
    serialized = original.to_serializable_obj()
    restored = ActionNode.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    # Verify fallback strategies are preserved
    assert isinstance(restored.action.target, TargetSpecWithFallback), \
        "Restored target should be TargetSpecWithFallback"
    
    original_strategies = original.action.target.strategies
    restored_strategies = restored.action.target.strategies
    
    assert len(restored_strategies) == len(original_strategies), \
        f"Strategy count mismatch: {len(restored_strategies)} != {len(original_strategies)}"
    
    for orig_spec, rest_spec in zip(original_strategies, restored_strategies):
        assert rest_spec.strategy == orig_spec.strategy, \
            f"Strategy mismatch: {rest_spec.strategy} != {orig_spec.strategy}"
        assert rest_spec.value == orig_spec.value, \
            f"Value mismatch: {rest_spec.value} != {orig_spec.value}"


def test_action_node_serialization_requires_action_executor():
    """
    Test that from_serializable_obj raises ValueError without action_executor.
    
    **Feature: serializable-mixin**
    **Validates: Context injection requirement**
    """
    action = Action(id='test_action', type='click', target='#btn')
    registry = ActionMetadataRegistry()
    
    original = ActionNode(
        action=action,
        action_executor=mock_executor,
        action_metadata=registry,
    )
    
    serialized = original.to_serializable_obj()
    
    # Should raise ValueError without action_executor
    try:
        ActionNode.from_serializable_obj(serialized)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "action_executor" in str(e), \
            f"Error message should mention action_executor: {e}"


# endregion


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running property tests for ActionNode...")
    
    test_action_node_is_workgraph_node()
    print("✓ test_action_node_is_workgraph_node passed")
    
    test_action_node_attribute_preservation()
    print("✓ test_action_node_attribute_preservation passed")
    
    test_action_node_execution_returns_action_result()
    print("✓ test_action_node_execution_returns_action_result passed")
    
    test_fallback_retry_count_matches_strategies()
    print("✓ test_fallback_retry_count_matches_strategies passed")
    
    test_fallback_stops_on_success()
    print("✓ test_fallback_stops_on_success passed")
    
    # Serialization property tests
    print("\nRunning serialization property tests...")
    
    test_action_node_serialization_preserves_action()
    print("✓ test_action_node_serialization_preserves_action passed")
    
    test_action_node_serialization_preserves_config()
    print("✓ test_action_node_serialization_preserves_config passed")
    
    test_action_node_serialization_preserves_fallback_strategies()
    print("✓ test_action_node_serialization_preserves_fallback_strategies passed")
    
    test_action_node_serialization_requires_action_executor()
    print("✓ test_action_node_serialization_requires_action_executor passed")
    
    print("\nAll property tests passed!")

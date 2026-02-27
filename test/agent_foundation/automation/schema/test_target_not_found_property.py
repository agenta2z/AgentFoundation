"""
Property-Based Tests for Action Model with Target Not Found Branch Fields

This module contains property-based tests using Hypothesis to verify
Action serialization round-trip with target_not_found_actions and
target_not_found_config fields.

**Feature: action-graph-target-not-found**
**Validates: Requirements 4.5**
"""
import sys
import json
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

from hypothesis import given, strategies as st, settings
import pytest

from agent_foundation.automation.schema.common import (
    Action,
    TargetSpec,
    TargetSpecWithFallback,
    TargetStrategy,
)


# =============================================================================
# Hypothesis Strategies for Action Model
# =============================================================================

# Strategy for generating valid action IDs
action_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip() and len(x.strip()) > 0)

# Strategy for generating valid action types
action_type_strategy = st.sampled_from([
    "click", "input_text", "scroll", "hover", "double_click",
    "visit_url", "wait", "select", "drag_and_drop"
])

# Strategy for generating valid TargetStrategy values
target_strategy_strategy = st.sampled_from(list(TargetStrategy))

# Strategy for generating valid target values
target_value_strategy = st.text(min_size=1, max_size=50).filter(lambda x: x.strip())

# Strategy for generating TargetSpec objects
target_spec_strategy = st.builds(
    TargetSpec,
    strategy=target_strategy_strategy,
    value=target_value_strategy,
    description=st.one_of(st.none(), st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
    options=st.one_of(st.none(), st.lists(st.text(min_size=1, max_size=20).filter(lambda x: x.strip()), max_size=3))
)

# Strategy for generating TargetSpecWithFallback objects
target_spec_with_fallback_strategy = st.builds(
    TargetSpecWithFallback,
    strategies=st.lists(target_spec_strategy, min_size=1, max_size=3)
)

# Strategy for generating target (string, TargetSpec, or TargetSpecWithFallback)
target_strategy = st.one_of(
    target_value_strategy,  # Simple string target
    target_spec_strategy,   # TargetSpec
    target_spec_with_fallback_strategy  # TargetSpecWithFallback
)

# Strategy for generating action args
action_args_strategy = st.one_of(
    st.none(),
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isidentifier()),
        values=st.one_of(
            st.text(max_size=50),
            st.integers(min_value=-1000, max_value=1000),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            st.booleans(),
        ),
        max_size=5
    )
)

# Strategy for generating target_not_found_config
target_not_found_config_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries({
        'retry_after_handling': st.booleans(),
        'max_retries': st.integers(min_value=0, max_value=10),
        'retry_delay': st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)
    })
)


# Recursive strategy for generating Action objects with nested target_not_found_actions
@st.composite
def action_strategy(draw, max_depth=2, current_depth=0):
    """Generate valid Action instances with optional nested target_not_found_actions.
    
    Args:
        draw: Hypothesis draw function
        max_depth: Maximum nesting depth for target_not_found_actions
        current_depth: Current nesting depth
    
    Returns:
        A valid Action instance
    """
    action_id = draw(action_id_strategy)
    action_type = draw(action_type_strategy)
    target = draw(st.one_of(st.none(), target_strategy))
    args = draw(action_args_strategy)
    no_action_if_target_not_found = draw(st.booleans())
    
    # Generate target_not_found_actions only if we haven't reached max depth
    # and the action has a target
    if current_depth < max_depth and target is not None:
        # Decide whether to include target_not_found_actions
        include_branch = draw(st.booleans())
        if include_branch:
            # Generate 0-3 nested actions
            num_nested = draw(st.integers(min_value=0, max_value=3))
            nested_actions = [
                draw(action_strategy(max_depth=max_depth, current_depth=current_depth + 1))
                for _ in range(num_nested)
            ]
            target_not_found_actions = nested_actions if nested_actions else None
            target_not_found_config = draw(target_not_found_config_strategy)
        else:
            target_not_found_actions = None
            target_not_found_config = None
    else:
        target_not_found_actions = None
        target_not_found_config = None
    
    return Action(
        id=action_id,
        type=action_type,
        target=target,
        args=args,
        no_action_if_target_not_found=no_action_if_target_not_found,
        target_not_found_actions=target_not_found_actions,
        target_not_found_config=target_not_found_config
    )


# Strategy for generating Action with guaranteed target_not_found_actions
@st.composite
def action_with_branch_strategy(draw, max_depth=2):
    """Generate Action instances that always have target_not_found_actions.
    
    This strategy ensures the action has a target and at least one nested action
    in target_not_found_actions.
    """
    action_id = draw(action_id_strategy)
    action_type = draw(action_type_strategy)
    target = draw(target_strategy)  # Always has a target
    args = draw(action_args_strategy)
    
    # Generate 1-3 nested actions
    num_nested = draw(st.integers(min_value=1, max_value=3))
    nested_actions = [
        draw(action_strategy(max_depth=max_depth, current_depth=1))
        for _ in range(num_nested)
    ]
    
    target_not_found_config = draw(target_not_found_config_strategy)
    
    return Action(
        id=action_id,
        type=action_type,
        target=target,
        args=args,
        target_not_found_actions=nested_actions,
        target_not_found_config=target_not_found_config
    )


# =============================================================================
# Helper Functions
# =============================================================================

def actions_are_equivalent(action1: Action, action2: Action) -> bool:
    """Check if two Action objects are equivalent.
    
    Compares all fields including nested target_not_found_actions.
    """
    # Compare basic fields
    if action1.id != action2.id:
        return False
    if action1.type != action2.type:
        return False
    if action1.args != action2.args:
        return False
    if action1.no_action_if_target_not_found != action2.no_action_if_target_not_found:
        return False
    
    # Compare targets
    if not targets_are_equivalent(action1.target, action2.target):
        return False
    
    # Compare target_not_found_config
    if action1.target_not_found_config != action2.target_not_found_config:
        return False
    
    # Compare target_not_found_actions
    if action1.target_not_found_actions is None and action2.target_not_found_actions is None:
        return True
    if action1.target_not_found_actions is None or action2.target_not_found_actions is None:
        return False
    if len(action1.target_not_found_actions) != len(action2.target_not_found_actions):
        return False
    
    for a1, a2 in zip(action1.target_not_found_actions, action2.target_not_found_actions):
        if not actions_are_equivalent(a1, a2):
            return False
    
    return True


def targets_are_equivalent(target1, target2) -> bool:
    """Check if two targets are equivalent."""
    if target1 is None and target2 is None:
        return True
    if target1 is None or target2 is None:
        return False
    
    # Both are strings
    if isinstance(target1, str) and isinstance(target2, str):
        return target1 == target2
    
    # Both are TargetSpec
    if isinstance(target1, TargetSpec) and isinstance(target2, TargetSpec):
        return (
            target1.strategy == target2.strategy and
            target1.value == target2.value and
            target1.description == target2.description and
            target1.options == target2.options
        )
    
    # Both are TargetSpecWithFallback
    if isinstance(target1, TargetSpecWithFallback) and isinstance(target2, TargetSpecWithFallback):
        if len(target1.strategies) != len(target2.strategies):
            return False
        for s1, s2 in zip(target1.strategies, target2.strategies):
            if not targets_are_equivalent(s1, s2):
                return False
        return True
    
    return False


# =============================================================================
# Task 12.1: Property 1 - Action serialization round-trip with branch fields
# =============================================================================

class TestActionSerializationRoundTrip:
    """Property-based tests for Action serialization round-trip.
    
    **Validates: Requirements 4.5**
    
    These tests verify that for any valid Action with target_not_found_actions
    and target_not_found_config:
    1. Serializing to JSON and deserializing back produces an equivalent Action
    2. All fields are preserved including nested actions
    3. The round-trip is idempotent (serialize -> deserialize -> serialize produces same JSON)
    """

    @settings(max_examples=100)
    @given(action=action_strategy())
    def test_dict_round_trip_preserves_all_fields(self, action: Action):
        """Property 1: For any valid Action, serializing to dict and deserializing
        back should produce an equivalent Action with all fields preserved.
        
        **Validates: Requirements 4.5**
        """
        # Serialize to dict
        action_dict = action.dict()
        
        # Deserialize back
        restored = Action(**action_dict)
        
        # Verify equivalence
        assert actions_are_equivalent(action, restored), \
            f"Round-trip failed: original={action.dict()}, restored={restored.dict()}"

    @settings(max_examples=100)
    @given(action=action_strategy())
    def test_json_round_trip_preserves_all_fields(self, action: Action):
        """Property 1: For any valid Action, serializing to JSON string and
        deserializing back should produce an equivalent Action.
        
        **Validates: Requirements 4.5**
        """
        # Serialize to JSON string
        json_str = action.json()
        
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict), "JSON string should parse to a dict"
        
        # Deserialize back
        restored = Action.parse_raw(json_str)
        
        # Verify equivalence
        assert actions_are_equivalent(action, restored), \
            f"JSON round-trip failed"

    @settings(max_examples=100)
    @given(action=action_with_branch_strategy())
    def test_round_trip_with_guaranteed_branch(self, action: Action):
        """Property 1: For any Action with target_not_found_actions, the round-trip
        should preserve all nested actions and config.
        
        **Validates: Requirements 4.5**
        """
        # Verify the action has branch (precondition)
        assert action.target_not_found_actions is not None
        assert len(action.target_not_found_actions) > 0
        
        # Serialize and deserialize
        json_str = action.json()
        restored = Action.parse_raw(json_str)
        
        # Verify branch is preserved
        assert restored.target_not_found_actions is not None
        assert len(restored.target_not_found_actions) == len(action.target_not_found_actions)
        
        # Verify config is preserved
        assert restored.target_not_found_config == action.target_not_found_config
        
        # Verify full equivalence
        assert actions_are_equivalent(action, restored)

    @settings(max_examples=100)
    @given(action=action_strategy())
    def test_round_trip_is_idempotent(self, action: Action):
        """Property 1: The round-trip should be idempotent - serialize -> deserialize
        -> serialize should produce the same JSON.
        
        **Validates: Requirements 4.5**
        """
        # First round-trip
        json_str1 = action.json()
        restored1 = Action.parse_raw(json_str1)
        
        # Second round-trip
        json_str2 = restored1.json()
        restored2 = Action.parse_raw(json_str2)
        
        # The JSON strings should be identical
        # (comparing parsed dicts to avoid formatting differences)
        dict1 = json.loads(json_str1)
        dict2 = json.loads(json_str2)
        assert dict1 == dict2, "Round-trip is not idempotent"
        
        # The restored actions should be equivalent
        assert actions_are_equivalent(restored1, restored2)

    @settings(max_examples=50)
    @given(action=action_with_branch_strategy(max_depth=3))
    def test_deeply_nested_actions_preserved(self, action: Action):
        """Property 1: Deeply nested target_not_found_actions should be preserved
        through serialization round-trip.
        
        **Validates: Requirements 4.5**
        """
        # Serialize and deserialize
        json_str = action.json()
        restored = Action.parse_raw(json_str)
        
        # Count total nested actions in original
        def count_nested(a: Action) -> int:
            if a.target_not_found_actions is None:
                return 0
            return len(a.target_not_found_actions) + sum(
                count_nested(nested) for nested in a.target_not_found_actions
            )
        
        original_count = count_nested(action)
        restored_count = count_nested(restored)
        
        assert original_count == restored_count, \
            f"Nested action count mismatch: {original_count} vs {restored_count}"
        
        # Full equivalence check
        assert actions_are_equivalent(action, restored)

    @settings(max_examples=100)
    @given(action=action_strategy())
    def test_target_not_found_config_values_preserved(self, action: Action):
        """Property 1: target_not_found_config values should be exactly preserved
        through serialization.
        
        **Validates: Requirements 4.5**
        """
        if action.target_not_found_config is None:
            # Skip if no config
            return
        
        # Serialize and deserialize
        json_str = action.json()
        restored = Action.parse_raw(json_str)
        
        # Verify config is preserved exactly
        assert restored.target_not_found_config is not None
        
        original_config = action.target_not_found_config
        restored_config = restored.target_not_found_config
        
        assert restored_config.get('retry_after_handling') == original_config.get('retry_after_handling')
        assert restored_config.get('max_retries') == original_config.get('max_retries')
        
        # For floats, use approximate comparison due to JSON serialization
        original_delay = original_config.get('retry_delay')
        restored_delay = restored_config.get('retry_delay')
        if original_delay is not None and restored_delay is not None:
            assert abs(original_delay - restored_delay) < 1e-10, \
                f"retry_delay mismatch: {original_delay} vs {restored_delay}"


# =============================================================================
# Main entry point for running tests directly
# =============================================================================

if __name__ == '__main__':
    print("Running property-based tests for Action serialization round-trip...")
    print()
    
    test_instance = TestActionSerializationRoundTrip()
    
    tests = [
        ("Property 1: Dict round-trip preserves all fields",
         test_instance.test_dict_round_trip_preserves_all_fields),
        ("Property 1: JSON round-trip preserves all fields",
         test_instance.test_json_round_trip_preserves_all_fields),
        ("Property 1: Round-trip with guaranteed branch",
         test_instance.test_round_trip_with_guaranteed_branch),
        ("Property 1: Round-trip is idempotent",
         test_instance.test_round_trip_is_idempotent),
        ("Property 1: Deeply nested actions preserved",
         test_instance.test_deeply_nested_actions_preserved),
        ("Property 1: target_not_found_config values preserved",
         test_instance.test_target_not_found_config_values_preserved),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}")
            print(f"  Error: {e}")
            failed += 1
    
    print()
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll property-based tests passed! ✓")


# =============================================================================
# Additional imports for Task 13: ActionGraph Builder Property Tests
# =============================================================================

from agent_foundation.automation.schema.action_graph import (
    ActionGraph,
    ActionChainHelper,
    TargetNotFoundContext,
)
from agent_foundation.automation.schema.common import (
    BranchAlreadyExistsError,
)


# =============================================================================
# Additional Hypothesis Strategies for ActionGraph Builder Tests
# =============================================================================

# Strategy for generating valid max_retries values (0-10)
valid_max_retries_strategy = st.integers(min_value=0, max_value=10)

# Strategy for generating invalid max_retries values (outside 0-10)
invalid_max_retries_strategy = st.one_of(
    st.integers(max_value=-1),  # Negative values
    st.integers(min_value=11)   # Values > 10
)

# Strategy for generating valid retry_delay values (0-60)
valid_retry_delay_strategy = st.floats(
    min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False
)

# Strategy for generating invalid retry_delay values (negative or > 60)
invalid_retry_delay_strategy = st.one_of(
    st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),  # Negative
    st.floats(min_value=60.001, max_value=1000.0, allow_nan=False, allow_infinity=False)  # > 60
)

# Strategy for generating number of actions to add inside context (1-10)
num_actions_strategy = st.integers(min_value=1, max_value=10)


# =============================================================================
# Task 13.1: Property 2 - Action placement - actions inside context go to branch list
# =============================================================================

class TestActionPlacementProperty:
    """Property-based tests for action placement in target_not_found context.
    
    **Validates: Requirements 2.3, 2.4, 11.1, 11.2**
    
    These tests verify that for any number of actions added inside a
    target_not_found() context:
    1. All actions go to the parent action's target_not_found_actions list
    2. Actions are NOT added to the current ActionSequenceNode
    3. The order of actions is preserved
    """

    @settings(max_examples=100)
    @given(
        num_actions=num_actions_strategy,
        retry_after_handling=st.booleans(),
        max_retries=valid_max_retries_strategy,
        retry_delay=valid_retry_delay_strategy
    )
    def test_actions_inside_context_go_to_branch_list(
        self, num_actions: int, retry_after_handling: bool, max_retries: int, retry_delay: float
    ):
        """Property 2: For any number of actions added inside a target_not_found()
        context, all actions should be added to the parent action's
        target_not_found_actions list.
        
        **Validates: Requirements 2.3, 2.4, 11.1, 11.2**
        """
        # Create a mock executor
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Create parent action with target
        parent_target = TargetSpec(strategy=TargetStrategy.ID, value="parent-btn")
        parent_helper = graph.action("click", target=parent_target)
        parent_action = parent_helper.action_obj
        
        # Add actions inside target_not_found context
        with parent_helper.target_not_found(
            retry_after_handling=retry_after_handling,
            max_retries=max_retries,
            retry_delay=retry_delay
        ):
            for i in range(num_actions):
                target = TargetSpec(strategy=TargetStrategy.ID, value=f"fallback-btn-{i}")
                graph.action("click", target=target)
        
        # Verify all actions are in the branch list
        assert parent_action.target_not_found_actions is not None
        assert len(parent_action.target_not_found_actions) == num_actions
        
        # Verify action order is preserved
        for i, action in enumerate(parent_action.target_not_found_actions):
            assert action.target.value == f"fallback-btn-{i}"


    @settings(max_examples=100)
    @given(
        num_branch_actions=num_actions_strategy,
        num_main_actions_before=st.integers(min_value=0, max_value=5),
        num_main_actions_after=st.integers(min_value=0, max_value=5)
    )
    def test_actions_not_added_to_main_sequence(
        self, num_branch_actions: int, num_main_actions_before: int, num_main_actions_after: int
    ):
        """Property 2: Actions inside target_not_found() context should NOT be
        added to the current ActionSequenceNode.
        
        **Validates: Requirements 2.3, 2.4, 11.1, 11.2**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Add actions before the context
        for i in range(num_main_actions_before):
            target = TargetSpec(strategy=TargetStrategy.ID, value=f"main-before-{i}")
            graph.action("click", target=target)
        
        # Create parent action with target_not_found context
        parent_target = TargetSpec(strategy=TargetStrategy.ID, value="parent-btn")
        parent_helper = graph.action("click", target=parent_target)
        
        with parent_helper.target_not_found():
            for i in range(num_branch_actions):
                target = TargetSpec(strategy=TargetStrategy.ID, value=f"branch-{i}")
                graph.action("click", target=target)
        
        # Add actions after the context
        for i in range(num_main_actions_after):
            target = TargetSpec(strategy=TargetStrategy.ID, value=f"main-after-{i}")
            graph.action("click", target=target)
        
        # Get the main sequence actions from the current node (private attribute _actions)
        main_actions = graph._current_node._actions
        
        # Expected count: before + parent + after (branch actions should NOT be in main)
        expected_main_count = num_main_actions_before + 1 + num_main_actions_after
        assert len(main_actions) == expected_main_count
        
        # Verify branch actions are NOT in main sequence
        main_target_values = [a.target.value for a in main_actions if a.target]
        for i in range(num_branch_actions):
            assert f"branch-{i}" not in main_target_values


# =============================================================================
# Task 13.2: Property 3 - Context manager lifecycle and stack integrity
# =============================================================================

class TestContextManagerLifecycleProperty:
    """Property-based tests for context manager lifecycle and stack integrity.
    
    **Validates: Requirements 2.3, 2.5, 7.1, 7.2, 7.3, 7.4**
    
    These tests verify that for any sequence of context enter/exit operations:
    1. The stack is properly managed (push on enter, pop on exit)
    2. Context is restored even if an exception occurs
    3. Partial state is cleaned up on exception
    """

    @settings(max_examples=100)
    @given(num_actions=num_actions_strategy)
    def test_stack_restored_after_normal_exit(self, num_actions: int):
        """Property 3: After exiting a target_not_found() context normally,
        the _action_branch_stack should be restored to its previous state.
        
        **Validates: Requirements 2.3, 2.5, 7.1, 7.2, 7.3, 7.4**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Verify stack is empty initially
        assert len(graph._action_branch_stack) == 0
        
        # Create parent action
        parent_target = TargetSpec(strategy=TargetStrategy.ID, value="parent-btn")
        parent_helper = graph.action("click", target=parent_target)
        
        # Enter and exit context
        with parent_helper.target_not_found():
            # Stack should have one entry
            assert len(graph._action_branch_stack) == 1
            
            # Add some actions
            for i in range(num_actions):
                target = TargetSpec(strategy=TargetStrategy.ID, value=f"branch-{i}")
                graph.action("click", target=target)
        
        # Stack should be empty after exit
        assert len(graph._action_branch_stack) == 0

    @settings(max_examples=50)
    @given(num_actions=num_actions_strategy)
    def test_stack_restored_after_exception(self, num_actions: int):
        """Property 3: After an exception inside a target_not_found() context,
        the _action_branch_stack should be restored to its previous state.
        
        **Validates: Requirements 2.3, 2.5, 7.1, 7.2, 7.3, 7.4**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Verify stack is empty initially
        assert len(graph._action_branch_stack) == 0
        
        # Create parent action
        parent_target = TargetSpec(strategy=TargetStrategy.ID, value="parent-btn")
        parent_helper = graph.action("click", target=parent_target)
        parent_action = parent_helper.action_obj
        
        # Enter context and raise exception
        try:
            with parent_helper.target_not_found():
                # Stack should have one entry
                assert len(graph._action_branch_stack) == 1
                
                # Add some actions before exception
                for i in range(num_actions):
                    target = TargetSpec(strategy=TargetStrategy.ID, value=f"branch-{i}")
                    graph.action("click", target=target)
                
                # Raise exception
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Stack should be empty after exception
        assert len(graph._action_branch_stack) == 0
        
        # Partial state should be cleaned up
        assert parent_action.target_not_found_actions is None
        assert parent_action.target_not_found_config is None


    @settings(max_examples=50)
    @given(
        num_outer_actions=st.integers(min_value=1, max_value=5),
        num_inner_actions=st.integers(min_value=1, max_value=5)
    )
    def test_nested_contexts_stack_integrity(
        self, num_outer_actions: int, num_inner_actions: int
    ):
        """Property 3: For nested target_not_found() contexts, the stack should
        be properly managed with correct push/pop order.
        
        **Validates: Requirements 2.3, 2.5, 7.1, 7.2, 7.3, 7.4**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Verify stack is empty initially
        assert len(graph._action_branch_stack) == 0
        
        # Create outer parent action
        outer_target = TargetSpec(strategy=TargetStrategy.ID, value="outer-btn")
        outer_helper = graph.action("click", target=outer_target)
        outer_action = outer_helper.action_obj
        
        with outer_helper.target_not_found():
            # Stack should have one entry
            assert len(graph._action_branch_stack) == 1
            
            # Add outer branch actions
            for i in range(num_outer_actions):
                target = TargetSpec(strategy=TargetStrategy.ID, value=f"outer-branch-{i}")
                inner_helper = graph.action("click", target=target)
                
                # Create nested context for first action only
                if i == 0:
                    inner_action = inner_helper.action_obj
                    with inner_helper.target_not_found():
                        # Stack should have two entries
                        assert len(graph._action_branch_stack) == 2
                        
                        # Add inner branch actions
                        for j in range(num_inner_actions):
                            inner_target = TargetSpec(
                                strategy=TargetStrategy.ID, value=f"inner-branch-{j}"
                            )
                            graph.action("click", target=inner_target)
                    
                    # After inner context exit, stack should have one entry
                    assert len(graph._action_branch_stack) == 1
                    
                    # Inner action should have its branch actions
                    assert inner_action.target_not_found_actions is not None
                    assert len(inner_action.target_not_found_actions) == num_inner_actions
        
        # Stack should be empty after all contexts exit
        assert len(graph._action_branch_stack) == 0
        
        # Outer action should have its branch actions
        assert outer_action.target_not_found_actions is not None
        assert len(outer_action.target_not_found_actions) == num_outer_actions


# =============================================================================
# Task 13.3: Property 4 - Branch uniqueness enforcement
# =============================================================================

class TestBranchUniquenessProperty:
    """Property-based tests for branch uniqueness enforcement.
    
    **Validates: Requirements 3.5**
    
    These tests verify that calling target_not_found() twice on the same
    action always raises BranchAlreadyExistsError.
    """

    @settings(max_examples=100)
    @given(
        retry_after_handling_1=st.booleans(),
        max_retries_1=valid_max_retries_strategy,
        retry_delay_1=valid_retry_delay_strategy,
        retry_after_handling_2=st.booleans(),
        max_retries_2=valid_max_retries_strategy,
        retry_delay_2=valid_retry_delay_strategy
    )
    def test_duplicate_branch_raises_error(
        self,
        retry_after_handling_1: bool, max_retries_1: int, retry_delay_1: float,
        retry_after_handling_2: bool, max_retries_2: int, retry_delay_2: float
    ):
        """Property 4: Calling target_not_found() twice on the same action
        should always raise BranchAlreadyExistsError, regardless of parameters.
        
        **Validates: Requirements 3.5**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Create action with target
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        helper = graph.action("click", target=target)
        
        # First call should succeed
        with helper.target_not_found(
            retry_after_handling=retry_after_handling_1,
            max_retries=max_retries_1,
            retry_delay=retry_delay_1
        ):
            pass
        
        # Second call should raise BranchAlreadyExistsError
        with pytest.raises(BranchAlreadyExistsError) as exc_info:
            helper.target_not_found(
                retry_after_handling=retry_after_handling_2,
                max_retries=max_retries_2,
                retry_delay=retry_delay_2
            )
        
        # Verify error attributes
        assert exc_info.value.condition == "target_not_found"
        assert exc_info.value.action_type == "click"

    @settings(max_examples=50)
    @given(action_type=action_type_strategy)
    def test_duplicate_branch_error_includes_action_type(self, action_type: str):
        """Property 4: BranchAlreadyExistsError should include the action type
        in its attributes.
        
        **Validates: Requirements 3.5**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Create action with target
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        helper = graph.action(action_type, target=target)
        
        # First call should succeed
        with helper.target_not_found():
            pass
        
        # Second call should raise BranchAlreadyExistsError with correct action_type
        with pytest.raises(BranchAlreadyExistsError) as exc_info:
            helper.target_not_found()
        
        assert exc_info.value.action_type == action_type


# =============================================================================
# Task 13.4: Property 5 - Parameter validation rejects out-of-range values
# =============================================================================

class TestParameterValidationProperty:
    """Property-based tests for parameter validation.
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    
    These tests verify that invalid parameter values always raise ValueError:
    1. max_retries outside 0-10
    2. retry_delay outside 0-60
    3. target=None
    """

    @settings(max_examples=100)
    @given(invalid_max_retries=invalid_max_retries_strategy)
    def test_invalid_max_retries_raises_value_error(self, invalid_max_retries: int):
        """Property 5: max_retries outside range 0-10 should always raise ValueError.
        
        **Validates: Requirements 3.1**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Create action with target
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        helper = graph.action("click", target=target)
        
        # Invalid max_retries should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found(max_retries=invalid_max_retries)
        
        # Error message should include the invalid value
        assert str(invalid_max_retries) in str(exc_info.value)
        assert "max_retries" in str(exc_info.value)

    @settings(max_examples=100)
    @given(invalid_retry_delay=invalid_retry_delay_strategy)
    def test_invalid_retry_delay_raises_value_error(self, invalid_retry_delay: float):
        """Property 5: retry_delay outside range 0-60 should always raise ValueError.
        
        **Validates: Requirements 3.2, 3.3**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Create action with target
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        helper = graph.action("click", target=target)
        
        # Invalid retry_delay should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found(retry_delay=invalid_retry_delay)
        
        # Error message should include the invalid value
        assert "retry_delay" in str(exc_info.value)

    @settings(max_examples=50)
    @given(action_type=action_type_strategy)
    def test_target_none_raises_value_error(self, action_type: str):
        """Property 5: Calling target_not_found() on an action with target=None
        should always raise ValueError.
        
        **Validates: Requirements 3.4**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Create action without target
        helper = graph.action(action_type, target=None)
        
        # target_not_found() should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found()
        
        # Error message should mention the action type and target
        assert action_type in str(exc_info.value)
        assert "target" in str(exc_info.value).lower()


    @settings(max_examples=100)
    @given(
        valid_max_retries=valid_max_retries_strategy,
        valid_retry_delay=valid_retry_delay_strategy,
        retry_after_handling=st.booleans()
    )
    def test_valid_parameters_accepted(
        self, valid_max_retries: int, valid_retry_delay: float, retry_after_handling: bool
    ):
        """Property 5: Valid parameter values should be accepted without error.
        
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        # Create action with target
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        helper = graph.action("click", target=target)
        
        # Valid parameters should not raise
        with helper.target_not_found(
            retry_after_handling=retry_after_handling,
            max_retries=valid_max_retries,
            retry_delay=valid_retry_delay
        ):
            pass
        
        # Verify config was stored correctly
        action = helper.action_obj
        assert action.target_not_found_config is not None
        assert action.target_not_found_config['retry_after_handling'] == retry_after_handling
        assert action.target_not_found_config['max_retries'] == valid_max_retries
        assert action.target_not_found_config['retry_delay'] == valid_retry_delay

    @settings(max_examples=50)
    @given(
        invalid_max_retries=invalid_max_retries_strategy,
        valid_retry_delay=valid_retry_delay_strategy
    )
    def test_invalid_max_retries_with_valid_delay(
        self, invalid_max_retries: int, valid_retry_delay: float
    ):
        """Property 5: Invalid max_retries should raise ValueError even with valid retry_delay.
        
        **Validates: Requirements 3.1**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        helper = graph.action("click", target=target)
        
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found(
                max_retries=invalid_max_retries,
                retry_delay=valid_retry_delay
            )
        
        assert "max_retries" in str(exc_info.value)

    @settings(max_examples=50)
    @given(
        valid_max_retries=valid_max_retries_strategy,
        invalid_retry_delay=invalid_retry_delay_strategy
    )
    def test_invalid_retry_delay_with_valid_max_retries(
        self, valid_max_retries: int, invalid_retry_delay: float
    ):
        """Property 5: Invalid retry_delay should raise ValueError even with valid max_retries.
        
        **Validates: Requirements 3.2, 3.3**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        helper = graph.action("click", target=target)
        
        with pytest.raises(ValueError) as exc_info:
            helper.target_not_found(
                max_retries=valid_max_retries,
                retry_delay=invalid_retry_delay
            )
        
        assert "retry_delay" in str(exc_info.value)


# =============================================================================
# Additional imports for Task 14: Execution Property Tests
# =============================================================================

from agent_foundation.automation.schema.action_node import ActionNode
from agent_foundation.automation.schema.action_metadata import ActionMetadataRegistry
from agent_foundation.automation.schema.common import (
    ExecutionRuntime,
    TargetNotFoundError,
)


# =============================================================================
# Custom Exception Classes for Testing
# =============================================================================

class ElementNotFoundError(Exception):
    """Custom exception for testing element-not-found detection."""
    pass


class ElementNotFoundException(Exception):
    """Alternative custom exception for testing element-not-found detection."""
    pass


class CustomElementNotFoundError(ElementNotFoundError):
    """Subclass of ElementNotFoundError for testing MRO detection."""
    pass


class SomeOtherNotFoundError(Exception):
    """Exception with 'NotFound' in name but not an exact match."""
    pass


# =============================================================================
# Additional Hypothesis Strategies for Execution Tests
# =============================================================================

# Strategy for generating number of branch actions (1-5)
num_branch_actions_strategy = st.integers(min_value=1, max_value=5)

# Strategy for generating number of execution attempts (1-10)
num_attempts_strategy = st.integers(min_value=1, max_value=10)


# =============================================================================
# Task 14.1: Property 6 - Branch execution on target not found
# =============================================================================

class TestBranchExecutionProperty:
    """Property-based tests for branch execution on target not found.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    
    These tests verify that for any action with target_not_found_actions:
    1. When executor raises ElementNotFoundError, branch actions execute
    2. All branch actions execute in order
    3. With retry_after_handling=False, execution returns success after branch
    4. With retry_after_handling=True, the parent action is retried
    """

    @settings(max_examples=50)
    @given(
        num_branch_actions=num_branch_actions_strategy,
        retry_after_handling=st.just(False),  # Test no-retry case
        max_retries=valid_max_retries_strategy,
    )
    def test_branch_executes_on_element_not_found_error(
        self, num_branch_actions: int, retry_after_handling: bool, max_retries: int
    ):
        """Property 6: When executor raises ElementNotFoundError and branch exists,
        all branch actions should execute in order.
        
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
        """
        execution_log = []
        
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            execution_log.append(target)
            if target == "main-btn":
                raise ElementNotFoundError("Element not found")
            return f"executed {target}"
        
        # Create branch actions
        branch_actions = [
            Action(
                id=f"branch_{i}",
                type="click",
                target=f"branch-btn-{i}"
            )
            for i in range(num_branch_actions)
        ]
        
        # Create main action with branch
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=branch_actions,
            target_not_found_config={
                "retry_after_handling": retry_after_handling,
                "max_retries": max_retries,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        # Verify branch executed
        assert result.success is True
        assert result.metadata.get("branch_executed") is True
        
        # Verify all branch actions executed in order
        expected_targets = ["main-btn"] + [f"branch-btn-{i}" for i in range(num_branch_actions)]
        assert execution_log == expected_targets

    @settings(max_examples=50)
    @given(
        num_branch_actions=num_branch_actions_strategy,
    )
    def test_branch_executes_on_element_not_found_exception(
        self, num_branch_actions: int
    ):
        """Property 6: When executor raises ElementNotFoundException (alternative name),
        branch actions should execute.
        
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
        """
        execution_log = []
        
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            execution_log.append(target)
            if target == "main-btn":
                raise ElementNotFoundException("Element not found")
            return f"executed {target}"
        
        # Create branch actions
        branch_actions = [
            Action(
                id=f"branch_{i}",
                type="click",
                target=f"branch-btn-{i}"
            )
            for i in range(num_branch_actions)
        ]
        
        # Create main action with branch
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=branch_actions,
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        # Verify branch executed
        assert result.success is True
        assert result.metadata.get("branch_executed") is True
        
        # Verify all branch actions executed
        assert len(execution_log) == num_branch_actions + 1


# =============================================================================
# Task 14.2: Property 7 - Retry behavior based on flag
# =============================================================================

class TestRetryBehaviorProperty:
    """Property-based tests for retry behavior based on flag.
    
    **Validates: Requirements 6.1, 6.2, 6.3**
    
    These tests verify that:
    1. retry_after_handling=True causes retries after branch execution
    2. retry_after_handling=False returns immediately after branch execution
    """

    @settings(max_examples=50)
    @given(
        max_retries=st.integers(min_value=1, max_value=5),
    )
    def test_retry_after_handling_true_causes_retries(
        self, max_retries: int
    ):
        """Property 7: When retry_after_handling=True, the parent action should
        be retried after branch execution.
        
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        main_call_count = 0
        branch_call_count = 0
        
        def mock_executor(**kwargs):
            nonlocal main_call_count, branch_call_count
            target = kwargs.get("action_target")
            if target == "main-btn":
                main_call_count += 1
                raise ElementNotFoundError("Element not found")
            branch_call_count += 1
            return "branch executed"
        
        # Create branch action
        branch_action = Action(
            id="branch_action",
            type="click",
            target="branch-btn"
        )
        
        # Create main action with retry
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=[branch_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": max_retries,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError):
            node.run(context)
        
        # Branch should execute max_retries + 1 times (1 initial + max_retries)
        assert branch_call_count == max_retries + 1

    @settings(max_examples=50)
    @given(
        max_retries=st.integers(min_value=1, max_value=5),
    )
    def test_retry_after_handling_false_returns_immediately(
        self, max_retries: int
    ):
        """Property 7: When retry_after_handling=False, execution should return
        immediately after branch execution without retrying.
        
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        main_call_count = 0
        branch_call_count = 0
        
        def mock_executor(**kwargs):
            nonlocal main_call_count, branch_call_count
            target = kwargs.get("action_target")
            if target == "main-btn":
                main_call_count += 1
                raise ElementNotFoundError("Element not found")
            branch_call_count += 1
            return "branch executed"
        
        # Create branch action
        branch_action = Action(
            id="branch_action",
            type="click",
            target="branch-btn"
        )
        
        # Create main action without retry
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=[branch_action],
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": max_retries,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        # Should succeed without retrying
        assert result.success is True
        assert result.metadata.get("branch_executed") is True
        
        # Main action called once, branch called once
        assert main_call_count == 1
        assert branch_call_count == 1


# =============================================================================
# Task 14.3: Property 8 - Max retries enforcement (N+1 attempts)
# =============================================================================

class TestMaxRetriesEnforcementProperty:
    """Property-based tests for max retries enforcement.
    
    **Validates: Requirements 6.2, 6.4, 6.5**
    
    These tests verify that:
    1. Total attempts = 1 initial + max_retries (N+1 total)
    2. TargetNotFoundError is raised with correct attempt_count and max_retries
    """

    @settings(max_examples=50)
    @given(
        max_retries=st.integers(min_value=0, max_value=10),
    )
    def test_total_attempts_equals_n_plus_one(
        self, max_retries: int
    ):
        """Property 8: Total attempts should equal 1 initial + max_retries (N+1).
        
        **Validates: Requirements 6.2, 6.4, 6.5**
        """
        branch_call_count = 0
        
        def mock_executor(**kwargs):
            nonlocal branch_call_count
            target = kwargs.get("action_target")
            if target == "main-btn":
                raise ElementNotFoundError("Element not found")
            branch_call_count += 1
            return "branch executed"
        
        # Create branch action
        branch_action = Action(
            id="branch_action",
            type="click",
            target="branch-btn"
        )
        
        # Create main action with retry
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=[branch_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": max_retries,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError) as exc_info:
            node.run(context)
        
        # Verify attempt count
        expected_attempts = max_retries + 1
        assert exc_info.value.attempt_count == expected_attempts
        assert exc_info.value.max_retries == max_retries
        
        # Branch should execute exactly N+1 times
        assert branch_call_count == expected_attempts

    @settings(max_examples=50)
    @given(
        max_retries=st.integers(min_value=0, max_value=10),
        action_type=action_type_strategy,
    )
    def test_error_message_format(
        self, max_retries: int, action_type: str
    ):
        """Property 8: TargetNotFoundError message should follow expected format.
        
        **Validates: Requirements 6.2, 6.4, 6.5**
        """
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            if target == "main-btn":
                raise ElementNotFoundError("Element not found")
            return "branch executed"
        
        # Create branch action
        branch_action = Action(
            id="branch_action",
            type="click",
            target="branch-btn"
        )
        
        # Create main action with retry
        action = Action(
            id="main_action",
            type=action_type,
            target="main-btn",
            target_not_found_actions=[branch_action],
            target_not_found_config={
                "retry_after_handling": True,
                "max_retries": max_retries,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        
        with pytest.raises(TargetNotFoundError) as exc_info:
            node.run(context)
        
        # Verify error attributes
        assert exc_info.value.action_type == action_type
        assert exc_info.value.max_retries == max_retries
        
        # Verify message contains expected parts
        error_msg = str(exc_info.value)
        assert f"Action: {action_type}" in error_msg
        assert f"{max_retries} retries allowed" in error_msg


# =============================================================================
# Task 14.4: Property 9 - Integration with no_action_if_target_not_found
# =============================================================================

class TestNoActionIfTargetNotFoundProperty:
    """Property-based tests for integration with no_action_if_target_not_found.
    
    **Validates: Requirements 5.5, 5.6, 10.2, 10.3**
    
    These tests verify that:
    1. When no branch exists and no_action_if_target_not_found=True, action is skipped
    2. When no branch exists and no_action_if_target_not_found=False, error is raised
    3. When branch exists, it takes precedence over no_action_if_target_not_found
    """

    @settings(max_examples=50)
    @given(
        action_type=action_type_strategy,
    )
    def test_no_branch_with_no_action_flag_skips_action(
        self, action_type: str
    ):
        """Property 9: When no branch exists and no_action_if_target_not_found=True,
        the action should be skipped without error.
        
        **Validates: Requirements 5.5, 5.6, 10.2, 10.3**
        """
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            no_action_flag = kwargs.get("no_action_if_target_not_found", False)
            if target == "main-btn":
                if no_action_flag:
                    return None  # Skipped
                raise ElementNotFoundError("Element not found")
            return "executed"
        
        # Create action without branch but with no_action_if_target_not_found=True
        action = Action(
            id="main_action",
            type=action_type,
            target="main-btn",
            no_action_if_target_not_found=True,
            target_not_found_actions=None,
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        # Should succeed (action skipped)
        assert result.success is True

    @settings(max_examples=50)
    @given(
        num_branch_actions=num_branch_actions_strategy,
    )
    def test_branch_takes_precedence_over_no_action_flag(
        self, num_branch_actions: int
    ):
        """Property 9: When branch exists, it should take precedence over
        no_action_if_target_not_found flag.
        
        **Validates: Requirements 5.5, 5.6, 10.2, 10.3**
        """
        execution_log = []
        
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            execution_log.append(target)
            if target == "main-btn":
                raise ElementNotFoundError("Element not found")
            return f"executed {target}"
        
        # Create branch actions
        branch_actions = [
            Action(
                id=f"branch_{i}",
                type="click",
                target=f"branch-btn-{i}"
            )
            for i in range(num_branch_actions)
        ]
        
        # Create action with both branch AND no_action_if_target_not_found=True
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            no_action_if_target_not_found=True,  # This should be ignored
            target_not_found_actions=branch_actions,
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        context = ExecutionRuntime()
        result = node.run(context)
        
        # Branch should execute (takes precedence)
        assert result.success is True
        assert result.metadata.get("branch_executed") is True
        
        # All branch actions should have executed
        assert len(execution_log) == num_branch_actions + 1


# =============================================================================
# Task 14.5: Property 10 - Graph reusability
# =============================================================================

class TestGraphReusabilityProperty:
    """Property-based tests for graph reusability.
    
    **Validates: Requirements 10.5**
    
    These tests verify that:
    1. ActionNode can be executed multiple times with same results
    2. State is properly reset between executions
    """

    @settings(max_examples=50)
    @given(
        num_executions=st.integers(min_value=2, max_value=5),
        num_branch_actions=num_branch_actions_strategy,
    )
    def test_action_node_reusable_with_branch(
        self, num_executions: int, num_branch_actions: int
    ):
        """Property 10: ActionNode should be executable multiple times with
        consistent results.
        
        **Validates: Requirements 10.5**
        """
        execution_counts = []
        
        def mock_executor(**kwargs):
            target = kwargs.get("action_target")
            if target == "main-btn":
                raise ElementNotFoundError("Element not found")
            return f"executed {target}"
        
        # Create branch actions
        branch_actions = [
            Action(
                id=f"branch_{i}",
                type="click",
                target=f"branch-btn-{i}"
            )
            for i in range(num_branch_actions)
        ]
        
        # Create action with branch
        action = Action(
            id="main_action",
            type="click",
            target="main-btn",
            target_not_found_actions=branch_actions,
            target_not_found_config={
                "retry_after_handling": False,
                "max_retries": 3,
                "retry_delay": 0.0
            }
        )
        
        node = ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        
        # Execute multiple times
        results = []
        for _ in range(num_executions):
            context = ExecutionRuntime()
            result = node.run(context)
            results.append(result)
        
        # All executions should succeed
        for result in results:
            assert result.success is True
            assert result.metadata.get("branch_executed") is True

    @settings(max_examples=50)
    @given(
        num_executions=st.integers(min_value=2, max_value=5),
    )
    def test_graph_reusable_with_target_not_found_context(
        self, num_executions: int
    ):
        """Property 10: ActionGraph with target_not_found context should be
        reusable for multiple executions.
        
        **Validates: Requirements 10.5**
        """
        execution_count = 0
        
        def mock_executor(**kwargs):
            nonlocal execution_count
            execution_count += 1
            return "executed"
        
        # Create graph with target_not_found context
        graph = ActionGraph(action_executor=mock_executor)
        
        parent_target = TargetSpec(strategy=TargetStrategy.ID, value="parent-btn")
        parent_helper = graph.action("click", target=parent_target)
        
        with parent_helper.target_not_found():
            fallback_target = TargetSpec(strategy=TargetStrategy.ID, value="fallback-btn")
            graph.action("click", target=fallback_target)
        
        # Execute multiple times
        for _ in range(num_executions):
            execution_count = 0
            result = graph.execute()
            # Each execution should call the executor at least once
            assert execution_count >= 1


# =============================================================================
# Task 14.6: Property 11 - Exception type detection
# =============================================================================

class TestExceptionTypeDetectionProperty:
    """Property-based tests for exception type detection.
    
    **Validates: Requirements 5.2, 9.2, 9.3**
    
    These tests verify that:
    1. _is_element_not_found_error() correctly identifies element-not-found exceptions
    2. Exact name matching works for ElementNotFoundError, ElementNotFoundException, TargetNotFoundError
    3. MRO check works for subclasses
    4. Non-matching exceptions return False
    """

    def _create_test_node(self):
        """Create a test ActionNode for testing _is_element_not_found_error."""
        def mock_executor(**kwargs):
            return "executed"
        
        action = Action(
            id="test_action",
            type="click",
            target="test-btn"
        )
        
        return ActionNode(
            action=action,
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )

    @settings(max_examples=50)
    @given(
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_element_not_found_error_detected(self, error_message: str):
        """Property 11: ElementNotFoundError should be detected as element-not-found.
        
        **Validates: Requirements 5.2, 9.2, 9.3**
        """
        node = self._create_test_node()
        error = ElementNotFoundError(error_message)
        
        assert node._is_element_not_found_error(error) is True

    @settings(max_examples=50)
    @given(
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_element_not_found_exception_detected(self, error_message: str):
        """Property 11: ElementNotFoundException should be detected as element-not-found.
        
        **Validates: Requirements 5.2, 9.2, 9.3**
        """
        node = self._create_test_node()
        error = ElementNotFoundException(error_message)
        
        assert node._is_element_not_found_error(error) is True

    @settings(max_examples=50)
    @given(
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_custom_subclass_detected_via_mro(self, error_message: str):
        """Property 11: Subclass of ElementNotFoundError should be detected via MRO.
        
        **Validates: Requirements 5.2, 9.2, 9.3**
        """
        node = self._create_test_node()
        error = CustomElementNotFoundError(error_message)
        
        assert node._is_element_not_found_error(error) is True

    @settings(max_examples=50)
    @given(
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_file_not_found_error_not_detected(self, error_message: str):
        """Property 11: FileNotFoundError should NOT be detected as element-not-found.
        
        **Validates: Requirements 5.2, 9.2, 9.3**
        """
        node = self._create_test_node()
        error = FileNotFoundError(error_message)
        
        assert node._is_element_not_found_error(error) is False

    @settings(max_examples=50)
    @given(
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_some_other_not_found_error_not_detected(self, error_message: str):
        """Property 11: Exception with 'NotFound' in name but not exact match
        should NOT be detected.
        
        **Validates: Requirements 5.2, 9.2, 9.3**
        """
        node = self._create_test_node()
        error = SomeOtherNotFoundError(error_message)
        
        assert node._is_element_not_found_error(error) is False

    @settings(max_examples=50)
    @given(
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_value_error_not_detected(self, error_message: str):
        """Property 11: ValueError should NOT be detected as element-not-found.
        
        **Validates: Requirements 5.2, 9.2, 9.3**
        """
        node = self._create_test_node()
        error = ValueError(error_message)
        
        assert node._is_element_not_found_error(error) is False

    @settings(max_examples=50)
    @given(
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_runtime_error_not_detected(self, error_message: str):
        """Property 11: RuntimeError should NOT be detected as element-not-found.
        
        **Validates: Requirements 5.2, 9.2, 9.3**
        """
        node = self._create_test_node()
        error = RuntimeError(error_message)
        
        assert node._is_element_not_found_error(error) is False

    def test_target_not_found_error_detected(self):
        """Property 11: TargetNotFoundError should be detected as element-not-found.
        
        **Validates: Requirements 5.2, 9.2, 9.3**
        """
        node = self._create_test_node()
        error = TargetNotFoundError(
            action_type="click",
            target="test-btn",
            attempt_count=1,
            max_retries=0
        )
        
        assert node._is_element_not_found_error(error) is True


# =============================================================================
# Extended main entry point for running all property-based tests
# =============================================================================

if __name__ == '__main__':
    # Remove the original main block and replace with comprehensive test runner
    pass


# =============================================================================
# Task 15.1: Property 12 - Exception attributes and messages
# =============================================================================

class TestExceptionAttributesProperty:
    """Property-based tests for exception attributes and messages.
    
    **Validates: Requirements 8.1, 8.2, 8.3**
    
    These tests verify that for any valid inputs:
    1. TargetNotFoundError stores all attributes correctly (action_type, target, attempt_count, max_retries)
    2. BranchAlreadyExistsError stores all attributes correctly (condition, action_type)
    3. Error messages are user-friendly and include relevant context
    """

    @settings(max_examples=100)
    @given(
        action_type=action_type_strategy,
        target_value=target_value_strategy,
        attempt_count=st.integers(min_value=1, max_value=100),
        max_retries=st.integers(min_value=0, max_value=100),
    )
    def test_target_not_found_error_stores_string_target_attributes(
        self, action_type: str, target_value: str, attempt_count: int, max_retries: int
    ):
        """Property 12: TargetNotFoundError should store all attributes correctly
        when target is a string.
        
        **Validates: Requirements 8.1, 8.3**
        """
        error = TargetNotFoundError(
            action_type=action_type,
            target=target_value,
            attempt_count=attempt_count,
            max_retries=max_retries
        )
        
        # Verify all attributes are stored correctly
        assert error.action_type == action_type
        assert error.target == target_value
        assert error.attempt_count == attempt_count
        assert error.max_retries == max_retries

    @settings(max_examples=100)
    @given(
        action_type=action_type_strategy,
        target=target_spec_strategy,
        attempt_count=st.integers(min_value=1, max_value=100),
        max_retries=st.integers(min_value=0, max_value=100),
    )
    def test_target_not_found_error_stores_target_spec_attributes(
        self, action_type: str, target: TargetSpec, attempt_count: int, max_retries: int
    ):
        """Property 12: TargetNotFoundError should store all attributes correctly
        when target is a TargetSpec.
        
        **Validates: Requirements 8.1, 8.3**
        """
        error = TargetNotFoundError(
            action_type=action_type,
            target=target,
            attempt_count=attempt_count,
            max_retries=max_retries
        )
        
        # Verify all attributes are stored correctly
        assert error.action_type == action_type
        assert error.target == target
        assert error.attempt_count == attempt_count
        assert error.max_retries == max_retries

    @settings(max_examples=100)
    @given(
        action_type=action_type_strategy,
        target=target_spec_with_fallback_strategy,
        attempt_count=st.integers(min_value=1, max_value=100),
        max_retries=st.integers(min_value=0, max_value=100),
    )
    def test_target_not_found_error_stores_fallback_target_attributes(
        self, action_type: str, target: TargetSpecWithFallback, attempt_count: int, max_retries: int
    ):
        """Property 12: TargetNotFoundError should store all attributes correctly
        when target is a TargetSpecWithFallback.
        
        **Validates: Requirements 8.1, 8.3**
        """
        error = TargetNotFoundError(
            action_type=action_type,
            target=target,
            attempt_count=attempt_count,
            max_retries=max_retries
        )
        
        # Verify all attributes are stored correctly
        assert error.action_type == action_type
        assert error.target == target
        assert error.attempt_count == attempt_count
        assert error.max_retries == max_retries

    @settings(max_examples=100)
    @given(
        action_type=action_type_strategy,
        target_value=target_value_strategy,
        attempt_count=st.integers(min_value=1, max_value=100),
        max_retries=st.integers(min_value=0, max_value=100),
    )
    def test_target_not_found_error_message_contains_context(
        self, action_type: str, target_value: str, attempt_count: int, max_retries: int
    ):
        """Property 12: TargetNotFoundError message should contain relevant context.
        
        **Validates: Requirements 8.1, 8.3**
        """
        error = TargetNotFoundError(
            action_type=action_type,
            target=target_value,
            attempt_count=attempt_count,
            max_retries=max_retries
        )
        
        error_msg = str(error)
        
        # Message should contain action type
        assert f"Action: {action_type}" in error_msg
        
        # Message should contain attempt count
        assert str(attempt_count) in error_msg
        
        # Message should contain max_retries
        assert f"{max_retries} retries allowed" in error_msg
        
        # Message should contain target info
        assert target_value in error_msg

    @settings(max_examples=100)
    @given(
        action_type=action_type_strategy,
        target=target_spec_strategy,
        attempt_count=st.integers(min_value=1, max_value=100),
        max_retries=st.integers(min_value=0, max_value=100),
    )
    def test_target_not_found_error_message_formats_target_spec(
        self, action_type: str, target: TargetSpec, attempt_count: int, max_retries: int
    ):
        """Property 12: TargetNotFoundError message should format TargetSpec correctly.
        
        **Validates: Requirements 8.1, 8.3**
        """
        error = TargetNotFoundError(
            action_type=action_type,
            target=target,
            attempt_count=attempt_count,
            max_retries=max_retries
        )
        
        error_msg = str(error)
        
        # Message should contain formatted target (strategy:value)
        expected_target_str = f"{target.strategy}:{target.value}"
        assert expected_target_str in error_msg

    @settings(max_examples=100)
    @given(
        action_type=action_type_strategy,
        target=target_spec_with_fallback_strategy,
        attempt_count=st.integers(min_value=1, max_value=100),
        max_retries=st.integers(min_value=0, max_value=100),
    )
    def test_target_not_found_error_message_formats_fallback_target(
        self, action_type: str, target: TargetSpecWithFallback, attempt_count: int, max_retries: int
    ):
        """Property 12: TargetNotFoundError message should format TargetSpecWithFallback correctly.
        
        **Validates: Requirements 8.1, 8.3**
        """
        error = TargetNotFoundError(
            action_type=action_type,
            target=target,
            attempt_count=attempt_count,
            max_retries=max_retries
        )
        
        error_msg = str(error)
        
        # Message should contain formatted fallback target
        expected_target_str = f"fallback[{len(target.strategies)} strategies]"
        assert expected_target_str in error_msg

    @settings(max_examples=100)
    @given(
        action_type=action_type_strategy,
        attempt_count=st.just(1),  # Single attempt
        max_retries=st.integers(min_value=0, max_value=100),
    )
    def test_target_not_found_error_singular_attempt_word(
        self, action_type: str, attempt_count: int, max_retries: int
    ):
        """Property 12: TargetNotFoundError message should use singular 'attempt' for count=1.
        
        **Validates: Requirements 8.3**
        """
        error = TargetNotFoundError(
            action_type=action_type,
            target="test-target",
            attempt_count=attempt_count,
            max_retries=max_retries
        )
        
        error_msg = str(error)
        
        # Should use singular "attempt" not "attempts"
        assert "1 attempt" in error_msg
        assert "1 attempts" not in error_msg

    @settings(max_examples=100)
    @given(
        action_type=action_type_strategy,
        attempt_count=st.integers(min_value=2, max_value=100),  # Multiple attempts
        max_retries=st.integers(min_value=0, max_value=100),
    )
    def test_target_not_found_error_plural_attempts_word(
        self, action_type: str, attempt_count: int, max_retries: int
    ):
        """Property 12: TargetNotFoundError message should use plural 'attempts' for count>1.
        
        **Validates: Requirements 8.3**
        """
        error = TargetNotFoundError(
            action_type=action_type,
            target="test-target",
            attempt_count=attempt_count,
            max_retries=max_retries
        )
        
        error_msg = str(error)
        
        # Should use plural "attempts"
        assert f"{attempt_count} attempts" in error_msg

    @settings(max_examples=100)
    @given(
        condition=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        action_type=action_type_strategy,
    )
    def test_branch_already_exists_error_stores_attributes(
        self, condition: str, action_type: str
    ):
        """Property 12: BranchAlreadyExistsError should store all attributes correctly.
        
        **Validates: Requirements 8.2, 8.3**
        """
        error = BranchAlreadyExistsError(
            condition=condition,
            action_type=action_type
        )
        
        # Verify all attributes are stored correctly
        assert error.condition == condition
        assert error.action_type == action_type

    @settings(max_examples=100)
    @given(
        condition=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        action_type=action_type_strategy,
    )
    def test_branch_already_exists_error_message_contains_context(
        self, condition: str, action_type: str
    ):
        """Property 12: BranchAlreadyExistsError message should contain relevant context.
        
        **Validates: Requirements 8.2, 8.3**
        """
        error = BranchAlreadyExistsError(
            condition=condition,
            action_type=action_type
        )
        
        error_msg = str(error)
        
        # Message should contain condition
        assert condition in error_msg
        
        # Message should contain action type
        assert action_type in error_msg
        
        # Message should be user-friendly
        assert "already exists" in error_msg

    @settings(max_examples=50)
    @given(
        action_type=action_type_strategy,
    )
    def test_branch_already_exists_error_with_target_not_found_condition(
        self, action_type: str
    ):
        """Property 12: BranchAlreadyExistsError with 'target_not_found' condition
        should have correct message format.
        
        **Validates: Requirements 8.2, 8.3**
        """
        error = BranchAlreadyExistsError(
            condition="target_not_found",
            action_type=action_type
        )
        
        error_msg = str(error)
        
        # Message should follow expected format
        assert f"Branch 'target_not_found' already exists on action '{action_type}'." == error_msg


# =============================================================================
# Task 15.2: Property 13 - Method equivalence (target_not_found vs on_target_not_found)
# =============================================================================

class TestMethodEquivalenceProperty:
    """Property-based tests for method equivalence.
    
    **Validates: Requirements 2.6**
    
    These tests verify that for any valid parameters:
    1. target_not_found() and on_target_not_found() produce equivalent results
    2. Both methods accept the same parameters
    3. Both methods raise the same errors for invalid inputs
    """

    @settings(max_examples=100)
    @given(
        retry_after_handling=st.booleans(),
        max_retries=valid_max_retries_strategy,
        retry_delay=valid_retry_delay_strategy,
    )
    def test_methods_produce_equivalent_context(
        self, retry_after_handling: bool, max_retries: int, retry_delay: float
    ):
        """Property 13: target_not_found() and on_target_not_found() should produce
        equivalent TargetNotFoundContext objects.
        
        **Validates: Requirements 2.6**
        """
        mock_executor = lambda **kwargs: "executed"
        
        # Create two graphs with identical setup
        graph1 = ActionGraph(action_executor=mock_executor)
        graph2 = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        
        helper1 = graph1.action("click", target=target)
        helper2 = graph2.action("click", target=target)
        
        # Use target_not_found() on first helper
        with helper1.target_not_found(
            retry_after_handling=retry_after_handling,
            max_retries=max_retries,
            retry_delay=retry_delay
        ):
            graph1.action("click", target=TargetSpec(strategy=TargetStrategy.ID, value="fallback-1"))
        
        # Use on_target_not_found() on second helper
        with helper2.on_target_not_found(
            retry_after_handling=retry_after_handling,
            max_retries=max_retries,
            retry_delay=retry_delay
        ):
            graph2.action("click", target=TargetSpec(strategy=TargetStrategy.ID, value="fallback-1"))
        
        # Both actions should have equivalent target_not_found_config
        action1 = helper1.action_obj
        action2 = helper2.action_obj
        
        assert action1.target_not_found_config == action2.target_not_found_config
        
        # Both should have same number of branch actions
        assert len(action1.target_not_found_actions) == len(action2.target_not_found_actions)

    @settings(max_examples=100)
    @given(
        invalid_max_retries=invalid_max_retries_strategy,
    )
    def test_both_methods_raise_same_error_for_invalid_max_retries(
        self, invalid_max_retries: int
    ):
        """Property 13: Both methods should raise ValueError for invalid max_retries.
        
        **Validates: Requirements 2.6**
        """
        mock_executor = lambda **kwargs: "executed"
        
        graph1 = ActionGraph(action_executor=mock_executor)
        graph2 = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        
        helper1 = graph1.action("click", target=target)
        helper2 = graph2.action("click", target=target)
        
        # Both should raise ValueError
        with pytest.raises(ValueError) as exc_info1:
            helper1.target_not_found(max_retries=invalid_max_retries)
        
        with pytest.raises(ValueError) as exc_info2:
            helper2.on_target_not_found(max_retries=invalid_max_retries)
        
        # Error messages should be identical
        assert str(exc_info1.value) == str(exc_info2.value)

    @settings(max_examples=100)
    @given(
        invalid_retry_delay=invalid_retry_delay_strategy,
    )
    def test_both_methods_raise_same_error_for_invalid_retry_delay(
        self, invalid_retry_delay: float
    ):
        """Property 13: Both methods should raise ValueError for invalid retry_delay.
        
        **Validates: Requirements 2.6**
        """
        mock_executor = lambda **kwargs: "executed"
        
        graph1 = ActionGraph(action_executor=mock_executor)
        graph2 = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        
        helper1 = graph1.action("click", target=target)
        helper2 = graph2.action("click", target=target)
        
        # Both should raise ValueError
        with pytest.raises(ValueError) as exc_info1:
            helper1.target_not_found(retry_delay=invalid_retry_delay)
        
        with pytest.raises(ValueError) as exc_info2:
            helper2.on_target_not_found(retry_delay=invalid_retry_delay)
        
        # Error messages should be identical
        assert str(exc_info1.value) == str(exc_info2.value)

    @settings(max_examples=50)
    @given(
        action_type=action_type_strategy,
    )
    def test_both_methods_raise_same_error_for_no_target(
        self, action_type: str
    ):
        """Property 13: Both methods should raise ValueError when action has no target.
        
        **Validates: Requirements 2.6**
        """
        mock_executor = lambda **kwargs: "executed"
        
        graph1 = ActionGraph(action_executor=mock_executor)
        graph2 = ActionGraph(action_executor=mock_executor)
        
        # Create actions without target
        helper1 = graph1.action(action_type, target=None)
        helper2 = graph2.action(action_type, target=None)
        
        # Both should raise ValueError
        with pytest.raises(ValueError) as exc_info1:
            helper1.target_not_found()
        
        with pytest.raises(ValueError) as exc_info2:
            helper2.on_target_not_found()
        
        # Error messages should be identical
        assert str(exc_info1.value) == str(exc_info2.value)

    @settings(max_examples=50)
    @given(
        action_type=action_type_strategy,
    )
    def test_both_methods_raise_same_error_for_duplicate_branch(
        self, action_type: str
    ):
        """Property 13: Both methods should raise BranchAlreadyExistsError for duplicate branch.
        
        **Validates: Requirements 2.6**
        """
        mock_executor = lambda **kwargs: "executed"
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        
        # Test target_not_found() followed by on_target_not_found()
        graph1 = ActionGraph(action_executor=mock_executor)
        helper1 = graph1.action(action_type, target=target)
        with helper1.target_not_found():
            pass
        
        with pytest.raises(BranchAlreadyExistsError) as exc_info1:
            helper1.on_target_not_found()
        
        # Test on_target_not_found() followed by target_not_found()
        graph2 = ActionGraph(action_executor=mock_executor)
        helper2 = graph2.action(action_type, target=target)
        with helper2.on_target_not_found():
            pass
        
        with pytest.raises(BranchAlreadyExistsError) as exc_info2:
            helper2.target_not_found()
        
        # Both should have same error attributes
        assert exc_info1.value.condition == exc_info2.value.condition
        assert exc_info1.value.action_type == exc_info2.value.action_type

    @settings(max_examples=100)
    @given(
        retry_after_handling=st.booleans(),
        max_retries=valid_max_retries_strategy,
        retry_delay=valid_retry_delay_strategy,
        num_branch_actions=st.integers(min_value=1, max_value=5),
    )
    def test_methods_store_same_config(
        self, retry_after_handling: bool, max_retries: int, retry_delay: float, num_branch_actions: int
    ):
        """Property 13: Both methods should store identical config on the action.
        
        **Validates: Requirements 2.6**
        """
        mock_executor = lambda **kwargs: "executed"
        
        graph1 = ActionGraph(action_executor=mock_executor)
        graph2 = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        
        helper1 = graph1.action("click", target=target)
        helper2 = graph2.action("click", target=target)
        
        # Use target_not_found() on first helper
        with helper1.target_not_found(
            retry_after_handling=retry_after_handling,
            max_retries=max_retries,
            retry_delay=retry_delay
        ):
            for i in range(num_branch_actions):
                graph1.action("click", target=TargetSpec(strategy=TargetStrategy.ID, value=f"fallback-{i}"))
        
        # Use on_target_not_found() on second helper
        with helper2.on_target_not_found(
            retry_after_handling=retry_after_handling,
            max_retries=max_retries,
            retry_delay=retry_delay
        ):
            for i in range(num_branch_actions):
                graph2.action("click", target=TargetSpec(strategy=TargetStrategy.ID, value=f"fallback-{i}"))
        
        action1 = helper1.action_obj
        action2 = helper2.action_obj
        
        # Config should be identical
        assert action1.target_not_found_config['retry_after_handling'] == action2.target_not_found_config['retry_after_handling']
        assert action1.target_not_found_config['max_retries'] == action2.target_not_found_config['max_retries']
        assert action1.target_not_found_config['retry_delay'] == action2.target_not_found_config['retry_delay']
        
        # Branch actions should have same count
        assert len(action1.target_not_found_actions) == len(action2.target_not_found_actions)

    def test_on_target_not_found_is_alias(self):
        """Property 13: on_target_not_found should be the same method as target_not_found.
        
        **Validates: Requirements 2.6**
        """
        mock_executor = lambda **kwargs: "executed"
        graph = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        helper = graph.action("click", target=target)
        
        # Both should reference the same underlying method
        assert helper.on_target_not_found == helper.target_not_found

    @settings(max_examples=50)
    @given(
        retry_after_handling=st.booleans(),
        max_retries=valid_max_retries_strategy,
        retry_delay=valid_retry_delay_strategy,
    )
    def test_methods_return_same_context_type(
        self, retry_after_handling: bool, max_retries: int, retry_delay: float
    ):
        """Property 13: Both methods should return TargetNotFoundContext.
        
        **Validates: Requirements 2.6**
        """
        mock_executor = lambda **kwargs: "executed"
        
        graph1 = ActionGraph(action_executor=mock_executor)
        graph2 = ActionGraph(action_executor=mock_executor)
        
        target = TargetSpec(strategy=TargetStrategy.ID, value="test-btn")
        
        helper1 = graph1.action("click", target=target)
        helper2 = graph2.action("click", target=target)
        
        ctx1 = helper1.target_not_found(
            retry_after_handling=retry_after_handling,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        ctx2 = helper2.on_target_not_found(
            retry_after_handling=retry_after_handling,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Both should return TargetNotFoundContext
        assert type(ctx1).__name__ == 'TargetNotFoundContext'
        assert type(ctx2).__name__ == 'TargetNotFoundContext'
        assert type(ctx1) == type(ctx2)

"""
Shared Test Fixtures and Hypothesis Strategies for Action Schema Tests

This module provides reusable pytest fixtures and Hypothesis strategies for
testing the ActionGraph target_not_found feature and related functionality.

Usage:
    # In test files, fixtures are automatically available via pytest
    def test_something(mock_executor, action_graph):
        ...
    
    # Hypothesis strategies can be imported directly
    from conftest import action_strategy, target_spec_strategy
    
    @given(action=action_strategy())
    def test_property(action):
        ...
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import strategies as st

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

from agent_foundation.automation.schema.common import (
    Action,
    TargetSpec,
    TargetSpecWithFallback,
    TargetStrategy,
    ExecutionRuntime,
)
from agent_foundation.automation.schema.action_graph import ActionGraph
from agent_foundation.automation.schema.action_metadata import ActionMetadataRegistry


# =============================================================================
# Custom Exception Classes for Testing
# =============================================================================

class ElementNotFoundError(Exception):
    """Custom exception for testing element-not-found detection.
    
    This exception name exactly matches one of the expected exception names
    in ActionNode._is_element_not_found_error().
    """
    pass


class ElementNotFoundException(Exception):
    """Alternative custom exception for testing element-not-found detection.
    
    This exception name exactly matches one of the expected exception names
    in ActionNode._is_element_not_found_error().
    """
    pass


class CustomElementNotFoundError(ElementNotFoundError):
    """Subclass of ElementNotFoundError for testing MRO detection.
    
    Used to verify that ActionNode._is_element_not_found_error() correctly
    checks the inheritance chain (MRO) for matching exception types.
    """
    pass


class SomeOtherNotFoundError(Exception):
    """Exception with 'NotFound' in name but not an exact match.
    
    Used to verify that ActionNode._is_element_not_found_error() does NOT
    match exceptions that merely contain 'NotFound' in their name.
    """
    pass


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def mock_executor():
    """Create a simple mock executor that returns 'executed' for all actions.
    
    Returns:
        A callable that can be used as action_executor for ActionGraph.
    
    Example:
        def test_something(mock_executor):
            graph = ActionGraph(action_executor=mock_executor)
            graph.action("click", target="btn")
            result = graph.execute()
    """
    def executor(**kwargs):
        return "executed"
    return executor


@pytest.fixture
def failing_executor():
    """Create a mock executor that raises ElementNotFoundError for specific targets.
    
    The executor raises ElementNotFoundError when the target is "main-btn",
    otherwise returns "executed".
    
    Returns:
        A callable that can be used as action_executor for ActionGraph.
    
    Example:
        def test_branch_execution(failing_executor):
            graph = ActionGraph(action_executor=failing_executor)
            with graph.action("click", target="main-btn").target_not_found():
                graph.action("click", target="fallback-btn")
            result = graph.execute()
    """
    def executor(**kwargs):
        target = kwargs.get("action_target")
        if target == "main-btn":
            raise ElementNotFoundError("Element not found")
        return f"executed {target}"
    return executor


@pytest.fixture
def tracking_executor():
    """Create a mock executor that tracks all calls.
    
    Returns:
        A tuple of (executor_callable, execution_log_list).
        The execution_log_list contains all targets that were executed.
    
    Example:
        def test_execution_order(tracking_executor):
            executor, log = tracking_executor
            graph = ActionGraph(action_executor=executor)
            graph.action("click", target="btn1")
            graph.action("click", target="btn2")
            graph.execute()
            assert log == ["btn1", "btn2"]
    """
    execution_log = []
    
    def executor(**kwargs):
        target = kwargs.get("action_target")
        execution_log.append(target)
        return f"executed {target}"
    
    return executor, execution_log


@pytest.fixture
def action_graph(mock_executor):
    """Create an ActionGraph instance with a mock executor.
    
    Args:
        mock_executor: The mock executor fixture (auto-injected by pytest).
    
    Returns:
        An ActionGraph instance ready for testing.
    
    Example:
        def test_action_graph(action_graph):
            action_graph.action("click", target="btn")
            result = action_graph.execute()
    """
    return ActionGraph(action_executor=mock_executor)


@pytest.fixture
def action_metadata():
    """Create an ActionMetadataRegistry instance.
    
    Returns:
        An ActionMetadataRegistry instance for testing.
    """
    return ActionMetadataRegistry()


@pytest.fixture
def execution_runtime():
    """Create an ExecutionRuntime instance.
    
    Returns:
        An ExecutionRuntime instance for testing action execution.
    """
    return ExecutionRuntime()


# =============================================================================
# Hypothesis Strategies - Basic Types
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

# Strategy for generating valid target values (non-empty strings)
target_value_strategy = st.text(min_size=1, max_size=50).filter(lambda x: x.strip())


# =============================================================================
# Hypothesis Strategies - Target Specifications
# =============================================================================

# Strategy for generating TargetSpec objects
target_spec_strategy = st.builds(
    TargetSpec,
    strategy=target_strategy_strategy,
    value=target_value_strategy,
    description=st.one_of(
        st.none(),
        st.text(min_size=1, max_size=50).filter(lambda x: x.strip())
    ),
    options=st.one_of(
        st.none(),
        st.lists(st.text(min_size=1, max_size=20).filter(lambda x: x.strip()), max_size=3)
    )
)

# Strategy for generating TargetSpecWithFallback objects
target_spec_with_fallback_strategy = st.builds(
    TargetSpecWithFallback,
    strategies=st.lists(target_spec_strategy, min_size=1, max_size=3)
)

# Strategy for generating any valid target (string, TargetSpec, or TargetSpecWithFallback)
target_strategy = st.one_of(
    target_value_strategy,  # Simple string target
    target_spec_strategy,   # TargetSpec
    target_spec_with_fallback_strategy  # TargetSpecWithFallback
)


# =============================================================================
# Hypothesis Strategies - Action Arguments and Config
# =============================================================================

# Strategy for generating action args dictionaries
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

# Strategy for generating target_not_found_config dictionaries
target_not_found_config_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries({
        'retry_after_handling': st.booleans(),
        'max_retries': st.integers(min_value=0, max_value=10),
        'retry_delay': st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)
    })
)


# =============================================================================
# Hypothesis Strategies - Parameter Validation
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


# =============================================================================
# Hypothesis Strategies - Action Generation
# =============================================================================

# Strategy for generating number of actions (1-10)
num_actions_strategy = st.integers(min_value=1, max_value=10)

# Strategy for generating number of branch actions (1-5)
num_branch_actions_strategy = st.integers(min_value=1, max_value=5)


@st.composite
def action_strategy(draw, max_depth: int = 2, current_depth: int = 0):
    """Generate valid Action instances with optional nested target_not_found_actions.
    
    This is a recursive strategy that can generate Actions with nested
    target_not_found_actions up to the specified max_depth.
    
    Args:
        draw: Hypothesis draw function (provided by @st.composite).
        max_depth: Maximum nesting depth for target_not_found_actions.
        current_depth: Current nesting depth (used internally for recursion).
    
    Returns:
        A valid Action instance.
    
    Example:
        @given(action=action_strategy())
        def test_action_serialization(action):
            json_str = action.json()
            restored = Action.parse_raw(json_str)
            assert action.id == restored.id
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


@st.composite
def action_with_branch_strategy(draw, max_depth: int = 2):
    """Generate Action instances that always have target_not_found_actions.
    
    This strategy ensures the action has a target and at least one nested action
    in target_not_found_actions. Useful for testing branch execution behavior.
    
    Args:
        draw: Hypothesis draw function (provided by @st.composite).
        max_depth: Maximum nesting depth for nested actions.
    
    Returns:
        An Action instance with guaranteed target_not_found_actions.
    
    Example:
        @given(action=action_with_branch_strategy())
        def test_branch_execution(action):
            assert action.target_not_found_actions is not None
            assert len(action.target_not_found_actions) > 0
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
    Useful for verifying serialization round-trips.
    
    Args:
        action1: First Action to compare.
        action2: Second Action to compare.
    
    Returns:
        True if the actions are equivalent, False otherwise.
    
    Example:
        original = Action(id="test", type="click", target="btn")
        restored = Action.parse_raw(original.json())
        assert actions_are_equivalent(original, restored)
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
    """Check if two targets are equivalent.
    
    Handles comparison of string targets, TargetSpec, and TargetSpecWithFallback.
    
    Args:
        target1: First target to compare.
        target2: Second target to compare.
    
    Returns:
        True if the targets are equivalent, False otherwise.
    """
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

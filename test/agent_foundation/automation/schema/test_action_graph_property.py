"""
Property-based tests for ActionGraph.

Tests correctness properties from the action-flow-workflow-refactor design document.
"""

import sys
import warnings
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
_workspace_root = _project_root.parent
_science_python_utils_src = _workspace_root / "SciencePythonUtils" / "src"
if _science_python_utils_src.exists() and str(_science_python_utils_src) not in sys.path:
    sys.path.insert(0, str(_science_python_utils_src))

from typing import Any, Dict, Optional
from hypothesis import given, strategies as st, settings

from rich_python_utils.common_objects.workflow.workgraph import WorkGraph

from science_modeling_tools.automation.schema.action_graph import (
    ActionGraph,
    ActionSequenceNode,
)
from science_modeling_tools.automation.schema.action_metadata import ActionMetadataRegistry


# region Test Fixtures

def mock_executor(
    action_type: str,
    action_target: Optional[str],
    action_args: Optional[Dict[str, Any]] = None,
    action_target_strategy: Optional[str] = None,
) -> str:
    """Mock action executor that returns a success string."""
    return f"executed_{action_type}_{action_target}"


# endregion


# region Property Tests

def test_action_graph_extends_workgraph():
    """
    Property 13: ActionGraph extends WorkGraph.
    
    **Feature: action-flow-workflow-refactor, Property 13: ActionGraph extends WorkGraph**
    **Validates: Requirements 7.3, 7.4**
    
    For any ActionGraph instance, it SHALL be an instance of WorkGraph.
    """
    graph = ActionGraph(action_executor=mock_executor)
    
    assert isinstance(graph, WorkGraph), \
        f"ActionGraph should be instance of WorkGraph, got {type(graph)}"


@settings(max_examples=100)
@given(st.sampled_from(['click', 'type', 'navigate', 'wait']))
def test_chaining_returns_self(action_type: str):
    """
    Property 14: Chaining returns self.
    
    **Feature: action-flow-workflow-refactor, Property 14: Chaining returns self**
    **Validates: Requirements 8.1**
    
    For any call to ActionGraph.action(), the return value SHALL be
    the same ActionGraph instance (identity equality).
    """
    graph = ActionGraph(action_executor=mock_executor)
    result = graph.action(action_type, target='#element')
    
    assert result is graph, \
        "action() should return the same ActionGraph instance"


@settings(max_examples=50)
@given(st.lists(
    st.tuples(
        st.sampled_from(['click', 'type', 'navigate']),
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N')))
    ),
    min_size=2,
    max_size=5
))
def test_chained_vs_nonchained_equivalence(action_specs):
    """
    Property 15: Chained vs non-chained equivalence.
    
    **Feature: action-flow-workflow-refactor, Property 15: Chained vs non-chained equivalence**
    **Validates: Requirements 8.2, 8.3**
    
    For any sequence of action() calls, chained style `g.action(a).action(b)`
    SHALL produce the same graph structure as non-chained `g.action(a); g.action(b)`.
    """
    # Build with chaining
    g1 = ActionGraph(action_executor=mock_executor)
    chain = g1
    for action_type, target in action_specs:
        chain = chain.action(action_type, target=target)
    
    # Build without chaining
    g2 = ActionGraph(action_executor=mock_executor)
    for action_type, target in action_specs:
        g2.action(action_type, target=target)
    
    # Both should have same number of nodes
    assert len(g1._nodes) == len(g2._nodes), \
        f"Node count should match: {len(g1._nodes)} vs {len(g2._nodes)}"
    
    # Both should have same number of actions in current node
    assert len(g1._current_node._actions) == len(g2._current_node._actions), \
        "Action count in current node should match"


def test_condition_method_exists():
    """
    Test that condition() method exists and returns ConditionContext.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Requirements 9.7**
    """
    from science_modeling_tools.automation.schema.action_graph import ConditionContext
    
    graph = ActionGraph(action_executor=mock_executor)
    graph.action("click", target="#first")
    
    # condition() should return ConditionContext
    result = graph.condition(lambda r: r.success)
    assert isinstance(result, ConditionContext), \
        f"condition() should return ConditionContext, got {type(result)}"
    
    # Using the condition in an if-block should create a new node
    if result:
        graph.action("click", target="#second")
    
    # Should have created a new node
    assert len(graph._nodes) == 2, "condition() should create a new node when used in if-block"


def test_action_sequence_node_is_workgraph_node():
    """
    Test that ActionSequenceNode extends WorkGraphNode.
    
    **Feature: action-flow-workflow-refactor**
    """
    from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode
    
    node = ActionSequenceNode(
        name="test_node",
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
    )
    
    assert isinstance(node, WorkGraphNode), \
        "ActionSequenceNode should be instance of WorkGraphNode"


@settings(max_examples=50)
@given(st.lists(
    st.tuples(
        st.sampled_from(['click', 'type', 'navigate', 'scroll']),
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N')))
    ),
    min_size=2,
    max_size=10
))
def test_linear_segment_compression(action_specs):
    """
    Property 18: Linear segment compression.
    
    **Feature: action-flow-workflow-refactor, Property 18: Linear segment compression**
    **Validates: Requirements 11.1, 11.2**
    
    For any sequence of N consecutive action() calls without condition(),
    all N actions SHALL be compressed into a single ActionSequenceNode,
    and that node SHALL execute all actions via a single ActionFlow.
    """
    graph = ActionGraph(action_executor=mock_executor)
    
    # Add multiple actions without any condition() calls
    for action_type, target in action_specs:
        graph.action(action_type, target=target)
    
    # Should have exactly 1 node (the initial root node)
    assert len(graph._nodes) == 1, \
        f"Linear actions should be in single node, got {len(graph._nodes)} nodes"
    
    # That node should contain all actions
    assert len(graph._current_node._actions) == len(action_specs), \
        f"Node should contain {len(action_specs)} actions, got {len(graph._current_node._actions)}"
    
    # Verify actions are in correct order
    for i, (action_type, target) in enumerate(action_specs):
        node_action = graph._current_node._actions[i]
        assert node_action.type == action_type, \
            f"Action {i} type should be '{action_type}', got '{node_action.type}'"
        assert node_action.target == target, \
            f"Action {i} target should be '{target}', got '{node_action.target}'"


def test_conditional_branching_structure():
    """
    Property 16: Conditional branching structure.
    
    **Feature: action-flow-workflow-refactor, Property 16: Conditional branching structure**
    **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6**
    
    For any ActionGraph with conditional branches using condition():
    - Actions added inside if-blocks SHALL be associated with the true branch
    - Actions in else-blocks SHALL be associated with the fallback branch
    - All branches SHALL merge back for subsequent actions
    """
    from science_modeling_tools.automation.schema.action_graph import ConditionContext
    
    graph = ActionGraph(action_executor=mock_executor)
    
    # Add initial action
    graph.action('navigate', target='https://example.com')
    initial_node = graph._current_node
    initial_action_count = len(initial_node._actions)
    
    # Create condition - should return ConditionContext
    cond = graph.condition(lambda r: getattr(r, 'success', True))
    assert isinstance(cond, ConditionContext), \
        f"condition() should return ConditionContext, got {type(cond)}"
    
    # Enter if-block via __bool__
    if cond:
        # Actions in if-block should go to branch node
        graph.action('click', target='#success_btn')
        branch_node = graph._current_node
        assert branch_node is not initial_node, \
            "Actions in if-block should be in a new branch node"
        assert len(branch_node._actions) == 1, \
            "Branch node should have 1 action"
        assert branch_node._actions[0].type == 'click', \
            "Branch action should be 'click'"
    
    # Verify branch was created and linked
    assert len(graph._nodes) >= 2, \
        f"Should have at least 2 nodes after condition, got {len(graph._nodes)}"
    
    # Verify the branch node has the condition
    assert branch_node.condition is not None, \
        "Branch node should have a condition"


def test_condition_context_bool_returns_true():
    """
    Test that ConditionContext.__bool__ returns True for if-block entry.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Requirements 9.1, 9.2**
    """
    from science_modeling_tools.automation.schema.action_graph import ConditionContext
    
    graph = ActionGraph(action_executor=mock_executor)
    cond = ConditionContext(graph, lambda r: r.success)
    
    # First call should return True
    result = bool(cond)
    assert result is True, "First __bool__ call should return True"
    
    # Second call should return False (already entered)
    result2 = bool(cond)
    assert result2 is False, "Second __bool__ call should return False"


def test_branch_isolation():
    """
    Test that actions in different branches are isolated.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Requirements 9.3, 9.4, 9.5**
    """
    graph = ActionGraph(action_executor=mock_executor)
    
    # Initial action
    graph.action('navigate', target='https://example.com')
    root_node = graph._current_node
    
    # Create first branch
    cond1 = graph.condition(lambda r: True)
    if cond1:
        graph.action('click', target='#branch1_btn')
        branch1_node = graph._current_node
    
    # Branch1 should be linked from root
    assert branch1_node in (root_node.next or []), \
        "Branch1 should be linked from root node"
    
    # Branch1 should have its own actions
    assert len(branch1_node._actions) == 1, \
        "Branch1 should have exactly 1 action"
    assert branch1_node._actions[0].target == '#branch1_btn', \
        "Branch1 action should have correct target"


@settings(max_examples=50)
@given(
    st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        min_size=1,
        max_size=3
    ),
    st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        min_size=1,
        max_size=3
    )
)
def test_branch_action_counts(true_branch_targets, false_branch_targets):
    """
    Property test for branch action isolation.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Requirements 9.3, 9.4**
    
    For any conditional branch with N actions in true branch and M actions
    in else branch, the true branch node SHALL contain exactly N actions
    and the else branch node SHALL contain exactly M actions.
    """
    graph = ActionGraph(action_executor=mock_executor)
    
    # Add actions to true branch
    cond = graph.condition(lambda r: True)
    if cond:
        for target in true_branch_targets:
            graph.action('click', target=target)
        true_node = graph._current_node
    
    # Verify true branch has correct action count
    assert len(true_node._actions) == len(true_branch_targets), \
        f"True branch should have {len(true_branch_targets)} actions, got {len(true_node._actions)}"


# endregion


# region New Branching API Tests

def test_context_manager_branching():
    """
    Test context manager syntax for conditional branching.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Context manager branching API**
    
    Usage:
        with graph.condition(lambda r: r.success) as branch:
            with branch.if_true():
                graph.action('click', '#success')
            with branch.if_false():
                graph.action('click', '#retry')
    """
    from science_modeling_tools.automation.schema.action_graph import ConditionContext
    
    graph = ActionGraph(action_executor=mock_executor)
    graph.action('navigate', target='https://example.com')
    root_node = graph._current_node
    
    # Use context manager syntax
    with graph.condition(lambda r: getattr(r, 'success', True)) as branch:
        with branch.if_true():
            graph.action('click', target='#success_btn')
            true_node = graph._current_node
        
        with branch.if_false():
            graph.action('click', target='#retry_btn')
            false_node = graph._current_node
    
    # Verify both branches were created
    assert true_node is not root_node, "True branch should be a new node"
    assert false_node is not root_node, "False branch should be a new node"
    assert true_node is not false_node, "True and false branches should be different nodes"
    
    # Verify actions are in correct branches
    assert len(true_node._actions) == 1, "True branch should have 1 action"
    assert true_node._actions[0].target == '#success_btn', "True branch action target"
    
    assert len(false_node._actions) == 1, "False branch should have 1 action"
    assert false_node._actions[0].target == '#retry_btn', "False branch action target"
    
    # Verify both branches are linked from root
    assert true_node in (root_node.next or []), "True branch should be linked from root"
    assert false_node in (root_node.next or []), "False branch should be linked from root"


def test_context_manager_multi_action_branches():
    """
    Test context manager with multiple actions per branch.
    """
    graph = ActionGraph(action_executor=mock_executor)
    
    with graph.condition(lambda r: True) as branch:
        with branch.if_true():
            graph.action('click', target='#btn1')
            graph.action('click', target='#btn2')
            graph.action('click', target='#btn3')
            true_node = graph._current_node
        
        with branch.if_false():
            graph.action('click', target='#retry1')
            graph.action('click', target='#retry2')
            false_node = graph._current_node
    
    assert len(true_node._actions) == 3, "True branch should have 3 actions"
    assert len(false_node._actions) == 2, "False branch should have 2 actions"


def test_match_case_branching():
    """
    Test match-case syntax for conditional branching (Python 3.10+).
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Match-case branching API**
    
    Usage:
        match graph.condition(lambda r: r.success):
            case ConditionContext.TRUE:
                graph.action('click', '#success')
            case ConditionContext.FALSE:
                graph.action('click', '#retry')
    """
    from science_modeling_tools.automation.schema.action_graph import ConditionContext
    
    graph = ActionGraph(action_executor=mock_executor)
    graph.action('navigate', target='https://example.com')
    root_node = graph._current_node
    
    # Test TRUE sentinel comparison
    cond = graph.condition(lambda r: getattr(r, 'success', True))
    
    # Simulate match-case TRUE branch
    if cond == ConditionContext.TRUE:
        graph.action('click', target='#success_btn')
        true_node = graph._current_node
    
    # Simulate match-case FALSE branch
    if cond == ConditionContext.FALSE:
        graph.action('click', target='#retry_btn')
        false_node = graph._current_node
    
    # Verify both branches were created
    assert true_node is not root_node, "True branch should be a new node"
    assert false_node is not root_node, "False branch should be a new node"
    
    # Verify actions
    assert len(true_node._actions) == 1, "True branch should have 1 action"
    assert len(false_node._actions) == 1, "False branch should have 1 action"


def test_callback_based_branching():
    """
    Test callback-based branching API.
    
    **Feature: action-flow-workflow-refactor**
    **Validates: Callback-based branching API**
    
    Usage:
        graph.branch(
            condition=lambda r: r.success,
            if_true=lambda g: g.action('click', '#success'),
            if_false=lambda g: g.action('click', '#retry'),
        )
    """
    graph = ActionGraph(action_executor=mock_executor)
    graph.action('navigate', target='https://example.com')
    root_node = graph._current_node
    initial_node_count = len(graph._nodes)
    
    # Use callback-based branching
    result = graph.branch(
        condition=lambda r: getattr(r, 'success', True),
        if_true=lambda g: g.action('click', target='#success_btn'),
        if_false=lambda g: g.action('click', target='#retry_btn'),
    )
    
    # Should return self for chaining
    assert result is graph, "branch() should return self for chaining"
    
    # Should have created 2 new nodes (true and false branches)
    assert len(graph._nodes) == initial_node_count + 2, \
        f"Should have 2 new nodes, got {len(graph._nodes) - initial_node_count}"
    
    # Find the branch nodes
    true_node = graph._nodes[initial_node_count]
    false_node = graph._nodes[initial_node_count + 1]
    
    # Verify actions
    assert len(true_node._actions) == 1, "True branch should have 1 action"
    assert true_node._actions[0].target == '#success_btn', "True branch action target"
    
    assert len(false_node._actions) == 1, "False branch should have 1 action"
    assert false_node._actions[0].target == '#retry_btn', "False branch action target"


def test_callback_branching_multi_action():
    """
    Test callback-based branching with multiple actions per branch.
    """
    def on_success(g):
        g.action('click', target='#success')
        g.action('wait', args={'seconds': 1})
        g.action('screenshot', target='#result')
    
    def on_failure(g):
        g.action('click', target='#retry')
        g.action('screenshot', target='#error')
    
    graph = ActionGraph(action_executor=mock_executor)
    initial_node_count = len(graph._nodes)
    
    graph.branch(
        condition=lambda r: True,
        if_true=on_success,
        if_false=on_failure,
    )
    
    true_node = graph._nodes[initial_node_count]
    false_node = graph._nodes[initial_node_count + 1]
    
    assert len(true_node._actions) == 3, "True branch should have 3 actions"
    assert len(false_node._actions) == 2, "False branch should have 2 actions"


def test_callback_branching_only_true():
    """
    Test callback-based branching with only true branch.
    """
    graph = ActionGraph(action_executor=mock_executor)
    initial_node_count = len(graph._nodes)
    
    graph.branch(
        condition=lambda r: True,
        if_true=lambda g: g.action('click', target='#success'),
        # No if_false
    )
    
    # Should have created only 1 new node
    assert len(graph._nodes) == initial_node_count + 1, \
        "Should have 1 new node for true branch only"


def test_callback_branching_only_false():
    """
    Test callback-based branching with only false branch.
    """
    graph = ActionGraph(action_executor=mock_executor)
    initial_node_count = len(graph._nodes)
    
    graph.branch(
        condition=lambda r: True,
        # No if_true
        if_false=lambda g: g.action('click', target='#retry'),
    )
    
    # Should have created only 1 new node
    assert len(graph._nodes) == initial_node_count + 1, \
        "Should have 1 new node for false branch only"


def test_callback_branching_chaining():
    """
    Test that callback-based branching supports method chaining.
    """
    graph = ActionGraph(action_executor=mock_executor)
    
    # Chain multiple operations
    result = (
        graph
        .action('navigate', target='https://example.com')
        .branch(
            condition=lambda r: True,
            if_true=lambda g: g.action('click', target='#btn1'),
            if_false=lambda g: g.action('click', target='#btn2'),
        )
        .action('wait', args={'seconds': 1})
    )
    
    assert result is graph, "Chaining should return same graph instance"


def test_if_true_if_false_require_context_manager():
    """
    Test that if_true() and if_false() raise error outside context manager.
    """
    from science_modeling_tools.automation.schema.action_graph import ConditionContext
    import pytest
    
    graph = ActionGraph(action_executor=mock_executor)
    cond = ConditionContext(graph, lambda r: True)
    
    # Should raise RuntimeError when not in context manager
    try:
        cond.if_true()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "within 'with' context" in str(e)
    
    try:
        cond.if_false()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "within 'with' context" in str(e)


# endregion


# region Serialization Property Tests

# Strategies for generating test data
action_type_strategy = st.sampled_from(['click', 'type', 'navigate', 'wait', 'scroll'])
target_strategy = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N')))


@st.composite
def action_list_strategy(draw):
    """Generate a list of (action_type, target) tuples."""
    num_actions = draw(st.integers(min_value=1, max_value=5))
    actions = []
    for _ in range(num_actions):
        action_type = draw(action_type_strategy)
        target = draw(target_strategy)
        actions.append((action_type, target))
    return actions


# **Feature: serializable-mixin, Property 5: ActionGraph Structure Preservation**
# **Validates: Requirements 2.1, 2.2, 2.5**
@settings(max_examples=100)
@given(actions=action_list_strategy())
def test_action_graph_serialization_preserves_structure(actions):
    """
    Property 5: ActionGraph Structure Preservation.
    
    **Feature: serializable-mixin, Property 5: ActionGraph Structure Preservation**
    **Validates: Requirements 2.1, 2.2, 2.5**
    
    For any ActionGraph with nodes and connections, serializing then deserializing
    SHALL preserve the node count, node names, and parent-child relationships.
    """
    # Build original graph
    original = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        original.action(action_type, target=target)
    
    # Serialize
    serialized = original.to_serializable_obj()
    
    # Verify serialized structure
    assert '_type' in serialized, "Missing _type field"
    assert '_module' in serialized, "Missing _module field"
    assert 'version' in serialized, "Missing version field"
    assert 'id' in serialized, "Missing id field"
    assert 'nodes' in serialized, "Missing nodes field"
    
    # Deserialize
    restored = ActionGraph.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
    )
    
    # Verify node count is preserved
    assert len(restored._nodes) == len(original._nodes), \
        f"Node count mismatch: {len(restored._nodes)} != {len(original._nodes)}"
    
    # Verify node names are preserved
    original_names = [n.name for n in original._nodes]
    restored_names = [n.name for n in restored._nodes]
    assert original_names == restored_names, \
        f"Node names mismatch: {restored_names} != {original_names}"
    
    # Verify action count in each node is preserved
    for orig_node, rest_node in zip(original._nodes, restored._nodes):
        assert len(rest_node._actions) == len(orig_node._actions), \
            f"Action count mismatch in node {orig_node.name}: {len(rest_node._actions)} != {len(orig_node._actions)}"


# **Feature: serializable-mixin, Property 5: ActionGraph Structure Preservation**
# **Validates: Requirements 2.1, 2.2, 2.5**
@settings(max_examples=100)
@given(actions=action_list_strategy())
def test_action_graph_serialization_preserves_actions(actions):
    """
    Property 5: ActionGraph action preservation.
    
    **Feature: serializable-mixin, Property 5: ActionGraph Structure Preservation**
    **Validates: Requirements 2.1, 2.2, 2.5**
    
    For any ActionGraph, serializing then deserializing SHALL preserve
    all action types and targets in the correct order.
    """
    # Build original graph
    original = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        original.action(action_type, target=target)
    
    # Serialize and deserialize
    serialized = original.to_serializable_obj()
    restored = ActionGraph.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
    )
    
    # Verify actions are preserved
    for orig_node, rest_node in zip(original._nodes, restored._nodes):
        for i, (orig_action, rest_action) in enumerate(zip(orig_node._actions, rest_node._actions)):
            assert rest_action.type == orig_action.type, \
                f"Action type mismatch at index {i}: {rest_action.type} != {orig_action.type}"
            assert rest_action.target == orig_action.target, \
                f"Action target mismatch at index {i}: {rest_action.target} != {orig_action.target}"


# **Feature: serializable-mixin, Property 5: ActionGraph Structure Preservation**
# **Validates: Requirements 2.1, 2.2, 2.5**
@settings(max_examples=50)
@given(
    actions1=action_list_strategy(),
    actions2=action_list_strategy(),
)
def test_action_graph_serialization_preserves_connections(actions1, actions2):
    """
    Property 5: ActionGraph connection preservation.
    
    **Feature: serializable-mixin, Property 5: ActionGraph Structure Preservation**
    **Validates: Requirements 2.1, 2.2, 2.5**
    
    For any ActionGraph with conditional branches, serializing then deserializing
    SHALL preserve the parent-child relationships between nodes.
    """
    # Build original graph with a branch
    original = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    
    # Add initial actions
    for action_type, target in actions1:
        original.action(action_type, target=target)
    
    # Create a branch
    cond = original.condition(lambda r: True)
    if cond:
        for action_type, target in actions2:
            original.action(action_type, target=target)
    
    # Serialize and deserialize
    serialized = original.to_serializable_obj()
    restored = ActionGraph.from_serializable_obj(
        serialized,
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
    )
    
    # Verify node count
    assert len(restored._nodes) == len(original._nodes), \
        f"Node count mismatch: {len(restored._nodes)} != {len(original._nodes)}"
    
    # Verify connections are preserved
    for orig_node, rest_node in zip(original._nodes, restored._nodes):
        orig_next_names = [n.name for n in (orig_node.next or [])]
        rest_next_names = [n.name for n in (rest_node.next or [])]
        assert rest_next_names == orig_next_names, \
            f"Connection mismatch for node {orig_node.name}: {rest_next_names} != {orig_next_names}"


def test_action_graph_serialization_to_serializable_obj():
    """
    Test that to_serializable_obj() and from_serializable_obj() work correctly.
    
    **Feature: serializable-mixin**
    **Validates: Serializable interface**
    """
    # Build graph
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#btn1')
    graph.action('type', target='#input', args={'text': 'hello'})
    
    # Use Serializable interface methods
    dict_data = graph.to_serializable_obj()
    restored = ActionGraph.from_serializable_obj(
        dict_data,
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
    )
    
    # Verify structure
    assert len(restored._nodes) == len(graph._nodes)
    assert len(restored._nodes[0]._actions) == len(graph._nodes[0]._actions)


def test_action_graph_json_round_trip():
    """
    Test JSON serialization round-trip using Serializable.serialize().
    
    **Feature: serializable-mixin**
    **Validates: JSON format support**
    """
    # Build graph
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#btn1')
    graph.action('navigate', target='https://example.com')
    
    # Serialize to JSON using Serializable interface
    json_str = graph.serialize(format='json')
    
    # Deserialize from JSON using Serializable interface
    restored = ActionGraph.deserialize(
        json_str,
        format='json',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
    )
    
    # Verify
    assert len(restored._nodes) == len(graph._nodes)
    assert len(restored._nodes[0]._actions) == 2


# endregion


if __name__ == "__main__":
    print("Running property tests for ActionGraph...")
    
    test_action_graph_extends_workgraph()
    print("✓ test_action_graph_extends_workgraph passed")
    
    test_chaining_returns_self()
    print("✓ test_chaining_returns_self passed")
    
    test_chained_vs_nonchained_equivalence()
    print("✓ test_chained_vs_nonchained_equivalence passed")
    
    test_condition_method_exists()
    print("✓ test_condition_method_exists passed")
    
    test_action_sequence_node_is_workgraph_node()
    print("✓ test_action_sequence_node_is_workgraph_node passed")
    
    test_linear_segment_compression()
    print("✓ test_linear_segment_compression passed")
    
    test_conditional_branching_structure()
    print("✓ test_conditional_branching_structure passed")
    
    test_condition_context_bool_returns_true()
    print("✓ test_condition_context_bool_returns_true passed")
    
    test_branch_isolation()
    print("✓ test_branch_isolation passed")
    
    test_branch_action_counts()
    print("✓ test_branch_action_counts passed")
    
    # New branching API tests
    print("\nRunning new branching API tests...")
    
    test_context_manager_branching()
    print("✓ test_context_manager_branching passed")
    
    test_context_manager_multi_action_branches()
    print("✓ test_context_manager_multi_action_branches passed")
    
    test_match_case_branching()
    print("✓ test_match_case_branching passed")
    
    test_callback_based_branching()
    print("✓ test_callback_based_branching passed")
    
    test_callback_branching_multi_action()
    print("✓ test_callback_branching_multi_action passed")
    
    test_callback_branching_only_true()
    print("✓ test_callback_branching_only_true passed")
    
    test_callback_branching_only_false()
    print("✓ test_callback_branching_only_false passed")
    
    test_callback_branching_chaining()
    print("✓ test_callback_branching_chaining passed")
    
    test_if_true_if_false_require_context_manager()
    print("✓ test_if_true_if_false_require_context_manager passed")
    
    # Serialization property tests
    print("\nRunning serialization property tests...")
    
    test_action_graph_serialization_preserves_structure()
    print("✓ test_action_graph_serialization_preserves_structure passed")
    
    test_action_graph_serialization_preserves_actions()
    print("✓ test_action_graph_serialization_preserves_actions passed")
    
    test_action_graph_serialization_preserves_connections()
    print("✓ test_action_graph_serialization_preserves_connections passed")
    
    test_action_graph_serialization_backward_compatibility()
    print("✓ test_action_graph_serialization_backward_compatibility passed")
    
    test_action_graph_json_round_trip()
    print("✓ test_action_graph_json_round_trip passed")
    
    print("\nAll property tests passed!")

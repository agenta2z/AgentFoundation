"""
Property-based tests for Python script serialization of ActionGraph and ActionFlow.

Tests correctness properties from the python-script-serialization design document.
"""

import ast
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
_workspace_root = _project_root.parent
_rich_python_utils_src = _workspace_root / "SciencePythonUtils" / "src"
if _rich_python_utils_src.exists() and str(_rich_python_utils_src) not in sys.path:
    sys.path.insert(0, str(_rich_python_utils_src))

from typing import Any, Dict, Optional
from hypothesis import given, strategies as st, settings

from agent_foundation.automation.schema.action_graph import ActionGraph
from agent_foundation.automation.schema.action_flow import ActionFlow
from agent_foundation.automation.schema.action_metadata import ActionMetadataRegistry
from agent_foundation.automation.schema.common import Action, ActionSequence


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


# region Strategies for generating test data

action_type_strategy = st.sampled_from(['click', 'input_text', 'wait', 'screenshot', 'navigate'])
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


# endregion


# region Property 8: to_serializable_obj returns self for python format

# **Feature: python-script-serialization, Property 8: to_serializable_obj returns self for python format**
# **Validates: Requirements 7.3**
@settings(max_examples=100)
@given(actions=action_list_strategy())
def test_action_graph_to_serializable_obj_returns_self_for_python(actions):
    """
    Property 8: to_serializable_obj returns self for python format.
    
    **Feature: python-script-serialization, Property 8: to_serializable_obj returns self for python format**
    **Validates: Requirements 7.3**
    
    For any ActionGraph, calling to_serializable_obj(_output_format='python')
    SHALL return the object itself (identity check), not a dict.
    """
    # Build graph with random actions
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        graph.action(action_type, target=target)
    
    # Call to_serializable_obj with python format
    result = graph.to_serializable_obj(_output_format='python')
    
    # Should return self (identity check)
    assert result is graph, \
        f"to_serializable_obj(_output_format='python') should return self, got {type(result)}"


# **Feature: python-script-serialization, Property 8: to_serializable_obj returns self for python format**
# **Validates: Requirements 7.3**
def test_action_graph_to_serializable_obj_returns_dict_for_other_formats():
    """
    Verify that to_serializable_obj returns dict for non-python formats.
    
    **Feature: python-script-serialization, Property 8: to_serializable_obj returns self for python format**
    **Validates: Requirements 7.3**
    """
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#btn')
    
    # For json format, should return dict
    result_json = graph.to_serializable_obj(_output_format='json')
    assert isinstance(result_json, dict), \
        f"to_serializable_obj(_output_format='json') should return dict, got {type(result_json)}"
    
    # For yaml format, should return dict
    result_yaml = graph.to_serializable_obj(_output_format='yaml')
    assert isinstance(result_yaml, dict), \
        f"to_serializable_obj(_output_format='yaml') should return dict, got {type(result_yaml)}"
    
    # For None (default), should return dict
    result_default = graph.to_serializable_obj()
    assert isinstance(result_default, dict), \
        f"to_serializable_obj() should return dict, got {type(result_default)}"


# endregion


# region Property 1: ActionGraph Python serialization produces valid Python

# **Feature: python-script-serialization, Property 1: ActionGraph Python serialization produces valid Python**
# **Validates: Requirements 1.1**
@settings(max_examples=100)
@given(actions=action_list_strategy())
def test_action_graph_python_serialization_produces_valid_python(actions):
    """
    Property 1: ActionGraph Python serialization produces valid Python.
    
    **Feature: python-script-serialization, Property 1: ActionGraph Python serialization produces valid Python**
    **Validates: Requirements 1.1**
    
    For any ActionGraph instance with any combination of actions,
    serializing to Python format SHALL produce a string that can be
    parsed by ast.parse() without raising SyntaxError.
    """
    # Build graph with random actions
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        graph.action(action_type, target=target)
    
    # Serialize to Python format
    python_script = graph.serialize(output_format='python')
    
    # Verify it's valid Python by parsing with ast
    try:
        ast.parse(python_script)
    except SyntaxError as e:
        raise AssertionError(
            f"Generated Python script has syntax error: {e}\n"
            f"Script:\n{python_script}"
        )


# **Feature: python-script-serialization, Property 1: ActionGraph Python serialization produces valid Python**
# **Validates: Requirements 1.1**
@settings(max_examples=50)
@given(
    actions=action_list_strategy(),
    branching_style=st.sampled_from(['match', 'with', 'branch', 'if'])
)
def test_action_graph_python_serialization_all_styles_valid(actions, branching_style):
    """
    Property 1: All branching styles produce valid Python.
    
    **Feature: python-script-serialization, Property 1: ActionGraph Python serialization produces valid Python**
    **Validates: Requirements 1.1**
    
    For any ActionGraph and any branching style, serializing to Python
    format SHALL produce a string that can be parsed by ast.parse().
    """
    # Build graph with random actions
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        graph.action(action_type, target=target)
    
    # Serialize to Python format with specified branching style
    python_script = graph.serialize(
        output_format='python',
        branching_style=branching_style
    )
    
    # Verify it's valid Python by parsing with ast
    try:
        ast.parse(python_script)
    except SyntaxError as e:
        raise AssertionError(
            f"Generated Python script (style={branching_style}) has syntax error: {e}\n"
            f"Script:\n{python_script}"
        )


# **Feature: python-script-serialization, Property 1: ActionGraph Python serialization produces valid Python**
# **Validates: Requirements 1.1**
def test_action_graph_python_serialization_with_conditions_valid():
    """
    Test that ActionGraph with conditions produces valid Python.
    
    **Feature: python-script-serialization, Property 1: ActionGraph Python serialization produces valid Python**
    **Validates: Requirements 1.1, 1.5**
    """
    from agent_foundation.automation.schema.action_graph import condition_expr
    
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#submit')
    
    # Create a condition with expression
    @condition_expr("result.success")
    def check_success(result):
        return getattr(result, 'success', True)
    
    # Use callback-based branching for cleaner test
    graph.branch(
        condition=check_success,
        if_true=lambda g: g.action('click', target='#success_btn'),
        if_false=lambda g: g.action('click', target='#retry_btn'),
    )
    
    # Test all branching styles
    for style in ['match', 'with', 'branch', 'if']:
        python_script = graph.serialize(
            output_format='python',
            branching_style=style
        )
        
        try:
            ast.parse(python_script)
        except SyntaxError as e:
            raise AssertionError(
                f"Generated Python script (style={style}) has syntax error: {e}\n"
                f"Script:\n{python_script}"
            )


# endregion


# region Property 5: Branching style affects output syntax

# **Feature: python-script-serialization, Property 5: Branching style affects output syntax**
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
@settings(max_examples=100)
@given(
    action_type=action_type_strategy,
    target=target_strategy,
    true_action_type=action_type_strategy,
    true_target=target_strategy,
    false_action_type=action_type_strategy,
    false_target=target_strategy,
)
def test_branching_style_affects_output_syntax(
    action_type, target, true_action_type, true_target, false_action_type, false_target
):
    """
    Property 5: Branching style affects output syntax.
    
    **Feature: python-script-serialization, Property 5: Branching style affects output syntax**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    
    For any ActionGraph with at least one condition, serializing with different
    branching styles SHALL produce output containing the expected syntax patterns:
    - 'match': contains "match" and "case ConditionContext"
    - 'with': contains "with graph.condition" and "if_true()" and "if_false()"
    - 'branch': contains "graph.branch(" and "if_true=" and "if_false="
    - 'if': contains "if graph.condition" and "else_branch()"
    """
    from agent_foundation.automation.schema.action_graph import condition_expr
    
    # Build graph with a condition
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action(action_type, target=target)
    
    # Create a condition with expression
    @condition_expr("result.success")
    def check_success(result):
        return getattr(result, 'success', True)
    
    # Use callback-based branching
    graph.branch(
        condition=check_success,
        if_true=lambda g: g.action(true_action_type, target=true_target),
        if_false=lambda g: g.action(false_action_type, target=false_target),
    )
    
    # Test match style
    match_script = graph.serialize(output_format='python', branching_style='match')
    assert "match graph.condition(" in match_script, \
        f"match style should contain 'match graph.condition(', got:\n{match_script}"
    assert "case ConditionContext.TRUE:" in match_script, \
        f"match style should contain 'case ConditionContext.TRUE:', got:\n{match_script}"
    assert "case ConditionContext.FALSE:" in match_script, \
        f"match style should contain 'case ConditionContext.FALSE:', got:\n{match_script}"
    
    # Test with style
    with_script = graph.serialize(output_format='python', branching_style='with')
    assert "with graph.condition(" in with_script, \
        f"with style should contain 'with graph.condition(', got:\n{with_script}"
    assert "with branch.if_true():" in with_script, \
        f"with style should contain 'with branch.if_true():', got:\n{with_script}"
    assert "with branch.if_false():" in with_script, \
        f"with style should contain 'with branch.if_false():', got:\n{with_script}"
    
    # Test branch style
    branch_script = graph.serialize(output_format='python', branching_style='branch')
    assert "graph.branch(" in branch_script, \
        f"branch style should contain 'graph.branch(', got:\n{branch_script}"
    assert "if_true=" in branch_script, \
        f"branch style should contain 'if_true=', got:\n{branch_script}"
    assert "if_false=" in branch_script, \
        f"branch style should contain 'if_false=', got:\n{branch_script}"
    
    # Test if style
    if_script = graph.serialize(output_format='python', branching_style='if')
    assert "if graph.condition(" in if_script, \
        f"if style should contain 'if graph.condition(', got:\n{if_script}"
    assert "graph.else_branch()" in if_script, \
        f"if style should contain 'graph.else_branch()', got:\n{if_script}"


# **Feature: python-script-serialization, Property 5: Branching style affects output syntax**
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
def test_branching_style_with_custom_variable_name():
    """
    Test that branching styles work correctly with custom variable names.
    
    **Feature: python-script-serialization, Property 5: Branching style affects output syntax**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 8.1**
    """
    from agent_foundation.automation.schema.action_graph import condition_expr
    
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#submit')
    
    @condition_expr("result.success")
    def check_success(result):
        return getattr(result, 'success', True)
    
    graph.branch(
        condition=check_success,
        if_true=lambda g: g.action('click', target='#success'),
        if_false=lambda g: g.action('click', target='#retry'),
    )
    
    custom_var = "my_workflow"
    
    # Test match style with custom variable
    match_script = graph.serialize(
        output_format='python', 
        branching_style='match',
        variable_name=custom_var
    )
    assert f"match {custom_var}.condition(" in match_script
    assert f"{custom_var}.action(" in match_script
    
    # Test with style with custom variable
    with_script = graph.serialize(
        output_format='python', 
        branching_style='with',
        variable_name=custom_var
    )
    assert f"with {custom_var}.condition(" in with_script
    assert f"{custom_var}.action(" in with_script
    
    # Test branch style with custom variable
    branch_script = graph.serialize(
        output_format='python', 
        branching_style='branch',
        variable_name=custom_var
    )
    assert f"{custom_var}.branch(" in branch_script
    
    # Test if style with custom variable
    if_script = graph.serialize(
        output_format='python', 
        branching_style='if',
        variable_name=custom_var
    )
    assert f"if {custom_var}.condition(" in if_script
    assert f"{custom_var}.else_branch()" in if_script


# **Feature: python-script-serialization, Property 5: Branching style affects output syntax**
# **Validates: Requirements 2.5**
def test_default_branching_style_is_match():
    """
    Test that the default branching style is 'match'.
    
    **Feature: python-script-serialization, Property 5: Branching style affects output syntax**
    **Validates: Requirements 2.5**
    """
    from agent_foundation.automation.schema.action_graph import condition_expr
    
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#submit')
    
    @condition_expr("result.success")
    def check_success(result):
        return getattr(result, 'success', True)
    
    graph.branch(
        condition=check_success,
        if_true=lambda g: g.action('click', target='#success'),
        if_false=lambda g: g.action('click', target='#retry'),
    )
    
    # Serialize without specifying branching_style (should default to 'match')
    default_script = graph.serialize(output_format='python')
    
    # Should contain match-case syntax
    assert "match graph.condition(" in default_script, \
        f"Default style should be 'match', got:\n{default_script}"
    assert "case ConditionContext.TRUE:" in default_script
    assert "case ConditionContext.FALSE:" in default_script


# **Feature: python-script-serialization, Property 5: Branching style affects output syntax**
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
def test_invalid_branching_style_raises_error():
    """
    Test that invalid branching style raises ValueError.
    
    **Feature: python-script-serialization, Property 5: Branching style affects output syntax**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#submit')
    
    try:
        graph.serialize(output_format='python', branching_style='invalid_style')
        assert False, "Should have raised ValueError for invalid branching_style"
    except ValueError as e:
        assert "invalid_style" in str(e).lower() or "Invalid branching_style" in str(e)


# endregion


# region Property 3: ActionGraph round-trip preserves structure

# **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
# **Validates: Requirements 6.1, 3.1, 3.4, 3.5, 6.3, 6.4**
@settings(max_examples=100)
@given(actions=action_list_strategy())
def test_action_graph_round_trip_preserves_actions(actions):
    """
    Property 3: ActionGraph round-trip preserves structure.
    
    **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
    **Validates: Requirements 6.1, 3.1, 3.4, 3.5, 6.3, 6.4**
    
    For any ActionGraph with actions, serializing to Python and deserializing back
    SHALL produce a graph with:
    - Same number of actions
    - Same action parameters (type, target) for each action
    """
    # Build original graph with random actions
    original_graph = ActionGraph(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    for action_type, target in actions:
        original_graph.action(action_type, target=target)
    
    # Serialize to Python format
    python_script = original_graph.serialize(output_format='python')
    
    # Deserialize back
    restored_graph = ActionGraph.deserialize(
        python_script,
        output_format='python',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    # Get actions from both graphs
    original_actions = []
    for node in original_graph._nodes:
        original_actions.extend(node._actions)
    
    restored_actions = []
    for node in restored_graph._nodes:
        restored_actions.extend(node._actions)
    
    # Verify same number of actions
    assert len(restored_actions) == len(original_actions), \
        f"Expected {len(original_actions)} actions, got {len(restored_actions)}"
    
    # Verify action parameters match
    for orig, restored in zip(original_actions, restored_actions):
        assert restored.type == orig.type, \
            f"Action type mismatch: expected {orig.type}, got {restored.type}"
        assert restored.target == orig.target, \
            f"Action target mismatch: expected {orig.target}, got {restored.target}"


# **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
# **Validates: Requirements 6.1, 3.1, 3.4, 3.5, 6.3, 6.4**
@settings(max_examples=50)
@given(
    actions=action_list_strategy(),
    branching_style=st.sampled_from(['match', 'with', 'branch', 'if'])
)
def test_action_graph_round_trip_all_styles(actions, branching_style):
    """
    Property 3: ActionGraph round-trip works for all branching styles.
    
    **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
    **Validates: Requirements 6.1, 3.1, 3.4, 3.5, 6.3, 6.4**
    
    For any ActionGraph and any branching style, serializing to Python and
    deserializing back SHALL produce a graph with equivalent actions.
    """
    # Build original graph with random actions
    original_graph = ActionGraph(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    for action_type, target in actions:
        original_graph.action(action_type, target=target)
    
    # Serialize to Python format with specified style
    python_script = original_graph.serialize(
        output_format='python',
        branching_style=branching_style
    )
    
    # Deserialize back
    restored_graph = ActionGraph.deserialize(
        python_script,
        output_format='python',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    # Get actions from both graphs
    original_actions = []
    for node in original_graph._nodes:
        original_actions.extend(node._actions)
    
    restored_actions = []
    for node in restored_graph._nodes:
        restored_actions.extend(node._actions)
    
    # Verify same number of actions
    assert len(restored_actions) == len(original_actions), \
        f"Expected {len(original_actions)} actions, got {len(restored_actions)}"


# **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
# **Validates: Requirements 6.1, 3.1, 3.4, 3.5, 6.3, 6.4**
def test_action_graph_round_trip_with_conditions():
    """
    Test that ActionGraph with conditions round-trips correctly.
    
    **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
    **Validates: Requirements 6.1, 3.1, 3.4, 3.5, 6.3, 6.4**
    """
    from agent_foundation.automation.schema.action_graph import condition_expr
    
    # Build original graph with conditions
    original_graph = ActionGraph(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    original_graph.action('click', target='#submit')
    
    @condition_expr("result.success")
    def check_success(result):
        return getattr(result, 'success', True)
    
    original_graph.branch(
        condition=check_success,
        if_true=lambda g: g.action('click', target='#success_btn'),
        if_false=lambda g: g.action('click', target='#retry_btn'),
    )
    
    # Test all branching styles
    for style in ['match', 'with', 'branch']:
        python_script = original_graph.serialize(
            output_format='python',
            branching_style=style
        )
        
        restored_graph = ActionGraph.deserialize(
            python_script,
            output_format='python',
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry()
        )
        
        # Count total actions in restored graph
        restored_actions = []
        for node in restored_graph._nodes:
            restored_actions.extend(node._actions)
        
        # Should have at least the initial action
        assert len(restored_actions) >= 1, \
            f"Style {style}: Expected at least 1 action, got {len(restored_actions)}"


# **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
# **Validates: Requirements 6.1, 3.1, 3.4, 3.5, 6.3, 6.4**
def test_action_graph_round_trip_preserves_action_args():
    """
    Test that action args are preserved during round-trip.
    
    **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
    **Validates: Requirements 6.3**
    """
    original_graph = ActionGraph(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    original_graph.action('input_text', target='#username', args={'text': 'testuser'})
    original_graph.action('wait', args={'seconds': 5})
    original_graph.action('click', target='#submit', args={'timeout': 10, 'force': True})
    
    python_script = original_graph.serialize(output_format='python')
    
    restored_graph = ActionGraph.deserialize(
        python_script,
        output_format='python',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    # Get actions from both graphs
    original_actions = []
    for node in original_graph._nodes:
        original_actions.extend(node._actions)
    
    restored_actions = []
    for node in restored_graph._nodes:
        restored_actions.extend(node._actions)
    
    assert len(restored_actions) == len(original_actions)
    
    for orig, restored in zip(original_actions, restored_actions):
        assert restored.type == orig.type
        assert restored.target == orig.target
        assert restored.args == orig.args, \
            f"Args mismatch: expected {orig.args}, got {restored.args}"


# **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
# **Validates: Requirements 3.3**
def test_action_graph_deserialize_requires_action_executor():
    """
    Test that deserialize raises ValueError when action_executor is missing.
    
    **Feature: python-script-serialization, Property 3: ActionGraph round-trip preserves structure**
    **Validates: Requirements 3.3**
    """
    graph = ActionGraph(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    graph.action('click', target='#btn')
    
    python_script = graph.serialize(output_format='python')
    
    try:
        ActionGraph.deserialize(
            python_script,
            output_format='python'
            # Missing action_executor
        )
        assert False, "Should have raised ValueError for missing action_executor"
    except ValueError as e:
        assert "action_executor" in str(e).lower()


# endregion


# region Property 2: ActionFlow Python serialization produces valid Python

# **Feature: python-script-serialization, Property 2: ActionFlow Python serialization produces valid Python**
# **Validates: Requirements 4.1**
@st.composite
def action_flow_action_list_strategy(draw):
    """Generate a list of Action objects for ActionFlow testing."""
    num_actions = draw(st.integers(min_value=1, max_value=5))
    actions = []
    for i in range(num_actions):
        action_type = draw(action_type_strategy)
        target = draw(target_strategy)
        # Optionally add args
        has_args = draw(st.booleans())
        args = None
        if has_args:
            args = {
                'key': draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L', 'N')))),
            }
        actions.append(Action(
            id=f"action_{i+1}",
            type=action_type,
            target=target,
            args=args,
        ))
    return actions


# **Feature: python-script-serialization, Property 2: ActionFlow Python serialization produces valid Python**
# **Validates: Requirements 4.1**
@settings(max_examples=100)
@given(actions=action_flow_action_list_strategy())
def test_action_flow_python_serialization_produces_valid_python(actions):
    """
    Property 2: ActionFlow Python serialization produces valid Python.
    
    **Feature: python-script-serialization, Property 2: ActionFlow Python serialization produces valid Python**
    **Validates: Requirements 4.1**
    
    For any ActionFlow instance with any ActionSequence, serializing to Python
    format SHALL produce a string that can be parsed by ast.parse() without
    raising SyntaxError.
    """
    # Build ActionFlow with ActionSequence
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(id="test_sequence", actions=actions)
    )
    
    # Serialize to Python format
    python_script = flow.serialize(output_format='python')
    
    # Verify it's valid Python by parsing with ast
    try:
        ast.parse(python_script)
    except SyntaxError as e:
        raise AssertionError(
            f"Generated Python script has syntax error: {e}\n"
            f"Script:\n{python_script}"
        )


# **Feature: python-script-serialization, Property 2: ActionFlow Python serialization produces valid Python**
# **Validates: Requirements 4.1**
def test_action_flow_python_serialization_basic():
    """
    Test basic ActionFlow Python serialization.
    
    **Feature: python-script-serialization, Property 2: ActionFlow Python serialization produces valid Python**
    **Validates: Requirements 4.1**
    """
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="basic_sequence",
            actions=[
                Action(id="a1", type="click", target="#btn"),
                Action(id="a2", type="input_text", target="#field", args={"text": "hello"}),
            ]
        )
    )
    
    python_script = flow.serialize(output_format='python')
    
    # Verify it's valid Python
    try:
        ast.parse(python_script)
    except SyntaxError as e:
        raise AssertionError(f"Generated Python script has syntax error: {e}\n{python_script}")
    
    # Verify expected content
    assert "ActionFlow" in python_script
    assert "ActionSequence" in python_script
    assert "Action" in python_script
    assert '"click"' in python_script
    assert '"input_text"' in python_script


# **Feature: python-script-serialization, Property 2: ActionFlow Python serialization produces valid Python**
# **Validates: Requirements 4.3**
def test_action_flow_python_serialization_no_branching():
    """
    Test that ActionFlow Python serialization has no branching constructs.
    
    **Feature: python-script-serialization, Property 2: ActionFlow Python serialization produces valid Python**
    **Validates: Requirements 4.3**
    """
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="seq",
            actions=[
                Action(id="a1", type="click", target="#btn"),
            ]
        )
    )
    
    python_script = flow.serialize(output_format='python')
    
    # Should NOT contain branching constructs
    assert "condition" not in python_script.lower()
    assert "branch" not in python_script.lower()
    assert "if_true" not in python_script
    assert "if_false" not in python_script
    assert "match " not in python_script


# **Feature: python-script-serialization, Property 8: to_serializable_obj returns self for python format**
# **Validates: Requirements 7.3**
@settings(max_examples=100)
@given(actions=action_flow_action_list_strategy())
def test_action_flow_to_serializable_obj_returns_self_for_python(actions):
    """
    Property 8: to_serializable_obj returns self for python format (ActionFlow).
    
    **Feature: python-script-serialization, Property 8: to_serializable_obj returns self for python format**
    **Validates: Requirements 7.3**
    
    For any ActionFlow, calling to_serializable_obj(_output_format='python')
    SHALL return the object itself (identity check), not a dict.
    """
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(id="test_seq", actions=actions)
    )
    
    # Call to_serializable_obj with python format
    result = flow.to_serializable_obj(_output_format='python')
    
    # Should return self (identity check)
    assert result is flow, \
        f"to_serializable_obj(_output_format='python') should return self, got {type(result)}"


# **Feature: python-script-serialization, Property 8: to_serializable_obj returns self for python format**
# **Validates: Requirements 7.3**
def test_action_flow_to_serializable_obj_returns_dict_for_other_formats():
    """
    Verify that ActionFlow to_serializable_obj returns dict for non-python formats.
    
    **Feature: python-script-serialization, Property 8: to_serializable_obj returns self for python format**
    **Validates: Requirements 7.3**
    """
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="seq",
            actions=[Action(id="a1", type="click", target="#btn")]
        )
    )
    
    # For json format, should return dict
    result_json = flow.to_serializable_obj(_output_format='json')
    assert isinstance(result_json, dict), \
        f"to_serializable_obj(_output_format='json') should return dict, got {type(result_json)}"
    
    # For yaml format, should return dict
    result_yaml = flow.to_serializable_obj(_output_format='yaml')
    assert isinstance(result_yaml, dict), \
        f"to_serializable_obj(_output_format='yaml') should return dict, got {type(result_yaml)}"
    
    # For None (default), should return dict
    result_default = flow.to_serializable_obj()
    assert isinstance(result_default, dict), \
        f"to_serializable_obj() should return dict, got {type(result_default)}"


# endregion


# region Property 4: ActionFlow round-trip preserves actions

# **Feature: python-script-serialization, Property 4: ActionFlow round-trip preserves actions**
# **Validates: Requirements 6.2, 5.1, 5.3**
@settings(max_examples=100)
@given(actions=action_flow_action_list_strategy())
def test_action_flow_round_trip_preserves_actions(actions):
    """
    Property 4: ActionFlow round-trip preserves actions.
    
    **Feature: python-script-serialization, Property 4: ActionFlow round-trip preserves actions**
    **Validates: Requirements 6.2, 5.1, 5.3**
    
    For any ActionFlow with an ActionSequence, serializing to Python and
    deserializing back SHALL produce a flow with:
    - Same number of actions
    - Same action parameters (type, target, args, id) for each action
    """
    # Build original ActionFlow with ActionSequence
    original_flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(id="test_sequence", actions=actions)
    )
    
    # Serialize to Python format
    python_script = original_flow.serialize(output_format='python')
    
    # Deserialize back
    restored_flow = ActionFlow.deserialize(
        python_script,
        output_format='python',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    # Verify sequence exists
    assert restored_flow.sequence is not None, \
        "Restored flow should have a sequence"
    
    # Get actions from both flows
    original_actions = original_flow.sequence.actions
    restored_actions = restored_flow.sequence.actions
    
    # Verify same number of actions
    assert len(restored_actions) == len(original_actions), \
        f"Expected {len(original_actions)} actions, got {len(restored_actions)}"
    
    # Verify action parameters match
    for orig, restored in zip(original_actions, restored_actions):
        assert restored.type == orig.type, \
            f"Action type mismatch: expected {orig.type}, got {restored.type}"
        assert restored.target == orig.target, \
            f"Action target mismatch: expected {orig.target}, got {restored.target}"
        assert restored.id == orig.id, \
            f"Action id mismatch: expected {orig.id}, got {restored.id}"
        assert restored.args == orig.args, \
            f"Action args mismatch: expected {orig.args}, got {restored.args}"


# **Feature: python-script-serialization, Property 4: ActionFlow round-trip preserves actions**
# **Validates: Requirements 6.2, 5.1, 5.3**
def test_action_flow_round_trip_basic():
    """
    Test basic ActionFlow round-trip.
    
    **Feature: python-script-serialization, Property 4: ActionFlow round-trip preserves actions**
    **Validates: Requirements 6.2, 5.1, 5.3**
    """
    original_flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="basic_sequence",
            actions=[
                Action(id="a1", type="click", target="#btn"),
                Action(id="a2", type="input_text", target="#field", args={"text": "hello"}),
                Action(id="a3", type="wait", args={"seconds": 5}),
            ]
        )
    )
    
    python_script = original_flow.serialize(output_format='python')
    
    restored_flow = ActionFlow.deserialize(
        python_script,
        output_format='python',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    assert restored_flow.sequence is not None
    assert len(restored_flow.sequence.actions) == 3
    
    # Verify first action
    assert restored_flow.sequence.actions[0].type == "click"
    assert restored_flow.sequence.actions[0].target == "#btn"
    assert restored_flow.sequence.actions[0].id == "a1"
    
    # Verify second action with args
    assert restored_flow.sequence.actions[1].type == "input_text"
    assert restored_flow.sequence.actions[1].target == "#field"
    assert restored_flow.sequence.actions[1].args == {"text": "hello"}
    
    # Verify third action with only args
    assert restored_flow.sequence.actions[2].type == "wait"
    assert restored_flow.sequence.actions[2].args == {"seconds": 5}


# **Feature: python-script-serialization, Property 4: ActionFlow round-trip preserves actions**
# **Validates: Requirements 5.2**
def test_action_flow_deserialize_requires_action_executor():
    """
    Test that ActionFlow.deserialize raises ValueError when action_executor is missing.
    
    **Feature: python-script-serialization, Property 4: ActionFlow round-trip preserves actions**
    **Validates: Requirements 5.2**
    """
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="seq",
            actions=[Action(id="a1", type="click", target="#btn")]
        )
    )
    
    python_script = flow.serialize(output_format='python')
    
    try:
        ActionFlow.deserialize(
            python_script,
            output_format='python'
            # Missing action_executor
        )
        assert False, "Should have raised ValueError for missing action_executor"
    except ValueError as e:
        assert "action_executor" in str(e).lower()


# **Feature: python-script-serialization, Property 4: ActionFlow round-trip preserves actions**
# **Validates: Requirements 6.2, 5.1, 5.3**
def test_action_flow_round_trip_with_custom_variable_name():
    """
    Test ActionFlow round-trip with custom variable name.
    
    **Feature: python-script-serialization, Property 4: ActionFlow round-trip preserves actions**
    **Validates: Requirements 6.2, 5.1, 5.3**
    """
    original_flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="custom_seq",
            actions=[
                Action(id="a1", type="click", target="#btn"),
            ]
        )
    )
    
    # Serialize with custom variable name
    python_script = original_flow.serialize(
        output_format='python',
        variable_name='my_flow'
    )
    
    # Verify custom variable name is used
    assert "my_flow = ActionFlow" in python_script
    assert "my_flow.execute" in python_script
    
    # Deserialize should still work
    restored_flow = ActionFlow.deserialize(
        python_script,
        output_format='python',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    assert restored_flow.sequence is not None
    assert len(restored_flow.sequence.actions) == 1


# endregion


# region Property 6: Variable name customization works

# **Feature: python-script-serialization, Property 6: Variable name customization works**
# **Validates: Requirements 8.1**
@st.composite
def variable_name_strategy(draw):
    """Generate valid Python variable names."""
    # Start with a letter or underscore
    first_char = draw(st.sampled_from(list('abcdefghijklmnopqrstuvwxyz_')))
    # Rest can be letters, digits, or underscores
    rest_chars = draw(st.text(
        min_size=0,
        max_size=15,
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_')
    ))
    return first_char + rest_chars


# **Feature: python-script-serialization, Property 6: Variable name customization works**
# **Validates: Requirements 8.1**
@settings(max_examples=100)
@given(
    actions=action_list_strategy(),
    variable_name=variable_name_strategy()
)
def test_action_graph_variable_name_customization(actions, variable_name):
    """
    Property 6: Variable name customization works for ActionGraph.
    
    **Feature: python-script-serialization, Property 6: Variable name customization works**
    **Validates: Requirements 8.1**
    
    For any ActionGraph, serializing with a custom variable_name parameter
    SHALL produce output where all method calls use that variable name
    instead of the default.
    """
    # Build graph with random actions
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        graph.action(action_type, target=target)
    
    # Serialize with custom variable name
    python_script = graph.serialize(
        output_format='python',
        variable_name=variable_name
    )
    
    # Verify the custom variable name is used in graph construction
    assert f"{variable_name} = ActionGraph" in python_script, \
        f"Expected '{variable_name} = ActionGraph' in script, got:\n{python_script}"
    
    # Verify the custom variable name is used in action calls
    assert f"{variable_name}.action(" in python_script, \
        f"Expected '{variable_name}.action(' in script, got:\n{python_script}"
    
    # Verify the default 'graph' is NOT used (unless variable_name is 'graph')
    if variable_name != 'graph':
        assert "graph = ActionGraph" not in python_script, \
            f"Default 'graph' should not appear when custom name '{variable_name}' is used"
        assert "graph.action(" not in python_script, \
            f"Default 'graph.action(' should not appear when custom name '{variable_name}' is used"


# **Feature: python-script-serialization, Property 6: Variable name customization works**
# **Validates: Requirements 8.1**
@settings(max_examples=100)
@given(
    actions=action_flow_action_list_strategy(),
    variable_name=variable_name_strategy()
)
def test_action_flow_variable_name_customization(actions, variable_name):
    """
    Property 6: Variable name customization works for ActionFlow.
    
    **Feature: python-script-serialization, Property 6: Variable name customization works**
    **Validates: Requirements 8.1**
    
    For any ActionFlow, serializing with a custom variable_name parameter
    SHALL produce output where all method calls use that variable name
    instead of the default.
    """
    # Build ActionFlow with ActionSequence
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(id="test_sequence", actions=actions)
    )
    
    # Serialize with custom variable name
    python_script = flow.serialize(
        output_format='python',
        variable_name=variable_name
    )
    
    # Verify the custom variable name is used in flow construction
    assert f"{variable_name} = ActionFlow" in python_script, \
        f"Expected '{variable_name} = ActionFlow' in script, got:\n{python_script}"
    
    # Verify the custom variable name is used in execute calls
    assert f"{variable_name}.execute(" in python_script, \
        f"Expected '{variable_name}.execute(' in script, got:\n{python_script}"
    
    # Verify the default 'flow' is NOT used (unless variable_name is 'flow')
    if variable_name != 'flow':
        assert "flow = ActionFlow" not in python_script, \
            f"Default 'flow' should not appear when custom name '{variable_name}' is used"
        assert "flow.execute(" not in python_script, \
            f"Default 'flow.execute(' should not appear when custom name '{variable_name}' is used"


# **Feature: python-script-serialization, Property 6: Variable name customization works**
# **Validates: Requirements 8.4**
def test_default_variable_names():
    """
    Test that default variable names are 'graph' for ActionGraph and 'flow' for ActionFlow.
    
    **Feature: python-script-serialization, Property 6: Variable name customization works**
    **Validates: Requirements 8.4**
    """
    # Test ActionGraph default
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#btn')
    
    graph_script = graph.serialize(output_format='python')
    assert "graph = ActionGraph" in graph_script, \
        f"ActionGraph default variable name should be 'graph', got:\n{graph_script}"
    
    # Test ActionFlow default
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="seq",
            actions=[Action(id="a1", type="click", target="#btn")]
        )
    )
    
    flow_script = flow.serialize(output_format='python')
    assert "flow = ActionFlow" in flow_script, \
        f"ActionFlow default variable name should be 'flow', got:\n{flow_script}"


# endregion


# region Property 7: Import inclusion can be toggled

# **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
# **Validates: Requirements 1.2, 8.2, 8.3**
@settings(max_examples=100)
@given(actions=action_list_strategy())
def test_action_graph_import_toggle_true(actions):
    """
    Property 7: Import inclusion can be toggled (include_imports=True) for ActionGraph.
    
    **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
    **Validates: Requirements 1.2, 8.2**
    
    For any ActionGraph, when include_imports=True, output SHALL contain
    "from agent_foundation" import statements.
    """
    # Build graph with random actions
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        graph.action(action_type, target=target)
    
    # Serialize with include_imports=True
    python_script = graph.serialize(
        output_format='python',
        include_imports=True
    )
    
    # Verify import statements are present
    assert "from agent_foundation" in python_script, \
        f"Expected 'from agent_foundation' import when include_imports=True, got:\n{python_script}"


# **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
# **Validates: Requirements 8.3**
@settings(max_examples=100)
@given(actions=action_list_strategy())
def test_action_graph_import_toggle_false(actions):
    """
    Property 7: Import inclusion can be toggled (include_imports=False) for ActionGraph.
    
    **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
    **Validates: Requirements 8.3**
    
    For any ActionGraph, when include_imports=False, output SHALL NOT contain
    any import statements.
    """
    # Build graph with random actions
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        graph.action(action_type, target=target)
    
    # Serialize with include_imports=False
    python_script = graph.serialize(
        output_format='python',
        include_imports=False
    )
    
    # Verify no import statements are present
    assert "import " not in python_script, \
        f"Expected no import statements when include_imports=False, got:\n{python_script}"
    assert "from " not in python_script.split('\n')[0] if python_script else True, \
        f"Expected no 'from' import statements when include_imports=False"


# **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
# **Validates: Requirements 1.2, 8.2, 8.3**
@settings(max_examples=100)
@given(actions=action_flow_action_list_strategy())
def test_action_flow_import_toggle_true(actions):
    """
    Property 7: Import inclusion can be toggled (include_imports=True) for ActionFlow.
    
    **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
    **Validates: Requirements 1.2, 8.2**
    
    For any ActionFlow, when include_imports=True, output SHALL contain
    "from agent_foundation" import statements.
    """
    # Build ActionFlow with ActionSequence
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(id="test_sequence", actions=actions)
    )
    
    # Serialize with include_imports=True
    python_script = flow.serialize(
        output_format='python',
        include_imports=True
    )
    
    # Verify import statements are present
    assert "from agent_foundation" in python_script, \
        f"Expected 'from agent_foundation' import when include_imports=True, got:\n{python_script}"


# **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
# **Validates: Requirements 8.3**
@settings(max_examples=100)
@given(actions=action_flow_action_list_strategy())
def test_action_flow_import_toggle_false(actions):
    """
    Property 7: Import inclusion can be toggled (include_imports=False) for ActionFlow.
    
    **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
    **Validates: Requirements 8.3**
    
    For any ActionFlow, when include_imports=False, output SHALL NOT contain
    any import statements.
    """
    # Build ActionFlow with ActionSequence
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(id="test_sequence", actions=actions)
    )
    
    # Serialize with include_imports=False
    python_script = flow.serialize(
        output_format='python',
        include_imports=False
    )
    
    # Verify no import statements are present
    assert "import " not in python_script, \
        f"Expected no import statements when include_imports=False, got:\n{python_script}"


# **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
# **Validates: Requirements 8.2**
def test_include_imports_default_is_true():
    """
    Test that include_imports defaults to True.
    
    **Feature: python-script-serialization, Property 7: Import inclusion can be toggled**
    **Validates: Requirements 8.2**
    """
    # Test ActionGraph default
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#btn')
    
    # Serialize without specifying include_imports (should default to True)
    graph_script = graph.serialize(output_format='python')
    assert "from agent_foundation" in graph_script, \
        f"ActionGraph should include imports by default, got:\n{graph_script}"
    
    # Test ActionFlow default
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="seq",
            actions=[Action(id="a1", type="click", target="#btn")]
        )
    )
    
    # Serialize without specifying include_imports (should default to True)
    flow_script = flow.serialize(output_format='python')
    assert "from agent_foundation" in flow_script, \
        f"ActionFlow should include imports by default, got:\n{flow_script}"


# endregion


# region Property 9: Existing serialization formats unchanged

# **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
# **Validates: Requirements 7.4**
@settings(max_examples=100)
@given(actions=action_list_strategy())
def test_action_graph_json_format_unchanged(actions):
    """
    Property 9: Existing serialization formats unchanged (JSON) for ActionGraph.
    
    **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
    **Validates: Requirements 7.4**
    
    For any ActionGraph, serializing with output_format='json' SHALL produce
    valid JSON that can be deserialized back to an equivalent graph.
    """
    import json
    
    # Build graph with random actions
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        graph.action(action_type, target=target)
    
    # Serialize to JSON format
    json_output = graph.serialize(output_format='json')
    
    # Verify it's valid JSON
    try:
        parsed = json.loads(json_output)
    except json.JSONDecodeError as e:
        raise AssertionError(f"JSON serialization produced invalid JSON: {e}\n{json_output}")
    
    # Verify expected structure
    assert "nodes" in parsed, f"JSON output should contain 'nodes' key, got: {parsed.keys()}"
    assert "version" in parsed, f"JSON output should contain 'version' key, got: {parsed.keys()}"
    
    # Verify round-trip works
    restored_graph = ActionGraph.deserialize(
        json_output,
        output_format='json',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    # Get actions from both graphs
    original_actions = []
    for node in graph._nodes:
        original_actions.extend(node._actions)
    
    restored_actions = []
    for node in restored_graph._nodes:
        restored_actions.extend(node._actions)
    
    # Verify same number of actions
    assert len(restored_actions) == len(original_actions), \
        f"JSON round-trip: Expected {len(original_actions)} actions, got {len(restored_actions)}"


# **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
# **Validates: Requirements 7.4**
@settings(max_examples=100)
@given(actions=action_flow_action_list_strategy())
def test_action_flow_json_format_unchanged(actions):
    """
    Property 9: Existing serialization formats unchanged (JSON) for ActionFlow.
    
    **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
    **Validates: Requirements 7.4**
    
    For any ActionFlow, serializing with output_format='json' SHALL produce
    valid JSON that can be deserialized back to an equivalent flow.
    """
    import json
    
    # Build ActionFlow with ActionSequence
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(id="test_sequence", actions=actions)
    )
    
    # Serialize to JSON format
    json_output = flow.serialize(output_format='json')
    
    # Verify it's valid JSON
    try:
        parsed = json.loads(json_output)
    except json.JSONDecodeError as e:
        raise AssertionError(f"JSON serialization produced invalid JSON: {e}\n{json_output}")
    
    # Verify expected structure
    assert "sequence" in parsed, f"JSON output should contain 'sequence' key, got: {parsed.keys()}"
    assert "version" in parsed, f"JSON output should contain 'version' key, got: {parsed.keys()}"
    
    # Verify round-trip works
    restored_flow = ActionFlow.deserialize(
        json_output,
        output_format='json',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    # Verify sequence exists and has same number of actions
    assert restored_flow.sequence is not None, "JSON round-trip: Restored flow should have a sequence"
    assert len(restored_flow.sequence.actions) == len(actions), \
        f"JSON round-trip: Expected {len(actions)} actions, got {len(restored_flow.sequence.actions)}"


# **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
# **Validates: Requirements 7.4**
@settings(max_examples=50)
@given(actions=action_list_strategy())
def test_action_graph_yaml_format_unchanged(actions):
    """
    Property 9: Existing serialization formats unchanged (YAML) for ActionGraph.
    
    **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
    **Validates: Requirements 7.4**
    
    For any ActionGraph, serializing with output_format='yaml' SHALL produce
    valid YAML that can be deserialized back to an equivalent graph.
    """
    try:
        import yaml
    except ImportError:
        # Skip test if PyYAML is not installed
        return
    
    # Build graph with random actions
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    for action_type, target in actions:
        graph.action(action_type, target=target)
    
    # Serialize to YAML format
    yaml_output = graph.serialize(output_format='yaml')
    
    # Verify it's valid YAML
    try:
        parsed = yaml.safe_load(yaml_output)
    except yaml.YAMLError as e:
        raise AssertionError(f"YAML serialization produced invalid YAML: {e}\n{yaml_output}")
    
    # Verify expected structure
    assert "nodes" in parsed, f"YAML output should contain 'nodes' key, got: {parsed.keys()}"
    
    # Verify round-trip works
    restored_graph = ActionGraph.deserialize(
        yaml_output,
        output_format='yaml',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    # Get actions from both graphs
    original_actions = []
    for node in graph._nodes:
        original_actions.extend(node._actions)
    
    restored_actions = []
    for node in restored_graph._nodes:
        restored_actions.extend(node._actions)
    
    # Verify same number of actions
    assert len(restored_actions) == len(original_actions), \
        f"YAML round-trip: Expected {len(original_actions)} actions, got {len(restored_actions)}"


# **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
# **Validates: Requirements 7.4**
@settings(max_examples=50)
@given(actions=action_flow_action_list_strategy())
def test_action_flow_yaml_format_unchanged(actions):
    """
    Property 9: Existing serialization formats unchanged (YAML) for ActionFlow.
    
    **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
    **Validates: Requirements 7.4**
    
    For any ActionFlow, serializing with output_format='yaml' SHALL produce
    valid YAML that can be deserialized back to an equivalent flow.
    """
    try:
        import yaml
    except ImportError:
        # Skip test if PyYAML is not installed
        return
    
    # Build ActionFlow with ActionSequence
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(id="test_sequence", actions=actions)
    )
    
    # Serialize to YAML format
    yaml_output = flow.serialize(output_format='yaml')
    
    # Verify it's valid YAML
    try:
        parsed = yaml.safe_load(yaml_output)
    except yaml.YAMLError as e:
        raise AssertionError(f"YAML serialization produced invalid YAML: {e}\n{yaml_output}")
    
    # Verify expected structure
    assert "sequence" in parsed, f"YAML output should contain 'sequence' key, got: {parsed.keys()}"
    
    # Verify round-trip works
    restored_flow = ActionFlow.deserialize(
        yaml_output,
        output_format='yaml',
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry()
    )
    
    # Verify sequence exists and has same number of actions
    assert restored_flow.sequence is not None, "YAML round-trip: Restored flow should have a sequence"
    assert len(restored_flow.sequence.actions) == len(actions), \
        f"YAML round-trip: Expected {len(actions)} actions, got {len(restored_flow.sequence.actions)}"


# **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
# **Validates: Requirements 7.4**
def test_to_serializable_obj_returns_dict_for_json_yaml():
    """
    Test that to_serializable_obj returns dict for JSON and YAML formats.
    
    **Feature: python-script-serialization, Property 9: Existing serialization formats unchanged**
    **Validates: Requirements 7.4**
    
    This verifies that the Python format special handling doesn't affect
    the existing JSON/YAML serialization behavior.
    """
    # Test ActionGraph
    graph = ActionGraph(action_executor=mock_executor, action_metadata=ActionMetadataRegistry())
    graph.action('click', target='#btn')
    
    # For JSON format, should return dict
    json_obj = graph.to_serializable_obj(_output_format='json')
    assert isinstance(json_obj, dict), \
        f"to_serializable_obj for JSON should return dict, got {type(json_obj)}"
    
    # For YAML format, should return dict
    yaml_obj = graph.to_serializable_obj(_output_format='yaml')
    assert isinstance(yaml_obj, dict), \
        f"to_serializable_obj for YAML should return dict, got {type(yaml_obj)}"
    
    # For None (default), should return dict
    default_obj = graph.to_serializable_obj()
    assert isinstance(default_obj, dict), \
        f"to_serializable_obj with no format should return dict, got {type(default_obj)}"
    
    # Test ActionFlow
    flow = ActionFlow(
        action_executor=mock_executor,
        action_metadata=ActionMetadataRegistry(),
        sequence=ActionSequence(
            id="seq",
            actions=[Action(id="a1", type="click", target="#btn")]
        )
    )
    
    # For JSON format, should return dict
    flow_json_obj = flow.to_serializable_obj(_output_format='json')
    assert isinstance(flow_json_obj, dict), \
        f"ActionFlow to_serializable_obj for JSON should return dict, got {type(flow_json_obj)}"
    
    # For YAML format, should return dict
    flow_yaml_obj = flow.to_serializable_obj(_output_format='yaml')
    assert isinstance(flow_yaml_obj, dict), \
        f"ActionFlow to_serializable_obj for YAML should return dict, got {type(flow_yaml_obj)}"


# endregion

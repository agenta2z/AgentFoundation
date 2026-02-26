"""
Property-based tests for ActionNode Agent-as-Action-Type feature.

Tests correctness properties from the agent-as-action-type design document.
Validates Properties 2 and 10 from the design.
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
_science_python_utils_src = _workspace_root / "SciencePythonUtils" / "src"
if _science_python_utils_src.exists() and str(_science_python_utils_src) not in sys.path:
    sys.path.insert(0, str(_science_python_utils_src))

from typing import Any, Dict, List, Optional
from hypothesis import given, strategies as st, settings, assume
import pytest

from science_modeling_tools.automation.schema.action_node import (
    ActionNode,
    AgentExecutionError,
)
from science_modeling_tools.automation.schema.common import (
    Action,
    ActionResult,
    ExecutionRuntime,
    TargetSpec,
    TargetStrategy,
)
from science_modeling_tools.automation.schema.action_metadata import ActionMetadataRegistry
from science_modeling_tools.automation.schema.action_executor import MultiActionExecutor
from science_modeling_tools.agents.agent import Agent


# region Mock Agent and Fixtures

class MockAgent(Agent):
    """
    Mock Agent for testing agent detection and execution.

    Records calls for verification and returns configurable results.
    """

    def __init__(
        self,
        return_value: Any = "mock_agent_result",
        should_fail: bool = False,
        fail_message: str = "Mock agent failure",
    ):
        """
        Initialize mock agent.

        Args:
            return_value: Value to return when called
            should_fail: If True, raise an exception when called
            fail_message: Exception message when should_fail is True
        """
        self._return_value = return_value
        self._should_fail = should_fail
        self._fail_message = fail_message
        self._call_history: List[Dict[str, Any]] = []

        # Initialize parent Agent with minimal config
        # We override __call__ so most agent internals aren't used
        super().__init__(
            reasoner=None,  # Not used in mock
            actor=None,     # Not used in mock
        )

    def __call__(self, *args, **kwargs) -> Any:
        """
        Record call and return configured result.
        """
        self._call_history.append({
            'args': args,
            'kwargs': kwargs,
        })

        if self._should_fail:
            raise RuntimeError(self._fail_message)

        return self._return_value

    @property
    def call_count(self) -> int:
        """Number of times the agent was called."""
        return len(self._call_history)

    @property
    def last_call(self) -> Optional[Dict[str, Any]]:
        """The last call arguments, or None if never called."""
        return self._call_history[-1] if self._call_history else None

    def reset(self):
        """Clear call history."""
        self._call_history.clear()


def mock_executor(
    action_type: str,
    action_target: Optional[str],
    action_args: Optional[Dict[str, Any]] = None,
    action_target_strategy: Optional[str] = None,
) -> str:
    """Mock action executor for standard (non-agent) actions."""
    return f"executed_{action_type}_{action_target}"


# endregion


# region Hypothesis Strategies

# Strategy for generating valid task descriptions
task_description_strategy = st.text(
    min_size=1,
    max_size=200,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S'))
).filter(lambda x: x.strip())  # Ensure non-empty after strip

# Strategy for generating action IDs
action_id_strategy = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=('L', 'N'))
).filter(lambda x: x.strip())

# Strategy for generating agent type identifiers
agent_type_strategy = st.sampled_from([
    'navigation_agent',
    'search_agent',
    'form_agent',
    'data_extraction_agent',
    'validation_agent',
])

# Strategy for generating context variables
context_variables_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    values=st.one_of(st.text(max_size=50), st.integers(), st.floats(allow_nan=False), st.booleans()),
    max_size=10,
)

# Strategy for TargetSpec with agent strategy
agent_target_spec_strategy = st.builds(
    TargetSpec,
    strategy=st.just('agent'),
    value=task_description_strategy,
    options=st.one_of(st.none(), st.lists(st.sampled_from(['static', 'dynamic', 'cached']), max_size=3)),
)

# endregion


# region Property 2: Agent Detection via isinstance

@settings(max_examples=100)
@given(
    agent_type=agent_type_strategy,
    task_description=task_description_strategy,
    action_id=action_id_strategy,
)
def test_agent_detection_via_isinstance(
    agent_type: str,
    task_description: str,
    action_id: str,
):
    """
    Property 2: Agent Detection via isinstance.

    **Feature: agent-as-action, Property 2: Agent Detection via isinstance**
    **Validates: Requirements 2.1, 2.5**

    For any executor resolved from MultiActionExecutor, ActionNode SHALL correctly
    identify Agent instances using isinstance checks and route to agent execution path.
    """
    # Create mock agent
    mock_agent = MockAgent(return_value="agent_result")

    # Create executor with agent registered
    executor = MultiActionExecutor({
        agent_type: mock_agent,
        'default': mock_executor,
    })

    # Create action using agent type
    action = Action(
        id=action_id,
        type=agent_type,
        target=task_description,
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Execute
    context = ExecutionRuntime()
    result = node.run(context)

    # Verify agent was called (detection worked)
    assert mock_agent.call_count == 1, \
        f"Agent should have been called exactly once, but was called {mock_agent.call_count} times"

    # Verify result
    assert isinstance(result, ActionResult), \
        f"Result should be ActionResult, got {type(result)}"
    assert result.success is True, \
        "Result should be successful"
    assert result.metadata.get("agent_action") is True, \
        "Result metadata should indicate agent action"
    assert result.metadata.get("agent_type") == agent_type, \
        f"Result metadata agent_type should be {agent_type}"


@settings(max_examples=50)
@given(
    agent_type=agent_type_strategy,
    task_description=task_description_strategy,
    context_vars=context_variables_strategy,
)
def test_agent_receives_context_variables(
    agent_type: str,
    task_description: str,
    context_vars: Dict[str, Any],
):
    """
    Property 5: Context Variable Passing (partial test).

    **Feature: agent-as-action, Property 5: Context Variable Passing**
    **Validates: Requirements 2.3, 5.1**

    For any ExecutionRuntime with variables, when an agent action is executed,
    the agent SHALL receive all variables from the context.
    """
    # Ensure we have at least one context variable
    if not context_vars:
        context_vars = {'test_key': 'test_value'}

    mock_agent = MockAgent(return_value="agent_result")

    executor = MultiActionExecutor({
        agent_type: mock_agent,
        'default': mock_executor,
    })

    action = Action(
        id='test_action',
        type=agent_type,
        target=task_description,
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Execute with context variables
    context = ExecutionRuntime(variables=context_vars.copy())
    result = node.run(context)

    # Verify agent received context
    assert mock_agent.call_count == 1
    call_kwargs = mock_agent.last_call['kwargs']

    assert 'context' in call_kwargs, \
        "Agent should receive 'context' in kwargs"

    # Verify all context variables were passed
    received_context = call_kwargs['context']
    for key, value in context_vars.items():
        assert key in received_context, \
            f"Context variable '{key}' should be in received context"
        assert received_context[key] == value, \
            f"Context variable '{key}' value mismatch"


@settings(max_examples=50)
@given(
    agent_type=agent_type_strategy,
    action_id=action_id_strategy,
)
def test_non_agent_executor_uses_standard_path(
    agent_type: str,
    action_id: str,
):
    """
    Verify that non-Agent executors use the standard execution path.

    **Feature: agent-as-action**
    **Validates: Agent detection is accurate - non-agents aren't mistaken for agents**
    """
    # Track standard executor calls
    standard_calls = []

    def tracking_executor(
        action_type: str,
        action_target: Optional[str],
        action_args: Optional[Dict[str, Any]] = None,
        action_target_strategy: Optional[str] = None,
    ) -> str:
        standard_calls.append({
            'action_type': action_type,
            'action_target': action_target,
        })
        return f"executed_{action_type}"

    # Register a regular callable, not an Agent
    executor = MultiActionExecutor({
        'click': tracking_executor,
        'default': tracking_executor,
    })

    # Create action with standard type
    action = Action(
        id=action_id,
        type='click',
        target='#button',
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Execute
    context = ExecutionRuntime()
    result = node.run(context)

    # Verify standard path was used
    assert len(standard_calls) == 1, \
        "Standard executor should have been called once"
    assert result.success is True
    assert result.metadata.get("agent_action") is not True, \
        "Result should not be marked as agent action"


# endregion


# region Property 10: Agent-Based Element Finding

@settings(max_examples=100)
@given(
    element_description=task_description_strategy,
    action_id=action_id_strategy,
    options=st.one_of(st.none(), st.lists(st.sampled_from(['static', 'cached']), max_size=2)),
)
def test_agent_based_element_finding(
    element_description: str,
    action_id: str,
    options: Optional[List[str]],
):
    """
    Property 10: Agent-Based Element Finding.

    **Feature: agent-as-action, Property 10: Agent-Based Element Finding**
    **Validates: Requirements 9.1, 9.2, 9.5**

    For any TargetSpec with strategy='agent', the ActionNode SHALL resolve
    the element using the registered find_element_agent before executing the action.
    """
    # The find_element_agent should return an element reference
    resolved_element = "#resolved_element_123"
    find_element_agent = MockAgent(return_value=resolved_element)

    # Track standard executor calls
    executor_calls = []

    def tracking_executor(
        action_type: str,
        action_target: Optional[str],
        action_args: Optional[Dict[str, Any]] = None,
        action_target_strategy: Optional[str] = None,
    ) -> str:
        executor_calls.append({
            'action_type': action_type,
            'action_target': action_target,
        })
        return f"clicked_{action_target}"

    executor = MultiActionExecutor({
        'find_element_agent': find_element_agent,
        'click': tracking_executor,
        'default': tracking_executor,
    })

    # Create action with agent-based target
    target_spec = TargetSpec(
        strategy='agent',
        value=element_description,
        options=options,
    )

    action = Action(
        id=action_id,
        type='click',
        target=target_spec,
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Execute
    context = ExecutionRuntime()
    result = node.run(context)

    # Verify find_element_agent was called
    assert find_element_agent.call_count == 1, \
        f"find_element_agent should have been called once, was called {find_element_agent.call_count} times"

    # Verify agent received correct input
    call_kwargs = find_element_agent.last_call['kwargs']
    assert call_kwargs.get('user_input') == element_description, \
        f"Agent should receive element description as user_input"

    # Verify options were passed if provided
    if options:
        assert call_kwargs.get('options') == options, \
            f"Agent should receive options: expected {options}, got {call_kwargs.get('options')}"

    # Verify the resolved element was used in the action
    assert len(executor_calls) == 1, \
        "Standard executor should have been called once with resolved element"
    assert executor_calls[0]['action_target'] == resolved_element, \
        f"Action should use resolved element '{resolved_element}', got '{executor_calls[0]['action_target']}'"

    # Verify result
    assert result.success is True


@settings(max_examples=50)
@given(
    element_description=task_description_strategy,
)
def test_agent_target_strategy_enum_and_string(element_description: str):
    """
    Verify that both TargetStrategy.AGENT enum and 'agent' string work.

    **Feature: agent-as-action**
    **Validates: Strategy matching works for both enum and string values**
    """
    resolved_element = "#element"
    find_element_agent = MockAgent(return_value=resolved_element)

    executor = MultiActionExecutor({
        'find_element_agent': find_element_agent,
        'default': mock_executor,
    })

    registry = ActionMetadataRegistry()

    # Test with string strategy
    target_string = TargetSpec(strategy='agent', value=element_description)
    action_string = Action(id='test_string', type='click', target=target_string)
    node_string = ActionNode(
        action=action_string,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()
    result = node_string.run(context)

    assert find_element_agent.call_count == 1, \
        "Agent should be called with string strategy 'agent'"

    find_element_agent.reset()

    # Test with enum strategy
    target_enum = TargetSpec(strategy=TargetStrategy.AGENT, value=element_description)
    action_enum = Action(id='test_enum', type='click', target=target_enum)
    node_enum = ActionNode(
        action=action_enum,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()
    result = node_enum.run(context)

    assert find_element_agent.call_count == 1, \
        "Agent should be called with TargetStrategy.AGENT enum"


def test_missing_find_element_agent_raises_error():
    """
    Verify that missing find_element_agent raises a clear error.

    **Feature: agent-as-action**
    **Validates: Proper error handling when find_element_agent is not registered**
    """
    # Executor without find_element_agent
    executor = MultiActionExecutor({
        'click': mock_executor,
        'default': mock_executor,
    })

    target_spec = TargetSpec(strategy='agent', value='find the button')
    action = Action(id='test_action', type='click', target=target_spec)

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()

    with pytest.raises(ValueError) as exc_info:
        node.run(context)

    assert "find_element_agent" in str(exc_info.value), \
        "Error should mention missing find_element_agent"


def test_agent_execution_error_contains_context():
    """
    Verify that AgentExecutionError contains proper context information.

    **Feature: agent-as-action**
    **Validates: Requirement 6.5 - Error context includes agent info and description**
    """
    failing_agent = MockAgent(should_fail=True, fail_message="Test failure")

    executor = MultiActionExecutor({
        'test_agent': failing_agent,
        'default': mock_executor,
    })

    task_description = "This is the task that failed"
    action = Action(id='failing_action', type='test_agent', target=task_description)

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()

    with pytest.raises(AgentExecutionError) as exc_info:
        node.run(context)

    error = exc_info.value
    assert error.agent_id == 'test_agent', \
        f"Error should contain agent_id 'test_agent', got '{error.agent_id}'"
    assert error.description == task_description, \
        f"Error should contain task description"
    assert error.original_error is not None, \
        "Error should contain original exception"


# endregion


# region Additional Property Tests

@settings(max_examples=50)
@given(
    agent_type=agent_type_strategy,
    task_description=task_description_strategy,
)
def test_agent_receives_previous_result(
    agent_type: str,
    task_description: str,
):
    """
    Property 6: Action Results Propagation.

    **Feature: agent-as-action, Property 6: Action Results Propagation**
    **Validates: Requirements 5.2**

    For any sequence of actions where an agent action follows other actions,
    the agent SHALL receive the results from previous actions.
    """
    mock_agent = MockAgent(return_value="agent_result")

    executor = MultiActionExecutor({
        agent_type: mock_agent,
        'default': mock_executor,
    })

    action = Action(
        id='test_action',
        type=agent_type,
        target=task_description,
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Set up context with previous result in '_' variable
    previous_result = {"status": "completed", "data": [1, 2, 3]}
    context = ExecutionRuntime(variables={'_': previous_result})

    result = node.run(context)

    # Verify agent received previous result
    call_kwargs = mock_agent.last_call['kwargs']
    assert 'action_results' in call_kwargs, \
        "Agent should receive 'action_results' in kwargs"
    assert call_kwargs['action_results'] == previous_result, \
        f"Agent should receive previous result: expected {previous_result}"


@settings(max_examples=50)
@given(
    agent_type=agent_type_strategy,
    task_description=task_description_strategy,
)
def test_agent_output_stored_in_context(
    agent_type: str,
    task_description: str,
):
    """
    Property 7: Agent Output Merging.

    **Feature: agent-as-action, Property 7: Agent Output Merging**
    **Validates: Requirements 5.3**

    For any agent action that completes successfully, the agent's output
    SHALL be available in the ExecutionRuntime context for subsequent actions.
    """
    agent_output = {"agent": "response", "data": 42}
    mock_agent = MockAgent(return_value=agent_output)

    executor = MultiActionExecutor({
        agent_type: mock_agent,
        'default': mock_executor,
    })

    action = Action(
        id='test_action',
        type=agent_type,
        target=task_description,
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()
    result = node.run(context)

    # Verify output is stored in '_' for subsequent actions
    assert '_' in context.variables, \
        "Agent output should be stored in context.variables['_']"
    assert context.variables['_'] == agent_output, \
        f"Stored output should match agent output"


# endregion


# region Template Variable Tests (Task 8)

@settings(max_examples=50)
@given(
    agent_type=agent_type_strategy,
    base_url=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
)
def test_agent_template_variables_in_target(
    agent_type: str,
    base_url: str,
):
    """
    Property: Template variables in agent target are resolved.

    **Feature: agent-as-action, Property: Template Variable Substitution**
    **Validates: Template variables in target field are resolved before agent execution**

    When an agent action's target contains template variables like '{base_url}/search',
    the ActionNode SHALL substitute these variables from context.variables before
    passing the task description to the agent.
    """
    mock_agent = MockAgent(return_value="agent_result")

    executor = MultiActionExecutor({
        agent_type: mock_agent,
        'default': mock_executor,
    })

    # Action with template variable in target
    action = Action(
        id='test_action',
        type=agent_type,
        target='{base_url}/search?q=test',  # Template variable
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Provide template variable value in context
    context = ExecutionRuntime(variables={'base_url': base_url})
    result = node.run(context)

    # Verify agent received substituted target
    call_kwargs = mock_agent.last_call['kwargs']
    expected_target = f'{base_url}/search?q=test'
    assert call_kwargs.get('user_input') == expected_target, \
        f"Agent should receive substituted target '{expected_target}', got '{call_kwargs.get('user_input')}'"


@settings(max_examples=50)
@given(
    agent_type=agent_type_strategy,
    query_text=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('L', 'N'))),
)
def test_agent_template_variables_in_args(
    agent_type: str,
    query_text: str,
):
    """
    Property: Template variables in agent args are resolved.

    **Feature: agent-as-action, Property: Template Variable Substitution**
    **Validates: Template variables in args field are resolved before agent execution**
    """
    mock_agent = MockAgent(return_value="agent_result")

    executor = MultiActionExecutor({
        agent_type: mock_agent,
        'default': mock_executor,
    })

    # Action with template variable in args
    action = Action(
        id='test_action',
        type=agent_type,
        target='perform search',
        args={'query': '{search_query}', 'limit': 10},  # Template variable in args
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Provide template variable value in context
    context = ExecutionRuntime(variables={'search_query': query_text})
    result = node.run(context)

    # Verify agent received substituted args
    call_kwargs = mock_agent.last_call['kwargs']
    assert call_kwargs.get('query') == query_text, \
        f"Agent should receive substituted query '{query_text}', got '{call_kwargs.get('query')}'"
    assert call_kwargs.get('limit') == 10, \
        "Non-template args should be passed through unchanged"


def test_agent_multiple_template_variables():
    """
    Verify multiple template variables are all resolved.

    **Feature: agent-as-action**
    **Validates: Multiple template variables work correctly**
    """
    mock_agent = MockAgent(return_value="agent_result")

    executor = MultiActionExecutor({
        'search_agent': mock_agent,
        'default': mock_executor,
    })

    # Action with multiple template variables
    action = Action(
        id='test_action',
        type='search_agent',
        target='{domain}/{path}',
        args={'user': '{username}', 'api_key': '{api_key}'},
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime(variables={
        'domain': 'example.com',
        'path': 'api/v1',
        'username': 'testuser',
        'api_key': 'secret123',
    })
    result = node.run(context)

    # Verify all variables were substituted
    call_kwargs = mock_agent.last_call['kwargs']
    assert call_kwargs.get('user_input') == 'example.com/api/v1'
    assert call_kwargs.get('user') == 'testuser'
    assert call_kwargs.get('api_key') == 'secret123'


# endregion


# region Error Handling Tests (Task 6)

def test_agent_on_error_continue_returns_failed_result():
    """
    Verify that on_error='continue' returns a failed ActionResult without raising.

    **Feature: agent-as-action, Property 8: Error Handling Policy**
    **Validates: Requirements 6.1, 6.2**

    When an agent action fails and on_error='continue', the ActionNode SHALL:
    1. Return ActionResult(success=False, error=e) instead of raising
    2. Store None in output variable and '_' context variable
    3. Allow subsequent actions to continue execution
    """
    failing_agent = MockAgent(should_fail=True, fail_message="Expected failure")

    executor = MultiActionExecutor({
        'test_agent': failing_agent,
        'default': mock_executor,
    })

    # Action with on_error='continue'
    action = Action(
        id='failing_action',
        type='test_agent',
        target='task description',
        on_error='continue',  # Key: should not raise
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()

    # Should NOT raise - returns failed result instead
    result = node.run(context)

    # Verify result indicates failure
    assert isinstance(result, ActionResult), \
        f"Result should be ActionResult, got {type(result)}"
    assert result.success is False, \
        "Result should indicate failure (success=False)"
    assert result.error is not None, \
        "Result should contain the error"
    assert result.metadata.get("on_error") == "continue", \
        "Result metadata should indicate on_error policy"

    # Verify context was updated (with None values since action failed)
    assert context.variables.get('_') is None, \
        "Context '_' should be None after failed action with on_error=continue"


def test_agent_on_error_stop_raises_exception():
    """
    Verify that on_error='stop' (default) raises AgentExecutionError.

    **Feature: agent-as-action, Property 8: Error Handling Policy**
    **Validates: Requirements 6.3**

    When an agent action fails and on_error='stop', the ActionNode SHALL
    raise AgentExecutionError to halt execution.
    """
    failing_agent = MockAgent(should_fail=True, fail_message="Expected failure")

    executor = MultiActionExecutor({
        'test_agent': failing_agent,
        'default': mock_executor,
    })

    # Action with default on_error='stop'
    action = Action(
        id='failing_action',
        type='test_agent',
        target='task description',
        # on_error defaults to 'stop'
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()

    # Should raise AgentExecutionError
    with pytest.raises(AgentExecutionError) as exc_info:
        node.run(context)

    assert exc_info.value.agent_id == 'test_agent'


@settings(max_examples=30)
@given(
    agent_type=agent_type_strategy,
    task_description=task_description_strategy,
)
def test_on_error_continue_allows_sequence_to_proceed(
    agent_type: str,
    task_description: str,
):
    """
    Property test: on_error='continue' enables graceful degradation.

    **Feature: agent-as-action**
    **Validates: Workflow can continue despite agent failure**

    This is important for workflows where some agent actions are optional
    or where partial results are acceptable.
    """
    failing_agent = MockAgent(should_fail=True, fail_message="Test failure")

    executor = MultiActionExecutor({
        agent_type: failing_agent,
        'default': mock_executor,
    })

    action = Action(
        id='optional_agent_action',
        type=agent_type,
        target=task_description,
        on_error='continue',
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()

    # Should not raise
    result = node.run(context)

    # Verify failure is captured in result
    assert result.success is False
    assert result.error is not None
    # Verify workflow can check the failure and decide how to proceed
    assert result.metadata.get("agent_action") is True


# endregion


# region Serialization Tests (Task 9)

def test_agent_action_serialization_roundtrip():
    """
    Verify that agent action serializes and deserializes correctly.

    **Feature: agent-as-action, Property: Serialization Support**
    **Validates: Agent actions can be persisted and restored**

    The ActionNode for agent actions should serialize to dict and deserialize
    back to a working ActionNode that can execute the agent.
    """
    mock_agent = MockAgent(return_value="agent_result")

    executor = MultiActionExecutor({
        'test_agent': mock_agent,
        'default': mock_executor,
    })

    # Create original action node
    action = Action(
        id='serializable_action',
        type='test_agent',
        target='perform task',
        args={'key': 'value'},
        on_error='continue',
    )

    registry = ActionMetadataRegistry()
    original_node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Serialize
    serialized = original_node.to_serializable_obj()

    # Verify serialized structure
    assert 'action' in serialized, "Serialized should contain action"
    assert serialized['action']['type'] == 'test_agent', \
        "Serialized action should preserve type"
    assert serialized['action']['target'] == 'perform task', \
        "Serialized action should preserve target"
    assert serialized['action']['args'] == {'key': 'value'}, \
        "Serialized action should preserve args"
    assert serialized['action']['on_error'] == 'continue', \
        "Serialized action should preserve on_error"

    # Deserialize
    restored_node = ActionNode.from_serializable_obj(
        serialized,
        action_executor=executor,
        action_metadata=registry,
    )

    # Verify restored node works
    context = ExecutionRuntime()
    result = restored_node.run(context)

    assert mock_agent.call_count == 1, \
        "Agent should be called after deserialization"
    assert result.success is True, \
        "Restored node should execute successfully"


def test_agent_target_spec_serialization():
    """
    Verify that agent TargetSpec serializes correctly.

    **Feature: agent-as-action**
    **Validates: TargetSpec with strategy='agent' serializes properly**
    """
    find_agent = MockAgent(return_value="#element")

    executor = MultiActionExecutor({
        'find_element_agent': find_agent,
        'click': mock_executor,
        'default': mock_executor,
    })

    # Create action with agent target strategy
    target_spec = TargetSpec(
        strategy='agent',
        value='the submit button',
        options=['static'],
    )

    action = Action(
        id='click_with_agent_target',
        type='click',
        target=target_spec,
    )

    registry = ActionMetadataRegistry()
    node = ActionNode(
        action=action,
        action_executor=executor,
        action_metadata=registry,
    )

    # Serialize
    serialized = node.to_serializable_obj()

    # Verify TargetSpec is serialized
    target_dict = serialized['action']['target']
    assert target_dict['strategy'] == 'agent', \
        "Target strategy should be serialized"
    assert target_dict['value'] == 'the submit button', \
        "Target value should be serialized"
    assert target_dict['options'] == ['static'], \
        "Target options should be serialized"

    # Deserialize and verify it works
    restored_node = ActionNode.from_serializable_obj(
        serialized,
        action_executor=executor,
        action_metadata=registry,
    )

    context = ExecutionRuntime()
    result = restored_node.run(context)

    assert find_agent.call_count == 1, \
        "find_element_agent should be called after deserialization"


# endregion


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running property tests for Agent-as-Action-Type feature...")

    print("\n=== Property 2: Agent Detection via isinstance ===")
    test_agent_detection_via_isinstance()
    print("✓ test_agent_detection_via_isinstance passed")

    test_agent_receives_context_variables()
    print("✓ test_agent_receives_context_variables passed")

    test_non_agent_executor_uses_standard_path()
    print("✓ test_non_agent_executor_uses_standard_path passed")

    print("\n=== Property 10: Agent-Based Element Finding ===")
    test_agent_based_element_finding()
    print("✓ test_agent_based_element_finding passed")

    test_agent_target_strategy_enum_and_string()
    print("✓ test_agent_target_strategy_enum_and_string passed")

    test_missing_find_element_agent_raises_error()
    print("✓ test_missing_find_element_agent_raises_error passed")

    test_agent_execution_error_contains_context()
    print("✓ test_agent_execution_error_contains_context passed")

    print("\n=== Additional Property Tests ===")
    test_agent_receives_previous_result()
    print("✓ test_agent_receives_previous_result passed")

    test_agent_output_stored_in_context()
    print("✓ test_agent_output_stored_in_context passed")

    print("\n=== Template Variable Tests (Task 8) ===")
    test_agent_template_variables_in_target()
    print("✓ test_agent_template_variables_in_target passed")

    test_agent_template_variables_in_args()
    print("✓ test_agent_template_variables_in_args passed")

    test_agent_multiple_template_variables()
    print("✓ test_agent_multiple_template_variables passed")

    print("\n=== Error Handling Tests (Task 6) ===")
    test_agent_on_error_continue_returns_failed_result()
    print("✓ test_agent_on_error_continue_returns_failed_result passed")

    test_agent_on_error_stop_raises_exception()
    print("✓ test_agent_on_error_stop_raises_exception passed")

    test_on_error_continue_allows_sequence_to_proceed()
    print("✓ test_on_error_continue_allows_sequence_to_proceed passed")

    print("\n=== Serialization Tests (Task 9) ===")
    test_agent_action_serialization_roundtrip()
    print("✓ test_agent_action_serialization_roundtrip passed")

    test_agent_target_spec_serialization()
    print("✓ test_agent_target_spec_serialization passed")

    print("\n✅ All property tests passed!")

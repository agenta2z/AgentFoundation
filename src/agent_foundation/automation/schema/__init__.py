"""
Action Sequence Schema - UI Automation Framework

Provides a generic, extensible action sequence execution system.
UI-agnostic core that can be implemented by platform-specific backends.

Example usage (Sequential):
    from agent_foundation.automation.schema import (
        ActionSequence, Action, TargetSpec,
        ActionFlow, ActionMetadataRegistry
    )

    # Execute from file, JSON string, or ActionSequence object
    executor = ActionFlow(
        action_executor=my_action_executor,  # Must satisfy ActionExecutor protocol
        action_metadata=ActionMetadataRegistry()
    )

    # Can execute from file path
    result = executor.execute("my_workflow.json")

    # Or from JSON string
    result = executor.execute('{"id": "test", "actions": [...]}')

    # Or from ActionSequence object
    sequence = ActionSequence(id="test", actions=[...])
    result = executor.execute(sequence)

Example usage (Graph-based DAG workflow):
    from agent_foundation.automation.schema import (
        ActionGraph, ActionMetadataRegistry, ConditionContext
    )

    # Build graph with action() and condition() methods
    graph = ActionGraph(
        action_executor=my_action_executor,
        action_metadata=ActionMetadataRegistry()
    )
    graph.action("click", target="login_btn")
    graph.action("input_text", target="username", args={"text": "user"})

    # Branching API Option 1: Context manager syntax (recommended)
    with graph.condition(lambda r: r.success) as branch:
        with branch.if_true():
            graph.action("click", target="dashboard")
        with branch.if_false():
            graph.action("click", target="retry")

    # Branching API Option 2: Match-case syntax (Python 3.10+)
    match graph.condition(lambda r: r.success):
        case ConditionContext.TRUE:
            graph.action("click", target="dashboard")
        case ConditionContext.FALSE:
            graph.action("click", target="retry")

    # Branching API Option 3: Callback-based syntax
    graph.branch(
        condition=lambda r: r.success,
        if_true=lambda g: g.action("click", target="dashboard"),
        if_false=lambda g: g.action("click", target="retry"),
    )

    # Branching API Option 4: If-statement with else_branch()
    # Note: Python's else doesn't invoke __bool__(), so use else_branch()
    if graph.condition(lambda r: r.success):
        graph.action("click", target="dashboard")
    graph.else_branch()
    graph.action("click", target="retry")

    result = graph.execute()

    # Save/load from JSON
    graph.to_json("workflow.json")
    loaded = ActionGraph.from_json("workflow.json", action_executor, action_metadata)

"""

# Common models, context, and protocols
from .common import (
    # Models
    Action,
    ActionSequence,
    TargetSpec,
    TargetSpecWithFallback,
    TargetStrategy,
    # Context and Results
    ActionResult,
    ExecutionRuntime,
    ExecutionResult,
    # Exceptions
    TargetNotFoundError,
    BranchAlreadyExistsError,
    # Protocols
    ActionExecutor,
    # Loader functions
    load_sequence,
    load_sequence_from_string,
)

# Action Metadata
from .action_metadata import (
    ActionMetadataRegistry,
    ActionTypeMetadata,
    CompositeActionConfig,
    CompositeActionStep,
    ActionMemoryMode,
    # Action name constants
    ACTION_NAME_CLICK,
    ACTION_NAME_INPUT_TEXT,
    ACTION_NAME_APPEND_TEXT,
    ACTION_NAME_SCROLL,
    ACTION_NAME_SCROLL_UP_TO_ELEMENT,
    ACTION_NAME_VISIT_URL,
    ACTION_NAME_WAIT,
    ACTION_NAME_NO_OP,
    ACTION_NAME_INPUT_AND_SUBMIT,
)

# ActionNode (WorkGraphNode subclass for single action execution)
from .action_node import ActionNode

# Multi-executor support (action_type → callable mapping)
from .action_executor import MultiActionExecutor

# Executor
from .action_flow import (
    ActionFlow,
    SequenceExecutor,  # Backward compatibility alias
)

# Graph Executor (DAG-based workflow orchestration)
from .action_graph import (
    ActionGraph,
    ActionSequenceNode,
    ConditionContext,
    BranchContext,
    BranchBlock,
    condition_expr,
    # Target not found context manager support
    ActionChainHelper,
    TargetNotFoundContext,
)

# Monitor support (Generic Layer - executor-agnostic)
from .monitor import (
    MonitorNode,
    MonitorResult,
    MonitorStatus,
)

__all__ = [
    # Models
    "Action",
    "ActionSequence",
    "TargetSpec",
    "TargetSpecWithFallback",
    "TargetStrategy",
    # Context and Results
    "ActionResult",
    "ExecutionRuntime",
    "ExecutionResult",
    # Exceptions
    "TargetNotFoundError",
    "BranchAlreadyExistsError",
    # Action Metadata
    "ActionMetadataRegistry",
    "ActionTypeMetadata",
    "CompositeActionConfig",
    "CompositeActionStep",
    "ActionMemoryMode",
    # Action name constants
    "ACTION_NAME_CLICK",
    "ACTION_NAME_INPUT_TEXT",
    "ACTION_NAME_APPEND_TEXT",
    "ACTION_NAME_SCROLL",
    "ACTION_NAME_SCROLL_UP_TO_ELEMENT",
    "ACTION_NAME_VISIT_URL",
    "ACTION_NAME_WAIT",
    "ACTION_NAME_NO_OP",
    "ACTION_NAME_INPUT_AND_SUBMIT",
    # ActionNode (WorkGraphNode subclass)
    "ActionNode",
    # Multi-executor support
    "MultiActionExecutor",
    # Executor
    "ActionFlow",
    "SequenceExecutor",  # Backward compatibility alias
    # Graph Executor
    "ActionGraph",
    "ActionSequenceNode",
    "ConditionContext",
    "BranchContext",
    "BranchBlock",
    "condition_expr",
    # Target not found context manager support
    "ActionChainHelper",
    "TargetNotFoundContext",
    # Monitor support (Generic Layer - executor-agnostic)
    "MonitorNode",
    "MonitorResult",
    "MonitorStatus",
    # Protocols
    "ActionExecutor",
    # Loader functions
    "load_sequence",
    "load_sequence_from_string",
]

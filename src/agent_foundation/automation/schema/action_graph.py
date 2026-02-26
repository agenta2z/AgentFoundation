"""
ActionGraph - Graph Construction + Execution for Action Workflows

Combines WorkGraph DAG orchestration with ActionFlow sequential execution.
Provides a builder pattern for constructing workflows via action() and condition() methods.

Renamed from ActionGraphExecutor for clarity and consistency.

Features:
- Fluent API: Call registered actions directly as methods (e.g., graph.click('#btn'))
- Callable workflows: Execute with template variables (e.g., workflow(base_url='...'))
- Template variable system: Use {var} placeholders with automatic type coercion
- Multi-engine support: Python str.format, Jinja2, Handlebars, string.Template
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

from attr import attrs, attrib

from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode, WorkGraph
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import ResultPassDownMode
from rich_python_utils.common_objects.workflow.common.worknode_base import NextNodesSelector
from rich_python_utils.common_objects.serializable import (
    Serializable,
    FIELD_TYPE,
    FIELD_MODULE,
    FIELD_SERIALIZATION,
    SERIALIZATION_DICT,
)

from .common import (
    Action,
    ActionSequence,
    BranchAlreadyExistsError,
    ExecutionResult,
    ExecutionRuntime,
    LoopExecutionError,
    TargetSpec,
    TargetSpecWithFallback,
    TargetStrategy,
)
from .action_flow import ActionFlow
from .action_metadata import ActionMetadataRegistry, ActionTypeMetadata
from .action_executor import MultiActionExecutor

# Generic monitor layer (executor-agnostic)
from .monitor import (
    MonitorNode,
    MonitorResult,
    MonitorStatus,
)

# Template engine support (used for validation)
SUPPORTED_TEMPLATE_ENGINES = ('python', 'jinja2', 'handlebars', 'string_template')


@attrs
class BranchContext:
    """Tracks branch state during graph construction."""
    branch_node: 'ActionSequenceNode' = attrib()
    is_true_branch: bool = attrib()
    parent_node: Optional['ActionSequenceNode'] = attrib(default=None)


class ConditionContext:
    """Context manager for conditional branching in ActionGraph.

    This class enables multiple branching syntaxes for building conditional
    branches in action graphs. At build time, all branches are recorded.
    At runtime, conditions are evaluated to determine which branch to execute.

    Supported syntaxes:

    1. Context manager with explicit conditions:
        with graph.condition(lambda r: r.success) as branch:
            with branch.if_true():
                graph.action('click', '#success_btn')
            with branch.if_false():
                graph.action('click', '#retry_btn')

    2. Multi-branch with elseif (context manager):
        with graph.condition(lambda r: score >= 90) as branch:
            with branch.if_true():
                graph.add(value=100)  # A
            with branch.elseif(lambda r: score >= 80):
                graph.add(value=80)   # B
            with branch.elseif(lambda r: score >= 70):
                graph.add(value=70)   # C
            with branch.if_false():
                graph.add(value=60)   # D

    3. Convenience comparison methods with value_extractor:
        with graph.condition(value_extractor=get_last_value) as branch:
            with branch.if_gte(90):       # if value >= 90
                graph.add(value=100)      # A
            with branch.elseif_gte(80):   # elif value >= 80
                graph.add(value=80)       # B
            with branch.elseif_gte(70):   # elif value >= 70
                graph.add(value=70)       # C
            with branch.else_():          # else
                graph.add(value=60)       # D

        Available comparison methods:
        - if_gt(v), elseif_gt(v)   : value > v
        - if_gte(v), elseif_gte(v) : value >= v
        - if_lt(v), elseif_lt(v)   : value < v
        - if_lte(v), elseif_lte(v) : value <= v
        - if_eq(v), elseif_eq(v)   : value == v
        - if_ne(v), elseif_ne(v)   : value != v

    4. Match-case syntax (Python 3.10+):
        match graph.condition(lambda r: r.success):
            case ConditionContext.TRUE:
                graph.action('click', '#success_btn')
            case ConditionContext.FALSE:
                graph.action('click', '#retry_btn')
    """
    
    # Sentinel values for match-case syntax
    TRUE = "CONDITION_TRUE"
    FALSE = "CONDITION_FALSE"

    def __init__(
        self,
        graph: 'ActionGraph',
        condition_func: Optional[Callable] = None,
        value_extractor: Optional[Callable] = None,
    ):
        self.graph = graph
        self.condition_func = condition_func
        self.value_extractor = value_extractor
        self._branch_node: Optional['ActionSequenceNode'] = None
        self._else_node: Optional['ActionSequenceNode'] = None
        self._entered = False
        self._parent_node: Optional['ActionSequenceNode'] = None
        self._in_context_manager = False
        self._prior_conditions: List[Callable] = []  # Track conditions for elseif exclusivity
    
    def __bool__(self) -> bool:
        """Called by Python's if statement at graph-build time.
        
        Always returns True so the if-block is entered and its actions
        are recorded. The actual condition is evaluated at runtime.
        
        Note: Python's else doesn't re-invoke __bool__(), so else blocks
        won't execute during build. Use else_branch() or context manager syntax.
        """
        if self._entered:
            # Already entered, return False for elif/else handling
            return False
        
        self._entered = True
        self._parent_node = self.graph._current_node
        
        # Create a branch node that will evaluate condition_func at runtime
        self._branch_node = self.graph._create_branch_node(self.condition_func)
        
        # Push branch context - subsequent actions go to this branch
        self.graph._push_branch(self._branch_node, is_true_branch=True)
        
        return True  # Always enter if-block during build
    
    def __enter__(self):
        """Enter conditional block (for 'with' statement compatibility).
        
        When used as context manager, enables if_true()/if_false() sub-contexts.
        """
        self._in_context_manager = True
        self._parent_node = self.graph._current_node
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit conditional block, pop branch context."""
        if not self._in_context_manager:
            self.graph._pop_branch()
        return False
    
    def if_true(self) -> 'BranchBlock':
        """Context manager for the true branch.

        Usage:
            with graph.condition(lambda r: r.success) as branch:
                with branch.if_true():
                    graph.action('click', '#success')
        """
        if not self._in_context_manager:
            raise RuntimeError("if_true() can only be used within 'with' context")

        # Track initial condition for elseif exclusivity
        self._prior_conditions.append(self.condition_func)

        # Create branch node for true condition
        self._branch_node = self.graph._create_branch_node(self.condition_func)
        return BranchBlock(self.graph, self._branch_node, self._parent_node)

    def elseif(self, condition_func: Callable) -> 'BranchBlock':
        """Context manager for an elif branch.

        Creates a branch that executes when all prior conditions are False
        AND this condition is True.

        Usage:
            with graph.condition(lambda r: score >= 90) as branch:
                with branch.if_true():
                    graph.add(value=100)  # A
                with branch.elseif(lambda r: score >= 80):
                    graph.add(value=80)   # B
                with branch.elseif(lambda r: score >= 70):
                    graph.add(value=70)   # C
                with branch.if_false():
                    graph.add(value=60)   # D
        """
        if not self._in_context_manager:
            raise RuntimeError("elseif() can only be used within 'with' context")

        # Build exclusive condition: not any prior AND this condition
        captured_priors = self._prior_conditions.copy()
        captured_cond = condition_func

        def elseif_exclusive_condition(result, _priors=captured_priors, _cond=captured_cond, **kwargs):
            # All prior conditions must be False
            for prior in _priors:
                if prior(result, **kwargs):
                    return False
            # And this condition must be True
            return _cond(result, **kwargs)

        # Track this condition for future elseif/if_false exclusivity
        self._prior_conditions.append(condition_func)

        elseif_node = self.graph._create_branch_node(elseif_exclusive_condition)
        return BranchBlock(self.graph, elseif_node, self._parent_node)

    def if_false(self) -> 'BranchBlock':
        """Context manager for the false branch (else).

        Executes when all prior conditions (if_true and elseif) are False.

        Usage:
            with graph.condition(lambda r: r.success) as branch:
                with branch.if_false():
                    graph.action('click', '#retry')
        """
        if not self._in_context_manager:
            raise RuntimeError("if_false() can only be used within 'with' context")

        # Build else condition: all prior conditions must be False
        if self._prior_conditions:
            captured_priors = self._prior_conditions.copy()

            def else_condition(result, _priors=captured_priors, **kwargs):
                for prior in _priors:
                    if prior(result, **kwargs):
                        return False
                return True

            self._else_node = self.graph._create_branch_node(else_condition)
        else:
            # Fallback: simple inversion if no prior conditions tracked
            def inverted_condition(result, **kwargs):
                return not self.condition_func(result, **kwargs)

            self._else_node = self.graph._create_branch_node(inverted_condition)

        return BranchBlock(self.graph, self._else_node, self._parent_node)

    # Alias for if_false
    def else_(self) -> 'BranchBlock':
        """Alias for if_false(). Context manager for the else branch."""
        return self.if_false()

    # ================================================================
    # Convenience comparison methods using value_extractor
    # ================================================================

    def _make_comparison_condition(self, op: str, value: Any) -> Callable:
        """Create a comparison condition function using value_extractor."""
        if self.value_extractor is None:
            raise RuntimeError(
                "Comparison methods require value_extractor. "
                "Use graph.condition(value_extractor=...) or provide a lambda."
            )
        extractor = self.value_extractor
        if op == 'gt':
            return lambda r, **kw: extractor(r) > value
        elif op == 'gte':
            return lambda r, **kw: extractor(r) >= value
        elif op == 'lt':
            return lambda r, **kw: extractor(r) < value
        elif op == 'lte':
            return lambda r, **kw: extractor(r) <= value
        elif op == 'eq':
            return lambda r, **kw: extractor(r) == value
        elif op == 'ne':
            return lambda r, **kw: extractor(r) != value
        else:
            raise ValueError(f"Unknown comparison operator: {op}")

    # if_* methods (first condition)
    def if_gt(self, value: Any) -> 'BranchBlock':
        """Branch if extracted value > given value."""
        cond = self._make_comparison_condition('gt', value)
        self._prior_conditions.append(cond)
        node = self.graph._create_branch_node(cond)
        return BranchBlock(self.graph, node, self._parent_node)

    def if_gte(self, value: Any) -> 'BranchBlock':
        """Branch if extracted value >= given value."""
        cond = self._make_comparison_condition('gte', value)
        self._prior_conditions.append(cond)
        node = self.graph._create_branch_node(cond)
        return BranchBlock(self.graph, node, self._parent_node)

    def if_lt(self, value: Any) -> 'BranchBlock':
        """Branch if extracted value < given value."""
        cond = self._make_comparison_condition('lt', value)
        self._prior_conditions.append(cond)
        node = self.graph._create_branch_node(cond)
        return BranchBlock(self.graph, node, self._parent_node)

    def if_lte(self, value: Any) -> 'BranchBlock':
        """Branch if extracted value <= given value."""
        cond = self._make_comparison_condition('lte', value)
        self._prior_conditions.append(cond)
        node = self.graph._create_branch_node(cond)
        return BranchBlock(self.graph, node, self._parent_node)

    def if_eq(self, value: Any) -> 'BranchBlock':
        """Branch if extracted value == given value."""
        cond = self._make_comparison_condition('eq', value)
        self._prior_conditions.append(cond)
        node = self.graph._create_branch_node(cond)
        return BranchBlock(self.graph, node, self._parent_node)

    def if_ne(self, value: Any) -> 'BranchBlock':
        """Branch if extracted value != given value."""
        cond = self._make_comparison_condition('ne', value)
        self._prior_conditions.append(cond)
        node = self.graph._create_branch_node(cond)
        return BranchBlock(self.graph, node, self._parent_node)

    # elseif_* methods (subsequent conditions with exclusivity)
    def _elseif_comparison(self, op: str, value: Any) -> 'BranchBlock':
        """Create elseif branch with comparison condition."""
        cond = self._make_comparison_condition(op, value)
        # Build exclusive condition
        captured_priors = self._prior_conditions.copy()
        captured_cond = cond

        def elseif_exclusive(result, _priors=captured_priors, _cond=captured_cond, **kwargs):
            for prior in _priors:
                if prior(result, **kwargs):
                    return False
            return _cond(result, **kwargs)

        self._prior_conditions.append(cond)
        node = self.graph._create_branch_node(elseif_exclusive)
        return BranchBlock(self.graph, node, self._parent_node)

    def elseif_gt(self, value: Any) -> 'BranchBlock':
        """Elseif branch: extracted value > given value."""
        return self._elseif_comparison('gt', value)

    def elseif_gte(self, value: Any) -> 'BranchBlock':
        """Elseif branch: extracted value >= given value."""
        return self._elseif_comparison('gte', value)

    def elseif_lt(self, value: Any) -> 'BranchBlock':
        """Elseif branch: extracted value < given value."""
        return self._elseif_comparison('lt', value)

    def elseif_lte(self, value: Any) -> 'BranchBlock':
        """Elseif branch: extracted value <= given value."""
        return self._elseif_comparison('lte', value)

    def elseif_eq(self, value: Any) -> 'BranchBlock':
        """Elseif branch: extracted value == given value."""
        return self._elseif_comparison('eq', value)

    def elseif_ne(self, value: Any) -> 'BranchBlock':
        """Elseif branch: extracted value != given value."""
        return self._elseif_comparison('ne', value)

    def __match_args__(self):
        """Support for match-case syntax (Python 3.10+)."""
        return ()
    
    def __eq__(self, other):
        """Support match-case comparison with TRUE/FALSE sentinels.
        
        Usage:
            match graph.condition(lambda r: r.success):
                case ConditionContext.TRUE:
                    graph.action('click', '#success')
                case ConditionContext.FALSE:
                    graph.action('click', '#retry')
        """
        if other == ConditionContext.TRUE:
            # Enter true branch
            if not self._entered:
                self._entered = True
                self._parent_node = self.graph._current_node
                self._branch_node = self.graph._create_branch_node(self.condition_func)
                self.graph._push_branch(self._branch_node, is_true_branch=True)
            return True
        elif other == ConditionContext.FALSE:
            # Pop true branch if active, enter false branch
            if self._branch_node and self.graph._branch_stack:
                self.graph._pop_branch()
            
            # Create else node with inverted condition
            def inverted_condition(result):
                return not self.condition_func(result)
            
            self._else_node = self.graph._create_branch_node(inverted_condition)
            self.graph._push_branch(self._else_node, is_true_branch=False)
            return True
        return False


class BranchBlock:
    """Context manager for a single branch block within a condition.
    
    Used by ConditionContext.if_true() and if_false() methods.
    """
    
    def __init__(
        self,
        graph: 'ActionGraph',
        branch_node: 'ActionSequenceNode',
        parent_node: 'ActionSequenceNode',
    ):
        self.graph = graph
        self.branch_node = branch_node
        self.parent_node = parent_node
    
    def __enter__(self):
        """Enter branch block, set current node to branch."""
        self.graph._push_branch(self.branch_node, is_true_branch=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit branch block, restore parent node."""
        self.graph._pop_branch()
        # Restore to parent for next branch
        self.graph._current_node = self.parent_node
        return False


class ActionChainHelper:
    """Helper for method chaining after graph.action().
    
    Enables the pattern:
        with graph.action("click", target=spec).target_not_found():
            graph.action("click", target=fallback_spec)
    
    Also supports continued chaining:
        graph.action("click", target=spec1).action("type", target=spec2)
    
    And the as-binding pattern:
        with graph.action("click", target=spec) as helper:
            # Access helper.action_obj if needed
            pass
    """
    
    def __init__(self, graph: 'ActionGraph', action: Action):
        """Initialize ActionChainHelper.
        
        Args:
            graph: The ActionGraph instance for method forwarding.
            action: The Action object created by graph.action().
        """
        self._graph = graph
        self._action = action
    
    @property
    def action_obj(self) -> Action:
        """Access the underlying Action object.
        
        Named action_obj (not action) to avoid conflict with the action() method.
        
        Returns:
            The Action object created by graph.action().
        """
        return self._action
    
    # Method forwarding for backward compatibility and chaining
    def action(self, *args, **kwargs) -> 'ActionChainHelper':
        """Forward to graph.action() for continued chaining.
        
        Returns:
            ActionChainHelper for the new action (or ActionGraph for monitor actions).
        """
        return self._graph.action(*args, **kwargs)
    
    def condition(self, *args, **kwargs) -> 'ConditionContext':
        """Forward to graph.condition() for continued chaining.
        
        Returns:
            ConditionContext for conditional branching.
        """
        return self._graph.condition(*args, **kwargs)
    
    def loop(self, *args, **kwargs) -> 'ActionGraph':
        """Forward to graph.loop() for continued chaining.
        
        Returns:
            ActionGraph for method chaining.
        """
        return self._graph.loop(*args, **kwargs)
    
    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Forward to graph.execute() for execution.
        
        Returns:
            ExecutionResult from graph execution.
        """
        return self._graph.execute(*args, **kwargs)
    
    def target_not_found(
        self,
        retry_after_handling: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> 'TargetNotFoundContext':
        """Define a branch for target-not-found condition.
        
        Creates a context manager that collects actions to execute when the
        parent action's target cannot be found.
        
        Args:
            retry_after_handling: If True, retry finding the target after branch executes.
                                  If False (default), continue to next action after branch.
            max_retries: Maximum retry attempts (0-10). Default is 3.
            retry_delay: Seconds to wait between retries (0-60). Default is 1.0.
        
        Returns:
            TargetNotFoundContext context manager for defining branch actions.
        
        Raises:
            ValueError: If max_retries is outside range 0-10.
            ValueError: If retry_delay is negative or exceeds 60 seconds.
            ValueError: If the action has no target (target is None).
            BranchAlreadyExistsError: If target_not_found() already called on this action.
        
        Example:
            with graph.action("click", target=spec).target_not_found():
                graph.action("click", target=fallback_spec)
            
            # With retry behavior
            with graph.action("click", target=spec).target_not_found(
                retry_after_handling=True,
                max_retries=3,
                retry_delay=1.0
            ):
                graph.action("click", target=fallback_spec)
        """
        # Validate action has a target
        if self._action.target is None:
            raise ValueError(
                f"Cannot define target_not_found branch on action '{self._action.type}' "
                f"with no target."
            )
        
        # Check for duplicate branch
        if self._action.target_not_found_actions is not None:
            raise BranchAlreadyExistsError(
                condition="target_not_found",
                action_type=self._action.type
            )
        
        # Validate parameters
        if not 0 <= max_retries <= 10:
            raise ValueError(f"max_retries must be 0-10, got {max_retries}")
        if retry_delay < 0:
            raise ValueError(f"retry_delay must be non-negative, got {retry_delay}")
        if retry_delay > 60:
            raise ValueError(f"retry_delay must be <= 60 seconds, got {retry_delay}")
        
        return TargetNotFoundContext(
            graph=self._graph,
            parent_action=self._action,
            retry_after_handling=retry_after_handling,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    # Alias for target_not_found
    on_target_not_found = target_not_found
    
    def __enter__(self) -> 'ActionChainHelper':
        """Support as-binding pattern: with graph.action(...) as helper.
        
        This is a no-op scope - entering the context doesn't change any state.
        It simply allows the pattern:
            with graph.action("click", target=spec) as helper:
                # Access helper.action_obj if needed
                pass
        
        Returns:
            self for the as-binding.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """No-op exit for as-binding pattern.
        
        Returns:
            False to not suppress any exceptions.
        """
        return False


class TargetNotFoundContext:
    """Context manager for target_not_found branch definition.
    
    Collects actions added inside the context into the parent action's
    target_not_found_actions list. When entering the context, the graph's
    action context switches to collect actions into the branch list. When
    exiting, the context is restored.
    
    Attributes:
        _graph: The ActionGraph instance for action collection.
        _parent_action: The Action that owns this branch.
        _retry_after_handling: If True, retry finding the target after branch executes.
        _max_retries: Maximum retry attempts (0-10).
        _retry_delay: Seconds to wait between retries.
        _branch_actions: List to collect branch actions.
        _context_pushed: Flag tracking whether context was successfully pushed.
    
    Example:
        with graph.action("click", target=spec).target_not_found():
            graph.action("click", target=fallback_spec)
        
        # With retry behavior
        with graph.action("click", target=spec).target_not_found(
            retry_after_handling=True,
            max_retries=3,
            retry_delay=1.0
        ):
            graph.action("click", target=fallback_spec)
    """
    
    def __init__(
        self,
        graph: 'ActionGraph',
        parent_action: Action,
        retry_after_handling: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize TargetNotFoundContext.
        
        Args:
            graph: The ActionGraph instance for action collection.
            parent_action: The Action that owns this branch.
            retry_after_handling: If True, retry finding the target after branch executes.
            max_retries: Maximum retry attempts (0-10).
            retry_delay: Seconds to wait between retries.
        """
        self._graph = graph
        self._parent_action = parent_action
        self._retry_after_handling = retry_after_handling
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._branch_actions: List[Action] = []
        self._context_pushed: bool = False
    
    def __enter__(self) -> 'TargetNotFoundContext':
        """Switch graph context to collect actions into this branch.
        
        Stores the configuration on the parent action and pushes a branch
        context so that subsequent graph.action() calls add actions to
        this branch's list instead of the main sequence.
        
        Returns:
            self for use in with-statement.
        """
        # Store config on parent action
        self._parent_action.target_not_found_config = {
            'retry_after_handling': self._retry_after_handling,
            'max_retries': self._max_retries,
            'retry_delay': self._retry_delay
        }
        
        # Initialize the branch list on parent action
        self._parent_action.target_not_found_actions = self._branch_actions
        
        # Push branch context so graph.action() adds to our list
        self._graph._push_action_branch_context(self._branch_actions)
        self._context_pushed = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Restore graph context. Always pops if pushed, even on exception.
        
        Args:
            exc_type: Exception type if an exception was raised, None otherwise.
            exc_val: Exception value if an exception was raised, None otherwise.
            exc_tb: Exception traceback if an exception was raised, None otherwise.
        
        Returns:
            False to not suppress any exceptions.
        """
        if self._context_pushed:
            self._graph._pop_action_branch_context()
            self._context_pushed = False
        
        # If exception occurred, clean up partial state on parent action
        if exc_type is not None:
            # Clear the branch to avoid partial state
            self._parent_action.target_not_found_actions = None
            self._parent_action.target_not_found_config = None
        
        return False  # Don't suppress exceptions
    
    # Forward common methods to graph for chaining inside context
    def action(self, *args, **kwargs) -> 'ActionChainHelper':
        """Add action to this branch. Forwards to graph.action().
        
        Returns:
            ActionChainHelper for the new action.
        """
        return self._graph.action(*args, **kwargs)
    
    def condition(self, *args, **kwargs) -> 'ConditionContext':
        """Forward to graph.condition() for nested conditions.
        
        Returns:
            ConditionContext for conditional branching.
        """
        return self._graph.condition(*args, **kwargs)


@attrs(slots=False, repr=False)
class ActionGraph(WorkGraph):
    """
    Graph construction + execution for action workflows.

    ActionGraph is a single-root-node DAG (Directed Acyclic Graph). All workflows
    start from a single root node created at initialization. Subsequent nodes are
    added via condition() and linked through add_next(), forming a tree-like
    structure where execution always begins from the root.

    Architecture:
    - Single root node: Created in __attrs_post_init__, always _nodes[0]
    - DAG structure: Nodes linked via add_next() for conditional branching
    - Execution: Starts from root, traverses based on condition evaluation

    Methods:
    - action(): Adds action to CURRENT node
    - condition(condition): Creates NEW node with condition, becomes current
    - execute(): Runs the built graph starting from root

    Usage:
        graph = ActionGraph(action_executor=driver)
        graph.action("click", target="login_btn")
        graph.action("input_text", target="username", args={"text": "user"})

        # condition creates NEW node with condition (evaluated at runtime)
        graph.condition(condition=lambda r: r.success)
        graph.action("click", target="dashboard")  # adds to condition node

        result = graph.execute()
    """

    # Override start_nodes from DirectedAcyclicGraph to provide a default
    start_nodes: List['ActionSequenceNode'] = attrib(factory=list)

    # The action executor callable (e.g., WebDriver instance)
    action_executor: Union[Callable, MultiActionExecutor] = attrib(default=None)

    # Action metadata registry
    action_metadata: ActionMetadataRegistry = attrib(factory=ActionMetadataRegistry)

    # Persistence settings (enable_result_save inherited from Resumable via WorkGraph)
    result_save_dir: Optional[str] = attrib(default=None, kw_only=True)

    # Internal state for building the graph
    _nodes: List['ActionSequenceNode'] = attrib(factory=list)
    _current_node: 'ActionSequenceNode' = attrib(default=None)
    _action_id_counter: int = attrib(default=0)
    
    # Branch tracking for conditional building
    _branch_stack: List[BranchContext] = attrib(factory=list)
    _else_branch_node: Optional['ActionSequenceNode'] = attrib(default=None)
    
    # Action branch stack for target_not_found context management
    # When non-empty, graph.action() adds actions to the top list instead of current node
    _action_branch_stack: List[List[Action]] = attrib(factory=list)

    # Template engine configuration
    template_engine: str = attrib(default='python', kw_only=True)

    def __attrs_post_init__(self):
        # Auto-wrap Mapping action_executor into MultiActionExecutor
        if isinstance(self.action_executor, Mapping) and not isinstance(self.action_executor, MultiActionExecutor):
            self.action_executor = MultiActionExecutor(self.action_executor)
        # Create initial root node (no condition)
        self._current_node = self._create_node(condition=None)
        # Validate template engine
        if self.template_engine not in SUPPORTED_TEMPLATE_ENGINES:
            raise ValueError(
                f"Unsupported template engine: {self.template_engine}. "
                f"Supported engines: {SUPPORTED_TEMPLATE_ENGINES}"
            )
        super().__attrs_post_init__()

    def __repr__(self) -> str:
        """
        Leverage DirectedAcyclicGraph.__repr__() with tree visualization.

        Sets start_nodes to enable str_all_descendants_of_nodes() to work.
        DirectedAcyclicGraph.__repr__ uses Node.str_all_descendants() which calls __str__()
        on each node, so our __str__ overrides make this display correctly.

        Note: We call DirectedAcyclicGraph.__repr__() directly because WorkGraph's
        @attrs decorator generates a __repr__ that overrides the inherited one.
        """
        from rich_python_utils.algorithms.graph.dag import DirectedAcyclicGraph

        # Ensure start_nodes is set for the visualization to work
        if not self.start_nodes and self._nodes:
            self.start_nodes = [self._nodes[0]]
        return DirectedAcyclicGraph.__repr__(self)

    def print_structure(self, **kwargs) -> str:
        """
        Print detailed DAG structure for debugging and analysis.

        Override to ensure start_nodes is set before calling parent method.
        Uses same pattern as __repr__ - ActionGraph builds nodes incrementally
        via builder pattern, so start_nodes isn't set until execute().

        Args:
            **kwargs: Passed to DirectedAcyclicGraph.print_structure()
                - ascii_tree: Use ASCII tree format (default: True)
                - include_degrees: Show in/out degrees (default: True)
                - include_adjacency: Show adjacency lists (default: True)

        Returns:
            String representation of the graph structure
        """
        from rich_python_utils.algorithms.graph.dag import DirectedAcyclicGraph

        # Ensure start_nodes is set for the visualization to work
        if not self.start_nodes and self._nodes:
            self.start_nodes = [self._nodes[0]]
        return DirectedAcyclicGraph.print_structure(self, **kwargs)

    @property
    def required_variables(self) -> Set[str]:
        """
        Set of all template variables required by this workflow.

        Aggregates required_variables from all ActionSequenceNodes.
        Each node in turn aggregates from its ActionNodes.

        Returns:
            Union of all required_variables from all nodes.
        """
        all_vars: Set[str] = set()
        for node in self._nodes:
            all_vars.update(node.required_variables)
        return all_vars

    def __getattr__(self, name: str):
        """
        Enable fluent API: graph.click(target) instead of graph.action('click', target).

        Dynamically dispatch method calls to registered action types.
        Unregistered actions raise AttributeError.

        Args:
            name: Method name (should match a registered action type)

        Returns:
            A callable that creates the action and returns self for chaining

        Raises:
            AttributeError: If name is not a registered action type

        Example:
            >>> graph = ActionGraph(action_executor=driver, action_metadata=registry)
            >>> graph.click('#submit').input_text('#search', text='hello').wait(seconds=2)
        """
        # Avoid infinite recursion during initialization
        if name.startswith('_') or name in ('action_metadata', 'template_engine'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if action type exists in registry
        try:
            metadata = self.action_metadata.get_metadata(name)
        except Exception:
            metadata = None

        if metadata is not None:
            def action_method(
                target: Optional[Union[TargetSpec, TargetSpecWithFallback, str]] = None,
                *,
                args: Optional[Dict[str, Any]] = None,
                action_id: Optional[str] = None,
                condition: Optional[str] = None,
                on_error: str = "stop",
                output: Optional[str] = None,
                timeout: Optional[float] = None,
                wait: Optional[Union[float, bool]] = None,
                **kwargs
            ) -> 'ActionGraph':
                """
                Fluent action method generated for registered action type.

                Args:
                    target: Target element (selector, description, or TargetSpec)
                    args: Action-specific arguments dict
                    action_id: Custom ID for the action
                    condition: Condition expression for execution
                    on_error: Error handling mode ('stop', 'continue', 'retry')
                    output: Variable name to store result
                    timeout: Action timeout in seconds
                    wait: Wait after action. float=seconds, True=human confirmation
                    **kwargs: Additional args merged with `args`

                Returns:
                    self for method chaining
                """
                # Merge kwargs into args (kwargs takes precedence)
                merged_args = {**(args or {}), **kwargs}
                return self.action(
                    action_type=name,
                    target=target,
                    args=merged_args if merged_args else None,
                    action_id=action_id,
                    condition=condition,
                    on_error=on_error,
                    output=output,
                    timeout=timeout,
                    wait=wait,
                )
            return action_method

        raise AttributeError(f"'{type(self).__name__}' has no action '{name}'")

    def _create_node(
        self,
        condition: Optional[Callable] = None,
        *,
        max_repeat: int = 1,
        repeat_condition: Optional[Callable] = None,
        fallback_result: Any = None,
        min_repeat_wait: float = 0,
        max_repeat_wait: float = 0,
        retry_on_exceptions: Optional[List[type]] = None,
        output_validator: Optional[Callable] = None,
    ) -> 'ActionSequenceNode':
        """Create a new ActionSequenceNode with empty action list."""
        node = ActionSequenceNode(
            name=f"node_{len(self._nodes)}",
            action_executor=self.action_executor,
            action_metadata=self.action_metadata,
            condition=condition,
            template_engine=self.template_engine,
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
            max_repeat=max_repeat,
            repeat_condition=repeat_condition,
            fallback_result=fallback_result,
            min_repeat_wait=min_repeat_wait,
            max_repeat_wait=max_repeat_wait,
            retry_on_exceptions=retry_on_exceptions,
            output_validator=output_validator,
            # Propagate persistence settings from graph to node
            enable_result_save=self.enable_result_save,
            result_save_dir=self.result_save_dir,
            # Propagate debug config from graph to node
            copy_debuggable_config_from=self,
        )
        self._nodes.append(node)
        return node

    def action(
        self,
        action_type: str,
        target: Optional[Union[TargetSpec, TargetSpecWithFallback, str, int, float]] = None,
        args: Optional[Dict[str, Any]] = None,
        *,
        action_id: Optional[str] = None,
        condition: Optional[str] = None,
        on_error: str = "stop",
        output: Optional[str] = None,
        timeout: Optional[float] = None,
        wait: Optional[Union[float, bool]] = None,
        no_action_if_target_not_found: bool = False,
        # Monitor-specific parameters
        event_condition: Optional[str] = None,
        event_confirmation_time: float = 0.0,
        max_checks: int = 60,
        interval: float = 0.5,
        continuous: bool = False,
        enable_auto_setup: bool = True,
        enable_verify_setup: bool = True,
    ) -> Union['ActionGraph', 'ActionChainHelper']:
        """
        Add an action to the CURRENT node's sequence.

        Template variables in target and args are detected by ActionNode at execution
        time. Each ActionNode autonomously handles its own template substitution.
        
        Returns ActionChainHelper for method chaining and target_not_found().
        
        NOTE: Special handling for "monitor" action type - the existing code
        calls _handle_monitor_action() BEFORE creating an Action object.
        This behavior must be preserved. Monitor actions return self, not ActionChainHelper.

        Args:
            action_type: Registered action type name (e.g., 'click', 'input_text', 'monitor')
            target: Target element selector, description, or TargetSpec
            args: Action-specific arguments dict
            action_id: Custom ID for the action (auto-generated if not provided)
            condition: Condition expression for conditional execution
            on_error: Error handling mode ('stop', 'continue', 'retry')
            output: Variable name to store action result
            timeout: Action timeout in seconds
            wait: Wait after action execution. float=seconds to wait, True=wait for
                  human confirmation via terminal input. Useful for debugging.
            no_action_if_target_not_found: If True, skip action gracefully when target
                  element is not found instead of raising an error. Default False.

            # Monitor-specific parameters (only used when action_type='monitor'):
            event_condition: Condition type (e.g., 'text_changed', 'element_present')
            event_confirmation_time: Debounce time in seconds
            max_checks: Maximum number of condition checks
            interval: Seconds between checks
            continuous: If True, enables continuous monitoring loop where monitor
                        re-runs after downstream actions complete. Default False.
            enable_auto_setup: If True (default), auto-run setup before each check
                               (e.g., switch to monitored tab). Set to False for demos
                               where manual control is desired.
            enable_verify_setup: If True (default), verify context is valid before
                                 checking condition. If verify fails, monitor is paused.

        Returns:
            ActionChainHelper for method chaining and target_not_found() (for non-monitor actions)
            self (ActionGraph) for monitor actions (preserving existing behavior)
        """
        # Handle "monitor" action type specially BEFORE creating Action
        # Monitor actions return self, not ActionChainHelper
        if action_type == "monitor":
            return self._handle_monitor_action(
                target=target,
                event_condition=event_condition,
                event_confirmation_time=event_confirmation_time,
                max_checks=max_checks,
                interval=interval,
                continuous=continuous,
                enable_auto_setup=enable_auto_setup,
                enable_verify_setup=enable_verify_setup,
            )
        
        # Create regular action
        self._action_id_counter += 1
        actual_action_id = action_id or f"action_{self._action_id_counter}"

        action_obj = Action(
            id=actual_action_id,
            type=action_type,
            target=target,
            args=args,
            condition=condition,
            on_error=on_error,
            output=output,
            timeout=timeout,
            wait=wait,
            no_action_if_target_not_found=no_action_if_target_not_found,
        )
        
        # Add to branch context if active, otherwise to current node
        if self._action_branch_stack:
            self._action_branch_stack[-1].append(action_obj)
        else:
            self._current_node.add_action(action_obj)
        
        return ActionChainHelper(self, action_obj)
    
    def _handle_monitor_action(
        self,
        target: Optional[Union[TargetSpec, TargetSpecWithFallback, str]] = None,
        event_condition: Optional[str] = None,
        event_confirmation_time: float = 0.0,
        max_checks: int = 60,
        interval: float = 0.5,
        continuous: bool = False,
        enable_auto_setup: bool = True,
        enable_verify_setup: bool = True,
    ) -> 'ActionGraph':
        """
        Handle monitor action type - monitors an element on the current tab.

        Creates a MonitorNode that watches an element for a condition. The monitor
        runs on the CURRENT tab (no new tab is created). Use `visit_url` action
        first if you need to navigate to a different page.

        Standard mode (continuous=False):
            Monitor waits for condition → downstream actions run once → done

        Continuous mode (continuous=True):
            Monitor waits for condition → downstream actions run → monitor re-runs
            → downstream actions run → ... (loop continues until manually stopped)

        Args:
            target: TargetSpec for the element to monitor
            event_condition: Condition type (e.g., 'text_changed', 'element_present')
            event_confirmation_time: Debounce time in seconds (condition must remain
                                     true for this duration before being reported as met)
            max_checks: Maximum number of condition checks (default 60)
            interval: Seconds between checks (default 0.5)
            continuous: If True, enables continuous monitoring loop where monitor
                        re-runs after downstream actions complete. Default False.
            enable_auto_setup: If True (default), auto-run setup before each check
                               (e.g., switch to monitored tab). Set to False for demos
                               where manual control is desired.
            enable_verify_setup: If True (default), verify context is valid before
                                 checking condition. If verify fails, monitor is paused.

        Returns:
            Self for method chaining

        Raises:
            ImportError: If webaxon package is not installed
            ValueError: If event_condition is not provided or invalid

        Example:
            >>> graph = ActionGraph(action_executor=webdriver)
            >>> graph.action("visit_url", target="https://www.google.com")
            >>> # For textarea/input, use "value_changed" (monitors get_attribute("value"))
            >>> # For regular elements (div/span), use "text_changed" (monitors element.text)
            >>> graph.action(
            ...     "monitor",
            ...     target=TargetSpec(strategy="xpath", value="//textarea[@title='Search']"),
            ...     event_condition="value_changed",  # Use value_changed for textarea/input
            ...     event_confirmation_time=5,  # Wait 5 seconds to confirm stable
            ...     max_checks=60,
            ...     interval=0.5,
            ...     continuous=True  # Enable continuous monitoring loop
            ... )
            >>> graph.action("click", target="#submit")
        """
        if event_condition is None:
            raise ValueError("event_condition is required for monitor action")
        
        # Lazy import concrete layer from WebAgent
        try:
            from webaxon.automation.monitor import (
                MonitorCondition,
                MonitorConditionType,
                create_monitor,
            )
        except ImportError as e:
            raise ImportError(
                "Element monitoring requires the webaxon package. "
                "Install it or use MonitorNode directly with a custom iteration callable."
            ) from e
        
        # Parse condition type
        try:
            condition_type = MonitorConditionType(event_condition)
        except ValueError:
            valid_types = [t.value for t in MonitorConditionType if t != MonitorConditionType.CUSTOM]
            raise ValueError(
                f"Unknown event_condition: '{event_condition}'. "
                f"Valid types: {valid_types}"
            )
        
        # Normalize string target to TargetSpec (create_monitor accepts TargetSpec or TargetSpecWithFallback)
        if isinstance(target, str):
            # String target → use default FRAMEWORK_ID strategy
            normalized_target = TargetSpec(strategy=TargetStrategy.FRAMEWORK_ID, value=target)
        elif isinstance(target, (TargetSpec, TargetSpecWithFallback)):
            normalized_target = target
        else:
            raise ValueError(f"target must be str, TargetSpec, or TargetSpecWithFallback, got {type(target)}")

        # Create MonitorCondition
        monitor_condition = MonitorCondition(
            condition_type=condition_type,
            event_confirmation_time=event_confirmation_time,
        )

        # Create the iteration callable, setup action, and verify_setup using factory function
        # Pass enable_auto_setup to enable auto-mode for use_visible_detection:
        # When continuous=True and enable_auto_setup=False, visibility detection is auto-enabled
        iteration, default_setup_action, default_verify_setup = create_monitor(
            webdriver=self.action_executor,
            target=normalized_target,
            condition=monitor_condition,
            interval=interval,
            continuous=continuous,
            enable_auto_setup=enable_auto_setup,
            # Pass action_executor for agent-based target resolution
            action_executor=self.action_executor,
            # Provide HTML context for xpath caching with options=['static']
            html_context_provider=lambda: self.action_executor.page_source if hasattr(self.action_executor, 'page_source') else None,
        )
        
        # Create MonitorNode (generic) with element iteration (concrete)
        from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import ResultPassDownMode

        if continuous:
            # continuous=True: Graph-based looping via self-edge + NextNodesSelector
            # Disable internal retry mechanism - the self-edge handles looping
            monitor_node = MonitorNode(
                name=f"monitor_{len(self._nodes)}",
                iteration=iteration,
                setup_action=default_setup_action,
                enable_auto_setup=enable_auto_setup,
                verify_setup=default_verify_setup,
                enable_verify_setup=enable_verify_setup,
                poll_interval=interval,  # Delay between polling iterations
                max_repeat=1,  # Run once per graph iteration (self-edge handles repeat)
                repeat_condition=None,  # No internal repeat
                output_validator=None,  # No output validation (NextNodesSelector controls flow)
                fallback_result=None,
                min_repeat_wait=0,
                max_repeat_wait=0,
                result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                # Propagate debug config from graph to node
                copy_debuggable_config_from=self,
            )
        else:
            # continuous=False: Internal looping via repeat_condition + max_repeat
            # The internal retry mechanism handles polling until condition is met
            def should_continue_monitoring(result, **kwargs):
                """Continue while condition NOT met."""
                if isinstance(result, NextNodesSelector):
                    monitor_result = result.result
                    if isinstance(monitor_result, MonitorResult):
                        return not monitor_result.success
                return True  # First iteration, continue

            def is_condition_met(result):
                """Valid when condition is met."""
                if isinstance(result, NextNodesSelector):
                    monitor_result = result.result
                    return isinstance(monitor_result, MonitorResult) and monitor_result.success
                return False

            fallback = MonitorResult(
                success=False,
                status=MonitorStatus.MAX_ITERATIONS,
                check_count=max_checks,
            )

            monitor_node = MonitorNode(
                name=f"monitor_{len(self._nodes)}",
                iteration=iteration,
                setup_action=default_setup_action,
                enable_auto_setup=enable_auto_setup,
                verify_setup=default_verify_setup,
                enable_verify_setup=enable_verify_setup,
                poll_interval=interval,  # Delay between polling iterations
                max_repeat=max_checks,
                repeat_condition=should_continue_monitoring,
                output_validator=is_condition_met,
                fallback_result=fallback,
                min_repeat_wait=0,
                max_repeat_wait=0,
                result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                # Propagate debug config from graph to node
                copy_debuggable_config_from=self,
            )
        
        self._nodes.append(monitor_node)

        # Link to current node
        self._current_node.add_next(monitor_node)

        # Create a new ActionSequenceNode for subsequent actions.
        # MonitorNode doesn't have add_action method, so we need a new node
        # for any actions that come after the monitor.
        next_node = self._create_node(condition=None)
        monitor_node.add_next(next_node)

        # For continuous monitoring, add self-edge to enable the loop:
        # monitor → downstream actions → monitor → downstream actions → ...
        if continuous:
            monitor_node.add_next(monitor_node)

        self._current_node = next_node

        return self

    def _push_action_branch_context(self, branch_actions: List[Action]) -> None:
        """Push a branch context for collecting actions.
        
        When a branch context is active, graph.action() calls will add actions
        to the top of the stack (branch_actions list) instead of the current node.
        
        This is used by TargetNotFoundContext to collect actions into the
        parent action's target_not_found_actions list.
        
        Args:
            branch_actions: The list to collect actions into.
        """
        self._action_branch_stack.append(branch_actions)

    def _pop_action_branch_context(self) -> None:
        """Pop a branch context.
        
        Restores the previous context so that graph.action() calls will add
        actions to the previous branch context (or to the current node if
        the stack is now empty).
        
        Safe to call even if the stack is empty (no-op in that case).
        """
        if self._action_branch_stack:
            self._action_branch_stack.pop()

    def condition(
        self,
        condition_func: Optional[Callable[[ExecutionResult], bool]] = None,
        *,
        value_extractor: Optional[Callable[[ExecutionResult], Any]] = None,
        max_repeat: int = 1,
        repeat_condition: Optional[Callable[..., bool]] = None,
        fallback_result: Any = None,
        min_repeat_wait: float = 0,
        max_repeat_wait: float = 0,
        retry_on_exceptions: Optional[List[type]] = None,
        output_validator: Optional[Callable[..., bool]] = None,
    ) -> 'ConditionContext':
        """
        Create a conditional branch point for if-elif-else syntax.

        Returns a ConditionContext that enables Python's native if-elif-else
        syntax for building conditional branches.

        At build time, all branches are recorded. At runtime, conditions are
        evaluated to determine which branch to execute.

        Args:
            condition_func: Callable evaluated at runtime; if False, branch is skipped.
                           Optional when using convenience methods like if_gte().
            value_extractor: Callable to extract a value from the result for
                            comparison methods (if_gt, if_gte, if_eq, etc.)

        Returns:
            ConditionContext for use with if-elif-else syntax

        Usage:
            # With explicit condition:
            with graph.condition(lambda r: r.success) as branch:
                with branch.if_true():
                    graph.action('click', '#success')
                with branch.if_false():
                    graph.action('click', '#retry')

            # With value_extractor and comparison methods:
            with graph.condition(value_extractor=get_last_value) as branch:
                with branch.if_gte(90):
                    graph.add(value=100)  # A
                with branch.elseif_gte(80):
                    graph.add(value=80)   # B
                with branch.else_():
                    graph.add(value=60)   # C
        """
        return ConditionContext(self, condition_func, value_extractor)
    
    def _create_branch_node(
        self,
        condition_func: Callable[[ExecutionResult], bool],
    ) -> 'ActionSequenceNode':
        """Create a branch node for conditional execution.
        
        Called by ConditionContext when entering an if-block.
        The condition_func is used both as:
        - condition: stored for serialization
        - repeat_condition: evaluated at runtime to determine if branch executes
        """
        new_node = self._create_node(
            condition=condition_func,
            repeat_condition=condition_func,
        )
        self._current_node.add_next(new_node)
        return new_node
    
    def _push_branch(self, branch_node: 'ActionSequenceNode', is_true_branch: bool):
        """Push a branch context onto the stack.
        
        Called by ConditionContext when entering an if-block.
        """
        parent = self._current_node
        self._branch_stack.append(BranchContext(
            branch_node=branch_node,
            is_true_branch=is_true_branch,
            parent_node=parent,
        ))
        self._current_node = branch_node
    
    def _pop_branch(self):
        """Pop a branch context from the stack.
        
        Called by ConditionContext when exiting an if-block.
        """
        if self._branch_stack:
            ctx = self._branch_stack.pop()
            # Restore to parent node for else/elif handling or continuation
            if not self._branch_stack:
                # No more branches, stay at current node for continuation
                pass
            else:
                # Still in nested branches
                self._current_node = self._branch_stack[-1].branch_node
    
    def else_branch(self) -> 'ActionGraph':
        """Start an else branch after a condition.
        
        Alternative to Python's else syntax for explicit else handling.
        
        Usage:
            if graph.condition(lambda r: r.success):
                graph.action('click', '#success')
            graph.else_branch()
            graph.action('click', '#retry')
        """
        if not self._branch_stack:
            raise RuntimeError("else_branch() called without a preceding condition()")
        
        ctx = self._branch_stack[-1]
        # Create else node (no condition = always executes if reached)
        else_node = self._create_node(condition=None)
        ctx.parent_node.add_next(else_node)
        self._current_node = else_node
        self._else_branch_node = else_node
        return self
    
    def branch(
        self,
        condition: Callable[[ExecutionResult], bool],
        if_true: Optional[Callable[['ActionGraph'], None]] = None,
        elseif: Optional[List[Tuple[Callable[[ExecutionResult], bool], Callable[['ActionGraph'], None]]]] = None,
        if_false: Optional[Callable[['ActionGraph'], None]] = None,
    ) -> 'ActionGraph':
        """Callback-based branching API with multi-branch support.

        Creates conditional branches using callback functions. All branches
        are recorded at build time; at runtime, conditions are evaluated
        in order to determine which branch executes.

        Args:
            condition: Callable evaluated at runtime for the first branch
            if_true: Callback for when condition is True
            elseif: List of (condition, callback) tuples for additional branches.
                    Each elseif condition is checked only if all prior conditions were False.
            if_false: Callback for when all conditions are False (else branch)

        Returns:
            self for method chaining

        Usage:
            # Simple 2-way branch:
            graph.branch(
                condition=lambda r: r.success,
                if_true=lambda g: g.action('click', '#success'),
                if_false=lambda g: g.action('click', '#retry'),
            )

            # Multi-way branch (if/elif/elif/else):
            graph.branch(
                condition=lambda r, **kw: get_value(r) >= 90,
                if_true=lambda g: g.add(value=100),  # A
                elseif=[
                    (lambda r, **kw: get_value(r) >= 80, lambda g: g.add(value=80)),  # B
                    (lambda r, **kw: get_value(r) >= 70, lambda g: g.add(value=70)),  # C
                ],
                if_false=lambda g: g.add(value=60),  # D (else)
            )
        """
        parent_node = self._current_node

        # Track all conditions for building exclusive branches
        prior_conditions: List[Callable] = []

        # Build true branch (if)
        if if_true is not None:
            true_node = self._create_branch_node(condition)
            self._push_branch(true_node, is_true_branch=True)
            if_true(self)
            self._pop_branch()
            self._current_node = parent_node
            prior_conditions.append(condition)

        # Build elseif branches
        if elseif is not None:
            for elseif_condition, elseif_callback in elseif:
                # Create condition: not any prior AND this condition
                captured_priors = prior_conditions.copy()
                captured_cond = elseif_condition

                def elseif_combined_condition(result, _priors=captured_priors, _cond=captured_cond, **kwargs):
                    # All prior conditions must be False
                    for prior in _priors:
                        if prior(result, **kwargs):
                            return False
                    # And this condition must be True
                    return _cond(result, **kwargs)

                elseif_node = self._create_branch_node(elseif_combined_condition)
                self._push_branch(elseif_node, is_true_branch=True)
                elseif_callback(self)
                self._pop_branch()
                self._current_node = parent_node
                prior_conditions.append(elseif_condition)

        # Build false branch (else) - when all prior conditions are False
        if if_false is not None:
            captured_all_priors = prior_conditions.copy()

            def else_condition(result, _priors=captured_all_priors, **kwargs):
                # All prior conditions must be False
                for prior in _priors:
                    if prior(result, **kwargs):
                        return False
                return True

            false_node = self._create_branch_node(else_condition)
            self._push_branch(false_node, is_true_branch=False)
            if_false(self)
            self._pop_branch()
            self._current_node = parent_node

        return self

    def loop(
        self,
        condition: Callable[[ExecutionResult], bool],
        max_loop: int = 1000,
        advance: Optional[Callable[[ExecutionResult], ExecutionResult]] = None,
    ) -> 'ActionGraph':
        """Create a blocking while loop.
        
        At runtime, uses WorkGraphNode's execute_with_retry to repeatedly
        check condition and execute advance (if provided) while condition is True.
        
        Implementation reuses:
        - WorkGraphNode's repeat_condition + max_repeat (via execute_with_retry)
        
        Args:
            condition: Callable that returns True to continue looping.
                       Signature: (result, **kwargs) -> bool
            max_loop: Safety limit on iterations (default 1000).
            advance: Optional callable executed each iteration.
                     Signature: (prev_result, **kwargs) -> result
                     Typically used for testing to simulate external changes.
                     In browser monitoring, often not needed as condition
                     checks external state directly.
        
        Returns:
            self for method chaining.
        
        Example (browser monitoring - no advance needed):
            graph.loop(
                condition=lambda r, **kw: not page_has_element(r, '#done'),
                max_loop=100,
            )
        
        Example (testing - advance simulates state change):
            counter = Counter()
            graph.loop(
                condition=lambda r, **kw: counter.value < 100,
                advance=lambda r, **kw: counter.increment(),
            )
        """
        # Create loop node using WorkGraphNode's repeat mechanism
        # 
        # execute_with_retry behavior:
        # - max_retry <= 1: single execution (1 iteration)
        # - max_retry >= 2: max_retry + 1 iterations (due to check-after-increment)
        #
        # To get exactly max_loop iterations:
        # - max_loop = 1: max_repeat = 1 (single execution path)
        # - max_loop >= 2: max_repeat = max_loop - 1 (gives max_loop iterations)
        #
        # However, max_repeat = 1 takes single execution path, so max_loop = 2
        # would only get 1 iteration. We handle this by using max_repeat = max(2, max_loop - 1)
        # and wrapping the condition to enforce the actual max_loop limit.
        
        # Track iteration count to enforce max_loop limit
        iteration_state = {'count': 0}
        user_condition = condition
        
        def wrapped_condition(result, _state=iteration_state, _max=max_loop, _cond=user_condition, **kwargs):
            """Wrap user condition to enforce max_loop limit."""
            if _state['count'] >= _max:
                return False
            return _cond(result, **kwargs)
        
        # Wrap advance to track iterations
        user_advance = advance
        
        def wrapped_advance(result, _state=iteration_state, _adv=user_advance, **kwargs):
            """Wrap advance to count iterations."""
            _state['count'] += 1
            if _adv is not None:
                return _adv(result, **kwargs)
            return result
        
        # Use max_repeat = max(2, max_loop) to ensure we enter the while loop
        # The wrapped_condition will enforce the actual max_loop limit
        actual_max_repeat = max(2, max_loop)
        
        loop_node = ActionSequenceNode(
            name=f"loop_{len(self._nodes)}",
            action_executor=self.action_executor,
            action_metadata=self.action_metadata,
            template_engine=self.template_engine,
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
            repeat_condition=wrapped_condition,
            max_repeat=actual_max_repeat,
            output_validator=lambda r: False,  # Force retry until condition fails
            # Propagate persistence settings from graph to node
            enable_result_save=self.enable_result_save,
            result_save_dir=self.result_save_dir,
            # Propagate debug config from graph to node
            copy_debuggable_config_from=self,
        )
        
        # Mark as loop node for serialization
        loop_node._is_loop_node = True
        loop_node._loop_max_loop = max_loop
        loop_node._loop_user_condition = user_condition
        loop_node._loop_user_advance = user_advance
        
        # Use wrapped advance
        loop_node.value = wrapped_advance
        
        # Add to graph
        self._current_node.add_next(loop_node)
        self._nodes.append(loop_node)
        self._current_node = loop_node
        
        return self

    def __call__(self, **variables) -> ExecutionResult:
        """
        Execute workflow with template variables (callable ActionGraph).

        Makes ActionGraph callable for convenient workflow invocation:
            workflow = ActionGraph(...).visit_url('{base_url}').click('#submit')
            result = workflow(base_url='https://example.com')

        Args:
            **variables: Values for template variables detected during build.
                Must provide all variables found in required_variables.

        Returns:
            ExecutionResult from workflow execution.

        Raises:
            ValueError: If required template variables are missing.

        Example:
            >>> search_workflow = (
            ...     ActionGraph(action_executor=driver, action_metadata=registry)
            ...     .visit_url('{base_url}')
            ...     .input_text('#search', text='{query}')
            ...     .wait(seconds='{delay}')  # Type coerced to float
            ... )
            >>> result = search_workflow(base_url='https://example.com', query='test', delay=2.0)
        """
        # Validate all required variables are provided
        required = self.required_variables
        missing = required - set(variables.keys())
        if missing:
            raise ValueError(
                f"Missing required template variables: {missing}. "
                f"Required variables: {required}"
            )

        return self.execute(initial_variables=variables)

    def execute(self, initial_variables: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute the built workflow.

        Args:
            initial_variables: Initial variable values for template substitution.

        Returns:
            ExecutionResult containing success status and final context.

        Raises:
            pydantic.ValidationError: If the graph has no actions to execute.
        """
        import logging
        _logger = logging.getLogger(__name__)

        from pydantic_core import InitErrorDetails, PydanticCustomError
        from pydantic import ValidationError
        from .common import ExecutionRuntime

        _logger.debug(f"[ActionGraph.execute] Starting execution with {len(self._nodes)} nodes")
        for i, node in enumerate(self._nodes):
            if hasattr(node, '_actions'):
                _logger.debug(f"[ActionGraph.execute]   Node {i} ({node.name}): {len(node._actions)} actions")
            else:
                _logger.debug(f"[ActionGraph.execute]   Node {i} ({node.name}): MonitorNode")

        # Validate that there are actions to execute
        # Only check ActionSequenceNodes with _actions attribute
        # Skip validation if loops or conditions are present (they don't require actions)
        has_actions = any(
            node._actions for node in self._nodes
            if hasattr(node, '_actions')
        )
        has_loops = any(
            getattr(node, 'max_repeat', 1) > 1 or getattr(node, 'repeat_condition', None) is not None
            for node in self._nodes
        )
        if not self._nodes or (not has_actions and not has_loops):
            raise ValidationError.from_exception_data(
                'ActionGraph',
                [
                    InitErrorDetails(
                        type=PydanticCustomError(
                            'value_error',
                            'Action sequence must contain at least one action'
                        ),
                        loc=('actions',),
                        input=[],
                    )
                ]
            )

        self._set_start_node()
        _logger.debug(f"[ActionGraph.execute] Start nodes set: {[n.name for n in self.start_nodes]}")

        # Create initial ExecutionResult with variables for first node
        # ActionSequenceNode._execute_sequence expects ExecutionResult as first arg
        initial_result = ExecutionResult(
            success=True,
            context=ExecutionRuntime(variables=initial_variables or {}),
        )
        _logger.debug(f"[ActionGraph.execute] Calling self.run() with initial_result")
        result = self.run(initial_result)
        _logger.debug(f"[ActionGraph.execute] self.run() returned: {type(result)}")
        return result

    def _set_start_node(self):
        """Set the root node as WorkGraph's start node for execution."""
        if self._nodes:
            self.start_nodes = [self._nodes[0]]

    # Serializable interface methods
    def to_serializable_obj(
        self,
        mode: str = 'auto',
        _output_format: Optional[str] = None
    ) -> Union[Dict[str, Any], 'ActionGraph']:
        """Convert ActionGraph to serializable Python object.
        
        Overrides Serializable.to_serializable_obj() to provide custom
        serialization that preserves graph structure, nodes, and connections.
        
        For Python format (_output_format='python'), returns self to indicate
        special handling is needed by serialize().
        
        Args:
            mode: Serialization mode ('auto', 'dict', 'pickle')
            _output_format: Target output format for conflict detection
        
        Returns:
            - self when _output_format='python' (special handling)
            - Dict containing version, id, nodes list, and config otherwise
        """
        # For Python format, return self to indicate special handling
        if _output_format == 'python':
            return self
        
        return {
            FIELD_TYPE: type(self).__name__,
            FIELD_MODULE: type(self).__module__,
            FIELD_SERIALIZATION: SERIALIZATION_DICT,
            "version": "1.0",
            "id": self.name or f"graph_{id(self)}",
            "nodes": [self._node_to_dict(node) for node in self._nodes],
            "config": {
                "enable_result_save": self.enable_result_save,
                "result_save_dir": self.result_save_dir,
            }
        }

    def serialize(
        self,
        output_format: str = 'json',
        path: Optional[Union[str, Path]] = None,
        serializable_obj_mode: str = 'auto',
        **kwargs
    ) -> str:
        """Serialize ActionGraph to specified format.
        
        Extended to support output_format='python' for Python script generation.
        
        Args:
            output_format: Output format ('json', 'yaml', 'pickle', or 'python')
            path: Optional file path to write result
            serializable_obj_mode: Mode for to_serializable_obj ('auto', 'dict', 'pickle')
            **kwargs: Format-specific options:
                - For 'python': branching_style, include_imports, variable_name
                - For 'json': indent
        
        Returns:
            Serialized string (Python script for 'python' format)
        
        Raises:
            ValueError: If output_format is not supported
        """
        # Handle Python format specially
        if output_format == 'python':
            return self._generate_python_script(
                path=path,
                branching_style=kwargs.get('branching_style', 'match'),
                include_imports=kwargs.get('include_imports', True),
                variable_name=kwargs.get('variable_name', 'graph'),
            )
        
        # Delegate to parent for other formats
        return super().serialize(
            output_format=output_format,
            path=path,
            serializable_obj_mode=serializable_obj_mode,
            **kwargs
        )
    
    @classmethod
    def from_serializable_obj(
        cls,
        obj: Dict[str, Any],
        action_executor: Union[Callable, MultiActionExecutor] = None,
        action_metadata: Optional[ActionMetadataRegistry] = None,
        **context
    ) -> 'ActionGraph':
        """Reconstruct ActionGraph from serializable dict.
        
        Overrides Serializable.from_serializable_obj() to provide custom
        deserialization that reconstructs graph structure with all nodes
        and connections.
        
        Args:
            obj: The serializable object (dict)
            action_executor: Callable for executing actions (required)
            action_metadata: Action type registry (optional)
            **context: Additional context parameters
        
        Returns:
            Reconstructed ActionGraph instance
        
        Raises:
            ValueError: If action_executor is not provided
        """
        if action_executor is None:
            action_executor = context.get('action_executor')
        if action_executor is None:
            raise ValueError("Required context parameter 'action_executor' not provided")
        
        if action_metadata is None:
            action_metadata = context.get('action_metadata', ActionMetadataRegistry())
        
        config = obj.get("config", {})
        
        graph = cls(
            action_executor=action_executor,
            action_metadata=action_metadata,
            enable_result_save=config.get("enable_result_save", False),
            result_save_dir=config.get("result_save_dir"),
        )
        graph._nodes.clear()
        graph._current_node = None

        node_map = {}
        for node_data in obj.get("nodes", []):
            node = cls._node_from_dict(node_data, action_executor, action_metadata)
            graph._nodes.append(node)
            node_map[node_data["id"]] = node

        for node_data in obj.get("nodes", []):
            node = node_map[node_data["id"]]
            for next_id in node_data.get("next_nodes", []):
                if next_id in node_map:
                    node.add_next(node_map[next_id])

        if graph._nodes:
            graph._current_node = graph._nodes[-1]

        return graph

    def _node_to_dict(self, node: 'ActionSequenceNode') -> Dict[str, Any]:
        """Convert a node to dictionary format."""
        result = {
            "id": node.name,
            "condition": self._condition_to_string(node.condition),
            "actions": [self._action_to_dict(a) for a in node._actions],
            "next_nodes": [n.name for n in (node.next or [])],
            "retry_config": {
                "max_repeat": node.max_repeat,
                "min_repeat_wait": node.min_repeat_wait,
                "max_repeat_wait": node.max_repeat_wait,
            }
        }
        
        # Handle loop nodes
        if getattr(node, '_is_loop_node', False):
            result["node_behavior"] = "loop"
            result["loop_config"] = {
                "max_loop": getattr(node, '_loop_max_loop', 1000),
                "condition": self._condition_to_string(getattr(node, '_loop_user_condition', None)),
                "has_advance": getattr(node, '_loop_user_advance', None) is not None,
            }
        
        return result

    def _condition_to_string(self, condition: Optional[Callable]) -> Optional[str]:
        """Convert condition callable to string expression."""
        if condition is None:
            return None
        return getattr(condition, '__condition_expr__', repr(condition))

    @staticmethod
    def _action_to_dict(action: Action) -> Dict[str, Any]:
        """Convert Action to dictionary."""
        return action.model_dump(exclude_none=True)

    @classmethod
    def _node_from_dict(
        cls,
        data: Dict[str, Any],
        action_executor: Union[Callable, MultiActionExecutor],
        action_metadata: ActionMetadataRegistry,
    ) -> 'ActionSequenceNode':
        """Create node from dictionary."""
        node_behavior = data.get("node_behavior")
        
        # Handle loop nodes
        if node_behavior == "loop":
            return cls._loop_node_from_dict(data, action_executor, action_metadata)
        
        # Regular node
        condition = cls._condition_from_string(data.get("condition"))
        retry_config = data.get("retry_config", {})

        node = ActionSequenceNode(
            name=data["id"],
            action_executor=action_executor,
            action_metadata=action_metadata,
            condition=condition,
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
            max_repeat=retry_config.get("max_repeat", 1),
            min_repeat_wait=retry_config.get("min_repeat_wait", 0),
            max_repeat_wait=retry_config.get("max_repeat_wait", 0),
        )

        for action_data in data.get("actions", []):
            node.add_action(Action(**action_data))

        return node
    
    @classmethod
    def _loop_node_from_dict(
        cls,
        data: Dict[str, Any],
        action_executor: Union[Callable, MultiActionExecutor],
        action_metadata: ActionMetadataRegistry,
    ) -> 'ActionSequenceNode':
        """Create loop node from dictionary."""
        loop_config = data.get("loop_config", {})
        max_loop = loop_config.get("max_loop", 1000)
        user_condition = cls._condition_from_string(loop_config.get("condition"))
        has_advance = loop_config.get("has_advance", False)
        
        # Recreate the wrapped condition and advance
        iteration_state = {'count': 0}
        
        def wrapped_condition(result, _state=iteration_state, _max=max_loop, _cond=user_condition, **kwargs):
            """Wrap user condition to enforce max_loop limit."""
            if _state['count'] >= _max:
                return False
            if _cond is not None:
                return _cond(result, **kwargs)
            return True
        
        def wrapped_advance(result, _state=iteration_state, **kwargs):
            """Wrap advance to count iterations."""
            _state['count'] += 1
            return result
        
        actual_max_repeat = max(2, max_loop)
        
        node = ActionSequenceNode(
            name=data["id"],
            action_executor=action_executor,
            action_metadata=action_metadata,
            result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
            repeat_condition=wrapped_condition,
            max_repeat=actual_max_repeat,
            output_validator=lambda r: False,
        )
        
        # Mark as loop node
        node._is_loop_node = True
        node._loop_max_loop = max_loop
        node._loop_user_condition = user_condition
        node._loop_user_advance = None  # Can't restore callable from serialization
        
        node.value = wrapped_advance
        
        return node

    @staticmethod
    def _condition_from_string(expr: Optional[str]) -> Optional[Callable]:
        """Create condition callable from string expression."""
        if expr is None:
            return None

        def condition_func(result: ExecutionResult) -> bool:
            return eval(expr, {"__builtins__": {}}, {"result": result})

        condition_func.__condition_expr__ = expr
        return condition_func

    # Python code generation helper methods
    def _action_to_python(self, action: Action, indent: int = 0) -> str:
        """Convert Action to Python method call string.
        
        Args:
            action: The Action object to convert
            indent: Number of spaces for indentation
        
        Returns:
            Python code string like: graph.action("click", target="submit_btn")
        
        Example output:
            graph.action("click", target="submit_btn", args={"timeout": 5})
        """
        indent_str = " " * indent
        parts = [f'"{action.type}"']
        
        if action.target is not None:
            target_str = self._target_to_python(action.target)
            parts.append(f"target={target_str}")
        
        if action.args:
            args_str = self._args_to_python(action.args)
            parts.append(f"args={args_str}")
        
        if action.id and not action.id.startswith("action_"):
            parts.append(f'action_id="{action.id}"')
        
        return f"{indent_str}graph.action({', '.join(parts)})"

    def _target_to_python(self, target: Union[TargetSpec, TargetSpecWithFallback, str]) -> str:
        """Convert target (str, TargetSpec, TargetSpecWithFallback) to Python code.
        
        Args:
            target: The target specification
        
        Returns:
            Python code string representing the target
        
        Examples:
            "submit_btn" -> '"submit_btn"'
            TargetSpec(selector=".btn") -> 'TargetSpec(selector=".btn")'
        """
        if isinstance(target, str):
            # Escape quotes in the string
            escaped = target.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(target, TargetSpec):
            parts = []
            if target.strategy is not None:
                parts.append(f'strategy="{target.strategy}"')
            if target.value is not None:
                escaped = target.value.replace('\\', '\\\\').replace('"', '\\"')
                parts.append(f'value="{escaped}"')
            if target.description is not None:
                escaped = target.description.replace('\\', '\\\\').replace('"', '\\"')
                parts.append(f'description="{escaped}"')
            return f"TargetSpec({', '.join(parts)})"
        elif isinstance(target, TargetSpecWithFallback):
            strategies_str = ", ".join(
                self._target_to_python(s) for s in target.strategies
            )
            return f"TargetSpecWithFallback(strategies=[{strategies_str}])"
        else:
            return repr(target)

    def _condition_to_python(self, condition: Optional[Callable]) -> str:
        """Convert condition to lambda expression using __condition_expr__ attribute.
        
        Args:
            condition: The condition callable
        
        Returns:
            Python code string for the condition lambda
        
        Example:
            condition with __condition_expr__ = "result.success"
            -> "lambda r: r.success"
        """
        if condition is None:
            return "None"
        
        expr = getattr(condition, '__condition_expr__', None)
        if expr:
            # Convert result.xxx to r.xxx for lambda
            if expr.startswith("result."):
                expr = "r." + expr[7:]
            return f"lambda r: {expr}"
        
        # Fallback to repr
        return repr(condition)

    def _args_to_python(self, args: Dict[str, Any]) -> str:
        """Convert args dict to Python dict literal string.
        
        Args:
            args: Dictionary of action arguments
        
        Returns:
            Python code string for the dict
        
        Example:
            {"text": "hello", "timeout": 5} -> '{"text": "hello", "timeout": 5}'
        """
        if not args:
            return "{}"
        
        parts = []
        for key, value in args.items():
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                parts.append(f'"{key}": "{escaped}"')
            elif isinstance(value, bool):
                parts.append(f'"{key}": {str(value)}')
            elif value is None:
                parts.append(f'"{key}": None')
            else:
                parts.append(f'"{key}": {repr(value)}')
        
        return "{" + ", ".join(parts) + "}"

    def _generate_python_script(
        self,
        path: Optional[Union[str, Path]] = None,
        branching_style: str = 'match',
        include_imports: bool = True,
        variable_name: str = 'graph',
    ) -> str:
        """Generate executable Python script from graph structure.
        
        Args:
            path: Optional file path to write the script
            branching_style: Style for conditional branches:
                - 'match': Python 3.10+ match-case syntax (default)
                - 'with': Context manager syntax with if_true()/if_false()
                - 'branch': Callback-based graph.branch() calls
                - 'if': If-statement syntax with else_branch()
            include_imports: Whether to include import statements
            variable_name: Variable name for the graph (default: 'graph')
        
        Returns:
            Generated Python script as string
        
        Raises:
            ValueError: If branching_style is not valid
        """
        valid_styles = ('match', 'with', 'branch', 'if')
        if branching_style not in valid_styles:
            raise ValueError(
                f"Invalid branching_style: {branching_style}. "
                f"Valid options: {valid_styles}"
            )
        
        lines = []
        
        # Generate imports
        if include_imports:
            lines.append("from agent_foundation.automation.schema import ActionGraph, ConditionContext")
            lines.append("from agent_foundation.automation.schema import Action, TargetSpec, TargetSpecWithFallback")
            lines.append("")
        
        # Generate graph construction
        lines.append(f"{variable_name} = ActionGraph(action_executor=driver, action_metadata=registry)")
        
        # Process nodes
        self._generate_nodes_python(
            lines=lines,
            variable_name=variable_name,
            branching_style=branching_style,
        )
        
        script = "\n".join(lines)
        
        # Write to file if path provided
        if path:
            Path(path).write_text(script, encoding='utf-8')
        
        return script

    def _generate_nodes_python(
        self,
        lines: List[str],
        variable_name: str,
        branching_style: str,
        indent: int = 0,
    ) -> None:
        """Generate Python code for all nodes in the graph.
        
        Args:
            lines: List to append generated lines to
            variable_name: Variable name for the graph
            branching_style: Style for conditional branches
            indent: Current indentation level
        """
        # Track which nodes have been processed
        processed = set()
        
        # Start with root node (first node)
        if self._nodes:
            self._generate_node_python(
                node=self._nodes[0],
                lines=lines,
                variable_name=variable_name,
                branching_style=branching_style,
                indent=indent,
                processed=processed,
            )

    def _generate_node_python(
        self,
        node: 'ActionSequenceNode',
        lines: List[str],
        variable_name: str,
        branching_style: str,
        indent: int,
        processed: set,
    ) -> None:
        """Generate Python code for a single node and its children.
        
        Args:
            node: The node to generate code for
            lines: List to append generated lines to
            variable_name: Variable name for the graph
            branching_style: Style for conditional branches
            indent: Current indentation level
            processed: Set of already processed node names
        """
        if node.name in processed:
            return
        processed.add(node.name)
        
        indent_str = "    " * indent
        
        # Generate actions for this node
        for action in node._actions:
            action_code = self._action_to_python(action, indent=indent * 4)
            # Replace 'graph' with actual variable name
            action_code = action_code.replace("graph.", f"{variable_name}.")
            lines.append(action_code)
        
        # Handle child nodes (branches)
        next_nodes = node.next or []
        if not next_nodes:
            return
        
        # Check if we have conditional branches
        conditional_nodes = [n for n in next_nodes if n.condition is not None]
        
        if conditional_nodes:
            self._generate_branches_python(
                parent_node=node,
                branch_nodes=conditional_nodes,
                lines=lines,
                variable_name=variable_name,
                branching_style=branching_style,
                indent=indent,
                processed=processed,
            )
        else:
            # Non-conditional continuation
            for next_node in next_nodes:
                self._generate_node_python(
                    node=next_node,
                    lines=lines,
                    variable_name=variable_name,
                    branching_style=branching_style,
                    indent=indent,
                    processed=processed,
                )

    def _generate_branches_python(
        self,
        parent_node: 'ActionSequenceNode',
        branch_nodes: List['ActionSequenceNode'],
        lines: List[str],
        variable_name: str,
        branching_style: str,
        indent: int,
        processed: set,
    ) -> None:
        """Generate Python code for conditional branches.
        
        Args:
            parent_node: The parent node containing the branches
            branch_nodes: List of branch nodes with conditions
            lines: List to append generated lines to
            variable_name: Variable name for the graph
            branching_style: Style for conditional branches
            indent: Current indentation level
            processed: Set of already processed node names
        """
        indent_str = "    " * indent
        
        # Get condition from first branch node
        if not branch_nodes:
            return
        
        first_branch = branch_nodes[0]
        condition_str = self._condition_to_python(first_branch.condition)
        
        # Find true and false branches
        true_branch = first_branch
        false_branch = branch_nodes[1] if len(branch_nodes) > 1 else None
        
        if branching_style == 'match':
            self._generate_match_style(
                condition_str, true_branch, false_branch,
                lines, variable_name, indent, processed
            )
        elif branching_style == 'with':
            self._generate_with_style(
                condition_str, true_branch, false_branch,
                lines, variable_name, indent, processed
            )
        elif branching_style == 'branch':
            self._generate_branch_style(
                condition_str, true_branch, false_branch,
                lines, variable_name, indent, processed
            )
        elif branching_style == 'if':
            self._generate_if_style(
                condition_str, true_branch, false_branch,
                lines, variable_name, indent, processed
            )

    def _generate_match_style(
        self,
        condition_str: str,
        true_branch: 'ActionSequenceNode',
        false_branch: Optional['ActionSequenceNode'],
        lines: List[str],
        variable_name: str,
        indent: int,
        processed: set,
    ) -> None:
        """Generate match-case style branching code."""
        indent_str = "    " * indent
        
        lines.append("")
        lines.append(f"{indent_str}match {variable_name}.condition({condition_str}):")
        lines.append(f"{indent_str}    case ConditionContext.TRUE:")
        
        # Generate true branch actions
        processed.add(true_branch.name)
        for action in true_branch._actions:
            action_code = self._action_to_python(action, indent=(indent + 2) * 4)
            action_code = action_code.replace("graph.", f"{variable_name}.")
            lines.append(action_code)
        
        if false_branch:
            lines.append(f"{indent_str}    case ConditionContext.FALSE:")
            processed.add(false_branch.name)
            for action in false_branch._actions:
                action_code = self._action_to_python(action, indent=(indent + 2) * 4)
                action_code = action_code.replace("graph.", f"{variable_name}.")
                lines.append(action_code)

    def _generate_with_style(
        self,
        condition_str: str,
        true_branch: 'ActionSequenceNode',
        false_branch: Optional['ActionSequenceNode'],
        lines: List[str],
        variable_name: str,
        indent: int,
        processed: set,
    ) -> None:
        """Generate context manager style branching code."""
        indent_str = "    " * indent
        
        lines.append("")
        lines.append(f"{indent_str}with {variable_name}.condition({condition_str}) as branch:")
        lines.append(f"{indent_str}    with branch.if_true():")
        
        # Generate true branch actions
        processed.add(true_branch.name)
        for action in true_branch._actions:
            action_code = self._action_to_python(action, indent=(indent + 2) * 4)
            action_code = action_code.replace("graph.", f"{variable_name}.")
            lines.append(action_code)
        
        if false_branch:
            lines.append(f"{indent_str}    with branch.if_false():")
            processed.add(false_branch.name)
            for action in false_branch._actions:
                action_code = self._action_to_python(action, indent=(indent + 2) * 4)
                action_code = action_code.replace("graph.", f"{variable_name}.")
                lines.append(action_code)

    def _generate_branch_style(
        self,
        condition_str: str,
        true_branch: 'ActionSequenceNode',
        false_branch: Optional['ActionSequenceNode'],
        lines: List[str],
        variable_name: str,
        indent: int,
        processed: set,
    ) -> None:
        """Generate callback-based branching code."""
        indent_str = "    " * indent
        
        lines.append("")
        lines.append(f"{indent_str}{variable_name}.branch(")
        lines.append(f"{indent_str}    condition={condition_str},")
        
        # Generate true branch
        processed.add(true_branch.name)
        if len(true_branch._actions) == 1:
            action = true_branch._actions[0]
            action_code = self._action_to_python(action, indent=0)
            action_code = action_code.replace("graph.", "g.")
            lines.append(f"{indent_str}    if_true=lambda g: {action_code},")
        else:
            lines.append(f"{indent_str}    if_true=lambda g: (")
            for action in true_branch._actions:
                action_code = self._action_to_python(action, indent=0)
                action_code = action_code.replace("graph.", "g.")
                lines.append(f"{indent_str}        {action_code},")
            lines.append(f"{indent_str}    ),")
        
        # Generate false branch
        if false_branch:
            processed.add(false_branch.name)
            if len(false_branch._actions) == 1:
                action = false_branch._actions[0]
                action_code = self._action_to_python(action, indent=0)
                action_code = action_code.replace("graph.", "g.")
                lines.append(f"{indent_str}    if_false=lambda g: {action_code},")
            else:
                lines.append(f"{indent_str}    if_false=lambda g: (")
                for action in false_branch._actions:
                    action_code = self._action_to_python(action, indent=0)
                    action_code = action_code.replace("graph.", "g.")
                    lines.append(f"{indent_str}        {action_code},")
                lines.append(f"{indent_str}    ),")
        
        lines.append(f"{indent_str})")

    def _generate_if_style(
        self,
        condition_str: str,
        true_branch: 'ActionSequenceNode',
        false_branch: Optional['ActionSequenceNode'],
        lines: List[str],
        variable_name: str,
        indent: int,
        processed: set,
    ) -> None:
        """Generate if-statement style branching code."""
        indent_str = "    " * indent
        
        lines.append("")
        lines.append(f"{indent_str}if {variable_name}.condition({condition_str}):")
        
        # Generate true branch actions
        processed.add(true_branch.name)
        for action in true_branch._actions:
            action_code = self._action_to_python(action, indent=(indent + 1) * 4)
            action_code = action_code.replace("graph.", f"{variable_name}.")
            lines.append(action_code)
        
        if false_branch:
            # else_branch() is called at the same indentation level as the if
            lines.append(f"{indent_str}{variable_name}.else_branch()")
            processed.add(false_branch.name)
            for action in false_branch._actions:
                # Actions after else_branch() are at the same level as the if block
                action_code = self._action_to_python(action, indent=indent * 4)
                action_code = action_code.replace("graph.", f"{variable_name}.")
                lines.append(action_code)

    # Python deserialization methods
    @classmethod
    def deserialize(
        cls,
        source: Union[str, Path, bytes],
        output_format: str = 'json',
        **context
    ) -> 'ActionGraph':
        """Deserialize ActionGraph from specified format.
        
        Extended to support output_format='python' for Python script parsing.
        
        Args:
            source: Source data (string, file path, or bytes)
            output_format: Input format ('json', 'yaml', 'pickle', or 'python')
            **context: Context parameters:
                - action_executor: Required callable for executing actions
                - action_metadata: Optional ActionMetadataRegistry
        
        Returns:
            Reconstructed ActionGraph instance
        
        Raises:
            ValueError: If output_format is not supported or action_executor missing
            SyntaxError: If Python script has invalid syntax
            FileNotFoundError: If source file doesn't exist
        """
        # Handle Python format specially
        if output_format == 'python':
            action_executor = context.pop('action_executor', None)
            if action_executor is None:
                raise ValueError("Required context parameter 'action_executor' not provided")
            
            action_metadata = context.pop('action_metadata', ActionMetadataRegistry())
            
            return cls._deserialize_python_script(
                source=source,
                action_executor=action_executor,
                action_metadata=action_metadata,
                **context
            )
        
        # Delegate to parent for other formats
        return super().deserialize(
            source=source,
            output_format=output_format,
            **context
        )

    @classmethod
    def _deserialize_python_script(
        cls,
        source: Union[str, Path],
        action_executor: Union[Callable, MultiActionExecutor],
        action_metadata: Optional[ActionMetadataRegistry] = None,
        **context
    ) -> 'ActionGraph':
        """Execute Python script and extract the ActionGraph object.

        Uses exec() to run the script with injected driver and registry,
        then extracts the constructed graph object.

        Args:
            source: Python script string or file path
            action_executor: Callable for executing actions (injected as 'driver')
            action_metadata: Action type registry (injected as 'registry')
            **context: Additional context parameters

        Returns:
            Reconstructed ActionGraph instance

        Raises:
            ValueError: If no ActionGraph found in script
            SyntaxError: If Python script has invalid syntax
            FileNotFoundError: If source file doesn't exist
        """
        if action_metadata is None:
            action_metadata = ActionMetadataRegistry()

        script_content = cls._read_python_source(source)

        # Create namespace with injected dependencies and imports
        namespace = {
            'driver': action_executor,
            'registry': action_metadata,
            # Classes needed by generated scripts
            'ActionGraph': cls,
            'ConditionContext': ConditionContext,
            'Action': Action,
            'ActionSequence': ActionSequence,
            'TargetSpec': TargetSpec,
            'TargetSpecWithFallback': TargetSpecWithFallback,
            'ActionMetadataRegistry': ActionMetadataRegistry,
        }

        # Execute script
        exec(script_content, namespace)

        # Find the ActionGraph instance
        for var_name in ['graph', 'g']:
            if var_name in namespace and isinstance(namespace[var_name], cls):
                return namespace[var_name]

        # Search all values
        for value in namespace.values():
            if isinstance(value, cls):
                return value

        raise ValueError("No ActionGraph object found in script")

    @classmethod
    def _read_python_source(cls, source: Union[str, Path]) -> str:
        """Read Python source from string or file path.
        
        Args:
            source: Python script string or file path
        
        Returns:
            Python script content as string
        
        Raises:
            FileNotFoundError: If source file doesn't exist
        """
        # Check if source is a file path
        if isinstance(source, Path):
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            return source.read_text(encoding='utf-8')
        
        # Check if source string is a file path
        if isinstance(source, str):
            path = Path(source)
            if path.exists() and path.is_file():
                return path.read_text(encoding='utf-8')
        
        # Treat as script content
        return source


@attrs(slots=False)
class ActionSequenceNode(WorkGraphNode):
    """
    WorkGraphNode that holds actions and uses repeat_condition for branching.

    Template Variable System:
    - Aggregates required_variables from all ActionNodes it contains
    - Exposes required_variables for ActionGraph to see what variables this node needs
    - Passes template_engine setting to child ActionNodes
    """
    action_executor: Union[Callable, MultiActionExecutor] = attrib(default=None)
    action_metadata: ActionMetadataRegistry = attrib(default=None)
    condition: Optional[Callable] = attrib(default=None, kw_only=True)
    template_engine: str = attrib(default='python', kw_only=True)
    result_save_dir: Optional[str] = attrib(default=None, kw_only=True)
    _actions: List[Action] = attrib(factory=list)

    # Cached required_variables (computed lazily)
    _cached_required_variables: Optional[Set[str]] = attrib(default=None, init=False)

    def __attrs_post_init__(self):
        self.value = self._execute_sequence
        super().__attrs_post_init__()

    def __str__(self) -> str:
        """Return node display string for inherited str_all_descendants()."""
        return f"[{self.name}] ({len(self._actions)} actions)"

    @property
    def required_variables(self) -> Set[str]:
        """
        Set of template variables required by all actions in this node.

        Aggregates required_variables from all ActionNodes. Computed lazily
        and cached for efficiency.

        Returns:
            Union of all required_variables from child ActionNodes.
        """
        if self._cached_required_variables is not None:
            return self._cached_required_variables

        # Import here to avoid circular import
        from .action_node import ActionNode

        all_vars: Set[str] = set()
        for action in self._actions:
            # Create temporary ActionNode to detect variables
            # (ActionFlow will create actual nodes during execution)
            node = ActionNode(
                action=action,
                action_executor=self.action_executor,
                action_metadata=self.action_metadata,
                template_engine=self.template_engine,
            )
            all_vars.update(node.required_variables)

        self._cached_required_variables = all_vars
        return all_vars

    def add_action(self, action: Action):
        """Add an action to this node's sequence."""
        import logging
        _logger = logging.getLogger(__name__)
        _logger.debug(
            f"[ActionSequenceNode.add_action] action_type={action.type}, "
            f"target={action.target}, target_type={type(action.target).__name__}"
        )
        self._actions.append(action)
        # Invalidate cache when actions change
        self._cached_required_variables = None

    def _get_fallback_result(self, *args, **kwargs):
        """Return prev_result for pass-through when repeat_condition is False."""
        return args[0] if args else None

    def _execute_sequence(self, *args, **kwargs) -> ExecutionResult:
        """Execute the action sequence via ActionFlow.

        If no actions are present, returns the input result unchanged (pass-through).
        """
        import logging
        _logger = logging.getLogger(__name__)
        _logger.debug(f"[ActionSequenceNode._execute_sequence] {self.name}: Starting with {len(self._actions)} actions")

        # Handle empty actions - just pass through the input result
        if not self._actions:
            _logger.debug(f"[ActionSequenceNode._execute_sequence] {self.name}: No actions, pass-through")
            if args and isinstance(args[0], ExecutionResult):
                return args[0]
            return ExecutionResult(
                success=True,
                context=ExecutionRuntime(),
            )

        sequence = ActionSequence(
            id=f"sequence_{self.name}",
            actions=self._actions
        )
        _logger.debug(f"[ActionSequenceNode._execute_sequence] {self.name}: Created sequence with actions: {[a.type for a in self._actions]}")

        variables = {}
        if args and isinstance(args[0], ExecutionResult):
            variables = args[0].context.variables if args[0].context else {}

        executor = ActionFlow(
            action_executor=self.action_executor,
            action_metadata=self.action_metadata,
            template_engine=self.template_engine,
            # Propagate persistence settings from parent node
            enable_result_save=self.enable_result_save,
            result_save_dir=self.result_save_dir,
        )
        _logger.debug(f"[ActionSequenceNode._execute_sequence] {self.name}: Calling ActionFlow.execute()")
        result = executor.execute(sequence=sequence, initial_variables=variables)
        _logger.debug(f"[ActionSequenceNode._execute_sequence] {self.name}: ActionFlow returned success={result.success}")
        return result


def condition_expr(expr: str):
    """
    Decorator to attach expression string to a condition function.
    Makes the condition serializable to JSON.
    """
    def decorator(func):
        func.__condition_expr__ = expr
        return func
    return decorator

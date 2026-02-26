"""
ActionNode - WorkGraphNode subclass for single action execution.

Wraps a single Action for execution within a Workflow or WorkGraph,
leveraging WorkGraphNode's built-in retry mechanism for fallback strategies.

Template Variable System:
- ActionNode autonomously handles template detection and substitution
- Exposes required_variables and output_variable for parent nodes to aggregate
- Supports type coercion for single-var templates based on action's arg_types
"""

import os
import pickle
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union, TYPE_CHECKING

from attr import attrs, attrib

if TYPE_CHECKING:
    from science_modeling_tools.agents.agent import Agent

from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode
from rich_python_utils.common_objects.serializable import (
    Serializable,
    FIELD_TYPE,
    FIELD_MODULE,
    FIELD_SERIALIZATION,
    SERIALIZATION_DICT,
)
from rich_python_utils.common_utils.typing_helper import coerce_to_type

from .common import (
    Action,
    ActionResult,
    ExecutionRuntime,
    TargetNotFoundError,
    TargetSpec,
    TargetSpecWithFallback,
    TargetStrategy,
)
from .action_metadata import ActionMetadataRegistry
from .action_executor import MultiActionExecutor


import logging
logger = logging.getLogger(__name__)

# Template engine support - default to python str.format
SUPPORTED_TEMPLATE_ENGINES = ('python', 'jinja2', 'handlebars', 'string_template')


def _get_template_utils(engine: str) -> Tuple[Callable, Callable]:
    """
    Get compile_template and format_template functions for the specified engine.

    Args:
        engine: Template engine name ('python', 'jinja2', 'handlebars', 'string_template')

    Returns:
        Tuple of (compile_template, format_template) callables
    """
    if engine == 'python':
        from rich_python_utils.string_utils.formatting.python_str_format import (
            compile_template, format_template
        )
    elif engine == 'jinja2':
        from rich_python_utils.string_utils.formatting.jinja2_format import (
            compile_template, format_template
        )
    elif engine == 'handlebars':
        from rich_python_utils.string_utils.formatting.handlebars_format import (
            compile_template, format_template
        )
    elif engine == 'string_template':
        from rich_python_utils.string_utils.formatting.string_template_format import (
            compile_template, format_template
        )
    else:
        raise ValueError(
            f"Unsupported template engine: {engine}. "
            f"Supported engines: {SUPPORTED_TEMPLATE_ENGINES}"
        )
    return compile_template, format_template


def _is_single_var_template(template: str, engine: str = 'python') -> Optional[str]:
    """
    Check if template is exactly a single variable placeholder.

    Single-var templates like '{var}' can be replaced with the actual value
    (preserving type), while multi-var templates '{a} and {b}' must be
    string-formatted.

    Args:
        template: The template string to check
        engine: Template engine name

    Returns:
        The variable name if template is exactly a single variable, None otherwise
    """
    import re
    template = template.strip()

    compile_fn, _ = _get_template_utils(engine)
    try:
        _, variables = compile_fn(template, return_variables=True)
    except Exception:
        return None

    if len(variables) != 1:
        return None

    var_name = next(iter(variables))

    if engine == 'python':
        if template == f'{{{var_name}}}':
            return var_name
    elif engine == 'jinja2':
        if re.match(rf'^{{\{{\s*{re.escape(var_name)}\s*\}}\}}$', template):
            return var_name
    elif engine == 'handlebars':
        if template == f'{{{{{var_name}}}}}':
            return var_name
    elif engine == 'string_template':
        if template == f'${var_name}' or template == f'${{{var_name}}}':
            return var_name

    return None


@attrs(slots=False)
class ActionNode(WorkGraphNode):
    """
    Single action execution as a WorkGraphNode.

    Wraps an Action for execution, leveraging WorkGraphNode's retry mechanism
    for fallback strategy support. Each ActionNode executes one action and
    returns an ActionResult.

    Template Variable System:
    - Autonomously detects template variables in target and args at construction
    - Exposes required_variables for parent nodes (ActionSequenceNode) to aggregate
    - Handles its own variable substitution during execution
    - Type coercion for single-var templates only (multi-var always returns string)

    Type Coercion Rules:
    - Single-var template '{delay}' → value passed through with type coercion
      Example: '{delay}' with delay=2.5 → 2.5 (float preserved)
    - Multi-var template '{base_url}/search' → always string (concatenation)
      Example: '{base_url}/api' with base_url='http://x.com' → 'http://x.com/api'

    Attributes:
        action: The Action to execute.
        action_executor: Callable that executes actions with signature
            (action_type, action_target, action_args, ...) -> result.
        action_metadata: Registry for action type configurations.
        template_engine: Template engine for variable substitution ('python', 'jinja2', etc.)
        enable_result_save: If True, save action results to disk for persistence.
        result_save_dir: Directory for saving results (uses temp dir if None).
        _required_variables: Set of template variables this action needs.
        _type_coercions: Maps variable name to type tuple for single-var coercion.
        _single_var_args: Maps arg name to variable name for single-var templates.
        _compiled_templates: Maps arg name to compiled template.

    Note:
        Fallback index for TargetSpecWithFallback is stored in ExecutionRuntime.node_states,
        not in this node. This keeps ActionNode stateless and reusable across executions.

    Example:
        >>> from agent_foundation.automation.schema.common import Action
        >>> action = Action(id='click_btn', type='click', target='#submit')
        >>> node = ActionNode(
        ...     action=action,
        ...     action_executor=mock_executor,
        ...     action_metadata=ActionMetadataRegistry(),
        ...     enable_result_save=True,
        ...     result_save_dir='/tmp/results',
        ... )
        >>> result = node.run(context)  # Returns ActionResult
        >>> node.required_variables  # Set of template vars this action needs
    """

    action: Action = attrib(kw_only=True)
    action_executor: Union[Callable, MultiActionExecutor] = attrib(kw_only=True)
    action_metadata: ActionMetadataRegistry = attrib(kw_only=True)
    template_engine: str = attrib(default='python', kw_only=True)
    result_save_dir: Optional[str] = attrib(default=None, kw_only=True)

    # Template tracking (populated in __attrs_post_init__)
    _required_variables: Set[str] = attrib(factory=set, init=False)
    _type_coercions: Dict[str, Tuple[type, ...]] = attrib(factory=dict, init=False)
    _single_var_args: Dict[str, str] = attrib(factory=dict, init=False)
    _compiled_templates: Dict[str, Any] = attrib(factory=dict, init=False)

    @property
    def required_variables(self) -> Set[str]:
        """
        Set of template variables this action requires.

        Exposed for parent nodes (ActionSequenceNode) to aggregate
        required variables from all child ActionNodes.
        """
        return self._required_variables

    @property
    def output_variable(self) -> Optional[str]:
        """
        Variable name where this action's result will be stored.

        Returns action.output if set, None otherwise.
        """
        return self.action.output

    def __attrs_post_init__(self):
        import logging
        _logger = logging.getLogger(__name__)
        _logger.debug(
            f"[ActionNode.__attrs_post_init__] action_id={self.action.id}, "
            f"action_type={self.action.type}, "
            f"target={self.action.target}, target_type={type(self.action.target).__name__}"
        )

        # Auto-wrap Mapping action_executor into MultiActionExecutor
        if isinstance(self.action_executor, Mapping) and not isinstance(self.action_executor, MultiActionExecutor):
            self.action_executor = MultiActionExecutor(self.action_executor)

        # Set the value callable to our execution method
        self.value = self._execute_action

        # Set node name from action ID
        self.name = self.action.id

        # Configure retry for fallback strategies
        if isinstance(self.action.target, TargetSpecWithFallback):
            self.max_repeat = len(self.action.target.strategies)
            self.retry_on_exceptions = [Exception]

        # Detect template variables in target and args
        self._detect_template_variables()

        super().__attrs_post_init__()

    def _detect_template_variables(self) -> None:
        """
        Scan target and args for template variables at construction time.

        Populates _required_variables, _type_coercions, _single_var_args,
        and _compiled_templates for efficient runtime substitution.
        """
        compile_fn, _ = _get_template_utils(self.template_engine)

        # Get action metadata for arg_types
        action_meta = self.action_metadata.get_metadata(self.action.type)
        parsed_arg_types = action_meta.parsed_arg_types if action_meta else {}

        # Scan target for template variables
        if isinstance(self.action.target, str):
            self._scan_template_value(
                value=self.action.target,
                arg_name='_target',
                compile_fn=compile_fn,
                parsed_arg_types={},  # target doesn't have arg_type
            )

        # Scan args for template variables
        if self.action.args:
            for arg_name, arg_value in self.action.args.items():
                if isinstance(arg_value, str):
                    self._scan_template_value(
                        value=arg_value,
                        arg_name=arg_name,
                        compile_fn=compile_fn,
                        parsed_arg_types=parsed_arg_types,
                    )

    def _scan_template_value(
        self,
        value: str,
        arg_name: str,
        compile_fn: Callable,
        parsed_arg_types: Dict[str, Tuple[type, ...]],
    ) -> None:
        """
        Scan a string value for template variables and update tracking dicts.

        Type coercion only applies to single-var templates because:
        - Single-var '{delay}' can return the raw value (e.g., float 2.5)
        - Multi-var '{a}/{b}' must be string since we're concatenating

        Args:
            value: The string value to scan
            arg_name: The argument name (for type lookup)
            compile_fn: Template compilation function
            parsed_arg_types: Pre-parsed arg types from action metadata
        """
        try:
            compiled, variables = compile_fn(value, return_variables=True)
        except Exception:
            return  # Not a valid template, skip

        if not variables:
            return  # No template variables found

        # Add variables to required set
        self._required_variables.update(variables)

        # Store compiled template for this arg
        self._compiled_templates[arg_name] = compiled

        # Check if this is a single-var template for type coercion
        # Only single-var templates can preserve type; multi-var always returns string
        single_var = _is_single_var_template(value, self.template_engine)
        if single_var:
            self._single_var_args[arg_name] = single_var

            # If arg has predefined type, track for coercion
            if arg_name in parsed_arg_types:
                self._type_coercions[single_var] = parsed_arg_types[arg_name]

    def substitute_variables(
        self,
        variables: Dict[str, Any],
    ) -> Tuple[Optional[Union[TargetSpec, TargetSpecWithFallback, str]], Optional[Dict[str, Any]]]:
        """
        Substitute template variables in target and args.

        Called during execution with current context variables.
        Single-var templates get type coercion; multi-var templates return strings.

        Args:
            variables: Current variable values from execution context.

        Returns:
            Tuple of (resolved_target, resolved_args) ready for action_executor.

        Example:
            Action: wait(seconds='{delay}') where delay has arg_type='float'
            variables: {'delay': '2.5'}
            Returns: (None, {'seconds': 2.5})  # '2.5' coerced to float
        """
        # No template variables for this action - return originals
        if not self._required_variables:
            return self.action.target, self.action.args

        _, format_fn = _get_template_utils(self.template_engine)

        # Substitute target
        resolved_target = self.action.target
        if isinstance(self.action.target, str) and '_target' in self._compiled_templates:
            resolved_target = self._substitute_value(
                value=self.action.target,
                arg_name='_target',
                variables=variables,
                format_fn=format_fn,
            )

        # Substitute args
        resolved_args = None
        if self.action.args:
            resolved_args = {}
            for arg_name, arg_value in self.action.args.items():
                if isinstance(arg_value, str) and arg_name in self._compiled_templates:
                    resolved_args[arg_name] = self._substitute_value(
                        value=arg_value,
                        arg_name=arg_name,
                        variables=variables,
                        format_fn=format_fn,
                    )
                else:
                    resolved_args[arg_name] = arg_value

        return resolved_target, resolved_args

    def _substitute_value(
        self,
        value: str,
        arg_name: str,
        variables: Dict[str, Any],
        format_fn: Callable,
    ) -> Any:
        """
        Substitute a single template value with type coercion support.

        For single-var templates (e.g., '{delay}'), returns the raw value
        with optional type coercion. For multi-var templates, returns
        the string-formatted result.

        Args:
            value: The template string
            arg_name: The argument name
            variables: Current variable values
            format_fn: Template formatting function

        Returns:
            Substituted value (typed for single-var, string for multi-var)
        """
        # Check if this is a single-var template
        var_name = self._single_var_args.get(arg_name)
        if var_name and var_name in variables:
            raw_value = variables[var_name]

            # Apply type coercion if tracked for this variable
            type_tuple = self._type_coercions.get(var_name)
            if type_tuple:
                return coerce_to_type(raw_value, type_tuple)

            return raw_value  # Return as-is for non-typed single-var

        # Multi-var template: use format_template for string substitution
        compiled = self._compiled_templates.get(arg_name, value)
        return format_fn(compiled, feed=variables, use_builtin_common_helpers=False)

    def has_saved_result(self) -> bool:
        """
        Check if a saved result exists for this action.
        
        Returns:
            True if a saved result file exists, False otherwise.
        """
        if not self.enable_result_save:
            return False
        result_path = self._get_result_path(self.action.id)
        return os.path.exists(result_path)
    
    def load_saved_result(self) -> Optional[ActionResult]:
        """
        Load a previously saved result for this action.
        
        Returns:
            ActionResult if saved result exists, None otherwise.
        """
        if not self.has_saved_result():
            return None
        
        result_path = self._get_result_path(self.action.id)
        try:
            with open(result_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def save_result(self, result: ActionResult) -> bool:
        """
        Save an action result to disk.
        
        Args:
            result: The ActionResult to save.
        
        Returns:
            True if save was successful, False otherwise.
        """
        if not self.enable_result_save:
            return False
        
        result_path = self._get_result_path(self.action.id)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        try:
            with open(result_path, 'wb') as f:
                pickle.dump(result, f)
            return True
        except Exception:
            return False
    
    def _execute_action(self, context: ExecutionRuntime) -> ActionResult:
        """
        Execute the action, cycling through fallback strategies on retry.

        Autonomously handles template variable substitution before execution.
        Detects Agent executors and routes to agent execution path.

        Args:
            context: Execution runtime context with variables and previous results.

        Returns:
            ActionResult with success status and value.
        """
        import logging
        _logger = logging.getLogger(__name__)

        try:
            # Substitute template variables using context.variables
            resolved_target, resolved_args = self.substitute_variables(context.variables)

            _logger.debug(
                f"[ActionNode._execute_action] action_id={self.action.id}, "
                f"action_type={self.action.type}, "
                f"original_target={self.action.target}, "
                f"resolved_target={resolved_target}, "
                f"resolved_target_type={type(resolved_target).__name__}"
            )

            # Resolve the executor for this action type
            executor = self._resolve_executor(self.action.type)

            # Check if executor is an Agent instance
            if self._is_agent_executor(executor):
                _logger.debug(
                    f"[ActionNode._execute_action] Detected agent executor for "
                    f"action_type={self.action.type}, delegating to _execute_agent_action"
                )
                return self._execute_agent_action(
                    context=context,
                    agent=executor,
                    resolved_target=resolved_target,
                    resolved_args=resolved_args,
                )

            # Check for agent-based element finding strategy
            if self._is_agent_target_strategy(resolved_target):
                _logger.debug(
                    f"[ActionNode._execute_action] Detected agent target strategy, "
                    f"resolving element via find_element_agent"
                )
                # Resolve element using find_element_agent
                resolved_element = self._resolve_agent_target(resolved_target, context)
                logger.info(f"[ActionNode._execute_action] Resolved element: {resolved_element}")
                # Execute action with resolved element
                result = self.action_executor(
                    action_type=self.action.type,
                    action_target=resolved_element,
                    action_args=resolved_args,
                    no_action_if_target_not_found=self.action.no_action_if_target_not_found,
                )
            # Standard execution path for non-agent executors
            elif resolved_target is None:
                # No target (e.g., visit_url with URL in args)
                result = self.action_executor(
                    action_type=self.action.type,
                    action_target=None,
                    action_args=resolved_args,
                    no_action_if_target_not_found=self.action.no_action_if_target_not_found,
                )
            elif isinstance(resolved_target, TargetSpecWithFallback):
                # Fallback handling: use current fallback index from context
                result = self._execute_with_current_fallback(context, resolved_args)
            else:
                # Simple target - extract value and strategy, pass through
                target_value = self._get_target_value(resolved_target)
                target_strategy = self._get_target_strategy(resolved_target)
                _logger.debug(
                    f"[ActionNode._execute_action] Calling executor with: "
                    f"action_type={self.action.type}, "
                    f"action_target={target_value}, "
                    f"action_target_strategy={target_strategy}"
                )
                result = self.action_executor(
                    action_type=self.action.type,
                    action_target=target_value,
                    action_args=resolved_args,
                    action_target_strategy=target_strategy,
                    no_action_if_target_not_found=self.action.no_action_if_target_not_found,
                )

            # Wrap in ActionResult
            action_result = ActionResult(
                success=True,
                value=result,
                metadata={
                    "source": getattr(result, "source", None),
                    "is_follow_up": getattr(result, "is_follow_up", False),
                },
            )

            # Store result in context if output variable specified
            if self.output_variable:
                context.set_variable(self.output_variable, action_result.value)

            # Always store last result as '_' for implicit reference
            context.set_variable('_', action_result.value)

            # Save result if persistence is enabled
            if self.enable_result_save:
                self.save_result(action_result)

            return action_result

        except Exception as e:
            # Handle target_not_found branch if configured.
            # IMPORTANT: This check RETURNS EARLY, so the fallback_index increment below
            # never executes when a target_not_found branch exists. This is intentional:
            # - target_not_found branches handle "element not found" by executing fallback
            #   actions, then retrying the SAME target (fallback_index stays at 0)
            # - TargetSpecWithFallback without a branch cycles through different strategies
            # - Combining both would create confusing semantics, so they're mutually exclusive
            if self._is_element_not_found_error(e) and self.action.target_not_found_actions:
                return self._handle_target_not_found_branch(context, e)
            
            # Increment fallback index for next retry attempt (WorkGraphNode retry mechanism).
            # This only runs when NO target_not_found branch exists (due to early return above).
            if isinstance(self.action.target, TargetSpecWithFallback):
                context.set_node_state(self.action.id, 'fallback_index', lambda x: (x or 0) + 1)
            raise  # Re-raise to trigger WorkGraphNode retry
    
    def _execute_with_current_fallback(
        self,
        context: ExecutionRuntime,
        resolved_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute action using the current fallback strategy.

        Args:
            context: Execution runtime context (holds fallback_index state).
            resolved_args: Pre-resolved args (after template substitution).

        Returns:
            Result from action_executor.

        Raises:
            Exception: If the current strategy fails.
        """
        target = self.action.target

        if not isinstance(target, TargetSpecWithFallback):
            raise ValueError("_execute_with_current_fallback called with non-fallback target")

        fallback_index = context.get_node_state(self.action.id, 'fallback_index', 0)
        if fallback_index >= len(target.strategies):
            raise ValueError("All fallback strategies exhausted")

        spec = target.strategies[fallback_index]
        target_value = self._get_target_value(spec)
        target_strategy = self._get_target_strategy(spec)

        # Use resolved_args if provided (already substituted), else use original
        args_to_use = resolved_args if resolved_args is not None else self.action.args

        return self.action_executor(
            action_type=self.action.type,
            action_target=target_value,
            action_args=args_to_use,
            action_target_strategy=target_strategy,
            no_action_if_target_not_found=self.action.no_action_if_target_not_found,
        )
    
    def _get_fallback_result(self, *args, **kwargs) -> ActionResult:
        """
        Return failure result when all fallback strategies are exhausted.
        
        Returns:
            ActionResult with success=False and the last error.
        """
        return ActionResult(
            success=False,
            error=ValueError(f"All fallback strategies failed for action '{self.action.id}'"),
            metadata={"fallback_exhausted": True},
        )
    
    def _get_target_value(
        self, target: Union[TargetSpec, str, int, float, None]
    ) -> Optional[Union[str, int, float]]:
        """
        Extract the target value from a target specification.

        Args:
            target: Target specification (TargetSpec, plain string, int, or float).

        Returns:
            Target value (string, int, or float), or None.
        """
        if target is None:
            return None
        if isinstance(target, (str, int, float)):
            return target
        if isinstance(target, TargetSpec):
            return target.value
        return None
    
    def _get_target_strategy(
        self, target: Union[TargetSpec, str, None]
    ) -> Optional[str]:
        """
        Extract the target strategy from a target specification.

        Args:
            target: Target specification (TargetSpec or plain string).

        Returns:
            Target strategy string, or None if not specified or plain string.
        """
        if target is None:
            return None
        if isinstance(target, str):
            return None  # Plain string has no explicit strategy
        if isinstance(target, TargetSpec):
            return target.strategy
        return None

    def _is_element_not_found_error(self, e: Exception) -> bool:
        """Check if exception is an element-not-found error.
        
        Uses exact type name matching to avoid false positives (e.g., FileNotFoundError).
        Also checks inheritance chain for subclasses.
        
        Args:
            e: The exception to check.
            
        Returns:
            True if the exception is an element-not-found error, False otherwise.
        """
        TARGET_NOT_FOUND_EXCEPTIONS = {
            'ElementNotFoundError',
            'ElementNotFoundException',
            'TargetNotFoundError'
        }
        
        exc_name = type(e).__name__
        if exc_name in TARGET_NOT_FOUND_EXCEPTIONS:
            return True
        
        # Check MRO for subclasses
        return any(
            cls.__name__ in TARGET_NOT_FOUND_EXCEPTIONS
            for cls in type(e).__mro__
        )

    def _handle_target_not_found_branch(
        self,
        context: ExecutionRuntime,
        original_error: Exception
    ) -> ActionResult:
        """Execute target_not_found branch with retry logic.
        
        The retry loop is HERE, not in _execute_action(). This method:
        1. Executes branch actions
        2. If retry_after_handling=False, returns success
        3. If retry_after_handling=True, retries the original action
        4. Repeats until max_retries exceeded
        
        Design Note on TargetSpecWithFallback:
            When the action's target is a TargetSpecWithFallback, all retries within
            this method use fallback_index=0 (the first strategy). This is intentional:
            the target_not_found branch is designed to execute fallback actions and
            retry the SAME target, not cycle through different strategies. If you need
            strategy cycling, use TargetSpecWithFallback without a target_not_found branch.
        
        Args:
            context: Execution runtime context with variables and previous results.
            original_error: The original exception that triggered the branch.
            
        Returns:
            ActionResult with success status and value.
            
        Raises:
            TargetNotFoundError: When max_retries is exceeded.
        """
        import logging
        import time
        _logger = logging.getLogger(__name__)
        
        config = self.action.target_not_found_config or {}
        max_retries = config.get('max_retries', 3)
        retry_delay = config.get('retry_delay', 1.0)
        retry_after_handling = config.get('retry_after_handling', False)
        
        attempt_count = 0
        
        while True:
            attempt_count += 1
            
            _logger.debug(
                f"[ActionNode._handle_target_not_found_branch] "
                f"action_id={self.action.id}, attempt={attempt_count}, "
                f"retry_after_handling={retry_after_handling}"
            )
            
            # Execute branch actions
            branch_results = self._execute_branch_actions(
                context, self.action.target_not_found_actions
            )
            
            # If not retrying, return success
            if not retry_after_handling:
                return ActionResult(
                    success=True,
                    value=None,
                    metadata={'branch_executed': True, 'branch_results': branch_results}
                )
            
            # Check retry limit
            if attempt_count > max_retries:
                raise TargetNotFoundError(
                    action_type=self.action.type,
                    target=self.action.target,
                    attempt_count=attempt_count,
                    max_retries=max_retries
                ) from original_error
            
            # Wait before retry
            if retry_delay > 0:
                time.sleep(retry_delay)
            
            # Retry original action (same logic as main try block)
            # NOTE on no_action_if_target_not_found=False below:
            # We intentionally force this to False during retry because:
            # 1. If the original action had no_action_if_target_not_found=True, the executor
            #    would have returned success (no exception), so we wouldn't be here
            # 2. The fact that we're in this branch means an exception WAS raised
            # 3. During retry, we want to actually attempt to find the target, not skip it
            # 4. If we respected the original setting, retry would immediately skip,
            #    defeating the purpose of the target_not_found branch
            try:
                resolved_target, resolved_args = self.substitute_variables(context.variables)
                
                # Resolve the executor for this action type
                executor = self._resolve_executor(self.action.type)
                
                # Check if executor is an Agent instance
                if self._is_agent_executor(executor):
                    return self._execute_agent_action(
                        context=context,
                        agent=executor,
                        resolved_target=resolved_target,
                        resolved_args=resolved_args,
                    )
                
                # Check for agent-based element finding strategy
                if self._is_agent_target_strategy(resolved_target):
                    resolved_element = self._resolve_agent_target(resolved_target, context)
                    result = self.action_executor(
                        action_type=self.action.type,
                        action_target=resolved_element,
                        action_args=resolved_args,
                        no_action_if_target_not_found=False,  # See NOTE above
                    )
                elif resolved_target is None:
                    result = self.action_executor(
                        action_type=self.action.type,
                        action_target=None,
                        action_args=resolved_args,
                        no_action_if_target_not_found=False,  # See NOTE above
                    )
                elif isinstance(resolved_target, TargetSpecWithFallback):
                    result = self._execute_with_current_fallback(context, resolved_args)
                else:
                    target_value = self._get_target_value(resolved_target)
                    target_strategy = self._get_target_strategy(resolved_target)
                    result = self.action_executor(
                        action_type=self.action.type,
                        action_target=target_value,
                        action_args=resolved_args,
                        action_target_strategy=target_strategy,
                        no_action_if_target_not_found=False,  # See NOTE above
                    )
                
                # Success! Return
                action_result = ActionResult(
                    success=True,
                    value=result,
                    metadata={
                        'branch_executed': True,
                        'retry_succeeded': True,
                        'retry_attempt': attempt_count
                    }
                )
                
                # Store in context
                if self.output_variable:
                    context.set_variable(self.output_variable, action_result.value)
                context.set_variable('_', action_result.value)
                
                return action_result
                
            except Exception as e:
                if not self._is_element_not_found_error(e):
                    raise  # Different error, propagate
                # Still not found, continue loop
                continue

    def _execute_branch_actions(
        self,
        context: ExecutionRuntime,
        branch_actions: List[Action]
    ) -> List[ActionResult]:
        """Execute a list of branch actions.
        
        Args:
            context: Execution runtime context with variables and previous results.
            branch_actions: List of Action objects to execute.
            
        Returns:
            List of ActionResult objects from each branch action.
        """
        results = []
        for branch_action in branch_actions:
            # Create ActionNode for branch action and execute
            branch_node = ActionNode(
                action=branch_action,
                action_executor=self.action_executor,
                action_metadata=self.action_metadata,
                template_engine=self.template_engine,
            )
            result = branch_node.run(context)  # Use run(), not _execute_action()
            results.append(result)
        return results

    def _resolve_executor(self, action_type: str) -> Any:
        """
        Resolve the executor for an action type.

        If action_executor is a MultiActionExecutor, resolves by action_type.
        Otherwise returns the action_executor directly if callable.

        Args:
            action_type: The action type to resolve (e.g., "click", "search_agent")

        Returns:
            The resolved executor (callable, Agent, or other executor type)

        Raises:
            ValueError: If no executor is available for the action type
        """
        if isinstance(self.action_executor, MultiActionExecutor):
            return self.action_executor.resolve(action_type)
        elif callable(self.action_executor):
            return self.action_executor
        else:
            raise ValueError(f"No executor available for action type: {action_type}")

    def _is_agent_executor(self, executor: Any) -> bool:
        """
        Check if the executor is an Agent instance.

        Uses lazy import to avoid circular dependency issues.
        Note: This is a strict check for Agent instances only, used to determine
        if agent-style execution should be used in _execute_action.

        Args:
            executor: The executor to check

        Returns:
            True if executor is an Agent instance, False otherwise
        """
        try:
            from science_modeling_tools.agents.agent import Agent
            return isinstance(executor, Agent)
        except ImportError:
            return False

    def _is_valid_find_element_agent(self, executor: Any) -> bool:
        """
        Check if the executor is valid for use as find_element_agent.

        Accepts both Agent instances and plain callables (functions).
        This is used in _resolve_agent_target to validate find_element_agent.

        Args:
            executor: The executor to check

        Returns:
            True if executor is an Agent instance or callable, False otherwise
        """
        # Accept any callable as a valid find_element_agent
        if callable(executor):
            return True
        try:
            from science_modeling_tools.agents.agent import Agent
            return isinstance(executor, Agent)
        except ImportError:
            return False

    def _is_agent_target_strategy(self, target: Any) -> bool:
        """
        Check if target uses the agent-based element finding strategy.

        Args:
            target: The target specification to check

        Returns:
            True if target uses TargetStrategy.AGENT, False otherwise
        """
        if not isinstance(target, TargetSpec):
            return False
        strategy = target.strategy
        if strategy is None:
            return False
        # Check for both enum value and string value
        if isinstance(strategy, TargetStrategy):
            return strategy == TargetStrategy.AGENT
        return str(strategy) == TargetStrategy.AGENT.value

    def _resolve_agent_target(
        self,
        target: TargetSpec,
        context: ExecutionRuntime,
    ) -> Any:
        """
        Resolve target using the find_element_agent.

        When a TargetSpec has strategy='agent', this method uses the registered
        find_element_agent to resolve the natural language element description
        to an actual element reference.

        Args:
            target: The TargetSpec with strategy='agent' and value containing
                    the natural language element description.
            context: Execution runtime context.

        Returns:
            The resolved element reference (selector, xpath, or element object).

        Raises:
            ValueError: If find_element_agent is not registered or not an Agent.
            AgentExecutionError: If the agent fails to resolve the element.
        """
        import logging
        _logger = logging.getLogger(__name__)

        _logger.debug(
            f"[ActionNode._resolve_agent_target] Resolving agent target: "
            f"value={target.value}, options={target.options}"
        )

        # Resolve the find_element_agent from executor
        try:
            find_agent = self._resolve_executor("find_element_agent")
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"find_element_agent not registered in executor. "
                f"Cannot resolve agent-based target: {target.value}"
            ) from e

        if not self._is_valid_find_element_agent(find_agent):
            raise ValueError(
                f"find_element_agent must be an Agent instance or callable, "
                f"got {type(find_agent).__name__}"
            )

        # Prepare input for agent
        task_input = {
            'user_input': target.value,
            'context': context.variables,
        }

        # Pass options if provided
        if target.options:
            task_input['options'] = target.options

        try:
            # Execute agent to find element
            result = find_agent(**task_input)

            _logger.debug(
                f"[ActionNode._resolve_agent_target] Agent resolved element: "
                f"result_type={type(result).__name__}"
            )

            # Return the agent's output as the resolved element reference
            # The agent should return a selector, xpath, or element object
            if hasattr(result, 'output'):
                return result.output
            return result

        except Exception as e:
            # Note: on_error handling for find_element_agent is typically "stop"
            # since we can't continue without a resolved element. The caller
            # (standard action execution) will handle the error appropriately.
            raise AgentExecutionError(
                agent_id="find_element_agent",
                description=target.value or "",
                original_error=e,
            )

    def _execute_agent_action(
        self,
        context: ExecutionRuntime,
        agent: Any,
        resolved_target: Any,
        resolved_args: Optional[Dict[str, Any]],
    ) -> ActionResult:
        """
        Execute an Agent action by delegating to the resolved agent.

        The target field value (after template substitution) is passed as the
        task description (user_input) to the agent.

        Args:
            context: Execution runtime context with variables and previous results.
            agent: The Agent instance to execute.
            resolved_target: The resolved target (task description for agent).
            resolved_args: Resolved action arguments after template substitution.

        Returns:
            ActionResult with success status and agent's output.

        Raises:
            AgentExecutionError: If the agent execution fails.
        """
        import logging
        _logger = logging.getLogger(__name__)

        # Get task description from resolved target
        task_description = str(resolved_target) if resolved_target is not None else ""

        # Get previous action result if available (stored in '_' variable)
        previous_result = context.variables.get('_')

        # Prepare task input for agent
        task_input = {
            'user_input': task_description,
            'context': context.variables,
            'action_results': previous_result,
        }

        # Add any additional args from the action
        if resolved_args:
            task_input.update(resolved_args)

        _logger.debug(
            f"[ActionNode._execute_agent_action] action_id={self.action.id}, "
            f"agent_type={self.action.type}, "
            f"task_description={task_description[:100]}..."
        )

        try:
            # Execute agent
            result = agent(**task_input)

            # Wrap in ActionResult
            action_result = ActionResult(
                success=True,
                value=result,
                metadata={
                    "agent_action": True,
                    "agent_type": self.action.type,
                },
            )

            # Store result in context if output variable specified
            if self.output_variable:
                context.set_variable(self.output_variable, action_result.value)

            # Always store last result as '_' for implicit reference
            context.set_variable('_', action_result.value)

            # Save result if persistence is enabled
            if self.enable_result_save:
                self.save_result(action_result)

            return action_result

        except Exception as e:
            _logger.error(
                f"[ActionNode._execute_agent_action] Agent execution failed: "
                f"action_id={self.action.id}, agent_type={self.action.type}, error={e}"
            )

            # Check on_error policy
            if self.action.on_error == "continue":
                # Return failure result without raising
                action_result = ActionResult(
                    success=False,
                    error=e,
                    metadata={
                        "agent_action": True,
                        "agent_type": self.action.type,
                        "on_error": "continue",
                    },
                )
                # Still store result in context even on failure
                if self.output_variable:
                    context.set_variable(self.output_variable, None)
                context.set_variable('_', None)
                return action_result

            # Default: on_error="stop" - re-raise with context
            raise AgentExecutionError(
                agent_id=self.action.type,
                description=task_description,
                original_error=e,
            )

    # Serializable interface methods
    def to_serializable_obj(
        self,
        mode: str = 'auto',
        _output_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert ActionNode to serializable Python object.
        
        Overrides Serializable.to_serializable_obj() to provide custom
        serialization that preserves the action definition and retry configuration.
        
        Args:
            mode: Serialization mode ('auto', 'dict', 'pickle')
            _output_format: Target output format for conflict detection
        
        Returns:
            Dict containing action definition and config
        """
        return {
            FIELD_TYPE: type(self).__name__,
            FIELD_MODULE: type(self).__module__,
            FIELD_SERIALIZATION: SERIALIZATION_DICT,
            "action": self.action.model_dump(exclude_none=True),
            "config": {
                "enable_result_save": self.enable_result_save,
                "result_save_dir": self.result_save_dir,
                "max_repeat": self.max_repeat,
            }
        }
    
    @classmethod
    def from_serializable_obj(
        cls,
        obj: Dict[str, Any],
        action_executor: Callable = None,
        action_metadata: Optional[ActionMetadataRegistry] = None,
        **context
    ) -> 'ActionNode':
        """Reconstruct ActionNode from serializable dict.
        
        Overrides Serializable.from_serializable_obj() to provide custom
        deserialization that reconstructs the action with context injection
        for action_executor and action_metadata.
        
        Args:
            obj: The serializable object (dict)
            action_executor: Callable for executing actions (required)
            action_metadata: Action type registry (optional)
            **context: Additional context parameters
        
        Returns:
            Reconstructed ActionNode instance
        
        Raises:
            ValueError: If action_executor is not provided
        """
        if action_executor is None:
            action_executor = context.get('action_executor')
        if action_executor is None:
            raise ValueError("Required context parameter 'action_executor' not provided")
        
        if action_metadata is None:
            action_metadata = context.get('action_metadata', ActionMetadataRegistry())
        
        # Reconstruct action from dict
        action = Action(**obj['action'])
        config = obj.get('config', {})
        
        return cls(
            action=action,
            action_executor=action_executor,
            action_metadata=action_metadata,
            enable_result_save=config.get('enable_result_save', False),
            result_save_dir=config.get('result_save_dir'),
        )

    def _get_result_path(self, name: str, *args, **kwargs) -> str:
        """
        Get the path for saving action results.
        
        Uses result_save_dir if configured, otherwise uses temp directory.
        
        Args:
            name: The name/id of the result to save.
        
        Returns:
            Path string for the result file.
        """
        base_dir = self.result_save_dir or tempfile.gettempdir()
        return os.path.join(base_dir, f"action_result_{name}.pkl")


class ActionExecutionError(Exception):
    """
    Raised when action execution fails.

    Attributes:
        action_id: ID of the failed action.
        original_error: The underlying exception that caused the failure.
    """

    def __init__(self, action_id: str, original_error: Exception):
        self.action_id = action_id
        self.original_error = original_error
        super().__init__(f"Action '{action_id}' failed: {original_error}")


class AgentExecutionError(Exception):
    """
    Raised when agent action execution fails.

    Provides detailed context about which agent failed and the task description.

    Attributes:
        agent_id: Identifier of the agent that failed (typically the action_type).
        description: The task description that was passed to the agent.
        original_error: The underlying exception that caused the failure.
    """

    def __init__(self, agent_id: str, description: str, original_error: Exception):
        self.agent_id = agent_id
        self.description = description
        self.original_error = original_error
        # Truncate description for message
        desc_preview = description[:100] + "..." if len(description) > 100 else description
        super().__init__(
            f"Agent '{agent_id}' failed on task: {desc_preview} "
            f"Error: {original_error}"
        )

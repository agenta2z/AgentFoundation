"""
Common Models and Protocols for Action Sequence Schema

Defines data models, runtime context, and protocol contracts for UI automation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable, TYPE_CHECKING
from pydantic import BaseModel, Field, validator

if TYPE_CHECKING:
    from .action_metadata import ActionMetadataRegistry


# region Target Specifications

class TargetStrategy(str, Enum):
    """
    Target resolution strategies for UI automation.

    This enum defines the SCHEMA for target resolution strategies. The actual
    resolution logic is implemented by the executor (e.g., WebDriver in WebAgent).
    ActionNode extracts the strategy from TargetSpec and passes it through to the
    executor via the `action_target_strategy` parameter.

    Strategies define how target elements are identified in the UI:

    - FRAMEWORK_ID: The automation framework assigns unique IDs to all UI elements.
      This provides stable, predictable element identification that works consistently
      across different UI frameworks. Value: '__id__'

    - ID: Uses the native identifier attribute of UI elements. In HTML, this is
      the 'id' attribute. In other UI frameworks, this corresponds to platform-specific
      identifier attributes. May not exist or be unique in all cases. Value: 'id'

    - XPATH: XPath expression for element location (web/XML-based UIs). Value: 'xpath'

    - CSS: CSS selector for element location (web UIs). Value: 'css'

    - TEXT: Text content matching for element identification. Value: 'text'

    - SOURCE: Source markup matching (e.g., HTML snippet, XAML fragment).
      The executor interprets this based on its UI framework. Value: 'source'

    - DESCRIPTION: AI-based natural language resolution. The executor uses AI/ML
      to find elements matching the natural language description. Value: 'description'

    - LITERAL: Literal value that is not an element identifier (e.g., URLs).
      The value is used as-is without element resolution. Value: 'literal'

    - AGENT: Agent-based element finding. Uses a registered 'find_element_agent'
      to resolve natural language element descriptions dynamically. The agent
      receives the target value and optional hints, returning an element reference.
      Value: 'agent'
    """
    FRAMEWORK_ID = '__id__'    # Framework-assigned unique identifier
    ID = 'id'                  # Native UI element identifier attribute
    XPATH = 'xpath'            # XPath expression (web/XML)
    CSS = 'css'                # CSS selector (web)
    TEXT = 'text'              # Text content matching
    SOURCE = 'source'          # Source markup matching (HTML, XAML, etc.)
    DESCRIPTION = 'description'  # AI-based natural language resolution
    LITERAL = 'literal'        # Literal value (e.g., URLs)
    AGENT = 'agent'            # Agent-based element finding (uses find_element_agent)


class TargetSpec(BaseModel):
    """
    Specification for locating a single UI element using one strategy.

    Can use either:
    - strategy + value: For explicit strategy-based resolution (e.g., id, xpath, css)
    - description: For AI-based natural language resolution (future feature)
    - Both: Strategy first, description as fallback
    - strategy='agent' + value: For agent-based element finding

    JSON Examples:
        # Simple with explicit strategy
        {"strategy": "id", "value": "submit-btn"}

        # Using XPath
        {"strategy": "xpath", "value": "//button[@type='submit']"}

        # Using CSS selector
        {"strategy": "css", "value": "button.submit"}

        # With description fallback
        {"strategy": "id", "value": "submit-btn", "description": "the submit button"}

        # Description only (for AI-based resolution)
        {"description": "the blue submit button at the bottom"}

        # Agent-based element finding
        {"strategy": "agent", "value": "the search input box"}

        # Agent-based with hints
        {"strategy": "agent", "value": "the login button", "options": ["static"]}

    Attributes:
        strategy: Resolution strategy (id, xpath, css, __id__, literal, agent, etc.).
                  If not specified, uses the action's default strategy.
        value: Strategy-specific value (element ID, XPath expression, CSS selector, etc.)
        description: Natural language description for AI-based fallback resolution.
        options: Optional hints for agent-based resolution (e.g., ["static"] for cacheable elements).
    """
    strategy: Optional[Union[TargetStrategy, str]] = None  # Optional - uses action's default if not specified
    value: Optional[str] = None  # Value for strategy-based resolution (e.g., element ID, XPath)
    description: Optional[str] = None  # Optional natural language description as fallback
    options: Optional[List[str]] = None  # Optional hints for agent-based resolution (e.g., ["static"] for cacheable)

    @validator('description', always=True)
    def must_have_value_or_description(cls, v, values):
        """Value or description must be provided."""
        value = values.get('value')
        if not value and not v:
            raise ValueError("Must specify at least 'value' or 'description'")
        return v

    class Config:
        use_enum_values = True


class TargetSpecWithFallback(BaseModel):
    """
    Target specification with multiple fallback strategies for resilient element location.

    Web pages are dynamic - an element might be findable by ID on one page load,
    but only by XPath on another. This class provides resilience by allowing
    multiple strategies to be tried in order until one succeeds.

    Resolution Behavior:
        1. Tries each strategy in the order specified in the 'strategies' list
        2. Returns the element as soon as one strategy succeeds
        3. If a strategy fails, catches the error and tries the next one
        4. If ALL strategies fail, raises ElementNotFoundError with the last error

    JSON Example:
        {
            "target": {
                "strategies": [
                    {"strategy": "id", "value": "submit-btn"},
                    {"strategy": "css", "value": "button.submit"},
                    {"strategy": "xpath", "value": "//button[@type='submit']"}
                ]
            }
        }

    Use Cases:
        - Elements with unstable IDs that change between page loads
        - Cross-browser compatibility where different selectors work better
        - A/B testing scenarios where element structure varies
        - Graceful degradation from fast (id) to slower but more robust (xpath) strategies

    Attributes:
        strategies: Ordered list of TargetSpec objects to try. First match wins.
    """
    strategies: List[TargetSpec]

    @validator('strategies')
    def must_have_at_least_one(cls, v):
        if not v:
            raise ValueError("Must specify at least one strategy")
        return v

# endregion


# region Action Models

class Action(BaseModel):
    """
    Base action specification for a single automation step.

    Design Principle:
    - Non-composite action: target = single element ID (string)
    - Composite action: target = space-separated element IDs (string)
    - WebDriver handles the composite vs non-composite distinction internally

    Attributes:
        id: Unique identifier for this action within the sequence.
        type: Action type (e.g., "click", "input_text", "visit_url", "scroll").
        target: Target element specification. Accepts three formats:
            - str: Simple string value (uses action's default strategy)
              For composite actions, use space-separated IDs: "123 456"
            - TargetSpec: Single strategy specification
              Example: {"strategy": "id", "value": "submit-btn"}
            - TargetSpecWithFallback: Multiple strategies tried in order
              Example: {"strategies": [{"strategy": "id", "value": "btn"}, {"strategy": "css", "value": ".btn"}]}
        args: Action-specific arguments (e.g., {"text": "hello"} for input_text).
        condition: Optional condition for execution (future feature).
        on_error: Error handling policy - "stop" or "continue" (future feature).
        output: Variable name to store action result (future feature).
        timeout: Action timeout in seconds (future feature).

    JSON Examples:
        # Simple string target (uses action's default strategy, typically __id__)
        {"id": "click_btn", "type": "click", "target": "123"}

        # Composite action with space-separated targets
        {"id": "input_submit", "type": "input_and_submit", "target": "123 456", "args": {"text": "hello"}}

        # Explicit strategy
        {"id": "click_btn", "type": "click", "target": {"strategy": "id", "value": "submit-btn"}}

        # Fallback strategies for resilience
        {
            "id": "click_btn",
            "type": "click",
            "target": {
                "strategies": [
                    {"strategy": "id", "value": "submit-btn"},
                    {"strategy": "css", "value": "button.submit"}
                ]
            }
        }
    """
    id: str  # Unique action identifier
    type: str  # Action type (e.g., "click", "input_text", "visit_url")
    target: Optional[Union[TargetSpec, TargetSpecWithFallback, str, int, float]] = None  # Target element(s) - space-separated for composite actions; int/float for wait duration
    args: Optional[Dict[str, Any]] = None  # Action-specific arguments
    condition: Optional[str] = None  # Optional condition for execution (for future increments)
    on_error: str = "stop"  # Error handling policy (for future increments)
    output: Optional[str] = None  # Output variable name (for future increments)
    timeout: Optional[float] = None  # Action timeout in seconds (for future increments)
    wait: Optional[Union[float, bool]] = None  # Wait after action: float=seconds, True=human confirmation
    no_action_if_target_not_found: bool = False  # Skip action if target element not found
    
    # Target not found branch - allows defining fallback actions when target cannot be resolved
    target_not_found_actions: Optional[List['Action']] = None  # List of actions to execute when target not found
    target_not_found_config: Optional[Dict[str, Any]] = None  # Config: retry_after_handling, max_retries, retry_delay

    class Config:
        extra = "forbid"  # Reject unknown fields


# Enable forward reference for recursive Action type in target_not_found_actions
Action.update_forward_refs()


class ActionSequence(BaseModel):
    """Complete action sequence specification."""
    version: str = "1.0"  # Schema version
    id: str  # Sequence identifier
    description: Optional[str] = None  # Human-readable description
    variables: Optional[Dict[str, Any]] = None  # Variables (for future increments)
    actions: List[Action]  # List of actions to execute
    outputs: Optional[Dict[str, str]] = None  # Output definitions (for future increments)

    @validator('actions')
    def must_have_actions(cls, v):
        if not v:
            raise ValueError("Action sequence must contain at least one action")
        return v

    @validator('actions')
    def action_ids_must_be_unique(cls, v):
        """Ensure all action IDs are unique."""
        ids = [action.id for action in v]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate action IDs found: {set(duplicates)}")
        return v

    class Config:
        extra = "forbid"  # Reject unknown fields

# endregion


# region Execution Context and Results

@dataclass
class ActionResult:
    """Result of action execution."""
    success: bool
    value: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionRuntime:
    """
    Runtime execution context for action sequences.

    Stores variables, action results, and per-action execution state.

    The node_states dict allows actions to store runtime state (like fallback_index)
    in the context rather than in the node itself. This keeps nodes stateless and
    reusable across multiple workflow executions.

    Attributes:
        variables: Template variables for substitution
        results: Final ActionResult for each action (keyed by action.id)
        current_action_id: ID of currently executing action
        node_states: Runtime state for each action during execution (keyed by action.id)
    """
    variables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, ActionResult] = field(default_factory=dict)
    current_action_id: Optional[str] = None
    node_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def set_result(self, action_id: str, result: ActionResult):
        """
        Store action result.

        Args:
            action_id: Action identifier
            result: Action execution result
        """
        self.results[action_id] = result

    def get_result(self, action_id: str) -> Optional[ActionResult]:
        """
        Get action result by ID.

        Args:
            action_id: Action identifier

        Returns:
            ActionResult or None if not found
        """
        return self.results.get(action_id)

    def set_variable(self, name: str, value: Any):
        """
        Set variable value.

        Args:
            name: Variable name
            value: Variable value
        """
        self.variables[name] = value

    def get_variable(self, name: str) -> Any:
        """
        Get variable value.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        if name not in self.variables:
            raise KeyError(f"Variable '{name}' not found in context")
        return self.variables[name]

    def get_node_state(self, action_id: str, key: str, default: Any = None) -> Any:
        """
        Get runtime state for an action.

        Args:
            action_id: Action identifier
            key: State key (e.g., 'fallback_index')
            default: Default value if not found

        Returns:
            State value or default
        """
        return self.node_states.get(action_id, {}).get(key, default)

    def set_node_state(self, action_id: str, key: str, value: Any):
        """
        Set runtime state for an action.

        Args:
            action_id: Action identifier
            key: State key (e.g., 'fallback_index')
            value: State value, or a callable that receives current value and returns new value.
                   If callable, it will be called with the current value (or None if not set).
                   Example: lambda x: (x or 0) + 1 to increment
        """
        if action_id not in self.node_states:
            self.node_states[action_id] = {}
        if callable(value):
            current = self.node_states[action_id].get(key)
            self.node_states[action_id][key] = value(current)
        else:
            self.node_states[action_id][key] = value

    def merge(self, other: 'ExecutionRuntime'):
        """Merge another runtime's results and variables into this one.

        Used by loop constructs to merge advance sequence results back
        into the main execution context.

        Args:
            other: Another ExecutionRuntime to merge from
        """
        self.variables.update(other.variables)
        self.results.update(other.results)
        # Note: node_states not merged - they are per-execution transient state


@dataclass
class ExecutionResult:
    """
    Final result of action sequence execution.

    Contains success status, final context state, and any outputs.
    """
    success: bool
    context: ExecutionRuntime
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    failed_action_id: Optional[str] = None


class LoopExecutionError(Exception):
    """Error during loop execution with iteration context.
    
    Provides detailed context about which loop and iteration failed,
    wrapping the original error for debugging.
    
    Attributes:
        loop_id: Identifier of the loop node that failed
        iteration: The iteration number (0-indexed) when failure occurred
        original_error: The underlying exception that caused the failure
    """
    def __init__(self, loop_id: str, iteration: int, original_error: Exception):
        self.loop_id = loop_id
        self.iteration = iteration
        self.original_error = original_error
        super().__init__(f"Loop '{loop_id}' failed at iteration {iteration}: {original_error}")


class TargetNotFoundError(Exception):
    """Raised when target element not found after all retries.
    
    This exception is raised when an action's target cannot be resolved
    after exhausting all retry attempts. It provides detailed context
    about the failed action and retry configuration.
    
    Attributes:
        action_type: The type of action that failed (e.g., "click", "input_text")
        target: The target specification that could not be resolved
        attempt_count: Total number of attempts made (1 initial + retries)
        max_retries: Maximum retries that were configured
    
    Example:
        >>> raise TargetNotFoundError(
        ...     action_type="click",
        ...     target=TargetSpec(strategy="id", value="submit-btn"),
        ...     attempt_count=4,
        ...     max_retries=3
        ... )
        TargetNotFoundError: Target not found after 4 attempts (1 initial + 3 retries allowed). Action: click, Target: id:submit-btn
    """
    def __init__(
        self,
        action_type: str,
        target: Union[TargetSpec, TargetSpecWithFallback, str],
        attempt_count: int,
        max_retries: int
    ):
        self.action_type = action_type
        self.target = target
        self.attempt_count = attempt_count
        self.max_retries = max_retries
        attempt_word = "attempt" if attempt_count == 1 else "attempts"
        
        # Format target for display
        if isinstance(target, TargetSpec):
            target_str = f"{target.strategy}:{target.value}"
        elif isinstance(target, TargetSpecWithFallback):
            target_str = f"fallback[{len(target.strategies)} strategies]"
        else:
            target_str = str(target)
        
        super().__init__(
            f"Target not found after {attempt_count} {attempt_word} "
            f"(1 initial + {max_retries} retries allowed). "
            f"Action: {action_type}, Target: {target_str}"
        )


class BranchAlreadyExistsError(Exception):
    """Raised when attempting to define a duplicate branch on an action.
    
    This exception is raised when `target_not_found()` is called twice
    on the same action, which would overwrite the existing branch definition.
    
    Attributes:
        condition: The branch condition type (e.g., "target_not_found")
        action_type: The type of action that already has this branch
    
    Example:
        >>> raise BranchAlreadyExistsError(
        ...     condition="target_not_found",
        ...     action_type="click"
        ... )
        BranchAlreadyExistsError: Branch 'target_not_found' already exists on action 'click'.
    """
    def __init__(self, condition: str, action_type: str):
        self.condition = condition
        self.action_type = action_type
        super().__init__(
            f"Branch '{condition}' already exists on action '{action_type}'."
        )

# endregion


# region Protocols

@runtime_checkable
class ActionExecutor(Protocol):
    """
    Protocol for executing actions on platform elements.

    Implementations handle platform-specific action execution
    (e.g., Selenium click, Appium tap, etc.).
    """

    def execute(
        self,
        action: Action,
        resolved_target: Any,
        action_metadata: "ActionMetadataRegistry",
        resolved_targets: Any = None
    ) -> ActionResult:
        """
        Execute action on resolved target.

        Args:
            action: Action specification to execute
            resolved_target: Platform-specific element (or None for targetless actions)
            action_metadata: Registry for looking up action metadata
            resolved_targets: List of resolved elements for composite actions (optional)

        Returns:
            ActionResult with execution outcome
        """
        ...

    def supports_action(self, action_type: str) -> bool:
        """
        Check if executor supports a specific action type.

        Args:
            action_type: Action type name (e.g., "click", "input_text")

        Returns:
            True if action type is supported, False otherwise
        """
        ...

# endregion


# region Loader Functions

def load_sequence(source: Union[str, "Path"]) -> ActionSequence:
    """
    Load action sequence from file path or JSON string.

    Auto-detects whether source is a file path or JSON string.

    Args:
        source: File path or JSON string

    Returns:
        Validated ActionSequence object

    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If JSON is invalid or doesn't match schema
    """
    import json
    from pathlib import Path
    
    # Try to interpret as file path first
    path = Path(source) if isinstance(source, str) else source

    if path.exists():
        # Load from file
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Assume it's a JSON string
        try:
            data = json.loads(str(source))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

    # Validate and parse with Pydantic
    try:
        return ActionSequence(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse action sequence: {e}") from e


def load_sequence_from_string(json_string: str) -> ActionSequence:
    """
    Load action sequence from JSON string.

    Args:
        json_string: JSON string containing action sequence

    Returns:
        Validated ActionSequence object

    Raises:
        ValueError: If JSON is invalid or doesn't match schema
    """
    import json
    
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    try:
        return ActionSequence(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse action sequence: {e}") from e

# endregion

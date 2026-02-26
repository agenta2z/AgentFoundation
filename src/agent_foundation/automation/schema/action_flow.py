"""
Action Flow Executor

Orchestrates action sequence execution with flow control.
Generic executor that works with any action_executor callable.

The name "ActionFlow" reflects the future support for branching,
conditionals, and other flow control mechanisms beyond simple sequential execution.

Design Principle:
- Non-composite action: single target (string)
- Composite action: target is space-separated element IDs (string)
- The action_executor handles the distinction internally
- ActionFlow just passes target through, handling only fallback logic

ActionFlow inherits from Workflow (for O(1) stack depth execution) and
Serializable (for JSON/YAML/pickle format support).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Mapping, Union, List

from attr import attrs, attrib

from rich_python_utils.common_objects.workflow.workflow import Workflow
from rich_python_utils.common_objects.serializable import (
    Serializable,
    FIELD_TYPE,
    FIELD_MODULE,
    FIELD_SERIALIZATION,
    SERIALIZATION_DICT,
)

from .common import ActionSequence, Action, TargetSpec, TargetSpecWithFallback, ExecutionRuntime, ExecutionResult, ActionResult
from .action_metadata import ActionMetadataRegistry
from .action_node import ActionNode, ActionExecutionError
from .action_executor import MultiActionExecutor


@attrs(slots=False)
class ActionFlow(Workflow, Serializable):
    """
    Orchestrates action sequence execution with flow control.

    For MVP: Sequential execution only (no variables, conditions, or branching).
    The name "ActionFlow" reflects future support for branching and conditionals.

    This class is generic and works with any action_executor that has the same
    interface as WebDriver.__call__:
        action_executor(action_type, action_target, action_args, ...) -> result

    Design Principle:
    - Non-composite action: action.target is a single element ID (string)
    - Composite action: action.target is space-separated element IDs (string)
    - The action_executor handles composite vs non-composite distinction
    - This executor only handles: sequential execution, fallback for TargetSpecWithFallback

    Template Variable System:
    - Passes template_engine to ActionNodes for autonomous template handling
    - Each ActionNode handles its own variable detection and substitution

    Inherits from Workflow for O(1) stack depth execution and Serializable for
    JSON, YAML, and pickle format support.

    Attributes:
        action_executor: Callable with signature (action_type, action_target, action_args) -> result.
            In web scenario, this is the WebDriver instance itself.
        action_metadata: Action metadata registry for looking up action configurations.
        template_engine: Template engine for variable substitution ('python', 'jinja2', etc.)
        sequence: Optional ActionSequence to execute (can also be passed to execute()).
        result_save_dir: Directory for saving results (uses temp dir if None).
        context: Runtime execution context (set during execute()).
        action_nodes: List of ActionNodes to execute (set during execute()).

    Inherited from Workflow:
        enable_result_save: If True, save action results to disk for persistence.
        resume_with_saved_results: If True, skip actions with existing saved results.
    """

    # Configuration attributes (kw_only=True to work with inherited attrs from Workflow)
    action_executor: Union[Callable, MultiActionExecutor] = attrib(kw_only=True)
    action_metadata: ActionMetadataRegistry = attrib(kw_only=True)
    template_engine: str = attrib(default='python', kw_only=True)
    sequence: Optional[ActionSequence] = attrib(default=None, kw_only=True)
    result_save_dir: Optional[str] = attrib(default=None, kw_only=True)

    # Runtime attributes (set in execute(), not in __init__)
    context: Optional[ExecutionRuntime] = attrib(default=None, init=False)
    action_nodes: List[ActionNode] = attrib(factory=list, init=False)

    def __attrs_post_init__(self):
        # Auto-wrap Mapping action_executor into MultiActionExecutor
        if isinstance(self.action_executor, Mapping) and not isinstance(self.action_executor, MultiActionExecutor):
            self.action_executor = MultiActionExecutor(self.action_executor)

    def _get_result_path(self, result_id, *args, **kwargs) -> str:
        """Return path for saving step results.

        Required by Workflow base class for result persistence.

        Args:
            result_id: Identifier for the result (typically action ID).

        Returns:
            File path for saving the result.
        """
        base_dir = self.result_save_dir or tempfile.gettempdir()
        return os.path.join(base_dir, f"action_flow_step_{result_id}.pkl")

    def _run(self, *args, **kwargs):
        """
        Execute all ActionNodes in sequence using iterative for-loop.

        Overrides Workflow._run to use ActionNodes directly with
        shared ExecutionRuntime context, maintaining O(1) stack depth.
        """
        import logging
        import time
        _logger = logging.getLogger(__name__)

        if not self.action_nodes:
            _logger.debug(f"[ActionFlow._run] No action nodes to execute, returning context")
            return self.context

        _logger.debug(f"[ActionFlow._run] Starting execution of {len(self.action_nodes)} action nodes")
        for i, node in enumerate(self.action_nodes):
            _logger.debug(f"[ActionFlow._run] Executing action {i+1}/{len(self.action_nodes)}: {node.action.type} (id={node.action.id})")
            try:
                # Execute the action node with the shared context
                result = node.run(self.context)
                _logger.debug(f"[ActionFlow._run] Action {node.action.id} completed: success={result.success}")

                # Store result in context for subsequent actions
                self.context.set_result(node.action.id, result)

                # Check for failure
                if not result.success:
                    _logger.debug(f"[ActionFlow._run] Action {node.action.id} failed, raising error")
                    raise ActionExecutionError(
                        action_id=node.action.id,
                        original_error=result.error or ValueError("Action failed"),
                    )

                # Handle wait option (for debugging)
                wait = node.action.wait
                if wait is not None:
                    if wait is True:
                        # Human confirmation mode
                        _logger.info(f"Action '{node.action.id}' ({node.action.type}) completed. Waiting for confirmation...")
                        input("Press Enter to continue to next action...")
                    elif isinstance(wait, (int, float)) and wait > 0:
                        # Timed wait mode
                        _logger.info(f"Action '{node.action.id}' completed. Waiting {wait}s...")
                        time.sleep(wait)

            except ActionExecutionError:
                raise  # Re-raise ActionExecutionError as-is
            except Exception as e:
                _logger.debug(f"[ActionFlow._run] Exception during action {node.action.id}: {e}")
                raise ActionExecutionError(
                    action_id=node.action.id,
                    original_error=e,
                )

        _logger.debug(f"[ActionFlow._run] All actions completed successfully")
        return self.context

    def execute(
        self,
        sequence: Optional[Union[ActionSequence, str, Path]] = None,
        initial_variables: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute action sequence from ActionSequence object, file path, or JSON string.

        Uses Workflow's iterative for-loop pattern internally for O(1) stack depth
        regardless of sequence length.

        Args:
            sequence: Action sequence to execute. Can be:
                - ActionSequence object
                - File path (str or Path) to JSON file
                - JSON string containing action sequence
                - None to use the stored self.sequence
            initial_variables: Optional initial variable values (for future increments)

        Returns:
            ExecutionResult with final state and outputs

        Raises:
            FileNotFoundError: If file path doesn't exist
            ValueError: If JSON is invalid or doesn't match schema
        """
        import logging
        _logger = logging.getLogger(__name__)

        # Use stored sequence if none provided
        if sequence is None:
            sequence = self.sequence

        if sequence is None:
            raise ValueError("No sequence provided and no stored sequence available")

        # Load sequence if it's a file path or JSON string
        if isinstance(sequence, (str, Path)):
            sequence = self._load_sequence(sequence)

        # Store the sequence for potential serialization
        self.sequence = sequence

        _logger.debug(f"[ActionFlow.execute] Sequence has {len(sequence.actions)} actions: {[a.type for a in sequence.actions]}")

        # Set up runtime state
        self.context = ExecutionRuntime(variables=initial_variables or {})

        # Build ActionNodes from the sequence with persistence and template settings
        self.action_nodes = [
            ActionNode(
                action=action,
                action_executor=self.action_executor,
                action_metadata=self.action_metadata,
                template_engine=self.template_engine,
                enable_result_save=self.enable_result_save,
                result_save_dir=self.result_save_dir,
            )
            for action in sequence.actions
        ]
        _logger.debug(f"[ActionFlow.execute] Built {len(self.action_nodes)} ActionNodes")

        # If resume is enabled, load saved results and filter nodes
        if self.resume_with_saved_results:
            _logger.debug(f"[ActionFlow.execute] Resume enabled, filtering nodes...")
            self.action_nodes = self._filter_nodes_with_resume(self.action_nodes, self.context)
            _logger.debug(f"[ActionFlow.execute] After filtering: {len(self.action_nodes)} nodes to execute")

        # Execute using inherited Workflow.run() -> _run()
        try:
            _logger.debug(f"[ActionFlow.execute] Calling self.run() to execute {len(self.action_nodes)} action nodes")
            self.run()
            _logger.debug(f"[ActionFlow.execute] self.run() completed successfully")
            return ExecutionResult(success=True, context=self.context)
        except ActionExecutionError as e:
            _logger.debug(f"[ActionFlow.execute] ActionExecutionError: {e.action_id}: {e.original_error}")
            return ExecutionResult(
                success=False,
                context=self.context,
                error=e.original_error,
                failed_action_id=e.action_id,
            )
        except Exception as e:
            _logger.debug(f"[ActionFlow.execute] Exception: {e}")
            return ExecutionResult(
                success=False,
                context=self.context,
                error=e,
                failed_action_id=self.context.current_action_id if self.context else None,
            )
    
    def _filter_nodes_with_resume(
        self,
        action_nodes: List[ActionNode],
        context: ExecutionRuntime,
    ) -> List[ActionNode]:
        """
        Filter action nodes by loading saved results for completed actions.
        
        Actions with saved results are skipped, and their results are loaded
        into the context. Only actions without saved results are returned
        for execution.
        
        Args:
            action_nodes: List of ActionNode instances to filter.
            context: ExecutionRuntime to populate with saved results.
        
        Returns:
            List of ActionNode instances that need to be executed.
        """
        nodes_to_execute = []
        
        for node in action_nodes:
            saved_result = node.load_saved_result()
            if saved_result is not None:
                # Load saved result into context
                context.set_result(node.action.id, saved_result)
            else:
                # No saved result, need to execute
                nodes_to_execute.append(node)
        
        return nodes_to_execute

    # Serializable interface methods
    def to_serializable_obj(
        self,
        mode: str = 'auto',
        _output_format: Optional[str] = None
    ) -> Union[Dict[str, Any], 'ActionFlow']:
        """Convert ActionFlow to serializable Python object.
        
        Overrides Serializable.to_serializable_obj() to provide custom
        serialization that preserves the action sequence structure.
        
        For Python format (_output_format='python'), returns self to indicate
        special handling is needed by serialize().
        
        Args:
            mode: Serialization mode ('auto', 'dict', 'pickle')
            _output_format: Target output format for conflict detection
        
        Returns:
            - self when _output_format='python' (special handling)
            - Dict containing version, sequence, and config otherwise
        """
        # For Python format, return self to indicate special handling
        if _output_format == 'python':
            return self
        
        return {
            FIELD_TYPE: type(self).__name__,
            FIELD_MODULE: type(self).__module__,
            FIELD_SERIALIZATION: SERIALIZATION_DICT,
            "version": "1.0",
            "sequence": self.sequence.model_dump() if self.sequence else None,
            "config": {
                "enable_result_save": self.enable_result_save,
                "resume_with_saved_results": self.resume_with_saved_results,
                "result_save_dir": self.result_save_dir,
            }
        }
    
    @classmethod
    def from_serializable_obj(
        cls,
        obj: Dict[str, Any],
        action_executor: Union[Callable, MultiActionExecutor] = None,
        action_metadata: Optional[ActionMetadataRegistry] = None,
        **context
    ) -> 'ActionFlow':
        """Reconstruct ActionFlow from serializable dict.
        
        Overrides Serializable.from_serializable_obj() to provide custom
        deserialization that reconstructs the action sequence with context
        injection for action_executor and action_metadata.
        
        Args:
            obj: The serializable object (dict)
            action_executor: Callable for executing actions (required)
            action_metadata: Action type registry (optional)
            **context: Additional context parameters
        
        Returns:
            Reconstructed ActionFlow instance
        
        Raises:
            ValueError: If action_executor is not provided
        """
        if action_executor is None:
            action_executor = context.get('action_executor')
        if action_executor is None:
            raise ValueError("Required context parameter 'action_executor' not provided")
        
        if action_metadata is None:
            action_metadata = context.get('action_metadata', ActionMetadataRegistry())
        
        # Reconstruct sequence from dict
        sequence_data = obj.get("sequence")
        sequence = ActionSequence(**sequence_data) if sequence_data else None
        
        config = obj.get("config", {})
        
        return cls(
            action_executor=action_executor,
            action_metadata=action_metadata,
            sequence=sequence,
            enable_result_save=config.get("enable_result_save", False),
            resume_with_saved_results=config.get("resume_with_saved_results", False),
            result_save_dir=config.get("result_save_dir"),
        )

    def serialize(
        self,
        output_format: str = 'json',
        path: Optional[Union[str, Path]] = None,
        serializable_obj_mode: str = 'auto',
        **kwargs
    ) -> str:
        """Serialize ActionFlow to specified format.
        
        Extended to support output_format='python' for Python script generation.
        
        Args:
            output_format: Output format ('json', 'yaml', 'pickle', or 'python')
            path: Optional file path to write result
            serializable_obj_mode: Mode for to_serializable_obj ('auto', 'dict', 'pickle')
            **kwargs: Format-specific options:
                - For 'python': include_imports, variable_name
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
                include_imports=kwargs.get('include_imports', True),
                variable_name=kwargs.get('variable_name', 'flow'),
            )
        
        # Delegate to parent for other formats
        return super().serialize(
            output_format=output_format,
            path=path,
            serializable_obj_mode=serializable_obj_mode,
            **kwargs
        )

    def _generate_python_script(
        self,
        path: Optional[Union[str, Path]] = None,
        include_imports: bool = True,
        variable_name: str = 'flow',
    ) -> str:
        """Generate executable Python script from ActionFlow structure.
        
        Generates a Python script that creates an ActionFlow and executes
        an ActionSequence with all actions. ActionFlow is sequential only,
        so no branching constructs are generated.
        
        Args:
            path: Optional file path to write the script
            include_imports: Whether to include import statements
            variable_name: Variable name for the flow (default: 'flow')
        
        Returns:
            Generated Python script as string
        """
        lines = []
        
        # Generate imports
        if include_imports:
            lines.append("from agent_foundation.automation.schema import ActionFlow, ActionSequence, Action")
            lines.append("from agent_foundation.automation.schema import TargetSpec, TargetSpecWithFallback")
            lines.append("")
        
        # Generate flow construction
        lines.append(f"{variable_name} = ActionFlow(action_executor=driver, action_metadata=registry)")
        
        # Generate ActionSequence with actions
        if self.sequence and self.sequence.actions:
            lines.append(f"{variable_name}.execute(ActionSequence(")
            
            # Add sequence id if present
            if self.sequence.id:
                escaped_id = self.sequence.id.replace('\\', '\\\\').replace('"', '\\"')
                lines.append(f'    id="{escaped_id}",')
            
            lines.append("    actions=[")
            
            for action in self.sequence.actions:
                action_str = self._action_to_python(action, indent=8)
                lines.append(action_str + ",")
            
            lines.append("    ]")
            lines.append("))")
        
        script = "\n".join(lines)
        
        # Write to file if path provided
        if path:
            Path(path).write_text(script, encoding='utf-8')
        
        return script

    def _action_to_python(self, action: Action, indent: int = 0) -> str:
        """Convert Action to Python constructor string.
        
        Args:
            action: The Action object to convert
            indent: Number of spaces for indentation
        
        Returns:
            Python code string like: Action(id="a1", type="click", target="btn")
        """
        indent_str = " " * indent
        parts = []
        
        # Add id if present and not auto-generated
        if action.id:
            escaped_id = action.id.replace('\\', '\\\\').replace('"', '\\"')
            parts.append(f'id="{escaped_id}"')
        
        # Add type (required)
        parts.append(f'type="{action.type}"')
        
        # Add target if present
        if action.target is not None:
            target_str = self._target_to_python(action.target)
            parts.append(f"target={target_str}")
        
        # Add args if present
        if action.args:
            args_str = self._args_to_python(action.args)
            parts.append(f"args={args_str}")
        
        return f"{indent_str}Action({', '.join(parts)})"

    def _target_to_python(self, target: Union[TargetSpec, TargetSpecWithFallback, str]) -> str:
        """Convert target (str, TargetSpec, TargetSpecWithFallback) to Python code.
        
        Args:
            target: The target specification
        
        Returns:
            Python code string representing the target
        """
        if isinstance(target, str):
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

    def _args_to_python(self, args: Dict[str, Any]) -> str:
        """Convert args dict to Python dict literal string.
        
        Args:
            args: Dictionary of action arguments
        
        Returns:
            Python code string for the dict
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

    # Python deserialization methods
    @classmethod
    def deserialize(
        cls,
        source: Union[str, Path, bytes],
        output_format: str = 'json',
        **context
    ) -> 'ActionFlow':
        """Deserialize ActionFlow from specified format.
        
        Extended to support output_format='python' for Python script parsing.
        
        Args:
            source: Source data (string, file path, or bytes)
            output_format: Input format ('json', 'yaml', 'pickle', or 'python')
            **context: Context parameters:
                - action_executor: Required callable for executing actions
                - action_metadata: Optional ActionMetadataRegistry
        
        Returns:
            Reconstructed ActionFlow instance
        
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
        action_executor: Callable,
        action_metadata: Optional[ActionMetadataRegistry] = None,
        **context
    ) -> 'ActionFlow':
        """Execute Python script and extract the ActionFlow object.

        Uses exec() to run the script with injected driver and registry,
        then extracts the constructed flow object.

        Args:
            source: Python script string or file path
            action_executor: Callable for executing actions (injected as 'driver')
            action_metadata: Action type registry (injected as 'registry')
            **context: Additional context parameters

        Returns:
            Reconstructed ActionFlow instance

        Raises:
            ValueError: If no ActionFlow found in script
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
            'ActionFlow': cls,
            'ActionSequence': ActionSequence,
            'Action': Action,
            'TargetSpec': TargetSpec,
            'TargetSpecWithFallback': TargetSpecWithFallback,
            'ActionMetadataRegistry': ActionMetadataRegistry,
        }

        # Execute script
        exec(script_content, namespace)

        # Find the ActionFlow instance
        for var_name in ['flow', 'f']:
            if var_name in namespace and isinstance(namespace[var_name], cls):
                return namespace[var_name]

        # Search all values
        for value in namespace.values():
            if isinstance(value, cls):
                return value

        raise ValueError("No ActionFlow object found in script")

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

    def _load_sequence(self, source: Union[str, Path]) -> ActionSequence:
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


# Backward compatibility alias
SequenceExecutor = ActionFlow
"""
Deprecated alias for ActionFlow.

Use ActionFlow instead. This alias is provided for backward compatibility
and will be removed in a future version.
"""

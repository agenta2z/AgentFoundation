from typing import Union, Callable, Mapping, Optional, Any, MutableMapping

from attr import attrs, attrib


@attrs
class MultiActionExecutor:
    """
    Unified executor that supports single callable or action_type → callable mapping.

    Reusable by Agent, ActionGraph, and ActionFlow.

    Usage:
        # Single executor
        executor = MultiActionExecutor(callable=my_executor)

        # Multi-executor with dict (auto-detected as callable_mapping)
        executor = MultiActionExecutor(
            {'Navigation.VisitURL': url_exec, 'default': fallback}
        )

        # Multi-executor with explicit callable_mapping
        executor = MultiActionExecutor(
            callable_mapping={'Navigation.VisitURL': url_exec, 'default': fallback}
        )

        # Multi-executor with string IDs
        executor = MultiActionExecutor(
            callable_mapping={'Navigation.VisitURL': 'browser', 'default': 'browser'},
            executor_ids={'browser': browser_executor}
        )
    """

    # Option 1: Single callable for all action types
    callable: Optional[Callable] = attrib(default=None)

    # Option 2: Mapping of action_type → callable or string ID
    callable_mapping: Optional[Mapping[str, Union[str, Callable]]] = attrib(default=None)

    # For resolving string IDs in mapping
    executor_ids: Optional[Mapping[str, Callable]] = attrib(default=None)

    # Fallback key when action_type not found in mapping
    default_key: str = attrib(default='default')

    # State management for stateful executors (e.g., browser instances)
    executor_states: Optional[Mapping[str, Any]] = attrib(default=None)

    def __attrs_post_init__(self):
        # Auto-detect: if callable is actually a Mapping, treat it as callable_mapping
        if self.callable is not None and isinstance(self.callable, Mapping):
            if self.callable_mapping is not None:
                raise ValueError("Cannot provide both 'callable' (as mapping) and 'callable_mapping'")
            self.callable_mapping = self.callable
            self.callable = None

        if self.callable is None and self.callable_mapping is None:
            raise ValueError("Must provide either 'callable' or 'callable_mapping'")
        if self.callable is not None and self.callable_mapping is not None:
            raise ValueError("Cannot provide both 'callable' and 'callable_mapping'")

    def resolve(self, action_type: str) -> Callable:
        """Resolve executor for given action type."""
        if self.callable is not None:
            return self.callable

        # Mapping mode
        if action_type in self.callable_mapping:
            executor = self.callable_mapping[action_type]
        elif self.default_key in self.callable_mapping:
            executor = self.callable_mapping[self.default_key]
        else:
            raise ValueError(
                f"No executor for action type '{action_type}' and no '{self.default_key}' fallback"
            )

        # Resolve string ID if needed
        if isinstance(executor, str):
            if not self.executor_ids:
                raise ValueError(f"No executor_ids provided but found string ID '{executor}'")
            if executor not in self.executor_ids:
                raise ValueError(f"Executor ID '{executor}' not found in executor_ids")
            return self.executor_ids[executor]

        return executor

    def __call__(self, action_type: str, *args, **kwargs) -> Any:
        """Execute action by resolving and calling appropriate executor."""
        executor = self.resolve(action_type)
        return executor(action_type=action_type, *args, **kwargs)

    def add_executor(self, action_type: str, executor: Union[str, Callable]):
        """Dynamically add an executor for an action type."""
        if self.callable is not None:
            # Convert from single to mapping mode
            self.callable_mapping = {self.default_key: self.callable}
            self.callable = None
        elif self.callable_mapping is None:
            self.callable_mapping = {}

        # Ensure mutable
        if not isinstance(self.callable_mapping, dict):
            self.callable_mapping = dict(self.callable_mapping)

        self.callable_mapping[action_type] = executor

    def get_state(self, action_type: str) -> Any:
        """Get state for a specific action type's executor.

        Args:
            action_type: The action type to get state for.

        Returns:
            The state for the executor, or None if not found.
        """
        if self.executor_states is None:
            return None
        if action_type in self.executor_states:
            return self.executor_states[action_type]
        return self.executor_states.get(self.default_key)

    def set_state(self, action_type: str, state: Any):
        """Set state for a specific action type's executor.

        Args:
            action_type: The action type to set state for.
            state: The state to set.
        """
        if self.executor_states is None:
            self.executor_states = {}
        if not isinstance(self.executor_states, MutableMapping):
            self.executor_states = dict(self.executor_states)
        self.executor_states[action_type] = state

    def copy(self, clear_states: bool = True) -> 'MultiActionExecutor':
        """Create a copy of the MultiActionExecutor.

        Args:
            clear_states: If True, clears executor_states in the copy.
                         If False, preserves states from the original.

        Returns:
            A new MultiActionExecutor instance with copied attributes.
        """
        return MultiActionExecutor(
            callable=self.callable,
            callable_mapping=dict(self.callable_mapping) if self.callable_mapping else None,
            executor_ids=self.executor_ids,
            default_key=self.default_key,
            executor_states=None if clear_states else (dict(self.executor_states) if self.executor_states else None)
        )

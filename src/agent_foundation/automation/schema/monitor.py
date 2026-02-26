"""
Monitor Module - Generic Layer (Executor-Agnostic)

This module provides the generic monitor support for ActionGraph/ActionFlow.
These components are executor-agnostic and can be used for any monitoring scenario
(browser tab monitoring, API polling, file watching, etc.).

Components:
- MonitorStatus: Enum for monitor completion status
- MonitorResult: Result dataclass for monitor execution
- MonitorNode: WorkGraphNode subclass for monitor execution

The concrete layer (WebDriver-specific) lives in WebAgent.automation.monitor:
- MonitorConditionType: Enum for built-in condition types
- MonitorCondition: Condition specification class
- create_monitor(): Factory function for element monitoring on current tab
"""

import time
from enum import Enum
from typing import Any, Callable, Optional, Union

from attr import attrs, attrib

from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode
from rich_python_utils.common_objects.workflow.common.worknode_base import NextNodesSelector


class MonitorStatus(str, Enum):
    """Status of monitor execution completion.
    
    Indicates how the monitor loop terminated:
    - CONDITION_MET: The monitored condition was satisfied
    - MAX_ITERATIONS: Maximum iteration count reached without condition met
    - TIMEOUT: Time limit exceeded (for future use)
    - ERROR: An error occurred during monitoring
    """
    CONDITION_MET = "condition_met"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    ERROR = "error"


@attrs
class MonitorResult:
    """
    Result of monitor execution - executor agnostic.
    
    This dataclass captures the outcome of a monitor operation, including
    whether the condition was met, how many checks were performed, and
    any matched content or error information.
    
    Attributes:
        success: True if the monitored condition was met
        status: MonitorStatus indicating how monitoring completed
        matched_content: Content that matched the condition (if any)
        check_count: Number of condition checks performed
        error_message: Error description if status is ERROR
        metadata: Optional dict for executor-specific data (e.g., tab_handle)
    
    Example:
        >>> result = MonitorResult(
        ...     success=True,
        ...     status=MonitorStatus.CONDITION_MET,
        ...     matched_content="Order Complete",
        ...     check_count=5
        ... )
        >>> result.success
        True
    """
    success: bool = attrib()
    status: MonitorStatus = attrib()
    matched_content: Optional[Any] = attrib(default=None)
    check_count: int = attrib(default=0)
    error_message: Optional[str] = attrib(default=None)
    metadata: Optional[dict] = attrib(default=None)


@attrs(slots=False)
class MonitorNode(WorkGraphNode):
    """
    Generic WorkGraphNode for monitor execution.

    MonitorNode provides a generic monitoring capability that can work with
    any callable that returns MonitorResult. The loop behavior is controlled
    by max_repeat/repeat_condition inherited from WorkGraphNode.

    Two key parts:
    1. iteration: Callable that performs one check, returns MonitorResult
    2. loop: Controlled by max_repeat/repeat_condition (inherited from WorkGraphNode)

    The iteration callable can be:
    - A simple function returning MonitorResult
    - An ActionGraph/ActionFlow (which is callable)
    - A factory-created callable with state (e.g., create_monitor())
    - Any callable that returns MonitorResult

    This design keeps MonitorNode executor-agnostic. The concrete implementation
    (browser tab monitoring, API polling, file watching) is passed as the iteration.

    Verify Setup:
        MonitorNode can verify the context is valid before checking the condition.
        This runs FIRST, before any setup or iteration.

        Use cases:
        - Browser monitoring: verify we're on the monitored tab before checking
        - API polling: verify connection is still valid

        Control via:
        - verify_setup: Callable that returns True if context is valid
        - enable_verify_setup: If True (default), check verify_setup before iteration

    Auto Setup:
        When verify_setup returns False, MonitorNode can auto-fix the context.
        This is useful for:
        - Browser monitoring: auto-switch to the monitored tab
        - API polling: refresh authentication tokens
        - File watching: verify directory exists

        Control via:
        - setup_action: Callable to fix context (no args, no return)
        - enable_auto_setup: If True (default), run setup_action when verify fails.
                             If False, monitor is "paused" until context becomes valid.

    Execution Order:
        1. verify_setup() - check if context is valid
        2. If verify fails AND enable_auto_setup: run setup_action() to fix
        3. If verify fails AND NOT enable_auto_setup: return "not met" (paused)
        4. Run iteration()

    Attributes:
        iteration: Callable that performs one monitoring check.
                   Must return MonitorResult.
        verify_setup: Optional callable that returns True if context is valid.
                      Checked FIRST before any setup or iteration.
        enable_verify_setup: If True (default), check verify_setup before iteration.
        setup_action: Optional callable to fix context when verify_setup fails.
                      Takes no arguments, returns nothing. Default None.
        enable_auto_setup: If True (default), run setup_action when verify fails.
                           If False, monitor is "paused" when context is invalid.

    Example:
        >>> def api_poll_iteration(prev_result=None) -> MonitorResult:
        ...     # Custom iteration that polls an API
        ...     import requests
        ...     try:
        ...         response = requests.get("https://api.example.com/status")
        ...         if response.json().get("ready"):
        ...             return MonitorResult(success=True, status=MonitorStatus.CONDITION_MET)
        ...         return MonitorResult(success=False, status=MonitorStatus.MAX_ITERATIONS)
        ...     except Exception as e:
        ...         return MonitorResult(success=False, status=MonitorStatus.ERROR, error_message=str(e))
        ...
        >>> monitor = MonitorNode(
        ...     name="api_monitor",
        ...     iteration=api_poll_iteration,
        ...     max_repeat=30,
        ... )

        # For browser monitoring, use the concrete layer from WebAgent:
        >>> from webaxon.automation.monitor import (
        ...     MonitorCondition, MonitorConditionType, create_monitor
        ... )
        >>> from agent_foundation.automation.schema import TargetSpec
        >>> iteration = create_monitor(webdriver, TargetSpec(strategy="xpath", value="//div"), condition)
        >>> monitor = MonitorNode(name="element_monitor", iteration=iteration, max_repeat=60)
    """
    # The callable that performs one monitoring iteration
    # Must return NextNodesSelector wrapping MonitorResult
    iteration: Callable[..., 'NextNodesSelector'] = attrib(default=None, kw_only=True)

    # Auto setup: run before each iteration (e.g., switch to monitored tab)
    setup_action: Optional[Callable[[], None]] = attrib(default=None, kw_only=True)
    enable_auto_setup: bool = attrib(default=True, kw_only=True)

    # Verify setup: check context is valid before condition check
    verify_setup: Optional[Callable[[], bool]] = attrib(default=None, kw_only=True)
    enable_verify_setup: bool = attrib(default=True, kw_only=True)

    # Poll interval: delay between iterations (seconds)
    # This delay is applied in _execute_iteration when:
    # - verify_setup fails and enable_auto_setup=False
    # - iteration returns "not met" (result.result.success=False)
    poll_interval: float = attrib(default=2.0, kw_only=True)

    # Internal counter for tracking iterations (for debugging)
    _loop_counter: int = attrib(default=0, init=False)

    def __attrs_post_init__(self):
        # Set value to execute the iteration callable
        self.value = self._execute_iteration
        super().__attrs_post_init__()

    def __str__(self) -> str:
        """Return node display string for inherited str_all_descendants()."""
        return f"[{self.name}] (monitor)"

    def _execute_iteration(self, prev_result=None, **kwargs) -> Union['MonitorResult', 'NextNodesSelector']:
        """Execute one monitor iteration.

        Execution order:
        1. Verify context is valid (if enable_verify_setup=True and verify_setup provided)
        2. If verify fails:
           - If enable_auto_setup=True: run setup_action to fix context, then proceed
           - If enable_auto_setup=False: return NextNodesSelector with include_others=False
        3. Run the actual iteration

        Args:
            prev_result: Result from previous iteration (if any)
            **kwargs: Additional keyword arguments (e.g., _is_self_loop from executor)

        Returns:
            NextNodesSelector wrapping MonitorResult (controls downstream execution)
        """
        import logging
        _logger = logging.getLogger(__name__)

        # Track and log loop execution
        self._loop_counter += 1
        is_self_loop = kwargs.get('_is_self_loop', False)
        _logger.debug(f"[MonitorNode._execute_iteration] ===== LOOP #{self._loop_counter} START (is_self_loop={is_self_loop}) =====")

        if self.iteration is None:
            return MonitorResult(
                success=False,
                status=MonitorStatus.ERROR,
                error_message="No iteration configured"
            )

        # Step 1: Verify context is valid (e.g., are we on the correct tab?)
        if self.enable_verify_setup and self.verify_setup is not None:
            verify_result = self.verify_setup()
            _logger.debug(f"[MonitorNode._execute_iteration] verify_setup() returned: {verify_result}")
            if not verify_result:
                # Context not valid - can we auto-fix it?
                if self.enable_auto_setup and self.setup_action is not None:
                    # Run setup to fix the context (e.g., switch to monitored tab)
                    _logger.debug(f"[MonitorNode._execute_iteration] Running setup_action to fix context")
                    self.setup_action()
                    # Proceed to iteration (assume setup fixed the context)
                else:
                    # Cannot auto-fix - return "not met" (monitor paused)
                    # IMPORTANT: Must wrap in NextNodesSelector to prevent downstream from running
                    # include_self=True: keep polling via self-edge
                    # include_others=False: DON'T run downstream actions
                    _logger.debug(f"[MonitorNode._execute_iteration] Cannot auto-fix, returning 'not met' with include_others=False")

                    # Apply poll interval delay before returning
                    time.sleep(self.poll_interval)

                    result = MonitorResult(
                        success=False,
                        status=MonitorStatus.MAX_ITERATIONS,  # Continues polling
                        error_message="verify_setup returned False - context not valid"
                    )
                    return NextNodesSelector(
                        include_self=True,
                        include_others=False,
                        result=result
                    )

        # Step 2: Run the actual iteration
        result = self.iteration(prev_result)

        # Apply poll interval delay if condition not met
        # iteration must return NextNodesSelector wrapping MonitorResult
        if not result.result.success:
            time.sleep(self.poll_interval)

        return result

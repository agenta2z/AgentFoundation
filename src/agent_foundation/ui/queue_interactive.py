from abc import ABC
from typing import Optional, Any, Union, Dict

from attr import attrs, attrib
from agent_foundation.ui.interactive_base import LOG_TYPE_USER_INPUT, LOG_TYPE_SYSTEM_RESPONSE, \
    InteractionFlags
from agent_foundation.ui.rich_interactive_base import RichInteractiveBase
from rich_python_utils.service_utils.queue_service.queue_service_base import QueueServiceBase

@attrs
class QueueInteractive(RichInteractiveBase, ABC):
    """
    A queue-based implementation of `InteractiveBase` for interacting with users through
    queue services for input and output.

    This class provides methods for capturing user input from a queue service, displaying
    responses to a queue service, and resetting the input state. It supports any queue
    service implementation that follows the QueueServiceBase interface.

    Attributes:
        pending_message: Message displayed when awaiting further input
        input_queue: QueueServiceBase instance for receiving user input
        response_queue: QueueServiceBase instance for sending agent responses
        input_queue_id: Queue identifier for input queue (default: 'input')
        response_queue_id: Queue identifier for response queue (default: 'response')
        blocking: Whether to block when getting input (default: True)
        timeout: Optional timeout in seconds for blocking get operations

    Methods:
        get_input() -> str:
            Gets user input from the input queue service.

        reset_input(flag: InteractionFlags) -> None:
            Resets the input state (currently a no-op).

        _send_response(response: str, flag: InteractionFlags) -> None:
            Sends the response to the response queue service with interaction flag metadata.

    Example:
        # Using StorageBasedQueueService for inter-process communication
        from rich_python_utils.service_utils.queue_service.storage_based_queue_service import (
            StorageBasedQueueService
        )

        # Create shared queue service
        queue_service = StorageBasedQueueService(root_path='/tmp/agent_queues')

        # Create interactive instance
        interactive = QueueInteractive(
            system_name="Assistant",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='user_input',
            response_queue_id='agent_response'
        )

        # In producer process: put user input
        queue_service.put('user_input', "Hello, how are you?")

        # In agent process: get input and send response
        user_input = interactive.get_input()
        interactive.send_response("I'm doing well, thank you!")

        # In consumer process: get response
        response = queue_service.get('agent_response')
    """

    # Queue service attributes (keyword-only to work with base class defaults)
    input_queue: Union[QueueServiceBase, Any] = attrib(kw_only=True)  # QueueServiceBase instance
    response_queue: Union[QueueServiceBase, Any] = attrib(kw_only=True)  # QueueServiceBase instance
    input_queue_id: str = attrib(default='input', kw_only=True)
    response_queue_id: str = attrib(default='response', kw_only=True)
    blocking: bool = attrib(default=True, kw_only=True)
    timeout: Optional[float] = attrib(default=None, kw_only=True)
    pending_message: str = attrib(default="Awaiting further input ...", kw_only=True)

    def _get_input(self) -> str:
        """
        Gets input from the user via the input queue service.

        Blocks until input is available (or timeout is reached if configured).

        Returns:
            str: The input text from the queue.

        Raises:
            RuntimeError: If the queue service is closed or unavailable.
        """
        self.log_debug(
            f"Getting input from queue '{self.input_queue_id}'",
            log_type=LOG_TYPE_USER_INPUT
        )

        return self.input_queue.get(
            queue_id=self.input_queue_id,
            blocking=self.blocking,
            timeout=self.timeout
        )

    def reset_input(self, flag: InteractionFlags) -> None:
        """
        Resets the input state.

        This is currently a no-op implementation for queue-based interactions.

        Args:
            flag (InteractionFlags): The interaction state flag.
                - PendingInput: Agent is waiting for user input
                - MessageOnly: Agent sent a message but will continue working
                - TurnCompleted: Agent's turn is complete
        """
        pass

    def _send_response(self, response: Union[str, Dict[str, Any]], flag: InteractionFlags = InteractionFlags.TurnCompleted) -> None:
        """
        Sends the response to the user via the response queue service.

        Puts the response (text or dict with metadata) into the configured response queue
        for consumption by other processes or threads. Includes the interaction flag in the
        message metadata to indicate the interaction state.

        Args:
            response (Union[str, Dict[str, Any]]): The response text or dict with metadata
                (e.g., {"session_id": "...", "response": "..."}) generated by the agent.
            flag (InteractionFlags): The interaction state flag. Defaults to TurnCompleted.
                - PendingInput: Agent is waiting for user input
                - MessageOnly: Agent sent a message but will continue working
                - TurnCompleted: Agent's turn is complete (default)

        Raises:
            RuntimeError: If the queue service is closed or unavailable.
        """
        self.log_debug(
            f"Sending response to queue '{self.response_queue_id}', "
            f"flag={flag.value}",
            log_type=LOG_TYPE_SYSTEM_RESPONSE
        )

        # Ensure response is always a dict with flag
        if isinstance(response, dict):
            response_message = {**response, 'flag': flag}
        else:
            response_message = {'response': response, 'flag': flag}

        # Serialize input_mode from instance state (set by RichInteractiveBase.send_response)
        input_mode = self._current_input_mode
        if input_mode is not None:
            if hasattr(input_mode, 'to_dict'):
                response_message['input_mode'] = input_mode.to_dict()
            elif isinstance(input_mode, dict):
                response_message['input_mode'] = input_mode

        self.response_queue.put(
            queue_id=self.response_queue_id,
            obj=response_message
        )

        # if is_html_element_string(response):
        #     import tempfile
        #     import webbrowser

        #     with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        #         f.write(response)
        #         temp_filename = f.name

        #     print(temp_filename)
        #     print("Displaying response in your default web browser ...")
        #     webbrowser.open('file://' + temp_filename)
        # else:
        #     print(self.get_system_response_string(response))

        # if is_pending:
        #     print(f"\n\n{self.pending_message}")

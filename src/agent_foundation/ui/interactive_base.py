from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Tuple, Any, Union, Dict, Iterable, List

from attr import attrs, attrib

from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_utils import iter_

LOG_TYPE_USER_INPUT = 'UserInput'
LOG_TYPE_SYSTEM_RESPONSE = 'SystemResponse'

class InteractionFlags(StrEnum):
    PendingInput = 'PendingInput'
    MessageOnly = 'MessageOnly'
    TurnCompleted = 'TurnCompleted'


@attrs
class InteractiveBase(Debuggable):
    """
    Abstract base class for creating interactive agents that facilitate communication with users
    through structured inputs and responses.

    This class outlines the core methods necessary for managing user interaction, including methods
    for capturing user input, resetting the input state, and sending responses. Subclasses should
    implement these methods based on the interaction format, which could range from text-based
    interfaces to more complex voice or GUI-based interactions.

    Attributes:
        system_name (str): A label representing the system's identity in interactions (e.g., "System" or "Agent").
        user_name (str): A label representing the user's identity in interactions (e.g., "User").

    Interaction Flags:
        The InteractionFlags enum indicates the interaction state:
        - PendingInput: Agent is waiting for user input (conversation expects user response)
        - MessageOnly: Agent sent an informational message but will continue working independently
        - TurnCompleted: Agent's turn is complete (conversation can end or user can respond)

    Methods:
        get_input() -> Any:
            Abstract method to retrieve input from the user. This method's implementation depends
            on the specific input format, such as text, voice, or other data types.

        reset_input(flag: InteractionFlags) -> None:
            Abstract method to reset or clear the current user input state, preparing the system for
            the next interaction. The flag indicates the interaction state.

        send_response(response: str, flag: InteractionFlags) -> None:
            Sends the response to the user and resets the input state for the next interaction. This
            method invokes `_send_response` to handle the output and then calls `reset_input`.

        _send_response(response: str, flag: InteractionFlags) -> None:
            Abstract protected method that subclasses must implement to handle the delivery of responses.
            The response format and flag usage will vary based on the interaction medium.
    """
    system_name: str = attrib(default="System")
    user_name: str = attrib(default="User")
    log_input_content: bool = attrib(default=True, kw_only=True)
    log_response_content: bool = attrib(default=True, kw_only=True)


    def get_user_input_string(self, user_input: Any) -> str:
        if user_input:
            return f'{self.user_name}: {user_input}'

    def get_system_response_string(self, system_response: Any) -> str:
        if system_response:
            return f'{self.system_name}: {system_response}'

    @abstractmethod
    def _get_input(self):
        raise NotImplementedError

    def get_input(self):
        """
        Retrieves input from the user.

        This method should be implemented to capture user input in the desired format,
        such as text, voice, or other types of input.

        Returns:
            Any: The raw input from the user.
        """

        retrieved_input = self._get_input()
        if retrieved_input is None:
            self.log_debug("No input available (timeout or empty queue)", log_type=LOG_TYPE_USER_INPUT)
        else:
            if self.log_input_content:
                self.log_debug(f"Received input: {retrieved_input}", log_type=LOG_TYPE_USER_INPUT)
            else:
                self.log_debug("Received input", log_type=LOG_TYPE_USER_INPUT)
        return retrieved_input

    @abstractmethod
    def reset_input(self, flag: InteractionFlags):
        """
        Resets or clears the current user input state, preparing for the next interaction.

        This method should be implemented to clear any stored input data or reset
        the interface. The flag provides context for the interaction state, which may
        influence how the reset occurs.

        Args:
            flag (InteractionFlags): The interaction state flag.
                - PendingInput: Agent is waiting for user input
                - MessageOnly: Agent sent a message but will continue working
                - TurnCompleted: Agent's turn is complete
        """
        raise NotImplementedError

    @abstractmethod
    def _send_response(self, response:  Any, flag: InteractionFlags = InteractionFlags.TurnCompleted) -> None:
        """
        Handles the delivery of a single response to the user.

        This abstract method must be implemented by subclasses to define the mechanism for
        delivering a response based on the interaction medium. For example, the response could
        be displayed as text, converted to speech, or rendered in a graphical user interface.

        Args:
            response (Any): The response to deliver to the user. The format of the response
                (e.g., string, audio, visual component) depends on the interaction medium.
            flag (InteractionFlags): The interaction state flag. Defaults to TurnCompleted.
                - PendingInput: Agent is waiting for user input
                - MessageOnly: Agent sent a message but will continue working
                - TurnCompleted: Agent's turn is complete (default)

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


    def _send_pending_message(self):
        """
        No-op placeholder for signaling a pending message.

        This method is currently a no-operation (no-op) implementation and does not perform
        any actions. Subclasses can override this method to define how a "pending" state is
        communicated to the user, such as displaying a loading message or visual indicator.

        Notes:
            - This method is invoked when the flag is truthy in the `send_response` method.
            - By default, this method does nothing (`pass`) but can be extended in subclasses
              as needed.
        """
        pass

    def send_response(self, response: Union[Any, List, Tuple], flag: InteractionFlags = InteractionFlags.TurnCompleted) -> None:
        """
        Sends one or more responses to the user and resets the input state for the next interaction.

        This method iterates through the provided responses, delivering each one using the
        `_send_response` method. If the flag is truthy, it invokes `_send_pending_message`
        to indicate a pending state. Finally, it resets the input state by calling `reset_input`.

        Args:
            response (Union[Any, List, Tuple]): A single response or a collection of responses
                (e.g., list or tuple) to deliver to the user. The format of the responses depends
                on the interaction medium.
            flag (InteractionFlags): The interaction state flag. Defaults to TurnCompleted.
                - PendingInput: Agent is waiting for user input
                - MessageOnly: Agent sent a message but will continue working
                - TurnCompleted: Agent's turn is complete (default)

        Returns:
            None

        Notes:
            - This method handles both single and multiple responses. For multiple responses,
              it iteratively calls `_send_response` for each item.
            - If the flag is truthy, `_send_pending_message` is called to signal a pending state.
            - After processing responses, `reset_input` is invoked to prepare the system for
              the next interaction.
        """
        self.log_debug(f"Sending response(s), flag={flag.value}", log_type=LOG_TYPE_SYSTEM_RESPONSE)

        if self.log_response_content:
            responses_list = []
            for _response in iter_(response):
                self._send_response(_response, flag=flag)
                responses_list.append(_response)
            self.log_debug(f"Response content: {responses_list}", log_type=LOG_TYPE_SYSTEM_RESPONSE)
        else:
            for _response in iter_(response):
                self._send_response(_response, flag=flag)

        self.log_debug("Response(s) sent successfully", log_type=LOG_TYPE_SYSTEM_RESPONSE)

        if flag:
            self._send_pending_message()

        self.reset_input(flag)

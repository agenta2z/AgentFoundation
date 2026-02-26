"""
Queue-based Dash Interactive App with Log Debugging.

This module provides a specialized version of DashInteractiveAppWithLogs that polls
responses from a shared queue service instead of per-session queues. This is useful
when you have a separate service managing agents and you want the UI to poll for
responses from a shared response queue.
"""
from typing import Callable, Optional, Tuple
import time
import dash
from dash.dependencies import Input, Output, State

from science_modeling_tools.ui.dash_interactive.dash_interactive_app_with_logs import DashInteractiveAppWithLogs


class QueueBasedDashInteractiveApp(DashInteractiveAppWithLogs):
    """
    Dash app that polls from a shared queue service instead of per-session queues.

    This is useful when you have a separate service managing agents and you want
    the UI to poll for responses from a shared response queue.

    Example:
        ```python
        def check_responses():
            # Your logic to check for responses from queue
            # Returns: (session_id, response_text, log_collector) or (None, None, None)
            response_data = queue_service.get(RESPONSE_QUEUE_ID, blocking=False)
            if response_data:
                session_id = response_data.get('session_id')
                response = response_data.get('response', '')
                return session_id, response, None
            return None, None, None

        app = QueueBasedDashApp(
            title="My App",
            response_checker=check_responses,
            special_waiting_message="__WAITING__"
        )
        app.set_message_handler(my_handler)
        app.run()
        ```
    """

    def __init__(
        self,
        title: str = "Interactive Debugger with Logs",
        port: int = 8050,
        debug: bool = True,
        message_handler: Optional[Callable[[str], str]] = None,
        response_checker: Optional[Callable[[], Tuple]] = None,
        special_waiting_message: str = "__WAITING_FOR_RESPONSE__",
        custom_monitor_tabs: list = None,
        custom_main_tabs: list = None
    ):
        """
        Initialize the queue-based Dash app.

        Args:
            title: Application title
            port: Port number to run the server on
            debug: Whether to run in debug mode
            message_handler: Optional callback function to handle user messages
            response_checker: Function that checks for responses from queue.
                             Should return (session_id, response, log_collector) or (None, None, None)
            special_waiting_message: Placeholder message to show while waiting for response
            custom_monitor_tabs: Optional list of custom monitor tab dicts with 'id', 'label', 'content'
            custom_main_tabs: Optional list of custom main tab dicts with 'id', 'label', 'content'
        """
        super().__init__(title, port, debug, message_handler, custom_monitor_tabs=custom_monitor_tabs, custom_main_tabs=custom_main_tabs)
        self.response_checker = response_checker
        self.special_waiting_message = special_waiting_message
        self._last_visibility_timestamp = 0

    def set_response_checker(self, checker: Callable[[], Tuple]):
        """
        Set the response checker function.

        Args:
            checker: Function that returns (session_id, response, log_collector) or (None, None, None)
        """
        self.response_checker = checker

    def _register_callbacks(self):
        """Register all callbacks including parent callbacks and polling callback.

        Note: The parent's _register_callbacks() will call self._register_polling_callback()
        via polymorphism, which will call this class's overridden version.
        """
        # Register parent callbacks (session management, message handling, visibility detection)
        # This will also call self._register_polling_callback() through polymorphism
        super()._register_callbacks()

    def _register_polling_callback(self):
        """Register custom polling callback that polls from shared queue."""
        @self.app.callback(
            Output('messages-store', 'data', allow_duplicate=True),
            [
                Input('response-poll-interval', 'n_intervals')
            ],
            [
                State('current-session-store', 'data'),
                State('messages-store', 'data'),
                State('page-visibility-store', 'data')
            ],
            prevent_initial_call=True
        )
        def poll_and_refresh_messages(n_intervals, session_id, messages_store, visibility_data):
            """Poll for queue responses and update messages store.

            The UI will automatically re-render when messages_store changes due to the
            existing callback that watches messages_store as an Input.
            """
            # visibility_data is now in State - no longer triggers callback
            # Callback only fires every 1 second (response-poll-interval)

            # Check for new response using the provided checker
            if not self.response_checker:
                return dash.no_update

            response_session_id, response, log_collector = self.response_checker()

            if response is None:
                return dash.no_update

            # Route response to the correct session
            if response_session_id and response_session_id in messages_store:
                # Create a deep copy to ensure Dash sees it as changed
                messages_store_copy = dict(messages_store)
                messages = list(messages_store_copy[response_session_id])

                # Replace the last message if it's the "waiting" placeholder
                if messages and messages[-1].get('role') == 'assistant':
                    last_content = messages[-1].get('content', '')
                    if last_content == self.special_waiting_message:
                        # Replace placeholder with actual response
                        messages[-1] = {
                            'role': 'assistant',
                            'content': response,
                            'timestamp': time.strftime('%H:%M:%S')
                        }
                    else:
                        # Add as new message
                        messages.append({
                            'role': 'assistant',
                            'content': response,
                            'timestamp': time.strftime('%H:%M:%S')
                        })
                else:
                    # Add new assistant message
                    messages.append({
                        'role': 'assistant',
                        'content': response,
                        'timestamp': time.strftime('%H:%M:%S')
                    })

                # Update the copy with modified messages
                messages_store_copy[response_session_id] = messages

                # Return updated store - the parent's load_session_messages callback
                # will automatically re-render the UI since it watches messages_store
                return messages_store_copy

            return dash.no_update

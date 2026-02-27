"""
Main Dash application for interactive debugging UI.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State

from agent_foundation.ui.dash_interactive.components.chat_history import (
    ChatHistoryList,
)
from agent_foundation.ui.dash_interactive.components.chat_window import ChatWindow


class DashInteractiveApp:
    """
    Main Dash application for GPT-like chat interface and log debugging.

    This application provides:
    - Left sidebar: Chat history and settings
    - Right panel: Chat window with message display and input
    - Tab support for switching between chat and log debugging (future)

    Attributes:
        app (dash.Dash): The Dash application instance
        chat_history (ChatHistoryList): Left sidebar component
        chat_window (ChatWindow): Right panel chat component
        sessions (List[Dict]): List of chat sessions
        current_session_id (str): ID of currently active session
        message_handler (Callable): Callback for processing user messages
    """

    def __init__(
        self,
        title: str = "Interactive Debugger",
        port: int = 8050,
        debug: bool = True,
        message_handler: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the Dash interactive app.

        Args:
            title: Application title
            port: Port number to run the server on
            debug: Whether to run in debug mode
            message_handler: Optional callback function to handle user messages
                           Should accept a string and return a string response
        """
        self.title = title
        self.port = port
        self.debug = debug
        self.message_handler = message_handler or self._default_message_handler

        # Initialize sessions storage
        self.sessions = []
        self.current_session_id = None
        self.session_messages = {}  # session_id -> List[messages]

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            title=self.title,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )

        # Create components
        self.chat_history = ChatHistoryList(
            component_id="chat-history", sessions=self.sessions
        )
        self.chat_window = ChatWindow(component_id="chat-window", messages=[])

        # Set up layout and callbacks
        self.app.layout = self._create_layout()
        self._register_callbacks()

    def _default_message_handler(self, message: str) -> str:
        """
        Default message handler that echoes back the message.

        Args:
            message: User input message

        Returns:
            Echo response
        """
        return f"Echo: {message}"

    def _create_layout(self) -> html.Div:
        """
        Create the main application layout.

        Returns:
            Dash Div containing the full layout
        """
        return html.Div(
            children=[
                # Store components for maintaining state
                dcc.Store(id="sessions-store", data=[]),
                dcc.Store(id="current-session-store", data=None),
                dcc.Store(id="messages-store", data={}),
                # Hidden button for chat tab switching (used by callbacks, not visible)
                html.Button(
                    id="main-panel-chat-btn",
                    n_clicks=0,
                    style={"display": "none"},
                ),
                # Main container with flexbox layout
                html.Div(
                    children=[
                        # Left sidebar - Chat history
                        self.chat_history.layout(),
                        # Right panel - Chat window
                        self.chat_window.layout(),
                    ],
                    style={
                        "display": "flex",
                        "height": "100vh",
                        "width": "100vw",
                        "overflow": "hidden",
                        "fontFamily": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    },
                ),
            ],
            style={"margin": "0", "padding": "0"},
        )

    def _register_callbacks(self):
        """Register all Dash callbacks for interactivity."""
        self._register_session_callbacks()
        self._register_message_callbacks()

    def _register_session_callbacks(self):
        """Register callbacks for session management (can be reused by subclasses)."""

        # Callback for creating new chat session
        @self.app.callback(
            [
                Output("sessions-store", "data"),
                Output("current-session-store", "data"),
                Output("messages-store", "data"),
                Output("main-panel-chat-btn", "n_clicks"),  # Trigger chat tab switch
            ],
            [Input("chat-history-new-chat-btn", "n_clicks")],
            [
                State("sessions-store", "data"),
                State("current-session-store", "data"),
                State("messages-store", "data"),
                State("main-panel-chat-btn", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def create_new_session(
            n_clicks, sessions, current_session, messages_store, chat_btn_clicks
        ):
            if n_clicks:
                # Create new session
                session_id = f"session_{len(sessions) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                new_session = {
                    "id": session_id,
                    "title": f"New Chat {len(sessions) + 1}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "active": True,
                }

                # Mark all other sessions as inactive
                for session in sessions:
                    session["active"] = False

                sessions.append(new_session)
                messages_store[session_id] = []

                # Trigger chat tab by incrementing its n_clicks
                return sessions, session_id, messages_store, (chat_btn_clicks or 0) + 1

            return sessions, current_session, messages_store, dash.no_update

        # Callback for updating chat history display
        @self.app.callback(
            Output("chat-history-session-list", "children"),
            [Input("sessions-store", "data")],
            prevent_initial_call=False,
        )
        def update_history_display(sessions):
            return self.chat_history.update_sessions(sessions)

        # Callback for selecting a session
        @self.app.callback(
            [
                Output("current-session-store", "data", allow_duplicate=True),
                Output("sessions-store", "data", allow_duplicate=True),
            ],
            [Input({"type": "chat-history-session-item", "index": ALL}, "n_clicks")],
            [State("sessions-store", "data"), State("current-session-store", "data")],
            prevent_initial_call=True,
        )
        def select_session(n_clicks_list, sessions, current_session):
            if not any(n_clicks_list):
                return current_session, sessions

            ctx = dash.callback_context
            if not ctx.triggered:
                return current_session, sessions

            # Get the clicked session ID
            clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]
            import json

            session_id = json.loads(clicked_id)["index"]

            # Update active status
            for session in sessions:
                session["active"] = session["id"] == session_id

            return session_id, sessions

    def _register_message_callbacks(self):
        """Register callbacks for message handling (can be overridden by subclasses)."""

        # Callback for loading messages of current session
        @self.app.callback(
            Output("chat-window-messages", "children"),
            [Input("current-session-store", "data"), Input("messages-store", "data")],
            prevent_initial_call=False,
        )
        def load_session_messages(session_id, messages_store):
            if session_id and session_id in messages_store:
                messages = messages_store[session_id]
                return self.chat_window.update_messages(messages)
            return self.chat_window.update_messages([])

        # Callback for sending messages
        @self.app.callback(
            [
                Output("messages-store", "data", allow_duplicate=True),
                Output("chat-window-input", "value"),
                Output("sessions-store", "data", allow_duplicate=True),
            ],
            [Input("chat-window-send-btn", "n_clicks")],
            [
                State("chat-window-input", "value"),
                State("current-session-store", "data"),
                State("messages-store", "data"),
                State("sessions-store", "data"),
            ],
            prevent_initial_call=True,
        )
        def send_message(
            send_clicks, message_text, session_id, messages_store, sessions
        ):
            if not message_text or not message_text.strip():
                return messages_store, "", sessions

            # Create session if none exists
            if not session_id:
                session_id = f"session_1_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                new_session = {
                    "id": session_id,
                    "title": "New Chat",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "active": True,
                }
                sessions.append(new_session)
                messages_store[session_id] = []

            # Add user message
            timestamp = datetime.now().strftime("%H:%M:%S")
            user_msg = {
                "role": "user",
                "content": message_text.strip(),
                "timestamp": timestamp,
            }

            if session_id not in messages_store:
                messages_store[session_id] = []

            messages_store[session_id].append(user_msg)

            # Get response from message handler
            try:
                response = self.message_handler(message_text.strip())
            except Exception as e:
                response = f"Error: {str(e)}"

            # Add assistant response
            assistant_msg = {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            messages_store[session_id].append(assistant_msg)

            # Update session title with first message if it's "New Chat"
            for session in sessions:
                if session["id"] == session_id and session["title"].startswith(
                    "New Chat"
                ):
                    # Use first few words of the message as title
                    words = message_text.split()[:5]
                    session["title"] = " ".join(words) + (
                        "..." if len(words) >= 5 else ""
                    )
                    break

            return messages_store, "", sessions

    def run(self, host: str = "0.0.0.0"):
        """
        Run the Dash application.

        Args:
            host: Host address to bind to
        """
        import os
        import socket

        # Get hostname for SSL certificate lookup
        hostname = socket.getfqdn()

        # Use devserver's built-in SSL certificates
        ssl_cert = f"/etc/pki/tls/certs/{hostname}.crt"
        ssl_key = f"/etc/pki/tls/certs/{hostname}.key"

        ssl_context = None
        if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
            ssl_context = (ssl_cert, ssl_key)
            print(f"\n{'='*60}")
            print(f"Starting {self.title} (HTTPS)")
            print(f"Server running at: https://{hostname}:{self.port}")
            print(f"SSL certificates: {ssl_cert}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"Starting {self.title}")
            print(f"Server running at: http://localhost:{self.port}")
            print(f"WARNING: SSL certificates not found at {ssl_cert}")
            print(f"{'='*60}\n")

        # Disable reloader - it doesn't work with Buck2 PAR files
        self.app.run(
            host=host,
            port=self.port,
            debug=self.debug,
            use_reloader=False,
            ssl_context=ssl_context,
        )

    def set_message_handler(self, handler: Callable[[str], str]):
        """
        Set a custom message handler function.

        Args:
            handler: Function that takes a message string and returns a response string
        """
        self.message_handler = handler

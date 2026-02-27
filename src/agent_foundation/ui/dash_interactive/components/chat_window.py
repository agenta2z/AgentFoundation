"""
Chat window component for displaying and sending messages.
"""
from typing import Any, Dict, List, Optional
from dash import html, dcc
from dash.dependencies import Input, Output, State

from agent_foundation.ui.dash_interactive.components.base import BaseComponent
from agent_foundation.ui.dash_interactive.ui_lib.chat_window import create_typing_indicator


class ChatWindow(BaseComponent):
    """
    Component for displaying chat messages and handling user input.

    This component shows the conversation messages in a scrollable area
    with a text input box at the bottom for sending new messages.

    Attributes:
        component_id (str): Unique identifier for this component
        messages (List[Dict]): List of message dictionaries with 'role', 'content', 'timestamp'
        placeholder (str): Placeholder text for input box
    """

    def __init__(
        self,
        component_id: str = "chat-window",
        messages: Optional[List[Dict[str, Any]]] = None,
        placeholder: str = "Send a message...",
        style: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the chat window component.

        Args:
            component_id: Unique identifier for this component
            messages: Initial list of chat messages
            placeholder: Placeholder text for the input box
            style: Optional CSS style overrides
        """
        super().__init__(component_id, style)
        self.messages = messages or []
        self.placeholder = placeholder

    def _get_default_style(self) -> Dict[str, Any]:
        """Get default styling for the chat window."""
        return {
            'flex': '1',
            'height': '100%',
            'backgroundColor': '#343541',
            'display': 'flex',
            'flexDirection': 'column',
            'position': 'relative',
            'overflow': 'hidden'
        }

    def layout(self) -> html.Div:
        """
        Generate the chat window layout.

        Returns:
            Dash Div containing messages area and input box
        """
        return html.Div(
            id=self.get_id(),
            children=[
                # Messages display area
                self._create_messages_area(),

                # Input box at bottom
                self._create_input_area()
            ],
            style=self.style
        )

    def _create_messages_area(self) -> html.Div:
        """Create scrollable messages display area."""
        return html.Div(
            id=self.get_id('messages'),
            children=self._render_messages(),
            style={
                'flex': '1',
                'minHeight': '0',
                'overflowY': 'auto',
                'padding': '20px',
                'display': 'flex',
                'flexDirection': 'column',
                'gap': '20px'
            }
        )

    def _render_messages(self) -> List[html.Div]:
        """
        Render individual chat messages.

        Returns:
            List of message Div elements
        """
        if not self.messages:
            return [
                html.Div(
                    children=[
                        html.H2(
                            "Welcome to Interactive Debugger",
                            style={
                                'color': '#ECECF1',
                                'fontSize': '32px',
                                'fontWeight': '600',
                                'marginBottom': '20px',
                                'textAlign': 'center'
                            }
                        ),
                        html.P(
                            "Start a conversation or load logs for debugging.",
                            style={
                                'color': '#8E8EA0',
                                'fontSize': '16px',
                                'textAlign': 'center'
                            }
                        )
                    ],
                    style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center',
                        'alignItems': 'center',
                        'height': '100%'
                    }
                )
            ]

        message_divs = []
        for msg in self.messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')

            # Different styling for user vs assistant messages
            if role == 'user':
                bg_color = '#343541'
                align = 'flex-end'
                msg_bg = '#40414F'
                max_width = '80%'
            else:  # assistant
                bg_color = '#444654'
                align = 'flex-start'
                msg_bg = '#444654'
                max_width = '100%'

            # Check if this is a special waiting message marker
            is_waiting = content == '__WAITING_FOR_RESPONSE__'
            if is_waiting:
                # Use reusable typing indicator from UI element library
                content_element = create_typing_indicator()
            else:
                content_element = content

            message_div = html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                '👤' if role == 'user' else '🤖',
                                style={
                                    'fontSize': '20px',
                                    'marginRight': '12px',
                                    'flexShrink': '0'
                                }
                            ),
                            html.Div(
                                children=[
                                    html.Div(
                                        content_element,
                                        style={
                                            'whiteSpace': 'pre-wrap',
                                            'wordBreak': 'break-word',
                                            'lineHeight': '1.6',
                                            'fontSize': '15px'
                                        }
                                    ),
                                    html.Div(
                                        timestamp,
                                        style={
                                            'fontSize': '11px',
                                            'color': '#8E8EA0',
                                            'marginTop': '8px'
                                        }
                                    ) if timestamp else None
                                ],
                                style={'flex': '1'}
                            )
                        ],
                        style={
                            'display': 'flex',
                            'padding': '16px',
                            'backgroundColor': msg_bg,
                            'borderRadius': '8px',
                            'maxWidth': max_width,
                            'color': '#ECECF1'
                        }
                    )
                ],
                style={
                    'display': 'flex',
                    'justifyContent': align,
                    'width': '100%'
                }
            )
            message_divs.append(message_div)

        return message_divs

    def _create_input_area(self) -> html.Div:
        """Create input area with text box and send button."""
        return html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Textarea(
                            id=self.get_id('input'),
                            placeholder=self.placeholder,
                            value='',
                            style={
                                'width': 'calc(100% - 50px)',
                                'height': '50px',
                                'minHeight': '50px',
                                'maxHeight': '150px',
                                'padding': '12px 16px',
                                'backgroundColor': '#40414F',
                                'color': '#ECECF1',
                                'border': '1px solid #565869',
                                'borderRadius': '8px',
                                'fontSize': '15px',
                                'resize': 'none',
                                'fontFamily': 'inherit',
                                'outline': 'none',
                                'boxSizing': 'border-box'
                            }
                        ),
                        html.Button(
                            '↑',
                            id=self.get_id('send-btn'),
                            n_clicks=0,
                            style={
                                'position': 'absolute',
                                'right': '20px',
                                'bottom': '20px',
                                'width': '36px',
                                'height': '36px',
                                'backgroundColor': '#19C37D',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontSize': '20px',
                                'cursor': 'pointer',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'transition': 'background-color 0.2s',
                                'flexShrink': '0'
                            }
                        )
                    ],
                    style={
                        'position': 'relative',
                        'padding': '15px 20px',
                        'maxWidth': '800px',
                        'margin': '0 auto',
                        'width': '100%',
                        'display': 'flex',
                        'alignItems': 'flex-end',
                        'gap': '10px',
                        'boxSizing': 'border-box'
                    }
                )
            ],
            style={
                'borderTop': '1px solid #565869',
                'backgroundColor': '#343541',
                'flexShrink': '0'
            }
        )

    def get_callback_inputs(self) -> List[Input]:
        """Get list of callback inputs."""
        return [
            Input(self.get_id('send-btn'), 'n_clicks')
        ]

    def get_callback_outputs(self) -> List[Output]:
        """Get list of callback outputs."""
        return [
            Output(self.get_id('messages'), 'children'),
            Output(self.get_id('input'), 'value')
        ]

    def get_callback_states(self) -> List[State]:
        """Get list of callback states."""
        return [
            State(self.get_id('input'), 'value')
        ]

    def update_messages(self, messages: List[Dict[str, Any]]) -> List[html.Div]:
        """
        Update the displayed messages.

        Args:
            messages: New list of message dictionaries

        Returns:
            List of rendered message Divs
        """
        self.messages = messages
        return self._render_messages()

    def add_message(self, role: str, content: str, timestamp: str = "") -> List[html.Div]:
        """
        Add a new message to the chat.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content text
            timestamp: Optional timestamp string

        Returns:
            Updated list of rendered message Divs
        """
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': timestamp
        })
        return self._render_messages()

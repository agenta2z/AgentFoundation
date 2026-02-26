"""
Chat history list component for displaying conversation sessions.
"""
from typing import Any, Dict, List, Optional
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL

from science_modeling_tools.ui.dash_interactive.components.base import BaseComponent


class ChatHistoryList(BaseComponent):
    """
    Component for displaying a list of chat sessions/conversations.

    This component shows a vertical list of chat sessions in the left sidebar,
    similar to ChatGPT's conversation history. Each session can be selected
    to view its messages.

    Attributes:
        component_id (str): Unique identifier for this component
        sessions (List[Dict]): List of session dictionaries with 'id', 'title', 'timestamp'
        show_settings (bool): Whether to show settings section
    """

    def __init__(
        self,
        component_id: str = "chat-history",
        sessions: Optional[List[Dict[str, Any]]] = None,
        show_settings: bool = True,
        style: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the chat history list component.

        Args:
            component_id: Unique identifier for this component
            sessions: Initial list of chat sessions
            show_settings: Whether to display settings section
            style: Optional CSS style overrides
        """
        super().__init__(component_id, style)
        self.sessions = sessions or []
        self.show_settings = show_settings

    def _get_default_style(self) -> Dict[str, Any]:
        """Get default styling for the chat history sidebar."""
        return {
            'width': '300px',
            'height': '100vh',
            'backgroundColor': '#202123',
            'color': '#ECECF1',
            'overflowY': 'auto',
            'display': 'flex',
            'flexDirection': 'column',
            'borderRight': '1px solid #4D4D4F'
        }

    def layout(self) -> html.Div:
        """
        Generate the chat history sidebar layout.

        Returns:
            Dash Div containing the history list and settings
        """
        children = []

        # Header with "New Chat" button
        children.append(self._create_header())

        # Session list container
        children.append(self._create_session_list())

        # Settings section at bottom (if enabled)
        if self.show_settings:
            children.append(self._create_settings())

        return html.Div(
            id=self.get_id(),
            children=children,
            style=self.style
        )

    def _create_header(self) -> html.Div:
        """Create header section with New Chat button."""
        return html.Div(
            children=[
                html.Button(
                    '+ New Chat',
                    id=self.get_id('new-chat-btn'),
                    style={
                        'width': '90%',
                        'margin': '15px 5%',
                        'padding': '12px',
                        'backgroundColor': 'transparent',
                        'color': '#ECECF1',
                        'border': '1px solid #4D4D4F',
                        'borderRadius': '6px',
                        'cursor': 'pointer',
                        'fontSize': '14px',
                        'fontWeight': '500',
                        'transition': 'background-color 0.2s'
                    },
                    n_clicks=0
                )
            ],
            style={'borderBottom': '1px solid #4D4D4F'}
        )

    def _create_session_list(self) -> html.Div:
        """Create scrollable list of chat sessions."""
        return html.Div(
            id=self.get_id('session-list'),
            children=self._render_sessions(),
            style={
                'flex': '1',
                'overflowY': 'auto',
                'padding': '10px 0'
            }
        )

    def _render_sessions(self) -> List[html.Div]:
        """
        Render individual session items.

        Returns:
            List of session Div elements
        """
        if not self.sessions:
            return [
                html.Div(
                    "No conversations yet",
                    style={
                        'padding': '20px',
                        'textAlign': 'center',
                        'color': '#8E8EA0',
                        'fontSize': '13px'
                    }
                )
            ]

        session_divs = []
        for session in self.sessions:
            # Build metadata display (timestamp and session_id)
            metadata_children = []
            if session.get('timestamp'):
                metadata_children.append(
                    html.Div(
                        session.get('timestamp', ''),
                        style={
                            'fontSize': '11px',
                            'color': '#8E8EA0'
                        }
                    )
                )
            # Add session_id display
            if session.get('id'):
                metadata_children.append(
                    html.Div(
                        f"ID: {session.get('id', '')}",
                        style={
                            'fontSize': '10px',
                            'color': '#6E6E80',
                            'marginTop': '2px',
                            'fontFamily': 'monospace',
                            'whiteSpace': 'nowrap',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis'
                        }
                    )
                )

            session_div = html.Div(
                children=[
                    html.Div(
                        session.get('title', 'Untitled Conversation'),
                        style={
                            'fontSize': '14px',
                            'fontWeight': '400',
                            'marginBottom': '4px',
                            'whiteSpace': 'nowrap',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis'
                        }
                    ),
                    html.Div(
                        children=metadata_children,
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'gap': '2px'
                        }
                    ) if metadata_children else None
                ],
                id={'type': self.get_id('session-item'), 'index': session['id']},
                n_clicks=0,
                style={
                    'padding': '12px 16px',
                    'margin': '0 8px',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'transition': 'background-color 0.2s',
                    'backgroundColor': '#343541' if session.get('active') else 'transparent'
                },
                className='session-item'
            )
            session_divs.append(session_div)

        return session_divs

    def _create_settings(self) -> html.Div:
        """Create settings section at the bottom."""
        return html.Div(
            children=[
                html.Hr(style={'border': '1px solid #4D4D4F', 'margin': '0'}),
                html.Div(
                    children=[
                        html.Div(
                            '⚙️ Settings',
                            id=self.get_id('settings-btn'),
                            n_clicks=0,
                            style={
                                'padding': '12px 16px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'transition': 'background-color 0.2s'
                            },
                            className='settings-item'
                        ),
                        html.Div(
                            '📊 Debug Mode',
                            id=self.get_id('debug-toggle'),
                            n_clicks=0,
                            style={
                                'padding': '12px 16px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'transition': 'background-color 0.2s'
                            },
                            className='settings-item'
                        )
                    ]
                )
            ],
            style={
                'marginTop': 'auto'
            }
        )

    def get_callback_inputs(self) -> List[Input]:
        """Get list of callback inputs."""
        return [
            Input(self.get_id('new-chat-btn'), 'n_clicks'),
            Input({'type': self.get_id('session-item'), 'index': ALL}, 'n_clicks'),
            Input(self.get_id('settings-btn'), 'n_clicks'),
            Input(self.get_id('debug-toggle'), 'n_clicks')
        ]

    def get_callback_outputs(self) -> List[Output]:
        """Get list of callback outputs."""
        return [
            Output(self.get_id('session-list'), 'children')
        ]

    def update_sessions(self, sessions: List[Dict[str, Any]]) -> List[html.Div]:
        """
        Update the session list.

        Args:
            sessions: New list of session dictionaries

        Returns:
            List of rendered session Divs
        """
        self.sessions = sessions
        return self._render_sessions()

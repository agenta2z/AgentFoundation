"""
Tabbed panel component for switching between chat and log debugging views.
"""
from typing import Any, Dict, List, Optional
from dash import html, dcc, clientside_callback, ClientsideFunction
from dash.dependencies import Input, Output, State

from science_modeling_tools.ui.dash_interactive.components.base import BaseComponent
from science_modeling_tools.ui.dash_interactive.components.chat_window import ChatWindow
from science_modeling_tools.ui.dash_interactive.components.log_graph import LogGraphVisualization
from science_modeling_tools.ui.dash_interactive.components.log_details import LogDetailsPanel


class TabbedPanel(BaseComponent):
    """
    Component providing tabbed interface for chat and log debugging.

    This component creates two tabs:
    1. Chat Interaction: Standard chat window
    2. Log Debugging: Split view with graph visualization and log details

    Attributes:
        component_id (str): Unique identifier for this component
        chat_window (ChatWindow): Chat interaction component
        log_graph (LogGraphVisualization): Log graph visualization component
        log_details (LogDetailsPanel): Log details panel component
    """

    def __init__(
        self,
        component_id: str = "tabbed-panel",
        style: Optional[Dict[str, Any]] = None,
        custom_monitor_tabs: Optional[List[Dict[str, Any]]] = None,
        custom_main_tabs: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the tabbed panel component.

        Args:
            component_id: Unique identifier for this component
            style: Optional CSS style overrides
            custom_monitor_tabs: List of custom tabs to add to monitor panel.
                Each tab is a dict with:
                - 'id': str (tab identifier, e.g., 'settings')
                - 'label': str (button label, e.g., 'Settings')
                - 'content': dash component (tab content)
            custom_main_tabs: List of custom tabs to add to main panel.
                Each tab is a dict with:
                - 'id': str (tab identifier, e.g., 'action-tester')
                - 'label': str (button label, e.g., 'Action Tester')
                - 'content': dash component (tab content)
        """
        super().__init__(component_id, style)

        # Store custom monitor tabs
        self.custom_monitor_tabs = custom_monitor_tabs or []
        # Store custom main tabs
        self.custom_main_tabs = custom_main_tabs or []

        # Create child components
        self.chat_window = ChatWindow(
            component_id=f"{component_id}-chat-window"
        )
        self.log_graph = LogGraphVisualization(
            component_id=f"{component_id}-log-graph"
        )
        self.log_details = LogDetailsPanel(
            component_id=f"{component_id}-log-details"
        )

    def _get_default_style(self) -> Dict[str, Any]:
        """Get default styling for the tabbed panel."""
        return {
            'flex': '1',
            'height': '100vh',
            'backgroundColor': '#343541',
            'display': 'flex',
            'flexDirection': 'column'
        }

    def layout(self) -> html.Div:
        """
        Generate the tabbed panel layout.

        Returns:
            Dash Div containing tabs and tab content
        """
        return html.Div(
            id=self.get_id(),
            children=[
                # Tab selection buttons
                self._create_tab_buttons(),

                # Tab content area
                html.Div(
                    id=self.get_id('content'),
                    children=[
                        # Chat tab (default visible)
                        html.Div(
                            id=self.get_id('chat-tab'),
                            children=[self.chat_window.layout()],
                            style={'display': 'block', 'height': '100%'}
                        ),
                        # Log debugging tab (hidden by default)
                        html.Div(
                            id=self.get_id('log-debug-tab'),
                            children=[self._create_log_debug_layout()],
                            style={'display': 'none', 'height': '100%'}
                        )
                    ] + [
                        # Custom main tab content (hidden by default)
                        html.Div(
                            id=self.get_id(f"{tab['id']}-tab"),
                            children=tab['content'],
                            style={'display': 'none', 'height': '100%'}
                        )
                        for tab in self.custom_main_tabs
                    ],
                    style={
                        'flex': '1',
                        'height': 'calc(100vh - 50px)',
                        'overflow': 'hidden'
                    }
                ),
                # Floating Monitor Panel (bottom-right, draggable, tabbed) - visible on all tabs
                html.Div(
                    id='main-panel-log-graph-monitor-panel',
                    children=[
                        html.Div(
                            children=[
                                # Drag handle and tabs
                                html.Div(
                                    children=[
                                        html.Div(
                                            id='main-panel-log-graph-monitor-drag-handle',
                                            children="📊 Monitor",
                                            style={
                                                'fontWeight': '600',
                                                'fontSize': '11px',
                                                'color': '#ECECF1',
                                                'cursor': 'move',
                                                'userSelect': 'none',
                                                'marginBottom': '6px'
                                            }
                                        ),
                                        # Tab buttons
                                        html.Div(
                                            children=[
                                                html.Button(
                                                    'Logs',
                                                    id='main-panel-log-graph-monitor-tab-logs-btn',
                                                    n_clicks=0,
                                                    style={
                                                        'flex': '1',
                                                        'padding': '4px 6px',
                                                        'backgroundColor': '#19C37D',
                                                        'color': '#FFFFFF',
                                                        'border': 'none',
                                                        'borderRadius': '3px 0 0 0',
                                                        'cursor': 'pointer',
                                                        'fontSize': '9px',
                                                        'fontWeight': '500',
                                                        'transition': 'all 0.2s'
                                                    }
                                                ),
                                                html.Button(
                                                    'Responses',
                                                    id='main-panel-log-graph-monitor-tab-responses-btn',
                                                    n_clicks=0,
                                                    style={
                                                        'flex': '1',
                                                        'padding': '4px 6px',
                                                        'backgroundColor': '#4A4A5A',
                                                        'color': '#8E8EA0',
                                                        'border': 'none',
                                                        'borderRadius': '0' if self.custom_monitor_tabs else '0 3px 3px 0',
                                                        'cursor': 'pointer',
                                                        'fontSize': '9px',
                                                        'fontWeight': '500',
                                                        'transition': 'all 0.2s'
                                                    }
                                                )
                                            ] + [
                                                # Add custom monitor tab buttons
                                                html.Button(
                                                    tab['label'],
                                                    id=f"main-panel-log-graph-monitor-tab-{tab['id']}-btn",
                                                    n_clicks=0,
                                                    style={
                                                        'flex': '1',
                                                        'padding': '4px 6px',
                                                        'backgroundColor': '#4A4A5A',
                                                        'color': '#8E8EA0',
                                                        'border': 'none',
                                                        'borderRadius': '0 3px 3px 0' if i == len(self.custom_monitor_tabs) - 1 else '0',
                                                        'cursor': 'pointer',
                                                        'fontSize': '9px',
                                                        'fontWeight': '500',
                                                        'transition': 'all 0.2s'
                                                    }
                                                )
                                                for i, tab in enumerate(self.custom_monitor_tabs)
                                            ] + [
                                            ],
                                            style={
                                                'display': 'flex',
                                                'marginBottom': '8px',
                                                'gap': '2px'
                                            }
                                        )
                                    ],
                                    style={
                                        'borderBottom': '1px solid rgba(255,255,255,0.1)',
                                        'paddingBottom': '6px',
                                        'marginBottom': '8px'
                                    }
                                ),

                                # Logs Tab Content
                                html.Div(
                                    id='main-panel-log-graph-monitor-logs-tab',
                                    children=[
                                        html.Div(
                                            id='main-panel-log-graph-monitor-status',
                                            children='Monitoring...',
                                            style={
                                                'fontSize': '11px',
                                                'color': '#8E8EA0',
                                                'marginBottom': '6px',
                                                'fontFamily': 'monospace'
                                            }
                                        ),
                                        html.Div(
                                            id='main-panel-log-graph-monitor-stats',
                                            children='No data',
                                            style={
                                                'fontSize': '11px',
                                                'color': '#8E8EA0',
                                                'marginBottom': '10px',
                                                'fontFamily': 'monospace'
                                            }
                                        ),
                                        html.Div([
                                            html.Div([
                                                html.Span("Monitor Log", style={'fontSize': '10px', 'color': '#8E8EA0', 'fontWeight': '500'}),
                                                html.A(
                                                    "hide",
                                                    id='main-panel-log-graph-monitor-messages-toggle',
                                                    n_clicks=0,
                                                    style={
                                                        'fontSize': '9px',
                                                        'color': '#19C37D',
                                                        'marginLeft': '8px',
                                                        'cursor': 'pointer',
                                                        'textDecoration': 'underline'
                                                    }
                                                )
                                            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'}),
                                            html.Div(
                                                id='main-panel-log-graph-monitor-messages',
                                                children='Waiting for messages...',
                                                style={
                                                    'fontSize': '9px',
                                                    'color': '#6E6E80',
                                                    'fontFamily': 'monospace',
                                                    'maxHeight': '100px',
                                                    'overflowY': 'auto',
                                                    'backgroundColor': 'rgba(0, 0, 0, 0.2)',
                                                    'padding': '6px',
                                                    'borderRadius': '3px',
                                                    'marginBottom': '10px',
                                                    'lineHeight': '1.3'
                                                }
                                            )
                                        ]),
                                        html.Button(
                                            id='main-panel-log-graph-refresh-btn',
                                            children='🔄 Refresh Graph',
                                            n_clicks=0,
                                            style={
                                                'width': '100%',
                                                'padding': '8px 12px',
                                                'backgroundColor': '#4A4A5A',
                                                'color': '#8E8EA0',
                                                'border': 'none',
                                                'borderRadius': '4px',
                                                'cursor': 'not-allowed',
                                                'fontSize': '12px',
                                                'fontWeight': '500',
                                                'transition': 'all 0.2s'
                                            }
                                        )
                                    ],
                                    style={'display': 'block'}
                                ),

                                # Responses Tab Content (split left-right)
                                html.Div(
                                    id='main-panel-log-graph-monitor-responses-tab',
                                    children=[
                                        html.Div(
                                            id='main-panel-log-graph-response-count',
                                            children='Total: 0',
                                            style={
                                                'fontSize': '10px',
                                                'color': '#8E8EA0',
                                                'marginBottom': '8px',
                                                'fontFamily': 'monospace',
                                                'fontWeight': '500'
                                            }
                                        ),
                                        # Split container
                                        html.Div(
                                            children=[
                                                # Left: Response list (selectable)
                                                html.Div(
                                                    id='main-panel-log-graph-response-list',
                                                    children='No responses',
                                                    style={
                                                        'flex': '0 0 120px',
                                                        'fontSize': '9px',
                                                        'color': '#6E6E80',
                                                        'fontFamily': 'monospace',
                                                        'maxHeight': '200px',
                                                        'overflowY': 'auto',
                                                        'backgroundColor': 'rgba(0, 0, 0, 0.2)',
                                                        'padding': '4px',
                                                        'borderRadius': '3px',
                                                        'lineHeight': '1.2',
                                                        'marginRight': '6px'
                                                    }
                                                ),
                                                # Right: Response details
                                                html.Div(
                                                    id='main-panel-log-graph-response-details',
                                                    children='Select a response',
                                                    style={
                                                        'flex': '1',
                                                        'fontSize': '9px',
                                                        'color': '#ECECF1',
                                                        'fontFamily': 'monospace',
                                                        'maxHeight': '200px',
                                                        'overflowY': 'auto',
                                                        'backgroundColor': 'rgba(0, 0, 0, 0.3)',
                                                        'padding': '6px',
                                                        'borderRadius': '3px',
                                                        'lineHeight': '1.4',
                                                        'whiteSpace': 'pre-wrap',
                                                        'wordBreak': 'break-word'
                                                    }
                                                )
                                            ],
                                            style={
                                                'display': 'flex',
                                                'gap': '0'
                                            }
                                        )
                                    ],
                                    style={'display': 'none'}
                                )
                            ] + [
                                # Custom monitor tab contents
                                html.Div(
                                    id=f"main-panel-log-graph-monitor-{tab['id']}-tab",
                                    children=tab['content'],
                                    style={'display': 'none'}
                                )
                                for tab in self.custom_monitor_tabs
                            ] + [
                            ],
                            style={
                                'padding': '12px',
                                'backgroundColor': 'rgba(44, 44, 44, 0.95)',
                                'borderRadius': '6px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.4)',
                                'border': '1px solid rgba(255,255,255,0.1)'
                            }
                        )
                    ],
                    style={
                        'position': 'fixed',
                        'bottom': '20px',
                        'right': '20px',
                        'width': '280px',
                        'zIndex': '3000',
                        'pointerEvents': 'auto'
                    }
                )
            ],
            style=self.style
        )

    def _create_tab_buttons(self) -> html.Div:
        """Create tab selection buttons."""
        # Base tabs
        buttons = [
            html.Button(
                'Chat Interaction',
                id=self.get_id('chat-btn'),
                n_clicks=0,
                style={
                    'padding': '12px 24px',
                    'backgroundColor': '#19C37D',
                    'color': '#ECECF1',
                    'border': 'none',
                    'borderBottom': '2px solid #19C37D',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'flex': '1'
                }
            ),
            html.Button(
                'Log Debugging',
                id=self.get_id('log-btn'),
                n_clicks=0,
                style={
                    'padding': '12px 24px',
                    'backgroundColor': '#40414F',
                    'color': '#8E8EA0',
                    'border': 'none',
                    'borderBottom': '2px solid transparent',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'flex': '1'
                }
            )
        ]
        
        # Add custom main tab buttons
        for tab in self.custom_main_tabs:
            buttons.append(
                html.Button(
                    tab['label'],
                    id=self.get_id(f"{tab['id']}-btn"),
                    n_clicks=0,
                    style={
                        'padding': '12px 24px',
                        'backgroundColor': '#40414F',
                        'color': '#8E8EA0',
                        'border': 'none',
                        'borderBottom': '2px solid transparent',
                        'cursor': 'pointer',
                        'fontSize': '14px',
                        'fontWeight': '500',
                        'flex': '1'
                    }
                )
            )
        
        return html.Div(
            children=buttons,
            style={
                'display': 'flex',
                'backgroundColor': '#40414F',
                'borderBottom': '1px solid #565869'
            }
        )

    def _create_log_debug_layout(self) -> html.Div:
        """
        Create the log debugging layout with adjustable horizontal split.

        Returns:
            Div with graph view on top and log details below, with draggable divider
        """
        return html.Div(
            children=[
                # Upper half - Log graph visualization
                html.Div(
                    id=self.get_id('log-graph-pane'),
                    children=[self.log_graph.layout()],
                    style={
                        'height': '65%',
                        'minHeight': '150px',
                        'overflow': 'auto',
                        'position': 'relative',
                        'flexShrink': '0'
                    }
                ),

                # Draggable divider
                html.Div(
                    id=self.get_id('resize-divider'),
                    style={
                        'height': '8px',
                        'backgroundColor': '#19C37D',
                        'cursor': 'row-resize',
                        'flexShrink': '0',
                        'position': 'relative',
                        'zIndex': '10',
                        'transition': 'background-color 0.2s'
                    }
                ),

                # Lower half - Log details
                html.Div(
                    id=self.get_id('log-details-pane'),
                    children=[self.log_details.layout()],
                    style={
                        'flex': '1',
                        'overflow': 'auto',
                        'minHeight': '100px'
                    }
                )
            ],
            style={
                'height': '100%',
                'display': 'flex',
                'flexDirection': 'column'
            }
        )

    def get_callback_inputs(self) -> List[Input]:
        """Get list of callback inputs."""
        return [
            Input(self.get_id('chat-btn'), 'n_clicks'),
            Input(self.get_id('log-btn'), 'n_clicks'),
            Input(self.get_id('execute-btn'), 'n_clicks')
        ]

    def get_callback_outputs(self) -> List[Output]:
        """Get list of callback outputs."""
        return [
            Output(self.get_id('chat-tab'), 'style'),
            Output(self.get_id('log-debug-tab'), 'style'),
            Output(self.get_id('chat-btn'), 'style'),
            Output(self.get_id('log-btn'), 'style')
        ]

    def get_callback_states(self) -> List[State]:
        """Get list of callback states."""
        return [
            State(self.get_id('graph-type-dropdown'), 'value')
        ]

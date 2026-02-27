"""
Dash application with both chat interaction and log debugging capabilities.
"""
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from threading import Thread
from queue import Queue, Empty
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from agent_foundation.ui.dash_interactive.dash_interactive_app import DashInteractiveApp
from agent_foundation.ui.dash_interactive.components.tabbed_panel import TabbedPanel
from agent_foundation.ui.dash_interactive.utils.dummy_graph_executor import execute_and_collect_logs
from agent_foundation.ui.queue_interactive import QueueInteractive


class DashInteractiveAppWithLogs(DashInteractiveApp):
    """
    Extended Dash application with chat interface and log debugging.

    This application provides:
    - Left sidebar: Chat history and settings
    - Right panel: Tabbed interface with:
      1. Chat Interaction tab
      2. Log Debugging tab (graph view + log details)

    Attributes:
        app (dash.Dash): The Dash application instance
        chat_history (ChatHistoryList): Left sidebar component
        tabbed_panel (TabbedPanel): Right panel with tabs
        sessions (List[Dict]): List of chat sessions
        current_session_id (str): ID of currently active session
        message_handler (Callable): Callback for processing user messages
        log_collector: Current log collector instance
    """

    def __init__(
        self,
        title: str = "Interactive Debugger with Logs",
        port: int = 8050,
        debug: bool = True,
        message_handler: Optional[Callable[[str], str]] = None,
        queue_service=None,  # Optional StorageBasedQueueService for web agent integration
        custom_monitor_tabs: list = None,  # Optional list of custom monitor tabs
        custom_main_tabs: list = None  # Optional list of custom main tabs
    ):
        """
        Initialize the Dash interactive app with log debugging.

        Args:
            title: Application title
            port: Port number to run the server on
            debug: Whether to run in debug mode
            message_handler: Optional callback function to handle user messages
            queue_service: Optional queue service for communicating with web agent service
            custom_monitor_tabs: Optional list of custom monitor tab dicts with 'id', 'label', 'content'
            custom_main_tabs: Optional list of custom main tab dicts with 'id', 'label', 'content'
        """
        # Log debugging state (before super().__init__)
        self.log_collector = None

        # Agent thread management
        self.session_agents = {}  # session_id -> Agent instance
        self.session_threads = {}  # session_id -> Thread
        self.session_interactives = {}  # session_id -> QueueInteractive
        self.session_reasoners = {}  # session_id -> reasoner_name
        self.agent_factory = None  # Callable that creates new agent

        # Web agent service integration
        self.queue_service = queue_service  # StorageBasedQueueService instance
        self.agent_status_messages = {}  # session_id -> list of status messages

        # Custom monitor tabs - use provided tabs or empty list
        self.custom_monitor_tabs = custom_monitor_tabs if custom_monitor_tabs is not None else []
        
        # Custom main tabs - use provided tabs or empty list
        self.custom_main_tabs = custom_main_tabs if custom_main_tabs is not None else []

        # Create tabbed_panel BEFORE calling super().__init__
        # (because parent's __init__ will call _create_layout which needs tabbed_panel)
        self.tabbed_panel = TabbedPanel(
            component_id="main-panel",
            custom_monitor_tabs=self.custom_monitor_tabs,
            custom_main_tabs=self.custom_main_tabs
        )

        # Call parent constructor (will create chat_history, chat_window, layout, and callbacks)
        super().__init__(title=title, port=port, debug=debug, message_handler=message_handler)

        # Add custom JavaScript for Ctrl+Enter keyboard shortcut and split pane resizer
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    /* Typing indicator animation - wave effect like ChatGPT/Claude */
                    @keyframes wave {
                        0%, 60%, 100% {
                            transform: translateY(0);
                            opacity: 0.7;
                        }
                        30% {
                            transform: translateY(-10px);
                            opacity: 1;
                        }
                    }
                </style>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        // Ctrl+Enter keyboard shortcut for sending messages
                        document.addEventListener('keydown', function(e) {
                            var activeElement = document.activeElement;
                            if (activeElement && activeElement.id === 'main-panel-chat-window-input') {
                                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                                    e.preventDefault();
                                    var button = document.getElementById('main-panel-chat-window-send-btn');
                                    if (button) {
                                        button.click();
                                    }
                                }
                            }
                        });

                        // Split pane resizer for Log Debugging tab
                        (function() {
                            var isResizing = false;
                            var startY = 0;
                            var startTopHeight = 0;
                            var container = null;
                            var topPane = null;
                            var divider = null;
                            var bottomPane = null;
                            var initialized = false;

                            // Global mousemove handler
                            function handleMouseMove(e) {
                                if (!isResizing) return;

                                var deltaY = e.clientY - startY;
                                var newTopHeight = startTopHeight + deltaY;
                                
                                // Get container dimensions
                                var containerHeight = container.offsetHeight;
                                
                                // Check if first child is control panel or the top pane itself
                                var controlPanelHeight = 0;
                                var firstChild = container.children[0];
                                if (firstChild && firstChild.id !== topPane.id) {
                                    controlPanelHeight = firstChild.offsetHeight;
                                }
                                
                                var dividerHeight = divider.offsetHeight;
                                var availableHeight = containerHeight - controlPanelHeight - dividerHeight;

                                var minHeight = 150;
                                var maxHeight = availableHeight - 150; // Leave at least 150px for bottom

                                if (newTopHeight >= minHeight && newTopHeight <= maxHeight) {
                                    topPane.style.height = newTopHeight + 'px';
                                    topPane.style.flexShrink = '0';
                                }

                                e.preventDefault();
                            }

                            // Global mouseup handler
                            function handleMouseUp() {
                                if (isResizing) {
                                    isResizing = false;
                                    document.body.style.cursor = 'default';
                                    topPane.style.userSelect = 'auto';
                                    bottomPane.style.userSelect = 'auto';
                                    divider.style.backgroundColor = '#19C37D';
                                    console.log('[Split Resizer] Drag ended, final height:', topPane.style.height);
                                }
                            }

                            function initSplitResize() {
                                topPane = document.getElementById('main-panel-log-graph-pane');
                                divider = document.getElementById('main-panel-resize-divider');
                                bottomPane = document.getElementById('main-panel-log-details-pane');
                                
                                if (!topPane || !divider || !bottomPane) {
                                    console.log('[Split Resizer] Elements not found, retrying...');
                                    setTimeout(initSplitResize, 200);
                                    return;
                                }

                                container = topPane.parentElement;

                                // Only add event listeners once (must be done before height calculation)
                                if (!initialized) {
                                    // Mousedown on divider
                                    divider.addEventListener('mousedown', function(e) {
                                        isResizing = true;
                                        startY = e.clientY;
                                        startTopHeight = topPane.offsetHeight;
                                        e.preventDefault();
                                        document.body.style.cursor = 'row-resize';
                                        topPane.style.userSelect = 'none';
                                        bottomPane.style.userSelect = 'none';
                                        divider.style.backgroundColor = '#10A37F';
                                        console.log('[Split Resizer] Drag started at Y:', startY, 'height:', startTopHeight);
                                    });

                                    // Hover effect on divider
                                    divider.addEventListener('mouseenter', function() {
                                        if (!isResizing) {
                                            divider.style.backgroundColor = '#10A37F';
                                        }
                                    });

                                    divider.addEventListener('mouseleave', function() {
                                        if (!isResizing) {
                                            divider.style.backgroundColor = '#19C37D';
                                        }
                                    });

                                    // Add global listeners
                                    document.addEventListener('mousemove', handleMouseMove);
                                    document.addEventListener('mouseup', handleMouseUp);

                                    initialized = true;
                                    console.log('[Split Resizer] Event listeners attached');
                                }

                                // Convert percentage height to pixel height
                                // Always recalculate if container height is available
                                var containerHeight = container.offsetHeight;
                                
                                if (containerHeight > 0 && (!topPane.style.height || topPane.style.height.includes('%') || !topPane.style.height.includes('px') || topPane.offsetHeight === 0)) {
                                    // Check if first child is control panel or the top pane itself
                                    var controlPanelHeight = 0;
                                    var firstChild = container.children[0];
                                    if (firstChild && firstChild.id !== topPane.id) {
                                        controlPanelHeight = firstChild.offsetHeight;
                                    }
                                    
                                    var dividerHeight = divider.offsetHeight;
                                    var availableHeight = containerHeight - controlPanelHeight - dividerHeight;
                                    var topHeight = Math.floor(availableHeight * 0.65); // 65% for graph, 35% for logs
                                    topPane.style.height = topHeight + 'px';
                                    topPane.style.flexShrink = '0';
                                    console.log('[Split Resizer] Set initial 65/35 split - Top:', topHeight + 'px', 'Bottom:', (availableHeight - topHeight) + 'px', 'Divider:', dividerHeight + 'px', 'Container:', containerHeight + 'px');
                                } else if (containerHeight === 0) {
                                    console.log('[Split Resizer] Container height is 0, tab might be hidden. Will retry when tab is shown.');
                                }

                                console.log('[Split Resizer] Initialized - topPane height:', topPane.style.height);
                            }

                            // Initial setup
                            setTimeout(initSplitResize, 100);

                            // Reinitialize when switching to Log Debugging tab
                            document.addEventListener('click', function(e) {
                                if (e.target && e.target.id === 'main-panel-log-btn') {
                                    // Force recalculation when tab becomes visible
                                    setTimeout(function() {
                                        if (topPane) {
                                            topPane.style.height = '65%'; // Reset to percentage to trigger recalculation
                                        }
                                        initSplitResize();
                                    }, 300);
                                }
                            });
                        })();
                    });
                </script>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

    def _create_layout(self) -> html.Div:
        """Create the main application layout with extended features."""
        return html.Div(
            children=[
                # Store components for maintaining state (from parent + extensions)
                dcc.Store(id='sessions-store', data=[]),
                dcc.Store(id='current-session-store', data=None),
                dcc.Store(id='messages-store', data={}),
                dcc.Store(id='log-data-store', data=None),
                dcc.Store(id='page-visibility-store', data={'visible': True, 'timestamp': 0}),

                # Interval components for auto-refresh
                dcc.Interval(
                    id='response-poll-interval',
                    interval=1000,  # Poll every 1 second for agent responses
                    n_intervals=0
                ),
                dcc.Interval(
                    id='log-refresh-interval',
                    interval=3000,  # Refresh logs every 3 seconds
                    n_intervals=0
                ),
                dcc.Interval(
                    id='visibility-check-interval',
                    interval=2000,  # Check visibility every 2 seconds
                    n_intervals=0
                ),
                dcc.Interval(
                    id='agent-status-poll-interval',
                    interval=1000,  # Poll for agent status updates every 1 second
                    n_intervals=0
                ),

                # Main container with flexbox layout
                html.Div(
                    children=[
                        # Left sidebar - Chat history (from parent)
                        self.chat_history.layout(),

                        # Right panel - Tabbed interface (overridden)
                        self.tabbed_panel.layout()
                    ],
                    style={
                        'display': 'flex',
                        'height': '100vh',
                        'width': '100vw',
                        'overflow': 'hidden',
                        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
                    }
                )
            ],
            style={
                'margin': '0',
                'padding': '0'
            }
        )

    def _register_polling_callback(self):
        """
        Register polling callback for agent responses.

        This is extracted as a separate method so subclasses can override
        the polling behavior (e.g., for queue-based communication).

        Triggers on both regular interval and page visibility changes to ensure
        responses are caught even when tab is inactive.
        """
        # Callback to poll response queue and add new messages to UI
        @self.app.callback(
            Output('messages-store', 'data', allow_duplicate=True),
            [
                Input('response-poll-interval', 'n_intervals'),
                Input('page-visibility-store', 'data')
            ],
            [
                State('current-session-store', 'data'),
                State('messages-store', 'data')
            ],
            prevent_initial_call=True
        )
        def poll_agent_responses(n_intervals, visibility_data, session_id, messages_store):
            """Poll response queue and add new messages to UI."""
            if not session_id:
                return messages_store

            # Check if using queue service mode (web agent service)
            if self.queue_service:
                # Poll from shared agent_response queue
                new_responses = []
                try:
                    while True:
                        response_data = self.queue_service.get('agent_response', blocking=False, timeout=0)
                        if response_data is None:
                            break

                        # Check if response is for current session
                        if isinstance(response_data, dict):
                            response_session_id = response_data.get('session_id')
                            if response_session_id == session_id:
                                response_content = response_data.get('response', '')
                                new_responses.append(response_content)
                        elif isinstance(response_data, str):
                            # Backward compatibility: plain string response
                            new_responses.append(response_data)
                except Exception as e:
                    print(f"[Dash UI] Error polling agent_response queue: {e}")

                # Add new responses to messages
                if new_responses and session_id in messages_store:
                    for response in new_responses:
                        if response == "[AGENT_COMPLETED]":
                            # Optional: Show completion indicator
                            assistant_msg = {
                                'role': 'system',
                                'content': 'Agent task completed.',
                                'timestamp': datetime.now().strftime('%H:%M:%S')
                            }
                            messages_store[session_id].append(assistant_msg)
                            continue

                        assistant_msg = {
                            'role': 'assistant',
                            'content': response,
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        }
                        messages_store[session_id].append(assistant_msg)

                return messages_store

            # Local agent mode - poll from per-session queue
            elif session_id in self.session_interactives:
                interactive = self.session_interactives[session_id]

                # Check for new responses (non-blocking)
                new_responses = []
                try:
                    while True:
                        response = interactive.response_queue.get_nowait()
                        new_responses.append(response)
                except Empty:
                    pass

                # Add new responses to messages
                if new_responses and session_id in messages_store:
                    for response in new_responses:
                        if response == "[AGENT_COMPLETED]":
                            # Optional: Show completion indicator
                            assistant_msg = {
                                'role': 'system',
                                'content': 'Agent task completed.',
                                'timestamp': datetime.now().strftime('%H:%M:%S')
                            }
                            messages_store[session_id].append(assistant_msg)
                            continue

                        assistant_msg = {
                            'role': 'assistant',
                            'content': response,
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        }
                        messages_store[session_id].append(assistant_msg)

                return messages_store

            return messages_store

    def _register_callbacks(self):
        """Register all Dash callbacks for interactivity (extends parent callbacks)."""

        # === Page visibility detection (clientside) ===
        # This detects when the browser tab becomes visible and triggers a poll
        self.app.clientside_callback(
            """
            function(n_intervals, currentData) {
                // Use the Page Visibility API to detect if page is visible
                const isVisible = !document.hidden;

                // Only update if visibility changed
                if (currentData && currentData.visible === isVisible) {
                    // No change, return current data without updating timestamp
                    return window.dash_clientside.no_update;
                }

                // Visibility changed - update with new timestamp
                const timestamp = Date.now();
                return {visible: isVisible, timestamp: timestamp};
            }
            """,
            Output('page-visibility-store', 'data'),
            Input('visibility-check-interval', 'n_intervals'),
            State('page-visibility-store', 'data')
        )

        # === Register parent session callbacks (reusable) ===
        # This handles: create_new_session, update_history_display, select_session
        self._register_session_callbacks()

        # === Override message callbacks (different component IDs for tabbed interface) ===
        # Callback for loading messages of current session
        @self.app.callback(
            Output('main-panel-chat-window-messages', 'children'),
            [
                Input('current-session-store', 'data'),
                Input('messages-store', 'data')
            ],
            prevent_initial_call=False
        )
        def load_session_messages(session_id, messages_store):
            if session_id and session_id in messages_store:
                messages = messages_store[session_id]
                return self.tabbed_panel.chat_window.update_messages(messages)
            return self.tabbed_panel.chat_window.update_messages([])

        # Callback for sending messages
        @self.app.callback(
            [
                Output('messages-store', 'data', allow_duplicate=True),
                Output('main-panel-chat-window-input', 'value'),
                Output('sessions-store', 'data', allow_duplicate=True),
                Output('current-session-store', 'data', allow_duplicate=True),
                Output('main-panel-chat-tab', 'style', allow_duplicate=True),
                Output('main-panel-log-debug-tab', 'style', allow_duplicate=True),
                Output('main-panel-chat-btn', 'style', allow_duplicate=True),
                Output('main-panel-log-btn', 'style', allow_duplicate=True)
            ],
            [
                Input('main-panel-chat-window-send-btn', 'n_clicks')
            ],
            [
                State('main-panel-chat-window-input', 'value'),
                State('current-session-store', 'data'),
                State('messages-store', 'data'),
                State('sessions-store', 'data')
            ],
            prevent_initial_call=True
        )
        def send_message(send_clicks, message_text, session_id, messages_store, sessions):
            # Styles for showing chat tab (active)
            chat_tab_active_styles = (
                {'display': 'block', 'height': '100%'},  # chat tab visible
                {'display': 'none', 'height': '100%'},   # log tab hidden
                {'padding': '12px 24px', 'backgroundColor': '#19C37D', 'color': '#ECECF1',
                 'border': 'none', 'borderBottom': '2px solid #19C37D', 'cursor': 'pointer',
                 'fontSize': '14px', 'fontWeight': '500', 'flex': '1'},  # chat button active
                {'padding': '12px 24px', 'backgroundColor': '#40414F', 'color': '#8E8EA0',
                 'border': 'none', 'borderBottom': '2px solid #40414F', 'cursor': 'pointer',
                 'fontSize': '14px', 'fontWeight': '500', 'flex': '1'}   # log button inactive
            )

            if not message_text or not message_text.strip():
                return messages_store, '', sessions, session_id, *chat_tab_active_styles

            # Create session if none exists
            if not session_id:
                session_id = f"session_1_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                new_session = {
                    'id': session_id,
                    'title': 'New Chat',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'active': True
                }
                sessions.append(new_session)
                messages_store[session_id] = []

            # Add user message
            timestamp = datetime.now().strftime('%H:%M:%S')
            user_msg = {
                'role': 'user',
                'content': message_text.strip(),
                'timestamp': timestamp
            }

            if session_id not in messages_store:
                messages_store[session_id] = []

            messages_store[session_id].append(user_msg)

            # Check if using queue service (web agent service mode)
            if self.queue_service:
                # Web agent service mode - delegate to message handler
                # The message handler is responsible for sending control messages and user messages
                try:
                    # Try calling with all parameters (newest signature)
                    import inspect
                    sig = inspect.signature(self.message_handler)
                    param_count = len(sig.parameters)

                    if param_count >= 3:
                        # Newest signature: handler(message, session_id, all_session_ids)
                        all_session_ids = list(messages_store.keys())
                        response = self.message_handler(message_text.strip(), session_id, all_session_ids)
                    elif param_count >= 2:
                        # Old signature: handler(message, session_id)
                        response = self.message_handler(message_text.strip(), session_id)
                    else:
                        # Oldest signature: handler(message)
                        response = self.message_handler(message_text.strip())
                except Exception as e:
                    response = f"Error: {str(e)}"

                # Don't add immediate response - polling callback will handle it

            # Check if using agent factory (local agent mode)
            elif self.agent_factory:
                # Local agent mode - start agent thread if needed
                if session_id not in self.session_threads:
                    self._start_agent_for_session(session_id)

                # Put user message in agent's input queue
                interactive = self.session_interactives[session_id]
                interactive.input_queue.put(message_text.strip())

                # Don't add immediate response - polling callback will handle it
            else:
                # Traditional message handler mode
                try:
                    # Try calling with all parameters (newest signature)
                    import inspect
                    sig = inspect.signature(self.message_handler)
                    param_count = len(sig.parameters)

                    if param_count >= 3:
                        # Newest signature: handler(message, session_id, all_session_ids)
                        all_session_ids = list(messages_store.keys())
                        response = self.message_handler(message_text.strip(), session_id, all_session_ids)
                    elif param_count >= 2:
                        # Old signature: handler(message, session_id)
                        response = self.message_handler(message_text.strip(), session_id)
                    else:
                        # Oldest signature: handler(message)
                        response = self.message_handler(message_text.strip())
                except Exception as e:
                    response = f"Error: {str(e)}"

                # Add assistant response
                assistant_msg = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                messages_store[session_id].append(assistant_msg)

            # Update session title with first message if it's "New Chat"
            for session in sessions:
                if session['id'] == session_id and session['title'].startswith('New Chat'):
                    words = message_text.split()[:5]
                    session['title'] = ' '.join(words) + ('...' if len(words) >= 5 else '')
                    break

            # Mark this session as active in the sessions list
            for session in sessions:
                session['active'] = (session['id'] == session_id)

            # Return with session_id and chat tab active styles to automatically select session and switch to Chat Interaction tab
            return messages_store, '', sessions, session_id, *chat_tab_active_styles

        # === Tab switching callbacks ===

        @self.app.callback(
            [
                Output('main-panel-chat-tab', 'style'),
                Output('main-panel-log-debug-tab', 'style'),
                Output('main-panel-chat-btn', 'style'),
                Output('main-panel-log-btn', 'style')
            ],
            [
                Input('main-panel-chat-btn', 'n_clicks'),
                Input('main-panel-log-btn', 'n_clicks')
            ],
            prevent_initial_call=False
        )
        def switch_tabs(chat_clicks, log_clicks):
            ctx = dash.callback_context

            # Default to chat tab
            if not ctx.triggered:
                return (
                    {'display': 'block', 'height': '100%'},
                    {'display': 'none', 'height': '100%'},
                    {
                        'padding': '12px 24px', 'backgroundColor': '#19C37D',
                        'color': '#ECECF1', 'border': 'none',
                        'borderBottom': '2px solid #19C37D', 'cursor': 'pointer',
                        'fontSize': '14px', 'fontWeight': '500', 'flex': '1'
                    },
                    {
                        'padding': '12px 24px', 'backgroundColor': '#40414F',
                        'color': '#8E8EA0', 'border': 'none',
                        'borderBottom': '2px solid transparent', 'cursor': 'pointer',
                        'fontSize': '14px', 'fontWeight': '500', 'flex': '1'
                    }
                )

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'main-panel-log-btn':
                # Show log tab
                return (
                    {'display': 'none', 'height': '100%'},
                    {'display': 'block', 'height': '100%'},
                    {
                        'padding': '12px 24px', 'backgroundColor': '#40414F',
                        'color': '#8E8EA0', 'border': 'none',
                        'borderBottom': '2px solid transparent', 'cursor': 'pointer',
                        'fontSize': '14px', 'fontWeight': '500', 'flex': '1'
                    },
                    {
                        'padding': '12px 24px', 'backgroundColor': '#19C37D',
                        'color': '#ECECF1', 'border': 'none',
                        'borderBottom': '2px solid #19C37D', 'cursor': 'pointer',
                        'fontSize': '14px', 'fontWeight': '500', 'flex': '1'
                    }
                )
            else:
                # Show chat tab
                return (
                    {'display': 'block', 'height': '100%'},
                    {'display': 'none', 'height': '100%'},
                    {
                        'padding': '12px 24px', 'backgroundColor': '#19C37D',
                        'color': '#ECECF1', 'border': 'none',
                        'borderBottom': '2px solid #19C37D', 'cursor': 'pointer',
                        'fontSize': '14px', 'fontWeight': '500', 'flex': '1'
                    },
                    {
                        'padding': '12px 24px', 'backgroundColor': '#40414F',
                        'color': '#8E8EA0', 'border': 'none',
                        'borderBottom': '2px solid transparent', 'cursor': 'pointer',
                        'fontSize': '14px', 'fontWeight': '500', 'flex': '1'
                    }
                )

        # === Log debugging callbacks ===

        # Update log graph visualization
        @self.app.callback(
            Output('main-panel-log-graph-graph', 'figure'),
            [
                Input('log-data-store', 'data'),
                Input('main-panel-log-graph-label-mode', 'value')
            ],
            prevent_initial_call=False
        )
        def update_log_graph(log_data, label_mode):
            # Default to 'name' if not specified
            if label_mode is None:
                label_mode = 'name'

            if log_data:
                # Check if this is graph data (WorkGraph structure) or hierarchy (tree structure)
                if 'graph_data' in log_data:
                    # Handle WorkGraph DAG structure
                    return self.tabbed_panel.log_graph.create_figure_from_graph(log_data['graph_data'], label_mode)
                elif 'hierarchy' in log_data:
                    # Handle standard tree hierarchy
                    return self.tabbed_panel.log_graph.create_figure(log_data['hierarchy'], label_mode)
            return self.tabbed_panel.log_graph.create_figure([])

        # Update log details when graph node is clicked
        @self.app.callback(
            [
                Output('main-panel-log-details-logs-container', 'children'),
                Output('main-panel-log-details-group-info', 'children'),
                Output('main-panel-log-details-pagination-controls', 'children'),
                Output('main-panel-log-details-pagination-state', 'data')
            ],
            [Input('main-panel-log-graph-graph', 'clickData')],
            [State('log-data-store', 'data')],
            prevent_initial_call=False
        )
        def update_log_details(click_data, log_data):
            if not click_data or not log_data:
                logs, group_info, pagination = self.tabbed_panel.log_details.update_logs([], "", log_group_id=None)
                return logs, group_info, pagination, {'page': 0, 'show_all': False}

            # Extract clicked node's log group ID
            point = click_data['points'][0]
            log_group_id = point.get('customdata', [''])[0]

            if log_group_id and log_group_id in log_data['log_groups']:
                logs = log_data['log_groups'][log_group_id]
                group_info = f"Log Group: {log_group_id} ({len(logs)} entries)"
                # Pass log_group_id to enable caching
                rendered_logs, group_info, pagination = self.tabbed_panel.log_details.update_logs(
                    logs, group_info, page=0, show_all=False, log_group_id=log_group_id
                )
                return rendered_logs, group_info, pagination, {'page': 0, 'show_all': False}

            logs, group_info, pagination = self.tabbed_panel.log_details.update_logs([], "No logs found", log_group_id=None)
            return logs, group_info, pagination, {'page': 0, 'show_all': False}

        # Toggle between Plotly and Cytoscape rendering modes
        @self.app.callback(
            [
                Output('main-panel-log-graph-plotly-container', 'style'),
                Output('main-panel-log-graph-cytoscape-container', 'style')
            ],
            [Input('main-panel-log-graph-rendering-mode', 'value')],
            prevent_initial_call=False
        )
        def toggle_graph_rendering_mode(mode):
            if mode == 'plotly':
                return {'display': 'block', 'height': 'calc(100% - 120px)'}, {'display': 'none', 'height': 'calc(100% - 120px)'}
            else:  # cytoscape
                return {'display': 'none', 'height': 'calc(100% - 120px)'}, {'display': 'block', 'height': 'calc(100% - 120px)'}

        # Update Cytoscape graph elements
        @self.app.callback(
            Output('main-panel-log-graph-cytoscape', 'elements'),
            [
                Input('log-data-store', 'data'),
                Input('main-panel-log-graph-label-mode', 'value')
            ],
            prevent_initial_call=False
        )
        def update_cytoscape_graph(log_data, label_mode):
            # Default to 'name' if not specified
            if label_mode is None:
                label_mode = 'name'

            if log_data:
                # Check if this is graph data (WorkGraph structure) or hierarchy (tree structure)
                if 'graph_data' in log_data:
                    # Handle WorkGraph DAG structure
                    graph_data = self.tabbed_panel.log_graph._process_dag_to_graph(
                        log_data['graph_data']['nodes'],
                        log_data['graph_data']['edges'],
                        log_data['graph_data']['agent']
                    )
                    return self.tabbed_panel.log_graph.convert_to_cytoscape_elements(graph_data, label_mode)
                elif 'hierarchy' in log_data:
                    # Handle standard tree hierarchy
                    graph_data = self.tabbed_panel.log_graph.process_hierarchy_to_graph(log_data['hierarchy'])
                    return self.tabbed_panel.log_graph.convert_to_cytoscape_elements(graph_data, label_mode)
            return []

        # Handle Cytoscape node clicks
        @self.app.callback(
            [
                Output('main-panel-log-details-logs-container', 'children', allow_duplicate=True),
                Output('main-panel-log-details-group-info', 'children', allow_duplicate=True),
                Output('main-panel-log-details-pagination-controls', 'children', allow_duplicate=True),
                Output('main-panel-log-details-pagination-state', 'data', allow_duplicate=True)
            ],
            [Input('main-panel-log-graph-cytoscape', 'tapNodeData')],
            [State('log-data-store', 'data')],
            prevent_initial_call=True
        )
        def update_log_details_from_cytoscape(tap_node_data, log_data):
            if not tap_node_data or not log_data:
                logs, group_info, pagination = self.tabbed_panel.log_details.update_logs([], "", log_group_id=None)
                return logs, group_info, pagination, {'page': 0, 'show_all': False}

            # Extract clicked node's log group ID
            log_group_id = tap_node_data.get('id', '')

            if log_group_id and log_group_id in log_data.get('log_groups', {}):
                logs = log_data['log_groups'][log_group_id]
                group_info = f"Log Group: {log_group_id} ({len(logs)} entries)"
                # Pass log_group_id to enable caching
                rendered_logs, group_info, pagination = self.tabbed_panel.log_details.update_logs(
                    logs, group_info, page=0, show_all=False, log_group_id=log_group_id
                )
                return rendered_logs, group_info, pagination, {'page': 0, 'show_all': False}

            logs, group_info, pagination = self.tabbed_panel.log_details.update_logs([], "No logs found", log_group_id=None)
            return logs, group_info, pagination, {'page': 0, 'show_all': False}

        # Reset log details when switching sessions
        @self.app.callback(
            [
                Output('main-panel-log-details-logs-container', 'children', allow_duplicate=True),
                Output('main-panel-log-details-group-info', 'children', allow_duplicate=True),
                Output('main-panel-log-details-pagination-controls', 'children', allow_duplicate=True),
                Output('main-panel-log-details-pagination-state', 'data', allow_duplicate=True)
            ],
            [Input('current-session-store', 'data')],
            prevent_initial_call=True
        )
        def reset_log_details_on_session_switch(session_id):
            # Clear log details when switching sessions
            logs, group_info, pagination = self.tabbed_panel.log_details.update_logs(
                [], "Select a node in the graph to view logs", log_group_id=None
            )
            return logs, group_info, pagination, {'page': 0, 'show_all': False}

        # Handle "Load More" button clicks for pagination
        @self.app.callback(
            [
                Output('main-panel-log-details-logs-container', 'children', allow_duplicate=True),
                Output('main-panel-log-details-pagination-controls', 'children', allow_duplicate=True),
                Output('main-panel-log-details-pagination-state', 'data', allow_duplicate=True)
            ],
            [Input('main-panel-log-details-load-more-btn', 'n_clicks')],
            [
                State('main-panel-log-details-pagination-state', 'data'),
                State('log-data-store', 'data'),
                State('main-panel-log-details-group-info', 'children')
            ],
            prevent_initial_call=True
        )
        def handle_load_more(n_clicks, pagination_state, log_data, group_info_text):
            if not n_clicks or not log_data or not pagination_state:
                return dash.no_update, dash.no_update, dash.no_update

            # Extract log_group_id from group_info text (e.g., "Log Group: xxx (123 entries)")
            import re
            match = re.search(r'Log Group: ([\w\-]+)', group_info_text or '')
            if not match:
                return dash.no_update, dash.no_update, dash.no_update

            log_group_id = match.group(1)
            if log_group_id not in log_data.get('log_groups', {}):
                return dash.no_update, dash.no_update, dash.no_update

            logs = log_data['log_groups'][log_group_id]
            current_page = pagination_state.get('page', 0)
            next_page = current_page + 1

            # Update logs in component and render next page (uses cache if available)
            group_info = f"Log Group: {log_group_id} ({len(logs)} entries)"
            rendered_logs, _, pagination_controls = self.tabbed_panel.log_details.update_logs(
                logs, group_info, page=next_page, show_all=False, log_group_id=log_group_id
            )

            return rendered_logs, pagination_controls, {'page': next_page, 'show_all': False}

        # Handle "Load All" button clicks
        @self.app.callback(
            [
                Output('main-panel-log-details-logs-container', 'children', allow_duplicate=True),
                Output('main-panel-log-details-pagination-controls', 'children', allow_duplicate=True),
                Output('main-panel-log-details-pagination-state', 'data', allow_duplicate=True)
            ],
            [Input('main-panel-log-details-load-all-btn', 'n_clicks')],
            [
                State('log-data-store', 'data'),
                State('main-panel-log-details-group-info', 'children')
            ],
            prevent_initial_call=True
        )
        def handle_load_all(n_clicks, log_data, group_info_text):
            if not n_clicks or not log_data:
                return dash.no_update, dash.no_update, dash.no_update

            # Extract log_group_id from group_info text
            import re
            match = re.search(r'Log Group: ([\w\-]+)', group_info_text or '')
            if not match:
                return dash.no_update, dash.no_update, dash.no_update

            log_group_id = match.group(1)
            if log_group_id not in log_data.get('log_groups', {}):
                return dash.no_update, dash.no_update, dash.no_update

            logs = log_data['log_groups'][log_group_id]

            # Update logs in component and render all logs (uses cache if available)
            group_info = f"Log Group: {log_group_id} ({len(logs)} entries)"
            rendered_logs, _, pagination_controls = self.tabbed_panel.log_details.update_logs(
                logs, group_info, page=0, show_all=True, log_group_id=log_group_id
            )

            return rendered_logs, pagination_controls, {'page': 0, 'show_all': True}

        # Clientside callback for expand/collapse log text
        # This handles the "Show more" / "Show less" button for long log entries
        self.app.clientside_callback(
            """
            function(n_clicks_list, store_data_list) {
                // Get the triggered input
                const ctx = dash_clientside.callback_context;
                if (!ctx.triggered || ctx.triggered.length === 0) {
                    return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
                }

                const triggered_prop = ctx.triggered[0].prop_id;
                const triggered_id_str = triggered_prop.split('.')[0];

                // Parse the triggered ID to get the index
                let triggered_idx = null;
                try {
                    const triggered_id = JSON.parse(triggered_id_str);
                    triggered_idx = triggered_id.index;
                } catch (e) {
                    console.error('Failed to parse triggered ID:', triggered_id_str, e);
                    return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
                }

                // Find the corresponding store data by matching index
                let store_data = null;
                let store_idx = -1;
                for (let i = 0; i < store_data_list.length; i++) {
                    if (store_data_list[i] !== null && store_data_list[i] !== undefined) {
                        // Check if this store corresponds to the triggered index
                        // Store data list is in the same order as the log entries
                        if (i === triggered_idx) {
                            store_data = store_data_list[i];
                            store_idx = i;
                            break;
                        }
                    }
                }

                if (!store_data) {
                    console.error('No store data found for index:', triggered_idx);
                    return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
                }

                // Toggle between full and truncated text
                const is_expanded = store_data.is_expanded || false;
                const new_text = is_expanded ? store_data.truncated : store_data.full;
                const button_text = is_expanded ? "Show more" : "Show less";

                // Update store data
                const new_store_data = {
                    full: store_data.full,
                    truncated: store_data.truncated,
                    is_expanded: !is_expanded
                };

                // Build outputs array (all no_update except for triggered index)
                const text_outputs = [];
                const button_outputs = [];
                const store_outputs = [];

                for (let i = 0; i < n_clicks_list.length; i++) {
                    if (i === store_idx) {
                        text_outputs.push(new_text);
                        button_outputs.push(button_text);
                        store_outputs.push(new_store_data);
                    } else {
                        text_outputs.push(window.dash_clientside.no_update);
                        button_outputs.push(window.dash_clientside.no_update);
                        store_outputs.push(window.dash_clientside.no_update);
                    }
                }

                return [text_outputs, button_outputs, store_outputs];
            }
            """,
            [
                Output({'type': 'main-panel-log-details-log-content', 'index': ALL}, 'children'),
                Output({'type': 'main-panel-log-details-expand-btn', 'index': ALL}, 'children'),
                Output({'type': 'main-panel-log-details-log-full-text', 'index': ALL}, 'data')
            ],
            [Input({'type': 'main-panel-log-details-expand-btn', 'index': ALL}, 'n_clicks')],
            [State({'type': 'main-panel-log-details-log-full-text', 'index': ALL}, 'data')],
            prevent_initial_call=True
        )

        # Clientside callback to fit Cytoscape graph when elements are loaded
        self.app.clientside_callback(
            """
            function(elements) {
                if (elements && elements.length > 0) {
                    // Wait a bit for Cytoscape to render, then fit
                    setTimeout(function() {
                        var cy = window.dash_cytoscape.getCytoscapeInstance('main-panel-log-graph-cytoscape');
                        if (cy) {
                            cy.fit(50); // Fit with 50px padding
                            console.log('[Cytoscape] Graph fitted to viewport');
                        }
                    }, 100);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('main-panel-log-graph-cytoscape', 'zoom'),
            Input('main-panel-log-graph-cytoscape', 'elements')
        )

        # Clientside callback for Reset View button
        # Resets zoom, pan, AND node positions to original
        self.app.clientside_callback(
            """
            function(n_clicks, elements) {
                console.log('[Cytoscape Reset] Button clicked, n_clicks:', n_clicks);
                if (n_clicks && n_clicks > 0 && elements && elements.length > 0) {
                    setTimeout(function() {
                        var cytoscapeDiv = document.getElementById('main-panel-log-graph-cytoscape');
                        if (cytoscapeDiv && cytoscapeDiv._cyreg && cytoscapeDiv._cyreg.cy) {
                            var cy = cytoscapeDiv._cyreg.cy;
                            
                            // Reset node positions to original
                            var nodeCount = 0;
                            elements.forEach(function(ele) {
                                if (ele.data && !ele.data.source && !ele.data.target) {
                                    // This is a node (not an edge)
                                    if (ele.position) {
                                        var node = cy.getElementById(ele.data.id);
                                        if (node.length > 0) {
                                            node.position({
                                                x: ele.position.x,
                                                y: ele.position.y
                                            });
                                            nodeCount++;
                                        }
                                    }
                                }
                            });
                            
                            // Reset zoom and pan
                            cy.zoom(1);
                            cy.pan({ x: 0, y: 0 });
                            
                            console.log('[Cytoscape Reset] Reset complete - ' + nodeCount + ' nodes repositioned, zoom=1, pan=(0,0)');
                        } else {
                            console.error('[Cytoscape Reset] Could not access Cytoscape instance');
                        }
                    }, 100);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('main-panel-log-graph-cytoscape', 'zoom', allow_duplicate=True),
            [Input('main-panel-log-graph-cytoscape-reset-btn', 'n_clicks')],
            [State('main-panel-log-graph-cytoscape', 'elements')],
            prevent_initial_call=True
        )

        # Clientside callback for Fit to Screen button
        # We need to access the Cytoscape instance directly via the DOM
        self.app.clientside_callback(
            """
            function(n_clicks, elements) {
                console.log('[Cytoscape Fit] Button clicked, n_clicks:', n_clicks);
                if (n_clicks && n_clicks > 0 && elements && elements.length > 0) {
                    // Access Cytoscape instance via the component's internal property
                    setTimeout(function() {
                        var cytoscapeDiv = document.getElementById('main-panel-log-graph-cytoscape');
                        if (cytoscapeDiv && cytoscapeDiv._cyreg && cytoscapeDiv._cyreg.cy) {
                            var cy = cytoscapeDiv._cyreg.cy;
                            cy.fit(50);
                            console.log('[Cytoscape Fit] Graph fitted to screen');
                        } else {
                            console.error('[Cytoscape Fit] Could not access Cytoscape instance');
                        }
                    }, 100);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('main-panel-log-graph-cytoscape', 'autoungrabify'),
            [Input('main-panel-log-graph-cytoscape-fit-btn', 'n_clicks')],
            [State('main-panel-log-graph-cytoscape', 'elements')],
            prevent_initial_call=True
        )

        # Clientside callback to auto-fit when switching to Cytoscape mode
        self.app.clientside_callback(
            """
            function(rendering_mode, elements) {
                console.log('[Cytoscape Auto-Fit] Rendering mode changed to:', rendering_mode);
                if (rendering_mode === 'cytoscape' && elements && elements.length > 0) {
                    // Wait a bit for container to be visible and Cytoscape to render
                    setTimeout(function() {
                        var cytoscapeDiv = document.getElementById('main-panel-log-graph-cytoscape');
                        if (cytoscapeDiv && cytoscapeDiv._cyreg && cytoscapeDiv._cyreg.cy) {
                            var cy = cytoscapeDiv._cyreg.cy;
                            cy.fit(50);
                            console.log('[Cytoscape Auto-Fit] Graph automatically fitted to screen on mode switch');
                        } else {
                            console.error('[Cytoscape Auto-Fit] Could not access Cytoscape instance');
                        }
                    }, 150);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('main-panel-log-graph-cytoscape', 'userPanningEnabled', allow_duplicate=True),
            [Input('main-panel-log-graph-rendering-mode', 'value')],
            [State('main-panel-log-graph-cytoscape', 'elements')],
            prevent_initial_call=True
        )

        # Clientside callback for Download PNG button
        self.app.clientside_callback(
            """
            function(n_clicks, elements) {
                console.log('[Cytoscape PNG] Button clicked, n_clicks:', n_clicks);
                if (n_clicks && n_clicks > 0 && elements && elements.length > 0) {
                    setTimeout(function() {
                        var cytoscapeDiv = document.getElementById('main-panel-log-graph-cytoscape');
                        if (cytoscapeDiv && cytoscapeDiv._cyreg && cytoscapeDiv._cyreg.cy) {
                            var cy = cytoscapeDiv._cyreg.cy;
                            
                            // Generate PNG from Cytoscape
                            var png64 = cy.png({
                                output: 'base64',
                                bg: '#2C2C2C',
                                full: true,
                                scale: 2  // Higher resolution
                            });
                            
                            // Create download link
                            var link = document.createElement('a');
                            link.href = png64;
                            link.download = 'log_graph_' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.png';
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            
                            console.log('[Cytoscape PNG] PNG downloaded');
                        } else {
                            console.error('[Cytoscape PNG] Could not access Cytoscape instance');
                        }
                    }, 100);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('main-panel-log-graph-cytoscape', 'boxSelectionEnabled'),
            [Input('main-panel-log-graph-cytoscape-download-png-btn', 'n_clicks')],
            [State('main-panel-log-graph-cytoscape', 'elements')],
            prevent_initial_call=True
        )

        # Clientside callback for Download JSON button
        self.app.clientside_callback(
            """
            function(n_clicks, elements) {
                console.log('[Cytoscape JSON] Button clicked, n_clicks:', n_clicks);
                if (n_clicks && n_clicks > 0 && elements) {
                    // Create JSON data
                    var jsonData = {
                        timestamp: new Date().toISOString(),
                        graph_type: 'cytoscape',
                        elements: elements
                    };
                    
                    // Convert to JSON string
                    var jsonStr = JSON.stringify(jsonData, null, 2);
                    
                    // Create download link
                    var blob = new Blob([jsonStr], { type: 'application/json' });
                    var url = URL.createObjectURL(blob);
                    var link = document.createElement('a');
                    link.href = url;
                    link.download = 'log_graph_' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.json';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                    
                    console.log('[Cytoscape JSON] JSON downloaded');
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('main-panel-log-graph-cytoscape', 'userZoomingEnabled'),
            [Input('main-panel-log-graph-cytoscape-download-json-btn', 'n_clicks')],
            [State('main-panel-log-graph-cytoscape', 'elements')],
            prevent_initial_call=True
        )

        # Add CSS-based hover effects to Cytoscape control buttons
        # Using CSS instead of JavaScript to avoid breaking Dash callbacks
        self.app.clientside_callback(
            """
            function(rendering_mode) {
                // Add CSS hover effects via style injection
                setTimeout(function() {
                    // Check if style already exists
                    if (!document.getElementById('cytoscape-button-hover-styles')) {
                        var style = document.createElement('style');
                        style.id = 'cytoscape-button-hover-styles';
                        style.textContent = `
                            #main-panel-log-graph-cytoscape-reset-btn:hover,
                            #main-panel-log-graph-cytoscape-fit-btn:hover,
                            #main-panel-log-graph-cytoscape-download-png-btn:hover,
                            #main-panel-log-graph-cytoscape-download-json-btn:hover {
                                background-color: rgba(255, 255, 255, 0.15) !important;
                            }
                        `;
                        document.head.appendChild(style);
                        console.log('[Cytoscape] Hover styles injected');
                    }
                }, 100);
                return window.dash_clientside.no_update;
            }
            """,
            Output('main-panel-log-graph-cytoscape', 'userPanningEnabled'),
            Input('main-panel-log-graph-rendering-mode', 'value'),
            prevent_initial_call=True
        )

        # === Monitor Panel Draggable Functionality ===

        # Clientside callback to make monitor panel draggable
        self.app.clientside_callback(
            """
            function(_) {
                // Initialize drag functionality on component mount
                const panel = document.getElementById('main-panel-log-graph-monitor-panel');
                const dragHandle = document.getElementById('main-panel-log-graph-monitor-drag-handle');

                if (!panel || !dragHandle) {
                    console.error('[Monitor Panel] Could not find panel or drag handle');
                    return window.dash_clientside.no_update;
                }

                // Check if already initialized
                if (panel.dataset.draggableInit === 'true') {
                    return window.dash_clientside.no_update;
                }

                let isDragging = false;
                let currentX;
                let currentY;
                let initialX;
                let initialY;
                let xOffset = 0;
                let yOffset = 0;

                dragHandle.addEventListener('mousedown', dragStart);
                document.addEventListener('mousemove', drag);
                document.addEventListener('mouseup', dragEnd);

                function dragStart(e) {
                    initialX = e.clientX - xOffset;
                    initialY = e.clientY - yOffset;

                    if (e.target === dragHandle || dragHandle.contains(e.target)) {
                        isDragging = true;
                    }
                }

                function drag(e) {
                    if (isDragging) {
                        e.preventDefault();

                        currentX = e.clientX - initialX;
                        currentY = e.clientY - initialY;

                        xOffset = currentX;
                        yOffset = currentY;

                        setTranslate(currentX, currentY, panel);
                    }
                }

                function dragEnd(e) {
                    initialX = currentX;
                    initialY = currentY;
                    isDragging = false;
                }

                function setTranslate(xPos, yPos, el) {
                    el.style.transform = 'translate3d(' + xPos + 'px, ' + yPos + 'px, 0)';
                }

                panel.dataset.draggableInit = 'true';
                console.log('[Monitor Panel] Drag functionality initialized');

                return window.dash_clientside.no_update;
            }
            """,
            Output('main-panel-log-graph-monitor-panel', 'data-draggable-init'),
            Input('main-panel-log-graph-monitor-panel', 'id')
        )

        # === Monitor Panel Tab Switching ===
        # NOTE: Tab switching callback for custom monitor tabs should be implemented
        # by concrete subclasses if needed, as outputs/inputs depend on which custom tabs are added

        # Callback to populate response list in the Response Monitor
        @self.app.callback(
            [
                Output('main-panel-log-graph-response-count', 'children'),
                Output('main-panel-log-graph-response-list', 'children')
            ],
            [
                Input('messages-store', 'data'),
                Input('current-session-store', 'data')
            ],
            prevent_initial_call=False
        )
        def update_response_list(messages_store, session_id):
            """Update the Response Monitor list with all agent responses."""
            if not session_id or not messages_store or session_id not in messages_store:
                return 'Total: 0', 'No responses'

            messages = messages_store[session_id]

            # Filter assistant messages only
            agent_responses = [msg for msg in messages if msg.get('role') == 'assistant']

            if not agent_responses:
                return 'Total: 0', 'No responses'

            count_text = f'Total: {len(agent_responses)}'

            # Build clickable response list items (most recent first)
            response_items = []
            for idx, msg in enumerate(reversed(agent_responses)):
                content = msg.get('content', '')
                timestamp = msg.get('timestamp', '')
                response_num = len(agent_responses) - idx

                # Extract first few characters for preview
                if isinstance(content, str):
                    preview = content[:20].replace('\n', ' ')
                elif isinstance(content, list):
                    preview = str(content[0])[:20].replace('\n', ' ') if content else '...'
                elif isinstance(content, dict):
                    response_content = content.get('response', content)
                    preview = str(response_content)[:20].replace('\n', ' ')
                else:
                    preview = str(content)[:20].replace('\n', ' ')

                # Create clickable item
                response_items.append(
                    html.Div(
                        children=[
                            html.Div(f"#{response_num}", style={'fontWeight': '600', 'color': '#19C37D', 'fontSize': '8px'}),
                            html.Div(preview + '...', style={'fontSize': '8px', 'color': '#ECECF1', 'marginTop': '2px'}),
                            html.Div(timestamp, style={'fontSize': '7px', 'color': '#6E6E80', 'marginTop': '2px'})
                        ],
                        id={'type': 'main-panel-log-graph-response-item', 'index': idx},
                        n_clicks=0,
                        style={
                            'padding': '4px',
                            'marginBottom': '3px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                            'borderRadius': '2px',
                            'cursor': 'pointer',
                            'transition': 'background-color 0.2s',
                            'borderLeft': '2px solid transparent'
                        }
                    )
                )

            return count_text, response_items

        # Callback to show response details when clicking an item
        @self.app.callback(
            Output('main-panel-log-graph-response-details', 'children'),
            [
                Input({'type': 'main-panel-log-graph-response-item', 'index': ALL}, 'n_clicks'),
                Input('messages-store', 'data'),
                Input('current-session-store', 'data')
            ],
            prevent_initial_call=True
        )
        def show_response_details(n_clicks_list, messages_store, session_id):
            """Show full details of selected response."""
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            triggered_prop = ctx.triggered[0]['prop_id']

            # Check if triggered by a response item click
            if 'response-item' not in triggered_prop:
                raise PreventUpdate

            # Parse which item was clicked
            import json
            triggered_id_str = triggered_prop.split('.')[0]
            try:
                triggered_id = json.loads(triggered_id_str)
                clicked_idx = triggered_id['index']
            except (json.JSONDecodeError, KeyError):
                raise PreventUpdate

            # Get messages
            if not session_id or not messages_store or session_id not in messages_store:
                return 'No data'

            messages = messages_store[session_id]
            agent_responses = [msg for msg in messages if msg.get('role') == 'assistant']

            if not agent_responses:
                return 'No responses'

            # Reverse to match list order (most recent first)
            agent_responses_reversed = list(reversed(agent_responses))

            if clicked_idx >= len(agent_responses_reversed):
                return 'Invalid response'

            selected_msg = agent_responses_reversed[clicked_idx]
            content = selected_msg.get('content', '')
            timestamp = selected_msg.get('timestamp', '')
            response_num = len(agent_responses) - clicked_idx

            # Format content for display
            if isinstance(content, str):
                display_content = content
            elif isinstance(content, list):
                # Format list nicely
                display_content = 'Response (list):\n'
                for i, item in enumerate(content):
                    display_content += f'\n[{i}]: {item}\n'
            elif isinstance(content, dict):
                # Format dict nicely
                import json
                display_content = 'Response (dict):\n' + json.dumps(content, indent=2)
            else:
                display_content = str(content)

            # Build details view
            return html.Div([
                html.Div(
                    f"Response #{response_num}",
                    style={
                        'fontSize': '10px',
                        'fontWeight': '600',
                        'color': '#19C37D',
                        'marginBottom': '4px',
                        'borderBottom': '1px solid rgba(255,255,255,0.1)',
                        'paddingBottom': '4px'
                    }
                ),
                html.Div(
                    f"Time: {timestamp}",
                    style={
                        'fontSize': '8px',
                        'color': '#8E8EA0',
                        'marginBottom': '8px'
                    }
                ),
                html.Div(
                    display_content,
                    style={
                        'fontSize': '9px',
                        'color': '#ECECF1',
                        'whiteSpace': 'pre-wrap',
                        'wordBreak': 'break-word'
                    }
                )
            ])

        # Clientside callback to highlight selected response item
        self.app.clientside_callback(
            """
            function(n_clicks_list) {
                const ctx = dash_clientside.callback_context;
                if (!ctx.triggered || ctx.triggered.length === 0) {
                    return window.dash_clientside.no_update;
                }

                const triggered_prop = ctx.triggered[0].prop_id;
                if (!triggered_prop.includes('response-item')) {
                    return window.dash_clientside.no_update;
                }

                // Parse triggered ID
                const triggered_id_str = triggered_prop.split('.')[0];
                let clicked_idx = -1;
                try {
                    const triggered_id = JSON.parse(triggered_id_str);
                    clicked_idx = triggered_id.index;
                } catch (e) {
                    return window.dash_clientside.no_update;
                }

                // Build style outputs for all items
                const style_outputs = [];
                for (let i = 0; i < n_clicks_list.length; i++) {
                    if (i === clicked_idx) {
                        // Highlight selected item
                        style_outputs.push({
                            'padding': '4px',
                            'marginBottom': '3px',
                            'backgroundColor': 'rgba(25, 195, 125, 0.2)',
                            'borderRadius': '2px',
                            'cursor': 'pointer',
                            'transition': 'background-color 0.2s',
                            'borderLeft': '2px solid #19C37D'
                        });
                    } else {
                        // Normal style
                        style_outputs.push({
                            'padding': '4px',
                            'marginBottom': '3px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                            'borderRadius': '2px',
                            'cursor': 'pointer',
                            'transition': 'background-color 0.2s',
                            'borderLeft': '2px solid transparent'
                        });
                    }
                }

                return style_outputs;
            }
            """,
            Output({'type': 'main-panel-log-graph-response-item', 'index': ALL}, 'style'),
            Input({'type': 'main-panel-log-graph-response-item', 'index': ALL}, 'n_clicks'),
            prevent_initial_call=True
        )

        # === Agent queue polling callbacks ===

        # Call the polling callback registration (can be overridden by subclasses)
        self._register_polling_callback()

        # Callback to auto-refresh logs
        @self.app.callback(
            Output('log-data-store', 'data', allow_duplicate=True),
            Input('log-refresh-interval', 'n_intervals'),
            State('current-session-store', 'data'),
            prevent_initial_call=True
        )
        def auto_refresh_logs(n_intervals, session_id):
            """Auto-refresh logs as agent executes."""
            if not session_id or session_id not in self.session_agents:
                return dash.no_update

            agent = self.session_agents[session_id]

            # Access agent's log collector if available
            if hasattr(agent, 'log_collector') and agent.log_collector:
                try:
                    graph_structure = agent.log_collector.get_graph_structure()
                    return {
                        'graph_data': {
                            'nodes': graph_structure['nodes'],
                            'edges': graph_structure['edges'],
                            'agent': graph_structure.get('agent', {})
                        },
                        'log_groups': {k: v for k, v in agent.log_collector.log_groups.items()}
                    }
                except Exception:
                    # If there's an error getting graph structure, just skip this update
                    pass

            return dash.no_update

    def set_agent_factory(self, factory: Callable):
        """
        Set factory function that creates new agent instances.

        Args:
            factory: Function that returns a configured Agent with QueueInteractive
        """
        self.agent_factory = factory

    def add_monitor_tab(self, tab_id: str, tab_label: str, tab_content: Any):
        """
        Add a custom tab to the monitor panel.

        This must be called BEFORE the app layout is created (i.e., before super().__init__()).
        Concrete implementations should subclass and add tabs in their __init__ before calling super().

        Args:
            tab_id: Unique identifier for the tab (e.g., 'settings')
            tab_label: Button label displayed for the tab (e.g., 'Settings')
            tab_content: Dash component(s) to display in the tab content area

        Example:
            >>> class CustomApp(DashInteractiveAppWithLogs):
            >>>     def __init__(self, **kwargs):
            >>>         # Add custom tabs before parent init
            >>>         self.custom_monitor_tabs = []
            >>>         self.add_monitor_tab('settings', 'Settings', self._create_settings_content())
            >>>         super().__init__(**kwargs)
        """
        tab = {
            'id': tab_id,
            'label': tab_label,
            'content': tab_content
        }
        self.custom_monitor_tabs.append(tab)

    def _start_agent_for_session(self, session_id: str):
        """Start background agent thread for a session."""
        if not self.agent_factory:
            raise ValueError("No agent_factory set. Call set_agent_factory() first.")

        # Create fresh QueueInteractive for this session
        interactive = QueueInteractive()
        self.session_interactives[session_id] = interactive

        # Start background thread that will create AND run the agent
        # This prevents UI blocking during agent creation
        thread = Thread(
            target=self._run_agent_in_background,
            args=(session_id,),
            daemon=True,
            name=f"Agent-{session_id}"
        )
        thread.start()
        self.session_threads[session_id] = thread

    def _run_agent_in_background(self, session_id: str):
        """Create and run agent in background thread."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            interactive = self.session_interactives[session_id]

            # Notify user that agent is initializing
            interactive.response_queue.put("🤖 Initializing agent (this may take a few seconds)...")

            logger.info(f"Creating agent for session {session_id}")
            # Create agent (this takes 5-10 seconds)
            agent = self.agent_factory()
            agent.interactive = interactive
            self.session_agents[session_id] = agent
            logger.info(f"Agent created for session {session_id}")

            # Notify user that agent is ready
            interactive.response_queue.put("✓ Agent ready! Processing your message...")

            # Run agent
            logger.info(f"Starting agent execution for session {session_id}")
            agent_results = agent({})  # Calls agent.__call__()
            logger.info(f"Agent completed for session {session_id}")

            # Put completion marker in response queue
            interactive.response_queue.put("[AGENT_COMPLETED]")

        except Exception as e:
            logger.exception(f"Agent crashed for session {session_id}")
            interactive = self.session_interactives.get(session_id)
            if interactive:
                interactive.response_queue.put(f"❌ ERROR: {str(e)}")

        finally:
            # Cleanup
            self.session_threads.pop(session_id, None)

    # run() and set_message_handler() are inherited from parent class

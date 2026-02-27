"""
Log details panel component for displaying individual log entries.
"""
from typing import Any, Dict, List, Optional
from dash import html, dcc
from dash.dependencies import Input, Output, State
import json

from agent_foundation.ui.dash_interactive.components.base import BaseComponent


class LogDetailsPanel(BaseComponent):
    """
    Component for displaying detailed log entries for a selected log group.

    This component shows individual log entries with their level, type,
    timestamp, and content in a readable format.

    Attributes:
        component_id (str): Unique identifier for this component
        logs (List[Dict]): List of log entries to display
    """

    def __init__(
        self,
        component_id: str = "log-details",
        logs: Optional[List[Dict[str, Any]]] = None,
        style: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the log details panel component.

        Args:
            component_id: Unique identifier for this component
            logs: Initial list of log entries
            style: Optional CSS style overrides
        """
        super().__init__(component_id, style)
        self.logs = logs or []
        self.page_size = 10  # Number of logs to show initially
        self.current_page = 0  # Current page index (0-based)
        self.show_all = False  # Whether to show all logs

        # Caching for rendered HTML elements
        # Cache key: (log_group_id, log_count, page, show_all) -> (rendered_logs, pagination_controls)
        self._render_cache = {}

        # Per-node pagination state
        # Key: log_group_id -> {'page': int, 'show_all': bool}
        self._node_pagination_state = {}

        # Current log group ID (to track which node is displayed)
        self._current_log_group_id = None

    def _get_default_style(self) -> Dict[str, Any]:
        """Get default styling for the log details panel."""
        return {
            'height': '100%',
            'backgroundColor': '#1E1E1E',
            'overflowY': 'auto',
            'padding': '10px'
        }

    def layout(self) -> html.Div:
        """
        Generate the log details panel layout.

        Returns:
            Dash Div containing the log details
        """
        return html.Div(
            id=self.get_id(),
            children=[
                html.Div(
                    children=[
                        html.H4(
                            "Log Details",
                            style={
                                'color': '#ECECF1',
                                'margin': '0 0 10px 0',
                                'fontSize': '16px',
                                'fontWeight': '600'
                            }
                        ),
                        html.Div(
                            id=self.get_id('group-info'),
                            style={
                                'color': '#8E8EA0',
                                'fontSize': '12px',
                                'marginBottom': '15px',
                                'padding': '8px',
                                'backgroundColor': '#2C2C2C',
                                'borderRadius': '4px'
                            }
                        )
                    ],
                    style={'borderBottom': '1px solid #4D4D4F', 'paddingBottom': '10px'}
                ),
                # Loading indicator (hidden by default)
                html.Div(
                    id=self.get_id('loading-indicator'),
                    children=[
                        html.Div(
                            "⏳ Rendering logs...",
                            style={
                                'color': '#8E8EA0',
                                'fontSize': '14px',
                                'textAlign': 'center',
                                'padding': '40px 20px'
                            }
                        )
                    ],
                    style={'display': 'none'}
                ),
                # Logs container
                html.Div(
                    id=self.get_id('logs-container'),
                    children=self._render_logs(),
                    style={
                        'marginTop': '10px'
                    }
                ),
                # Pagination controls
                html.Div(
                    id=self.get_id('pagination-controls'),
                    children=self._render_pagination_controls(),
                    style={
                        'marginTop': '15px',
                        'paddingTop': '15px',
                        'borderTop': '1px solid #4D4D4F',
                        'display': 'flex',
                        'justifyContent': 'center',
                        'gap': '10px'
                    }
                ),
                # Hidden store for pagination state
                dcc.Store(id=self.get_id('pagination-state'), data={'page': 0, 'show_all': False})
            ],
            style=self.style
        )

    def _render_logs(self, page: int = 0, show_all: bool = False) -> List[html.Div]:
        """
        Render individual log entries with pagination.

        Args:
            page: Current page number (0-based)
            show_all: Whether to show all logs (ignores pagination)

        Returns:
            List of Div elements for each log entry
        """
        if not self.logs:
            return [
                html.Div(
                    "Select a log group in the graph above to view details",
                    style={
                        'padding': '40px 20px',
                        'textAlign': 'center',
                        'color': '#8E8EA0',
                        'fontSize': '14px'
                    }
                )
            ]

        # Determine which logs to show based on pagination
        if show_all:
            logs_to_render = self.logs
        else:
            # Accumulative pagination: show from start up to (page+1) * page_size
            # This way "Load More" adds more logs instead of replacing
            start_idx = 0
            end_idx = (page + 1) * self.page_size
            logs_to_render = self.logs[start_idx:end_idx]

        log_divs = []
        for i, log in enumerate(logs_to_render):
            # Index is simply i since we always start from 0
            log_div = self._create_log_entry(log, i)
            log_divs.append(log_div)

        return log_divs

    def _render_pagination_controls(self, page: int = 0, show_all: bool = False) -> List:
        """
        Render pagination controls (Load More, Load All buttons).

        Args:
            page: Current page number (0-based)
            show_all: Whether all logs are currently shown

        Returns:
            List of button elements
        """
        if not self.logs or len(self.logs) <= self.page_size:
            # No pagination needed if logs fit in one page
            return []

        total_logs = len(self.logs)
        shown_logs = total_logs if show_all else min((page + 1) * self.page_size, total_logs)
        has_more = shown_logs < total_logs

        controls = []

        if not show_all and has_more:
            # Show "Load More" button
            controls.append(
                html.Button(
                    f"Load More ({shown_logs}/{total_logs} shown)",
                    id=self.get_id('load-more-btn'),
                    n_clicks=0,
                    style={
                        'padding': '8px 16px',
                        'backgroundColor': '#19C37D',
                        'color': '#1E1E1E',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontSize': '12px',
                        'fontWeight': '500'
                    }
                )
            )

        if not show_all and total_logs > self.page_size * 2:
            # Show "Load All" button if there are many logs
            controls.append(
                html.Button(
                    f"Load All ({total_logs} total)",
                    id=self.get_id('load-all-btn'),
                    n_clicks=0,
                    style={
                        'padding': '8px 16px',
                        'backgroundColor': '#4A4A5A',
                        'color': '#ECECF1',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontSize': '12px',
                        'fontWeight': '500'
                    }
                )
            )

        if show_all:
            # Show info that all logs are loaded
            controls.append(
                html.Div(
                    f"All {total_logs} logs loaded",
                    style={
                        'color': '#8E8EA0',
                        'fontSize': '12px',
                        'padding': '8px 16px'
                    }
                )
            )

        return controls

    def _create_log_entry(self, log: Dict[str, Any], index: int) -> html.Div:
        """
        Create a single log entry display with left/right split view.

        Left side: Metadata (level, name, time)
        Right side: Log details/content

        Args:
            log: Log entry dictionary
            index: Index of the log entry

        Returns:
            Div element for the log entry
        """
        # Determine log level color
        level = log.get('level', 20)
        level_colors = {
            10: '#00BFFF',  # DEBUG - cyan
            20: '#19C37D',  # INFO - green
            30: '#FFA500',  # WARNING - orange
            40: '#FF4500',  # ERROR - red
            50: '#DC143C',  # CRITICAL - crimson
        }
        level_color = level_colors.get(level, '#8E8EA0')

        # Level name
        level_names = {
            10: 'DEBUG',
            20: 'INFO',
            30: 'WARNING',
            40: 'ERROR',
            50: 'CRITICAL'
        }
        level_name = level_names.get(level, 'INFO')

        # Format timestamp
        timestamp = log.get('timestamp', log.get('time', ''))
        if not timestamp and 'item' in log:
            # Try to extract from item if it's a timestamp
            timestamp = ''

        # Format log item
        log_item = log.get('item', '')
        if isinstance(log_item, (dict, list)):
            log_item = json.dumps(log_item, indent=2)
        else:
            log_item = str(log_item)

        # Check if log item is long and should be truncated
        char_limit = 200
        is_long = len(log_item) > char_limit
        truncated_item = log_item[:char_limit] + '...' if is_long else log_item

        # Format log type
        log_type = log.get('type', 'Unknown')
        log_name = log.get('name', 'Logger')

        return html.Div(
            children=[
                html.Div(
                    children=[
                        # LEFT SIDE - Metadata
                        html.Div(
                            children=[
                                # Level badge
                                html.Div(
                                    level_name,
                                    style={
                                        'color': level_color,
                                        'fontWeight': 'bold',
                                        'fontSize': '11px',
                                        'padding': '3px 8px',
                                        'backgroundColor': f"{level_color}22",
                                        'borderRadius': '3px',
                                        'marginBottom': '8px',
                                        'display': 'inline-block'
                                    }
                                ),
                                # Type
                                html.Div(
                                    children=[
                                        html.Span(
                                            f"[{log_type}]",
                                            style={
                                                'color': '#ECECF1',
                                                'fontSize': '11px',
                                                'fontFamily': 'monospace'
                                            }
                                        )
                                    ],
                                    style={'marginBottom': '6px'}
                                ),
                                # Timestamp
                                html.Div(
                                    children=[
                                        html.Span(
                                            timestamp if timestamp else 'No timestamp',
                                            style={
                                                'color': '#8E8EA0',
                                                'fontSize': '10px',
                                                'fontFamily': 'monospace'
                                            }
                                        )
                                    ],
                                    style={'marginBottom': '6px'}
                                ),
                                # Logger name
                                html.Div(
                                    children=[
                                        html.Span(
                                            f"Logger: {log_name}",
                                            style={
                                                'color': '#8E8EA0',
                                                'fontSize': '10px'
                                            }
                                        )
                                    ]
                                ),
                                # Expandable metadata at bottom of left side
                                self._create_expandable_metadata(log, index) if log.get('full_log_group_id') else None
                            ],
                            style={
                                'flex': '0 0 220px',
                                'padding': '12px',
                                'backgroundColor': '#2A2A2A',
                                'borderRight': f'3px solid {level_color}',
                                'borderTopLeftRadius': '6px',
                                'borderBottomLeftRadius': '6px'
                            }
                        ),
                        # RIGHT SIDE - Log content
                        html.Div(
                            children=[
                                # Truncated/full log content (initially shows truncated if long)
                                html.Pre(
                                    truncated_item,
                                    id={'type': self.get_id('log-content'), 'index': index},
                                    style={
                                        'color': '#ECECF1',
                                        'fontSize': '12px',
                                        'margin': '0',
                                        'padding': '0',
                                        'whiteSpace': 'pre-wrap',
                                        'wordBreak': 'break-word',
                                        'fontFamily': 'Consolas, Monaco, "Courier New", monospace',
                                        'lineHeight': '1.5'
                                    }
                                ),
                                # Expand/collapse button for long logs
                                html.Button(
                                    "Show more",
                                    id={'type': self.get_id('expand-btn'), 'index': index},
                                    n_clicks=0,
                                    style={
                                        'marginTop': '8px',
                                        'padding': '4px 10px',
                                        'fontSize': '11px',
                                        'backgroundColor': '#4A4A5A',
                                        'color': '#ECECF1',
                                        'border': 'none',
                                        'borderRadius': '3px',
                                        'cursor': 'pointer',
                                        'display': 'inline-block' if is_long else 'none'
                                    }
                                ),
                                # Hidden store for full log text
                                dcc.Store(
                                    id={'type': self.get_id('log-full-text'), 'index': index},
                                    data={'full': log_item, 'truncated': truncated_item, 'is_expanded': False}
                                )
                            ],
                            style={
                                'flex': '1',
                                'padding': '12px 16px',
                                'backgroundColor': '#1E1E1E',
                                'borderTopRightRadius': '6px',
                                'borderBottomRightRadius': '6px',
                                'overflowX': 'auto'
                            }
                        )
                    ],
                    style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'minHeight': '80px'
                    }
                )
            ],
            style={
                'marginBottom': '12px',
                'borderRadius': '6px',
                'border': '1px solid #3C3C3C',
                'overflow': 'hidden'
            }
        )

    def _create_expandable_metadata(self, log: Dict[str, Any], index: int) -> html.Details:
        """
        Create expandable metadata section for additional log hierarchy information.

        Args:
            log: Log entry dictionary
            index: Index of the log entry

        Returns:
            Details element with hierarchy information
        """
        metadata = {
            'Full Log Group ID': log.get('full_log_group_id', 'N/A'),
            'Log Group ID': log.get('log_group_id', 'N/A'),
            'Parent Log Group ID': log.get('parent_log_group_id', 'N/A')
        }

        return html.Details(
            children=[
                html.Summary(
                    "▼ Hierarchy",
                    style={
                        'color': '#8E8EA0',
                        'fontSize': '10px',
                        'cursor': 'pointer',
                        'marginTop': '12px',
                        'userSelect': 'none'
                    }
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    f"{key}:",
                                    style={
                                        'color': '#8E8EA0',
                                        'fontSize': '9px',
                                        'marginBottom': '2px'
                                    }
                                ),
                                html.Div(
                                    str(value),
                                    style={
                                        'color': '#ECECF1',
                                        'fontSize': '9px',
                                        'fontFamily': 'monospace',
                                        'wordBreak': 'break-all',
                                        'marginBottom': '6px'
                                    }
                                )
                            ]
                        )
                        for key, value in metadata.items()
                    ],
                    style={
                        'marginTop': '6px',
                        'padding': '6px',
                        'backgroundColor': '#1E1E1E',
                        'borderRadius': '3px',
                        'fontSize': '9px'
                    }
                )
            ]
        )

    def get_callback_inputs(self) -> List[Input]:
        """Get list of callback inputs."""
        return []

    def get_callback_outputs(self) -> List[Output]:
        """Get list of callback outputs."""
        return [
            Output(self.get_id('logs-container'), 'children'),
            Output(self.get_id('group-info'), 'children')
        ]

    def _get_cache_key(self, log_group_id: str, log_count: int, page: int, show_all: bool) -> tuple:
        """
        Generate cache key for rendered logs.

        Args:
            log_group_id: ID of the log group
            log_count: Total number of logs in the group
            page: Current page number
            show_all: Whether showing all logs

        Returns:
            Tuple to use as cache key
        """
        return (log_group_id, log_count, page, show_all)

    def _invalidate_cache_for_node(self, log_group_id: str):
        """
        Invalidate all cached renders for a specific node.

        Args:
            log_group_id: ID of the log group to invalidate
        """
        keys_to_remove = [key for key in self._render_cache.keys() if key[0] == log_group_id]
        for key in keys_to_remove:
            del self._render_cache[key]

    def _get_or_restore_pagination_state(self, log_group_id: str, default_page: int = 0, default_show_all: bool = False) -> tuple:
        """
        Get saved pagination state for a node, or use defaults.

        Args:
            log_group_id: ID of the log group
            default_page: Default page if no state saved
            default_show_all: Default show_all if no state saved

        Returns:
            Tuple of (page, show_all)
        """
        if log_group_id in self._node_pagination_state:
            state = self._node_pagination_state[log_group_id]
            return state['page'], state['show_all']
        return default_page, default_show_all

    def _save_pagination_state(self, log_group_id: str, page: int, show_all: bool):
        """
        Save pagination state for a node.

        Args:
            log_group_id: ID of the log group
            page: Current page number
            show_all: Whether showing all logs
        """
        self._node_pagination_state[log_group_id] = {'page': page, 'show_all': show_all}

    def update_logs(self, logs: List[Dict[str, Any]], group_info: str = "", page: int = 0, show_all: bool = False, log_group_id: str = None, use_cache: bool = True) -> tuple:
        """
        Update the displayed logs with pagination support and caching.

        Args:
            logs: New list of log entries
            group_info: Information about the selected log group
            page: Current page number (0-based)
            show_all: Whether to show all logs
            log_group_id: ID of the log group (for caching)
            use_cache: Whether to use cached renders (default True)

        Returns:
            Tuple of (rendered logs, group info text, pagination controls)
        """
        self.logs = logs
        log_count = len(logs)

        # Extract log_group_id from group_info if not provided
        if log_group_id is None and group_info:
            import re
            match = re.search(r'Log Group: ([\w\-]+)', group_info)
            if match:
                log_group_id = match.group(1)

        # If switching to a different node, restore its pagination state
        if log_group_id and log_group_id != self._current_log_group_id:
            # Switching nodes - restore saved pagination state
            saved_page, saved_show_all = self._get_or_restore_pagination_state(log_group_id)
            # Only use saved state if not explicitly overridden
            if page == 0 and not show_all:  # User didn't specify explicit state
                page = saved_page
                show_all = saved_show_all
            self._current_log_group_id = log_group_id

        # Check cache if enabled and we have a log_group_id
        if use_cache and log_group_id:
            cache_key = self._get_cache_key(log_group_id, log_count, page, show_all)

            if cache_key in self._render_cache:
                # Cache hit!
                cached_logs, cached_pagination = self._render_cache[cache_key]
                self.current_page = page
                self.show_all = show_all
                # Save pagination state
                self._save_pagination_state(log_group_id, page, show_all)
                return cached_logs, group_info, cached_pagination

        # Cache miss or caching disabled - render fresh
        self.current_page = page
        self.show_all = show_all

        rendered_logs = self._render_logs(page, show_all)
        pagination_controls = self._render_pagination_controls(page, show_all)

        # Store in cache if we have a log_group_id
        if log_group_id:
            cache_key = self._get_cache_key(log_group_id, log_count, page, show_all)
            self._render_cache[cache_key] = (rendered_logs, pagination_controls)

            # Save pagination state
            self._save_pagination_state(log_group_id, page, show_all)

            # Limit cache size (keep last 50 entries)
            if len(self._render_cache) > 50:
                # Remove oldest entries (first 10)
                keys_to_remove = list(self._render_cache.keys())[:10]
                for key in keys_to_remove:
                    del self._render_cache[key]

        return rendered_logs, group_info, pagination_controls

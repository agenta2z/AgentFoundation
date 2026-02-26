LogDetailsPanel
===============

The ``LogDetailsPanel`` component displays detailed log entries for a selected log group,
with support for pagination, caching, and expandable log content.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

``LogDetailsPanel`` provides a detailed view of individual log entries within a selected log group.
It works in conjunction with :doc:`log_graph` to display log details when a node is selected in
the visualization. The component features color-coded log levels, pagination for large log sets,
and expandable metadata sections.

Class Reference
---------------

.. py:class:: LogDetailsPanel(component_id="log-details", logs=None, style=None)

   Component for displaying detailed log entries with pagination and caching.

   :param str component_id: Unique identifier for this component
   :param list logs: Initial list of log entry dictionaries
   :param dict style: Optional CSS style overrides

   .. py:attribute:: page_size
      :type: int
      :value: 10

      Number of log entries to display per page.

   .. py:attribute:: current_page
      :type: int
      :value: 0

      Current page index (0-based).

   .. py:attribute:: show_all
      :type: bool
      :value: False

      Whether to display all logs without pagination.

Basic Usage
-----------

Creating a Log Details Panel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive.components.log_details import LogDetailsPanel

   # Create panel with default settings
   log_details = LogDetailsPanel(
       component_id="my-log-details"
   )

   # Create panel with initial logs
   log_details = LogDetailsPanel(
       component_id="my-log-details",
       logs=[
           {
               'level': 20,
               'type': 'LLM Call',
               'timestamp': '2024-01-15 10:30:45',
               'name': 'agent.llm',
               'item': 'Processing user query...'
           }
       ]
   )

   # Add to Dash layout
   app.layout = html.Div([
       log_details.layout()
   ])

Updating Logs
^^^^^^^^^^^^^

.. code-block:: python

   # Update displayed logs with new data
   rendered_logs, group_info, pagination = log_details.update_logs(
       logs=new_log_entries,
       group_info="Log Group: agent-1234 | Total: 25 logs",
       page=0,
       show_all=False,
       log_group_id="agent-1234"
   )

Log Entry Format
----------------

Log entries are dictionaries with the following fields:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``level``
     - int
     - Python logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)
   * - ``type``
     - str
     - Category of the log (e.g., "LLM Call", "Tool Use", "Agent Step")
   * - ``timestamp``
     - str
     - Timestamp string for when the log was created
   * - ``name``
     - str
     - Logger name that created the entry
   * - ``item``
     - str/dict/list
     - Actual log content (strings are displayed directly, dicts/lists are JSON formatted)
   * - ``full_log_group_id``
     - str
     - Full hierarchical ID of the log group (optional, enables metadata expansion)
   * - ``log_group_id``
     - str
     - ID of the immediate log group (optional)
   * - ``parent_log_group_id``
     - str
     - ID of the parent log group (optional)

Log Level Styling
-----------------

Each log entry is color-coded based on its level:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Level
     - Value
     - Color
     - Use Case
   * - DEBUG
     - 10
     - Cyan (#00BFFF)
     - Detailed debugging information
   * - INFO
     - 20
     - Green (#19C37D)
     - General informational messages
   * - WARNING
     - 30
     - Orange (#FFA500)
     - Warning conditions
   * - ERROR
     - 40
     - Red (#FF4500)
     - Error conditions
   * - CRITICAL
     - 50
     - Crimson (#DC143C)
     - Critical failures

Layout Structure
----------------

The log details panel uses a split view for each log entry:

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────────┐
   │  Log Details                                                      │
   │  ┌──────────────────────────────────────────────────────────────┐│
   │  │ Log Group: agent-1234 | Total: 25 logs | Type: Agent         ││
   │  └──────────────────────────────────────────────────────────────┘│
   │                                                                   │
   │  ┌────────────────────┬─────────────────────────────────────────┐│
   │  │ [INFO]             │ Processing user request...              ││
   │  │ [Agent Step]       │                                         ││
   │  │ 10:30:45           │                                         ││
   │  │ Logger: agent.llm  │                                         ││
   │  │ ▼ Hierarchy        │                                         ││
   │  └────────────────────┴─────────────────────────────────────────┘│
   │                                                                   │
   │  ┌────────────────────┬─────────────────────────────────────────┐│
   │  │ [DEBUG]            │ {"model": "gpt-4", "tokens": 150}       ││
   │  │ [LLM Call]         │ [Show more]                              ││
   │  │ 10:30:46           │                                         ││
   │  │ Logger: llm.api    │                                         ││
   │  └────────────────────┴─────────────────────────────────────────┘│
   │                                                                   │
   │           [Load More (10/25 shown)]    [Load All (25 total)]     │
   └──────────────────────────────────────────────────────────────────┘

Pagination System
-----------------

The component implements accumulative pagination, where "Load More" adds additional
logs rather than replacing the current view.

Default Behavior
^^^^^^^^^^^^^^^^

- Shows 10 logs initially (configurable via ``page_size``)
- "Load More" button loads the next page of logs
- "Load All" button displays all logs at once (shown when total > 2 × page_size)
- Pagination state is preserved per log group when switching between nodes

Pagination Methods
^^^^^^^^^^^^^^^^^^

.. py:method:: _render_pagination_controls(page=0, show_all=False)

   Render pagination buttons based on current state.

   :param int page: Current page number (0-based)
   :param bool show_all: Whether all logs are currently shown
   :return: List of button elements

.. py:method:: _get_or_restore_pagination_state(log_group_id, default_page=0, default_show_all=False)

   Restore saved pagination state when returning to a previously viewed log group.

   :param str log_group_id: ID of the log group
   :param int default_page: Default page if no saved state
   :param bool default_show_all: Default show_all if no saved state
   :return: Tuple of (page, show_all)

.. py:method:: _save_pagination_state(log_group_id, page, show_all)

   Save current pagination state for a log group.

   :param str log_group_id: ID of the log group
   :param int page: Current page number
   :param bool show_all: Whether showing all logs

Caching System
--------------

The component includes a render caching system to improve performance when
navigating between log groups.

Cache Key Structure
^^^^^^^^^^^^^^^^^^^

Cache keys are tuples of ``(log_group_id, log_count, page, show_all)``:

.. code-block:: python

   def _get_cache_key(self, log_group_id: str, log_count: int,
                      page: int, show_all: bool) -> tuple:
       return (log_group_id, log_count, page, show_all)

Cache Management
^^^^^^^^^^^^^^^^

- Cache stores rendered HTML and pagination controls
- Maximum cache size: 50 entries (oldest entries removed when exceeded)
- Cache is automatically invalidated when log data changes
- Caching can be disabled per-call via ``use_cache=False``

.. code-block:: python

   # Force fresh render without cache
   rendered_logs, group_info, pagination = log_details.update_logs(
       logs=logs,
       group_info="Log Group: test",
       use_cache=False
   )

Expandable Content
------------------

Log Content Expansion
^^^^^^^^^^^^^^^^^^^^^

Long log entries (>200 characters) are automatically truncated with a "Show more" button:

.. code-block:: python

   # Truncation logic
   char_limit = 200
   is_long = len(log_item) > char_limit
   truncated_item = log_item[:char_limit] + '...' if is_long else log_item

Each log entry includes a hidden store for the full content:

.. code-block:: python

   dcc.Store(
       id={'type': self.get_id('log-full-text'), 'index': index},
       data={'full': log_item, 'truncated': truncated_item, 'is_expanded': False}
   )

Metadata Expansion
^^^^^^^^^^^^^^^^^^

When log entries include hierarchy information (``full_log_group_id``), an expandable
metadata section is shown:

.. code-block:: python

   def _create_expandable_metadata(self, log: Dict[str, Any], index: int):
       metadata = {
           'Full Log Group ID': log.get('full_log_group_id', 'N/A'),
           'Log Group ID': log.get('log_group_id', 'N/A'),
           'Parent Log Group ID': log.get('parent_log_group_id', 'N/A')
       }
       return html.Details(...)

Integration with LogGraphVisualization
--------------------------------------

The ``LogDetailsPanel`` is typically used with ``LogGraphVisualization`` to show
details when a graph node is selected:

.. code-block:: python

   from dash.dependencies import Input, Output, State
   from dash import callback_context

   @app.callback(
       [Output('log-details-logs-container', 'children'),
        Output('log-details-group-info', 'children'),
        Output('log-details-pagination-controls', 'children')],
       [Input('log-graph-cytoscape', 'tapNodeData')],
       [State('logs-store', 'data')]
   )
   def update_log_details(tap_data, logs_data):
       if not tap_data:
           return log_details.update_logs([], "")

       # Get logs for the selected node
       log_group_id = tap_data.get('id')
       node_logs = get_logs_for_group(logs_data, log_group_id)

       return log_details.update_logs(
           logs=node_logs,
           group_info=f"Log Group: {log_group_id} | Total: {len(node_logs)} logs",
           log_group_id=log_group_id
       )

Component IDs
-------------

The component generates the following element IDs:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - ID Pattern
     - Description
   * - ``{component_id}``
     - Root container element
   * - ``{component_id}-group-info``
     - Log group information display
   * - ``{component_id}-loading-indicator``
     - Loading spinner (hidden by default)
   * - ``{component_id}-logs-container``
     - Container for rendered log entries
   * - ``{component_id}-pagination-controls``
     - Pagination button container
   * - ``{component_id}-pagination-state``
     - Hidden store for pagination state
   * - ``{component_id}-load-more-btn``
     - "Load More" button
   * - ``{component_id}-load-all-btn``
     - "Load All" button

Pattern-matching IDs (for individual log entries):

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ID Pattern
     - Description
   * - ``{'type': '{component_id}-log-content', 'index': n}``
     - Pre element containing log content
   * - ``{'type': '{component_id}-expand-btn', 'index': n}``
     - Expand/collapse button for long logs
   * - ``{'type': '{component_id}-log-full-text', 'index': n}``
     - Hidden store with full/truncated text

Styling Customization
---------------------

Default Styles
^^^^^^^^^^^^^^

.. code-block:: python

   def _get_default_style(self) -> Dict[str, Any]:
       return {
           'height': '100%',
           'backgroundColor': '#1E1E1E',
           'overflowY': 'auto',
           'padding': '10px'
       }

Custom Styling
^^^^^^^^^^^^^^

.. code-block:: python

   log_details = LogDetailsPanel(
       component_id="custom-log-details",
       style={
           'height': '600px',
           'backgroundColor': '#2D2D2D',
           'borderRadius': '8px',
           'border': '1px solid #4D4D4F'
       }
   )

Complete Example
----------------

.. code-block:: python

   from dash import Dash, html, dcc
   from dash.dependencies import Input, Output, State
   from science_modeling_tools.ui.dash_interactive.components.log_details import LogDetailsPanel
   from science_modeling_tools.ui.dash_interactive.components.log_graph import LogGraphVisualization

   app = Dash(__name__)

   # Initialize components
   log_graph = LogGraphVisualization(component_id="log-graph")
   log_details = LogDetailsPanel(component_id="log-details")

   # Layout
   app.layout = html.Div([
       dcc.Store(id="logs-store", data={}),
       html.Div([
           html.Div(log_graph.layout(), style={'height': '300px'}),
           html.Div(log_details.layout(), style={'height': '400px'})
       ])
   ])

   @app.callback(
       [Output('log-details-logs-container', 'children'),
        Output('log-details-group-info', 'children'),
        Output('log-details-pagination-controls', 'children')],
       [Input('log-graph-cytoscape', 'tapNodeData')],
       [State('logs-store', 'data')]
   )
   def show_node_details(tap_data, logs_data):
       if not tap_data:
           return log_details.update_logs([], "Select a node to view logs")

       log_group_id = tap_data.get('id')
       node_logs = logs_data.get(log_group_id, {}).get('logs', [])
       node_type = logs_data.get(log_group_id, {}).get('type', 'Unknown')

       return log_details.update_logs(
           logs=node_logs,
           group_info=f"Log Group: {log_group_id} | Type: {node_type} | Total: {len(node_logs)} logs",
           log_group_id=log_group_id
       )

   @app.callback(
       [Output('log-details-logs-container', 'children', allow_duplicate=True),
        Output('log-details-pagination-controls', 'children', allow_duplicate=True)],
       [Input('log-details-load-more-btn', 'n_clicks')],
       [State('log-details-pagination-state', 'data')],
       prevent_initial_call=True
   )
   def load_more_logs(n_clicks, pagination_state):
       if not n_clicks:
           return dash.no_update, dash.no_update

       current_page = pagination_state.get('page', 0)
       rendered_logs, _, pagination = log_details.update_logs(
           logs=log_details.logs,
           group_info="",
           page=current_page + 1,
           show_all=False
       )
       return rendered_logs, pagination

   if __name__ == '__main__':
       app.run_server(debug=True)

See Also
--------

- :doc:`log_graph` - Graph visualization component that works with LogDetailsPanel
- :doc:`tabbed_panel` - Container component for organizing log views
- :doc:`../applications/dash_interactive_app_with_logs` - Complete application using log visualization

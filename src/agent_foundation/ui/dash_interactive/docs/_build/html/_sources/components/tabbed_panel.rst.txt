===========
TabbedPanel
===========

.. module:: agent_foundation.ui.dash_interactive.components.tabbed_panel
   :synopsis: Tab container for chat and log debugging views

The ``TabbedPanel`` component provides a tabbed interface that contains both the
chat interaction and log debugging views.


Overview
========

This component provides:

* **Tab buttons** - Switch between Chat Interaction and Log Debugging
* **Chat tab** - Contains the ``ChatWindow`` component
* **Log Debug tab** - Contains graph visualization and log details
* **Resizable split pane** - Adjustable divider between graph and log details
* **Floating monitor panel** - Draggable panel for real-time monitoring
* **Custom tabs support** - Add custom main tabs and monitor tabs


Class Definition
================

.. code-block:: python

   class TabbedPanel(BaseComponent):
       """
       Component providing tabbed interface for chat and log debugging.

       This component creates two main tabs:
       1. Chat Interaction: Standard chat window
       2. Log Debugging: Split view with graph visualization and log details
       """


Constructor
-----------

.. code-block:: python

   def __init__(
       self,
       component_id: str = "tabbed-panel",
       style: Optional[Dict[str, Any]] = None,
       custom_monitor_tabs: Optional[List[Dict[str, Any]]] = None,
       custom_main_tabs: Optional[List[Dict[str, Any]]] = None
   )

**Parameters:**

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``component_id``
     - str
     - Unique identifier for this component (default: "tabbed-panel")
   * - ``style``
     - Dict
     - Optional CSS style overrides
   * - ``custom_monitor_tabs``
     - List[Dict]
     - Custom tabs for the floating monitor panel
   * - ``custom_main_tabs``
     - List[Dict]
     - Custom tabs for the main tab bar


Custom Tab Structure
====================

Custom tabs are specified as dictionaries:

.. code-block:: python

   custom_tab = {
       'id': 'settings',         # Unique identifier
       'label': 'Settings',      # Button label
       'content': html.Div(...)  # Dash component for tab content
   }


Layout Structure
================

The component generates this structure:

.. code-block:: text

   TabbedPanel
   ├── Tab Buttons
   │   ├── "Chat Interaction" button
   │   ├── "Log Debugging" button
   │   └── Custom main tab buttons...
   ├── Tab Content Area
   │   ├── Chat Tab (visible by default)
   │   │   └── ChatWindow
   │   ├── Log Debug Tab (hidden by default)
   │   │   ├── Log Graph Pane
   │   │   │   └── LogGraphVisualization
   │   │   ├── Resize Divider
   │   │   └── Log Details Pane
   │   │       └── LogDetailsPanel
   │   └── Custom main tabs...
   └── Floating Monitor Panel
       ├── Drag Handle
       ├── Tab Buttons (Logs, Responses, custom...)
       └── Tab Contents
           ├── Logs Tab
           │   ├── Status
           │   ├── Stats
           │   ├── Monitor Log
           │   └── Refresh Button
           ├── Responses Tab
           │   ├── Response Count
           │   ├── Response List
           │   └── Response Details
           └── Custom monitor tabs...


Generated Element IDs
---------------------

With ``component_id="main-panel"``, generates these IDs:

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Element ID
     - Purpose
   * - ``main-panel``
     - Main container
   * - ``main-panel-chat-btn``
     - Chat tab button
   * - ``main-panel-log-btn``
     - Log debugging tab button
   * - ``main-panel-chat-tab``
     - Chat tab content container
   * - ``main-panel-log-debug-tab``
     - Log debug tab content container
   * - ``main-panel-log-graph-pane``
     - Log graph visualization pane
   * - ``main-panel-resize-divider``
     - Draggable resize divider
   * - ``main-panel-log-details-pane``
     - Log details pane
   * - ``main-panel-log-graph-monitor-panel``
     - Floating monitor panel


Child Components
================

The ``TabbedPanel`` creates and contains these child components:

.. code-block:: python

   # Created automatically in __init__
   self.chat_window = ChatWindow(
       component_id=f"{component_id}-chat-window"
   )
   self.log_graph = LogGraphVisualization(
       component_id=f"{component_id}-log-graph"
   )
   self.log_details = LogDetailsPanel(
       component_id=f"{component_id}-log-details"
   )


Methods
=======

layout()
--------

.. code-block:: python

   def layout(self) -> html.Div:
       """Generate the tabbed panel layout."""

Returns a Dash ``html.Div`` containing the complete tabbed interface.


get_callback_inputs()
---------------------

.. code-block:: python

   def get_callback_inputs(self) -> List[Input]:
       """Get list of callback inputs."""

Returns inputs for:

* Chat tab button clicks
* Log tab button clicks
* Execute button clicks


get_callback_outputs()
----------------------

.. code-block:: python

   def get_callback_outputs(self) -> List[Output]:
       """Get list of callback outputs."""

Returns outputs for:

* Chat tab style
* Log debug tab style
* Chat button style
* Log button style


Resizable Split Pane
====================

The Log Debugging tab features a resizable split pane:

.. code-block:: text

   ┌────────────────────────────────────┐
   │     Log Graph Visualization        │
   │         (65% default)              │
   ├════════════════════════════════════┤  ← Drag here
   │        Log Details Panel           │
   │         (35% default)              │
   └────────────────────────────────────┘

**JavaScript Implementation:**

The split pane uses JavaScript for smooth dragging:

* Mousedown on divider starts resize
* Mousemove updates pane heights
* Mouseup ends resize
* Minimum height of 150px for each pane


Floating Monitor Panel
======================

The monitor panel is a floating, draggable widget:

Features:

* **Draggable** - Grab the header to move
* **Tabbed** - Switch between Logs and Responses views
* **Expandable** - Add custom tabs

**Position:**

.. code-block:: python

   style = {
       'position': 'fixed',
       'bottom': '20px',
       'right': '20px',
       'width': '280px',
       'zIndex': '3000'
   }


Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   from agent_foundation.ui.dash_interactive.components.tabbed_panel import (
       TabbedPanel
   )

   panel = TabbedPanel(component_id="main-panel")
   app.layout = html.Div([panel.layout()])


With Custom Monitor Tab
-----------------------

.. code-block:: python

   # Create custom settings tab content
   settings_content = html.Div([
       html.H4("Settings"),
       dcc.Checklist(
           options=[
               {'label': 'Auto-refresh', 'value': 'auto'},
               {'label': 'Show timestamps', 'value': 'timestamps'}
           ],
           value=['auto']
       )
   ])

   panel = TabbedPanel(
       component_id="main-panel",
       custom_monitor_tabs=[
           {
               'id': 'settings',
               'label': 'Settings',
               'content': settings_content
           }
       ]
   )


With Custom Main Tab
--------------------

.. code-block:: python

   # Create custom analytics tab
   analytics_content = html.Div([
       html.H2("Analytics Dashboard"),
       dcc.Graph(id='analytics-graph')
   ])

   panel = TabbedPanel(
       component_id="main-panel",
       custom_main_tabs=[
           {
               'id': 'analytics',
               'label': 'Analytics',
               'content': analytics_content
           }
       ]
   )


Tab Switching Callback
----------------------

.. code-block:: python

   @app.callback(
       [
           Output('main-panel-chat-tab', 'style'),
           Output('main-panel-log-debug-tab', 'style'),
           Output('main-panel-chat-btn', 'style'),
           Output('main-panel-log-btn', 'style')
       ],
       [
           Input('main-panel-chat-btn', 'n_clicks'),
           Input('main-panel-log-btn', 'n_clicks')
       ]
   )
   def switch_tabs(chat_clicks, log_clicks):
       ctx = dash.callback_context

       # Determine which button was clicked
       if not ctx.triggered:
           # Default to chat tab
           button_id = 'main-panel-chat-btn'
       else:
           button_id = ctx.triggered[0]['prop_id'].split('.')[0]

       if button_id == 'main-panel-log-btn':
           # Show log tab
           return (
               {'display': 'none'},      # Hide chat
               {'display': 'block'},     # Show log
               inactive_btn_style,       # Chat btn inactive
               active_btn_style          # Log btn active
           )
       else:
           # Show chat tab
           return (
               {'display': 'block'},     # Show chat
               {'display': 'none'},      # Hide log
               active_btn_style,         # Chat btn active
               inactive_btn_style        # Log btn inactive
           )


Accessing Child Components
--------------------------

.. code-block:: python

   panel = TabbedPanel(component_id="main-panel")

   # Access the chat window
   chat_messages = panel.chat_window.update_messages(messages)

   # Access the log graph
   figure = panel.log_graph.create_figure(hierarchy)

   # Access the log details
   logs, info, pagination = panel.log_details.update_logs(logs, group_info)


Styling
=======

Default Styles
--------------

.. code-block:: python

   default_style = {
       'flex': '1',
       'height': '100vh',
       'backgroundColor': '#343541',
       'display': 'flex',
       'flexDirection': 'column'
   }


Tab Button Styles
-----------------

.. code-block:: python

   # Active tab button
   active_style = {
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

   # Inactive tab button
   inactive_style = {
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


Resize Divider Style
--------------------

.. code-block:: python

   divider_style = {
       'height': '8px',
       'backgroundColor': '#19C37D',
       'cursor': 'row-resize',
       'transition': 'background-color 0.2s'
   }


Integration
===========

The ``TabbedPanel`` is the main content component in ``DashInteractiveAppWithLogs``:

.. code-block:: python

   class DashInteractiveAppWithLogs(DashInteractiveApp):
       def __init__(self, ...):
           self.tabbed_panel = TabbedPanel(
               component_id="main-panel",
               custom_monitor_tabs=custom_monitor_tabs,
               custom_main_tabs=custom_main_tabs
           )
           super().__init__(...)

       def _create_layout(self):
           return html.Div([
               self.chat_history.layout(),   # Left sidebar
               self.tabbed_panel.layout()    # Right panel (tabbed)
           ], style={'display': 'flex'})


See Also
========

* :doc:`base` - BaseComponent class
* :doc:`chat_window` - ChatWindow component
* :doc:`log_graph` - LogGraphVisualization component
* :doc:`log_details` - LogDetailsPanel component
* :doc:`../applications/dash_interactive_app_with_logs` - Extended application class

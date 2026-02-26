API Reference
=============

This is the complete API reference for the ``dash_interactive`` module.

.. contents:: On This Page
   :local:
   :depth: 2

Module Overview
---------------

The ``dash_interactive`` module provides a modular framework for building interactive
chat interfaces with log debugging capabilities using Plotly Dash.

Quick Reference
---------------

Applications
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``DashInteractiveApp``
     - Base chat application with session management
   * - ``DashInteractiveAppWithLogs``
     - Extended app with log visualization
   * - ``QueueBasedDashInteractiveApp``
     - Queue-based async communication app

Components
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``BaseComponent``
     - Abstract base class for all components
   * - ``ChatHistoryList``
     - Sidebar with session management
   * - ``ChatWindow``
     - Chat message display and input
   * - ``TabbedPanel``
     - Tabbed interface container
   * - ``LogGraphVisualization``
     - Graph visualization (Plotly/Cytoscape)
   * - ``LogDetailsPanel``
     - Log details with pagination

Utilities
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``LogCollector``
     - Log collection and graph building

Main Entry Points
-----------------

Importing
^^^^^^^^^

.. code-block:: python

   # Applications
   from science_modeling_tools.ui.dash_interactive import (
       DashInteractiveApp,
       DashInteractiveAppWithLogs,
       QueueBasedDashInteractiveApp
   )

   # Components
   from science_modeling_tools.ui.dash_interactive.components import (
       BaseComponent,
       ChatHistoryList,
       ChatWindow,
       TabbedPanel,
       LogGraphVisualization,
       LogDetailsPanel
   )

   # Utilities
   from science_modeling_tools.ui.dash_interactive.utils import LogCollector

Application Classes
-------------------

DashInteractiveApp
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class DashInteractiveApp(
       title: str = "Interactive Debugger",
       port: int = 8050,
       debug: bool = True,
       message_handler: Optional[Callable[[str], str]] = None
   )

**Methods:**

- ``run(host='0.0.0.0')``: Start the Dash server
- ``set_message_handler(handler)``: Set custom message handler

**See:** :doc:`applications/dash_interactive_app`

DashInteractiveAppWithLogs
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class DashInteractiveAppWithLogs(
       title: str = "Interactive Debugger with Logs",
       port: int = 8050,
       debug: bool = True,
       message_handler: Optional[Callable] = None,
       queue_service = None,
       custom_monitor_tabs: list = None,
       custom_main_tabs: list = None
   )

**Methods:**

- ``run(host='0.0.0.0')``: Start the Dash server
- ``set_message_handler(handler)``: Set custom message handler
- ``set_agent_factory(factory)``: Set agent factory function
- ``add_monitor_tab(tab_id, tab_label, tab_content)``: Add custom monitor tab

**See:** :doc:`applications/dash_interactive_app_with_logs`

QueueBasedDashInteractiveApp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class QueueBasedDashInteractiveApp(
       title: str = "Interactive Debugger with Logs",
       port: int = 8050,
       debug: bool = True,
       message_handler: Optional[Callable] = None,
       response_checker: Optional[Callable[[], Tuple]] = None,
       special_waiting_message: str = "__WAITING_FOR_RESPONSE__",
       custom_monitor_tabs: list = None,
       custom_main_tabs: list = None
   )

**Methods:**

- ``run(host='0.0.0.0')``: Start the Dash server
- ``set_message_handler(handler)``: Set custom message handler
- ``set_response_checker(checker)``: Set response checker function

**See:** :doc:`applications/queue_based_app`

Component Classes
-----------------

BaseComponent
^^^^^^^^^^^^^

.. code-block:: python

   class BaseComponent(
       component_id: str = "base",
       style: Optional[Dict[str, Any]] = None
   )

**Methods:**

- ``layout() -> html.Div``: Generate component layout (abstract)
- ``get_id(suffix='') -> str``: Get prefixed element ID
- ``get_callback_inputs() -> List[Input]``: Get callback inputs
- ``get_callback_outputs() -> List[Output]``: Get callback outputs
- ``_get_default_style() -> Dict``: Get default styling (override)

**See:** :doc:`components/base`

ChatHistoryList
^^^^^^^^^^^^^^^

.. code-block:: python

   class ChatHistoryList(
       component_id: str = "chat-history",
       sessions: Optional[List[Dict]] = None,
       style: Optional[Dict] = None
   )

**Methods:**

- ``layout() -> html.Div``: Generate sidebar layout
- ``update_sessions(sessions) -> List[html.Div]``: Update session list

**See:** :doc:`components/chat_history`

ChatWindow
^^^^^^^^^^

.. code-block:: python

   class ChatWindow(
       component_id: str = "chat-window",
       messages: Optional[List[Dict]] = None,
       style: Optional[Dict] = None
   )

**Methods:**

- ``layout() -> html.Div``: Generate chat window layout
- ``update_messages(messages) -> List[html.Div]``: Update message display

**See:** :doc:`components/chat_window`

TabbedPanel
^^^^^^^^^^^

.. code-block:: python

   class TabbedPanel(
       component_id: str = "main-panel",
       style: Optional[Dict] = None,
       custom_monitor_tabs: list = None,
       custom_main_tabs: list = None
   )

**Attributes:**

- ``chat_window``: Embedded ChatWindow component
- ``log_graph``: Embedded LogGraphVisualization component
- ``log_details``: Embedded LogDetailsPanel component

**See:** :doc:`components/tabbed_panel`

LogGraphVisualization
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class LogGraphVisualization(
       component_id: str = "log-graph",
       hierarchy: Optional[List[Dict]] = None,
       style: Optional[Dict] = None
   )

**Methods:**

- ``layout() -> html.Div``: Generate graph visualization layout
- ``create_figure(hierarchy, label_mode='name')``: Create Plotly figure
- ``create_figure_from_graph(graph_data, label_mode)``: Create from graph data
- ``convert_to_cytoscape_elements(graph_data, label_mode)``: Convert to Cytoscape

**See:** :doc:`components/log_graph`

LogDetailsPanel
^^^^^^^^^^^^^^^

.. code-block:: python

   class LogDetailsPanel(
       component_id: str = "log-details",
       logs: Optional[List[Dict]] = None,
       style: Optional[Dict] = None
   )

**Methods:**

- ``layout() -> html.Div``: Generate log details layout
- ``update_logs(logs, group_info, page, show_all, log_group_id, use_cache)``: Update log display

**See:** :doc:`components/log_details`

Utility Classes
---------------

LogCollector
^^^^^^^^^^^^

.. code-block:: python

   class LogCollector()

**Methods:**

- ``__call__(log_data)``: Collect a log entry
- ``get_graph_structure() -> Dict``: Get graph structure
- ``get_logs_for_node(node_id) -> List``: Get logs for node
- ``get_all_node_ids() -> List``: Get all node IDs
- ``get_log_graph_statistics() -> Dict``: Get graph statistics
- ``clear()``: Clear all logs
- ``to_dict() -> Dict``: Serialize to dictionary
- ``from_dict(data) -> LogCollector``: Deserialize from dictionary
- ``from_json_logs(log_path, pattern) -> LogCollector``: Load from JSON files

**See:** :doc:`utilities/log_collector`

Data Structures
---------------

Session Object
^^^^^^^^^^^^^^

.. code-block:: python

   {
       'id': str,           # Unique session identifier
       'title': str,        # Display title
       'timestamp': str,    # Creation timestamp
       'active': bool       # Whether currently selected
   }

Message Object
^^^^^^^^^^^^^^

.. code-block:: python

   {
       'role': str,         # 'user', 'assistant', or 'system'
       'content': str,      # Message content
       'timestamp': str     # Message timestamp
   }

Log Entry Object
^^^^^^^^^^^^^^^^

.. code-block:: python

   {
       'id': str,           # Debuggable node ID
       'name': str,         # Display name
       'type': str,         # Log type category
       'item': Any,         # Log content
       'parent_ids': List,  # Parent node IDs (optional)
       'timestamp': str,    # ISO timestamp (optional)
       'level': int         # Python logging level (optional)
   }

Graph Structure Object
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   {
       'nodes': [
           {
               'id': str,
               'name': str,
               'label': str,
               'log_count': int,
               'node_type': str
           }
       ],
       'edges': [
           {'source': str, 'target': str}
       ],
       'agent': {
           'id': str,
           'name': str,
           'log_count': int
       }
   }

Custom Tab Object
^^^^^^^^^^^^^^^^^

.. code-block:: python

   {
       'id': str,           # Unique tab identifier
       'label': str,        # Tab button label
       'content': Component # Dash component for tab content
   }

Constants
---------

Color Palette
^^^^^^^^^^^^^

.. code-block:: python

   # Primary colors
   ACCENT_COLOR = '#19C37D'        # Green accent
   BACKGROUND_DARK = '#1E1E1E'     # Dark background
   BACKGROUND_MEDIUM = '#2C2C2C'   # Medium background
   BACKGROUND_SIDEBAR = '#202123'  # Sidebar background
   TEXT_PRIMARY = '#ECECF1'        # Primary text
   TEXT_SECONDARY = '#8E8EA0'      # Secondary text
   BORDER_COLOR = '#4D4D4F'        # Border color

Log Level Colors
^^^^^^^^^^^^^^^^

.. code-block:: python

   LOG_LEVEL_COLORS = {
       10: '#00BFFF',  # DEBUG - cyan
       20: '#19C37D',  # INFO - green
       30: '#FFA500',  # WARNING - orange
       40: '#FF4500',  # ERROR - red
       50: '#DC143C',  # CRITICAL - crimson
   }

Detailed Documentation
----------------------

For detailed documentation on each module, see:

.. toctree::
   :maxdepth: 1

   applications/index
   components/index
   utilities/index
   examples/index

============
Architecture
============

This document describes the architecture and design principles of the Dash Interactive framework.


Design Philosophy
=================

Dash Interactive follows several key design principles:

1. **Modularity** - Components are self-contained and reusable
2. **Separation of Concerns** - UI, state management, and business logic are separated
3. **Extensibility** - Easy to extend through inheritance and composition
4. **Consistency** - Uniform patterns across all components


High-Level Architecture
=======================

Directory Structure
-------------------

.. code-block:: text

   dash_interactive/
   ├── __init__.py                      # Main export: DashInteractiveApp
   ├── dash_interactive_app.py          # Basic chat application
   ├── dash_interactive_app_with_logs.py # Extended app with log debugging
   ├── queue_based_dash_interactive_app.py # Queue-based polling variant
   │
   ├── components/                       # Reusable UI components
   │   ├── __init__.py
   │   ├── base.py                       # Abstract BaseComponent class
   │   ├── chat_history.py               # Left sidebar component
   │   ├── chat_window.py                # Chat messages and input
   │   ├── tabbed_panel.py               # Tab container
   │   ├── log_graph.py                  # Graph visualization
   │   └── log_details.py                # Log entry viewer
   │
   ├── ui_lib/                           # Low-level UI elements
   │   └── chat_window/
   │       ├── input_ui.py
   │       ├── message_ui.py
   │       └── wait_for_response_ui.py
   │
   ├── utils/                            # Utility modules
   │   ├── log_collector.py              # Log collection and graph building
   │   └── dummy_graph_executor.py       # Demo utilities
   │
   └── examples/                         # Demo applications
       ├── basic_chat_demo.py
       └── agent_chat_demo.py


Component Hierarchy
-------------------

The framework uses a layered architecture:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                    Application Layer                            │
   │  ┌──────────────────┐  ┌─────────────────────────────────────┐ │
   │  │ DashInteractive  │  │ DashInteractiveAppWithLogs          │ │
   │  │      App         │──│                                     │ │
   │  └──────────────────┘  │  ┌─────────────────────────────┐   │ │
   │                        │  │ QueueBasedDashInteractiveApp│   │ │
   │                        │  └─────────────────────────────┘   │ │
   │                        └─────────────────────────────────────┘ │
   ├─────────────────────────────────────────────────────────────────┤
   │                    Component Layer                              │
   │  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
   │  │ ChatHistory   │  │ ChatWindow   │  │ TabbedPanel         │  │
   │  │    List       │  │              │  │ ┌─────────────────┐ │  │
   │  └───────────────┘  └──────────────┘  │ │LogGraphVisual.  │ │  │
   │                                       │ ├─────────────────┤ │  │
   │                                       │ │LogDetailsPanel  │ │  │
   │                                       │ └─────────────────┘ │  │
   │                                       └─────────────────────┘  │
   ├─────────────────────────────────────────────────────────────────┤
   │                    Base Layer                                   │
   │  ┌─────────────────────────────────────────────────────────┐   │
   │  │                    BaseComponent                         │   │
   │  │  - component_id    - layout()                           │   │
   │  │  - style           - get_callback_inputs()              │   │
   │  │  - get_id()        - get_callback_outputs()             │   │
   │  └─────────────────────────────────────────────────────────┘   │
   ├─────────────────────────────────────────────────────────────────┤
   │                    Utility Layer                                │
   │  ┌─────────────────┐  ┌──────────────────────────────────┐    │
   │  │  LogCollector   │  │  UI Element Library (ui_lib)     │    │
   │  └─────────────────┘  └──────────────────────────────────┘    │
   └─────────────────────────────────────────────────────────────────┘


Class Inheritance
-----------------

.. code-block:: text

   BaseComponent (Abstract)
       │
       ├── ChatHistoryList
       ├── ChatWindow
       ├── TabbedPanel
       ├── LogGraphVisualization
       └── LogDetailsPanel

   DashInteractiveApp
       │
       └── DashInteractiveAppWithLogs
               │
               └── QueueBasedDashInteractiveApp


Component Architecture
======================

BaseComponent Pattern
---------------------

All UI components inherit from ``BaseComponent``, which provides:

.. code-block:: python

   class BaseComponent(ABC):
       """Abstract base class for all UI components."""

       def __init__(self, component_id: str, style: Dict = None):
           self.component_id = component_id
           self.style = {**self._get_default_style(), **(style or {})}

       @abstractmethod
       def _get_default_style(self) -> Dict[str, Any]:
           """Return default CSS styles."""
           pass

       @abstractmethod
       def layout(self) -> Any:
           """Generate Dash layout."""
           pass

       @abstractmethod
       def get_callback_inputs(self) -> List[Input]:
           """Get Dash Input objects for callbacks."""
           pass

       @abstractmethod
       def get_callback_outputs(self) -> List[Output]:
           """Get Dash Output objects for callbacks."""
           pass

       def get_id(self, suffix: str = "") -> str:
           """Generate unique ID for sub-components."""
           return f"{self.component_id}-{suffix}" if suffix else self.component_id

This pattern ensures:

* **Consistent ID generation** - All components use the same ID scheme
* **Style merging** - Default styles can be overridden
* **Standard interface** - All components provide the same methods


State Management
================

Dash Store Components
---------------------

State is managed using Dash's ``dcc.Store`` components:

.. code-block:: python

   # Session state stores
   dcc.Store(id='sessions-store', data=[])           # List of chat sessions
   dcc.Store(id='current-session-store', data=None)  # Active session ID
   dcc.Store(id='messages-store', data={})           # Messages per session
   dcc.Store(id='log-data-store', data=None)         # Log debugging data

**Data Flow:**

.. code-block:: text

   User Input
       │
       ▼
   ┌───────────────────┐
   │  Dash Callback    │
   │  (Input trigger)  │
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │  State Access     │
   │  (Read stores)    │
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │  Process Logic    │
   │  (Update data)    │
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │  Output Update    │
   │  (Write stores)   │
   └─────────┬─────────┘
             │
             ▼
   UI Re-render


Session Management
------------------

Sessions are managed as a list of dictionaries:

.. code-block:: python

   session = {
       'id': 'session_1_20241216120000',  # Unique identifier
       'title': 'New Chat',                # Display title
       'timestamp': '2024-12-16 12:00',    # Creation time
       'active': True                      # Whether currently selected
   }


Callback Architecture
=====================

Callback Registration
---------------------

Callbacks are registered in the application's ``_register_callbacks()`` method:

.. code-block:: python

   class DashInteractiveApp:
       def _register_callbacks(self):
           self._register_session_callbacks()
           self._register_message_callbacks()

This separation allows subclasses to override specific callback groups while
inheriting others.


Callback Patterns
-----------------

**1. Simple Callback:**

.. code-block:: python

   @self.app.callback(
       Output('output-id', 'children'),
       Input('input-id', 'n_clicks')
   )
   def handle_click(n_clicks):
       return f"Clicked {n_clicks} times"

**2. Callback with State:**

.. code-block:: python

   @self.app.callback(
       Output('messages-store', 'data'),
       Input('send-btn', 'n_clicks'),
       State('input-field', 'value'),
       State('messages-store', 'data')
   )
   def send_message(n_clicks, message, messages):
       messages.append({'role': 'user', 'content': message})
       return messages

**3. Pattern-Matching Callback:**

.. code-block:: python

   @self.app.callback(
       Output('current-session-store', 'data'),
       Input({'type': 'session-item', 'index': ALL}, 'n_clicks')
   )
   def select_session(n_clicks_list):
       ctx = dash.callback_context
       # Handle dynamic session selection
       ...

**4. Clientside Callback:**

.. code-block:: python

   self.app.clientside_callback(
       """
       function(n_intervals, currentData) {
           const isVisible = !document.hidden;
           return {visible: isVisible, timestamp: Date.now()};
       }
       """,
       Output('visibility-store', 'data'),
       Input('interval', 'n_intervals'),
       State('visibility-store', 'data')
   )


Agent Integration
=================

Communication Patterns
----------------------

Dash Interactive supports multiple agent communication patterns:

**1. Synchronous Handler:**

.. code-block:: python

   def handler(message: str) -> str:
       return process(message)  # Blocking call

   app.set_message_handler(handler)

**2. Queue-Based Async:**

.. code-block:: python

   # Agent sends responses to queue
   interactive.response_queue.put(response)

   # UI polls queue periodically
   @self.app.callback(...)
   def poll_responses(...):
       response = queue.get_nowait()
       # Update UI

**3. External Service Integration:**

.. code-block:: python

   app = QueueBasedDashInteractiveApp(
       response_checker=check_external_queue
   )


Log Collection Architecture
===========================

LogCollector Design
-------------------

The ``LogCollector`` class captures hierarchical logs:

.. code-block:: text

   Debuggable Objects
         │
         │ log_data = {
         │     'id': 'node_123',
         │     'parent_ids': ['parent_456'],
         │     'item': 'Log message',
         │     'level': 20
         │ }
         │
         ▼
   ┌─────────────────┐
   │  LogCollector   │
   │  .__call__()    │
   └────────┬────────┘
            │
            ├──────────────────────────────┐
            │                              │
            ▼                              ▼
   ┌─────────────────┐          ┌─────────────────┐
   │   log_groups    │          │  graph_nodes    │
   │   (by node ID)  │          │  graph_edges    │
   └─────────────────┘          └─────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ get_graph_      │
                  │   structure()   │
                  └─────────────────┘
                           │
                           ▼
                  {nodes, edges, agent}


Graph Structure
---------------

The graph structure returned by ``get_graph_structure()``:

.. code-block:: python

   {
       'nodes': [
           {
               'id': 'Agent_123',
               'name': 'Main Agent',
               'label': 'Main Agent',
               'log_count': 15,
               'node_type': 'normal'
           },
           ...
       ],
       'edges': [
           {'source': 'Agent_123', 'target': 'SubTask_456'},
           ...
       ],
       'agent': {
           'id': 'Agent_123',
           'name': 'Main Agent',
           'log_count': 15
       }
   }


Extension Points
================

Creating Custom Components
--------------------------

To create a custom component:

.. code-block:: python

   from agent_foundation.ui.dash_interactive.components.base import BaseComponent
   from dash import html

   class MyCustomComponent(BaseComponent):
       def __init__(self, component_id: str = "my-component", **kwargs):
           super().__init__(component_id)
           # Custom initialization

       def _get_default_style(self):
           return {
               'backgroundColor': '#343541',
               'padding': '20px'
           }

       def layout(self):
           return html.Div(
               id=self.get_id(),
               children=[
                   html.H3("My Component"),
                   # ... more elements
               ],
               style=self.style
           )

       def get_callback_inputs(self):
           return [Input(self.get_id('button'), 'n_clicks')]

       def get_callback_outputs(self):
           return [Output(self.get_id('output'), 'children')]


Adding Custom Tabs
------------------

Add custom tabs to the monitor panel or main panel:

.. code-block:: python

   app = DashInteractiveAppWithLogs(
       custom_monitor_tabs=[
           {
               'id': 'settings',
               'label': 'Settings',
               'content': create_settings_panel()
           }
       ],
       custom_main_tabs=[
           {
               'id': 'analytics',
               'label': 'Analytics',
               'content': create_analytics_dashboard()
           }
       ]
   )


Creating Custom Applications
----------------------------

Extend the base application class:

.. code-block:: python

   class MyCustomApp(DashInteractiveAppWithLogs):
       def __init__(self, **kwargs):
           # Add custom initialization before parent
           self.custom_data = {}
           super().__init__(**kwargs)

       def _register_callbacks(self):
           # Register parent callbacks
           super()._register_callbacks()
           # Add custom callbacks
           self._register_custom_callbacks()

       def _register_custom_callbacks(self):
           @self.app.callback(...)
           def my_custom_callback(...):
               ...


Performance Considerations
==========================

Optimization Strategies
-----------------------

1. **Pagination** - Log details panel uses pagination for large log sets
2. **Caching** - Rendered log entries are cached by log group ID
3. **Clientside Callbacks** - UI-only operations use JavaScript callbacks
4. **Lazy Loading** - Components load data on demand


Best Practices
--------------

* Use ``prevent_initial_call=True`` for callbacks that shouldn't fire on page load
* Use ``allow_duplicate=True`` sparingly - prefer single-output callbacks
* Minimize store data size - don't store rendered components
* Use polling intervals appropriate for your use case (1-3 seconds typical)


See Also
========

* :doc:`components/index` - Detailed component documentation
* :doc:`applications/index` - Application class documentation
* :doc:`examples/custom_components` - Custom component examples

DashInteractiveAppWithLogs
==========================

The ``DashInteractiveAppWithLogs`` extends the base application with log visualization
and debugging capabilities, featuring a tabbed interface for chat and log debugging.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

``DashInteractiveAppWithLogs`` provides:

- All features from :doc:`dash_interactive_app`
- Tabbed interface (Chat Interaction + Log Debugging)
- Real-time log graph visualization (Plotly and Cytoscape)
- Log details panel with pagination
- Response monitoring panel
- Agent thread management for background processing
- Web agent service integration via queue service

Class Reference
---------------

.. py:class:: DashInteractiveAppWithLogs(title="Interactive Debugger with Logs", port=8050, debug=True, message_handler=None, queue_service=None, custom_monitor_tabs=None, custom_main_tabs=None)

   Extended Dash application with chat interface and log debugging.

   :param str title: Application title
   :param int port: Port number for the server
   :param bool debug: Enable debug mode with hot reloading
   :param callable message_handler: Custom message handler function
   :param queue_service: Optional ``StorageBasedQueueService`` for web agent integration
   :param list custom_monitor_tabs: Optional list of custom monitor tab dicts
   :param list custom_main_tabs: Optional list of custom main tab dicts

   **Additional Attributes:**

   .. py:attribute:: tabbed_panel
      :type: TabbedPanel

      The main tabbed interface component.

   .. py:attribute:: log_collector
      :type: LogCollector

      Current log collector instance for capturing execution logs.

   .. py:attribute:: session_agents
      :type: Dict[str, Agent]

      Mapping of session IDs to Agent instances.

   .. py:attribute:: session_threads
      :type: Dict[str, Thread]

      Mapping of session IDs to background threads.

   .. py:attribute:: session_interactives
      :type: Dict[str, QueueInteractive]

      Mapping of session IDs to QueueInteractive instances.

   .. py:attribute:: agent_factory
      :type: Callable

      Factory function for creating new agent instances.

Quick Start
-----------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs

   app = DashInteractiveAppWithLogs(
       title="Agent Debugger",
       port=8050
   )
   app.run()

With Log Visualization
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector

   def handler_with_logs(message: str, session_id: str) -> str:
       log_collector = LogCollector()

       # Start a log group
       log_collector.log_group_start("agent", {"query": message})

       # Add logs during execution
       log_collector.log(20, "llm", {"prompt": message}, "LLM Call")

       # End the group
       log_collector.log_group_end("agent")

       return "Processed your message!"

   app = DashInteractiveAppWithLogs(
       message_handler=handler_with_logs
   )
   app.run()

With Agent Factory
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs

   def create_agent():
       """Factory function that creates a configured agent."""
       agent = MyAgent(config=agent_config)
       return agent

   app = DashInteractiveAppWithLogs(title="Agent Chat")
   app.set_agent_factory(create_agent)
   app.run()

Application Layout
------------------

The application extends the base layout with a tabbed interface:

.. code-block:: text

   ┌────────────────────────────────────────────────────────────────────────┐
   │                                                                        │
   │  ┌───────────────┐  ┌────────────────────────────────────────────────┐│
   │  │               │  │  [Chat Interaction]  [Log Debugging]           ││
   │  │ ChatHistoryList│  │  ┌────────────────────────────────────────────┐││
   │  │               │  │  │                                            │││
   │  │ [+ New Chat]  │  │  │         Tab Content Area                   │││
   │  │               │  │  │                                            │││
   │  │ Session 1     │  │  │   (ChatWindow or Log Debug Panels)         │││
   │  │ Session 2     │  │  │                                            │││
   │  │               │  │  └────────────────────────────────────────────┘││
   │  │               │  └────────────────────────────────────────────────┘│
   │  └───────────────┘                                                    │
   │                                                                        │
   └────────────────────────────────────────────────────────────────────────┘

Log Debugging Tab Layout
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   ┌───────────────────────────────────────────────────────────────────────┐
   │  [Plotly ▾] [Label: name ▾]                                           │
   │  ┌─────────────────────────────────────────────────────────────────┐ │
   │  │                                                                  │ │
   │  │              Log Graph Visualization                             │ │
   │  │                     (65%)                                        │ │
   │  │                                                                  │ │
   │  ├──────────────────────── Resize Handle ──────────────────────────┤ │
   │  │                                                                  │ │
   │  │              Log Details Panel                                   │ │
   │  │                     (35%)                                        │ │
   │  │  [Load More]  [Load All]                                        │ │
   │  │                                                                  │ │
   │  └─────────────────────────────────────────────────────────────────┘ │
   │                                                                       │
   │  ┌──────────────────────┐                                            │
   │  │   Monitor Panel      │ (Draggable)                                │
   │  │   [Responses]        │                                            │
   │  │   Response #1        │                                            │
   │  │   Response #2        │                                            │
   │  └──────────────────────┘                                            │
   └───────────────────────────────────────────────────────────────────────┘

Message Handler Signatures
--------------------------

The application supports multiple handler signatures:

Single Parameter
^^^^^^^^^^^^^^^^

.. code-block:: python

   def handler(message: str) -> str:
       return f"Response to: {message}"

Two Parameters
^^^^^^^^^^^^^^

.. code-block:: python

   def handler(message: str, session_id: str) -> str:
       return f"Session {session_id}: {message}"

Three Parameters
^^^^^^^^^^^^^^^^

.. code-block:: python

   def handler(message: str, session_id: str, all_session_ids: list) -> str:
       return f"Session {session_id} of {len(all_session_ids)} sessions"

Agent Integration
-----------------

Using Agent Factory
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def create_my_agent():
       from my_agents import Agent
       return Agent(
           model="gpt-4",
           interactive=True
       )

   app = DashInteractiveAppWithLogs()
   app.set_agent_factory(create_my_agent)
   app.run()

Agent Lifecycle:

1. User sends first message in a session
2. If no agent exists for session, ``_start_agent_for_session()`` is called
3. Background thread created and ``agent_factory()`` invoked
4. Agent runs in background, responses polled via ``response_queue``

Web Agent Service
^^^^^^^^^^^^^^^^^

For integration with external agent services:

.. code-block:: python

   from queue_service import StorageBasedQueueService

   queue_service = StorageBasedQueueService()

   app = DashInteractiveAppWithLogs(
       queue_service=queue_service,
       message_handler=my_handler
   )
   app.run()

With ``queue_service`` set:

- Messages sent to ``agent_response`` queue are polled
- Responses include ``session_id`` for routing
- ``[AGENT_COMPLETED]`` marker signals task completion

Data Stores
-----------

Additional stores beyond the base application:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Store ID
     - Purpose
   * - ``log-data-store``
     - Hierarchical log structure and log groups
   * - ``page-visibility-store``
     - Browser tab visibility state for polling optimization

Interval Components
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Interval ID
     - Period
     - Purpose
   * - ``response-poll-interval``
     - 1 second
     - Poll agent response queues
   * - ``log-refresh-interval``
     - 3 seconds
     - Auto-refresh log visualization
   * - ``visibility-check-interval``
     - 2 seconds
     - Check browser tab visibility
   * - ``agent-status-poll-interval``
     - 1 second
     - Poll for agent status updates

Log Data Format
---------------

The ``log-data-store`` expects data in this format:

.. code-block:: python

   {
       # For tree hierarchy
       'hierarchy': [
           {
               'id': 'agent-1',
               'name': 'Main Agent',
               'type': 'agent',
               'children': [
                   {'id': 'tool-1', 'name': 'Search', 'type': 'tool', 'children': []}
               ]
           }
       ],

       # OR for DAG graph
       'graph_data': {
           'nodes': [...],
           'edges': [...],
           'agent': {...}
       },

       # Log entries by group ID
       'log_groups': {
           'agent-1': [
               {'level': 20, 'type': 'Agent Step', 'item': 'Processing...', 'timestamp': '10:30:45'}
           ],
           'tool-1': [
               {'level': 10, 'type': 'Tool Call', 'item': 'search("query")', 'timestamp': '10:30:46'}
           ]
       }
   }

Custom Tabs
-----------

Monitor Panel Tabs
^^^^^^^^^^^^^^^^^^

Add custom tabs to the monitor panel (before initialization):

.. code-block:: python

   class MyApp(DashInteractiveAppWithLogs):
       def __init__(self, **kwargs):
           # Prepare custom tabs
           custom_tabs = [
               {
                   'id': 'settings',
                   'label': 'Settings',
                   'content': self._create_settings_content()
               }
           ]
           super().__init__(custom_monitor_tabs=custom_tabs, **kwargs)

       def _create_settings_content(self):
           return html.Div([
               html.Label("Custom Setting"),
               dcc.Input(id="setting-input", type="text")
           ])

Main Panel Tabs
^^^^^^^^^^^^^^^

Add custom tabs to the main tabbed panel:

.. code-block:: python

   custom_main_tabs = [
       {
           'id': 'metrics',
           'label': 'Metrics',
           'content': html.Div("Metrics dashboard content")
       }
   ]

   app = DashInteractiveAppWithLogs(
       custom_main_tabs=custom_main_tabs
   )

Keyboard Shortcuts
------------------

The application includes JavaScript for keyboard shortcuts:

- **Ctrl+Enter** (or **Cmd+Enter** on Mac): Send message

Split Pane Resizing
-------------------

The Log Debugging tab features a resizable split between the graph and details:

- Default split: 65% graph, 35% details
- Drag the green divider to resize
- Minimum height: 150px for each pane
- Position resets when switching tabs

Callbacks Reference
-------------------

Session Callbacks
^^^^^^^^^^^^^^^^^

Inherited from base application.

Message Callbacks
^^^^^^^^^^^^^^^^^

- ``send_message``: Handles message sending with agent/queue service integration
- ``poll_agent_responses``: Polls response queues and updates UI

Tab Callbacks
^^^^^^^^^^^^^

- ``switch_tabs``: Switches between Chat and Log Debugging tabs
- ``toggle_graph_rendering_mode``: Switches between Plotly and Cytoscape

Log Visualization Callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``update_log_graph``: Updates Plotly graph visualization
- ``update_cytoscape_graph``: Updates Cytoscape graph elements
- ``update_log_details``: Shows log details when node clicked (Plotly)
- ``update_log_details_from_cytoscape``: Shows log details when node clicked (Cytoscape)

Pagination Callbacks
^^^^^^^^^^^^^^^^^^^^

- ``handle_load_more``: Loads next page of logs
- ``handle_load_all``: Loads all logs at once

Monitor Panel Callbacks
^^^^^^^^^^^^^^^^^^^^^^^

- ``update_response_list``: Populates response list
- ``show_response_details``: Shows full response on click

Methods Reference
-----------------

.. py:method:: set_agent_factory(factory)

   Set factory function that creates new agent instances.

   :param callable factory: Function that returns a configured Agent

.. py:method:: add_monitor_tab(tab_id, tab_label, tab_content)

   Add a custom tab to the monitor panel. Must be called before layout creation.

   :param str tab_id: Unique identifier for the tab
   :param str tab_label: Button label for the tab
   :param tab_content: Dash component(s) for tab content

.. py:method:: _start_agent_for_session(session_id)

   Start background agent thread for a session.

   :param str session_id: Session identifier

.. py:method:: _run_agent_in_background(session_id)

   Create and run agent in background thread.

   :param str session_id: Session identifier

.. py:method:: _register_polling_callback()

   Register polling callback for agent responses. Override in subclasses.

Complete Example
----------------

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector

   # Global log collector for demonstration
   log_collector = None

   def agent_handler(message: str, session_id: str) -> str:
       global log_collector
       log_collector = LogCollector()

       # Simulate agent execution with logging
       log_collector.log_group_start("agent", {"type": "agent", "query": message})

       # LLM call
       log_collector.log_group_start("llm-1", {"type": "llm_call"})
       log_collector.log(20, "llm", {"model": "gpt-4", "prompt": message}, "LLM Call")
       log_collector.log_group_end("llm-1")

       # Tool usage
       if "search" in message.lower():
           log_collector.log_group_start("tool-1", {"type": "tool"})
           log_collector.log(20, "tool", {"name": "search", "query": message}, "Tool Call")
           log_collector.log_group_end("tool-1")

       log_collector.log_group_end("agent")

       return f"Processed: {message}"

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Agent Debugger",
           port=8050,
           debug=True,
           message_handler=agent_handler
       )

       # Callback to update log data after each message
       @app.app.callback(
           Output('log-data-store', 'data'),
           Input('messages-store', 'data'),
           prevent_initial_call=True
       )
       def update_logs_after_message(messages):
           global log_collector
           if log_collector:
               return {
                   'hierarchy': log_collector.get_hierarchy(),
                   'log_groups': log_collector.log_groups
               }
           return None

       app.run()

See Also
--------

- :doc:`dash_interactive_app` - Base application class
- :doc:`queue_based_app` - Queue-based variant for async communication
- :doc:`../components/tabbed_panel` - TabbedPanel component
- :doc:`../components/log_graph` - LogGraphVisualization component
- :doc:`../components/log_details` - LogDetailsPanel component
- :doc:`../utilities/log_collector` - LogCollector utility

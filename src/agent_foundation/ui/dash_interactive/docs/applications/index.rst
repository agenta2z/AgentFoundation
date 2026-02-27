Applications
============

The ``dash_interactive`` module provides three ready-to-use application classes,
each offering different levels of functionality for building interactive chat interfaces.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

The applications are designed as progressive enhancements:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Application
     - Description
   * - :doc:`dash_interactive_app`
     - Basic chat application with sidebar and chat window
   * - :doc:`dash_interactive_app_with_logs`
     - Extended application with log visualization and debugging tools
   * - :doc:`queue_based_app`
     - Async-capable application using queue-based communication

Choosing an Application
-----------------------

Use this decision tree to select the right application:

.. code-block:: text

   Need agent execution logs?
   ├── No → DashInteractiveApp (basic chat)
   └── Yes → Need async agent communication?
             ├── No → DashInteractiveAppWithLogs (sync with logs)
             └── Yes → QueueBasedDashInteractiveApp (async with logs)

DashInteractiveApp
^^^^^^^^^^^^^^^^^^

Best for:

- Simple chat interfaces
- Direct function-based message handlers
- Minimal complexity requirements

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveApp

   def handle_message(message, session_id, history):
       return f"You said: {message}"

   app = DashInteractiveApp(
       message_handler=handle_message,
       title="Simple Chat"
   )
   app.run(debug=True)

DashInteractiveAppWithLogs
^^^^^^^^^^^^^^^^^^^^^^^^^^

Best for:

- Agent debugging and development
- Visualizing execution flow
- Monitoring LLM calls and tool usage

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs

   def handle_with_logs(message, session_id, history, log_collector):
       log_collector.log_group_start("agent", {"query": message})
       # Agent execution with logging
       log_collector.log_group_end("agent")
       return response

   app = DashInteractiveAppWithLogs(
       message_handler=handle_with_logs
   )
   app.run(debug=True)

QueueBasedDashInteractiveApp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Best for:

- Long-running agent tasks
- Streaming responses
- Background processing
- Production deployments

.. code-block:: python

   from agent_foundation.ui.dash_interactive import QueueBasedDashInteractiveApp
   from queue import Queue

   input_queue = Queue()
   output_queue = Queue()

   app = QueueBasedDashInteractiveApp(
       input_queue=input_queue,
       output_queue=output_queue
   )

Application Class Hierarchy
---------------------------

.. code-block:: text

   DashInteractiveApp
   │
   └── DashInteractiveAppWithLogs
       │
       └── (QueueBasedDashInteractiveApp extends the pattern)

All applications share:

- Dark theme styling (ChatGPT-like)
- Session management
- Chat history persistence
- Component-based architecture

Documentation
-------------

.. toctree::
   :maxdepth: 2

   dash_interactive_app
   dash_interactive_app_with_logs
   queue_based_app

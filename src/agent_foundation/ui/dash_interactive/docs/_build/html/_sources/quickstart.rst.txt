============
Quick Start
============

This guide will help you get started with Dash Interactive quickly.


Prerequisites
=============

Before you begin, ensure you have:

* Python 3.8 or higher
* pip package manager
* A web browser (Chrome, Firefox, Safari, or Edge)


Installation
============

Dash Interactive is part of the ScienceModelingTools package. Install the required dependencies:

.. code-block:: bash

   pip install dash dash-bootstrap-components dash-cytoscape plotly attrs


Basic Usage
===========

1. Simple Echo Chat
-------------------

The simplest way to use Dash Interactive is with a basic chat interface:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp

   # Create a simple message handler
   def echo_handler(message: str) -> str:
       return f"Echo: {message}"

   # Create and run the app
   app = DashInteractiveApp(
       title="Echo Chat",
       port=8050,
       debug=True
   )
   app.set_message_handler(echo_handler)
   app.run()

Open your browser to ``http://localhost:8050`` to see the chat interface.


2. Chat with Custom Logic
-------------------------

You can implement more complex message handling:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp
   import random

   def smart_handler(message: str) -> str:
       message_lower = message.lower()

       if "hello" in message_lower or "hi" in message_lower:
           return random.choice([
               "Hello! How can I help you?",
               "Hi there! What can I do for you?",
               "Hey! Nice to see you!"
           ])
       elif "bye" in message_lower:
           return "Goodbye! Have a great day!"
       elif "?" in message:
           return "That's a great question! Let me think about it..."
       else:
           return f"I received your message: '{message}'"

   app = DashInteractiveApp(title="Smart Chat")
   app.set_message_handler(smart_handler)
   app.run()


3. Chat with Log Debugging
--------------------------

For applications that need to visualize execution logs:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive.dash_interactive_app_with_logs import (
       DashInteractiveAppWithLogs
   )

   app = DashInteractiveAppWithLogs(
       title="Log Debugging Demo",
       port=8050,
       debug=True
   )

   def handler(message: str) -> str:
       # Your logic here - logs will be visualized in the Log Debugging tab
       return f"Processed: {message}"

   app.set_message_handler(handler)
   app.run()


Understanding the Interface
===========================

Main Layout
-----------

The Dash Interactive interface consists of:

1. **Left Sidebar** - Chat history panel

   * "New Chat" button to create new sessions
   * List of previous chat sessions
   * Settings and debug mode toggles

2. **Right Panel** - Main content area

   * For ``DashInteractiveApp``: Chat window only
   * For ``DashInteractiveAppWithLogs``: Tabbed interface with:

     - **Chat Interaction** tab - Standard chat interface
     - **Log Debugging** tab - Graph visualization and log details


Keyboard Shortcuts
------------------

* **Ctrl+Enter** (or **Cmd+Enter** on Mac) - Send message


Session Management
------------------

* Click "New Chat" to start a fresh conversation
* Click on any session in the sidebar to switch to it
* Sessions are preserved during the application runtime


Next Steps
==========

Now that you have the basics, explore more advanced topics:

* :doc:`architecture` - Understand the system design
* :doc:`components/index` - Learn about individual components
* :doc:`examples/agent_integration` - Integrate with LLM agents
* :doc:`examples/log_debugging` - Set up log visualization


Troubleshooting
===============

Common Issues
-------------

**Port already in use**
   Change the port number: ``app = DashInteractiveApp(port=8051)``

**Module not found errors**
   Ensure all dependencies are installed and the package is in your Python path

**UI not updating**
   Enable debug mode: ``app = DashInteractiveApp(debug=True)``

**Callbacks not firing**
   Check browser console for JavaScript errors; ensure component IDs are unique

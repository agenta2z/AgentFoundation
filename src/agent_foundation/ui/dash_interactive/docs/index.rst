.. Dash Interactive documentation master file

==========================================
Dash Interactive Documentation
==========================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/framework-Dash-green.svg
   :alt: Dash Framework

**Dash Interactive** is a modular Dash-based UI framework for building GPT-like chat interfaces
and hierarchical log debugging tools. It provides a comprehensive set of components for creating
interactive debugging applications with real-time log visualization.

.. note::
   This documentation covers version 1.0 of Dash Interactive.


Key Features
============

* **GPT-like Chat Interface** - Clean, modern chat UI similar to ChatGPT
* **Session Management** - Multiple conversation sessions with history
* **Log Debugging** - Interactive visualization of hierarchical logs from WorkGraph execution
* **Graph Visualization** - Clickable tree view of log groups with Plotly and Cytoscape
* **Detailed Log View** - Color-coded log entries with expandable metadata
* **Modular Components** - Reusable, well-structured components following OOP principles
* **Agent Integration** - Easy integration with LLM agents via queue-based communication


Quick Start
===========

Basic Chat Application
----------------------

Here's a minimal example to get started with a basic chat interface:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp

   def my_handler(message: str) -> str:
       return f"You said: {message}"

   app = DashInteractiveApp(
       title="My Chat App",
       port=8050,
       debug=True
   )
   app.set_message_handler(my_handler)
   app.run()

Then navigate to ``http://localhost:8050`` in your browser.


Chat with Log Debugging
-----------------------

For applications that need log debugging capabilities:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive.dash_interactive_app_with_logs import (
       DashInteractiveAppWithLogs
   )

   app = DashInteractiveAppWithLogs(
       title="Log Debugging Demo",
       port=8050,
       debug=True
   )
   app.run()


Installation
============

The dash_interactive module is part of the ScienceModelingTools package. Dependencies include:

* ``dash`` - Core Dash framework
* ``dash-bootstrap-components`` - Bootstrap components for Dash
* ``dash-cytoscape`` - Cytoscape graph visualization
* ``plotly`` - Interactive plotting library
* ``attrs`` - Python classes without boilerplate


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   architecture
   styling

.. toctree::
   :maxdepth: 2
   :caption: Components

   components/index
   components/base
   components/chat_history
   components/chat_window
   components/tabbed_panel
   components/log_graph
   components/log_details

.. toctree::
   :maxdepth: 2
   :caption: Applications

   applications/index
   applications/dash_interactive_app
   applications/dash_interactive_app_with_logs
   applications/queue_based_app

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   utilities/index
   utilities/log_collector

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   examples/index
   examples/basic_chat
   examples/agent_integration
   examples/log_debugging
   examples/custom_components

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Support
=======

For questions, issues, or contributions, please refer to the ScienceModelingTools repository.


License
=======

Part of the ScienceModelingTools package.

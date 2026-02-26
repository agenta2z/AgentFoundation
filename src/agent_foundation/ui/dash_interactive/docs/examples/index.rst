Examples
========

This section provides comprehensive examples demonstrating how to use the
``dash_interactive`` module for various use cases.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

The examples cover:

- **Basic Chat**: Simple chat interface with custom message handlers
- **Agent Integration**: Connecting agents for interactive chat
- **Log Debugging**: Visualizing agent execution with log graphs
- **Custom Components**: Extending the framework with custom UI elements

Getting Started
---------------

All examples assume you have installed the required dependencies:

.. code-block:: bash

   pip install dash dash-bootstrap-components plotly dash-cytoscape

Then import from the module:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import (
       DashInteractiveApp,
       DashInteractiveAppWithLogs,
       QueueBasedDashInteractiveApp
   )

Documentation
-------------

.. toctree::
   :maxdepth: 2

   basic_chat
   agent_integration
   log_debugging
   custom_components

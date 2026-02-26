==========
Components
==========

This section documents the reusable UI components that make up the Dash Interactive framework.


Overview
========

All components inherit from the abstract ``BaseComponent`` class and follow a consistent
pattern for defining layouts, styles, and callbacks.


Component List
==============

.. list-table:: Available Components
   :widths: 30 70
   :header-rows: 1

   * - Component
     - Description
   * - :doc:`base`
     - Abstract base class for all UI components
   * - :doc:`chat_history`
     - Left sidebar with session list and controls
   * - :doc:`chat_window`
     - Main chat interface with messages and input
   * - :doc:`tabbed_panel`
     - Tab container for chat and log debugging views
   * - :doc:`log_graph`
     - Interactive graph visualization for log hierarchy
   * - :doc:`log_details`
     - Detailed log entry viewer with pagination


Component Hierarchy
===================

.. code-block:: text

   BaseComponent (Abstract)
       │
       ├── ChatHistoryList
       │       └── Session list, New Chat button, Settings
       │
       ├── ChatWindow
       │       └── Messages area, Input field, Send button
       │
       ├── TabbedPanel
       │       ├── Chat tab (contains ChatWindow)
       │       ├── Log Debug tab
       │       │       ├── LogGraphVisualization
       │       │       └── LogDetailsPanel
       │       └── Monitor panel (floating)
       │
       ├── LogGraphVisualization
       │       └── Plotly graph, Cytoscape graph, Controls
       │
       └── LogDetailsPanel
               └── Log entries, Pagination, Expand/collapse


Component Patterns
==================

Each component follows these patterns:


ID Generation
-------------

Components use the ``get_id()`` method to generate unique IDs for sub-elements:

.. code-block:: python

   class MyComponent(BaseComponent):
       def layout(self):
           return html.Div([
               html.Button(id=self.get_id('button')),     # "my-component-button"
               html.Div(id=self.get_id('container')),     # "my-component-container"
               html.Input(id=self.get_id('input'))        # "my-component-input"
           ], id=self.get_id())                           # "my-component"


Style Merging
-------------

Default styles are merged with custom styles:

.. code-block:: python

   component = ChatWindow(
       component_id="chat",
       style={'backgroundColor': '#custom'}  # Overrides default
   )


Callback Interface
------------------

Components declare their callback requirements:

.. code-block:: python

   # Get inputs that trigger callbacks
   inputs = component.get_callback_inputs()

   # Get outputs that callbacks update
   outputs = component.get_callback_outputs()

   # Get state values (read but don't trigger)
   states = component.get_callback_states()


Contents
========

.. toctree::
   :maxdepth: 2

   base
   chat_history
   chat_window
   tabbed_panel
   log_graph
   log_details


See Also
========

* :doc:`../architecture` - Overall system architecture
* :doc:`../applications/index` - How components are used in applications
* :doc:`../examples/custom_components` - Creating custom components

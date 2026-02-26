=====================
LogGraphVisualization
=====================

.. module:: science_modeling_tools.ui.dash_interactive.components.log_graph
   :synopsis: Interactive graph visualization for hierarchical log structure

The ``LogGraphVisualization`` component displays hierarchical log groups as an
interactive graph using both Plotly and Cytoscape rendering options.


Overview
========

This component provides:

* **Dual rendering modes** - Plotly (static pan/zoom) and Cytoscape (draggable nodes)
* **Hierarchical tree layout** - Visualize parent-child relationships
* **Color-coded nodes** - Normal vs exit nodes
* **Interactive clicking** - Click nodes to view their logs
* **Export options** - Download as PNG or JSON
* **Label mode switching** - Show node names or IDs


Class Definition
================

.. code-block:: python

   class LogGraphVisualization(BaseComponent):
       """
       Component for visualizing hierarchical log groups as a graph.

       This component displays the hierarchical structure of log groups
       using an interactive tree/graph visualization.
       """


Constructor
-----------

.. code-block:: python

   def __init__(
       self,
       component_id: str = "log-graph",
       graph_data: Optional[Dict[str, Any]] = None,
       style: Optional[Dict[str, Any]] = None
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
     - Unique identifier (default: "log-graph")
   * - ``graph_data``
     - Dict
     - Hierarchical graph structure data
   * - ``style``
     - Dict
     - Optional CSS style overrides


Graph Data Structure
====================

The component accepts graph data in this format:

.. code-block:: python

   graph_data = {
       'nodes': [
           {
               'id': 'Agent_123',        # Unique node ID
               'name': 'Main Agent',      # Display name
               'log_count': 15,           # Number of logs
               'x': 0,                    # X position
               'y': 0,                    # Y position
               'level': 0,                # Hierarchy level
               'node_type': 'normal'      # 'normal' or 'exit'
           },
           ...
       ],
       'edges': [
           {
               'source': 'Agent_123',     # Parent node ID
               'target': 'SubTask_456',   # Child node ID
               'source_pos': (0, 0),      # Source position
               'target_pos': (100, -100)  # Target position
           },
           ...
       ]
   }


Layout Structure
================

.. code-block:: text

   LogGraphVisualization
   ├── Header
   │   ├── "Log Group Graph" title
   │   ├── Rendering Mode Dropdown (Plotly/Cytoscape)
   │   └── Label Mode Dropdown (Name/ID)
   ├── Selected Info Display
   ├── Plotly Container (visible by default)
   │   ├── dcc.Graph
   │   └── Loading Overlay
   └── Cytoscape Container (hidden by default)
       ├── cyto.Cytoscape
       ├── Loading Overlay
       └── Control Buttons
           ├── Reset View
           ├── Fit to Screen
           ├── Download PNG
           └── Download JSON


Generated Element IDs
---------------------

With ``component_id="log-graph"``:

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Element ID
     - Purpose
   * - ``log-graph``
     - Main container
   * - ``log-graph-graph``
     - Plotly graph component
   * - ``log-graph-cytoscape``
     - Cytoscape graph component
   * - ``log-graph-rendering-mode``
     - Rendering mode dropdown
   * - ``log-graph-label-mode``
     - Label mode dropdown
   * - ``log-graph-selected-info``
     - Selected node info display
   * - ``log-graph-plotly-container``
     - Plotly container div
   * - ``log-graph-cytoscape-container``
     - Cytoscape container div
   * - ``log-graph-cytoscape-reset-btn``
     - Reset view button
   * - ``log-graph-cytoscape-fit-btn``
     - Fit to screen button


Methods
=======

layout()
--------

.. code-block:: python

   def layout(self) -> html.Div:
       """Generate the log graph layout with dual rendering options."""


create_figure()
---------------

.. code-block:: python

   def create_figure(
       self,
       hierarchy: List[Dict[str, Any]],
       label_mode: str = 'name'
   ) -> go.Figure:
       """
       Create a figure from hierarchical log structure.

       Args:
           hierarchy: Hierarchical structure from LogCollector
           label_mode: 'name' for log name or 'id' for node ID

       Returns:
           Plotly Figure object
       """


create_figure_from_graph()
--------------------------

.. code-block:: python

   def create_figure_from_graph(
       self,
       graph_data: Dict[str, Any],
       label_mode: str = 'name'
   ) -> go.Figure:
       """
       Create a figure directly from graph data (nodes and edges).

       Used for WorkGraph structures where we have a DAG with
       convergence nodes, rather than a pure tree hierarchy.

       Args:
           graph_data: Dictionary with 'nodes', 'edges', and 'agent'
           label_mode: 'name' for log name or 'id' for node ID

       Returns:
           Plotly Figure object
       """


update_graph()
--------------

.. code-block:: python

   def update_graph(
       self,
       graph_data: Dict[str, Any],
       label_mode: str = 'name'
   ) -> go.Figure:
       """
       Update the graph with new data.

       Args:
           graph_data: New graph structure data
           label_mode: 'name' or 'id'

       Returns:
           Updated Plotly Figure
       """


convert_to_cytoscape_elements()
-------------------------------

.. code-block:: python

   def convert_to_cytoscape_elements(
       self,
       graph_data: Dict[str, Any],
       label_mode: str = 'name'
   ) -> List[Dict[str, Any]]:
       """
       Convert graph data to Cytoscape elements format.

       Args:
           graph_data: Dictionary with 'nodes' and 'edges'
           label_mode: 'name' or 'id'

       Returns:
           List of Cytoscape element dictionaries
       """


Static Methods
--------------

.. code-block:: python

   @staticmethod
   def process_hierarchy_to_graph(
       hierarchy: List[Dict[str, Any]]
   ) -> Dict[str, Any]:
       """
       Process hierarchical log structure into graph visualization data.

       Args:
           hierarchy: Hierarchical structure from LogCollector

       Returns:
           Dictionary with 'nodes' and 'edges'
       """


Rendering Modes
===============

Plotly Mode (Default)
---------------------

**Features:**

* Static node positions
* Pan and zoom with mouse drag
* Hover information on nodes
* Click to select nodes
* Standard Plotly modebar

**Best for:**

* Quick overview of graph structure
* Screenshots and exports
* Consistent node layout


Cytoscape Mode
--------------

**Features:**

* Draggable nodes
* Pan and zoom
* Click to select
* Custom control buttons
* Node repositioning preserved during session

**Best for:**

* Interactive exploration
* Rearranging node positions
* Detailed graph inspection


Node Types
==========

Normal Nodes
------------

Regular execution nodes:

.. code-block:: python

   # Plotly styling
   marker = {
       'size': 30,
       'color': '#19C37D',  # Green
       'line': {'color': '#ECECF1', 'width': 2}
   }

Exit Nodes
----------

Exit/completion points (created for ``AgentWorkstreamCompleted``):

.. code-block:: python

   # Plotly styling
   marker = {
       'size': 35,
       'color': '#FF6B6B',   # Red/coral
       'symbol': 'diamond',
       'line': {'color': '#FFD93D', 'width': 2}  # Yellow border
   }


Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive.components.log_graph import (
       LogGraphVisualization
   )

   graph = LogGraphVisualization(component_id="my-graph")
   app.layout = html.Div([graph.layout()])


From Log Collector
------------------

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive.utils.log_collector import LogCollector

   # Collect logs
   collector = LogCollector()
   collector({'id': 'Agent', 'name': 'Main Agent', 'item': 'Start'})
   collector({'id': 'Task1', 'name': 'Task 1', 'item': 'Work', 'parent_ids': ['Agent']})

   # Get graph structure
   graph_structure = collector.get_graph_structure()

   # Create visualization
   graph_viz = LogGraphVisualization()
   figure = graph_viz.create_figure_from_graph(graph_structure)


Updating Graph via Callback
---------------------------

.. code-block:: python

   @app.callback(
       Output('my-graph-graph', 'figure'),
       [
           Input('log-data-store', 'data'),
           Input('my-graph-label-mode', 'value')
       ]
   )
   def update_log_graph(log_data, label_mode):
       if not log_data:
           return graph_viz.create_figure([])

       if 'graph_data' in log_data:
           return graph_viz.create_figure_from_graph(
               log_data['graph_data'],
               label_mode or 'name'
           )

       return graph_viz.create_figure(
           log_data.get('hierarchy', []),
           label_mode or 'name'
       )


Handling Node Clicks
--------------------

.. code-block:: python

   @app.callback(
       Output('log-details-container', 'children'),
       Input('my-graph-graph', 'clickData'),
       State('log-data-store', 'data')
   )
   def handle_node_click(click_data, log_data):
       if not click_data or not log_data:
           return "Select a node to view logs"

       # Get clicked node ID
       point = click_data['points'][0]
       node_id = point.get('customdata', [''])[0]

       # Get logs for this node
       if node_id in log_data.get('log_groups', {}):
           logs = log_data['log_groups'][node_id]
           return f"Node {node_id}: {len(logs)} log entries"

       return "No logs found for this node"


Toggling Rendering Mode
-----------------------

.. code-block:: python

   @app.callback(
       [
           Output('my-graph-plotly-container', 'style'),
           Output('my-graph-cytoscape-container', 'style')
       ],
       Input('my-graph-rendering-mode', 'value')
   )
   def toggle_rendering(mode):
       if mode == 'plotly':
           return (
               {'display': 'block', 'height': 'calc(100% - 120px)'},
               {'display': 'none', 'height': 'calc(100% - 120px)'}
           )
       else:  # cytoscape
           return (
               {'display': 'none', 'height': 'calc(100% - 120px)'},
               {'display': 'block', 'height': 'calc(100% - 120px)'}
           )


Cytoscape Stylesheet
====================

The component uses this stylesheet for Cytoscape rendering:

.. code-block:: python

   stylesheet = [
       # Normal nodes
       {
           'selector': 'node',
           'style': {
               'content': 'data(label)',
               'background-color': '#19C37D',
               'color': '#ECECF1',
               'border-width': '2px',
               'border-color': '#ECECF1'
           }
       },
       # Exit nodes
       {
           'selector': 'node.exit',
           'style': {
               'background-color': '#FF6B6B',
               'border-color': '#FFD93D',
               'shape': 'octagon'
           }
       },
       # Edges
       {
           'selector': 'edge',
           'style': {
               'curve-style': 'bezier',
               'target-arrow-shape': 'triangle',
               'target-arrow-color': '#565869',
               'line-color': '#565869'
           }
       }
   ]


See Also
========

* :doc:`base` - BaseComponent class
* :doc:`log_details` - LogDetailsPanel component
* :doc:`../utilities/log_collector` - LogCollector utility

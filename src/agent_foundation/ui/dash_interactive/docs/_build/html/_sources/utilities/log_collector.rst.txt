LogCollector
============

The ``LogCollector`` utility captures logs from debuggable objects and builds
execution graph structures for visualization.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

``LogCollector`` provides:

- Chronological log storage
- Log grouping by debuggable ID
- Automatic graph structure building from parent-child relationships
- Serialization/deserialization support
- Statistics calculation

It integrates with the log visualization components (:doc:`../components/log_graph`,
:doc:`../components/log_details`) to provide interactive debugging experiences.

Class Reference
---------------

.. py:class:: LogCollector()

   Collects logs from debuggable objects and builds execution graph structure.

   .. py:attribute:: logs
      :type: List[Dict[str, Any]]

      All collected logs in chronological order.

   .. py:attribute:: log_groups
      :type: Dict[str, List[Dict[str, Any]]]

      Logs organized by debuggable node ID.

   .. py:attribute:: graph_nodes
      :type: Dict[str, Dict[str, Any]]

      Graph nodes indexed by debuggable ID.

   .. py:attribute:: graph_edges
      :type: Set[Tuple[str, str]]

      Set of ``(parent_id, child_id)`` edge tuples.

Quick Start
-----------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector

   # Create collector
   collector = LogCollector()

   # Add log entries
   collector({
       'id': 'Agent_1',
       'name': 'Main Agent',
       'type': 'AgentStarted',
       'item': {'query': 'Hello world'}
   })

   collector({
       'id': 'LLM_1',
       'name': 'GPT-4',
       'type': 'LLMCall',
       'item': {'prompt': 'Process query...'},
       'parent_ids': ['Agent_1']  # Links to parent node
   })

   # Get graph structure for visualization
   graph = collector.get_graph_structure()
   print(f"Nodes: {len(graph['nodes'])}")
   print(f"Edges: {len(graph['edges'])}")

Callable Interface
^^^^^^^^^^^^^^^^^^

``LogCollector`` is callable, allowing it to be used directly as a logging callback:

.. code-block:: python

   def debuggable_operation(log_callback):
       log_callback({
           'id': 'operation_1',
           'name': 'My Operation',
           'item': 'Processing...'
       })

   collector = LogCollector()
   debuggable_operation(collector)  # Pass collector as callback

Log Entry Format
----------------

Required Fields
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``id``
     - str
     - Unique identifier for the debuggable node
   * - ``name``
     - str
     - Display name for the node

Optional Fields
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``type``
     - str
     - Log type category (e.g., "LLMCall", "ToolUse", "AgentStep")
   * - ``item``
     - Any
     - Log content (string, dict, list, etc.)
   * - ``parent_ids``
     - List[str]
     - IDs of parent nodes for graph edges
   * - ``timestamp``
     - str
     - ISO format timestamp (auto-generated if not provided)
   * - ``level``
     - int
     - Python logging level (10=DEBUG, 20=INFO, etc.)

Example Log Entry
^^^^^^^^^^^^^^^^^

.. code-block:: python

   {
       'id': 'Tool_search_1',
       'name': 'Web Search',
       'type': 'ToolCall',
       'item': {
           'tool': 'search',
           'query': 'Python documentation',
           'results': ['result1', 'result2']
       },
       'parent_ids': ['Agent_1', 'LLM_1'],  # Multiple parents allowed
       'timestamp': '2024-01-15T10:30:45.123456',
       'level': 20
   }

Graph Building
--------------

How Graph Structure is Built
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Each unique ``id`` becomes a node in the graph
2. Entries in ``parent_ids`` create edges from parent to child
3. If a parent node doesn't exist, a placeholder is created
4. ``AgentWorkstreamCompleted`` type creates exit nodes

Graph Structure Output
^^^^^^^^^^^^^^^^^^^^^^

The ``get_graph_structure()`` method returns:

.. code-block:: python

   {
       'nodes': [
           {
               'id': 'Agent_1',
               'name': 'Main Agent',
               'label': 'Main Agent',  # For compatibility
               'log_count': 5,
               'node_type': 'normal'
           },
           {
               'id': 'LLM_1',
               'name': 'GPT-4',
               'label': 'GPT-4',
               'log_count': 3,
               'node_type': 'normal'
           }
       ],
       'edges': [
           {'source': 'Agent_1', 'target': 'LLM_1'}
       ],
       'agent': {
           'id': 'Agent_1',
           'name': 'Main Agent',
           'log_count': 5
       }
   }

Node Types
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``normal``
     - Standard execution node
   * - ``exit``
     - Symbolic exit node (created from ``AgentWorkstreamCompleted``)

Methods Reference
-----------------

Collecting Logs
^^^^^^^^^^^^^^^

.. py:method:: __call__(log_data)

   Collect a log entry. Automatically adds timestamp if not present.

   :param dict log_data: Log data dictionary with ``id``, ``name``, and optional fields

.. py:method:: clear()

   Clear all collected logs and graph data.

Querying Data
^^^^^^^^^^^^^

.. py:method:: get_graph_structure()

   Get the complete graph structure for visualization.

   :return: Dictionary with ``nodes``, ``edges``, and ``agent`` keys

.. py:method:: get_logs_for_node(node_id)

   Get all logs for a specific debuggable node.

   :param str node_id: The debuggable ID
   :return: List of log entries for that node

.. py:method:: get_all_node_ids()

   Get all debuggable node IDs (sorted).

   :return: List of all node IDs

.. py:method:: get_log_graph_statistics()

   Calculate statistics about the log graph structure.

   :return: Dictionary with node_count, edge_count, max_depth, total_log_count, etc.

Serialization
^^^^^^^^^^^^^

.. py:method:: to_dict()

   Convert collected logs to a dictionary for storage.

   :return: Dictionary with logs, log_groups, graph_nodes, graph_edges, graph_structure

.. py:staticmethod:: from_dict(data)

   Create a LogCollector from a dictionary.

   :param dict data: Dictionary with logs and metadata
   :return: LogCollector instance

.. py:staticmethod:: from_json_logs(log_path, json_file_pattern='*')

   Create a LogCollector by reading logs from JSON files.

   :param str log_path: Path to a JSON file or folder containing JSON files
   :param str json_file_pattern: File pattern for JSON file search
   :return: LogCollector with all logs loaded and graph built

Integration Examples
--------------------

With DashInteractiveAppWithLogs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector

   collector = LogCollector()

   def handler(message: str) -> str:
       # Log the processing
       collector({
           'id': 'Agent_1',
           'name': 'Chat Agent',
           'type': 'UserMessage',
           'item': message
       })

       # Process and log LLM call
       collector({
           'id': 'LLM_1',
           'name': 'GPT-4',
           'type': 'LLMCall',
           'item': {'prompt': message},
           'parent_ids': ['Agent_1']
       })

       response = "Processed response"

       collector({
           'id': 'Agent_1',
           'name': 'Chat Agent',
           'type': 'AgentResponse',
           'item': response
       })

       return response

   app = DashInteractiveAppWithLogs(message_handler=handler)

   # Callback to update log data
   @app.app.callback(
       Output('log-data-store', 'data'),
       Input('messages-store', 'data'),
       prevent_initial_call=True
   )
   def update_log_data(_):
       return {
           'graph_data': collector.get_graph_structure(),
           'log_groups': dict(collector.log_groups)
       }

   app.run()

Loading Logs from Files
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load from a single JSON file
   collector = LogCollector.from_json_logs("logs/execution.json")

   # Load from a directory of JSON files
   collector = LogCollector.from_json_logs("logs/agent_runs/")

   # Load with file pattern
   collector = LogCollector.from_json_logs(
       "logs/",
       json_file_pattern="agent_*.json"
   )

   # Get statistics
   stats = collector.get_log_graph_statistics()
   print(f"Total nodes: {stats['node_count']}")
   print(f"Total edges: {stats['edge_count']}")
   print(f"Max depth: {stats['max_depth']}")
   print(f"Total logs: {stats['total_log_count']}")

Saving and Restoring
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import json

   # Save collector state
   with open("collector_state.json", "w") as f:
       json.dump(collector.to_dict(), f, indent=2)

   # Restore collector state
   with open("collector_state.json", "r") as f:
       data = json.load(f)

   restored_collector = LogCollector.from_dict(data)

Statistics Example
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   collector = LogCollector()

   # Add some logs...
   collector({'id': 'A', 'name': 'Root', 'item': 'start'})
   collector({'id': 'B', 'name': 'Child1', 'item': 'step1', 'parent_ids': ['A']})
   collector({'id': 'C', 'name': 'Child2', 'item': 'step2', 'parent_ids': ['A']})
   collector({'id': 'D', 'name': 'Grandchild', 'item': 'step3', 'parent_ids': ['B']})

   # Get statistics
   stats = collector.get_log_graph_statistics()
   print(stats)
   # Output:
   # {
   #     'node_count': 4,
   #     'edge_count': 3,
   #     'total_log_count': 4,
   #     'max_depth': 2,
   #     'agent_id': 'A',
   #     'agent_name': 'Root'
   # }

Graph Visualization Integration
-------------------------------

The ``LogCollector`` output is designed to work with
:doc:`../components/log_graph`:

.. code-block:: python

   from agent_foundation.ui.dash_interactive.components.log_graph import LogGraphVisualization

   log_graph = LogGraphVisualization()
   graph_structure = collector.get_graph_structure()

   # Create Plotly figure
   figure = log_graph.create_figure_from_graph(graph_structure, label_mode='name')

   # Or create Cytoscape elements
   elements = log_graph.convert_to_cytoscape_elements(
       log_graph._process_dag_to_graph(
           graph_structure['nodes'],
           graph_structure['edges'],
           graph_structure['agent']
       ),
       label_mode='name'
   )

Best Practices
--------------

1. **Consistent IDs**: Use consistent ID patterns (e.g., ``{Type}_{unique_id}``)

2. **Meaningful Names**: Use descriptive names for better visualization

3. **Parent Tracking**: Always include ``parent_ids`` to build accurate graphs

4. **Type Categories**: Use consistent type strings for filtering/styling

5. **Timestamp Handling**: Let LogCollector auto-generate timestamps unless you need precise control

See Also
--------

- :doc:`../components/log_graph` - LogGraphVisualization component
- :doc:`../components/log_details` - LogDetailsPanel component
- :doc:`../applications/dash_interactive_app_with_logs` - Application with log visualization

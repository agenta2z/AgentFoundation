Log Debugging Example
======================

This example demonstrates how to visualize agent execution with log graphs
and detailed log inspection.

.. contents:: On This Page
   :local:
   :depth: 2

Basic Log Visualization
-----------------------

Simple example showing log collection and visualization:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector
   from dash.dependencies import Input, Output

   # Global log collector
   collector = LogCollector()

   def handler(message: str, session_id: str) -> str:
       """Handler that logs execution steps."""
       agent_id = f'Agent_{session_id[:8]}'

       # Start agent
       collector({
           'id': agent_id,
           'name': 'Chat Agent',
           'type': 'AgentStarted',
           'item': {'query': message},
           'level': 20
       })

       # Simulate processing steps
       collector({
           'id': f'Step_1_{session_id[:8]}',
           'name': 'Parse Input',
           'type': 'ProcessingStep',
           'item': {'input': message},
           'parent_ids': [agent_id],
           'level': 10
       })

       collector({
           'id': f'Step_2_{session_id[:8]}',
           'name': 'Generate Response',
           'type': 'ProcessingStep',
           'item': {'status': 'completed'},
           'parent_ids': [f'Step_1_{session_id[:8]}'],
           'level': 20
       })

       response = f"Processed: {message}"

       # Log completion
       collector({
           'id': agent_id,
           'name': 'Chat Agent',
           'type': 'AgentCompleted',
           'item': {'response': response},
           'level': 20
       })

       return response

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Log Debugger",
           message_handler=handler
       )

       # Update log visualization after each message
       @app.app.callback(
           Output('log-data-store', 'data'),
           Input('messages-store', 'data'),
           prevent_initial_call=True
       )
       def update_logs(_):
           return {
               'graph_data': collector.get_graph_structure(),
               'log_groups': dict(collector.log_groups)
           }

       app.run()

Hierarchical Agent Logging
--------------------------

Example with nested agent calls:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector
   from dash.dependencies import Input, Output
   import time
   import random

   collector = LogCollector()

   def simulate_llm_call(prompt: str, parent_id: str) -> str:
       """Simulate an LLM call with logging."""
       llm_id = f'LLM_{random.randint(1000, 9999)}'

       collector({
           'id': llm_id,
           'name': 'GPT-4',
           'type': 'LLMCallStart',
           'item': {'prompt': prompt[:50] + '...'},
           'parent_ids': [parent_id],
           'level': 10
       })

       time.sleep(0.3)  # Simulate API call

       response = f"LLM response to: {prompt[:20]}..."

       collector({
           'id': llm_id,
           'name': 'GPT-4',
           'type': 'LLMCallEnd',
           'item': {'response': response, 'tokens': random.randint(50, 200)},
           'parent_ids': [parent_id],
           'level': 20
       })

       return response

   def simulate_tool_call(tool_name: str, args: dict, parent_id: str) -> str:
       """Simulate a tool call with logging."""
       tool_id = f'Tool_{tool_name}_{random.randint(1000, 9999)}'

       collector({
           'id': tool_id,
           'name': tool_name,
           'type': 'ToolCallStart',
           'item': {'args': args},
           'parent_ids': [parent_id],
           'level': 10
       })

       time.sleep(0.2)  # Simulate tool execution

       result = f"Result from {tool_name}"

       collector({
           'id': tool_id,
           'name': tool_name,
           'type': 'ToolCallEnd',
           'item': {'result': result},
           'parent_ids': [parent_id],
           'level': 20
       })

       return result

   def handler(message: str, session_id: str) -> str:
       """Handler with hierarchical logging."""
       agent_id = f'Agent_{session_id[:8]}'

       # Start main agent
       collector({
           'id': agent_id,
           'name': 'ReAct Agent',
           'type': 'AgentStarted',
           'item': {'query': message},
           'level': 20
       })

       # First LLM call to plan
       plan = simulate_llm_call(f"Plan for: {message}", agent_id)

       # Execute tools based on keywords
       if 'search' in message.lower():
           simulate_tool_call('WebSearch', {'query': message}, agent_id)
       if 'calculate' in message.lower():
           simulate_tool_call('Calculator', {'expression': message}, agent_id)
       if 'file' in message.lower():
           simulate_tool_call('FileReader', {'path': '/data'}, agent_id)

       # Final LLM call
       final = simulate_llm_call("Generate final response", agent_id)

       # Complete agent
       collector({
           'id': agent_id,
           'name': 'ReAct Agent',
           'type': 'AgentWorkstreamCompleted',  # Creates exit node
           'item': {'result': 'success'},
           'level': 20
       })

       return f"Completed task: {message}"

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Hierarchical Logs",
           message_handler=handler
       )

       @app.app.callback(
           Output('log-data-store', 'data'),
           Input('messages-store', 'data'),
           prevent_initial_call=True
       )
       def update_logs(_):
           return {
               'graph_data': collector.get_graph_structure(),
               'log_groups': dict(collector.log_groups)
           }

       app.run()

Loading Logs from Files
-----------------------

Visualize previously saved execution logs:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector
   from dash.dependencies import Input, Output
   import os

   # Load logs from file or directory
   LOG_PATH = "logs/agent_execution/"

   collector = None
   if os.path.exists(LOG_PATH):
       collector = LogCollector.from_json_logs(LOG_PATH)
       print(f"Loaded {len(collector.logs)} log entries")
   else:
       collector = LogCollector()
       print(f"Log path not found: {LOG_PATH}")

   def handler(message: str, session_id: str) -> str:
       """Simple handler for testing."""
       return f"Viewer mode - logs loaded from: {LOG_PATH}"

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Log Viewer",
           message_handler=handler
       )

       # Set initial log data from loaded logs
       @app.app.callback(
           Output('log-data-store', 'data'),
           Input('current-session-store', 'data'),
           prevent_initial_call=False
       )
       def set_initial_logs(_):
           if collector and collector.logs:
               return {
                   'graph_data': collector.get_graph_structure(),
                   'log_groups': dict(collector.log_groups)
               }
           return None

       # Print statistics
       if collector:
           stats = collector.get_log_graph_statistics()
           print(f"Graph statistics:")
           print(f"  Nodes: {stats['node_count']}")
           print(f"  Edges: {stats['edge_count']}")
           print(f"  Max depth: {stats['max_depth']}")

       app.run()

Real-Time Log Updates
---------------------

Pattern for updating logs during long-running operations:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector
   from dash.dependencies import Input, Output
   from threading import Thread
   import time

   collector = LogCollector()
   processing_complete = False

   def background_processor(message: str, agent_id: str):
       """Background task that logs as it processes."""
       global processing_complete

       steps = ['Parsing', 'Analyzing', 'Processing', 'Generating']

       for i, step in enumerate(steps):
           step_id = f'{step}_{agent_id}'

           collector({
               'id': step_id,
               'name': step,
               'type': 'Step',
               'item': {'progress': f'{i+1}/{len(steps)}'},
               'parent_ids': [agent_id] if i == 0 else [f'{steps[i-1]}_{agent_id}'],
               'level': 20
           })

           time.sleep(1)  # Simulate work

       processing_complete = True

   def handler(message: str, session_id: str) -> str:
       global processing_complete
       processing_complete = False

       agent_id = f'Agent_{session_id[:8]}'

       collector({
           'id': agent_id,
           'name': 'Background Agent',
           'type': 'Started',
           'item': {'message': message},
           'level': 20
       })

       # Start background processing
       thread = Thread(target=background_processor, args=(message, agent_id))
       thread.start()

       return "Processing started - watch the Log Debugging tab!"

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Real-Time Logs",
           message_handler=handler
       )

       # Polling callback for log updates
       @app.app.callback(
           Output('log-data-store', 'data'),
           Input('log-refresh-interval', 'n_intervals'),
           prevent_initial_call=False
       )
       def refresh_logs(_):
           return {
               'graph_data': collector.get_graph_structure(),
               'log_groups': dict(collector.log_groups)
           }

       app.run()

Custom Log Styling
------------------

Using log levels for visual differentiation:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector
   from dash.dependencies import Input, Output

   collector = LogCollector()

   def handler(message: str, session_id: str) -> str:
       agent_id = f'Agent_{session_id[:8]}'

       # INFO level - green
       collector({
           'id': agent_id,
           'name': 'Main Agent',
           'type': 'Info',
           'item': 'Starting process',
           'level': 20  # INFO
       })

       # DEBUG level - cyan
       collector({
           'id': f'Debug_{agent_id}',
           'name': 'Debug Info',
           'type': 'Debug',
           'item': {'internal_state': 'value'},
           'parent_ids': [agent_id],
           'level': 10  # DEBUG
       })

       # WARNING level - orange
       if len(message) > 50:
           collector({
               'id': f'Warning_{agent_id}',
               'name': 'Warning',
               'type': 'Warning',
               'item': 'Input is quite long',
               'parent_ids': [agent_id],
               'level': 30  # WARNING
           })

       # Simulate potential error
       if 'error' in message.lower():
           collector({
               'id': f'Error_{agent_id}',
               'name': 'Error',
               'type': 'Error',
               'item': 'Simulated error occurred',
               'parent_ids': [agent_id],
               'level': 40  # ERROR
           })
           return "Error occurred during processing!"

       return f"Processed: {message}"

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Log Levels Demo",
           message_handler=handler
       )

       @app.app.callback(
           Output('log-data-store', 'data'),
           Input('messages-store', 'data'),
           prevent_initial_call=True
       )
       def update_logs(_):
           return {
               'graph_data': collector.get_graph_structure(),
               'log_groups': dict(collector.log_groups)
           }

       app.run()

See Also
--------

- :doc:`../utilities/log_collector` - LogCollector reference
- :doc:`../components/log_graph` - LogGraphVisualization component
- :doc:`../components/log_details` - LogDetailsPanel component
- :doc:`../applications/dash_interactive_app_with_logs` - Application with log support

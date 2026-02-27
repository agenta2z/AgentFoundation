"""
Demo of the Dash Interactive UI with log debugging capabilities using a REAL Agent.

This demonstrates:
- Chat interaction tab for conversing with a bot
- Log debugging tab for visualizing hierarchical logs from REAL WorkGraph execution
- Interactive graph visualization with clickable nodes (constructed from logged JSON files)
- Detailed log viewing for selected log groups from actual Agent execution
- Uses write_json logger to persist logs to disk
- Reconstructs execution graph from parent_ids in logged JSON files

Run this script and navigate to http://localhost:8050 to see the UI.
"""
import sys
from functools import partial
from pathlib import Path

# Add source to path if needed
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add SciencePythonUtils and ScienceModelingTools src paths
rich_python_utils_src = project_root / "SciencePythonUtils" / "src"
agent_foundation_src = project_root / "ScienceModelingTools" / "src"
for path in [rich_python_utils_src, agent_foundation_src]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_foundation.ui.dash_interactive.dash_interactive_app_with_logs import DashInteractiveAppWithLogs
from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector
from rich_python_utils.io_utils.json_io import write_json, iter_json_objs
from rich_python_utils.datetime_utils.common import timestamp

# Import mock agent components from separate module
from mock_agent import (
    MockInteractive,
    ComplexAgentReasoner,
    TrackingActor,
    ComplexAgent
)

# No longer need log_graph_construction - everything is in LogCollector now


# Store log file paths and collectors from executions
_recent_log_collectors = {}
_latest_log_collector = None  # Store the most recent log_collector for easy access
_latest_log_file_path = None  # Store the path to the latest log file
_runtime_dir = Path(__file__).parent / '_runtime'  # Directory for storing log files

def execute_complex_agent_with_logs(user_input: str) -> tuple[str, str]:
    """
    Execute a real complex Agent and log to JSON files in a folder.

    Creates an Agent with parallel branching, executes it, and logs all
    execution details to JSON files using write_json logger with 'space' parameter.
    This creates one JSON file per debuggable object (identified by 'id').

    Args:
        user_input: The user's input message

    Returns:
        Tuple of (log_folder_path, agent_id)
    """
    # Create a timestamped log folder
    log_name = f'agent_{timestamp()}'
    log_path = _runtime_dir / log_name / 'logs'

    # Create write_json logger with 'space' parameter
    # This will create separate JSON files for each debuggable ID
    json_logger = partial(
        write_json,
        file_path=str(log_path),
        append=True
    )

    # Create mock components
    reasoner = ComplexAgentReasoner()
    actor = TrackingActor()
    interactive = MockInteractive(inputs=[user_input])

    # Create the agent with the JSON logger
    agent = ComplexAgent(
        reasoner=reasoner,
        actor=actor,
        interactive=interactive,
        log_time=True,  # Enable time logging
        logger=json_logger,  # Use write_json logger with space='id'
        always_add_logging_based_logger=False,  # Only use JSON logger
        debug_mode=True,  # Enable debug mode for more logs
        branching_agent_start_as_new=True,  # Each branch starts fresh
        only_keep_parent_debuggable_ids=True  # Only store parent IDs, not objects
    )

    # Get agent ID before execution
    agent_id = agent.id

    # Execute the agent
    try:
        result = agent(user_input)
    except Exception as e:
        print(f"[WARNING] Agent execution encountered an issue: {e}")
        # Continue - logs were still written to files

    return str(log_path), agent_id


def smart_agent_handler(message: str) -> str:
    """
    Message handler that executes a real complex Agent with JSON logging.

    Each user message triggers a real Agent execution with parallel branching,
    generating detailed hierarchical logs written to JSON files that can be
    viewed in the Log Debugging tab.

    Args:
        message: User input message

    Returns:
        Response message
    """
    import datetime
    global _latest_log_collector, _latest_log_file_path

    # Execute real Agent and get log folder path + agent ID
    log_folder_path, agent_id = execute_complex_agent_with_logs(message)

    # Load logs from all JSON files in the folder
    # LogCollector automatically builds the graph from parent_ids
    log_collector = LogCollector.from_json_logs(log_folder_path)

    # Store as the latest for UI access
    _latest_log_collector = log_collector
    _latest_log_file_path = log_folder_path

    # Also store with a timestamp key for history
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    session_key = f"agent_{timestamp_str}"
    _recent_log_collectors[session_key] = log_collector

    # Keep only last 10 executions
    if len(_recent_log_collectors) > 10:
        oldest_key = sorted(_recent_log_collectors.keys())[0]
        del _recent_log_collectors[oldest_key]

    # Generate response based on the execution
    num_log_groups = len(log_collector.log_groups)
    num_logs = len(log_collector.logs)

    # Get statistics from log collector
    stats = log_collector.get_log_graph_statistics()

    # Count JSON files in the folder
    json_file_count = len(list(Path(log_folder_path).glob('*.json')))

    response = f"""Processed your message: "{message[:100]}{'...' if len(message) > 100 else ''}"

**Real Agent Execution Summary:**
- Agent: {stats['agent_name']} (ID: {agent_id})
- Execution graph nodes: {stats['node_count']}
- Execution graph edges: {stats['edge_count']}
- Max graph depth: {stats['max_depth']}
- Total log entries: {num_logs} ({stats['total_log_count']} in graph nodes)
- JSON files created: {json_file_count} (one per debuggable)
- Session: {session_key}
- Log folder: {Path(log_folder_path).name}

**Graph constructed from ParentChildLink logs!** Switch to the **Log Debugging** tab to see the execution graph!
The graph is built from ParentChildLink entries that explicitly record parent-child relationships.

**Tip**: Click on any node in the graph to view its specific logs. You'll see the parallel execution structure!"""

    return response


def get_latest_agent_logs(graph_type: str = None):
    """
    Custom graph executor that returns the latest Agent's logs.

    This replaces the dummy graph executor with real Agent logs loaded from JSON files.

    Args:
        graph_type: Graph type selector (ignored - always returns latest)

    Returns:
        LogCollector with the latest Agent execution logs
    """
    global _latest_log_collector

    if _latest_log_collector:
        return _latest_log_collector

    # Fallback: return the most recent from history
    if _recent_log_collectors:
        latest_key = sorted(_recent_log_collectors.keys())[-1]
        return _recent_log_collectors[latest_key]

    # No logs available - return empty collector
    return LogCollector()


def main():
    """Run the log debugging demo with real Agent execution."""
    # Create the app
    app = DashInteractiveAppWithLogs(
        title="Real Agent Log Debugging Demo",
        port=8050,
        debug=True
    )

    # Set custom message handler
    app.set_message_handler(smart_agent_handler)

    # Add custom callback to auto-load logs when switching to Log Debugging tab
    from dash.dependencies import Input, Output, State
    import dash

    @app.app.callback(
        Output('log-data-store', 'data', allow_duplicate=True),  # Allow duplicate output
        [Input('main-panel-log-btn', 'n_clicks')],
        [State('log-data-store', 'data')],
        prevent_initial_call=True
    )
    def auto_load_agent_logs_on_tab_switch(log_btn_clicks, current_data):
        """Automatically load Agent logs when switching to Log Debugging tab."""
        global _latest_log_collector, _latest_log_file_path

        if log_btn_clicks and _latest_log_collector:
            # Get graph structure from log collector
            graph_structure = _latest_log_collector.get_graph_structure()
            
            # Return graph structure built from parent_ids in logs
            return {
                'graph_data': {  # Use 'graph_data' key instead of 'hierarchy'
                    'nodes': graph_structure['nodes'],
                    'edges': graph_structure['edges'],
                    'agent': graph_structure['agent'],  # Required by UI
                    'log_file': Path(_latest_log_file_path).name if _latest_log_file_path else 'unknown'
                },
                'log_groups': {k: v for k, v in _latest_log_collector.log_groups.items()}
            }

        # Return existing data if no new logs
        return current_data if current_data else dash.no_update

    # Run the app
    print(f"""
================================================================
        REAL AGENT LOG DEBUGGING DEMO
================================================================

Features:
  * Chat Interaction: Executes a REAL Agent with parallel branching
  * JSON Logging: Uses write_json logger to persist logs to disk
  * Graph from Logs: Constructs execution graph from parent_ids in JSON logs
  * AUTOMATIC Log Loading: Graph loads automatically when you switch tabs!
  * No In-Memory Traversal: Graph built directly from logged data
  * Real Logs: Displays actual logs from Agent and WorkGraphNode execution

Technical Implementation:
  * Logger: write_json with append=True (like e2e_dev_grocery_planning.py)
  * Graph Construction: Reads parent_ids from JSON logs (no BFS/DFS needed)
  * Parent Tracking: Debuggable.parent_debuggables with only_keep_parent_debuggable_ids=True
  * Log Directory: {_runtime_dir}

Usage (Simple!):
  1. Navigate to http://localhost:8050
  2. Type a message in chat (e.g., "Research quantum computing")
  3. Switch to 'Log Debugging' tab
  4. Graph AUTOMATICALLY loads from the JSON log file!
  5. You'll see nodes from the actual execution with parent-child relationships
  6. Click on any node to see its logs

Agent Execution Flow:
  * Iteration 1: SearchAction (quantum_computing)
  * Iteration 2: 3 Parallel Branches
    - AnalyzeAlgorithms (algorithms)
    - AnalyzeHardware (hardware)
    - AnalyzeUseCases (use_cases)
  * Iteration 3: WriteReport (final_report)
  * Iteration 4: Complete

This demonstrates actual branched recursion with graph reconstruction from logs!

NEW ARCHITECTURE:
  * No in-memory graph traversal - reads directly from JSON logs!
  * Parent-child relationships tracked via parent_ids in log entries
  * Simpler and more reliable than traversing Python object graphs
  * Logs are persistent and can be analyzed offline

""")

    app.run()


if __name__ == '__main__':
    main()

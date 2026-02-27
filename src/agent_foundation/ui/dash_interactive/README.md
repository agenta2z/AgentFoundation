# Dash Interactive UI

A modular Dash-based UI framework for building GPT-like chat interfaces and hierarchical log debugging tools.

## Features

- **GPT-like Chat Interface**: Clean, modern chat UI similar to ChatGPT
- **Session Management**: Multiple conversation sessions with history
- **Log Debugging**: Interactive visualization of hierarchical logs from WorkGraph execution
- **Graph Visualization**: Clickable tree view of log groups with Plotly
- **Detailed Log View**: Color-coded log entries with expandable metadata
- **Modular Components**: Reusable, well-structured components
- **Agent Integration**: Easy integration with LLM agents

## Architecture

```
dash_interactive/
├── components/              # Reusable UI components
│   ├── base.py             # Abstract base component class
│   ├── chat_history.py     # Left sidebar with session list
│   ├── chat_window.py      # Right panel with messages and input
│   ├── tabbed_panel.py     # Tab container for chat and log debugging
│   ├── log_graph.py        # Interactive graph visualization
│   └── log_details.py      # Detailed log entry viewer
├── utils/                  # Utility modules
│   ├── log_collector.py    # Hierarchical log collection
│   └── dummy_graph_executor.py  # Demo graph generator
├── examples/               # Demo scripts
│   ├── basic_chat_demo.py
│   └── log_debugging_demo.py
├── app.py                  # Basic chat application
└── app_with_logs.py        # Full app with log debugging
```

## Components

### BaseComponent

Abstract base class for all UI components. Provides:
- Consistent styling interface
- Standard layout/callback patterns
- ID generation utilities

### ChatHistoryList

Left sidebar component showing:
- New Chat button
- List of chat sessions
- Settings section

### ChatWindow

Right panel component with:
- Scrollable message display
- User/Assistant message formatting
- Input text area with send button

### TabbedPanel

Tab container component with:
- Chat Interaction tab
- Log Debugging tab with horizontal split
- Graph visualization in upper panel
- Log details in lower panel

### LogGraphVisualization

Interactive graph component using Plotly that:
- Displays hierarchical log group structure as a tree
- Color-codes nodes by log level
- Provides clickable nodes to select log groups
- Shows hover info with log group IDs

### LogDetailsPanel

Log details viewer that:
- Displays individual log entries for selected group
- Color-codes entries by level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Shows timestamp, type, and content
- Provides expandable metadata section

### DashInteractiveApp

Basic chat application class that:
- Combines chat components
- Manages session state
- Handles callbacks
- Provides message handler interface

### DashInteractiveAppWithLogs

Extended application with log debugging that:
- Includes all chat features
- Adds tabbed interface for log debugging
- Executes WorkGraphs and collects logs
- Connects graph clicks to log details

## Quick Start

### Basic Chat Demo

```python
from agent_foundation.ui.dash_interactive import DashInteractiveApp

def my_handler(message: str) -> str:
    return f"You said: {message}"

app = DashInteractiveApp(
    title="My Chat App",
    port=8050,
    debug=True
)
app.set_message_handler(my_handler)
app.run()
```

Then navigate to `http://localhost:8050`

### Log Debugging Demo

```python
from agent_foundation.ui.dash_interactive.app_with_logs import DashInteractiveAppWithLogs

app = DashInteractiveAppWithLogs(
    title="Log Debugging Demo",
    port=8050,
    debug=True
)
app.run()
```

Then navigate to `http://localhost:8050` and:
1. Use the Chat tab for conversation
2. Switch to "Log Debugging" tab
3. Select a graph type (Sequential, Parallel, or Complex)
4. Click "Execute Graph" to run a WorkGraph
5. Click nodes in the graph to view their detailed logs

### Run Examples

```bash
# Basic echo demo
cd ScienceModelingTools/src/agent_foundation/ui/dash_interactive/examples
python basic_chat_demo.py

# Full log debugging demo
python log_debugging_demo.py
```

## Integration with Agents

To integrate with an actual agent:

```python
from agent_foundation.agents.agent import Agent
from agent_foundation.ui.dash_interactive import DashInteractiveApp

# Create your agent
agent = Agent(...)

# Create handler function
def agent_handler(message: str) -> str:
    result = agent.run(message)
    return result

# Set up UI
app = DashInteractiveApp()
app.set_message_handler(agent_handler)
app.run()
```

## Integration with WorkGraphs

To use log debugging with your own WorkGraphs:

```python
from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector
from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode

# Create log collector
log_collector = LogCollector()

# Create your WorkGraph with hierarchical logging
node = WorkGraphNode(
    my_task,
    logger=log_collector,
    debug_mode=True,
    always_add_logging_based_logger=False
)

# Child nodes automatically inherit parent log group ID
child_node = WorkGraphNode(
    child_task,
    parent_log_group_id=node,  # Establishes hierarchy
    logger=log_collector,
    debug_mode=True
)

# Execute graph
graph = WorkGraph(...)
graph.run(...)

# Access collected logs
hierarchy = log_collector.get_graph_structure()
log_groups = log_collector.log_groups
```

## Styling

The UI uses a dark theme similar to ChatGPT with:
- **Backgrounds**: #202123 (sidebar), #343541 (main), #444654 (messages)
- **Text**: #ECECF1 (primary), #8E8EA0 (secondary)
- **Accents**: #19C37D (green for buttons/active)
- **Log Levels**:
  - DEBUG: #00BFFF (cyan)
  - INFO: #19C37D (green)
  - WARNING: #FFA500 (orange)
  - ERROR: #FF4500 (red)
  - CRITICAL: #DC143C (crimson)
- Smooth transitions and hover effects

## Dependencies

- dash
- dash-bootstrap-components
- plotly
- python >= 3.8
- rich_python_utils (for WorkGraph and Debuggable)
- attrs

## Graph Types

The demo includes three types of WorkGraphs to demonstrate hierarchical logging:

### Sequential Graph
Linear chain of tasks: Task1 → Task2 → Task3

Each task processes the output of the previous task, with clear parent-child relationships in logs.

### Parallel Graph
Branching execution with merge: Start → (ParallelA & ParallelB) → Summarizer

Demonstrates how parallel branches are logged with a common parent and then merged.

### Complex Graph
Multi-level nested hierarchy with 4 levels

Shows deep nesting with multiple branches at each level, perfect for testing complex hierarchical logging.

## Future Enhancements

- [ ] Log filtering and search
- [ ] Export chat history
- [ ] Export logs to file
- [ ] Theme customization
- [ ] Streaming responses
- [ ] File upload support
- [ ] Real-time log streaming during execution
- [ ] Log level filtering in graph view

## Development

### Adding New Components

1. Create new file in `components/`
2. Inherit from `BaseComponent`
3. Implement required abstract methods
4. Add to `components/__init__.py`

### Adding Callbacks

Callbacks are registered in `DashInteractiveApp._register_callbacks()`. Each component provides:
- `get_callback_inputs()` - Input dependencies
- `get_callback_outputs()` - Output targets
- `get_callback_states()` - State values (optional)

## License

Part of ScienceModelingTools package.

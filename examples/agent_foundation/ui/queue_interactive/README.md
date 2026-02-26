# QueueInteractive Examples

Comprehensive examples demonstrating the `QueueInteractive` class for queue-based agent communication.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

`QueueInteractive` is a queue-based implementation of `InteractiveBase` that enables agent communication through queue services. The KEY advantage is support for **TRUE cross-process communication** using `QueueServiceBase` implementations.

### What's New?

The updated `QueueInteractive` now:
- Accepts any `QueueServiceBase` implementation (StorageBasedQueueService, ThreadQueueService, RedisQueueService)
- Works across **separate Python processes** (using StorageBasedQueueService)
- Supports configurable queue IDs for flexible routing
- Provides blocking/non-blocking and timeout controls
- Enables distributed agent systems

### Why Use QueueInteractive?

```python
# OLD WAY: Limited to same process
from queue import Queue

queue = Queue()  # Only works within same process

# NEW WAY: Works across processes!
from rich_python_utils.service_utils.queue_service.storage_based_queue_service import StorageBasedQueueService

queue_service = StorageBasedQueueService(root_path='/tmp/queues')
# Now multiple processes can communicate!
```

## Key Features

### 1. True Cross-Process Communication

```python
# Process 1 (Agent)
interactive = QueueInteractive(
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id='questions',
    response_queue_id='answers'
)
user_input = interactive.get_input()  # Blocks until question arrives
interactive.send_response("Here's my answer", flag=False)

# Process 2 (User) - completely separate process!
queue_service.put('questions', "What is AI?")
answer = queue_service.get('answers')  # Gets response from agent
```

### 2. Multiple Queue Services

- **StorageBasedQueueService**: File-based, persistent, works across processes
- **ThreadQueueService**: In-memory, fast, same-process only
- **RedisQueueService**: External Redis, distributed systems

### 3. Flexible Queue Routing

```python
# Different agents on different queues
customer_agent = QueueInteractive(..., input_queue_id='customer_queue')
tech_agent = QueueInteractive(..., input_queue_id='tech_queue')
```

### 4. Timeout and Non-blocking Support

```python
# Blocking with timeout
interactive = QueueInteractive(..., blocking=True, timeout=5.0)

# Non-blocking
interactive = QueueInteractive(..., blocking=False)
```

## Prerequisites

```bash
# Install rich_python_utils
pip install science-python-utils

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/SciencePythonUtils/src:$PYTHONPATH
```

## Quick Start

### Basic Usage (Single Process)

```python
from rich_python_utils.service_utils.queue_service.storage_based_queue_service import (
    StorageBasedQueueService
)
from science_modeling_tools.ui.queue_interactive import QueueInteractive

# Create queue service
queue_service = StorageBasedQueueService(root_path='/tmp/my_agent')

# Create interactive
interactive = QueueInteractive(
    system_name="Assistant",
    user_name="User",
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id='input',
    response_queue_id='output'
)

# Send input
queue_service.put('input', "Hello!")

# Agent gets input
message = interactive.get_input()

# Agent sends response
interactive.send_response("Hi there!", flag=False)

# Get response
response = queue_service.get('output')

queue_service.close()
```

### Cross-Process Usage

```python
import multiprocessing as mp


def agent_process(root_path):
    queue_service = StorageBasedQueueService(root_path=root_path)
    interactive = QueueInteractive(
        system_name="Agent",
        user_name="User",
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id='questions',
        response_queue_id='answers'
    )

    while True:
        question = interactive.get_input()
        if not question or question == 'exit':
            break
        interactive.send_response(f"You asked: {question}", flag=False)

    queue_service.close()


def user_process(root_path):
    queue_service = StorageBasedQueueService(root_path=root_path)

    questions = ["Hello?", "How are you?", "exit"]
    for q in questions:
        queue_service.put('questions', q)
        answer = queue_service.get('answers', timeout=5.0)
        print(f"Q: {q} -> A: {answer}")

    queue_service.close()


# Run in separate processes
root = '/tmp/agent_demo'
agent = mp.Process(target=agent_process, args=(root,))
user = mp.Process(target=user_process, args=(root,))

agent.start()
time.sleep(0.5)  # Let agent start first
user.start()

agent.join()
user.join()
```

## Examples

### Example 1: Simple Usage ([example_simple_usage.py](example_simple_usage.py))

Demonstrates:
- Basic usage with StorageBasedQueueService
- Multi-turn conversation
- Timeout and non-blocking behavior
- Multiple queue IDs
- ThreadQueueService usage

```bash
python example_simple_usage.py
```

### Example 2: Multiprocessing Agent ([example_multiprocessing_agent.py](example_multiprocessing_agent.py))

Demonstrates:
- Simple Q&A across processes
- Agent pool with load balancing
- Specialized agents with routing

```bash
python example_multiprocessing_agent.py
```

## API Reference

### QueueInteractive

```python
class QueueInteractive(InteractiveBase):
    """
    Queue-based implementation of InteractiveBase.

    Attributes:
        system_name (str): Agent name (inherited from InteractiveBase)
        user_name (str): User name (inherited from InteractiveBase)
        input_queue (QueueServiceBase): Queue service for receiving input
        response_queue (QueueServiceBase): Queue service for sending responses
        input_queue_id (str): Queue ID for input (default: 'input')
        response_queue_id (str): Queue ID for responses (default: 'response')
        blocking (bool): Whether to block on get_input (default: True)
        timeout (float): Timeout in seconds for blocking get (default: None)
        pending_message (str): Message for pending state
    """
```

#### Constructor

```python
interactive = QueueInteractive(
    system_name="Agent",           # Optional, default="System"
    user_name="User",              # Optional, default="User"
    input_queue=queue_service,     # Required: QueueServiceBase instance
    response_queue=queue_service,  # Required: QueueServiceBase instance
    input_queue_id='my_input',     # Optional, default='input'
    response_queue_id='my_output', # Optional, default='response'
    blocking=True,                 # Optional, default=True
    timeout=10.0,                  # Optional, default=None
    pending_message="Waiting..."   # Optional
)
```

#### Methods

##### get_input()

```python
def get_input(self) -> str:
    """
    Get input from the input queue.

    Returns:
        str: Input message, or empty string if timeout/empty queue

    Raises:
        RuntimeError: If queue service is closed
    """
```

**Example:**

```python
# Blocking with timeout
message = interactive.get_input()  # Uses configured timeout

# To change behavior:
interactive.blocking = False
message = interactive.get_input()  # Non-blocking
```

##### send_response()

```python
def send_response(self, response: Any, is_pending: bool) -> None:
    """
    Send response(s) to the response queue.

    Args:
        response: Single response or list/tuple of responses
        is_pending: Whether more input is expected

    Raises:
        RuntimeError: If queue service is closed
    """
```

**Example:**

```python
# Single response
interactive.send_response("Hello!", flag=False)

# Multiple responses
interactive.send_response(
    ["First message", "Second message"],
    flag=True
)
```

##### reset_input()

```python
def reset_input(self, is_pending: bool) -> None:
    """
    Reset input state (currently a no-op).

    Args:
        is_pending: Whether more input is expected
    """
```

## Use Cases

### 1. Single Agent, Single User

```python
# Simple Q&A bot
agent = QueueInteractive(
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id='questions',
    response_queue_id='answers'
)
```

### 2. Multiple Agents, Load Balancing

```python
# Pool of workers processing from shared queue
for worker_id in range(num_workers):
    agent = QueueInteractive(
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id='task_queue',      # Shared input
        response_queue_id='result_queue'  # Shared output
    )
    # Each worker gets tasks in round-robin fashion
```

### 3. Specialized Agents

```python
# Different agents for different task types
math_agent = QueueInteractive(
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id='math_tasks',
    response_queue_id='math_results'
)

text_agent = QueueInteractive(
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id='text_tasks',
    response_queue_id='text_results'
)
```

### 4. Agent Pipelines

```python
# Agent 1 -> Agent 2 -> Agent 3
agent1 = QueueInteractive(
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id='stage1_input',
    response_queue_id='stage2_input'  # Output goes to next stage
)

agent2 = QueueInteractive(
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id='stage2_input',
    response_queue_id='stage3_input'
)

agent3 = QueueInteractive(
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id='stage3_input',
    response_queue_id='final_output'
)
```

### 5. Request-Response Pattern

```python
# User sends request, waits for response
user_queue = f'user_{user_id}_requests'
response_queue = f'user_{user_id}_responses'

agent = QueueInteractive(
    input_queue=queue_service,
    response_queue=queue_service,
    input_queue_id=user_queue,
    response_queue_id=response_queue
)
```

## Best Practices

### 1. Always Close Queue Services

```python
try:
    queue_service = StorageBasedQueueService(root_path='/tmp/queues')
    interactive = QueueInteractive(...)
    # ... do work ...
finally:
    queue_service.close()
```

Or use context managers (if supported):

```python
with StorageBasedQueueService(root_path='/tmp/queues') as queue_service:
    interactive = QueueInteractive(...)
    # ... do work ...
```

### 2. Use Descriptive Queue IDs

```python
# Good
input_queue_id='customer_support_questions'
response_queue_id='customer_support_answers'

# Bad
input_queue_id='input'
response_queue_id='output'
```

### 3. Set Appropriate Timeouts

```python
# For real-time applications
interactive = QueueInteractive(..., timeout=5.0)

# For batch processing
interactive = QueueInteractive(..., timeout=60.0)

# For always-on services
interactive = QueueInteractive(..., timeout=None)  # Wait forever
```

### 4. Handle Empty/Timeout Gracefully

```python
message = interactive.get_input()
if not message:
    # Timeout or empty queue
    logger.warning("No input received")
    continue
```

### 5. Use Same Queue Service Instance When Possible

```python
# Good: Reuse service instance
queue_service = StorageBasedQueueService(root_path='/tmp/queues')
agent1 = QueueInteractive(input_queue=queue_service, ...)
agent2 = QueueInteractive(input_queue=queue_service, ...)

# Avoid: Creating multiple instances unnecessarily
# (Each instance creates its own lock files)
```

### 6. For Multiprocessing: Set Start Method

```python
import multiprocessing as mp

if sys.platform == 'win32':
    mp.set_start_method('spawn', force=True)
```

### 7. Give Agents Time to Start

```python
# Start agent first
agent_proc = mp.Process(target=agent_function, args=(...))
agent_proc.start()

# Wait a bit before sending messages
time.sleep(0.5)

# Start user/producer
user_proc = mp.Process(target=user_function, args=(...))
user_proc.start()
```

## Troubleshooting

### Issue: "No module named 'science_python_utils'"

**Solution:** Install or add to PYTHONPATH:

```bash
pip install science-python-utils
# OR
export PYTHONPATH=/path/to/SciencePythonUtils/src:$PYTHONPATH
```

### Issue: "RuntimeError: Service is closed"

**Cause:** Trying to use queue service after calling `close()`.

**Solution:** Don't call `close()` until you're completely done:

```python
queue_service = StorageBasedQueueService(...)
interactive = QueueInteractive(input_queue=queue_service, ...)

# Do all work here...

# Only close when completely done
queue_service.close()
```

### Issue: get_input() returns empty string

**Causes:**
1. Timeout reached (blocking mode)
2. Queue is empty (non-blocking mode)

**Solution:** Check your timeout and ensure messages are being sent:

```python
# Increase timeout
interactive.timeout = 30.0

# Or check if queue has items
if queue_service.size('my_queue') > 0:
    message = interactive.get_input()
```

### Issue: Messages not received across processes

**Cause:** Using ThreadQueueService (doesn't support cross-process).

**Solution:** Use StorageBasedQueueService for cross-process communication:

```python
# Don't use this for cross-process:
# queue_service = ThreadQueueService()

# Use this instead:
queue_service = StorageBasedQueueService(root_path='/tmp/queues')
```

### Issue: "PermissionError: [WinError 32]" when deleting temp directory

**Cause:** Windows hasn't released lock file yet.

**Solution:** Add delay before cleanup:

```python
queue_service.close()
time.sleep(0.2)  # Let Windows release file handles
shutil.rmtree(tmpdir)
```

### Issue: Process hangs on get_input()

**Cause:** Blocking with no timeout, and no messages arriving.

**Solution:** Always use timeout in production:

```python
interactive = QueueInteractive(..., timeout=10.0)

message = interactive.get_input()
if not message:
    # Handle timeout
    logger.warning("Timeout waiting for input")
```

### Issue: Multiple processes getting same message

**Cause:** Using same queue_id for input across multiple agents (expected for load balancing).

**Solution:** If you want separate queues per agent:

```python
# Each agent gets its own queue
agent1 = QueueInteractive(..., input_queue_id='agent1_input')
agent2 = QueueInteractive(..., input_queue_id='agent2_input')
```

## Comparison with Alternatives

### vs. Python's queue.Queue

| Feature | queue.Queue | QueueInteractive + StorageBasedQueueService |
|---------|------------|---------------------------------------------|
| Cross-process | ❌ No | ✅ Yes |
| Persistent | ❌ No | ✅ Yes |
| Setup complexity | Simple | Moderate |
| Performance | Very fast | Moderate |
| Best for | Threading | Multiprocessing, distributed systems |

### vs. multiprocessing.Queue

| Feature | multiprocessing.Queue | QueueInteractive + StorageBasedQueueService |
|---------|----------------------|---------------------------------------------|
| Cross-process | ✅ Yes (same parent) | ✅ Yes (any process) |
| Persistent | ❌ No | ✅ Yes |
| Setup complexity | Simple | Moderate |
| Survives restart | ❌ No | ✅ Yes |
| Best for | Process pools | Independent processes, long-running agents |

### vs. Redis Queue

| Feature | Redis Queue | QueueInteractive + StorageBasedQueueService |
|---------|------------|---------------------------------------------|
| Cross-machine | ✅ Yes | ❌ No (needs shared filesystem) |
| External dependency | ✅ Redis server | ❌ None |
| Setup complexity | High | Low |
| Performance | Very fast | Moderate |
| Best for | Distributed systems | Single-machine multiprocessing |

## Advanced Patterns

### Pattern 1: Agent Pool with Monitoring

```python
def monitor(root_path):
    queue_service = StorageBasedQueueService(root_path=root_path)
    while True:
        pending = queue_service.size('tasks')
        completed = queue_service.size('results')
        print(f"Pending: {pending}, Completed: {completed}")
        time.sleep(1)
```

### Pattern 2: Priority Queues

```python
# Process high priority first
while True:
    # Try high priority
    msg = queue_service.get('high_priority', blocking=False)
    if msg:
        process(msg)
        continue

    # Then normal priority
    msg = queue_service.get('normal_priority', blocking=True, timeout=1.0)
    if msg:
        process(msg)
```

### Pattern 3: Request Routing

```python
def router(root_path):
    queue_service = StorageBasedQueueService(root_path=root_path)

    while True:
        request = queue_service.get('incoming', blocking=True)
        if not request:
            continue

        # Route based on request type
        if request['type'] == 'math':
            queue_service.put('math_queue', request)
        elif request['type'] == 'text':
            queue_service.put('text_queue', request)
```

## License

This code is part of the ScienceModelingTools project.

## Support

For issues or questions:
1. Check this README
2. Review the examples
3. Check test files for additional usage patterns
4. Consult the QueueServiceBase documentation

## Changelog

### v2.0.0 (Current)
- Updated to use QueueServiceBase instead of Python's Queue
- Added support for cross-process communication
- Added configurable queue IDs
- Added timeout and blocking controls
- Backward incompatible: Requires queue service instances instead of Queue objects

### v1.0.0 (Previous)
- Initial implementation with Python's queue.Queue
- Single-process only

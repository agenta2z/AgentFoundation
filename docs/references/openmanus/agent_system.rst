====================================
Agent System
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

The OpenManus agent system implements a hierarchical class structure based
on the **ReAct (Reason + Act)** paradigm. Agents iteratively reason about
their task by calling an LLM, then act by executing tools, observing
results, and repeating until the task is complete or a step limit is
reached.

Class Hierarchy
===============

::

    BaseAgent (abstract)
    │   Manages: state machine, memory, step loop, stuck detection
    │
    └── ReActAgent (abstract)
        │   Adds: think() + act() decomposition
        │
        └── ToolCallAgent (concrete)
            │   Implements: LLM function calling, tool execution
            │
            ├── Manus
            │     General-purpose agent with MCP support
            │
            ├── BrowserAgent
            │     Browser-focused with context helper
            │
            ├── SWEAgent
            │     Software engineering (bash + file editing)
            │
            ├── MCPAgent
            │     Dynamic MCP server tool discovery
            │
            ├── DataAnalysis
            │     Data analysis + chart visualization
            │
            └── SandboxManus
                  Cloud sandbox with VNC and remote tools


BaseAgent
=========

**File**: ``app/agent/base.py``

The ``BaseAgent`` is the abstract foundation for all agents. It manages
the agent lifecycle, state machine, memory, and execution loop.

State Machine
-------------

Agents transition through four states defined by the ``AgentState`` enum:

::

    IDLE ──► RUNNING ──► FINISHED
                │
                └──────► ERROR

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - State
     - Description
   * - ``IDLE``
     - Initial state. Agent is created but not yet running.
   * - ``RUNNING``
     - Agent is actively executing steps (think + act loop).
   * - ``FINISHED``
     - Agent has completed its task (via ``Terminate`` tool or manual
       state set).
   * - ``ERROR``
     - An unrecoverable error occurred during execution.

Key Attributes
--------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``name``
     - ``str``
     - Agent identifier (default: ``"base"``)
   * - ``description``
     - ``str``
     - Human-readable description
   * - ``system_prompt``
     - ``str``
     - System message sent to LLM on every call
   * - ``next_step_prompt``
     - ``str``
     - Appended as user message before each ``think()`` call
   * - ``memory``
     - ``Memory``
     - Conversation history (bounded list of ``Message`` objects)
   * - ``state``
     - ``AgentState``
     - Current state in the state machine
   * - ``max_steps``
     - ``int``
     - Maximum steps before forced termination (default: 10)
   * - ``current_step``
     - ``int``
     - Counter for the current step number
   * - ``duplicate_threshold``
     - ``int``
     - Number of identical messages before stuck detection (default: 2)
   * - ``llm``
     - ``LLM``
     - LLM client instance

Execution Loop
--------------

The ``run()`` method implements the main execution loop:

.. code-block:: python

    async def run(self, request: str | None = None) -> str:
        # 1. Validate and transition to RUNNING state
        if self.state != AgentState.IDLE:
            raise RuntimeError("Agent not in IDLE state")
        self.state = AgentState.RUNNING

        # 2. Add user request to memory
        if request:
            self.update_memory("user", request)

        # 3. Execute step loop
        results = []
        while (self.current_step < self.max_steps
               and self.state != AgentState.FINISHED):
            self.current_step += 1
            step_result = await self.step()
            results.append(step_result)

        # 4. Handle step limit exceeded
        if self.current_step >= self.max_steps:
            self.current_step = 0
            self.state = AgentState.IDLE

        # 5. Return result
        return results[-1] if results else ""

Stuck Detection
---------------

The ``is_stuck()`` method detects when an agent is repeating the same
response. It counts how many previous assistant messages have content
identical to the latest message across the entire conversation history.
If the count reaches ``duplicate_threshold`` (default 2), it returns
``True`` and the ``handle_stuck_state()`` method prepends a
strategy-change prompt to ``next_step_prompt``.

This is a critical safety mechanism against infinite loops where the LLM
keeps producing the same output.


ReActAgent
==========

**File**: ``app/agent/react.py``

The ``ReActAgent`` adds the **think-then-act** decomposition to the base
agent. It is still abstract -- subclasses must implement both ``think()``
and ``act()``.

.. code-block:: python

    class ReActAgent(BaseAgent, ABC):
        @abstractmethod
        async def think(self) -> bool:
            """Process current state and decide next action."""

        @abstractmethod
        async def act(self) -> str:
            """Execute decided actions."""

        async def step(self) -> str:
            should_act = await self.think()
            if not should_act:
                return "Thinking complete - no action needed"
            return await self.act()

The ``think()`` method returns a boolean: ``True`` if the agent decided
on actions to take, ``False`` if thinking is complete without needing
action (e.g., the LLM returned a text response with no tool calls).


ToolCallAgent
=============

**File**: ``app/agent/toolcall.py``

The ``ToolCallAgent`` is the first concrete agent class. It implements
the ReAct pattern using **LLM function calling** (OpenAI tool_calls API).

Key Attributes
--------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``available_tools``
     - ``ToolCollection``
     - Registry of tools the agent can use
   * - ``tool_calls``
     - ``List[ToolCall]``
     - Tool calls from the last LLM response
   * - ``tool_choices``
     - ``ToolChoice``
     - How the LLM selects tools: ``NONE``, ``AUTO``, ``REQUIRED``
   * - ``special_tool_names``
     - ``List[str]``
     - Tools that trigger state changes (default: ``["terminate"]``)
   * - ``max_steps``
     - ``int``
     - Override: 30 (higher than BaseAgent's 10)
   * - ``max_observe``
     - ``int``
     - Maximum characters in tool output before truncation

Think Phase
-----------

The ``think()`` method:

1. Appends ``next_step_prompt`` as a user message to memory
2. Calls ``self.llm.ask_tool()`` with:
   - Full message history (``self.memory.messages``)
   - System prompt
   - Available tool schemas (``self.available_tools.to_params()``)
   - Tool choice mode
3. Parses the LLM response:
   - If ``tool_calls`` present → stores them, returns ``True``
   - If text-only response → adds to memory, returns ``False``

Act Phase
---------

The ``act()`` method:

1. Iterates over ``self.tool_calls``
2. For each tool call:
   a. Parses JSON arguments from ``tool_call.function.arguments``
   b. Calls ``self.execute_tool(tool_call)``
   c. Handles special tools (terminate → ``AgentState.FINISHED``)
   d. Truncates result to ``max_observe`` characters if set
   e. Adds result as ``Message.tool_message()`` to memory
3. Returns concatenated results

Tool Execution
--------------

.. code-block:: python

    async def execute_tool(self, command: ToolCall) -> str:
        name = command.function.name
        args = json.loads(command.function.arguments or "{}")

        result = await self.available_tools.execute(
            name=name,
            tool_input=args
        )
        return str(result)


Manus Agent
===========

**File**: ``app/agent/manus.py``

The ``Manus`` agent is the **flagship general-purpose agent**. It combines
multiple tools with MCP server integration for maximum versatility.

Configuration
-------------

.. list-table::
   :widths: 25 60
   :header-rows: 1

   * - Property
     - Value
   * - ``name``
     - ``"manus"``
   * - ``max_steps``
     - 20
   * - ``system_prompt``
     - From ``app/prompt/manus.py`` -- identity + workspace directory
   * - ``next_step_prompt``
     - Tool selection guidance with current directory context

Tools
-----

Default tools (statically configured):

1. **PythonExecute** -- Execute Python code
2. **BrowserUseTool** -- Browser automation (16 actions)
3. **StrReplaceEditor** -- File viewing and editing
4. **AskHuman** -- Request human input
5. **Terminate** -- Signal task completion

Plus any **MCP server tools** discovered at initialization from the
``[mcp]`` configuration section.

Async Factory Pattern
---------------------

The ``Manus`` agent uses an async factory method because MCP connection
setup is asynchronous:

.. code-block:: python

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        instance = cls(**kwargs)
        # Connect to MCP servers from config
        mcp_config = config.mcp
        for server in mcp_config.servers:
            await instance.mcp_clients.connect(server)
            instance.available_tools.add_tools(*instance.mcp_clients.tools)
        return instance


BrowserAgent
============

**File**: ``app/agent/browser.py``

Specialized for browser automation tasks. Uses a ``BrowserContextHelper``
to capture and format browser state (URL, tabs, scroll position,
screenshot) as context for each LLM call.

Key Features:

- 94-line system prompt with detailed JSON response format
- Browser state tracking (current URL, open tabs, scroll position)
- Screenshot-based visual context for multimodal LLM reasoning
- Memory tracking field in structured responses


SWEAgent
========

**File**: ``app/agent/swe.py``

Software engineering agent with tools for shell commands and file editing:

- **Bash** -- Persistent shell session
- **StrReplaceEditor** -- File viewing and editing with undo
- **Terminate** -- Task completion

System prompt describes the file editor interface, response format
requirements, and indentation handling.


MCPAgent
========

**File**: ``app/agent/mcp.py``

Connects to MCP servers for dynamic tool discovery:

- Tools are discovered at connection time from external MCP servers
- Periodic tool refresh every ``_refresh_tools_interval`` steps (default: 5)
- Handles multimedia responses (base64 images)
- Graceful shutdown when MCP service becomes unavailable
- Supports both SSE and stdio transport protocols


DataAnalysis Agent
==================

**File**: ``app/agent/data_analysis.py``

Specialized for data analysis and visualization:

- **NormalPythonExecute** -- Extended Python execution with code_type
- **VisualizationPrepare** -- Prepare metadata for chart generation
- **DataVisualization** -- Create charts from CSV using VMind
- **Terminate** -- Task completion

Uses a visualization-specific system prompt that requires analysis
reports.


SandboxManus Agent
==================

**File**: ``app/agent/sandbox_agent.py``

Cloud sandbox variant of ``Manus`` using Daytona:

- Runs in isolated cloud environments with VNC
- Tools: ``SandboxBrowserTool``, ``SandboxFilesTool``,
  ``SandboxShellTool``, ``SandboxVisionTool``
- Also supports MCP tools and human interaction
- Resource limits: 2 CPU, 4GB RAM, 5GB disk
- Auto-stop after 15 minutes of inactivity


Agent Lifecycle
===============

Creation
--------

1. **Instantiation**: Agent class is created (Pydantic model init)
2. **Configuration**: System prompt, tools, LLM instance are configured
3. **MCP Connection** (if applicable): ``create()`` factory connects to
   MCP servers

Execution
---------

1. **``run(prompt)``**: Entry point, transitions to RUNNING state
2. **Step Loop**: Iterates ``think()`` → ``act()`` up to ``max_steps``
3. **Termination**: Via ``Terminate`` tool, max_steps, or error

Cleanup
-------

1. **``cleanup()``**: Called in ``finally`` block after ``run()``
2. **Browser cleanup**: Closes browser contexts and connections
3. **MCP disconnect**: Closes all MCP server connections
4. **Sandbox cleanup**: Stops and removes Docker containers


Memory System
=============

Each agent has its own ``Memory`` instance (defined in ``app/schema.py``):

.. code-block:: python

    class Memory(BaseModel):
        messages: List[Message] = Field(default_factory=list)
        max_messages: int = Field(default=100)

        def add_message(self, message: Message):
            self.messages.append(message)
            # Trim oldest if exceeded
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]

Key characteristics:

- **Bounded**: Sliding window of up to 100 messages (configurable)
- **Ephemeral**: Lost when the process exits
- **Per-agent**: Each agent has its own isolated memory
- **Not shared**: No mechanism for inter-agent memory sharing
- **Linear**: Flat chronological list, no indexing or categorization

Message Types
-------------

The ``Message`` class supports multiple roles:

- ``system`` -- System prompt messages
- ``user`` -- User input and next_step prompts
- ``assistant`` -- LLM responses (text and/or tool calls)
- ``tool`` -- Tool execution results

Messages can include:

- Text content
- Tool calls (function name + arguments)
- Tool results (output, error, base64 images)
- Base64-encoded images for multimodal context

====================================
Tutorials
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

This document provides step-by-step tutorials for common tasks with
OpenManus, from basic usage to advanced customization.


Tutorial 1: Your First Agent Run
==================================

This tutorial walks through running OpenManus for the first time.

Prerequisites
-------------

- OpenManus installed (see :doc:`getting_started`)
- API key configured in ``config/config.toml``

Steps
-----

1. **Start the agent**:

   .. code-block:: bash

       python main.py

2. **Enter a prompt** when asked:

   .. code-block:: text

       Enter your prompt: What is the current weather in San Francisco?

3. **Observe the execution**:

   The agent will:

   - ``think()``: Decide to use ``web_search``
   - ``act()``: Execute the search, get results
   - ``think()``: Decide to present findings
   - ``act()``: Call ``terminate`` with the answer

4. **Check the output** in the terminal.

Understanding the Output
-------------------------

The agent logs show the think-act cycle:

::

    [Step 1] Thinking...
    → LLM decided to call: web_search("current weather San Francisco")

    [Step 1] Acting...
    → web_search result: "Currently 62°F, partly cloudy..."

    [Step 2] Thinking...
    → LLM decided to call: terminate(status="success")

    [Step 2] Acting...
    → Task completed.


Tutorial 2: Creating a Custom Tool
====================================

This tutorial shows how to create and register a custom tool.

Step 1: Define the Tool
-------------------------

Create ``app/tool/my_calculator.py``:

.. code-block:: python

    from app.tool.base import BaseTool, ToolResult


    class Calculator(BaseTool):
        """A simple calculator tool."""

        name: str = "calculator"
        description: str = (
            "Evaluates mathematical expressions. "
            "Use for arithmetic calculations."
        )
        parameters: dict = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate, "
                        "e.g., '2 + 3 * 4'"
                    ),
                }
            },
            "required": ["expression"],
        }

        async def execute(self, expression: str) -> ToolResult:
            try:
                # Safe evaluation (numbers and operators only)
                allowed = set("0123456789+-*/.() ")
                if not all(c in allowed for c in expression):
                    return self.fail_response(
                        "Expression contains invalid characters"
                    )
                result = eval(expression)  # noqa: S307
                return self.success_response(str(result))
            except Exception as e:
                return self.fail_response(f"Calculation error: {e}")

Step 2: Register with an Agent
-------------------------------

Create a custom agent or modify an existing one:

.. code-block:: python

    # my_agent.py
    from app.agent.manus import Manus
    from app.tool.my_calculator import Calculator
    from app.tool.tool_collection import ToolCollection
    from app.tool.terminate import Terminate

    class MyAgent(Manus):
        available_tools: ToolCollection = ToolCollection(
            Calculator(),
            Terminate(),
        )

Step 3: Use the Agent
----------------------

.. code-block:: python

    import asyncio
    from my_agent import MyAgent

    async def main():
        agent = MyAgent()
        result = await agent.run("What is 15 * 7 + 23?")
        print(result)

    asyncio.run(main())


Tutorial 3: Multi-Agent Planning Flow
=======================================

This tutorial demonstrates the planning flow with multiple agents.

Step 1: Configure Agents
--------------------------

In ``config/config.toml``:

.. code-block:: toml

    [runflow]
    use_data_analysis_agent = true

Step 2: Run the Planning Flow
-------------------------------

.. code-block:: bash

    python run_flow.py

Step 3: Enter a Complex Task
------------------------------

.. code-block:: text

    Enter your prompt: Research the top 5 programming languages by
    popularity in 2025, create a comparison table, and generate a
    bar chart visualization.

Step 4: Observe the Plan
-------------------------

The system will:

1. **Create a plan** with steps like:
   - Step 1: Search for programming language rankings
   - Step 2: Extract and organize data
   - Step 3: Create comparison table
   - Step 4: Generate visualization
   - Step 5: Compile final report

2. **Execute each step** with the appropriate agent:
   - Steps 1-3: ``Manus`` agent (web search, data processing)
   - Step 4: ``DataAnalysis`` agent (chart generation)
   - Step 5: ``Manus`` agent (report compilation)


Tutorial 4: Connecting to MCP Servers
=======================================

This tutorial shows how to use external tools via MCP.

Step 1: Set Up an MCP Server
------------------------------

Using the official MCP filesystem server:

.. code-block:: bash

    npm install -g @modelcontextprotocol/server-filesystem

Step 2: Configure the Connection
----------------------------------

Create ``config/mcp.json``:

.. code-block:: json

    {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/path/to/allowed/directory"
                ],
                "type": "stdio"
            }
        }
    }

Step 3: Run the MCP Agent
---------------------------

.. code-block:: bash

    python run_mcp.py --connection stdio --interactive

The agent will discover all tools from the filesystem MCP server and
make them available for use.

Step 4: Use the Tools
----------------------

.. code-block:: text

    Enter your prompt: List all Python files in the project and
    show me the contents of the main entry point.

The agent will use the MCP filesystem tools to browse and read files.


Tutorial 5: Exposing OpenManus as an MCP Server
=================================================

This tutorial makes OpenManus tools available to other applications.

Step 1: Start the Server
--------------------------

.. code-block:: bash

    python run_mcp_server.py

This exposes Bash, BrowserUseTool, StrReplaceEditor, and Terminate
as MCP tools via stdio transport.

Step 2: Connect from Another Application
------------------------------------------

In Claude Desktop's MCP configuration (``claude_desktop_config.json``):

.. code-block:: json

    {
        "mcpServers": {
            "openmanus": {
                "command": "python",
                "args": ["/path/to/OpenManus/run_mcp_server.py"]
            }
        }
    }

Now Claude Desktop can use OpenManus tools for bash commands, browser
automation, and file editing.


Tutorial 6: Building a Custom Flow
====================================

This tutorial shows how to create a custom multi-agent orchestration.

Step 1: Define Custom Agents
------------------------------

.. code-block:: python

    from app.agent.toolcall import ToolCallAgent
    from app.tool.tool_collection import ToolCollection
    from app.tool.web_search import WebSearch
    from app.tool.python_execute import PythonExecute
    from app.tool.terminate import Terminate

    class ResearchAgent(ToolCallAgent):
        name: str = "researcher"
        system_prompt: str = (
            "You are a research specialist. "
            "Search for information and summarize findings."
        )
        available_tools: ToolCollection = ToolCollection(
            WebSearch(),
            Terminate(),
        )

    class CoderAgent(ToolCallAgent):
        name: str = "coder"
        system_prompt: str = (
            "You are a coding specialist. "
            "Write clean, efficient Python code."
        )
        available_tools: ToolCollection = ToolCollection(
            PythonExecute(),
            Terminate(),
        )

Step 2: Create the Flow
-------------------------

.. code-block:: python

    from app.flow.flow_factory import FlowFactory, FlowType

    agents = {
        "researcher": ResearchAgent,
        "coder": CoderAgent,
    }

    flow = FlowFactory.create_flow(
        flow_type=FlowType.PLANNING,
        agents=agents,
    )

Step 3: Execute
----------------

.. code-block:: python

    import asyncio

    async def main():
        result = await flow.execute(
            "Research the best sorting algorithms, then implement "
            "the top 3 in Python with benchmarks."
        )
        print(result)

    asyncio.run(main())


Tutorial 7: Using the Sandbox
================================

This tutorial shows how to run code in an isolated Docker sandbox.

Step 1: Enable Sandbox Mode
-----------------------------

In ``config/config.toml``:

.. code-block:: toml

    [sandbox]
    use_sandbox = true
    image = "python:3.12-slim"
    work_dir = "/workspace"
    memory_limit = "512m"
    network_enabled = false

Step 2: Run Normally
---------------------

.. code-block:: bash

    python main.py

The ``StrReplaceEditor`` will automatically use the Docker sandbox
for file operations instead of the local filesystem.

Step 3: Verify Isolation
--------------------------

.. code-block:: text

    Enter your prompt: Create a Python script that writes to
    /workspace/test.py and execute it.

Files will be created inside the Docker container, not on your
host filesystem.


Tutorial 8: Understanding the Think-Act Loop
==============================================

This tutorial explains the core execution pattern by tracing through
a real example.

Given the prompt: "Create a file called hello.py with a hello world program"

**Step 1 - Think**:

The agent sends to the LLM:

::

    System: You are OpenManus, an all-capable AI assistant...
    User: Create a file called hello.py with a hello world program
    User: Based on user needs, proactively select the most appropriate
          tool... Current working directory: /path/to/workspace

The LLM responds with a tool call:

::

    tool_calls: [{
        function: {
            name: "str_replace_editor",
            arguments: {
                "command": "create",
                "path": "workspace/hello.py",
                "file_text": "print('Hello, World!')\n"
            }
        }
    }]

**Step 1 - Act**:

The agent executes:

::

    StrReplaceEditor.execute(
        command="create",
        path="workspace/hello.py",
        file_text="print('Hello, World!')\n"
    )
    → ToolResult(output="File created successfully at workspace/hello.py")

**Step 2 - Think**:

The agent sends the updated conversation (including the tool result)
back to the LLM. The LLM decides the task is complete:

::

    tool_calls: [{
        function: {
            name: "terminate",
            arguments: {"status": "success"}
        }
    }]

**Step 2 - Act**:

::

    Terminate.execute(status="success")
    → Agent state transitions to FINISHED
    → Loop exits

Result: ``workspace/hello.py`` contains ``print('Hello, World!')``

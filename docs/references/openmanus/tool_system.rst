====================================
Tool System
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

The OpenManus tool system provides a unified abstraction for agent
capabilities. Tools are the bridge between LLM reasoning and real-world
actions -- they allow agents to execute code, browse the web, edit files,
search the internet, and more.

Architecture
============

::

    BaseTool (abstract, Pydantic BaseModel)
    │   name, description, parameters, execute()
    │
    ├── ToolResult (return type)
    │     output, error, base64_image, system
    │
    └── ToolCollection (registry + dispatcher)
          tool_map, to_params(), execute(), add_tool()


BaseTool
--------

**File**: ``app/tool/base.py``

.. code-block:: python

    class BaseTool(ABC, BaseModel):
        name: str
        description: str
        parameters: dict = {}

        @abstractmethod
        async def execute(self, **kwargs) -> Any:
            """Execute the tool with given arguments."""

        def to_param(self) -> dict:
            """Convert to OpenAI function-calling format."""
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                }
            }

        def __call__(self, **kwargs):
            return self.execute(**kwargs)

ToolResult
----------

.. code-block:: python

    class ToolResult(BaseModel):
        output: Any = None          # Success output
        error: Optional[str] = None # Error message
        base64_image: Optional[str] = None  # Screenshot/image
        system: Optional[str] = None  # System message

    class CLIResult(ToolResult):
        """Result from CLI operations."""

    class ToolFailure(ToolResult):
        """Marker for tool execution failures."""

ToolCollection
--------------

**File**: ``app/tool/tool_collection.py``

.. code-block:: python

    class ToolCollection:
        tools: Tuple[BaseTool, ...] = ()
        tool_map: Dict[str, BaseTool] = {}

        def __init__(self, *tools: BaseTool):
            # Build tool_map from tools
            self.tool_map = {t.name: t for t in tools}

        def to_params(self) -> List[dict]:
            """All tools in OpenAI function-calling format."""
            return [t.to_param() for t in self.tool_map.values()]

        async def execute(self, name: str, tool_input: dict) -> ToolResult:
            tool = self.tool_map.get(name)
            if not tool:
                return ToolResult(error=f"Tool '{name}' not found")
            return await tool(**tool_input)

        def add_tool(self, tool: BaseTool):
            self.tool_map[tool.name] = tool
            self.tools = tuple(self.tool_map.values())


Complete Tool Catalog
=====================

Shell and Code Execution
------------------------

PythonExecute
^^^^^^^^^^^^^

**File**: ``app/tool/python_execute.py``

Executes Python code in a separate process with timeout protection.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``python_execute``
   * - **Parameters**
     - ``code`` (str, required): Python code to execute
   * - **Timeout**
     - 5 seconds (configurable)
   * - **Isolation**
     - Runs in ``multiprocessing.Process`` with ``exec()``
   * - **Output**
     - Only captures ``print()`` output; return values are not captured
   * - **Safety**
     - Weak -- full ``__builtins__`` available; arbitrary code execution
       possible

Bash
^^^^

**File**: ``app/tool/bash.py``

Persistent bash shell session with interactive support.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``bash``
   * - **Parameters**
     - ``command`` (str): Shell command; ``restart`` (bool): Reset session
   * - **Timeout**
     - 120 seconds
   * - **Features**
     - Persistent session, sentinel-based output detection, stdin support,
       Ctrl+C handling
   * - **Used By**
     - SWEAgent, MCP Server

Browser and Web Tools
---------------------

BrowserUseTool
^^^^^^^^^^^^^^

**File**: ``app/tool/browser_use_tool.py``

Full browser automation using the ``browser-use`` library (Playwright).

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``browser_use``
   * - **Actions**
     - ``go_to_url``, ``click_element``, ``input_text``, ``scroll_down``,
       ``scroll_up``, ``scroll_to_text``, ``send_keys``,
       ``get_dropdown_options``, ``select_dropdown_option``, ``go_back``,
       ``web_search``, ``wait``, ``extract_content``, ``switch_tab``,
       ``open_tab``, ``close_tab``
   * - **Features**
     - Screenshot capture, thread-safe (asyncio Lock), LLM-powered
       content extraction, embedded WebSearch
   * - **Used By**
     - Manus, BrowserAgent

WebSearch
^^^^^^^^^

**File**: ``app/tool/web_search.py``

Multi-engine web search with automatic fallback.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``web_search``
   * - **Parameters**
     - ``query`` (str), ``num_results`` (int, default 10),
       ``fetch_content`` (bool, default False)
   * - **Engines**
     - Google → DuckDuckGo → Baidu → Bing (configurable fallback order)
   * - **Features**
     - Per-engine retry logic (tenacity), content fetching with HTML
       parsing, language/country filtering

Crawl4aiTool
^^^^^^^^^^^^

**File**: ``app/tool/crawl4ai.py``

Web crawler for AI-optimized content extraction.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``crawl4ai``
   * - **Parameters**
     - ``url`` (str): URL to crawl
   * - **Features**
     - Clean markdown extraction, caching, word count thresholds,
       multiple URL support

File Editing
------------

StrReplaceEditor
^^^^^^^^^^^^^^^^

**File**: ``app/tool/str_replace_editor.py``

File editor with view, create, edit, and undo capabilities.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``str_replace_editor``
   * - **Commands**
     - ``view`` (read file/directory), ``create`` (new file),
       ``str_replace`` (find-and-replace), ``insert`` (add line),
       ``undo_edit`` (revert last change)
   * - **Features**
     - Edit history per file, line number display, directory listing,
       supports local and sandbox file operations
   * - **File Operators**
     - ``LocalFileOperator`` (direct filesystem),
       ``SandboxFileOperator`` (Docker container)
   * - **Limits**
     - Output truncated at 16KB

Planning and Control
--------------------

PlanningTool
^^^^^^^^^^^^

**File**: ``app/tool/planning.py``

Plan CRUD operations for multi-step task management.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``planning``
   * - **Commands**
     - ``create``, ``update``, ``list``, ``get``, ``set_active``,
       ``mark_step``, ``delete``
   * - **Step Statuses**
     - ``not_started``, ``in_progress``, ``completed``, ``blocked``
   * - **Storage**
     - In-memory dictionary (ephemeral)

Terminate
^^^^^^^^^

**File**: ``app/tool/terminate.py``

Signals task completion.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``terminate``
   * - **Parameters**
     - ``status`` (str): ``"success"`` or ``"failure"``
   * - **Behavior**
     - Triggers ``AgentState.FINISHED`` as a "special tool"

AskHuman
^^^^^^^^

**File**: ``app/tool/ask_human.py``

Human-in-the-loop interaction.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``ask_human``
   * - **Parameters**
     - ``inquire`` (str): Question to ask
   * - **Mechanism**
     - Python ``input()`` call (blocking, terminal-based)

CreateChatCompletion
^^^^^^^^^^^^^^^^^^^^

**File**: ``app/tool/create_chat_completion.py``

Structured LLM output formatting.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``create_chat_completion``
   * - **Features**
     - Dynamic JSON schema from Python types, Pydantic model support,
       Union/List/Dict type handling

Desktop Automation
------------------

ComputerUseTool
^^^^^^^^^^^^^^^

**File**: ``app/tool/computer_use_tool.py``

Desktop GUI automation via API.

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``computer_use``
   * - **Actions**
     - ``move_to``, ``click``, ``scroll``, ``typing``, ``press``,
       ``wait``, ``mouse_down``, ``mouse_up``, ``drag_to``, ``hotkey``,
       ``screenshot``
   * - **Requirements**
     - Daytona sandbox with automation service on port 8000

Data Visualization
------------------

DataVisualization
^^^^^^^^^^^^^^^^^

**File**: ``app/tool/chart_visualization/data_visualization.py``

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``data_visualization``
   * - **Features**
     - Creates charts from CSV via VMind (TypeScript), outputs PNG/HTML,
       supports insights overlay

VisualizationPrepare
^^^^^^^^^^^^^^^^^^^^

**File**: ``app/tool/chart_visualization/chart_prepare.py``

.. list-table::
   :widths: 20 80

   * - **Name**
     - ``visualization_preparation``
   * - **Purpose**
     - Prepares metadata JSON for DataVisualization tool

Sandbox Tools
-------------

These tools operate inside Daytona cloud sandboxes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Tool
     - Name
     - Description
   * - SandboxBrowserTool
     - ``sandbox_browser``
     - Browser automation via curl to localhost:8003; 15 actions
       including navigate, click, drag_drop
   * - SandboxFilesTool
     - ``sandbox_files``
     - File CRUD in /workspace; create, str_replace, full rewrite,
       delete; auto-detects index.html
   * - SandboxShellTool
     - ``sandbox_shell``
     - Shell via tmux sessions; blocking/non-blocking execution;
       configurable timeout
   * - SandboxVisionTool
     - ``sandbox_vision``
     - Read and compress images from sandbox; max 1920x1080; returns
       base64

MCP Tools
---------

MCPClientTool
^^^^^^^^^^^^^

**File**: ``app/tool/mcp.py``

Proxy for remote MCP server tools.

.. list-table::
   :widths: 20 80

   * - **Name**
     - Dynamic: ``mcp_{server_id}_{original_name}``
   * - **Mechanism**
     - Forwards ``execute()`` calls to remote MCP server via session
   * - **Features**
     - Server identification, name sanitization (max 64 chars)

MCPClients
^^^^^^^^^^

Manager for multiple MCP server connections. Extends ``ToolCollection``.

.. list-table::
   :widths: 20 80

   * - **Transports**
     - SSE (``connect_sse``) and stdio (``connect_stdio``)
   * - **Discovery**
     - Auto-discovers tools from servers via ``session.list_tools()``
   * - **Lifecycle**
     - Connect, initialize, list tools, disconnect with cleanup


Search Engine Backends
======================

**Directory**: ``app/tool/search/``

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Engine
     - File
     - Library
   * - Google
     - ``google_search.py``
     - ``googlesearch-python``
   * - DuckDuckGo
     - ``duckduckgo_search.py``
     - ``duckduckgo_search``
   * - Baidu
     - ``baidu_search.py``
     - ``baidusearch``
   * - Bing
     - ``bing_search.py``
     - Custom implementation

All implement the ``WebSearchEngine`` base class:

.. code-block:: python

    class WebSearchEngine(ABC):
        @abstractmethod
        def perform_search(
            self, query: str, num_results: int = 10, **kwargs
        ) -> List[SearchItem]:
            """Execute search and return results."""


Tool Integration with Agents
==============================

Tools are bound to agents through the ``available_tools`` attribute:

.. code-block:: python

    class Manus(ToolCallAgent):
        available_tools: ToolCollection = Field(
            default_factory=lambda: ToolCollection(
                PythonExecute(),
                BrowserUseTool(),
                StrReplaceEditor(),
                AskHuman(),
                Terminate(),
            )
        )

The ``ToolCallAgent.think()`` method passes ``available_tools.to_params()``
to the LLM, which returns tool calls referencing tools by name. The
``ToolCallAgent.act()`` method dispatches calls through
``available_tools.execute(name, args)``.

Adding Custom Tools
-------------------

To add a custom tool:

1. Create a subclass of ``BaseTool``
2. Define ``name``, ``description``, and ``parameters``
3. Implement ``execute()``
4. Add to an agent's ``ToolCollection``

.. code-block:: python

    from app.tool.base import BaseTool, ToolResult

    class MyTool(BaseTool):
        name: str = "my_tool"
        description: str = "Does something useful"
        parameters: dict = {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input data"}
            },
            "required": ["input"]
        }

        async def execute(self, input: str) -> ToolResult:
            result = process(input)
            return ToolResult(output=result)

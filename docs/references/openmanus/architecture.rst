====================================
System Architecture
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

OpenManus follows a layered architecture with clear separation of concerns.
The system is designed around a **ReAct (Reason + Act) paradigm** where
LLM-powered agents iteratively reason about tasks and take actions using
tools.

Architecture Diagram
--------------------

::

    ┌─────────────────────────────────────────────────────────────┐
    │                      Entry Points                          │
    │  main.py │ run_flow.py │ run_mcp.py │ sandbox_main.py      │
    └─────────────┬──────────────┬──────────────┬────────────────┘
                  │              │              │
    ┌─────────────▼──────────────▼──────────────▼────────────────┐
    │                    Flow Layer                               │
    │  BaseFlow ──► PlanningFlow (multi-agent task decomposition) │
    │                       │                                     │
    │              FlowFactory (creates flows)                    │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │                    Agent Layer                               │
    │                                                              │
    │  BaseAgent (abstract)                                        │
    │    └── ReActAgent (abstract: think + act)                    │
    │          └── ToolCallAgent (LLM function calling)            │
    │                ├── Manus (general-purpose + MCP)              │
    │                ├── BrowserAgent (browser automation)          │
    │                ├── SWEAgent (software engineering)            │
    │                ├── MCPAgent (MCP server tools)                │
    │                ├── DataAnalysis (data + visualization)        │
    │                └── SandboxManus (cloud sandbox)               │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │                    Tool Layer                                │
    │                                                              │
    │  BaseTool (abstract) ─► ToolCollection (registry/dispatch)   │
    │    ├── PythonExecute     ├── BrowserUseTool                   │
    │    ├── Bash              ├── StrReplaceEditor                 │
    │    ├── WebSearch         ├── PlanningTool                     │
    │    ├── Terminate         ├── AskHuman                        │
    │    ├── ComputerUseTool   ├── Crawl4aiTool                    │
    │    ├── MCPClientTool     └── CreateChatCompletion             │
    │    └── Sandbox Tools (Browser, Files, Shell, Vision)         │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │                  Infrastructure Layer                        │
    │                                                              │
    │  LLM (OpenAI/Azure/Bedrock) │ Config (TOML)                  │
    │  Memory (Message history)   │ Logger (loguru)                │
    │  Sandbox (Docker/Daytona)   │ MCP (Client + Server)          │
    └─────────────────────────────────────────────────────────────┘


Design Patterns
===============

Singleton Pattern
-----------------

Several core components use the singleton pattern to ensure a single shared
instance across the application:

- **Config**: ``Config()`` returns the same instance on every call. Loads
  configuration from ``config/config.toml`` once.

  .. code-block:: python

      # app/config.py
      class Config:
          _instance = None
          def __new__(cls):
              if cls._instance is None:
                  cls._instance = super().__new__(cls)
                  cls._instance._load_config()
              return cls._instance

- **LLM**: One instance per ``config_name``, stored in ``LLM._instances``.
  The default instance uses the ``[llm]`` section of the TOML config.

- **SANDBOX_CLIENT**: A global ``LocalSandboxClient`` singleton created at
  module import time in ``app/sandbox/client.py``.

Template Method Pattern
-----------------------

The agent execution loop uses the Template Method pattern:

- ``BaseAgent.run()`` defines the overall loop structure (initialize, step
  loop, finalize)
- ``BaseAgent.step()`` is abstract -- subclasses define what happens in
  each step
- ``ReActAgent.step()`` further decomposes into ``think()`` + ``act()``
- ``ToolCallAgent`` provides concrete ``think()`` (LLM call) and ``act()``
  (tool execution) implementations

Strategy Pattern
----------------

- **FileOperator**: The ``StrReplaceEditor`` uses a ``FileOperator``
  protocol with two implementations: ``LocalFileOperator`` (direct
  filesystem) and ``SandboxFileOperator`` (Docker container). The strategy
  is selected based on ``config.sandbox.use_sandbox``.

- **Search Engine**: ``WebSearch`` supports multiple search engines
  (Google, Baidu, DuckDuckGo, Bing) with automatic fallback.

Factory Pattern
---------------

- **FlowFactory**: Creates flow instances by type (currently only
  ``PLANNING``).
- **Manus.create()**: Async factory method for agent initialization
  (connects MCP servers, initializes browser contexts).

Proxy Pattern
-------------

- **MCPClientTool**: Acts as a proxy for remote MCP server tools. It
  stores a reference to the MCP session and forwards ``execute()`` calls
  to the remote server via ``session.call_tool()``.

Observer Pattern
----------------

- **ReActAgent**: Separates the observation of LLM output (``think()``)
  from action execution (``act()``), enabling agents to observe results
  and feed them back into the next thinking cycle.


Directory Structure
===================

::

    OpenManus/
    ├── app/                           # Main application package
    │   ├── __init__.py                # Python 3.11-3.13 version check
    │   ├── config.py                  # Configuration system (TOML-based)
    │   ├── llm.py                     # LLM client wrapper
    │   ├── bedrock.py                 # AWS Bedrock adapter
    │   ├── schema.py                  # Core data models
    │   ├── logger.py                  # Logging configuration
    │   ├── exceptions.py              # Custom exceptions
    │   ├── agent/                     # Agent implementations
    │   │   ├── __init__.py
    │   │   ├── base.py                # BaseAgent (abstract)
    │   │   ├── react.py               # ReActAgent (abstract)
    │   │   ├── toolcall.py            # ToolCallAgent (concrete)
    │   │   ├── manus.py               # Manus (flagship agent)
    │   │   ├── browser.py             # BrowserAgent
    │   │   ├── swe.py                 # SWEAgent
    │   │   ├── mcp.py                 # MCPAgent
    │   │   ├── data_analysis.py       # DataAnalysis agent
    │   │   └── sandbox_agent.py       # SandboxManus
    │   ├── flow/                      # Multi-agent orchestration
    │   │   ├── base.py                # BaseFlow (abstract)
    │   │   ├── planning.py            # PlanningFlow
    │   │   └── flow_factory.py        # FlowFactory
    │   ├── tool/                      # Tool implementations
    │   │   ├── base.py                # BaseTool, ToolResult
    │   │   ├── tool_collection.py     # ToolCollection
    │   │   ├── python_execute.py      # Python code execution
    │   │   ├── bash.py                # Bash shell tool
    │   │   ├── browser_use_tool.py    # Browser automation
    │   │   ├── str_replace_editor.py  # File editor
    │   │   ├── web_search.py          # Web search
    │   │   ├── terminate.py           # Task termination
    │   │   ├── ask_human.py           # Human-in-the-loop
    │   │   ├── planning.py            # Plan management
    │   │   ├── crawl4ai.py            # Web crawling
    │   │   ├── computer_use_tool.py   # Desktop automation
    │   │   ├── create_chat_completion.py # Structured output
    │   │   ├── file_operators.py      # File operation strategies
    │   │   ├── mcp.py                 # MCP client tools
    │   │   ├── search/                # Search engine backends
    │   │   ├── sandbox/               # Sandbox-specific tools
    │   │   └── chart_visualization/   # Data visualization
    │   ├── prompt/                    # Prompt templates
    │   │   ├── manus.py               # Manus agent prompts
    │   │   ├── toolcall.py            # ToolCallAgent prompts
    │   │   ├── browser.py             # BrowserAgent prompts
    │   │   ├── swe.py                 # SWEAgent prompts
    │   │   ├── planning.py            # Planning flow prompts
    │   │   ├── mcp.py                 # MCPAgent prompts
    │   │   └── visualization.py       # DataAnalysis prompts
    │   ├── mcp/                       # MCP server implementation
    │   │   └── server.py              # MCPServer (FastMCP-based)
    │   ├── sandbox/                   # Docker sandbox system
    │   │   ├── client.py              # Sandbox client
    │   │   └── core/                  # Sandbox internals
    │   ├── daytona/                   # Daytona cloud sandbox
    │   │   ├── sandbox.py             # Sandbox management
    │   │   └── tool_base.py           # Sandbox tool base
    │   └── utils/                     # Utility functions
    │       ├── files_utils.py         # File utilities
    │       └── logger.py              # Alternative logger
    ├── config/                        # Configuration files
    │   ├── config.example.toml        # Main config template
    │   ├── config.example-*.toml      # Provider-specific configs
    │   └── mcp.example.json           # MCP server config
    ├── protocol/                      # Protocol integrations
    │   └── a2a/                       # Agent-to-Agent protocol
    ├── tests/                         # Test suite
    ├── examples/                      # Usage examples
    ├── workspace/                     # Agent workspace directory
    ├── main.py                        # Single agent entry point
    ├── run_flow.py                    # Multi-agent flow entry point
    ├── run_mcp.py                     # MCP agent entry point
    ├── run_mcp_server.py              # MCP server entry point
    └── sandbox_main.py                # Sandbox agent entry point


Data Flow
=========

Configuration Flow
------------------

::

    config/config.toml
         │
         ▼
    Config singleton (app/config.py)
         │
         ├──► AppConfig (Pydantic model)
         │      ├── llm: Dict[str, LLMSettings]
         │      ├── browser_config: BrowserSettings
         │      ├── search_config: SearchSettings
         │      ├── sandbox_config: SandboxSettings
         │      └── mcp_config: MCPSettings
         │
         ├──► LLM instances (per config_name)
         ├──► Agent configurations
         └──► Tool configurations

Request Flow (Single Agent)
---------------------------

::

    User Input (prompt string)
         │
         ▼
    Agent.run(prompt)
         │
         ├── Initialize: state = RUNNING, add prompt to memory
         │
         ▼
    ┌── Step Loop (max_steps iterations) ──────────────────────┐
    │                                                           │
    │   think()                                                 │
    │     │  Append next_step_prompt to memory                  │
    │     │  Call LLM.ask_tool(messages, tools)                 │
    │     │  Parse tool_calls from response                     │
    │     │  Add assistant message to memory                    │
    │     ▼                                                     │
    │   act()                                                   │
    │     │  For each tool_call:                                │
    │     │    Parse JSON arguments                             │
    │     │    Call ToolCollection.execute(name, args)           │
    │     │    Handle special tools (terminate → FINISHED)       │
    │     │    Add tool result to memory                        │
    │     ▼                                                     │
    │   Check: state == FINISHED? → exit loop                   │
    │   Check: is_stuck()? → modify prompt                      │
    │   Check: step >= max_steps? → exit loop                   │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
         │
         ▼
    Return final result string

Request Flow (Multi-Agent Planning)
------------------------------------

::

    User Input (prompt string)
         │
         ▼
    PlanningFlow.execute(prompt)
         │
         ├── Create initial plan via LLM + PlanningTool
         │     (decompose task into numbered steps)
         │
         ▼
    ┌── Plan Execution Loop ────────────────────────────────────┐
    │                                                           │
    │   Get current step from active plan                       │
    │     │                                                     │
    │     ▼                                                     │
    │   Select executor agent for step                          │
    │     │  (get_executor → agents dict lookup)                │
    │     │                                                     │
    │     ▼                                                     │
    │   Agent.run(step_prompt)                                  │
    │     │  (inner think-act loop)                              │
    │     │                                                     │
    │     ▼                                                     │
    │   Mark step completed via PlanningTool                    │
    │   Advance to next step                                    │
    │                                                           │
    │   Check: all steps completed? → exit loop                 │
    │   Check: max_steps exceeded? → exit loop                  │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
         │
         ▼
    Return aggregated result


Technology Stack
================

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Category
     - Technology
     - Purpose
   * - Language
     - Python 3.11-3.13
     - Core implementation language
   * - LLM API
     - OpenAI SDK (``openai``)
     - LLM communication (supports OpenAI, Azure, compatible APIs)
   * - Cloud LLM
     - AWS Bedrock (``boto3``)
     - Alternative LLM provider via Bedrock Converse API
   * - Data Validation
     - Pydantic v2
     - Configuration, schema, message validation
   * - Browser
     - Playwright + browser-use
     - Browser automation and web interaction
   * - Web Crawling
     - Crawl4AI
     - AI-optimized web content extraction
   * - Search
     - googlesearch, duckduckgo_search, baidusearch
     - Multi-engine web search
   * - Tokenization
     - tiktoken
     - Token counting for budget management
   * - Retry Logic
     - tenacity
     - Exponential backoff for API calls
   * - Containerization
     - Docker SDK
     - Sandboxed code execution
   * - Cloud Sandbox
     - Daytona SDK
     - Cloud-based sandboxed environments
   * - MCP
     - mcp library
     - Model Context Protocol client/server
   * - API Server
     - FastAPI + Uvicorn
     - A2A protocol server
   * - Logging
     - loguru
     - Structured logging with rotation
   * - Configuration
     - TOML (tomli)
     - Human-readable configuration files

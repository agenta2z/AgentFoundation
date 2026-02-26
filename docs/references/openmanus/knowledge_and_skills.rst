====================================
Knowledge and Skills System
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

.. note::

   This document provides a thorough, critical analysis of how OpenManus
   handles knowledge and skills. A key finding is that OpenManus
   **does not have a dedicated knowledge management system or formalized
   skill library**. This document explains what exists, what is absent,
   and the architectural implications.


Executive Summary
=================

OpenManus follows an **LLM-centric, stateless architecture** where:

- **"Knowledge"** exists only in prompt templates and conversation context
- **"Skills"** are hardcoded tool definitions with no learning mechanism
- **All state is ephemeral** -- lost when the process exits
- There is **no persistent memory**, no vector database, no RAG pipeline,
  no skill library that grows over time

This design keeps the system simple and predictable but limits its ability
to improve over time or leverage past interactions.


Knowledge Management
====================

Knowledge Creation
------------------

OpenManus does not produce, index, or store knowledge artifacts in the
traditional sense. The closest analogs to "knowledge creation" are:

**Conversation Memory**

During execution, messages accumulate in the agent's ``Memory`` object.
Each message captures a unit of interaction (user input, LLM reasoning,
tool results).

.. code-block:: python

    # app/schema.py, lines 159-188
    class Memory(BaseModel):
        messages: List[Message] = Field(default_factory=list)
        max_messages: int = Field(default=100)

        def add_message(self, message: Message):
            self.messages.append(message)
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]

This is purely ephemeral -- the memory is a Python list that exists only
for the duration of one ``agent.run()`` call.

**Plan Data**

The ``PlanningTool`` stores plans as in-memory dictionaries:

.. code-block:: python

    # app/tool/planning.py, line 69
    plans: dict = {}  # In-memory only

Plans include titles, numbered steps, step statuses, and step notes.
This structured data could be considered "generated knowledge" about
task decomposition, but it is lost when the process exits.

**File Edit History**

The ``StrReplaceEditor`` maintains undo history:

.. code-block:: python

    # app/tool/str_replace_editor.py, line 101
    _file_history: DefaultDict[PathLike, List[str]]

Again, this is in-memory only and serves a functional purpose (undo)
rather than knowledge management.

Knowledge Processing
--------------------

There is **no structured knowledge processing pipeline**. The LLM itself
serves as the knowledge processor. The closest mechanisms are:

**Web Content Extraction**

The ``BrowserUseTool.extract_content`` action uses an LLM call to
extract structured content from web pages:

::

    Web Page HTML
         │
         ▼
    markdownify (HTML → Markdown)
         │
         ▼
    LLM call with extraction goal
         │
         ▼
    Structured JSON output (consumed immediately, not stored)

**Web Search Results**

The ``WebSearch`` tool returns structured ``SearchResult`` objects with
title, URL, and optional content. The optional ``fetch_content`` flag
parses raw HTML with BeautifulSoup.

**Web Crawling**

The ``Crawl4aiTool`` extracts clean markdown from web pages using the
Crawl4AI library. Results are returned as tool output.

.. important::

   In all three cases, the extracted information enters the conversation
   memory as a tool result message. It is **never indexed, stored, or
   made searchable** beyond the current conversation context window.

Knowledge Storage
-----------------

**There is no persistent knowledge storage.** Specifically:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Storage Type
     - Status
   * - Relational database
     - Not present
   * - NoSQL database
     - Not present
   * - Vector database
     - Not present
   * - File-based knowledge store
     - Not present
   * - Embedding storage
     - Not present
   * - RAG pipeline
     - Not present
   * - Knowledge graph
     - Not present
   * - Session persistence
     - Not present
   * - Cross-session memory
     - Not present

All state exists only as in-memory Python objects:

- ``Memory.messages`` -- conversation history
- ``PlanningTool.plans`` -- plan data
- ``StrReplaceEditor._file_history`` -- file edit undo
- ``LLM._instances`` -- singleton LLM instances
- ``MCPClients.sessions`` -- MCP server connections

Knowledge Retrieval
-------------------

There is **no internal knowledge retrieval system**. The system provides
two external retrieval mechanisms:

1. **Web Search** (``WebSearch`` tool): Real-time internet search with
   engine fallback (Google → DuckDuckGo → Baidu → Bing)

2. **Browser Content Extraction**: Scraping and parsing specific web
   pages via ``BrowserUseTool`` and ``Crawl4aiTool``

The only "internal retrieval" is the LLM seeing the full conversation
context (up to ``max_messages=100`` messages). The LLM uses this context
implicitly to inform its responses.

.. note::

   The word "embedding" does not appear in any source file. There is
   no vector similarity search, no semantic retrieval, and no indexing
   of past interactions.

How Agents Access Knowledge
----------------------------

Agents access information through exactly **two channels**:

**1. System Prompts (Static, Baked-In Knowledge)**

Each agent type has a static system prompt that defines its identity,
capabilities, and behavioral instructions. These prompts are the
"hardcoded knowledge" of each agent.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Agent
     - File
     - Prompt Size
   * - Manus
     - ``app/prompt/manus.py``
     - ~4 lines (minimal identity + workspace)
   * - Browser
     - ``app/prompt/browser.py``
     - ~94 lines (detailed JSON format, action rules)
   * - Planning
     - ``app/prompt/planning.py``
     - ~27 lines (plan creation instructions)
   * - SWE
     - ``app/prompt/swe.py``
     - ~22 lines (editor interface description)
   * - MCP
     - ``app/prompt/mcp.py``
     - ~43 lines (multi-prompt set)
   * - ToolCall
     - ``app/prompt/toolcall.py``
     - ~1 line (minimal base prompt)
   * - Visualization
     - ``app/prompt/visualization.py``
     - ~5 lines (workspace + report requirement)

**2. Conversation Memory (Dynamic, Ephemeral Context)**

The accumulated ``Message`` objects in ``Memory.messages`` are passed to
the LLM on every ``think()`` call. The LLM sees the full history within
token limits and uses it as working context.


Skill System
============

Skill Definition
----------------

In OpenManus, **"skills" are tools** -- instances of ``BaseTool``
subclasses. Each tool is defined by:

.. code-block:: python

    class BaseTool(ABC, BaseModel):
        name: str               # Unique identifier
        description: str        # Natural language description for LLM
        parameters: dict = {}   # JSON Schema for input parameters

        @abstractmethod
        async def execute(self, **kwargs) -> Any:
            """Execute the tool."""

Tools are converted to OpenAI function-calling format:

.. code-block:: json

    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Executes Python code...",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }

Skill Discovery
---------------

Skills are **statically configured** per agent type. Each agent class
declares its tools at class definition time:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Agent
     - Tools
   * - Manus
     - PythonExecute, BrowserUseTool, StrReplaceEditor, AskHuman,
       Terminate + MCP tools
   * - BrowserAgent
     - BrowserUseTool, Terminate
   * - SWEAgent
     - Bash, StrReplaceEditor, Terminate
   * - DataAnalysis
     - NormalPythonExecute, VisualizationPrepare, DataVisualization,
       Terminate
   * - MCPAgent
     - Dynamic from MCP server
   * - SandboxManus
     - AskHuman, Terminate + SandboxBrowser, SandboxFiles,
       SandboxShell, SandboxVision + MCP tools

The **only dynamic discovery mechanism** is MCP (Model Context Protocol):

.. code-block:: python

    # app/tool/mcp.py
    async def _initialize_and_list_tools(self, server_id):
        await session.initialize()
        response = await session.list_tools()
        for tool in response.tools:
            server_tool = MCPClientTool(
                name=f"mcp_{server_id}_{tool.name}",
                description=tool.description,
                parameters=tool.inputSchema,
                session=session,
            )
            self.tool_map[tool_name] = server_tool

Skill Learning and Acquisition
-------------------------------

.. warning::

   **OpenManus cannot learn, create, or acquire new skills at runtime.**

There is:

- No skill generation from demonstrations
- No few-shot learning pipeline for skills
- No skill library that grows over time
- No mechanism to save successful action sequences as reusable skills
- No reinforcement learning for skill improvement within the framework

The only quasi-dynamic capability is MCP server tools, which are
discovered at connection time but defined externally.

.. note::

   The README mentions a separate project, `OpenManus-RL
   <https://github.com/OpenManus/OpenManus-RL>`_, for reinforcement
   learning-based tuning of LLM agents. This is a separate project,
   not integrated into the OpenManus codebase.

Skill Execution Flow
--------------------

The complete skill (tool) execution flow:

::

    1. think() — LLM Call
       │
       │  Messages + tool schemas → LLM
       │  LLM returns tool_calls (name + JSON args)
       │
       ▼
    2. act() — Tool Dispatch
       │
       │  For each tool_call:
       │    Parse JSON arguments
       │    Look up tool in ToolCollection.tool_map
       │    Call tool.execute(**args)
       │
       ▼
    3. Result Handling
       │
       │  ToolResult(output, error, base64_image)
       │  Truncate to max_observe characters
       │  Add as Message.tool_message() to memory
       │
       ▼
    4. Special Tool Check
       │
       │  If tool name in special_tool_names (e.g., "terminate"):
       │    Set agent.state = AgentState.FINISHED

Skill Composition
-----------------

There is **no formal skill composition framework**. However, several
implicit composition patterns exist:

1. **Sequential Composition via LLM**: The LLM can call multiple tools
   in a single ``think()`` step. The ``act()`` method executes them
   sequentially.

2. **BrowserUseTool embeds WebSearch**: The ``web_search`` action
   internally calls ``WebSearch.execute()`` then navigates the browser
   to the first result.

3. **VisualizationPrepare → DataVisualization**: A two-tool pipeline
   where the first prepares metadata and the second generates charts.

4. **PlanningFlow orchestrates multi-agent tool use**: Different agents
   with different tools are coordinated through the planning system.


Memory System Deep Dive
========================

Architecture
------------

::

    ┌──────────────────────────────────────────────────────────┐
    │                    BaseAgent                             │
    │                                                          │
    │  memory: Memory                                          │
    │    └── messages: List[Message]  (max 100)                │
    │                                                          │
    │  update_memory(role, content)                            │
    │    └── memory.add_message(Message(...))                   │
    │                                                          │
    │  is_stuck() → bool                                       │
    │    └── Compares last N assistant messages for duplicates  │
    └──────────────────────────────────────────────────────────┘

Message Structure
-----------------

.. code-block:: python

    class Message(BaseModel):
        role: Role          # "system", "user", "assistant", "tool"
        content: str | None
        tool_calls: List[ToolCall] | None
        name: str | None
        tool_call_id: str | None
        base64_image: str | None

        @classmethod
        def user_message(cls, content: str) -> "Message": ...

        @classmethod
        def assistant_message(cls, content: str) -> "Message": ...

        @classmethod
        def tool_message(cls, content: str, tool_call_id: str,
                         base64_image: str | None = None) -> "Message": ...

Memory Lifecycle
----------------

1. **Initialization**: Empty ``Memory`` with ``max_messages=100``
2. **User Prompt**: First message added as ``user`` role
3. **Think Loop**: Each ``think()`` adds:
   - ``next_step_prompt`` as ``user`` message
   - LLM response as ``assistant`` message
4. **Act Loop**: Each tool result added as ``tool`` message
5. **Overflow**: When messages exceed ``max_messages``, oldest are
   dropped (sliding window)
6. **Termination**: Memory is abandoned when ``run()`` completes

Cross-Agent Memory
------------------

In the ``PlanningFlow``, agents do **not** share memory directly.
Instead, step results are communicated through:

- Plan step notes (written to ``PlanningTool``)
- Step prompts that include context from previous steps

Each agent starts with a fresh ``Memory`` for each step execution.


Prompt Templates as Baked-In Knowledge
========================================

All agent prompts are static strings defined in ``app/prompt/``:

Manus Prompt
------------

.. code-block:: text

    SYSTEM_PROMPT:
    You are OpenManus, an all-capable AI assistant, aimed at
    solving any task presented by the user. You have various
    tools at your disposal that you can call upon to
    efficiently complete complex requests.

    NEXT_STEP_PROMPT:
    Based on user needs, proactively select the most
    appropriate tool or combination of tools...
    Current working directory: {directory}

Browser Prompt (Detailed)
--------------------------

The browser prompt is the most detailed (~94 lines), specifying:

- JSON response format with required fields
- Action rules (one action per step, use indexes not coordinates)
- Memory tracking requirements
- Visual context handling (screenshots)
- Error recovery patterns

Planning Prompt
---------------

Instructs the LLM to create structured, actionable plans with
dependency tracking and verification steps.

SWE Prompt
----------

Describes the file editor interface, response format requirements,
and indentation handling for code editing tasks.

.. note::

   There are **zero few-shot examples** anywhere in the codebase.
   The browser prompt includes JSON format specifications but not
   complete task-solution demonstrations.


Architectural Analysis and Limitations
========================================

What OpenManus Does Well
------------------------

1. **Simplicity**: The stateless design is easy to understand, debug,
   and deploy. There are no complex persistence layers to manage.

2. **LLM-Centric Intelligence**: By delegating all reasoning to the
   LLM, the system benefits from the LLM's general capabilities
   without needing domain-specific knowledge engineering.

3. **Tool Extensibility**: Adding new tools is straightforward --
   subclass ``BaseTool``, implement ``execute()``, and register with
   an agent's ``ToolCollection``.

4. **MCP Interoperability**: The MCP integration enables dynamic tool
   discovery from external servers, providing runtime extensibility.

What OpenManus Lacks
---------------------

1. **No Learning**: The system cannot improve from experience. Every
   session starts from scratch with the same prompts and tools.

2. **No Persistent Memory**: There is no way to remember information
   across sessions. Users must re-provide context each time.

3. **No Knowledge Base**: There is no structured repository of facts,
   procedures, or domain knowledge that agents can query.

4. **No Skill Library**: There is no mechanism to define, store, and
   reuse complex multi-tool procedures as named skills.

5. **No Semantic Retrieval**: Without embeddings or vector search,
   agents cannot efficiently search through large knowledge corpora.

6. **No Inter-Agent Memory Sharing**: In multi-agent flows, agents
   communicate only through the planning tool's step notes.

Recommendations for Enhancement
---------------------------------

For users who want to extend OpenManus with knowledge and skill
capabilities, the following enhancements would be impactful:

1. **Persistent Memory**

   Add a vector database (e.g., ChromaDB, FAISS) for storing and
   retrieving past interactions, learned facts, and task outcomes.

   .. code-block:: python

       class PersistentMemory:
           def __init__(self, db_path: str):
               self.db = chromadb.PersistentClient(db_path)
               self.collection = self.db.get_or_create_collection("memory")

           async def store(self, text: str, metadata: dict):
               embedding = await self.embed(text)
               self.collection.add(embeddings=[embedding], ...)

           async def retrieve(self, query: str, k: int = 5):
               results = self.collection.query(query_texts=[query], n_results=k)
               return results

2. **RAG Pipeline**

   Augment the ``think()`` method to retrieve relevant context before
   each LLM call:

   .. code-block:: python

       async def think(self):
           # Retrieve relevant knowledge
           context = await self.knowledge_base.retrieve(
               self.memory.messages[-1].content
           )
           # Inject into system prompt
           augmented_prompt = f"{self.system_prompt}\n\nRelevant context:\n{context}"
           # Call LLM with augmented context
           response = await self.llm.ask_tool(messages, augmented_prompt, ...)

3. **Skill Library**

   Define reusable multi-step procedures in YAML/JSON:

   .. code-block:: yaml

       skills:
         - name: "web_research"
           steps:
             - tool: web_search
               args: {query: "{topic}"}
             - tool: browser_use
               args: {action: extract_content, goal: "{goal}"}
             - tool: str_replace_editor
               args: {command: create, path: "research.md"}

4. **Cross-Session Memory**

   Serialize ``Memory`` objects for session continuity:

   .. code-block:: python

       class SessionManager:
           def save_session(self, session_id: str, memory: Memory):
               data = memory.model_dump_json()
               Path(f"sessions/{session_id}.json").write_text(data)

           def load_session(self, session_id: str) -> Memory:
               data = Path(f"sessions/{session_id}.json").read_text()
               return Memory.model_validate_json(data)

.. _knowledge-memory-tools:

============
Memory Tools
============

The memory system exposes tools to agents for searching, reading, and adding
to the knowledge base during conversations.

memory_search
=============

The primary retrieval tool — performs semantic search across memory files.

**Tool name**: ``memory_search``

**Description**: *"Mandatory recall step: semantically search MEMORY.md +
memory/\*.md (and optional session transcripts) before answering questions
about prior work, decisions, dates, people, preferences, or todos; returns top
snippets with path + lines."*

Parameters
----------

.. code-block:: typescript

   {
     query: string;            // Search query text (required)
     maxResults?: number;      // Max results to return (default: 6)
     minScore?: number;        // Min relevance score (default: 0.35)
   }

Return Value
------------

Returns a JSON object with:

.. code-block:: typescript

   {
     results: Array<{
       path: string;           // e.g., "memory/2026-02-23.md"
       startLine: number;      // Chunk start line
       endLine: number;        // Chunk end line
       score: number;          // Relevance score (0-1)
       snippet: string;        // Matching text content
       source: "memory" | "sessions";
       citation?: string;      // e.g., "Source: memory/2026-02-23.md#L5-L8"
     }>;
     disabled?: boolean;       // true if memory is unavailable
     error?: string;           // Error message if search failed
     backend: "builtin" | "qmd";
     provider: string;         // Embedding provider used
   }

Citation Modes
--------------

Citations are controlled by ``memory.citations`` in ``openclaw.json``:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Mode
     - Behavior
   * - ``auto``
     - Include citations in direct chats, suppress in groups/channels (default)
   * - ``on``
     - Always include citations
   * - ``off``
     - Never include citations

Citation logic checks the session key to determine context: direct
sessions get citations, group/channel sessions do not (in ``auto`` mode).

Result Budget
-------------

The ``maxInjectedChars`` setting limits how much memory content is injected
into the conversation to prevent overwhelming the context window.

"Mandatory Recall Step"
-----------------------

The tool description explicitly labels ``memory_search`` as a "mandatory
recall step." This instruction in the system prompt encourages the LLM to
call ``memory_search`` proactively before answering questions about prior
conversations, user preferences, or stored knowledge.

Source: ``src/agents/tools/memory-tool.ts:40-53``

memory_get
==========

Safe snippet reader for memory files with pagination.

**Tool name**: ``memory_get``

Parameters
----------

.. code-block:: typescript

   {
     path: string;             // Relative file path (required)
     from?: number;            // Start line number (1-indexed)
     lines?: number;           // Number of lines to read
   }

Returns the requested text content with the resolved absolute path. Files are
read relative to the agent's workspace directory.

This tool is used by agents to read specific sections of memory files after
``memory_search`` returns a relevant snippet with line references.

Source: ``src/agents/tools/memory-tool.ts``

Unavailability Handling
=======================

When the memory system is unavailable (no embedding provider, database
corruption, etc.), ``memory_search`` returns:

.. code-block:: json

   {
     "disabled": true,
     "error": "Memory search unavailable: no embedding provider configured"
   }

The tool description instructs the agent to surface this to the user rather
than silently ignoring it.

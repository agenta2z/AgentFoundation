.. _tutorial-understanding-memory:

====================
Understanding Memory
====================

This tutorial walks through the complete memory lifecycle — from an agent
writing knowledge to that knowledge being retrieved in a future conversation.

Overview
========

OpenClaw's memory system gives agents persistent knowledge across sessions.
The lifecycle has four stages:

.. code-block:: text

   Write → Index → Store → Retrieve
     │        │       │        │
     │        │       │        └─ Hybrid search (vector + keyword)
     │        │       └─ SQLite database (chunks, FTS5, vec0)
     │        └─ Chunk + embed markdown files
     └─ Agent writes MEMORY.md or memory/YYYY-MM-DD.md

Let's trace each stage with a concrete example.

Stage 1: Writing Memory
=========================

Scenario: During a conversation, the user tells the agent their preferred
programming language and project structure.

The Agent's Perspective
------------------------

The agent has two memory files it can write to:

1. **MEMORY.md** — Evergreen knowledge (preferences, facts, long-term context)
2. **memory/YYYY-MM-DD.md** — Episodic entries (what happened today)

The agent decides to write:

**MEMORY.md**:

.. code-block:: markdown

   ## User Preferences

   - Preferred language: TypeScript
   - Style: functional, minimal dependencies
   - Testing: Vitest with >80% coverage

**memory/2026-02-23.md**:

.. code-block:: markdown

   ## Session Notes

   - Discussed project architecture for the new API gateway
   - User prefers Express v5 over Fastify
   - Decided on a monorepo structure with pnpm workspaces

Automatic Memory Flush
-----------------------

Memory isn't only written explicitly. When the conversation grows long (token
usage exceeds ``DEFAULT_MEMORY_FLUSH_SOFT_TOKENS = 4000``), the system
triggers an automatic memory flush:

1. ``shouldRunMemoryFlush()`` checks if enough new conversation has occurred
   since the last flush
2. The agent receives a special prompt asking it to summarize important
   information
3. The agent writes key facts to memory files
4. The compaction counter is updated to prevent repeated flushes

Source: ``src/auto-reply/reply/memory-flush.ts``

Stage 2: Indexing
==================

Once memory files are written (or modified), the indexing system processes
them into searchable chunks.

File Watching
--------------

The memory manager watches for file changes using chokidar with a 1500ms
debounce. When a change is detected:

1. The file's hash is compared against the stored hash in the ``files`` table
2. If different, the file is re-chunked and re-embedded
3. Old chunks for the file are replaced with new ones

Chunking
---------

The ``chunkMarkdown()`` function splits files into overlapping chunks:

- **Target size**: ~400 tokens (~1600 characters)
- **Overlap**: ~80 tokens between adjacent chunks
- **Boundaries**: Chunks split at markdown heading boundaries when possible

For our ``MEMORY.md`` file (small), the entire content becomes a single chunk.

For ``memory/2026-02-23.md``, it also becomes one chunk since it's short.

.. code-block:: text

   MEMORY.md
   ├── Chunk 1: "## User Preferences\n- Preferred language: TypeScript..."
   │   (lines 1-5, 127 chars)
   └── (single chunk — file is small)

   memory/2026-02-23.md
   ├── Chunk 1: "## Session Notes\n- Discussed project architecture..."
   │   (lines 1-5, 182 chars)
   └── (single chunk — file is small)

Source: ``src/memory/internal.ts``

Embedding
----------

Each chunk is converted to a vector embedding — a numerical representation
that captures semantic meaning. The embedding provider is selected
automatically:

.. code-block:: text

   Auto-selection cascade:
   1. Local (embeddinggemma-300m via node-llama-cpp)  ← preferred
   2. OpenAI (text-embedding-3-small)
   3. Gemini (gemini-embedding-001)
   4. Voyage (voyage-4-large)
   5. Mistral (mistral-embed)

The local provider runs entirely on the user's machine with no API calls.
All embeddings are L2-normalized for consistent cosine similarity calculations.

Source: ``src/memory/embeddings.ts``

Stage 3: Storage
=================

Indexed chunks are stored in a SQLite database at
``~/.openclaw/state/memory/<agentId>.sqlite``.

Database Tables
----------------

.. code-block:: text

   ┌──────────────────────────────────────────────────────┐
   │                    SQLite Database                    │
   │                                                      │
   │  meta         │ key-value index metadata             │
   │  files        │ tracked files (path, hash, mtime)    │
   │  chunks       │ text chunks with embeddings          │
   │  chunks_fts   │ FTS5 virtual table (BM25 search)     │
   │  chunks_vec   │ vec0 virtual table (vector search)   │
   │  embedding_cache │ cached embeddings by content hash │
   └──────────────────────────────────────────────────────┘

Our two chunks are stored in the ``chunks`` table:

.. code-block:: sql

   INSERT INTO chunks (id, path, source, start_line, end_line, hash, model,
                       text, embedding, updated_at)
   VALUES
     ('chunk-001', 'MEMORY.md', 'memory', 1, 5, 'abc123...', 'local',
      '## User Preferences...', <768-dim vector>, 1708700000),
     ('chunk-002', 'memory/2026-02-23.md', 'memory', 1, 5, 'def456...',
      'local', '## Session Notes...', <768-dim vector>, 1708700000);

Simultaneously, the text is indexed in the FTS5 table for keyword search,
and the vector is indexed in the vec0 table for similarity search.

Source: ``src/memory/memory-schema.ts``

Safe Reindex
-------------

When a full reindex is needed (e.g., embedding model change), the system:

1. Creates a temporary database
2. Indexes all files into the temp DB
3. Atomically swaps the temp DB with the live DB

This prevents data loss if the process crashes mid-reindex.

Source: ``src/memory/manager-sync-ops.ts``

Stage 4: Retrieval
===================

Days later, the user starts a new conversation and asks: "What testing
framework do I prefer?"

The ``memory_search`` Tool
---------------------------

The agent's system prompt includes ``memory_search`` described as a
"mandatory recall step." When the agent encounters a question about prior
preferences, it calls:

.. code-block:: json

   {
     "tool": "memory_search",
     "params": {
       "query": "testing framework preference",
       "maxResults": 6,
       "minScore": 0.35
     }
   }

The Retrieval Pipeline
-----------------------

The query flows through a multi-stage pipeline:

.. code-block:: text

   "testing framework preference"
     │
     ├── Query Expansion
     │   └── extractKeywords() → ["testing", "framework", "preference"]
     │
     ├── Vector Search (weight: 0.7)
     │   ├── Embed query → 768-dim vector
     │   ├── sqlite-vec cosine distance search
     │   └── Returns: MEMORY.md chunk (score: 0.82)
     │
     ├── Keyword Search (weight: 0.3)
     │   ├── FTS5 BM25 query: "testing OR framework OR preference"
     │   └── Returns: MEMORY.md chunk (score: 0.71)
     │
     ├── Hybrid Merge
     │   ├── Normalize scores to [0, 1]
     │   ├── Combined = 0.7 × vector + 0.3 × keyword
     │   └── MEMORY.md chunk: 0.7 × 0.82 + 0.3 × 0.71 = 0.787
     │
     ├── Temporal Decay
     │   ├── MEMORY.md: no date → evergreen (no decay)
     │   ├── memory/2026-02-23.md: 0 days old → decay factor ≈ 1.0
     │   └── (30-day half-life: score × 0.5^(age/30))
     │
     ├── MMR Re-ranking
     │   ├── Maximal Marginal Relevance (Jaccard-based diversity)
     │   └── Prevents near-duplicate chunks from dominating results
     │
     ├── Threshold Filter
     │   └── Remove results below minScore (0.35)
     │
     └── Top-K Selection
         └── Return top 6 results

**Result returned to the agent:**

.. code-block:: json

   {
     "results": [
       {
         "path": "MEMORY.md",
         "startLine": 1,
         "endLine": 5,
         "score": 0.787,
         "snippet": "## User Preferences\n- Preferred language: TypeScript\n- Style: functional, minimal dependencies\n- Testing: Vitest with >80% coverage",
         "source": "memory",
         "citation": "Source: MEMORY.md#L1-L5"
       }
     ],
     "backend": "builtin",
     "provider": "local"
   }

Source: ``src/memory/manager-search.ts``, ``src/memory/hybrid.ts``

The Agent Responds
-------------------

With the memory search result, the agent can now answer confidently:

   "You prefer Vitest for testing, with a coverage threshold above 80%."

The citation (``Source: MEMORY.md#L1-L5``) may be included depending on the
``memory.citations`` setting (auto/on/off).

Graceful Degradation
=====================

The memory system degrades gracefully when components are unavailable:

.. code-block:: text

   Full hybrid search (vector + keyword)
     ↓ (no embedding provider)
   Vector-only search (pre-computed embeddings still work)
     ↓ (no vectors at all)
   FTS-only search (keyword matching via BM25)
     ↓ (FTS unavailable)
   Memory disabled (returns { disabled: true })

Even without an embedding provider, previously computed embeddings remain
usable. New content simply won't get embeddings until a provider is
configured.

Session Transcript Indexing
============================

Beyond explicit memory files, the system can also index session transcripts
(past conversations). The ``buildSessionEntry()`` function converts JSONL
transcript files into Markdown-like content that can be chunked and embedded
alongside memory files.

This allows agents to search not just curated knowledge, but the full
history of past conversations.

Source: ``src/memory/session-files.ts``

Summary
========

The memory lifecycle:

1. **Write**: Agent creates/updates ``MEMORY.md`` or ``memory/YYYY-MM-DD.md``
   (manually or via automatic memory flush)
2. **Index**: Files are chunked (~400 tokens), embedded (768-dim vectors),
   and indexed in SQLite (chunks + FTS5 + vec0)
3. **Store**: SQLite database at ``~/.openclaw/state/memory/<agentId>.sqlite``
   with safe reindex via atomic swap
4. **Retrieve**: Hybrid search (0.7 vector + 0.3 keyword) → temporal decay
   → MMR diversity → threshold (0.35) → top-K (6)

Key source files:

- ``src/memory/manager.ts`` — Main memory manager
- ``src/memory/manager-sync-ops.ts`` — File sync and indexing
- ``src/memory/manager-search.ts`` — Search implementation
- ``src/memory/hybrid.ts`` — Hybrid merge algorithm
- ``src/memory/temporal-decay.ts`` — Time-based score decay
- ``src/memory/mmr.ts`` — Diversity re-ranking
- ``src/memory/embeddings.ts`` — Embedding providers
- ``src/memory/memory-schema.ts`` — Database schema
- ``src/agents/tools/memory-tool.ts`` — Agent-facing tools
